"""
goal_templates.py — Persistent goal-decomposition template registry.

A goal template is the backward-chaining dual of a rule:
  Rule  (forward): observation  → response
  Goal (backward): desired_outcome :- required_sub_tasks

Templates are stored in goal_templates.json keyed by (game_id, level).
At episode start, _inject_initial_goals() instantiates the relevant template
into the live GoalManager, replacing the hardcoded fallback goals.

Design
------
Each template entry holds:
  - A flat list of *nodes* with parent references (tree by convention).
  - A *variables* dict — values filled in at bootstrap time from level meta
    and the BFS observational tour (n_rot_visits, goal_rot, …).
  - Node descriptions may use {variable} placeholders; ``instantiate()``
    formats them before handing off to GoalManager.push().

This keeps the template human-readable in JSON while letting the runtime
substitute episode-specific values (e.g. current level number) at push time.

Cross-game generality
---------------------
Templates tagged game_id="*" apply to any game.  Game-specific templates
are tried first; the generic fallback is used otherwise.

Usage
-----
    registry = GoalTemplateRegistry(path="goal_templates.json")

    # Bootstrap writes:
    registry.record_template(
        game_id="ls20", level=1,
        nodes=[
            {"id": "n0", "parent": None,  "priority": 1,
             "description": "Complete {game_id} level {level} …"},
            {"id": "n1", "parent": "n0", "priority": 2,
             "description": "Transform player: visit ROT_CHANGER {n_rot_visits} time(s)"},
            …
        ],
        variables={"goal_rot": 0, "n_rot_visits": 1, …},
    )

    # Episode start reads:
    nodes = registry.instantiate("ls20", level=1, extra={"level": 1})
    # → [{"description": "…", "priority": 1, "parent_node_id": None}, …]
    # Push into GoalManager in order, mapping node_id → GoalManager id.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_FALLBACK_KEY = "*"


class GoalTemplateRegistry:
    """
    JSON-backed store for goal decomposition templates.

    On-disk format::

        {
          "ls20:1": {
            "game_id": "ls20",
            "level": 1,
            "source": "bootstrap",
            "variables": {"goal_rot": 0, "n_rot_visits": 1, …},
            "nodes": [
              {"id": "n0", "parent": null, "priority": 1,
               "description": "Complete {game_id} level {level}…"},
              {"id": "n1", "parent": "n0", "priority": 2,
               "description": "Visit ROT_CHANGER {n_rot_visits} time(s)…"},
              …
            ]
          }
        }
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        # key → template dict
        self._data: dict[str, dict] = {}
        if self.path and self.path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record_template(
        self,
        game_id: str,
        level: int,
        nodes: list[dict],
        variables: dict | None = None,
        source: str = "bootstrap",
    ) -> None:
        """Store (or replace) the goal template for (game_id, level).

        Parameters
        ----------
        nodes
            Ordered list of node dicts.  Each node must have at minimum:
              - ``id``      — unique within this template (str)
              - ``parent``  — parent node id, or None for root
              - ``priority``— int; lower = higher priority
              - ``description`` — may contain {var} placeholders
        variables
            Dict used to format {placeholder} strings in descriptions.
            Stored with the template so future instantiations can re-use
            them without re-running the bootstrap.
        """
        key = f"{game_id}:{level}"
        self._data[key] = {
            "game_id":   game_id,
            "level":     level,
            "source":    source,
            "variables": variables or {},
            "nodes":     nodes,
        }
        if self.path:
            self._save()

    # ------------------------------------------------------------------
    # Read / instantiate
    # ------------------------------------------------------------------

    def get_template(self, game_id: str, level: int) -> dict | None:
        """Return raw template dict for (game_id, level), or None."""
        return (
            self._data.get(f"{game_id}:{level}")
            or self._data.get(f"{_FALLBACK_KEY}:{level}")
        )

    def instantiate(
        self,
        game_id: str,
        level: int,
        extra: dict | None = None,
    ) -> list[dict]:
        """Return a list of node dicts ready to push into GoalManager.

        Each returned dict has:
          - ``description``    — formatted string (placeholders resolved)
          - ``priority``       — int
          - ``parent_node_id`` — template node id of parent, or None

        Caller is responsible for pushing nodes in order and building the
        node_id → GoalManager.goal_id mapping.

        Returns [] if no template is stored for (game_id, level).
        """
        tmpl = self.get_template(game_id, level)
        if not tmpl:
            return []

        variables = dict(tmpl.get("variables", {}))
        variables.setdefault("game_id", game_id)
        variables.setdefault("level",   level)
        if extra:
            variables.update(extra)

        result = []
        for node in tmpl["nodes"]:
            desc = node.get("description", "")
            try:
                desc = desc.format(**variables)
            except (KeyError, ValueError):
                pass  # leave unresolved placeholders as-is
            result.append({
                "description":    desc,
                "priority":       node.get("priority", 5),
                "parent_node_id": node.get("parent"),
                "metadata":       node.get("metadata", {}),
                "node_id":        node["id"],
            })
        return result

    def all_keys(self) -> list[str]:
        return list(self._data.keys())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        assert self.path is not None
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        assert self.path is not None
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._data = {}


# ---------------------------------------------------------------------------
# GoalManager integration helper
# ---------------------------------------------------------------------------

def push_template_into_manager(
    manager: Any,
    nodes: list[dict],
    activate_first: bool = True,
) -> dict[str, str]:
    """Push instantiated template nodes into a GoalManager.

    Parameters
    ----------
    manager
        A ``GoalManager`` instance.
    nodes
        Output of ``GoalTemplateRegistry.instantiate()``.
    activate_first
        If True, activate the root goal immediately.

    Returns
    -------
    dict mapping template node_id → GoalManager goal id.
    """
    node_to_goal: dict[str, str] = {}

    for node in nodes:
        parent_goal_id = (
            node_to_goal.get(node["parent_node_id"])
            if node["parent_node_id"]
            else None
        )
        goal = manager.push(
            description=node["description"],
            priority=node["priority"],
            parent_id=parent_goal_id,
            metadata=node.get("metadata", {}),
        )
        node_to_goal[node["node_id"]] = goal.id

    if activate_first and node_to_goal:
        first_id = node_to_goal[nodes[0]["node_id"]]
        manager.activate(first_id)

    return node_to_goal
