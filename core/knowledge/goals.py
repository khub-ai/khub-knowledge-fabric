"""
goals.py — GoalManager for the Knowledge Fabric framework.

Maintains a dynamic goal tree that agents can push sub-goals onto and
resolve/abandon as solving progresses.  Goals are the primary driver of
agent behavior in domains where a sequence of sub-tasks must be planned
and executed (e.g. ARC-AGI-3 interactive tasks).

Design principles:
  - Flat list with `parent_id` links — simple tree without extra libraries.
  - Five statuses: pending | active | succeeded | failed | abandoned
  - Only ONE goal can be active at a time (the "current focus").
  - `format_for_prompt()` renders the tree as an indented outline.
  - `parse_agent_updates()` extracts goal/state update blocks from raw LLM text.
  - `apply_updates()` applies the parsed list of goal mutations.

Goal update JSON (emitted by agents inside a ```json ... ``` block):
  {
    "goal_updates": [
      {"action": "push",    "description": "...", "priority": 1, "parent_id": null},
      {"action": "resolve", "id": "g-3", "result": "found color=5"},
      {"action": "abandon", "id": "g-2", "reason": "approach failed"},
      {"action": "activate","id": "g-5"}
    ],
    "state_updates": {
      "description": "recorded color",
      "set": {"dominant_color": 5},
      "delete": []
    }
  }

Both keys are optional; the block may contain only `goal_updates` or only
`state_updates`.
"""

from __future__ import annotations

import json
import re
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Goal status constants
# ---------------------------------------------------------------------------

STATUS_PENDING   = "pending"
STATUS_ACTIVE    = "active"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED    = "failed"
STATUS_ABANDONED = "abandoned"

_TERMINAL = {STATUS_SUCCEEDED, STATUS_FAILED, STATUS_ABANDONED}


# ---------------------------------------------------------------------------
# Goal dataclass
# ---------------------------------------------------------------------------

@dataclass
class Goal:
    """
    A single node in the goal tree.

    Attributes
    ----------
    id : str
        Unique identifier (auto-generated if not provided).
    description : str
        Human-readable description of what this goal is trying to achieve.
    status : str
        One of: pending | active | succeeded | failed | abandoned.
    parent_id : str | None
        ID of the parent goal (None for top-level goals).
    priority : int
        Lower number = higher priority.  Used to order active/pending goals.
    result : str
        Free-text outcome recorded on resolve/fail.
    metadata : dict
        Any extra domain-specific data the agent wants to attach.
    created : str
        ISO-8601 timestamp when the goal was created.
    resolved : str | None
        ISO-8601 timestamp when the goal reached a terminal status.
    """

    description: str
    id: str = field(default_factory=lambda: f"g-{uuid.uuid4().hex[:8]}")
    status: str = STATUS_PENDING
    parent_id: Optional[str] = None
    priority: int = 5
    result: str = ""
    metadata: dict = field(default_factory=dict)
    created: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    resolved: Optional[str] = None

    def is_terminal(self) -> bool:
        return self.status in _TERMINAL

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "parent_id": self.parent_id,
            "priority": self.priority,
            "result": self.result,
            "metadata": self.metadata,
            "created": self.created,
            "resolved": self.resolved,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Goal":
        return cls(
            id=d.get("id", f"g-{uuid.uuid4().hex[:8]}"),
            description=d.get("description", ""),
            status=d.get("status", STATUS_PENDING),
            parent_id=d.get("parent_id"),
            priority=d.get("priority", 5),
            result=d.get("result", ""),
            metadata=d.get("metadata", {}),
            created=d.get("created", datetime.now(timezone.utc).isoformat()),
            resolved=d.get("resolved"),
        )


# ---------------------------------------------------------------------------
# GoalManager
# ---------------------------------------------------------------------------

class GoalManager:
    """
    Manages the goal tree for a single task run.

    Goals are ephemeral — they are not persisted to disk and exist only for
    the lifetime of the task.

    Parameters
    ----------
    task_id : str
        Identifier of the current task (for logging).
    dataset_tag : str
        Dataset/use-case tag (e.g. "arc-agi-3").
    root_description : str | None
        If provided, a root goal is automatically pushed with this description
        and immediately activated.
    """

    def __init__(
        self,
        task_id: str = "",
        dataset_tag: str = "",
        root_description: str | None = None,
    ) -> None:
        self.task_id = task_id
        self.dataset_tag = dataset_tag
        self._goals: list[Goal] = []
        self._counter = 0

        if root_description:
            root = self.push(root_description, priority=0)
            self.activate(root.id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _by_id(self, goal_id: str) -> Optional[Goal]:
        for g in self._goals:
            if g.id == goal_id:
                return g
        return None

    def _children(self, parent_id: str) -> list[Goal]:
        return [g for g in self._goals if g.parent_id == parent_id]

    def _depth(self, goal: Goal) -> int:
        depth = 0
        current = goal
        while current.parent_id:
            parent = self._by_id(current.parent_id)
            if parent is None:
                break
            current = parent
            depth += 1
        return depth

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def push(
        self,
        description: str,
        priority: int = 5,
        parent_id: Optional[str] = None,
        metadata: dict | None = None,
    ) -> Goal:
        """
        Add a new pending goal to the tree.

        Returns the newly created Goal.
        """
        goal = Goal(
            description=description,
            priority=priority,
            parent_id=parent_id,
            metadata=metadata or {},
        )
        self._goals.append(goal)
        return goal

    def activate(self, goal_id: str) -> bool:
        """
        Mark a goal as active.  Any currently active goal is demoted back to
        pending (only one goal active at a time).

        Returns True if the goal was found and activated.
        """
        # Demote existing active goals
        for g in self._goals:
            if g.status == STATUS_ACTIVE:
                g.status = STATUS_PENDING

        target = self._by_id(goal_id)
        if target is None or target.is_terminal():
            return False
        target.status = STATUS_ACTIVE
        return True

    def resolve(self, goal_id: str, result: str = "") -> bool:
        """
        Mark a goal as succeeded.

        Returns True if found and updated.
        """
        g = self._by_id(goal_id)
        if g is None:
            return False
        g.status = STATUS_SUCCEEDED
        g.result = result
        g.resolved = datetime.now(timezone.utc).isoformat()
        return True

    def fail(self, goal_id: str, result: str = "") -> bool:
        """
        Mark a goal as failed.

        Returns True if found and updated.
        """
        g = self._by_id(goal_id)
        if g is None:
            return False
        g.status = STATUS_FAILED
        g.result = result
        g.resolved = datetime.now(timezone.utc).isoformat()
        return True

    def abandon(self, goal_id: str, reason: str = "") -> bool:
        """
        Mark a goal (and all its descendants) as abandoned.

        Returns True if the root goal was found and abandoned.
        """
        g = self._by_id(goal_id)
        if g is None:
            return False

        # Recursively abandon descendants first
        for child in self._children(goal_id):
            self.abandon(child.id, reason=reason)

        g.status = STATUS_ABANDONED
        g.result = reason
        g.resolved = datetime.now(timezone.utc).isoformat()
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def all_goals(self) -> list[Goal]:
        return list(self._goals)

    def active_goals(self) -> list[Goal]:
        return [g for g in self._goals if g.status == STATUS_ACTIVE]

    def pending_goals(self) -> list[Goal]:
        return sorted(
            [g for g in self._goals if g.status == STATUS_PENDING],
            key=lambda g: g.priority,
        )

    def top_goal(self) -> Optional[Goal]:
        """Return the single active goal, or the highest-priority pending goal."""
        active = self.active_goals()
        if active:
            return active[0]
        pending = self.pending_goals()
        return pending[0] if pending else None

    def open_goals(self) -> list[Goal]:
        """Active + pending goals, sorted by priority."""
        return sorted(
            [g for g in self._goals if g.status in (STATUS_ACTIVE, STATUS_PENDING)],
            key=lambda g: (g.priority, g.created),
        )

    def is_complete(self) -> bool:
        """True when all goals are in a terminal state (or no goals exist)."""
        return all(g.is_terminal() for g in self._goals)

    # ------------------------------------------------------------------
    # Prompt serialization
    # ------------------------------------------------------------------

    def format_for_prompt(
        self,
        include_terminal: bool = False,
        max_goals: int = 20,
    ) -> str:
        """
        Render the goal tree as an indented outline for LLM prompts.

        Parameters
        ----------
        include_terminal : bool
            Whether to include succeeded/failed/abandoned goals.
        max_goals : int
            Cap on total goals rendered (most recent first if over limit).
        """
        if not self._goals:
            return "Goals: (none)"

        # Collect goals to show
        if include_terminal:
            to_show = list(self._goals)
        else:
            to_show = [g for g in self._goals if not g.is_terminal()]

        if not to_show:
            return "Goals: (all resolved)"

        # Limit
        if len(to_show) > max_goals:
            to_show = to_show[-max_goals:]

        # Build indented outline
        lines = ["Goals:"]

        def _render(goals: list[Goal], parent_id: Optional[str], indent: int) -> None:
            children = [g for g in goals if g.parent_id == parent_id]
            children.sort(key=lambda g: (g.priority, g.created))
            prefix = "  " * indent
            marker_map = {
                STATUS_ACTIVE:    "[*]",
                STATUS_PENDING:   "[ ]",
                STATUS_SUCCEEDED: "[+]",
                STATUS_FAILED:    "[x]",
                STATUS_ABANDONED: "[-]",
            }
            for g in children:
                marker = marker_map.get(g.status, "[?]")
                result_str = f"  -> {g.result}" if g.result else ""
                lines.append(f"{prefix}{marker} ({g.id}) {g.description}{result_str}")
                _render(goals, g.id, indent + 1)

        # Find roots among the goals we're showing
        shown_ids = {g.id for g in to_show}
        roots = [g for g in to_show if g.parent_id is None or g.parent_id not in shown_ids]
        root_ids = {g.id for g in roots}

        # Render root-level then recurse
        for g in sorted(roots, key=lambda x: (x.priority, x.created)):
            marker_map = {
                STATUS_ACTIVE:    "[*]",
                STATUS_PENDING:   "[ ]",
                STATUS_SUCCEEDED: "[+]",
                STATUS_FAILED:    "[x]",
                STATUS_ABANDONED: "[-]",
            }
            marker = marker_map.get(g.status, "[?]")
            result_str = f"  -> {g.result}" if g.result else ""
            lines.append(f"  {marker} ({g.id}) {g.description}{result_str}")
            _render(to_show, g.id, indent=2)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Agent update parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_agent_updates(text: str) -> Optional[dict]:
        """
        Extract the first JSON block from `text` that contains
        `goal_updates` and/or `state_updates` keys.

        Returns the parsed dict, or None if no valid block is found.
        """
        # Search for ```json ... ``` or ``` ... ``` fenced blocks
        for raw in re.findall(
            r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE
        ):
            raw = raw.strip()
            if not raw.startswith("{"):
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and (
                "goal_updates" in obj or "state_updates" in obj
            ):
                return obj

        # Also try bare JSON objects (no fences) as a fallback
        for raw in re.findall(r"\{[^{}]*\"goal_updates\"[^{}]*\}", text, re.DOTALL):
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue

        return None

    def apply_updates(self, updates: dict) -> list[str]:
        """
        Apply a list of goal mutations from a parsed agent update dict.

        Expected shape:
            {"goal_updates": [
                {"action": "push",    "description": "...", "priority": 1, "parent_id": null},
                {"action": "resolve", "id": "g-3", "result": "..."},
                {"action": "fail",    "id": "g-4", "result": "..."},
                {"action": "abandon", "id": "g-2", "reason": "..."},
                {"action": "activate","id": "g-5"},
            ]}

        Returns a list of human-readable log strings describing what was done.
        """
        log: list[str] = []
        for item in updates.get("goal_updates", []):
            action = item.get("action", "")
            if action == "push":
                g = self.push(
                    description=item.get("description", "(no description)"),
                    priority=item.get("priority", 5),
                    parent_id=item.get("parent_id"),
                    metadata=item.get("metadata", {}),
                )
                log.append(f"pushed goal {g.id}: {g.description!r}")
            elif action == "activate":
                gid = item.get("id", "")
                ok = self.activate(gid)
                log.append(f"activate {gid}: {'ok' if ok else 'NOT FOUND'}")
            elif action == "resolve":
                gid = item.get("id", "")
                ok = self.resolve(gid, result=item.get("result", ""))
                log.append(f"resolve {gid}: {'ok' if ok else 'NOT FOUND'}")
            elif action == "fail":
                gid = item.get("id", "")
                ok = self.fail(gid, result=item.get("result", ""))
                log.append(f"fail {gid}: {'ok' if ok else 'NOT FOUND'}")
            elif action == "abandon":
                gid = item.get("id", "")
                ok = self.abandon(gid, reason=item.get("reason", ""))
                log.append(f"abandon {gid}: {'ok' if ok else 'NOT FOUND'}")
            else:
                log.append(f"unknown goal action: {action!r}")
        return log

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        counts = {s: 0 for s in (STATUS_PENDING, STATUS_ACTIVE, STATUS_SUCCEEDED,
                                  STATUS_FAILED, STATUS_ABANDONED)}
        for g in self._goals:
            counts[g.status] = counts.get(g.status, 0) + 1
        return (
            f"GoalManager(task_id={self.task_id!r}, "
            f"pending={counts[STATUS_PENDING]}, "
            f"active={counts[STATUS_ACTIVE]}, "
            f"succeeded={counts[STATUS_SUCCEEDED]}, "
            f"failed={counts[STATUS_FAILED]}, "
            f"abandoned={counts[STATUS_ABANDONED]})"
        )
