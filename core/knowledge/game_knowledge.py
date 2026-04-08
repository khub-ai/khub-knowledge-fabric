"""
game_knowledge.py — Per-game positional memory store.

Stores point facts learned from guided tours (knowledge_bootstrap.py) and
from in-episode observations:
  - Changer positions (rotation, color, shape changers) with nearby colors
  - Win target positions
  - Player state at win (rot_idx, color_idx)
  - Step budget and starting state per level

This is intentionally SEPARATE from rules.json.  Rules express generalizable
event-pattern principles; game_knowledge.json stores instance facts that are
specific to a (game_id, level) pair and would never fire as rule conditions
in other games.

MEDIATOR context injection
--------------------------
Call GameKnowledgeRegistry.context_for(game_id, level) to get a short
natural-language summary of known positions for the current game/level.
Inject this into the MEDIATOR system prompt so it can use positional hints
without the hints polluting the generalizable rule base.

Usage
-----
    registry = GameKnowledgeRegistry(path="game_knowledge.json")
    registry.record_level(
        game_id="ls20",
        level=1,
        rot_changers=[{"x": 19, "y": 30, "nearby_colors": [3, 5]}],
        color_changers=[],
        shape_changers=[],
        win_target={"x": 34, "y": 10},
        player_at_win={"rot_idx": 0, "color_idx": 2},
        step_budget=42,
        start_state={"rot_idx": 3, "color_idx": 2},
    )
    hint = registry.context_for("ls20", 1)
    # "Level 1: rotation changer at (19,30); win target at (34,10) ..."
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class GameKnowledgeRegistry:
    """
    JSON-backed store for per-game, per-level positional facts.

    The on-disk format is::

        {
          "<game_id>": {
            "<level_number>": {
              "rot_changers":   [{"x": int, "y": int, "nearby_colors": [...]}],
              "color_changers": [...],
              "shape_changers": [...],
              "win_target":     {"x": int, "y": int} | null,
              "player_at_win":  {"rot_idx": int, "color_idx": int} | null,
              "step_budget":    int | null,
              "start_state":    {"rot_idx": int, "color_idx": int} | null
            }
          }
        }

    Level numbers are stored as strings (JSON keys must be strings).
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        # {game_id: {level_str: level_dict}}
        self._data: dict[str, dict[str, dict]] = {}
        if self.path and self.path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record_level(
        self,
        game_id: str,
        level: int,
        *,
        rot_changers:   list[dict] | None = None,
        color_changers: list[dict] | None = None,
        shape_changers: list[dict] | None = None,
        win_target:     dict | None = None,
        player_at_win:  dict | None = None,
        step_budget:    int  | None = None,
        start_state:    dict | None = None,
    ) -> None:
        """Record (or update) positional facts for one game/level pair.

        Existing entries are MERGED: passing rot_changers=[...] replaces
        only that list; other fields already on disk are left unchanged.
        Pass an empty list [] to explicitly clear a list.
        """
        level_str = str(level)
        game_data = self._data.setdefault(game_id, {})
        entry     = game_data.setdefault(level_str, {})

        if rot_changers   is not None: entry["rot_changers"]   = rot_changers
        if color_changers is not None: entry["color_changers"] = color_changers
        if shape_changers is not None: entry["shape_changers"] = shape_changers
        if win_target     is not None: entry["win_target"]     = win_target
        if player_at_win  is not None: entry["player_at_win"]  = player_at_win
        if step_budget    is not None: entry["step_budget"]    = step_budget
        if start_state    is not None: entry["start_state"]    = start_state

        if self.path:
            self._save()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_level(self, game_id: str, level: int) -> dict | None:
        """Return the stored fact dict for (game_id, level), or None."""
        return self._data.get(game_id, {}).get(str(level))

    def context_for(self, game_id: str, level: int) -> str:
        """Return a human-readable summary for MEDIATOR context injection.

        Returns an empty string if nothing is known for this game/level.
        """
        entry = self.get_level(game_id, level)
        if not entry:
            return ""

        lines = [f"Positional memory for {game_id} level {level}:"]

        start = entry.get("start_state")
        if start:
            lines.append(
                f"  Player starts at rot_idx={start.get('rot_idx','?')}, "
                f"color_idx={start.get('color_idx','?')}."
            )

        budget = entry.get("step_budget")
        if budget:
            lines.append(f"  Step budget: {budget}.")

        for key, label in [
            ("rot_changers",   "Rotation changer"),
            ("color_changers", "Color changer"),
            ("shape_changers", "Shape changer"),
        ]:
            for pos in entry.get(key, []):
                nearby = pos.get("nearby_colors", [])
                color_note = (
                    f" (nearby colors: {nearby})" if nearby else ""
                )
                lines.append(
                    f"  {label} last seen at game coord "
                    f"({pos['x']},{pos['y']}){color_note}."
                )

        wt = entry.get("win_target")
        paw = entry.get("player_at_win")
        if wt:
            state_note = ""
            if paw:
                state_note = (
                    f" — player must have rot_idx={paw.get('rot_idx','?')}, "
                    f"color_idx={paw.get('color_idx','?')}"
                )
            lines.append(
                f"  Win target last seen at game coord "
                f"({wt['x']},{wt['y']}){state_note}."
            )

        return "\n".join(lines)

    def all_games(self) -> list[str]:
        return list(self._data.keys())

    def all_levels(self, game_id: str) -> list[int]:
        return sorted(int(k) for k in self._data.get(game_id, {}).keys())

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
