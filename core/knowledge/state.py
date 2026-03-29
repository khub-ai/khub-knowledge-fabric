"""
state.py — StateManager for the Knowledge Fabric framework.

Tracks the mutable, task-scoped environment state across agent rounds.
State is ephemeral (not persisted to disk); it lives for the duration of a
single task run and is discarded afterward.

Design principles:
  - Free-form `data: dict` so any use-case can store what it needs.
  - Append-only history so agents can see how the state evolved.
  - `rollback(n)` to undo the last n updates (useful when a revision fails).
  - `schema` hint string shown to agents so they understand the data shape.
  - `format_for_prompt()` produces a concise, human-readable summary suitable
    for injection into LLM prompts.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# History entry
# ---------------------------------------------------------------------------

@dataclass
class StateChange:
    """A single recorded state mutation."""
    description: str
    before: dict
    after: dict
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "before": self.before,
            "after": self.after,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# StateManager
# ---------------------------------------------------------------------------

class StateManager:
    """
    Task-scoped mutable environment state with history and rollback.

    Parameters
    ----------
    task_id : str
        Identifier of the current task (for logging and prompt injection).
    dataset_tag : str
        Dataset/use-case tag (e.g. "arc-agi-2", "arc-agi-3").
    schema : str
        Human-readable description of the expected shape of `data`.
        Shown to agents so they know how to interpret and update state.
    initial_data : dict | None
        Optional initial values for `data`.
    max_history : int
        Maximum number of history entries to keep (FIFO, oldest dropped first).
    """

    def __init__(
        self,
        task_id: str = "",
        dataset_tag: str = "",
        schema: str = "",
        initial_data: dict | None = None,
        max_history: int = 50,
    ) -> None:
        self.task_id = task_id
        self.dataset_tag = dataset_tag
        self.schema = schema
        self._data: dict = deepcopy(initial_data) if initial_data else {}
        self._history: list[StateChange] = []
        self._max_history = max_history

    # ------------------------------------------------------------------
    # Read access
    # ------------------------------------------------------------------

    @property
    def data(self) -> dict:
        """Read-only view of current state data (returns a shallow copy)."""
        return dict(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a single key from the current state."""
        return self._data.get(key, default)

    @property
    def history(self) -> list[StateChange]:
        """Ordered list of all recorded state changes (oldest first)."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Write access
    # ------------------------------------------------------------------

    def update(self, changes: dict, description: str = "") -> None:
        """
        Merge `changes` into the current state and record in history.

        Only the keys present in `changes` are modified; all other keys are
        left unchanged.
        """
        before = deepcopy(self._data)
        self._data.update(changes)
        after = deepcopy(self._data)
        entry = StateChange(
            description=description or f"update({list(changes.keys())})",
            before=before,
            after=after,
        )
        self._history.append(entry)
        if len(self._history) > self._max_history:
            self._history.pop(0)

    def replace(self, new_data: dict, description: str = "") -> None:
        """
        Replace the entire state with `new_data` and record in history.
        """
        before = deepcopy(self._data)
        self._data = deepcopy(new_data)
        after = deepcopy(self._data)
        entry = StateChange(
            description=description or "replace(all)",
            before=before,
            after=after,
        )
        self._history.append(entry)
        if len(self._history) > self._max_history:
            self._history.pop(0)

    def rollback(self, n: int = 1) -> int:
        """
        Undo the last `n` updates by restoring the `before` snapshot of each.

        Returns the number of rollbacks actually applied (may be less than `n`
        if history is shorter than `n`).
        """
        applied = 0
        for _ in range(min(n, len(self._history))):
            last = self._history.pop()
            self._data = deepcopy(last.before)
            applied += 1
        return applied

    # ------------------------------------------------------------------
    # Prompt serialization
    # ------------------------------------------------------------------

    def format_for_prompt(self, include_history: int = 3) -> str:
        """
        Return a concise string representation suitable for LLM prompts.

        Parameters
        ----------
        include_history : int
            How many recent history entries to include (0 = current state only).
        """
        lines: list[str] = []

        if self.schema:
            lines.append(f"[State schema: {self.schema}]")

        # Current state — pretty-print as key: value lines
        if self._data:
            lines.append("Current state:")
            for k, v in self._data.items():
                v_str = str(v)
                if len(v_str) > 200:
                    v_str = v_str[:200] + "..."
                lines.append(f"  {k}: {v_str}")
        else:
            lines.append("Current state: (empty)")

        # Recent history
        if include_history > 0 and self._history:
            recent = self._history[-include_history:]
            lines.append(f"Recent changes ({len(recent)} of {len(self._history)} total):")
            for ch in recent:
                lines.append(f"  - {ch.description}  [{ch.timestamp[:19]}]")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # JSON update parsing (from agent output)
    # ------------------------------------------------------------------

    def apply_agent_updates(self, updates: dict) -> None:
        """
        Apply a state-update dict emitted by an agent.

        Expected format (from `parse_agent_updates` in GoalManager or inline):
            {"description": "...", "set": {"key": value, ...}, "delete": ["key", ...]}

        All fields are optional.
        """
        description = updates.get("description", "agent state update")
        to_set: dict = updates.get("set", {})
        to_delete: list = updates.get("delete", [])

        before = deepcopy(self._data)
        if to_set:
            self._data.update(to_set)
        for key in to_delete:
            self._data.pop(key, None)
        after = deepcopy(self._data)

        entry = StateChange(description=description, before=before, after=after)
        self._history.append(entry)
        if len(self._history) > self._max_history:
            self._history.pop(0)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"StateManager(task_id={self.task_id!r}, "
            f"keys={list(self._data.keys())}, "
            f"history_len={len(self._history)})"
        )
