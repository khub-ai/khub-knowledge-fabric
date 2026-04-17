"""
distillation_recorder.py — Record OBSERVER/MEDIATOR I/O for model distillation.

Captures the input prompts and output responses from OBSERVER and MEDIATOR
calls, organized by game and level. This data will later be used to distill
smaller, specialized models using the dialogic distillation approach.

Directory structure:
  distillation_data/{game_id}/level_{N}/
    observer_{step:04d}.json    — OBSERVER frame + prompt + response
    mediator_{step:04d}.json    — MEDIATOR prompt + response + action plan
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional


_BASE_DIR = Path(__file__).parent / "distillation_data"


class DistillationRecorder:
    """Records OBSERVER and MEDIATOR I/O for a single episode."""

    def __init__(self, game_id: str, enabled: bool = True) -> None:
        self.game_id = game_id
        self.enabled = enabled
        self._current_level: int = 1
        self._step: int = 0
        self._observer_count: int = 0
        self._mediator_count: int = 0
        self._level_dir: Optional[Path] = None
        if enabled:
            self._ensure_level_dir()

    def _ensure_level_dir(self) -> Path:
        d = _BASE_DIR / self.game_id / f"level_{self._current_level}"
        d.mkdir(parents=True, exist_ok=True)
        self._level_dir = d
        return d

    def set_level(self, level: int) -> None:
        """Update the current level (1-indexed)."""
        if level != self._current_level:
            self._current_level = level
            self._observer_count = 0
            self._mediator_count = 0
            if self.enabled:
                self._ensure_level_dir()

    def set_step(self, step: int) -> None:
        """Update the current step counter."""
        self._step = step

    def record_observer(
        self,
        frame: list[list[int]],
        system_prompt: str,
        user_message: str,
        response: str,
        duration_ms: int,
        level_solved: bool = False,
    ) -> Optional[Path]:
        """Record an OBSERVER call."""
        if not self.enabled:
            return None

        self._observer_count += 1
        d = self._ensure_level_dir()
        path = d / f"observer_{self._observer_count:04d}.json"

        record = {
            "type": "observer",
            "game_id": self.game_id,
            "level": self._current_level,
            "step": self._step,
            "timestamp": time.time(),
            "level_solved": level_solved,
            "duration_ms": duration_ms,
            "frame": frame,
            "system_prompt": system_prompt,
            "user_message": user_message,
            "response": response,
        }

        path.write_text(json.dumps(record, indent=2, default=_json_safe),
                        encoding="utf-8")
        return path

    def record_mediator(
        self,
        system_prompt: str,
        user_message: str,
        response: str,
        action_plan: list[dict],
        duration_ms: int,
        level_solved: bool = False,
    ) -> Optional[Path]:
        """Record a MEDIATOR call."""
        if not self.enabled:
            return None

        self._mediator_count += 1
        d = self._ensure_level_dir()
        path = d / f"mediator_{self._mediator_count:04d}.json"

        record = {
            "type": "mediator",
            "game_id": self.game_id,
            "level": self._current_level,
            "step": self._step,
            "timestamp": time.time(),
            "level_solved": level_solved,
            "duration_ms": duration_ms,
            "system_prompt": system_prompt,
            "user_message": user_message,
            "response": response,
            "action_plan": action_plan,
        }

        path.write_text(json.dumps(record, indent=2, default=_json_safe),
                        encoding="utf-8")
        return path

    def mark_level_solved(self, level: int) -> None:
        """
        Mark all records for this level as belonging to a solved level.
        Updates the level_solved flag in all existing records.
        """
        d = _BASE_DIR / self.game_id / f"level_{level}"
        if not d.exists():
            return
        for f in d.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                data["level_solved"] = True
                f.write_text(json.dumps(data, indent=2, default=_json_safe),
                             encoding="utf-8")
            except Exception:
                pass

    def stats(self) -> dict:
        return {
            "game_id": self.game_id,
            "level": self._current_level,
            "observer_records": self._observer_count,
            "mediator_records": self._mediator_count,
        }


def _json_safe(v: Any) -> Any:
    if isinstance(v, set):
        return sorted(v)
    if isinstance(v, tuple):
        return list(v)
    return str(v)
