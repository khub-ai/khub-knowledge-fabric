"""
knowledge.py — Persistent knowledge base for ARC-AGI ensemble.

Stores generalizable patterns, failure modes, and human insights discovered
across puzzle runs. The MEDIATOR reads and writes this base so each new task
benefits from prior experience.

Schema of knowledge.json:
{
  "version": 1,
  "patterns": [
    {
      "id": "p001",
      "name": "column-gravity",
      "description": "Non-background cells fall to the bottom of their column.",
      "trigger_cues": ["cells floating above empty space", "consistent downward shift"],
      "confirmed_tasks": ["1e0a9b12"],
      "added": "2026-03-21T17:30:00",
      "source": "MEDIATOR"
    }
  ],
  "failure_modes": [
    {
      "id": "f001",
      "description": "Mistook row-gravity for column-gravity due to symmetric test input.",
      "tasks": ["xxxx"],
      "lesson": "Always verify the axis by checking BOTH rows and columns in demos.",
      "added": "2026-03-21T18:00:00"
    }
  ],
  "human_insights": [
    {
      "id": "h001",
      "content": "In stamp tasks, the stamp is always the smaller object.",
      "tasks": [],
      "added": "2026-03-21T19:00:00",
      "source": "human"
    }
  ]
}
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Default location — override via KNOWLEDGE_FILE env var or constructor arg
DEFAULT_PATH = Path(__file__).parent / "knowledge.json"


class KnowledgeBase:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.environ.get("KNOWLEDGE_FILE", DEFAULT_PATH))
        self._data: dict[str, Any] = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"version": 1, "patterns": [], "failure_modes": [], "human_insights": []}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def reload(self) -> None:
        """Reload from disk (e.g. after human edits)."""
        self._data = self._load()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @property
    def patterns(self) -> list[dict]:
        return self._data["patterns"]

    @property
    def failure_modes(self) -> list[dict]:
        return self._data["failure_modes"]

    @property
    def human_insights(self) -> list[dict]:
        return self._data["human_insights"]

    def format_for_prompt(self, max_patterns: int = 10, max_failures: int = 5,
                           max_insights: int = 10) -> str:
        """Return a compact string suitable for injecting into an agent prompt."""
        parts: list[str] = []

        if self.patterns:
            parts.append("## Known Patterns")
            for p in self.patterns[-max_patterns:]:
                parts.append(f"- **{p['name']}**: {p['description']}")
                if p.get("trigger_cues"):
                    parts.append(f"  Cues: {', '.join(p['trigger_cues'])}")

        if self.failure_modes:
            parts.append("\n## Known Failure Modes")
            for f in self.failure_modes[-max_failures:]:
                parts.append(f"- {f['description']}")
                if f.get("lesson"):
                    parts.append(f"  Lesson: {f['lesson']}")

        if self.human_insights:
            parts.append("\n## Human Insights")
            for h in self.human_insights[-max_insights:]:
                parts.append(f"- {h['content']}")

        if not parts:
            return "(no prior knowledge)"

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def _next_id(self, prefix: str, collection: list[dict]) -> str:
        if not collection:
            return f"{prefix}001"
        last = collection[-1].get("id", f"{prefix}000")
        try:
            num = int(last[len(prefix):]) + 1
        except ValueError:
            num = len(collection) + 1
        return f"{prefix}{num:03d}"

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    def add_pattern(self, name: str, description: str,
                    trigger_cues: list[str] | None = None,
                    confirmed_tasks: list[str] | None = None,
                    source: str = "MEDIATOR") -> str:
        pid = self._next_id("p", self.patterns)
        self.patterns.append({
            "id": pid,
            "name": name,
            "description": description,
            "trigger_cues": trigger_cues or [],
            "confirmed_tasks": confirmed_tasks or [],
            "added": self._now_iso(),
            "source": source,
        })
        self.save()
        return pid

    def add_failure_mode(self, description: str, lesson: str,
                          tasks: list[str] | None = None) -> str:
        fid = self._next_id("f", self.failure_modes)
        self.failure_modes.append({
            "id": fid,
            "description": description,
            "lesson": lesson,
            "tasks": tasks or [],
            "added": self._now_iso(),
        })
        self.save()
        return fid

    def add_human_insight(self, content: str, tasks: list[str] | None = None) -> str:
        hid = self._next_id("h", self.human_insights)
        self.human_insights.append({
            "id": hid,
            "content": content,
            "tasks": tasks or [],
            "added": self._now_iso(),
            "source": "human",
        })
        self.save()
        return hid

    def confirm_pattern_on_task(self, pattern_name: str, task_id: str) -> bool:
        """Add task_id to a pattern's confirmed_tasks list."""
        for p in self.patterns:
            if p["name"] == pattern_name:
                if task_id not in p.get("confirmed_tasks", []):
                    p.setdefault("confirmed_tasks", []).append(task_id)
                self.save()
                return True
        return False

    def parse_mediator_update(self, mediator_text: str, task_id: str) -> dict[str, int]:
        """
        Parse a fenced JSON block from MEDIATOR output and apply it to the KB.

        Expected format inside ```json ... ```:
        {
          "new_patterns": [{"name": "...", "description": "...", "trigger_cues": [...]}],
          "confirmed_patterns": ["column-gravity"],
          "new_failure_modes": [{"description": "...", "lesson": "..."}],
          "new_insights": []
        }

        Returns counts of items added.
        """
        import re
        block_re = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
        matches = block_re.findall(mediator_text)
        counts = {"patterns": 0, "failure_modes": 0, "insights": 0}
        for raw in matches:
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            # Only process blocks that look like KB updates
            if not any(k in obj for k in ("new_patterns", "confirmed_patterns",
                                           "new_failure_modes", "new_insights")):
                continue
            for p in obj.get("new_patterns", []):
                self.add_pattern(
                    name=p["name"],
                    description=p["description"],
                    trigger_cues=p.get("trigger_cues", []),
                    confirmed_tasks=[task_id],
                )
                counts["patterns"] += 1
            for name in obj.get("confirmed_patterns", []):
                self.confirm_pattern_on_task(name, task_id)
            for f in obj.get("new_failure_modes", []):
                self.add_failure_mode(
                    description=f["description"],
                    lesson=f.get("lesson", ""),
                    tasks=[task_id],
                )
                counts["failure_modes"] += 1
            for h in obj.get("new_insights", []):
                self.add_human_insight(
                    content=h if isinstance(h, str) else h.get("content", str(h)),
                    tasks=[task_id],
                )
                counts["insights"] += 1
        return counts

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        return {
            "patterns": len(self.patterns),
            "failure_modes": len(self.failure_modes),
            "human_insights": len(self.human_insights),
        }
