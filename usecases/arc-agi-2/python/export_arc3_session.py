import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional


def decode_frame(value: Any) -> Optional[List[List[int]]]:
    if not value:
        return None
    if isinstance(value, list) and value and isinstance(value[0], list):
        if value[0] and isinstance(value[0][0], list):
            return value[0]
        return value
    return None


def compute_diff(prev_frame: Optional[List[List[int]]], next_frame: Optional[List[List[int]]]) -> Dict[str, Any]:
    if not prev_frame or not next_frame:
        return {"kind": "grid-cell-diff", "cells": [], "components": [], "change_types": {}}

    cells: List[List[int]] = []
    change_types: Dict[str, int] = {}
    height = len(next_frame)
    width = len(next_frame[0])

    for y in range(height):
        for x in range(width):
            before = prev_frame[y][x]
            after = next_frame[y][x]
            if before != after:
                cells.append([x, y, before, after])
                key = f"{before}->{after}"
                change_types[key] = change_types.get(key, 0) + 1

    components = connected_components(cells)
    return {
        "kind": "grid-cell-diff",
        "cells": cells,
        "components": components,
        "change_types": change_types,
    }


def connected_components(cells: List[List[int]]) -> List[Dict[str, int]]:
    if not cells:
        return []

    points = {(x, y) for x, y, _, _ in cells}
    seen = set()
    components = []

    for start in list(points):
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        x_min = x_max = start[0]
        y_min = y_max = start[1]

        while queue:
            x, y = queue.popleft()
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (x + dx, y + dy)
                if nxt in points and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)

        components.append({"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max})

    components.sort(key=lambda item: (item["y_min"], item["x_min"]))
    return components


def load_playlog(playlog_dir: Path) -> List[Path]:
    files = sorted(playlog_dir.glob("[0-9][0-9][0-9]-*.json"))
    if not files:
        raise FileNotFoundError(f"No playlog step JSON files found in {playlog_dir}")
    return files


def make_goal(levels_completed: Optional[int], win_levels: Optional[int]) -> Dict[str, Any]:
    total = int(win_levels or 0)
    value = int(levels_completed or 0)
    status = "completed" if total and value >= total else "active"
    return {
        "id": "goal-complete-levels",
        "label": "Complete all required levels",
        "status": status,
        "description": "Primary objective for this ARC-AGI-3 game session.",
        "progress": {"value": value, "total": total},
        "tags": ["arc-agi-3", "objective"],
    }


def make_state_entities(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "id": f"state-status-{raw.get('observation_state', 'unknown')}",
            "label": f"Status: {raw.get('observation_state', 'unknown')}",
            "status": "active",
            "description": "Current environment status reported by the ARC runtime.",
            "tags": ["runtime-state"],
        },
        {
            "id": f"state-level-progress-{raw.get('levels_completed', 0)}",
            "label": "Level progress",
            "status": "active",
            "description": f"{raw.get('levels_completed', 0)} of {raw.get('win_levels', 0)} required levels completed.",
            "progress": {
                "value": int(raw.get('levels_completed', 0) or 0),
                "total": int(raw.get('win_levels', 0) or 0),
            },
            "tags": ["progress"],
        },
    ]


def export_session(playlog_dir: Path, output_path: Path) -> None:
    step_files = load_playlog(playlog_dir)
    raw_steps = [json.loads(path.read_text()) for path in step_files]

    first = raw_steps[0]
    game_id = first.get("returned", {}).get("game_id", playlog_dir.name)
    top_goal = make_goal(first.get("levels_completed"), first.get("win_levels"))

    session = {
        "schema_version": "0.2",
        "session_type": "arc-agi-3",
        "session_id": f"{game_id}-{playlog_dir.name}",
        "title": f"{game_id} exploratory play session",
        "renderer": "arc-grid",
        "metadata": {
            "game_id": game_id,
            "playlog_dir": str(playlog_dir),
        },
        "goals": [top_goal],
        "entities": {
            "rules": [],
            "tools": [],
            "states": [
                {
                    "id": "state-arc-interactive-session",
                    "label": "Interactive ARC session",
                    "status": "active",
                    "description": "This session uses an interactive ARC-AGI-3 environment rather than static grid prediction.",
                    "tags": ["session-type"],
                }
            ],
            "goals": [top_goal],
        },
        "steps": [],
    }

    prev_frame = None
    for raw in raw_steps:
        frame = decode_frame(raw.get("returned", {}).get("frame"))
        index = raw.get("step_number", 0)
        goal = make_goal(raw.get("levels_completed"), raw.get("win_levels"))
        step = {
            "index": index,
            "timestamp": None,
            "action": {
                "type": raw.get("action_name", "RESET"),
                "label": raw.get("action_name", "RESET"),
            },
            "commentary": {
                "note": raw.get("decision_note", ""),
            },
            "artifacts": {
                "primary": {
                    "kind": "grid-2d",
                    "data": frame,
                }
            },
            "transition": {
                "from_step": max(index - 1, 0),
                "diff": compute_diff(prev_frame, frame),
            },
            "state": {
                "status": raw.get("observation_state"),
                "levels_completed": raw.get("levels_completed"),
                "win_levels": raw.get("win_levels"),
            },
            "goals": [goal],
            "entities": {
                "rules": [],
                "tools": [],
                "states": make_state_entities(raw),
                "goals": [goal],
            },
            "metadata": {
                "source_file": f"{index:03d}-{raw.get('label', 'step')}.json",
                "guid": raw.get("returned", {}).get("guid"),
                "available_actions": raw.get("returned", {}).get("available_actions"),
            },
        }
        session["steps"].append(step)
        prev_frame = frame

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(session, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an ARC-AGI-3 playlog to KF session schema JSON.")
    parser.add_argument("--playlog-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_session(args.playlog_dir, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
