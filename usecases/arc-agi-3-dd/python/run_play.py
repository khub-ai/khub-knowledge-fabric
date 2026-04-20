"""Drive TUTOR through an ls20 play session using high-level commands.

TUTOR issues ONE command per call.  The harness executes it autonomously
(BFS navigation, probing, etc.) and reports back a COMMAND_RESULT.
TUTOR is NOT called on every individual game step -- only once per command.

Commands:
  PROBE_DIRECTIONS -- execute each action once, report (dr,dc) per action
  MOVE_TO          -- BFS-navigate avatar to (row, col)
  STAMP_AT         -- MOVE_TO + fire one action at destination
  RAW_ACTION       -- single low-level action
  RESET            -- reset level (costs a life, refills budget)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

import backends
from play_prompts import (
    SYSTEM_PLAY, build_play_user_message,
    SYSTEM_POSTGAME, build_postgame_user_message,
)
from dsl_executor import _build_change_report, _normalise_frame
from navigator import bfs_navigate, nearest_reachable
from preview_html import render_play_session, grid_to_png_b64

TRAINING_DATA_DIR = HERE.parents[2] / ".tmp" / "training_data"
TUTOR_MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict:
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    first = text.find("{")
    if first < 0:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    depth = 0
    for i in range(first, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[first:i + 1])
    raise ValueError("unterminated JSON in reply")


def _format_frame_text(grid: np.ndarray) -> str:
    rows = [", ".join(f"{int(v):2d}" for v in row) for row in grid]
    return "[\n" + ",\n".join(f"  [{r}]" for r in rows) + "\n]"


def _bbox_centre_int(bbox) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return (round((r0 + r1) / 2), round((c0 + c1) / 2))


def _read_budget(cr: dict) -> int | None:
    """Extract current budget fill from counter_changes in CHANGE_REPORT."""
    for c in (cr.get("counter_changes") or []):
        name = (c.get("name") or "").lower()
        if "progress" in name or "budget" in name or "bar" in name:
            val = c.get("after_fill")
            if val is not None:
                return int(val)
    return None


def _update_action_effects(
    action_effects: dict[str, tuple[int, int]],
    action: str,
    cr: dict,
) -> None:
    """Update action_effects from a reliable CHANGE_REPORT primary_motion."""
    pm = cr.get("primary_motion")
    if not pm or pm.get("tracker_unreliable") or not pm.get("moved"):
        return
    dr = pm.get("dr", 0)
    dc = pm.get("dc", 0)
    if dr == 0 and dc == 0:
        return
    if action not in action_effects or action_effects[action] == (dr, dc):
        action_effects[action] = (dr, dc)


def _update_cursor_pos(
    cursor_pos: tuple[int, int] | None,
    actual_dr: int | None,
    actual_dc: int | None,
    cr: dict,
) -> tuple[int, int] | None:
    """Advance cursor_pos using the OBSERVED (dr, dc) from this step.

    Uses actual_dr/dc from the step's motion report — not the stored
    action_effects — so blocked moves (dr=0,dc=0) don't drift the position.
    Falls back to primary_motion post_bbox only for initial bootstrap.
    """
    if cursor_pos is not None and actual_dr is not None and actual_dc is not None:
        r = max(0, min(63, cursor_pos[0] + actual_dr))
        c = max(0, min(63, cursor_pos[1] + actual_dc))
        return (r, c)
    # Bootstrap: use primary_motion only when we have no observed dr/dc
    pm = cr.get("primary_motion")
    if pm and not pm.get("tracker_unreliable") and pm.get("moved"):
        post = pm.get("post_bbox")
        if post:
            return _bbox_centre_int(post)
    return cursor_pos


# ---------------------------------------------------------------------------
# Working knowledge loader
# ---------------------------------------------------------------------------

def load_working_knowledge(
    round2_dir: Path, lessons_path: Path | None = None,
) -> tuple[str, dict, tuple[int, int] | None]:
    """Returns (working_knowledge_text, element_records, initial_cursor_pos)."""
    r2 = json.loads((round2_dir / "tutor_round2_reply.json").read_text(encoding="utf-8"))
    assess = r2.get("assessment") or {}

    elements = assess.get("elements") or []
    element_records: dict[int, dict] = {}
    agent_bbox = None
    for e in elements:
        eid = e.get("id")
        if eid is None:
            continue
        bbox = e.get("bbox")
        element_records[int(eid)] = {
            "bbox":     bbox,
            "name":     e.get("name"),
            "function": e.get("function", "unknown"),
        }
        fn = (e.get("function") or "").lower()
        name = (e.get("name") or "").lower()
        if "agent" in fn or "cursor" in fn or "agent" in name or "cursor" in name:
            if bbox:
                agent_bbox = bbox

    initial_cursor_pos: tuple[int, int] | None = None
    if agent_bbox:
        initial_cursor_pos = _bbox_centre_int(agent_bbox)

    lines: list[str] = []
    if lessons_path is not None and lessons_path.exists():
        lines += [
            "LESSONS_FROM_LAST_RUN (YOU wrote this; takes precedence over everything below):",
            lessons_path.read_text(encoding="utf-8").strip(),
            "", "---", "",
        ]
    lines.append("ELEMENTS (from your Round-2 revised assessment):")
    for e in elements:
        lines.append(
            f"  #{e.get('id')} {e.get('name','?')} "
            f"bbox={e.get('bbox')} fn={e.get('function','?')} "
            f"-- {e.get('rationale','')}"
        )
    strat = assess.get("initial_strategy") or {}
    lines += ["", f"PRIMARY_GOAL: {strat.get('primary_goal','?')}"]
    if strat.get("rationale"):
        lines.append(f"STRATEGY_NOTES: {strat['rationale']}")
    qs = strat.get("open_questions") or []
    if qs:
        lines.append("OPEN_QUESTIONS:")
        lines += [f"  - {q}" for q in qs]
    prior_path = round2_dir / "prior_knowledge.txt"
    if prior_path.exists():
        lines += ["", "PRIOR_KNOWLEDGE:", prior_path.read_text(encoding="utf-8")]

    return "\n".join(lines), element_records, initial_cursor_pos


# ---------------------------------------------------------------------------
# Element-space helpers
# ---------------------------------------------------------------------------

def _detect_element_overlaps(
    cursor_pos: tuple[int, int] | None,
    element_records: dict,
) -> list[dict]:
    """Return elements whose bbox contains cursor_pos."""
    if not cursor_pos:
        return []
    r, c = cursor_pos
    hits = []
    for eid, rec in element_records.items():
        bbox = rec.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        r0, c0, r1, c1 = bbox
        if r0 <= r <= r1 and c0 <= c <= c1:
            hits.append({
                "id": eid,
                "name": rec.get("name", "?"),
                "function": rec.get("function", "?"),
                "bbox": bbox,
            })
    return hits


def _harness_coordinate_note(
    command: str,
    target_pos: tuple[int, int] | None,
    element_records: dict,
) -> str:
    """Return a correction string when target_pos is empty space.

    If the target falls inside a known element's bbox → empty string (no issue).
    Otherwise the harness names the nearest element and its actual coordinates.
    """
    if command not in ("MOVE_TO", "STAMP_AT") or not target_pos:
        return ""
    tr, tc = target_pos
    for rec in element_records.values():
        bbox = rec.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        r0, c0, r1, c1 = bbox
        if r0 <= tr <= r1 and c0 <= tc <= c1:
            fn = (rec.get("function") or "").lower()
            if "hazard" in fn or "trap" in fn or "death" in fn:
                return (
                    f"HARNESS WARNING: target {list(target_pos)} is inside "
                    f"'{rec.get('name','?')}' (function={rec.get('function','?')}). "
                    f"This element may reset the level or cost a life."
                )
            return ""   # target IS inside a known (non-hazard) element
    nearby = _find_nearby_elements(target_pos, element_records, radius=20)
    if not nearby:
        return f"HARNESS: target {list(target_pos)} is not inside any known element (no nearby element found within radius 20)."
    n = nearby[0]
    cr, cc = n["center"]
    if abs(cr - tr) <= 1 and abs(cc - tc) <= 1:
        return ""   # off by 1 — not worth flagging
    return (
        f"HARNESS CORRECTION: target {list(target_pos)} is empty space "
        f"(not inside any known element). "
        f"Nearest element: '{n['name']}' (fn={n['function']}, id={n['id']}) "
        f"actual center={n['center']}, bbox={n['bbox']}. "
        f"Please update WORKING_KNOWLEDGE with the corrected coordinates."
    )


def _find_nearby_elements(
    target_pos: tuple[int, int],
    element_records: dict,
    radius: int = 8,
) -> list[dict]:
    """Return elements within Manhattan radius of target_pos, sorted by distance."""
    tr, tc = target_pos
    nearby = []
    for eid, rec in element_records.items():
        bbox = rec.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        r0, c0, r1, c1 = bbox
        cr = round((r0 + r1) / 2)
        cc = round((c0 + c1) / 2)
        dist = abs(cr - tr) + abs(cc - tc)
        if dist <= radius:
            nearby.append({
                "id": eid,
                "name": rec.get("name", "?"),
                "function": rec.get("function", "?"),
                "bbox": bbox,
                "center": [cr, cc],
                "dist_manhattan": dist,
            })
    nearby.sort(key=lambda x: x["dist_manhattan"])
    return nearby


# ---------------------------------------------------------------------------
# Command executors
# ---------------------------------------------------------------------------

def _step_env(env, action_label: str):
    from arcengine import GameAction
    if action_label == "RESET":
        return env.reset()
    return env.step(GameAction[action_label])


def exec_raw_action(
    env, action: str, prev_grid: np.ndarray, element_records: dict,
    action_effects: dict, cursor_pos,
):
    """Execute one low-level action; return (obs, new_grid, cr, step_log_entry)."""
    obs = _step_env(env, action)
    cur_grid = _normalise_frame(obs.frame)
    cr = _build_change_report(prev_grid, cur_grid, element_records)
    _update_action_effects(action_effects, action, cr)
    pm = cr.get("primary_motion") or {}
    if not pm.get("tracker_unreliable"):
        actual_dr: int | None = pm.get("dr", 0)
        actual_dc: int | None = pm.get("dc", 0)
    else:
        actual_dr = None
        actual_dc = None
    new_cursor = _update_cursor_pos(cursor_pos, actual_dr, actual_dc, cr)
    entry = {
        "action": action,
        "dr": (cr.get("primary_motion") or {}).get("dr"),
        "dc": (cr.get("primary_motion") or {}).get("dc"),
        "reliable": not (cr.get("primary_motion") or {}).get("tracker_unreliable", True),
        "diff_cells": (cr.get("totals") or {}).get("diff_cells"),
    }
    return obs, cur_grid, cr, new_cursor, entry


def exec_probe_directions(
    env, available_actions: list[str],
    prev_grid: np.ndarray, element_records: dict,
    action_effects: dict, cursor_pos,
):
    """Execute each action once; return bundled result."""
    motion_log = []
    cur_grid = prev_grid
    obs = None
    for action in available_actions:
        obs, cur_grid, cr, cursor_pos, entry = exec_raw_action(
            env, action, cur_grid, element_records, action_effects, cursor_pos,
        )
        motion_log.append(entry)

    return obs, cur_grid, motion_log, cursor_pos


def exec_move_to(
    env, target_pos: tuple[int, int],
    prev_grid: np.ndarray, element_records: dict,
    action_effects: dict, cursor_pos,
    budget_remaining: int,
    walls: set | None = None,
    stamp_action: str | None = None,
):
    """BFS-navigate to target_pos, optionally fire stamp_action there.

    Detects wall-blocked moves during execution: records the wall, aborts
    the path early, and returns an error describing the actual vs intended
    position so TUTOR can self-correct its spatial model.
    """
    if not cursor_pos:
        return None, prev_grid, [], cursor_pos, "cursor_pos unknown", {}

    path = bfs_navigate(cursor_pos, target_pos, action_effects, walls=walls)
    if path is None:
        result = nearest_reachable(cursor_pos, target_pos, action_effects, walls=walls)
        if result is None:
            return None, prev_grid, [], cursor_pos, "unreachable and no movement actions known", {}
        actual_target, path = result

    if stamp_action:
        path = path + [stamp_action]

    if len(path) > budget_remaining - 2:
        return None, prev_grid, [], cursor_pos, (
            f"path length {len(path)} exceeds remaining budget {budget_remaining}"
        ), {}

    motion_log = []
    cur_grid = prev_grid
    obs = None
    exec_error: str | None = None
    walls_hit: list[dict] = []

    for action in path:
        pos_before = cursor_pos
        obs, cur_grid, cr, cursor_pos, entry = exec_raw_action(
            env, action, cur_grid, element_records, action_effects, cursor_pos,
        )
        motion_log.append(entry)

        # Wall detection: planned move expected non-zero displacement but got zero
        if action != stamp_action:
            planned_dr, planned_dc = action_effects.get(action, (0, 0))
            actual_dr = entry.get("dr") or 0
            actual_dc = entry.get("dc") or 0
            if (planned_dr != 0 or planned_dc != 0) and actual_dr == 0 and actual_dc == 0:
                wall_key = (pos_before[0], pos_before[1], action) if pos_before else None
                if wall_key and walls is not None:
                    walls.add(wall_key)
                wall_info = {
                    "action": action,
                    "blocked_at": list(pos_before) if pos_before else None,
                    "expected_dr": planned_dr,
                    "expected_dc": planned_dc,
                }
                walls_hit.append(wall_info)
                exec_error = (
                    f"wall: {action} blocked at {pos_before} "
                    f"(expected dr={planned_dr},dc={planned_dc}, got 0,0). "
                    f"Path aborted. Agent at {cursor_pos}, target was {target_pos}."
                )
                break

        state = obs.state.name if hasattr(obs.state, "name") else str(obs.state)
        if state in ("WIN", "GAME_OVER"):
            break

    # Report target-not-reached even when no wall explicitly blocked (e.g. wrong coords)
    if exec_error is None and cursor_pos != target_pos and stamp_action is None:
        exec_error = (
            f"target not reached: requested {list(target_pos)}, "
            f"agent ended at {list(cursor_pos) if cursor_pos else None}. "
            f"Check nearby_elements in COMMAND_RESULT for actual element positions."
        )

    target_analysis = {
        "requested_pos":   list(target_pos),
        "actual_pos":      list(cursor_pos) if cursor_pos else None,
        "reached":         cursor_pos == target_pos,
        "walls_hit":       walls_hit,
        "nearby_elements": _find_nearby_elements(target_pos, element_records),
    }

    return obs, cur_grid, motion_log, cursor_pos, exec_error, target_analysis


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def _dump_training_data(
    *, game_id: str, trial_id: str, system_prompt: str, session_dir: Path,
) -> None:
    out_dir = TRAINING_DATA_DIR / game_id / trial_id
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = session_dir / "play_log.jsonl"
    if not log_path.exists():
        return

    entries = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    turn_entries = [e for e in entries if "command" in e]
    wk_text = (session_dir / "working_knowledge.md").read_text(encoding="utf-8") if (session_dir / "working_knowledge.md").exists() else ""
    total_cost = sum(e.get("cost_usd", 0) for e in turn_entries)

    (out_dir / "metadata.json").write_text(json.dumps({
        "game_id": game_id, "trial_id": trial_id, "outcome": "WIN",
        "turns": len(turn_entries), "total_cost_usd": round(total_cost, 6),
        "session_dir": str(session_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")

    for e in turn_entries:
        turn = e.get("turn", 0)
        record = {
            "turn": turn,
            "system": system_prompt,
            "user": f"PLAY TURN {turn}\nWORKING_KNOWLEDGE:\n{wk_text}\n\nLAST_COMMAND_RESULT:\n{json.dumps(e.get('command_result') or {}, indent=2)}",
            "assistant": json.dumps({
                "command": e.get("command"),
                "args": e.get("args"),
                "rationale": e.get("rationale"),
                "predict": e.get("predict"),
                "revise_knowledge": e.get("revise_knowledge"),
                "done": e.get("done"),
            }),
            "frame_b64": e.get("frame_b64", ""),
            "metadata": {k: e.get(k) for k in
                ("state", "levels_completed", "cost_usd", "latency_ms",
                 "input_tokens", "output_tokens", "turn_start_iso")},
        }
        (out_dir / f"turn_{turn:03d}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8",
        )
    print(f"Training data ({len(turn_entries)} turns) -> {out_dir}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round2-session", required=True)
    ap.add_argument("--lessons",
                    help="Prior post_game_knowledge.md to inject as LESSONS_FROM_LAST_RUN")
    ap.add_argument("--game",         default="ls20-9607627b")
    ap.add_argument("--max-turns",    type=int, default=20,
                    help="Max TUTOR calls (not game steps)")
    ap.add_argument("--sessions-dir", default=str(HERE.parent / "benchmarks" / "sessions"))
    ap.add_argument("--frames-dir",   default=str(HERE.parent / "benchmarks" / "frames"))
    ap.add_argument("--max-tokens",   type=int, default=1500)
    a = ap.parse_args()

    ARC_REPO = Path(os.environ.get("ARC_AGI_3_REPO", r"C:\_backup\github\arc-agi-3"))
    sys.path.insert(0, str(ARC_REPO))
    from arc_agi import Arcade, OperationMode

    round2_dir  = Path(a.round2_session)
    lessons_path = Path(a.lessons) if a.lessons else None
    working_knowledge, element_records, cursor_pos = load_working_knowledge(
        round2_dir, lessons_path=lessons_path,
    )
    frames_dir = Path(a.frames_dir)

    trial_id    = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_dir = Path(a.sessions_dir) / f"trial_{trial_id}_play"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "working_knowledge.md").write_text(working_knowledge, encoding="utf-8")

    arc = Arcade(operation_mode=OperationMode.OFFLINE,
                 environments_dir=str(ARC_REPO / "environment_files"))
    env = arc.make(a.game)
    obs = env.reset()

    prev_grid       = _normalise_frame(obs.frame)
    action_effects: dict[str, tuple[int, int]] = {}
    walls: set[tuple[int, int, str]] = set()   # (row, col, action) → blocked
    budget_remaining = 84  # will be updated from counter_changes

    # Pre-populate action_effects and walls from prior session's manifest
    if lessons_path is not None:
        prior_manifest = lessons_path.parent / "manifest.json"
        if prior_manifest.exists():
            try:
                pm_data = json.loads(prior_manifest.read_text(encoding="utf-8"))
                for a_name, a_effect in (pm_data.get("action_effects_learned") or {}).items():
                    action_effects[a_name] = (int(a_effect[0]), int(a_effect[1]))
                for w in (pm_data.get("walls_learned") or []):
                    walls.add((int(w[0]), int(w[1]), str(w[2])))
                if action_effects:
                    print(f"Pre-loaded action_effects: {action_effects}")
                if walls:
                    print(f"Pre-loaded walls ({len(walls)}): {sorted(walls)}")
            except Exception as e:  # noqa: BLE001
                print(f"Could not load prior action_effects/walls: {e}")

    log_path = session_dir / "play_log.jsonl"
    log_fh   = log_path.open("w", encoding="utf-8")

    command_trace: list[dict] = []
    recent_history: list[dict] = []
    command_result: dict | None = None
    final_state = obs.state.name if hasattr(obs.state, "name") else str(obs.state)
    t0 = time.time()

    for turn in range(1, a.max_turns + 1):
        cur_grid = _normalise_frame(obs.frame)
        state    = obs.state.name if hasattr(obs.state, "name") else str(obs.state)
        avail    = [f"ACTION{int(x)}" for x in obs.available_actions]
        lc       = int(obs.levels_completed)

        frame_b64 = grid_to_png_b64(cur_grid)
        user_msg  = build_play_user_message(
            turn              = turn,
            game_id           = a.game,
            state             = state,
            levels_completed  = lc,
            win_levels        = int(obs.win_levels),
            budget_remaining  = budget_remaining,
            cursor_pos        = cursor_pos,
            action_effects    = action_effects,
            working_knowledge = working_knowledge,
            recent_history    = recent_history,
            command_result    = command_result,
            frame_text        = _format_frame_text(cur_grid),
        )

        turn_start = datetime.now(timezone.utc)
        try:
            rsp = backends.call_anthropic(
                model=TUTOR_MODEL, system=SYSTEM_PLAY, user=user_msg,
                image_b64=None, max_tokens=a.max_tokens,
            )
            reply_text    = rsp["reply"]
            latency_ms    = rsp["latency_ms"]
            input_tokens  = rsp.get("input_tokens", 0)
            output_tokens = rsp.get("output_tokens", 0)
            cost_usd      = rsp.get("cost_usd", 0.0)
        except Exception as e:  # noqa: BLE001
            print(f"turn {turn}: TUTOR call failed: {e}")
            break

        try:
            decision = extract_json(reply_text)
        except Exception as e:  # noqa: BLE001
            print(f"turn {turn}: JSON parse error: {e}\n{reply_text[:300]}")
            break

        command  = decision.get("command", "").upper().strip()
        args     = decision.get("args") or {}
        rationale = decision.get("rationale", "")
        predict  = decision.get("predict", {})
        revise   = decision.get("revise_knowledge", "")
        done_flag = bool(decision.get("done"))

        print(f"turn {turn:>2} state={state:<12} cmd={command:<20} "
              f"({latency_ms} ms, ${cost_usd:.4f})")
        if rationale:
            print(f"         {rationale[:100]}")
        if revise:
            print(f"         REVISE: {revise[:100]}")

        # ---- Execute command ------------------------------------------------
        motion_log: list[dict] = []
        exec_error: str | None = None
        target_analysis: dict = {}

        if command == "PROBE_DIRECTIONS":
            obs, cur_grid, motion_log, cursor_pos = exec_probe_directions(
                env, avail, cur_grid, element_records, action_effects, cursor_pos,
            )

        elif command in ("MOVE_TO", "STAMP_AT"):
            raw_target = args.get("target_pos")
            if not raw_target or len(raw_target) != 2:
                exec_error = "target_pos missing or invalid"
            else:
                target = (int(raw_target[0]), int(raw_target[1]))
                stamp  = args.get("action") if command == "STAMP_AT" else None
                cr_before = _build_change_report(prev_grid, cur_grid, {})
                b_before = _read_budget(cr_before)
                if b_before is not None:
                    budget_remaining = b_before
                obs, cur_grid, motion_log, cursor_pos, exec_error, target_analysis = exec_move_to(
                    env, target, cur_grid, element_records, action_effects,
                    cursor_pos, budget_remaining, walls=walls, stamp_action=stamp,
                )

        elif command == "RAW_ACTION":
            raw_act = str(args.get("action", "")).upper().strip()
            if not raw_act:
                exec_error = "action missing"
            else:
                obs, cur_grid, cr, cursor_pos, entry = exec_raw_action(
                    env, raw_act, cur_grid, element_records, action_effects, cursor_pos,
                )
                motion_log = [entry]

        elif command == "RESET":
            obs = env.reset()
            cur_grid = _normalise_frame(obs.frame)
            motion_log = [{"action": "RESET", "dr": None, "dc": None}]
            cursor_pos = None  # reset position unknown

        else:
            exec_error = f"unknown command: {command!r}"

        # ---- Build COMMAND_RESULT for next turn ----------------------------
        final_cr = {}
        if obs is not None:
            final_cr = _build_change_report(prev_grid, cur_grid, element_records)
        budget_update = _read_budget(final_cr)
        if budget_update is not None:
            budget_remaining = budget_update

        # agent_pos_after: harness-detected position from bbox (more reliable than
        # cursor_pos arithmetic for confirming where the avatar actually ended up)
        agent_pos_after = final_cr.get("agent_pos_after")
        if agent_pos_after:
            cursor_pos = tuple(agent_pos_after)

        steps_taken  = len(motion_log)
        budget_spent = sum(1 for e in motion_log if e.get("action") != "RESET")

        element_overlaps = _detect_element_overlaps(cursor_pos, element_records)
        harness_note = _harness_coordinate_note(
            command, args.get("target_pos") and tuple(int(x) for x in args["target_pos"]),
            element_records,
        )

        command_result = {
            "command_executed":  command,
            "args":              args,
            "steps_taken":       steps_taken,
            "budget_spent":      budget_spent,
            "budget_remaining":  budget_remaining,
            "cursor_pos_after":  list(cursor_pos) if cursor_pos else None,
            "agent_pos_after":   agent_pos_after,
            "element_overlaps":  element_overlaps,
            "target_analysis":   target_analysis,
            "harness_note":      harness_note or None,
            "motion_log":        motion_log,
            "final_state":       obs.state.name if obs and hasattr(obs.state, "name") else state,
            "error":             exec_error,
        }
        if exec_error:
            print(f"         EXEC ERROR: {exec_error}")
        if harness_note:
            print(f"         HARNESS: {harness_note[:120]}")
        if element_overlaps:
            names = [e["name"] for e in element_overlaps]
            print(f"         OVERLAPS: {names}")

        # ---- Log -------------------------------------------------------
        log_entry = {
            "turn": turn, "state": state, "levels_completed": lc,
            "win_levels": int(obs.win_levels) if obs is not None else 1,
            "game_id": a.game,
            "command": command, "args": args,
            "rationale": rationale, "predict": predict,
            "revise_knowledge": revise, "done": done_flag,
            "steps_taken": steps_taken, "budget_spent": budget_spent,
            "budget_remaining": budget_remaining,
            "cursor_pos": list(cursor_pos) if cursor_pos else None,
            "action_effects": {k: list(v) for k, v in action_effects.items()},
            "command_result": command_result,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "turn_start_iso": turn_start.isoformat(),
            "frame_b64": frame_b64,
        }
        log_fh.write(json.dumps(log_entry) + "\n")
        log_fh.flush()
        render_play_session(session_dir, frames_dir, live=True)

        command_trace.append({
            "turn": turn, "command": command, "args": args,
            "rationale": rationale, "state": state,
            "steps_taken": steps_taken,
        })
        recent_history.append({
            "turn": turn, "command": command, "args": args,
            "state": state,
            "cursor_pos_after": list(cursor_pos) if cursor_pos else None,
            "steps_taken": steps_taken,
            "budget_spent": budget_spent,
        })

        final_state = obs.state.name if obs and hasattr(obs.state, "name") else state
        prev_grid = cur_grid

        if done_flag and final_state in ("WIN", "GAME_OVER"):
            break
        if final_state in ("WIN", "GAME_OVER"):
            break

    log_fh.close()
    render_play_session(session_dir, frames_dir, live=False)

    # Post-game knowledge capture
    outcome = {"WIN": "WIN", "GAME_OVER": "LOSS"}.get(final_state, final_state)
    try:
        rsp = backends.call_anthropic(
            model=TUTOR_MODEL, system=SYSTEM_POSTGAME,
            user=build_postgame_user_message(
                game_id=a.game, outcome=outcome, turns=len(command_trace),
                final_state=final_state,
                levels_completed=int(obs.levels_completed) if obs else 0,
                win_levels=int(obs.win_levels) if obs else 0,
                action_effects=action_effects,
                working_knowledge=working_knowledge,
                command_trace=command_trace,
            ),
            image_b64=None, max_tokens=1500,
        )
        note = rsp["reply"].strip()
    except Exception as e:  # noqa: BLE001
        note = f"(post-game call failed: {e})"
    (session_dir / "post_game_knowledge.md").write_text(note, encoding="utf-8")

    manifest = {
        "trial_id": trial_id, "game_id": a.game,
        "round2_session": str(round2_dir),
        "lessons_from": str(lessons_path) if lessons_path else None,
        "tutor_model": TUTOR_MODEL,
        "max_turns": a.max_turns,
        "turns_played": len(command_trace),
        "final_state": final_state, "outcome": outcome,
        "action_effects_learned": {k: list(v) for k, v in action_effects.items()},
        "walls_learned": [list(w) for w in sorted(walls)],
        "wall_time_s": round(time.time() - t0, 1),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": ["working_knowledge.md", "play_log.jsonl",
                  "post_game_knowledge.md"],
    }
    (session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    render_play_session(session_dir, frames_dir, live=False)

    if outcome == "WIN":
        _dump_training_data(
            game_id=a.game, trial_id=trial_id,
            system_prompt=SYSTEM_PLAY, session_dir=session_dir,
        )

    total_cost = sum(e.get("cost_usd", 0)
                     for line in log_path.read_text(encoding="utf-8").splitlines()
                     for e in [json.loads(line)] if "cost_usd" in e)
    print(f"\nOutcome: {outcome} ({final_state}), {len(command_trace)} TUTOR calls, "
          f"${total_cost:.3f}, wrote {session_dir}")
    print(f"Action effects learned: {action_effects}")


if __name__ == "__main__":
    main()
