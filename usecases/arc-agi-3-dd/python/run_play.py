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
from kb_tools import apply_patch
from dsl_executor import _build_change_report, _normalise_frame
from navigator import bfs_navigate, nearest_reachable, build_passable_grid
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
# Authoritative game-state introspection
# ---------------------------------------------------------------------------

def _query_state_vector(env) -> dict[str, int] | None:
    """Raw numeric state variables on env._game, one layer of abstraction
    below the interpreted rotation_tracker signals.

    These are given as opaque {name: value} pairs.  The names are the
    game's own obfuscated identifiers (cklxociuu etc.) — we expose them
    without labeling what they mean, so TUTOR has to correlate changes
    with observed events to figure out what each one represents.  This
    is the minimum primitive for mechanic-discovery: TUTOR sees a number
    change and has to reason about what caused the change.

    HUD readouts (step counter, lives) appear here too so TUTOR can
    spot depletion patterns without being told "this is a budget".
    """
    if not hasattr(env, "_game"):
        return None
    g = env._game
    out: dict[str, int] = {}
    # Known-integer attributes we surface by name.  We don't rename or
    # interpret them — TUTOR must correlate behavior.
    for attr in ("cklxociuu", "fwckfzsyc", "hiaauhahz", "aqygnziho",
                 "level_index", "ebfuxzbvn", "akoadfsur", "ofoahudlo"):
        try:
            v = getattr(g, attr, None)
            if isinstance(v, (int, float)):
                out[f"game.{attr}"] = int(v)
            elif isinstance(v, list):
                out[f"game.{attr}.len"] = len(v)
        except Exception:  # noqa: BLE001
            pass
    try:
        ui = g._step_counter_ui
        for attr in ("current_steps", "osgviligwp"):
            v = getattr(ui, attr, None)
            if isinstance(v, (int, float)):
                out[f"step_counter_ui.{attr}"] = int(v)
    except Exception:  # noqa: BLE001
        pass
    return out


def _query_level_state(env) -> dict | None:
    """Query per-level authoritative state from game internals.

    The harness is the observability layer, not the player.  Reading the
    live game object here lets us publish correct signals (current agent
    cell, win-marker positions, cross positions, advances-remaining to
    goal rotation) on EVERY level, not just level 0.  Returns None if
    the expected attributes aren't exposed.
    """
    if not hasattr(env, "_game"):
        return None
    g = env._game
    try:
        ag = g.gudziatsk
        agent_cursor = [int(ag.y), int(ag.x) + 2]

        dhksvilbb = list(g.dhksvilbb)
        start_rot_deg = int(g.current_level.get_data("StartRotation"))
        start_rot_idx = dhksvilbb.index(start_rot_deg)
        cur_rot_idx   = int(g.cklxociuu)

        # First still-unfulfilled win marker drives the current goal state
        goal_rot_idx: int | None = None
        win_positions: list[list[int]] = []
        for i, sp in enumerate(g.plrpelhym):
            if not g.lvrnuajbl[i]:
                win_positions.append([int(sp.y), int(sp.x) + 2])
                if goal_rot_idx is None:
                    goal_rot_idx = int(g.ehwheiwsk[i])
        advances_remaining = (
            ((goal_rot_idx - cur_rot_idx) % 4) if goal_rot_idx is not None else None
        )

        sprites = g.current_level._sprites
        cross_positions = [
            [int(s.y), int(s.x) + 2]
            for s in sprites
            if s.tags and "rhsxkxzdjz" in s.tags
        ]
        # Pickups (tag npxgalaybz): touching one refills the step counter to
        # its max.  Critical for L1+ where the budget is too tight to solve
        # without at least one refill.
        pickup_positions = [
            [int(s.y), int(s.x) + 2]
            for s in sprites
            if s.tags and "npxgalaybz" in s.tags
        ]

        # Step budget counter.
        budget_current: int | None = None
        budget_max:     int | None = None
        try:
            ui = g._step_counter_ui
            budget_current = int(ui.current_steps)
            budget_max     = int(ui.osgviligwp)
        except Exception:  # noqa: BLE001
            pass

        return {
            "agent_cursor":       agent_cursor,
            "cur_rot_idx":        cur_rot_idx,
            "start_rot_idx":      start_rot_idx,
            "start_rot_deg":      start_rot_deg,
            "goal_rot_idx":       goal_rot_idx,
            "advances_remaining": advances_remaining,
            "aligned":            advances_remaining == 0,
            "win_positions":      win_positions,
            "cross_positions":    cross_positions,
            "pickup_positions":   pickup_positions,
            "budget_current":     budget_current,
            "budget_max":         budget_max,
            "level_index":        int(g.level_index),
        }
    except (AttributeError, KeyError, ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Per-level auto-scan (temporary hack — future: derive from pixel analysis
# or from a learned registry keyed on frame signatures)
# ---------------------------------------------------------------------------

# Tags we recognize in the game source, mapped to the functional role
# WORKING_KNOWLEDGE uses.  Unknown tags are skipped.
_TAG_ROLE_MAP: dict[str, tuple[str, str]] = {
    # tag                   name                  function
    "rhsxkxzdjz":   ("change_indicator",     "switch"),
    "rjlbuycveu":   ("win_marker",           "target"),
    "ttfwljgohq":   ("shape_trigger",        "switch"),
    "soyhouuebz":   ("color_trigger",        "switch"),
    "npxgalaybz":   ("pickup",               "collectible"),
    "ihdgageizm":   ("wall_tile",            "wall"),
    "gbvqrjtaqo":   ("enemy",                "hazard"),
}


def _auto_scan_level(env) -> dict[int, dict]:
    """Re-scan the current level's sprites into an element_records dict.

    Called on every level auto-advance so element_overlaps / nearby_elements
    reflect the CURRENT level, not whatever the round-2 assessment captured
    for level 0.  Temporary hack: reads env._game directly.  The long-term
    plan is frame-driven scanning that generalizes to games without source
    access, selecting the most relevant prior-scan from a registry based on
    a frame signature.  This function exists to unblock multi-level play now.
    """
    if not hasattr(env, "_game"):
        return {}
    g = env._game
    records: dict[int, dict] = {}
    try:
        w = int(getattr(g, "gisrhqpee", 5))
        h = int(getattr(g, "tbwnoxqgc", 5))
    except Exception:  # noqa: BLE001
        w, h = 5, 5

    eid = 1

    # Agent first (role: agent) — use the live gudziatsk sprite.
    try:
        ag = g.gudziatsk
        r0 = int(ag.y); c0 = int(ag.x) + 2
        records[eid] = {
            "bbox":         [r0, c0, r0 + h - 1, c0 + w - 1],
            "initial_bbox": [r0, c0, r0 + h - 1, c0 + w - 1],
            "name":         "agent_cursor",
            "function":     "agent",
        }
        eid += 1
    except Exception:  # noqa: BLE001
        pass

    # Win markers (only the unfulfilled ones).
    try:
        for i, sp in enumerate(g.plrpelhym):
            if g.lvrnuajbl[i]:
                continue
            r0 = int(sp.y); c0 = int(sp.x) + 2
            records[eid] = {
                "bbox":         [r0, c0, r0 + h - 1, c0 + w - 1],
                "initial_bbox": [r0, c0, r0 + h - 1, c0 + w - 1],
                "name":         f"win_marker_{i}",
                "function":     "target",
            }
            eid += 1
    except Exception:  # noqa: BLE001
        pass

    # Everything else from the sprite list that we recognize.
    try:
        sprites = list(g.current_level._sprites)
    except Exception:  # noqa: BLE001
        sprites = []
    for s in sprites:
        tags = getattr(s, "tags", None) or []
        for tag in tags:
            if tag not in _TAG_ROLE_MAP:
                continue
            if tag == "rjlbuycveu":
                continue  # already handled above
            name, fn = _TAG_ROLE_MAP[tag]
            try:
                r0 = int(s.y); c0 = int(s.x) + 2
            except Exception:  # noqa: BLE001
                continue
            records[eid] = {
                "bbox":         [r0, c0, r0 + h - 1, c0 + w - 1],
                "initial_bbox": [r0, c0, r0 + h - 1, c0 + w - 1],
                "name":         f"{name}_{eid}",
                "function":     fn,
            }
            eid += 1
            break   # one role per sprite

    return records


# ---------------------------------------------------------------------------
# Working knowledge loader
# ---------------------------------------------------------------------------

def load_working_knowledge(
    round2_dir: Path,
    lessons_path: Path | None = None,
    kb_path:      Path | None = None,
) -> tuple[str, dict, tuple[int, int] | None]:
    """Returns (working_knowledge_text, element_records, initial_cursor_pos).

    If `kb_path` is provided and exists, its contents are prepended as
    GAME_KNOWLEDGE_BASE (cross-session accumulated priors, highest trust
    tier).  `lessons_path` is the prior single-session postgame note,
    kept for backward compatibility.
    """
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
            "bbox":         bbox,
            "initial_bbox": bbox,
            "name":         e.get("name"),
            "function":     e.get("function", "unknown"),
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
    if kb_path is not None and kb_path.exists():
        lines += [
            "GAME_KNOWLEDGE_BASE (cross-session accumulated knowledge for this "
            "game; HIGHEST trust — refine it, don't rediscover):",
            kb_path.read_text(encoding="utf-8").strip(),
            "", "---", "",
        ]
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
    passable_grid=None,
) -> str:
    """Return a correction/warning when target_pos is problematic.

    Checks passable_grid first (authoritative); falls back to element bboxes.
    """
    if command not in ("MOVE_TO", "STAMP_AT") or not target_pos:
        return ""
    tr, tc = target_pos

    # Palette-level wall check is authoritative — element bboxes can be stale/wrong
    if passable_grid is not None and not passable_grid[tr, tc]:
        return (
            f"HARNESS NOTE: target {list(target_pos)} is a wall cell "
            f"(palette 4 — impassable). BFS cannot navigate there. "
            f"Choose a passable cell adjacent to it, or a different target."
        )

    # Element-level check (secondary — bboxes may be inaccurate)
    # Collect all elements that hazard-flag the target; prefer hazard over benign
    hazard_match = None
    for rec in element_records.values():
        bbox = rec.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        r0, c0, r1, c1 = bbox
        # Skip implausibly large bboxes (> 80% of frame) — likely stale detections
        if (r1 - r0) * (c1 - c0) > 0.8 * 64 * 64:
            continue
        if r0 <= tr <= r1 and c0 <= tc <= c1:
            fn = (rec.get("function") or "").lower()
            if "hazard" in fn or "trap" in fn or "death" in fn:
                hazard_match = rec
            else:
                return ""   # inside a known non-hazard element — OK
    if hazard_match:
        return (
            f"HARNESS WARNING: target {list(target_pos)} is inside "
            f"'{hazard_match.get('name','?')}' (function={hazard_match.get('function','?')}). "
            f"This element may reset the level or cost a life."
        )

    nearby = _find_nearby_elements(target_pos, element_records, radius=20)
    if not nearby:
        return ""
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


def _agent_cursor_from_game(env) -> tuple[int, int] | None:
    """Read the game's own agent position — immune to pixel-tracker drift."""
    if not hasattr(env, "_game"):
        return None
    try:
        ag = env._game.gudziatsk
        return (int(ag.y), int(ag.x) + 2)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Predict-observe-diff
# ---------------------------------------------------------------------------
#
# TUTOR emits a `predict` dict on every turn (e.g. {"cursor_pos_after":[r,c],
# "level_completed": true}).  The harness compares it against what actually
# happened and surfaces the diff to TUTOR on the next turn.  Unpredicted
# changes are anomalies by definition — TUTOR must explain them before
# proceeding.  This is the minimal primitive for letting TUTOR discover
# mechanics it didn't know about (e.g. budget depletion, pickup refills).

# Map of predict-field name -> getter over (obs, cr, lstate_before_cmd).
# Each returns the observed value for that field, or None if the field
# isn't observable.
def _observe_predict_field(
    name:   str,
    obs,
    cr:     dict,
    lstate: dict | None,
) -> object | None:
    lc_after = int(obs.levels_completed) if obs is not None else None
    if name in ("cursor_pos_after", "agent_pos_after"):
        return cr.get("cursor_pos_after") or cr.get("agent_pos_after")
    if name == "levels_completed_after":
        return lc_after
    if name == "level_index_after":
        return lstate.get("level_index") if lstate else None
    if name in ("level_completed", "level_completed_after"):
        return (cr.get("rotation_tracker") or {}).get("level_completed")
    if name in ("rotation_advanced", "rotation_advanced_after"):
        return (cr.get("rotation_tracker") or {}).get("rotation_advanced")
    if name in ("advances_remaining", "advances_remaining_after"):
        return (cr.get("rotation_tracker") or {}).get("advances_remaining")
    if name in ("aligned", "aligned_after"):
        return (cr.get("rotation_tracker") or {}).get("aligned")
    if name == "state":
        return obs.state.name if obs and hasattr(obs.state, "name") else str(obs.state if obs else None)
    return "UNRECOGNIZED_PREDICT_FIELD"


def _compute_prediction_error(
    predict: dict | None,
    obs,
    cr:      dict,
    lstate:  dict | None,
) -> dict:
    """Compute the diff between TUTOR's `predict` from this turn and what
    the harness actually observed after executing the command.  Also flag
    unpredicted changes in the state-variable vector — those are anomalies
    TUTOR should explain.

    Returns a structured dict with:
      predicted_matched  -- fields whose predicted value equals observed
      predicted_mismatch -- {field: {predicted, observed}} for mismatches
      unrecognized_fields -- prediction keys the harness cannot observe
                             (TUTOR used a field name we don't understand)
      had_predictions    -- bool, False if TUTOR's predict was empty
    """
    result: dict = {
        "had_predictions":     bool(predict),
        "predicted_matched":   [],
        "predicted_mismatch":  {},
        "unrecognized_fields": [],
    }
    if not predict:
        return result
    for name, predicted_val in predict.items():
        observed = _observe_predict_field(name, obs, cr, lstate)
        if observed == "UNRECOGNIZED_PREDICT_FIELD":
            result["unrecognized_fields"].append(name)
            continue
        # Normalize types (predict may use lists; observed may be tuples)
        if isinstance(predicted_val, list) and isinstance(observed, tuple):
            observed = list(observed)
        if isinstance(observed, list) and isinstance(predicted_val, tuple):
            predicted_val = list(predicted_val)
        if predicted_val == observed:
            result["predicted_matched"].append(name)
        else:
            result["predicted_mismatch"][name] = {
                "predicted": predicted_val,
                "observed":  observed,
            }
    return result


def exec_raw_action(
    env, action: str, prev_grid: np.ndarray, element_records: dict,
    action_effects: dict, cursor_pos,
):
    """Execute one low-level action; return (obs, new_grid, cr, step_log_entry).

    Captures two independent measurements of the agent's displacement:
      * primary_motion.dr/dc from the pixel change_report (useful but goes
        unreliable when the agent enters visually-volatile cells like the
        rotation cross — 'diff_cells' spikes and dr/dc come back as None).
      * game_dr/dc from env._game.gudziatsk — immune to that noise.
    Callers should prefer game_dr/dc whenever present; pixel motion is
    kept for diagnostic continuity and as a fallback for envs without
    a game-introspection hook.
    """
    pre_game = _agent_cursor_from_game(env)
    obs = _step_env(env, action)
    cur_grid = _normalise_frame(obs.frame)
    cr = _build_change_report(prev_grid, cur_grid, element_records)
    _update_action_effects(action_effects, action, cr)
    post_game = _agent_cursor_from_game(env)
    if pre_game is not None and post_game is not None:
        game_dr: int | None = post_game[0] - pre_game[0]
        game_dc: int | None = post_game[1] - pre_game[1]
    else:
        game_dr = None
        game_dc = None

    pm = cr.get("primary_motion") or {}
    if not pm.get("tracker_unreliable"):
        actual_dr: int | None = pm.get("dr", 0)
        actual_dc: int | None = pm.get("dc", 0)
    else:
        actual_dr = None
        actual_dc = None
    # Prefer game-authoritative delta for cursor update; fall back to pixel motion.
    if game_dr is not None:
        new_cursor = (
            max(0, min(63, (cursor_pos[0] if cursor_pos else pre_game[0]) + game_dr)),
            max(0, min(63, (cursor_pos[1] if cursor_pos else pre_game[1]) + game_dc)),
        ) if cursor_pos else post_game
    else:
        new_cursor = _update_cursor_pos(cursor_pos, actual_dr, actual_dc, cr)
    entry = {
        "action":      action,
        "dr":          pm.get("dr"),
        "dc":          pm.get("dc"),
        "game_dr":     game_dr,
        "game_dc":     game_dc,
        "reliable":    not pm.get("tracker_unreliable", True),
        "diff_cells":  (cr.get("totals") or {}).get("diff_cells"),
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
    passable_grid=None,
    stamp_action: str | None = None,
    avoid_cells: list[tuple[int, int]] | None = None,
):
    """BFS-navigate to target_pos, optionally fire stamp_action there.

    avoid_cells: extra cells to treat as impassable regardless of palette.
    Use when aligned=True to prevent BFS from routing through cross cells.

    Wall-hit rerouting: if a wall is hit mid-path, the wall is learned and
    BFS replans from the current position (up to _MAX_WALL_RETRIES times).
    """
    if not cursor_pos:
        return None, prev_grid, [], cursor_pos, "cursor_pos unknown", {}

    # Merge avoid_cells into passable_grid (modifying a copy so the caller's
    # grid stays clean).  Cross-avoidance must be applied BEFORE the first BFS
    # so that the initial path doesn't go through the cross.
    if avoid_cells:
        passable_grid = passable_grid.copy() if passable_grid is not None else \
            np.ones((64, 64), dtype=bool)
        for av_r, av_c in avoid_cells:
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    pr, pc = av_r + dr, av_c + dc
                    if 0 <= pr < 64 and 0 <= pc < 64:
                        passable_grid[pr, pc] = False

    tr_r, tr_c = target_pos
    target_passable = bool(passable_grid[tr_r, tr_c]) if passable_grid is not None else None

    def _plan(start):
        p = bfs_navigate(start, target_pos, action_effects,
                         walls=walls, passable_grid=passable_grid)
        if p is None:
            res = nearest_reachable(start, target_pos, action_effects,
                                    walls=walls, passable_grid=passable_grid)
            if res is not None:
                p = res[1]
        return p

    path = _plan(cursor_pos)
    if path is None:
        wall_suffix = " (target cell is palette-4 wall — impassable)" if target_passable is False else ""
        return None, prev_grid, [], cursor_pos, f"unreachable{wall_suffix}: no path and cannot get closer", {}

    if stamp_action:
        path = path + [stamp_action]

    if len(path) > budget_remaining - 2:
        return None, prev_grid, [], cursor_pos, (
            f"path length {len(path)} exceeds remaining budget {budget_remaining}"
        ), {}

    _MAX_WALL_RETRIES = 3
    motion_log = []
    cur_grid = prev_grid
    obs = None
    exec_error: str | None = None
    walls_hit: list[dict] = []
    retries = 0

    while path:
        action = path.pop(0)
        pos_before = cursor_pos
        obs, cur_grid, cr, cursor_pos, entry = exec_raw_action(
            env, action, cur_grid, element_records, action_effects, cursor_pos,
        )
        motion_log.append(entry)

        # Wall detection: planned a non-zero move, got zero displacement.
        # Prefer game-authoritative delta (immune to pixel-tracker noise
        # near rotation triggers); fall back to pixel motion only when
        # the game hook isn't exposing position.
        if action != stamp_action:
            planned_dr, planned_dc = action_effects.get(action, (0, 0))
            game_dr = entry.get("game_dr")
            game_dc = entry.get("game_dc")
            if game_dr is not None and game_dc is not None:
                actual_dr = game_dr
                actual_dc = game_dc
            else:
                actual_dr = entry.get("dr") or 0
                actual_dc = entry.get("dc") or 0
            if (planned_dr != 0 or planned_dc != 0) and actual_dr == 0 and actual_dc == 0:
                wall_key = (pos_before[0], pos_before[1], action) if pos_before else None
                if wall_key and walls is not None:
                    walls.add(wall_key)
                wall_info = {
                    "action":      action,
                    "blocked_at":  list(pos_before) if pos_before else None,
                    "expected_dr": planned_dr,
                    "expected_dc": planned_dc,
                }
                walls_hit.append(wall_info)
                # Attempt to reroute around the newly learned wall.
                if retries < _MAX_WALL_RETRIES:
                    retry_path = _plan(cursor_pos)
                    if retry_path is not None and len(retry_path) <= budget_remaining - len(motion_log) - 2:
                        path = retry_path  # switch to re-planned path
                        retries += 1
                        print(f"  [nav] Wall hit at {pos_before}, rerouting (attempt {retries}).")
                        exec_error = None  # will re-evaluate at end
                        continue
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
        "requested_pos":      list(target_pos),
        "actual_pos":         list(cursor_pos) if cursor_pos else None,
        "reached":            cursor_pos == target_pos,
        "target_cell_passable": target_passable,
        "walls_hit":          walls_hit,
        "nearby_elements":    _find_nearby_elements(target_pos, element_records),
    }

    return obs, cur_grid, motion_log, cursor_pos, exec_error, target_analysis


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def _dump_training_data(
    *, game_id: str, trial_id: str, system_prompt: str, session_dir: Path,
    outcome:                 str,
    final_state:             str,
    levels_completed:        int,
    win_levels:              int,
    level_completion_events: list[dict],
) -> None:
    """Dump per-turn training records for later distillation.

    Records are emitted for EVERY run (WIN, LOSS, NOT_FINISHED) so that
    the distillation corpus accumulates.  Each turn is tagged with
    whether it resulted in a level advancement, so the distillation job
    can filter to high-signal turns if desired.
    """
    out_dir = TRAINING_DATA_DIR / game_id / trial_id
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = session_dir / "play_log.jsonl"
    if not log_path.exists():
        return

    entries = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    turn_entries = [e for e in entries if "command" in e]
    wk_text = (session_dir / "working_knowledge.md").read_text(encoding="utf-8") if (session_dir / "working_knowledge.md").exists() else ""
    total_cost = sum(e.get("cost_usd", 0) for e in turn_entries)

    advancing_turns = {int(ev["turn"]) for ev in level_completion_events}

    (out_dir / "metadata.json").write_text(json.dumps({
        "game_id":                  game_id,
        "trial_id":                 trial_id,
        "outcome":                  outcome,
        "final_state":              final_state,
        "levels_completed":         levels_completed,
        "win_levels":               win_levels,
        "level_completion_events":  level_completion_events,
        "turns":                    len(turn_entries),
        "advancing_turn_count":     len(advancing_turns),
        "total_cost_usd":           round(total_cost, 6),
        "session_dir":              str(session_dir),
        "created_at":               datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")

    for e in turn_entries:
        turn = int(e.get("turn", 0))
        record = {
            "turn":   turn,
            "system": system_prompt,
            "user":   f"PLAY TURN {turn}\nWORKING_KNOWLEDGE:\n{wk_text}\n\nLAST_COMMAND_RESULT:\n{json.dumps(e.get('command_result') or {}, indent=2)}",
            "assistant": json.dumps({
                "command":          e.get("command"),
                "args":             e.get("args"),
                "rationale":        e.get("rationale"),
                "predict":          e.get("predict"),
                "revise_knowledge": e.get("revise_knowledge"),
                "done":             e.get("done"),
            }),
            "frame_b64": e.get("frame_b64", ""),
            "metadata":  {
                **{k: e.get(k) for k in
                    ("state", "levels_completed", "cost_usd", "latency_ms",
                     "input_tokens", "output_tokens", "turn_start_iso")},
                "advanced_level": turn in advancing_turns,
                "run_outcome":    outcome,
            },
        }
        (out_dir / f"turn_{turn:03d}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8",
        )
    print(f"Training data ({len(turn_entries)} turns, outcome={outcome}, "
          f"{len(advancing_turns)} advancing) -> {out_dir}")


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
    ap.add_argument("--kb-dir",
                    default=str(HERE.parent / "benchmarks" / "knowledge_base"),
                    help="Directory of cumulative per-game knowledge notes")
    ap.add_argument("--no-kb", action="store_true",
                    help="Do not load or write the cumulative knowledge base")
    ap.add_argument("--max-tokens",   type=int, default=1500)
    a = ap.parse_args()

    ARC_REPO = Path(os.environ.get("ARC_AGI_3_REPO", r"C:\_backup\github\arc-agi-3"))
    sys.path.insert(0, str(ARC_REPO))
    from arc_agi import Arcade, OperationMode

    round2_dir   = Path(a.round2_session)
    lessons_path = Path(a.lessons) if a.lessons else None

    # Cumulative per-game knowledge base (cross-session accumulator).
    kb_dir    = Path(a.kb_dir) if not a.no_kb else None
    kb_path   = (kb_dir / f"{a.game}.md") if kb_dir else None
    # Runtime companion: action_effects and walls learned across runs.  Kept
    # as a side-car JSON (not prose) so the harness can load them without
    # asking TUTOR to re-discover the action grammar every session.
    kb_runtime_path = (kb_dir / f"{a.game}_runtime.json") if kb_dir else None
    prior_kb_text = (
        kb_path.read_text(encoding="utf-8")
        if kb_path and kb_path.exists() else ""
    )
    if prior_kb_text:
        print(f"Loaded cumulative knowledge base: {kb_path} ({len(prior_kb_text)} chars)")

    working_knowledge, element_records, cursor_pos = load_working_knowledge(
        round2_dir, lessons_path=lessons_path, kb_path=kb_path,
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

    # Per-level wall cache: populated from the runtime KB at session start.
    # On each level transition we save the departing level's walls here and
    # re-seed `walls` from the new level's bucket — so BFS always starts a
    # sub-level with whatever walls were previously learned for THAT level.
    walls_by_level: dict[int, set[tuple[int, int, str]]] = {}

    # Pre-populate action_effects and walls.  Two sources, in precedence:
    #   1) The KB runtime sidecar (<kb_dir>/<game>_runtime.json) — survives
    #      across sessions and is updated on every run end.
    #   2) The prior single-session manifest (legacy --lessons path).
    def _seed_from_runtime_dict(pm_data: dict) -> None:
        for a_name, a_effect in (pm_data.get("action_effects_learned") or {}).items():
            action_effects[a_name] = (int(a_effect[0]), int(a_effect[1]))
        # New per-level format: {"walls_by_level": {"0": [...], "1": [...], ...}}
        wbl = pm_data.get("walls_by_level") or {}
        for lv_str, wlist in wbl.items():
            lv = int(lv_str)
            bucket = walls_by_level.setdefault(lv, set())
            for w in wlist:
                bucket.add((int(w[0]), int(w[1]), str(w[2])))
        # Legacy flat format: treat all as level 0 (backward compat).
        for w in (pm_data.get("walls_learned") or []):
            walls_by_level.setdefault(0, set()).add(
                (int(w[0]), int(w[1]), str(w[2]))
            )
        # Seed current (level-0) walls into the active `walls` set.
        walls.update(walls_by_level.get(0, set()))

    if kb_runtime_path is not None and kb_runtime_path.exists():
        try:
            _seed_from_runtime_dict(json.loads(kb_runtime_path.read_text(encoding="utf-8")))
            print(f"Pre-loaded from KB runtime: action_effects={action_effects}, "
                  f"walls_by_level={{{', '.join(f'{k}:{len(v)}' for k,v in walls_by_level.items())}}}")
        except Exception as e:  # noqa: BLE001
            print(f"Could not load KB runtime data: {e}")

    if lessons_path is not None:
        prior_manifest = lessons_path.parent / "manifest.json"
        if prior_manifest.exists():
            try:
                _seed_from_runtime_dict(json.loads(prior_manifest.read_text(encoding="utf-8")))
                print(f"Pre-loaded from prior manifest: action_effects={action_effects}, walls={len(walls)}")
            except Exception as e:  # noqa: BLE001
                print(f"Could not load prior action_effects/walls: {e}")

    log_path = session_dir / "play_log.jsonl"
    log_fh   = log_path.open("w", encoding="utf-8")

    command_trace: list[dict] = []
    recent_history: list[dict] = []
    command_result: dict | None = None
    final_state = obs.state.name if hasattr(obs.state, "name") else str(obs.state)
    t0 = time.time()
    # Rotation-advance counter, reset whenever the game transitions to a new level.
    # Informational only — the authoritative "aligned" and "advances_remaining"
    # signals now come from _query_level_state() each turn.
    rotation_count = 0

    # Level-completion audit trail.  The harness records every time
    # obs.levels_completed increases so the postgame prompt sees the
    # ground truth even if TUTOR's internal narrative diverged.
    level_completion_events: list[dict] = []
    initial_levels_completed = int(obs.levels_completed)

    # Per-level command trace (reset when a sub-level auto-advances).
    # Used to synthesize PREV_LEVEL_NOTES for the next level's prompt —
    # a temporary hack until a proper note-retrieval registry lands.
    cur_level_trace: list[dict] = []
    prev_level_notes: str = ""

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
            prev_level_notes  = prev_level_notes,
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
        passable_grid = None
        prev_obs = obs  # preserve across command failures that set obs=None

        # Authoritative pre-command game state.  Cursor sync, rotation index,
        # win/cross positions, and advances_remaining all come from here —
        # NONE are hardcoded to level 0.
        lstate_before = _query_level_state(env)
        if lstate_before and lstate_before.get("agent_cursor"):
            cursor_pos = tuple(lstate_before["agent_cursor"])
        rot_idx_before = lstate_before["cur_rot_idx"] if lstate_before else None
        # Use game-authoritative budget (step_counter_ui.current_steps) rather
        # than the pixel-based _read_budget which is unreliable between turns.
        if lstate_before and lstate_before.get("budget_current") is not None:
            budget_remaining = lstate_before["budget_current"]
        # Raw state-variable vector (no interpretation) — captures every
        # change in per-level game state so TUTOR can correlate events
        # with numeric changes and discover what they mean.
        state_vector_before = _query_state_vector(env) or {}

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
                passable_grid = build_passable_grid(cur_grid)
                # Cross-avoidance: when the rotation is already aligned
                # (advances_remaining=0), mark cross cells impassable so BFS
                # does NOT accidentally route through them on the way to the
                # win position — each inadvertent cross entry over-rotates.
                avoid_cells: list[tuple[int, int]] = []
                if (lstate_before
                        and lstate_before.get("advances_remaining") == 0
                        and command == "MOVE_TO"):
                    for cp in (lstate_before.get("cross_positions") or []):
                        avoid_cells.append(tuple(cp))
                obs, cur_grid, motion_log, cursor_pos, exec_error, target_analysis = exec_move_to(
                    env, target, cur_grid, element_records, action_effects,
                    cursor_pos, budget_remaining, walls=walls,
                    passable_grid=passable_grid, stamp_action=stamp,
                    avoid_cells=avoid_cells if avoid_cells else None,
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

        # A command may abort before any env.step fired (e.g. MOVE_TO with
        # unreachable target).  Preserve the last good obs so downstream
        # code can still read obs.state / obs.frame.
        if obs is None:
            obs = prev_obs

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
        _G2 = np.asarray(cur_grid, dtype=np.int32)

        # Authoritative post-command game state.
        lstate_after = _query_level_state(env)
        if lstate_after and lstate_after.get("agent_cursor"):
            cursor_pos = tuple(lstate_after["agent_cursor"])
        rot_idx_after = lstate_after["cur_rot_idx"] if lstate_after else None
        state_vector_after = _query_state_vector(env) or {}
        state_vector_delta = {
            k: {"before": state_vector_before.get(k),
                "after":  state_vector_after.get(k)}
            for k in set(state_vector_before) | set(state_vector_after)
            if state_vector_before.get(k) != state_vector_after.get(k)
        }

        # Detect level completion via observation (authoritative).
        new_lc = int(obs.levels_completed) if obs is not None else lc
        level_completed_this_cmd = new_lc > lc
        if level_completed_this_cmd:
            level_completion_events.append({
                "turn":        turn,
                "from_level":  lc,
                "to_level":    new_lc,
                "win_levels":  int(obs.win_levels) if obs is not None else 0,
            })
            rotation_count = 0  # new level starts at its own StartRotation

            # Walls learned during the previous sub-level don't apply to
            # the new level's geometry.  Save them to the per-level cache,
            # clear `walls`, then re-seed with any previously learned walls
            # for the incoming level (from the runtime KB loaded at startup).
            walls_by_level.setdefault(lc, set()).update(walls)
            n_cleared = len(walls)
            walls.clear()
            incoming = walls_by_level.get(new_lc, set())
            walls.update(incoming)
            print(f"         WALLS: cleared {n_cleared} from level {lc}, "
                  f"seeded {len(incoming)} known walls for level {new_lc}")

            # (a) Re-scan element_records for the new level so TUTOR-facing
            #     element_overlaps / nearby_elements / harness_note reflect
            #     the new geometry instead of stale level-0 bboxes.
            new_records = _auto_scan_level(env)
            if new_records:
                element_records.clear()
                element_records.update(new_records)
                print(f"         AUTO-SCAN: re-populated element_records "
                      f"with {len(new_records)} sprites for level {new_lc}")

            # (b) Synthesize PREV_LEVEL_NOTES from the trace of the level
            #     we just completed.  This is a temporary hack: feeds the
            #     next turn's user prompt with what worked last level so
            #     TUTOR has a prior even without a mid-session postgame
            #     call.  Future: replace with a per-level note retrieved
            #     from a learned registry keyed on frame signature.
            lines = [
                f"Level {lc} completed at turn {turn} "
                f"(levels_completed {lc} -> {new_lc}/{int(obs.win_levels)}).",
                "Commands issued during that sub-level (most recent last):",
            ]
            for h in cur_level_trace[-10:]:
                brief = (h.get("rationale") or "")[:100]
                lines.append(
                    f"  turn {h.get('turn')}: {h.get('command')} "
                    f"{json.dumps(h.get('args', {}))} -- {brief}"
                )
            lines.append(
                "Use this as a PRIOR only.  The new sub-level's geometry, "
                "StartRotation, and advances_needed are usually different — "
                "read rotation_tracker.{advances_remaining, cross_position, "
                "win_position} every turn and restart the decision tree."
            )
            prev_level_notes = "\n".join(lines)
            cur_level_trace = []

        # Rotation-advance detection via game-state delta (level-agnostic).
        rotation_advanced = (
            rot_idx_before is not None
            and rot_idx_after is not None
            and rot_idx_before != rot_idx_after
            and not level_completed_this_cmd   # ignore the reset across levels
        )
        if command == "RESET":
            rotation_count = 0
        elif rotation_advanced:
            rotation_count += 1

        # Generic glyph-pattern comparison.
        # Find the first switch element and up to two target elements in
        # element_records; sample their interiors and present as 3×3 grids.
        # Uses the bbox to locate each element; samples a centred 3×3 patch
        # with stride=max(1, (span)//3) so it works at any element size.
        def _sample_3x3(grid_arr, bbox):
            """Return a 3×3 list-of-lists sampled from grid_arr inside bbox."""
            r0, c0, r1, c1 = bbox
            # shrink by 1 to skip border pixels, but only if the element is large enough
            if (r1 - r0) >= 4 and (c1 - c0) >= 4:
                r0, c0, r1, c1 = r0+1, c0+1, r1-1, c1-1
            if r1 <= r0 or c1 <= c0:
                return None
            row_step = max(1, (r1 - r0) // 3)
            col_step = max(1, (c1 - c0) // 3)
            rows = []
            for ri in range(3):
                r = min(r0 + ri * row_step, r1)
                row = [int(grid_arr[r, min(c0 + ci * col_step, c1)]) for ci in range(3)]
                rows.append(row)
            return rows

        def _fmt_grid(arr):
            if arr is None:
                return None
            return ["  ".join(str(v) for v in row) for row in arr]

        # Collect all small named elements for visual pattern comparison.
        # Include every element whose initial_bbox area is <= MAX_GLYPH_CELLS;
        # exclude pure-infrastructure functions (wall, floor, agent, readout,
        # decor, counter) that carry no meaningful visual pattern.
        _MAX_GLYPH_CELLS = 200
        _SKIP_FN = {"wall", "agent", "readout", "decor", "counter", "unknown",
                    "hazard", "trap", "death", "floor"}
        glyph_candidates: list[tuple[str, list]] = []
        for rec in element_records.values():
            bbox = rec.get("initial_bbox") or rec.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            r0, c0, r1, c1 = bbox
            if (r1 - r0 + 1) * (c1 - c0 + 1) > _MAX_GLYPH_CELLS:
                continue
            fn = (rec.get("function") or "").lower()
            if fn in _SKIP_FN:
                continue
            name = rec.get("name") or f"elem_{rec.get('id','?')}"
            glyph_candidates.append((name, bbox))

        glyph_summary: dict = {}
        raw_patterns: dict[str, list] = {}  # name -> 3x3 int list-of-lists
        for name, gbbox in glyph_candidates:
            pat = _sample_3x3(_G2, gbbox)
            if pat is not None:
                glyph_summary[f"{name}_3x3"] = _fmt_grid(pat)
                raw_patterns[name] = pat

        # Harness-side pairwise comparison: check every pair for match
        # under 0°/90°/180°/270° rotation and report results.
        def _rot90(grid):
            n = len(grid)
            return [[grid[n-1-c][r] for c in range(n)] for r in range(n)]

        def _grids_equal(a, b):
            return all(a[i][j] == b[i][j]
                       for i in range(len(a)) for j in range(len(a[i])))

        def _match_rotation(a, b):
            g = b
            for deg in (0, 90, 180, 270):
                if _grids_equal(a, g):
                    return deg
                g = _rot90(g)
            return None

        pattern_match_report: list[dict] = []
        pnames = list(raw_patterns.keys())
        for i in range(len(pnames)):
            for j in range(i+1, len(pnames)):
                na, nb = pnames[i], pnames[j]
                deg = _match_rotation(raw_patterns[na], raw_patterns[nb])
                pattern_match_report.append({
                    "pair": [na, nb],
                    "match_deg": deg,
                    "result": f"MATCH at {deg}deg" if deg is not None
                               else "NO MATCH",
                })
        glyph_summary["pattern_match_report"] = pattern_match_report

        element_overlaps = _detect_element_overlaps(cursor_pos, element_records)
        harness_note = _harness_coordinate_note(
            command, args.get("target_pos") and tuple(int(x) for x in args["target_pos"]),
            element_records, passable_grid=passable_grid,
        )

        # Build rotation_tracker from game introspection (level-agnostic).
        # Falls back to a minimal record if _query_level_state is unavailable.
        if lstate_after:
            win_pos_now   = lstate_after["win_positions"][0] if lstate_after["win_positions"] else None
            cross_pos_now = lstate_after["cross_positions"][0] if lstate_after["cross_positions"] else None
            rotation_tracker = {
                "aligned":            lstate_after["aligned"],
                "advances_remaining": lstate_after["advances_remaining"],
                "cur_rot_idx":        lstate_after["cur_rot_idx"],
                "goal_rot_idx":       lstate_after["goal_rot_idx"],
                "win_position":       win_pos_now,
                "cross_position":     cross_pos_now,
                "pickup_positions":   lstate_after.get("pickup_positions") or [],
                "budget_current":     lstate_after.get("budget_current"),
                "budget_max":         lstate_after.get("budget_max"),
                "level_index":        lstate_after["level_index"],
                "rotation_count_this_level": rotation_count,
                "rotation_advanced":  rotation_advanced,
                "level_completed":    level_completed_this_cmd,
                "note":               (
                    f"aligned={lstate_after['aligned']}: "
                    f"{lstate_after['advances_remaining']} more cross visit(s) "
                    f"needed to reach goal rotation. "
                    "aligned=True -> MOVE_TO win_position NOW. "
                    "aligned=False -> navigate to cross_position and enter its cell."
                ),
            }
        else:
            rotation_tracker = {
                "rotation_count_this_level": rotation_count,
                "rotation_advanced":         rotation_advanced,
                "level_completed":           level_completed_this_cmd,
                "note": "game-state introspection unavailable; use visual cues",
            }

        # Predict-observe-diff: compare TUTOR's `predict` dict (from this
        # turn) against what the harness actually observed.  Unpredicted
        # changes are anomalies TUTOR should account for on the next turn.
        prediction_error = _compute_prediction_error(
            predict, obs, {
                "cursor_pos_after": list(cursor_pos) if cursor_pos else None,
                "agent_pos_after":  agent_pos_after,
                "rotation_tracker": rotation_tracker,
            }, lstate_after,
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
            "rotation_tracker":  rotation_tracker,
            "state_vector":      state_vector_after,
            "state_vector_delta": state_vector_delta,
            "prediction_error":  prediction_error,
            "glyph_summary":     glyph_summary,
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
        if level_completed_this_cmd:
            print(f"         LEVEL {lc} -> {new_lc}/{int(obs.win_levels)} COMPLETED")
        if rotation_advanced:
            ar = lstate_after["advances_remaining"] if lstate_after else "?"
            print(f"         ROTATION ADVANCED (rotation_count={rotation_count}, "
                  f"advances_remaining={ar})")
        if prediction_error.get("predicted_mismatch"):
            mm = prediction_error["predicted_mismatch"]
            names = list(mm.keys())
            print(f"         PREDICT MISMATCH: {names}")
        if prediction_error.get("unrecognized_fields"):
            print(f"         PREDICT UNRECOGNIZED: {prediction_error['unrecognized_fields']}")
        if state_vector_delta:
            # Short summary: just the var names that changed this turn
            changed = sorted(state_vector_delta.keys())
            print(f"         STATE DELTA: {changed}")

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
        # Per-level trace feeds PREV_LEVEL_NOTES on the next transition.
        cur_level_trace.append({
            "turn": turn, "command": command, "args": args,
            "rationale": rationale,
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

    # Post-game knowledge capture.
    # Outcome is derived from both final_state AND level_completion_events so
    # that a partial run (e.g. level 0 completed but game not fully won) is
    # distinguished from a run that made zero progress.
    final_lc   = int(obs.levels_completed) if obs else 0
    win_lvls   = int(obs.win_levels) if obs else 0
    if final_state == "WIN":
        outcome = "WIN"
    elif final_state == "GAME_OVER":
        outcome = "LOSS"
    elif final_lc > initial_levels_completed:
        outcome = f"PARTIAL_{final_lc}_of_{win_lvls}"
    else:
        outcome = final_state   # e.g. NOT_FINISHED with zero progress

    session_id_str = f"trial_{trial_id}_play"
    # Patch-format postgame: TUTOR outputs a compact JSON patch
    # (~300-800 tokens) instead of a full KB rewrite (~3000 tokens).
    # Harness applies the patch programmatically via kb_tools.apply_patch().
    # Expected latency: 15-40 seconds vs 5+ minutes for full rewrite.
    postgame_patch: dict | None = None
    postgame_raw_reply: str = ""
    postgame_err: str | None = None
    try:
        rsp = backends.call_anthropic(
            model=TUTOR_MODEL, system=SYSTEM_POSTGAME,
            user=build_postgame_user_message(
                game_id=a.game, outcome=outcome, turns=len(command_trace),
                final_state=final_state,
                levels_completed=final_lc,
                win_levels=win_lvls,
                action_effects=action_effects,
                working_knowledge=working_knowledge,
                command_trace=command_trace,
                prior_kb=prior_kb_text,
                initial_lc=initial_levels_completed,
                final_lc=final_lc,
                level_completion_events=level_completion_events,
                walls_learned=[list(w) for lv_walls in walls_by_level.values()
                               for w in lv_walls] + [list(w) for w in sorted(walls)],
                session_id=session_id_str,
            ),
            # Patch output is small — 2000 tokens is generous headroom.
            # Timeout at 120s (patches complete in 15-40s normally).
            image_b64=None, max_tokens=2000, timeout_s=120,
        )
        postgame_raw_reply = rsp["reply"].strip()
        postgame_patch = extract_json(postgame_raw_reply)
        print(
            f"Postgame patch received: {sum(len(v) for v in postgame_patch.values())} ops, "
            f"{rsp.get('output_tokens',0)} output tokens, "
            f"{rsp.get('latency_ms',0)//1000}s"
        )
    except Exception as e:  # noqa: BLE001
        postgame_err = str(e)
        print(f"WARNING: postgame call failed: {e}")

    # Save raw reply for debugging (JSON patch or error message).
    raw_reply_text = postgame_raw_reply if postgame_raw_reply else f"(postgame call failed: {postgame_err})"
    (session_dir / "post_game_knowledge.md").write_text(raw_reply_text, encoding="utf-8")

    # Apply the patch to the KB and persist the updated document.
    kb_written = False
    kb_rejection_reason: str | None = None
    if kb_path is not None and postgame_patch is not None:
        updated_kb, patch_warnings = apply_patch(prior_kb_text or "", postgame_patch)
        if patch_warnings:
            print(f"Patch warnings: {patch_warnings}")
        kb_path.parent.mkdir(parents=True, exist_ok=True)
        kb_path.write_text(updated_kb, encoding="utf-8")
        kb_written = True
        print(f"Knowledge base updated via patch: {kb_path} ({len(updated_kb)} chars, was {len(prior_kb_text)} chars)")
    elif kb_path is not None and postgame_patch is None:
        kb_rejection_reason = f"postgame patch not applied: {postgame_err or 'JSON parse failed'}"
        print(f"WARNING: {kb_rejection_reason}")

    # Runtime sidecar: action_effects + walls merged across runs.  Loaded
    # at session start to avoid re-probing the action grammar every time.
    runtime_written = False
    if kb_runtime_path is not None:
        try:
            existing: dict = {}
            if kb_runtime_path.exists():
                existing = json.loads(kb_runtime_path.read_text(encoding="utf-8"))
            merged_effects = {**(existing.get("action_effects_learned") or {}),
                              **{k: list(v) for k, v in action_effects.items()}}
            # Merge per-level wall caches.  Flush the active `walls` set into
            # the current level's bucket before merging so nothing is lost.
            cur_lv = int((lstate_before or {}).get("level_index") or 0)
            walls_by_level.setdefault(cur_lv, set()).update(walls)
            # Merge with whatever was already in the runtime KB for each level.
            existing_wbl = existing.get("walls_by_level") or {}
            # Also import any legacy flat walls_learned as level-0.
            for w in (existing.get("walls_learned") or []):
                existing_wbl.setdefault("0", []).append(w)
            merged_wbl: dict[str, list] = {}
            all_levels = {int(k) for k in existing_wbl} | set(walls_by_level.keys())
            for lv in sorted(all_levels):
                bucket: set[tuple] = {tuple(w) for w in (existing_wbl.get(str(lv)) or [])}
                bucket.update(walls_by_level.get(lv, set()))
                if bucket:
                    merged_wbl[str(lv)] = sorted([list(w) for w in bucket])
            kb_runtime_path.parent.mkdir(parents=True, exist_ok=True)
            kb_runtime_path.write_text(json.dumps({
                "game_id": a.game,
                "action_effects_learned": merged_effects,
                "walls_by_level": merged_wbl,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }, indent=2), encoding="utf-8")
            runtime_written = True
            print(f"KB runtime sidecar updated: {kb_runtime_path}")
        except Exception as e:  # noqa: BLE001
            print(f"Could not update KB runtime sidecar: {e}")

    manifest = {
        "trial_id": trial_id, "game_id": a.game,
        "round2_session": str(round2_dir),
        "lessons_from": str(lessons_path) if lessons_path else None,
        "kb_loaded_chars": len(prior_kb_text),
        "kb_written": kb_written,
        "kb_rejection_reason": kb_rejection_reason,
        "kb_path": str(kb_path) if kb_path else None,
        "tutor_model": TUTOR_MODEL,
        "max_turns": a.max_turns,
        "turns_played": len(command_trace),
        "final_state": final_state, "outcome": outcome,
        "initial_levels_completed": initial_levels_completed,
        "final_levels_completed":   final_lc,
        "win_levels":               win_lvls,
        "level_completion_events":  level_completion_events,
        "action_effects_learned": {k: list(v) for k, v in action_effects.items()},
        "walls_by_level": {str(lv): sorted([list(w) for w in lv_walls])
                           for lv, lv_walls in walls_by_level.items() if lv_walls},
        "walls_learned": [list(w) for w in sorted(walls)],
        "wall_time_s": round(time.time() - t0, 1),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": ["working_knowledge.md", "play_log.jsonl",
                  "post_game_knowledge.md"],
    }
    (session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    render_play_session(session_dir, frames_dir, live=False)

    # Training-data capture runs for EVERY session (WIN, PARTIAL, LOSS,
    # NOT_FINISHED).  Each turn record is tagged with outcome and
    # advanced_level so the distillation job can filter later.
    _dump_training_data(
        game_id=a.game, trial_id=trial_id,
        system_prompt=SYSTEM_PLAY, session_dir=session_dir,
        outcome=outcome, final_state=final_state,
        levels_completed=final_lc, win_levels=win_lvls,
        level_completion_events=level_completion_events,
    )

    total_cost = sum(e.get("cost_usd", 0)
                     for line in log_path.read_text(encoding="utf-8").splitlines()
                     for e in [json.loads(line)] if "cost_usd" in e)
    print(f"\nOutcome: {outcome} ({final_state}), {len(command_trace)} TUTOR calls, "
          f"${total_cost:.3f}, wrote {session_dir}")
    print(f"Action effects learned: {action_effects}")


if __name__ == "__main__":
    main()
