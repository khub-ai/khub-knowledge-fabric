"""
record_run.py — Record a full LS20 run to a rich JSON file for playback.

Captures every env.step() with:
  - 64x64 pixel frame (before and after)
  - Full game state (player pos, shape/color/rot, counter, lives)
  - Events fired (push pads, changers, resets, level advances, blocks)
  - Per-level metadata (maze layout, interactables, targets)
  - Pre-solve vs agent source tagging

Output: <output>.json   (loadable by playback_viewer.html)

Usage:
  python record_run.py --out playlogs/run_001.json
  python record_run.py --out playlogs/run_001.json --start-level 5
"""

from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")
sys.path.insert(0, "C:/_backup/github/khub-knowledge-fabric")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test")

import arc_agi
from ensemble import _KNOWN_SUBPLANS, obs_levels_completed, obs_frame


# ---------------------------------------------------------------------------
# ARC color palette (index → "r,g,b")
# ---------------------------------------------------------------------------
ARC_PALETTE = {
    0:  [0,   0,   0],
    1:  [0,   116, 217],
    2:  [255, 65,  54],
    3:  [46,  204, 113],
    4:  [170, 170, 170],
    5:  [200, 200, 200],
    6:  [255, 220, 0],
    7:  [255, 133, 27],
    8:  [127, 219, 255],
    9:  [255, 255, 255],
    10: [61,  153, 112],
    11: [0,   31,  63],
    12: [255, 133, 27],
    13: [255, 41,  117],
    14: [133, 20,  75],
    15: [57,  204, 204],
}


def _frame_to_list(frame_arr) -> list[list[int]]:
    """Convert numpy frame to Python list-of-lists."""
    if hasattr(frame_arr, "tolist"):
        return frame_arr.tolist()
    return [list(row) for row in frame_arr]


def _game_state(game) -> dict:
    return {
        "player_x":   game.gudziatsk.x,
        "player_y":   game.gudziatsk.y,
        "shape_idx":  game.fwckfzsyc,
        "color_idx":  game.hiaauhahz,
        "rot_idx":    game.cklxociuu,
        "counter":    game._step_counter_ui.current_steps,
        "counter_max":game._step_counter_ui.osgviligwp,
        "lives":      game.aqygnziho,
    }


def _level_meta(game, level_idx: int) -> dict:
    """Extract static metadata for one level."""
    lv = game._levels[level_idx]
    data = lv._data if hasattr(lv, "_data") else {}

    walls, targets, resets, shape_ch, color_ch, rot_ch, push_pads = [], [], [], [], [], [], []

    for s in lv._sprites:
        tags = s.tags or []
        pos  = {"x": s.x, "y": s.y}
        if "ihdgageizm" in tags:
            walls.append(pos)
        elif "rjlbuycveu" in tags:
            targets.append(pos)
        elif "npxgalaybz" in tags:
            resets.append(pos)
        elif "ttfwljgohq" in tags:
            shape_ch.append(pos)
        elif "soyhouuebz" in tags:
            color_ch.append(pos)
        elif "rhsxkxzdjz" in tags:
            rot_ch.append(pos)
        elif "gbvqrjtaqo" in tags:
            push_pads.append({"name": s.name, "x": s.x, "y": s.y})

    return {
        "level":        level_idx + 1,
        "step_counter": data.get("StepCounter", 42),
        "steps_dec":    data.get("StepsDecrement", 2),
        "goal_shape":   data.get("kvynsvxbpi"),
        "goal_color":   data.get("GoalColor"),
        "goal_rot":     data.get("GoalRotation"),
        "start_shape":  data.get("StartShape"),
        "start_color":  data.get("StartColor"),
        "start_rot":    data.get("StartRotation"),
        "fog":          data.get("Fog", False),
        "walls":        walls,
        "targets":      targets,
        "resets":       resets,
        "shape_changers":  shape_ch,
        "color_changers":  color_ch,
        "rot_changers":    rot_ch,
        "push_pads":       push_pads,
    }


def _detect_events(state_before: dict, state_after: dict, level_num: int, blocked: bool) -> list[str]:
    events = []
    if blocked:
        events.append("BLOCKED")
    if state_after["counter"] > state_before["counter"] and not blocked:
        events.append("COUNTER_RESET")
    if state_after["shape_idx"] != state_before["shape_idx"]:
        events.append(f"SHAPE_CHANGE:{state_before['shape_idx']}->{state_after['shape_idx']}")
    if state_after["color_idx"] != state_before["color_idx"]:
        events.append(f"COLOR_CHANGE:{state_before['color_idx']}->{state_after['color_idx']}")
    if state_after["rot_idx"] != state_before["rot_idx"]:
        events.append(f"ROT_CHANGE:{state_before['rot_idx']}->{state_after['rot_idx']}")
    if state_after["player_x"] != state_before["player_x"] or state_after["player_y"] != state_before["player_y"]:
        dx = abs(state_after["player_x"] - state_before["player_x"])
        dy = abs(state_after["player_y"] - state_before["player_y"])
        if dx > 5 or dy > 5:
            events.append(f"PUSH_PAD:({state_before['player_x']},{state_before['player_y']})->"
                          f"({state_after['player_x']},{state_after['player_y']})")
    if state_after["lives"] < state_before["lives"]:
        events.append(f"LIFE_LOST:lives={state_after['lives']}")
    return events


def record(env_id: str, start_level: int, out_path: Path, verbose: bool = True) -> None:
    arc = arc_agi.Arcade()
    env = arc.make(env_id, render_mode=None)
    obs = env.reset()
    AS  = {a.name: a for a in env.action_space}

    game = env._game
    total_levels = len(game._levels)
    color_names  = {v: f"color{k}" for k, v in enumerate(game.tnkekoeuk)}

    # Build per-level metadata
    levels_meta = [_level_meta(game, i) for i in range(total_levels)]

    steps_data: list[dict] = []
    plan    = _KNOWN_SUBPLANS.get(env_id, [])
    step_n  = 0

    def take_step(action_name: str, source: str) -> bool:
        """Execute one step, append record. Returns True if level advanced."""
        nonlocal step_n
        step_n += 1

        frame_before = _frame_to_list(obs_frame(obs))
        state_before = _game_state(game)
        level_before = obs_levels_completed(obs)

        obs2 = env.step(AS[action_name])
        frame_after  = _frame_to_list(obs_frame(obs2))
        state_after  = _game_state(game)
        level_after  = obs_levels_completed(obs2)

        blocked = (state_after["player_x"] == state_before["player_x"] and
                   state_after["player_y"] == state_before["player_y"] and
                   not any(e in ["COUNTER_RESET","PUSH_PAD","LIFE_LOST"]
                           for e in _detect_events(state_before, state_after, level_before, False)))
        events  = _detect_events(state_before, state_after, level_before, blocked)
        advanced = level_after > level_before
        if advanced:
            events.append(f"LEVEL_ADVANCE:{level_before+1}->{level_after+1}")

        # Goal match status at step end
        if game.plrpelhym:
            goal_ok = {
                "shape": state_after["shape_idx"] == (game.ldxlnycps[0] if game.ldxlnycps else -1),
                "color": state_after["color_idx"] == (game.yjdexjsoa[0] if game.yjdexjsoa else -1),
                "rot":   state_after["rot_idx"]   == (game.ehwheiwsk[0] if game.ehwheiwsk else -1),
            }
        else:
            goal_ok = {}

        rec = {
            "step":          step_n,
            "level":         level_before + 1,
            "action":        action_name,
            "source":        source,
            "frame_before":  frame_before,
            "frame_after":   frame_after,
            "state_before":  state_before,
            "state_after":   state_after,
            "events":        events,
            "goal_match":    goal_ok,
            "level_advanced":advanced,
        }
        steps_data.append(rec)

        if verbose:
            ev_str = "  " + " | ".join(events) if events else ""
            print(f"  [{source}] step {step_n:4d} {action_name}: "
                  f"({state_before['player_x']},{state_before['player_y']})"
                  f"->({state_after['player_x']},{state_after['player_y']})"
                  f"  ctr={state_after['counter']:2d}"
                  f"  lv={level_after+1}{ev_str}")

        # Update obs for next step
        import ctypes
        # We can't rebind 'obs' in the outer nonlocal since it's a loop var,
        # but we store the latest obs in a mutable container
        obs_holder[0] = obs2
        return advanced

    # mutable holder for obs across closure calls
    obs_holder = [obs]

    # Rebind take_step to use obs_holder
    def take_step2(action_name: str, source: str) -> bool:
        nonlocal step_n
        step_n += 1

        frame_before = _frame_to_list(obs_frame(obs_holder[0]))
        state_before = _game_state(game)
        level_before = obs_levels_completed(obs_holder[0])

        obs2 = env.step(AS[action_name])
        obs_holder[0] = obs2

        frame_after = _frame_to_list(obs_frame(obs2))
        state_after = _game_state(game)
        level_after = obs_levels_completed(obs2)

        blocked = (state_after["player_x"] == state_before["player_x"] and
                   state_after["player_y"] == state_before["player_y"])
        events  = _detect_events(state_before, state_after, level_before, blocked)
        advanced = level_after > level_before
        if advanced:
            events.append(f"LEVEL_ADVANCE:{level_before+1}->{level_after+1}")

        if game.plrpelhym and not advanced:
            goal_ok = {
                "shape": state_after["shape_idx"] == (game.ldxlnycps[0] if game.ldxlnycps else -1),
                "color": state_after["color_idx"] == (game.yjdexjsoa[0] if game.yjdexjsoa else -1),
                "rot":   state_after["rot_idx"]   == (game.ehwheiwsk[0] if game.ehwheiwsk else -1),
            }
        else:
            goal_ok = {}

        rec = {
            "step":           step_n,
            "level":          level_before + 1,
            "action":         action_name,
            "source":         source,
            "frame_before":   frame_before,
            "frame_after":    frame_after,
            "state_before":   state_before,
            "state_after":    state_after,
            "events":         events,
            "goal_match":     goal_ok,
            "level_advanced": advanced,
        }
        steps_data.append(rec)

        if verbose:
            ev_str = ("  [" + " | ".join(events) + "]") if events else ""
            print(f"  [{source:8s}] step {step_n:4d} {action_name}: "
                  f"({state_before['player_x']:2d},{state_before['player_y']:2d})"
                  f"->({state_after['player_x']:2d},{state_after['player_y']:2d})"
                  f"  sh={state_after['shape_idx']} col={state_after['color_idx']}"
                  f"  ctr={state_after['counter']:2d}  lv={level_after+1}{ev_str}")
        return advanced

    # --- Pre-solve ---
    print(f"Recording pre-solve ({len(plan)} steps)...")
    for action in plan:
        take_step2(action, "presolve")

    final_level = obs_levels_completed(obs_holder[0]) + 1
    won = (final_level > total_levels)

    # --- Write output ---
    payload = {
        "meta": {
            "env_id":       env_id,
            "total_levels": total_levels,
            "start_level":  start_level,
            "final_level":  final_level,
            "won":          won,
            "total_steps":  step_n,
            "presolve_steps": len(plan),
            "palette":      ARC_PALETTE,
            "color_names":  {str(k): v for k, v in enumerate(game.tnkekoeuk)},
            "rot_values":   list(game.dhksvilbb),
            "shape_count":  len(game.ijessuuig),
            "run_date":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "levels": levels_meta,
        "steps":  steps_data,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nRecorded {step_n} steps, final level {final_level}/{total_levels}.")
    print(f"Output: {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env",         default="ls20")
    p.add_argument("--out",         default="playlogs/presolve_record.json")
    p.add_argument("--start-level", dest="start_level", type=int, default=1)
    p.add_argument("--quiet",       action="store_true")
    args = p.parse_args()
    record(args.env, args.start_level, Path(args.out), verbose=not args.quiet)
