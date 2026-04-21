"""Offline level solver: runs BFS+ planner and writes the optimal solution
to the per-game solutions.json.

Usage:
  python solve_and_record.py --game ls20-9607627b --target-level 1

"--target-level 1" means sub-level index 1 (i.e. level 2/7 in the user's
numbering).  The script:
  1. Instantiates the game env.
  2. Replays any already-recorded solutions to reach the target level.
  3. Queries _query_level_state for crosses, pickups, win, budget.
  4. Builds passable_grid from the current frame.
  5. Calls planner.solve_level.
  6. VERIFIES the returned sequence by executing it on a fresh env.
  7. Writes (or updates) the solutions.json entry for that level.

NO LLM calls.  Deterministic.  Can be re-run any time the level geometry
changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Plumbing: point at the ARC env repo so we can import arc_agi / arcengine.
ARC_REPO = Path(os.environ.get(
    "ARC_AGI_3_REPO", r"C:\_backup\github\arc-agi-3"
))
sys.path.insert(0, str(ARC_REPO))

from arc_agi import Arcade, OperationMode                        # noqa: E402
from arcengine import GameAction                                  # noqa: E402

from dsl_executor import _normalise_frame                         # noqa: E402
from navigator import build_passable_grid                         # noqa: E402
from planner import (                                                  # noqa: E402
    solve_level, solve_align_phase, solve_align_phase_all,
    solve_win_phase, problem_from_env_query,
)

# _query_level_state lives in run_play.py; we import it rather than
# duplicating the attribute-probing logic.
from run_play import _query_level_state                            # noqa: E402


KB_DIR   = HERE.parent / "benchmarks" / "knowledge_base"
ACTION_EFFECTS_DEFAULT = {
    "ACTION1": (-5, 0), "ACTION2": (5, 0),
    "ACTION3": (0, -5), "ACTION4": (0, 5),
}


# -----------------------------------------------------------------------------

def _make_env(game_id: str):
    arc = Arcade(
        operation_mode   = OperationMode.OFFLINE,
        environments_dir = str(ARC_REPO / "environment_files"),
    )
    env = arc.make(game_id)
    env.reset()
    return env


def _load_solutions(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_solutions(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _frame_hash(grid: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(grid, dtype=np.int8).tobytes()
    ).hexdigest()[:16]


def _load_runtime_walls(kb_runtime_path: Path, level_key: str) -> set:
    if not kb_runtime_path.exists():
        return set()
    try:
        rt = json.loads(kb_runtime_path.read_text(encoding="utf-8"))
        return {tuple(w) for w in rt.get("walls_by_level", {}).get(level_key, [])}
    except Exception:
        return set()


def _replay_prior_levels(env, solutions_map: dict, target_lc: int):
    """Replay recorded solutions for levels 0..target_lc-1 so env is at
    lc=target_lc when we start planning.  Returns the last obs (the entry
    frame for target_lc) on success, None on failure."""
    if target_lc == 0:
        # We're already at lc=0 immediately after env.reset(); need a no-op
        # step to get a fresh obs.  Since there's no no-op action, we just
        # return None and the caller re-resets.
        return None
    obs = None
    for lc in range(target_lc):
        key = str(lc)
        if key not in solutions_map.get("levels", {}):
            print(f"  [replay] no recorded solution for L{lc} -- cannot reach L{target_lc}")
            return None
        steps = solutions_map["levels"][key]["game_steps"]
        for step in steps:
            action_idx = int(step.replace("ACTION", ""))
            obs = env.step(action_idx)
        if obs is None or int(obs.levels_completed) != lc + 1:
            print(f"  [replay] L{lc} replay failed (expected lc={lc+1}, got {obs.levels_completed if obs else None})")
            return None
        print(f"  [replay] L{lc} advanced via {len(steps)} steps")
    return obs


def _execute_sequence(env, initial_lc: int, steps: list[str], verbose: bool = False) -> tuple[bool, int]:
    """Run the solution; return (did_level_advance, steps_used).  Caller
    passes the env already positioned at initial_lc."""
    prev_frame = None
    for i, action_name in enumerate(steps):
        action_idx = int(action_name.replace("ACTION", ""))
        obs = env.step(action_idx)
        if verbose:
            grid = _normalise_frame(obs.frame)
            diff = int(np.sum(prev_frame != grid)) if prev_frame is not None else -1
            moved = diff > 10 if diff >= 0 else True
            print(f"    step {i+1:2d} {action_name}: lc={obs.levels_completed} "
                  f"diff={diff} {'MOVED' if moved else 'BLOCKED'}")
            prev_frame = grid
        state_str = obs.state.name if hasattr(obs.state, "name") else str(obs.state)
        if state_str != "NOT_FINISHED":
            return False, i + 1
        if int(obs.levels_completed) > initial_lc:
            return True, i + 1
    return False, len(steps)


# -----------------------------------------------------------------------------

def solve_one_level(game_id: str, target_lc: int, dry_run: bool = False) -> dict:
    """Solve and (optionally) record the solution for target_lc."""
    t0 = time.time()
    solutions_path = KB_DIR / f"{game_id}_solutions.json"
    kb_runtime_path = KB_DIR / f"{game_id}_runtime.json"

    solutions_data = _load_solutions(solutions_path)
    solutions_data.setdefault("game_id", game_id)
    solutions_data.setdefault("levels", {})

    # Stage 1: drive env to target level.
    env = _make_env(game_id)
    obs = _replay_prior_levels(env, solutions_data, target_lc)
    if obs is None and target_lc > 0:
        print(f"FATAL: cannot reach L{target_lc} -- need prior level solutions first.")
        return {"ok": False, "reason": "prior-level-missing"}

    # Stage 2: gather level parameters.
    ls = _query_level_state(env)
    print(f"  [query] L{target_lc} state: {ls}")

    # Frame: we captured obs from the last step of the replay.  For
    # target_lc==0, call env.reset() to get obs0 (already done by _make_env,
    # but we need the returned obs).  Re-reset to guarantee a fresh obs.
    if obs is None:
        obs = env.reset()

    grid = _normalise_frame(obs.frame)
    passable = build_passable_grid(grid)
    walls    = _load_runtime_walls(kb_runtime_path, str(target_lc))
    print(f"  [env] passable cells on-grid: {int(passable.sum())}/{passable.size}")
    print(f"  [env] walls for L{target_lc}: {walls}")

    # Stage 3a: plan+execute alignment phase (using current passable grid).
    spawn = tuple(ls["agent_cursor"])
    advances_req = int(ls["advances_remaining"])
    budget_max_v = int(ls["budget_max"])
    budget_cur_v = int(ls.get("budget_current", budget_max_v))

    def _snap_to_step_grid(pt: tuple[int, int]) -> tuple[int, int]:
        """Snap (r,c) to the nearest cell on the agent's 5-step grid from spawn
        that is passable. Sprite-reported positions are frequently off-grid
        (e.g. [51,42] for a sprite the agent actually reaches at [50,41])."""
        sr, sc = spawn
        r, c = pt
        best = None
        best_d = 1e9
        for dr in range(-4, 5):
            for dc in range(-4, 5):
                nr, nc = r + dr, c + dc
                if 0 <= nr < 64 and 0 <= nc < 64 \
                   and (nr - sr) % 5 == 0 and (nc - sc) % 5 == 0 \
                   and passable[nr, nc]:
                    d = abs(dr) + abs(dc)
                    if d < best_d:
                        best_d = d
                        best = (nr, nc)
        return best or pt

    cross_list  = [_snap_to_step_grid(tuple(c)) for c in ls["cross_positions"]]
    pickup_list = [_snap_to_step_grid(tuple(p)) for p in ls["pickup_positions"]]
    win_cell    = _snap_to_step_grid(tuple(ls["win_positions"][0]))
    print(f"  [snap] cross {ls['cross_positions']} -> {cross_list}")
    print(f"  [snap] pickups {ls['pickup_positions']} -> {pickup_list}")
    print(f"  [snap] win {ls['win_positions'][0]} -> {win_cell}")

    print(f"  [plan:A] aligning... spawn={spawn} crosses={cross_list} "
          f"advances_req={advances_req} budget={budget_cur_v}/{budget_max_v}")
    t_plan = time.time()
    phaseA_candidates = solve_align_phase_all(
        spawn             = spawn,
        cross_positions   = cross_list,
        pickup_positions  = pickup_list,
        advances_required = advances_req,
        budget_max        = budget_max_v,
        budget_current    = budget_cur_v,
        action_effects    = ACTION_EFFECTS_DEFAULT,
        passable_grid     = passable,
        walls             = walls,
        budget_per_action = 2,
    )
    if not phaseA_candidates:
        print("  [plan:A] UNSOLVABLE (no alignment path)")
        return {"ok": False, "reason": "align-unsolvable"}
    print(f"  [plan:A] found {len(phaseA_candidates)} alignment candidates; trying in score order")

    # Try each candidate until phase B succeeds.
    sequence = None
    phaseA_seq = None
    phaseB_seq = None
    for idx, (candA_seq, candA_pos, candA_budget, candA_pickups) in enumerate(phaseA_candidates):
        print(f"  [plan:A cand #{idx+1}] {len(candA_seq)} steps -> pos={candA_pos} "
              f"budget={candA_budget} pickups_left={list(candA_pickups)}")

        # Execute on a fresh env snapshot (replay prior levels, then phase A).
        env_snap = _make_env(game_id)
        obs_snap = _replay_prior_levels(env_snap, solutions_data, target_lc)
        if obs_snap is None:
            obs_snap = env_snap.reset()
        for a in candA_seq:
            obs_snap = env_snap.step(int(a.replace("ACTION", "")))
        lsA = _query_level_state(env_snap)
        live_budget_A = int(lsA.get("budget_current", 0))
        if not lsA.get("aligned") or live_budget_A <= 0:
            print(f"    skip: live post-exec state aligned={lsA.get('aligned')} "
                  f"budget={live_budget_A}")
            continue

        gridA = _normalise_frame(obs_snap.frame)
        passableA = build_passable_grid(gridA)

        print(f"    [plan:B] winning... start={candA_pos} win={win_cell} "
              f"budget={live_budget_A}/{budget_max_v} pickups={list(candA_pickups)}")
        candB_seq = solve_win_phase(
            start_pos         = candA_pos,
            pickup_positions  = list(candA_pickups),
            win_position      = win_cell,
            budget_current    = live_budget_A,
            budget_max        = budget_max_v,
            action_effects    = ACTION_EFFECTS_DEFAULT,
            passable_grid     = passableA,
            walls             = walls,
            budget_per_action = 2,
        )
        if candB_seq is None:
            print(f"    [plan:B cand #{idx+1}] UNSOLVABLE from this align state")
            continue
        print(f"    [plan:B cand #{idx+1}] win-phase: {len(candB_seq)} steps  -> TOTAL "
              f"{len(candA_seq) + len(candB_seq)} steps")
        phaseA_seq = candA_seq
        phaseB_seq = candB_seq
        sequence = list(candA_seq) + list(candB_seq)
        break

    dt_plan = time.time() - t_plan
    if sequence is None:
        print(f"  [planner] UNSOLVABLE -- no candidate yielded feasible phase B "
              f"({dt_plan*1000:.1f} ms, {len(phaseA_candidates)} candidates tried)")
        return {"ok": False, "reason": "no-feasible-candidate"}
    print(f"  [planner] total: {len(sequence)} steps in {dt_plan*1000:.1f} ms")
    print(f"  [sequence] {sequence}")

    # Stage 4: verify on a fresh env.
    env_v = _make_env(game_id)
    obs_v = _replay_prior_levels(env_v, solutions_data, target_lc)
    if obs_v is None:
        obs_v = env_v.reset()
    grid_v = _normalise_frame(obs_v.frame)
    entry_hash = _frame_hash(grid_v)

    advanced, steps_used = _execute_sequence(env_v, target_lc, sequence, verbose=False)
    if not advanced:
        print(f"  [verify] FAILED: sequence did not advance level after {steps_used} steps")
        return {"ok": False, "reason": "verify-failed", "sequence": sequence}
    print(f"  [verify] OK: advanced to L{target_lc+1} in {steps_used} steps")

    # Stage 5: write solutions.json.
    if dry_run:
        print("  [write] dry_run=True, not writing solutions.json")
    else:
        solutions_data["levels"][str(target_lc)] = {
            "game_steps":         list(sequence),
            "budget_cost":        len(sequence),
            "frame_hash_on_entry": entry_hash,
            "step_count":         len(sequence),
            "solver":             "planner.solve_level",
            "solved_at":          time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _save_solutions(solutions_path, solutions_data)
        print(f"  [write] recorded L{target_lc} solution ({len(sequence)} steps)")

    return {
        "ok": True, "steps": len(sequence), "plan_ms": dt_plan * 1000,
        "wall_time_s": time.time() - t0, "sequence": sequence,
    }


# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game",          required=True,
                    help="Game id, e.g. ls20-9607627b")
    ap.add_argument("--target-level",  type=int, required=True,
                    help="Sub-level index to solve (0-indexed; target_lc=1 means solve 2/7)")
    ap.add_argument("--dry-run",       action="store_true",
                    help="Don't write solutions.json; just plan and verify")
    a = ap.parse_args()

    result = solve_one_level(a.game, a.target_level, dry_run=a.dry_run)
    if not result.get("ok"):
        sys.exit(1)


if __name__ == "__main__":
    main()
