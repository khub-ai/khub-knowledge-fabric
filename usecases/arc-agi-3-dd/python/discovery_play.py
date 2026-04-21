"""Strict-mode play loop.

Runs under the prime directive: no env._game access, no privileged
signals, no hardcoded function tags, no authored KB.  Only obs fields
and pixel analysis.

Architecture:
  1. Reset env; extract initial frame components.
  2. Bootstrap: mechanically probe each available direction action to
     learn (dr, dc) effects.  Zero LLM calls.
  3. Main loop: for each turn, send TUTOR the untagged components +
     known effects + recent outcomes; TUTOR picks a MOVE_TO target.
  4. Harness executes MOVE_TO via BFS over an EMPIRICALLY-DISCOVERED
     passable grid (every cell tentatively passable; failed moves
     accumulate (r, c, action) walls).
  5. After each TUTOR-decided command, observe obs.levels_completed
     and obs.state; stop on level advance or game over or max turns.

Budget cost per session:
  - Bootstrap: ~4 env.step calls (one per direction), 0 LLM.
  - Each TUTOR turn: ~$0.08-0.12 depending on prompt size.
  - BFS navigation costs env.step calls but no LLM.

Writes:
  - session_dir/play_log.jsonl     (per-turn records)
  - session_dir/working_kb.md      (postgame accumulated)
  - session_dir/manifest.json       (outcome summary)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ARC_REPO = Path(os.environ.get("ARC_AGI_3_REPO", r"C:\_backup\github\arc-agi-3"))
sys.path.insert(0, str(ARC_REPO))

from arc_agi import Arcade, OperationMode                           # noqa: E402

from dsl_executor import _normalise_frame                           # noqa: E402
from pixel_elements import extract_components, summarize_frame       # noqa: E402
from discovery_bootstrap import bootstrap_action_effects             # noqa: E402
from discovery_prompts import SYSTEM_DISCOVERY, USER_DISCOVERY_TEMPLATE  # noqa: E402
import backends                                                      # noqa: E402

TUTOR_MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _format_frame_text(grid: np.ndarray) -> str:
    rows = [", ".join(f"{int(v):2d}" for v in row) for row in grid]
    return "[\n" + ",\n".join(f"  [{r}]" for r in rows) + "\n]"


def _extract_json(text: str) -> dict:
    import re
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    first = text.find("{")
    if first < 0:
        raise ValueError(f"no JSON in reply: {text[:200]!r}")
    depth = 0
    for i in range(first, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[first:i + 1])
    raise ValueError("unterminated JSON")


def _components_summary(
    comps:      list[dict],
    agent_fp:   tuple[int, int, tuple] | None,
    max_n:      int = 15,
) -> str:
    """Render components sorted by distinctiveness.

    distinctiveness = smaller total frame-area for that palette + smaller size.
    The agent component is excluded (agent position is shown separately).
    """
    if not comps:
        return "  (none)"
    # Compute total pixels per palette (a palette common across many pixels
    # is "common" / likely structural; rare palettes are more salient).
    palette_total: dict[int, int] = {}
    for c in comps:
        palette_total[c["palette"]] = palette_total.get(c["palette"], 0) + c["size"]

    filtered = []
    for c in comps:
        if agent_fp is not None and (
            c["palette"] == agent_fp[0] and c["size"] == agent_fp[1]
            and tuple(c["extent"]) == tuple(agent_fp[2])
        ):
            continue   # skip agent itself
        filtered.append(c)

    # Distinctiveness score: lower = more distinctive.
    # Primary: palette frequency across the whole frame (rare = distinctive).
    # Secondary: component size (small = more likely a landmark, big = structural).
    filtered.sort(key=lambda c: (palette_total[c["palette"]], c["size"]))

    out = []
    for c in filtered[:max_n]:
        rare_note = f"(pal_total={palette_total[c['palette']]})"
        out.append(
            f"  id={c['id']:2d} pal={c['palette']:2d} size={c['size']:4d} "
            f"bbox={c['bbox']} centroid={c['centroid']} extent={c['extent']} "
            f"fill={c['fill_ratio']} {rare_note}"
        )
    if len(filtered) > max_n:
        out.append(f"  ... ({len(filtered) - max_n} more)")
    return "\n".join(out)


def _action_effects_text(effects: dict[str, tuple[int, int]]) -> str:
    if not effects:
        return "  (none learned yet)"
    return "\n".join(f"  {a}: dr={dr:+d}, dc={dc:+d}" for a, (dr, dc) in effects.items())


def _history_text(hist: list[dict], n: int = 3) -> str:
    if not hist:
        return "  (none)"
    lines = []
    for h in hist[-n:]:
        lines.append(
            f"  turn {h['turn']}: MOVE_TO {h.get('target')} -> "
            f"reached={h.get('reached')} "
            f"lc={h.get('lc_before')}->{h.get('lc_after')} "
            f"frame_change={h.get('frame_diff_cells')} "
            f"rationale='{h.get('rationale', '')[:60]}'"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Harness-level BFS navigator for STRICT mode
# ---------------------------------------------------------------------------
#
# No hardcoded wall palettes: walls are accumulated EMPIRICALLY -- every
# (r, c, action) for which the agent failed to move (frame changed less
# than a movement-threshold).  Passable grid is "every cell tentatively
# passable".  If BFS can't find a path, we return the closest reachable.

def _bfs_plan(
    start:          tuple[int, int],
    target:         tuple[int, int],
    action_effects: dict[str, tuple[int, int]],
    walls:          set[tuple[int, int, str]],
    grid_shape:     tuple[int, int] = (64, 64),
    max_steps:      int = 60,
) -> Optional[list[str]]:
    H, W = grid_shape
    move_actions = {a: (dr, dc) for a, (dr, dc) in action_effects.items()
                    if dr != 0 or dc != 0}
    if not move_actions or start == target:
        return None if start != target else []

    queue: deque = deque([(start, [])])
    visited: dict[tuple[int, int], int] = {start: 0}

    best_pos = start
    best_dist = (start[0] - target[0]) ** 2 + (start[1] - target[1]) ** 2
    best_path: list[str] = []

    while queue:
        (r, c), path = queue.popleft()
        d = (r - target[0]) ** 2 + (c - target[1]) ** 2
        if d < best_dist:
            best_dist = d
            best_pos = (r, c)
            best_path = path
        if len(path) >= max_steps:
            continue
        for action, (dr, dc) in move_actions.items():
            if (r, c, action) in walls:
                continue
            nr = max(0, min(H - 1, r + dr))
            nc = max(0, min(W - 1, c + dc))
            if (nr, nc) == (r, c):
                continue   # boundary clamp no-op
            steps = len(path) + 1
            if (nr, nc) in visited and visited[(nr, nc)] <= steps:
                continue
            visited[(nr, nc)] = steps
            new_path = path + [action]
            if (nr, nc) == target:
                return new_path
            queue.append(((nr, nc), new_path))

    # Target unreachable; return nearest-reachable path (may be empty).
    return best_path if best_path else None


# ---------------------------------------------------------------------------
# Pixel-derived cursor tracking (strict-mode replacement for _agent_cursor_from_game)
# ---------------------------------------------------------------------------

def _agent_cursor_from_frame(
    frame: np.ndarray,
    agent_fingerprint: tuple[int, int, tuple] | None,
) -> tuple[int, int] | None:
    """Find the agent's centroid by matching the known fingerprint
    (palette, size, extent) against current-frame components.
    Returns None if no unique match."""
    if agent_fingerprint is None:
        return None
    pal, size, extent = agent_fingerprint
    comps = extract_components(frame, min_size=2)
    matches = [c for c in comps
               if c["palette"] == pal and c["size"] == size
               and tuple(c["extent"]) == tuple(extent)]
    if len(matches) != 1:
        return None
    return (matches[0]["centroid"][0], matches[0]["centroid"][1])


# ---------------------------------------------------------------------------
# Main play loop
# ---------------------------------------------------------------------------

def run_session(
    game_id:        str,
    max_turns:      int = 8,
    session_dir:    Path | None = None,
    max_tokens:     int = 1500,
    model:          str = TUTOR_MODEL,
) -> dict:
    arc = Arcade(
        operation_mode   = OperationMode.OFFLINE,
        environments_dir = str(ARC_REPO / "environment_files"),
    )
    env = arc.make(game_id)
    obs = env.reset()
    frame0 = _normalise_frame(obs.frame)

    if session_dir:
        session_dir.mkdir(parents=True, exist_ok=True)
        log_path     = session_dir / "play_log.jsonl"
        manifest_path = session_dir / "manifest.json"
        log_path.write_text("")
    else:
        log_path = None
        manifest_path = None

    # ---- BOOTSTRAP (no LLM) ----
    available = [int(a) for a in obs.available_actions if int(a) != 0]
    print(f"[strict] probing actions {available}...")
    boot = bootstrap_action_effects(env, _normalise_frame, available)
    action_effects: dict[str, tuple[int, int]] = {
        a: tuple(e) for a, e in boot["action_effects_learned"].items()
    }
    print(f"[strict] action_effects learned: {action_effects}")
    agent_fp = None
    if boot["agent_candidates"]:
        agent_fp = boot["agent_candidates"][0][0]   # top consensus fingerprint
        print(f"[strict] agent fingerprint: palette={agent_fp[0]} size={agent_fp[1]} extent={agent_fp[2]}")

    # Post-bootstrap state
    obs_after_boot = env.reset()   # reset after probing so turn 1 starts at spawn
    frame = _normalise_frame(obs_after_boot.frame)
    obs = obs_after_boot
    walls: set[tuple[int, int, str]] = set()

    # ---- TURN LOOP ----
    history: list[dict] = []
    cost_usd_total = 0.0
    turns_used = 0
    level_advanced = False
    final_state = "NOT_FINISHED"
    initial_lc = int(obs.levels_completed)

    for turn in range(1, max_turns + 1):
        turns_used = turn
        if obs.state.name != "NOT_FINISHED":
            final_state = obs.state.name
            break
        if int(obs.levels_completed) > initial_lc:
            level_advanced = True
            final_state = obs.state.name
            break

        # Build turn prompt ingredients
        frame = _normalise_frame(obs.frame)
        comps = extract_components(frame, min_size=2)
        agent_pos = _agent_cursor_from_frame(frame, agent_fp)
        agent_comp = next((c for c in comps
                          if agent_fp and c["palette"] == agent_fp[0]
                          and c["size"] == agent_fp[1]
                          and tuple(c["extent"]) == tuple(agent_fp[2])), None)

        # Build tried-targets list (failed or no-advance targets this session).
        tried = []
        for h in history:
            if not h.get("reached") or h.get("lc_after", 0) <= h.get("lc_before", 0):
                t = h.get("target")
                if t and tuple(t) not in {tuple(x) for x in tried}:
                    tried.append(list(t))
        tried_text = "  (none yet)" if not tried else \
            "\n".join(f"  - {t}" for t in tried)

        user_msg = USER_DISCOVERY_TEMPLATE.format(
            turn            = turn,
            state           = obs.state.name,
            lc              = int(obs.levels_completed),
            win_levels      = int(obs.win_levels),
            actions         = [int(a) for a in obs.available_actions],
            agent_pal       = agent_comp["palette"] if agent_comp else "?",
            agent_size      = agent_comp["size"] if agent_comp else "?",
            agent_extent    = agent_comp["extent"] if agent_comp else "?",
            agent_r         = agent_pos[0] if agent_pos else "?",
            agent_c         = agent_pos[1] if agent_pos else "?",
            action_effects  = _action_effects_text(action_effects),
            components      = _components_summary(comps, agent_fp),
            hist_n          = min(3, len(history)),
            history         = _history_text(history),
            tried_targets   = tried_text,
        )

        print(f"\n[turn {turn}] calling TUTOR (agent at {agent_pos})...")
        t0 = time.time()
        rsp = backends.call_anthropic(
            model      = model,
            system     = SYSTEM_DISCOVERY,
            user       = user_msg,
            max_tokens = max_tokens,
        )
        latency_ms = int((time.time() - t0) * 1000)
        reply_text = rsp.get("reply", "") or ""
        in_tok  = rsp.get("input_tokens",  0)
        out_tok = rsp.get("output_tokens", 0)
        cost    = rsp.get("cost_usd", 0.0)
        cost_usd_total += cost

        try:
            cmd = _extract_json(reply_text)
        except Exception as e:
            print(f"[turn {turn}] bad JSON from TUTOR: {e}")
            break

        target = cmd.get("args", {}).get("target_pos")
        rationale = cmd.get("rationale", "")
        revise = cmd.get("revise", "")
        print(f"[turn {turn}] rationale: {rationale[:100]}")
        print(f"[turn {turn}] target:    {target}  (${cost:.3f}, {in_tok}+{out_tok} tok, {latency_ms}ms)")

        if not target or len(target) != 2:
            print(f"[turn {turn}] missing target_pos; stopping")
            break

        target_t = (int(target[0]), int(target[1]))
        if agent_pos is None:
            print(f"[turn {turn}] agent_pos unknown; stopping")
            break

        # Plan via BFS
        path = _bfs_plan(agent_pos, target_t, action_effects, walls)
        if not path:
            print(f"[turn {turn}] BFS found no path from {agent_pos} to {target_t}")
            history.append({
                "turn":             turn,
                "target":           list(target_t),
                "reached":          False,
                "rationale":        rationale,
                "revise":           revise,
                "lc_before":        int(obs.levels_completed),
                "lc_after":         int(obs.levels_completed),
                "frame_diff_cells": 0,
                "cost_usd":         cost,
                "note":             "no path",
            })
            continue

        # Execute path; re-plan on wall hits up to _MAX_REROUTES times.
        lc_before = int(obs.levels_completed)
        cur_pos = agent_pos
        exec_frame_before = frame.copy()
        _MAX_REROUTES = 4
        _MAX_TOTAL_STEPS = 35
        total_steps_this_cmd = 0
        reroutes_done = 0
        current_plan = list(path)
        while current_plan and total_steps_this_cmd < _MAX_TOTAL_STEPS:
            action_name = current_plan.pop(0)
            act_int = int(action_name.replace("ACTION", ""))
            obs = env.step(act_int)
            total_steps_this_cmd += 1
            cur_frame = _normalise_frame(obs.frame)
            new_pos = _agent_cursor_from_frame(cur_frame, agent_fp)
            if new_pos is None:
                break
            if new_pos == cur_pos:
                # Blocked. Learn wall and re-plan from here.
                walls.add((cur_pos[0], cur_pos[1], action_name))
                if reroutes_done < _MAX_REROUTES:
                    reroutes_done += 1
                    current_plan = _bfs_plan(cur_pos, target_t, action_effects, walls) or []
                    continue
                break
            cur_pos = new_pos
            if obs.state.name != "NOT_FINISHED" or int(obs.levels_completed) > lc_before:
                break
            if not current_plan:
                # Reached end of plan without hitting target -- re-plan
                # in case BFS's nearest-reachable left us short.
                if cur_pos != target_t and reroutes_done < _MAX_REROUTES:
                    reroutes_done += 1
                    current_plan = _bfs_plan(cur_pos, target_t, action_effects, walls) or []

        lc_after = int(obs.levels_completed)
        reached = (cur_pos == target_t)
        frame_after = _normalise_frame(obs.frame)
        diff_cells = int(np.sum(exec_frame_before != frame_after))
        history.append({
            "turn":             turn,
            "target":           list(target_t),
            "reached":          reached,
            "rationale":        rationale,
            "revise":           revise,
            "lc_before":        lc_before,
            "lc_after":         lc_after,
            "frame_diff_cells": diff_cells,
            "cost_usd":         cost,
        })
        print(f"[turn {turn}] executed {len(path)}-step path; "
              f"reached={reached} cur_pos={cur_pos} lc={lc_before}->{lc_after} "
              f"diff_cells={diff_cells} walls_learned_so_far={len(walls)}")

        if log_path:
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "turn":      turn,
                    "target":    list(target_t),
                    "rationale": rationale,
                    "revise":    revise,
                    "path":      path,
                    "reached":   reached,
                    "lc_before": lc_before,
                    "lc_after":  lc_after,
                    "diff_cells": diff_cells,
                    "cost_usd":  cost,
                }) + "\n")

        if lc_after > initial_lc:
            level_advanced = True
            final_state = obs.state.name
            print(f"[turn {turn}] *** LEVEL {lc_before} -> {lc_after} ADVANCED ***")
            break

    # ---- WRAP-UP ----
    result = {
        "game_id":             game_id,
        "outcome":             "LEVEL_ADVANCED" if level_advanced else "NO_ADVANCE",
        "turns_used":          turns_used,
        "initial_lc":          initial_lc,
        "final_lc":            int(obs.levels_completed),
        "final_state":         final_state,
        "cost_usd_total":      round(cost_usd_total, 4),
        "action_effects":      {a: list(e) for a, e in action_effects.items()},
        "walls_learned":       [list(w) for w in walls],
        "history":             history,
    }
    print("\n" + "=" * 60)
    print(f"RESULT: {result['outcome']}  turns={result['turns_used']} "
          f"lc={result['initial_lc']}->{result['final_lc']} "
          f"cost=${result['cost_usd_total']:.4f}")
    print("=" * 60)

    if manifest_path:
        manifest_path.write_text(json.dumps(result, indent=2))

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game",        default="ls20-9607627b")
    ap.add_argument("--max-turns",   type=int, default=8)
    ap.add_argument("--session-dir",
                    default=None,
                    help="Where to write log + manifest; defaults to benchmarks/sessions/trial_<ts>_strict")
    ap.add_argument("--max-tokens",  type=int, default=1500)
    a = ap.parse_args()

    if a.session_dir:
        sd = Path(a.session_dir)
    else:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        sd = HERE.parent / "benchmarks" / "sessions" / f"trial_{ts}_strict"

    result = run_session(
        game_id     = a.game,
        max_turns   = a.max_turns,
        session_dir = sd,
        max_tokens  = a.max_tokens,
    )
    sys.exit(0 if result["outcome"] == "LEVEL_ADVANCED" else 1)


if __name__ == "__main__":
    main()
