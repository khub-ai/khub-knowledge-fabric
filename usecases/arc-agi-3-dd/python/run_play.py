"""Drive TUTOR through an ls20 L1 play session.

Bootstraps from a Round-2 session (revised assessment + optional prior
knowledge), then loops one-action-at-a-time using the live ARC env.

At each turn:
  1. Capture current frame.
  2. Build a CHANGE_REPORT vs the previous frame (using Round-1/2
     elements so CHANGE_REPORT tracks named entities).
  3. Call TUTOR with working_knowledge + recent_history + frame +
     CHANGE_REPORT.
  4. Execute the action.  Record.

Terminates on WIN / GAME_OVER / max_steps / repeated no-ops.

At end, call TUTOR once more with POSTGAME prompt and save the
resulting knowledge note alongside the play log.
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


ARC_REPO = Path(os.environ.get(
    "ARC_AGI_3_REPO", r"C:\_backup\github\arc-agi-3"
))
sys.path.insert(0, str(ARC_REPO))

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402


TUTOR_MODEL = "claude-sonnet-4-6"


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


def load_working_knowledge(
    round2_dir: Path, lessons_path: Path | None = None,
) -> tuple[str, dict]:
    """Build a compact prose 'working_knowledge' string from a Round-2 TUTOR
    reply, plus return the element records dict used for CHANGE_REPORT.

    If `lessons_path` is given, its contents are prepended as a
    LESSONS_FROM_LAST_RUN block that takes precedence over the Round-2
    assessment — this is how post-game notes from a prior play feed back
    into the next attempt.
    """
    r2 = json.loads((round2_dir / "tutor_round2_reply.json").read_text(encoding="utf-8"))
    assess = r2.get("assessment") or {}

    elements = assess.get("elements") or []
    element_records = {}
    for e in elements:
        eid = e.get("id")
        if eid is None:
            continue
        element_records[int(eid)] = {
            "bbox":     e.get("bbox"),
            "name":     e.get("name"),
            "function": e.get("function", "unknown"),
        }

    lines: list[str] = []
    if lessons_path is not None and lessons_path.exists():
        lines.append("LESSONS_FROM_LAST_RUN (YOU wrote this at the end of your")
        lines.append("previous play session.  It REPLACES any contradictory")
        lines.append("claim below — trust it over the Round-2 assessment and")
        lines.append("over any prior_knowledge):")
        lines.append(lessons_path.read_text(encoding="utf-8").strip())
        lines.append("")
        lines.append("---")
        lines.append("")
    lines.append("ELEMENTS (from your Round-2 revised assessment):")
    for e in elements:
        lines.append(
            f"  #{e.get('id')} {e.get('name','?')} "
            f"bbox={e.get('bbox')} fn={e.get('function','?')} "
            f"— {e.get('rationale','')}"
        )
    strat = assess.get("initial_strategy") or {}
    lines.append("")
    lines.append(f"PRIMARY_GOAL: {strat.get('primary_goal','?')}")
    if strat.get("rationale"):
        lines.append(f"STRATEGY_NOTES: {strat['rationale']}")
    open_qs = strat.get("open_questions") or []
    if open_qs:
        lines.append("OPEN_QUESTIONS:")
        for q in open_qs:
            lines.append(f"  - {q}")
    # Include prior_knowledge if Round-2 session had one.
    prior_path = round2_dir / "prior_knowledge.txt"
    if prior_path.exists():
        lines.append("")
        lines.append("PRIOR_KNOWLEDGE (operator-injected, treat as given):")
        lines.append(prior_path.read_text(encoding="utf-8"))
    return "\n".join(lines), element_records


def _step_env(env, action_label: str):
    if action_label == "RESET":
        return env.reset()
    return env.step(GameAction[action_label])


def _cr_summary(cr: dict) -> dict:
    """One-line summary of a CHANGE_REPORT for the recent-history block."""
    motions = cr.get("element_motions") or []
    reliable_moved = [m for m in motions
                      if m.get("moved") and not m.get("tracker_unreliable")]
    counters = cr.get("counter_changes") or []
    counter_summary = (
        ",".join(f"{c.get('name','?')}:{c.get('before_fill','?')}->{c.get('after_fill','?')}"
                 for c in counters) or "none"
    )
    totals = cr.get("totals") or {}
    pm = cr.get("primary_motion")
    pm_summary = "none"
    if pm:
        pm_summary = (f"{pm.get('name','?')} dr={pm.get('dr')} dc={pm.get('dc')} "
                      f"-> {pm.get('post_bbox')}")
    return {
        "motions_count":   len(reliable_moved),
        "diff_cells":      totals.get("diff_cells"),
        "counter_summary": counter_summary,
        "primary_motion":  pm_summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round2-session", required=True,
                    help="Round-2 session dir (provides working_knowledge)")
    ap.add_argument("--lessons",
                    help="Path to a prior post_game_knowledge.md to inject as "
                         "LESSONS_FROM_LAST_RUN (takes precedence over Round-2)")
    ap.add_argument("--game", default="ls20-9607627b")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--sessions-dir", default=str(HERE.parent / "benchmarks" / "sessions"))
    ap.add_argument("--max-tokens", type=int, default=1500)
    a = ap.parse_args()

    round2_dir = Path(a.round2_session)
    lessons_path = Path(a.lessons) if a.lessons else None
    working_knowledge, element_records = load_working_knowledge(
        round2_dir, lessons_path=lessons_path,
    )

    trial_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_dir = Path(a.sessions_dir) / f"trial_{trial_id}_play"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "working_knowledge.md").write_text(working_knowledge, encoding="utf-8")

    arc = Arcade(
        operation_mode   = OperationMode.OFFLINE,
        environments_dir = str(ARC_REPO / "environment_files"),
    )
    env = arc.make(a.game)
    obs = env.reset()
    prev_grid = _normalise_frame(obs.frame)

    log_path = session_dir / "play_log.jsonl"
    log_fh = log_path.open("w", encoding="utf-8")

    action_trace: list[dict] = []
    recent_history: list[dict] = []
    final_state = obs.state.name if hasattr(obs.state, "name") else str(obs.state)
    t0 = time.time()

    for turn in range(1, a.max_steps + 1):
        cur_grid = _normalise_frame(obs.frame)
        state    = obs.state.name if hasattr(obs.state, "name") else str(obs.state)
        actions  = [f"ACTION{int(x)}" for x in obs.available_actions]
        lc       = int(obs.levels_completed)

        # CHANGE_REPORT vs previous frame.
        if turn == 1:
            change_report = None
        else:
            change_report = _build_change_report(prev_grid, cur_grid, element_records)

        user_msg = build_play_user_message(
            turn              = turn,
            game_id           = a.game,
            state             = state,
            levels_completed  = lc,
            win_levels        = int(obs.win_levels),
            action_labels     = actions,
            working_knowledge = working_knowledge,
            recent_history    = recent_history,
            change_report     = change_report,
            frame_text        = _format_frame_text(cur_grid),
        )

        try:
            rsp = backends.call_anthropic(
                model=TUTOR_MODEL, system=SYSTEM_PLAY, user=user_msg,
                image_b64=None, max_tokens=a.max_tokens,
            )
            reply_text = rsp["reply"]
            latency_ms = rsp["latency_ms"]
        except Exception as e:  # noqa: BLE001
            print(f"turn {turn}: TUTOR call failed: {e}")
            break

        try:
            decision = extract_json(reply_text)
        except Exception as e:  # noqa: BLE001
            print(f"turn {turn}: JSON parse error: {e}")
            log_fh.write(json.dumps({
                "turn": turn, "parse_error": str(e), "raw_reply": reply_text,
            }) + "\n")
            break

        action = decision.get("action", "").upper().strip()
        seq = decision.get("action_sequence") or [action]
        seq = [str(s).upper().strip() for s in seq if s]
        if not seq or seq[0] != action:
            seq = [action] + seq
        # Dedup adjacency: keep as-is but cap at 5.
        seq = seq[:5]
        rationale = decision.get("rationale", "")
        predict = decision.get("predict", {})
        revise  = decision.get("revise_knowledge", "")
        done_flag = bool(decision.get("done"))

        summary = _cr_summary(change_report or {})
        print(f"turn {turn:>2} state={state:<12} action={action:<10} "
              f"motions={summary['motions_count']} "
              f"diff={summary['diff_cells']} counter={summary['counter_summary']} "
              f"({latency_ms} ms)")
        if revise:
            print(f"         revise: {revise}")

        log_fh.write(json.dumps({
            "turn": turn, "state": state, "levels_completed": lc,
            "action": action, "rationale": rationale, "predict": predict,
            "revise_knowledge": revise, "done": done_flag,
            "change_report_summary": summary,
            "change_report": change_report,
            "latency_ms": latency_ms,
        }) + "\n")
        log_fh.flush()

        action_trace.append({"turn": turn, "action": action,
                             "rationale": rationale, "state": state})
        recent_history.append({
            "turn": turn, "action": action, "state": state, **summary,
        })

        if done_flag and state in ("WIN", "GAME_OVER"):
            final_state = state
            break

        prev_grid = cur_grid
        # Execute the full action_sequence back-to-back; stop early on
        # WIN / GAME_OVER / unavailable-action.
        seq_executed: list[str] = []
        step_err = None
        for act in seq:
            if act != "RESET" and act not in actions:
                step_err = f"action {act!r} not in AVAILABLE_ACTIONS {actions}"
                break
            try:
                obs = _step_env(env, act)
            except Exception as e:  # noqa: BLE001
                step_err = f"env.step({act}) failed: {e}"
                break
            seq_executed.append(act)
            _st = obs.state.name if hasattr(obs.state, "name") else str(obs.state)
            if _st in ("WIN", "GAME_OVER"):
                break
        if len(seq_executed) < len(seq):
            print(f"turn {turn}: executed {len(seq_executed)}/{len(seq)} of {seq} "
                  f"({step_err})")
        elif len(seq) > 1:
            print(f"         executed sequence {seq_executed}")

        final_state = obs.state.name if hasattr(obs.state, "name") else str(obs.state)
        if final_state in ("WIN", "GAME_OVER"):
            last_grid = _normalise_frame(obs.frame)
            last_cr   = _build_change_report(prev_grid, last_grid, element_records)
            log_fh.write(json.dumps({
                "turn": turn + 1, "final_state": final_state,
                "levels_completed": int(obs.levels_completed),
                "post_action_change_report": last_cr,
                "sequence_executed": seq_executed,
            }) + "\n")
            break

    log_fh.close()

    # Post-game knowledge capture.
    outcome = {"WIN": "WIN", "GAME_OVER": "LOSS"}.get(final_state, final_state)
    postgame_user = build_postgame_user_message(
        game_id           = a.game,
        outcome           = outcome,
        turns             = len(action_trace),
        final_state       = final_state,
        levels_completed  = int(obs.levels_completed),
        win_levels        = int(obs.win_levels),
        working_knowledge = working_knowledge,
        action_trace      = action_trace,
    )
    try:
        rsp = backends.call_anthropic(
            model=TUTOR_MODEL, system=SYSTEM_POSTGAME, user=postgame_user,
            image_b64=None, max_tokens=1500,
        )
        note = rsp["reply"].strip()
    except Exception as e:  # noqa: BLE001
        note = f"(post-game knowledge call failed: {e})"
    (session_dir / "post_game_knowledge.md").write_text(note, encoding="utf-8")

    manifest = {
        "trial_id":    trial_id,
        "game_id":     a.game,
        "round2_session": str(round2_dir),
        "lessons_from":   str(lessons_path) if lessons_path else None,
        "tutor_model": TUTOR_MODEL,
        "max_steps":   a.max_steps,
        "turns_played": len(action_trace),
        "final_state":  final_state,
        "outcome":      outcome,
        "wall_time_s":  round(time.time() - t0, 1),
        "created_at":   datetime.now(timezone.utc).isoformat(),
        "files": [
            "working_knowledge.md", "play_log.jsonl", "post_game_knowledge.md",
        ],
    }
    (session_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    print(f"\nOutcome: {outcome} ({final_state}), "
          f"{len(action_trace)} turns, wrote {session_dir}")


if __name__ == "__main__":
    main()
