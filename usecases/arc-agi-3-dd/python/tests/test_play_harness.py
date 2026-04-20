"""Regression tests for the ls20 play harness.

Covers the harness-layer introspection that the multi-level fixes depend on:
  - _query_level_state  (authoritative per-level rotation / win / cross data)
  - _auto_scan_level    (per-level element_records refresh on transition)
  - build_postgame_user_message + build_play_user_message (prompt templates
    accept the new kwargs: prior_kb, level_completion_events, prev_level_notes)

These are integration-leaning tests that spin up a real ls20 Arcade env and
drive it through the level-0 solution we confirmed by direct simulation.
They are cheap (fractions of a second) and deterministic.

Run (from the python/ dir or repo root, with ARC-AGI-3 repo on sys.path):
    python -m pytest tests/test_play_harness.py -v
or directly:
    python tests/test_play_harness.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY_DIR = HERE.parent
sys.path.insert(0, str(PY_DIR))

ARC_REPO = Path(os.environ.get("ARC_AGI_3_REPO", r"C:\_backup\github\arc-agi-3"))
if str(ARC_REPO) not in sys.path:
    sys.path.insert(0, str(ARC_REPO))


def _make_env():
    from arc_agi import Arcade, OperationMode
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ARC_REPO / "environment_files"),
    )
    return arc.make("ls20-9607627b")


def _drive_to_level_1(env):
    """Execute the confirmed L0 solution: agent ends on L1 start."""
    from arcengine import GameAction
    env.reset()
    path = (
        [GameAction.ACTION3] * 3
        + [GameAction.ACTION1] * 3
        + [GameAction.ACTION1, GameAction.ACTION4, GameAction.ACTION4,
           GameAction.ACTION4, GameAction.ACTION1, GameAction.ACTION1,
           GameAction.ACTION1]
    )
    obs = None
    for a in path:
        obs = env.step(a)
    return obs


# ---------------------------------------------------------------------------
# _query_level_state
# ---------------------------------------------------------------------------

def test_query_level_state_at_reset():
    from run_play import _query_level_state
    env = _make_env()
    env.reset()
    s = _query_level_state(env)
    assert s is not None
    assert s["level_index"] == 0
    assert s["agent_cursor"] == [45, 36]
    assert s["cross_positions"] == [[30, 21]]
    assert s["win_positions"] == [[10, 36]]
    assert s["start_rot_deg"] == 270
    assert s["start_rot_idx"] == 3
    assert s["cur_rot_idx"] == 3
    assert s["goal_rot_idx"] == 0
    assert s["advances_remaining"] == 1
    assert s["aligned"] is False


def test_query_level_state_after_cross_visit_aligns():
    from run_play import _query_level_state
    from arcengine import GameAction
    env = _make_env()
    env.reset()
    for a in [GameAction.ACTION3] * 3 + [GameAction.ACTION1] * 3:
        env.step(a)
    s = _query_level_state(env)
    assert s["cur_rot_idx"] == 0
    assert s["advances_remaining"] == 0
    assert s["aligned"] is True
    # Win position is still [10,36] on L0 — verified by prior evidence.
    assert s["win_positions"][0] == [10, 36]


def test_query_level_state_refreshes_after_auto_advance():
    """After L0 win, game auto-advances; all L1 values differ from L0."""
    from run_play import _query_level_state
    env = _make_env()
    obs = _drive_to_level_1(env)
    assert obs.levels_completed == 1, "L0 should be completed by the canonical path"
    s = _query_level_state(env)
    assert s["level_index"] == 1
    assert s["agent_cursor"] == [40, 31]
    assert s["start_rot_deg"] == 0
    assert s["goal_rot_idx"] == 3
    assert s["advances_remaining"] == 3, (
        "L1 needs 3 cross visits (StartRot 0° → GoalRot 270°), "
        "NOT 1 like L0"
    )
    assert s["aligned"] is False
    assert s["win_positions"] == [[40, 16]]
    assert s["cross_positions"] == [[45, 51]]


# ---------------------------------------------------------------------------
# _auto_scan_level
# ---------------------------------------------------------------------------

def test_auto_scan_level_produces_agent_and_win_records():
    from run_play import _auto_scan_level
    env = _make_env()
    env.reset()
    records = _auto_scan_level(env)
    assert records, "auto-scan should return at least the agent record"
    # Exactly one agent record, named "agent_cursor", function="agent"
    agents = [r for r in records.values() if r["function"] == "agent"]
    assert len(agents) == 1
    assert agents[0]["name"] == "agent_cursor"
    # bbox is a 5-wide/tall block centered on the agent's cursor cell
    r0, c0, r1, c1 = agents[0]["bbox"]
    assert (r1 - r0, c1 - c0) == (4, 4)

    # At least one switch (cross) and one target (win marker)
    assert any(r["function"] == "switch" for r in records.values())
    assert any(r["function"] == "target" for r in records.values())


def test_auto_scan_level_refreshes_after_advance():
    """L0 and L1 scans must differ — positions change."""
    from run_play import _auto_scan_level
    env = _make_env()
    env.reset()
    l0 = _auto_scan_level(env)
    _drive_to_level_1(env)
    l1 = _auto_scan_level(env)
    # Agent position differs
    a0 = next(r for r in l0.values() if r["function"] == "agent")["bbox"]
    a1 = next(r for r in l1.values() if r["function"] == "agent")["bbox"]
    assert a0 != a1, "agent bbox should change across the level transition"
    # Cross position differs
    c0 = [r for r in l0.values() if r["function"] == "switch"][0]["bbox"]
    c1 = [r for r in l1.values() if r["function"] == "switch"][0]["bbox"]
    assert c0 != c1, "cross bbox should change across the level transition"


# ---------------------------------------------------------------------------
# Prompt builders accept new kwargs
# ---------------------------------------------------------------------------

def test_build_postgame_user_message_includes_observed_facts():
    from play_prompts import build_postgame_user_message
    msg = build_postgame_user_message(
        game_id="ls20-test",
        outcome="PARTIAL_1_of_7",
        turns=2, final_state="NOT_FINISHED",
        levels_completed=1, win_levels=7,
        action_effects={"ACTION1": (-5, 0)},
        working_knowledge="(wk)",
        command_trace=[{"turn": 1, "command": "MOVE_TO", "steps_taken": 6,
                        "rationale": "to cross"}],
        prior_kb="PRIOR KB TEXT",
        initial_lc=0, final_lc=1,
        level_completion_events=[{"turn": 2, "from_level": 0, "to_level": 1,
                                  "win_levels": 7}],
        walls_learned=[(15, 36, "ACTION1")],
    )
    assert "PRIOR_KNOWLEDGE_BASE" in msg
    assert "PRIOR KB TEXT" in msg
    assert "HARNESS_OBSERVED_FACTS" in msg
    assert "level_completion_events" in msg
    assert '"from_level": 0' in msg
    assert '"to_level": 1' in msg
    assert "final_levels_completed:   1" in msg
    assert "win_levels_required:      7" in msg
    assert "ACTION1" in msg


def test_build_postgame_user_message_blank_kb_is_marked():
    from play_prompts import build_postgame_user_message
    msg = build_postgame_user_message(
        game_id="x", outcome="NOT_FINISHED", turns=0,
        final_state="NOT_FINISHED", levels_completed=0, win_levels=7,
        action_effects={}, working_knowledge="", command_trace=[],
    )
    assert "(none — this is the first recorded session for this game)" in msg


def test_build_play_user_message_prev_level_notes_roundtrip():
    from play_prompts import build_play_user_message
    notes = "Level 0 completed at turn 2.\nStrategy: cross once, then win."
    msg = build_play_user_message(
        turn=3, game_id="g", state="NOT_FINISHED",
        levels_completed=1, win_levels=7, budget_remaining=35,
        cursor_pos=(40, 31), action_effects={"ACTION1": (-5, 0)},
        working_knowledge="wk",
        recent_history=[], command_result=None, frame_text="[frame]",
        prev_level_notes=notes,
    )
    assert "PREV_LEVEL_NOTES" in msg
    assert "Level 0 completed at turn 2." in msg


def test_build_play_user_message_empty_prev_level_notes_marked():
    from play_prompts import build_play_user_message
    msg = build_play_user_message(
        turn=1, game_id="g", state="NOT_FINISHED",
        levels_completed=0, win_levels=7, budget_remaining=42,
        cursor_pos=(45, 36), action_effects={},
        working_knowledge="wk",
        recent_history=[], command_result=None, frame_text="[frame]",
    )
    assert "(no prior sub-level completed this session)" in msg


# ---------------------------------------------------------------------------
# Direct-runner entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback
    tests = [
        test_query_level_state_at_reset,
        test_query_level_state_after_cross_visit_aligns,
        test_query_level_state_refreshes_after_auto_advance,
        test_auto_scan_level_produces_agent_and_win_records,
        test_auto_scan_level_refreshes_after_advance,
        test_build_postgame_user_message_includes_observed_facts,
        test_build_postgame_user_message_blank_kb_is_marked,
        test_build_play_user_message_prev_level_notes_roundtrip,
        test_build_play_user_message_empty_prev_level_notes_marked,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
