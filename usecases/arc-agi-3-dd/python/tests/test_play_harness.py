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
        session_id="trial_TEST_play",
    )
    # Template renamed PRIOR_KNOWLEDGE_BASE -> PRIOR_KB when the postgame
    # was restructured around the provenance-tagged KB format.
    assert "PRIOR_KB" in msg
    assert "PRIOR KB TEXT" in msg
    assert "HARNESS_OBSERVED_FACTS" in msg
    assert "level_completion_events" in msg
    assert '"from_level": 0' in msg
    assert '"to_level": 1' in msg
    assert "final_levels_completed:   1" in msg
    assert "win_levels_required:      7" in msg
    assert "ACTION1" in msg
    assert "trial_TEST_play" in msg   # session_id passed through


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
# Raw state-variable vector
# ---------------------------------------------------------------------------

def test_query_state_vector_at_reset():
    from run_play import _query_state_vector
    env = _make_env()
    env.reset()
    sv = _query_state_vector(env)
    assert sv is not None
    # At reset: rotation index 3 (StartRotation=270°), 3 lives, step counter 42
    assert sv.get("game.cklxociuu") == 3
    assert sv.get("game.aqygnziho") == 3
    assert sv.get("game.level_index") == 0
    assert sv.get("step_counter_ui.current_steps") == 42
    assert sv.get("step_counter_ui.osgviligwp") == 42


def test_query_state_vector_tracks_rotation_and_budget():
    """After cross visit: cklxociuu changes + step counter decreases."""
    from run_play import _query_state_vector
    from arcengine import GameAction
    env = _make_env()
    env.reset()
    sv0 = _query_state_vector(env)
    # Any action consumes step-counter budget (whether motion succeeds or not)
    for a in [GameAction.ACTION3] * 3 + [GameAction.ACTION1] * 3:
        env.step(a)
    sv1 = _query_state_vector(env)
    assert sv1["step_counter_ui.current_steps"] < sv0["step_counter_ui.current_steps"]
    assert sv1["game.cklxociuu"] == 0, "cross visit should advance rotation 3->0"


# ---------------------------------------------------------------------------
# Predict-observe-diff
# ---------------------------------------------------------------------------

def test_prediction_error_matches_observed_cursor_pos():
    from run_play import _compute_prediction_error
    # Fake obs + cr: cursor ended at [30, 21]
    class FakeObs:
        levels_completed = 0
        class state:
            name = "NOT_FINISHED"
    cr = {"cursor_pos_after": [30, 21],
          "rotation_tracker": {"level_completed": False}}
    pe = _compute_prediction_error(
        predict={"cursor_pos_after": [30, 21]},
        obs=FakeObs(),
        cr=cr,
        lstate={"level_index": 0},
    )
    assert pe["had_predictions"] is True
    assert "cursor_pos_after" in pe["predicted_matched"]
    assert pe["predicted_mismatch"] == {}


def test_prediction_error_flags_mismatch():
    from run_play import _compute_prediction_error
    class FakeObs:
        levels_completed = 0
        class state:
            name = "NOT_FINISHED"
    cr = {"cursor_pos_after": [35, 21],
          "rotation_tracker": {"level_completed": False}}
    pe = _compute_prediction_error(
        predict={"cursor_pos_after": [30, 21], "level_completed": True},
        obs=FakeObs(),
        cr=cr,
        lstate={"level_index": 0},
    )
    assert "cursor_pos_after" in pe["predicted_mismatch"]
    assert "level_completed" in pe["predicted_mismatch"]
    assert pe["predicted_mismatch"]["cursor_pos_after"]["observed"] == [35, 21]


def test_prediction_error_flags_unrecognized_field():
    from run_play import _compute_prediction_error
    class FakeObs:
        levels_completed = 0
        class state:
            name = "NOT_FINISHED"
    pe = _compute_prediction_error(
        predict={"some_made_up_field": "value"},
        obs=FakeObs(),
        cr={"cursor_pos_after": [0, 0]},
        lstate=None,
    )
    assert "some_made_up_field" in pe["unrecognized_fields"]


def test_prediction_error_empty_when_no_predictions():
    from run_play import _compute_prediction_error
    class FakeObs:
        levels_completed = 0
    pe = _compute_prediction_error(
        predict=None, obs=FakeObs(), cr={}, lstate=None,
    )
    assert pe["had_predictions"] is False
    assert pe["predicted_matched"] == []
    assert pe["predicted_mismatch"] == {}


# ---------------------------------------------------------------------------
# kb_tools: strip-authored and provenance-summary
# ---------------------------------------------------------------------------

_TEST_KB = """\
# test KB

prose goes here.

```yaml
version: 1
game_id: test
beliefs:
  - id: B1
    provenance: discovered
    discovered_in: [session_A]
    statement: TUTOR saw the action effects
  - id: B2
    provenance: authored-by-claude-code
    statement: from source reading
  - id: B3
    provenance: corroborated
    corroborating_sessions: [session_A, session_B]
    statement: authored-then-verified twice
  - id: B4
    provenance: corroborated
    corroborating_sessions: [session_A]
    statement: authored-then-verified once (not enough)
hypotheses:
  - id: H1
    provenance: discovered
    status: proposed
    statement: TUTOR raised this from an anomaly
  - id: H2
    provenance: authored-interpretation
    status: supported
    statement: claude-code interpretation of raw sensors
traps:
  - id: T1
    provenance: discovered
    description: TUTOR hit this
  - id: T2
    provenance: authored-by-claude-code
    description: from source
```

trailing prose.
"""


def test_strip_authored_removes_authored_and_unconfirmed_corroborated():
    from kb_tools import strip_authored
    stripped = strip_authored(_TEST_KB)
    # Kept (discovered): B1, H1, T1
    assert "TUTOR saw the action effects" in stripped
    assert "TUTOR raised this from an anomaly" in stripped
    assert "TUTOR hit this" in stripped
    # Kept (corroborated with >=2 sessions): B3
    assert "authored-then-verified twice" in stripped
    # Removed: B2, B4 (not enough corroboration), H2, T2
    assert "from source reading" not in stripped
    assert "authored-then-verified once" not in stripped
    assert "claude-code interpretation" not in stripped
    # Prose outside YAML block preserved
    assert "prose goes here" in stripped
    assert "trailing prose" in stripped


def test_strip_authored_returns_unchanged_when_no_yaml_block():
    from kb_tools import strip_authored
    text = "just prose, no yaml"
    assert strip_authored(text) == text


def test_provenance_summary_counts_correctly():
    from kb_tools import provenance_summary
    summary = provenance_summary(_TEST_KB)
    assert summary["beliefs"]["discovered"] == 1
    assert summary["beliefs"]["authored-by-claude-code"] == 1
    assert summary["beliefs"]["corroborated"] == 2
    assert summary["hypotheses"]["discovered"] == 1
    assert summary["hypotheses"]["authored-interpretation"] == 1
    assert summary["traps"]["discovered"] == 1
    assert summary["traps"]["authored-by-claude-code"] == 1


def test_strip_authored_on_current_ls20_kb():
    """Smoke test against the real shipped KB so format drift is caught."""
    from kb_tools import strip_authored, provenance_summary
    kb_path = PY_DIR.parent / "benchmarks" / "knowledge_base" / "ls20-9607627b.md"
    if not kb_path.exists():
        return   # don't fail if KB hasn't been authored yet
    text = kb_path.read_text(encoding="utf-8")
    summary_full = provenance_summary(text)
    stripped = strip_authored(text)
    summary_stripped = provenance_summary(stripped)
    # After stripping, there should be strictly fewer authored entries
    full_authored = sum(
        counts.get("authored-by-claude-code", 0) + counts.get("authored-interpretation", 0)
        for counts in summary_full.values()
    )
    stripped_authored = sum(
        counts.get("authored-by-claude-code", 0) + counts.get("authored-interpretation", 0)
        for counts in summary_stripped.values()
    )
    assert full_authored > 0, "current KB should have some authored items"
    assert stripped_authored == 0, "stripped KB must have zero authored items"
    # Discovered items should survive
    full_discovered = sum(counts.get("discovered", 0) for counts in summary_full.values())
    stripped_discovered = sum(counts.get("discovered", 0) for counts in summary_stripped.values())
    assert stripped_discovered == full_discovered


# ---------------------------------------------------------------------------
# apply_patch tests
# ---------------------------------------------------------------------------

_PATCHABLE_KB = """
Prose header.

```yaml
version: 1
game_id: test
beliefs:
  - id: B1
    provenance: discovered
    discovered_in: [s1]
    statement: original belief
hypotheses:
  - id: H1
    provenance: discovered
    status: proposed
    statement: hypothesis to be supported
    evidence:
      supporting: []
traps: []
open_experiments:
  - id: E1
    provenance: discovered
    description: run test X
```

Prose footer.
"""


def test_apply_patch_adds_belief():
    from kb_tools import apply_patch
    patch = {"belief_updates": [
        {"op": "add", "id": "B2", "provenance": "discovered",
         "discovered_in": ["s2"], "statement": "new belief from session s2"},
    ], "hypothesis_updates": [], "trap_updates": [], "open_experiment_updates": []}
    updated, warns = apply_patch(_PATCHABLE_KB, patch)
    assert "new belief from session s2" in updated
    assert "B2" in updated
    assert not warns


def test_apply_patch_skips_duplicate_add():
    from kb_tools import apply_patch
    patch = {"belief_updates": [
        {"op": "add", "id": "B1", "provenance": "discovered",
         "discovered_in": ["s2"], "statement": "duplicate id"},
    ], "hypothesis_updates": [], "trap_updates": [], "open_experiment_updates": []}
    updated, warns = apply_patch(_PATCHABLE_KB, patch)
    # B1 already exists — should not add
    assert updated.count("original belief") == 1  # only once
    assert updated.count("duplicate id") == 0
    assert any("already exists" in w for w in warns)


def test_apply_patch_updates_hypothesis_status():
    from kb_tools import apply_patch
    patch = {"belief_updates": [], "hypothesis_updates": [
        {"op": "update_status", "id": "H1", "status": "supported",
         "evidence_session": "s2", "note": "prediction confirmed"},
    ], "trap_updates": [], "open_experiment_updates": []}
    updated, warns = apply_patch(_PATCHABLE_KB, patch)
    assert "supported" in updated
    assert "prediction confirmed" in updated
    assert not warns


def test_apply_patch_corroborates_belief():
    from kb_tools import apply_patch
    patch = {"belief_updates": [
        {"op": "corroborate", "id": "B1", "session_id": "s2"},
    ], "hypothesis_updates": [], "trap_updates": [], "open_experiment_updates": []}
    updated, warns = apply_patch(_PATCHABLE_KB, patch)
    assert "s2" in updated
    assert "corroborated" in updated
    assert not warns


def test_apply_patch_closes_experiment():
    from kb_tools import apply_patch
    patch = {"belief_updates": [], "hypothesis_updates": [],
             "trap_updates": [],
             "open_experiment_updates": [
                 {"op": "close", "id": "E1", "session_id": "s2",
                  "result": "test confirmed X works"},
             ]}
    updated, warns = apply_patch(_PATCHABLE_KB, patch)
    assert "closed" in updated
    assert "test confirmed X works" in updated
    assert not warns


def test_apply_patch_warns_on_missing_id():
    from kb_tools import apply_patch
    patch = {"belief_updates": [], "hypothesis_updates": [
        {"op": "update_status", "id": "H99", "status": "refuted",
         "evidence_session": "s2"},
    ], "trap_updates": [], "open_experiment_updates": []}
    updated, warns = apply_patch(_PATCHABLE_KB, patch)
    assert any("H99" in w for w in warns)


def test_apply_patch_empty_patch_unchanged():
    from kb_tools import apply_patch, provenance_summary
    patch = {"belief_updates": [], "hypothesis_updates": [],
             "trap_updates": [], "open_experiment_updates": []}
    updated, warns = apply_patch(_PATCHABLE_KB, patch)
    # Content should be equivalent (modulo YAML reserialisation)
    orig_summary  = provenance_summary(_PATCHABLE_KB)
    patch_summary = provenance_summary(updated)
    assert orig_summary == patch_summary
    assert not warns


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
        test_query_state_vector_at_reset,
        test_query_state_vector_tracks_rotation_and_budget,
        test_prediction_error_matches_observed_cursor_pos,
        test_prediction_error_flags_mismatch,
        test_prediction_error_flags_unrecognized_field,
        test_prediction_error_empty_when_no_predictions,
        test_strip_authored_removes_authored_and_unconfirmed_corroborated,
        test_strip_authored_returns_unchanged_when_no_yaml_block,
        test_provenance_summary_counts_correctly,
        test_strip_authored_on_current_ls20_kb,
        test_apply_patch_adds_belief,
        test_apply_patch_skips_duplicate_add,
        test_apply_patch_updates_hypothesis_status,
        test_apply_patch_corroborates_belief,
        test_apply_patch_closes_experiment,
        test_apply_patch_warns_on_missing_id,
        test_apply_patch_empty_patch_unchanged,
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
