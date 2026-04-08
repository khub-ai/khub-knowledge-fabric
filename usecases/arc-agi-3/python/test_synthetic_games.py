"""
test_synthetic_games.py — Tests for the synthetic game generator.

Validates:
  - Determinism: same seed → same game.
  - Solvability: ground-truth solver succeeds within budget.
  - Step semantics: applying solve()'s plan actually wins the game.
  - Frame shape and content invariants (agent visible, goal visible, etc.).
  - Batch generation produces distinct games.
  - Decoy NO_OP actions don't accidentally affect state.

Plain-assertion runner — no pytest. Run with:

    python test_synthetic_games.py
"""

from __future__ import annotations

import sys
import traceback

from proposer_schema import EffectType, GoalPredicateType, PreconditionType
from synthetic_games import (
    AGENT,
    GOAL,
    NavigationGame,
    TogglePuzzle,
    generate_batch,
    generate_navigation_game,
    generate_toggle_puzzle,
)


# =============================================================================
# Navigation tests
# =============================================================================

def test_navigation_determinism() -> None:
    g1 = generate_navigation_game(seed=42)
    g2 = generate_navigation_game(seed=42)
    assert g1.agent_start == g2.agent_start
    assert g1.goal_pos == g2.goal_pos
    assert g1.walls == g2.walls
    assert g1.ground_truth.action_models.keys() == g2.ground_truth.action_models.keys()
    for aid in g1.ground_truth.action_models:
        e1 = g1.ground_truth.action_models[aid]
        e2 = g2.ground_truth.action_models[aid]
        assert e1.effect_type == e2.effect_type
        assert e1.params == e2.params


def test_navigation_render_invariants() -> None:
    g = generate_navigation_game(seed=7)
    frame = g.reset()
    assert len(frame) == g.grid_h
    assert all(len(row) == g.grid_w for row in frame)
    # Agent visible.
    assert any(AGENT in row for row in frame)
    # Goal visible (unless overlapped by agent — generator avoids this).
    assert any(GOAL in row for row in frame)


def test_navigation_solvable_and_solution_executes() -> None:
    for seed in range(20):
        g = generate_navigation_game(seed=seed)
        plan = g.solve()
        assert plan is not None, f"seed {seed}: no solution found"
        assert len(plan) <= 64
        # Execute the plan and verify the game is won.
        g.reset()
        won = False
        for aid in plan:
            _, won, _ = g.step(aid)
        assert won, f"seed {seed}: executing plan did not win"


def test_navigation_blocked_action_is_noop() -> None:
    """If an action would walk into a wall, agent stays put."""
    walls = {(0, 1)}
    gt = NavigationGame(
        grid_h=3, grid_w=3, walls=walls,
        agent_start=(0, 0), goal_pos=(2, 2),
        ground_truth=__import__("synthetic_games").GroundTruth(
            action_models={
                "ACTION1": __import__("synthetic_games").HiddenEffect(
                    effect_type=EffectType.TRANSLATE,
                    precondition=PreconditionType.IF_BLOCKED,
                    params={"dr": 0, "dc": 1},
                ),
            },
            goal_predicate=GoalPredicateType.REACH,
            goal_params={"target_pos": (2, 2)},
        ),
    )
    gt.reset()
    _, won, _ = gt.step("ACTION1")
    assert gt.agent_pos == (0, 0), "agent should not have moved into wall"
    assert not won


# =============================================================================
# Toggle puzzle tests
# =============================================================================

def test_toggle_determinism() -> None:
    a = generate_toggle_puzzle(seed=11)
    b = generate_toggle_puzzle(seed=11)
    assert a.target_index == b.target_index
    assert a.start_index == b.start_index
    for aid in a.ground_truth.action_models:
        ea = a.ground_truth.action_models[aid]
        eb = b.ground_truth.action_models[aid]
        assert ea.effect_type == eb.effect_type
        assert ea.params == eb.params


def test_toggle_solvable_and_solution_executes() -> None:
    for seed in range(30):
        g = generate_toggle_puzzle(seed=seed)
        plan = g.solve()
        assert plan is not None, f"seed {seed}: no plan"
        g.reset()
        won = False
        for aid in plan:
            _, won, _ = g.step(aid)
        assert won, f"seed {seed}: plan failed to win"


def test_toggle_decoy_actions_are_inert() -> None:
    """Calling NO_OP decoy actions never wins or changes state."""
    g = generate_toggle_puzzle(seed=3)
    # Find a decoy.
    decoy = None
    for aid, eff in g.ground_truth.action_models.items():
        if eff.effect_type == EffectType.NO_OP:
            decoy = aid
            break
    if decoy is None:
        return  # generator chose all-toggle (cycle_len 1) — skip
    g.reset()
    before = g.index
    for _ in range(10):
        _, won, _ = g.step(decoy)
        assert not won
        assert g.index == before


def test_toggle_target_never_trivial() -> None:
    """Generator never produces a game already in goal state."""
    for seed in range(30):
        g = generate_toggle_puzzle(seed=seed)
        g.reset()
        assert not g.won


# =============================================================================
# Batch generation
# =============================================================================

def test_batch_generates_distinct_games() -> None:
    games = generate_batch("navigation", n=10)
    assert len(games) == 10
    starts = {g.agent_start for g in games}
    # Not all 10 should collapse to a single starting position.
    assert len(starts) > 1


def test_batch_toggle_distinct() -> None:
    games = generate_batch("toggle", n=10)
    assert len(games) == 10
    targets = {g.target_index for g in games}
    assert len(targets) >= 2


# =============================================================================
# Runner
# =============================================================================

def main() -> int:
    tests = [v for k, v in globals().items()
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
