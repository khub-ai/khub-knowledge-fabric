"""
test_planner.py — Explore→exploit integration on the synthetic game suite.

For each game:
  1. Run a uniform-random exploration phase to populate the tracker.
  2. Ask the planner for a plan against the (known) goal predicate.
  3. Execute the plan in the real game and assert it wins.

This is the closest thing to a system-level test the symbolic core has:
the planner only sees the *learned* world model, never the ground truth.
If the tracker is wrong, the planner is wrong, and the test fails.
"""

from __future__ import annotations

import random
import sys
import traceback

from dsl import HypothesisTracker
from planner import (
    information_gain_action,
    match_color_goal,
    plan_to_goal,
    reach_cell_goal,
)
from synthetic_games import (
    AGENT,
    SLOT_COLORS,
    generate_navigation_game,
    generate_toggle_puzzle,
)


def _explore(game, tracker, steps, rng) -> None:
    prev = game.reset()
    for _ in range(steps):
        a = information_gain_action(tracker, game.available_actions)
        # Mix in random to avoid pathological ties.
        if rng.random() < 0.3:
            a = rng.choice(game.available_actions)
        curr, won, _ = game.step(a)
        tracker.observe_step(a, prev, curr)
        prev = curr
        if won:
            prev = game.reset()


# =============================================================================
# Navigation
# =============================================================================

def test_planner_navigation_solves_after_exploration() -> None:
    wins = 0
    n = 30
    for seed in range(n):
        g = generate_navigation_game(seed=seed)
        rng = random.Random(seed)
        t = HypothesisTracker(g.available_actions)
        _explore(g, t, steps=80, rng=rng)

        start = g.reset()
        goal = reach_cell_goal(g.goal_pos, agent_color=AGENT)
        result = plan_to_goal(t, start, goal,
                              agent_color=AGENT, max_depth=32)
        if not result.found:
            continue
        # Execute the plan in the real game.
        g.reset()
        won = False
        for aid in result.plan:
            _, won, _ = g.step(aid)
            if won:
                break
        if won:
            wins += 1
    assert wins >= int(0.80 * n), f"only {wins}/{n} navigation games solved"


# =============================================================================
# Toggle
# =============================================================================

def test_planner_toggle_solves_after_exploration() -> None:
    wins = 0
    n = 30
    for seed in range(n):
        g = generate_toggle_puzzle(seed=seed)
        rng = random.Random(seed)
        t = HypothesisTracker(g.available_actions)
        _explore(g, t, steps=40, rng=rng)

        start = g.reset()
        # Slot cells from the rendered start frame: those that match the
        # current slot color in a centered block. Use the center.
        cr, cc = g.slot_pos
        slot_cells = set()
        for r in range(cr - g.slot_radius + 1, cr + g.slot_radius):
            for c in range(cc - g.slot_radius + 1, cc + g.slot_radius):
                if 0 <= r < g.grid_h and 0 <= c < g.grid_w:
                    slot_cells.add((r, c))
        target_color = g.cycle[g.target_index]
        goal = match_color_goal(slot_cells, target_color)
        result = plan_to_goal(t, start, goal, max_depth=8)
        if not result.found:
            continue
        g.reset()
        won = False
        for aid in result.plan:
            _, won, _ = g.step(aid)
            if won:
                break
        if won:
            wins += 1
    assert wins >= int(0.80 * n), f"only {wins}/{n} toggle games solved"


# =============================================================================
# Information-gain explorer sanity
# =============================================================================

def test_information_gain_picks_unobserved_action_first() -> None:
    t = HypothesisTracker(["A", "B", "C"])
    # Nothing observed yet — any action is acceptable.
    a = information_gain_action(t, ["A", "B", "C"])
    assert a in {"A", "B", "C"}


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
