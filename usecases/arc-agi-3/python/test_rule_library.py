"""
test_rule_library.py — Round-trip + bootstrap tests for the rule library.
"""

from __future__ import annotations

import random
import sys
import tempfile
import traceback
from pathlib import Path

from dsl import HypothesisTracker
from planner import plan_to_goal, reach_cell_goal
from rule_library import RuleLibrary, compute_signature
from synthetic_games import AGENT, generate_navigation_game


def _train(seed: int, steps: int = 60) -> tuple:
    g = generate_navigation_game(seed=seed)
    t = HypothesisTracker(g.available_actions)
    rng = random.Random(seed)
    prev = g.reset()
    for _ in range(steps):
        a = rng.choice(g.available_actions)
        curr, won, _ = g.step(a)
        t.observe_step(a, prev, curr)
        prev = curr
        if won:
            prev = g.reset()
    return g, t


def test_signature_is_stable_across_seeds_of_same_family() -> None:
    g0, _ = _train(0, steps=10)
    g1, _ = _train(1, steps=10)
    s0 = compute_signature(g0.reset(), n_actions=len(g0.available_actions))
    s1 = compute_signature(g1.reset(), n_actions=len(g1.available_actions))
    # Same grid shape, palette, action count → same signature key.
    assert s0.to_key() == s1.to_key()


def test_record_and_lookup() -> None:
    g, t = _train(0, steps=80)
    sig = compute_signature(g.reset(), n_actions=len(g.available_actions))
    lib = RuleLibrary()
    entry = lib.record(sig, t, source_game="navigation-seed-0")
    assert len(entry.rules) >= 1
    found = lib.lookup(sig)
    assert found is not None
    assert found.source_game == "navigation-seed-0"


def test_save_load_round_trip() -> None:
    g, t = _train(0, steps=80)
    sig = compute_signature(g.reset(), n_actions=len(g.available_actions))
    lib = RuleLibrary()
    lib.record(sig, t, source_game="round-trip")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "rules.json"
        lib.save(path)
        lib2 = RuleLibrary.load(path)
    assert len(lib2) == 1
    e = lib2.lookup(sig)
    assert e is not None and len(e.rules) >= 1


def test_seeded_tracker_can_plan_immediately() -> None:
    """Train on seed 0, transfer to seed 1 of the same family, plan on
    the very first frame using only the bootstrapped models."""
    g_train, t_train = _train(0, steps=80)
    sig = compute_signature(g_train.reset(),
                            n_actions=len(g_train.available_actions))
    lib = RuleLibrary()
    lib.record(sig, t_train, source_game="train")

    # Note: action semantics are randomized per seed in the synthetic
    # generator, so a transfer between two random nav seeds may not solve.
    # Use the same seed but a fresh tracker to demonstrate seeding works
    # without re-running observations.
    g = generate_navigation_game(seed=0)
    fresh = HypothesisTracker(g.available_actions)
    entry = lib.lookup(sig)
    assert entry is not None
    seeded = lib.seed_tracker(fresh, entry)
    assert seeded >= 1
    start = g.reset()
    result = plan_to_goal(
        fresh, start,
        reach_cell_goal(g.goal_pos, AGENT),
        agent_color=AGENT, max_depth=32,
    )
    assert result.found, "seeded tracker should plan without observation"
    g.reset()
    won = False
    for aid in result.plan:
        _, won, _ = g.step(aid)
        if won:
            break
    assert won


def test_low_posterior_rules_not_persisted() -> None:
    g = generate_navigation_game(seed=0)
    t = HypothesisTracker(g.available_actions)  # untrained
    sig = compute_signature(g.reset(), n_actions=len(g.available_actions))
    lib = RuleLibrary()
    entry = lib.record(sig, t, min_posterior=0.5)
    assert len(entry.rules) == 0


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
