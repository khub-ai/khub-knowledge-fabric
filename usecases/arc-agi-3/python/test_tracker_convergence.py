"""
test_tracker_convergence.py — Regression bounds for the v1 Bayesian updater.

Pins the convergence rates measured by eval_tracker.py so future tracker
edits can't silently regress. Bounds are deliberately a bit below the
currently observed numbers to absorb minor noise.
"""

from __future__ import annotations

import sys
import traceback

from eval_tracker import run_batch


def test_navigation_k30() -> None:
    r = run_batch("navigation", n=50, steps=30, base_seed=0)
    assert r.mean_discovery_rate >= 0.90, r.mean_discovery_rate
    assert r.pct_games_fully_correct >= 0.80, r.pct_games_fully_correct


def test_navigation_k60() -> None:
    r = run_batch("navigation", n=50, steps=60, base_seed=0)
    assert r.mean_discovery_rate >= 0.95, r.mean_discovery_rate
    assert r.pct_games_fully_correct >= 0.90, r.pct_games_fully_correct


def test_toggle_k30() -> None:
    r = run_batch("toggle", n=50, steps=30, base_seed=0)
    assert r.mean_discovery_rate >= 0.95, r.mean_discovery_rate
    assert r.pct_games_fully_correct >= 0.95, r.pct_games_fully_correct


def test_toggle_k60() -> None:
    r = run_batch("toggle", n=50, steps=60, base_seed=0)
    assert r.mean_discovery_rate >= 0.98, r.mean_discovery_rate
    assert r.pct_games_fully_correct == 1.0, r.pct_games_fully_correct


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
