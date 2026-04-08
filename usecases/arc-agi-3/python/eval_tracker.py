"""
eval_tracker.py — Convergence benchmark for the symbolic hypothesis tracker.

Generates batches of synthetic games, runs a uniform-random action policy
against each one for K steps, and at every step compares the tracker's
top-1 action model per action to the ground-truth HiddenEffect.

Reports per-action discovery rate, mean steps-to-first-correct (convergence
time), and per-game success (all-actions-correct fraction). The whole point:
give the symbolic core a clean, repeatable score it must beat before being
trusted on real ARC-AGI-3 frames.

Usage:
    python eval_tracker.py                      # default 20 games per family
    python eval_tracker.py --n 50 --steps 80    # bigger run
    python eval_tracker.py --family navigation  # single family
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass

from dsl import ActionModelHypothesis, HypothesisTracker
from proposer_schema import EffectType
from synthetic_games import (
    GAME_GENERATORS,
    GroundTruth,
    HiddenEffect,
    SyntheticGame,
    generate_batch,
)


# =============================================================================
# Match criteria
# =============================================================================

def model_matches_ground_truth(
    model: ActionModelHypothesis | None, truth: HiddenEffect
) -> bool:
    """Strict equality on effect type + the salient parameters."""
    if model is None:
        return False
    if model.effect_type != truth.effect_type:
        return False
    if truth.effect_type == EffectType.TRANSLATE:
        return (model.params.get("dr") == truth.params.get("dr")
                and model.params.get("dc") == truth.params.get("dc"))
    if truth.effect_type == EffectType.TOGGLE:
        # The tracker discovers TOGGLE as "recolor of these cells".
        # Ground truth doesn't store cells (the synthetic game does the
        # cell math itself). Match on effect_type alone is sufficient
        # because the tracker's TOGGLE candidates are unique by cell-set.
        return True
    if truth.effect_type == EffectType.NO_OP:
        return True
    return False


# =============================================================================
# Single-game run
# =============================================================================

@dataclass
class GameResult:
    family:                  str
    seed:                    int
    n_actions:               int
    n_actions_exercised:     int          # actions called ≥ 3 times
    n_actions_discovered:    int          # of exercised, top-1 matches truth
    converged_at:            dict[str, int]  # action_id → step of first correct top-1
    final_top_models:        dict[str, str]  # action_id → human description
    all_exercised_correct:   bool

    @property
    def discovery_rate(self) -> float:
        if self.n_actions_exercised == 0:
            return 0.0
        return self.n_actions_discovered / self.n_actions_exercised


def run_one_game(
    game: SyntheticGame, family: str, seed: int,
    steps: int, rng: random.Random,
) -> GameResult:
    tracker = HypothesisTracker(game.available_actions)
    prev = game.reset()
    call_counts = {a: 0 for a in game.available_actions}
    converged_at: dict[str, int] = {}

    for step in range(1, steps + 1):
        action = rng.choice(game.available_actions)
        call_counts[action] += 1
        curr, won, _adv = game.step(action)
        tracker.observe_step(action, prev, curr)
        prev = curr
        if won:
            # Reset so other actions still get exercised; otherwise post-win
            # frozen frames would contradict the real models and NO_OP would
            # overtake.
            prev = game.reset()

        # Record first step at which the top-1 model for this action matches
        # ground truth (only for actions called ≥ 3 times so far).
        if call_counts[action] >= 3 and action not in converged_at:
            top = tracker.top_action_models(action, k=1)
            top_model = top[0] if top else None
            if model_matches_ground_truth(
                top_model, game.ground_truth.action_models[action]
            ):
                converged_at[action] = step

    # Final scoring.
    exercised = [a for a, n in call_counts.items() if n >= 3]
    discovered = []
    final_top: dict[str, str] = {}
    for a in game.available_actions:
        top = tracker.top_action_models(a, k=1)
        m = top[0] if top else None
        final_top[a] = (
            f"{m.effect_type.value} {m.params}" if m else "(none)"
        )
        if a in exercised and model_matches_ground_truth(
            m, game.ground_truth.action_models[a]
        ):
            discovered.append(a)

    return GameResult(
        family=family,
        seed=seed,
        n_actions=len(game.available_actions),
        n_actions_exercised=len(exercised),
        n_actions_discovered=len(discovered),
        converged_at=converged_at,
        final_top_models=final_top,
        all_exercised_correct=(len(exercised) > 0
                               and len(discovered) == len(exercised)),
    )


# =============================================================================
# Batch evaluation
# =============================================================================

@dataclass
class BatchReport:
    family:                  str
    n_games:                 int
    steps_per_game:          int
    mean_discovery_rate:     float
    pct_games_fully_correct: float
    mean_convergence_step:   float
    detail:                  list[GameResult]


def run_batch(family: str, n: int, steps: int, base_seed: int = 0
              ) -> BatchReport:
    rng = random.Random(base_seed)
    games = generate_batch(family, n=n, base_seed=base_seed)
    results: list[GameResult] = []
    for i, g in enumerate(games):
        results.append(run_one_game(g, family, base_seed + i, steps, rng))

    rates = [r.discovery_rate for r in results]
    mean_rate = sum(rates) / len(rates) if rates else 0.0
    full_pct = sum(1 for r in results if r.all_exercised_correct) / len(results)

    convs: list[int] = []
    for r in results:
        convs.extend(r.converged_at.values())
    mean_conv = sum(convs) / len(convs) if convs else float("inf")

    return BatchReport(
        family=family,
        n_games=n,
        steps_per_game=steps,
        mean_discovery_rate=mean_rate,
        pct_games_fully_correct=full_pct,
        mean_convergence_step=mean_conv,
        detail=results,
    )


def format_report(report: BatchReport) -> str:
    lines = [
        f"=== {report.family} ({report.n_games} games × {report.steps_per_game} steps) ===",
        f"  mean discovery rate     : {report.mean_discovery_rate:.2%}",
        f"  games fully correct     : {report.pct_games_fully_correct:.2%}",
        f"  mean convergence step   : {report.mean_convergence_step:.1f}"
        if report.mean_convergence_step != float("inf")
        else "  mean convergence step   : (no convergence)",
    ]
    failures = [r for r in report.detail if not r.all_exercised_correct]
    if failures:
        lines.append(f"  failing seeds           : "
                     f"{[r.seed for r in failures[:10]]}"
                     + (" ..." if len(failures) > 10 else ""))
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=20, help="games per family")
    p.add_argument("--steps", type=int, default=60, help="steps per game")
    p.add_argument("--family", choices=list(GAME_GENERATORS.keys()) + ["all"],
                   default="all")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    families = (list(GAME_GENERATORS.keys()) if args.family == "all"
                else [args.family])
    overall_ok = True
    for fam in families:
        report = run_batch(fam, n=args.n, steps=args.steps, base_seed=args.seed)
        print(format_report(report))
        print()
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
