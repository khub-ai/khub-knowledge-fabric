"""
kf_symbolic_agent.py — End-to-end orchestration of the symbolic core.

Wires the four layers built so far into one game-playing loop:

  HypothesisTracker  (dsl.py)         — what the world does
  Planner            (planner.py)     — exploit mode
  RuleLibrary        (rule_library.py)— bootstrap from past games
  Proposer           (proposer_*.py)  — LLM/VLM judgment calls

The agent is *environment-agnostic*: it talks to a thin `GameEnv`
protocol with `reset() -> Frame`, `step(action_id) -> (Frame, won)`,
`available_actions: list[str]`. The synthetic_games families satisfy
this protocol directly, and a small adapter (~20 lines) is enough to
wire it to the real ARC-AGI-3 harness — that adapter lives in
harness.py / agents.py and is out of scope for this module.

Loop policy:

  Phase A (bootstrap):     If the rule library has an entry for our
                           structural signature, seed it.
  Phase B (explore):       Run information_gain_action for E steps to
                           let the tracker discover/refine action models.
  Phase C (plan):          Ask the planner for a goal-directed plan.
                           If found, execute it.
  Phase D (replan):        On a failed plan or stalled execution, return
                           to Phase B with a larger budget. After K
                           consecutive stalls, optionally call the
                           Proposer's EXPLAIN_STALL.
  On win:                  Snapshot the tracker into the rule library.

The agent works with MockProposer for tests; FrontierProposer or
ConstrainedProposer drop in unchanged.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from dsl import HypothesisTracker
from planner import (
    GoalPredicate,
    information_gain_action,
    plan_to_goal,
)
from proposer_schema import Proposer, ProposalContext
from rule_library import RuleLibrary, compute_signature


Frame = list[list[int]]


# =============================================================================
# Environment protocol
# =============================================================================

class GameEnv(Protocol):
    available_actions: list[str]
    def reset(self) -> Frame: ...
    def step(self, action_id: str) -> tuple[Frame, bool, bool]: ...


# =============================================================================
# Result record
# =============================================================================

@dataclass
class EpisodeResult:
    won:              bool
    steps_taken:      int
    explore_steps:    int
    plan_attempts:    int
    plans_executed:   int
    library_seeded:   int
    library_recorded: bool
    stall_explanations: int = 0
    final_signature:  str = ""


# =============================================================================
# Agent
# =============================================================================

class KFSymbolicAgent:
    """The integration point. Holds the tracker, planner config, library,
    and (optionally) a Proposer. Each call to `play_episode` resets the
    tracker and runs one game to completion or budget exhaustion."""

    def __init__(
        self,
        rule_library:        Optional[RuleLibrary] = None,
        proposer:             Optional[Proposer] = None,
        explore_steps:       int = 60,
        max_steps:           int = 300,
        plan_max_depth:      int = 32,
        plan_max_nodes:      int = 20000,
        explore_random_p:    float = 0.3,
        agent_color:         Optional[int] = None,
        stall_threshold:     int = 3,
        seed:                int = 0,
    ) -> None:
        self.rule_library     = rule_library if rule_library is not None else RuleLibrary()
        self.proposer         = proposer
        self.explore_steps    = explore_steps
        self.max_steps        = max_steps
        self.plan_max_depth   = plan_max_depth
        self.plan_max_nodes   = plan_max_nodes
        self.explore_random_p = explore_random_p
        self.agent_color      = agent_color
        self.stall_threshold  = stall_threshold
        self.rng              = random.Random(seed)

    # ----------------------------------------------------------------------

    async def play_episode(
        self,
        env:        GameEnv,
        goal:       GoalPredicate,
        source_tag: str = "",
    ) -> EpisodeResult:
        """Play one episode end-to-end and return summary metrics.

        `goal` is provided by the caller. In the real harness it would be
        constructed from the top-1 GoalBelief; for synthetic games and
        tests it's passed in directly.
        """
        tracker = HypothesisTracker(env.available_actions)
        prev = env.reset()

        # ---- Phase A: bootstrap from the rule library ---------------------
        sig = compute_signature(prev, n_actions=len(env.available_actions))
        seeded = 0
        entry = self.rule_library.lookup(sig)
        if entry is not None:
            seeded = self.rule_library.seed_tracker(
                tracker, entry,
                prior_weight=(self.proposer.prior_weight
                              if self.proposer else 0.3),
            )

        result = EpisodeResult(
            won=False,
            steps_taken=0,
            explore_steps=0,
            plan_attempts=0,
            plans_executed=0,
            library_seeded=seeded,
            library_recorded=False,
            final_signature=sig.to_key(),
        )

        stalls = 0
        # ---- Main loop: explore → plan → execute → replan ----------------
        while result.steps_taken < self.max_steps:
            # Phase B: explore
            won, prev = await self._explore(
                env, tracker, prev, self.explore_steps, result
            )
            if won:
                result.won = True
                break

            # Phase C: plan
            result.plan_attempts += 1
            plan_result = plan_to_goal(
                tracker, prev, goal,
                agent_color=self.agent_color,
                max_depth=self.plan_max_depth,
                max_nodes=self.plan_max_nodes,
            )
            if not plan_result.found:
                stalls += 1
                if (self.proposer is not None
                        and stalls >= self.stall_threshold):
                    await self._consult_stall_proposer(
                        tracker, env, result
                    )
                    stalls = 0
                # Increase explore budget on failure.
                continue

            # Execute the plan. Bail out and re-plan as soon as a step
            # produces a frame change different from what the simulator
            # predicted (tracked here as: any equal-frame outcome → blocked).
            result.plans_executed += 1
            divergent = False
            for aid in plan_result.plan:
                if result.steps_taken >= self.max_steps:
                    break
                curr, won, _ = env.step(aid)
                tracker.observe_step(aid, prev, curr)
                if curr == prev:
                    divergent = True
                prev = curr
                result.steps_taken += 1
                if won:
                    result.won = True
                    break
                if divergent:
                    break
            if result.won:
                break
            stalls += 1

        # ---- Post-game: persist tracker into the rule library on win -----
        if result.won:
            self.rule_library.record(sig, tracker, source_game=source_tag)
            result.library_recorded = True
        return result

    # ----------------------------------------------------------------------

    async def _explore(
        self,
        env:     GameEnv,
        tracker: HypothesisTracker,
        prev:    Frame,
        budget:  int,
        result:  EpisodeResult,
    ) -> tuple[bool, Frame]:
        for _ in range(budget):
            if result.steps_taken >= self.max_steps:
                return False, prev
            if self.rng.random() < self.explore_random_p:
                action = self.rng.choice(env.available_actions)
            else:
                action = information_gain_action(tracker, env.available_actions)
            curr, won, _ = env.step(action)
            tracker.observe_step(action, prev, curr)
            prev = curr
            result.steps_taken += 1
            result.explore_steps += 1
            if won:
                return True, prev
        return False, prev

    # ----------------------------------------------------------------------

    async def _consult_stall_proposer(
        self,
        tracker: HypothesisTracker,
        env:     GameEnv,
        result:  EpisodeResult,
    ) -> None:
        """Send EXPLAIN_STALL to the Proposer. Currently advisory-only:
        we log the suggestion but don't yet act on it. Wiring the
        suggested action into the next explore phase is the next step
        once we observe how a real backend behaves on stuck games."""
        from proposer_schema import ExplainStallRequest
        ctx = tracker.to_proposal_context() \
            if hasattr(tracker, "to_proposal_context") \
            else ProposalContext(
                frame_shape=tracker.frame_shape,
                available_actions=env.available_actions,
                objects=[], regions=[], gestalt=[],
            )
        req = ExplainStallRequest(
            context=ctx,
            cycles_stalled=self.stall_threshold,
            last_planner_failure_reason="planner exhausted budget",
        )
        try:
            await self.proposer.propose(req, image_png=None)
        except Exception:
            pass
        result.stall_explanations += 1
