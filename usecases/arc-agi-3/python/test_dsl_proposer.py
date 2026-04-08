"""
test_dsl_proposer.py — Unit tests for dsl.py + proposer_schema.py + mock_proposer.py.

Plain-assertion test runner (no pytest dependency). Run directly:

    python test_dsl_proposer.py

Each test is a function returning None. Failures raise AssertionError; the
runner counts pass/fail and exits non-zero on any failure.

Coverage:
  - Schema round-trips (Pydantic validation, JSON serialization).
  - DslObject geometry (area, centroid, bbox).
  - HypothesisTracker ingestion and projection to ProposalContext.
  - MockProposer default policies for every request kind.
  - MockProposer scripted-response queue.
  - End-to-end loop: tracker → request → mock proposer → integrate response.
  - Constrained vs Frontier prior-weight asymmetry on integration.
"""

from __future__ import annotations

import asyncio
import json
import sys
import traceback

from proposer_schema import (
    BindRolesRequest,
    BindRolesResponse,
    Confidence,
    EffectType,
    ExplainStallRequest,
    ExplainStallResponse,
    GoalCandidate,
    GoalPredicateType,
    GoalRanking,
    Label,
    PreconditionType,
    ProposalContext,
    ProposeActionModelRequest,
    ProposeActionModelResponse,
    ProposePreconditionRequest,
    ProposePreconditionResponse,
    RankGoalsRequest,
    RankGoalsResponse,
    RequestKind,
    Role,
    RoleHypothesis,
)
from dsl import (
    ActionModelHypothesis,
    DslGestaltFeature,
    DslObject,
    DslRegion,
    DslTransition,
    GoalBelief,
    HypothesisTracker,
    RoleBelief,
)
from mock_proposer import MockProposer


# =============================================================================
# Helpers
# =============================================================================

def _make_tracker() -> HypothesisTracker:
    t = HypothesisTracker(["ACTION1", "ACTION2", "ACTION3"])
    t.set_frame_shape((16, 16))
    # Background
    bg = DslObject(id=1, color=0,
                   cells={(r, c) for r in range(16) for c in range(16)},
                   is_background=True)
    # Player-like
    player = DslObject(id=2, color=12,
                       cells={(8, 8), (8, 9), (9, 8), (9, 9)},
                       observed_effects={"ACTION1": [EffectType.TRANSLATE]})
    # Counter-like
    counter = DslObject(id=3, color=11,
                        cells={(15, c) for c in range(0, 8)})
    t.add_object(bg)
    t.add_object(player)
    t.add_object(counter)
    t.add_region(DslRegion(id=1, kind="bounded_box", bbox=(2, 2, 6, 6)))
    t.add_gestalt(DslGestaltFeature(kind="progress_bar", score=0.8,
                                     location=(15, 4), detail="counter strip"))
    t.observe(DslTransition(step=1, action_id="ACTION1", objects_moved=[2],
                            pixel_diff_count=4))
    t.seed_action_model(ActionModelHypothesis(
        action_id="ACTION1", effect_type=EffectType.TRANSLATE,
        support=4, contradictions=0, description="ACTION1 translates obj 2"))
    t.seed_goal(GoalBelief(predicate=GoalPredicateType.REACH, posterior=0.4,
                           description="reach bounded_box", target_object_ids=[1]))
    t.seed_goal(GoalBelief(predicate=GoalPredicateType.MATCH, posterior=0.35,
                           description="match reference", target_object_ids=[]))
    return t


# =============================================================================
# Tests
# =============================================================================

def test_dsl_object_geometry() -> None:
    o = DslObject(id=1, color=5, cells={(2, 3), (2, 4), (3, 3)})
    assert o.area == 3
    assert o.centroid == ((2 + 2 + 3) / 3, (3 + 4 + 3) / 3)
    assert o.bbox == (2, 3, 3, 4)


def test_role_belief_uniform_prior_then_update() -> None:
    b = RoleBelief(object_id=1)
    # Initial top should be DECORATION (slightly favored).
    role, _ = b.top()
    assert role == Role.DECORATION
    # Strong update should flip the top.
    b.update(Role.AGENT, weight=10.0)
    role, _ = b.top()
    assert role == Role.AGENT


def test_action_model_posterior_monotonic() -> None:
    weak = ActionModelHypothesis(action_id="ACTION1",
                                  effect_type=EffectType.TRANSLATE,
                                  support=0, contradictions=0)
    strong = ActionModelHypothesis(action_id="ACTION1",
                                    effect_type=EffectType.TRANSLATE,
                                    support=20, contradictions=0)
    assert strong.posterior > weak.posterior


def test_tracker_projects_to_proposal_context() -> None:
    t = _make_tracker()
    ctx = t.to_proposal_context()
    assert isinstance(ctx, ProposalContext)
    assert ctx.frame_shape == (16, 16)
    assert len(ctx.objects) == 3
    assert any(o.is_background for o in ctx.objects)
    assert len(ctx.regions) == 1
    assert len(ctx.gestalt) == 1
    assert len(ctx.recent_transitions) == 1
    assert len(ctx.current_action_models) == 1
    assert len(ctx.current_role_hypotheses) == 2  # bg excluded
    assert len(ctx.current_goal_candidates) == 2


def test_proposal_context_round_trips_json() -> None:
    t = _make_tracker()
    ctx = t.to_proposal_context()
    payload = ctx.model_dump(mode="json")
    roundtripped = ProposalContext.model_validate(payload)
    assert roundtripped.frame_shape == ctx.frame_shape
    assert len(roundtripped.objects) == len(ctx.objects)
    # Ensure valid JSON serialization end-to-end.
    text = json.dumps(payload)
    assert "frame_shape" in text


async def _async_test_mock_proposer_defaults() -> None:
    t = _make_tracker()
    ctx = t.to_proposal_context()
    mp = MockProposer(mode="constrained")

    # BIND_ROLES default
    r1 = await mp.propose(BindRolesRequest(context=ctx), b"")
    assert r1.request_kind == RequestKind.BIND_ROLES
    assert isinstance(r1, BindRolesResponse)
    assert all(b.role == Role.UNKNOWN for b in r1.bindings)
    assert len(r1.bindings) == 2  # excludes background

    # RANK_GOALS default
    r2 = await mp.propose(RankGoalsRequest(context=ctx), b"")
    assert isinstance(r2, RankGoalsResponse)
    assert [g.rank for g in r2.rankings] == [0, 1]

    # PROPOSE_ACTION_MODEL default
    r3 = await mp.propose(
        ProposeActionModelRequest(context=ctx, action_id="ACTION2",
                                   failing_transition_index=0), b"")
    assert isinstance(r3, ProposeActionModelResponse)
    assert r3.effect_type == EffectType.NO_OP

    # PROPOSE_PRECONDITION default
    r4 = await mp.propose(
        ProposePreconditionRequest(context=ctx, action_id="ACTION1",
                                    effect_type=EffectType.TRANSLATE,
                                    fires_under_transitions=[0],
                                    nofires_under_transitions=[]), b"")
    assert isinstance(r4, ProposePreconditionResponse)
    assert r4.precondition == PreconditionType.ALWAYS

    # EXPLAIN_STALL default
    r5 = await mp.propose(
        ExplainStallRequest(context=ctx, cycles_stalled=5,
                            last_planner_failure_reason="depth exceeded"), b"")
    assert isinstance(r5, ExplainStallResponse)
    assert r5.suggested_action_id == "ACTION1"

    assert len(mp.received_requests) == 5


def test_mock_proposer_defaults() -> None:
    asyncio.run(_async_test_mock_proposer_defaults())


async def _async_test_mock_proposer_scripted() -> None:
    t = _make_tracker()
    ctx = t.to_proposal_context()
    canned = BindRolesResponse(bindings=[
        RoleHypothesis(object_id=2, role=Role.AGENT,
                       confidence=Confidence.HIGH, label=Label.CONFIRMED),
    ], rationale="scripted")
    mp = MockProposer(mode="frontier", scripted=[canned])
    r = await mp.propose(BindRolesRequest(context=ctx), b"")
    assert isinstance(r, BindRolesResponse)
    assert r.bindings[0].role == Role.AGENT
    assert r.bindings[0].label == Label.CONFIRMED
    assert r.rationale == "scripted"


def test_mock_proposer_scripted() -> None:
    asyncio.run(_async_test_mock_proposer_scripted())


def test_integration_role_bindings_constrained_vs_frontier() -> None:
    """Same proposer suggestion should affect frontier tracker more strongly."""
    bindings = [RoleHypothesis(object_id=2, role=Role.AGENT,
                               confidence=Confidence.HIGH, label=Label.GUESS)]

    t_c = _make_tracker()
    t_c.integrate_role_bindings(bindings, prior_weight=0.3)
    p_c = t_c.role_beliefs[2].distribution[Role.AGENT]

    t_f = _make_tracker()
    t_f.integrate_role_bindings(bindings, prior_weight=0.6)
    p_f = t_f.role_beliefs[2].distribution[Role.AGENT]

    assert p_f > p_c, f"frontier ({p_f}) should exceed constrained ({p_c})"


async def _async_test_end_to_end_loop() -> None:
    """Tracker projects → MockProposer responds → tracker integrates."""
    t = _make_tracker()
    mp = MockProposer(mode="frontier", scripted=[
        BindRolesResponse(bindings=[
            RoleHypothesis(object_id=2, role=Role.AGENT,
                           confidence=Confidence.HIGH, label=Label.GUESS),
            RoleHypothesis(object_id=3, role=Role.COUNTER,
                           confidence=Confidence.MEDIUM, label=Label.GUESS),
        ], rationale="end-to-end"),
        ProposeActionModelResponse(
            action_id="ACTION2",
            effect_type=EffectType.ROTATE,
            precondition=PreconditionType.IF_FOCUSED,
            description="rotate focused slot",
        ),
    ])

    # Round 1: ask BIND_ROLES, integrate.
    ctx = t.to_proposal_context()
    resp1 = await mp.propose(BindRolesRequest(context=ctx), b"")
    assert isinstance(resp1, BindRolesResponse)
    t.integrate_role_bindings(resp1.bindings, mp.prior_weight)

    # The agent role for object 2 should now be at least near the top.
    role, _ = t.role_beliefs[2].top()
    assert role == Role.AGENT, f"expected AGENT, got {role}"

    # Round 2: ask PROPOSE_ACTION_MODEL, integrate.
    ctx2 = t.to_proposal_context()
    resp2 = await mp.propose(
        ProposeActionModelRequest(context=ctx2, action_id="ACTION2",
                                   failing_transition_index=0), b"")
    assert isinstance(resp2, ProposeActionModelResponse)
    before = len(t.action_models["ACTION2"])
    t.integrate_action_model(
        action_id=resp2.action_id,
        effect_type=resp2.effect_type,
        precondition=resp2.precondition,
        description=resp2.description,
        prior_weight=mp.prior_weight,
    )
    after = len(t.action_models["ACTION2"])
    assert after == before + 1
    assert t.action_models["ACTION2"][-1].effect_type == EffectType.ROTATE


def test_end_to_end_loop() -> None:
    asyncio.run(_async_test_end_to_end_loop())


def test_top_k_queries() -> None:
    t = _make_tracker()
    t.seed_action_model(ActionModelHypothesis(
        action_id="ACTION1", effect_type=EffectType.NO_OP,
        support=0, contradictions=10))
    top = t.top_action_models("ACTION1", k=2)
    assert len(top) == 2
    # The translate model has support=4, no contradictions; should rank first.
    assert top[0].effect_type == EffectType.TRANSLATE

    goals = t.top_goals(k=2)
    assert goals[0].posterior >= goals[1].posterior


# =============================================================================
# Runner
# =============================================================================

def main() -> int:
    tests = [v for k, v in globals().items()
             if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for fn in tests:
        name = fn.__name__
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except Exception:
            print(f"FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
