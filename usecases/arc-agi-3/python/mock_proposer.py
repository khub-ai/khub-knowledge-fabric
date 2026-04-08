"""
mock_proposer.py — Deterministic Proposer for tests and symbolic-core dev.

Implements the `Proposer` interface from `proposer_schema.py` without any
LLM. Two ways to use it:

  1. Scripted: pre-load a queue of canned responses; each propose() call
     pops the next one. Used in unit tests where you want to verify the
     symbolic core's reaction to specific Proposer outputs.

  2. Default-policy: with no scripted responses, returns a sensible
     deterministic default for each request kind. Used during symbolic-core
     development when you just need *some* response so the pipeline runs.

The MockProposer also records every received request in `received_requests`
so tests can assert on what the symbolic core actually asked.
"""

from __future__ import annotations

from typing import Literal

from proposer_schema import (
    BindRolesRequest,
    BindRolesResponse,
    Confidence,
    EffectType,
    ExplainStallRequest,
    ExplainStallResponse,
    GoalPredicateType,
    GoalRanking,
    Label,
    PreconditionType,
    Proposer,
    ProposalRequest,
    ProposalResponse,
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


class MockProposer(Proposer):
    """Deterministic Proposer. Backend-free, network-free, instant."""

    def __init__(
        self,
        mode: Literal["constrained", "frontier"] = "constrained",
        scripted: list[ProposalResponse] | None = None,
    ) -> None:
        self.mode = mode
        self.prior_weight = 0.3 if mode == "constrained" else 0.6
        self._scripted: list[ProposalResponse] = list(scripted or [])
        self.received_requests: list[ProposalRequest] = []

    def queue(self, response: ProposalResponse) -> None:
        """Append a canned response to the script."""
        self._scripted.append(response)

    async def propose(
        self,
        request: ProposalRequest,
        image_png: bytes,
    ) -> ProposalResponse:
        self.received_requests.append(request)
        if self._scripted:
            return self._scripted.pop(0)
        return self._default_for(request)

    # ------------------------------------------------------------------
    # Default policies — boring but valid responses for each request kind.
    # ------------------------------------------------------------------

    def _default_for(self, req: ProposalRequest) -> ProposalResponse:
        kind = req.request_kind

        if kind == RequestKind.BIND_ROLES:
            assert isinstance(req, BindRolesRequest)
            ids = req.target_object_ids or [
                o.id for o in req.context.objects if not o.is_background
            ]
            return BindRolesResponse(
                bindings=[
                    RoleHypothesis(
                        object_id=oid,
                        role=Role.UNKNOWN,
                        confidence=Confidence.LOW,
                        label=Label.GUESS,
                    )
                    for oid in ids
                ],
                rationale="mock: default UNKNOWN binding",
            )

        if kind == RequestKind.RANK_GOALS:
            assert isinstance(req, RankGoalsRequest)
            # Identity ranking — keep existing order.
            return RankGoalsResponse(
                rankings=[
                    GoalRanking(predicate=g.predicate, rank=i, rationale="")
                    for i, g in enumerate(req.context.current_goal_candidates)
                ],
                rationale="mock: identity ranking",
            )

        if kind == RequestKind.PROPOSE_ACTION_MODEL:
            assert isinstance(req, ProposeActionModelRequest)
            return ProposeActionModelResponse(
                action_id=req.action_id,
                effect_type=EffectType.NO_OP,
                precondition=PreconditionType.ALWAYS,
                description=f"mock: {req.action_id} = no_op",
                rationale="mock: default no_op fallback",
            )

        if kind == RequestKind.PROPOSE_PRECONDITION:
            assert isinstance(req, ProposePreconditionRequest)
            return ProposePreconditionResponse(
                action_id=req.action_id,
                precondition=PreconditionType.ALWAYS,
                rationale="mock: default ALWAYS",
            )

        if kind == RequestKind.EXPLAIN_STALL:
            assert isinstance(req, ExplainStallRequest)
            return ExplainStallResponse(
                suspected_missing_primitive="",
                suggested_exploration_target_object_id=None,
                suggested_action_id=(req.context.available_actions[0]
                                     if req.context.available_actions else None),
                rationale="mock: try first action",
            )

        raise ValueError(f"unknown request kind: {kind}")
