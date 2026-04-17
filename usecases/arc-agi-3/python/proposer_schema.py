"""
proposer_schema.py — Wire-level schema for the LLM/VLM Proposer interface.

This is the single source of truth for the typed contract described in
`docs/symbolic_dsl.md` Layer 8. Both deployment modes implement against
these types:

  Constrained mode: Qwen3-VL-8B (or similar) with grammar-constrained
                    decoding (llama.cpp GBNF, outlines, lm-format-enforcer).
                    The grammar is generated from these Pydantic models.

  Frontier mode:    Claude / GPT / Gemini via structured-output APIs
                    (Anthropic tool use, OpenAI structured outputs).
                    Tool schemas are generated from these Pydantic models.

The symbolic core is identical across modes. Only the backend that fulfills
`propose()` differs. Any divergence in behavior between modes is therefore
attributable to the Proposer alone.

Run this file directly to print example payloads for every request kind:

    python proposer_schema.py

Requires: pydantic>=2.0
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


# =============================================================================
# Enumerated vocabularies
# =============================================================================
# These are the closed sets that make grammar-constrained decoding feasible
# for small open VLMs. Every field in the schema that an LLM is asked to
# emit must come from one of these enums or be a numeric/string slot with
# a tightly bounded range.

class Role(str, Enum):
    """Generic object roles. Game-agnostic by design."""
    AGENT              = "agent"
    OBSTACLE           = "obstacle"
    TARGET             = "target"
    TRIGGER            = "trigger"
    CONTAINER          = "container"
    KEY                = "key"
    COUNTER            = "counter"
    PROGRESS_INDICATOR = "progress_indicator"
    CURSOR             = "cursor"
    SLOT               = "slot"
    REFERENCE          = "reference"
    GOAL_STATE         = "goal_state"
    DECORATION         = "decoration"
    UNKNOWN            = "unknown"   # explicit "I don't know" — preferred over guessing


class Confidence(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


class Label(str, Enum):
    GUESS     = "guess"
    CONFIRMED = "confirmed"


class EffectType(str, Enum):
    """Causal primitives — see DSL Layer 2.1/2.2."""
    TRANSLATE      = "translate"
    ROTATE         = "rotate"
    REFLECT        = "reflect"
    RECOLOR        = "recolor"
    RESIZE         = "resize"
    SPAWN          = "spawn"
    DESTROY        = "destroy"
    TOGGLE         = "toggle"
    SWAP           = "swap"
    MERGE          = "merge"
    SPLIT          = "split"
    REGION_FILL    = "region_fill"
    COUNTER_TICK   = "counter_tick"
    LEVEL_ADVANCE  = "level_advance"
    STATE_RESET    = "state_reset"
    NO_OP          = "no_op"


class PreconditionType(str, Enum):
    """Conditional firing rules — see DSL Layer 2.3."""
    ALWAYS         = "always"
    IF_ROLE        = "if_role"
    IF_ADJACENT    = "if_adjacent"
    IF_INSIDE      = "if_inside"
    IF_FOCUSED     = "if_focused"
    IF_COUNTER     = "if_counter"
    IF_BLOCKED     = "if_blocked"


class GoalPredicateType(str, Enum):
    """Goal hypothesis families — see DSL Layer 4."""
    REACH                       = "reach"
    MATCH                       = "match"
    MATCH_ALL                   = "match_all"
    COVER                       = "cover"
    ELIMINATE                   = "eliminate"
    EQUAL_COUNT                 = "equal_count"
    SEQUENCE                    = "sequence"
    BEFORE_EQUALS_AFTER_UNDER   = "before_equals_after_under"
    COUNTER_REACHES             = "counter_reaches"
    PATTERN_COMPLETE            = "pattern_complete"


class RequestKind(str, Enum):
    """The five question types the Proposer can be asked. See DSL §8.2."""
    BIND_ROLES              = "bind_roles"
    RANK_GOALS              = "rank_goals"
    PROPOSE_ACTION_MODEL    = "propose_action_model"
    PROPOSE_PRECONDITION    = "propose_precondition"
    EXPLAIN_STALL           = "explain_stall"


# =============================================================================
# Symbolic perception summary (request payload — what the Proposer sees)
# =============================================================================
# These types describe the *current symbolic state* that the symbolic core
# passes into the Proposer. They are produced by object_tracker and the
# hypothesis tracker, never by the LLM. The LLM consumes them as context.

class ObjectSummary(BaseModel):
    """One detected object in the current frame."""
    id:        int                              # stable across frames via tracking
    color:     int                              # palette index 0-19+
    area:      int                              # cell count
    centroid:  tuple[float, float]              # (row, col), float-valued
    bbox:      tuple[int, int, int, int]        # (r0, c0, r1, c1) inclusive
    is_background: bool = False
    # Behavioral fingerprint accumulated by the tracker (optional, may be empty
    # on the first frame of a level). Each entry: action_id → effect_type seen.
    observed_effects: dict[str, list[str]] = Field(default_factory=dict)


class RegionSummary(BaseModel):
    """A named structural subdivision of the frame (DSL §1.3)."""
    id:    int
    kind:  Literal[
        "border", "interior", "bounded_box", "slot_strip",
        "reference_pair", "quadrant", "connected_open_space",
    ]
    bbox:  tuple[int, int, int, int]
    # Object ids contained within this region (computed by symbolic layer).
    contains_object_ids: list[int] = Field(default_factory=list)


class GestaltFeature(BaseModel):
    """A single-frame gestalt feature (DSL §1.4)."""
    kind: Literal[
        "symmetry_axis", "repetition", "unique_cell",
        "progress_bar", "counter_digits", "miniature",
    ]
    score:    float = Field(ge=0.0, le=1.0)     # detector confidence
    location: tuple[int, int] | None = None     # (row, col) anchor, if applicable
    detail:   str | None = None                 # short symbolic description


class TransitionRecord(BaseModel):
    """One observed (state, action, next_state) transition, summarized."""
    step:           int
    action_id:      str                          # e.g. "ACTION1"
    objects_moved:  list[int]                    # object ids that moved
    objects_appeared: list[int] = Field(default_factory=list)
    objects_disappeared: list[int] = Field(default_factory=list)
    pixel_diff_count: int = 0
    level_advanced: bool = False


class ActionModelSummary(BaseModel):
    """A current top-k hypothesis about what an action does."""
    action_id:      str
    effect_type:    EffectType
    precondition:   PreconditionType = PreconditionType.ALWAYS
    posterior:      float = Field(ge=0.0, le=1.0)
    support:        int = 0                      # observations consistent
    contradictions: int = 0                      # observations inconsistent
    description:    str                          # short human-readable summary


class GoalCandidate(BaseModel):
    """A current top-k hypothesis about the goal predicate."""
    predicate:    GoalPredicateType
    posterior:    float = Field(ge=0.0, le=1.0)
    description:  str
    target_object_ids: list[int] = Field(default_factory=list)


class RoleHypothesis(BaseModel):
    """Current belief about an object's role."""
    object_id:  int
    role:       Role
    confidence: Confidence
    label:      Label = Label.GUESS


# =============================================================================
# ProposalRequest — what the symbolic core sends to the Proposer
# =============================================================================

class ProposalContext(BaseModel):
    """Common context payload included in every request kind.

    The image is delivered out-of-band by the backend wrapper (as a base64
    PNG content block for vision APIs); this object only carries symbolic
    state. Image rendering is handled by the same code that calls propose().
    """
    frame_shape:    tuple[int, int]                  # (rows, cols)
    available_actions: list[str]                     # e.g. ["ACTION1", "ACTION2", ...]
    objects:        list[ObjectSummary]
    regions:        list[RegionSummary]              = Field(default_factory=list)
    gestalt:        list[GestaltFeature]             = Field(default_factory=list)
    recent_transitions: list[TransitionRecord]      = Field(default_factory=list)
    current_action_models: list[ActionModelSummary] = Field(default_factory=list)
    current_role_hypotheses: list[RoleHypothesis]   = Field(default_factory=list)
    current_goal_candidates: list[GoalCandidate]    = Field(default_factory=list)


class BindRolesRequest(BaseModel):
    request_kind: Literal[RequestKind.BIND_ROLES] = RequestKind.BIND_ROLES
    context:      ProposalContext
    # Restrict the question to specific objects, or leave None for all.
    target_object_ids: list[int] | None = None


class RankGoalsRequest(BaseModel):
    request_kind: Literal[RequestKind.RANK_GOALS] = RequestKind.RANK_GOALS
    context:      ProposalContext
    # The current top-k goals (also in context.current_goal_candidates) that
    # are tied or near-tied; the Proposer should reorder them.
    tied_goal_indices: list[int] = Field(default_factory=list)


class ProposeActionModelRequest(BaseModel):
    request_kind: Literal[RequestKind.PROPOSE_ACTION_MODEL] = RequestKind.PROPOSE_ACTION_MODEL
    context:      ProposalContext
    # Action whose existing models all contradict the latest transition.
    action_id:    str
    # The contradicting transition (also in context.recent_transitions[-1]).
    failing_transition_index: int


class ProposePreconditionRequest(BaseModel):
    request_kind: Literal[RequestKind.PROPOSE_PRECONDITION] = RequestKind.PROPOSE_PRECONDITION
    context:      ProposalContext
    action_id:    str
    effect_type:  EffectType
    # Description of when the effect fired vs. when it didn't.
    fires_under_transitions: list[int]
    nofires_under_transitions: list[int]


class ExplainStallRequest(BaseModel):
    request_kind: Literal[RequestKind.EXPLAIN_STALL] = RequestKind.EXPLAIN_STALL
    context:      ProposalContext
    cycles_stalled: int
    last_planner_failure_reason: str


# Discriminated union for dispatch.
ProposalRequest = Annotated[
    Union[
        BindRolesRequest,
        RankGoalsRequest,
        ProposeActionModelRequest,
        ProposePreconditionRequest,
        ExplainStallRequest,
    ],
    Field(discriminator="request_kind"),
]


# =============================================================================
# ProposalResponse — what the Proposer returns to the symbolic core
# =============================================================================
# Every response is grammar-enforced. No free-form prose anywhere outside
# of bounded `rationale` strings (which are advisory; the symbolic core
# logs them but does not act on them).

class BindRolesResponse(BaseModel):
    request_kind: Literal[RequestKind.BIND_ROLES] = RequestKind.BIND_ROLES
    bindings:     list[RoleHypothesis]
    rationale:    str = Field(default="", max_length=500)


class GoalRanking(BaseModel):
    predicate:   GoalPredicateType
    rank:        int                                  # 0 = most likely
    rationale:   str = Field(default="", max_length=200)


class RankGoalsResponse(BaseModel):
    request_kind: Literal[RequestKind.RANK_GOALS] = RequestKind.RANK_GOALS
    rankings:     list[GoalRanking]
    new_candidate: GoalCandidate | None = None       # optional fourth option
    rationale:    str = Field(default="", max_length=500)


class ProposeActionModelResponse(BaseModel):
    request_kind: Literal[RequestKind.PROPOSE_ACTION_MODEL] = RequestKind.PROPOSE_ACTION_MODEL
    action_id:    str
    effect_type:  EffectType
    precondition: PreconditionType = PreconditionType.ALWAYS
    description:  str = Field(default="", max_length=300)
    rationale:    str = Field(default="", max_length=500)


class ProposePreconditionResponse(BaseModel):
    request_kind: Literal[RequestKind.PROPOSE_PRECONDITION] = RequestKind.PROPOSE_PRECONDITION
    action_id:    str
    precondition: PreconditionType
    rationale:    str = Field(default="", max_length=500)


class ExplainStallResponse(BaseModel):
    request_kind: Literal[RequestKind.EXPLAIN_STALL] = RequestKind.EXPLAIN_STALL
    suspected_missing_primitive: str = Field(default="", max_length=200)
    suggested_exploration_target_object_id: int | None = None
    suggested_action_id: str | None = None
    rationale: str = Field(default="", max_length=500)


ProposalResponse = Annotated[
    Union[
        BindRolesResponse,
        RankGoalsResponse,
        ProposeActionModelResponse,
        ProposePreconditionResponse,
        ExplainStallResponse,
    ],
    Field(discriminator="request_kind"),
]


# =============================================================================
# Backend interface
# =============================================================================
# Every Proposer backend (Constrained / Frontier) implements this signature.
# The symbolic core never imports a backend directly — it imports the type
# below and is wired to a concrete implementation at startup.

class Proposer:
    """Abstract Proposer — implementations must override propose()."""

    mode: Literal["constrained", "frontier"]
    prior_weight: float            # 0.3 in constrained, 0.6 in frontier (DSL §8.3)

    async def propose(
        self,
        request: "ProposalRequest",
        image_png: bytes,
    ) -> "ProposalResponse":
        raise NotImplementedError


# =============================================================================
# Example payloads
# =============================================================================

def _example_context() -> ProposalContext:
    return ProposalContext(
        frame_shape=(64, 64),
        available_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
        objects=[
            ObjectSummary(id=1, color=2, area=2400, centroid=(31.5, 31.5),
                          bbox=(0, 0, 63, 63), is_background=True),
            ObjectSummary(id=2, color=12, area=4, centroid=(46.0, 36.0),
                          bbox=(45, 35, 47, 37),
                          observed_effects={"ACTION1": ["translate"]}),
            ObjectSummary(id=3, color=11, area=52, centroid=(62.0, 32.0),
                          bbox=(61, 6, 62, 58),
                          observed_effects={"ACTION1": ["counter_tick"]}),
        ],
        regions=[
            RegionSummary(id=1, kind="bounded_box", bbox=(2, 2, 20, 20),
                          contains_object_ids=[]),
        ],
        gestalt=[
            GestaltFeature(kind="progress_bar", score=0.8, location=(62, 32),
                           detail="horizontal strip shrinking ~4 cells/step"),
        ],
        recent_transitions=[
            TransitionRecord(step=12, action_id="ACTION1", objects_moved=[2],
                             pixel_diff_count=8, level_advanced=False),
        ],
        current_action_models=[
            ActionModelSummary(action_id="ACTION1", effect_type=EffectType.TRANSLATE,
                               posterior=0.85, support=8, contradictions=1,
                               description="ACTION1 translates object 2 by (0,+1)"),
        ],
        current_role_hypotheses=[
            RoleHypothesis(object_id=2, role=Role.AGENT, confidence=Confidence.HIGH,
                           label=Label.CONFIRMED),
        ],
        current_goal_candidates=[
            GoalCandidate(predicate=GoalPredicateType.REACH, posterior=0.4,
                          description="Reach the bounded_box region", target_object_ids=[1]),
            GoalCandidate(predicate=GoalPredicateType.MATCH, posterior=0.35,
                          description="Match the reference pattern", target_object_ids=[]),
        ],
    )


def _print_examples() -> None:
    import json

    ctx = _example_context()
    requests: list[BaseModel] = [
        BindRolesRequest(context=ctx, target_object_ids=[2, 3]),
        RankGoalsRequest(context=ctx, tied_goal_indices=[0, 1]),
        ProposeActionModelRequest(context=ctx, action_id="ACTION3",
                                   failing_transition_index=0),
        ProposePreconditionRequest(context=ctx, action_id="ACTION2",
                                    effect_type=EffectType.TRANSLATE,
                                    fires_under_transitions=[3, 5],
                                    nofires_under_transitions=[7, 9]),
        ExplainStallRequest(context=ctx, cycles_stalled=12,
                            last_planner_failure_reason="no plan within depth 8"),
    ]
    responses: list[BaseModel] = [
        BindRolesResponse(bindings=[
            RoleHypothesis(object_id=2, role=Role.AGENT, confidence=Confidence.HIGH,
                           label=Label.CONFIRMED),
            RoleHypothesis(object_id=3, role=Role.COUNTER, confidence=Confidence.MEDIUM,
                           label=Label.GUESS),
        ], rationale="object 2 moves under ACTION1; object 3 shrinks each step"),
        RankGoalsResponse(rankings=[
            GoalRanking(predicate=GoalPredicateType.MATCH, rank=0,
                        rationale="reference pair visible in upper region"),
            GoalRanking(predicate=GoalPredicateType.REACH, rank=1, rationale=""),
        ]),
        ProposeActionModelResponse(action_id="ACTION3", effect_type=EffectType.ROTATE,
                                    precondition=PreconditionType.IF_FOCUSED,
                                    description="rotates the focused slot 90° clockwise"),
        ProposePreconditionResponse(action_id="ACTION2",
                                     precondition=PreconditionType.IF_BLOCKED,
                                     rationale="fires only when adjacent to wall"),
        ExplainStallResponse(suspected_missing_primitive="counter precondition on ACTION4",
                              suggested_exploration_target_object_id=3,
                              suggested_action_id="ACTION4"),
    ]

    for req, resp in zip(requests, responses):
        print(f"\n===== {req.__class__.__name__} =====")
        print(json.dumps(req.model_dump(mode="json"), indent=2)[:1200], "...")
        print(f"\n----- {resp.__class__.__name__} -----")
        print(json.dumps(resp.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    _print_examples()
