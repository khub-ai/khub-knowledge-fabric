"""Core data types of the cognitive engine.

This module consolidates the top-level types the rest of the engine
operates on: Observation, Hypothesis, Rule, Goal, Plan, WorldState, and
the Observer query protocol.  Lower-level types (Condition, Claim,
Credence) live in their own modules and are re-exported via the package
``__init__``.

All types in this module are pure data structures.  Phase 1 contains no
behaviour beyond simple accessors and factory helpers; learning,
planning, and execution are implemented in later phases.

Standing directive: nothing in this file may encode the mechanics of
any specific domain.  ``Observation.raw_frame`` is typed as ``Any``
deliberately — adapters decide what a "frame" is (2-D image, video
segment, point cloud, ROS message).  Event types are generic
(AgentMoved, ResourceChanged, EntityStateChanged); per-domain semantics
live in the adapter that emits them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from .claims import Claim
from .conditions import Condition
from .credence import Credence
from .tools import (
    ToolInvocation,
    ToolProposal,
    ToolRegistry,
    ToolResult,
    ToolSignature,
)


# ===========================================================================
# Scope — where/when a hypothesis, rule, or goal applies
# ===========================================================================


class ScopeKind(Enum):
    """Temporal extent of a scoped object.

    These kinds form an implicit hierarchy; clearing a broader scope
    (e.g. ``LEVEL``) cascades to any contained narrower scopes
    (``STEP``/``LIFE``/``EPISODE``).  The clear-on-transition policy is
    enforced by the WorldState at scope boundaries.
    """

    STEP     = "step"       # single observation
    LIFE     = "life"       # between death-resets within an episode
    EPISODE  = "episode"    # one run attempt on one level/task
    LEVEL    = "level"      # one game level or task variant
    GAME     = "game"       # one game/task family
    GLOBAL   = "global"     # persistent across everything


@dataclass(frozen=True)
class Scope:
    """Structural scope of a belief.

    ``kind`` gives the temporal extent.  The optional filters narrow
    applicability further: a hypothesis scoped to a specific
    ``position_region`` is evaluated only when the agent / target entity
    is in that region.
    """

    kind:            ScopeKind = ScopeKind.EPISODE
    position_region: Optional[Tuple[Any, ...]] = None   # implementation-defined bounding region
    entity_filter:   Optional[frozenset] = None         # restrict to these entity IDs
    time_range:      Optional[Tuple[int, int]] = None   # [start_step, end_step]


# ===========================================================================
# Events — per-step occurrences emitted by Adapter
# ===========================================================================


class Event:
    """Base class for all events.  Concrete events are frozen dataclasses.

    The ``step`` attribute is set by the adapter when emitting.  Events
    are the unit of symbolic information flowing from adapter to engine.
    New event types can be added for new domains; miners consume events
    they recognise and ignore the rest.
    """

    step: int


@dataclass(frozen=True)
class AgentMoved(Event):
    """Agent's position changed from ``from_pos`` to ``to_pos``."""
    step: int
    from_pos: Tuple[Any, ...]
    to_pos:   Tuple[Any, ...]


@dataclass(frozen=True)
class AgentDied(Event):
    """Agent lost a life or was terminated.  ``cause`` is an adapter-
    provided reason string (e.g. ``"lethal_cell"``, ``"budget_exhausted"``,
    ``"task_failure"``); the engine treats cause strings opaquely for
    symbolic correlation — it does not parse them for specific domains."""
    step: int
    cause: str


@dataclass(frozen=True)
class ResourceChanged(Event):
    """Tracked resource went from ``old_val`` to ``new_val``.

    Both agent-owned resources (budget, energy, time) and world-owned
    resources (e.g. a shared tank level) are emitted through this event.
    The adapter determines the resource identity.
    """
    step: int
    resource_id: str
    old_val: float
    new_val: float


@dataclass(frozen=True)
class EntityStateChanged(Event):
    """A property of ``entity_id`` changed value from ``old`` to ``new``.

    The general-purpose way to signal "something about this entity is
    different now".  Miners consume these events to form PropertyClaims,
    CausalClaims, and RelationalClaims.
    """
    step: int
    entity_id: str
    property: str
    old: Any
    new: Any


@dataclass(frozen=True)
class GoalConditionMet(Event):
    """A tracked goal's success condition became true this step."""
    step: int
    goal_id: str


@dataclass(frozen=True)
class EntityAppeared(Event):
    """A new entity became visible / trackable for the first time."""
    step: int
    entity_id: str
    initial_state: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EntityDisappeared(Event):
    """A previously-tracked entity is no longer visible / reachable."""
    step: int
    entity_id: str


@dataclass(frozen=True)
class SurpriseEvent(Event):
    """A committed prediction was violated.

    Emitted either by the adapter (when it detects a gross discrepancy)
    or synthesised by the engine itself (when an observation contradicts
    a committed TransitionClaim).  Triggers the abductive proposer and
    may trigger replanning.
    """
    step: int
    expected: Any
    actual:   Any
    context:  str = ""


# ===========================================================================
# Observation — what the adapter emits each step
# ===========================================================================


@dataclass
class Observation:
    """One step of symbolic data from the adapter.

    ``raw_frame`` is preserved so the engine can issue an ObserverQuery
    later and the adapter can answer it visually.  The engine itself
    MUST NOT inspect ``raw_frame`` — only hand it to the Observer via an
    ObserverQuery.  Metadata is a free-form dict for adapter-specific
    extensions (timestamps, sensor IDs, etc.).
    """

    step:             int
    agent_state:      Dict[str, Any]
    events:           List[Event]
    entity_snapshots: Dict[str, Dict[str, Any]]   # entity_id → {property: value}
    raw_frame:        Any = None
    metadata:         Dict[str, Any] = field(default_factory=dict)


# ===========================================================================
# Entities
# ===========================================================================


@dataclass
class EntityModel:
    """The engine's view of one tracked entity.

    Properties accumulate over time as the adapter reports them.  A
    ``kind`` label is optional and provides a coarse semantic category
    (e.g. ``"agent"``, ``"obstacle"``, ``"resource"``, ``"gate"``) when
    the adapter can supply one; without it the engine works from
    property-by-property evidence alone.
    """

    id:               str
    properties:       Dict[str, Any] = field(default_factory=dict)
    first_seen_step:  int = -1
    last_seen_step:   int = -1
    kind:             Optional[str] = None


# ===========================================================================
# Hypothesis — the learning unit
# ===========================================================================


@dataclass
class Hypothesis:
    """A candidate claim held with some credence.

    The hypothesis store maintains a lattice of Hypotheses linked by
    parent/child relations (generalisation/specialisation).  Siblings
    are reachable via parent.child_ids.  Competing hypotheses (same
    canonical key, different parameters) are linked via the
    Credence.competing field.
    """

    id:                  str
    claim:               Claim
    credence:            Credence
    scope:               Scope
    source:              str                                  # "miner:Name" | "adapter:seed" | ...
    supporting_steps:    List[int]            = field(default_factory=list)
    contradicting_steps: List[int]            = field(default_factory=list)
    expires_at:          Optional[int]        = None
    # Lattice links
    parent_id:           Optional[str]        = None
    child_ids:           List[str]            = field(default_factory=list)
    # Metadata
    created_at:          int                  = 0
    rationale:           Optional[str]        = None


# ===========================================================================
# Rule — externally imposed, authority-weighted
# ===========================================================================


class PrincipalKind(Enum):
    """Category of the entity whose will a Rule represents.

    For ARC-AGI-3 the only principal is SYSTEM; robotics requires the
    full taxonomy for conflict resolution across multiple humans.
    """
    OWNER        = "owner"
    OPERATOR     = "operator"
    GUEST        = "guest"
    STRANGER     = "stranger"
    BYSTANDER    = "bystander"
    SYSTEM       = "system"
    SAFETY_SPEC  = "safety_spec"


@dataclass(frozen=True)
class Principal:
    """An entity whose directives produce Rules.

    ``authority`` is a 0..1 base level; the effective authority at
    decision time may be modulated by the ``context`` condition
    (e.g. the owner's authority is full at home but limited in public).
    """
    id:        str
    kind:      PrincipalKind
    authority: float
    context:   Optional[Condition] = None


class Violability(Enum):
    """How strictly a Rule must be followed.

    INVIOLABLE
        Never violate under any circumstance; planner treats as hard
        filter (prunes all plans that violate the rule).
    DEFEASIBLE
        May be violated if a higher-authority or inviolable rule forces
        it; otherwise hard filter.
    ADVISORY
        Soft preference; violation adds a cost penalty proportional to
        ``priority`` but does not prune.
    """
    INVIOLABLE = "inviolable"
    DEFEASIBLE = "defeasible"
    ADVISORY   = "advisory"


class ConstraintKind(Enum):
    """Prohibit an action/condition, require it, or merely prefer it."""
    PROHIBIT = "prohibit"
    REQUIRE  = "require"
    PREFER   = "prefer"


@dataclass(frozen=True)
class RuleConstraint:
    """The content of a Rule: what action or condition it targets.

    ``target`` is either a Condition (for state-based rules: "never
    allow Condition C to hold") or an action-name string (for
    action-based rules: "never execute action A").  ``weight`` is used
    only for PREFER-kind constraints.
    """
    kind:   ConstraintKind
    target: Union[Condition, str]
    weight: float = 1.0


@dataclass
class Rule:
    """An externally imposed constraint.

    Unlike hypotheses, Rules are authoritative — they do not need
    empirical support to be binding, and their violation is not a
    disconfirmation but an action the planner avoids.  ``priority``
    is typically derived as ``principal.authority * <rule-level weight>``
    and is recomputed when the active set of principals changes.
    """
    id:           str
    condition:    Condition                # when does this rule apply?
    constraint:   RuleConstraint           # what does it require or prohibit?
    principal:    Principal
    violability:  Violability
    priority:     float
    scope:        Scope
    source:       str
    expires_at:   Optional[int] = None
    rationale:    Optional[str] = None
    created_at:   int = 0


# ===========================================================================
# Goal tree: AND-OR-CHANCE with robotics extensions
# ===========================================================================


class NodeType(Enum):
    """Types of goal-decomposition nodes.

    Active (used by the planner today):
        ATOM    — a leaf: directly achievable condition
        AND     — all children must be achieved
        OR      — any one child suffices
        CHANCE  — environment resolves with a prior distribution over outcomes

    Future (reserved for robotics / multi-agent extensions; not yet
    implemented by the planner):
        OPTION       — macro-action / sub-plan
        MAINTAIN     — ongoing condition to preserve, not achieve once
        LOOP         — cyclic sub-plan with exit condition
        ADVERSARIAL  — opposing agent's decision
        INFO_SET     — indistinguishable states, single strategy
    """
    ATOM         = "atom"
    AND          = "and"
    OR           = "or"
    CHANCE       = "chance"
    OPTION       = "option"
    MAINTAIN     = "maintain"
    LOOP         = "loop"
    ADVERSARIAL  = "adversarial"
    INFO_SET     = "info_set"


class Ordering(Enum):
    """For AND nodes: whether children must be achieved in order or any order."""
    SEQUENTIAL = "sequential"
    UNORDERED  = "unordered"


class GoalStatus(Enum):
    OPEN      = "open"         # not yet pursued
    ACTIVE    = "active"       # currently being pursued
    ACHIEVED  = "achieved"
    BLOCKED   = "blocked"      # no plan found; may become unblocked
    PRUNED    = "pruned"       # ruled out by evidence / higher-authority rule
    ABANDONED = "abandoned"    # given up permanently (e.g. deadline missed)


@dataclass
class GoalNode:
    """Node in the AND-OR-CHANCE goal tree.

    Leaf nodes are ATOM with a Condition.  Composite nodes (AND/OR/
    CHANCE) carry children.  The planner walks this tree to produce a
    :class:`Plan`; nodes carry both their current ``status`` and, for OR
    nodes, the currently-selected ``active_branch`` child.

    ``supporting_hypothesis_ids`` records which hypotheses motivated the
    insertion of this node so that if they are later falsified the
    planner knows to prune the branch.
    """
    id:         str
    node_type:  NodeType
    condition:  Optional[Condition]         = None   # required for ATOM nodes
    children:   List["GoalNode"]            = field(default_factory=list)
    ordering:   Ordering                    = Ordering.SEQUENTIAL
    status:     GoalStatus                  = GoalStatus.OPEN
    priority:   float                       = 0.5
    deadline:   Optional[int]               = None
    active_branch: Optional[str]            = None   # child_id currently selected (OR nodes only)
    # CHANCE nodes: probability prior over named outcomes
    outcome_priors: Dict[str, float]        = field(default_factory=dict)
    # Which hypotheses justified this node's existence
    supporting_hypothesis_ids: List[str]    = field(default_factory=list)
    source:     str                         = "engine:derived"
    created_at: int                         = 0


@dataclass
class Goal:
    """Top-level wrapper around a goal tree.

    A Goal has a unique ID, a principal (whose goal it is — important for
    robotics conflict resolution), a priority, and a ``root`` GoalNode
    that is the AND-OR-CHANCE tree to be satisfied.
    """
    id:          str
    root:        GoalNode
    priority:    float
    deadline:    Optional[int]     = None
    source:      str               = "adapter:primary"
    principal:   Optional[Principal] = None
    created_at:  int               = 0

    @property
    def condition(self) -> Optional[Condition]:
        """Success condition of an atomic top-level goal, if applicable."""
        return self.root.condition if self.root.node_type == NodeType.ATOM else None

    @property
    def status(self) -> GoalStatus:
        return self.root.status


# --- conflicts ---


class ConflictType(Enum):
    MUTEX       = "mutex"        # success conditions logically incompatible
    RESOURCE    = "resource"     # compete for the same limited resource
    ADVERSARIAL = "adversarial"  # one principal opposes another's goal
    TEMPORAL    = "temporal"     # deadlines incompatible with sequential pursuit


class ResolutionPolicy(Enum):
    PRIORITY        = "priority"
    INTERLEAVE      = "interleave"
    USER_ARBITRATE  = "user_arbitrate"
    FAIL            = "fail"


@dataclass
class GoalConflict:
    """A detected conflict between two goals in the forest.

    Populated by :class:`GoalForest` conflict-detection logic (later
    phase).  The ``rationale`` is a human-readable explanation for
    audit / user display.
    """
    goal_a:             str
    goal_b:             str
    conflict_type:      ConflictType
    resolution_policy:  ResolutionPolicy
    detected_at:        int
    rationale:          str


@dataclass
class GoalForest:
    """Collection of active goal trees with conflict tracking.

    Only ``active_goal_id`` receives planning attention at any moment;
    the goal-selection policy (later phase) determines which goal is
    active based on priorities, deadlines, and unresolved conflicts.
    """
    goals:          Dict[str, Goal] = field(default_factory=dict)
    conflicts:      List[GoalConflict] = field(default_factory=list)
    active_goal_id: Optional[str] = None


# ===========================================================================
# Actions and plans
# ===========================================================================


@dataclass(frozen=True)
class Action:
    """An action the adapter can execute.

    ``parameters`` is a canonical (sorted) tuple of kv pairs so that two
    actions constructed from the same logical arguments hash and compare
    equal regardless of construction order.  The adapter is responsible
    for translating a canonical Action back to its domain-native form
    (e.g. an API call, a motor command, a game controller input).
    """
    id:         str
    name:       str
    parameters: Tuple[Tuple[str, Any], ...] = ()

    @classmethod
    def make(cls, name: str, **params: Any) -> "Action":
        sorted_params = tuple(sorted(params.items()))
        # Default canonical id: name + repr of params — adapters may choose
        # their own ID scheme (e.g. domain-native command strings).
        aid = f"{name}({','.join(f'{k}={v}' for k, v in sorted_params)})"
        return cls(id=aid, name=name, parameters=sorted_params)


class PlanStatus(Enum):
    ACTIVE      = "active"
    EXECUTING   = "executing"
    COMPLETE    = "complete"
    INVALIDATED = "invalidated"
    FAILED      = "failed"


@dataclass
class PlannedAction:
    """One step in a Plan: the action to take, plus metadata.

    ``depends_on_hypotheses`` lists hypothesis IDs whose truth the step
    relies on; when any of them is demoted below commit, the plan is
    invalidated.  ``expected_effects`` are the TransitionClaims the
    planner used to forecast the outcome — their post-conditions are
    checked against actual observation and any mismatch feeds back into
    hypothesis evidence.
    """
    action:                Action
    expected_effects:      List[Claim] = field(default_factory=list)
    depends_on_hypotheses: List[str]   = field(default_factory=list)
    pre_condition:         Optional[Condition] = None


@dataclass
class Plan:
    """A selected path through the goal AND-OR tree expressed as an
    ordered sequence of PlannedActions.

    ``branch_selections`` records which child was chosen at each OR
    node encountered during planning — used to back-track efficiently
    when a plan is invalidated.  ``assumptions`` is the flattened set
    of hypothesis IDs every step depends on, cached here so the
    invalidation check is an O(|assumptions|) membership test each
    step, not a walk through the full tree.
    """
    goal_id:            str
    steps:              List[PlannedAction]
    computed_at:        int
    assumptions:        List[str]           = field(default_factory=list)
    branch_selections:  Dict[str, str]      = field(default_factory=dict)
    status:             PlanStatus          = PlanStatus.ACTIVE
    current_step_index: int                 = 0


# ===========================================================================
# Option — learned macro-action (tool creation via composition)
# ===========================================================================


@dataclass
class Option:
    """A learned macro-action: a parameterised sub-plan promoted to a
    single callable unit.

    Once the engine has executed a successful plan fragment several times
    across varying contexts, an OptionSynthesiser miner abstracts the
    varying parts into parameters and registers the fragment as an
    Option.  The Option joins the action space; future plans can invoke
    it as a single step instead of searching through its constituent
    actions, collapsing the branching factor in later planning.

    Attributes
    ----------
    id
        Stable identifier.
    name
        Human-readable name (may appear in logs and Mediator contexts).
    parameters
        Ordered tuple of ``(param_name, type_hint)`` pairs describing
        inputs the caller must supply.
    internal_plan
        The concrete sub-plan with parameter placeholders.  Execution
        substitutes actual arguments at invocation time.
    applicability
        Condition under which invoking the Option is sensible.  The
        Planner uses this as a guard; invocations whose preconditions
        are not met are rejected.
    expected_effects
        Claims the Option is believed to produce on success.  Feed into
        Planner forecasting and into post-hoc evidence comparison.
    success_rate
        Empirical success rate over past invocations; updated after each
        use.  Used by the OR-node branch selector (via StrategyClaim) to
        compare competing Options.
    n_uses
        Total invocations recorded.
    source
        Provenance tag: ``"synthesiser:RepeatedFragment"`` for code-side
        synthesis, ``"mediator:synthesis"`` for Mediator-proposed options,
        ``"user:teach"`` for operator-taught macros.
    scope
        Defaults to ``ScopeKind.GAME`` — re-usable in future runs of the
        same game/task.  Promoted to ``ScopeKind.GLOBAL`` once the
        Option has proved useful across multiple games (i.e. it
        crossed a cross-domain reuse threshold).
    """

    id:                  str
    name:                str
    parameters:          Tuple[Tuple[str, str], ...]
    internal_plan:       "Plan"
    applicability:       Condition
    expected_effects:    List[Claim]         = field(default_factory=list)
    success_rate:        float               = 0.5
    n_uses:              int                 = 0
    source:              str                 = "synthesiser:RepeatedFragment"
    scope:               "Scope"             = field(default_factory=lambda: Scope(kind=ScopeKind.GAME))
    created_at:          int                 = 0
    rationale:           Optional[str]       = None


# ===========================================================================
# CachedSolution — recorded executable sequence for a known task
# ===========================================================================


@dataclass
class CachedSolution:
    """A recorded sequence of actions known to achieve a specific task.

    This is the substrate for two superficially different capabilities
    that are structurally the same:

    **Game-level replay (ARC-AGI-3)** — an action sequence that solved
    a specific level.  Loaded in training mode to skip past already-
    solved levels; purged in competition mode.  Typically
    ``scope = LEVEL``.

    **Procedural / muscle-memory skills (robotics)** — a rehearsed
    motor sequence for a task like "walk to pose X" or "grasp cup",
    executed efficiently with minimal cognitive monitoring.  Typically
    ``scope = GAME`` (task-family-specific) or ``GLOBAL``
    (cross-task locomotion primitives like "walk").

    Distinct from :class:`Hypothesis` because it is a *recording* or
    *rehearsed procedure*, not a probabilistic belief.  A single
    failure does not falsify a CachedSolution — especially a
    non-deterministic one — it just updates the empirical success
    rate.  Removal happens by explicit abandonment (low success rate
    dropping below a threshold, or mode-gated purging at episode
    start).

    Distinct from :class:`Option` because:

    * An Option is a first-class entry in the action space; the
      Planner composes it with other actions during search.
    * A CachedSolution is a *complete* procedure for a task and is
      typically invoked as a whole unit with reduced monitoring.
    * Options are always parameterised abstractions; CachedSolutions
      may be concrete (exact action sequence for one level) or
      parameterised (muscle-memory skills).

    Attributes
    ----------
    id
        Stable identifier, typically ``f"{task_id}:{params_hash}"``.
    task_id
        Adapter-defined task identifier.  Examples:
        ``"arc:ls20:L2"``, ``"robotics:walk_to"``,
        ``"robotics:grasp_from_table"``.  The engine does not parse
        this string; the adapter owns the naming convention.
    task_parameters
        For parameterised skills (e.g. walk-to-pose-X), the sorted
        tuple of ``(name, value)`` pairs.  Empty for concrete
        recordings.
    plan
        The action sequence.  For muscle-memory skills the sequence
        may reference parameters by name; for concrete recordings the
        sequence is literal.
    deterministic
        If True, repeated execution from the same start state yields
        the same outcome (exact replay).  If False, the environment
        is stochastic — the solution should be treated as a strong
        prior that informs planning rather than a guaranteed outcome.
        **Even ARC-AGI-3 levels may be stochastic** (timing-dependent
        mechanics); robotics is almost always stochastic.  Callers
        must not assume determinism by default.
    monitor_level
        How tightly the executor watches for deviation during replay:

        * ``"low"`` — muscle-memory mode.  Execute the sequence with
          minimal checking; only hard failure signals (life lost,
          safety violation) interrupt.  Fastest; best for well-
          rehearsed skills in predictable contexts.
        * ``"moderate"`` — check preconditions at key waypoints
          declared by the plan.  Catches mid-sequence drift while
          still running most steps without scrutiny.
        * ``"full"`` — standard planner monitoring throughout.  Each
          step's expected effects are compared to actual observations
          and divergence triggers replanning.  Slowest but safest for
          new or stochastic contexts.
    n_uses / n_successes
        Empirical usage statistics; updated by PostMortem after each
        invocation.  See :meth:`success_rate`.
    scope
        Default ``LEVEL`` for game-specific recordings; ``GAME`` or
        ``GLOBAL`` for transferable procedural skills.  Critical for
        competition-mode purging — see
        :class:`core.cognitive_os.engine.config.OperatingMode`.
    source
        Provenance tag.  Examples:
        ``"postmortem:recording"`` (captured from a successful episode),
        ``"rehearsal:training"`` (deliberately practised), or
        ``"user:teach"`` (taught by operator demonstration).
    """

    id:               str
    task_id:          str
    plan:             "Plan"
    task_parameters:  Tuple[Tuple[str, Any], ...] = ()
    recorded_at:      int                         = 0
    n_uses:           int                         = 0
    n_successes:      int                         = 0
    deterministic:    bool                        = True
    monitor_level:    str                         = "low"
    scope:            "Scope"                     = field(
        default_factory=lambda: Scope(kind=ScopeKind.LEVEL))
    source:           str                         = "postmortem:recording"
    rationale:        Optional[str]               = None

    @property
    def success_rate(self) -> float:
        """Empirical success rate; defaults to 0.5 (uninformative prior)
        until at least one use is recorded."""
        if self.n_uses == 0:
            return 0.5
        return self.n_successes / self.n_uses


# ===========================================================================
# PostMortem — retrospective analysis at episode end
# ===========================================================================


@dataclass
class PostMortem:
    """Retrospective summary produced at episode end, driving
    cross-episode learning.

    The PostMortemAnalyzer runs once per episode (including on
    successful completion).  Its output is the mechanism by which the
    system **accumulates knowledge** across episodes: lessons are
    written back into the hypothesis store at ``Scope(kind=GAME)`` or
    broader, synthesised Options are added to the action registry,
    and failure signatures feed the Mediator at the next impasse so
    the same dead ends are avoided.

    Attributes
    ----------
    episode_id
        Stable identifier for this run.
    final_status
        ``"success"``, ``"failure"``, ``"timeout"``, or ``"abandoned"``.
        Free-form to allow adapters to introduce finer categories
        (e.g. ``"failure:budget_exhausted"`` vs ``"failure:death"``).
    final_step
        Step at which the episode terminated.
    goal_outcomes
        Final status of each top-level goal.
    failed_plans
        Plans that were invalidated or failed during the episode.
        Used by the analyser to extract ConstraintClaims and
        StrategyClaim updates.
    contradicted_hypotheses
        IDs of hypotheses demoted during the episode.  Helps the
        Mediator avoid the same dead ends in future runs.
    surprises
        Surprise events recorded.  High-surprise episodes deserve
        Mediator consultation for abductive hypothesis generation
        before the next run.
    lessons
        Claims extracted by the analyser — typically new
        ``StrategyClaim``\\s and ``ConstraintClaim``\\s to be written
        back into the store at broader scope.
    options_synthesised
        IDs of Options newly created during this episode.
    mediator_usage / observer_usage
        Usage counters by question type, for budget review and
        retuning.
    total_steps
        Episode length.
    wall_time_seconds
        Real-time cost of the episode; populated by the runner.
    """

    episode_id:              str
    final_status:            str
    final_step:              int
    goal_outcomes:           Dict[str, "GoalStatus"]  = field(default_factory=dict)
    failed_plans:            List["Plan"]              = field(default_factory=list)
    contradicted_hypotheses: List[str]                 = field(default_factory=list)
    surprises:               List["SurpriseEvent"]     = field(default_factory=list)
    lessons:                 List[Claim]               = field(default_factory=list)
    options_synthesised:     List[str]                 = field(default_factory=list)
    mediator_usage:          Dict[str, int]            = field(default_factory=dict)
    observer_usage:          Dict[str, int]            = field(default_factory=dict)
    total_steps:             int = 0
    wall_time_seconds:       float = 0.0


# ===========================================================================
# Oracle protocols — Observer (visual) and Mediator (common-sense)
# ===========================================================================
#
# The engine is code-centric: hypothesis formation, credence updates,
# planning, and action selection are algorithms, not LLM calls.  But two
# capabilities are genuinely hard to code and genuinely useful to hand to
# an LLM:
#
#   1. OBSERVER — visual / perceptual oracle.  Given one or more raw
#      frames and a typed question ("are these two entities still
#      similar?", "classify this object"), return a typed answer.  Used
#      by the engine to resolve appearance-based RelationalClaims and
#      to bootstrap initial visual structure at episode start.
#
#   2. MEDIATOR — common-sense / world-knowledge oracle.  Given a
#      SYMBOLIC summary of the current WorldState and a typed question
#      ("what roles do these entities likely play?", "what strategy is
#      reasonable at this impasse?"), return structured Claims, Goals,
#      or Rules the engine should consider.  Used when miner-based
#      learning is inadequate: new entity types, impasses, surprises
#      without local explanation.
#
# Both oracles share the same discipline:
#
#   * Typed inputs and typed outputs — no free-form text in the
#     decision path (free text is allowed in `explanation` fields for
#     audit/logging only).
#   * Stateless across calls — the Oracle receives whatever context it
#     needs inside the query; the engine retains no Oracle state.
#   * Budgeted via ResourceTracker — LLMBudget splits the cap between
#     Observer and Mediator so that one cannot starve the other.
#   * Outputs are treated as evidence, not commands — a Mediator-proposed
#     Claim enters the HypothesisStore with an LLM-source prior credence
#     and must still be validated by subsequent observation.  A
#     Mediator hallucination fails to accumulate support and is pruned.
#
# Adapters implement the oracles however suits their domain: a VLM for
# Observer, a text LLM for Mediator, a hand-written rules engine, or a
# human-in-the-loop operator.  The engine does not care.
#
# ---------------------------------------------------------------------------
# Observer — visual questions about specific frames
# ---------------------------------------------------------------------------


class QuestionType(Enum):
    """What the engine is asking the Observer about.

    STILL_SIMILAR    — are two entities still visually similar?
                       (revalidating a cached RelationalClaim)
    CLASSIFY         — what category does this entity belong to?
    COMPARE          — compare two entities / regions and report salient
                       differences
    DESCRIBE         — free-form description (for logging / audit only;
                       output is NEVER in the decision path)
    STRUCTURE_MAP    — does source group map to target group?
    """
    STILL_SIMILAR = "still_similar"
    CLASSIFY      = "classify"
    COMPARE       = "compare"
    DESCRIBE      = "describe"
    STRUCTURE_MAP = "structure_map"


@dataclass
class ObserverQuery:
    """A question the engine poses to the adapter's Observer.

    ``frames`` is a list of raw frame references; the adapter resolves
    them to domain-native form before handing them to its Observer
    implementation (VLM, classical vision pipeline, or human).
    """
    query_id:  str
    question:  QuestionType
    targets:   List[str]            # entity_ids the question is about
    frames:    List[Any]            # frames to inspect (adapter-specific type)
    claim_id:  Optional[str] = None # hypothesis this resolves, if any
    urgency:   float = 0.5
    context:   str = ""


@dataclass
class ObserverAnswer:
    """The adapter's answer to an ObserverQuery.

    ``result`` is a typed value: bool for yes/no questions, a string
    for classification, a structured dict for comparisons and mappings.
    The engine uses ``confidence`` to weight the answer as evidence; an
    Observer that is itself uncertain should pass a lower value.
    """
    query_id:    str
    result:      Any
    confidence:  float
    explanation: str = ""


# ---------------------------------------------------------------------------
# Mediator — common-sense guidance given a symbolic world summary
# ---------------------------------------------------------------------------


class MediatorQuestion(Enum):
    """What kind of common-sense guidance the engine is seeking.

    IDENTIFY_ROLES
        Given these entities and their observed properties, what role
        does each likely play?  (e.g. "this one looks like the agent,
        this like a wall, this like a life counter, this like a goal").
        Output: ``entity_roles`` dict plus PropertyClaims encoding the
        role assignments.
    SUGGEST_MECHANICS
        Given inferred roles, what mechanics are typical?  (e.g. "life
        counters usually decrement on agent damage; walls usually block
        movement").  Output: CausalClaims and TransitionClaims with
        LLM-source prior credence.
    SUGGEST_STRATEGY
        Given the current impasse (plan exhausted or repeatedly failing),
        what high-level strategy is reasonable?  Output: a small number
        of candidate GoalNodes representing alternative strategies to
        attempt, possibly with a suggested OR-node addition.
    WARN_HAZARDS
        Given observed entities / events, what hazards does common sense
        suggest?  Output: PropertyClaims (e.g. dangerous, lethal,
        time-pressure) and optional Rules (never-enter zones).  Extra
        critical in robotics; informational in game domains.
    PROPOSE_GOALS
        Given the current WorldState, what subgoals should the agent
        consider pursuing?  Used by the engine at cold start or when
        no adapter-seeded primary goal is known.  Output: GoalNodes
        and optionally top-level Goals.
    EXPLAIN_SURPRISE
        Given a SurpriseEvent that local miners can't explain, propose
        an abductive hypothesis.  Output: Claims (typically CausalClaims
        linking the surprise to some prior event).
    PROPOSE_TOOL
        Given the current impasse and the list of tools already
        available, propose a new tool (ToolProposal) the adapter could
        implement to unblock progress.  Output: one or more
        ToolProposals.  The adapter decides whether to implement the
        proposal and its adoption gate must pass before the tool
        enters the ToolRegistry.
    """
    IDENTIFY_ROLES    = "identify_roles"
    SUGGEST_MECHANICS = "suggest_mechanics"
    SUGGEST_STRATEGY  = "suggest_strategy"
    WARN_HAZARDS      = "warn_hazards"
    PROPOSE_GOALS     = "propose_goals"
    EXPLAIN_SURPRISE  = "explain_surprise"
    PROPOSE_TOOL      = "propose_tool"


@dataclass
class WorldStateSummary:
    """A curated, symbolic digest of WorldState passed to the Mediator.

    The engine (not the adapter) is responsible for constructing this
    summary: filtering entities to those relevant to the question,
    trimming hypotheses to committed + currently-contested, truncating
    event history to the last N steps.  The adapter's job is then to
    serialise this structured summary into whatever text format the
    underlying LLM expects — keeping domain-specific formatting choices
    in the adapter.

    ``impasse_context`` is a short, structured reason for why the
    engine is consulting the Mediator (e.g. ``"plan exhausted for goal
    g1 after 3 failed attempts"``).  It seeds the LLM's framing; it is
    NOT a free-form prompt injected into the decision path.
    """

    step:                 int
    agent:                Dict[str, Any]
    entities:             Dict[str, EntityModel]      = field(default_factory=dict)
    committed_hypotheses: List["Hypothesis"]          = field(default_factory=list)
    contested_hypotheses: List["Hypothesis"]          = field(default_factory=list)
    active_goals:         List["Goal"]                = field(default_factory=list)
    active_rules:         List["Rule"]                = field(default_factory=list)
    recent_events:        List[Event]                 = field(default_factory=list)
    impasse_context:      Optional[str]               = None
    attempted_plans:      List["Plan"]                = field(default_factory=list)
    # What the adapter can already do — prevents the Mediator from
    # proposing tools that duplicate existing primitives.
    available_tools:      Optional[ToolRegistry]      = None
    # Previously-learned Options available as macro-actions.  Useful
    # context for SUGGEST_STRATEGY and PROPOSE_TOOL questions.
    available_options:    List["Option"]              = field(default_factory=list)


@dataclass
class MediatorQuery:
    """A common-sense question the engine poses to the Mediator.

    The Mediator is stateless across calls — everything it may need to
    answer is in ``world_summary`` plus the question-specific focus
    fields.  ``focus_entities`` / ``focus_goals`` / ``surprise`` narrow
    attention so the LLM is not forced to reason about the entire
    WorldState when only part of it is relevant.
    """
    query_id:       str
    question:       MediatorQuestion
    world_summary:  WorldStateSummary
    focus_entities: List[str]                = field(default_factory=list)
    focus_goals:    List[str]                = field(default_factory=list)
    surprise:       Optional[SurpriseEvent]  = None
    urgency:        float                    = 0.5
    context:        str                      = ""


@dataclass
class MediatorAnswer:
    """The Mediator's answer — structured, not free text.

    All substantive outputs are typed engine objects (Claims, GoalNodes,
    Rules, role assignments).  The engine adds proposed_claims to the
    HypothesisStore with LLM-source prior credence, inserts
    proposed_goals under the appropriate parent, and routes
    proposed_rules through the governance pipeline (which may require
    user approval for INVIOLABLE rules).

    ``entity_roles`` is a convenience channel for the common
    IDENTIFY_ROLES question — a dict mapping entity_id to a
    common-sense role name.  The engine converts these into
    PropertyClaims internally so they flow through the same
    evidence-tracking pipeline as any other claim.

    ``tool_invocations`` lets the Mediator request an immediate
    tool call as part of its answer — useful when common-sense
    guidance is "you should probe the grid structure here" and the
    Mediator wants to drive that probe directly rather than describe
    it.  The engine enqueues each invocation through the normal
    tool-dispatch pipeline.

    ``tool_proposals`` carries new ToolProposals returned in response
    to a PROPOSE_TOOL question.  Adoption is gated by the adapter.

    ``explanation`` is free text; it is logged for audit and may be
    shown to users, but it is never parsed for decision-making.
    """
    query_id:         str
    proposed_claims:  List[Claim]            = field(default_factory=list)
    proposed_goals:   List["GoalNode"]       = field(default_factory=list)
    proposed_rules:   List["Rule"]           = field(default_factory=list)
    entity_roles:     Dict[str, str]         = field(default_factory=dict)
    tool_invocations: List[ToolInvocation]   = field(default_factory=list)
    tool_proposals:   List[ToolProposal]     = field(default_factory=list)
    confidence:       float                  = 0.5
    explanation:      str                    = ""


# ===========================================================================
# WorldState — the engine's current model
# ===========================================================================


@dataclass
class WorldState:
    """The agent's complete current model of the environment.

    Updated each step by the engine from the incoming :class:`Observation`.
    The planner, explorer, and miners all read from this structure;
    nothing else in the engine retains state (the miners are stateless
    functions over ``observation_history`` and current hypotheses).

    ``observation_history`` is retained in full for evidence-gathering
    miners (some patterns are only detectable over long windows);
    adapters may configure an eviction policy to cap memory in very long
    episodes.
    """
    step:                 int                         = 0
    agent:                Dict[str, Any]              = field(default_factory=dict)
    entities:             Dict[str, EntityModel]      = field(default_factory=dict)
    hypotheses:           Dict[str, Hypothesis]       = field(default_factory=dict)
    rules:                Dict[str, Rule]             = field(default_factory=dict)
    goal_forest:          GoalForest                  = field(default_factory=GoalForest)
    observation_history:  List[Observation]           = field(default_factory=list)
    # Tools the adapter exposes (populated at episode start) and
    # Options learned across prior episodes at scope GAME or broader.
    tool_registry:        ToolRegistry                = field(default_factory=ToolRegistry)
    options:              Dict[str, Option]           = field(default_factory=dict)
    # Pending async tool invocations (invocation_id → invocation).  The
    # runtime tracks in-flight calls here so the planner can reason
    # about latency while awaiting results.
    pending_tool_calls:   Dict[str, ToolInvocation]   = field(default_factory=dict)
    # Cached solutions / rehearsed procedures.  Loaded at episode start
    # in TRAINING mode (all scopes); in COMPETITION mode only those
    # with scope GAME or GLOBAL are loaded (LEVEL-scoped solutions are
    # purged so the agent must re-solve each level from first
    # principles).  See config.OperatingMode.
    cached_solutions:     Dict[str, "CachedSolution"] = field(default_factory=dict)
    # The config is held here so subsystems can read thresholds without
    # plumbing it through every call signature.  ``Any`` to avoid a
    # circular import with the config module at typing time.
    config:               Any = None

    # --- convenience queries ---

    def committed_hypotheses(self) -> List[Hypothesis]:
        """Return hypotheses whose credence is at or above commit threshold.

        Requires ``self.config`` to be an :class:`EngineConfig`; returns an
        empty list if config is not set (defensive for early construction
        before config is attached).
        """
        if self.config is None:
            return []
        cfg = self.config.credence
        return [h for h in self.hypotheses.values() if h.credence.is_committed(cfg)]

    def active_goal(self) -> Optional[Goal]:
        """Return the currently-active goal, if any."""
        gid = self.goal_forest.active_goal_id
        return self.goal_forest.goals.get(gid) if gid else None

    def hypothesis_by_claim_canonical(self, canonical_key: tuple) -> List[Hypothesis]:
        """Return all hypotheses whose claim shares the given canonical key.

        Used by the HypothesisStore's dedup logic to find competitors
        when a new claim is proposed.  Linear scan; acceptable for
        current store sizes.  A future optimisation may index by
        canonical key if stores grow past a few thousand entries.
        """
        return [h for h in self.hypotheses.values()
                if h.claim.canonical_key() == canonical_key]
