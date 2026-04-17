"""Cognitive engine — domain-agnostic symbolic reasoning substrate.

This package implements a generic cognitive engine that operates purely on
symbolic observations, hypotheses, goals, and plans.  It has no knowledge of
any specific domain (ARC-AGI-3, robotics, etc.) — domain-specific perception
and action translation live in Adapters (see `core.cognitive_os.engine.adapters`).

Design principle (standing directive):
    Game-specific or task-specific solutions MUST NOT be injected into this
    package.  Every capability needed to solve a particular problem must be
    expressed as a generalisation of the substrate — a new miner, a new
    claim type, a new planner heuristic — that would remain useful for an
    unrelated future domain.  The engine is a learning system: knowledge
    accumulated from one episode should benefit future episodes across the
    same and other domains.

LLM seams:
    Two typed oracle protocols allow an LLM (or any equivalent oracle) to
    be consulted at bounded, audited points in the loop:

    * OBSERVER  — visual / perceptual queries (frame + typed question →
                  typed answer).  For appearance-based questions the
                  engine cannot answer symbolically.
    * MEDIATOR  — common-sense / world-knowledge queries (symbolic
                  WorldStateSummary + typed question → structured
                  Claims/Goals/Rules, optionally including
                  ToolProposals or ToolInvocations).  Used at impasses,
                  cold starts, unexplained surprises, hazard assessment,
                  and tool-creation requests.

    Both oracles produce typed outputs; free-form text is confined to
    `explanation` fields that never enter the decision path.  Budgets
    are enforced via `LLMBudget` and tracked by the ResourceTracker.

Tool system:
    The adapter exposes domain primitives (grid BFS, symmetry detection,
    motion planning, etc.) through a generic ToolRegistry.  Miners,
    Planner, Explorer, and Mediator invoke tools by name via
    ToolInvocation; the adapter dispatches and returns ToolResults.
    Sync and async modes are both supported (signature carries
    `is_async` and `typical_latency_ms`).  Learned Options — macro-
    actions synthesised from recurring successful plan fragments — are
    added to the action space across episodes, so the system's toolkit
    grows over time.

Phase 1 (current): data types only — Observation, Hypothesis, Goal, Plan,
Rule, WorldState, the Claim/Condition/Credence machinery they reference,
and the Observer + Mediator query/answer protocols.  No behaviour beyond
canonicalisation and trivial accessors.
"""

from .config import (
    CredenceConfig,
    SourcePriors,
    ExplorerConfig,
    LLMBudget,
    PlannerConfig,
    EngineConfig,
    OperatingMode,
)
from .conditions import (
    Condition,
    AlwaysTrue,
    AtPosition,
    EntityInState,
    ResourceAbove,
    ResourceBelow,
    EntityProbed,
    ActionTried,
    Conjunction,
    Disjunction,
    Negation,
)
from .claims import (
    Claim,
    PropertyClaim,
    CausalClaim,
    TransitionClaim,
    RelationalClaim,
    ConstraintClaim,
    StructureMappingClaim,
    StrategyClaim,
    RelationType,
    MappingKind,
    RelationPattern,
    Asymmetry,
)
from .credence import (
    Credence,
    update_on_support,
    update_on_contradict,
    apply_decay,
)
from . import hypothesis_store
from .hypothesis_store import (
    propose,
    update_credence_from_events,
    apply_staleness_decay_all,
    prune_abandoned,
    committed,
    contested_groups,
    by_canonical_key,
    by_full_key,
    event_evidence_for_claim,
)
from . import refinement
from .refinement import (
    specialize_on_contradiction,
    detect_generalization_candidates,
    link_parent_child,
    prune_subsumed_children,
)
from . import goal_forest
from .goal_forest import (
    add_goal,
    select_active_goal,
    derive_subgoals_from_causal,
    atomic_leaves,
    is_achieved,
    mark_status,
    refresh_status,
    detect_conflicts,
)
from . import planner
from .planner import (
    compute_plan,
    apply_rules_filter,
    advisory_penalty_for_action,
)
from . import explorer
from .explorer import (
    claim_coverage,
    info_gain,
    propose_curiosity_goals,
    choose_exploration_action,
)
from . import adapters
from .adapters import Adapter
from . import miners
from .miners import (
    Miner,
    PropertyObservedMiner,
    TransitionMiner,
    FutilePatternMiner,
    SurpriseMiner,
    default_miners,
)
from . import postmortem
from .postmortem import (
    run_post_mortem,
    extract_lessons,
    OptionSynthesiser,
)
from . import episode_runner
from .episode_runner import run_episode
from .tools import (
    ToolSignature,
    ToolRegistry,
    ToolInvocation,
    ToolResult,
    ToolProposal,
    ToolCallback,
)
from .types import (
    # Scope / time
    Scope,
    ScopeKind,
    # Events
    Event,
    AgentMoved,
    AgentDied,
    ResourceChanged,
    EntityStateChanged,
    GoalConditionMet,
    EntityAppeared,
    EntityDisappeared,
    SurpriseEvent,
    # Observation
    Observation,
    # Entities
    EntityModel,
    # Hypothesis
    Hypothesis,
    # Rule
    Rule,
    Principal,
    PrincipalKind,
    Violability,
    RuleConstraint,
    ConstraintKind,
    # Goal
    Goal,
    GoalNode,
    NodeType,
    Ordering,
    GoalStatus,
    GoalForest,
    GoalConflict,
    ConflictType,
    ResolutionPolicy,
    # Action / Plan
    Action,
    PlannedAction,
    Plan,
    PlanStatus,
    # Learned macro-actions, cached procedures, and post-episode analysis
    Option,
    CachedSolution,
    PostMortem,
    # Observer (visual oracle)
    ObserverQuery,
    ObserverAnswer,
    QuestionType,
    # Mediator (common-sense oracle)
    MediatorQuery,
    MediatorAnswer,
    MediatorQuestion,
    WorldStateSummary,
    # World
    WorldState,
)

__all__ = [
    # config
    "CredenceConfig", "SourcePriors", "ExplorerConfig",
    "LLMBudget", "PlannerConfig", "EngineConfig",
    "OperatingMode",
    # conditions
    "Condition", "AlwaysTrue", "AtPosition", "EntityInState",
    "ResourceAbove", "ResourceBelow", "EntityProbed", "ActionTried",
    "Conjunction", "Disjunction", "Negation",
    # claims
    "Claim", "PropertyClaim", "CausalClaim", "TransitionClaim",
    "RelationalClaim", "ConstraintClaim", "StructureMappingClaim",
    "StrategyClaim", "RelationType", "MappingKind",
    "RelationPattern", "Asymmetry",
    # credence
    "Credence", "update_on_support", "update_on_contradict", "apply_decay",
    # hypothesis store (Phase 2)
    "hypothesis_store",
    "propose", "update_credence_from_events", "apply_staleness_decay_all",
    "prune_abandoned", "committed", "contested_groups",
    "by_canonical_key", "by_full_key", "event_evidence_for_claim",
    # refinement (Phase 2)
    "refinement",
    "specialize_on_contradiction", "detect_generalization_candidates",
    "link_parent_child", "prune_subsumed_children",
    # goal forest (Phase 3)
    "goal_forest",
    "add_goal", "select_active_goal", "derive_subgoals_from_causal",
    "atomic_leaves", "is_achieved", "mark_status", "refresh_status",
    "detect_conflicts",
    # planner (Phase 3)
    "planner",
    "compute_plan", "apply_rules_filter", "advisory_penalty_for_action",
    # explorer (Phase 3)
    "explorer",
    "claim_coverage", "info_gain",
    "propose_curiosity_goals", "choose_exploration_action",
    # adapter protocol (Phase 4)
    "adapters", "Adapter",
    # miners (Phase 4)
    "miners", "Miner",
    "PropertyObservedMiner", "TransitionMiner",
    "FutilePatternMiner", "SurpriseMiner", "default_miners",
    # postmortem (Phase 4)
    "postmortem", "run_post_mortem", "extract_lessons", "OptionSynthesiser",
    # episode runner (Phase 4)
    "episode_runner", "run_episode",
    # tools
    "ToolSignature", "ToolRegistry", "ToolInvocation", "ToolResult",
    "ToolProposal", "ToolCallback",
    # scope
    "Scope", "ScopeKind",
    # events
    "Event", "AgentMoved", "AgentDied", "ResourceChanged",
    "EntityStateChanged", "GoalConditionMet", "EntityAppeared",
    "EntityDisappeared", "SurpriseEvent",
    # observation
    "Observation",
    # entities / hypothesis
    "EntityModel", "Hypothesis",
    # rule
    "Rule", "Principal", "PrincipalKind", "Violability",
    "RuleConstraint", "ConstraintKind",
    # goal
    "Goal", "GoalNode", "NodeType", "Ordering", "GoalStatus",
    "GoalForest", "GoalConflict", "ConflictType", "ResolutionPolicy",
    # action / plan
    "Action", "PlannedAction", "Plan", "PlanStatus",
    # learned macro-actions, cached procedures, and post-episode analysis
    "Option", "CachedSolution", "PostMortem",
    # observer (visual oracle)
    "ObserverQuery", "ObserverAnswer", "QuestionType",
    # mediator (common-sense oracle)
    "MediatorQuery", "MediatorAnswer", "MediatorQuestion", "WorldStateSummary",
    # world
    "WorldState",
]
