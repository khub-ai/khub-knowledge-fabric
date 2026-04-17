"""Engine configuration — all tunable parameters in one place.

Nothing in the engine algorithm code should hardcode a threshold, rate, or
budget.  All such values live here, composed into an `EngineConfig` that is
passed down to subsystems.

The `curiosity_level` parameter is the single most important knob for
tuning exploration behaviour.  It modulates several derived parameters so
that a user can dial one number rather than coordinating five.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Operating mode
# ---------------------------------------------------------------------------


class OperatingMode(Enum):
    """Top-level operating mode of the engine — determines which
    persisted state is loaded at episode start.

    All modes load **knowledge**: hypotheses at scope ``GAME``/``GLOBAL``,
    learned Options, and cached solutions at scope ``GAME``/``GLOBAL``
    (e.g. muscle-memory skills in robotics, cross-game strategies).
    This is the accumulated competence of the agent and is legitimately
    usable in competition.

    Only ``TRAINING`` and ``DEBUG`` load **level-specific solutions**:
    CachedSolutions at scope ``LEVEL``.  These are concrete recordings
    of how a specific (game, level) was solved; loading them allows
    the agent to skip past already-solved levels in order to reach a
    target level.  In competition the agent must solve each level
    from first principles — replaying a stored recording of that same
    level would be cheating.

    TRAINING
        Full access.  Load all persisted hypotheses, Options, and
        CachedSolutions (all scopes).  Used for development,
        multi-level training runs, and rapid iteration toward a target
        level via solution replay.
    COMPETITION
        Knowledge only.  LEVEL-scoped CachedSolutions are purged before
        the episode begins.  GAME-scoped and GLOBAL-scoped items are
        loaded normally.  The agent brings all it has *learned* but
        none of the specific answers it has *memorised*.
    EVALUATION
        Same purging rules as COMPETITION but with verbose logging
        turned on for benchmark analysis.  Distinguished from
        COMPETITION only so that telemetry can be filtered.
    DEBUG
        Same load rules as TRAINING but with verbose logging and
        extra diagnostics.  Used for reproducing failures.
    """

    TRAINING    = "training"
    COMPETITION = "competition"
    EVALUATION  = "evaluation"
    DEBUG       = "debug"

    def loads_level_solutions(self) -> bool:
        """Whether this mode loads LEVEL-scoped CachedSolutions at
        episode start.  Used at the single load gate in the runtime."""
        return self in (OperatingMode.TRAINING, OperatingMode.DEBUG)

    def verbose_logging(self) -> bool:
        """Whether this mode enables extra diagnostic output."""
        return self in (OperatingMode.DEBUG, OperatingMode.EVALUATION)


# ---------------------------------------------------------------------------
# Credence dynamics
# ---------------------------------------------------------------------------


@dataclass
class CredenceConfig:
    """Thresholds and rates for the Credence update rule.

    A hypothesis with ``point >= commit_threshold`` is treated as committed
    by the planner.  One with ``point <= abandon_threshold`` is pruned.
    """

    commit_threshold: float = 0.85
    abandon_threshold: float = 0.15
    learning_rate: float = 0.15
    decay_per_step: float = 0.001
    staleness_window: int = 50


# ---------------------------------------------------------------------------
# Hypothesis source priors
# ---------------------------------------------------------------------------


@dataclass
class SourcePriors:
    """Initial credence assigned when a hypothesis is proposed by a source.

    User corrections start nearly committed; speculative LLM or abductive
    proposals start low and must earn credence through evidence.
    """

    user_correction:       float = 0.95
    adapter_seed:          float = 0.80
    observer_full_scan:    float = 0.70
    miner_confirmed:       float = 0.60
    analogy_transfer:      float = 0.40
    llm_proposer:          float = 0.30
    abductive_speculation: float = 0.25

    def for_source(self, source: str) -> float:
        """Lookup initial credence for a named source; defaults to 0.5.

        The ``source`` string follows ``"<kind>:<detail>"`` convention
        (e.g. ``"miner:FutilePattern"``, ``"user:correction"``).  Only the
        ``<kind>`` prefix is used for routing.
        """
        kind = source.split(":", 1)[0]
        return {
            "user":       self.user_correction,
            "adapter":    self.adapter_seed,
            "observer":   self.observer_full_scan,
            "miner":      self.miner_confirmed,
            "analogy":    self.analogy_transfer,
            "llm":        self.llm_proposer,
            "abductive":  self.abductive_speculation,
        }.get(kind, 0.5)


# ---------------------------------------------------------------------------
# Explorer / curiosity
# ---------------------------------------------------------------------------


@dataclass
class ExplorerConfig:
    """Exploration and curiosity parameters.

    ``curiosity_level`` is the primary knob.  Setting it alone via
    :meth:`from_curiosity_level` gives a coherent default for all derived
    parameters.  Individual parameters can still be overridden afterward
    for fine-grained tuning.

    Parameters
    ----------
    curiosity_level
        0.0 = never explore for its own sake; 1.0 = maximally curious.
    curiosity_threshold
        Claim-coverage below this fraction marks an entity as "unknown"
        and therefore a candidate for curiosity-driven probing.
    novelty_base
        Base priority assigned to a newly generated curiosity goal.
    info_gain_weight
        Relative importance of discriminating between competing
        hypotheses vs. probing wholly unknown entities.
    idle_boost
        Multiplier applied to curiosity goal priority when no
        higher-priority goal is currently making progress.  Encourages
        the agent to "look around" when not busy rather than sitting still.
    generate_curiosity_goals
        Master switch.  When False the explorer produces only
        info-gain goals (never raw novelty-seeking goals).
    """

    curiosity_level:          float = 0.3
    curiosity_threshold:      float = 0.2
    novelty_base:             float = 0.1
    info_gain_weight:         float = 1.0
    idle_boost:               float = 3.0
    generate_curiosity_goals: bool  = True

    @classmethod
    def from_curiosity_level(cls, level: float) -> "ExplorerConfig":
        """Derive a coherent ExplorerConfig from a single 0..1 curiosity knob.

        level=0.0  → no curiosity goals generated at all (pure exploit).
        level=0.5  → moderate exploration when idle, balanced with exploit.
        level=1.0  → aggressive exploration; will prefer unknowns even when
                     progress on the primary goal is possible.
        """
        level = max(0.0, min(1.0, level))
        return cls(
            curiosity_level          = level,
            curiosity_threshold      = 0.1 + 0.3 * level,
            novelty_base             = 0.05 + 0.25 * level,
            info_gain_weight         = 0.5 + 1.0 * level,
            idle_boost               = 1.0 + 4.0 * (1.0 - level),   # low-curiosity agents rely more on idle boost
            generate_curiosity_goals = level > 0.0,
        )


# ---------------------------------------------------------------------------
# LLM budgets
# ---------------------------------------------------------------------------


@dataclass
class LLMBudget:
    """Hard caps on LLM invocations, tracked by the ResourceTracker.

    The engine is code-centric; LLM calls go through two typed oracle
    seams:

    * OBSERVER — visual Q&A (frame-in, typed-answer-out).  Per-call cost
      is low (VLM call with a small frame region); calls are frequent
      during initial visual scans and whenever a cached visual relation
      needs re-validating.

    * MEDIATOR — common-sense guidance given a symbolic WorldStateSummary.
      Per-call cost is higher (large-context text LLM call); calls are
      infrequent — triggered by impasses, unexplained surprises, cold
      starts, and hazard queries.

    The two budgets are tracked separately so a burst of visual
    revalidation cannot starve a later impasse consultation (or vice
    versa).  The ``per_goal`` and ``per_hypothesis`` caps apply to the
    *combined* oracle usage attributable to a specific goal or hypothesis.
    """

    # Observer (visual) budgets
    observer_per_episode:   int = 50

    # Mediator (common-sense) budgets — typically smaller because
    # per-call cost is higher and the engine should prefer learned
    # patterns over re-consulting the Mediator on the same impasse.
    mediator_per_episode:   int = 10

    # Combined caps — apply to both oracles together
    per_goal:               int = 10
    per_hypothesis:         int = 2

    # Tolerance for budget overrun before hard-stopping
    overrun_tolerance:      int = 3   # episodes in a row we will tolerate exceeding budget before hard-stopping


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


@dataclass
class PlannerConfig:
    """Planner configuration — controls replanning cadence and search limits."""

    # Always-replan conditions (cheap) are unconditional.
    # These are optional extensions, off by default for ARC, on for robotics.
    replan_on_surprise: bool = False
    replan_periodic:    bool = False
    replan_interval:    int  = 10

    max_plan_depth: int = 200       # max AO* search depth (guards runaway search)
    branch_budget:  int = 10_000    # max nodes expanded per planning call


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class EngineConfig:
    """Top-level engine configuration.

    The two factory methods provide sane defaults for the two target
    domains; any individual sub-config can be overridden after construction.
    """

    credence:       CredenceConfig = field(default_factory=CredenceConfig)
    source_priors:  SourcePriors   = field(default_factory=SourcePriors)
    explorer:       ExplorerConfig = field(default_factory=ExplorerConfig)
    llm_budget:     LLMBudget      = field(default_factory=LLMBudget)
    planner:        PlannerConfig  = field(default_factory=PlannerConfig)
    # Default to TRAINING: safer for development.  Harness/CLI must
    # explicitly set COMPETITION for benchmarking runs.  The runtime's
    # single load gate reads this value and filters persisted
    # CachedSolutions by scope accordingly.
    operating_mode: OperatingMode  = OperatingMode.TRAINING

    @classmethod
    def arc_agi3_default(cls) -> "EngineConfig":
        """Defaults tuned for ARC-AGI-3 gameplay.

        - Low-to-moderate curiosity: exploration only when stuck.
        - No periodic/surprise replanning: plans are stable until invalidated.
        - Tight LLM budget: most knowledge must come from miners.
        """
        return cls(
            explorer   = ExplorerConfig.from_curiosity_level(0.3),
            planner    = PlannerConfig(replan_on_surprise=False, replan_periodic=False),
            llm_budget = LLMBudget(observer_per_episode=30, mediator_per_episode=5),
        )

    @classmethod
    def robotics_default(cls) -> "EngineConfig":
        """Defaults tuned for embodied robotics.

        - Higher curiosity: open-ended environments reward exploration.
        - Replan on surprise AND periodically: safety-critical drift detection.
        - Larger LLM budget: visual queries and common-sense calls are routine.
        """
        return cls(
            explorer   = ExplorerConfig.from_curiosity_level(0.5),
            planner    = PlannerConfig(replan_on_surprise=True,
                                        replan_periodic=True,
                                        replan_interval=20),
            llm_budget = LLMBudget(observer_per_episode=200,
                                    mediator_per_episode=40,
                                    per_goal=40),
        )
