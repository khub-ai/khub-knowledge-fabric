"""
core/cognitive_os — Cross-domain Cognitive OS layer.

Provides the domain-agnostic infrastructure that sits above any specific
environment (ARC-AGI game, robot simulator, physical robot) or skill layer:

  StateStore           — evidence-tracked, scope-managed fact store
  Environment          — ABC for all environment adapters (ARC, AI2-THOR, ...)
  Observation          — canonical sensor snapshot passed between env and agent
  Perception           — ABC for perception modules (pixel analysis, camera, ...)
  Hypothesis /
    HypothesisRegistry — testable claims with provisional/confirmed/falsified
                         lifecycle, support/against evidence counters
  Tick /
    StreamRecorder     — append-only sensory-action-decision-outcome log with
                         pluggable miners running on every record
  PatternMiner,
    FutilePatternMiner,
    SurpriseMiner      — base miner + two concrete ones covering wall-banging
                         and unexpected-outcome detection
  SafetyEnsemble /
    SafetyChecker /
    ActionProposal /
    SafetyVerdict      — safety-governance layer: vet any proposed action
                         through a composable ensemble of checkers.  Seed
                         of a future Safety Agent / Ensemble.
  CausalAttributor /
    Surprise /
    resource_exhaustion_hook /
    seed_resource_decay_hypothesis
                       — competing-cause attribution.  Prevents a
                         surface-level "cell X killed me" from being
                         blindly trusted when a slow-burning cause
                         (resource exhaustion) is the real culprit.
  ResourceTracker /
    Resource /
    BudgetStatus       — first-class depleting resources (step counter,
                         battery, tokens).  Planner consults via
                         can_afford / nearest_refill.
  DistalEffectMiner    — detects actions at cell A that consistently
                         change cell B (devices / buttons / switches).
  GoalPreconditionMiner — learns which events must precede a goal's
                         success (required_drawer_open, orientation_match).
  SimilarityOracle /
    PixelSimilarityOracle /
    VLMSimilarityOracle /
    CascadingOracle /
    SimilarityMiner    — "two things look alike → maybe they're
                         connected" hypotheses.  Pixel baseline for
                         easy cases, VLM fallback for semantic
                         similarity (rotation / scale / colour
                         invariance).

Usage
-----
    from core.cognitive_os import (
        StateStore, Environment, Observation, Perception,
        Hypothesis, HypothesisRegistry,
        Tick, StreamRecorder,
        FutilePatternMiner, SurpriseMiner,
    )

Implementing a new environment adapter:
    from core.cognitive_os import Environment, Observation
    class MyRobotEnv(Environment):
        def reset(self) -> Observation: ...
        def step(self, action) -> Observation: ...
        ...
"""

from .state_store import (
    StateStore,
    StateFact,
    RelFact,
    Delta,
    EventFact,
)
from .env_interface import Environment, Observation
from .perception import Perception
from .hypothesis import (
    Hypothesis,
    HypothesisRegistry,
    PROVISIONAL,
    CONFIRMED,
    FALSIFIED,
)
from .stream import Tick, StreamRecorder
from .miners import (
    PatternMiner,
    FutilePatternMiner,
    SurpriseMiner,
    TemporalInstabilityMiner,
    DistalEffectMiner,
    GoalPreconditionMiner,
)
from .safety import (
    ActionProposal,
    SafetyVerdict,
    SafetyChecker,
    SafetyEnsemble,
    LethalPosChecker,
    strict_block_aggregator,
)
from .causal import (
    Surprise,
    CauseCandidate,
    AlternativeCauseHook,
    CausalAttributor,
    resource_exhaustion_hook,
    seed_resource_decay_hypothesis,
)
from .resources import (
    BudgetStatus,
    Resource,
    ResourceTracker,
)
from .similarity import (
    RegionDescriptor,
    SimilarityResult,
    SimilarityOracle,
    PixelSimilarityOracle,
    VLMSimilarityOracle,
    VLMCallable,
    CascadingOracle,
    SimilarityMiner,
)

__all__ = [
    "StateStore",
    "StateFact",
    "RelFact",
    "Delta",
    "EventFact",
    "Environment",
    "Observation",
    "Perception",
    "Hypothesis",
    "HypothesisRegistry",
    "PROVISIONAL",
    "CONFIRMED",
    "FALSIFIED",
    "Tick",
    "StreamRecorder",
    "PatternMiner",
    "FutilePatternMiner",
    "SurpriseMiner",
    "TemporalInstabilityMiner",
    "DistalEffectMiner",
    "GoalPreconditionMiner",
    "ActionProposal",
    "SafetyVerdict",
    "SafetyChecker",
    "SafetyEnsemble",
    "LethalPosChecker",
    "strict_block_aggregator",
    "Surprise",
    "CauseCandidate",
    "AlternativeCauseHook",
    "CausalAttributor",
    "resource_exhaustion_hook",
    "seed_resource_decay_hypothesis",
    "BudgetStatus",
    "Resource",
    "ResourceTracker",
    "RegionDescriptor",
    "SimilarityResult",
    "SimilarityOracle",
    "PixelSimilarityOracle",
    "VLMSimilarityOracle",
    "VLMCallable",
    "CascadingOracle",
    "SimilarityMiner",
]
