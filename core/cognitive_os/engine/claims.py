"""Claims — the contents of a hypothesis.

A :class:`Hypothesis` wraps a :class:`Claim` with credence, scope, evidence,
and lattice links (parent/children).  The Claim itself is the *structural*
statement being believed.

Two orthogonal keys are exposed on every Claim:

* ``canonical_key()`` — structural identity.  Two claims with equal
  canonical keys refer to the *same phenomenon*; they compete for the same
  evidence even if their specific parameters differ.  Used by the
  HypothesisStore to link competitors.

* ``full_key()`` — exact identity.  Two claims with equal full keys are
  literally the same claim; proposing one when the other exists merges
  evidence rather than creating a duplicate.

The distinction is what lets the system *learn parameters*.  Three
``CausalClaim``\\s with the same trigger/effect but ``min_occurrences`` of
2, 3, and 4 all share the same canonical key; they are competitors; evidence
will push credence onto one and drop the others below the abandon
threshold.  No separate parameter-search mechanism is needed.

Standing directive: no Claim type encodes any domain-specific mechanic.
Every Claim form shown here is one a robot, an ARC solver, or any other
symbolic agent would equally benefit from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, Optional, Tuple

from .conditions import Condition, _hashable


# ---------------------------------------------------------------------------
# Supporting enums / value types
# ---------------------------------------------------------------------------


class RelationType(Enum):
    """Binary relations between two entities.

    ``APPEARS_*`` relations are normally resolved by an Observer (visual
    oracle).  ``CO_OCCURS_WITH`` and ``SPATIALLY_NEAR`` are resolved
    symbolically from event history and entity positions.
    ``STRUCTURALLY_LINKED`` is resolved by accumulated CausalClaims between
    the two entities.  ``SAME_CLASS`` is a taxonomic grouping that may be
    resolved by either route.
    """

    APPEARS_SIMILAR     = "appears_similar"
    APPEARS_IDENTICAL   = "appears_identical"
    APPEARS_DISTINCT    = "appears_distinct"
    CO_OCCURS_WITH      = "co_occurs_with"
    STRUCTURALLY_LINKED = "structurally_linked"
    SAME_CLASS          = "same_class"
    SPATIALLY_NEAR      = "spatially_near"


class MappingKind(Enum):
    """What basis a StructureMappingClaim rests on."""

    VISUAL     = "visual"       # appearance-driven; resolved via Observer
    STRUCTURAL = "structural"   # relation-topology driven
    FUNCTIONAL = "functional"   # role-based correspondence


@dataclass(frozen=True)
class RelationPattern:
    """A relation schema that might hold in both source and target groups
    of a :class:`StructureMappingClaim`.

    ``relation`` names the relation (e.g. "adjacent", "larger_than",
    "triggers").  ``arity`` is typically 2 but may be higher.  The pattern
    does not instantiate the roles — the mapping itself supplies the
    concrete entities.  For example, ``RelationPattern("adjacent", 2)`` in
    a source group ``{A, B, C}`` with mapping ``{A→X, B→Y, C→Z}`` is
    preserved iff ``adjacent(X,Y)`` holds whenever ``adjacent(A,B)`` does,
    and likewise for every other pair.
    """

    relation: str
    arity:    int = 2


@dataclass(frozen=True)
class Asymmetry:
    """An element in one group with no counterpart in the other, or a
    relation that holds on one side but fails on the other under the
    current mapping.

    Asymmetries are predictive: a source entity with no target counterpart
    may indicate the target group is incomplete, suggesting an exploration
    subgoal to look for the missing element.
    """

    side: str                              # "source" or "target"
    entity_id: Optional[str] = None        # unmapped entity, if element-kind
    relation:  Optional[str] = None        # relation that fails to project, if relation-kind
    note:      str = ""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Claim:
    """Base class for all claim types.

    Subclasses MUST override :meth:`canonical_key` and :meth:`full_key`.
    They SHOULD override :meth:`referenced_entities` if they name any
    entities.  Equality and hashing derive from full_key so that two
    identical claims compare equal and can be deduplicated by
    dictionary/set membership.
    """

    def canonical_key(self) -> tuple:
        raise NotImplementedError

    def full_key(self) -> tuple:
        raise NotImplementedError

    def referenced_entities(self) -> FrozenSet[str]:
        return frozenset()

    def __hash__(self) -> int:
        return hash(self.full_key())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Claim):
            return NotImplemented
        return self.full_key() == other.full_key()


# ---------------------------------------------------------------------------
# 1. PropertyClaim — an entity has a property with a value
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PropertyClaim(Claim):
    """Claim: ``entity.property == value``.

    Canonical key identifies the (entity, property) pair; full key adds
    the value.  So claims asserting different values of the same property
    on the same entity compete.
    """

    entity_id: str
    property:  str
    value:     Any

    def canonical_key(self) -> tuple:
        return ("PropertyClaim", self.entity_id, self.property)

    def full_key(self) -> tuple:
        return ("PropertyClaim", self.entity_id, self.property, _hashable(self.value))

    def referenced_entities(self) -> FrozenSet[str]:
        return frozenset([self.entity_id])


# ---------------------------------------------------------------------------
# 2. CausalClaim — trigger condition causes effect condition
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class CausalClaim(Claim):
    """Claim: when ``trigger`` holds (``min_occurrences`` times, optionally
    with a ``delay``), ``effect`` becomes true.

    ``min_occurrences`` and ``delay`` are learnable parameters — multiple
    CausalClaims with the same trigger/effect but different parameter
    values are competitors and the store will drive one to commitment.

    ``delay`` is measured in steps; a zero delay means the effect should
    be observable on the same step as the trigger firing.
    """

    trigger:         Condition
    effect:          Condition
    min_occurrences: int = 1
    delay:           int = 0

    def canonical_key(self) -> tuple:
        return ("CausalClaim",
                self.trigger.canonical_key(),
                self.effect.canonical_key())

    def full_key(self) -> tuple:
        return (*self.canonical_key(), self.min_occurrences, self.delay)

    def referenced_entities(self) -> FrozenSet[str]:
        return self.trigger.variables() | self.effect.variables()


# ---------------------------------------------------------------------------
# 3. TransitionClaim — action × pre-condition → post-condition
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class TransitionClaim(Claim):
    """Claim: executing ``action`` when ``pre`` holds yields ``post``.

    The backbone of the planner's forward model.  Canonical key is
    (action, pre) so multiple TransitionClaims with the same (action, pre)
    but different post conditions are competitors — the store converges
    on the best predictor of action outcomes.
    """

    action: str
    pre:    Condition
    post:   Condition

    def canonical_key(self) -> tuple:
        return ("TransitionClaim", self.action, self.pre.canonical_key())

    def full_key(self) -> tuple:
        return (*self.canonical_key(), self.post.canonical_key())

    def referenced_entities(self) -> FrozenSet[str]:
        return self.pre.variables() | self.post.variables()


# ---------------------------------------------------------------------------
# 4. RelationalClaim — a binary relation between two entities
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class RelationalClaim(Claim):
    """Claim: entities ``a`` and ``b`` stand in ``relation``.

    ``a`` and ``b`` are always stored in canonical order (lexicographic)
    so that the relation is undirected by default.  Use the factory
    :meth:`make` to construct — it normalises the argument order.  For
    directed relations, include direction as part of the ``relation``
    value (e.g. distinct relation types ``"triggers"`` vs ``"triggered_by"``).
    """

    a:          str
    b:          str
    relation:   RelationType
    properties: Tuple[Tuple[str, Any], ...] = ()   # immutable kv pairs

    @classmethod
    def make(cls,
             e1: str,
             e2: str,
             relation: RelationType,
             **properties: Any) -> "RelationalClaim":
        a, b = sorted([e1, e2])
        props = tuple(sorted((k, _hashable(v)) for k, v in properties.items()))
        return cls(a=a, b=b, relation=relation, properties=props)

    def canonical_key(self) -> tuple:
        return ("RelationalClaim", self.a, self.b, self.relation.value)

    def full_key(self) -> tuple:
        return (*self.canonical_key(), self.properties)

    def referenced_entities(self) -> FrozenSet[str]:
        return frozenset([self.a, self.b])


# ---------------------------------------------------------------------------
# 5. ConstraintClaim — a structural limitation with an implication
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ConstraintClaim(Claim):
    """Claim: when ``condition`` holds, ``implication`` (described
    informally as a string) follows.

    Used for structural/meta-level observations such as "when
    ``ResourceBelow('budget', 5)`` holds, the goal ``reach(target)`` is
    unreachable".  The planner uses ConstraintClaims as pruning hints,
    not as hard filters.

    ``implication`` is intentionally a free-form string because this
    class is a catch-all for constraints that don't fit the other four;
    a subsequent refinement step may replace it with a structured
    ConstraintClaim subtype.
    """

    condition:   Condition
    implication: str

    def canonical_key(self) -> tuple:
        return ("ConstraintClaim", self.condition.canonical_key())

    def full_key(self) -> tuple:
        return (*self.canonical_key(), self.implication)

    def referenced_entities(self) -> FrozenSet[str]:
        return self.condition.variables()


# ---------------------------------------------------------------------------
# 6. StructureMappingClaim — partial correspondence between two entity groups
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class StructureMappingClaim(Claim):
    """Claim: entities in ``source_entities`` correspond to entities in
    ``target_entities`` according to ``mapping``, with the preserved
    relations in ``preserved_relations`` supporting the correspondence.

    Gentner-style structure mapping.  Distinct from :class:`RelationalClaim`
    because it is a *group-to-group* projection rather than a binary
    relation, and because it explicitly tracks asymmetries that drive
    prediction and exploration.

    A committed StructureMappingClaim lets the planner *transfer* a
    previously successful plan from the source group to the target group
    by substituting mapped entities.  Asymmetries generate exploration
    subgoals: "what should be in the target position corresponding to
    this unmapped source element?"

    Canonical key is (source_entities, target_entities, mapping_kind) —
    different mappings between the same two groups compete for evidence.
    """

    source_entities:      FrozenSet[str]
    target_entities:      FrozenSet[str]
    mapping:              Tuple[Tuple[str, str], ...]          # (src_id, tgt_id) pairs, sorted
    preserved_relations:  Tuple[RelationPattern, ...] = ()
    asymmetries:          Tuple[Asymmetry, ...]       = ()
    mapping_kind:         MappingKind = MappingKind.STRUCTURAL

    @classmethod
    def make(cls,
             source: FrozenSet[str],
             target: FrozenSet[str],
             mapping: Dict[str, str],
             preserved: Tuple[RelationPattern, ...] = (),
             asymmetries: Tuple[Asymmetry, ...] = (),
             kind: MappingKind = MappingKind.STRUCTURAL) -> "StructureMappingClaim":
        sorted_mapping = tuple(sorted(mapping.items()))
        return cls(
            source_entities     = frozenset(source),
            target_entities     = frozenset(target),
            mapping             = sorted_mapping,
            preserved_relations = preserved,
            asymmetries         = asymmetries,
            mapping_kind        = kind,
        )

    def canonical_key(self) -> tuple:
        return ("StructureMappingClaim",
                tuple(sorted(self.source_entities)),
                tuple(sorted(self.target_entities)),
                self.mapping_kind.value)

    def full_key(self) -> tuple:
        return (*self.canonical_key(),
                self.mapping,
                tuple((r.relation, r.arity) for r in self.preserved_relations))

    def referenced_entities(self) -> FrozenSet[str]:
        return self.source_entities | self.target_entities


# ---------------------------------------------------------------------------
# 7. StrategyClaim — meta-claim about OR-node branch success
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class StrategyClaim(Claim):
    """Claim: in contexts matching ``context_pattern``, choosing strategy
    ``strategy_type`` at an OR-node succeeds with approximately the stored
    rate.

    This is how the engine learns branch-selection heuristics empirically.
    StrategyClaims live in the same HypothesisStore as every other claim,
    accumulate evidence from executed plans, and are consulted by the
    planner when selecting among OR-branches.

    ``context_pattern`` is a Condition that describes when this strategy
    has been observed to work — e.g. "when
    ``ResourceAbove('budget', 20)`` holds, the 'direct' strategy succeeds".

    Canonical key is (context_pattern, strategy_type); the learned
    ``success_rate`` and trial count are parameters.
    """

    context_pattern: Condition
    strategy_type:   str
    success_rate:    float = 0.5
    n_trials:        int   = 0

    def canonical_key(self) -> tuple:
        return ("StrategyClaim",
                self.context_pattern.canonical_key(),
                self.strategy_type)

    def full_key(self) -> tuple:
        return (*self.canonical_key(),
                round(self.success_rate, 3),
                self.n_trials)

    def referenced_entities(self) -> FrozenSet[str]:
        return self.context_pattern.variables()
