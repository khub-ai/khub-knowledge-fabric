"""Conditions — logical predicates the engine evaluates against a WorldState.

A :class:`Condition` is a structural, hashable predicate.  Two conditions
with the same :meth:`canonical_key` are considered the *same* condition for
dedup / lookup purposes.  Conditions are immutable (frozen dataclasses) so
they can be safely shared across hypotheses, goals, and plans.

Every Condition must implement:

* ``canonical_key()`` — hashable tuple identifying structural form.  Used as
  the key for dedup across the hypothesis lattice.
* ``evaluate(world)`` — returns ``True``/``False`` if the predicate's truth
  value is determinable from ``world``, or ``None`` if information is
  insufficient.  Tri-valued logic is required because many conditions
  reference entities or properties the agent has never observed.
* ``variables()`` — the set of entity IDs the condition references, used by
  the planner for subgoal expansion and by miners for co-occurrence
  tracking.

No Condition class contains any domain-specific logic.  Position tuples are
of arbitrary dimensionality (2-D grids for ARC, 3-D coords for robotics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, FrozenSet, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .types import WorldState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hashable(value: Any) -> Any:
    """Coerce a value to something hashable.

    Lists/dicts/sets become tuples with sorted keys so that logically equal
    values produce equal canonical keys regardless of original ordering.
    """
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, set) or isinstance(value, frozenset):
        return tuple(sorted(_hashable(v) for v in value))
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    return value


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Condition:
    """Base class for all condition predicates.

    Subclasses are frozen dataclasses with ``eq=False`` (equality is defined
    on :meth:`canonical_key`, not dataclass field-by-field).  This lets two
    subclasses with different structural forms but equivalent canonical
    keys compare equal — useful for dedup of parameterised predicates.
    """

    def canonical_key(self) -> tuple:
        raise NotImplementedError

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        raise NotImplementedError

    def variables(self) -> FrozenSet[str]:
        return frozenset()

    # Hash / equality delegate to canonical_key so the same predicate used in
    # different places deduplicates correctly.
    def __hash__(self) -> int:
        return hash(self.canonical_key())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Condition):
            return NotImplemented
        return self.canonical_key() == other.canonical_key()


# ---------------------------------------------------------------------------
# Leaf conditions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class AlwaysTrue(Condition):
    """Tautological condition — always satisfied.

    Useful as a placeholder for Rule.condition when a rule applies
    unconditionally, or as the trigger of a CausalClaim whose effect is
    believed to hold throughout an episode.
    """

    def canonical_key(self) -> tuple:
        return ("AlwaysTrue",)

    def evaluate(self, world: "WorldState") -> bool:
        return True


@dataclass(frozen=True, eq=False)
class AtPosition(Condition):
    """Predicate: a given entity is at a given position.

    ``pos`` is a tuple of arbitrary dimensionality — 2-D for grid domains,
    3-D (or more with orientation) for robotics.  Convention: the first
    ``entity_id`` defaults to ``"agent"``, the canonical name used by the
    engine for the acting subject; adapters may choose to override it.
    """

    pos: Tuple[Any, ...]
    entity_id: str = "agent"

    def canonical_key(self) -> tuple:
        return ("AtPosition", self.entity_id, tuple(self.pos))

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        if self.entity_id == "agent":
            observed = world.agent.get("position")
        else:
            ent = world.entities.get(self.entity_id)
            observed = ent.properties.get("position") if ent is not None else None
        if observed is None:
            return None
        return tuple(observed) == tuple(self.pos)

    def variables(self) -> FrozenSet[str]:
        return frozenset([self.entity_id])


@dataclass(frozen=True, eq=False)
class EntityInState(Condition):
    """Predicate: ``entity.property == value``.

    ``value`` may be any hashable-coercible value (see :func:`_hashable`).
    """

    entity_id: str
    property:  str
    value:     Any

    def canonical_key(self) -> tuple:
        return ("EntityInState", self.entity_id, self.property, _hashable(self.value))

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        ent = world.entities.get(self.entity_id)
        if ent is None:
            return None
        if self.property not in ent.properties:
            return None
        return ent.properties[self.property] == self.value

    def variables(self) -> FrozenSet[str]:
        return frozenset([self.entity_id])


@dataclass(frozen=True, eq=False)
class ResourceAbove(Condition):
    """Predicate: a resource's current value is strictly above a threshold."""

    resource_id: str
    threshold:   float

    def canonical_key(self) -> tuple:
        return ("ResourceAbove", self.resource_id, float(self.threshold))

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        val = world.agent.get("resources", {}).get(self.resource_id)
        if val is None:
            return None
        return float(val) > self.threshold


@dataclass(frozen=True, eq=False)
class ResourceBelow(Condition):
    """Predicate: a resource's current value is strictly below a threshold."""

    resource_id: str
    threshold:   float

    def canonical_key(self) -> tuple:
        return ("ResourceBelow", self.resource_id, float(self.threshold))

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        val = world.agent.get("resources", {}).get(self.resource_id)
        if val is None:
            return None
        return float(val) < self.threshold


@dataclass(frozen=True, eq=False)
class EntityProbed(Condition):
    """Curiosity-goal condition: the entity has been observed / interacted
    with enough to raise its claim-coverage above ``curiosity_threshold``.

    The ``coverage`` attribute is the required coverage; actual coverage
    is computed by the explorer at evaluation time.  Canonical key does
    NOT include coverage — two EntityProbed conditions for the same
    entity deduplicate even if they were created at different thresholds.
    """

    entity_id: str
    coverage:  float = 0.5

    def canonical_key(self) -> tuple:
        return ("EntityProbed", self.entity_id)

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        # The explorer (not this class) implements coverage computation;
        # at Condition-level we can only answer None until the explorer
        # writes a coverage field onto the EntityModel.
        ent = world.entities.get(self.entity_id)
        if ent is None:
            return None
        cov = ent.properties.get("_claim_coverage")
        if cov is None:
            return None
        return float(cov) >= self.coverage

    def variables(self) -> FrozenSet[str]:
        return frozenset([self.entity_id])


@dataclass(frozen=True, eq=False)
class ActionTried(Condition):
    """Curiosity-goal condition: an action has been executed at least once.

    Used by the explorer when an action's transition dynamics are
    completely unknown — a single attempt yields a first TransitionClaim,
    after which ordinary evidence accumulation takes over.
    """

    action_id: str

    def canonical_key(self) -> tuple:
        return ("ActionTried", self.action_id)

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        tried = world.agent.get("_actions_tried")
        if tried is None:
            return None
        return self.action_id in tried


# ---------------------------------------------------------------------------
# Composite conditions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Conjunction(Condition):
    """Logical AND over a set of conditions.

    Child conditions are stored as a tuple so the dataclass stays hashable;
    canonicalisation sorts child keys so that the order of construction
    does not affect identity.
    """

    conditions: Tuple[Condition, ...]

    def canonical_key(self) -> tuple:
        keys = sorted(c.canonical_key() for c in self.conditions)
        return ("Conjunction", tuple(keys))

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        results = [c.evaluate(world) for c in self.conditions]
        if any(r is False for r in results):
            return False
        if any(r is None for r in results):
            return None
        return True

    def variables(self) -> FrozenSet[str]:
        out: FrozenSet[str] = frozenset()
        for c in self.conditions:
            out = out | c.variables()
        return out


@dataclass(frozen=True, eq=False)
class Disjunction(Condition):
    """Logical OR over a set of conditions."""

    conditions: Tuple[Condition, ...]

    def canonical_key(self) -> tuple:
        keys = sorted(c.canonical_key() for c in self.conditions)
        return ("Disjunction", tuple(keys))

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        results = [c.evaluate(world) for c in self.conditions]
        if any(r is True for r in results):
            return True
        if any(r is None for r in results):
            return None
        return False

    def variables(self) -> FrozenSet[str]:
        out: FrozenSet[str] = frozenset()
        for c in self.conditions:
            out = out | c.variables()
        return out


@dataclass(frozen=True, eq=False)
class Negation(Condition):
    """Logical NOT of a condition.  Preserves tri-valued semantics."""

    condition: Condition

    def canonical_key(self) -> tuple:
        return ("Negation", self.condition.canonical_key())

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        v = self.condition.evaluate(world)
        if v is None:
            return None
        return not v

    def variables(self) -> FrozenSet[str]:
        return self.condition.variables()
