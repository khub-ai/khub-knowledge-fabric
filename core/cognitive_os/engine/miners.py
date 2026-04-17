"""Miners — pattern detectors over the event stream.

Miners are stateless callables that observe each step's events,
along with the running :class:`WorldState`, and propose hypotheses
into the :class:`hypothesis_store`.  They are the bridge between raw
symbolic events and the structured hypothesis layer the planner
operates on.

Design
------
Every miner implements :class:`Miner.step(ws, events, step)`.  The
method inspects the events and the current WorldState and calls
:func:`hypothesis_store.propose` for each hypothesis it wants to
register.  The store takes care of dedup, competitor linking, and
credence updates — the miner only decides *what to propose*.

Miners are stateless in the engine sense: all historical context they
need is in ``ws.observation_history``.  A miner that wants to remember
something across steps must encode it as a hypothesis — that's the
entire point of the hypothesis layer.

Phase 4 miners
--------------
The four miners below cover the common patterns needed to populate a
hypothesis store from a live event stream:

* :class:`PropertyObservedMiner` — converts ``EntityStateChanged``
  events directly into ``PropertyClaim``\\s for the new value.  The
  fastest source of hypothesis bootstrap: every entity change seen
  becomes an explicit belief about current state.

* :class:`TransitionMiner` — detects (pre-state, action, post-state)
  triples in the observation/action history and proposes
  ``TransitionClaim``\\s.  Builds the transition model the planner
  uses for BFS.

* :class:`FutilePatternMiner` — detects "action in context yields no
  change" patterns and proposes a ``TransitionClaim`` with the
  trivial post == pre.  Lets the planner avoid known-ineffective
  actions.

* :class:`SurpriseMiner` — detects events that contradict committed
  hypotheses and emits a :class:`SurpriseEvent` into
  ``ws.observation_history`` so that downstream miners /
  Mediator / refinement can react to it.  This is the detector
  behind the specialisation-on-contradiction loop.

Capability audit
----------------
* **Debugging** — PRIMARY.  Miners are how the engine forms
  candidate explanations from observation.  SurpriseMiner
  specifically catches "my prediction was wrong" events that drive
  specialisation and Mediator consultation.
* **Problem-solving** — secondary.  TransitionMiner populates the
  planner's transition model; FutilePatternMiner prunes the
  effective action space.
* **Tool creation** — minor.  Patterns detected here are the raw
  material for the :class:`OptionSynthesiser` in
  :mod:`postmortem`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

from .claims import (
    PropertyClaim,
    TransitionClaim,
)
from .conditions import (
    AtPosition,
    Condition,
    EntityInState,
)
from . import hypothesis_store as _store
from .types import (
    AgentMoved,
    EntityStateChanged,
    Event,
    Scope,
    ScopeKind,
    SurpriseEvent,
    WorldState,
)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Miner(ABC):
    """Abstract base class for all miners.  Subclasses implement
    :meth:`step`.

    Each miner carries a ``name`` (used as the source tag prefix on
    proposed hypotheses) and a ``default_scope`` (applied when the
    miner doesn't have a better scope hint for a given hypothesis).
    Both default to sensible values that subclasses may override.
    """

    name: str = "miner"
    default_scope: Scope = Scope(kind=ScopeKind.GAME)

    @abstractmethod
    def step(self,
             ws:     WorldState,
             events: List[Event],
             step:   int) -> None:
        """Inspect the current step's events and (optionally) the
        :class:`WorldState` history, and propose hypotheses via
        :func:`hypothesis_store.propose`.

        Must not mutate ``ws`` directly — only through ``propose()``.
        """


# ---------------------------------------------------------------------------
# PropertyObservedMiner
# ---------------------------------------------------------------------------


class PropertyObservedMiner(Miner):
    """Convert each :class:`EntityStateChanged` event into a
    :class:`PropertyClaim` asserting the new value.

    Rationale: an observed value *is* evidence that the value holds
    right now.  Competing claims about the same (entity, property)
    appear automatically as canonical competitors and the store
    reconciles them as more observations accumulate.

    This is the simplest possible miner and yields the fastest
    hypothesis-store bootstrap.  It runs on every step.
    """

    name = "miner:PropertyObserved"

    def step(self,
             ws:     WorldState,
             events: List[Event],
             step:   int) -> None:
        for evt in events:
            if not isinstance(evt, EntityStateChanged):
                continue
            claim = PropertyClaim(
                entity_id = evt.entity_id,
                property  = evt.property,
                value     = evt.new,
            )
            _store.propose(
                ws,
                claim  = claim,
                source = self.name,
                scope  = self.default_scope,
                step   = step,
            )


# ---------------------------------------------------------------------------
# TransitionMiner
# ---------------------------------------------------------------------------


class TransitionMiner(Miner):
    """Detect (pre-position, action, post-position) triples from
    :class:`AgentMoved` events and propose matching
    :class:`TransitionClaim`\\s.

    This is the miner that populates the planner's transition model.
    Without it the planner has nothing to BFS over, so the engine
    reduces to pure exploration until enough transitions accumulate
    through trial and error.

    Extension path: handle resource-changing transitions (via
    :class:`ResourceChanged`) and property-changing transitions
    (via :class:`EntityStateChanged` caused by a specific action).
    Phase 4 keeps it positional because that's the minimum viable
    model for grid / locomotion domains.
    """

    name = "miner:Transition"

    def step(self,
             ws:     WorldState,
             events: List[Event],
             step:   int) -> None:
        # Look for an AgentMoved in this step's events.  If the last
        # action was recorded on ws.agent['_last_action'], pair them.
        last_action = ws.agent.get("_last_action")
        if last_action is None:
            return

        for evt in events:
            if not isinstance(evt, AgentMoved):
                continue
            claim = TransitionClaim(
                action = str(last_action),
                pre    = AtPosition(tuple(evt.from_pos), entity_id="agent"),
                post   = AtPosition(tuple(evt.to_pos), entity_id="agent"),
            )
            _store.propose(
                ws,
                claim  = claim,
                source = self.name,
                scope  = self.default_scope,
                step   = step,
            )


# ---------------------------------------------------------------------------
# FutilePatternMiner
# ---------------------------------------------------------------------------


class FutilePatternMiner(Miner):
    """Detect "action in context yields no observable change" patterns.

    When an action is executed but no :class:`AgentMoved`,
    :class:`EntityStateChanged`, or :class:`ResourceChanged` event is
    produced, the miner proposes a :class:`TransitionClaim` with
    identical pre and post conditions — effectively "this action is
    a no-op in this state".  The planner avoids no-op actions
    automatically because their BFS doesn't advance.

    Rationale: wall-banging (repeatedly trying actions that don't work)
    is one of the biggest wastes of episode budget.  Detecting futile
    patterns early lets the planner prune them from its action space.
    """

    name = "miner:FutilePattern"

    # Event classes that constitute "something happened"
    _SIGNIFICANT_EVENT_TYPES: Tuple = (AgentMoved, EntityStateChanged)

    def step(self,
             ws:     WorldState,
             events: List[Event],
             step:   int) -> None:
        last_action = ws.agent.get("_last_action")
        if last_action is None:
            return
        # Significant change observed this step?
        if any(isinstance(e, self._SIGNIFICANT_EVENT_TYPES) for e in events):
            return
        # No change — propose a futile-transition claim
        position = tuple(ws.agent.get("position") or ())
        if not position:
            return
        pre = AtPosition(position, entity_id="agent")
        claim = TransitionClaim(
            action = str(last_action),
            pre    = pre,
            post   = pre,  # no-op
        )
        _store.propose(
            ws,
            claim  = claim,
            source = self.name,
            scope  = self.default_scope,
            step   = step,
        )


# ---------------------------------------------------------------------------
# SurpriseMiner
# ---------------------------------------------------------------------------


class SurpriseMiner(Miner):
    """Detect observations that contradict committed hypotheses.

    On each step, walk committed :class:`PropertyClaim` and
    :class:`TransitionClaim` hypotheses and check them against the
    latest events.  When a committed claim is contradicted (the
    store's :func:`event_evidence_for_claim` returns ``False``),
    emit a :class:`SurpriseEvent` so downstream consumers — the
    Mediator (``EXPLAIN_SURPRISE`` question), the refinement layer
    (``specialize_on_contradiction``), and the runner's replan
    trigger — can react.

    The miner does *not* demote the contradicted hypothesis itself —
    that happens through the normal credence-update pipeline during
    the same step.  SurpriseMiner's contribution is purely
    surfacing the event so higher-level subsystems know "a
    committed prediction just failed", which is different from the
    slow cumulative credence drift the store already handles.

    Rationale: contradiction-driven learning is the engine's
    equivalent of Claude-Code-style iterative debugging.  Without
    explicit surprise detection, the system would silently slide
    through broken predictions instead of pausing to consult the
    Mediator or specialise.
    """

    name = "miner:Surprise"

    def step(self,
             ws:     WorldState,
             events: List[Event],
             step:   int) -> None:
        committed_hyps = _store.committed(ws)
        if not committed_hyps:
            return
        new_surprises: List[SurpriseEvent] = []

        for h in committed_hyps:
            for evt in events:
                verdict = _store.event_evidence_for_claim(evt, h.claim, ws)
                if verdict is False:
                    # A committed claim was contradicted.
                    new_surprises.append(SurpriseEvent(
                        step     = step,
                        expected = h.claim.canonical_key(),
                        actual   = _event_signature(evt),
                        context  = (f"committed hypothesis {h.id} "
                                    f"contradicted by {type(evt).__name__}"),
                    ))
                    break  # one surprise per hypothesis per step

        # Append surprise events to the latest observation in history.
        # (The runner appends observations; we piggyback on the last.)
        if new_surprises and ws.observation_history:
            ws.observation_history[-1].events.extend(new_surprises)


def _event_signature(evt: Event) -> Any:
    """Compact, hashable-ish description of an event for surprise
    logging.  Intentionally loose: we just need something the
    Mediator can look at."""
    sig = {
        "type": type(evt).__name__,
        "step": getattr(evt, "step", None),
    }
    if hasattr(evt, "__dataclass_fields__"):
        for k in evt.__dataclass_fields__:
            if k != "step":
                sig[k] = getattr(evt, k)
    return sig


# ---------------------------------------------------------------------------
# Default miner suite
# ---------------------------------------------------------------------------


def default_miners() -> List[Miner]:
    """Return the canonical Phase 4 miner suite in run order.

    Order matters: :class:`PropertyObservedMiner` and
    :class:`TransitionMiner` run first so their output is visible
    to the credence-update pass in the same step;
    :class:`FutilePatternMiner` runs next, also contributing
    hypotheses; :class:`SurpriseMiner` runs last because it needs
    access to the just-updated WorldState to evaluate committed
    hypotheses against events.
    """
    return [
        PropertyObservedMiner(),
        TransitionMiner(),
        FutilePatternMiner(),
        SurpriseMiner(),
    ]
