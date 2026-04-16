"""
safety.py — Cognitive-OS safety-governance layer.

Today this is a lightweight set of data classes + a simple composable
ensemble of ``SafetyChecker`` instances.  The intent, and the reason the
interface is shaped the way it is, is that this is the seed of a full
**Safety Agent / Safety Ensemble** — a supervisor responsible for
vetting every proposed action the rest of the cognitive OS wants to
execute.  Future evolutions (any of which should land without breaking
callers):

* A checker can become an *agent* — backed by an LLM, a learned
  classifier, or a simulator — as long as it returns a ``SafetyVerdict``.
* ``SafetyEnsemble.evaluate`` can grow from "strict-block on any veto"
  to a learned arbiter that weighs checkers by track record.
* ``ActionProposal`` can carry richer metadata (confidence, counterfactual
  alternatives, source-agent trajectory) that more sophisticated
  checkers can reason over.

Architectural intent
--------------------
* **Caller asks, doesn't tell.**  The caller describes what it wants to
  do (``ActionProposal``); the ensemble decides whether to allow it
  (``SafetyVerdict``).  What the caller does with a block (skip,
  substitute, break-and-replan) is policy of the caller, not the
  safety layer.
* **Domain-agnostic.**  ``ActionProposal`` takes an opaque ``action``
  and a ``predicted_effect`` dict whose *keys* are a soft convention
  shared between callers and checkers.  The convention:

    ``target_pos``    — expected post-action position ``(x, y)`` in grid
                        or continuous space.
    ``ee_pose``       — expected end-effector pose (robot arm).
    ``token_output``  — expected emitted text (LLM action).
    ``resource_cost`` — expected resource spend (budget, energy).

  Checkers look up only the keys they care about and return
  ``allow=True`` when the key is absent — unknown effects are not
  blocked by default.
* **Composable.**  Multiple checkers → one ensemble → one verdict.
  Checkers can be stateless or stateful.  Order-independence is *not*
  required; the ensemble's aggregator is responsible for breaking ties.

Two-usecase fit
---------------
* ARC-AGI-3: a ``LethalPosChecker`` loaded with cells that killed the
  agent in prior episodes.  When the LLM mediator fallback proposes
  stepping onto one of them, the ensemble vetoes, the caller breaks
  the plan chunk, and the next cycle re-plans.
* Robotics: a ``CollisionChecker`` (future) reading from a persistent
  collision-risk cell map; a ``JointLimitChecker`` reading kinematic
  constraints; a ``ResourceGuard`` watching battery.  All plug into
  the same ``SafetyEnsemble`` via ``register``.

Design non-goals
----------------
* The ensemble does **not** pick a replacement action.  That is
  planning, not safety.  A checker MAY suggest alternatives via
  ``SafetyVerdict.alternatives`` — the planner chooses.
* The ensemble does **not** learn from verdict outcomes (yet).  A
  future meta-checker can subscribe to a feedback loop, but that's
  orthogonal to the current interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Protocol, Tuple, Set, Dict, Iterable


# --------------------------------------------------------------------------- types


@dataclass(frozen=True)
class ActionProposal:
    """A proposed action awaiting safety evaluation.

    The caller fills in what it knows.  Checkers look up only the
    ``predicted_effect`` keys they care about; absent keys mean the
    checker abstains (``allow=True``).

    Attributes
    ----------
    agent_id
        Short identifier of the agent/planner that produced this
        proposal (``"mediator"``, ``"bfs-planner"``, ``"skill-xyz"``).
        Useful for audit and for future per-source checker policies
        (e.g. "be stricter with LLM proposals than with BFS ones").
    action
        Opaque domain-specific handle.  ARC: action name string.
        Robotics: skill name / trajectory handle / joint-space vector.
    predicted_effect
        What the caller expects to happen if this action executes.
        Convention on keys (callers should emit, checkers should read):
        ``target_pos`` — grid or (x, y) cell.
        ``ee_pose``    — robot end-effector pose.
        ``token_output`` — LLM emission text.
        ``resource_cost`` — numeric resource drain.
    context
        Freeform metadata.  Examples: ``{"source": "llm-fallback"}``,
        ``{"confidence": 0.3}``.  Checkers may inspect but should not
        depend on any particular key being present.
    """

    agent_id: str
    action: Any
    predicted_effect: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SafetyVerdict:
    """Outcome of safety evaluation.

    Attributes
    ----------
    allow
        True → execute, False → do not.  This is the only boolean the
        caller needs to branch on.
    severity
        ``"info"`` | ``"warn"`` | ``"block"`` | ``"critical"``.
        ``info``/``warn`` may coexist with ``allow=True`` (the checker
        wants to log something without stopping execution).  ``block``
        and ``critical`` imply ``allow=False``.
    checker
        Name of the checker (or ensemble) that issued this verdict.
        The ensemble concatenates child names when aggregating.
    reason
        Human-readable explanation — surface it in logs, audit trails,
        and telemetry.  The planner may also read it to decide whether
        to try alternatives.
    alternatives
        Optional list of suggested safe alternative actions the
        checker believes would pass.  Planner may or may not consult.
    """

    allow: bool
    severity: str = "info"
    checker: str = ""
    reason: str = ""
    alternatives: List[Any] = field(default_factory=list)


class SafetyChecker(Protocol):
    """Protocol for a single safety rule.

    Implementers can be simple callables bundled as dataclasses, or
    whole agents.  Only the ``name`` attribute and ``check`` method
    matter to the ensemble.
    """

    name: str

    def check(self, proposal: ActionProposal) -> SafetyVerdict: ...


# --------------------------------------------------------------------- ensemble


Aggregator = Callable[[List[SafetyVerdict]], SafetyVerdict]


def strict_block_aggregator(verdicts: List[SafetyVerdict]) -> SafetyVerdict:
    """Default aggregation: any ``allow=False`` blocks.

    When multiple checkers block, the verdict's ``reason`` concatenates
    their individual reasons so the caller (and log reader) can see
    *why* an action was rejected by name.
    """
    if not verdicts:
        return SafetyVerdict(allow=True, checker="ensemble", reason="no checkers")
    blocks = [v for v in verdicts if not v.allow]
    if blocks:
        # Most-severe first (critical > block)
        def _rank(v: SafetyVerdict) -> int:
            return {"critical": 3, "block": 2, "warn": 1, "info": 0}.get(v.severity, 0)
        blocks.sort(key=_rank, reverse=True)
        worst = blocks[0]
        names = ",".join(v.checker or "?" for v in blocks)
        reason = " | ".join(
            f"{v.checker or '?'}: {v.reason or '(no reason given)'}" for v in blocks
        )
        # Union alternatives — planner may iterate through them.
        alts: List[Any] = []
        for v in blocks:
            for a in v.alternatives:
                if a not in alts:
                    alts.append(a)
        return SafetyVerdict(
            allow=False, severity=worst.severity,
            checker=f"ensemble({names})",
            reason=reason, alternatives=alts,
        )
    return SafetyVerdict(allow=True, severity="info",
                         checker="ensemble", reason="all clear")


class SafetyEnsemble:
    """Composable set of ``SafetyChecker`` instances.

    Callers hold a single ensemble, register checkers at init (or later
    via ``register``), and invoke ``evaluate(proposal)`` once per
    action they want to vet.  Aggregation across checker verdicts is
    pluggable via ``aggregator`` — default is strict-block.

    Usage::

        safety = SafetyEnsemble()
        safety.register(LethalPosChecker(lethal=loaded_set))
        # ... later, in the action loop:
        verdict = safety.evaluate(proposal)
        if not verdict.allow:
            log(f"[SAFETY] {verdict.severity}: {verdict.reason}")
            # caller-policy: break, substitute, skip...

    Scale-up path (future):
    * Wrap a learned arbiter as ``aggregator``.
    * Register LLM-backed checkers that run in parallel (the Protocol
      does not require sync; implement ``check`` however you want).
    * Add a feedback channel so checkers can observe which of their
      vetoes were overridden and adjust.
    """

    def __init__(
        self,
        checkers: Optional[Iterable[SafetyChecker]] = None,
        aggregator: Optional[Aggregator] = None,
        name: str = "SafetyEnsemble",
    ) -> None:
        self.name = name
        self._checkers: List[SafetyChecker] = list(checkers or [])
        self._aggregator: Aggregator = aggregator or strict_block_aggregator

    def register(self, checker: SafetyChecker) -> None:
        """Add a checker.  Idempotent by identity (same object not added twice)."""
        if checker not in self._checkers:
            self._checkers.append(checker)

    def unregister(self, checker: SafetyChecker) -> bool:
        """Remove a checker by identity.  Returns True if removed."""
        try:
            self._checkers.remove(checker)
            return True
        except ValueError:
            return False

    @property
    def checkers(self) -> List[SafetyChecker]:
        """Read-only view of registered checkers (for introspection / tests)."""
        return list(self._checkers)

    def evaluate(self, proposal: ActionProposal) -> SafetyVerdict:
        """Run every checker and aggregate verdicts.

        A checker raising an exception is logged (via the returned
        verdict) and treated as *allow=True, severity=warn* — a safety
        bug must not brick the agent.  (Callers who want stricter
        behaviour can wrap the ensemble.)
        """
        verdicts: List[SafetyVerdict] = []
        for c in self._checkers:
            try:
                verdicts.append(c.check(proposal))
            except Exception as e:  # pragma: no cover — defensive
                verdicts.append(SafetyVerdict(
                    allow=True, severity="warn",
                    checker=getattr(c, "name", c.__class__.__name__),
                    reason=f"checker raised {type(e).__name__}: {e}",
                ))
        return self._aggregator(verdicts)

    # Convenience: make the ensemble itself satisfy SafetyChecker so
    # ensembles can be nested.
    def check(self, proposal: ActionProposal) -> SafetyVerdict:
        return self.evaluate(proposal)


# ---------------------------------------------------------------- LethalPosChecker


class LethalPosChecker:
    """Block actions whose predicted ``target_pos`` falls in a set of
    cells learned to be deadly in prior episodes.

    The canonical use case is a moving hazard (ARC "changer" sprite,
    patrolling robot, moving crowd) whose spatial footprint has been
    mapped by observed life-loss events.  The checker is deliberately
    stateless about *how* the set was assembled — any
    ``update_lethal(new_set)`` caller can overwrite or extend it.

    Parameters
    ----------
    lethal
        Initial set of lethal positions.  May be empty.
    name
        Checker name, surfaced in verdicts.

    Expected ``ActionProposal.predicted_effect`` key
    ------------------------------------------------
    ``target_pos`` : ``(int, int)`` — grid cell (x, y).  Missing key
    causes the checker to abstain (``allow=True``).
    """

    def __init__(self, lethal: Optional[Iterable[Tuple[int, int]]] = None,
                 name: str = "lethal_at_pos"):
        self.name = name
        self._lethal: Set[Tuple[int, int]] = {tuple(p) for p in (lethal or [])}  # type: ignore[misc]

    @property
    def lethal(self) -> Set[Tuple[int, int]]:
        return set(self._lethal)

    def update_lethal(self, lethal: Iterable[Tuple[int, int]]) -> None:
        """Replace the lethal set with a fresh snapshot."""
        self._lethal = {tuple(p) for p in lethal}  # type: ignore[misc]

    def extend_lethal(self, extra: Iterable[Tuple[int, int]]) -> None:
        """Union additional cells into the lethal set."""
        for p in extra:
            self._lethal.add(tuple(p))  # type: ignore[arg-type]

    def check(self, proposal: ActionProposal) -> SafetyVerdict:
        target = proposal.predicted_effect.get("target_pos")
        if target is None:
            return SafetyVerdict(
                allow=True, severity="info", checker=self.name,
                reason="no target_pos predicted — abstaining",
            )
        t = (int(target[0]), int(target[1])) if hasattr(target, "__iter__") else None
        if t is None:
            return SafetyVerdict(
                allow=True, severity="warn", checker=self.name,
                reason=f"target_pos of unexpected shape: {target!r}",
            )
        if t in self._lethal:
            return SafetyVerdict(
                allow=False, severity="block", checker=self.name,
                reason=f"target {t} is in persisted lethal set "
                       f"({len(self._lethal)} cells)",
            )
        return SafetyVerdict(
            allow=True, severity="info", checker=self.name,
            reason="target clear",
        )


__all__ = [
    "ActionProposal",
    "SafetyVerdict",
    "SafetyChecker",
    "SafetyEnsemble",
    "LethalPosChecker",
    "strict_block_aggregator",
    "Aggregator",
]
