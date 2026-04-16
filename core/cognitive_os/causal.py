"""
causal.py — Competing-cause attribution for the Cognitive OS.

Problem this module solves
--------------------------
When something bad happens — the agent loses a life, the robot e-stops, a
plan fails — the naive attribution is "whatever I was doing right before
caused it". That attribution lands in some blacklist (lethal cells, unsafe
skills, banned waypoints) and then the planner routes around the blamed
thing forever.

The failure mode is *misattribution* — the surface cause (I was standing
at cell X) was coincidental with the actual cause (my battery was at 5%
and would have quit no matter where I was standing). A substrate that
can't tell these apart pollutes its own safety knowledge.

The fix is generic and applies far beyond ARC or robotics: **before a
miner attributes a surprise to the most-recent (state, action), poll the
hypothesis registry for *competing explanations* whose values at the
time of the surprise would also predict it. If any is in its danger
regime, attribute to that instead.**

Two-usecase fit
---------------
* **ARC-AGI-3 (ls20 L2):** agent dies; `lethal_at_pos` miner attributes to
  the cell. But the agent has a ``resource_decay`` hypothesis about
  ``color11`` being the step counter, and the counter was near zero on
  the fatal tick. Attribution should go to resource-exhaustion; the cell
  must NOT enter the lethal set.
* **Robotics:** robot face-plants; a naive miner logs "trajectory-shape
  X is unsafe" and blocks it. But a ``resource_decay`` hypothesis tracks
  battery, and battery was at 4%. Attribution goes to power-exhaustion;
  the trajectory is fine.

Design principles
-----------------
* **Competing hypotheses, not rules.** The alternative causes are held
  in the same ``HypothesisRegistry`` as everything else. Any module can
  register a new one by writing a ``resource_decay`` (or future
  ``slow_fault``, ``phase_hazard``, ...) hypothesis with the right
  metadata. The attributor is agnostic to the domain of each.
* **Evidence-based confidence.** The attributor returns a ranked list
  of candidate causes with confidence scores, not a single answer.
  Callers pick the top candidate (default policy) or apply their own
  policy (e.g. always-attribute-conservatively).
* **Separable from the miner.** The miner asks the attributor; the
  attributor consults the registry. Miners stay simple; swapping the
  attributor (better logic, learned classifier) doesn't touch the
  miner.
* **Graceful unknown.** If no competing hypothesis applies, the
  attributor reports ``cause="unknown"`` with the surface attribution
  as a fallback. Callers decide whether to trust the surface or abstain.

Interface contract
------------------
``Surprise`` describes *what* went wrong, including the tick snapshot.
``CauseCandidate`` is one possible explanation. ``CausalAttributor``
takes a Surprise and returns a ranked list of candidates.

Scale-up path
-------------
* **Today.** Hand-coded alternative-cause rules that read
  ``resource_decay`` hypotheses and check a single threshold.
* **Near-term.** More hypothesis kinds: ``phase_hazard`` (cycle-driven
  danger), ``cumulative_wear`` (slow-accumulating faults),
  ``environmental_precondition`` (e.g. light level).
* **Mid-term.** Learned attributor — a classifier trained on
  (surprise, competing-hypothesis-states) → correct-cause labels
  produced by human feedback or counterfactual replays.
* **Long-term.** The attributor itself becomes an agent that can
  *request new observations* (probe actions) to disambiguate ties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple

from .hypothesis import Hypothesis, HypothesisRegistry, PROVISIONAL, CONFIRMED


# --------------------------------------------------------------------------- types


@dataclass(frozen=True)
class Surprise:
    """A negative/unexpected outcome whose cause needs attribution.

    Attributes
    ----------
    kind
        Short tag — ``"life_loss"``, ``"collision"``, ``"plan_failure"``,
        ``"e_stop"``. Attributors may specialise on this.
    subject
        What the surface evidence points at: e.g. ``(cell_x, cell_y)``
        for a spatial event, a skill name for a motion failure.
    tick_idx
        When it happened — used to inspect resource states as of that
        tick.
    observation_snapshot
        Dict of observable state at the tick — ``{"step_counter": 2,
        "battery_pct": 4, ...}``. Attributors look up the keys they
        care about.
    context
        Freeform metadata (planner cycle, agent_id, …).
    """

    kind: str
    subject: Any
    tick_idx: int = -1
    observation_snapshot: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CauseCandidate:
    """One possible explanation for a Surprise.

    Attributes
    ----------
    cause
        Short tag — ``"resource_exhaustion"``, ``"position"``,
        ``"phase_hazard"``, ``"unknown"``. Callers branch on this.
    confidence
        0.0–1.0. The attributor's subjective belief that this is the
        right explanation.
    source_hypothesis_key
        Optional registry key of the hypothesis that fired this
        candidate — for audit.
    subject
        The thing the cause points at: a resource name, a position, …
        Mirrors ``Surprise.subject`` semantics but may be different
        when the cause shifts attribution.
    reason
        Human-readable explanation suitable for logs.
    """

    cause: str
    confidence: float = 0.0
    source_hypothesis_key: Optional[tuple] = None
    subject: Any = None
    reason: str = ""


class AlternativeCauseHook(Protocol):
    """Callable that checks one family of alternative causes.

    Implementations read the registry + observation_snapshot and
    return CauseCandidates for any alternative cause that applies.
    Return an empty list to abstain.
    """

    def __call__(
        self,
        surprise: Surprise,
        registry: HypothesisRegistry,
    ) -> List[CauseCandidate]: ...


# ------------------------------------------------------------------- attributor


class CausalAttributor:
    """Consult competing hypotheses before accepting a surface attribution.

    Typical wiring::

        attributor = CausalAttributor(registry)
        attributor.register_hook(resource_exhaustion_hook)
        attributor.register_hook(phase_hazard_hook)   # future
        ...
        # In a miner that previously did:
        #     lethal_set.add(target_cell)
        # Do instead:
        candidates = attributor.attribute(Surprise(
            kind="life_loss", subject=target_cell,
            tick_idx=i, observation_snapshot=snap))
        top = candidates[0]
        if top.cause == "position":
            lethal_set.add(target_cell)
        elif top.cause == "resource_exhaustion":
            # Do NOT blame the cell. Log for diagnosis.
            pass

    Parameters
    ----------
    registry
        Shared HypothesisRegistry. Hooks read from it.
    surface_threshold
        Minimum confidence for a competing cause to displace the
        surface cause. Below this, surface-attribution wins but the
        competing candidate is still returned as a runner-up.
    default_surface_confidence
        Confidence the attributor assigns to the surface attribution
        when no competing cause fires. Tuned to be modest so future
        hooks can overtake it if they speak up.
    """

    SURFACE_CAUSE = "surface"  # sentinel cause tag when no hook fires

    def __init__(
        self,
        registry: HypothesisRegistry,
        *,
        surface_threshold: float = 0.6,
        default_surface_confidence: float = 0.5,
    ) -> None:
        self.registry = registry
        self._hooks: List[AlternativeCauseHook] = []
        self.surface_threshold = surface_threshold
        self.default_surface_confidence = default_surface_confidence

    def register_hook(self, hook: AlternativeCauseHook) -> None:
        """Add a cause family. Idempotent by identity."""
        if hook not in self._hooks:
            self._hooks.append(hook)

    def attribute(
        self,
        surprise: Surprise,
        *,
        surface_candidate: Optional[CauseCandidate] = None,
    ) -> List[CauseCandidate]:
        """Rank explanations for ``surprise``.

        Returns a list of CauseCandidates sorted by confidence
        descending. The first element is the preferred attribution;
        callers who want strict surface-only can inspect the list
        instead of blindly taking [0].

        ``surface_candidate`` is what the miner would have attributed
        in the absence of competing causes — usually ``CauseCandidate(
        cause="position", subject=surprise.subject, …)``. If omitted
        a default is synthesised so callers don't have to.
        """
        if surface_candidate is None:
            surface_candidate = CauseCandidate(
                cause="surface",
                confidence=self.default_surface_confidence,
                subject=surprise.subject,
                reason="surface attribution (no alternative found)",
            )

        competing: List[CauseCandidate] = []
        for hook in self._hooks:
            try:
                competing.extend(hook(surprise, self.registry))
            except Exception as e:  # pragma: no cover — defensive
                # A broken hook must not brick the attributor.
                competing.append(CauseCandidate(
                    cause="unknown",
                    confidence=0.0,
                    reason=f"hook raised {type(e).__name__}: {e}",
                ))

        # If any competing candidate beats the threshold, it takes the
        # top slot. Otherwise surface wins but we still keep the competing
        # list so callers can see what almost fired.
        competing.sort(key=lambda c: c.confidence, reverse=True)
        top_competitor = competing[0] if competing else None
        if top_competitor is not None and top_competitor.confidence >= self.surface_threshold:
            return [top_competitor] + [surface_candidate] + competing[1:]
        return [surface_candidate] + competing


# --------------------------------------------------- resource-exhaustion hook


def resource_exhaustion_hook(
    surprise: Surprise,
    registry: HypothesisRegistry,
) -> List[CauseCandidate]:
    """Check every ``resource_decay`` hypothesis for critical-level firing.

    A ``resource_decay`` hypothesis has:
      * ``subject`` = resource name (e.g. ``"step_counter"``,
        ``"battery"``)
      * ``conditions["critical_threshold"]`` (numeric) — below this the
        resource is deemed "about to cause failure"
      * ``conditions["observation_key"]`` — key in
        ``surprise.observation_snapshot`` that carries the current
        value (e.g. ``"step_counter_value"``)
      * ``conditions["decay_per_action"]`` (numeric, optional) — how
        much the value drops per action. Used to look one step ahead.

    Look-ahead semantics
    --------------------
    The surprise's ``observation_snapshot`` carries the **pre-fatal**
    value (the value AFTER the action just before the death — i.e.
    what was read when the planner picked the next action). The fatal
    action then decays the value by one more ``decay_per_action`` step.
    So the hook fires when ``v - decay_per_action ≤ critical_threshold``
    (the next action would have crossed the line) — not just
    ``v ≤ critical_threshold``. With ``decay_per_action=0`` (or
    unspecified) it falls back to the strict ≤ check.

    Without this look-ahead the hook silently misses every
    one-step-out-of-budget death — exactly the failure mode that lets
    a counter-exhaustion death poison the lethal-cell set with the
    cell the agent happened to be entering.

    Confidence scaling: 0.95 when the post-fatal value is at or below
    half the critical threshold (clearly exhausted), 0.7 when just at
    threshold, scaled in between.
    """
    out: List[CauseCandidate] = []
    decay_hs = registry.query(
        predicate="resource_decay",
        status_in=(PROVISIONAL, CONFIRMED),
    )
    snap = surprise.observation_snapshot
    for h in decay_hs:
        cond = h.conditions or {}
        obs_key = cond.get("observation_key")
        thresh = cond.get("critical_threshold")
        if obs_key is None or thresh is None:
            continue
        val = snap.get(obs_key)
        if val is None:
            continue
        try:
            v = float(val)
            t = float(thresh)
            decay = float(cond.get("decay_per_action") or 0.0)
        except (TypeError, ValueError):
            continue
        # Project the fatal action's effect forward by one decay step,
        # so a pre-fatal observation can still flag the impending
        # exhaustion the next action will cause.
        v_after = v - max(0.0, decay)
        if v_after > t:
            continue
        # Scale confidence on the projected post-fatal value: at
        # threshold -> 0.7; at half or below -> 0.95.
        if t <= 0:
            conf = 0.95 if v_after <= 0 else 0.7
        else:
            ratio = max(0.0, min(1.0, v_after / t))
            conf = 0.95 - 0.25 * ratio  # 0 -> 0.95, t -> 0.70
        out.append(CauseCandidate(
            cause="resource_exhaustion",
            confidence=conf,
            source_hypothesis_key=h.key(),
            subject=h.subject,
            reason=(
                f"resource {h.subject!r} = {v} (projected next-step "
                f"{v_after}) ≤ critical threshold {t} at tick "
                f"{surprise.tick_idx}"
            ),
        ))
    return out


# ----------------------------------------------------- hypothesis-seed helper


def seed_resource_decay_hypothesis(
    registry: HypothesisRegistry,
    *,
    resource_name: str,
    observation_key: str,
    critical_threshold: float,
    decay_per_action: Optional[float] = None,
    step_seen: int = -1,
    label: str = "",
) -> Hypothesis:
    """Convenience: register (or fetch) a ``resource_decay`` hypothesis.

    Callers from any adapter can use this to declare "I believe X is a
    depleting resource that causes failure when it hits Y" and
    subsequent cause attribution + resource budgeting will pick it up
    automatically.

    Parameters mirror the conditions used by the hooks and the
    ResourceTracker — one declaration feeds both subsystems.
    """
    conditions: Dict[str, Any] = {
        "observation_key": observation_key,
        "critical_threshold": float(critical_threshold),
    }
    if decay_per_action is not None:
        conditions["decay_per_action"] = float(decay_per_action)
    existing = registry.get("resource_decay", resource_name, conditions)
    if existing is not None:
        return existing
    h = Hypothesis(
        predicate="resource_decay",
        subject=resource_name,
        status=PROVISIONAL,
        support=1,
        against=0,
        conditions=conditions,
        first_seen_step=step_seen,
        last_updated_step=step_seen,
        label=label or f"{resource_name} depletes; failure at ≤ {critical_threshold}",
    )
    return registry.add(h)


__all__ = [
    "Surprise",
    "CauseCandidate",
    "AlternativeCauseHook",
    "CausalAttributor",
    "resource_exhaustion_hook",
    "seed_resource_decay_hypothesis",
]
