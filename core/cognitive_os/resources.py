"""
resources.py — Depleting-resource tracking and budget-aware planner consultation.

Problem this module solves
--------------------------
Many tasks have a bounded resource that depletes with action and, when
it hits a floor, causes failure: step counter in a puzzle, battery on a
robot, time-on-task, tokens of LLM budget, gripper-actuation cycles.
A substrate that doesn't reason about such resources will cheerfully
plan a 50-step detour when 25 steps remain, then "inexplicably" fail
and misattribute the failure to whatever it was doing at the moment.

This module gives the substrate two things:

1. A **first-class representation** of depleting resources — name,
   current value, critical threshold, decay-per-action estimate,
   refill triggers.
2. A **budget-aware planner consultation** — given a proposed path
   cost, does the current budget cover it? If not, does a refill
   subgoal fit? What's the nearest refill?

Both are domain-agnostic; both are exercised by both use cases.

Two-usecase fit
---------------
* **ARC-AGI-3 (ls20 L2):** step counter (HUD bar `color11`) depletes
  each step; hitting zero resets the level (one of two lives). Refill
  stations restore it. The planner must refill before a long leg.
* **Robotics:** battery pack depletes; dock-charging restores. Also
  gripper-cycle wear (replace tool before N cycles exhausted),
  conversation/LLM budget (switch to a smaller model when under 10%
  remaining). All plug into the same interface.

Relationship to ``causal.py``
-----------------------------
A ``resource_decay`` hypothesis in the registry — declared by
``seed_resource_decay_hypothesis`` — is the single source of truth for:
  * ``CausalAttributor`` / ``resource_exhaustion_hook`` (Gap 1)
  * ``ResourceTracker`` (this module — Gap 2)
Both read the same metadata (observation_key, critical_threshold,
decay_per_action). Changing a resource's definition once updates
attribution and budgeting in lockstep.

Design principles
-----------------
* **Observed value wins over modeled value.** The tracker estimates
  decay-per-action as a heuristic, but when a fresh observation
  arrives (the actual HUD bar size, the actual battery %) it
  overwrites the estimate. Model error doesn't compound.
* **Refills are hypotheses, not hard-coded cells.** Refill triggers
  live in the registry as ``refill_at_pos`` (ARC) / ``dock_at_pose``
  (robotics) hypotheses. The tracker never stores refill locations
  itself — it queries and ranks.
* **Abstention over guessing.** If a resource's decay rate is
  unknown and no observation has been taken since the last proposal,
  the tracker reports ``BudgetStatus.UNKNOWN`` rather than making up
  a number. Callers decide whether to proceed cautiously or wait.
* **Robotics-analogue integrity.** Every public method is named so
  a robotics reviewer recognises it: ``can_afford`` /
  ``nearest_refill`` / ``update_from_observation``.

Scale-up path
-------------
* **Today.** One resource at a time; flat path-cost = step count.
* **Near-term.** Multi-resource budgets (both battery AND time);
  Pareto-ranked plans.
* **Mid-term.** Probabilistic decay (distribution, not point);
  risk-aware planning ("fit with 95% confidence").
* **Long-term.** Tracker asks planner for counterfactual: "would a
  detour through the refill give me a better-budgeted alternative?"
  — full closed loop with goal replacement, still no caller code
  change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .hypothesis import HypothesisRegistry, PROVISIONAL, CONFIRMED


# --------------------------------------------------------------------------- types


class BudgetStatus(str, Enum):
    """Outcome of a budget check. Callers branch on these values."""

    AFFORDABLE = "affordable"        # path fits in current budget safely
    TIGHT = "tight"                  # fits but under safety margin
    OVERRUN = "overrun"              # path exceeds current budget
    UNKNOWN = "unknown"              # decay rate or current value not yet known


@dataclass
class Resource:
    """One depleting resource.

    Attributes
    ----------
    name
        Short identifier — ``"step_counter"``, ``"battery"``,
        ``"gripper_cycles"``. Shared with the ``resource_decay``
        hypothesis subject.
    current
        Most recently observed value.
    critical_threshold
        Value at or below which failure / e-stop is expected.
    decay_per_action
        Estimated drop per action. May be refined from observations.
    safety_margin
        Extra budget to keep on top of a plan's cost before calling it
        affordable. Expressed in the same units as ``current``.
    max_known
        Highest value ever observed — proxy for "full tank" used when
        planning refills.
    observation_key
        Key in an observation snapshot that carries the live value.
        Lets ``update_from_observation`` be called with a plain dict.
    refill_predicate
        Hypothesis predicate under which refill locations are stored
        (default ``"refill_at_pos"`` — ARC). Robotics might set
        ``"dock_at_pose"``.
    """

    name: str
    current: float = float("nan")
    critical_threshold: float = 0.0
    decay_per_action: Optional[float] = None
    safety_margin: float = 0.0
    max_known: float = float("nan")
    observation_key: Optional[str] = None
    refill_predicate: str = "refill_at_pos"

    def is_known(self) -> bool:
        return not _is_nan(self.current)

    def headroom(self) -> Optional[float]:
        """Distance above critical threshold. None if unknown."""
        if not self.is_known():
            return None
        return max(0.0, float(self.current) - float(self.critical_threshold))


def _is_nan(x: Any) -> bool:
    try:
        return x != x  # nan is the only value that is not equal to itself
    except Exception:
        return False


# ------------------------------------------------------------------- tracker


class ResourceTracker:
    """Per-resource budget and refill awareness.

    Typical wiring::

        tracker = ResourceTracker(registry)
        tracker.register(Resource(
            name="step_counter",
            critical_threshold=0,
            decay_per_action=1.0,
            safety_margin=4.0,
            observation_key="step_counter_value",
            refill_predicate="refill_at_pos",
        ))
        # Each cycle:
        tracker.update_from_observation({"step_counter_value": 23, ...})
        status = tracker.can_afford("step_counter", path_cost=18)
        if status is BudgetStatus.OVERRUN:
            refill = tracker.nearest_refill("step_counter",
                                            from_pos=player_pos,
                                            distance_fn=bfs_distance)
            if refill is not None:
                planner.insert_subgoal(goto=refill)

    Parameters
    ----------
    registry
        Shared HypothesisRegistry. Used to (a) auto-discover resources
        already declared via ``seed_resource_decay_hypothesis`` and
        (b) query refill locations.
    autoseed_from_registry
        If True, on construction the tracker scans the registry for
        ``resource_decay`` hypotheses and registers a Resource for
        each. Reduces boilerplate.
    """

    def __init__(
        self,
        registry: HypothesisRegistry,
        *,
        autoseed_from_registry: bool = True,
    ) -> None:
        self.registry = registry
        self._resources: Dict[str, Resource] = {}
        if autoseed_from_registry:
            self._autoseed()

    # ---- registration ------------------------------------------------------

    def register(self, resource: Resource) -> None:
        """Add or replace a Resource.

        Replacing is allowed so that a later, better-specified Resource
        (e.g. learned decay rate) can supersede an autoseeded one.
        """
        self._resources[resource.name] = resource

    def has(self, name: str) -> bool:
        return name in self._resources

    def get(self, name: str) -> Optional[Resource]:
        return self._resources.get(name)

    @property
    def resources(self) -> List[Resource]:
        return list(self._resources.values())

    def _autoseed(self) -> None:
        for h in self.registry.query(predicate="resource_decay",
                                     status_in=(PROVISIONAL, CONFIRMED)):
            cond = h.conditions or {}
            name = h.subject if isinstance(h.subject, str) else str(h.subject)
            if name in self._resources:
                continue
            self._resources[name] = Resource(
                name=name,
                critical_threshold=float(cond.get("critical_threshold", 0.0)),
                decay_per_action=(
                    float(cond["decay_per_action"])
                    if "decay_per_action" in cond else None
                ),
                observation_key=cond.get("observation_key"),
                refill_predicate=cond.get("refill_predicate", "refill_at_pos"),
                safety_margin=float(cond.get("safety_margin", 0.0)),
            )

    # ---- observation ingest ------------------------------------------------

    def update_from_observation(self, snapshot: Dict[str, Any]) -> None:
        """Refresh each Resource's current value from an observation dict.

        Resources whose ``observation_key`` is not present are left
        unchanged — the tracker does NOT fabricate decay by subtracting
        decay_per_action; that's what predictive models are for, not
        the tracker of current truth.
        """
        for r in self._resources.values():
            if r.observation_key is None:
                continue
            if r.observation_key not in snapshot:
                continue
            raw = snapshot[r.observation_key]
            try:
                v = float(raw)
            except (TypeError, ValueError):
                continue
            r.current = v
            if _is_nan(r.max_known) or v > r.max_known:
                r.max_known = v

    def observe_decay(self, name: str, pre: float, post: float) -> None:
        """Update ``decay_per_action`` from one (pre, post) observation.

        Uses a simple exponential moving average so repeated
        observations stabilise the estimate. Ignored if pre <= post
        (treated as a refill, not a decay sample).
        """
        r = self._resources.get(name)
        if r is None or pre <= post:
            return
        sample = float(pre) - float(post)
        if r.decay_per_action is None:
            r.decay_per_action = sample
        else:
            alpha = 0.2  # weight on new sample
            r.decay_per_action = (1 - alpha) * r.decay_per_action + alpha * sample

    # ---- budgeting ---------------------------------------------------------

    def can_afford(
        self,
        name: str,
        path_cost: float,
    ) -> BudgetStatus:
        """Can the agent execute a path of ``path_cost`` actions?

        ``path_cost`` is in *action units* — concretely, "1.0" = one
        action whose decay is ``decay_per_action``. Callers who already
        know their cost in resource units can pass ``path_cost =
        resource_cost / decay_per_action`` or invert the check through
        ``remaining_actions``.
        """
        r = self._resources.get(name)
        if r is None:
            return BudgetStatus.UNKNOWN
        if not r.is_known() or r.decay_per_action is None:
            return BudgetStatus.UNKNOWN
        drain = float(path_cost) * float(r.decay_per_action)
        post = float(r.current) - drain
        critical = float(r.critical_threshold)
        if post < critical:
            return BudgetStatus.OVERRUN
        if post < critical + float(r.safety_margin):
            return BudgetStatus.TIGHT
        return BudgetStatus.AFFORDABLE

    def remaining_actions(self, name: str) -> Optional[float]:
        """Estimated number of actions until critical. None if unknown."""
        r = self._resources.get(name)
        if r is None or not r.is_known() or not r.decay_per_action:
            return None
        headroom = r.headroom()
        if headroom is None:
            return None
        return headroom / float(r.decay_per_action)

    # ---- refill discovery --------------------------------------------------

    def refill_subjects(self, name: str) -> List[Any]:
        """All known refill-location subjects for a resource.

        Queries the registry for ``<refill_predicate>`` hypotheses.
        Returns subjects (positions, poses, dock ids) — the caller
        decides reachability.
        """
        r = self._resources.get(name)
        if r is None:
            return []
        return [
            h.subject
            for h in self.registry.query(predicate=r.refill_predicate,
                                         status_in=(PROVISIONAL, CONFIRMED))
        ]

    def nearest_refill(
        self,
        name: str,
        *,
        from_pos: Any,
        distance_fn: Callable[[Any, Any], Optional[float]],
    ) -> Optional[Tuple[Any, float]]:
        """Find the refill subject with smallest ``distance_fn(from_pos, s)``.

        ``distance_fn`` is domain-supplied:
          * ARC: BFS path length in grid cells.
          * Robotics: great-circle distance / dock-travel estimate.
        Returning None from ``distance_fn`` means unreachable; those
        subjects are skipped.
        """
        best: Optional[Tuple[Any, float]] = None
        for s in self.refill_subjects(name):
            d = distance_fn(from_pos, s)
            if d is None:
                continue
            if best is None or d < best[1]:
                best = (s, float(d))
        return best

    # ---- introspection -----------------------------------------------------

    def status_line(self, name: str) -> str:
        r = self._resources.get(name)
        if r is None:
            return f"{name}: unregistered"
        cur = "?" if not r.is_known() else f"{r.current:g}"
        rate = "?" if r.decay_per_action is None else f"{r.decay_per_action:g}"
        remaining = self.remaining_actions(name)
        rem_s = "?" if remaining is None else f"{remaining:.1f}"
        return (f"{r.name}: current={cur} decay/act={rate} crit={r.critical_threshold:g} "
                f"safety={r.safety_margin:g} actions_left={rem_s}")


__all__ = [
    "BudgetStatus",
    "Resource",
    "ResourceTracker",
]
