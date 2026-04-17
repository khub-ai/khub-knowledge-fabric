"""
dsl.py — Internal symbolic state for the KF agent.

This is the *mutable* internal representation that the symbolic core
maintains and updates. It is distinct from `proposer_schema.py`, which
defines the *immutable wire format* exchanged with the LLM/VLM Proposer.

Separation of concerns:

  proposer_schema.py    — what the Proposer sees (frozen, schema-bound)
  dsl.py                — what the symbolic core owns (mutable, richer)

The two layers share enum vocabularies (Role, EffectType, PreconditionType,
GoalPredicateType, …) so that DSL state can be projected into a
`ProposalContext` for the LLM with no vocabulary translation. Internal types
carry richer fields (full cell masks, transition histories, mutable
posteriors) that are deliberately *not* exposed to the LLM.

This file defines:
  - DslObject, DslRegion, DslGestaltFeature, DslTransition
  - ActionModelHypothesis    (one candidate world-model for an action)
  - RoleBelief               (multinomial distribution over Role for an object)
  - GoalBelief               (one candidate goal predicate with posterior)
  - HypothesisTracker        (the Bayesian state machine; v0 is a stub that
                              accumulates observations and exposes top-k
                              queries, no real Bayesian update yet)

The tracker is intentionally a *skeleton*: enough structure for the
MockProposer and the unit tests to exercise the contract end-to-end, but
without the actual Bayesian arithmetic. That arithmetic comes in a later
implementation step (see DSL doc Layer 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from proposer_schema import (
    # Enums — single source of truth
    Confidence,
    EffectType,
    GoalPredicateType,
    Label,
    PreconditionType,
    Role,
    # Wire-format types we project into
    ActionModelSummary,
    GestaltFeature as WireGestaltFeature,
    GoalCandidate,
    ObjectSummary,
    ProposalContext,
    RegionSummary,
    RoleHypothesis,
    TransitionRecord,
)


# =============================================================================
# Perception state (mutable, owned by the symbolic core)
# =============================================================================

@dataclass
class DslObject:
    """Internal object record. Richer than the wire ObjectSummary."""
    id:    int
    color: int
    cells: set[tuple[int, int]] = field(default_factory=set)
    is_background: bool = False
    # Per-action effect history: action_id -> ordered list of EffectType seen.
    observed_effects: dict[str, list[EffectType]] = field(default_factory=dict)
    # Step at which this object was first detected; used for stability.
    first_seen_step: int = 0

    @property
    def area(self) -> int:
        return len(self.cells)

    @property
    def centroid(self) -> tuple[float, float]:
        if not self.cells:
            return (0.0, 0.0)
        rs = sum(r for r, _ in self.cells)
        cs = sum(c for _, c in self.cells)
        n = len(self.cells)
        return (rs / n, cs / n)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        if not self.cells:
            return (0, 0, 0, 0)
        rs = [r for r, _ in self.cells]
        cs = [c for _, c in self.cells]
        return (min(rs), min(cs), max(rs), max(cs))

    def to_summary(self) -> ObjectSummary:
        return ObjectSummary(
            id=self.id,
            color=self.color,
            area=self.area,
            centroid=self.centroid,
            bbox=self.bbox,
            is_background=self.is_background,
            observed_effects={k: [e.value for e in v]
                              for k, v in self.observed_effects.items()},
        )


@dataclass
class DslRegion:
    id:   int
    kind: str                                 # see RegionSummary.kind literal
    bbox: tuple[int, int, int, int]
    contains_object_ids: list[int] = field(default_factory=list)

    def to_summary(self) -> RegionSummary:
        return RegionSummary(
            id=self.id, kind=self.kind, bbox=self.bbox,
            contains_object_ids=list(self.contains_object_ids),
        )


@dataclass
class DslGestaltFeature:
    kind:     str
    score:    float
    location: tuple[int, int] | None = None
    detail:   str | None = None

    def to_summary(self) -> WireGestaltFeature:
        return WireGestaltFeature(kind=self.kind, score=self.score,
                                  location=self.location, detail=self.detail)


@dataclass
class DslTransition:
    """One observed (state, action, next_state) transition, internal form."""
    step:                int
    action_id:           str
    objects_moved:       list[int] = field(default_factory=list)
    objects_appeared:    list[int] = field(default_factory=list)
    objects_disappeared: list[int] = field(default_factory=list)
    pixel_diff_count:    int = 0
    level_advanced:      bool = False

    def to_record(self) -> TransitionRecord:
        return TransitionRecord(
            step=self.step,
            action_id=self.action_id,
            objects_moved=list(self.objects_moved),
            objects_appeared=list(self.objects_appeared),
            objects_disappeared=list(self.objects_disappeared),
            pixel_diff_count=self.pixel_diff_count,
            level_advanced=self.level_advanced,
        )


# =============================================================================
# Frame-level diff primitives — used by the Bayesian updater
# =============================================================================

def _dominant_color(frame: list[list[int]]) -> int:
    counts: dict[int, int] = {}
    for row in frame:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _frame_equal(a: list[list[int]], b: list[list[int]]) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if ra != rb:
            return False
    return True


def _detect_translate(
    prev: list[list[int]], curr: list[list[int]], bg_color: int
) -> tuple[int, int] | None:
    """Detect a uniform translation of a single non-background object.
    Returns (dr, dc) or None.
    """
    h = len(prev)
    w = len(prev[0]) if h else 0
    if h == 0 or (h, w) != (len(curr), len(curr[0]) if curr else 0):
        return None
    from_by: dict[int, set[tuple[int, int]]] = {}
    to_by:   dict[int, set[tuple[int, int]]] = {}
    for r in range(h):
        for c in range(w):
            p, q = prev[r][c], curr[r][c]
            if p != q:
                from_by.setdefault(p, set()).add((r, c))
                to_by.setdefault(q, set()).add((r, c))
    if not from_by:
        return None
    candidates: list[tuple[int, int]] = []
    for color, fc in from_by.items():
        if color == bg_color:
            continue
        tc = to_by.get(color)
        if tc is None or len(tc) != len(fc) or not fc:
            continue
        fr_r = sum(r for r, _ in fc) / len(fc)
        fr_c = sum(c for _, c in fc) / len(fc)
        to_r = sum(r for r, _ in tc) / len(tc)
        to_c = sum(c for _, c in tc) / len(tc)
        dr = round(to_r - fr_r)
        dc = round(to_c - fr_c)
        if (dr, dc) == (0, 0):
            continue
        if all((r + dr, c + dc) in tc for r, c in fc):
            candidates.append((dr, dc))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Multiple consistent candidates — prefer the smallest magnitude.
    return min(candidates, key=lambda d: abs(d[0]) + abs(d[1]))


def _detect_uniform_recolor(
    prev: list[list[int]], curr: list[list[int]], bg_color: int
) -> tuple[int, int, set[tuple[int, int]]] | None:
    """Detect a contiguous region whose color uniformly changed C1 → C2.
    Returns (from_color, to_color, cell_set) or None.
    """
    h = len(prev)
    w = len(prev[0]) if h else 0
    if (h, w) != (len(curr), len(curr[0]) if curr else 0):
        return None
    cells: set[tuple[int, int]] = set()
    from_color: int | None = None
    to_color:   int | None = None
    for r in range(h):
        for c in range(w):
            p, q = prev[r][c], curr[r][c]
            if p == q:
                continue
            if from_color is None:
                from_color, to_color = p, q
            elif p != from_color or q != to_color:
                return None
            cells.add((r, c))
    if not cells or from_color is None or to_color is None:
        return None
    return (from_color, to_color, cells)


def _enumerate_candidate_models(
    action_id: str,
    prev_frame: list[list[int]],
    curr_frame: list[list[int]],
    bg_color:   int,
) -> list["ActionModelHypothesis"]:
    """Generate plausible new hypotheses consistent with one observed transition."""
    out: list[ActionModelHypothesis] = []
    if _frame_equal(prev_frame, curr_frame):
        out.append(ActionModelHypothesis(
            action_id=action_id, effect_type=EffectType.NO_OP,
            description="no observed change",
        ))
        return out
    t = _detect_translate(prev_frame, curr_frame, bg_color)
    if t is not None:
        dr, dc = t
        out.append(ActionModelHypothesis(
            action_id=action_id,
            effect_type=EffectType.TRANSLATE,
            precondition=PreconditionType.IF_BLOCKED,
            params={"dr": dr, "dc": dc},
            description=f"translate ({dr},{dc}) IF_BLOCKED",
        ))
    rc = _detect_uniform_recolor(prev_frame, curr_frame, bg_color)
    if rc is not None:
        _, _, cells = rc
        out.append(ActionModelHypothesis(
            action_id=action_id,
            effect_type=EffectType.TOGGLE,
            params={"cells": frozenset(cells)},
            description=f"recolor {len(cells)} cells",
        ))
    return out


def _models_equivalent(a: "ActionModelHypothesis",
                       b: "ActionModelHypothesis") -> bool:
    if a.effect_type != b.effect_type:
        return False
    if a.precondition != b.precondition:
        return False
    return a.params == b.params


# =============================================================================
# Hypothesis state (the Bayesian beliefs the tracker maintains)
# =============================================================================

@dataclass
class ActionModelHypothesis:
    """One candidate model for what an action does."""
    action_id:      str
    effect_type:    EffectType
    precondition:   PreconditionType = PreconditionType.ALWAYS
    support:        int = 0
    contradictions: int = 0
    description:    str = ""
    # Sticky prior, used to seed posterior before any evidence.
    prior_weight:   float = 0.1
    # Effect-specific parameters. Mirrors the synthetic_games HiddenEffect schema:
    #   TRANSLATE: {"dr": int, "dc": int}
    #   TOGGLE:    {"cells": frozenset[(r,c)]}
    #   NO_OP:     {}
    params:         dict = field(default_factory=dict)

    @property
    def posterior(self) -> float:
        """Beta(α+support, β+contradictions) mean — proper conjugate update."""
        alpha = 1.0 + self.prior_weight * 10
        beta  = 1.0 + (1.0 - self.prior_weight) * 5
        s = alpha + self.support
        f = beta  + self.contradictions
        return s / (s + f)

    def consistent_with(
        self,
        prev_frame: list[list[int]],
        curr_frame: list[list[int]],
        bg_color:   int,
    ) -> bool:
        """Return True if the observed prev→curr transition does not
        contradict this hypothesis. Used by the Bayesian updater."""
        et = self.effect_type
        equal = _frame_equal(prev_frame, curr_frame)

        if et == EffectType.NO_OP:
            return equal

        if et == EffectType.TRANSLATE:
            want = (self.params.get("dr", 0), self.params.get("dc", 0))
            if equal:
                # IF_BLOCKED preconditions allow "no movement" as consistent.
                return self.precondition == PreconditionType.IF_BLOCKED
            detected = _detect_translate(prev_frame, curr_frame, bg_color)
            return detected is not None and detected == want

        if et == EffectType.TOGGLE:
            if equal:
                return False
            detected = _detect_uniform_recolor(prev_frame, curr_frame, bg_color)
            if detected is None:
                return False
            _, _, cells = detected
            want_cells = self.params.get("cells")
            return want_cells is None or frozenset(cells) == want_cells

        # Unknown / unsupported effect — treat as inconsistent so it gets pruned.
        return False

    def to_summary(self) -> ActionModelSummary:
        return ActionModelSummary(
            action_id=self.action_id,
            effect_type=self.effect_type,
            precondition=self.precondition,
            posterior=self.posterior,
            support=self.support,
            contradictions=self.contradictions,
            description=self.description or
                f"{self.action_id} → {self.effect_type.value}"
                f" ({self.precondition.value})",
        )


@dataclass
class RoleBelief:
    """Multinomial distribution over Role for one object."""
    object_id:    int
    distribution: dict[Role, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.distribution:
            # Uniform prior over all roles, slightly favoring DECORATION
            # so unknown objects default to "background until proven otherwise".
            self.distribution = {r: 1.0 for r in Role}
            self.distribution[Role.DECORATION] = 2.0
        self._normalize()

    def _normalize(self) -> None:
        total = sum(self.distribution.values())
        if total > 0:
            for k in list(self.distribution.keys()):
                self.distribution[k] /= total

    def update(self, role: Role, weight: float) -> None:
        self.distribution[role] = self.distribution.get(role, 0.0) + weight
        self._normalize()

    def top(self) -> tuple[Role, float]:
        return max(self.distribution.items(), key=lambda kv: kv[1])

    def to_hypothesis(self) -> RoleHypothesis:
        role, p = self.top()
        if p > 0.7:
            conf = Confidence.HIGH
        elif p > 0.4:
            conf = Confidence.MEDIUM
        else:
            conf = Confidence.LOW
        return RoleHypothesis(
            object_id=self.object_id, role=role, confidence=conf,
            label=Label.GUESS,
        )


@dataclass
class GoalBelief:
    predicate:   GoalPredicateType
    posterior:   float
    description: str
    target_object_ids: list[int] = field(default_factory=list)

    def to_candidate(self) -> GoalCandidate:
        return GoalCandidate(
            predicate=self.predicate, posterior=self.posterior,
            description=self.description,
            target_object_ids=list(self.target_object_ids),
        )


# =============================================================================
# Hypothesis tracker — Bayesian state machine (v0 skeleton)
# =============================================================================

class HypothesisTracker:
    """Owns the symbolic core's beliefs.

    v0 responsibilities:
      - Hold objects, regions, gestalt features, transitions.
      - Hold action-model hypotheses, role beliefs, goal candidates.
      - Project current state into a ProposalContext for the LLM.
      - Integrate ProposalResponse content back into beliefs (with a
        configurable prior_weight per the Constrained/Frontier mode policy).
      - Expose top-k queries for the planner.

    Out of scope for v0 (deferred to Layer 5 implementation):
      - Real Bayesian updates with proper likelihoods.
      - Lazy enumeration of new candidate models on contradiction.
      - Hypothesis pruning under a posterior floor.
      - Information-gain computation for explore-mode action selection.
    """

    def __init__(self, available_actions: Iterable[str]) -> None:
        self.available_actions: list[str] = list(available_actions)
        self.frame_shape: tuple[int, int] = (0, 0)
        self.objects: dict[int, DslObject] = {}
        self.regions: dict[int, DslRegion] = {}
        self.gestalt: list[DslGestaltFeature] = []
        self.transitions: list[DslTransition] = []
        self.action_models: dict[str, list[ActionModelHypothesis]] = {
            a: [] for a in self.available_actions
        }
        self.role_beliefs: dict[int, RoleBelief] = {}
        self.goal_beliefs: list[GoalBelief] = []

    # ------ ingestion ------------------------------------------------------

    def set_frame_shape(self, shape: tuple[int, int]) -> None:
        self.frame_shape = shape

    def add_object(self, obj: DslObject) -> None:
        self.objects[obj.id] = obj
        if obj.id not in self.role_beliefs and not obj.is_background:
            self.role_beliefs[obj.id] = RoleBelief(object_id=obj.id)

    def add_region(self, region: DslRegion) -> None:
        self.regions[region.id] = region

    def add_gestalt(self, feat: DslGestaltFeature) -> None:
        self.gestalt.append(feat)

    def observe(self, transition: DslTransition) -> None:
        """Record a transition record. Used by tests and for replay logging."""
        self.transitions.append(transition)

    def observe_step(
        self,
        action_id:  str,
        prev_frame: list[list[int]],
        curr_frame: list[list[int]],
    ) -> None:
        """Bayesian update over candidate ActionModelHypotheses for `action_id`.

        For each existing candidate model, increment `support` if the
        observation is consistent with the model, otherwise increment
        `contradictions`. If no surviving model has positive net evidence,
        lazily enumerate new candidates from the observed diff.
        """
        bg = _dominant_color(prev_frame)
        models = self.action_models.setdefault(action_id, [])

        # 1. Update existing candidates. Equal frames are ambiguous between
        #    NO_OP and a blocked TRANSLATE; if any non-NO_OP candidate exists
        #    we don't credit NO_OP for them, otherwise NO_OP runs away with
        #    support whenever a TRANSLATE action repeatedly hits walls.
        equal = _frame_equal(prev_frame, curr_frame)
        has_non_noop = any(m.effect_type != EffectType.NO_OP for m in models)
        for m in models:
            if m.consistent_with(prev_frame, curr_frame, bg):
                if equal and m.effect_type == EffectType.NO_OP and has_non_noop:
                    continue  # ambiguous; don't credit NO_OP
                m.support += 1
            else:
                # A single counter-example refutes NO_OP entirely: drop its
                # accumulated support so a freshly-spawned TRANSLATE can win.
                if m.effect_type == EffectType.NO_OP and not equal:
                    m.support = 0
                m.contradictions += 1

        # 2. Lazy enumeration: spawn new candidates if all current models
        #    are net-negative (or none exist yet).
        net_negative = (not models) or all(
            m.contradictions >= m.support for m in models
        )
        if net_negative:
            for new_m in _enumerate_candidate_models(
                action_id, prev_frame, curr_frame, bg
            ):
                if not any(_models_equivalent(new_m, m) for m in models):
                    new_m.support = 1   # seed with the spawning observation
                    models.append(new_m)

        # 3. Prune to top-K to keep enumeration tractable.
        self._prune_action_models(action_id, max_k=8)

        # 4. Record a lightweight transition record for replay/audit.
        self.transitions.append(DslTransition(
            step=len(self.transitions) + 1,
            action_id=action_id,
            pixel_diff_count=sum(
                1
                for r in range(len(prev_frame))
                for c in range(len(prev_frame[0]))
                if prev_frame[r][c] != curr_frame[r][c]
            ),
        ))

    def _prune_action_models(self, action_id: str, max_k: int = 8) -> None:
        models = self.action_models.get(action_id, [])
        if len(models) <= max_k:
            return
        models.sort(key=lambda m: -m.posterior)
        self.action_models[action_id] = models[:max_k]

    def seed_action_model(self, model: ActionModelHypothesis) -> None:
        self.action_models.setdefault(model.action_id, []).append(model)

    def seed_goal(self, goal: GoalBelief) -> None:
        self.goal_beliefs.append(goal)

    # ------ queries --------------------------------------------------------

    def top_action_models(self, action_id: str, k: int = 3
                          ) -> list[ActionModelHypothesis]:
        models = self.action_models.get(action_id, [])
        return sorted(models, key=lambda m: -m.posterior)[:k]

    def top_role(self, object_id: int) -> tuple[Role, float] | None:
        b = self.role_beliefs.get(object_id)
        return b.top() if b else None

    def top_goals(self, k: int = 3) -> list[GoalBelief]:
        return sorted(self.goal_beliefs, key=lambda g: -g.posterior)[:k]

    # ------ projection to wire format -------------------------------------

    def to_proposal_context(self, recent_n: int = 8) -> ProposalContext:
        return ProposalContext(
            frame_shape=self.frame_shape,
            available_actions=list(self.available_actions),
            objects=[o.to_summary() for o in self.objects.values()],
            regions=[r.to_summary() for r in self.regions.values()],
            gestalt=[g.to_summary() for g in self.gestalt],
            recent_transitions=[t.to_record()
                                for t in self.transitions[-recent_n:]],
            current_action_models=[
                m.to_summary()
                for ms in self.action_models.values() for m in ms
            ],
            current_role_hypotheses=[b.to_hypothesis()
                                     for b in self.role_beliefs.values()],
            current_goal_candidates=[g.to_candidate()
                                     for g in self.goal_beliefs],
        )

    # ------ integration of Proposer responses -----------------------------

    def integrate_role_bindings(
        self, bindings: list[RoleHypothesis], prior_weight: float
    ) -> None:
        """Fold proposer-suggested role bindings into role beliefs.

        prior_weight is the trust factor from DSL §8.3 (0.3 constrained,
        0.6 frontier). The proposer's suggestion never overwrites — it
        nudges the multinomial.
        """
        for b in bindings:
            belief = self.role_beliefs.setdefault(
                b.object_id, RoleBelief(object_id=b.object_id))
            # Stronger Confidence → larger nudge, scaled by prior_weight.
            conf_scale = {Confidence.HIGH: 3.0, Confidence.MEDIUM: 1.5,
                          Confidence.LOW: 0.5}[b.confidence]
            belief.update(b.role, prior_weight * conf_scale)

    def integrate_action_model(
        self, action_id: str, effect_type: EffectType,
        precondition: PreconditionType, description: str,
        prior_weight: float,
    ) -> None:
        """Add a proposer-suggested ActionModel to the candidate pool."""
        self.action_models.setdefault(action_id, []).append(
            ActionModelHypothesis(
                action_id=action_id,
                effect_type=effect_type,
                precondition=precondition,
                description=description,
                prior_weight=prior_weight,
            )
        )

    def integrate_goal_ranking(
        self, ranked: list[tuple[GoalPredicateType, int]], prior_weight: float
    ) -> None:
        """Reorder existing goal candidates per proposer ranking.

        Adjusts posteriors monotonically; does not introduce new goals here
        (that's handled by RankGoalsResponse.new_candidate separately).
        """
        rank_map = {p: r for p, r in ranked}
        for g in self.goal_beliefs:
            if g.predicate in rank_map:
                # Higher rank (lower number) → larger posterior boost.
                boost = max(0.0, 1.0 - 0.2 * rank_map[g.predicate])
                g.posterior = (1 - prior_weight) * g.posterior \
                              + prior_weight * boost
