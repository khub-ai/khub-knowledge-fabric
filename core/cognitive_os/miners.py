"""
miners.py — Pattern / hypothesis miners that run over the sensory-action
stream and update a HypothesisRegistry.

Each miner is a small callable ``miner(stream, tick)`` that a
``StreamRecorder`` runs on every new tick.  Miners are deliberately
narrow: each encapsulates *one* detector.  Composing them gives the
agent a richer-than-any-single-one pattern-recognition capability.

Generic miners live here. Domain-specific ones (e.g. "every time the ARC
maze rotates, invalidate pos-delta patterns") live in the usecase
adapter and register themselves on the same StreamRecorder.

Two mining styles
-----------------
1. **Regularity miners** produce *behavioural rules* — statements of the
   form "condition C repeats outcome O with high consistency". Example:
   ``FutilePatternMiner`` produces rules like "(pos, action) is a no-op"
   after 2+ consistent no-op observations.

2. **Surprise miners** produce *provisional hypotheses* — statements of
   the form "outcome O was observed under C but was not predicted by the
   baseline model; worth testing further". Example: ``SurpriseMiner``
   sees an outcome whose magnitude is an outlier and registers a
   "something special at subject S" hypothesis for the planner to probe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .hypothesis import Hypothesis, HypothesisRegistry, PROVISIONAL
from .stream import StreamRecorder, Tick


# ---------------------------------------------------------------- base class


class PatternMiner:
    """Base miner. Subclass and override ``observe`` for tick-by-tick logic.

    Miners hold a reference to the HypothesisRegistry they write into.
    They must be idempotent: being called twice on the same tick must
    not double-count evidence (enforced by tick-idx bookkeeping in the
    subclasses that need it).
    """

    def __init__(self, registry: HypothesisRegistry, *, name: str = ""):
        self.registry = registry
        self.name     = name or self.__class__.__name__
        self._last_seen_step: int = -1

    # stream calls us as miner(stream, tick)
    def __call__(self, stream: StreamRecorder, tick: Tick) -> None:
        # Skip if we've already processed this tick (idempotence guard).
        if tick.step_idx <= self._last_seen_step:
            return
        try:
            self.observe(stream, tick)
        finally:
            self._last_seen_step = tick.step_idx

    def observe(self, stream: StreamRecorder, tick: Tick) -> None:  # noqa: D401
        """Override in subclasses."""
        raise NotImplementedError


# -------------------------------------------------------- futile-pattern miner


class FutilePatternMiner(PatternMiner):
    """Detect repeated (state, action) -> no-op, register as a blocking rule.

    The canonical wall-banging case: the agent takes action A from
    position P and the position does not change.  After ``min_count``
    such observations with NO movement observation mixed in, register a
    hypothesis ``blocked_action_at_pos`` with subject ``(pos, action)``.

    * Domain generality:
       - ARC:      position == player_pos, action == ACTION1..4
       - robotics: position == discretised pose cell, action == skill name
      Both produce the same (pos, action) key; only the adapter differs.

    * Invalidation: when a tick contains a notable_event indicating the
      world-model ground truth changed (ARC: ``maze_rotation``; robotics:
      ``env_reconfigured``), the miner resets its internal counters so
      stale wall knowledge doesn't leak across changes.

    Tick requirements
    -----------------
    * ``observation["player_pos"]`` or ``observation["pos"]``  (pre-state)
    * ``action``
    * ``outcome["post_pos"]`` or ``outcome["pos"]``
    If any are missing, the tick is skipped.
    """

    PREDICATE = "blocked_action_at_pos"

    def __init__(
        self,
        registry: HypothesisRegistry,
        *,
        min_count: int = 2,
        invalidating_events: tuple[str, ...] = ("maze_rotation", "env_reconfigured"),
    ):
        super().__init__(registry, name="futile")
        self.min_count = min_count
        self.invalidating_events = set(invalidating_events)
        # (pos, action) -> {"noop": int, "moved": int, "first_step": int}
        self._stats: dict[tuple, dict] = {}

    def observe(self, stream: StreamRecorder, tick: Tick) -> None:
        # Invalidate on world-model-changing events.
        if any(ev in self.invalidating_events for ev in tick.notable_events):
            self._stats.clear()
            # Also drop existing hypotheses whose evidence is now stale.
            # We do this by falsifying — rather than deleting — so that
            # if evidence returns post-rotation the same registry slot
            # gets new life without reviving outdated counts.
            for h in list(self.registry.query(predicate=self.PREDICATE)):
                # Use observe_against to push the ratio into falsification
                # territory deliberately.
                h.against = max(h.against, h.support + 1)
                self.registry._maybe_transition(h)  # noqa: SLF001
            return

        pre  = _pos(tick.observation)
        post = _pos(tick.outcome) or pre
        action = tick.action
        if pre is None or action is None:
            return

        key = (pre, action)
        rec = self._stats.setdefault(key, {"noop": 0, "moved": 0, "first_step": tick.step_idx})
        if post == pre:
            rec["noop"] += 1
        else:
            rec["moved"] += 1

        # Rule: noop-only and threshold reached.
        if rec["noop"] >= self.min_count and rec["moved"] == 0:
            subject = list(key)              # (pos, action_name)
            existing = self.registry.get(self.PREDICATE, subject)
            if existing is None:
                # First threshold crossing: seed the hypothesis with the
                # evidence count we have so far.
                self.registry.add(Hypothesis(
                    predicate=self.PREDICATE,
                    subject=subject,
                    status=PROVISIONAL,
                    support=rec["noop"],
                    against=0,
                    first_seen_step=rec["first_step"],
                    last_updated_step=tick.step_idx,
                    label=f"{action}@{pre} is a no-op",
                ))
            else:
                # Later corroborations: single +1 so the count tracks
                # real observations instead of inflating via add/merge.
                self.registry.observe_support(
                    self.PREDICATE, subject,
                    step=tick.step_idx,
                    note="repeat no-op",
                )
        elif rec["moved"] > 0:
            # Mixed evidence — don't claim blocked; and if we previously did,
            # push counter-evidence into the registry.
            h = self.registry.get(self.PREDICATE, list(key))
            if h is not None:
                self.registry.observe_against(
                    self.PREDICATE, list(key), step=tick.step_idx,
                    note="movement observed after earlier no-op",
                )


# -------------------------------------------------------------- surprise miner


class SurpriseMiner(PatternMiner):
    """Detect outcomes that a baseline predictor did not foresee; register
    them as provisional hypotheses worth probing.

    Baseline predictor (kept minimal by design): for ARC-style environments
    the expected ``diff_magnitude`` for a normal walk step is small
    (usually < 80). Any outcome whose diff jumps outside the "ordinary
    movement" band AND is not a classified reset event is a surprise.

    The surprise *target* (subject) is the position where it happened,
    keyed by the domain-agnostic position. Caller may override ``classify``
    to translate raw outcome into a predicate/subject pair.

    Config
    ------
    classify
        Callable ``(tick) -> Optional[(predicate, subject, label, conditions)]``.
        Return None if the tick is unremarkable. The default classifier
        maps a ``"counter_jump"`` notable_event at a position into a
        ``refill_at_pos`` hypothesis.  Callers typically pass their own
        classifier that reflects their domain.
    """

    def __init__(
        self,
        registry: HypothesisRegistry,
        classify: Optional[Callable[[Tick], Optional[tuple]]] = None,
    ):
        super().__init__(registry, name="surprise")
        self.classify = classify or _default_classify

    def observe(self, stream: StreamRecorder, tick: Tick) -> None:
        result = self.classify(tick)
        if result is None:
            return
        predicate, subject, label, conditions = result
        # First observation: register (or merge support into) a provisional
        # hypothesis.  Further corroborations come via
        # ``registry.observe_support`` — see the adapter.
        existing = self.registry.get(predicate, subject, conditions)
        if existing is None:
            self.registry.add(Hypothesis(
                predicate=predicate,
                subject=subject,
                status=PROVISIONAL,
                support=1,
                against=0,
                conditions=conditions,
                first_seen_step=tick.step_idx,
                last_updated_step=tick.step_idx,
                label=label,
                evidence=[{"step": tick.step_idx, "kind": "support", "note": "first_seen"}],
            ))
        else:
            self.registry.observe_support(
                predicate, subject,
                conditions=conditions,
                step=tick.step_idx,
                note="repeat corroboration",
            )


def _default_classify(tick: Tick) -> Optional[tuple]:
    """Default surprise classifier.

    If the tick is labelled with ``counter_jump`` as a notable event,
    emit a ``refill_at_pos`` hypothesis whose subject is the position
    where the jump occurred. Conditions carry any level context so that
    refill claims are scoped to a level (different levels may have
    different mechanics).
    """
    if "counter_jump" not in tick.notable_events:
        return None
    post_pos = _pos(tick.outcome) or _pos(tick.observation)
    if post_pos is None:
        return None
    conditions = {}
    level = tick.observation.get("level")
    if level is not None:
        conditions["level"] = level
    return ("refill_at_pos", post_pos, f"counter jump at {post_pos}", conditions)


# -------------------------------------------------- temporal-instability miner


class TemporalInstabilityMiner(PatternMiner):
    """Detect ``(pos, action)`` pairs whose outcome varies over time.

    Core signal: the same ``(pre_pos, action)`` combination produced a
    **safe** outcome (normal transition) at one step and a **fatal** outcome
    (``life_loss_reset``) at a later step.  This is the abductive signature
    of a *moving hazard* — e.g. an animated sprite whose position coincides
    with the player's target at some times but not others.

    The miner registers ``unstable_at_pos_action`` hypotheses whose subject
    is ``(pos, action)``.  A downstream planner can consult the registry
    and (a) avoid re-issuing the same action from that position, or
    (b) insert additional route verification before committing.

    Domain generality
    -----------------
    * ARC: pos == player_pos, action == ACTION1..4 — the mechanic in
      ls20 level 2 is the "changer" sprite that tracks the player.
    * Robotics: pos == pose cell, action == skill-or-trajectory name —
      the analogue is a moving obstacle (another robot, a person in the
      workspace) that makes a previously-safe command intermittently
      lethal.

    The miner is deliberately conservative: a single death with at least
    one prior safe observation at the same key is enough to flag.  False
    positives are cheap (planner detours slightly); false negatives are
    expensive (another death in a 2-life game).

    Tick requirements
    -----------------
    * ``observation["player_pos"]`` or ``observation["pos"]``  (pre-state)
    * ``action``
    * Either ``outcome["diff_magnitude"] >= life_loss_diff`` (death) or
      any lower diff (safe).  If ``diff_magnitude`` is absent, the tick is
      skipped.
    """

    PREDICATE = "unstable_at_pos_action"

    def __init__(
        self,
        registry: HypothesisRegistry,
        *,
        life_loss_diff: int = 3000,
    ):
        super().__init__(registry, name="temporal_instability")
        self.life_loss_diff = life_loss_diff
        # key (pos, action) -> {"safe": int, "death": int,
        #                       "first_step": int, "last_step": int}
        self._stats: dict[tuple, dict] = {}

    def observe(self, stream: StreamRecorder, tick: Tick) -> None:
        pre = _pos(tick.observation)
        action = tick.action
        if pre is None or action is None:
            return
        diff_raw = tick.outcome.get("diff_magnitude")
        if diff_raw is None:
            return
        diff = int(diff_raw)
        # Classify outcome.
        if diff >= self.life_loss_diff:
            kind = "death"
        elif diff > 0:
            # Any non-zero diff counts as a "safe observation" — the
            # commanded move took effect without killing the agent.
            kind = "safe"
        else:
            # diff == 0: no-op / stuck.  Not informative for instability;
            # let FutilePatternMiner handle it.
            return

        key = (pre, action)
        rec = self._stats.setdefault(
            key,
            {"safe": 0, "death": 0,
             "first_step": tick.step_idx, "last_step": tick.step_idx},
        )
        rec[kind] += 1
        rec["last_step"] = tick.step_idx

        # Flag instability as soon as we have at least one of each.
        if rec["safe"] >= 1 and rec["death"] >= 1:
            subject = [list(pre), action]   # JSON-serialisable
            existing = self.registry.get(self.PREDICATE, subject)
            label = (
                f"{action}@{pre}: safe×{rec['safe']} death×{rec['death']} "
                f"(moving-hazard suspect)"
            )
            if existing is None:
                self.registry.add(Hypothesis(
                    predicate=self.PREDICATE,
                    subject=subject,
                    status=PROVISIONAL,
                    support=rec["safe"] + rec["death"],
                    against=0,
                    first_seen_step=rec["first_step"],
                    last_updated_step=tick.step_idx,
                    label=label,
                ))
            else:
                # Fresh observation arrived — push the count via the
                # registry's observe_support so evidence is attributed
                # to this specific step.
                self.registry.observe_support(
                    self.PREDICATE, subject,
                    step=tick.step_idx,
                    note=f"{kind}@step{tick.step_idx}",
                )


# ------------------------------------------------------------- distal-effect miner


class DistalEffectMiner(PatternMiner):
    """Detect actions at cell A that consistently change distant cell B.

    Core signal: the agent acts at cell A, and the *global observation
    diff* between pre- and post-frames shows consistent change at cell
    B (with ``|B - A| > local_radius``). After ``min_count`` such
    observations clustering on the same (A, B) pair, register a
    ``distal_effect`` hypothesis whose subject is
    ``[source_cell, affected_region]``.

    This is the signature of a **device** — button, lever, RC changer,
    NPC — whose activation has a non-local effect. A planner that can
    read ``distal_effect`` hypotheses can use them as tools: to change
    state at B, act at A.

    Domain generality
    -----------------
    * ARC: A is the RC-changer cell, B is the orientation-indicator
      sprite on the other side of the board.
    * Robotics: A is the button on cabinet #3, B is the latch on
      drawer #7 three metres away; A is the wall switch, B is the
      light in another room.

    Tick requirements
    -----------------
    * ``tick.observation["player_pos"]`` or ``["pos"]``  (= A)
    * ``tick.action``
    * ``tick.outcome["distal_changes"]`` — a list of cells/regions whose
      state changed this tick and are NOT at A itself. The domain
      adapter is responsible for extracting this (ARC adapter diffs
      colored regions; robotics adapter diffs object poses). Absence
      means the tick has nothing to mine.

    Parameters
    ----------
    min_count
        Corroborations at the same (A, action, B) key needed before
        emitting the hypothesis.
    invalidating_events
        Events that wipe the internal state — e.g. a level change or
        environment reconfiguration that invalidates the spatial
        mapping.
    """

    PREDICATE = "distal_effect"

    def __init__(
        self,
        registry: HypothesisRegistry,
        *,
        min_count: int = 2,
        invalidating_events: tuple[str, ...] = ("maze_rotation", "env_reconfigured",
                                                "level_advance"),
    ):
        super().__init__(registry, name="distal_effect")
        self.min_count = min_count
        self.invalidating_events = set(invalidating_events)
        # key (source_cell, action, affected_cell) -> {"count": int,
        #                                              "first_step": int}
        self._stats: dict[tuple, dict] = {}

    def observe(self, stream: StreamRecorder, tick: Tick) -> None:
        if any(ev in self.invalidating_events for ev in tick.notable_events):
            self._stats.clear()
            return

        source = _pos(tick.observation)
        action = tick.action
        if source is None or action is None:
            return

        distal = tick.outcome.get("distal_changes")
        if not distal:
            return

        for ch in distal:
            aff = _coerce_region(ch)
            if aff is None or aff == source:
                continue
            key = (source, action, aff)
            rec = self._stats.setdefault(key, {"count": 0,
                                               "first_step": tick.step_idx})
            rec["count"] += 1
            if rec["count"] < self.min_count:
                continue

            subject = [list(source), action, list(aff) if isinstance(aff, tuple) else aff]
            label = (f"{action}@{source} causes change at {aff} "
                     f"(device-like)")
            existing = self.registry.get(self.PREDICATE, subject)
            if existing is None:
                self.registry.add(Hypothesis(
                    predicate=self.PREDICATE,
                    subject=subject,
                    status=PROVISIONAL,
                    support=rec["count"],
                    against=0,
                    first_seen_step=rec["first_step"],
                    last_updated_step=tick.step_idx,
                    label=label,
                ))
            else:
                self.registry.observe_support(
                    self.PREDICATE, subject,
                    step=tick.step_idx,
                    note="repeat distal change",
                )


def _coerce_region(ch: Any) -> Optional[tuple]:
    """Accept a 2-tuple/list, a dict with 'cell'/'pos'/'pose' key, or a
    hashable region id, and canonicalise to a hashable value."""
    if ch is None:
        return None
    if isinstance(ch, (list, tuple)) and len(ch) >= 2 and \
            all(isinstance(x, (int, float)) for x in ch[:2]):
        return (int(ch[0]), int(ch[1]))
    if isinstance(ch, dict):
        for k in ("cell", "pos", "pose", "region", "id"):
            v = ch.get(k)
            if v is not None:
                return _coerce_region(v)
    # Fall through: treat as hashable id (e.g. "drawer_7") if possible.
    try:
        hash(ch)
        return ch  # type: ignore[return-value]
    except TypeError:
        return None


# ---------------------------------------------------- goal-precondition miner


class GoalPreconditionMiner(PatternMiner):
    """Infer preconditions for goals that succeed only after some
    prerequisite event has been observed.

    Core signal: a goal G has been attempted N times (via the
    ``goal_attempt`` notable event) and failed N-1 of those attempts.
    On the one successful attempt, some notable event E happened in
    the intervening ticks that did *not* happen in the failures.
    Register a ``precondition_of`` hypothesis: "E is a precondition
    of G".

    Tick requirements
    -----------------
    * ``tick.notable_events`` should contain one of:
        - ``"goal_attempt:<goal_id>"``  — attempt resolved
        - ``"goal_success:<goal_id>"``  — attempt resolved as success
        - ``"goal_failure:<goal_id>"``  — attempt resolved as failure
      Domain adapters emit these from their goal stack.

    Hypothesis shape
    ----------------
    * ``predicate`` = ``"precondition_of"``
    * ``subject``  = ``[goal_id, event_label]``
    * ``support``  = number of successes where the event was present
    * ``against``  = number of successes where the event was absent
      (strong signal it isn't really a precondition)

    Domain generality
    -----------------
    * ARC: goal = "reach WIN_TARGET"; prerequisite event =
      ``"orientation_matched"`` (fired by DistalEffectMiner-derived
      logic when small-L orientation equals large-L indicator).
    * Robotics: goal = "pick knife from drawer"; prerequisite event =
      ``"drawer_open"`` (fired when the drawer sensor sees open state).
    """

    PREDICATE = "precondition_of"

    def __init__(
        self,
        registry: HypothesisRegistry,
        *,
        window: int = 64,
    ):
        super().__init__(registry, name="goal_precondition")
        self.window = window
        # goal_id -> list of event labels observed since last goal_attempt
        self._events_since_attempt: dict[str, list[str]] = {}
        # goal_id -> {"success_events": set[str], "failure_events": set[str]}
        self._summary: dict[str, dict] = {}

    def observe(self, stream: StreamRecorder, tick: Tick) -> None:
        # Accumulate all non-goal events across ALL goals being tracked
        non_goal_events = [
            ev for ev in tick.notable_events
            if not ev.startswith(("goal_attempt:", "goal_success:",
                                  "goal_failure:"))
        ]
        for gid in list(self._events_since_attempt):
            self._events_since_attempt[gid].extend(non_goal_events)
            if len(self._events_since_attempt[gid]) > self.window:
                self._events_since_attempt[gid] = \
                    self._events_since_attempt[gid][-self.window:]

        for ev in tick.notable_events:
            if ev.startswith("goal_attempt:"):
                gid = ev.split(":", 1)[1]
                self._events_since_attempt.setdefault(gid, [])
            elif ev.startswith("goal_success:"):
                gid = ev.split(":", 1)[1]
                self._resolve(gid, success=True, step=tick.step_idx)
            elif ev.startswith("goal_failure:"):
                gid = ev.split(":", 1)[1]
                self._resolve(gid, success=False, step=tick.step_idx)

    def _resolve(self, gid: str, *, success: bool, step: int) -> None:
        seen = set(self._events_since_attempt.pop(gid, []))
        summary = self._summary.setdefault(gid, {"success_events": [],
                                                 "failure_events": []})
        target = "success_events" if success else "failure_events"
        summary[target].append(seen)

        # Candidate preconditions: events present in every success and in
        # no failure (or at least not in the majority).
        succ = summary["success_events"]
        fail = summary["failure_events"]
        if not succ:
            return
        common_in_succ = set.intersection(*succ) if succ else set()
        for ev in common_in_succ:
            n_fail_with = sum(1 for f in fail if ev in f)
            if n_fail_with >= len(fail) and len(fail) > 0:
                # Event is present in every failure too — useless.
                continue
            subject = [gid, ev]
            existing = self.registry.get(self.PREDICATE, subject)
            if existing is None:
                self.registry.add(Hypothesis(
                    predicate=self.PREDICATE,
                    subject=subject,
                    status=PROVISIONAL,
                    support=len(succ),
                    against=n_fail_with,
                    first_seen_step=step,
                    last_updated_step=step,
                    label=f"{ev} is precondition of goal {gid}",
                ))
            else:
                # Incremental update via observe_support / observe_against
                # so evidence is attributed to steps.
                if success:
                    self.registry.observe_support(
                        self.PREDICATE, subject, step=step,
                        note="success with event present",
                    )
                else:
                    if ev in seen:
                        self.registry.observe_against(
                            self.PREDICATE, subject, step=step,
                            note="failure with event present",
                        )


# ------------------------------------------------------------------- helpers


def _pos(d: dict) -> Optional[tuple]:
    """Extract a position tuple from a dict using either 'player_pos' or
    'pos' key. Accepts tuples, lists, or None."""
    if not isinstance(d, dict):
        return None
    v = d.get("player_pos", d.get("pos"))
    if v is None:
        return None
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return (int(v[0]), int(v[1]))
    return None
