"""
co_occurrence.py — Cross-object attribute co-occurrence registry.

Records pairs of attribute changes that consistently co-occur within the
same environment step, then promotes strong pairs to candidate KF rules.

Design principles
-----------------
Two-way
    Every observation "A changed with B" also records "B changed with A".
    Neither side is pre-judged as cause or effect; the system accumulates
    evidence in both directions and lets the rule engine decide.

Role-indexed, not color-indexed
    Changes are keyed by *role name* (e.g. "player_piece") resolved from
    concept_bindings at observe time, not by raw color value.  If no role
    is bound, falls back to "color{N}".  This means the same co-occurrence
    record accumulates evidence even when the role migrates to a different
    color in a future level or game.

Named events
    Non-attribute triggers such as "level_advanced" or "episode_won" are
    represented as ChangeEvent(subject="__event__", attr=<event_name>,
    delta=None).  Pairing them with attribute changes enables win-condition
    hypothesis generation: "level advances co-occurs with player at row X".

Persistence
    Registry state is written to co_occurrences.json alongside rules.json.
    Raw counts survive episode resets; once a pair is promoted to a
    candidate rule the key is recorded in `promoted_keys` so the same
    rule is not emitted twice.

Cross-domain
    This module has no ARC-AGI-3-specific logic.  Any ensemble that calls
    observe_step() with a list of ChangeEvent objects and periodically calls
    promote_to_rules() will benefit from it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChangeEvent:
    """A single attribute change observed within one step.

    subject
        Role name (e.g. "player_piece") if known, else "color{N}".
        Use "__event__" for named binary events (level_advanced, etc.).
    attr
        Attribute name: "size", "row", "col", "width", or for events the
        event name itself (e.g. "level_advanced").
    delta
        Numeric change (may be fractional).  None for binary events.
    """
    subject: str
    attr: str
    delta: float | None

    @property
    def delta_sign(self) -> int:
        """Summarise direction: +1 (increase), -1 (decrease), 0 (none/event)."""
        if self.delta is None or self.delta == 0:
            return 0
        return 1 if self.delta > 0 else -1

    @property
    def key(self) -> str:
        return f"{self.subject}|{self.attr}|{self.delta_sign:+d}"


@dataclass
class CoOccurrenceRecord:
    """Running statistics for one directed (A → B) co-occurrence pair."""
    subject: str        # e.g. "player_piece"
    subject_attr: str   # e.g. "row"
    subject_sign: int   # -1 / 0 / +1
    object: str         # e.g. "step_counter"
    object_attr: str    # e.g. "size"
    object_sign: int    # -1 / 0 / +1
    subject_deltas: list[float] = field(default_factory=list)
    object_deltas: list[float]  = field(default_factory=list)
    count: int = 0

    # How many steps the subject change was observed (denominator for
    # computing consistency: count / subject_fires).
    subject_fires: int = 0

    @property
    def pair_key(self) -> str:
        return (f"{self.subject}|{self.subject_attr}|{self.subject_sign:+d}|"
                f"{self.object}|{self.object_attr}|{self.object_sign:+d}")

    @property
    def consistency(self) -> float:
        """Fraction of steps where subject fired that object also fired."""
        return self.count / self.subject_fires if self.subject_fires else 0.0

    @property
    def mean_subject_delta(self) -> float:
        return sum(self.subject_deltas) / len(self.subject_deltas) if self.subject_deltas else 0.0

    @property
    def mean_object_delta(self) -> float:
        return sum(self.object_deltas) / len(self.object_deltas) if self.object_deltas else 0.0

    def to_rule_text(self) -> tuple[str, str]:
        """Return (condition, action) natural-language text for a KF rule."""
        def _delta_phrase(sign: int, mean: float, attr: str, subj: str) -> str:
            if subj == "__event__":
                return attr   # e.g. "level_advanced"
            direction = "increases" if sign > 0 else ("decreases" if sign < 0 else "changes")
            return f"{subj}.{attr} {direction} by ~{abs(mean):.1g}"

        condition = _delta_phrase(
            self.subject_sign, self.mean_subject_delta,
            self.subject_attr, self.subject
        )
        object_phrase = _delta_phrase(
            self.object_sign, self.mean_object_delta,
            self.object_attr, self.object
        )
        action = (
            f"{object_phrase} in the same step  "
            f"[co-occurrence · {self.count} observations · "
            f"consistency {self.consistency:.0%} · two-way]"
        )
        return condition, action

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "subject_attr": self.subject_attr,
            "subject_sign": self.subject_sign,
            "object": self.object,
            "object_attr": self.object_attr,
            "object_sign": self.object_sign,
            "subject_deltas": self.subject_deltas,
            "object_deltas": self.object_deltas,
            "count": self.count,
            "subject_fires": self.subject_fires,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CoOccurrenceRecord":
        r = cls(
            subject=d["subject"],
            subject_attr=d["subject_attr"],
            subject_sign=d["subject_sign"],
            object=d["object"],
            object_attr=d["object_attr"],
            object_sign=d["object_sign"],
        )
        r.subject_deltas = d.get("subject_deltas", [])
        r.object_deltas  = d.get("object_deltas", [])
        r.count          = d.get("count", 0)
        r.subject_fires  = d.get("subject_fires", 0)
        return r


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class CoOccurrenceRegistry:
    """
    Accumulates co-occurrence evidence across steps and episodes.

    Usage (per step)::

        events = [
            ChangeEvent("player_piece", "row", -5),
            ChangeEvent("step_counter", "size", -2),
            ChangeEvent("action_counter", "size", +2),
        ]
        registry.observe_step(events)

    Usage (for a named event such as level advance)::

        events = [
            ChangeEvent("__event__", "level_advanced", None),
            ChangeEvent("player_piece", "row", player_row_at_win),
            ChangeEvent("player_piece", "col", player_col_at_win),
        ]
        registry.observe_step(events)

    Promotion::

        new_rules = registry.promote_to_rules(rule_engine, min_count=5,
                                              min_consistency=0.80)
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        # pair_key → CoOccurrenceRecord
        self._records: dict[str, CoOccurrenceRecord] = {}
        # pair keys already emitted as rules (no re-promotion)
        self._promoted: set[str] = set()
        if self.path and self.path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe_step(self, events: list[ChangeEvent]) -> None:
        """Record all pairwise co-occurrences from a single step.

        Every ordered pair (A, B) where A ≠ B is recorded.  Both (A→B) and
        (B→A) are stored, making the relationship two-way by construction.
        """
        if len(events) < 2:
            return
        for i, ev_a in enumerate(events):
            # Increment subject_fires for every pair involving ev_a as subject
            for j, ev_b in enumerate(events):
                if i == j:
                    continue
                self._record_pair(ev_a, ev_b)

        if self.path:
            self._save()

    def _record_pair(self, subject: ChangeEvent, obj: ChangeEvent) -> None:
        """Record one directed observation: subject co-occurred with obj."""
        # Build or retrieve record
        key = (f"{subject.subject}|{subject.attr}|{subject.delta_sign:+d}|"
               f"{obj.subject}|{obj.attr}|{obj.delta_sign:+d}")
        if key not in self._records:
            self._records[key] = CoOccurrenceRecord(
                subject=subject.subject,
                subject_attr=subject.attr,
                subject_sign=subject.delta_sign,
                object=obj.subject,
                object_attr=obj.attr,
                object_sign=obj.delta_sign,
            )
        rec = self._records[key]
        rec.count += 1
        rec.subject_fires += 1
        if subject.delta is not None:
            rec.subject_deltas.append(subject.delta)
        if obj.delta is not None:
            rec.object_deltas.append(obj.delta)

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    def promote_to_rules(
        self,
        rule_engine: Any,
        min_count: int = 5,
        min_consistency: float = 0.80,
        ns_tag: str = "",
        source_task: str = "",
        max_rules: int = 30,
    ) -> list[dict]:
        """Emit candidate rules for all pairs that meet the thresholds.

        Returns the list of newly created rule dicts.
        Already-promoted pairs are skipped (no duplicates).

        Parameters
        ----------
        min_count
            Minimum number of co-occurrence observations required.
        min_consistency
            Minimum fraction of subject-fired steps where the object also
            changed.  Higher = stricter; 0.90 recommended for noisy games.
        max_rules
            Maximum new rules emitted per call.  Candidates are ranked by
            (consistency desc, count desc) so the strongest pairs are
            promoted first.  0 means unlimited (not recommended).
        """
        new_rules: list[dict] = []

        # Rank eligible pairs: strongest (consistency, then count) first.
        # This ensures the cap retains the most-evidenced pairs.
        eligible = [
            (key, rec) for key, rec in self._records.items()
            if key not in self._promoted
            and rec.count >= min_count
            and rec.consistency >= min_consistency
        ]
        eligible.sort(key=lambda kv: (-kv[1].consistency, -kv[1].count))

        for key, rec in eligible:
            if max_rules and len(new_rules) >= max_rules:
                break
            condition, action = rec.to_rule_text()
            rule = rule_engine.add_rule(
                condition=condition,
                action=action,
                source="co_occurrence",
                source_task=source_task,
                tags=["co-occurrence", ns_tag] if ns_tag else ["co-occurrence"],
                lineage={
                    "type": "co_occurrence",
                    "parent_ids": [],
                    "reason": (
                        f"Observed {rec.count}x with "
                        f"{rec.consistency:.0%} consistency"
                    ),
                },
                status="candidate",
            )
            if rule:
                self._promoted.add(key)
                new_rules.append(rule)

        if new_rules and self.path:
            self._save()
        return new_rules

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def summary(self) -> list[dict]:
        """Return all records sorted by count descending — useful for logging."""
        return sorted(
            [r.to_dict() | {"pair_key": k, "consistency": r.consistency}
             for k, r in self._records.items()],
            key=lambda x: -x["count"],
        )

    def top(self, n: int = 10) -> list[CoOccurrenceRecord]:
        """Return the N most-observed records."""
        return sorted(self._records.values(), key=lambda r: -r.count)[:n]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        assert self.path is not None
        data = {
            "records":       {k: v.to_dict() for k, v in self._records.items()},
            "promoted_keys": sorted(self._promoted),
        }
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        assert self.path is not None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for k, v in data.get("records", {}).items():
            self._records[k] = CoOccurrenceRecord.from_dict(v)
        self._promoted = set(data.get("promoted_keys", []))


# ---------------------------------------------------------------------------
# Helpers: build ChangeEvent lists from object-tracker output
# ---------------------------------------------------------------------------

def events_from_step(
    obj_diff: Any,
    concept_bindings: dict,
    levels_delta: int = 0,
) -> list[ChangeEvent]:
    """Convert one step's object diff into a list of ChangeEvent objects.

    Parameters
    ----------
    obj_diff:
        ObjectDiff returned by object_tracker.diff_objects().
    concept_bindings:
        {color_int: role_name | {"role":..., "confidence":...}} dict from
        the current episode's OBSERVER bindings.  Used to label changes by
        role rather than by color.
    levels_delta:
        How many levels were completed this step (usually 0 or 1).
    """
    events: list[ChangeEvent] = []

    def _role(color: int) -> str:
        raw = concept_bindings.get(color)
        if isinstance(raw, dict):
            return raw.get("role", f"color{color}")
        return raw or f"color{color}"

    # Position moves
    for mv in getattr(obj_diff, "moved", []):
        color = mv.obj.color
        role  = _role(color)
        dr = mv.delta_r
        dc = mv.delta_c
        if dr:
            events.append(ChangeEvent(role, "row", float(dr)))
        if dc:
            events.append(ChangeEvent(role, "col", float(dc)))

    # Attribute changes (size, width, height, orientation, …)
    for ac in getattr(obj_diff, "attribute_changes", []):
        role = _role(ac.color)
        for attr in ac.changed:
            before_val = getattr(ac.before, attr, None)
            after_val  = getattr(ac.after,  attr, None)
            if before_val is None or after_val is None:
                continue
            try:
                delta = float(after_val) - float(before_val)
            except (TypeError, ValueError):
                continue
            if delta != 0:
                events.append(ChangeEvent(role, attr, delta))

    # Named event for level advance
    if levels_delta > 0:
        events.append(ChangeEvent("__event__", "level_advanced", None))

    return events
