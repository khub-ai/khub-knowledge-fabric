"""
hypothesis.py — Cross-domain hypothesis registry for the Cognitive OS.

Stores and manages *predictive claims about the world* that the agent is still
collecting evidence for. A hypothesis is any statement of the form
"under conditions C, action A applied to subject S produces outcome O" whose
truth value is not yet proven and needs to be validated by further observation.

The registry is the single place where miners deposit new hypotheses, the
planner queries them to make routing decisions, and the evidence loop
promotes or falsifies them based on subsequent observations.

Design principles
-----------------
* Domain-agnostic. No ARC vocabulary. No robotics vocabulary.
  ARC calls it "refill tile"; a robotics stack calls it "recharge station";
  the registry stores a predicate string and lets domains agree on names.
* Evidence-tracked. Every hypothesis carries `support` and `against` counts
  plus a list of tick references so we can audit *why* it is what it is.
* Lifecycle: provisional -> confirmed | falsified. No silent mutations —
  transitions go through promote/falsify so they can be logged and replayed.
* Persistence is optional. `save` / `load` are JSON; callers decide when.

Typical flow
------------
    reg = HypothesisRegistry()
    reg.add(Hypothesis(
        predicate="refill_at_pos", subject=(39, 50),
        conditions={"level": 2}, first_seen_step=42,
    ))
    # later, on a second corroborating observation:
    reg.observe_support("refill_at_pos", (39, 50), step=57)
    # or when the predicted outcome fails to materialise:
    reg.observe_against("refill_at_pos", (39, 50), step=70)
    # planner side:
    hs = reg.query(predicate="refill_at_pos", status_in=("provisional", "confirmed"))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Optional


# --------------------------------------------------------------------- status


PROVISIONAL = "provisional"
CONFIRMED   = "confirmed"
FALSIFIED   = "falsified"

_VALID_STATUS = {PROVISIONAL, CONFIRMED, FALSIFIED}


# ---------------------------------------------------------------- data class


@dataclass
class Hypothesis:
    """A testable claim about the world.

    Fields
    ------
    predicate
        Short kebab/snake identifier naming *what* is claimed — e.g.
        ``"refill_at_pos"``, ``"blocked_action_at_pos"``,
        ``"graspable_object"``. Domains agree on predicate names.
    subject
        The entity the claim is about. For positions we use a tuple
        ``(col, row)``; for objects an id; for compound subjects a tuple.
        Must be JSON-serialisable (tuple, str, int, list of these).
    status
        One of PROVISIONAL, CONFIRMED, FALSIFIED.
    support
        Count of observations consistent with the claim.
    against
        Count of observations inconsistent with the claim.
    conditions
        Contextual keys that scope the claim — e.g. ``{"level": 2,
        "rotation": 0}``. Two hypotheses with the same predicate and
        subject but different conditions are different hypotheses.
    evidence
        References to supporting observations — usually a list of tick
        indices or dicts ``{"step": N, "kind": "support"|"against"}``.
    label
        Optional human-readable short description for logs.
    first_seen_step / last_updated_step
        Bookkeeping for decay / debugging.
    """

    predicate:          str
    subject:            Any
    status:             str  = PROVISIONAL
    support:            int  = 1
    against:            int  = 0
    conditions:         dict = field(default_factory=dict)
    evidence:           list = field(default_factory=list)
    label:              str  = ""
    first_seen_step:    int  = -1
    last_updated_step:  int  = -1

    # ---- identity ----------------------------------------------------------

    def key(self) -> tuple:
        """Identity key: (predicate, subject, frozen_conditions).

        Used by the registry to deduplicate.  We freeze `subject` to a
        hashable form (tuple if it was a list) and `conditions` to a
        sorted tuple of items.
        """
        return (self.predicate, _freeze(self.subject), _freeze_conditions(self.conditions))

    # ---- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        # Make subject JSON-safe (tuples -> lists).
        d["subject"] = _to_jsonable(self.subject)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Hypothesis":
        # Restore subject tuples where it makes sense (e.g. positions).
        subj = d.get("subject")
        if isinstance(subj, list) and all(isinstance(x, (int, float)) for x in subj):
            subj = tuple(subj)
        return cls(
            predicate         = d["predicate"],
            subject           = subj,
            status            = d.get("status", PROVISIONAL),
            support           = int(d.get("support", 1)),
            against           = int(d.get("against", 0)),
            conditions        = dict(d.get("conditions", {})),
            evidence          = list(d.get("evidence", [])),
            label             = d.get("label", ""),
            first_seen_step   = int(d.get("first_seen_step", -1)),
            last_updated_step = int(d.get("last_updated_step", -1)),
        )


# -------------------------------------------------------------- helpers ---


def _freeze(x: Any) -> Any:
    if isinstance(x, list):
        return tuple(_freeze(e) for e in x)
    if isinstance(x, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in x.items()))
    return x


def _freeze_conditions(c: dict) -> tuple:
    return tuple(sorted((k, _freeze(v)) for k, v in c.items()))


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, tuple):
        return [_to_jsonable(e) for e in x]
    if isinstance(x, list):
        return [_to_jsonable(e) for e in x]
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    return x


# --------------------------------------------------------------- registry ---


class HypothesisRegistry:
    """In-memory hypothesis store with optional JSON persistence.

    Thresholds
    ----------
    promote_at
        `support >= promote_at and against == 0` -> PROVISIONAL to CONFIRMED.
    falsify_ratio
        If `against >= support * falsify_ratio and against >= 1`,
        demote to FALSIFIED.  Defaults mean a single counter-example after
        one supporting observation does NOT falsify (gives flaky effects a
        chance), but once there is clear majority-against evidence, the
        hypothesis is falsified.
    """

    def __init__(self, promote_at: int = 2, falsify_ratio: float = 1.5):
        self._by_key: dict[tuple, Hypothesis] = {}
        self.promote_at    = promote_at
        self.falsify_ratio = falsify_ratio

    # ---- mutation ----------------------------------------------------------

    def add(self, h: Hypothesis) -> Hypothesis:
        """Insert a new hypothesis, or fold evidence into an existing one.

        If a hypothesis with the same (predicate, subject, conditions) key
        already exists, we merge evidence instead of replacing it.  This
        lets miners emit the same hypothesis repeatedly without inflating
        the registry.
        """
        k = h.key()
        if k in self._by_key:
            return self._merge(self._by_key[k], h)
        self._by_key[k] = h
        return h

    def observe_support(
        self,
        predicate: str,
        subject: Any,
        *,
        conditions: Optional[dict] = None,
        step: int = -1,
        note: str = "",
    ) -> Optional[Hypothesis]:
        """Record one supporting observation.  Promotes if threshold met."""
        h = self._get_or_none(predicate, subject, conditions or {})
        if h is None:
            return None
        h.support += 1
        h.last_updated_step = step
        h.evidence.append({"step": step, "kind": "support", "note": note})
        self._maybe_transition(h)
        return h

    def observe_against(
        self,
        predicate: str,
        subject: Any,
        *,
        conditions: Optional[dict] = None,
        step: int = -1,
        note: str = "",
    ) -> Optional[Hypothesis]:
        """Record one contradicting observation.  Falsifies if majority."""
        h = self._get_or_none(predicate, subject, conditions or {})
        if h is None:
            return None
        h.against += 1
        h.last_updated_step = step
        h.evidence.append({"step": step, "kind": "against", "note": note})
        self._maybe_transition(h)
        return h

    def promote(self, h: Hypothesis) -> None:
        h.status = CONFIRMED

    def falsify(self, h: Hypothesis) -> None:
        h.status = FALSIFIED

    # ---- query -------------------------------------------------------------

    def get(
        self,
        predicate: str,
        subject: Any,
        conditions: Optional[dict] = None,
    ) -> Optional[Hypothesis]:
        return self._get_or_none(predicate, subject, conditions or {})

    def query(
        self,
        *,
        predicate: Optional[str]          = None,
        subject:   Any                    = None,
        status_in: Iterable[str]          = (PROVISIONAL, CONFIRMED),
        conditions_match: Optional[dict]  = None,
    ) -> list[Hypothesis]:
        """Return hypotheses matching the filter; status_in defaults to
        non-falsified (both provisional and confirmed)."""
        status_set = set(status_in)
        out: list[Hypothesis] = []
        for h in self._by_key.values():
            if predicate is not None and h.predicate != predicate:
                continue
            if subject is not None and _freeze(h.subject) != _freeze(subject):
                continue
            if h.status not in status_set:
                continue
            if conditions_match and not _conditions_compatible(h.conditions, conditions_match):
                continue
            out.append(h)
        return out

    def subjects(
        self,
        predicate: str,
        *,
        status_in: Iterable[str] = (PROVISIONAL, CONFIRMED),
    ) -> list[Any]:
        """Convenience: return subjects of all matching hypotheses."""
        return [h.subject for h in self.query(predicate=predicate, status_in=status_in)]

    def __len__(self) -> int:
        return len(self._by_key)

    def __iter__(self):
        return iter(self._by_key.values())

    def summary(self, top_n: int = 8) -> str:
        if not self._by_key:
            return "HypothesisRegistry(empty)"
        ranked = sorted(
            self._by_key.values(),
            key=lambda h: (h.status != CONFIRMED, -(h.support - h.against)),
        )[:top_n]
        parts = [
            f"{h.predicate}/{_to_jsonable(h.subject)}:{h.status}(+{h.support}/-{h.against})"
            for h in ranked
        ]
        return "HypothesisRegistry(" + ", ".join(parts) + ")"

    # ---- persistence -------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [h.to_dict() for h in self._by_key.values()]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: Path) -> int:
        path = Path(path)
        if not path.exists():
            return 0
        data = json.loads(path.read_text(encoding="utf-8"))
        n = 0
        for d in data:
            h = Hypothesis.from_dict(d)
            self._by_key[h.key()] = h
            n += 1
        return n

    # ---- private -----------------------------------------------------------

    def _get_or_none(
        self,
        predicate: str,
        subject: Any,
        conditions: dict,
    ) -> Optional[Hypothesis]:
        k = (predicate, _freeze(subject), _freeze_conditions(conditions))
        return self._by_key.get(k)

    def _merge(self, existing: Hypothesis, incoming: Hypothesis) -> Hypothesis:
        existing.support           += incoming.support
        existing.against           += incoming.against
        existing.last_updated_step  = max(
            existing.last_updated_step, incoming.last_updated_step
        )
        # Extend evidence list but cap to avoid unbounded growth.
        existing.evidence.extend(incoming.evidence)
        if len(existing.evidence) > 64:
            existing.evidence = existing.evidence[-64:]
        # Preserve earliest first_seen.
        if incoming.first_seen_step >= 0 and (
            existing.first_seen_step < 0
            or incoming.first_seen_step < existing.first_seen_step
        ):
            existing.first_seen_step = incoming.first_seen_step
        if not existing.label and incoming.label:
            existing.label = incoming.label
        self._maybe_transition(existing)
        return existing

    def _maybe_transition(self, h: Hypothesis) -> None:
        if h.status == FALSIFIED:
            return
        # Falsification: against is dominant.
        if h.against >= 1 and h.against >= h.support * self.falsify_ratio:
            h.status = FALSIFIED
            return
        # Promotion: enough support with no counter-evidence yet.
        if h.status == PROVISIONAL and h.support >= self.promote_at and h.against == 0:
            h.status = CONFIRMED


def _conditions_compatible(hyp_conds: dict, match: dict) -> bool:
    """A hypothesis matches a condition filter when every (k,v) in `match`
    is present in the hypothesis conditions with the same value. Missing
    keys in the hypothesis are treated as 'unspecified' and do NOT match.
    This is strict-intersection semantics — callers who want loose matching
    can query with a smaller `match` dict."""
    for k, v in match.items():
        if k not in hyp_conds or hyp_conds[k] != v:
            return False
    return True
