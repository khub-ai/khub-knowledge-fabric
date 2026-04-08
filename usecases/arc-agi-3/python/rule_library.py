"""
rule_library.py — Persistent library of learned action models, keyed by
structural signature.

Motivation (DSL doc Layer 7): once the tracker has converged on action
models for a game, that knowledge should not be thrown away. Future games
with a similar structural signature (grid shape, palette, object
cardinalities, gestalt hints) can bootstrap from the library instead of
re-learning from scratch.

The library is deliberately small and JSON-serializable: signature → list
of (action_id → effect_type, params, precondition) entries, plus a
confidence score drawn from the originating tracker's posterior. Matching
is exact on signature for v1; fuzzy nearest-neighbor lookup is future
work.

The library is *advisory*: loaded rules seed the tracker as priors
(prior_weight=0.3, same as a Constrained proposer), and a single frame of
contradicting evidence will demote them. This keeps the system honest
when the "same" signature masks different mechanics.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from dsl import ActionModelHypothesis, HypothesisTracker
from proposer_schema import EffectType, PreconditionType


# =============================================================================
# Signature
# =============================================================================

@dataclass(frozen=True)
class StructuralSignature:
    """A compact, hashable fingerprint of a game's perceptual surface.

    Fields are chosen to be cheap to compute from a single frame and
    discriminative enough to distinguish game families without being so
    specific that every seed of the same family gets its own entry.
    """
    grid_shape:     tuple[int, int]
    palette:        tuple[int, ...]       # sorted unique colors
    n_actions:      int
    dominant_color: int                   # background proxy

    def to_key(self) -> str:
        return (f"{self.grid_shape[0]}x{self.grid_shape[1]}"
                f"|p={','.join(str(c) for c in self.palette)}"
                f"|a={self.n_actions}"
                f"|bg={self.dominant_color}")


def compute_signature(
    frame: list[list[int]], n_actions: int
) -> StructuralSignature:
    counts: dict[int, int] = {}
    for row in frame:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    dom = max(counts.items(), key=lambda kv: kv[1])[0]
    return StructuralSignature(
        grid_shape=(len(frame), len(frame[0]) if frame else 0),
        palette=tuple(sorted(counts.keys())),
        n_actions=n_actions,
        dominant_color=dom,
    )


# =============================================================================
# Rule entries
# =============================================================================

@dataclass
class ActionRule:
    action_id:   str
    effect_type: str              # EffectType.value
    precondition: str             # PreconditionType.value
    params:      dict
    posterior:   float


@dataclass
class LibraryEntry:
    signature_key: str
    rules:         list[ActionRule] = field(default_factory=list)
    source_game:   str = ""       # free-form label for provenance


# =============================================================================
# Library
# =============================================================================

class RuleLibrary:
    """In-memory + on-disk rule library.

    Usage:
        lib = RuleLibrary.load("rules.json")
        entry = lib.lookup(signature)
        if entry:
            lib.seed_tracker(tracker, entry, prior_weight=0.3)
        ...
        lib.record(signature, tracker, source_game="navigation-seed-12")
        lib.save("rules.json")
    """

    def __init__(self, entries: dict[str, LibraryEntry] | None = None) -> None:
        self._entries: dict[str, LibraryEntry] = entries or {}

    # ---- lookup & write ---------------------------------------------------

    def lookup(self, sig: StructuralSignature) -> LibraryEntry | None:
        return self._entries.get(sig.to_key())

    def record(
        self,
        sig:           StructuralSignature,
        tracker:       HypothesisTracker,
        source_game:   str = "",
        min_posterior: float = 0.5,
    ) -> LibraryEntry:
        """Snapshot the tracker's top-1 action model per action into the
        library under `sig`'s key. Only models above `min_posterior` are
        kept — we refuse to persist hypotheses we're not confident in.
        """
        rules: list[ActionRule] = []
        for aid in tracker.available_actions:
            top = tracker.top_action_models(aid, k=1)
            if not top:
                continue
            m = top[0]
            if m.posterior < min_posterior:
                continue
            rules.append(ActionRule(
                action_id=aid,
                effect_type=m.effect_type.value,
                precondition=m.precondition.value,
                params=_jsonable_params(m.params),
                posterior=m.posterior,
            ))
        entry = LibraryEntry(
            signature_key=sig.to_key(),
            rules=rules,
            source_game=source_game,
        )
        self._entries[sig.to_key()] = entry
        return entry

    def seed_tracker(
        self,
        tracker:      HypothesisTracker,
        entry:        LibraryEntry,
        prior_weight: float = 0.3,
    ) -> int:
        """Seed the tracker with hypotheses from a library entry.

        Returns the number of rules seeded. Each seeded hypothesis enters
        the tracker with `support=0, contradictions=0` and the given
        `prior_weight`, so a single contradicting observation already
        drops its posterior below a cleanly-observed alternative.
        """
        seeded = 0
        for r in entry.rules:
            try:
                eff = EffectType(r.effect_type)
                pre = PreconditionType(r.precondition)
            except ValueError:
                continue
            params = _revive_params(r.params, eff)
            hyp = ActionModelHypothesis(
                action_id=r.action_id,
                effect_type=eff,
                precondition=pre,
                params=params,
                prior_weight=prior_weight,
                description=f"seeded from rule library ({entry.source_game})",
            )
            tracker.seed_action_model(hyp)
            seeded += 1
        return seeded

    # ---- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        payload = {
            k: {
                "signature_key": e.signature_key,
                "source_game":   e.source_game,
                "rules":         [asdict(r) for r in e.rules],
            }
            for k, e in self._entries.items()
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "RuleLibrary":
        p = Path(path)
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text())
        entries: dict[str, LibraryEntry] = {}
        for k, v in raw.items():
            entries[k] = LibraryEntry(
                signature_key=v["signature_key"],
                source_game=v.get("source_game", ""),
                rules=[ActionRule(**r) for r in v.get("rules", [])],
            )
        return cls(entries)

    def __len__(self) -> int:
        return len(self._entries)


# =============================================================================
# Param (de)serialization — frozensets aren't JSON-native
# =============================================================================

def _jsonable_params(params: dict) -> dict:
    out: dict = {}
    for k, v in params.items():
        if isinstance(v, (frozenset, set)):
            out[k] = sorted(list(v))
        else:
            out[k] = v
    return out


def _revive_params(params: dict, effect: EffectType) -> dict:
    out: dict = {}
    for k, v in params.items():
        if k == "cells" and isinstance(v, list):
            out[k] = frozenset(tuple(x) for x in v)
        else:
            out[k] = v
    return out
