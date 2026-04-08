"""
concept_registry.py — Domain-agnostic persistent store for learned concepts.

A *concept* is a named, typed record that captures something an agent has
learned and may want to recall later — possibly in a different domain. The
registry is intentionally generic: it stores concepts, retrieves them by
filters, and tracks confidence over time. It does NOT interpret the
`signature` or `abstraction` payloads — those are opaque dicts whose shape
is defined by each consuming use case.

Design rules (deliberate, do not relax without discussion):

  - No domain nouns. The registry never mentions frames, objects, pixels,
    actions, episodes, levels, classes, labels, robots, or any other
    use-case-specific terminology. The only first-class fields are id,
    name, domain, kind, signature, abstraction, confidence, provenance,
    timestamps.

  - `domain` and `kind` are caller-supplied free-form strings. The registry
    indexes them but does not enforce a vocabulary. Each use case defines
    its own kind taxonomy in its own module.

  - `signature` and `abstraction` are opaque `dict[str, Any]`. The registry
    persists them verbatim and never reads inside them.

  - Persistence is a single JSON file with atomic rewrites. Cheap, portable,
    no DB dependency. Upgradeable to SQLite later without changing the API.

Public API:

    registry = ConceptRegistry(store_path)
    cid = registry.record(name=..., domain=..., kind=..., signature=...,
                          abstraction=..., provenance=..., confidence=0.5)
    hits = registry.recall(domain="...", kind="...", name_query="...",
                           min_confidence=0.0, limit=10)
    registry.confirm(cid, evidence={...}, confidence_delta=0.05)
    registry.deprecate(cid, reason="...")
    concept = registry.get(cid)
    all_concepts = registry.all()
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Concept:
    """A single stored concept. Payload fields are opaque to the registry."""

    id: str
    name: str
    domain: str
    kind: str
    signature: dict[str, Any]
    abstraction: dict[str, Any]
    confidence: float
    provenance: dict[str, Any]
    created_at: str
    updated_at: str
    deprecated: bool = False
    deprecation_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Concept":
        return cls(
            id=d["id"],
            name=d["name"],
            domain=d["domain"],
            kind=d["kind"],
            signature=d.get("signature", {}),
            abstraction=d.get("abstraction", {}),
            confidence=float(d.get("confidence", 0.5)),
            provenance=d.get("provenance", {}),
            created_at=d["created_at"],
            updated_at=d["updated_at"],
            deprecated=bool(d.get("deprecated", False)),
            deprecation_reason=d.get("deprecation_reason"),
        )


class ConceptRegistry:
    """Persistent store of concepts with thread-safe atomic writes."""

    def __init__(self, store_path: str | Path):
        self._path = Path(store_path)
        self._lock = threading.RLock()
        self._concepts: dict[str, Concept] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with self._path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return
        for entry in data.get("concepts", []):
            try:
                c = Concept.from_dict(entry)
                self._concepts[c.id] = c
            except (KeyError, ValueError):
                continue

    def _flush(self) -> None:
        """Atomic write: temp file in same dir, then os.replace."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "concepts": [c.to_dict() for c in self._concepts.values()],
        }
        fd, tmp_path = tempfile.mkstemp(
            prefix=".concepts_", suffix=".json", dir=str(self._path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=False)
            os.replace(tmp_path, self._path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        name: str,
        domain: str,
        kind: str,
        signature: dict[str, Any],
        abstraction: dict[str, Any],
        provenance: Optional[dict[str, Any]] = None,
        confidence: float = 0.5,
    ) -> str:
        """Insert a new concept; returns its assigned id."""
        if not name or not domain or not kind:
            raise ValueError("name, domain, and kind are required")
        if not isinstance(signature, dict) or not isinstance(abstraction, dict):
            raise TypeError("signature and abstraction must be dicts")
        confidence = max(0.0, min(1.0, float(confidence)))

        with self._lock:
            cid = f"c_{uuid.uuid4().hex[:12]}"
            now = _utcnow()
            self._concepts[cid] = Concept(
                id=cid,
                name=name,
                domain=domain,
                kind=kind,
                signature=dict(signature),
                abstraction=dict(abstraction),
                confidence=confidence,
                provenance=dict(provenance or {}),
                created_at=now,
                updated_at=now,
            )
            self._flush()
            return cid

    def confirm(
        self,
        concept_id: str,
        *,
        evidence: Optional[dict[str, Any]] = None,
        confidence_delta: float = 0.05,
    ) -> None:
        """Bump confidence and append evidence to provenance."""
        with self._lock:
            c = self._concepts.get(concept_id)
            if c is None or c.deprecated:
                return
            c.confidence = max(0.0, min(1.0, c.confidence + confidence_delta))
            c.updated_at = _utcnow()
            if evidence:
                hist = c.provenance.setdefault("evidence_history", [])
                hist.append({"at": c.updated_at, **evidence})
            self._flush()

    def deprecate(self, concept_id: str, *, reason: str) -> None:
        with self._lock:
            c = self._concepts.get(concept_id)
            if c is None:
                return
            c.deprecated = True
            c.deprecation_reason = reason
            c.updated_at = _utcnow()
            self._flush()

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get(self, concept_id: str) -> Optional[Concept]:
        with self._lock:
            return self._concepts.get(concept_id)

    def all(self, *, include_deprecated: bool = False) -> list[Concept]:
        with self._lock:
            return [
                c for c in self._concepts.values()
                if include_deprecated or not c.deprecated
            ]

    def recall(
        self,
        *,
        domain: Optional[str] = None,
        kind: Optional[str] = None,
        name_query: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 10,
        include_cross_domain: bool = True,
        cross_domain_kinds: Optional[list[str]] = None,
    ) -> list[Concept]:
        """Retrieve concepts ordered by relevance.

        Ranking:
          1. Same-domain matches first, ordered by confidence desc.
          2. If `include_cross_domain` and `cross_domain_kinds` is given,
             cross-domain concepts whose kind is in that list are appended,
             also ordered by confidence desc.

        Filters: `kind` (exact), `name_query` (case-insensitive substring on
        name and on abstraction.summary if present), `min_confidence`.
        """
        q = (name_query or "").strip().lower()

        def matches_kind(c: Concept) -> bool:
            return kind is None or c.kind == kind

        def matches_query(c: Concept) -> bool:
            if not q:
                return True
            if q in c.name.lower():
                return True
            summary = str(c.abstraction.get("summary", "")).lower()
            return q in summary

        def keep(c: Concept) -> bool:
            return (
                not c.deprecated
                and c.confidence >= min_confidence
                and matches_kind(c)
                and matches_query(c)
            )

        with self._lock:
            same_domain: list[Concept] = []
            other_domain: list[Concept] = []
            for c in self._concepts.values():
                if not keep(c):
                    continue
                if domain is None or c.domain == domain:
                    same_domain.append(c)
                elif include_cross_domain and (
                    cross_domain_kinds is None or c.kind in cross_domain_kinds
                ):
                    other_domain.append(c)

            same_domain.sort(key=lambda c: c.confidence, reverse=True)
            other_domain.sort(key=lambda c: c.confidence, reverse=True)
            return (same_domain + other_domain)[:limit]
