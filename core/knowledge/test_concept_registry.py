"""Standalone tests for ConceptRegistry. Run with: python test_concept_registry.py"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.knowledge.concept_registry import Concept, ConceptRegistry


def _fresh(tmp: Path) -> ConceptRegistry:
    return ConceptRegistry(tmp / "concepts.json")


def test_record_and_get():
    with tempfile.TemporaryDirectory() as d:
        r = _fresh(Path(d))
        cid = r.record(
            name="step_counter",
            domain="domA",
            kind="behavioral_signature",
            signature={"trigger": "monotonic decrease"},
            abstraction={"summary": "counts down to zero"},
            confidence=0.6,
        )
        c = r.get(cid)
        assert c is not None
        assert c.name == "step_counter"
        assert c.domain == "domA"
        assert c.confidence == 0.6
        assert c.signature == {"trigger": "monotonic decrease"}


def test_persistence_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "concepts.json"
        r1 = ConceptRegistry(path)
        cid = r1.record(
            name="x", domain="domA", kind="k1",
            signature={"a": 1}, abstraction={"summary": "s"},
        )
        r2 = ConceptRegistry(path)
        c = r2.get(cid)
        assert c is not None
        assert c.signature == {"a": 1}


def test_recall_filters_and_ranking():
    with tempfile.TemporaryDirectory() as d:
        r = _fresh(Path(d))
        r.record(name="alpha", domain="domA", kind="k1",
                 signature={}, abstraction={"summary": "first"}, confidence=0.9)
        r.record(name="beta",  domain="domA", kind="k2",
                 signature={}, abstraction={"summary": "second"}, confidence=0.5)
        r.record(name="gamma", domain="domB", kind="k1",
                 signature={}, abstraction={"summary": "third"}, confidence=0.95)

        same = r.recall(domain="domA", limit=10, include_cross_domain=False)
        assert [c.name for c in same] == ["alpha", "beta"]

        cross = r.recall(domain="domA", cross_domain_kinds=["k1"], limit=10)
        names = [c.name for c in cross]
        assert names[:2] == ["alpha", "beta"]
        assert "gamma" in names

        kind1 = r.recall(kind="k1", limit=10)
        assert {c.name for c in kind1} == {"alpha", "gamma"}

        hi = r.recall(min_confidence=0.8, limit=10)
        assert {c.name for c in hi} == {"alpha", "gamma"}

        q = r.recall(name_query="seco", limit=10)
        assert [c.name for c in q] == ["beta"]


def test_confirm_bumps_confidence():
    with tempfile.TemporaryDirectory() as d:
        r = _fresh(Path(d))
        cid = r.record(name="x", domain="domA", kind="k",
                       signature={}, abstraction={}, confidence=0.5)
        r.confirm(cid, evidence={"obs": "matched"}, confidence_delta=0.1)
        c = r.get(cid)
        assert abs(c.confidence - 0.6) < 1e-9
        assert c.provenance["evidence_history"][0]["obs"] == "matched"
        for _ in range(20):
            r.confirm(cid, confidence_delta=0.1)
        assert r.get(cid).confidence == 1.0


def test_deprecate_hides_from_recall():
    with tempfile.TemporaryDirectory() as d:
        r = _fresh(Path(d))
        cid = r.record(name="x", domain="domA", kind="k",
                       signature={}, abstraction={})
        r.deprecate(cid, reason="bad evidence")
        assert r.recall(domain="domA") == []
        assert len(r.all(include_deprecated=True)) == 1
        assert r.get(cid).deprecated is True


def test_validation():
    with tempfile.TemporaryDirectory() as d:
        r = _fresh(Path(d))
        try:
            r.record(name="", domain="d", kind="k", signature={}, abstraction={})
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError on empty name")
        try:
            r.record(name="x", domain="d", kind="k",
                     signature="not a dict", abstraction={})  # type: ignore
        except TypeError:
            pass
        else:
            raise AssertionError("expected TypeError on bad signature")


def main():
    tests = [
        test_record_and_get,
        test_persistence_roundtrip,
        test_recall_filters_and_ranking,
        test_confirm_bumps_confidence,
        test_deprecate_hides_from_recall,
        test_validation,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
