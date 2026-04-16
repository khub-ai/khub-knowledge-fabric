"""
similarity.py — Similarity-driven connection hypotheses.

Problem this module solves
--------------------------
Human minds do something no pixel-diff substrate does by default: if
two objects in a scene *look alike* — same shape class, same colour,
same role-coded iconography — we immediately hypothesise that they are
**connected**. They might be paired (keys and locks), they might be
instances of the same category (all refill stations), one might be a
miniature of the other (small-L = indicator of orientation-goal;
large-L = target to match).

A cognitive substrate that can't pick up on "those two things resemble
each other; maybe they matter together" will miss whole classes of
puzzles and whole categories of task structure.

The trouble: **pixel-level** similarity is too weak. Two L-shapes at
different scales, rotations, or colours *look* similar to a human but
have near-zero pixel overlap. Hand-engineered descriptors (HOG, ORB,
histogram matching) cover some cases but fail on the ones that depend
on semantic abstraction.

This module provides:

1. A narrow **``SimilarityOracle`` Protocol** — "given two regions,
   tell me how similar they are and what kind of connection that
   might imply". Implementations range from pixel-baseline to
   VLM-backed.
2. A default **``PixelSimilarityOracle``** that catches the easy
   cases (same-colour blob, same-shape blob at same scale).
3. A **``VLMSimilarityOracle``** that delegates to an adapter which
   asks the OBSERVER / MEDIATOR (VLM-based) for judgment. The oracle
   *itself* owns caching, rate-limiting, and result parsing — the
   adapter is a thin pipe to the VLM.
4. A **``SimilarityMiner``** that periodically selects salient region
   pairs, queries the oracle, and registers
   ``possible_connection_between`` hypotheses for pairs the oracle
   flags.

Two-usecase fit
---------------
* **ARC-AGI-3 (ls20 L2):** the small-L sprite near the RC changer and
  the large-L indicator on the HUD are both *L-shaped*. A VLM
  observer can see that. A ``possible_connection_between`` hypothesis
  on the pair lets the agent plan the experiment "does hitting the
  RC changer affect the shape of the small L in a way that makes it
  match the large L?" — which is the solution.
* **Robotics:** two identical pill bottles on different shelves →
  probably interchangeable inventory. A pair of arrow-shaped markers
  on the floor → probably waypoints of the same route. The operator's
  coffee mug on the desk and the one in the cupboard → probably the
  same object category. Every one of those becomes a planning
  affordance once flagged.

Design principles
-----------------
* **Oracle abstraction.** Callers never know whether similarity was
  computed via pixels, hand-features, or VLM. The oracle is a black
  box with a single ``compare(a, b) -> SimilarityResult`` method.
* **Connection types are soft labels.** The oracle reports
  ``connection_kind`` (``"identity"``, ``"instance_of_category"``,
  ``"paired"``, ``"miniature_indicator"``, ``"unknown"``). The
  planner reads them as hints, not constraints. New kinds can be
  added by new oracles without changing callers.
* **Budget-respecting.** VLM calls are expensive. The oracle
  implementations expose a ``cost_hint`` and the miner throttles by
  budget. Skipping a call is safe — the substrate just doesn't learn
  that connection this cycle.
* **Cache first.** A pair's similarity doesn't change between two
  ticks of the same frame. The oracle MUST cache by region-signature
  so the miner can query liberally.
* **Pluggable into the adapter.** The VLM oracle wraps a callable
  the adapter supplies — ``vlm_fn(prompt, crops) -> str`` — so ARC
  can route to its OBSERVER/MEDIATOR and a robot stack can route to
  its own VLM without the oracle module knowing which.

Scale-up path
-------------
* **Today.** PixelSimilarityOracle + VLMSimilarityOracle; a few
  connection kinds; the miner looks at a user-supplied list of
  salient regions.
* **Near-term.** The oracle grows a ``find_all_similar(needle,
  haystack)`` batch API so a VLM call can evaluate dozens of pairs
  in one round trip.
* **Mid-term.** Region salience is itself learned — the substrate
  picks which regions to compare based on which pairs previously
  yielded useful connections.
* **Long-term.** The oracle *is* an agent that can ask the VLM
  follow-up questions ("is this thing a miniature of that one? a
  slider indicator? an icon key?"), and its ``SimilarityResult``
  carries the dialogue trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple,
)

from .hypothesis import Hypothesis, HypothesisRegistry, PROVISIONAL
from .miners import PatternMiner
from .stream import StreamRecorder, Tick


# --------------------------------------------------------------------------- types


@dataclass(frozen=True)
class RegionDescriptor:
    """Opaque handle for a region of an observation.

    The oracle doesn't interpret these — it just uses ``signature``
    for caching and passes ``crop`` / ``metadata`` to the downstream
    comparator (pixel-feature computation or VLM).

    Attributes
    ----------
    signature
        A hashable, stable identifier for this region. Used as cache
        key — two regions with the same signature are assumed to be
        the same data. Typical: ``("frame_hash", bbox_tuple)``.
    crop
        Actual image data or structured handle — a 2-D numpy/list grid
        for ARC, a PIL.Image or camera crop for robotics. The oracle
        passes this to its comparator.
    metadata
        Freeform — colour palette, detected class, bbox, role guess.
    """

    signature: Any
    crop: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimilarityResult:
    """Oracle's answer for one region pair.

    Attributes
    ----------
    score
        0.0–1.0 — the oracle's subjective similarity belief.
    connection_kind
        Soft label — ``"identity"`` (same thing, different place),
        ``"instance_of_category"`` (same kind of thing),
        ``"paired"`` (interact / correspond), ``"miniature_indicator"``
        (one is a small reference version of the other),
        ``"unknown"``. Callers treat as a hint.
    rationale
        Human-readable explanation — from the VLM or synthesised from
        pixel features.
    oracle
        Name of the oracle that produced the result (audit).
    extra
        Freeform structured data from the oracle (VLM trace,
        feature-vector distances, etc.)
    """

    score: float
    connection_kind: str = "unknown"
    rationale: str = ""
    oracle: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


class SimilarityOracle(Protocol):
    """Compare two region descriptors and return a SimilarityResult.

    Attributes
    ----------
    name
        Short tag for audit logs.
    cost_hint
        Relative call-cost — ``"cheap"``, ``"medium"``, ``"expensive"``.
        The miner uses this to decide how aggressively to probe.
    """

    name: str
    cost_hint: str

    def compare(
        self,
        a: RegionDescriptor,
        b: RegionDescriptor,
    ) -> SimilarityResult: ...


# --------------------------------------------------------- pixel-baseline oracle


class PixelSimilarityOracle:
    """Cheap pixel-level baseline.

    Strengths:
      * Same-colour blob detection.
      * Same-shape blob at the same scale.
      * Zero-cost / no network calls.

    Weaknesses:
      * Scale-invariant, rotation-invariant, colour-invariant
        semantic match (the hard cases) — weak to zero.
      * Will return high scores for accidentally-similar pixel
        layouts.

    Use as the fall-through layer beneath a ``VLMSimilarityOracle``
    (see ``CascadingOracle``): run pixel first, run VLM only when
    pixel is ambiguous. This saves VLM budget on easy cases.
    """

    name = "pixel_baseline"
    cost_hint = "cheap"

    def __init__(self, *, same_size_bonus: float = 0.2) -> None:
        self.same_size_bonus = same_size_bonus
        self._cache: Dict[Tuple[Any, Any], SimilarityResult] = {}

    def compare(
        self,
        a: RegionDescriptor,
        b: RegionDescriptor,
    ) -> SimilarityResult:
        key = (a.signature, b.signature)
        hit = self._cache.get(key) or self._cache.get((key[1], key[0]))
        if hit is not None:
            return hit

        score = _pixel_overlap_score(a.crop, b.crop)
        if _same_shape(a.crop, b.crop):
            score = min(1.0, score + self.same_size_bonus)

        kind = "unknown"
        if score >= 0.85:
            kind = "identity"
        elif score >= 0.6:
            kind = "instance_of_category"

        res = SimilarityResult(
            score=score,
            connection_kind=kind,
            rationale=f"pixel overlap {score:.2f}",
            oracle=self.name,
        )
        self._cache[key] = res
        return res


def _pixel_overlap_score(a: Any, b: Any) -> float:
    """Element-wise equality rate for two same-shape 2-D grids; 0
    if shapes differ or crops missing."""
    if a is None or b is None:
        return 0.0
    try:
        ra = len(a); ca = len(a[0]) if ra else 0
        rb = len(b); cb = len(b[0]) if rb else 0
    except TypeError:
        return 0.0
    if ra == 0 or rb == 0 or ra != rb or ca != cb:
        return 0.0
    matches = 0
    total = 0
    for i in range(ra):
        for j in range(ca):
            total += 1
            if a[i][j] == b[i][j]:
                matches += 1
    return matches / total if total else 0.0


def _same_shape(a: Any, b: Any) -> bool:
    try:
        return len(a) == len(b) and (len(a) == 0 or len(a[0]) == len(b[0]))
    except TypeError:
        return False


# ----------------------------------------------------------- VLM oracle


VLMCallable = Callable[[str, List[Any], Dict[str, Any]], str]
"""
Adapter-supplied VLM entry point.

Signature
    ``vlm_fn(prompt: str, crops: list, opts: dict) -> str``

The adapter is responsible for:
  * Converting ``crops`` into the image format the VLM understands.
  * Dispatching to the underlying VLM (OBSERVER, MEDIATOR, whatever).
  * Returning the raw text reply.

The oracle handles prompt construction and output parsing.
"""


class VLMSimilarityOracle:
    """Delegate similarity judgment to a VLM via an adapter-supplied
    callable.

    Why this shape
    --------------
    * The oracle stays in ``core/cognitive_os`` (domain-agnostic).
    * The adapter layer (``usecases/arc-agi-3``,
      ``usecases/robotics/rai``) owns *which* VLM is called and
      *how*. Adapters pass a ``vlm_fn`` that routes into their
      OBSERVER/MEDIATOR stack.
    * Oracle can therefore be unit-tested with a stub vlm_fn.

    Parameters
    ----------
    vlm_fn
        Adapter callable — see ``VLMCallable`` for the contract.
    prompt_template
        Format string with ``{a}`` / ``{b}`` placeholders for region
        metadata summaries. Override when a use case has richer
        vocabulary.
    parse_result
        Callable ``(raw_text) -> SimilarityResult``. Override to
        customise parsing logic for a specific VLM's output format.
    budget_per_cycle
        Soft cap on calls per planning cycle. The oracle enforces
        it by returning a cached "budget_exhausted" result when
        exceeded; the miner should see ``cost_hint="expensive"``
        and throttle separately on top.
    """

    name = "vlm_oracle"
    cost_hint = "expensive"

    DEFAULT_PROMPT = (
        "Compare these two regions and report their similarity.\n"
        "Region A: {a}\nRegion B: {b}\n"
        "Reply with one line in the format:\n"
        "  score=<0..1> kind=<identity|instance_of_category|"
        "paired|miniature_indicator|unknown> "
        "rationale=<short explanation>\n"
    )

    def __init__(
        self,
        vlm_fn: VLMCallable,
        *,
        prompt_template: str = DEFAULT_PROMPT,
        parse_result: Optional[Callable[[str], SimilarityResult]] = None,
        budget_per_cycle: int = 3,
    ) -> None:
        self._vlm_fn = vlm_fn
        self._prompt = prompt_template
        self._parse = parse_result or _parse_default_vlm_reply
        self._cache: Dict[Tuple[Any, Any], SimilarityResult] = {}
        self._calls_this_cycle = 0
        self.budget_per_cycle = budget_per_cycle

    def reset_cycle_budget(self) -> None:
        """Reset the budget counter at the start of each planning cycle."""
        self._calls_this_cycle = 0

    def compare(
        self,
        a: RegionDescriptor,
        b: RegionDescriptor,
    ) -> SimilarityResult:
        key = (a.signature, b.signature)
        hit = self._cache.get(key) or self._cache.get((key[1], key[0]))
        if hit is not None:
            return hit

        if self._calls_this_cycle >= self.budget_per_cycle:
            return SimilarityResult(
                score=0.0, connection_kind="unknown",
                rationale="vlm budget exhausted this cycle",
                oracle=self.name,
            )

        prompt = self._prompt.format(a=_describe(a), b=_describe(b))
        try:
            reply = self._vlm_fn(prompt, [a.crop, b.crop], {})
        except Exception as e:  # pragma: no cover — defensive
            return SimilarityResult(
                score=0.0, connection_kind="unknown",
                rationale=f"vlm_fn raised {type(e).__name__}: {e}",
                oracle=self.name,
            )
        self._calls_this_cycle += 1
        res = self._parse(reply)
        res = SimilarityResult(
            score=res.score, connection_kind=res.connection_kind,
            rationale=res.rationale, oracle=self.name,
            extra={"raw_reply": reply, **res.extra},
        )
        self._cache[key] = res
        return res


def _describe(d: RegionDescriptor) -> str:
    md = d.metadata or {}
    if "label" in md:
        return str(md["label"])
    if "color" in md and "bbox" in md:
        return f"color={md['color']} bbox={md['bbox']}"
    return str(md)


def _parse_default_vlm_reply(raw: str) -> SimilarityResult:
    """Parse a ``score=... kind=... rationale=...`` line.

    Very permissive — accepts extra whitespace, missing fields, and
    degrades gracefully. Callers who need a stricter parse should
    supply their own.
    """
    score = 0.0
    kind = "unknown"
    rationale = raw.strip()
    for tok in raw.replace("\n", " ").split():
        if tok.startswith("score="):
            try:
                score = float(tok.split("=", 1)[1])
            except ValueError:
                pass
        elif tok.startswith("kind="):
            kind = tok.split("=", 1)[1].strip() or "unknown"
    return SimilarityResult(score=score, connection_kind=kind,
                            rationale=rationale)


# ------------------------------------------------------------- cascading oracle


class CascadingOracle:
    """Pixel-first, VLM-fallback chain.

    Runs the cheap oracle; if score is confidently high or confidently
    low, returns it. Otherwise escalates to the expensive oracle.
    Lets the miner probe liberally without burning VLM budget on
    obvious cases.
    """

    name = "cascading"
    cost_hint = "mixed"

    def __init__(
        self,
        cheap: SimilarityOracle,
        expensive: SimilarityOracle,
        *,
        ambiguous_low: float = 0.35,
        ambiguous_high: float = 0.75,
    ) -> None:
        self._cheap = cheap
        self._expensive = expensive
        self._lo = ambiguous_low
        self._hi = ambiguous_high

    def compare(
        self,
        a: RegionDescriptor,
        b: RegionDescriptor,
    ) -> SimilarityResult:
        res = self._cheap.compare(a, b)
        if res.score <= self._lo or res.score >= self._hi:
            return res
        # Ambiguous — ask the expensive oracle.
        return self._expensive.compare(a, b)


# --------------------------------------------------------- similarity miner


class SimilarityMiner(PatternMiner):
    """Probe region pairs each cycle; register ``possible_connection_between``
    hypotheses for pairs the oracle flags as similar.

    Because computing salient regions is domain-specific, the miner is
    fed pairs via the ``pairs`` attribute on ticks or via an explicit
    ``pair_supplier`` callback.

    Tick-driven mode
    ----------------
    If ``tick.outcome["similarity_pairs"]`` is present (a list of
    ``(a, b)`` RegionDescriptor tuples), the miner queries the oracle
    on each and registers hypotheses.

    Supplier mode
    -------------
    If a ``pair_supplier`` callable is provided
    (``fn(stream, tick) -> Iterable[(RegionDescriptor, RegionDescriptor)]``),
    the miner calls it each tick. Useful for adapters that prefer to
    select pairs from StateStore rather than piggy-back on the tick.

    Hypothesis shape
    ----------------
    * ``predicate`` = ``"possible_connection_between"``
    * ``subject``  = ``[a.signature, b.signature]``
    * ``conditions`` includes ``connection_kind``, ``oracle``,
      ``score``
    * ``support`` increments every time the oracle re-confirms above
      ``min_score``. ``against`` increments on contradicting low
      scores from a later oracle pass.

    Parameters
    ----------
    min_score
        Below this, the oracle's opinion is not evidence.
    pair_supplier
        Optional callable; see above.
    """

    PREDICATE = "possible_connection_between"

    def __init__(
        self,
        registry: HypothesisRegistry,
        oracle: SimilarityOracle,
        *,
        min_score: float = 0.6,
        pair_supplier: Optional[
            Callable[[StreamRecorder, Tick],
                     Sequence[Tuple[RegionDescriptor, RegionDescriptor]]]
        ] = None,
    ):
        super().__init__(registry, name="similarity")
        self.oracle = oracle
        self.min_score = min_score
        self.pair_supplier = pair_supplier

    def observe(self, stream: StreamRecorder, tick: Tick) -> None:
        pairs: List[Tuple[RegionDescriptor, RegionDescriptor]] = []
        tick_pairs = tick.outcome.get("similarity_pairs")
        if tick_pairs:
            for p in tick_pairs:
                if isinstance(p, (list, tuple)) and len(p) == 2:
                    pairs.append((p[0], p[1]))
        if self.pair_supplier is not None:
            try:
                pairs.extend(list(self.pair_supplier(stream, tick)))
            except Exception:
                pass  # supplier errors never break the miner

        for a, b in pairs:
            try:
                res = self.oracle.compare(a, b)
            except Exception:
                continue
            subject = [_tojson(a.signature), _tojson(b.signature)]
            conditions = {
                "connection_kind": res.connection_kind,
                "oracle": res.oracle,
            }
            existing = self.registry.get(self.PREDICATE, subject, conditions)
            if res.score >= self.min_score:
                if existing is None:
                    self.registry.add(Hypothesis(
                        predicate=self.PREDICATE,
                        subject=subject,
                        status=PROVISIONAL,
                        support=1,
                        against=0,
                        conditions=conditions,
                        first_seen_step=tick.step_idx,
                        last_updated_step=tick.step_idx,
                        label=(f"{res.connection_kind}: "
                               f"{a.signature} ~ {b.signature} "
                               f"(score={res.score:.2f}, {res.oracle})"),
                    ))
                else:
                    self.registry.observe_support(
                        self.PREDICATE, subject,
                        conditions=conditions,
                        step=tick.step_idx,
                        note=f"score={res.score:.2f}",
                    )
            else:
                if existing is not None:
                    self.registry.observe_against(
                        self.PREDICATE, subject,
                        conditions=conditions,
                        step=tick.step_idx,
                        note=f"low score={res.score:.2f}",
                    )


def _tojson(x: Any) -> Any:
    """Coerce a region signature to JSON-safe form for hypothesis subjects."""
    if isinstance(x, tuple):
        return [_tojson(e) for e in x]
    if isinstance(x, list):
        return [_tojson(e) for e in x]
    if isinstance(x, dict):
        return {str(k): _tojson(v) for k, v in x.items()}
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    return str(x)


__all__ = [
    "RegionDescriptor",
    "SimilarityResult",
    "SimilarityOracle",
    "PixelSimilarityOracle",
    "VLMSimilarityOracle",
    "VLMCallable",
    "CascadingOracle",
    "SimilarityMiner",
]
