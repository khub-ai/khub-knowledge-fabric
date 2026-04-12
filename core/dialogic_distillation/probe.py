"""
probe.py — PUPIL Domain Readiness Probe for Dialogic Distillation.

Determines whether a PUPIL model has sufficient visual and verbal capability
to benefit from the DD process in a given domain — before any DD sessions
are run. Produces a structured capability profile with a go/partial/no-go
readiness verdict.

This is a pre-flight check, not DD itself. It answers:
  "Is the PUPIL in the right capability regime for DD to be applicable here?"

Four capability dimensions are assessed:
  1. Perception vocabulary   — does PUPIL spontaneously describe domain features?
  2. Feature detection       — can PUPIL find specific features when queried?
  3. Rule comprehension      — does rule injection improve PUPIL's accuracy?
  4. Consistency             — does PUPIL give stable answers on the same image?

Caching:
  TUTOR and VALIDATOR outputs are cached (they are model+prompt deterministic)
  so the same probe images can be reused across many PUPIL models at low cost.
  Cache is keyed on (model, prompt, image_hash). Enable disk cache by setting
  the LLM_CACHE_DIR environment variable before calling probe().

Cost tracking:
  Per-model token usage and USD cost are recorded separately for TUTOR,
  VALIDATOR, and PUPIL. Call get_probe_costs() after probe() returns.

Usage:
    from core.dialogic_distillation.probe import probe, get_probe_costs

    report = await probe(
        pupil_model     = "qwen/qwen3-vl-8b-instruct",
        tutor_model     = "claude-opus-4-6",
        validator_model = "claude-sonnet-4-6",
        domain_config   = ROAD_SURFACE_CONFIG,
        probe_images    = [
            ProbeImage(path="...", true_class="Dry Road",   difficulty="easy"),
            ProbeImage(path="...", true_class="Wet Road",   difficulty="hard"),
            ...
        ],
        pair_info       = {"class_a": "Dry Road", "class_b": "Wet Road",
                           "pair_id": "dry_vs_wet"},
        seed_rule       = {...},   # optional: known-good rule from prior domain
    )
    print(report["verdict"])          # "go" | "partial" | "no-go"
    print(report["feature_profile"])  # per-feature detection rates
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .protocols import DomainConfig
from .agents import (
    image_block,
    parse_json_block,
    parse_json_array,
    encode_image_b64,
    _get_default_call_agent,
)


# ---------------------------------------------------------------------------
# Readiness thresholds
# ---------------------------------------------------------------------------

VERDICT_GO      = "go"
VERDICT_PARTIAL = "partial"
VERDICT_NO_GO   = "no-go"

# Minimum scores for a "go" verdict
_GO_PERCEPTION_MIN         = 0.60
_GO_RULE_COMPREHENSION_MIN = 0.15   # absolute accuracy gain from rule injection
_GO_CONSISTENCY_MIN        = 0.75

# "no-go" hard floors — failing either of these overrides everything
_NOGO_PERCEPTION_MAX       = 0.30
_NOGO_CONSISTENCY_MAX      = 0.50

# Number of repeat runs for consistency check
_CONSISTENCY_REPEATS = 3
_CONSISTENCY_N_IMAGES = 5

# Feature query difficulty labels
DIFFICULTY_EASY   = "easy"
DIFFICULTY_MEDIUM = "medium"
DIFFICULTY_HARD   = "hard"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProbeImage:
    """A single labeled image for use in the readiness probe."""
    path: str                          # absolute path to JPEG
    true_class: str                    # ground-truth class label
    difficulty: str = DIFFICULTY_MEDIUM  # "easy" | "medium" | "hard"
    notes: str = ""                    # optional human annotation

    @property
    def image_hash(self) -> str:
        """MD5 hash of image bytes — used for cache keying."""
        return hashlib.md5(Path(self.path).read_bytes()).hexdigest()


@dataclass
class FeatureQuery:
    """A specific feature detection query authored by the TUTOR."""
    feature_id: str          # e.g. "specular_sheen"
    description: str         # e.g. "Is there a uniform specular sheen visible?"
    expected_class: str      # which class this feature is diagnostic for
    difficulty: str = DIFFICULTY_MEDIUM


@dataclass
class ProbeRoleCosts:
    """Token usage and cost for one role (TUTOR / VALIDATOR / PUPIL)."""
    role: str
    input_tokens:  int   = 0
    output_tokens: int   = 0
    api_calls:     int   = 0
    cost_usd:      float = 0.0

    def add(self, input_tok: int, output_tok: int, cost: float) -> None:
        self.input_tokens  += input_tok
        self.output_tokens += output_tok
        self.api_calls     += 1
        self.cost_usd      += cost

    def to_dict(self) -> dict:
        return asdict(self)


# Module-level cost accumulators — reset by reset_probe_costs()
_role_costs: Dict[str, ProbeRoleCosts] = {}


def reset_probe_costs() -> None:
    """Reset all per-role cost accumulators."""
    _role_costs.clear()


def get_probe_costs() -> Dict[str, dict]:
    """Return per-role cost summaries as plain dicts."""
    return {role: rc.to_dict() for role, rc in _role_costs.items()}


def _record_cost(role: str, model: str,
                 input_tok: int, output_tok: int) -> None:
    """Accumulate token usage for a role, estimating cost from model string."""
    cost = _estimate_cost(model, input_tok, output_tok)
    if role not in _role_costs:
        _role_costs[role] = ProbeRoleCosts(role=role)
    _role_costs[role].add(input_tok, output_tok, cost)


def _estimate_cost(model: str, input_tok: int, output_tok: int) -> float:
    """Rough USD cost estimate based on model string."""
    # Anthropic pricing (April 2026, per 1M tokens)
    ANTHROPIC = {
        "claude-opus":   (15.00, 75.00),
        "claude-sonnet": ( 3.00, 15.00),
        "claude-haiku":  ( 0.80,  4.00),
    }
    for prefix, (inp, out) in ANTHROPIC.items():
        if prefix in model.lower():
            return (input_tok * inp + output_tok * out) / 1_000_000

    # Together / OpenRouter open-source defaults
    TOGETHER = {
        "qwen3-vl-8b": (0.10, 0.15),
        "qwen":        (0.10, 0.15),
        "llama":       (0.88, 0.88),
    }
    for prefix, (inp, out) in TOGETHER.items():
        if prefix in model.lower():
            return (input_tok * inp + output_tok * out) / 1_000_000

    # Unknown model — return 0, don't crash
    return 0.0


# ---------------------------------------------------------------------------
# Probe-level LLM cache (separate from pipeline cache)
# ---------------------------------------------------------------------------
# Cache is keyed on (model, role, image_hash, prompt_hash).
# TUTOR and VALIDATOR results are cached across PUPIL runs — the same
# expert descriptions and feature queries are reused at zero cost.
# PUPIL results are NOT cached by default (you want fresh PUPIL responses).

_PROBE_MEM_CACHE: Dict[str, str] = {}

_PROBE_DISK_CACHE_DIR: Optional[Path] = (
    Path(d) / "probe" if (d := os.environ.get("LLM_CACHE_DIR", "")) else None
)


def _probe_cache_key(model: str, role: str,
                     image_hash: str, prompt: str) -> str:
    raw = f"{model}\x00{role}\x00{image_hash}\x00{prompt}"
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()


def _probe_cache_get(key: str) -> Optional[str]:
    if key in _PROBE_MEM_CACHE:
        return _PROBE_MEM_CACHE[key]
    if _PROBE_DISK_CACHE_DIR is not None:
        path = _PROBE_DISK_CACHE_DIR / f"{key}.pkl"
        if path.exists():
            try:
                val = pickle.loads(path.read_bytes())
                _PROBE_MEM_CACHE[key] = val
                return val
            except Exception:
                pass
    return None


def _probe_cache_put(key: str, value: str) -> None:
    _PROBE_MEM_CACHE[key] = value
    if _PROBE_DISK_CACHE_DIR is not None:
        _PROBE_DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _PROBE_DISK_CACHE_DIR / f"{key}.pkl"
        try:
            path.write_bytes(pickle.dumps(value))
        except Exception:
            pass


def clear_probe_cache(disk: bool = False) -> None:
    """Clear the probe in-memory cache (and optionally disk cache).

    Args:
        disk: If True, also delete all .pkl files in the probe disk cache dir.
    """
    _PROBE_MEM_CACHE.clear()
    if disk and _PROBE_DISK_CACHE_DIR is not None and _PROBE_DISK_CACHE_DIR.exists():
        for f in _PROBE_DISK_CACHE_DIR.glob("*.pkl"):
            try:
                f.unlink()
            except Exception:
                pass


# Max concurrent PUPIL calls in asyncio.gather — prevents OpenRouter rate limits
_MAX_CONCURRENT_PUPIL = 5


async def _bounded_gather(coros, max_concurrent: int = _MAX_CONCURRENT_PUPIL):
    """Run coroutines concurrently with a semaphore cap.

    Replacement for raw asyncio.gather() when calling external APIs that may
    impose per-model concurrency limits (e.g. OpenRouter free-tier models).
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def _bounded(coro):
        async with sem:
            return await coro

    return list(await asyncio.gather(*[_bounded(c) for c in coros]))


# ---------------------------------------------------------------------------
# Internal LLM call helper
# ---------------------------------------------------------------------------

async def _call(
    role: str,
    model: str,
    system: str,
    content: list,
    max_tokens: int = 1024,
    call_agent_fn: Optional[Callable] = None,
    cache: bool = True,
    image_hash: str = "",
) -> str:
    """Call LLM, optionally caching result. Records cost for role."""
    fn = call_agent_fn or _get_default_call_agent()

    # Build prompt hash for cache key (text portion only)
    text_parts = [b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"]
    prompt_hash = hashlib.sha256((system + "\n".join(text_parts)).encode()).hexdigest()[:16]

    if cache:
        key = _probe_cache_key(model, role, image_hash, prompt_hash)
        cached = _probe_cache_get(key)
        if cached is not None:
            return cached

    raw, _ = await fn(
        f"PROBE_{role.upper()}",
        content,
        system_prompt=system,
        model=model,
        max_tokens=max_tokens,
    )

    # Token counting — best effort (pipeline may not expose exact counts)
    # Estimate from character lengths as fallback
    est_input  = sum(len(b.get("text", "")) for b in content
                     if isinstance(b, dict) and b.get("type") == "text") // 4
    est_output = len(raw) // 4
    _record_cost(role, model, est_input, est_output)

    if cache:
        _probe_cache_put(key, raw)

    return raw


# ---------------------------------------------------------------------------
# Step 1: TUTOR generates expert descriptions and feature queries
# ---------------------------------------------------------------------------

async def _tutor_describe_image(
    image: ProbeImage,
    domain_config: DomainConfig,
    tutor_model: str,
    call_agent_fn: Optional[Callable],
) -> str:
    """TUTOR produces expert description of an image."""
    system = (
        f"You are a {domain_config.expert_role}. "
        f"Describe {domain_config.item_noun}s using precise domain vocabulary."
    )
    content = [
        image_block(image.path),
        {"type": "text", "text": (
            f"Describe this {domain_config.item_noun} as an expert would, "
            f"using the specific observational vocabulary of your field. "
            f"Focus on: {domain_config.observation_guidance}. "
            f"Do NOT include: {domain_config.non_visual_exclusions}. "
            f"Ground truth class: {image.true_class}. "
            f"Be specific and precise. 3-5 sentences."
        )},
    ]
    return await _call(
        "TUTOR", tutor_model, system, content,
        max_tokens=512, call_agent_fn=call_agent_fn,
        cache=True, image_hash=image.image_hash,
    )


async def _tutor_generate_feature_queries(
    images: List[ProbeImage],
    pair_info: dict,
    domain_config: DomainConfig,
    tutor_model: str,
    call_agent_fn: Optional[Callable],
    n_queries: int = 12,
) -> List[FeatureQuery]:
    """TUTOR generates feature detection queries covering easy→hard range."""
    class_a = pair_info["class_a"]
    class_b = pair_info["class_b"]

    system = f"You are a {domain_config.expert_role}."
    content = [{"type": "text", "text": (
        f"You are helping assess whether a vision model can detect domain-specific "
        f"features in {domain_config.item_noun_plural}.\n\n"
        f"Generate {n_queries} feature detection queries for distinguishing "
        f"{class_a} from {class_b}.\n\n"
        f"Cover a range of difficulty:\n"
        f"  - easy (4 queries): broad features any camera can see "
        f"(overall brightness, obvious color, large patterns)\n"
        f"  - medium (4 queries): moderate features requiring some domain knowledge\n"
        f"  - hard (4 queries): subtle features requiring expert visual vocabulary\n\n"
        f"For each query, provide:\n"
        f"  - feature_id: short snake_case name\n"
        f"  - question: a yes/no question a vision model can answer from the image\n"
        f"  - diagnostic_for: '{class_a}' or '{class_b}' (which class the feature "
        f"being present is diagnostic for)\n"
        f"  - difficulty: 'easy' | 'medium' | 'hard'\n\n"
        f"Respond with a JSON array of objects with those four fields.\n"
        f"Focus on: {domain_config.observation_guidance}."
    )}]

    # Use image_hash of first image as cache anchor (queries are domain-level, not per-image)
    anchor_hash = images[0].image_hash if images else "no_image"
    raw = await _call(
        "TUTOR", tutor_model, system, content,
        max_tokens=2048, call_agent_fn=call_agent_fn,
        cache=True, image_hash=anchor_hash + "_queries",
    )

    # TUTOR returns a JSON array — parse_json_block only finds objects,
    # so try parse_json_array first, then fall back to parse_json_block.
    parsed = parse_json_array(raw) or parse_json_block(raw)
    if not isinstance(parsed, list):
        parsed = []

    queries = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        queries.append(FeatureQuery(
            feature_id=item.get("feature_id", f"feature_{len(queries)}"),
            description=item.get("question", ""),
            expected_class=item.get("diagnostic_for", class_a),
            difficulty=item.get("difficulty", DIFFICULTY_MEDIUM),
        ))

    return queries[:n_queries]


# ---------------------------------------------------------------------------
# Step 2: PUPIL free description (vocabulary overlap scoring)
# ---------------------------------------------------------------------------

async def _pupil_describe_image(
    image: ProbeImage,
    domain_config: DomainConfig,
    pupil_model: str,
    call_agent_fn: Optional[Callable],
) -> str:
    """PUPIL freely describes the image with no domain hints."""
    system = "You are a visual analysis assistant."
    content = [
        image_block(image.path),
        {"type": "text", "text": (
            f"Describe what you see in this image in detail. "
            f"Be specific about textures, patterns, colors, and any notable features."
        )},
    ]
    # PUPIL results not cached — fresh responses required
    return await _call(
        "PUPIL", pupil_model, system, content,
        max_tokens=512, call_agent_fn=call_agent_fn,
        cache=False, image_hash=image.image_hash,
    )


async def _validator_score_vocabulary_overlap(
    pupil_description: str,
    expert_description: str,
    domain_config: DomainConfig,
    validator_model: str,
    call_agent_fn: Optional[Callable],
) -> float:
    """VALIDATOR scores vocabulary overlap between PUPIL and expert descriptions."""
    system = "You are an objective evaluator comparing two descriptions of the same image."
    content = [{"type": "text", "text": (
        f"Compare these two descriptions of the same {domain_config.item_noun}:\n\n"
        f"EXPERT DESCRIPTION:\n{expert_description}\n\n"
        f"MODEL DESCRIPTION:\n{pupil_description}\n\n"
        f"Score the vocabulary overlap on a scale of 0.0 to 1.0:\n"
        f"  0.0 = completely generic description, no domain-specific terms\n"
        f"  0.5 = some overlap with domain vocabulary\n"
        f"  1.0 = uses same domain-specific observational terms as the expert\n\n"
        f"Consider: does the model mention similar features, textures, patterns "
        f"or properties as the expert, even if phrased differently?\n\n"
        f"Respond with JSON: {{\"score\": 0.0-1.0, \"reason\": \"brief explanation\"}}"
    )}]

    # Cache on (expert_desc_hash, pupil_desc_hash) — stable across reruns
    img_hash = hashlib.md5((expert_description + pupil_description).encode()).hexdigest()
    raw = await _call(
        "VALIDATOR", validator_model, system, content,
        max_tokens=256, call_agent_fn=call_agent_fn,
        cache=True, image_hash=img_hash,
    )
    result = parse_json_block(raw)
    if result and "score" in result:
        return float(result["score"])
    # Fallback: look for a number in raw text
    import re
    m = re.search(r'\b(0\.\d+|1\.0)\b', raw)
    return float(m.group()) if m else 0.0


# ---------------------------------------------------------------------------
# Step 3: PUPIL feature detection under query
# ---------------------------------------------------------------------------

async def _pupil_answer_feature_query(
    image: ProbeImage,
    query: FeatureQuery,
    domain_config: DomainConfig,
    pupil_model: str,
    call_agent_fn: Optional[Callable],
) -> Tuple[bool, str]:
    """PUPIL answers one yes/no feature detection query."""
    system = "You are a visual analysis assistant. Answer questions about images precisely."
    content = [
        image_block(image.path),
        {"type": "text", "text": (
            f"Look carefully at this {domain_config.item_noun}.\n\n"
            f"Question: {query.description}\n\n"
            f"Respond with JSON: "
            f'{{\"answer\": \"yes\" or \"no\", \"observation\": \"what you see\"}}'
        )},
    ]
    raw = await _call(
        "PUPIL", pupil_model, system, content,
        max_tokens=256, call_agent_fn=call_agent_fn,
        cache=False,
    )
    result = parse_json_block(raw)
    if result:
        answer = result.get("answer", "").lower().strip()
        observation = result.get("observation", raw[:200])
        return answer.startswith("y"), observation
    return "yes" in raw.lower(), raw[:200]


async def _validator_verify_feature_detection(
    image: ProbeImage,
    query: FeatureQuery,
    pupil_answer: bool,
    pupil_observation: str,
    domain_config: DomainConfig,
    validator_model: str,
    call_agent_fn: Optional[Callable],
) -> Tuple[bool, bool]:
    """VALIDATOR independently answers the same query to provide ground truth.

    Returns (validator_answer, pupil_correct).
    """
    system = f"You are a {domain_config.expert_role} with expert visual analysis capability."
    content = [
        image_block(image.path),
        {"type": "text", "text": (
            f"Look carefully at this {domain_config.item_noun}.\n\n"
            f"Question: {query.description}\n\n"
            f"Respond with JSON: "
            f'{{\"answer\": \"yes\" or \"no\", \"observation\": \"what you see\"}}'
        )},
    ]
    img_hash = f"{image.image_hash}_{query.feature_id}"
    raw = await _call(
        "VALIDATOR", validator_model, system, content,
        max_tokens=256, call_agent_fn=call_agent_fn,
        cache=True, image_hash=img_hash,
    )
    result = parse_json_block(raw)
    validator_answer = False
    if result:
        validator_answer = result.get("answer", "").lower().strip().startswith("y")
    else:
        validator_answer = "yes" in raw.lower()

    pupil_correct = (pupil_answer == validator_answer)
    return validator_answer, pupil_correct


# ---------------------------------------------------------------------------
# Step 4: Rule comprehension (zero-shot vs rule-injected classification)
# ---------------------------------------------------------------------------

async def _classify_zero_shot(
    image: ProbeImage,
    pair_info: dict,
    domain_config: DomainConfig,
    pupil_model: str,
    call_agent_fn: Optional[Callable],
) -> str:
    """PUPIL classifies image zero-shot (no rules)."""
    class_a = pair_info["class_a"]
    class_b = pair_info["class_b"]
    system = f"You are a visual classification assistant."
    content = [
        image_block(image.path),
        {"type": "text", "text": (
            f"Classify this {domain_config.item_noun} as one of:\n"
            f"  A) {class_a}\n"
            f"  B) {class_b}\n\n"
            f"Respond with JSON: "
            f'{{\"classification\": \"{class_a}\" or \"{class_b}\", '
            f'\"reasoning\": \"brief\"}}'
        )},
    ]
    raw = await _call(
        "PUPIL", pupil_model, system, content,
        max_tokens=256, call_agent_fn=call_agent_fn, cache=False,
    )
    result = parse_json_block(raw)
    if result:
        c = result.get("classification", "")
        if class_a.lower() in c.lower():
            return class_a
        if class_b.lower() in c.lower():
            return class_b
    return class_a if class_a.lower() in raw.lower() else class_b


async def _classify_with_rule(
    image: ProbeImage,
    pair_info: dict,
    domain_config: DomainConfig,
    pupil_model: str,
    seed_rule: dict,
    call_agent_fn: Optional[Callable],
) -> str:
    """PUPIL classifies image with a DD rule injected into context."""
    class_a = pair_info["class_a"]
    class_b = pair_info["class_b"]
    rule_text = seed_rule.get("rule", "")
    preconditions = seed_rule.get("preconditions", [])
    preconds_str = "\n".join(f"  - {p}" for p in preconditions)

    system = (
        f"You are a visual classification assistant. "
        f"Apply the provided classification rule precisely."
    )
    content = [
        image_block(image.path),
        {"type": "text", "text": (
            f"Classify this {domain_config.item_noun} using the rule below.\n\n"
            f"RULE: {rule_text}\n"
            f"PRECONDITIONS (all must be met to apply rule):\n{preconds_str}\n\n"
            f"If the preconditions are met, apply the rule. "
            f"Otherwise use your best visual judgment.\n\n"
            f"Options:\n  A) {class_a}\n  B) {class_b}\n\n"
            f"Respond with JSON: "
            f'{{\"classification\": \"{class_a}\" or \"{class_b}\", '
            f'\"rule_applied\": true/false, \"reasoning\": \"brief\"}}'
        )},
    ]
    raw = await _call(
        "PUPIL", pupil_model, system, content,
        max_tokens=256, call_agent_fn=call_agent_fn, cache=False,
    )
    result = parse_json_block(raw)
    if result:
        c = result.get("classification", "")
        if class_a.lower() in c.lower():
            return class_a
        if class_b.lower() in c.lower():
            return class_b
    return class_a if class_a.lower() in raw.lower() else class_b


# ---------------------------------------------------------------------------
# Step 5: Consistency check
# ---------------------------------------------------------------------------

async def _check_consistency(
    images: List[ProbeImage],
    pair_info: dict,
    domain_config: DomainConfig,
    pupil_model: str,
    seed_rule: Optional[dict],
    call_agent_fn: Optional[Callable],
    n_images: int = _CONSISTENCY_N_IMAGES,
    n_repeats: int = _CONSISTENCY_REPEATS,
) -> float:
    """Run PUPIL n_repeats times on the same n_images; return fraction consistent."""
    subset = images[:n_images]
    consistent = 0
    total = 0
    for image in subset:
        results = []
        for _ in range(n_repeats):
            if seed_rule:
                pred = await _classify_with_rule(
                    image, pair_info, domain_config, pupil_model,
                    seed_rule, call_agent_fn,
                )
            else:
                pred = await _classify_zero_shot(
                    image, pair_info, domain_config, pupil_model, call_agent_fn,
                )
            results.append(pred)
        # Consistent if all n_repeats agree
        if len(set(results)) == 1:
            consistent += 1
        total += 1
    return consistent / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def _compute_verdict(
    perception_score: float,
    rule_comprehension_delta: float,
    consistency_score: float,
    feature_profile: dict,
) -> Tuple[str, List[str], List[str]]:
    """Compute readiness verdict and generate recommendations.

    Returns (verdict, weak_points, recommendations).
    """
    weak_points = []
    recommendations = []

    # Hard no-go conditions
    if perception_score < _NOGO_PERCEPTION_MAX:
        weak_points.append(
            f"Perception score {perception_score:.2f} < {_NOGO_PERCEPTION_MAX} "
            f"— PUPIL cannot describe domain features in any recognisable form"
        )
    if consistency_score < _NOGO_CONSISTENCY_MAX:
        weak_points.append(
            f"Consistency score {consistency_score:.2f} < {_NOGO_CONSISTENCY_MAX} "
            f"— PUPIL gives unstable answers, suggesting perception is near-random"
        )

    if weak_points:
        recommendations.append(
            "PUPIL visual encoder likely lacks domain-relevant feature representation. "
            "Consider fine-tuning the visual encoder on domain images before attempting DD."
        )
        recommendations.append(
            "Alternatively, try a PUPIL model with a stronger visual backbone "
            "(e.g. larger parameter count or domain-pretrained)."
        )
        return VERDICT_NO_GO, weak_points, recommendations

    # Check go conditions
    go_flags = []
    partial_flags = []

    if perception_score >= _GO_PERCEPTION_MIN:
        go_flags.append("perception")
    else:
        partial_flags.append(f"perception {perception_score:.2f} < {_GO_PERCEPTION_MIN}")
        recommendations.append(
            "Perception score is borderline. Simplify rule vocabulary to use "
            "coarser, more basic visual terms the PUPIL clearly responds to."
        )

    if rule_comprehension_delta >= _GO_RULE_COMPREHENSION_MIN:
        go_flags.append("rule_comprehension")
    else:
        partial_flags.append(
            f"rule comprehension delta {rule_comprehension_delta:.2f} "
            f"< {_GO_RULE_COMPREHENSION_MIN}"
        )
        recommendations.append(
            "Rule injection does not reliably improve PUPIL accuracy. "
            "Try simpler, more explicit rule phrasing. "
            "Consider adding a PUPIL re-classification gate in the DD protocol."
        )

    if consistency_score >= _GO_CONSISTENCY_MIN:
        go_flags.append("consistency")
    else:
        partial_flags.append(f"consistency {consistency_score:.2f} < {_GO_CONSISTENCY_MIN}")
        recommendations.append(
            "PUPIL shows moderate inconsistency on repeated runs. "
            "Use temperature=0 for PUPIL calls and increase pool validation size."
        )

    # Feature profile weak points
    for feat_id, rate in feature_profile.items():
        parts = feat_id.rsplit("_", 1)
        difficulty = parts[-1] if parts[-1] in (DIFFICULTY_EASY, DIFFICULTY_MEDIUM, DIFFICULTY_HARD) else ""
        if difficulty == DIFFICULTY_EASY and rate < 0.60:
            weak_points.append(
                f"Cannot reliably detect easy feature '{feat_id}' (rate={rate:.2f})"
            )
        elif difficulty == DIFFICULTY_MEDIUM and rate < 0.40:
            weak_points.append(
                f"Low detection rate on medium feature '{feat_id}' (rate={rate:.2f})"
            )

    if weak_points:
        recommendations.append(
            "DD rules should avoid the feature types listed in weak_points. "
            "The TUTOR should be informed of these limitations before authoring rules."
        )

    if partial_flags:
        return VERDICT_PARTIAL, weak_points, recommendations

    return VERDICT_GO, weak_points, recommendations


# ---------------------------------------------------------------------------
# Main probe entry point
# ---------------------------------------------------------------------------

async def probe(
    pupil_model: str,
    tutor_model: str,
    validator_model: str,
    domain_config: DomainConfig,
    probe_images: List[ProbeImage],
    pair_info: dict,
    seed_rule: Optional[dict] = None,
    call_agent_fn: Optional[Callable] = None,
    n_feature_queries: int = 12,
    console=None,
) -> dict:
    """Run the PUPIL Domain Readiness Probe.

    Args:
        pupil_model:       Model to probe (e.g. "qwen/qwen3-vl-8b-instruct")
        tutor_model:       Expert model for descriptions + queries
                           (e.g. "claude-opus-4-6")
        validator_model:   Validation model for scoring
                           (e.g. "claude-sonnet-4-6")
        domain_config:     DomainConfig for the target domain
        probe_images:      20-30 labeled ProbeImages (balanced across classes)
        pair_info:         {"class_a": ..., "class_b": ..., "pair_id": ...}
        seed_rule:         Known-good DD rule (from prior domain or prior session)
                           used in rule comprehension and consistency steps.
                           If None, a synthetic rule is generated by the TUTOR.
        call_agent_fn:     LLM call function. Defaults to core pipeline call_agent.
        n_feature_queries: Number of feature detection queries to generate (default 12)
        console:           Optional rich Console for progress output.

    Returns:
        Full probe report dict with keys:
          verdict, perception_score, feature_profile, zero_shot_accuracy,
          rule_aided_accuracy, rule_comprehension_delta, consistency_score,
          weak_points, recommendations, per_image_results, costs, metadata
    """
    _print = console.print if console else lambda *a, **kw: None
    reset_probe_costs()

    _print(f"\n[bold]PUPIL Domain Readiness Probe[/bold]")
    _print(f"  PUPIL:     [cyan]{pupil_model}[/cyan]")
    _print(f"  TUTOR:     [cyan]{tutor_model}[/cyan]")
    _print(f"  VALIDATOR: [cyan]{validator_model}[/cyan]")
    _print(f"  Domain:    [cyan]{domain_config.item_noun_plural}[/cyan]")
    _print(f"  Pair:      [cyan]{pair_info['class_a']} vs {pair_info['class_b']}[/cyan]")
    _print(f"  Images:    {len(probe_images)}")

    report: dict = {
        "metadata": {
            "pupil_model": pupil_model,
            "tutor_model": tutor_model,
            "validator_model": validator_model,
            "domain": domain_config.item_noun_plural,
            "pair_id": pair_info.get("pair_id", ""),
            "n_images": len(probe_images),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "per_image_results": [],
    }

    # ------------------------------------------------------------------
    # Step 1: TUTOR descriptions (cached — reused across PUPIL runs)
    # ------------------------------------------------------------------
    _print("\n  [bold]Step 1/5[/bold] TUTOR expert descriptions...")
    tutor_descs = {}
    for img in probe_images:
        desc = await _tutor_describe_image(
            img, domain_config, tutor_model, call_agent_fn
        )
        tutor_descs[img.path] = desc

    # ------------------------------------------------------------------
    # Step 2: PUPIL free descriptions + vocabulary overlap scoring
    # ------------------------------------------------------------------
    _print("  [bold]Step 2/5[/bold] PUPIL vocabulary probe...")
    perception_scores = []
    for img in probe_images:
        pupil_desc = await _pupil_describe_image(
            img, domain_config, pupil_model, call_agent_fn
        )
        overlap = await _validator_score_vocabulary_overlap(
            pupil_desc, tutor_descs[img.path],
            domain_config, validator_model, call_agent_fn,
        )
        perception_scores.append(overlap)

    perception_score = sum(perception_scores) / len(perception_scores) if perception_scores else 0.0
    _print(f"    Perception score: [{'green' if perception_score >= _GO_PERCEPTION_MIN else 'yellow'}]"
           f"{perception_score:.3f}[/]")
    report["perception_score"] = round(perception_score, 4)
    report["perception_per_image"] = [round(s, 4) for s in perception_scores]

    # ------------------------------------------------------------------
    # Step 3: Feature detection queries (TUTOR generates, cached)
    # ------------------------------------------------------------------
    _print("  [bold]Step 3/5[/bold] Feature detection probe...")
    feature_queries = await _tutor_generate_feature_queries(
        probe_images, pair_info, domain_config,
        tutor_model, call_agent_fn, n_queries=n_feature_queries,
    )
    _print(f"    {len(feature_queries)} queries generated")

    # Run PUPIL + VALIDATOR on a representative subset of images
    query_images = probe_images[:min(10, len(probe_images))]
    feature_results: Dict[str, List[bool]] = {}

    for query in feature_queries:
        correct_list = []
        for img in query_images:
            # PUPIL answers
            pupil_ans, pupil_obs = await _pupil_answer_feature_query(
                img, query, domain_config, pupil_model, call_agent_fn
            )
            # VALIDATOR provides ground truth
            _, pupil_correct = await _validator_verify_feature_detection(
                img, query, pupil_ans, pupil_obs,
                domain_config, validator_model, call_agent_fn,
            )
            correct_list.append(pupil_correct)

        feat_key = f"{query.feature_id}_{query.difficulty}"
        feature_results[feat_key] = correct_list

    feature_profile = {
        k: round(sum(v) / len(v), 4) if v else 0.0
        for k, v in feature_results.items()
    }
    report["feature_profile"] = feature_profile
    report["feature_queries"] = [
        {"feature_id": q.feature_id, "question": q.description,
         "diagnostic_for": q.expected_class, "difficulty": q.difficulty}
        for q in feature_queries
    ]

    avg_by_difficulty = {}
    for diff in (DIFFICULTY_EASY, DIFFICULTY_MEDIUM, DIFFICULTY_HARD):
        scores = [v for k, v in feature_profile.items() if k.endswith(f"_{diff}")]
        avg_by_difficulty[diff] = round(sum(scores) / len(scores), 4) if scores else None
    report["feature_detection_by_difficulty"] = avg_by_difficulty
    _print(f"    Easy: {avg_by_difficulty.get('easy')!s:>5}  "
           f"Medium: {avg_by_difficulty.get('medium')!s:>5}  "
           f"Hard: {avg_by_difficulty.get('hard')!s:>5}")

    # ------------------------------------------------------------------
    # Step 4: Rule comprehension (zero-shot vs rule-injected)
    # ------------------------------------------------------------------
    _print("  [bold]Step 4/5[/bold] Rule comprehension probe...")

    # Generate a seed rule if none provided
    if seed_rule is None:
        _print("    No seed_rule provided — TUTOR generating synthetic rule...")
        class_a = pair_info["class_a"]
        class_b = pair_info["class_b"]
        system = f"You are a {domain_config.expert_role}."
        rule_content = [{"type": "text", "text": (
            f"Write one classification rule for distinguishing "
            f"{class_a} from {class_b} in {domain_config.item_noun_plural}.\n"
            f"The rule must reference only visually observable features.\n"
            f"Format as JSON: {{\"rule\": \"...\", "
            f"\"preconditions\": [\"...\", \"...\"], "
            f"\"favors\": \"{class_a}\"}}"
        )}]
        anchor_hash = probe_images[0].image_hash if probe_images else "no_img"
        raw = await _call(
            "TUTOR", tutor_model, system, rule_content,
            max_tokens=512, call_agent_fn=call_agent_fn,
            cache=True, image_hash=anchor_hash + "_seed_rule",
        )
        seed_rule = parse_json_block(raw) or {
            "rule": raw[:200], "preconditions": [], "favors": class_a
        }
        _print(f"    Seed rule: [dim]{seed_rule.get('rule', '')[:100]}[/dim]")

    # Zero-shot accuracy
    zero_shot_correct = 0
    rule_aided_correct = 0
    cls_images = probe_images  # all images

    # Use _bounded_gather (max 5 concurrent) rather than raw asyncio.gather
    # to avoid hitting OpenRouter per-model concurrency limits.
    zero_shot_results = await _bounded_gather([
        _classify_zero_shot(img, pair_info, domain_config, pupil_model, call_agent_fn)
        for img in cls_images
    ])
    for img, pred in zip(cls_images, zero_shot_results):
        if pred == img.true_class:
            zero_shot_correct += 1

    rule_aided_results = await _bounded_gather([
        _classify_with_rule(img, pair_info, domain_config, pupil_model,
                            seed_rule, call_agent_fn)
        for img in cls_images
    ])
    for img, pred in zip(cls_images, rule_aided_results):
        if pred == img.true_class:
            rule_aided_correct += 1

    n_cls = len(cls_images)
    zero_shot_acc    = zero_shot_correct    / n_cls if n_cls else 0.0
    rule_aided_acc   = rule_aided_correct   / n_cls if n_cls else 0.0
    comprehension_delta = rule_aided_acc - zero_shot_acc

    report["zero_shot_accuracy"]        = round(zero_shot_acc, 4)
    report["rule_aided_accuracy"]       = round(rule_aided_acc, 4)
    report["rule_comprehension_delta"]  = round(comprehension_delta, 4)
    report["seed_rule_used"]            = seed_rule

    _print(f"    Zero-shot:  {zero_shot_acc:.3f}  "
           f"Rule-aided: {rule_aided_acc:.3f}  "
           f"Delta: [{'green' if comprehension_delta >= _GO_RULE_COMPREHENSION_MIN else 'yellow'}]"
           f"{comprehension_delta:+.3f}[/]")

    # ------------------------------------------------------------------
    # Step 5: Consistency check
    # ------------------------------------------------------------------
    _print("  [bold]Step 5/5[/bold] Consistency check...")
    consistency_score = await _check_consistency(
        probe_images, pair_info, domain_config,
        pupil_model, seed_rule, call_agent_fn,
    )
    report["consistency_score"] = round(consistency_score, 4)
    _print(f"    Consistency: [{'green' if consistency_score >= _GO_CONSISTENCY_MIN else 'yellow'}]"
           f"{consistency_score:.3f}[/]")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    verdict, weak_points, recommendations = _compute_verdict(
        perception_score, comprehension_delta,
        consistency_score, feature_profile,
    )
    report["verdict"] = verdict
    report["weak_points"] = weak_points
    report["recommendations"] = recommendations

    colour = {"go": "green", "partial": "yellow", "no-go": "red"}[verdict]
    _print(f"\n  Verdict: [{colour}][bold]{verdict.upper()}[/bold][/{colour}]")
    for wp in weak_points:
        _print(f"    [red]⚠ {wp}[/red]")
    for rec in recommendations:
        _print(f"    [yellow]→ {rec}[/yellow]")

    # ------------------------------------------------------------------
    # Costs
    # ------------------------------------------------------------------
    report["costs"] = get_probe_costs()
    total_cost = sum(rc["cost_usd"] for rc in report["costs"].values())
    report["total_cost_usd"] = round(total_cost, 6)
    _print(f"\n  Cost: [cyan]${total_cost:.4f}[/cyan]")

    return report


# ---------------------------------------------------------------------------
# Convenience: save / load probe report
# ---------------------------------------------------------------------------

def save_report(report: dict, path: str | Path) -> None:
    """Save probe report as JSON."""
    Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def load_report(path: str | Path) -> dict:
    """Load a previously saved probe report."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
