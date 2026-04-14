"""
pupil_eval.py — PUPIL classifier evaluation using Claude Haiku for PyroWatch.

Runs claude-haiku-4-5-20251001 as the ground_sentinel-tier PUPIL classifier for
FIgLib wildfire frames.  Supports:

  - Baseline evaluation (no rules): measure raw error distribution
  - Rule-injected evaluation: measure uplift after a DD session produces rules
  - Confident-miss mining: find frames where Haiku is confidently wrong
  - Summary metrics: accuracy, precision, recall, class distribution

Known failure mode (FIgLib early smoke):
  - early_smoke_signature → heat_shimmer_artifact at 0.75–0.90
    (faint early smoke on hot terrain, before visible flame crown)
  - early_smoke_signature → atmospheric_haze
    (very faint smoke blends with distant haze layer)

Integration:
    This module is called by run_pupil_dd_experiment.py and can also be used
    standalone for offline evaluation.

    from pupil_eval import run_baseline_eval, eval_summary
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Model constant
# ---------------------------------------------------------------------------

PUPIL_MODEL = "claude-haiku-4-5-20251001"   # Cost-efficient Haiku for bulk eval
TUTOR_MODEL = "claude-opus-4-6"             # High-quality TUTOR for DD session
VALIDATOR_MODEL = "claude-sonnet-4-6"       # Validator for pool validation

# ---------------------------------------------------------------------------
# Classification system prompt
# ---------------------------------------------------------------------------

CLASSIFICATION_SYSTEM = """\
You are a wildfire early-detection classifier running on a ground-sentinel
mountaintop camera system (fixed PTZ, RGB 4K, wide-angle).

Your task is to classify a HPWREN/ALERTCalifornia hillside or chaparral camera
frame captured during dry-season fire weather into EXACTLY ONE of the following
classes:

  early_smoke_signature   — Genuine combustion smoke from a wildland fire,
                            typically faint and translucent in early stages.
                            Key features:
                            • Diffuse, slightly opaque plume with soft edges
                            • Consistent drift direction aligned with prevailing wind
                            • Opacity increases upward from point of origin
                            • Slight brownish or grayish tint distinct from terrain
                            • In sequences: plume expands and drifts across frames
                            IMPORTANT: At ignition time the smoke may be very faint —
                            a small, translucent wisp above the ridge line still counts
                            as early_smoke_signature if the opacity gradient is present.

  heat_shimmer_artifact   — Optical distortion caused by hot-air convection above
                            sun-heated terrain. NOT a fire.
                            Key features:
                            • Wavy, turbulent distortion directly above rock or
                              dark soil surfaces (not in open air or above vegetation)
                            • No consistent horizontal drift direction
                            • Distortion is transparent (background terrain visible
                              through it) with no opacity gradient
                            • Oscillates vertically in place; does not expand
                            • Most intense at midday over exposed rock/pavement

  atmospheric_haze        — Ambient haze, smog, or aerosol from distant sources.
                            No local ignition point.
                            Key features:
                            • Uniform, horizon-wide reduction in visibility
                            • No localised plume geometry or origin point
                            • No opacity gradient (haze is equally dense throughout)
                            • Background terrain visible but softened uniformly

  dust_plume              — Windblown mineral dust raised from dry terrain.
                            Key features:
                            • Tan, beige, or light-brown opaque cloud
                            • Often originates from road, dry riverbed, or field
                            • Moves rapidly with wind, lacks upward buoyancy
                            • No thermal signature (does not billow upward)

  fog_patch               — Ground-level moisture cloud. Typically morning/evening.
                            Key features:
                            • Brilliant white, high-contrast plume or bank
                            • Does not drift up from terrain (settles or dissipates)
                            • Often fills valleys and canyons
                            • Not present at peak dry-season midday temperatures

  no_fire                 — Clear terrain with no smoke, haze, shimmer, or dust
                            visible above threshold. Normal dry-season appearance.

Output ONLY a raw JSON object with no markdown fences, no preamble, no
explanation outside the JSON.  The exact required schema is:

{"class": "<one of the six class names>", "confidence": <0.0 to 1.0>, "reasoning": "<one sentence>"}

Do not output anything before or after the JSON object.\
"""

# ---------------------------------------------------------------------------
# Rule formatting helper
# ---------------------------------------------------------------------------

def _format_rules(rules: list[dict]) -> str:
    """Format a list of rule dicts into the knowledge injection block."""
    lines: list[str] = []
    for r in rules:
        rule_text = r.get("rule", str(r))
        preconditions = r.get("preconditions", [])
        context = r.get("context_preconditions", {})
        lines.append(f"Rule: {rule_text}")
        if preconditions:
            lines.append("Preconditions: " + "; ".join(preconditions))
        else:
            lines.append("Preconditions: (none specified)")
        if context:
            for cset, ctx_text in context.items():
                lines.append(f"Context ({cset}): {ctx_text}")
    return "\n".join(lines)


def _build_system_with_rules(rules: list[dict]) -> str:
    """Return the classification system prompt extended with injected rules."""
    rule_block = _format_rules(rules)
    return (
        CLASSIFICATION_SYSTEM
        + "\n\nKNOWLEDGE RULES (apply before classifying):\n"
        + rule_block
    )


# ---------------------------------------------------------------------------
# Default call_agent resolver
# ---------------------------------------------------------------------------

_default_call_agent_fn: Optional[Callable] = None


def _get_call_agent() -> Callable:
    global _default_call_agent_fn
    if _default_call_agent_fn is None:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
        from core.pipeline.agents import call_agent
        _default_call_agent_fn = call_agent
    return _default_call_agent_fn


# ---------------------------------------------------------------------------
# Image encoding helper
# ---------------------------------------------------------------------------

def _image_block(image_path: str) -> dict:
    """Return an Anthropic-style image content block for a file."""
    import base64
    ext = Path(image_path).suffix.lower()
    _MEDIA = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = _MEDIA.get(ext, "image/jpeg")
    data = base64.standard_b64encode(Path(image_path).read_bytes()).decode("ascii")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": data,
        },
    }


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Optional[dict]:
    """Extract the first complete JSON object from model output."""
    import json
    import re

    # Strip thinking tags if model emits them
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Try fenced block first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Bracket-counting parse (handles multiline JSON without fences)
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(cleaned[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start: i + 1])
                except json.JSONDecodeError:
                    pass
                break
    return None


# ---------------------------------------------------------------------------
# Core classification call
# ---------------------------------------------------------------------------

async def classify_frame(
    image_path: str,
    model: str = PUPIL_MODEL,
    rules: Optional[list[dict]] = None,
    call_agent_fn: Optional[Callable] = None,
) -> dict:
    """Classify a single wildfire surveillance frame using the PUPIL model.

    Parameters
    ----------
    image_path:
        Absolute path to the frame image (JPEG or PNG).
    model:
        Anthropic model to use; defaults to PUPIL_MODEL (Haiku).
    rules:
        Optional list of DD rule dicts to inject into the system prompt.
        When provided, the classifier is told to apply these rules before
        classifying.  Each dict should contain at minimum "rule" (str) and
        optionally "preconditions" (list[str]) and
        "context_preconditions" (dict[str, str]).
    call_agent_fn:
        The LLM call function; defaults to core.pipeline.agents.call_agent.

    Returns
    -------
    dict with keys:
        image_path, predicted_class, confidence, reasoning, duration_ms
    """
    _call = call_agent_fn or _get_call_agent()

    system_prompt = _build_system_with_rules(rules) if rules else CLASSIFICATION_SYSTEM

    content = [
        _image_block(image_path),
        {"type": "text", "text": "Classify this wildfire surveillance frame."},
    ]

    t0 = time.time()
    text, ms = await _call(
        "PUPIL_CLASSIFY",
        content,
        system_prompt=system_prompt,
        model=model,
        max_tokens=256,
    )
    duration_ms = ms or int((time.time() - t0) * 1000)

    parsed = _extract_json(text)
    if parsed and "class" in parsed:
        return {
            "image_path": image_path,
            "predicted_class": parsed["class"],
            "confidence": float(parsed.get("confidence", 0.0)),
            "reasoning": parsed.get("reasoning", ""),
            "duration_ms": duration_ms,
            "raw_response": text,
        }

    # Parse failure: return a sentinel result so the eval doesn't crash
    return {
        "image_path": image_path,
        "predicted_class": "parse_error",
        "confidence": 0.0,
        "reasoning": f"(JSON parse failed — raw: {text[:300]})",
        "duration_ms": duration_ms,
        "raw_response": text,
    }


# ---------------------------------------------------------------------------
# Batch evaluation helpers
# ---------------------------------------------------------------------------

async def run_baseline_eval(
    labeled_frames: list[tuple[str, str]],
    model: str = PUPIL_MODEL,
    n_concurrent: int = 5,
    call_agent_fn: Optional[Callable] = None,
) -> list[dict]:
    """Evaluate PUPIL on a labeled frame set without any DD rules.

    Parameters
    ----------
    labeled_frames:
        List of (image_path, ground_truth_label) tuples.
    model:
        Anthropic model string.
    n_concurrent:
        Maximum number of parallel classify_frame calls.
    call_agent_fn:
        LLM backend; defaults to core.pipeline.agents.call_agent.

    Returns
    -------
    List of result dicts, each containing:
        image_path, ground_truth, predicted_class, confidence, reasoning,
        correct (bool), duration_ms
    """
    sem = asyncio.Semaphore(n_concurrent)

    async def _eval_one(image_path: str, ground_truth: str) -> dict:
        async with sem:
            result = await classify_frame(
                image_path=image_path,
                model=model,
                rules=None,
                call_agent_fn=call_agent_fn,
            )
        return {
            **result,
            "ground_truth": ground_truth,
            "correct": result["predicted_class"] == ground_truth,
        }

    tasks = [_eval_one(img, gt) for img, gt in labeled_frames]
    return await asyncio.gather(*tasks)


async def run_eval_with_rules(
    labeled_frames: list[tuple[str, str]],
    rules: list[dict],
    model: str = PUPIL_MODEL,
    n_concurrent: int = 5,
    call_agent_fn: Optional[Callable] = None,
) -> list[dict]:
    """Evaluate PUPIL on labeled frames with DD rules injected.

    Parameters
    ----------
    labeled_frames:
        List of (image_path, ground_truth_label) tuples.
    rules:
        List of rule dicts from a DD session transcript["final_rules"]["ground_sentinel"].
    model:
        Anthropic model string.
    n_concurrent:
        Maximum number of parallel calls.
    call_agent_fn:
        LLM backend; defaults to core.pipeline.agents.call_agent.

    Returns
    -------
    Same structure as run_baseline_eval but each result includes the injected
    rules count in "rules_injected".
    """
    sem = asyncio.Semaphore(n_concurrent)

    async def _eval_one(image_path: str, ground_truth: str) -> dict:
        async with sem:
            result = await classify_frame(
                image_path=image_path,
                model=model,
                rules=rules,
                call_agent_fn=call_agent_fn,
            )
        return {
            **result,
            "ground_truth": ground_truth,
            "correct": result["predicted_class"] == ground_truth,
            "rules_injected": len(rules),
        }

    tasks = [_eval_one(img, gt) for img, gt in labeled_frames]
    return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def find_confident_misses(
    eval_results: list[dict],
    ground_truth: str = "early_smoke_signature",
    confidence_min: float = 0.70,
) -> list[dict]:
    """Return results where the true class is ground_truth but model is wrong
    and confident.

    Parameters
    ----------
    eval_results:
        Output of run_baseline_eval or run_eval_with_rules.
    ground_truth:
        The positive class we care about (false negatives for this class).
        Defaults to "early_smoke_signature".
    confidence_min:
        Minimum confidence to consider a result a "confident miss".

    Returns
    -------
    Subset of eval_results, sorted by confidence descending (most confident
    wrong answers first).
    """
    misses = [
        r for r in eval_results
        if r.get("ground_truth") == ground_truth
        and r.get("predicted_class") != ground_truth
        and r.get("confidence", 0.0) >= confidence_min
    ]
    return sorted(misses, key=lambda x: x.get("confidence", 0.0), reverse=True)


def eval_summary(
    results: list[dict],
    ground_truth_class: str = "early_smoke_signature",
) -> dict:
    """Compute evaluation summary with binary precision/recall metrics.

    Parameters
    ----------
    results:
        Output of run_baseline_eval or run_eval_with_rules.
    ground_truth_class:
        The positive class for binary precision/recall metrics.
        Defaults to "early_smoke_signature".

    Returns
    -------
    dict with keys:
        total, correct, accuracy,
        tp, fp, fn, tn,
        precision, recall, f1,
        confident_misses (count of wrong predictions with conf >= 0.70),
        class_distribution (dict mapping predicted class -> count),
        confusion (dict mapping predicted class -> count for ground_truth frames only)
    """
    total = len(results)

    tp = sum(
        1 for r in results
        if r.get("ground_truth") == ground_truth_class
        and r.get("predicted_class") == ground_truth_class
    )
    fp = sum(
        1 for r in results
        if r.get("ground_truth") != ground_truth_class
        and r.get("predicted_class") == ground_truth_class
    )
    fn = sum(
        1 for r in results
        if r.get("ground_truth") == ground_truth_class
        and r.get("predicted_class") != ground_truth_class
    )
    tn = sum(
        1 for r in results
        if r.get("ground_truth") != ground_truth_class
        and r.get("predicted_class") != ground_truth_class
    )
    correct = tp + tn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    confident_misses = sum(
        1 for r in results
        if r.get("ground_truth") == ground_truth_class
        and r.get("predicted_class") != ground_truth_class
        and r.get("confidence", 0.0) >= 0.70
    )

    class_dist: dict[str, int] = {}
    for r in results:
        cls = r.get("predicted_class", "unknown")
        class_dist[cls] = class_dist.get(cls, 0) + 1

    # Confusion breakdown: for ground_truth frames, what did model predict?
    confusion: dict[str, int] = {}
    for r in results:
        if r.get("ground_truth") == ground_truth_class:
            predicted = r.get("predicted_class", "unknown")
            confusion[predicted] = confusion.get(predicted, 0) + 1

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confident_misses": confident_misses,
        "class_distribution": dict(sorted(class_dist.items())),
        "confusion": dict(sorted(confusion.items())),
    }
