"""
qwen_pupil_eval.py — PUPIL classifier evaluation using Qwen3-VL on OpenRouter.

Runs Qwen/Qwen3-VL-8B-Instruct as the scout-tier PUPIL classifier for maritime
SAR sea-surface frames.  Supports:

  - Baseline evaluation (no rules): measure raw error distribution
  - Rule-injected evaluation: measure uplift after a DD session produces rules
  - Confident-miss mining: find frames where Qwen is confidently wrong
  - Summary metrics: accuracy, precision, recall, class distribution
  - Partial-credit scoring: uncertain_investigate on a true person frame scores
    PARTIAL_CREDIT_SCORE (default 0.5) rather than 0.0 — because a real fleet
    would dispatch a commander drone to investigate, likely recovering the person

Known failure modes (SeaDronesSee val):
  - person_in_water → life_ring_unoccupied at 0.95–0.97 (tractable via DD)
  - person_in_water → other at 0.98 when swimmer is <20px in 5456px frame
    (scale/attention failure; not a resolution limit — fixed by detect-first
    pipeline; uncertain_investigate class added to make this actionable)

Integration:
    This module is called by run_pupil_dd_experiment.py and can also be used
    standalone for offline evaluation.

    from qwen_pupil_eval import run_baseline_eval, eval_summary
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Model constant
# ---------------------------------------------------------------------------

PUPIL_MODEL = "qwen/qwen3-vl-8b-instruct"   # OpenRouter (lowercase = OpenRouter routing)

# Partial credit awarded when uncertain_investigate is predicted on a true
# person_in_water frame.  Rationale: in a real fleet the drone would be
# dispatched for a closer look, likely recovering the person — this is
# strictly better than a confident wrong prediction that triggers no action.
PARTIAL_CREDIT_SCORE: float = 0.5

# ---------------------------------------------------------------------------
# Classification system prompt
# ---------------------------------------------------------------------------

CLASSIFICATION_SYSTEM = """\
You are a UAV maritime surveillance classifier running on a scout-tier drone.
Your task is to classify a sea-surface image captured by a 12MP RGB camera
at 20–40 m altitude (AGL).

Classify the primary object of interest into EXACTLY ONE of the following classes:

  person_in_water      — A human at the water surface. Look for an upright oval
                         silhouette (head and shoulders), bilateral symmetry,
                         skin-tone or life-vest colour contrast, and V-shaped
                         water disturbance flanking the object.

  whitecap             — A breaking wave crest. Characterised by an irregular,
                         frothy bright patch that fades radially from peak
                         brightness to surrounding water colour within its own
                         boundary. Asymmetric and diffuse.

  floating_debris      — Non-human floating material (packaging, clothing, wood,
                         foam). Irregular outline, no bilateral symmetry, no
                         flanking water disturbance.

  life_ring_unoccupied — A ring-shaped flotation device without a person.
                         Torus geometry: bright outer ring with a dark central
                         void. Distinct circular profile.

  uncertain_investigate — A small, isolated object that cannot be confirmed at
                          current altitude but shows one or more person-consistent
                          features: orange/red colouring consistent with a PFD or
                          survival suit, isolated bright spot above the water
                          surface, or position near SAR vessels in open water.
                          IMPORTANT: use this class instead of 'other' whenever
                          ANY person-consistent feature is visible, even faintly.
                          A commander drone will be dispatched to investigate.
                          Also provide investigation_urgency (0.0–1.0): how
                          strongly the visible features suggest a person in water.

  other                — Anything with NO person-consistent features: vessel,
                         bird, seaweed, image artefact, open water, etc.
                         Do NOT use this class if any orange/red isolated object,
                         PFD-coloured spot, or person-sized anomaly is visible.

Output ONLY a raw JSON object with no markdown fences, no preamble, no
explanation outside the JSON.  The exact required schema is:

{"class": "<one of the six class names>", "confidence": <0.0 to 1.0>, "reasoning": "<one sentence>", "investigation_urgency": <0.0 to 1.0 or null>}

Set investigation_urgency to a float when class is uncertain_investigate,
otherwise set it to null.
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
        lines.append(f"Rule: {rule_text}")
        if preconditions:
            lines.append("Preconditions: " + "; ".join(preconditions))
        else:
            lines.append("Preconditions: (none specified)")
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
        from core.pipeline.agents import call_agent
        _default_call_agent_fn = call_agent
    return _default_call_agent_fn


# ---------------------------------------------------------------------------
# Image encoding helper (reuse core utility when available, else inline)
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

    # Strip common thinking tags that Qwen3 models emit before the JSON
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
    """Classify a single sea-surface frame using the PUPIL model.

    Parameters
    ----------
    image_path:
        Absolute path to the frame image (JPEG or PNG).
    model:
        OpenRouter model to use; defaults to PUPIL_MODEL.
    rules:
        Optional list of DD rule dicts to inject into the system prompt.
        When provided, the classifier is told to apply these rules before
        classifying.  Each dict should contain at minimum "rule" (str) and
        optionally "preconditions" (list[str]).
    call_agent_fn:
        The LLM call function; defaults to core.pipeline.agents.call_agent.

    Returns
    -------
    dict with keys:
        image_path, predicted_class, confidence, reasoning, duration_ms
    """
    _call = call_agent_fn or _get_call_agent()

    if rules is not None:
        system_prompt = _build_system_with_rules(rules)
    else:
        system_prompt = CLASSIFICATION_SYSTEM

    content = [
        _image_block(image_path),
        {"type": "text", "text": "Classify this sea-surface frame."},
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
            "investigation_urgency": (
                float(parsed["investigation_urgency"])
                if parsed.get("investigation_urgency") is not None
                else None
            ),
            "duration_ms": duration_ms,
            "raw_response": text,
        }

    # Parse failure: return a sentinel result so the eval doesn't crash
    return {
        "image_path": image_path,
        "predicted_class": "parse_error",
        "confidence": 0.0,
        "reasoning": f"(JSON parse failed — raw: {text[:300]})",
        "investigation_urgency": None,
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
        OpenRouter model string.
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
        List of rule dicts from a DD session transcript["final_rules"]["scout"]
        or transcript["final_rules"]["commander"].
    model:
        OpenRouter model string.
    n_concurrent:
        Maximum number of parallel calls.
    call_agent_fn:
        LLM backend; defaults to core.pipeline.agents.call_agent.

    Returns
    -------
    Same structure as run_baseline_eval but each result includes the injected
    rules in "rules_injected".
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
    ground_truth: str = "person_in_water",
    confidence_min: float = 0.70,
) -> list[dict]:
    """Return results where the true class is ground_truth but the model is wrong
    and confident.

    Parameters
    ----------
    eval_results:
        Output of run_baseline_eval or run_eval_with_rules.
    ground_truth:
        The class label we care about (false negatives for this class).
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
    ground_truth_class: str = "person_in_water",
    partial_credit: float = PARTIAL_CREDIT_SCORE,
) -> dict:
    """Compute a comprehensive evaluation summary with partial-credit scoring.

    Parameters
    ----------
    results:
        Output of run_baseline_eval or run_eval_with_rules.
    ground_truth_class:
        The positive class for binary precision/recall metrics.
    partial_credit:
        Score (0–1) awarded when the true class is ground_truth_class but the
        model predicts uncertain_investigate.  Rationale: in a real fleet, an
        uncertain_investigate prediction triggers a commander drone dispatch —
        the person would likely be found.  Default: PARTIAL_CREDIT_SCORE (0.5).
        Set to 0.0 to use strict binary scoring only.

    Returns
    -------
    dict with keys:
        total, correct, accuracy,
        tp, fp, fn, tn,
        partial (uncertain_investigate on true positive frames),
        precision, recall,
        weighted_recall  (tp + partial*partial_credit) / (tp + fn)
        weighted_score   (tp + tn + partial*partial_credit) / total
        avg_urgency      mean investigation_urgency for uncertain_investigate results
        confident_misses (count of wrong non-uncertain predictions with conf >= 0.70),
        class_distribution (dict mapping predicted class -> count)
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
        and r.get("predicted_class") != "uncertain_investigate"
    )
    tn = sum(
        1 for r in results
        if r.get("ground_truth") != ground_truth_class
        and r.get("predicted_class") != ground_truth_class
    )

    # Partial: uncertain_investigate on a true positive frame
    partial_results = [
        r for r in results
        if r.get("ground_truth") == ground_truth_class
        and r.get("predicted_class") == "uncertain_investigate"
    ]
    partial = len(partial_results)

    # Average urgency score for uncertain_investigate predictions
    urgency_values = [
        r["investigation_urgency"]
        for r in results
        if r.get("predicted_class") == "uncertain_investigate"
        and r.get("investigation_urgency") is not None
    ]
    avg_urgency = round(sum(urgency_values) / len(urgency_values), 4) if urgency_values else None

    # Strict accuracy (TP + TN only, no partial credit)
    correct = tp + tn
    accuracy = correct / total if total > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn + partial) if (tp + fn + partial) > 0 else 0.0

    # Weighted metrics counting partial credit
    weighted_recall_num = tp + partial * partial_credit
    weighted_recall_den = tp + fn + partial
    weighted_recall = weighted_recall_num / weighted_recall_den if weighted_recall_den > 0 else 0.0

    weighted_score = (tp + tn + partial * partial_credit) / total if total > 0 else 0.0

    confident_misses = len(find_confident_misses(results, ground_truth_class, 0.70))

    class_distribution: dict[str, int] = {}
    for r in results:
        cls = r.get("predicted_class", "unknown")
        class_distribution[cls] = class_distribution.get(cls, 0) + 1

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "partial": partial,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "weighted_recall": round(weighted_recall, 4),
        "weighted_score": round(weighted_score, 4),
        "avg_urgency": avg_urgency,
        "confident_misses": confident_misses,
        "class_distribution": dict(sorted(class_distribution.items())),
    }
