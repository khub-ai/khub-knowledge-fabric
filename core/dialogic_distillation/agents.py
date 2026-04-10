"""
agents.py — Domain-independent agent runner functions for dialogic distillation.

Every function takes a `config: DomainConfig` for prompt vocabulary and a
`call_agent_fn` callable for the actual LLM call.  This decouples the
distillation protocol from any specific LLM backend (Anthropic, OpenAI,
OpenRouter, claude-tutor, etc.).

The default call_agent_fn is core.pipeline.agents.call_agent.
"""
from __future__ import annotations
import asyncio
import base64
import json
import re
from pathlib import Path
from typing import Callable, Optional, Union

from .protocols import DomainConfig
from .constants import DEFAULT_EARLY_EXIT_FP, DEFAULT_PRECISION_GATE, DEFAULT_MAX_FP
from . import prompts as _prompts


# ---------------------------------------------------------------------------
# Default call_agent — imported lazily to avoid circular import
# ---------------------------------------------------------------------------

_default_call_agent = None


def _get_default_call_agent():
    global _default_call_agent
    if _default_call_agent is None:
        from core.pipeline.agents import call_agent
        _default_call_agent = call_agent
    return _default_call_agent


# ---------------------------------------------------------------------------
# Utilities (domain-independent)
# ---------------------------------------------------------------------------

def encode_image_b64(image_path: str | Path) -> str:
    """Read an image file and return its base64-encoded string."""
    return base64.standard_b64encode(Path(image_path).read_bytes()).decode("ascii")


_MEDIA_TYPES = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".gif":  "image/gif",
    ".webp": "image/webp",
}


def image_block(image_path: str | Path) -> dict:
    """Return an Anthropic content block for an image file.

    Media type is inferred from the file extension; defaults to image/jpeg.
    """
    ext = Path(image_path).suffix.lower()
    media_type = _MEDIA_TYPES.get(ext, "image/jpeg")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": encode_image_b64(image_path),
        },
    }


def parse_json_block(text: str) -> Optional[dict]:
    """Extract the first complete JSON object from LLM output and parse it.

    Handles fenced ```json ... ``` blocks and raw JSON at any nesting depth.
    """
    # Try fenced block first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Fallback: bracket counting for deeply-nested responses
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
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
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    pass
                break
    return None


# ---------------------------------------------------------------------------
# Agent runner functions
# ---------------------------------------------------------------------------

async def run_expert_rule_author(
    task: dict,
    wrong_prediction: str,
    correct_label: str,
    config: DomainConfig,
    model_reasoning: str = "",
    model: str = "",
    prior_context: str = "",
    call_agent_fn: Callable | None = None,
) -> tuple[dict, int]:
    """Call the expert VLM with a failure case and ask it to author a corrective rule.

    task must have: class_a, class_b, test_image_path
    prior_context: optional string for multi-round dialogic exchanges.
    Returns (candidate_rule_dict, duration_ms).
    """
    _call = call_agent_fn or _get_default_call_agent()
    class_a = task["class_a"]
    class_b = task["class_b"]
    image_path = task["test_image_path"]
    reasoning_snippet = model_reasoning[:400] if model_reasoning else "(not available)"

    if prior_context:
        preamble = (
            f"You are the TUTOR in an ongoing teaching dialogue with a PUPIL VLM.\n\n"
            f"In a previous round you gave the pupil a rule, but the pupil "
            f"STILL got the {config.classification_noun} wrong. Below is a record of what happened:\n\n"
            f"{prior_context}\n\n"
            f"Your task now is to author a NEW or REVISED rule that addresses the "
            f"specific gap the pupil expressed — the feature it could not see or the "
            f"reasoning step it got wrong this time. Do not simply repeat the previous "
            f"rule. The new rule must target a DIFFERENT observable feature that is "
            f"visible in this {config.item_noun} and not covered by any prior rule."
        )
    else:
        preamble = (
            f"You are the TUTOR. A weaker PUPIL VLM just looked at this "
            f"{config.item_noun} and got the {config.classification_noun} wrong. Your job is to "
            f"(a) notice exactly where the pupil went off track, and (b) give the "
            f"pupil a corrective visual rule it can apply to this and similar future cases."
        )

    content = [
        image_block(image_path),
        {
            "type": "text",
            "text": (
                f"{preamble}\n\n"
                f"{config.class_noun.capitalize()} pair: {class_a} vs {class_b}\n"
                f"Ground truth: {correct_label}\n"
                f"Pupil's prediction: {wrong_prediction}  <- WRONG\n"
                f"Pupil's stated reasoning: {reasoning_snippet}\n\n"
                f"First, look at the {config.item_noun} carefully and diagnose the pupil's mistake:\n"
                f"  - Did the pupil hallucinate features that aren't actually visible?\n"
                f"  - Did the pupil overweight an ambiguous or shared feature?\n"
                f"  - Did the pupil miss a salient {config.feature_noun} that would have flipped "
                f"the {config.classification_noun}?\n\n"
                f"Then author a corrective visual rule that:\n"
                f"  1. Would have led to the correct {config.classification_noun} ('{correct_label}')\n"
                f"  2. Targets the specific failure mode you identified (not generic advice)\n"
                f"  3. Uses pre-conditions the pupil can actually check from the "
                f"{config.item_noun} — concrete visible features, not abstract reasoning\n"
                f"  4. Will generalize to similar cases without firing on "
                f"{class_a if correct_label != class_a else class_b}"
            ),
        },
    ]

    text, ms = await _call(
        "EXPERT_RULE_AUTHOR",
        content,
        system_prompt=_prompts.expert_rule_author_system(config),
        model=model,
        max_tokens=4096,
    )

    rule = parse_json_block(text)
    if rule and "rule" in rule and "favors" in rule:
        rule["raw_response"] = text
        return rule, ms

    return {
        "rule": text, "feature": "unknown", "favors": correct_label,
        "confidence": "low", "preconditions": [], "raw_response": text,
    }, ms


async def run_rule_validator_on_image(
    image_path: str,
    ground_truth: str,
    candidate_rule: dict,
    config: DomainConfig,
    model: str = "",
    call_agent_fn: Callable | None = None,
) -> tuple[dict, int]:
    """Test whether a candidate rule applies to a single labeled image.

    Returns (result_dict, duration_ms).
    """
    _call = call_agent_fn or _get_default_call_agent()
    rule_text = candidate_rule.get("rule", "")
    preconditions = candidate_rule.get("preconditions", [])
    favors = candidate_rule.get("favors", "")

    precond_text = (
        "\n".join(f"  - {p}" for p in preconditions)
        if preconditions else "  (none specified)"
    )

    content = [
        image_block(image_path),
        {
            "type": "text",
            "text": (
                f"Candidate rule: {rule_text}\n\n"
                f"Pre-conditions that must ALL hold:\n{precond_text}\n\n"
                f"If pre-conditions are met, this rule predicts: {favors}\n\n"
                f"Does this rule apply to this {config.item_noun}? "
                "Answer strictly based on what you can see."
            ),
        },
    ]

    text, ms = await _call(
        "RULE_VALIDATOR",
        content,
        system_prompt=_prompts.rule_validator_system(config),
        model=model,
        max_tokens=512,
    )

    result = parse_json_block(text)
    if result and "precondition_met" in result:
        fires = result.get("precondition_met", False)
        predicted = result.get("would_predict") if fires else None
        correct = (predicted == ground_truth) if fires else True
        return {
            "precondition_met": fires,
            "would_predict": predicted,
            "correct": correct,
            "ground_truth": ground_truth,
            "observations": result.get("observations", ""),
        }, ms

    return {
        "precondition_met": False, "would_predict": None,
        "correct": True, "ground_truth": ground_truth, "observations": text,
    }, ms


async def validate_candidate_rule(
    candidate_rule: dict,
    validation_images: list,
    trigger_image_path: str,
    trigger_correct_label: str,
    config: DomainConfig,
    model: str = "",
    early_exit_fp: int = DEFAULT_EARLY_EXIT_FP,
    call_agent_fn: Callable | None = None,
) -> dict:
    """Test a candidate rule against a pool of labeled images.

    Pool images are validated concurrently (asyncio.gather) for speed.
    Trigger image is checked first; pool validation is skipped if it fails.

    Returns a dict with TP, FP, TN, FN counts, precision, recall, accept flag,
    and per-image case lists for contrastive analysis.
    """
    tp_cases: list[dict] = []
    fp_cases: list[dict] = []

    # Check trigger image first — short-circuit if it doesn't fire
    trigger_result, _ = await run_rule_validator_on_image(
        trigger_image_path, trigger_correct_label, candidate_rule,
        config=config, model=model, call_agent_fn=call_agent_fn,
    )
    fires_on_trigger = (
        trigger_result["precondition_met"] and trigger_result["correct"]
    )

    if not fires_on_trigger:
        favors = candidate_rule.get("favors", "")
        fn = sum(1 for _, gt in validation_images if gt == favors)
        tn = len(validation_images) - fn
        return {
            "fires_on_trigger": False,
            "tp": 0, "fp": 0, "tn": tn, "fn": fn,
            "precision": 0.0, "recall": 0.0,
            "accepted": False, "rejection_reason": "did not fire on trigger",
            "tp_cases": [], "fp_cases": [],
            "early_exited": False,
        }

    favors = candidate_rule.get("favors", "")

    # Use per-domain gate overrides if set, otherwise fall back to global defaults
    precision_gate = (
        config.precision_gate
        if config.precision_gate is not None
        else DEFAULT_PRECISION_GATE
    )
    max_fp_gate = (
        config.max_fp
        if config.max_fp is not None
        else DEFAULT_MAX_FP
    )

    # Validate all pool images concurrently
    pool_tasks = [
        run_rule_validator_on_image(
            img_path, gt, candidate_rule,
            config=config, model=model, call_agent_fn=call_agent_fn,
        )
        for img_path, gt in validation_images
    ]
    results = await asyncio.gather(*pool_tasks)

    tp = fp = tn = fn = 0
    for (res, _), (img_path, gt) in zip(results, validation_images):
        case = {
            "image_path": img_path, "ground_truth": gt,
            "observations": res.get("observations", ""),
        }
        if res["precondition_met"]:
            if gt == favors:
                tp += 1
                tp_cases.append(case)
            else:
                fp += 1
                fp_cases.append(case)
        else:
            if gt == favors:
                fn += 1
            else:
                tn += 1

    total_fires = tp + fp
    precision = tp / total_fires if total_fires > 0 else 0.0
    total_positive = tp + fn
    recall = tp / total_positive if total_positive > 0 else 0.0

    accepted = (fires_on_trigger and precision >= precision_gate
                and fp <= max_fp_gate)

    return {
        "fires_on_trigger": fires_on_trigger,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "accepted": accepted,
        "rejection_reason": (
            f"fp={fp} > {max_fp_gate}" if fp > max_fp_gate
            else f"precision={precision:.2f} < {precision_gate}" if precision < precision_gate
            else None
        ),
        "tp_cases": tp_cases,
        "fp_cases": fp_cases,
        "early_exited": False,  # No early exit with concurrent validation
    }


async def validate_candidate_rules_batch(
    candidate_rules: list[dict],
    validation_images: list,
    trigger_image_path: str,
    trigger_correct_label: str,
    config: DomainConfig,
    model: str = "",
    call_agent_fn: Callable | None = None,
) -> list[dict]:
    """Validate a list of candidate rules (spectrum levels) against the same pool.

    Returns a list of validation result dicts, one per rule (same order).
    """
    tasks = [
        validate_candidate_rule(
            rule, validation_images, trigger_image_path,
            trigger_correct_label, config=config, model=model,
            call_agent_fn=call_agent_fn,
        )
        for rule in candidate_rules
    ]
    return list(await asyncio.gather(*tasks))


async def run_contrastive_feature_analysis(
    tp_cases: list[dict],
    fp_cases: list[dict],
    candidate_rule: dict,
    pair_info: dict,
    config: DomainConfig,
    model: str = "",
    call_agent_fn: Callable | None = None,
) -> tuple[dict, int]:
    """Identify the visual feature that best distinguishes TP from FP cases.

    Returns (contrastive_result_dict, duration_ms).
    """
    _call = call_agent_fn or _get_default_call_agent()
    rule_text = candidate_rule.get("rule", "")
    preconditions = candidate_rule.get("preconditions", [])
    favors = candidate_rule.get("favors", "")
    class_a = pair_info.get("class_a", "")
    class_b = pair_info.get("class_b", "")

    precond_text = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(preconditions))

    def _fmt_cases(cases, label):
        lines = []
        for i, c in enumerate(cases, 1):
            obs = c.get("observations", "(no observations recorded)")
            lines.append(
                f"  [{label} {i}] Ground truth: {c['ground_truth']}\n"
                f"    Validator observations: {obs}"
            )
        return "\n".join(lines) if lines else "  (none)"

    user_msg = (
        f"{config.class_noun.capitalize()} pair: {class_a} vs {class_b}\n"
        f"Rule favors: {favors}\n\n"
        f"Rule: {rule_text}\n\n"
        f"Pre-conditions:\n{precond_text}\n\n"
        f"TRUE POSITIVE cases (rule fired correctly):\n{_fmt_cases(tp_cases, 'TP')}\n\n"
        f"FALSE POSITIVE cases (rule fired on wrong {config.class_noun}):\n{_fmt_cases(fp_cases, 'FP')}\n\n"
        f"What single visual feature ({config.feature_noun}) most reliably "
        f"distinguishes the TP cases from the FP cases? "
        "Focus on what the validator *observed* in each case."
    )

    text, ms = await _call(
        "EXPERT_RULE_AUTHOR",
        user_msg,
        system_prompt=_prompts.contrastive_analysis_system(config),
        model=model,
        max_tokens=1024,
    )

    result = parse_json_block(text)
    if result and "discriminating_feature" in result:
        result["raw_response"] = text
        return result, ms

    return {
        "discriminating_feature": None,
        "description": text,
        "present_in": None,
        "confidence": "low",
        "rationale": "Parse failed.",
        "raw_response": text,
    }, ms


async def run_rule_spectrum_generator(
    candidate_rule: dict,
    tp_cases: list[dict],
    fp_cases: list[dict],
    contrastive_result: dict,
    pair_info: dict,
    config: DomainConfig,
    model: str = "",
    call_agent_fn: Callable | None = None,
) -> tuple[list[dict], int]:
    """Generate four specificity variants of a candidate rule.

    Returns (list_of_rule_dicts ordered level 1->4, duration_ms).
    """
    _call = call_agent_fn or _get_default_call_agent()
    rule_text = candidate_rule.get("rule", "")
    preconditions = candidate_rule.get("preconditions", [])
    favors = candidate_rule.get("favors", "")
    class_a = pair_info.get("class_a", "")
    class_b = pair_info.get("class_b", "")
    disc_desc = contrastive_result.get("description", "(not yet analyzed)")
    disc_present = contrastive_result.get("present_in", "?")
    disc_rationale = contrastive_result.get("rationale", "")

    precond_text = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(preconditions))

    def _fmt(cases, label):
        lines = []
        for i, c in enumerate(cases[:4], 1):
            obs = c.get("observations", "")
            lines.append(f"  [{label} {i}] {c['ground_truth']}: {obs[:120]}")
        return "\n".join(lines) if lines else "  (none)"

    user_msg = (
        f"{config.class_noun.capitalize()} pair: {class_a} vs {class_b}\n"
        f"Rule favors: {favors}\n\n"
        f"Original rule: {rule_text}\n\n"
        f"Original pre-conditions:\n{precond_text}\n\n"
        f"Contrastive analysis — discriminating feature:\n"
        f"  {disc_desc} (present in {disc_present} cases)\n"
        f"  Rationale: {disc_rationale}\n\n"
        f"TRUE POSITIVE observations (rule fired correctly):\n{_fmt(tp_cases, 'TP')}\n\n"
        f"FALSE POSITIVE observations (rule misfired):\n{_fmt(fp_cases, 'FP')}\n\n"
        "Please produce the four-level specificity spectrum as described."
    )

    text, ms = await _call(
        "EXPERT_RULE_AUTHOR",
        user_msg,
        system_prompt=_prompts.spectrum_system(config),
        model=model,
        max_tokens=4096,
    )

    result = parse_json_block(text)
    if result and "levels" in result:
        levels = result["levels"]
        for lv in levels:
            lv.setdefault("favors", favors)
            lv["raw_response"] = text
        return levels, ms

    return [candidate_rule], ms


async def run_rule_completer(
    candidate_rule: dict,
    pair_info: dict,
    config: DomainConfig,
    model: str = "",
    call_agent_fn: Callable | None = None,
) -> tuple[dict, int]:
    """Enrich a candidate rule with implicit background conditions the expert omitted.

    Text-only — no images. Returns (completed_rule_dict, duration_ms).
    """
    _call = call_agent_fn or _get_default_call_agent()
    rule_text = candidate_rule.get("rule", "")
    preconditions = candidate_rule.get("preconditions", [])
    favors = candidate_rule.get("favors", "")
    class_a = pair_info.get("class_a", "")
    class_b = pair_info.get("class_b", "")
    other_class = class_b if favors == class_a else class_a

    precond_text = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(preconditions))

    user_msg = (
        f"Favored {config.class_noun}: {favors}\n"
        f"Confusable {config.class_noun}: {other_class}\n\n"
        f"Rule:\n{rule_text}\n\n"
        f"Existing pre-conditions ({len(preconditions)}):\n"
        f"{precond_text if precond_text else '  (none)'}\n\n"
        "Please complete the rule by adding any implicit background conditions "
        "the expert assumed but did not state."
    )

    text, ms = await _call(
        "RULE_COMPLETER",
        user_msg,
        system_prompt=_prompts.rule_completer_system(config),
        model=model,
        max_tokens=2048,
    )

    result = parse_json_block(text)
    if result and "preconditions" in result:
        completed = {**candidate_rule, **result}
        completed.setdefault("added_preconditions", [])
        completed.setdefault("completion_rationale", "")
        return completed, ms

    return {
        **candidate_rule,
        "added_preconditions": [],
        "completion_rationale": f"(parse error — raw: {text[:200]})",
    }, ms


async def run_semantic_rule_validator(
    candidate_rule: dict,
    pair_info: dict,
    config: DomainConfig,
    model: str = "",
    call_agent_fn: Callable | None = None,
) -> tuple[dict, int]:
    """Text-only semantic check of a candidate rule's domain logic.

    Returns (semantic_result_dict, duration_ms).
    """
    _call = call_agent_fn or _get_default_call_agent()
    rule_text = candidate_rule.get("rule", "")
    preconditions = candidate_rule.get("preconditions", [])
    favors = candidate_rule.get("favors", "")
    class_a = pair_info.get("class_a", "")
    class_b = pair_info.get("class_b", "")
    other_class = class_b if favors == class_a else class_a

    precond_text = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(preconditions))

    user_msg = (
        f"Favored {config.class_noun}: {favors}\n"
        f"Confusable {config.class_noun}: {other_class}\n\n"
        f"Rule:\n{rule_text}\n\n"
        f"Pre-conditions:\n{precond_text}\n\n"
        "Please evaluate each pre-condition and provide an overall recommendation."
    )

    text, ms = await _call(
        "SEMANTIC_RULE_VALIDATOR",
        user_msg,
        system_prompt=_prompts.semantic_validator_system(config),
        model=model,
        max_tokens=2048,
    )

    result = parse_json_block(text)
    if result and "overall" in result:
        return result, ms

    return {
        "precondition_ratings": [],
        "overall": "accept",
        "rationale": f"(parse error — raw: {text[:200]})",
    }, ms


async def run_rule_reviser(
    candidate_rule: dict,
    contrastive_result: dict,
    tp_cases: list[dict],
    fp_cases: list[dict],
    pair_info: dict,
    config: DomainConfig,
    model: str = "",
    call_agent_fn: Callable | None = None,
) -> tuple[dict, int]:
    """Propose a revised rule with one additional pre-condition.

    Returns (revised_rule_dict, duration_ms).
    """
    _call = call_agent_fn or _get_default_call_agent()
    rule_text = candidate_rule.get("rule", "")
    preconditions = candidate_rule.get("preconditions", [])
    favors = candidate_rule.get("favors", "")
    class_a = pair_info.get("class_a", "")
    class_b = pair_info.get("class_b", "")
    disc_feature = contrastive_result.get("description", "")
    present_in = contrastive_result.get("present_in", "")
    rationale = contrastive_result.get("rationale", "")

    precond_text = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(preconditions))

    def _fmt(cases, label):
        lines = []
        for i, c in enumerate(cases, 1):
            obs = c.get("observations", "")
            lines.append(f"  [{label} {i}] {c['ground_truth']}: {obs}")
        return "\n".join(lines) if lines else "  (none)"

    user_msg = (
        f"{config.class_noun.capitalize()} pair: {class_a} vs {class_b}\n"
        f"Rule favors: {favors}\n\n"
        f"Current rule: {rule_text}\n\n"
        f"Current pre-conditions:\n{precond_text}\n\n"
        f"Discriminating feature identified by contrastive analysis:\n"
        f"  Feature: {disc_feature}\n"
        f"  Present in: {present_in} cases\n"
        f"  Rationale: {rationale}\n\n"
        f"TRUE POSITIVE observations:\n{_fmt(tp_cases, 'TP')}\n\n"
        f"FALSE POSITIVE observations:\n{_fmt(fp_cases, 'FP')}\n\n"
        "Please add one pre-condition to the rule that incorporates the "
        "discriminating feature and will prevent the rule from firing on the "
        "false positive cases."
    )

    text, ms = await _call(
        "EXPERT_RULE_AUTHOR",
        user_msg,
        system_prompt=_prompts.rule_reviser_system(config),
        model=model,
        max_tokens=2048,
    )

    result = parse_json_block(text)
    if result and "rule" in result and "preconditions" in result:
        result["raw_response"] = text
        return result, ms

    candidate_rule["raw_response"] = text
    return candidate_rule, ms
