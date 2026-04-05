"""
agents.py — UC200 (bird image classification) agent runners.

All agent runner functions are thin wrappers over call_agent() from core.
Each function:
  - Builds a domain-specific system prompt and user message
  - Calls call_agent() (text or vision)
  - Parses the LLM response into a typed dict
  - Returns (parsed_result, raw_text, duration_ms) or (parsed_result, duration_ms)

Agents in this use case:
  run_schema_generator   Round 0.5 — generate a per-pair feature observation form
  run_observer           Round 1   — VLM fills in the feature form from the image
  run_mediator_classify  Round 2   — classify based on feature record + rules
  run_mediator_revise    Round 2R  — revise classification after verifier rejection
  run_verifier           Round 3   — check decision vs few-shot labeled images
  run_rule_extractor     Post-task — extract new visual rules from success/failure

Re-exported from core (used by ensemble.py and harness.py):
  call_agent, DEFAULT_MODEL, reset_cost_tracker, get_cost_tracker
"""

from __future__ import annotations
import base64
import json
import re
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# KF core imports
# ---------------------------------------------------------------------------
_KF_ROOT = Path(__file__).resolve().parents[3]
if str(_KF_ROOT) not in sys.path:
    sys.path.insert(0, str(_KF_ROOT))

from core.pipeline.agents import (  # noqa: F401
    call_agent,
    DEFAULT_MODEL,
    reset_cost_tracker,
    get_cost_tracker,
    SHOW_PROMPTS,
)
import core.pipeline.agents as _agents_mod  # for SHOW_PROMPTS write access

SHOW_PROMPTS = False  # harness sets this to True via agents.SHOW_PROMPTS = True


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def encode_image_b64(image_path: str | Path) -> str:
    """Read an image file and return its base64-encoded string."""
    return base64.standard_b64encode(Path(image_path).read_bytes()).decode("ascii")


def _image_block(image_path: str | Path) -> dict:
    """Return an Anthropic content block for a JPEG image."""
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": encode_image_b64(image_path),
        },
    }


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def format_pair_for_prompt(task: dict) -> str:
    """Produce a short task description for rule matching and agent prompts."""
    return (
        f"Pair: {task['class_a']} vs {task['class_b']}\n"
        f"Task type: Fine-grained visual bird species classification\n"
        f"Task ID: {task.get('_task_id', task.get('pair_id', ''))}"
    )


def _parse_json_block(text: str) -> Optional[dict]:
    """Extract the first ```json ... ``` block from LLM output and parse it.

    Falls back to raw JSON parse if no fenced block is found.
    """
    # Try fenced block first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try raw JSON anywhere in the text
    match = re.search(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Round 0.5 — Schema generator
# ---------------------------------------------------------------------------

_SCHEMA_SYSTEM = """\
You are an expert ornithologist designing a structured visual observation form.
Given a confusable species pair and their key visual discriminators, generate a \
feature observation schema — a JSON questionnaire for a vision model to fill out \
from a single bird photograph.

CRITICAL: Include ONLY features that are directly visible in a static image:
- Plumage coloration and pattern
- Bill shape, color, length, curvature
- Eye color and orbital ring
- Tail shape and pattern
- Wing markings (wingbars, rufous patches, primary projection)
- Facial pattern (mask, supercilium, crown stripe)
- Body size/proportions relative to features visible in the image
- Bare parts (facial skin, leg color)

Do NOT include: vocalizations, calls, habitat, range, season, behavior, feeding, \
or size relative to other birds (unless both birds are in the same frame).

Output ONLY a JSON object in this format:
{
  "fields": [
    {
      "name": "snake_case_field_name",
      "question": "What is the ... of this bird?",
      "options": ["option_1", "option_2", "uncertain/not visible"]
    }
  ]
}

Include 6-10 fields that maximally discriminate the two species visually.
Every field MUST have "uncertain/not visible" as the last option.
"""


async def run_schema_generator(task: dict, matched_rules: list) -> tuple[dict, int]:
    """Generate a feature observation schema for the confusable pair.

    Returns (schema_dict, duration_ms).  schema_dict has key "fields".
    On parse failure returns a minimal fallback schema.
    """
    pair_desc = format_pair_for_prompt(task)
    rules_hint = ""
    if matched_rules:
        actions = "\n".join(f"- {m.rule.get('action', '')}" for m in matched_rules[:8])
        rules_hint = f"\nKnown expert rules (use to guide field selection):\n{actions}"

    user_msg = (
        f"{pair_desc}\n\n"
        f"Class A: {task['class_a']}\n"
        f"Class B: {task['class_b']}\n"
        f"{rules_hint}\n\n"
        "Generate the feature observation schema for classifying images of these two species."
    )

    text, ms = await call_agent(
        "SCHEMA_GENERATOR",
        user_msg,
        system_prompt=_SCHEMA_SYSTEM,
        max_tokens=1024,
    )

    schema = _parse_json_block(text)
    if schema and "fields" in schema:
        return schema, ms

    # Fallback: minimal schema so OBSERVER can still run
    return {
        "fields": [
            {"name": "overall_appearance", "question": "Describe the overall appearance.", "options": ["uncertain/not visible"]},
            {"name": "bill_color", "question": "What color is the bill?", "options": ["dark", "yellow", "pale", "red/orange", "uncertain/not visible"]},
            {"name": "eye_color", "question": "What color is the eye?", "options": ["dark", "pale yellow", "red", "uncertain/not visible"]},
            {"name": "tail_pattern", "question": "Describe the tail pattern.", "options": ["plain", "spotted", "banded", "uncertain/not visible"]},
            {"name": "wing_markings", "question": "Are wingbars or wing patches visible?", "options": ["none", "two wingbars", "rufous patch", "uncertain/not visible"]},
        ]
    }, ms


# ---------------------------------------------------------------------------
# Round 1 — OBSERVER (VLM)
# ---------------------------------------------------------------------------

_OBSERVER_SYSTEM = """\
You are a careful bird species observer. You will be shown a photograph of a bird \
and a structured feature observation form. Your task is to fill in each field based \
ONLY on what you can directly observe in the image.

Rules:
- Assign a value from the provided options for each field.
- Assign a confidence score from 0.0 (completely invisible/uncertain) to 1.0 (clearly visible).
- If a feature is not visible, obscured, or ambiguous, set confidence to 0.0 or very low.
- Do NOT guess species identity yet — only report observed features.
- Do NOT use prior knowledge about which species is more common or likely.

Output ONLY a JSON object:
{
  "features": {
    "field_name": {"value": "option_string", "confidence": 0.0},
    ...
  },
  "notes": "Any additional visual observations not captured by the form."
}
"""


async def run_observer(
    task: dict,
    schema: dict,
    matched_rules: list,
) -> tuple[dict, int]:
    """Call the VLM with the test image and feature schema.

    Returns (feature_record_dict, duration_ms).
    """
    schema_text = json.dumps(schema, indent=2)

    # Build content blocks: image first, then instructions
    content_blocks = [
        _image_block(task["test_image_path"]),
        {
            "type": "text",
            "text": (
                f"Species pair: {task['class_a']} vs {task['class_b']}\n\n"
                f"Feature observation form:\n{schema_text}\n\n"
                "Fill in every field in the form based on what you can see in this image. "
                "Return a JSON object with the structure shown in the system prompt."
            ),
        },
    ]

    text, ms = await call_agent(
        "OBSERVER",
        content_blocks,
        system_prompt=_OBSERVER_SYSTEM,
        max_tokens=1024,
    )

    record = _parse_json_block(text)
    if record and "features" in record:
        record["raw_response"] = text
        return record, ms

    return {"features": {}, "notes": text, "raw_response": text}, ms


# ---------------------------------------------------------------------------
# Round 2 — MEDIATOR (classify)
# ---------------------------------------------------------------------------

_MEDIATOR_SYSTEM = """\
You are an expert ornithologist making a fine-grained species classification.
You will receive a structured feature observation record (filled in from a photograph) \
and a set of expert visual discrimination rules.

Classification procedure:
1. Review each feature in the observation record.
2. Skip any feature with confidence < 0.5 — it is unreliable.
3. Apply the expert rules to the high-confidence features.
4. Weigh the evidence and choose the more likely species.
5. If evidence is genuinely insufficient (all high-confidence features are neutral), \
   return "uncertain" as the label.

Output ONLY a JSON object:
{
  "label": "<class_a_name>" | "<class_b_name>" | "uncertain",
  "confidence": 0.0,
  "reasoning": "Step-by-step chain of evidence from feature observations to decision.",
  "applied_rules": ["r_001", "r_002"],
  "features_used": ["field_name_1", "field_name_2"]
}

Do NOT include any text outside the JSON block.
"""


def _format_rules_for_mediator(matched_rules: list) -> str:
    if not matched_rules:
        return "No rules matched for this pair."
    lines = []
    for m in matched_rules:
        r = m.rule
        lines.append(
            f"[{m.rule_id}] IF {r.get('condition', '')} THEN {r.get('action', '')} "
            f"(confidence: {m.confidence})"
        )
    return "\n".join(lines)


def _format_feature_record(record: dict) -> str:
    features = record.get("features", {})
    if not features:
        return "No features recorded."
    lines = []
    for name, obs in features.items():
        val = obs.get("value", "?")
        conf = obs.get("confidence", 0.0)
        conf_label = "HIGH" if conf >= 0.7 else ("MED" if conf >= 0.5 else "LOW")
        lines.append(f"  {name}: {val!r}  [{conf_label} conf={conf:.2f}]")
    notes = record.get("notes", "")
    if notes:
        lines.append(f"  notes: {notes}")
    return "\n".join(lines)


async def run_mediator_classify(
    task: dict,
    feature_record: dict,
    matched_rules: list,
) -> tuple[dict, str, int]:
    """Classify using feature record + expert rules.

    Returns (decision_dict, raw_text, duration_ms).
    """
    rules_text = _format_rules_for_mediator(matched_rules)
    features_text = _format_feature_record(feature_record)

    user_msg = (
        f"Classify as either '{task['class_a']}' or '{task['class_b']}'.\n\n"
        f"Feature observation record:\n{features_text}\n\n"
        f"Expert visual rules:\n{rules_text}\n\n"
        "Apply the rules to the observed features and return your classification decision."
    )

    text, ms = await call_agent(
        "MEDIATOR",
        user_msg,
        system_prompt=_MEDIATOR_SYSTEM,
        max_tokens=1024,
    )

    decision = _parse_json_block(text)
    if decision and "label" in decision:
        return decision, text, ms

    # Fallback: try to find a species name in the response
    label = "uncertain"
    for cls in (task["class_a"], task["class_b"]):
        if cls.lower() in text.lower():
            label = cls
            break
    return {"label": label, "confidence": 0.0, "reasoning": text, "applied_rules": []}, text, ms


# ---------------------------------------------------------------------------
# Round 2R — MEDIATOR (revise)
# ---------------------------------------------------------------------------

_MEDIATOR_REVISE_SYSTEM = """\
You are an expert ornithologist revising a species classification after a consistency check \
revealed a problem with the initial decision.

You will receive:
- The original feature observation record
- The initial classification decision
- Feedback from the consistency checker explaining why the decision may be wrong
- Expert visual rules

Reconsider the classification in light of the feedback. You may:
- Change the label if the feedback reveals a better-supported interpretation
- Return "uncertain" if the evidence is genuinely ambiguous

Output ONLY a JSON object with the same structure as before:
{
  "label": "<class_a_name>" | "<class_b_name>" | "uncertain",
  "confidence": 0.0,
  "reasoning": "...",
  "applied_rules": [],
  "features_used": []
}
"""


async def run_mediator_revise(
    task: dict,
    feature_record: dict,
    matched_rules: list,
    prior_decision: dict,
    verifier_feedback: dict,
) -> tuple[dict, str, int]:
    """Revise the classification after verifier rejection.

    Returns (decision_dict, raw_text, duration_ms).
    """
    rules_text = _format_rules_for_mediator(matched_rules)
    features_text = _format_feature_record(feature_record)

    user_msg = (
        f"Classify as either '{task['class_a']}' or '{task['class_b']}'.\n\n"
        f"Feature observation record:\n{features_text}\n\n"
        f"Expert visual rules:\n{rules_text}\n\n"
        f"Prior decision: {json.dumps(prior_decision, indent=2)}\n\n"
        f"Consistency check feedback:\n"
        f"  Consistent: {verifier_feedback.get('consistent', '?')}\n"
        f"  Revision signal: {verifier_feedback.get('revision_signal', '')}\n"
        f"  Notes: {verifier_feedback.get('notes', '')}\n\n"
        "Revise your classification decision in light of this feedback."
    )

    text, ms = await call_agent(
        "MEDIATOR_REVISE",
        user_msg,
        system_prompt=_MEDIATOR_REVISE_SYSTEM,
        max_tokens=1024,
    )

    decision = _parse_json_block(text)
    if decision and "label" in decision:
        return decision, text, ms

    label = prior_decision.get("label", "uncertain")
    return {"label": label, "confidence": 0.0, "reasoning": text, "applied_rules": []}, text, ms


# ---------------------------------------------------------------------------
# Round 3 — VERIFIER
# ---------------------------------------------------------------------------

_VERIFIER_SYSTEM = """\
You are a visual consistency checker for fine-grained bird species classification.

You will be shown:
1. A test image (the image being classified)
2. A proposed label and feature observation record
3. A set of labeled reference images (few-shot examples of each species)

Your task is to check whether the proposed label is visually consistent with the \
reference images — NOT to reclassify from scratch.

Specifically, check:
- Does the test bird share key visual features with the reference images of the proposed label?
- Does it clearly LACK features that distinguish the other species?

Output ONLY a JSON object:
{
  "consistent": true | false,
  "confidence": 0.0,
  "revision_signal": "Explain what specific feature makes the label seem wrong, if inconsistent.",
  "notes": "Any additional observations."
}

If you are unsure, set consistent=true and note the uncertainty.
"""


async def run_verifier(
    task: dict,
    decision: dict,
    feature_record: dict,
) -> tuple[dict, int]:
    """Check classification consistency against few-shot labeled images.

    Uses few_shot_a and few_shot_b image paths from the task dict.
    Returns (verification_dict, duration_ms).
    """
    few_shot_a = task.get("few_shot_a", [])
    few_shot_b = task.get("few_shot_b", [])

    # Build content: test image, then labeled reference images
    content_blocks: list[dict] = [
        {"type": "text", "text": f"TEST IMAGE — proposed label: {decision.get('label', '?')}"},
        _image_block(task["test_image_path"]),
    ]

    if few_shot_a:
        content_blocks.append({
            "type": "text",
            "text": f"\nREFERENCE IMAGES — {task['class_a']} (Class A):",
        })
        for p in few_shot_a[:3]:
            content_blocks.append(_image_block(p))

    if few_shot_b:
        content_blocks.append({
            "type": "text",
            "text": f"\nREFERENCE IMAGES — {task['class_b']} (Class B):",
        })
        for p in few_shot_b[:3]:
            content_blocks.append(_image_block(p))

    features_text = _format_feature_record(feature_record)
    content_blocks.append({
        "type": "text",
        "text": (
            f"\nFeature record for the test image:\n{features_text}\n\n"
            f"Decision reasoning: {decision.get('reasoning', '')}\n\n"
            "Is this classification visually consistent with the reference images? "
            "Return the JSON consistency check."
        ),
    })

    text, ms = await call_agent(
        "VERIFIER",
        content_blocks,
        system_prompt=_VERIFIER_SYSTEM,
        max_tokens=512,
    )

    result = _parse_json_block(text)
    if result and "consistent" in result:
        return result, ms

    return {"consistent": True, "confidence": 0.5, "revision_signal": "", "notes": text}, ms


# ---------------------------------------------------------------------------
# Post-task — Rule extractor
# ---------------------------------------------------------------------------

_RULE_EXTRACTOR_SYSTEM = """\
You are a knowledge engineer extracting visual discrimination rules for fine-grained \
bird species classification.

You will receive:
- The confusable pair being classified
- The feature observation record (what the VLM saw)
- The decision that was made
- The correct label (ground truth)
- Whether the decision was correct

Your task: extract 0-3 new visual rules that would help future classifications of \
this pair. Only extract rules if something meaningful can be learned.

Rules MUST be:
- Purely visual (observable in a static image)
- Species-specific (clearly favor one of the two species)
- Generalizable (not just this one image)

Do NOT extract rules about:
- Vocalizations, calls, or sounds
- Habitat, range, or season
- Behavior, feeding, or flocking
- Size relative to other birds (unless a proportional feature visible in the image)

Output a JSON block:
```json
{
  "rule_updates": [
    {
      "action": "new",
      "condition": "If [visual feature description] is observed in [species pair] classification...",
      "rule_action": "Classify as [species name]",
      "tags": ["bird-uc200"]
    }
  ]
}
```

If no new rules can be extracted, return: ```json {"rule_updates": []} ```
"""


async def run_rule_extractor(
    task: dict,
    feature_record: dict,
    decision: dict,
    correct_label: str,
    is_correct: bool,
) -> tuple[str, int]:
    """Extract new visual rules from a classified example.

    Returns (raw_mediator_text_with_rule_updates, duration_ms).
    The caller passes this to rule_engine.parse_mediator_rule_updates().
    """
    features_text = _format_feature_record(feature_record)
    outcome = "CORRECT" if is_correct else f"WRONG (predicted {decision.get('label', '?')}, actual {correct_label})"

    user_msg = (
        f"Pair: {task['class_a']} vs {task['class_b']}\n\n"
        f"Feature observation record:\n{features_text}\n\n"
        f"Decision: {decision.get('label', '?')} (confidence: {decision.get('confidence', 0):.2f})\n"
        f"Reasoning: {decision.get('reasoning', '')[:400]}\n\n"
        f"Ground truth: {correct_label}\n"
        f"Outcome: {outcome}\n\n"
        "Extract 0-3 new visual rules that would help future classifications of this pair."
    )

    text, ms = await call_agent(
        "RULE_EXTRACTOR",
        user_msg,
        system_prompt=_RULE_EXTRACTOR_SYSTEM,
        max_tokens=768,
    )

    return text, ms
