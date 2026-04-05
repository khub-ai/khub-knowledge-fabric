"""
agents.py — HAM10000 (dermoscopy lesion classification) agent runners.

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
_KF_ROOT = Path(__file__).resolve().parents[4]
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
        f"Task type: Fine-grained dermoscopic lesion classification\n"
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
You are an expert dermatologist designing a structured dermoscopic observation form.
Given a confusable lesion pair and their key visual discriminators, generate a \
feature observation schema — a JSON questionnaire for a vision model to fill out \
from a single dermoscopic image.

CRITICAL: Include ONLY features that are directly visible in a dermoscopic image:
- Pigment network (typical/atypical meshwork, regularity, peripheral fade)
- Border characteristics (regular/irregular, notching, abrupt cutoff)
- Color variation (number of distinct colors, distribution, uniformity)
- Globules and dots (distribution: symmetric/asymmetric, peripheral clustering)
- Regression structures (white scar-like areas, blue-gray peppering)
- Blue-white veil (structureless blue-white area over raised lesion)
- Vascular structures (arborizing/dotted/hairpin/polymorphous vessels)
- BCC-specific structures (blue-gray ovoid nests, leaf-like areas, spoke-wheel)
- Keratosis-specific structures (milia-like cysts, comedo-like openings, cerebriform pattern)
- Symmetry (shape symmetry, color distribution symmetry across axes)

Do NOT include: patient history, symptoms, age, body location, palpation findings, \
vocalizations, habitat, or any non-visual clinical information.

Output ONLY a JSON object in this format:
{
  "fields": [
    {
      "name": "snake_case_field_name",
      "question": "What is the ... of this lesion?",
      "options": ["option_1", "option_2", "uncertain/not visible"]
    }
  ]
}

Include 6-10 fields that maximally discriminate the two lesion types dermoscopically.
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
        "Generate the feature observation schema for classifying dermoscopic images of these two lesion types."
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
            {"name": "symmetry", "question": "Is the lesion symmetric in shape and color?", "options": ["symmetric", "asymmetric in one axis", "asymmetric in two axes", "uncertain/not visible"]},
            {"name": "border", "question": "How is the lesion border?", "options": ["regular and smooth", "irregular or notched", "uncertain/not visible"]},
            {"name": "color_variation", "question": "How many distinct colors are present?", "options": ["1-2 colors", "3 or more colors", "uncertain/not visible"]},
            {"name": "pigment_network", "question": "Is a pigment network visible?", "options": ["typical/regular", "atypical/irregular", "absent", "uncertain/not visible"]},
            {"name": "special_structures", "question": "Are any special structures visible?", "options": ["milia-like cysts", "comedo-like openings", "arborizing vessels", "blue-white veil", "regression structures", "none visible", "uncertain/not visible"]},
        ]
    }, ms


# ---------------------------------------------------------------------------
# Round 1 — OBSERVER (VLM)
# ---------------------------------------------------------------------------

_OBSERVER_SYSTEM = """\
You are a careful dermoscopy observer. You will be shown a dermoscopic image of a \
skin lesion and a structured feature observation form. Your task is to fill in each \
field based ONLY on what you can directly observe in the image.

Rules:
- Assign a value from the provided options for each field.
- Assign a confidence score from 0.0 (completely invisible/uncertain) to 1.0 (clearly visible).
- If a feature is not visible, obscured, or ambiguous, set confidence to 0.0 or very low.
- Note dermoscopic features: pigment network, globules, dots, vessels, regression, \
  blue-white veil, special structures, color variation, symmetry.
- Do NOT guess lesion diagnosis yet — only report observed dermoscopic features.
- Do NOT use prior knowledge about which lesion type is more common or likely.

Output ONLY a JSON object:
{
  "features": {
    "field_name": {"value": "option_string", "confidence": 0.0},
    ...
  },
  "notes": "Any additional dermoscopic observations not captured by the form."
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
                f"Lesion pair: {task['class_a']} vs {task['class_b']}\n\n"
                f"Feature observation form:\n{schema_text}\n\n"
                "Fill in every field in the form based on what you can see in this dermoscopic image. "
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
You are an expert dermatologist making a fine-grained dermoscopic lesion classification.
You will receive a structured feature observation record (filled in from a dermoscopic \
image) and a set of expert visual discrimination rules.

Classification procedure:
1. Review each feature in the observation record.
2. Skip any feature with confidence < 0.35 — it is too unreliable to use.
   Features with confidence 0.35–0.5 may be used with reduced weight.
3. Apply the expert rules to the high-confidence features.
4. Weigh the evidence and commit to the more likely lesion type.

IMPORTANT — you must choose one of the two class labels. This is a binary classification \
task: the ground truth is always one of the two classes, never "uncertain". \
Return "uncertain" ONLY if you observe strong positive evidence for BOTH classes \
simultaneously (a genuine contradiction). Weak, absent, or ambiguous evidence is \
normal in dermoscopy — when evidence is weak, lean toward the class with even \
slightly more support. Do not abstain.

Output ONLY a JSON object:
{
  "label": "<class_a_name>" | "<class_b_name>" | "uncertain",
  "confidence": 0.0,
  "reasoning": "Step-by-step chain of evidence from dermoscopic feature observations to decision.",
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
        f"Dermoscopic feature observation record:\n{features_text}\n\n"
        f"Expert visual rules:\n{rules_text}\n\n"
        "Apply the rules to the observed dermoscopic features and return your classification decision."
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

    # Fallback: try to find a lesion type name in the response
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
You are an expert dermatologist revising a dermoscopic lesion classification after a \
consistency check revealed a hard contradiction with the initial decision.

You will receive:
- The original dermoscopic feature observation record
- The initial classification decision
- Feedback from the consistency checker naming the specific contradicting feature
- Expert visual rules

Reconsider the classification in light of the feedback. You MUST commit to one of the \
two class labels — do not return "uncertain" unless you see strong positive evidence \
for BOTH classes at the same time. If the feedback reveals the other class is better \
supported, switch to it. If the feedback is inconclusive, keep the original label.

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
        f"Dermoscopic feature observation record:\n{features_text}\n\n"
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
You are a visual consistency checker for fine-grained dermoscopic lesion classification.

You will be shown:
1. A test dermoscopic image (the image being classified)
2. A proposed label and dermoscopic feature observation record
3. A set of labeled reference images (few-shot examples of each lesion type)

Your task is to catch HARD CONTRADICTIONS only — NOT to second-guess uncertain evidence.

Dermoscopy is inherently ambiguous. Overlapping features, subtle structures, and low-confidence \
observations are NORMAL and are NOT grounds for marking a decision inconsistent. Only set \
consistent=false if you see a clear factual contradiction, such as:
- A pathognomonic feature of the OTHER class is unmistakably present (e.g., arborizing vessels \
  in a lesion labeled Benign Keratosis, or milia-like cysts in a lesion labeled Basal Cell Carcinoma)
- The test image is clearly not a dermoscopic skin lesion image at all

Do NOT set consistent=false merely because:
- The image is ambiguous or low-quality
- Features are subtle or partially visible
- You personally would have chosen differently
- The decision confidence is low

Output ONLY a JSON object:
{
  "consistent": true | false,
  "confidence": 0.0,
  "revision_signal": "Name the specific pathognomonic feature of the OTHER class that is unmistakably present, if inconsistent.",
  "notes": "Any additional dermoscopic observations."
}

When in doubt, set consistent=true.
"""


async def run_verifier(
    task: dict,
    decision: dict,
    feature_record: dict,
) -> tuple[dict, int]:
    """Check classification consistency against few-shot labeled dermoscopic images.

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
            f"\nDermoscopic feature record for the test image:\n{features_text}\n\n"
            f"Decision reasoning: {decision.get('reasoning', '')}\n\n"
            "Is this dermoscopic classification visually consistent with the reference images? "
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
You are a knowledge engineer extracting visual dermoscopic discrimination rules for \
fine-grained skin lesion classification.

You will receive:
- The confusable lesion pair being classified
- The dermoscopic feature observation record (what the VLM saw)
- The decision that was made
- The correct label (ground truth)
- Whether the decision was correct

Your task: extract 0-3 new visual rules that would help future classifications of \
this pair. Only extract rules if something meaningful can be learned.

Rules MUST be:
- Purely visual and dermoscopic (observable in a dermoscopic image)
- Lesion-type-specific (clearly favor one of the two lesion types)
- Generalizable (not just this one image)

Do NOT extract rules about:
- Patient history, symptoms, or clinical context
- Body location, age, or gender
- Palpation findings or non-visual characteristics
- Any information not visible in the dermoscopic image

Output a JSON block:
```json
{
  "rule_updates": [
    {
      "action": "new",
      "condition": "If [dermoscopic feature description] is observed in [lesion pair] classification...",
      "rule_action": "Classify as [lesion type name]",
      "tags": ["derm-ham10000"]
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
    pair_id: str = "",
) -> tuple[str, int]:
    """Extract new visual dermoscopic rules from a classified example.

    Returns (raw_text_with_rule_updates, duration_ms).
    The caller passes this to rule_engine.parse_mediator_rule_updates().
    """
    features_text = _format_feature_record(feature_record)
    outcome = "CORRECT" if is_correct else f"WRONG (predicted {decision.get('label', '?')}, actual {correct_label})"
    pid = pair_id or task.get("pair_id", "")

    user_msg = (
        f"Pair: {task['class_a']} vs {task['class_b']}\n\n"
        f"Dermoscopic feature observation record:\n{features_text}\n\n"
        f"Decision: {decision.get('label', '?')} (confidence: {decision.get('confidence', 0):.2f})\n"
        f"Reasoning: {decision.get('reasoning', '')[:400]}\n\n"
        f"Ground truth: {correct_label}\n"
        f"Outcome: {outcome}\n\n"
        "Extract 0-3 new visual dermoscopic rules that would help future classifications of this pair.\n"
        f"Use tags: [\"derm-ham10000\", \"{pid}\"]"
    )

    text, ms = await call_agent(
        "RULE_EXTRACTOR",
        user_msg,
        system_prompt=_RULE_EXTRACTOR_SYSTEM,
        max_tokens=768,
    )

    return text, ms


# ---------------------------------------------------------------------------
# Baseline — zero-shot and few-shot (no rules, no schema)
# ---------------------------------------------------------------------------

_BASELINE_ZERO_SHOT_SYSTEM = """\
You are an expert dermatologist. You will be shown a dermoscopic image.
Classify it as one of the two specified lesion types based solely on the image.

Output ONLY a JSON object:
{
  "label": "<class_a_name>" | "<class_b_name>",
  "confidence": 0.0,
  "reasoning": "Brief dermoscopic rationale."
}
"""

_BASELINE_FEW_SHOT_SYSTEM = """\
You are an expert dermatologist. You will be shown a dermoscopic test image \
followed by labeled reference images of each lesion type.
Classify the TEST IMAGE as one of the two specified lesion types.

Output ONLY a JSON object:
{
  "label": "<class_a_name>" | "<class_b_name>",
  "confidence": 0.0,
  "reasoning": "Brief dermoscopic rationale."
}
"""


async def run_baseline(
    task: dict,
    mode: str = "zero_shot",
) -> tuple[dict, int]:
    """Run a zero-shot or few-shot baseline (no rules, no schema).

    Args:
        task: task dict with class_a, class_b, test_image_path, few_shot_a/b
        mode: "zero_shot" or "few_shot"

    Returns:
        (decision_dict, duration_ms)
    """
    class_a = task["class_a"]
    class_b = task["class_b"]

    if mode == "zero_shot":
        content = [
            _image_block(task["test_image_path"]),
            {
                "type": "text",
                "text": (
                    f"Classify this dermoscopic image as either '{class_a}' or '{class_b}'.\n"
                    "Return the JSON object specified in the system prompt."
                ),
            },
        ]
        system = _BASELINE_ZERO_SHOT_SYSTEM
    else:  # few_shot
        content: list[dict] = [
            {"type": "text", "text": f"TEST IMAGE — classify as '{class_a}' or '{class_b}':"},
            _image_block(task["test_image_path"]),
        ]
        for p in task.get("few_shot_a", [])[:3]:
            content.append({"type": "text", "text": f"\nREFERENCE — {class_a}:"})
            content.append(_image_block(p))
        for p in task.get("few_shot_b", [])[:3]:
            content.append({"type": "text", "text": f"\nREFERENCE — {class_b}:"})
            content.append(_image_block(p))
        content.append({
            "type": "text",
            "text": "Classify the TEST IMAGE. Return the JSON object specified in the system prompt.",
        })
        system = _BASELINE_FEW_SHOT_SYSTEM

    text, ms = await call_agent(
        f"BASELINE_{mode.upper()}",
        content,
        system_prompt=system,
        max_tokens=512,
    )

    result = _parse_json_block(text)
    if result and "label" in result:
        return result, ms

    label = "uncertain"
    for cls in (class_a, class_b):
        if cls.lower() in text.lower():
            label = cls
            break
    return {"label": label, "confidence": 0.0, "reasoning": text}, ms
