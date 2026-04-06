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

Multi-backend:
  Set ACTIVE_MODEL to switch between Anthropic and OpenAI backends.
  Anthropic: "claude-sonnet-4-6" (default)
  OpenAI:    "gpt-4o", "o4-mini"
"""

from __future__ import annotations
import asyncio
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, Union

# ---------------------------------------------------------------------------
# KF core imports
# ---------------------------------------------------------------------------
_KF_ROOT = Path(__file__).resolve().parents[4]
if str(_KF_ROOT) not in sys.path:
    sys.path.insert(0, str(_KF_ROOT))

import core.pipeline.agents as _core_agents
from core.pipeline.agents import (  # noqa: F401
    get_cost_tracker,
    CostTracker,
)


def reset_cost_tracker() -> None:
    """Reset cost tracker and clear any OpenAI-specific cost accumulator."""
    _core_agents.reset_cost_tracker()
    tracker = _core_agents.get_cost_tracker()
    tracker._openai_cost = 0.0
    # Patch cost_usd() to return OpenAI cost when running against OpenAI
    def _cost_usd_patched(self=tracker) -> float:
        oc = getattr(self, "_openai_cost", 0.0)
        if oc > 0.0:
            return oc
        return (
            self.input_tokens          * _core_agents._PRICE_INPUT_PER_TOKEN +
            self.cache_creation_tokens * _core_agents._PRICE_CACHE_CREATION_PER_TOKEN +
            self.cache_read_tokens     * _core_agents._PRICE_CACHE_READ_PER_TOKEN +
            self.output_tokens         * _core_agents._PRICE_OUTPUT_PER_TOKEN
        )
    import types
    tracker.cost_usd = types.MethodType(lambda self: _cost_usd_patched(), tracker)

SHOW_PROMPTS = False  # harness sets this to True via agents.SHOW_PROMPTS = True

# Active model — harness sets this before running tasks.
# Supported values: any claude-* model, or "gpt-4o" / "gpt-4-turbo"
ACTIVE_MODEL: str = "claude-sonnet-4-6"
DEFAULT_MODEL: str = "claude-sonnet-4-6"  # kept for harness display compatibility

# ---------------------------------------------------------------------------
# OpenAI pricing (USD per token — verify at platform.openai.com)
# ---------------------------------------------------------------------------
_GPT4O_PRICE_INPUT_PER_TOKEN   = 2.50  / 1_000_000
_GPT4O_PRICE_OUTPUT_PER_TOKEN  = 10.00 / 1_000_000
_O4MINI_PRICE_INPUT_PER_TOKEN  = 1.10  / 1_000_000
_O4MINI_PRICE_OUTPUT_PER_TOKEN = 4.40  / 1_000_000


def _openai_pricing(model: str) -> tuple[float, float]:
    """Return (input_price, output_price) per token for the given OpenAI model."""
    if model.startswith("o4"):
        return _O4MINI_PRICE_INPUT_PER_TOKEN, _O4MINI_PRICE_OUTPUT_PER_TOKEN
    return _GPT4O_PRICE_INPUT_PER_TOKEN, _GPT4O_PRICE_OUTPUT_PER_TOKEN

_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        import openai as _openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")
        _openai_client = _openai.AsyncOpenAI(api_key=api_key)
    return _openai_client


def _anthropic_blocks_to_openai(blocks: list) -> list:
    """Convert Anthropic content blocks to OpenAI message content format."""
    result = []
    for block in blocks:
        if not isinstance(block, dict):
            result.append({"type": "text", "text": str(block)})
            continue
        if block.get("type") == "text":
            result.append({"type": "text", "text": block.get("text", "")})
        elif block.get("type") == "image":
            src = block.get("source", {})
            media_type = src.get("media_type", "image/jpeg")
            data = src.get("data", "")
            result.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{data}"},
            })
    return result


async def _call_agent_openai(
    agent_id: str,
    user_message: Union[str, list],
    system_prompt: str = "",
    model: str = "gpt-4o",
    max_tokens: int = 4096,
    max_retries: int = 5,
) -> tuple[str, int]:
    """OpenAI-backed call_agent with the same signature as the Anthropic version."""
    import openai as _openai

    client = _get_openai_client()

    if isinstance(user_message, list):
        content = _anthropic_blocks_to_openai(user_message)
    else:
        content = user_message  # plain string accepted by OpenAI

    # o4-mini (and other o-series reasoning models) reject the "system" role;
    # they accept "developer" role instead.
    is_reasoning = model.startswith("o4") or model.startswith("o1") or model.startswith("o3")
    system_role = "developer" if is_reasoning else "system"

    messages = []
    if system_prompt:
        messages.append({"role": system_role, "content": system_prompt})
    messages.append({"role": "user", "content": content})

    if SHOW_PROMPTS:
        print(f"\n=== {agent_id} ({model}) ===")
        print(f"SYSTEM: {system_prompt[:400]}")
        if isinstance(content, str):
            print(f"USER: {content[:800]}")
        else:
            parts = [b.get("text", "[image]") if b.get("type") == "text" else "[image]" for b in content]
            print(f"USER: {' '.join(parts)[:800]}")

    # Reasoning models use max_completion_tokens; others use max_tokens.
    token_limit_kwarg = "max_completion_tokens" if is_reasoning else "max_tokens"

    t0 = time.time()
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                **{token_limit_kwarg: max_tokens},
                messages=messages,
            )
            duration_ms = int((time.time() - t0) * 1000)
            text = response.choices[0].message.content or ""
            if response.usage:
                u = response.usage
                tracker = get_cost_tracker()
                # Map to Anthropic tracker — no cache tokens for OpenAI
                tracker.input_tokens  += u.prompt_tokens
                tracker.output_tokens += u.completion_tokens
                tracker.api_calls     += 1
                # Override cost_usd calculation for openai pricing
                price_in, price_out = _openai_pricing(model)
                tracker._openai_cost = getattr(tracker, "_openai_cost", 0.0)
                tracker._openai_cost += (
                    u.prompt_tokens     * price_in +
                    u.completion_tokens * price_out
                )
            return text, duration_ms
        except _openai.RateLimitError:
            if attempt < max_retries - 1:
                wait = 60 * (attempt + 1)
                print(f"  [rate-limit] {agent_id} retry {attempt+1}/{max_retries-1} in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise
        except _openai.APIStatusError as e:
            if e.status_code in (429, 503) and attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  [overloaded] {agent_id} retry {attempt+1}/{max_retries-1} in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise


def _is_openai_model(model: str) -> bool:
    return model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4")


async def call_agent(
    agent_id: str,
    user_message: Union[str, list],
    system_prompt: str = "",
    model: str = "",
    max_tokens: int = 4096,
    max_retries: int = 5,
) -> tuple[str, int]:
    """Route call to Anthropic or OpenAI backend based on ACTIVE_MODEL."""
    active = model or ACTIVE_MODEL
    if _is_openai_model(active):
        return await _call_agent_openai(
            agent_id, user_message, system_prompt,
            model=active, max_tokens=max_tokens, max_retries=max_retries,
        )
    return await _core_agents.call_agent(
        agent_id, user_message, system_prompt,
        model=active, max_tokens=max_tokens, max_retries=max_retries,
    )


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
    """Extract the first complete JSON object from LLM output and parse it.

    Handles fenced ```json ... ``` blocks and raw JSON at any nesting depth.
    """
    # Try fenced block first (Claude typically wraps in fences)
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Fallback: find the first complete JSON object using bracket counting.
    # The limited 2-level regex fails on deeply-nested responses (e.g. from
    # OpenAI models that don't fence their output).
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


# Per-pair absence checklists: fields always appended to the OBSERVER schema,
# independent of rule retrieval. Ensures the OBSERVER always reports presence/absence
# of the most discriminating markers for each pair — composite-absence rules depend on
# these being recorded even when no retrieval-matched rule mentions them.
_ABSENCE_CHECKLIST: dict[str, list[dict]] = {
    "melanoma_vs_melanocytic_nevus": [
        {"name": "blue_white_veil_present", "question": "Is a blue-white veil present (diffuse blue-white structureless area over a raised area)?", "options": ["present", "absent", "uncertain/not visible"]},
        {"name": "regression_structures_present", "question": "Are regression structures visible (white scar-like areas or blue-gray peppering)?", "options": ["present", "absent", "uncertain/not visible"]},
        {"name": "atypical_network_present", "question": "Is an atypical pigment network present (irregular meshwork with variable thickness or abrupt endings)?", "options": ["present — atypical/irregular", "present — typical/regular", "absent", "uncertain/not visible"]},
        {"name": "peripheral_dots_globules", "question": "Are dots or globules distributed asymmetrically or clustered at the periphery?", "options": ["yes — asymmetric or peripheral clustering", "no — absent or symmetrically distributed", "uncertain/not visible"]},
    ],
    "basal_cell_carcinoma_vs_benign_keratosis": [
        {"name": "arborizing_vessels_present", "question": "Are arborizing (tree-like branching) vessels visible?", "options": ["present", "absent", "uncertain/not visible"]},
        {"name": "blue_gray_ovoid_nests_present", "question": "Are blue-gray ovoid nests present?", "options": ["present", "absent — none visible", "uncertain/not visible"]},
        {"name": "leaf_like_areas_present", "question": "Are leaf-like areas or spoke-wheel structures visible?", "options": ["present", "absent", "uncertain/not visible"]},
        {"name": "milia_like_cysts_present", "question": "Are milia-like cysts visible (white or yellowish well-defined round dots)?", "options": ["present — multiple", "present — rare/single", "absent", "uncertain/not visible"]},
        {"name": "comedo_like_openings_present", "question": "Are comedo-like openings visible (dark plugged pore-like structures)?", "options": ["present — multiple", "present — rare/few", "absent", "uncertain/not visible"]},
        {"name": "cerebriform_pattern_present", "question": "Is a cerebriform (brain-like gyri and sulci) pattern present?", "options": ["present", "absent", "uncertain/not visible"]},
    ],
    "actinic_keratosis_vs_benign_keratosis": [
        {"name": "strawberry_pattern_present", "question": "Is a strawberry pattern visible (red pseudonetwork around follicular openings with white halos)?", "options": ["present", "absent", "uncertain/not visible"]},
        {"name": "dotted_vessels_erythema", "question": "Are dotted or glomerular vessels present on a pink/erythematous background?", "options": ["yes — dotted/coiled vessels on erythematous base", "no — vessels absent or background not erythematous", "uncertain/not visible"]},
        {"name": "background_erythema", "question": "Is the background clearly pink or erythematous (inflammatory redness)?", "options": ["yes — clearly pink/erythematous", "no — tan, brown, gray, or skin-colored", "uncertain/not visible"]},
        {"name": "milia_like_cysts_present", "question": "Are milia-like cysts visible (white or yellowish well-defined round dots)?", "options": ["present — multiple", "present — rare/single", "absent", "uncertain/not visible"]},
        {"name": "comedo_like_openings_present", "question": "Are comedo-like openings visible (dark plugged pore-like structures)?", "options": ["present", "absent", "uncertain/not visible"]},
        {"name": "regression_or_peppering", "question": "Are regression structures or blue-gray peppering visible?", "options": ["present", "absent", "uncertain/not visible"]},
    ],
}


def _pair_id_from_task(task: dict) -> str:
    """Derive a normalized pair_id from task class names."""
    a = task.get("class_a", "").lower().replace(" ", "_")
    b = task.get("class_b", "").lower().replace(" ", "_")
    return f"{a}_vs_{b}"


def _append_absence_checklist(schema: dict, pair_id: str) -> dict:
    """Append per-pair absence-checklist fields to schema, skipping duplicates."""
    checklist = _ABSENCE_CHECKLIST.get(pair_id, [])
    if not checklist:
        return schema
    existing_names = {f["name"] for f in schema.get("fields", [])}
    new_fields = [f for f in checklist if f["name"] not in existing_names]
    if new_fields:
        schema = dict(schema)
        schema["fields"] = list(schema.get("fields", [])) + new_fields
    return schema


async def run_schema_generator(task: dict, matched_rules: list) -> tuple[dict, int]:
    """Generate a feature observation schema for the confusable pair.

    Returns (schema_dict, duration_ms).  schema_dict has key "fields".
    On parse failure returns a minimal fallback schema.
    Always appends a per-pair absence-checklist so the OBSERVER records
    presence/absence of key markers independent of rule retrieval results.
    """
    pair_id = _pair_id_from_task(task)
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
    )

    schema = _parse_json_block(text)
    if schema and "fields" in schema:
        return _append_absence_checklist(schema, pair_id), ms

    # Fallback: minimal schema so OBSERVER can still run
    fallback = {
        "fields": [
            {"name": "symmetry", "question": "Is the lesion symmetric in shape and color?", "options": ["symmetric", "asymmetric in one axis", "asymmetric in two axes", "uncertain/not visible"]},
            {"name": "border", "question": "How is the lesion border?", "options": ["regular and smooth", "irregular or notched", "uncertain/not visible"]},
            {"name": "color_variation", "question": "How many distinct colors are present?", "options": ["1-2 colors", "3 or more colors", "uncertain/not visible"]},
            {"name": "pigment_network", "question": "Is a pigment network visible?", "options": ["typical/regular", "atypical/irregular", "absent", "uncertain/not visible"]},
            {"name": "special_structures", "question": "Are any special structures visible?", "options": ["milia-like cysts", "comedo-like openings", "arborizing vessels", "blue-white veil", "regression structures", "none visible", "uncertain/not visible"]},
        ]
    }
    return _append_absence_checklist(fallback, pair_id), ms


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
   For rules marked "HARD GATE": verify every listed pre-condition is met by
   the observed features before applying the rule. If any pre-condition is not
   confirmed, skip the rule entirely.
4. Weigh the evidence and commit to the more likely lesion type.

IMPORTANT — you MUST choose one of the two class labels. This is a strict binary \
classification task: the ground truth is always one of the two classes. \
"uncertain" is NOT an allowed output. Weak, absent, or ambiguous evidence is \
normal in dermoscopy — when evidence is weak or contradictory, lean toward the \
class with even slightly more support and output that class. You MUST commit. \
Do not abstain. Do not output "uncertain".

Output ONLY a JSON object:
{
  "label": "<class_a_name>" | "<class_b_name>",
  "confidence": 0.0,
  "reasoning": "Step-by-step chain of evidence from dermoscopic feature observations to decision.",
  "applied_rules": ["r_001", "r_002"],
  "features_used": ["field_name_1", "field_name_2"]
}

The "label" field MUST be exactly one of the two class names provided in the user message. \
Do NOT include any text outside the JSON block.
"""


def _format_rules_for_mediator(matched_rules: list) -> str:
    if not matched_rules:
        return "No rules matched for this pair."
    lines = []
    for m in matched_rules:
        r = m.rule
        condition = r.get("condition", "")
        action    = r.get("action", "")

        # Patch rules have the form "[Patch rule — pair_id] precond1; precond2; ..."
        # Format them as a hard-gate checklist so the MEDIATOR cannot apply them
        # unless ALL pre-conditions are satisfied by the observed features.
        if condition.startswith("[Patch rule"):
            bracket_end = condition.find("]")
            header = condition[: bracket_end + 1]          # "[Patch rule — pair_id]"
            rest   = condition[bracket_end + 2 :].strip()  # "precond1; precond2; ..."
            preconditions = [p.strip() for p in rest.split(";") if p.strip()]
            lines.append(f"[{m.rule_id}] {header}")
            lines.append(f"  → Action if rule applies: {action}")
            lines.append(f"  ⚠ HARD GATE — apply this rule ONLY when ALL conditions below")
            lines.append(f"    are confirmed by the feature record (skip it if ANY is absent):")
            for i, pc in enumerate(preconditions, 1):
                lines.append(f"    {i}. {pc}")
        else:
            lines.append(
                f"[{m.rule_id}] IF {condition} THEN {action} "
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
        max_tokens=4096,
    )

    decision = _parse_json_block(text)
    if decision and "label" in decision:
        label = decision["label"]
        # If the model still returned "uncertain" despite instructions, salvage from reasoning
        if label == "uncertain":
            reasoning_text = decision.get("reasoning", "") + " " + text
            for cls in (task["class_a"], task["class_b"]):
                if cls.lower() in reasoning_text.lower():
                    decision["label"] = cls
                    decision["confidence"] = 0.2
                    break
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
two class labels — "uncertain" is NOT an allowed output. If the feedback reveals the \
other class is better supported, switch to it. If the feedback is inconclusive, keep \
the original label.

Output ONLY a JSON object with the same structure as before:
{
  "label": "<class_a_name>" | "<class_b_name>",
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
        max_tokens=4096,
    )

    decision = _parse_json_block(text)
    if decision and "label" in decision:
        label = decision["label"]
        if label == "uncertain":
            reasoning_text = decision.get("reasoning", "") + " " + text
            for cls in (task["class_a"], task["class_b"]):
                if cls.lower() in reasoning_text.lower():
                    decision["label"] = cls
                    decision["confidence"] = 0.2
                    break
            else:
                # Fall back to the prior decision rather than staying uncertain
                decision["label"] = prior_decision.get("label", task["class_a"])
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
        max_tokens=2048,
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


# ---------------------------------------------------------------------------
# Dialogic patching — expert rule authoring + validation
# ---------------------------------------------------------------------------

_EXPERT_RULE_AUTHOR_SYSTEM = """\
You are a senior dermoscopy expert and knowledge engineer.

A classification model made an error on a dermoscopic image. Your job is to author
a precise visual rule that would have led to the correct diagnosis — and that will
generalize to similar cases in the future.

The rule must be:
1. Purely visual and dermoscopic (observable in a dermoscopic image only)
2. Expressed as a pre-condition + prediction: "When [pre-condition features are met],
   classify as [class]"
3. The pre-condition must be specific enough to EXCLUDE false positives — it should
   NOT apply to typical cases of the opposing class
4. Generalizable: it must describe a pattern that applies to a class of similar images,
   not just this one image

Output ONLY a JSON object:
{
  "rule": "Natural language: When [pre-condition], classify as [class].",
  "feature": "snake_case_feature_name",
  "favors": "<exact class name>",
  "confidence": "high" | "medium" | "low",
  "preconditions": [
    "Condition 1 that must hold for this rule to apply",
    "Condition 2 ...",
    ...
  ],
  "rationale": "Why this pattern distinguishes the two classes."
}
"""


async def run_expert_rule_author(
    task: dict,
    wrong_prediction: str,
    correct_label: str,
    model_reasoning: str = "",
    model: str = "claude-opus-4-6",
) -> tuple[dict, int]:
    """Call the expert VLM with a failure case and ask it to author a corrective rule.

    Args:
        task:             task dict (must have class_a, class_b, test_image_path)
        wrong_prediction: what the cheap model predicted
        correct_label:    the ground-truth label
        model_reasoning:  the cheap model's reasoning (for context)
        model:            expert VLM model identifier

    Returns:
        (candidate_rule_dict, duration_ms)
    """
    class_a = task["class_a"]
    class_b = task["class_b"]
    image_path = task["test_image_path"]

    reasoning_snippet = model_reasoning[:400] if model_reasoning else "(not available)"

    content = [
        _image_block(image_path),
        {
            "type": "text",
            "text": (
                f"Lesion pair: {class_a} vs {class_b}\n\n"
                f"Ground truth: {correct_label}\n"
                f"Model prediction: {wrong_prediction}  ← WRONG\n"
                f"Model reasoning: {reasoning_snippet}\n\n"
                "The model made an error on this image. "
                "Please author a corrective visual rule that would have led to the "
                f"correct diagnosis ('{correct_label}') and that will generalize to "
                "similar cases."
            ),
        },
    ]

    text, ms = await call_agent(
        "EXPERT_RULE_AUTHOR",
        content,
        system_prompt=_EXPERT_RULE_AUTHOR_SYSTEM,
        model=model,
        max_tokens=4096,
    )

    rule = _parse_json_block(text)
    if rule and "rule" in rule and "favors" in rule:
        rule["raw_response"] = text
        return rule, ms

    return {"rule": text, "feature": "unknown", "favors": correct_label,
            "confidence": "low", "preconditions": [], "raw_response": text}, ms


_RULE_VALIDATOR_SYSTEM = """\
You are a dermoscopy expert assessing whether a visual rule applies to a given image.

You will be shown a dermoscopic image and a candidate rule with its pre-conditions.
Your job is to answer two questions:
1. Do the rule's pre-conditions hold for this image?
2. If yes, what class would the rule predict?

Be strict about pre-conditions: only mark them as met if you can clearly observe
the required pattern. When in doubt, mark as NOT met.

Output ONLY a JSON object:
{
  "precondition_met": true | false,
  "would_predict": "<class_name>" | null,
  "observations": "Brief note on what you saw that led to this assessment."
}
"""


async def run_rule_validator_on_image(
    image_path: str,
    ground_truth: str,
    candidate_rule: dict,
    model: str = "",
) -> tuple[dict, int]:
    """Test whether a candidate rule applies to a single labeled image.

    Returns:
        ({"precondition_met": bool, "would_predict": str|None,
          "correct": bool, "ground_truth": str}, duration_ms)
    """
    rule_text = candidate_rule.get("rule", "")
    preconditions = candidate_rule.get("preconditions", [])
    favors = candidate_rule.get("favors", "")

    precond_text = "\n".join(f"  - {p}" for p in preconditions) if preconditions else "  (none specified)"

    content = [
        _image_block(image_path),
        {
            "type": "text",
            "text": (
                f"Candidate rule: {rule_text}\n\n"
                f"Pre-conditions that must ALL hold:\n{precond_text}\n\n"
                f"If pre-conditions are met, this rule predicts: {favors}\n\n"
                "Does this rule apply to this image? "
                "Answer strictly based on what you can see."
            ),
        },
    ]

    text, ms = await call_agent(
        "RULE_VALIDATOR",
        content,
        system_prompt=_RULE_VALIDATOR_SYSTEM,
        model=model or ACTIVE_MODEL,
        max_tokens=512,
    )

    result = _parse_json_block(text)
    if result and "precondition_met" in result:
        fires = result.get("precondition_met", False)
        predicted = result.get("would_predict") if fires else None
        correct = (predicted == ground_truth) if fires else True  # non-firing is not an error
        return {
            "precondition_met": fires,
            "would_predict": predicted,
            "correct": correct,
            "ground_truth": ground_truth,
            "observations": result.get("observations", ""),
        }, ms

    return {"precondition_met": False, "would_predict": None,
            "correct": True, "ground_truth": ground_truth, "observations": text}, ms


async def validate_candidate_rule(
    candidate_rule: dict,
    validation_images: list,   # list of (image_path: str, ground_truth: str)
    trigger_image_path: str,   # the failure image that triggered this rule
    trigger_correct_label: str,
    model: str = "",
) -> dict:
    """Test a candidate rule against a pool of labeled images.

    Returns a dict with TP, FP, TN, FN counts, precision, recall,
    whether the rule fires correctly on the trigger image, an accept flag,
    and the per-image case lists for contrastive analysis.
    """
    tp = fp = tn = fn = 0
    fires_on_trigger = False
    tp_cases: list[dict] = []   # {"image_path", "ground_truth", "observations"}
    fp_cases: list[dict] = []

    # Check the trigger image first
    trigger_result, _ = await run_rule_validator_on_image(
        trigger_image_path, trigger_correct_label, candidate_rule, model=model
    )
    fires_on_trigger = trigger_result["precondition_met"] and trigger_result["correct"]

    # Check validation pool
    favors = candidate_rule.get("favors", "")
    for img_path, gt in validation_images:
        res, _ = await run_rule_validator_on_image(img_path, gt, candidate_rule, model=model)
        case = {"image_path": img_path, "ground_truth": gt,
                "observations": res.get("observations", "")}
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

    # Accept if: fires on the trigger AND precision is acceptable AND not too many FP
    # fp <= 1 is the binding gate in practice — a rule with 6 TP / 2 FP fails here
    # even if precision == 0.75.
    accepted = fires_on_trigger and precision >= 0.75 and fp <= 1
    rejection_reason = (
        "did not fire on trigger"   if not fires_on_trigger else
        f"fp={fp} > 1"              if fp > 1 else
        f"precision={precision:.2f} < 0.75" if precision < 0.75 else
        None
    )

    return {
        "fires_on_trigger": fires_on_trigger,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "accepted": accepted,
        "rejection_reason": rejection_reason,
        "tp_cases": tp_cases,
        "fp_cases": fp_cases,
    }


# ---------------------------------------------------------------------------
# Contrastive feature analysis and rule revision
# ---------------------------------------------------------------------------

_CONTRASTIVE_ANALYSIS_SYSTEM = """\
You are a senior dermoscopy expert and knowledge engineer.

You will be shown a candidate rule that fires correctly on some images (TRUE POSITIVES)
but incorrectly on others (FALSE POSITIVES). Your task is to identify the single most
discriminating visual feature that distinguishes the TP images from the FP images —
i.e., a feature that is consistently present in TPs but absent in FPs, or vice versa.

This feature will be used to tighten the rule's pre-conditions.

Output ONLY a JSON object:
{
  "discriminating_feature": "snake_case_feature_name",
  "description": "Plain-language description of the feature.",
  "present_in": "tp" | "fp",
  "confidence": "high" | "medium" | "low",
  "rationale": "Why this feature distinguishes TPs from FPs."
}

If you cannot identify a reliable discriminating feature, output:
{
  "discriminating_feature": null,
  "description": "Cannot identify a reliable discriminating feature.",
  "present_in": null,
  "confidence": "low",
  "rationale": "Explanation of why the distinction cannot be reliably made."
}
"""


async def run_contrastive_feature_analysis(
    tp_cases: list[dict],
    fp_cases: list[dict],
    candidate_rule: dict,
    pair_info: dict,
    model: str = "",
) -> tuple[dict, int]:
    """Identify the visual feature that distinguishes TP from FP cases.

    Uses the validator's per-image observations (text) rather than re-running
    OBSERVER on each image, keeping cost low.

    Args:
        tp_cases:  list of {"image_path", "ground_truth", "observations"}
        fp_cases:  list of {"image_path", "ground_truth", "observations"}
        candidate_rule: the rule being analyzed
        pair_info: {"class_a", "class_b", "pair_id"}
        model:     VLM model to use

    Returns:
        (contrastive_result_dict, duration_ms)
    """
    rule_text     = candidate_rule.get("rule", "")
    preconditions = candidate_rule.get("preconditions", [])
    favors        = candidate_rule.get("favors", "")
    class_a       = pair_info.get("class_a", "")
    class_b       = pair_info.get("class_b", "")

    precond_text = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(preconditions))

    def _format_cases(cases: list[dict], label: str) -> str:
        lines = []
        for i, c in enumerate(cases, 1):
            obs = c.get("observations", "(no observations recorded)")
            lines.append(f"  [{label} {i}] Ground truth: {c['ground_truth']}\n"
                         f"    Validator observations: {obs}")
        return "\n".join(lines) if lines else "  (none)"

    tp_block = _format_cases(tp_cases, "TP")
    fp_block = _format_cases(fp_cases, "FP")

    user_msg = (
        f"Pair: {class_a} vs {class_b}\n"
        f"Rule favors: {favors}\n\n"
        f"Rule: {rule_text}\n\n"
        f"Pre-conditions:\n{precond_text}\n\n"
        f"TRUE POSITIVE cases (rule fired correctly):\n{tp_block}\n\n"
        f"FALSE POSITIVE cases (rule fired on wrong class):\n{fp_block}\n\n"
        "What single visual feature most reliably distinguishes the TP cases from the FP cases? "
        "Focus on what the validator *observed* in each case."
    )

    text, ms = await call_agent(
        "EXPERT_RULE_AUTHOR",
        user_msg,
        system_prompt=_CONTRASTIVE_ANALYSIS_SYSTEM,
        model=model or ACTIVE_MODEL,
        max_tokens=1024,
    )

    result = _parse_json_block(text)
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


_RULE_REVISER_SYSTEM = """\
You are a senior dermoscopy expert and knowledge engineer.

You have authored a rule that passes validation on true positive cases but fires
incorrectly on false positive cases. A contrastive analysis has identified a
discriminating visual feature that is present in one group but not the other.

Your task is to add ONE new pre-condition to the rule that incorporates this
discriminating feature, so the rule no longer fires on false positives.

Rules:
- Add exactly one pre-condition. Do not remove or rewrite existing ones.
- The new pre-condition must be observable in a dermoscopic image.
- It must be phrased as a positive assertion ("Feature X is present") or a
  negative assertion ("Feature Y is absent"), not as a comparison.
- It must be specific enough that the RULE_VALIDATOR can answer yes/no reliably.

Output ONLY a JSON object with the full updated rule (same schema as before,
with the new pre-condition appended to the preconditions list):
{
  "rule": "<updated natural-language rule>",
  "feature": "<snake_case_feature_name>",
  "favors": "<exact class name>",
  "confidence": "high" | "medium" | "low",
  "preconditions": ["existing 1", "existing 2", ..., "NEW pre-condition"],
  "rationale": "<updated rationale explaining the revision>",
  "revision_note": "One sentence: what was added and why."
}
"""


async def run_rule_reviser(
    candidate_rule: dict,
    contrastive_result: dict,
    tp_cases: list[dict],
    fp_cases: list[dict],
    pair_info: dict,
    model: str = "",
) -> tuple[dict, int]:
    """Propose a revised rule with one additional pre-condition.

    Args:
        candidate_rule:     the rejected rule
        contrastive_result: output of run_contrastive_feature_analysis
        tp_cases:           TP cases from validation
        fp_cases:           FP cases from validation
        pair_info:          {"class_a", "class_b", "pair_id"}
        model:              VLM model to use

    Returns:
        (revised_rule_dict, duration_ms)
    """
    rule_text          = candidate_rule.get("rule", "")
    preconditions      = candidate_rule.get("preconditions", [])
    favors             = candidate_rule.get("favors", "")
    class_a            = pair_info.get("class_a", "")
    class_b            = pair_info.get("class_b", "")
    disc_feature       = contrastive_result.get("description", "")
    present_in         = contrastive_result.get("present_in", "")
    rationale          = contrastive_result.get("rationale", "")

    precond_text = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(preconditions))

    def _fmt(cases, label):
        lines = []
        for i, c in enumerate(cases, 1):
            obs = c.get("observations", "")
            lines.append(f"  [{label} {i}] {c['ground_truth']}: {obs}")
        return "\n".join(lines) if lines else "  (none)"

    user_msg = (
        f"Pair: {class_a} vs {class_b}\n"
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

    text, ms = await call_agent(
        "EXPERT_RULE_AUTHOR",
        user_msg,
        system_prompt=_RULE_REVISER_SYSTEM,
        model=model or ACTIVE_MODEL,
        max_tokens=2048,
    )

    result = _parse_json_block(text)
    if result and "rule" in result and "preconditions" in result:
        result["raw_response"] = text
        return result, ms

    # Fall back: return original rule unchanged
    candidate_rule["raw_response"] = text
    return candidate_rule, ms


# ---------------------------------------------------------------------------
# Specificity spectrum
# ---------------------------------------------------------------------------

_SPECTRUM_SYSTEM = """\
You are a senior dermoscopy expert and knowledge engineer.

You are given a candidate rule and evidence about where it works (TRUE POSITIVES)
and where it misfires (FALSE POSITIVES). Your task is to produce FOUR versions of
the rule at different levels of specificity — from most general to most specific —
so that the tightest version that still passes a precision gate can be selected.

The four levels must all favor the same class and describe the same underlying
visual phenomenon, varying only in how many pre-conditions are required:

  Level 1 — MOST GENERAL: single essential pre-condition. The one feature that
    is most diagnostic of the favored class and most absent from FP cases.
    This should fire broadly — accept some FP risk.

  Level 2 — MODERATE: core pre-condition PLUS one supporting condition that
    begins to exclude FP cases.

  Level 3 — SPECIFIC: the original expert rule as-is (copy it unchanged).

  Level 4 — MOST SPECIFIC: the original rule PLUS one additional pre-condition
    derived from the contrastive analysis that should eliminate the observed FPs.
    This may over-tighten (low recall) but should have highest precision.

Output ONLY a JSON object with a "levels" array of exactly 4 rule objects:
{
  "levels": [
    {
      "level": 1,
      "label": "most_general",
      "rule": "When [single essential condition], classify as [class].",
      "feature": "snake_case_feature_name",
      "favors": "<exact class name>",
      "confidence": "high" | "medium" | "low",
      "preconditions": ["Single essential pre-condition"],
      "rationale": "Why this is the core diagnostic signal."
    },
    {
      "level": 2,
      "label": "moderate",
      ...
    },
    {
      "level": 3,
      "label": "original",
      ...  (copy the original rule exactly)
    },
    {
      "level": 4,
      "label": "most_specific",
      ...  (original + one tightening condition from contrastive analysis)
    }
  ]
}
"""


async def run_rule_spectrum_generator(
    candidate_rule: dict,
    tp_cases: list[dict],
    fp_cases: list[dict],
    contrastive_result: dict,
    pair_info: dict,
    model: str = "",
) -> tuple[list[dict], int]:
    """Generate four specificity variants of a candidate rule.

    Returns (list_of_rule_dicts ordered level 1→4, duration_ms).
    The list is ordered from most general to most specific so the caller
    can pick the first passing entry.
    """
    rule_text     = candidate_rule.get("rule", "")
    preconditions = candidate_rule.get("preconditions", [])
    favors        = candidate_rule.get("favors", "")
    class_a       = pair_info.get("class_a", "")
    class_b       = pair_info.get("class_b", "")
    disc_desc     = contrastive_result.get("description", "(not yet analyzed)")
    disc_present  = contrastive_result.get("present_in", "?")
    disc_rationale = contrastive_result.get("rationale", "")

    precond_text = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(preconditions))

    def _fmt(cases, label):
        lines = []
        for i, c in enumerate(cases[:4], 1):  # cap at 4 to keep prompt short
            obs = c.get("observations", "")
            lines.append(f"  [{label} {i}] {c['ground_truth']}: {obs[:120]}")
        return "\n".join(lines) if lines else "  (none)"

    user_msg = (
        f"Pair: {class_a} vs {class_b}\n"
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

    text, ms = await call_agent(
        "EXPERT_RULE_AUTHOR",
        user_msg,
        system_prompt=_SPECTRUM_SYSTEM,
        model=model or ACTIVE_MODEL,
        max_tokens=4096,
    )

    result = _parse_json_block(text)
    if result and "levels" in result:
        levels = result["levels"]
        # Ensure each level inherits favors from the original if missing
        for lv in levels:
            lv.setdefault("favors", favors)
            lv["raw_response"] = text  # shared; useful for debugging
        return levels, ms

    # Fallback: return just the original as level 3
    return [candidate_rule], ms


async def validate_candidate_rules_batch(
    rules: list[dict],
    validation_images: list,
    trigger_image_path: str,
    trigger_correct_label: str,
    model: str = "",
) -> list[dict]:
    """Validate a list of rules in parallel against the same image pool.

    Returns a list of validation result dicts (same schema as
    validate_candidate_rule), one per input rule, in the same order.
    """
    coros = [
        validate_candidate_rule(
            candidate_rule=rule,
            validation_images=validation_images,
            trigger_image_path=trigger_image_path,
            trigger_correct_label=trigger_correct_label,
            model=model,
        )
        for rule in rules
    ]
    results = await asyncio.gather(*coros)
    # validate_candidate_rule returns a plain dict (not a tuple)
    return list(results)
