"""
agents.py — N-way agent runners for the KF dermatology-multiclass use case.

All 7 HAM10000 classes are in scope simultaneously. The core pipeline is
identical to the 2-way version (Observer → Mediator → Verifier) but every
prompt is generalised from "class A vs class B" to "one of N classes".

Key differences from 2-way agents.py:
  - task dict uses `categories` (list) and `category_set_id` (str) instead
    of `class_a`/`class_b`/`pair_id`.
  - task dict uses `few_shot` (dict: class_name → list[path]) instead of
    `few_shot_a` / `few_shot_b`.
  - Rules carry an optional `contra` list of class names they rule out;
    this is surfaced prominently in Mediator prompts.
  - Schema is generated for all N classes at once and cached under
    `{category_set_id}_schema`.
  - Verifier receives 1 reference image per class (up to N images total).
  - Mediator must pick exactly one of the N class names.

Backend infrastructure (Anthropic / OpenAI / OpenRouter / claude-tutor) is
re-imported from core.pipeline.agents and the 2-way agents module so it is
not duplicated here.  Only the N-way domain logic lives in this file.
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
from typing import Optional

# ---------------------------------------------------------------------------
# KF core + 2-way backend — imported, not duplicated
# ---------------------------------------------------------------------------
_HERE       = Path(__file__).resolve().parent
_KF_ROOT    = _HERE.parents[4]
_DERM2_PY   = _HERE.parents[1] / "dermatology" / "python"

# _HERE and _KF_ROOT get priority (inserted at front).
# _DERM2_PY is appended (lowest priority) so it never shadows local modules
# (dataset.py, domain_config.py) with the 2-way versions of the same file.
for _p in (str(_KF_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_DERM2_PY) not in sys.path:
    sys.path.append(str(_DERM2_PY))

# Import the entire 2-way backend under an alias to avoid name collision.
# We re-export what callers of this module need.
import importlib.util as _ilu

def _load_derm2_agents():
    spec = _ilu.spec_from_file_location("_derm2_agents", _DERM2_PY / "agents.py")
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_d2 = _load_derm2_agents()

# Re-export backend symbols callers expect on this module
call_agent        = _d2.call_agent
reset_cost_tracker = _d2.reset_cost_tracker
get_cost_tracker  = _d2.get_cost_tracker
ACTIVE_MODEL      = _d2.ACTIVE_MODEL
DEFAULT_MODEL     = _d2.DEFAULT_MODEL
SHOW_PROMPTS      = _d2.SHOW_PROMPTS
_image_block      = _d2._image_block
_parse_json_block = _d2._parse_json_block
_format_feature_record = _d2._format_feature_record

# Allow harness to set these (mirrors 2-way interface)
import types as _types

def _set_active_model(val: str) -> None:
    _d2.ACTIVE_MODEL = val

def _set_default_model(val: str) -> None:
    _d2.DEFAULT_MODEL = val

def _set_show_prompts(val: bool) -> None:
    _d2.SHOW_PROMPTS = val

# ---------------------------------------------------------------------------
# Dataset imports
# ---------------------------------------------------------------------------
from dataset import (
    CATEGORY_SET_ID,
    CATEGORY_NAMES,
    DX_TO_NAME,
    NAME_TO_DX,
)


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def format_categories_for_prompt(task: dict) -> str:
    """Short header listing all N candidate classes for use in agent prompts."""
    cats = task.get("categories", CATEGORY_NAMES)
    cats_str = ", ".join(cats)
    return (
        f"Category set: {task.get('category_set_id', CATEGORY_SET_ID)}\n"
        f"Candidates ({len(cats)}): {cats_str}\n"
        f"Task type: Fine-grained dermoscopic lesion classification (N-way)\n"
        f"Task ID: {task.get('_task_id', '')}"
    )


# ---------------------------------------------------------------------------
# Round 0.5 — N-way Schema Generator
# ---------------------------------------------------------------------------

_SCHEMA_SYSTEM_NWAY = """\
You are an expert dermatologist designing a structured dermoscopic observation form.
Given a set of N lesion classes and their key visual discriminators, generate a \
feature observation schema — a JSON questionnaire for a vision model to fill out \
from a single dermoscopic image.

The form must be able to help distinguish between ALL provided classes, not just \
any specific pair.

CRITICAL: Include ONLY features that are directly visible in a dermoscopic image:
- Pigment network (typical/atypical meshwork, regularity, peripheral fade)
- Border characteristics (regular/irregular, notching, abrupt cutoff)
- Color variation (number of distinct colors, distribution, uniformity)
- Globules and dots (distribution: symmetric/asymmetric, peripheral clustering)
- Regression structures (white scar-like areas, blue-gray peppering)
- Blue-white veil (structureless blue-white area over raised lesion)
- Vascular structures (arborizing/dotted/hairpin/coiled/polymorphous vessels)
- BCC-specific: blue-gray ovoid nests, leaf-like areas, spoke-wheel structures
- Keratosis-specific: milia-like cysts, comedo-like openings, cerebriform pattern
- AK-specific: strawberry pattern, dotted vessels on erythematous base
- Vascular lesion-specific: red/purple lacunae, blood vessel lagoons
- Dermatofibroma-specific: central white scar-like patch, peripheral light-brown ring

Do NOT include: patient history, symptoms, age, body location, palpation findings.

Output ONLY a JSON object:
{
  "fields": [
    {
      "name": "snake_case_field_name",
      "question": "What is the ... of this lesion?",
      "options": ["option_1", "option_2", "uncertain/not visible"]
    }
  ]
}

Include 12-16 fields that maximally discriminate across ALL provided lesion classes.
Every field MUST have "uncertain/not visible" as the last option.
"""

# Comprehensive N-way absence checklist — covers all 7 class discriminators.
# These fields are always appended to ensure the Observer consistently records
# presence/absence of every key marker regardless of which rules fired.
_ABSENCE_CHECKLIST_NWAY: list[dict] = [
    # Melanoma markers
    {"name": "blue_white_veil",
     "question": "Is a blue-white veil present (diffuse blue-white structureless area over a raised portion)?",
     "options": ["present", "absent", "uncertain/not visible"]},
    {"name": "regression_structures",
     "question": "Are regression structures visible (white scar-like areas or blue-gray peppering)?",
     "options": ["present", "absent", "uncertain/not visible"]},
    {"name": "pigment_network_character",
     "question": "How is the pigment network characterized?",
     "options": ["atypical/irregular (variable thickness, branching, abrupt endings)",
                 "typical/regular (uniform meshwork, smooth fade)",
                 "absent", "uncertain/not visible"]},
    {"name": "peripheral_dots_globules",
     "question": "Are dots or globules present at the periphery in an asymmetric distribution?",
     "options": ["yes — asymmetric or peripheral clustering",
                 "no — absent or symmetrically distributed", "uncertain/not visible"]},
    # BCC markers
    {"name": "arborizing_vessels",
     "question": "Are arborizing (tree-like branching) vessels visible?",
     "options": ["present", "absent", "uncertain/not visible"]},
    {"name": "blue_gray_ovoid_nests",
     "question": "Are blue-gray ovoid nests or large blue-gray blotches present?",
     "options": ["present", "absent", "uncertain/not visible"]},
    {"name": "leaf_like_spoke_wheel",
     "question": "Are leaf-like areas or spoke-wheel structures visible?",
     "options": ["present", "absent", "uncertain/not visible"]},
    # Keratosis markers (BKL / AK shared)
    {"name": "milia_like_cysts",
     "question": "Are milia-like cysts visible (white/yellowish well-defined round structures)?",
     "options": ["present — multiple", "present — rare/single", "absent", "uncertain/not visible"]},
    {"name": "comedo_like_openings",
     "question": "Are comedo-like openings visible (dark plugged pore-like structures)?",
     "options": ["present — multiple", "present — rare/few", "absent", "uncertain/not visible"]},
    # AK-specific
    {"name": "strawberry_pattern",
     "question": "Is a strawberry pattern visible (red pseudonetwork with follicular white halos)?",
     "options": ["present", "absent", "uncertain/not visible"]},
    {"name": "erythematous_background",
     "question": "Is the lesion background clearly pink or erythematous?",
     "options": ["yes — clearly pink/erythematous", "no — tan, brown, gray, or skin-toned",
                 "uncertain/not visible"]},
    # Vascular lesion markers
    {"name": "red_purple_lacunae",
     "question": "Are red or purple sharply demarcated round/oval lacunae visible?",
     "options": ["present", "absent", "uncertain/not visible"]},
    # Dermatofibroma markers
    {"name": "central_white_scar",
     "question": "Is a central white scar-like area present surrounded by a peripheral pigment ring?",
     "options": ["present", "absent", "uncertain/not visible"]},
]


def _is_fine_schema(task: dict) -> bool:
    """Return True when the task is a fine-class (7-way or L2) schema.

    Returns False for coarse L1 group schemas (category_set_id starts with
    'derm_h_level1') so the fine-class absence checklist and visual reference
    are NOT injected into group-level observation prompts.
    """
    cset = task.get("category_set_id", "")
    return not cset.startswith("derm_h_level1")


# ---------------------------------------------------------------------------
# Level-1 group schema — hand-authored per-group feature brief
# ---------------------------------------------------------------------------

# Explicit per-group visual definitions injected into the L1 schema-generator
# user message so the generated schema knows WHAT each group contains and
# WHICH dermoscopic features define each group boundary.
_L1_GROUP_BRIEFS = """\
You are generating a feature schema for COARSE GROUP classification, not fine-class
classification.  Each group may contain multiple fine classes.  The schema must
capture the key visual signals that separate these 5 groups from each other.

Group definitions and their primary dermoscopic discriminators:

  Melanocytic (contains: Melanoma, Melanocytic Nevus)
    - Pigment network present (brown meshwork of lines)
    - Melanocytic architecture: dots, globules, or streaks within pigmented areas
    - May show blue-white veil, regression (white scar zones), atypical network
    - Key signal: presence of any organized pigment network or melanocytic dots/globules

  Keratosis-type (contains: Benign Keratosis, Actinic Keratosis)
    - Absent or markedly reduced true pigment network
    - Surface features: milia-like cysts (bright white round dots), comedo-like
      openings (dark plugged pores), cerebriform/warty/stuck-on texture
    - OR: erythematous (pink/red) background with dotted vessels or diffuse redness
    - Key signal: keratotic surface structures OR erythematous base without lacunae

  Basal Cell Carcinoma
    - Arborizing (tree-like branching) bright red vessels on pearly background
    - Blue-gray ovoid nests or large gray-blue structureless blobs
    - Leaf-like areas or spoke-wheel pigmented structures
    - Key signal: arborizing vessels and/or blue-gray nests — NO pigment network

  Vascular Lesion (contains: Haemangioma, Angiokeratoma)
    - Sharply demarcated red, dark-red, or purple round/oval lacunae (blood-filled spaces)
    - No pigment network; overall reddish/purplish colour with lacunar pattern
    - Key signal: clearly visible lacunae (like blood bubbles)

  Dermatofibroma
    - Central white or ivory scar-like structureless area (depressed, fibrous)
    - Peripheral delicate light-brown pigment ring or network surrounding the centre
    - Symmetric shape; no atypical pigment network or regression
    - Key signal: central white scar patch + surrounding peripheral pigment ring

Generate 10-14 fields that maximally separate these 5 groups using the above signals.
"""

# Hard-wired L1 absence checklist — one decisive field per group boundary.
# These are appended regardless of what the schema generator produces, ensuring
# the Observer always records the primary discriminating signal for every group.
_L1_ABSENCE_CHECKLIST: list[dict] = [
    {"name": "pigment_network_present",
     "question": "Is a pigment network (brown meshwork of lines) visible anywhere in the lesion?",
     "options": ["clearly present", "faint or partial", "absent", "uncertain/not visible"]},
    {"name": "arborizing_vessels_l1",
     "question": "Are arborizing (bright-red tree-like branching) vessels visible?",
     "options": ["present", "absent", "uncertain/not visible"]},
    {"name": "blue_gray_nests_l1",
     "question": "Are blue-gray ovoid nests or large structureless blue-gray blobs present?",
     "options": ["present", "absent", "uncertain/not visible"]},
    {"name": "red_purple_lacunae_l1",
     "question": "Are sharply demarcated red or purple round/oval lacunae (blood-filled spaces) visible?",
     "options": ["present", "absent", "uncertain/not visible"]},
    {"name": "keratotic_surface_l1",
     "question": "Are keratotic surface structures visible (milia-like cysts, comedo openings, or warty/stuck-on texture)?",
     "options": ["present", "absent", "uncertain/not visible"]},
    {"name": "erythematous_base_l1",
     "question": "Is the overall lesion background clearly erythematous (pink or red) rather than brown or pigmented?",
     "options": ["yes — clearly pink/red base", "no — brown, gray, or pigmented", "uncertain/not visible"]},
    {"name": "central_white_patch_l1",
     "question": "Is there a central white or ivory scar-like structureless patch (NOT regression peppering)?",
     "options": ["present with peripheral pigment ring", "present without peripheral ring",
                 "absent", "uncertain/not visible"]},
]


def _merge_l1_checklist(schema: dict) -> dict:
    """Append L1 group-boundary checklist fields, skipping duplicates."""
    existing = {f["name"] for f in schema.get("fields", [])}
    new_fields = [f for f in _L1_ABSENCE_CHECKLIST if f["name"] not in existing]
    if new_fields:
        schema = dict(schema)
        schema["fields"] = list(schema.get("fields", [])) + new_fields
    return schema


async def run_schema_generator(task: dict, matched_rules: list) -> tuple[dict, int]:
    """Generate a feature observation schema for the given category set.

    For fine-class schemas (7-way or L2 sub-problems) the N-way absence
    checklist is appended so all key class markers are consistently recorded.

    For coarse L1 group schemas:
      - The user message is replaced with a group-aware brief (_L1_GROUP_BRIEFS)
        that tells the schema generator exactly what each group contains and
        which features define each group boundary.
      - The L1 absence checklist (_L1_ABSENCE_CHECKLIST) is appended instead
        of the fine-class checklist, ensuring one decisive field per group.

    Returns (schema_dict, duration_ms).
    """
    fine = _is_fine_schema(task)
    cats = task.get("categories", CATEGORY_NAMES)

    if fine:
        # Fine-class or L2 sub-problem — original behaviour
        cats_str = "\n".join(f"  - {c}" for c in cats)
        rules_hint = ""
        if matched_rules:
            actions = "\n".join(f"- {m.rule.get('action', '')}" for m in matched_rules[:8])
            rules_hint = f"\nKnown expert rules (use to guide field selection):\n{actions}"
        user_msg = (
            f"Generate a dermoscopic feature observation schema for classifying images "
            f"into ONE of the following {len(cats)} lesion classes:\n{cats_str}\n"
            f"{rules_hint}\n\n"
            "The schema must surface features that discriminate across ALL classes simultaneously."
        )
    else:
        # L1 coarse group schema — use the hand-authored group brief
        user_msg = _L1_GROUP_BRIEFS

    text, ms = await call_agent(
        "SCHEMA_GENERATOR",
        user_msg,
        system_prompt=_SCHEMA_SYSTEM_NWAY,
    )

    schema = _parse_json_block(text)
    if schema and "fields" in schema:
        if fine:
            return _merge_absence_checklist(schema), ms
        else:
            return _merge_l1_checklist(schema), ms

    # Fallback: minimal schema
    fallback = {"fields": [
        {"name": "symmetry",        "question": "Is the lesion symmetric in shape and color distribution?",
         "options": ["symmetric", "asymmetric in one axis", "asymmetric in two axes", "uncertain/not visible"]},
        {"name": "border",          "question": "How is the lesion border?",
         "options": ["regular and smooth", "irregular or notched", "uncertain/not visible"]},
        {"name": "color_count",     "question": "How many distinct colors are present?",
         "options": ["1 color", "2 colors", "3 or more colors", "uncertain/not visible"]},
        {"name": "pigment_network", "question": "Is a pigment network visible?",
         "options": ["typical/regular", "atypical/irregular", "absent", "uncertain/not visible"]},
        {"name": "vascular_pattern","question": "What vascular pattern is visible?",
         "options": ["arborizing", "dotted/coiled", "hairpin", "polymorphous", "none visible",
                     "uncertain/not visible"]},
        {"name": "special_structure","question": "Which special structure is most prominent?",
         "options": ["milia-like cysts", "comedo-like openings", "blue-gray ovoid nests",
                     "blue-white veil", "regression structures", "red/purple lacunae",
                     "central white scar", "none", "uncertain/not visible"]},
    ]}
    if fine:
        return _merge_absence_checklist(fallback), ms
    else:
        return _merge_l1_checklist(fallback), ms


def _merge_absence_checklist(schema: dict) -> dict:
    """Append N-way absence-checklist fields, skipping any already present.

    Only called for fine-class schemas (see _is_fine_schema).
    """
    existing = {f["name"] for f in schema.get("fields", [])}
    new_fields = [f for f in _ABSENCE_CHECKLIST_NWAY if f["name"] not in existing]
    if new_fields:
        schema = dict(schema)
        schema["fields"] = list(schema.get("fields", [])) + new_fields
    return schema


# ---------------------------------------------------------------------------
# Round 1 — OBSERVER (unchanged — already class-agnostic)
# ---------------------------------------------------------------------------

_OBSERVER_SYSTEM = """\
You are a careful dermoscopy observer. You will be shown a dermoscopic image of a \
skin lesion and a structured feature observation form. Fill in each field based ONLY \
on what you can directly observe in the image.

Rules:
- Assign a value from the provided options for each field.
- Assign a confidence score from 0.0 (completely invisible/uncertain) to 1.0 (clearly visible).
- If a feature is not visible, obscured, or ambiguous, set confidence low (< 0.35).
- Report dermoscopic features only: pigment network, globules, dots, vessels, regression,
  blue-white veil, special structures, color variation, symmetry.
- Do NOT guess the diagnosis — only report observed features.

Output ONLY a JSON object:
{
  "features": {
    "field_name": {"value": "option_string", "confidence": 0.0},
    ...
  },
  "notes": "Any additional dermoscopic observations not captured by the form."
}
"""

# Visual reference injected into the Observer user message so the VLM knows
# exactly what each special structure looks like before filling in the form.
# This is pure perceptual grounding — no diagnostic steering.
_OBSERVER_VISUAL_REFERENCE = """\
## Visual feature reference — what to actively look for

KERATOSIS markers (milia-like cysts / comedo-like openings):
  • Milia-like cysts: bright white or yellowish, sharply defined round structures 0.2–1 mm,
    scattered within the lesion — like tiny trapped pearls under the skin surface.
  • Comedo-like openings: dark brown-black circular or irregular plugged pore-like depressions,
    resembling blackheads embedded in the lesion.
  • Cerebriform/fingerprint pattern: convoluted ridge-and-furrow surface texture.
  • Stuck-on / verrucous appearance: rough or warty surface, well-demarcated border.

ACTINIC KERATOSIS markers (strawberry pattern):
  • Strawberry pattern: diffuse pinkish-red background ("pseudonetwork") with white halos
    surrounding follicular openings — looks like a dotted red lattice on a pink ground.
  • Surface scale: white or yellowish flaky or rough surface overlying the lesion.
  • Erythematous background: clearly pink or red lesion colour distinct from surrounding skin.

BASAL CELL CARCINOMA markers:
  • Arborizing vessels: bright red tree-like branching vessels in sharp focus on a
    pearly white/translucent background — like a river delta in red.
  • Blue-gray ovoid nests: structureless rounded or oval blue-gray to gray-white blobs
    within the lesion, larger than globules.
  • Leaf-like areas: bulbous brown-gray projections at the lesion periphery, like maple leaves.
  • Spoke-wheel structures: pigmented spokes radiating from a central hub.

DERMATOFIBROMA markers:
  • Central white scar-like patch: central stellate or irregular white/shiny fibrous zone,
    often depressed, surrounded by pigmentation — like a scar in the lesion centre.
  • Peripheral delicate pigment network: fine, light-brown regular network ring at the
    outer edge framing the central white area.

VASCULAR LESION markers (haemangioma / angiokeratoma):
  • Lacunae: sharply defined round or oval spaces filled with red, dark-red, purple, or
    near-black colour — like blood-filled compartments or bubbles.
  • No pigment network elsewhere: the lesion lacks any brown pigment meshwork.

MELANOMA / HIGH-RISK markers:
  • Blue-white veil: confluent, hazy blue-white zone over a raised area — like frosted glass.
  • Regression structures: white scar-like structureless zones (depigmented) OR
    blue-gray peppering (fine slate-gray granules or dots).
  • Atypical pigment network: thickened, branched, or irregularly distributed meshwork
    with abrupt endings at the periphery.
  • Irregular streaks / pseudopods: radial projections at the border.

MELANOCYTIC NEVUS markers:
  • Regular/typical pigment network: uniform meshwork with smooth peripheral fade.
  • Symmetric structure: even colour and shape distribution around the centre.
  • Regular globules: uniform dots/globules evenly distributed or forming a cobblestone pattern.
"""


async def run_observer(
    task: dict,
    schema: dict,
    matched_rules: list,
) -> tuple[dict, int]:
    """VLM fills in the feature schema from the test image.

    Returns (feature_record_dict, duration_ms).
    """
    schema_text = json.dumps(schema, indent=2)
    cats_str    = ", ".join(task.get("categories", CATEGORY_NAMES))

    # Inject the visual feature reference only for fine-class schemas (7-way
    # or L2 sub-problems).  For coarse L1 group classification the reference
    # contains fine-class feature descriptions (lacunae, central white scar,
    # milia cysts …) that bias the Observer toward incorrect fine-class
    # predictions before the group has even been determined.
    visual_ref_block = (
        f"{_OBSERVER_VISUAL_REFERENCE}\n\n"
        "Carefully examine the image. Use the visual reference above to actively "
        "search for each special structure before filling in the form.\n\n"
    ) if _is_fine_schema(task) else ""

    content_blocks = [
        _image_block(task["test_image_path"]),
        {
            "type": "text",
            "text": (
                f"Candidate classes: {cats_str}\n\n"
                f"{visual_ref_block}"
                f"Feature observation form:\n{schema_text}\n\n"
                "Fill in every field based on what you can see in this dermoscopic image. "
                "Return a JSON object with the structure shown in the system prompt."
            ),
        },
    ]

    text, ms = await call_agent("OBSERVER", content_blocks, system_prompt=_OBSERVER_SYSTEM)

    record = _parse_json_block(text)
    if record and "features" in record:
        record["raw_response"] = text
        return record, ms
    return {"features": {}, "notes": text, "raw_response": text}, ms


# ---------------------------------------------------------------------------
# Rule formatting — N-way (includes `contra`)
# ---------------------------------------------------------------------------

def _format_rules_for_mediator(matched_rules: list) -> str:
    """Format rules for the N-way Mediator, prominently showing `contra`."""
    if not matched_rules:
        return "No rules matched for this category set."
    lines = []
    for m in matched_rules:
        r         = m.rule
        condition = r.get("condition", "")
        action    = r.get("action", "")
        favors    = r.get("favors", "")
        contra    = r.get("contra", [])

        if condition.startswith("[Patch rule"):
            # N-way patch rule — extract preconditions and show contra prominently
            bracket_end   = condition.find("]")
            header        = condition[: bracket_end + 1]
            rest          = condition[bracket_end + 2:].strip()
            preconditions = [p.strip() for p in rest.split(";") if p.strip()]
            lines.append(f"[{m.rule_id}] {header}")
            lines.append(f"  → Action: {action}")
            if favors:
                lines.append(f"  → FAVORS: {favors}")
            if contra:
                lines.append(f"  → CONTRA-INDICATES: {', '.join(contra)}")
            lines.append("  ⚠ HARD GATE — apply ONLY when ALL conditions are confirmed by the feature record:")
            for i, pc in enumerate(preconditions, 1):
                lines.append(f"    {i}. {pc}")
        else:
            line = f"[{m.rule_id}] IF {condition} THEN {action} (confidence: {m.confidence})"
            if favors:
                line += f" | FAVORS: {favors}"
            if contra:
                line += f" | CONTRA-INDICATES: {', '.join(contra)}"
            lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Round 2 — MEDIATOR (N-way classify)
# ---------------------------------------------------------------------------

_MEDIATOR_SYSTEM_NWAY = """\
You are an expert dermatologist making a fine-grained dermoscopic lesion classification.
You will receive:
  1. A structured dermoscopic feature observation record.
  2. Expert visual discrimination rules (may be empty).
  3. The exact list of N candidate class labels to choose from.

Classification procedure:
1. Review each feature in the observation record.
2. Skip features with confidence < 0.35 — too unreliable.
   Features with confidence 0.35–0.5 may be used at reduced weight.
3. Apply the expert rules to the high-confidence features.
   For rules marked HARD GATE: verify every listed pre-condition before applying.
   Note FAVORS and CONTRA-INDICATES fields — a rule ruling out several classes
   simultaneously is strong evidence for its FAVORS class.
4. Eliminate classes ruled out by CONTRA-INDICATES signals.
5. Weigh remaining evidence and commit to the single most likely class.

IMPORTANT — you MUST output exactly one of the N class labels provided in the \
user message. "uncertain" is NOT an allowed output. When evidence is weak or \
contradictory, lean toward the class with even slightly more support and commit. \
Do not abstain.

Output ONLY a JSON object:
{
  "label": "<one of the N class names>",
  "confidence": 0.0,
  "reasoning": "Step-by-step reasoning from features to decision, noting any rules applied and classes eliminated.",
  "applied_rules": ["r_001"],
  "eliminated_classes": ["Class Y", "Class Z"],
  "features_used": ["field_name_1"]
}
"""


async def run_mediator_classify(
    task: dict,
    feature_record: dict,
    matched_rules: list,
) -> tuple[dict, str, int]:
    """N-way classification using feature record + rules.

    Returns (decision_dict, raw_text, duration_ms).
    """
    cats         = task.get("categories", CATEGORY_NAMES)
    cats_str     = "\n".join(f"  - {c}" for c in cats)
    rules_text   = _format_rules_for_mediator(matched_rules)
    features_text = _format_feature_record(feature_record)

    user_msg = (
        f"Classify into exactly ONE of the following {len(cats)} classes:\n{cats_str}\n\n"
        f"Dermoscopic feature observation record:\n{features_text}\n\n"
        f"Expert visual rules:\n{rules_text}\n\n"
        "Apply the rules, eliminate ruled-out classes, and return your classification."
    )

    text, ms = await call_agent(
        "MEDIATOR", user_msg, system_prompt=_MEDIATOR_SYSTEM_NWAY, max_tokens=4096,
    )

    decision = _parse_json_block(text)
    if decision and "label" in decision:
        label = decision["label"]
        # Salvage "uncertain" by scanning reasoning for a class name
        if label == "uncertain" or label not in cats:
            for cls in cats:
                if cls.lower() in (decision.get("reasoning", "") + " " + text).lower():
                    decision["label"] = cls
                    decision["confidence"] = 0.2
                    break
            else:
                decision["label"] = cats[0]
                decision["confidence"] = 0.1
        return decision, text, ms

    # Full fallback
    label = "uncertain"
    for cls in cats:
        if cls.lower() in text.lower():
            label = cls
            break
    return {"label": label, "confidence": 0.0, "reasoning": text, "applied_rules": []}, text, ms


# ---------------------------------------------------------------------------
# Round 2R — MEDIATOR (N-way revise)
# ---------------------------------------------------------------------------

_MEDIATOR_REVISE_SYSTEM_NWAY = """\
You are an expert dermatologist revising an N-way dermoscopic classification after \
a consistency check revealed a contradiction.

You will receive:
- The feature observation record
- The initial classification decision and its reasoning
- Feedback from the verifier naming the contradicting feature
- Expert rules
- The full list of candidate classes

Reconsider in light of the feedback. Eliminate the class that the contradiction rules \
out, and choose the next best supported class. You MUST commit to one of the N class \
labels — "uncertain" is NOT allowed.

Output ONLY a JSON object with the same structure as the Mediator:
{
  "label": "<one of the N class names>",
  "confidence": 0.0,
  "reasoning": "...",
  "applied_rules": [],
  "eliminated_classes": [],
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
    """Revise N-way classification after verifier rejection.

    Returns (decision_dict, raw_text, duration_ms).
    """
    cats          = task.get("categories", CATEGORY_NAMES)
    cats_str      = "\n".join(f"  - {c}" for c in cats)
    rules_text    = _format_rules_for_mediator(matched_rules)
    features_text = _format_feature_record(feature_record)

    user_msg = (
        f"Candidate classes:\n{cats_str}\n\n"
        f"Dermoscopic feature observation record:\n{features_text}\n\n"
        f"Expert rules:\n{rules_text}\n\n"
        f"Prior decision: {json.dumps(prior_decision, indent=2)}\n\n"
        f"Verifier feedback:\n"
        f"  Consistent: {verifier_feedback.get('consistent', '?')}\n"
        f"  Revision signal: {verifier_feedback.get('revision_signal', '')}\n"
        f"  Notes: {verifier_feedback.get('notes', '')}\n\n"
        "Revise your classification. Eliminate the class the feedback rules out and commit."
    )

    text, ms = await call_agent(
        "MEDIATOR_REVISE", user_msg, system_prompt=_MEDIATOR_REVISE_SYSTEM_NWAY, max_tokens=4096,
    )

    decision = _parse_json_block(text)
    if decision and "label" in decision:
        label = decision["label"]
        if label == "uncertain" or label not in cats:
            for cls in cats:
                if cls.lower() in (decision.get("reasoning", "") + " " + text).lower():
                    decision["label"] = cls
                    decision["confidence"] = 0.2
                    break
            else:
                decision["label"] = prior_decision.get("label", cats[0])
        return decision, text, ms

    label = prior_decision.get("label", cats[0])
    return {"label": label, "confidence": 0.0, "reasoning": text, "applied_rules": []}, text, ms


# ---------------------------------------------------------------------------
# Round 3 — VERIFIER (N-way, 1 reference image per class)
# ---------------------------------------------------------------------------

_VERIFIER_SYSTEM_NWAY = """\
You are a visual consistency checker for fine-grained N-way dermoscopic lesion classification.

You will be shown:
1. A test dermoscopic image with a proposed label and feature record.
2. One labeled reference image per candidate class.

Your task: catch HARD CONTRADICTIONS only.
- Set consistent=false ONLY when a pathognomonic feature of a DIFFERENT class is \
  unmistakably present in the test image (e.g., arborizing vessels labeled as Melanoma, \
  or clear red lacunae labeled as Benign Keratosis).
- Do NOT flag ambiguous, subtle, or low-confidence differences.
- Dermoscopy is inherently ambiguous — overlap is normal and is NOT contradictory.
- When in doubt, set consistent=true.

If inconsistent, name the specific class whose pathognomonic feature is unmistakably present.

Output ONLY a JSON object:
{
  "consistent": true | false,
  "confidence": 0.0,
  "revision_signal": "Name the specific feature and the class it belongs to, if inconsistent.",
  "notes": "Any additional observations."
}
"""


async def run_verifier(
    task: dict,
    decision: dict,
    feature_record: dict,
) -> tuple[dict, int]:
    """Check N-way classification consistency against 1 reference image per class.

    Uses task["few_shot"] dict: {class_name: [path, ...]}
    Returns (verification_dict, duration_ms).
    """
    cats      = task.get("categories", CATEGORY_NAMES)
    few_shot  = task.get("few_shot", {})

    content_blocks: list[dict] = [
        {"type": "text", "text": f"TEST IMAGE — proposed label: {decision.get('label', '?')}"},
        _image_block(task["test_image_path"]),
    ]

    for cls in cats:
        paths = few_shot.get(cls, [])
        if paths:
            content_blocks.append({"type": "text", "text": f"\nREFERENCE — {cls}:"})
            content_blocks.append(_image_block(paths[0]))  # 1 image per class

    features_text = _format_feature_record(feature_record)
    cats_str      = ", ".join(cats)
    content_blocks.append({
        "type": "text",
        "text": (
            f"\nCandidate classes: {cats_str}\n"
            f"Dermoscopic feature record:\n{features_text}\n\n"
            f"Decision reasoning: {decision.get('reasoning', '')}\n\n"
            "Is this N-way classification visually consistent with the reference images?"
        ),
    })

    text, ms = await call_agent(
        "VERIFIER", content_blocks, system_prompt=_VERIFIER_SYSTEM_NWAY, max_tokens=2048,
    )

    result = _parse_json_block(text)
    if result and "consistent" in result:
        return result, ms
    return {"consistent": True, "confidence": 0.5, "revision_signal": "", "notes": text}, ms


# ---------------------------------------------------------------------------
# Post-task — N-way Rule Extractor
# ---------------------------------------------------------------------------

_RULE_EXTRACTOR_SYSTEM_NWAY = """\
You are a knowledge engineer extracting visual dermoscopic discrimination rules for \
fine-grained N-way skin lesion classification.

You will receive the full N-way task context: all candidate classes, the observed \
dermoscopic features, the decision made, the correct label, and the outcome.

Extract 0-3 new visual rules that would help future N-way classifications.  Each rule:
- MUST be purely visual and dermoscopic (observable in a dermoscopic image)
- MUST clearly favor one of the N classes
- SHOULD specify which classes it contra-indicates when that is clear and reliable
- MUST be generalizable beyond this one image

Do NOT extract rules about patient history, body location, or non-visual information.

Output a JSON block:
```json
{
  "rule_updates": [
    {
      "action": "new",
      "condition": "If [dermoscopic feature description] is observed...",
      "rule_action": "Classify as [class name]",
      "favors": "[class name]",
      "contra": ["[class name 1]", "[class name 2]"],
      "tags": ["derm-ham10000", "dermatology_7class"]
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
    category_set_id: str = "",
) -> tuple[str, int]:
    """Extract new N-way visual dermoscopic rules from a classified example.

    Returns (raw_text_with_rule_updates, duration_ms).
    """
    cats          = task.get("categories", CATEGORY_NAMES)
    cats_str      = ", ".join(cats)
    features_text = _format_feature_record(feature_record)
    outcome       = "CORRECT" if is_correct else (
        f"WRONG (predicted '{decision.get('label', '?')}', actual '{correct_label}')"
    )
    cset_id = category_set_id or task.get("category_set_id", CATEGORY_SET_ID)

    user_msg = (
        f"Category set: {cset_id}\n"
        f"All candidate classes: {cats_str}\n\n"
        f"Dermoscopic feature observation record:\n{features_text}\n\n"
        f"Decision: {decision.get('label', '?')} (confidence: {decision.get('confidence', 0):.2f})\n"
        f"Reasoning: {decision.get('reasoning', '')[:400]}\n\n"
        f"Ground truth: {correct_label}\n"
        f"Outcome: {outcome}\n\n"
        f"Extract 0-3 new visual rules. Tag them: [\"derm-ham10000\", \"{cset_id}\"]"
    )

    text, ms = await call_agent(
        "RULE_EXTRACTOR", user_msg, system_prompt=_RULE_EXTRACTOR_SYSTEM_NWAY, max_tokens=1024,
    )
    return text, ms


# ---------------------------------------------------------------------------
# Baseline — N-way zero-shot and few-shot
# ---------------------------------------------------------------------------

_BASELINE_ZERO_SHOT_SYSTEM_NWAY = """\
You are an expert dermatologist. You will be shown a dermoscopic image.
Classify it as exactly one of the N lesion types provided.

Output ONLY a JSON object:
{
  "label": "<one of the N class names>",
  "confidence": 0.0,
  "reasoning": "Brief dermoscopic rationale."
}
"""

_BASELINE_FEW_SHOT_SYSTEM_NWAY = """\
You are an expert dermatologist. You will be shown a test dermoscopic image and \
one labeled reference image per candidate lesion type.
Classify the TEST IMAGE as exactly one of the N lesion types.

Output ONLY a JSON object:
{
  "label": "<one of the N class names>",
  "confidence": 0.0,
  "reasoning": "Brief dermoscopic rationale."
}
"""


async def run_baseline(
    task: dict,
    mode: str = "zero_shot",
) -> tuple[dict, int]:
    """Run a zero-shot or few-shot N-way baseline (no rules, no schema).

    Returns (decision_dict, duration_ms).
    """
    cats     = task.get("categories", CATEGORY_NAMES)
    cats_str = ", ".join(f"'{c}'" for c in cats)

    if mode == "zero_shot":
        content = [
            _image_block(task["test_image_path"]),
            {"type": "text",
             "text": f"Classify this dermoscopic image as one of: {cats_str}.\n"
                     "Return the JSON object specified in the system prompt."},
        ]
        system = _BASELINE_ZERO_SHOT_SYSTEM_NWAY
    else:  # few_shot
        content: list[dict] = [
            {"type": "text", "text": f"TEST IMAGE — classify as one of: {cats_str}:"},
            _image_block(task["test_image_path"]),
        ]
        few_shot = task.get("few_shot", {})
        for cls in cats:
            paths = few_shot.get(cls, [])
            if paths:
                content.append({"type": "text", "text": f"\nREFERENCE — {cls}:"})
                content.append(_image_block(paths[0]))
        content.append({"type": "text",
                         "text": "Classify the TEST IMAGE. Return the JSON object specified."})
        system = _BASELINE_FEW_SHOT_SYSTEM_NWAY

    text, ms = await call_agent(
        f"BASELINE_{mode.upper()}", content, system_prompt=system, max_tokens=512,
    )

    result = _parse_json_block(text)
    if result and "label" in result:
        # Ensure label is in the valid set
        if result["label"] not in cats:
            for cls in cats:
                if cls.lower() in text.lower():
                    result["label"] = cls
                    break
            else:
                result["label"] = cats[0]
                result["confidence"] = 0.1
        return result, ms

    label = cats[0]
    for cls in cats:
        if cls.lower() in text.lower():
            label = cls
            break
    return {"label": label, "confidence": 0.0, "reasoning": text}, ms


# ---------------------------------------------------------------------------
# Dialogic patching — N-way expert rule authoring
# ---------------------------------------------------------------------------

from core.dialogic_distillation import agents as _dd_agents
from domain_config import DERM_CONFIG as _DERM_CONFIG


async def run_expert_rule_author(
    task: dict,
    wrong_prediction: str,
    correct_label: str,
    model_reasoning: str = "",
    model: str = "claude-sonnet-4-6",
) -> tuple[dict, int]:
    """Author a corrective N-way rule for a failure case.

    Adapts the 2-way dialogic distillation author by injecting N-way context.
    The rule dict returned includes `favors` and `contra` fields.
    """
    # Build a pseudo pair_info that includes all N categories as context.
    cats    = task.get("categories", CATEGORY_NAMES)
    cset_id = task.get("category_set_id", CATEGORY_SET_ID)
    other_classes = [c for c in cats if c not in (wrong_prediction, correct_label)]

    # Augment the task with pair-like fields so core can build its prompts,
    # while adding an nway_context hint for the model to use.
    nway_task = dict(task)
    nway_task["class_a"]    = correct_label
    nway_task["class_b"]    = wrong_prediction
    nway_task["pair_id"]    = cset_id
    nway_task["_nway_all"]  = cats
    nway_task["_nway_other"] = other_classes

    # Inject N-way hint into model_reasoning
    nway_hint = (
        f"\n\n[N-way context] All candidate classes: {', '.join(cats)}. "
        f"The rule should indicate which classes it contra-indicates, "
        f"in addition to favoring '{correct_label}'. "
        f"Other classes beyond '{wrong_prediction}' to consider ruling out: "
        f"{', '.join(other_classes) if other_classes else 'none'}."
    )

    result, ms = await _dd_agents.run_expert_rule_author(
        nway_task, wrong_prediction, correct_label,
        config=_DERM_CONFIG,
        model_reasoning=model_reasoning + nway_hint,
        model=model,
        call_agent_fn=call_agent,
    )

    # Post-process: set contra to include wrong_prediction + any other classes
    # explicitly mentioned in the rule text as being ruled out.
    if isinstance(result, dict):
        if "contra" not in result:
            result["contra"] = [wrong_prediction]
        result["category_set_id"] = cset_id

    return result, ms


async def run_rule_validator_on_image(
    image_path: str,
    ground_truth: str,
    candidate_rule: dict,
    model: str = "",
) -> tuple[dict, int]:
    return await _dd_agents.run_rule_validator_on_image(
        image_path, ground_truth, candidate_rule, config=_DERM_CONFIG,
        model=model or _d2.ACTIVE_MODEL, call_agent_fn=call_agent,
    )


async def validate_candidate_rule(
    candidate_rule: dict,
    validation_images: list,
    trigger_image_path: str,
    trigger_correct_label: str,
    model: str = "",
    early_exit_fp: int = 2,
) -> dict:
    return await _dd_agents.validate_candidate_rule(
        candidate_rule, validation_images, trigger_image_path,
        trigger_correct_label, config=_DERM_CONFIG,
        model=model or _d2.ACTIVE_MODEL, early_exit_fp=early_exit_fp,
        call_agent_fn=call_agent,
    )


async def validate_candidate_rules_batch(
    rules: list[dict],
    validation_images: list,
    trigger_image_path: str,
    trigger_correct_label: str,
    model: str = "",
) -> list[dict]:
    return await _dd_agents.validate_candidate_rules_batch(
        rules, validation_images, trigger_image_path, trigger_correct_label,
        config=_DERM_CONFIG, model=model or _d2.ACTIVE_MODEL,
        call_agent_fn=call_agent,
    )


async def run_contrastive_feature_analysis(
    tp_cases: list[dict],
    fp_cases: list[dict],
    candidate_rule: dict,
    pair_info: dict,
    model: str = "",
) -> tuple[dict, int]:
    return await _dd_agents.run_contrastive_feature_analysis(
        tp_cases, fp_cases, candidate_rule, pair_info, config=_DERM_CONFIG,
        model=model or _d2.ACTIVE_MODEL, call_agent_fn=call_agent,
    )


async def run_rule_spectrum_generator(
    candidate_rule: dict,
    tp_cases: list[dict],
    fp_cases: list[dict],
    contrastive_result: dict,
    pair_info: dict,
    model: str = "",
) -> tuple[list[dict], int]:
    return await _dd_agents.run_rule_spectrum_generator(
        candidate_rule, tp_cases, fp_cases, contrastive_result, pair_info,
        config=_DERM_CONFIG, model=model or _d2.ACTIVE_MODEL, call_agent_fn=call_agent,
    )


async def run_rule_completer(
    candidate_rule: dict,
    pair_info: dict,
    model: str = "",
) -> tuple[dict, int]:
    return await _dd_agents.run_rule_completer(
        candidate_rule, pair_info, config=_DERM_CONFIG,
        model=model or _d2.ACTIVE_MODEL, call_agent_fn=call_agent,
    )


async def run_semantic_rule_validator(
    candidate_rule: dict,
    pair_info: dict,
    model: str = "",
) -> tuple[dict, int]:
    return await _dd_agents.run_semantic_rule_validator(
        candidate_rule, pair_info, config=_DERM_CONFIG,
        model=model or _d2.ACTIVE_MODEL, call_agent_fn=call_agent,
    )
