"""
agents.py — Bird patching agent functions for the KF dialogic loop.

Uses ornithological vocabulary in all prompts (replacing dermoscopy language
from the dermatology module).

Infrastructure (call_agent, OpenRouter, ClaudeTutor, image encoding) is
re-used from the dermatology agents module to avoid duplication.

Patching functions exported:
  run_expert_rule_author         Step 1a — author corrective rule from failure
  run_rule_completer             Step 1b — fill implicit background conditions
  run_semantic_rule_validator    Step 1c — text-only logic check before image testing
  run_rule_validator_on_image    Step 3  — binary precondition check on one image
  validate_candidate_rule        Step 3  — validate rule against a pool
  run_contrastive_feature_analysis  Step 3b — identify TP/FP discriminating feature
  run_rule_spectrum_generator    Step 3c — generate 4-level specificity spectrum
  validate_candidate_rules_batch Step 3c — batch-validate all spectrum levels
"""

from __future__ import annotations
import asyncio
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Re-use call_agent infrastructure from dermatology (via importlib to avoid
# circular import — Python would find THIS file as "agents" via sys.path)
# ---------------------------------------------------------------------------
import importlib.util as _ilu

_DERM_AGENTS_PATH = (
    Path(__file__).resolve().parents[2] / "dermatology" / "python" / "agents.py"
)
_derm_spec = _ilu.spec_from_file_location("derm_agents", _DERM_AGENTS_PATH)
_derm_agents = _ilu.module_from_spec(_derm_spec)
_derm_spec.loader.exec_module(_derm_agents)  # type: ignore[union-attr]

call_agent        = _derm_agents.call_agent
encode_image_b64  = _derm_agents.encode_image_b64
_image_block      = _derm_agents._image_block        # pylint: disable=protected-access
_parse_json_block = _derm_agents._parse_json_block   # pylint: disable=protected-access
ACTIVE_MODEL      = _derm_agents.ACTIVE_MODEL
reset_cost_tracker   = _derm_agents.reset_cost_tracker
get_cost_tracker     = _derm_agents.get_cost_tracker
get_call_cache_stats = _derm_agents.get_call_cache_stats


# ---------------------------------------------------------------------------
# System prompts — ornithology vocabulary
# ---------------------------------------------------------------------------

_EXPERT_RULE_AUTHOR_SYSTEM = """\
You are a senior ornithologist and bird identification expert.

A classification model made an error on a bird photograph. Your job is to author
a precise visual rule that would have led to the correct species identification —
and that will generalize to similar photographs in the future.

The rule must be:
1. Purely visual — observable in a photograph only (no range, habitat, vocalizations,
   behavior, or seasonal context unless clearly visible in the image)
2. Expressed as a pre-condition + prediction: "When [field mark features are met],
   identify as [species]"
3. The pre-condition must be specific enough to EXCLUDE false positives — it should
   NOT apply to typical specimens of the confusable species
4. Generalizable: it must describe a plumage pattern or structural feature that applies
   to a class of similar photographs, not just this one image

Output ONLY a JSON object:
{
  "rule": "Natural language: When [pre-condition], identify as [species].",
  "feature": "snake_case_feature_name",
  "favors": "<exact species name>",
  "confidence": "high" | "medium" | "low",
  "preconditions": [
    "Condition 1 that must hold for this rule to apply",
    "Condition 2 ...",
    ...
  ],
  "rationale": "Why this field mark pattern distinguishes the two species."
}
"""

_RULE_VALIDATOR_SYSTEM = """\
You are an expert ornithologist assessing whether a visual identification rule
applies to a given bird photograph.

You will be shown a photograph and a candidate rule with its pre-conditions.
Your job is to answer two questions:
1. Do the rule's pre-conditions hold for this photograph?
2. If yes, what species would the rule identify?

Be strict about pre-conditions: only mark them as met if you can clearly observe
the required field mark or plumage feature. When in doubt, mark as NOT met.

Output ONLY a JSON object:
{
  "precondition_met": true | false,
  "would_predict": "<species_name>" | null,
  "observations": "Brief note on what you saw that led to this assessment."
}
"""

_CONTRASTIVE_ANALYSIS_SYSTEM = """\
You are a senior ornithologist and bird identification expert.

You will be shown a candidate rule that fires correctly on some bird photographs
(TRUE POSITIVES) but incorrectly on others (FALSE POSITIVES). Your task is to
identify the single most discriminating visual feature that distinguishes the TP
photographs from the FP photographs — i.e., a field mark or plumage feature that
is consistently present in TPs but absent in FPs, or vice versa.

This feature will be used to tighten the rule's pre-conditions.

Output ONLY a JSON object:
{
  "discriminating_feature": "snake_case_feature_name",
  "description": "Plain-language description of the field mark or plumage feature.",
  "present_in": "tp" | "fp",
  "confidence": "high" | "medium" | "low",
  "rationale": "Why this feature distinguishes TPs from FPs in ornithological terms."
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

_SPECTRUM_SYSTEM = """\
You are a senior ornithologist and bird identification expert.

You are given a candidate species identification rule and evidence about where it works
(TRUE POSITIVES) and where it misfires (FALSE POSITIVES). Your task is to produce FOUR
versions of the rule at different levels of specificity — from most general to most
specific — so that the tightest version that still passes a precision gate can be selected.

The four levels must all favor the same species and describe the same underlying field mark
or plumage phenomenon, varying only in how many pre-conditions are required:

  Level 1 — MOST GENERAL: single essential pre-condition. The one field mark that
    is most diagnostic of the favored species and most absent from FP specimens.
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
      "rule": "When [single essential field mark], identify as [species].",
      "feature": "snake_case_feature_name",
      "favors": "<exact species name>",
      "confidence": "high" | "medium" | "low",
      "preconditions": ["Single essential pre-condition"],
      "rationale": "Why this is the core diagnostic field mark."
    },
    { "level": 2, "label": "moderate", ... },
    { "level": 3, "label": "original", ...  },
    { "level": 4, "label": "most_specific", ... }
  ]
}
"""

_RULE_COMPLETER_SYSTEM = """\
You are a senior ornithologist completing a species identification rule.

BACKGROUND
The rule was authored by an expert responding to a specific misidentification. Experts
naturally write DIAGNOSTIC rules: they describe what was distinctive about that one
photograph. But they omit BACKGROUND conditions — field marks so obvious to a trained
ornithologist that they go without saying. When the rule is evaluated by a naive
classifier that checks only what is explicitly listed, those omitted conditions create
loopholes: the rule fires on photographs that share the distinctive field mark but lack
the expected background features of the favored species.

YOUR TASK
Identify the implicit pre-conditions the expert assumed but did not write down.
These are conditions that:
  1. Are standard, well-established field marks expected to be PRESENT for
     the favored species (positive background conditions).
  2. Are standard markers expected to be ABSENT for the favored species — i.e.,
     features that would instead indicate the confusable species — that the rule
     does not already exclude (negative background conditions).
  3. Are NOT already covered, even implicitly, by the existing pre-conditions.

DO NOT add:
  - Conditions that could plausibly occur in both species.
  - Conditions already implied by the existing pre-conditions.
  - Highly specific conditions that would rarely be met — do not over-tighten.
  - Non-visual conditions (range, vocalizations, behavior, season) unless visible.

Keep the original rule text and feature key unchanged.
Add the new pre-conditions to the existing list.

Output ONLY a JSON object (no markdown fences, no commentary outside the JSON):
{
  "rule": "<original rule text — unchanged>",
  "feature": "<original feature key — unchanged>",
  "favors": "<unchanged>",
  "confidence": "<unchanged>",
  "preconditions": ["<full list: original pre-conditions + new ones>"],
  "added_preconditions": ["<only the newly added pre-conditions>"],
  "completion_rationale": "<2–3 sentences explaining what background knowledge was
                           implicit and why it needed to be made explicit>"
}

If the existing pre-conditions are already complete and you have nothing meaningful
to add, return the rule unchanged with "added_preconditions": [] and explain why.
"""

_SEMANTIC_VALIDATOR_SYSTEM = """\
You are a senior ornithologist reviewing a proposed species identification rule before
it is tested on photographs.

You will be given:
1. A candidate rule and its pre-conditions
2. The pair of species it is meant to distinguish (favored species vs confusable species)

Your task: evaluate whether each pre-condition is a reliable visual discriminator.

For each pre-condition, rate it as one of:
- "reliable"          — field mark consistently separates the favored species from the
                        confusable one; rarely or never present in the other species
- "unreliable"        — feature can easily occur in both species, is too vague,
                        or points in the wrong direction
- "context_dependent" — only discriminating under specific co-occurring conditions,
                        lighting, age class, or season; risky as a stand-alone gate

Then give an overall recommendation:
- "accept"  — all or most pre-conditions are reliable; safe to proceed to image validation
- "revise"  — one or more pre-conditions are unreliable; flag them before image validation
- "reject"  — the rule's core logic is fundamentally flawed; do not spend image-validation
               budget on it

Output ONLY a JSON object (no markdown, no commentary):
{
  "precondition_ratings": [
    {
      "precondition": "<exact text of pre-condition>",
      "rating": "reliable|unreliable|context_dependent",
      "comment": "<one-sentence ornithological justification>"
    }
  ],
  "overall": "accept|revise|reject",
  "rationale": "<two-to-three sentence overall assessment>"
}
"""


# ---------------------------------------------------------------------------
# Patching agent functions — bird vocabulary
# ---------------------------------------------------------------------------

async def run_expert_rule_author(
    task: dict,
    wrong_prediction: str,
    correct_label: str,
    model_reasoning: str = "",
    model: str = "",
) -> tuple[dict, int]:
    """Call the expert VLM with a failure case and ask it to author a corrective rule.

    task must have: class_a, class_b, test_image_path
    Returns (candidate_rule_dict, duration_ms).
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
                f"You are the TUTOR. A weaker PUPIL VLM just looked at this bird "
                f"photograph and got the identification wrong. Your job is to (a) notice "
                f"exactly where the pupil went off track, and (b) give the pupil a "
                f"corrective visual rule it can apply to this and similar future cases.\n\n"
                f"Species pair: {class_a} vs {class_b}\n"
                f"Ground truth: {correct_label}\n"
                f"Pupil's prediction: {wrong_prediction}  ← WRONG\n"
                f"Pupil's stated reasoning: {reasoning_snippet}\n\n"
                "First, look at the photograph carefully and diagnose the pupil's mistake:\n"
                "  • Did the pupil hallucinate field marks that aren't actually visible "
                "(e.g. a wing bar or eye ring that doesn't exist in this photo)?\n"
                "  • Did the pupil overweight an ambiguous or shared feature?\n"
                "  • Did the pupil miss a salient field mark that would have flipped "
                "the identification?\n\n"
                "Then author a corrective visual rule that:\n"
                f"  1. Would have led to the correct identification ('{correct_label}')\n"
                "  2. Targets the specific failure mode you identified (not generic advice)\n"
                "  3. Uses pre-conditions the pupil can actually check from the photograph "
                "— concrete visible features, not knowledge of range or vocalizations\n"
                f"  4. Will generalize to similar cases without firing on "
                f"{class_a if correct_label != class_a else class_b}"
            ),
        },
    ]

    text, ms = await call_agent(
        "EXPERT_RULE_AUTHOR",
        content,
        system_prompt=_EXPERT_RULE_AUTHOR_SYSTEM,
        model=model or ACTIVE_MODEL,
        max_tokens=4096,
    )

    rule = _parse_json_block(text)
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
    model: str = "",
) -> tuple[dict, int]:
    """Test whether a candidate rule applies to a single labeled bird photograph.

    Returns (result_dict, duration_ms).
    """
    rule_text = candidate_rule.get("rule", "")
    preconditions = candidate_rule.get("preconditions", [])
    favors = candidate_rule.get("favors", "")

    precond_text = (
        "\n".join(f"  - {p}" for p in preconditions)
        if preconditions else "  (none specified)"
    )

    content = [
        _image_block(image_path),
        {
            "type": "text",
            "text": (
                f"Candidate rule: {rule_text}\n\n"
                f"Pre-conditions that must ALL hold:\n{precond_text}\n\n"
                f"If pre-conditions are met, this rule identifies the bird as: {favors}\n\n"
                "Does this rule apply to this photograph? "
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
    validation_images: list,          # list of (image_path: str, ground_truth: str)
    trigger_image_path: str,
    trigger_correct_label: str,
    model: str = "",
    early_exit_fp: int = 2,
) -> dict:
    """Test a candidate rule against a pool of labeled bird photographs.

    Returns a dict with TP, FP, TN, FN counts, precision, recall, accept flag,
    and per-image case lists for contrastive analysis.
    """
    tp = fp = tn = fn = 0
    fires_on_trigger = False
    tp_cases: list[dict] = []
    fp_cases: list[dict] = []

    # Check trigger image first
    trigger_result, _ = await run_rule_validator_on_image(
        trigger_image_path, trigger_correct_label, candidate_rule, model=model
    )
    fires_on_trigger = (
        trigger_result["precondition_met"] and trigger_result["correct"]
    )

    if not fires_on_trigger:
        remaining = len(validation_images)
        favors = candidate_rule.get("favors", "")
        fn = sum(1 for _, gt in validation_images if gt == favors)
        tn = remaining - fn
        return {
            "fires_on_trigger": False,
            "tp": 0, "fp": 0, "tn": tn, "fn": fn,
            "precision": 0.0, "recall": 0.0,
            "accepted": False, "rejection_reason": "did not fire on trigger",
            "tp_cases": [], "fp_cases": [],
        }

    favors = candidate_rule.get("favors", "")
    for img_path, gt in validation_images:
        result, _ = await run_rule_validator_on_image(
            img_path, gt, candidate_rule, model=model
        )
        fires = result["precondition_met"]
        if fires:
            predicted = result.get("would_predict")
            if predicted == gt:
                tp += 1
                tp_cases.append({
                    "image_path": img_path, "ground_truth": gt,
                    "observations": result.get("observations", ""),
                })
            else:
                fp += 1
                fp_cases.append({
                    "image_path": img_path, "ground_truth": gt,
                    "observations": result.get("observations", ""),
                })
                if fp > early_exit_fp:
                    # Definite rejection — count remaining as TN conservatively
                    break
        else:
            if gt == favors:
                fn += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    accepted = fires_on_trigger and fp <= 1 and precision >= 0.75

    return {
        "fires_on_trigger": fires_on_trigger,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall,
        "accepted": accepted,
        "rejection_reason": None if accepted else (
            "did not fire on trigger" if not fires_on_trigger
            else f"FP={fp} precision={precision:.2f}"
        ),
        "tp_cases": tp_cases,
        "fp_cases": fp_cases,
    }


async def validate_candidate_rules_batch(
    candidate_rules: list[dict],
    validation_images: list,
    trigger_image_path: str,
    trigger_correct_label: str,
    model: str = "",
) -> list[dict]:
    """Validate a list of candidate rules (spectrum levels) against the same pool.

    Returns a list of validation result dicts, one per rule (same order).
    Runs all validations concurrently.
    """
    tasks = [
        validate_candidate_rule(
            rule, validation_images, trigger_image_path,
            trigger_correct_label, model=model,
        )
        for rule in candidate_rules
    ]
    return list(await asyncio.gather(*tasks))


async def run_contrastive_feature_analysis(
    tp_cases: list[dict],
    fp_cases: list[dict],
    candidate_rule: dict,
    pair_info: dict,
    model: str = "",
) -> tuple[dict, int]:
    """Identify the visual feature that best distinguishes TP from FP cases.

    Uses validator's per-image observations (text) — no re-running OBSERVER.
    Returns (contrastive_result_dict, duration_ms).
    """
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
        f"Species pair: {class_a} vs {class_b}\n"
        f"Rule favors: {favors}\n\n"
        f"Rule: {rule_text}\n\n"
        f"Pre-conditions:\n{precond_text}\n\n"
        f"TRUE POSITIVE cases (rule fired correctly):\n{_fmt_cases(tp_cases, 'TP')}\n\n"
        f"FALSE POSITIVE cases (rule fired on wrong species):\n{_fmt_cases(fp_cases, 'FP')}\n\n"
        "What single visual feature (field mark or plumage pattern) most reliably "
        "distinguishes the TP cases from the FP cases? "
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
    """
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
        f"Species pair: {class_a} vs {class_b}\n"
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
        for lv in levels:
            lv.setdefault("favors", favors)
            lv["raw_response"] = text
        return levels, ms

    return [candidate_rule], ms


async def run_rule_completer(
    candidate_rule: dict,
    pair_info: dict,
    model: str = "",
) -> tuple[dict, int]:
    """Enrich a candidate rule with implicit background conditions the expert omitted.

    Text-only — no images. Returns (completed_rule_dict, duration_ms).
    """
    rule_text = candidate_rule.get("rule", "")
    preconditions = candidate_rule.get("preconditions", [])
    favors = candidate_rule.get("favors", "")
    class_a = pair_info.get("class_a", "")
    class_b = pair_info.get("class_b", "")
    other_class = class_b if favors == class_a else class_a

    precond_text = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(preconditions))

    user_msg = (
        f"Favored species: {favors}\n"
        f"Confusable species: {other_class}\n\n"
        f"Rule:\n{rule_text}\n\n"
        f"Existing pre-conditions ({len(preconditions)}):\n"
        f"{precond_text if precond_text else '  (none)'}\n\n"
        "Please complete the rule by adding any implicit background conditions "
        "the expert assumed but did not state."
    )

    text, ms = await call_agent(
        "RULE_COMPLETER",
        user_msg,
        system_prompt=_RULE_COMPLETER_SYSTEM,
        model=model or ACTIVE_MODEL,
        max_tokens=2048,
    )

    result = _parse_json_block(text)
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
    model: str = "",
) -> tuple[dict, int]:
    """Text-only semantic check of a candidate rule's ornithological logic.

    Returns (semantic_result_dict, duration_ms).
    """
    rule_text = candidate_rule.get("rule", "")
    preconditions = candidate_rule.get("preconditions", [])
    favors = candidate_rule.get("favors", "")
    class_a = pair_info.get("class_a", "")
    class_b = pair_info.get("class_b", "")
    other_class = class_b if favors == class_a else class_a

    precond_text = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(preconditions))

    user_msg = (
        f"Favored species: {favors}\n"
        f"Confusable species: {other_class}\n\n"
        f"Rule:\n{rule_text}\n\n"
        f"Pre-conditions:\n{precond_text}\n\n"
        "Please evaluate each pre-condition and provide an overall recommendation."
    )

    text, ms = await call_agent(
        "SEMANTIC_RULE_VALIDATOR",
        user_msg,
        system_prompt=_SEMANTIC_VALIDATOR_SYSTEM,
        model=model or ACTIVE_MODEL,
        max_tokens=2048,
    )

    result = _parse_json_block(text)
    if result and "overall" in result:
        return result, ms

    return {
        "precondition_ratings": [],
        "overall": "revise",
        "rationale": f"(parse error — raw: {text[:200]})",
    }, ms
