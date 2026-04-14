"""
agents.py — PyroWatch wildfire DD agent wrappers.

Extends core/dialogic_distillation with four wildfire-specific adaptations:

  1. Cross-modal TUTOR prompt: expert bridges MWIR confirmation to optical RGB
     observables (README §7, DESIGN.md §3.1)
  2. Three-tier grounding check: validates rule criteria per hardware tier
     (ground_sentinel, scout_drone, commander_aircraft). Commander tier is
     informational only — MWIR provides definitive confirmation there.
  3. Temporal reformulation: converts "smoke drift consistent across frames"
     (available on fixed sentinel PTZ) to a within-frame proxy (wind-axis
     elongation with opacity gradient) for single-pass scout drones.
  4. Environmental context injection: accepted rules are augmented with RAWS
     meteorological preconditions evaluated at inference time, making the
     same optical rule more or less aggressive based on fire weather conditions
     without any weight change.

Entry point: run_wildfire_dd_session()
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Callable, Optional
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from core.dialogic_distillation import agents as _core_agents
from core.dialogic_distillation.protocols import DomainConfig

_PYTHON_DIR = Path(__file__).resolve().parent
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from domain_config import (
    WILDFIRE_DETECTION_CONFIG,
    TIER_OBSERVABILITY,
    CROSS_MODAL_TUTOR_PROMPT,
    CONFUSABLE_PAIRS,
    RED_FLAG_PRECONDITIONS,
    ELEVATED_RISK_PRECONDITIONS,
    CONDITION_SET_URGENCY,
)

# ---------------------------------------------------------------------------
# Tier display names (used in prompts)
# ---------------------------------------------------------------------------

TIER_DESCRIPTION: dict[str, str] = {
    "ground_sentinel": (
        "fixed mountaintop PTZ camera (RGB 4K, continuous 360° coverage)"
    ),
    "scout_drone": (
        "patrol drone (RGB 20MP, single-pass route, single frame per location)"
    ),
    "commander_aircraft": (
        "long-loiter aircraft (RGB 60MP + calibrated MWIR thermal camera)"
    ),
}

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

def _tier_grounding_system(tier: str, tier_observability: str) -> str:
    return f"""\
You are evaluating whether the preconditions of a visual classification rule
are observable by a specific camera sensor system.

SENSOR SYSTEM ({tier} tier):
{tier_observability}

For each precondition, determine:

  OBSERVABLE    — the feature can be confirmed or denied from a SINGLE frame
                  captured by this sensor; the sensor has sufficient resolution
                  and the feature is physically present in the image
  TEMPORAL      — confirming this feature requires comparing MULTIPLE frames
                  over time (e.g., "drifts consistently", "oscillates",
                  "maintains direction across frames")
  UNOBSERVABLE  — the feature is physically impossible to detect with this
                  sensor (e.g., requires MWIR, higher resolution than sensor
                  provides, or external data not present in the image)

Output ONLY a JSON object:
{{
  "tier": "{tier}",
  "criteria": [
    {{
      "precondition": "<exact text>",
      "classification": "observable" | "temporal" | "unobservable",
      "reason": "<one sentence: why this classification>"
    }}
  ],
  "summary": "accept_all" | "remove_some" | "reformulate_temporal"
}}
"""


def _temporal_reformulation_system(config: DomainConfig) -> str:
    return f"""\
You are a {config.expert_role}.

A temporal precondition in a wildfire smoke classification rule cannot be used
by a single-frame classifier (scout drone passing over a location once). Your
task is to propose a within-frame spatial proxy that approximates the same
discriminating signal using only features visible in a single image.

A good proxy for wildfire smoke:
- Is observable in one frame (no "drifts consistently", "oscillates" etc.)
- Approximates the physical reason the temporal feature is diagnostic
  (e.g., consistent wind-direction drift → asymmetric elongation along wind axis)
- Uses concrete visual terms: color, shape, opacity gradient, spatial extent
- Is specific enough to exclude heat shimmer and atmospheric haze

Output ONLY a JSON object:
{{
  "temporal_criterion": "<original temporal precondition>",
  "proxy": "<single-frame within-image spatial proxy>",
  "rationale": "<why this proxy approximates the temporal signal>",
  "confidence": "high" | "medium" | "low"
}}
"""


def _cross_modal_tutor_system(config: DomainConfig) -> str:
    return f"""\
You are a {config.expert_role}.

A wildfire detection fleet is misclassifying early ignition smoke as heat
shimmer in camera frames. You have access to confirmation from a calibrated
MWIR sensor (3–5 μm band, absolute temperature) that reveals the ground truth.

Your task: describe what features ARE VISIBLE IN THE OPTICAL RGB IMAGE that
should lead to the correct classification. You are bridging from thermal
confirmation to optical observables.

Rules:
- Describe only features visible in the RGB camera frame
- Do not reference MWIR temperature values, RAWS weather data, GPS coordinates,
  or any sensor output other than the RGB image itself
- For scout drone tier: the classifier processes SINGLE FRAMES — do not describe
  temporal patterns like "drifts consistently" or "oscillates differently".
  Instead, describe what those temporal phenomena leave as spatial traces in a
  single frame (e.g., asymmetric elongation, opacity gradient along wind axis)
- Be concrete: color hue, opacity level, spatial origin point, column shape

PRECONDITION QUALITY — CRITICAL:
Write exactly 3 preconditions. Follow these rules strictly:

1. POSITIVE PRESENCE ONLY. Every precondition must state a feature that IS
   visibly present. Never write absence conditions ("no X", "lacks X",
   "without X", "does not show X").

   BAD:  "The haze has no broad distributed source"
   GOOD: "The haze has a concentrated point of origin where opacity is highest"

2. NO MEASUREMENTS. No pixel sizes, no percentages, no distances, no angles.
   Use qualitative words: "concentrated", "faint", "blue-gray", "elongated".

   BAD:  "Smoke occupies approximately 2% of frame area"
   GOOD: "A faint haze is visible in one region of the frame"

3. CERTAIN AND CONFIRMABLE. Only write a precondition if a third-party observer
   could confirm it immediately from this single image. When in doubt, leave it
   out — 3 strong conditions beat 5 uncertain ones.

   BAD:  "A barely perceptible blue tint may be present at the haze boundary"
   GOOD: "The haze has a blue or blue-gray color distinct from the terrain"

Output ONLY a JSON object:
{{
  "rule": "When [preconditions], classify as [class].",
  "feature": "snake_case_feature_name",
  "favors": "<exact class name>",
  "confidence": "high" | "medium" | "low",
  "preconditions": [
    "<Condition 1 — positive presence, no measurements, clearly confirmable>",
    "<Condition 2 — ...>",
    "<Condition 3 — ...>"
  ],
  "rationale": "<Why these optical features correspond to the MWIR confirmation.>"
}}
"""


def _env_context_injection_system() -> str:
    return """\
You are a wildfire risk analyst reviewing an accepted visual classification rule
for early smoke detection.

Your task: add a meteorological context block to the rule that specifies the
RAWS weather conditions under which this rule should activate with elevated
sensitivity. The same optical signature is more significant under red flag
conditions than in spring.

The context block must use only fields available from a standard RAWS weather
station: wind_speed_mph, relative_humidity_pct, temperature_f, fire_weather_watch.

Output ONLY a JSON object with three fields:
{
  "red_flag": {
    "raws_wind_speed_mph": "> 20",
    "raws_relative_humidity_pct": "< 15",
    "raws_temperature_f": "> 85",
    "fire_weather_watch": "true",
    "urgency_multiplier": 1.0
  },
  "elevated_risk": {
    "raws_wind_speed_mph": "> 10",
    "raws_relative_humidity_pct": "< 25",
    "raws_temperature_f": "> 75",
    "urgency_multiplier": 0.7
  },
  "normal": {
    "urgency_multiplier": 0.4,
    "note": "Rule active but low urgency; likely atmospheric or agricultural source"
  }
}
"""


# ---------------------------------------------------------------------------
# Cross-modal TUTOR call
# ---------------------------------------------------------------------------

async def run_cross_modal_tutor(
    failure_image_path: str,
    confirmation_modality: str,
    confirmation_details: str,
    ground_truth_class: str,
    pupil_classification: str,
    pupil_confidence: float,
    tier: str,
    config: DomainConfig,
    model: str = "",
    call_agent_fn: Callable | None = None,
) -> tuple[dict, int]:
    """Ask the TUTOR expert to bridge from MWIR confirmation to optical RGB observables.

    Returns (candidate_rule_dict, duration_ms).
    """
    _call = call_agent_fn or _core_agents._get_default_call_agent()

    tier_description = TIER_DESCRIPTION.get(tier, tier)
    prompt = CROSS_MODAL_TUTOR_PROMPT.format(
        expert_role=config.expert_role,
        pupil_classification=pupil_classification,
        pupil_confidence=pupil_confidence,
        confirmation_modality=confirmation_modality,
        ground_truth_class=ground_truth_class,
        confirmation_details=confirmation_details,
        tier_description=tier_description,
    )

    content = [
        _core_agents.image_block(failure_image_path),
        {"type": "text", "text": prompt},
    ]

    text, ms = await _call(
        "CROSS_MODAL_TUTOR",
        content,
        system_prompt=_cross_modal_tutor_system(config),
        model=model,
        max_tokens=2048,
    )

    rule = _core_agents.parse_json_block(text)
    if rule and "rule" in rule and "preconditions" in rule:
        rule["raw_response"] = text
        rule.setdefault("favors", ground_truth_class)
        return rule, ms

    return {
        "rule": text,
        "feature": "unknown",
        "favors": ground_truth_class,
        "confidence": "low",
        "preconditions": [],
        "rationale": "",
        "raw_response": text,
    }, ms


# ---------------------------------------------------------------------------
# Tier-aware grounding check
# ---------------------------------------------------------------------------

async def run_tier_grounding_check(
    candidate_rule: dict,
    tier: str,
    tier_observability: str,
    config: DomainConfig,
    model: str = "",
    call_agent_fn: Callable | None = None,
) -> tuple[dict, int]:
    """Evaluate which preconditions are observable, temporal, or unobservable
    for a specific hardware tier.

    Returns (grounding_result_dict, duration_ms).
    """
    _call = call_agent_fn or _core_agents._get_default_call_agent()

    preconditions = candidate_rule.get("preconditions", [])
    precond_text = "\n".join(f"{i+1}. {p}" for i, p in enumerate(preconditions))

    user_msg = (
        f"Rule: {candidate_rule.get('rule', '')}\n\n"
        f"Preconditions to evaluate:\n{precond_text}\n\n"
        f"Evaluate each precondition for the {tier} tier sensor described above."
    )

    text, ms = await _call(
        "TIER_GROUNDING_CHECK",
        user_msg,
        system_prompt=_tier_grounding_system(tier, tier_observability),
        model=model,
        max_tokens=1024,
    )

    result = _core_agents.parse_json_block(text)
    if result and "criteria" in result:
        criteria = result["criteria"]
        observable = [c["precondition"] for c in criteria if c["classification"] == "observable"]
        temporal = [c["precondition"] for c in criteria if c["classification"] == "temporal"]
        unobservable = [c["precondition"] for c in criteria if c["classification"] == "unobservable"]
        result["observable"] = observable
        result["temporal"] = temporal
        result["unobservable"] = unobservable
        return result, ms

    # Fallback: treat all as observable if parse fails
    return {
        "tier": tier,
        "criteria": [{"precondition": p, "classification": "observable", "reason": "(parse error)"} for p in preconditions],
        "summary": "accept_all",
        "observable": preconditions,
        "temporal": [],
        "unobservable": [],
    }, ms


# ---------------------------------------------------------------------------
# Temporal feature reformulation
# ---------------------------------------------------------------------------

async def run_temporal_reformulation(
    temporal_criterion: str,
    ground_truth_class: str,
    wrong_class: str,
    config: DomainConfig,
    model: str = "",
    call_agent_fn: Callable | None = None,
) -> tuple[dict, int]:
    """Convert a temporal precondition to a within-frame spatial proxy.

    Returns (reformulation_dict, duration_ms).
    """
    _call = call_agent_fn or _core_agents._get_default_call_agent()

    user_msg = (
        f"Confusable pair: {ground_truth_class} vs {wrong_class}\n\n"
        f"Temporal precondition (cannot be used in a single-frame classifier "
        f"on a scout drone):\n"
        f"  \"{temporal_criterion}\"\n\n"
        f"Propose a within-frame spatial proxy for this temporal criterion."
    )

    text, ms = await _call(
        "TEMPORAL_REFORMULATION",
        user_msg,
        system_prompt=_temporal_reformulation_system(config),
        model=model,
        max_tokens=512,
    )

    result = _core_agents.parse_json_block(text)
    if result and "proxy" in result:
        return result, ms

    return {
        "temporal_criterion": temporal_criterion,
        "proxy": None,
        "rationale": f"(parse error — raw: {text[:200]})",
        "confidence": "low",
    }, ms


# ---------------------------------------------------------------------------
# Environmental context injection
# ---------------------------------------------------------------------------

async def run_env_context_injection(
    accepted_rule: dict,
    raws_conditions: dict | None,
    config: DomainConfig,
    model: str = "",
    call_agent_fn: Callable | None = None,
) -> tuple[dict, int]:
    """Augment an accepted rule with RAWS meteorological context preconditions.

    Asks the model to produce the three condition-set blocks (red_flag,
    elevated_risk, normal) with urgency multipliers.

    If raws_conditions is provided, also evaluates which condition set is
    currently active and sets rule['active_condition_set'].

    Returns (augmented_rule, duration_ms).
    """
    _call = call_agent_fn or _core_agents._get_default_call_agent()

    user_msg = (
        f"Accepted rule: {accepted_rule.get('rule', '')}\n\n"
        f"Feature: {accepted_rule.get('feature', '')}\n"
        f"Rationale: {accepted_rule.get('rationale', '')}\n\n"
        f"Add meteorological context blocks for this smoke detection rule."
    )

    text, ms = await _call(
        "ENV_CONTEXT_INJECTION",
        user_msg,
        system_prompt=_env_context_injection_system(),
        model=model,
        max_tokens=512,
    )

    context_blocks = _core_agents.parse_json_block(text)
    augmented = {**accepted_rule}

    if context_blocks and "red_flag" in context_blocks:
        augmented["context_preconditions"] = context_blocks
    else:
        # Fallback to static defaults
        augmented["context_preconditions"] = {
            "red_flag": {**RED_FLAG_PRECONDITIONS, "urgency_multiplier": 1.0},
            "elevated_risk": {**ELEVATED_RISK_PRECONDITIONS, "urgency_multiplier": 0.7},
            "normal": {"urgency_multiplier": 0.4},
        }

    # Evaluate active condition set from live RAWS data if provided
    if raws_conditions:
        augmented["active_condition_set"] = _evaluate_condition_set(raws_conditions)
        augmented["raws_snapshot"] = raws_conditions

    return augmented, ms


def _evaluate_condition_set(raws: dict) -> str:
    """Return the active condition set name given a RAWS conditions dict.

    Expected keys (all optional): wind_speed_mph, relative_humidity_pct,
    temperature_f, fire_weather_watch.
    """
    wind = raws.get("wind_speed_mph", 0)
    rh = raws.get("relative_humidity_pct", 100)
    temp = raws.get("temperature_f", 60)
    watch = raws.get("fire_weather_watch", False)

    if (wind > 20 and rh < 15 and temp > 85) or watch:
        return "red_flag"
    if wind > 10 and rh < 25 and temp > 75:
        return "elevated_risk"
    return "normal"


# ---------------------------------------------------------------------------
# Tier adaptation: grounding check + temporal reformulation
# ---------------------------------------------------------------------------

async def adapt_rule_for_tier(
    candidate_rule: dict,
    tier: str,
    tier_observability: str,
    ground_truth_class: str,
    wrong_class: str,
    config: DomainConfig,
    validator_model: str = "",
    tutor_model: str = "",
    call_agent_fn: Callable | None = None,
) -> tuple[dict, dict]:
    """Produce a tier-adapted rule from a candidate rule.

    Steps:
      1. Run tier grounding check — classify each precondition
      2. Remove unobservable criteria
      3. Reformulate temporal criteria as within-frame spatial proxies
      4. Return adapted rule + grounding report

    Returns (adapted_rule, grounding_report).
    """
    grounding_result, _ = await run_tier_grounding_check(
        candidate_rule=candidate_rule,
        tier=tier,
        tier_observability=tier_observability,
        config=config,
        model=validator_model,
        call_agent_fn=call_agent_fn,
    )

    observable = grounding_result["observable"]
    temporal = grounding_result["temporal"]
    unobservable = grounding_result["unobservable"]

    # Reformulate temporal criteria concurrently
    proxy_tasks = [
        run_temporal_reformulation(
            temporal_criterion=tc,
            ground_truth_class=ground_truth_class,
            wrong_class=wrong_class,
            config=config,
            model=tutor_model,
            call_agent_fn=call_agent_fn,
        )
        for tc in temporal
    ]
    proxy_results = await asyncio.gather(*proxy_tasks)

    proxies = []
    for (reform, _) in proxy_results:
        if reform.get("proxy"):
            proxies.append(reform["proxy"])

    adapted_preconditions = observable + proxies
    removed = unobservable
    reformulated = [
        {"temporal": tc, "proxy": r.get("proxy")}
        for tc, (r, _) in zip(temporal, proxy_results)
    ]

    adapted_rule = {
        **candidate_rule,
        "preconditions": adapted_preconditions,
        "tier": tier,
        "tier_adapted": True,
    }
    if adapted_preconditions != candidate_rule.get("preconditions", []):
        adapted_rule["rule"] = (
            "When [" + "; ".join(adapted_preconditions) + f"], classify as {ground_truth_class}."
        )

    grounding_report = {
        "tier": tier,
        "original_count": len(candidate_rule.get("preconditions", [])),
        "observable": observable,
        "temporal_reformulated": reformulated,
        "unobservable_removed": removed,
        "adapted_count": len(adapted_preconditions),
        "summary": grounding_result.get("summary"),
    }

    return adapted_rule, grounding_report


# ---------------------------------------------------------------------------
# Full wildfire DD session
# ---------------------------------------------------------------------------

async def run_wildfire_dd_session(
    failure_image_path: str,
    confirmation_modality: str,
    confirmation_details: str,
    ground_truth_class: str,
    pupil_classification: str,
    pupil_confidence: float,
    pool_images: list[tuple[str, str]],
    pair_info: dict,
    raws_conditions: dict | None = None,
    config: DomainConfig = WILDFIRE_DETECTION_CONFIG,
    tier_observability: dict[str, str] = TIER_OBSERVABILITY,
    tutor_model: str = "claude-opus-4-6",
    validator_model: str = "claude-sonnet-4-6",
    tiers: list[str] | None = None,
    primary_tier: str = "ground_sentinel",
    skip_pool_validation: bool = False,
    call_agent_fn: Callable | None = None,
    console=None,
) -> dict:
    """Run a complete wildfire DD session.

    Steps:
      1. Cross-modal TUTOR call (expert bridges MWIR → RGB observables)
      2. Pool validation on base rule
      3. Spectrum tightening if pool fails with FPs
      4. Per-tier adaptation (grounding check + temporal reformulation)
      5. Environmental context injection (RAWS precondition blocks)

    primary_tier: the tier whose sensor captured the failure image. Used to
    select the appropriate TUTOR prompt phrasing. Default: ground_sentinel.

    Returns a session transcript dict.
    """
    if tiers is None:
        tiers = ["ground_sentinel", "scout_drone"]

    _print = console.print if console else lambda *a, **kw: None
    t0 = time.monotonic()

    transcript: dict = {
        "failure_image": failure_image_path,
        "confirmation_modality": confirmation_modality,
        "ground_truth_class": ground_truth_class,
        "pupil_classification": pupil_classification,
        "pupil_confidence": pupil_confidence,
        "raws_conditions": raws_conditions,
        "steps": [],
        "initial_rule": None,
        "pool_result": None,
        "tighten_history": [],
        "grounding_reports": {},
        "final_rules": {},
        "outcome": "pending",
    }

    # ------------------------------------------------------------------
    # Step 1: Cross-modal TUTOR
    # ------------------------------------------------------------------
    _print("\n[bold]Step 1: Cross-modal TUTOR[/bold]", style="cyan")
    initial_rule, ms = await run_cross_modal_tutor(
        failure_image_path=failure_image_path,
        confirmation_modality=confirmation_modality,
        confirmation_details=confirmation_details,
        ground_truth_class=ground_truth_class,
        pupil_classification=pupil_classification,
        pupil_confidence=pupil_confidence,
        tier=primary_tier,
        config=config,
        model=tutor_model,
        call_agent_fn=call_agent_fn,
    )
    transcript["initial_rule"] = {k: v for k, v in initial_rule.items() if k != "raw_response"}
    transcript["steps"].append({"step": "cross_modal_tutor", "duration_ms": ms})

    _print(f"  Rule: [italic]{initial_rule.get('rule', '')[:120]}[/italic]")
    for pc in initial_rule.get("preconditions", []):
        _print(f"    pre: {pc[:100]}")

    active_rule = initial_rule

    # ------------------------------------------------------------------
    # Step 2: Pool validation
    # When skip_pool_validation=True the TUTOR's initial rule is accepted
    # directly without calling the validator model (useful when the
    # validator model is rate-limited or unavailable).
    # ------------------------------------------------------------------
    if skip_pool_validation:
        _print("\n[bold]Step 2: Pool validation SKIPPED[/bold]", style="yellow")
        pool_result = {
            "fires_on_trigger": True, "accepted": True, "skipped": True,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0, "precision": 1.0, "recall": 1.0,
            "rejection_reason": None,
        }
        transcript["pool_result"] = pool_result
        accepted = True
    else:
        _print(f"\n[bold]Step 2: Pool validation[/bold] ({len(pool_images)} frames)", style="cyan")
        pool_result = await _core_agents.validate_candidate_rule(
        candidate_rule=active_rule,
        validation_images=pool_images,
        trigger_image_path=failure_image_path,
        trigger_correct_label=ground_truth_class,
        config=config,
        model=validator_model,
        call_agent_fn=call_agent_fn,
    )
    transcript["pool_result"] = {k: v for k, v in pool_result.items() if k not in ("tp_cases", "fp_cases")}

    prec = pool_result["precision"]
    fp = pool_result["fp"]
    tp = pool_result["tp"]
    accepted = pool_result["accepted"]
    _print(f"  TP={tp} FP={fp} precision={prec:.2f} "
           f"{'[green]PASS[/green]' if accepted else '[red]FAIL[/red]'}")

    # ------------------------------------------------------------------
    # Step 3: Spectrum tightening if pool failed with FPs
    # ------------------------------------------------------------------
    if not accepted and fp > 0 and pool_result.get("tp_cases"):
        _print("\n[bold]Step 3: Spectrum tightening[/bold]", style="cyan")

        tp_cases = pool_result.get("tp_cases", [])
        fp_cases = pool_result.get("fp_cases", [])

        contrastive_result, _ = await _core_agents.run_contrastive_feature_analysis(
            tp_cases=tp_cases,
            fp_cases=fp_cases,
            candidate_rule=active_rule,
            pair_info=pair_info,
            config=config,
            model=tutor_model,
            call_agent_fn=call_agent_fn,
        )
        disc = contrastive_result.get("description", "(none)")
        _print(f"  Contrastive: [italic]{disc[:100]}[/italic]")
        transcript["tighten_history"].append({"step": "contrastive_analysis", "description": disc})

        spectrum_levels, _ = await _core_agents.run_rule_spectrum_generator(
            candidate_rule=active_rule,
            tp_cases=tp_cases,
            fp_cases=fp_cases,
            contrastive_result=contrastive_result,
            pair_info=pair_info,
            config=config,
            model=tutor_model,
            call_agent_fn=call_agent_fn,
        )
        _print(f"  Generated {len(spectrum_levels)} spectrum levels")

        batch_results = await _core_agents.validate_candidate_rules_batch(
            candidate_rules=spectrum_levels,
            validation_images=pool_images,
            trigger_image_path=failure_image_path,
            trigger_correct_label=ground_truth_class,
            config=config,
            model=validator_model,
            call_agent_fn=call_agent_fn,
        )

        for lv, res in zip(spectrum_levels, batch_results):
            _print(f"    L{lv.get('level','?')}: TP={res['tp']} FP={res['fp']} "
                   f"prec={res['precision']:.2f} "
                   f"{'[green]PASS[/green]' if res['accepted'] else '[red]FAIL[/red]'}")

        passing = [(lv, res) for lv, res in zip(spectrum_levels, batch_results) if res["accepted"]]
        if passing:
            best_lv, best_res = max(passing, key=lambda x: x[0].get("level", 0))
            active_rule = best_lv
            accepted = True
            _print(f"  [green]Accepted L{best_lv.get('level')} ({best_lv.get('label')})[/green]")
            transcript["pool_result_after_tighten"] = {
                k: v for k, v in best_res.items() if k not in ("tp_cases", "fp_cases")
            }
            transcript["tighten_history"].append({
                "step": "selected_level",
                "level": best_lv.get("level"),
                "label": best_lv.get("label"),
            })
        else:
            _print("  [red]No spectrum level passed pool gate[/red]")
            transcript["outcome"] = "pool_failed"

    if not accepted:
        transcript["outcome"] = "pool_failed"
        transcript["final_rules"] = {}
        elapsed = int((time.monotonic() - t0) * 1000)
        transcript["total_duration_ms"] = elapsed
        return transcript

    # ------------------------------------------------------------------
    # Step 4: Per-tier adaptation (concurrent)
    # ------------------------------------------------------------------
    _print("\n[bold]Step 4: Tier adaptation[/bold]", style="cyan")

    tier_tasks = [
        adapt_rule_for_tier(
            candidate_rule=active_rule,
            tier=tier,
            tier_observability=tier_observability[tier],
            ground_truth_class=ground_truth_class,
            wrong_class=pupil_classification,
            config=config,
            validator_model=validator_model,
            tutor_model=tutor_model,
            call_agent_fn=call_agent_fn,
        )
        for tier in tiers
        if tier in tier_observability
    ]
    tier_results = await asyncio.gather(*tier_tasks)

    base_rules: dict[str, dict] = {}
    for tier, (adapted_rule, grounding_report) in zip(tiers, tier_results):
        base_rules[tier] = {k: v for k, v in adapted_rule.items() if k != "raw_response"}
        transcript["grounding_reports"][tier] = grounding_report
        n_removed = len(grounding_report["unobservable_removed"])
        n_reformed = len([r for r in grounding_report["temporal_reformulated"] if r["proxy"]])
        _print(f"  {tier}: {grounding_report['adapted_count']} preconditions "
               f"({n_removed} removed, {n_reformed} reformulated from temporal)")

    # ------------------------------------------------------------------
    # Step 5: Environmental context injection
    # ------------------------------------------------------------------
    _print("\n[bold]Step 5: Environmental context injection[/bold]", style="cyan")

    if raws_conditions:
        active_set = _evaluate_condition_set(raws_conditions)
        _print(f"  RAWS conditions: {raws_conditions}")
        _print(f"  Active condition set: [bold]{active_set}[/bold]")
    else:
        active_set = "unknown"

    env_tasks = [
        run_env_context_injection(
            accepted_rule=base_rules[tier],
            raws_conditions=raws_conditions,
            config=config,
            model=validator_model,
            call_agent_fn=call_agent_fn,
        )
        for tier in tiers
        if tier in base_rules
    ]
    env_results = await asyncio.gather(*env_tasks)

    final_rules: dict[str, dict] = {}
    for tier, (augmented_rule, _) in zip(tiers, env_results):
        final_rules[tier] = augmented_rule
        sets = list(augmented_rule.get("context_preconditions", {}).keys())
        _print(f"  {tier}: context sets {sets}")

    transcript["final_rules"] = final_rules
    transcript["outcome"] = "accepted"
    transcript["active_condition_set"] = active_set

    elapsed = int((time.monotonic() - t0) * 1000)
    transcript["total_duration_ms"] = elapsed
    _print(f"\n[green]Session complete[/green] — {elapsed}ms total")

    return transcript
