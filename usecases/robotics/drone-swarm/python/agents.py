"""
agents.py — Maritime SAR DD agent wrappers.

Extends core/dialogic_distillation with three maritime-specific adaptations:

  1. Cross-modal TUTOR prompt: expert bridges thermal confirmation to RGB
     observables (DESIGN.md §3.1)
  2. Tier-aware grounding check: validates rule criteria per hardware tier,
     flags temporal features and unobservable criteria (DESIGN.md §3.2)
  3. Temporal reformulation: converts temporal criteria to within-frame
     proxies for single-frame scout classifiers (DESIGN.md §3.2)

Entry point: run_maritime_dd_session()
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
    MARITIME_SAR_CONFIG,
    TIER_OBSERVABILITY,
    CROSS_MODAL_TUTOR_PROMPT,
    CONFUSABLE_PAIRS,
)


# ---------------------------------------------------------------------------
# System prompts for maritime-specific passes
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
                  over time (e.g., "remains stable", "does not disperse",
                  "maintains position across frames")
  UNOBSERVABLE  — the feature is physically impossible to detect with this
                  sensor (e.g., requires higher resolution, different modality,
                  or external data not present in the image)

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

A temporal precondition in a classification rule cannot be used by a
single-frame classifier. Your task is to propose a within-frame optical
proxy that approximates the same discriminating signal using only features
visible in a single image.

A good proxy:
- Is observable in one frame (no "remains stable", "does not disperse" etc.)
- Approximates the underlying physical reason the temporal feature is diagnostic
- Uses concrete visual terms: brightness, shape, geometry, contrast, texture
- Is specific enough to exclude the confusable class

Output ONLY a JSON object:
{{
  "temporal_criterion": "<original temporal precondition>",
  "proxy": "<single-frame within-image proxy>",
  "rationale": "<why this proxy approximates the temporal signal>",
  "confidence": "high" | "medium" | "low"
}}
"""


def _cross_modal_tutor_system(config: DomainConfig) -> str:
    return f"""\
You are a {config.expert_role}.

A fleet of optical scout drones is misclassifying objects in sea-surface
images. You have access to confirmation from a different sensor modality
(thermal FLIR camera on a commander drone) that reveals the ground truth.

Your task: describe what features ARE VISIBLE IN THE RGB OPTICAL IMAGE
that should lead to the correct classification. You are bridging from
thermal knowledge to optical observables.

Rules:
- Describe only features a {config.item_noun} can show
- Do not reference thermal values, AIS data, or any non-optical sensor
- The scout classifier processes SINGLE FRAMES — do not describe temporal
  features like "remains stable" or "does not move". Instead, describe
  what those temporal phenomena leave as traces in a single frame.
- Be concrete: brightness levels, shapes, aspect ratios, spatial arrangements

PRECONDITION QUALITY — CRITICAL:
Write exactly 3 preconditions. Follow these rules strictly:

1. POSITIVE PRESENCE ONLY. Every precondition must state a feature that IS
   visibly present. Never write absence conditions ("no X", "lacks X",
   "without X", "does not show X") — a validator cannot reliably confirm
   that something is absent.

   BAD:  "The object lacks a dark central void"
   GOOD: "The object is solid-filled with bright material visible at its centre"

2. NO MEASUREMENTS. No pixel sizes, no aspect ratio numbers, no distances.
   Use qualitative words: "small", "compact", "roughly oval", "bright".

   BAD:  "Aspect ratio approximately 1:1 to 1:1.5"
   GOOD: "The object has a compact, roughly oval or circular shape"

3. CERTAIN AND CONFIRMABLE. Only write a precondition if a third-party
   observer could confirm it immediately from this single image. When in
   doubt, leave it out — 3 strong conditions beat 5 uncertain ones.

   BAD:  "A faint localized ripple disturbance surrounds the object"
   GOOD: "A small bright oval region is visible against the dark water surface"

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
  "rationale": "<Why these optical features correspond to the thermal confirmation.>"
}}
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
    config: DomainConfig,
    model: str = "",
    call_agent_fn: Callable | None = None,
) -> tuple[dict, int]:
    """Ask the expert TUTOR to bridge from cross-modal confirmation to RGB observables.

    Returns (candidate_rule_dict, duration_ms).
    """
    _call = call_agent_fn or _core_agents._get_default_call_agent()

    prompt = CROSS_MODAL_TUTOR_PROMPT.format(
        expert_role=config.expert_role,
        pupil_classification=pupil_classification,
        pupil_confidence=pupil_confidence,
        confirmation_modality=confirmation_modality,
        ground_truth_class=ground_truth_class,
        confirmation_details=confirmation_details,
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
    grounding_result keys: tier, criteria (list), summary, observable, temporal, unobservable
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
    """Convert a temporal precondition to a within-frame proxy for single-frame classifiers.

    Returns (reformulation_dict, duration_ms).
    reformulation_dict keys: temporal_criterion, proxy, rationale, confidence
    """
    _call = call_agent_fn or _core_agents._get_default_call_agent()

    user_msg = (
        f"Confusable pair: {ground_truth_class} vs {wrong_class}\n\n"
        f"Temporal precondition (cannot be used in a single-frame classifier):\n"
        f"  \"{temporal_criterion}\"\n\n"
        f"Propose a within-frame optical proxy for this temporal criterion."
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
# Tier adaptation: combine grounding check + temporal reformulation
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
      1. Run tier grounding check — classify preconditions
      2. Remove unobservable criteria
      3. Reformulate temporal criteria as within-frame proxies
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
# Full maritime DD session
# ---------------------------------------------------------------------------

async def run_maritime_dd_session(
    failure_image_path: str,
    confirmation_modality: str,
    confirmation_details: str,
    ground_truth_class: str,
    pupil_classification: str,
    pupil_confidence: float,
    pool_images: list[tuple[str, str]],
    pair_info: dict,
    config: DomainConfig = MARITIME_SAR_CONFIG,
    tier_observability: dict[str, str] = TIER_OBSERVABILITY,
    tutor_model: str = "claude-opus-4-6",
    validator_model: str = "claude-sonnet-4-6",
    tiers: list[str] | None = None,
    call_agent_fn: Callable | None = None,
    console=None,
) -> dict:
    """Run a complete maritime DD session.

    Steps:
      1. Cross-modal TUTOR call (expert bridges thermal → RGB observables)
      2. Pool validation on base rule
      3. Spectrum tightening if pool fails with FPs
      4. Per-tier adaptation (grounding check + temporal reformulation)

    Returns a session transcript dict with:
      initial_rule, pool_result, final_rules (per tier), grounding_reports, outcome
    """
    if tiers is None:
        tiers = ["scout", "commander"]

    _print = console.print if console else lambda *a, **kw: None
    t0 = time.monotonic()

    transcript: dict = {
        "failure_image": failure_image_path,
        "confirmation_modality": confirmation_modality,
        "ground_truth_class": ground_truth_class,
        "pupil_classification": pupil_classification,
        "pupil_confidence": pupil_confidence,
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
    # ------------------------------------------------------------------
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
    # Step 4: Per-tier adaptation
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

    final_rules: dict[str, dict] = {}
    for tier, (adapted_rule, grounding_report) in zip(tiers, tier_results):
        final_rules[tier] = {k: v for k, v in adapted_rule.items() if k != "raw_response"}
        transcript["grounding_reports"][tier] = grounding_report
        n_removed = len(grounding_report["unobservable_removed"])
        n_reformed = len([r for r in grounding_report["temporal_reformulated"] if r["proxy"]])
        _print(f"  {tier}: {grounding_report['adapted_count']} preconditions "
               f"({n_removed} removed, {n_reformed} reformulated from temporal)")

    transcript["final_rules"] = final_rules
    transcript["outcome"] = "accepted"

    elapsed = int((time.monotonic() - t0) * 1000)
    transcript["total_duration_ms"] = elapsed
    _print(f"\n[green]Session complete[/green] — {elapsed}ms total")

    return transcript
