"""
dialogic.py — Multi-round dialogic distillation protocol.

Three parties collaborate:
  PUPIL  — the cheap VLM that failed (provides wrong prediction + reasoning)
  TUTOR  — the expert model (authors corrective rules)
  KF     — the orchestrator (steers dialog, validates, registers rules)

The protocol:
  1. PUPIL fails on a case
  2. KF shows the failure to TUTOR
  3. TUTOR proposes a corrective rule (Round 1)
  4. KF immediately tests the rule on the trigger image (grounding check)
  5. If preconditions don't fire -> KF feeds validator observations
     back to TUTOR with specific guidance
  6. TUTOR refines the rule (Round 2+)
  7. Once grounded -> KF runs the pool gate
  8. If pool fails with FPs -> contrastive tightening rounds
"""
from __future__ import annotations
from typing import Callable, Optional

from .protocols import DomainConfig
from .constants import DEFAULT_MAX_TIGHTENING_ROUNDS
from . import agents as _agents
from . import prompts as _prompts


async def run_dialogic_distillation(
    image_path: str,
    image_id: str,
    correct_label: str,
    wrong_prediction: str,
    pupil_reasoning: str,
    pair_info: dict,
    config: DomainConfig,
    tutor_model: str,
    validator_model: str,
    max_rounds: int,
    pool_images: list,
    max_tightening_rounds: int = DEFAULT_MAX_TIGHTENING_ROUNDS,
    call_agent_fn: Callable | None = None,
    console=None,
) -> dict:
    """Run multi-round dialogic distillation for a single failure image.

    Returns a complete transcript dict with per-round evidence.
    """
    class_a = pair_info["class_a"]
    class_b = pair_info["class_b"]
    opposing = class_a if correct_label != class_a else class_b

    # Optional rich console for progress output
    _print = console.print if console else lambda *a, **kw: None

    transcript = {
        "image_id": image_id,
        "correct_label": correct_label,
        "wrong_prediction": wrong_prediction,
        "rounds": [],
        "final_rule": None,
        "grounded_at_round": None,
        "pool_result": None,
        "pool_result_after_tighten": None,
        "outcome": "pending",
    }

    active_rule = None

    for round_num in range(1, max_rounds + 1):
        _print(f"\n  [bold]Round {round_num}[/bold]", style="cyan")
        round_record = {"round": round_num, "party_actions": []}

        # -- TUTOR turn --
        if round_num == 1:
            # Build vocabulary examples block
            vocab_lines = ""
            if config.good_vocabulary_examples or config.bad_vocabulary_examples:
                parts = []
                for ex in config.good_vocabulary_examples:
                    parts.append(f"   - GOOD: \"{ex}\"")
                for ex in config.bad_vocabulary_examples:
                    parts.append(f"   - BAD: \"{ex}\"")
                vocab_lines = "\n".join(parts) + "\n"

            prompt_text = _prompts.ROUND1_PROMPT.format(
                class_a=class_a,
                class_b=class_b,
                correct_label=correct_label,
                wrong_prediction=wrong_prediction,
                pupil_reasoning=pupil_reasoning or "(not available)",
                item_noun=config.item_noun,
                class_noun=config.class_noun,
                vocab_examples=vocab_lines,
            )
        else:
            prev = transcript["rounds"][-1]
            prev_rule = active_rule
            prev_obs = prev.get("validator_observations", "")
            prev_met = "MET" if prev.get("fires_on_trigger") else "NOT MET"

            kf_guidance = generate_kf_guidance(
                prev_rule, prev, round_num, config
            )

            prompt_text = _prompts.REFINEMENT_PROMPT.format(
                round_num=round_num,
                previous_rule=prev_rule.get("rule", ""),
                previous_preconditions="\n".join(
                    f"  - {p}" for p in prev_rule.get("preconditions", [])),
                validator_observations=prev_obs,
                met_status=prev_met,
                kf_guidance=kf_guidance,
                item_noun=config.item_noun,
            )

        content = [
            _agents.image_block(image_path),
            {"type": "text", "text": prompt_text},
        ]

        raw_text, ms = await (_agents._get_default_call_agent()
                              if call_agent_fn is None else call_agent_fn)(
            "DIALOGIC_TUTOR",
            content,
            system_prompt=_prompts.dialogic_tutor_system(config),
            model=tutor_model,
            max_tokens=2048,
        )

        rule = _agents.parse_json_block(raw_text)
        if not rule or "preconditions" not in rule:
            rule = {"rule": raw_text, "feature": "unknown",
                    "favors": correct_label, "confidence": "low",
                    "preconditions": [], "rationale": ""}
        rule["raw_response"] = raw_text
        active_rule = rule

        round_record["party_actions"].append({
            "party": "TUTOR",
            "action": "author_rule" if round_num == 1 else "refine_rule",
            "rule": rule.get("rule", ""),
            "preconditions": rule.get("preconditions", []),
            "rationale": rule.get("rationale", ""),
        })

        _print(f"    TUTOR -> rule: [italic]{rule.get('rule', '')[:120]}[/italic]")
        for pc in rule.get("preconditions", []):
            _print(f"      pre: {pc[:100]}")

        # -- KF grounding check --
        _print("    KF -> grounding check on trigger image...")
        val_result, _ = await _agents.run_rule_validator_on_image(
            image_path=image_path,
            ground_truth=correct_label,
            candidate_rule=rule,
            config=config,
            model=validator_model,
            call_agent_fn=call_agent_fn,
        )

        fires = val_result.get("precondition_met", False)
        observations = val_result.get("observations", "")

        round_record["party_actions"].append({
            "party": "KF",
            "action": "grounding_check",
            "fires_on_trigger": fires,
            "validator_observations": observations,
        })
        round_record["fires_on_trigger"] = fires
        round_record["validator_observations"] = observations

        status = "[green]FIRES[/green]" if fires else "[red]DOES NOT FIRE[/red]"
        _print(f"    KF -> {status}")
        _print(f"    Validator saw: [dim]{observations[:150]}[/dim]")

        transcript["rounds"].append(round_record)

        if fires:
            transcript["grounded_at_round"] = round_num
            _print(f"    [green]Rule grounded at round {round_num}![/green]")
            break
        elif round_num < max_rounds:
            _print(f"    KF -> steering TUTOR for round {round_num + 1}...")
        else:
            _print(f"    [red]Max rounds ({max_rounds}) reached without grounding[/red]")

    # -- Pool gate (if grounded) --
    if transcript["grounded_at_round"] is not None:
        _print(f"\n  [bold]Pool gate[/bold] -- validating on {len(pool_images)} images...")
        pool_result = await _agents.validate_candidate_rule(
            candidate_rule=active_rule,
            validation_images=pool_images,
            trigger_image_path=image_path,
            trigger_correct_label=correct_label,
            config=config,
            model=validator_model,
            call_agent_fn=call_agent_fn,
        )
        transcript["pool_result"] = {
            k: v for k, v in pool_result.items()
            if k not in ("tp_cases", "fp_cases")
        }

        prec = pool_result["precision"]
        fp = pool_result["fp"]
        tp = pool_result["tp"]
        accepted = pool_result["accepted"]

        _print(f"    TP={tp} FP={fp} precision={prec:.2f} "
               f"{'[green]PASS[/green]' if accepted else '[red]FAIL[/red]'}")

        # -- Spectrum-based tightening (if pool fails with FPs) --
        #
        # Instead of the crude single-precondition loop, we use the full
        # patch pipeline: contrastive analysis -> 4-level spectrum ->
        # batch validate all levels -> pick tightest passing level.
        # This is the same approach used in the production patch loop.
        tighten_history = []
        if not accepted and fp > 0 and pool_result.get("tp_cases"):
            _print(f"\n  [bold]Spectrum tightening[/bold] -- "
                   f"FP={fp}, running contrastive analysis...")

            tp_cases = pool_result.get("tp_cases", [])
            fp_cases = pool_result.get("fp_cases", [])

            # Step 1: contrastive analysis -- find the TP/FP discriminating feature
            contrastive_result, _ = await _agents.run_contrastive_feature_analysis(
                tp_cases=tp_cases,
                fp_cases=fp_cases,
                candidate_rule=active_rule,
                pair_info=pair_info,
                config=config,
                model=tutor_model,
                call_agent_fn=call_agent_fn,
            )
            disc = contrastive_result.get("description", "(none)")
            _print(f"    Contrastive: [italic]{disc[:100]}[/italic] "
                   f"(present in {contrastive_result.get('present_in', '?')})")

            tighten_history.append({
                "step": "contrastive_analysis",
                "discriminating_feature": contrastive_result.get("discriminating_feature"),
                "description": disc,
                "present_in": contrastive_result.get("present_in"),
                "confidence": contrastive_result.get("confidence"),
            })

            # Step 2: generate 4-level specificity spectrum
            _print("    Generating 4-level specificity spectrum...")
            spectrum_levels, _ = await _agents.run_rule_spectrum_generator(
                candidate_rule=active_rule,
                tp_cases=tp_cases,
                fp_cases=fp_cases,
                contrastive_result=contrastive_result,
                pair_info=pair_info,
                config=config,
                model=tutor_model,
                call_agent_fn=call_agent_fn,
            )
            _print(f"    Generated {len(spectrum_levels)} spectrum levels")
            for lv in spectrum_levels:
                _print(f"      L{lv.get('level','?')} ({lv.get('label','?')}): "
                       f"{len(lv.get('preconditions', []))} preconditions")

            tighten_history.append({
                "step": "spectrum_generation",
                "n_levels": len(spectrum_levels),
                "levels": [
                    {"level": lv.get("level"), "label": lv.get("label"),
                     "n_preconditions": len(lv.get("preconditions", []))}
                    for lv in spectrum_levels
                ],
            })

            # Step 3: batch validate all levels against the pool
            _print(f"    Batch validating {len(spectrum_levels)} levels "
                   f"on {len(pool_images)} pool images...")
            batch_results = await _agents.validate_candidate_rules_batch(
                candidate_rules=spectrum_levels,
                validation_images=pool_images,
                trigger_image_path=image_path,
                trigger_correct_label=correct_label,
                config=config,
                model=validator_model,
                call_agent_fn=call_agent_fn,
            )

            for lv, res in zip(spectrum_levels, batch_results):
                _print(f"      L{lv.get('level','?')}: "
                       f"TP={res['tp']} FP={res['fp']} "
                       f"prec={res['precision']:.2f} "
                       f"{'[green]PASS[/green]' if res['accepted'] else '[red]FAIL[/red]'}")

            tighten_history.append({
                "step": "batch_validation",
                "results": [
                    {"level": lv.get("level"), "label": lv.get("label"),
                     "tp": res["tp"], "fp": res["fp"],
                     "precision": res["precision"], "accepted": res["accepted"]}
                    for lv, res in zip(spectrum_levels, batch_results)
                ],
            })

            # Step 4: pick the tightest level that passes
            passing = [(lv, res) for lv, res in zip(spectrum_levels, batch_results)
                       if res["accepted"]]
            if passing:
                # tightest = highest level number among passing
                best_lv, best_res = max(passing, key=lambda x: x[0].get("level", 0))
                active_rule = best_lv
                accepted = True
                _print(f"    [green]Accepted spectrum level "
                       f"L{best_lv.get('level')} ({best_lv.get('label')})[/green]")
                tighten_history.append({
                    "step": "selected_level",
                    "level": best_lv.get("level"),
                    "label": best_lv.get("label"),
                    "pool": {k: v for k, v in best_res.items()
                             if k not in ("tp_cases", "fp_cases")},
                })
                transcript["pool_result_after_tighten"] = {
                    k: v for k, v in best_res.items()
                    if k not in ("tp_cases", "fp_cases")
                }
            else:
                _print("    [red]No spectrum level passed the pool gate[/red]")
                tighten_history.append({"step": "selected_level", "level": None,
                                        "outcome": "no_level_passed"})
                transcript["pool_result_after_tighten"] = None

            transcript["tighten_history"] = tighten_history

        transcript["final_rule"] = {
            k: v for k, v in active_rule.items() if k != "raw_response"
        }
        transcript["outcome"] = "accepted" if accepted else "grounded_but_pool_failed"
    else:
        transcript["final_rule"] = {
            k: v for k, v in active_rule.items() if k != "raw_response"
        } if active_rule else None
        transcript["outcome"] = "not_grounded"

    return transcript


def generate_kf_guidance(
    prev_rule: dict,
    prev_round: dict,
    round_num: int,
    config: DomainConfig,
) -> str:
    """Generate KF's steering guidance for the next TUTOR round.

    This is where KF adds value as orchestrator — it doesn't just relay
    the validator's observations, it diagnoses *why* the rule didn't fire
    and gives the TUTOR targeted advice.
    """
    obs = prev_round.get("validator_observations", "").lower()
    preconditions = prev_rule.get("preconditions", [])

    guidance_parts = []

    guidance_parts.append(
        "The validator model describes images differently than you might. "
        "Use the EXACT phrases from the validator's observations where possible."
    )

    if "not" in obs or "no " in obs or "absence" in obs:
        guidance_parts.append(
            "The validator explicitly noted the ABSENCE of certain features. "
            "Your preconditions may reference features that are genuinely not "
            f"visible at the resolution/quality of this {config.item_noun}."
        )

    if len(preconditions) > 3:
        guidance_parts.append(
            f"You had {len(preconditions)} preconditions — too many increases "
            "the chance that one fails. Consolidate to 2-3 strong ones."
        )

    if round_num >= 3:
        guidance_parts.append(
            "We are on round 3+. Try a fundamentally different visual signal "
            "rather than rephrasing the same features. What ELSE do you see "
            f"in this {config.item_noun} that distinguishes it?"
        )

    return "\n".join(f"- {g}" for g in guidance_parts)
