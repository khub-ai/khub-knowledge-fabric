"""
ensemble.py — derm-ham10000 (dermoscopy skin lesion classification) ensemble orchestrator.

Implements the 4-round KF pipeline for fine-grained dermoscopic lesion classification.

Round structure:
  Round 0   RuleEngine retrieves visual rules for this confusable pair.
  Round 0.5 Retrieve (or generate) the feature observation schema from ToolRegistry.
  Round 1   OBSERVER (VLM) fills in the feature schema from the test image.
  Round 2   MEDIATOR classifies based on feature record + matched rules.
  Round 3   VERIFIER checks consistency against few-shot images; may trigger revision.

Post-task:
  Rule stats updated (fires/successes/failures).
  Rule extractor proposes new visual rules from the outcome.
  New rules are stored via parse_mediator_rule_updates() with observability_filter=True.

Entry point:
  await run_ensemble(task, task_id, rule_engine, tool_registry, ...)
"""

from __future__ import annotations
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# KF core imports
# ---------------------------------------------------------------------------
_KF_ROOT = Path(__file__).resolve().parents[4]
if str(_KF_ROOT) not in sys.path:
    sys.path.insert(0, str(_KF_ROOT))

from core.knowledge.state import StateManager
from core.knowledge.goals import GoalManager

from rules import RuleEngine, RuleMatch
from tools import ToolRegistry
from agents import (
    call_agent,
    DEFAULT_MODEL,
    reset_cost_tracker,
    get_cost_tracker,
    format_pair_for_prompt,
    run_schema_generator,
    run_observer,
    run_mediator_classify,
    run_mediator_revise,
    run_verifier,
    run_rule_extractor,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_TAG   = "derm-ham10000"
MAX_REVISIONS = 1           # verifier can trigger at most this many revisions
TWO_STAGE_THRESHOLD = 999   # derm-ham10000 is small (<100 rules); always single-stage


# ---------------------------------------------------------------------------
# Round 0 — Rule matching
# ---------------------------------------------------------------------------

async def match_rules(rule_engine: RuleEngine, task: dict) -> list[RuleMatch]:
    """Retrieve active rules relevant to this confusable pair.

    Uses two-stage retrieval when rule count exceeds TWO_STAGE_THRESHOLD.
    """
    active_task_rules = rule_engine.active_task_rules()
    if not active_task_rules:
        return []

    task_text = format_pair_for_prompt(task)

    if len(active_task_rules) > TWO_STAGE_THRESHOLD:
        cat_prompt = rule_engine.build_category_filter_prompt(task_text)
        if cat_prompt:
            cat_text, _ = await call_agent("MEDIATOR", cat_prompt, max_tokens=256)
            subset = rule_engine.filter_rules_by_categories(cat_text, max_rules=25)
        else:
            subset = active_task_rules[:25]
        user_msg = rule_engine.build_match_prompt(task_text, rules_subset=subset)
    else:
        user_msg = rule_engine.build_match_prompt(task_text)

    text, _ = await call_agent("MEDIATOR", user_msg, max_tokens=1024)
    return rule_engine.parse_match_response(text)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

async def run_ensemble(
    task: dict,
    task_id: str = "unknown",
    rule_engine: Optional[RuleEngine] = None,
    tool_registry: Optional[ToolRegistry] = None,
    verbose: bool = True,
    dataset: str = "derm-ham10000",
    dataset_tag: str = DATASET_TAG,
    max_revisions: int = MAX_REVISIONS,
    test_mode: bool = False,
) -> dict:
    """Run the full derm-ham10000 ensemble on a single image classification task.

    Args:
        task: Dict with keys:
            pair_id          str   e.g. "melanoma_vs_melanocytic_nevus"
            class_a          str   e.g. "Melanoma"
            class_b          str   e.g. "Melanocytic Nevus"
            dx_a             str   e.g. "mel"
            dx_b             str   e.g. "nv"
            test_image_path  str   absolute path to the test JPEG image
            test_label       str   ground truth lesion type name
            few_shot_a       list[str]  paths to labeled class_a images (for verifier)
            few_shot_b       list[str]  paths to labeled class_b images
        task_id: Unique identifier for this image (e.g. "melanoma_vs_melanocytic_nevus_ISIC_0024306")
        rule_engine: RuleEngine instance (created if None)
        tool_registry: ToolRegistry instance (created if None)
        verbose: Print round-by-round progress.
        dataset: Dataset label for result tracking.
        dataset_tag: Namespace tag ("derm-ham10000").
        max_revisions: Maximum MEDIATOR revisions after verifier rejection.
        test_mode: If True, no learning — rules/tools are read-only.

    Returns:
        Result dict with: task_id, pair_id, predicted_label, correct_label,
        correct, confidence, rounds_completed, duration_ms, cost_usd,
        feature_record, decision, verification, rule_ids_fired,
        input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens,
        api_calls, model, dataset.
    """
    if rule_engine is None:
        rule_engine = RuleEngine(dataset_tag=dataset_tag)
    if tool_registry is None:
        tool_registry = ToolRegistry(dataset_tag=dataset_tag)

    task["_task_id"] = task_id
    pair_id = task["pair_id"]
    correct_label = task.get("test_label", "")

    reset_cost_tracker()
    t_start = time.time()
    rounds_completed = 0

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log(f"\n{'─'*60}")
    log(f"Task: {task_id}")
    log(f"Pair: {task['class_a']} vs {task['class_b']}")
    log(f"Ground truth: {correct_label}")
    log(f"Rules: {rule_engine.stats_summary()}")

    # ------------------------------------------------------------------
    # Round 0 — Rule matching
    # ------------------------------------------------------------------
    log("Round 0: matching rules...")
    matched_rules = await match_rules(rule_engine, task)
    fired_ids = [m.rule_id for m in matched_rules]
    rounds_completed += 1

    if matched_rules:
        log(f"  Matched {len(matched_rules)} rule(s): {fired_ids}")
    else:
        log("  No rules matched.")

    # ------------------------------------------------------------------
    # Round 0.5 — Feature schema (from ToolRegistry or generated)
    # ------------------------------------------------------------------
    schema_name = f"{pair_id}_schema"
    schema = tool_registry.get_schema(schema_name)

    if schema:
        log(f"  Schema: loaded '{schema_name}' from registry ({len(schema.get('fields', []))} fields)")
    else:
        log(f"  Schema: generating '{schema_name}'...")
        schema, schema_ms = await run_schema_generator(task, matched_rules)
        log(f"  Schema: generated {len(schema.get('fields', []))} fields")
        if not test_mode and schema.get("fields"):
            tool_registry.register(
                name=schema_name,
                code=json.dumps(schema),
                verified=True,
                tool_type="schema",
                source_task=pair_id,
                description=f"Feature observation schema for {task['class_a']} vs {task['class_b']}",
            )

    # ------------------------------------------------------------------
    # Round 1 — OBSERVER fills in the feature form
    # ------------------------------------------------------------------
    log("Round 1: OBSERVER filling feature schema from dermoscopic image...")
    t_r1 = time.time()
    feature_record, obs_ms = await run_observer(task, schema, matched_rules)
    rounds_completed += 1
    log(f"  Done in {time.time()-t_r1:.1f}s  |  "
        f"{len(feature_record.get('features', {}))} features recorded")

    high_conf = {k: v for k, v in feature_record.get("features", {}).items()
                 if v.get("confidence", 0) >= 0.35}
    _hc_parts = [f"{k}={v['value']!r}" for k, v in list(high_conf.items())[:5]]
    log(f"  High-confidence features ({len(high_conf)}): {', '.join(_hc_parts)}")

    # ------------------------------------------------------------------
    # Round 2 — MEDIATOR classifies
    # ------------------------------------------------------------------
    log("Round 2: MEDIATOR classifying...")
    t_r2 = time.time()
    decision, mediator_text, med_ms = await run_mediator_classify(task, feature_record, matched_rules)
    rounds_completed += 1
    log(f"  Done in {time.time()-t_r2:.1f}s")
    log(f"  Decision: {decision.get('label', '?')}  "
        f"(confidence={decision.get('confidence', 0):.2f})")
    log(f"  Reasoning: {decision.get('reasoning', '')[:120]}")

    # ------------------------------------------------------------------
    # Round 3 — VERIFIER + revision loop
    # ------------------------------------------------------------------
    verification: dict = {}
    revision = 0

    if task.get("few_shot_a") or task.get("few_shot_b"):
        log("Round 3: VERIFIER checking consistency...")
        t_r3 = time.time()
        verification, ver_ms = await run_verifier(task, decision, feature_record)
        rounds_completed += 1
        log(f"  Done in {time.time()-t_r3:.1f}s  |  "
            f"consistent={verification.get('consistent', '?')}")

        while not verification.get("consistent", True) and revision < max_revisions:
            revision += 1
            log(f"  Revision {revision}/{max_revisions}: "
                f"{verification.get('revision_signal', '')[:80]}")
            decision, mediator_text, rev_ms = await run_mediator_revise(
                task, feature_record, matched_rules, decision, verification
            )
            log(f"  Revised decision: {decision.get('label', '?')}  "
                f"(confidence={decision.get('confidence', 0):.2f})")
            # Re-verify
            verification, ver_ms = await run_verifier(task, decision, feature_record)
            log(f"  Re-check: consistent={verification.get('consistent', '?')}")
    else:
        log("Round 3: skipped (no few-shot images provided)")

    # ------------------------------------------------------------------
    # Compute result
    # ------------------------------------------------------------------
    predicted_label = decision.get("label", "uncertain")
    is_correct = (predicted_label == correct_label) if correct_label else None

    ct = get_cost_tracker()
    duration_ms = int((time.time() - t_start) * 1000)

    log(f"\n{'─'*60}")
    log(f"Result: {'CORRECT' if is_correct else 'WRONG'}  "
        f"predicted={predicted_label!r}  actual={correct_label!r}")
    log(f"Cost: ${ct.cost_usd():.4f}  |  "
        f"API calls: {ct.api_calls}  |  "
        f"Duration: {duration_ms/1000:.1f}s")

    # ------------------------------------------------------------------
    # Post-task learning (skipped in test_mode)
    # ------------------------------------------------------------------
    if not test_mode and correct_label:
        # Update rule fire stats
        for m in matched_rules:
            if is_correct:
                rule_engine.record_success(m.rule_id, task_id)
            else:
                rule_engine.record_failure(m.rule_id, task_id)

        rule_engine.increment_tasks_seen(fired_ids=set(fired_ids))

        # Extract new visual dermoscopic rules from this outcome
        extractor_text, _ = await run_rule_extractor(
            task, feature_record, decision, correct_label, is_correct,
            pair_id=pair_id,
        )
        if extractor_text:
            new_rules = rule_engine.parse_mediator_rule_updates(extractor_text, task_id)
            if new_rules:
                new_ids = [r["id"] for r in new_rules]
                log(f"  Extracted {len(new_rules)} new rule(s): {new_ids}")

        # Candidate promotion: promote candidate rules that fired and were confirmed
        if is_correct and matched_rules:
            for m in matched_rules:
                rule = rule_engine.get(m.rule_id)
                if rule and rule.get("status") == "candidate" and rule.get("source_task") != task_id:
                    if rule_engine.promote_candidate(m.rule_id):
                        log(f"  Promoted candidate->active: {m.rule_id}")

        # Pruning
        pruned = rule_engine.auto_deprecate()
        if pruned:
            log(f"  Auto-pruned {len(pruned)} rule(s): {pruned}")

    return {
        "task_id":               task_id,
        "pair_id":               pair_id,
        "predicted_label":       predicted_label,
        "correct_label":         correct_label,
        "correct":               is_correct,
        "confidence":            decision.get("confidence", 0.0),
        "rounds_completed":      rounds_completed,
        "revisions":             revision,
        "duration_ms":           duration_ms,
        "cost_usd":              round(ct.cost_usd(), 6),
        "input_tokens":          ct.input_tokens,
        "cache_creation_tokens": ct.cache_creation_tokens,
        "cache_read_tokens":     ct.cache_read_tokens,
        "output_tokens":         ct.output_tokens,
        "api_calls":             ct.api_calls,
        "model":                 DEFAULT_MODEL,
        "dataset":               dataset,
        "rule_ids_fired":        fired_ids,
        "feature_record":        {k: v for k, v in feature_record.items() if k != "raw_response"},
        "decision":              decision,
        "verification":          verification,
    }
