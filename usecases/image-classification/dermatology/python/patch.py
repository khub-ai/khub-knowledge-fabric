"""
patch.py — Dialogic patching loop for the KF dermatology pipeline.

Workflow:
  1. Run a cheap VLM zero-shot to identify failures.
  2. For each failure, call an expert VLM to author a corrective rule.
  3. Validate the candidate rule against the labeled training pool.
  4. Register accepted rules in the knowledge base.
  5. Re-run the cheap VLM with the new rules and report the improvement.

This demonstrates the core KF value proposition: a cheap VLM + expert-authored
rules can approach the performance of a much more expensive VLM, with the
expensive model's cost incurred only once per rule (not per image).

Usage:
  # Full loop: zero-shot → patch → re-test
  python patch.py --cheap-model o4-mini --expert-model claude-opus-4-6

  # Start from an existing zero-shot results file
  python patch.py --failures-from results_baseline_o4mini_zeroshot.json \\
                  --cheap-model o4-mini --expert-model claude-opus-4-6

  # Dry-run: author and validate rules without registering them
  python patch.py --cheap-model o4-mini --expert-model claude-opus-4-6 --dry-run
"""

from __future__ import annotations
import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_KF_ROOT = _HERE.parents[4]
if str(_KF_ROOT) not in sys.path:
    sys.path.insert(0, str(_KF_ROOT))

import agents
from dataset import load as load_ham10000
from harness import CONFUSABLE_PAIRS

from rich.console import Console
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_TAG = "derm-ham10000"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_api_keys() -> None:
    key_file = Path("P:/_access/Security/api_keys.env")
    if key_file.exists():
        for line in key_file.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip(); v = v.strip()
                if k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY") and not os.environ.get(k):
                    os.environ[k] = v


def _task_from_results_entry(entry: dict, ds, pair_info: dict) -> dict | None:
    """Reconstruct a minimal task dict from a results entry + dataset."""
    task_id = entry["task_id"]
    # task_id format: {pair_id}_{image_id}
    pair_id = entry["pair_id"]
    image_id = task_id[len(pair_id) + 1:]  # strip "pair_id_" prefix

    # Find image in dataset
    dx_a = pair_info.get("dx_a", "")
    dx_b = pair_info.get("dx_b", "")

    for dx in (dx_a, dx_b):
        for split in ("test", "train"):
            for img in ds.images_for_class(dx, split=split):
                if img.image_id == image_id:
                    return {
                        "class_a": pair_info["class_a"],
                        "class_b": pair_info["class_b"],
                        "pair_id": pair_id,
                        "test_image_path": str(img.file_path),
                        "test_label": entry["correct_label"],
                    }
    return None


def _pair_info_for_id(pair_id: str) -> dict | None:
    for cp in CONFUSABLE_PAIRS:
        if cp["pair_id"] == pair_id:
            return cp
    return None


def _init_patch_rules(patch_rules_path: Path) -> "RuleEngine":
    """Create a fresh isolated patch rules file and return its RuleEngine.

    Never touches the main rules.json or any KB JSON file.
    """
    from rules import RuleEngine as _RE
    if patch_rules_path.exists():
        patch_rules_path.unlink()
    return _RE(str(patch_rules_path), dataset_tag=DATASET_TAG)


def _register_rule(rule_engine: "RuleEngine", candidate_rule: dict,
                   pair_id: str, validation: dict,
                   triggered_by: str, cheap_model: str, expert_model: str) -> bool:
    """Register an accepted patch rule into the isolated rule engine.

    Returns True if the rule was added.
    """
    rule_text     = candidate_rule["rule"]
    favors        = candidate_rule["favors"]
    confidence    = candidate_rule.get("confidence", "medium")
    preconditions = candidate_rule.get("preconditions", [])

    # Encode pre-conditions in the condition string so the MEDIATOR sees them.
    precond_str = "; ".join(preconditions) if preconditions else rule_text
    condition   = f"[Patch rule — {pair_id}] {precond_str}"
    action      = (f"Classify as {favors}. Confidence: {confidence}. "
                   f"Rule: {rule_text}")

    result = rule_engine.add_rule(
        condition=condition,
        action=action,
        source=f"expert_vlm:{expert_model}",
        source_task=triggered_by,
        tags=[DATASET_TAG, pair_id, f"patch:{cheap_model}",
              f"triggered_by:{triggered_by}"],
        lineage={"type": "new", "parent_ids": [],
                 "reason": f"patch rule for {pair_id} triggered by {triggered_by}"},
        observability_filter=False,
    )
    return result is not None


def _run_zero_shot_baseline(cheap_model: str, data_dir: str, output: str,
                             max_per_class: int) -> list[dict]:
    """Run cheap model zero-shot baseline and return task results."""
    result = subprocess.run(
        [
            sys.executable, "harness.py",
            "--all", "--max-per-class", str(max_per_class),
            "--mode", "test",
            "--model", cheap_model,
            "--baseline", "zero_shot",
            "--output", output,
            "--data-dir", data_dir,
        ],
        cwd=_HERE,
        capture_output=False,  # show progress to user
        text=True,
    )
    if result.returncode != 0:
        console.print("[red]Zero-shot baseline run failed[/red]")
        sys.exit(1)
    out_path = Path(_HERE) / output
    if not out_path.exists():
        return []
    with open(out_path) as f:
        data = json.load(f)
    return data.get("tasks", [])


def _run_pipeline_with_patch_rules(cheap_model: str, data_dir: str, output: str,
                                    max_per_class: int,
                                    patch_rules_path: Path,
                                    pair: str = "",
                                    failure_task_ids: list[str] | None = None,
                                    ) -> list[dict]:
    """Run cheap model + isolated patch rules through the KF pipeline.

    If *failure_task_ids* is given, only those tasks are re-run (via --task-list).
    This avoids replacing zero-shot successes with potentially wrong KF answers.
    """
    cmd = [
        sys.executable, "harness.py",
        "--max-per-class", str(max_per_class),
        "--mode", "test",
        "--model", cheap_model,
        "--rules", str(patch_rules_path),
        "--output", output,
        "--data-dir", data_dir,
    ]

    # If we have specific failure IDs, write them to a temp file and use --task-list
    task_list_path: Path | None = None
    if failure_task_ids:
        task_list_path = Path(_HERE) / "_patch_rerun_tasks.json"
        with open(task_list_path, "w") as f:
            json.dump(failure_task_ids, f)
        cmd += ["--task-list", str(task_list_path), "--all"]
    elif pair:
        cmd += ["--pair", pair]
    else:
        cmd += ["--all"]

    result = subprocess.run(cmd, cwd=_HERE, capture_output=False, text=True)
    if result.returncode != 0:
        console.print("[red]Pipeline re-run failed[/red]")
        sys.exit(1)

    # Clean up temp task list
    if task_list_path and task_list_path.exists():
        task_list_path.unlink()

    out_path = Path(_HERE) / output
    if not out_path.exists():
        return []
    with open(out_path) as f:
        data = json.load(f)
    return data.get("tasks", [])


# ---------------------------------------------------------------------------
# Core patching loop
# ---------------------------------------------------------------------------

async def run_patch_loop(
    failures: list[dict],
    ds,
    cheap_model: str,
    expert_model: str,
    rule_engine,
    max_val_per_class: int,
    max_authoring_per_class: int,
    max_confirm_per_class: int,
    min_precision: float,
    dry_run: bool,
    validator_model: str = "",
) -> list[dict]:
    """Author, validate, and register rules for each failure.

    Rules are written ONLY to the isolated rule_engine (patch_rules_clean.json).
    The main rules.json and KB JSON files are never touched.

    validator_model: model used for pool validation (RULE_VALIDATOR calls).
      Defaults to expert_model when not specified.  Use a cheaper model here
      (e.g. claude-sonnet-4-6) to avoid routing all pool images through the
      expensive expert or the claude-tutor human-in-the-loop path.

    Returns list of patch records.
    """
    # Pool validation uses a potentially cheaper model than rule authoring.
    # This separates the expensive expert (authoring, semantic validation) from
    # the high-volume binary precondition checks (held-out, authoring, confirm pools).
    _val_model = validator_model or expert_model

    patch_records = []

    for i, failure in enumerate(failures, 1):
        pair_id = failure["pair_id"]
        task_id = failure["task_id"]
        wrong   = failure["predicted_label"]
        correct = failure["correct_label"]

        console.rule(f"[{i}/{len(failures)}] {task_id}")
        console.print(f"  Wrong: [red]{wrong}[/red]  →  Correct: [green]{correct}[/green]")

        pair_info = _pair_info_for_id(pair_id)
        if not pair_info:
            console.print(f"  [yellow]Pair info not found for {pair_id}, skipping[/yellow]")
            continue

        task = _task_from_results_entry(failure, ds, pair_info)
        if not task:
            console.print(f"  [yellow]Image not found for {task_id}, skipping[/yellow]")
            continue

        # --- Step 1: Expert VLM authors a rule ---
        console.print(f"  Calling expert VLM ({expert_model}) to author rule...")
        candidate_rule, _ = await agents.run_expert_rule_author(
            task=task,
            wrong_prediction=wrong,
            correct_label=correct,
            model_reasoning=failure.get("reasoning", ""),
            model=expert_model,
        )
        console.print(f"  Rule: [italic]{candidate_rule.get('rule', '')[:120]}[/italic]")
        console.print(f"  Favors: {candidate_rule.get('favors')} | "
                      f"Confidence: {candidate_rule.get('confidence')}")
        for pc in candidate_rule.get("preconditions", []):
            console.print(f"    Pre-condition: {pc}")

        # --- Step 1b: Rule completion — fill in implicit background conditions ---
        # Save the pre-completion rule as a fallback: if the completer over-tightens
        # (fires_on_trigger=False on the completed rule), it becomes Level 0 in the
        # spectrum — the loosest possible baseline.
        pre_completion_rule = {**candidate_rule, "label": "pre_completion"}
        console.print(f"  Completing rule (filling implicit background conditions)...")
        candidate_rule, _ = await agents.run_rule_completer(
            candidate_rule=candidate_rule,
            pair_info=pair_info,
            model=expert_model,
        )
        added = candidate_rule.get("added_preconditions", [])
        if added:
            console.print(f"  [cyan]{len(added)} background condition(s) added:[/cyan]")
            for pc in added:
                console.print(f"    + {pc}")
            console.print(f"  Completion note: "
                          f"{candidate_rule.get('completion_rationale', '')[:120]}")
        else:
            console.print("  Rule already complete — no additions needed.")

        # --- Step 1d: Semantic rule validation (text-only, no images) ---
        # Now runs on the COMPLETED rule (original + added background conditions).
        console.print(f"  Running semantic validation ({expert_model})...")
        semantic_result, _ = await agents.run_semantic_rule_validator(
            candidate_rule=candidate_rule,
            pair_info=pair_info,
            model=expert_model,
        )
        semantic_overall = semantic_result.get("overall", "accept")
        console.print(f"  Semantic: [bold]{semantic_overall.upper()}[/bold] — "
                      f"{semantic_result.get('rationale', '')[:120]}")
        for pr in semantic_result.get("precondition_ratings", []):
            rating = pr.get("rating", "?")
            color  = "green" if rating == "reliable" else ("red" if rating == "unreliable" else "yellow")
            console.print(f"    [{color}]{rating}[/{color}]: {pr.get('precondition', '')[:80]}")

        if semantic_overall == "reject":
            console.print("  [red]SKIPPING image validation — semantic validator rejected this rule.[/red]")
            _surface_to_expert(candidate_rule,
                               {"fires_on_trigger": False, "tp": 0, "fp": 0,
                                "tn": 0, "fn": 0, "precision": 0.0, "recall": 0.0,
                                "accepted": False,
                                "rejection_reason": "semantic_validation_rejected",
                                "tp_cases": [], "fp_cases": []},
                               pair_info, task_id)
            patch_records.append({
                "task_id":          task_id,
                "pair_id":          pair_id,
                "wrong_prediction": wrong,
                "correct_label":    correct,
                "candidate_rule":   {k: v for k, v in candidate_rule.items()
                                     if k != "raw_response"},
                "semantic_validation": semantic_result,
                "validation":       {"fires_on_trigger": False, "tp": 0, "fp": 0,
                                     "tn": 0, "fn": 0, "precision": 0.0, "recall": 0.0,
                                     "accepted": False,
                                     "rejection_reason": "semantic_validation_rejected"},
                "spectrum_history": [],
                "accepted":         False,
                "registered":       False,
            })
            continue

        # --- Step 2: Sample image pools ---
        #   Authoring pool is sampled lazily — only when contrastive analysis is needed.
        #   This saves 16+ validation calls when the rule passes the held-out gate.
        authoring_pool_cache: dict = {}  # populated lazily by _get_authoring_cases()

        async def _get_authoring_cases(rule_to_check):
            """Lazily sample and validate the authoring pool. Returns (tp_cases, fp_cases)."""
            cache_key = id(rule_to_check)
            if cache_key in authoring_pool_cache:
                return authoring_pool_cache[cache_key]
            auth_imgs_a = [(str(img.file_path), pair_info["class_a"])
                           for img in ds.sample_images(dx_a, max_authoring_per_class,
                                                       split="train", seed=0)]
            auth_imgs_b = [(str(img.file_path), pair_info["class_b"])
                           for img in ds.sample_images(dx_b, max_authoring_per_class,
                                                       split="train", seed=0)]
            authoring_images = auth_imgs_a + auth_imgs_b
            console.print(
                f"  Authoring pool: {len(authoring_images)} images "
                f"({len(auth_imgs_a)} {pair_info['class_a']}, "
                f"{len(auth_imgs_b)} {pair_info['class_b']})"
            )
            authoring_val = await agents.validate_candidate_rule(
                candidate_rule=rule_to_check,
                validation_images=authoring_images,
                trigger_image_path=task["test_image_path"],
                trigger_correct_label=correct,
                model=_val_model,
            )
            tp_c = authoring_val.get("tp_cases", [])
            fp_c = authoring_val.get("fp_cases", [])
            console.print(
                f"  Authoring pool: TP={authoring_val['tp']} FP={authoring_val['fp']} "
                f"precision={authoring_val['precision']:.2f}"
            )
            authoring_pool_cache[cache_key] = (tp_c, fp_c)
            return tp_c, fp_c
        dx_a = pair_info.get("dx_a", "")
        dx_b = pair_info.get("dx_b", "")
        held_imgs_a = [(str(img.file_path), pair_info["class_a"])
                       for img in ds.sample_images(dx_a, max_val_per_class,
                                                   split="train", seed=42)]
        held_imgs_b = [(str(img.file_path), pair_info["class_b"])
                       for img in ds.sample_images(dx_b, max_val_per_class,
                                                   split="train", seed=42)]
        held_out_images = held_imgs_a + held_imgs_b

        # Confirmation pool — sampled lazily after spectrum picks a winner.
        # Different seed (123) and larger size, so a passing rule must
        # generalize beyond the small held-out pool that drove its design.
        confirm_pool_cache: list = []

        async def _get_confirmation_pool() -> list:
            if confirm_pool_cache:
                return confirm_pool_cache
            cf_a = [(str(img.file_path), pair_info["class_a"])
                    for img in ds.sample_images(dx_a, max_confirm_per_class,
                                                split="train", seed=123)]
            cf_b = [(str(img.file_path), pair_info["class_b"])
                    for img in ds.sample_images(dx_b, max_confirm_per_class,
                                                split="train", seed=123)]
            # Exclude images already seen in held-out or authoring pools to keep
            # the confirmation set genuinely fresh.
            seen = {p for p, _ in held_out_images}
            confirm_pool_cache.extend([(p, lbl) for p, lbl in (cf_a + cf_b)
                                        if p not in seen])
            return confirm_pool_cache

        console.print(f"  Held-out pool: {len(held_out_images)} images")

        # --- Step 3: Held-out gate FIRST (binding acceptance gate) ---
        # Run this before the authoring pool — if the rule passes, we're done
        # and save 16 authoring-pool calls.  Early exit on FP > 1.
        console.print("  Running held-out gate...")
        validation = await agents.validate_candidate_rule(
            candidate_rule=candidate_rule,
            validation_images=held_out_images,
            trigger_image_path=task["test_image_path"],
            trigger_correct_label=correct,
            model=_val_model,
        )

        fires_on_trigger = validation["fires_on_trigger"]
        precision        = validation["precision"]
        fp               = validation["fp"]
        accepted         = validation["accepted"] and precision >= min_precision

        console.print(
            f"  Held-out gate: "
            f"TP={validation['tp']} FP={fp} "
            f"TN={validation['tn']} FN={validation['fn']} | "
            f"precision={precision:.2f} recall={validation['recall']:.2f} | "
            f"fires_on_trigger={fires_on_trigger}"
            f"{' (early exit)' if validation.get('early_exited') else ''}"
        )

        spectrum_history: list[dict] = []
        active_rule  = candidate_rule
        best_level   = None   # winning level dict, if any

        # confirmation_validation will be set whenever a confirmation pool run
        # is performed; recorded in the patch record for traceability.
        confirmation_validation: dict | None = None

        if accepted:
            best_level = {"level": 3, "label": "original", **candidate_rule}
        elif not fires_on_trigger:
            # Check whether the completer caused the rejection: if the pre-completion
            # rule fires on the trigger, the completer over-tightened.  In that case,
            # run the spectrum with the pre-completion rule as Level 0 (loosest).
            console.print(
                "  [yellow]fires_on_trigger=False[/yellow] — "
                "checking whether rule completion over-tightened..."
            )
            pre_val = await agents.validate_candidate_rule(
                candidate_rule=pre_completion_rule,
                validation_images=held_out_images,
                trigger_image_path=task["test_image_path"],
                trigger_correct_label=correct,
                model=_val_model,
            )
            if pre_val["fires_on_trigger"]:
                console.print(
                    "  Pre-completion rule fires on trigger "
                    f"(TP={pre_val['tp']} FP={pre_val['fp']} "
                    f"precision={pre_val['precision']:.2f}) — "
                    "completion over-tightened. Running spectrum from pre-completion base..."
                )
                # Re-run spectrum search with pre-completion rule as the anchor.
                # Lazily fetch authoring pool observations for contrastive analysis.
                tp_cases, fp_cases = await _get_authoring_cases(pre_completion_rule)
                contrastive, _ = await agents.run_contrastive_feature_analysis(
                    tp_cases=tp_cases,
                    fp_cases=fp_cases,
                    candidate_rule=pre_completion_rule,
                    pair_info=pair_info,
                    model=expert_model,
                )
                disc_feature = contrastive.get("discriminating_feature")
                console.print(
                    f"  Discriminating feature: "
                    f"[italic]{contrastive['description'][:100]}[/italic] "
                    f"(confidence={contrastive.get('confidence','?')})"
                )
                spectrum_levels, _ = await agents.run_rule_spectrum_generator(
                    candidate_rule=pre_completion_rule,
                    tp_cases=tp_cases,
                    fp_cases=fp_cases,
                    contrastive_result=contrastive,
                    pair_info=pair_info,
                    model=expert_model,
                )
                # Prepend the bare pre-completion rule as Level 0
                level_0 = {**pre_completion_rule, "level": 0, "label": "pre_completion"}
                spectrum_levels = [level_0] + spectrum_levels

                validations = await agents.validate_candidate_rules_batch(
                    rules=spectrum_levels,
                    validation_images=held_out_images,
                    trigger_image_path=task["test_image_path"],
                    trigger_correct_label=correct,
                    model=_val_model,
                )
                for lv, vr in zip(spectrum_levels, validations):
                    lv_accepted = vr["accepted"] and vr["precision"] >= min_precision
                    n_pc = len(lv.get("preconditions", []))
                    console.print(
                        f"    Level {lv['level']} ({lv.get('label','?')}, {n_pc} pre-cond): "
                        f"TP={vr['tp']} FP={vr['fp']} "
                        f"precision={vr['precision']:.2f} "
                        f"fires={vr['fires_on_trigger']} "
                        f"→ {'[green]PASS[/green]' if lv_accepted else '[red]FAIL[/red]'}"
                    )
                    spectrum_history.append({
                        "level": lv["level"], "label": lv.get("label", ""),
                        "n_preconditions": n_pc,
                        "rule": {k: v for k, v in lv.items() if k != "raw_response"},
                        "validation": {k: v for k, v in vr.items()
                                       if k not in ("tp_cases", "fp_cases")},
                        "accepted": lv_accepted,
                    })
                    if lv_accepted and best_level is None:
                        best_level = lv
                        active_rule = lv
                        validation  = vr
                        accepted    = True

                if best_level is None:
                    console.print("  [red]No level passed[/red] — escalating to expert.")
                    best_vr = max(zip(spectrum_levels, validations),
                                  key=lambda x: x[1]["precision"])
                    _surface_to_expert(best_vr[0], best_vr[1], pair_info, task_id)
                else:
                    console.print(
                        f"  [green]ACCEPTED[/green] level {best_level['level']} "
                        f"({best_level.get('label','?')}) — "
                        f"{len(best_level.get('preconditions',[]))} pre-condition(s)"
                    )
            else:
                console.print(
                    "  Pre-completion rule also fails on trigger — "
                    "not a completion artifact. [red]REJECTED[/red]."
                )
        elif fp == 0:
            console.print(f"  [red]REJECTED[/red] — precision {precision:.2f} < {min_precision:.2f} (no FP to analyze)")
        else:
            # --- Step 3c: Spectrum search ---
            # Use the HELD-OUT pool's actual TP/FP cases as the contrastive input —
            # these are the exact rejection-causing samples (the rule wrongly fired
            # on the FP image, that's why we're here).  This gives the contrastive
            # step concrete evidence to reason about, instead of asking it to guess
            # from the authoring pool which had no FPs.
            tp_cases = validation.get("tp_cases", [])
            fp_cases = validation.get("fp_cases", [])

            console.print(
                f"  [yellow]REJECTED[/yellow] — FP={fp}, precision={precision:.2f}. "
                f"Running contrastive analysis + spectrum on the actual "
                f"held-out FP case(s)..."
            )

            # Contrastive analysis to find discriminating feature
            contrastive, _ = await agents.run_contrastive_feature_analysis(
                tp_cases=tp_cases,
                fp_cases=fp_cases,
                candidate_rule=candidate_rule,
                pair_info=pair_info,
                model=expert_model,
            )
            disc_feature = contrastive.get("discriminating_feature")
            console.print(
                f"  Discriminating feature: "
                f"[italic]{contrastive['description'][:100]}[/italic] "
                f"(present in {contrastive.get('present_in','?')} cases, "
                f"confidence={contrastive.get('confidence','?')})"
            )

            if not disc_feature:
                console.print("  Cannot identify discriminating feature — escalating to expert.")
                _surface_to_expert(candidate_rule, validation, pair_info, task_id)
            else:
                # Generate 4-level spectrum in one call
                console.print("  Generating 4-level specificity spectrum...")
                spectrum_levels, _ = await agents.run_rule_spectrum_generator(
                    candidate_rule=candidate_rule,
                    tp_cases=tp_cases,
                    fp_cases=fp_cases,
                    contrastive_result=contrastive,
                    pair_info=pair_info,
                    model=expert_model,
                )
                console.print(f"  Spectrum: {len(spectrum_levels)} level(s) generated")

                # Validate all levels in parallel against the held-out pool
                # (expert never saw these images — genuine held-out gate)
                validations = await agents.validate_candidate_rules_batch(
                    rules=spectrum_levels,
                    validation_images=held_out_images,
                    trigger_image_path=task["test_image_path"],
                    trigger_correct_label=correct,
                    model=_val_model,
                )

                # Report and find the most general passing level
                for lv, vr in zip(spectrum_levels, validations):
                    lv_accepted = vr["accepted"] and vr["precision"] >= min_precision
                    n_pc = len(lv.get("preconditions", []))
                    console.print(
                        f"    Level {lv['level']} ({lv.get('label','?')}, {n_pc} pre-cond): "
                        f"TP={vr['tp']} FP={vr['fp']} "
                        f"precision={vr['precision']:.2f} "
                        f"fires={vr['fires_on_trigger']} "
                        f"→ {'[green]PASS[/green]' if lv_accepted else '[red]FAIL[/red]'}"
                    )
                    spectrum_history.append({
                        "level":      lv["level"],
                        "label":      lv.get("label", ""),
                        "n_preconditions": n_pc,
                        "rule":       {k: v for k, v in lv.items()
                                       if k not in ("raw_response",)},
                        "validation": {k: v for k, v in vr.items()
                                       if k not in ("tp_cases", "fp_cases")},
                        "accepted":   lv_accepted,
                    })
                    # Pick the most general (first in list, ordered 1→4) that passes
                    if lv_accepted and best_level is None:
                        best_level = lv
                        active_rule = lv
                        validation  = vr
                        accepted    = True

                if best_level is None:
                    console.print(
                        f"  [red]No level passed[/red] — "
                        f"escalating to expert."
                    )
                    # Use the best-precision level for the expert surface
                    best_vr = max(zip(spectrum_levels, validations),
                                  key=lambda x: x[1]["precision"])
                    _surface_to_expert(best_vr[0], best_vr[1], pair_info, task_id)
                else:
                    console.print(
                        f"  [green]ACCEPTED[/green] level {best_level['level']} "
                        f"({best_level.get('label','?')}) — "
                        f"{len(best_level.get('preconditions',[]))} pre-condition(s)"
                    )

        # --- Step 3d: Confirmation pool ---
        # Any rule that survived the held-out gate (whether the original or a
        # spectrum-tightened variant) must also pass a LARGER, FRESH pool before
        # we trust it.  The held-out pool is small (16 images by default), and a
        # passing precision there could be a sampling fluke.  The confirmation
        # pool uses a different seed and a larger size to test generalization.
        if accepted and max_confirm_per_class > 0:
            confirm_images = await _get_confirmation_pool()
            console.print(
                f"  Confirmation pool: {len(confirm_images)} fresh images "
                f"(seed=123, {max_confirm_per_class}/class) — "
                f"validating winning rule..."
            )
            confirmation_validation = await agents.validate_candidate_rule(
                candidate_rule=active_rule,
                validation_images=confirm_images,
                trigger_image_path=task["test_image_path"],
                trigger_correct_label=correct,
                model=_val_model,
            )
            cf_precision = confirmation_validation["precision"]
            cf_fp = confirmation_validation["fp"]
            console.print(
                f"  Confirmation: TP={confirmation_validation['tp']} "
                f"FP={cf_fp} TN={confirmation_validation['tn']} "
                f"FN={confirmation_validation['fn']} | "
                f"precision={cf_precision:.2f}"
            )
            if cf_precision < min_precision:
                console.print(
                    f"  [red]CONFIRMATION FAILED[/red] — precision "
                    f"{cf_precision:.2f} < {min_precision:.2f}. "
                    f"Rule did not generalize beyond the held-out pool. "
                    f"Rejecting."
                )
                accepted = False
            else:
                console.print(
                    f"  [green]CONFIRMED[/green] — rule generalizes "
                    f"({cf_precision:.2f} ≥ {min_precision:.2f})"
                )

        record = {
            "task_id":           task_id,
            "pair_id":           pair_id,
            "wrong_prediction":  wrong,
            "correct_label":     correct,
            "candidate_rule":    {k: v for k, v in active_rule.items()
                                  if k != "raw_response"},
            "semantic_validation": semantic_result,
            "validation":        {k: v for k, v in validation.items()
                                  if k not in ("tp_cases", "fp_cases")},
            "confirmation_validation": (
                {k: v for k, v in confirmation_validation.items()
                 if k not in ("tp_cases", "fp_cases")}
                if confirmation_validation else None
            ),
            "spectrum_history":  spectrum_history,
            "accepted":          accepted,
            "registered":        False,
        }

        # --- Step 4: Register in isolated patch rules file ---
        if accepted and not dry_run:
            ok = _register_rule(
                rule_engine=rule_engine,
                candidate_rule=active_rule,
                pair_id=pair_id,
                validation=validation,
                triggered_by=task_id,
                cheap_model=cheap_model,
                expert_model=expert_model,
            )
            record["registered"] = ok
            if ok:
                console.print("  Rule registered in patch rules file.")
            else:
                console.print("  [yellow]Registration failed (observability filter?)[/yellow]")
        elif not accepted:
            console.print(
                f"  [red]REJECTED[/red] — "
                f"precision={validation['precision']:.2f} FP={validation['fp']} "
                f"({'spectrum tried' if spectrum_history else 'no spectrum'})"
            )

        patch_records.append(record)

    return patch_records


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _surface_to_expert(active_rule: dict, validation: dict,
                        pair_info: dict, task_id: str) -> None:
    """Print a structured question for the human expert (or superior VLM) to review.

    This is the dialogic moment: the system cannot automatically resolve the
    conflict and surfaces it with enough context for the expert to make a decision.
    """
    fp_cases   = validation.get("fp_cases", [])
    tp_cases   = validation.get("tp_cases", [])
    favors     = active_rule.get("favors", "?")
    class_a    = pair_info.get("class_a", "?")
    class_b    = pair_info.get("class_b", "?")
    pair_id    = pair_info.get("pair_id", "?")

    console.print("\n  [bold yellow]⚑ EXPERT REVIEW REQUIRED[/bold yellow]")
    console.print(f"  Task:   {task_id}")
    console.print(f"  Rule favors: {favors} (authored for {pair_id})")
    console.print(f"  Problem: rule also fired on {len(fp_cases)} "
                  f"training image(s) of the wrong class:")
    for c in fp_cases:
        console.print(f"    • {c['ground_truth']}: {c.get('observations','')[:120]}")
    if tp_cases:
        console.print(f"  Correct firings ({len(tp_cases)} TP):")
        for c in tp_cases[:3]:
            console.print(f"    • {c['ground_truth']}: {c.get('observations','')[:80]}")
    console.print(
        f"\n  [bold]Question for expert[/bold]: Does the rule pattern genuinely "
        f"distinguish {favors} from {class_a if favors == class_b else class_b}?\n"
        f"  If yes: the pre-conditions need to be tightened to exclude these cases.\n"
        f"  If no: this rule may generalize beyond {pair_id} and should be reviewed "
        f"before cross-pair deployment.\n"
    )


def _detect_cross_pair_firing(rerun_tasks: list[dict],
                               patch_records: list[dict]) -> list[dict]:
    """Check whether patch rules fired on tasks outside their intended pair.

    Builds a map from rule condition prefix to intended pair_id, then checks
    each rerun task's fired rule IDs against that map.

    Returns a list of cross-pair firing events.
    """
    # Build map: rule_id → intended pair_id using the condition prefix
    # registered rules have condition: "[Patch rule — {pair_id}] ..."
    # We match by the pair_id embedded in the registered record.
    # Since we don't store rule_ids in patch_records, we flag any rerun task
    # where rules fired AND the task's pair_id differs from a registered record's pair_id.
    # This is conservative: if only one pair was patched and rerun covers all pairs,
    # any cross-pair firing is meaningful.

    registered_pairs = {rec["pair_id"] for rec in patch_records if rec.get("registered")}
    events = []

    for task in rerun_tasks:
        task_pair   = task.get("pair_id", "")
        fired_rules = task.get("rule_ids_fired", [])
        if not fired_rules:
            continue
        for intended_pair in registered_pairs:
            if intended_pair != task_pair:
                for rid in fired_rules:
                    events.append({
                        "rule_id":       rid,
                        "intended_pair": intended_pair,
                        "fired_on_pair": task_pair,
                        "task_id":       task.get("task_id", ""),
                        "correct":       task.get("correct"),
                        "flag":          (
                            "QUALITY PROBLEM — pre-conditions too broad"
                            if not task.get("correct")
                            else "POSSIBLE GENERALIZATION — worth expert review"
                        ),
                    })

    if events:
        console.print("\n[bold yellow]Cross-pair firing detected:[/bold yellow]")
        for e in events:
            icon = "[red]✗[/red]" if e["flag"].startswith("QUALITY") else "[yellow]?[/yellow]"
            console.print(
                f"  {icon} Rule {e['rule_id']} (for {e['intended_pair']}) "
                f"fired on {e['task_id']} ({e['fired_on_pair']}) — "
                f"correct={e['correct']}"
            )
            console.print(f"    → {e['flag']}")

    return events


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KF dialogic patching loop")
    p.add_argument("--cheap-model",       dest="cheap_model",       default="o4-mini")
    p.add_argument("--expert-model",      dest="expert_model",      default="gpt-4o")
    p.add_argument("--validator-model",   dest="validator_model",   default="",
                   help="Model for pool validation (RULE_VALIDATOR calls). "
                        "Defaults to expert-model when not set. "
                        "Use a cheaper model (e.g. claude-sonnet-4-6) to reduce cost "
                        "since pool validation is high-volume binary precondition checking, "
                        "not creative rule authoring.")
    p.add_argument("--failures-from",     dest="failures_from",     default="",
                   help="Load failures from an existing results JSON")
    p.add_argument("--pair",              default="",
                   help="Limit zero-shot + rerun to a single pair ID")
    p.add_argument("--max-per-class",          dest="max_per_class",          type=int, default=3)
    p.add_argument("--max-val-per-class",      dest="max_val_per_class",      type=int, default=8,
                   help="Held-out validation pool size per class (expert never sees)")
    p.add_argument("--max-authoring-per-class", dest="max_authoring_per_class", type=int, default=8,
                   help="Authoring pool size per class (shown to expert for contrastive analysis)")
    p.add_argument("--max-confirm-per-class",   dest="max_confirm_per_class",   type=int, default=16,
                   help="Confirmation pool size per class (fresh seed, used after spectrum picks a winner; 0 disables)")
    p.add_argument("--min-precision",     dest="min_precision",     type=float, default=0.75)
    p.add_argument("--data-dir",          dest="data_dir",
                   default="C:/_backup/ml/data/DermaMNIST_HAM10000")
    p.add_argument("--patch-rules",       dest="patch_rules",
                   default="patch_rules_clean.json",
                   help="Isolated rules file for this session (never touches main rules.json)")
    p.add_argument("--dry-run",           action="store_true",
                   help="Author and validate rules but do not register them")
    p.add_argument("--output",            default="patch_session.json")
    p.add_argument("--skip-rerun",        dest="skip_rerun", action="store_true")
    p.add_argument("--zero-shot-only",    dest="zero_shot_only", action="store_true",
                   help="Run zero-shot baseline only; save {tasks:[...]} to --output and exit.")
    p.add_argument("--rerun-only",        dest="rerun_only", action="store_true",
                   help="Load --patch-rules and re-classify --failures-from; no patching. "
                        "Use --max-per-class matching the baseline to include all failure IDs.")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    _load_api_keys()

    patch_rules_path = Path(_HERE) / args.patch_rules

    effective_validator = args.validator_model or args.expert_model
    console.rule("[bold]KF Dialogic Patching Loop[/bold]")
    console.print(f"  Cheap model:            [cyan]{args.cheap_model}[/cyan]")
    console.print(f"  Expert model:           [cyan]{args.expert_model}[/cyan]")
    console.print(f"  Validator model:        [cyan]{effective_validator}[/cyan]"
                  + (" [dim](same as expert)[/dim]" if not args.validator_model else ""))
    console.print(f"  Patch rules file:       [cyan]{patch_rules_path}[/cyan]")
    console.print(f"  Authoring pool/class:   {args.max_authoring_per_class} "
                  f"(expert sees; contrastive analysis context)")
    console.print(f"  Held-out pool/class:    {args.max_val_per_class} "
                  f"(expert never sees; precision gate)")
    console.print(f"  Dry-run:                {args.dry_run}")

    # --rerun-only: apply existing rules to failures from a baseline file; no patching
    if args.rerun_only:
        if not args.failures_from:
            console.print("[red]--rerun-only requires --failures-from[/red]")
            sys.exit(1)
        if not patch_rules_path.exists():
            console.print(f"[red]Rules file not found: {patch_rules_path}[/red]")
            sys.exit(1)
        with open(args.failures_from) as f:
            prev_ro = json.load(f)
        all_tasks_ro = prev_ro.get("tasks", [])
        if args.pair:
            all_tasks_ro = [t for t in all_tasks_ro if t.get("pair_id") == args.pair]
        total_ro = len(all_tasks_ro)
        failures_ro = [t for t in all_tasks_ro if not t["correct"]]
        before_correct_ro = total_ro - len(failures_ro)
        console.print(f"\nBaseline: {before_correct_ro}/{total_ro} correct | "
                      f"[red]{len(failures_ro)} failure(s)[/red]")
        console.print(f"Rules: {patch_rules_path.name}")
        if not failures_ro:
            console.print("[green]No failures to rerun.[/green]")
            return
        rerun_out = f"_rerun_only_{Path(args.output).stem}.json"
        step_tasks = _run_pipeline_with_patch_rules(
            cheap_model=args.cheap_model,
            data_dir=args.data_dir,
            output=rerun_out,
            max_per_class=args.max_per_class,
            patch_rules_path=patch_rules_path,
            failure_task_ids=[t["task_id"] for t in failures_ro],
        )
        fixed_ro = sum(1 for t in step_tasks if t["correct"])
        after_correct_ro = before_correct_ro + fixed_ro
        console.rule("[bold]Rerun Summary[/bold]")
        from rich.table import Table as _Table
        tbl = _Table(show_header=True)
        tbl.add_column("Phase"); tbl.add_column("Correct"); tbl.add_column("Accuracy")
        tbl.add_row("Zero-shot baseline", f"{before_correct_ro}/{total_ro}",
                    f"{before_correct_ro/total_ro*100:.1f}%")
        tbl.add_row("After applying rules", f"{after_correct_ro}/{total_ro}",
                    f"{after_correct_ro/total_ro*100:.1f}%")
        tbl.add_row("Delta", f"+{fixed_ro}", f"+{fixed_ro/total_ro*100:.1f}pp")
        console.print(tbl)
        return

    # Load dataset
    console.print(f"\n[dim]Loading HAM10000 from {args.data_dir}...[/dim]")
    ds = load_ham10000(args.data_dir)

    # Step 1: Get failures (zero-shot or from file)
    if args.failures_from:
        console.print(f"\nLoading failures from [cyan]{args.failures_from}[/cyan]...")
        with open(args.failures_from) as f:
            prev = json.load(f)
        all_tasks = prev.get("tasks", [])
    else:
        console.print(f"\n[bold]Step 1[/bold]: Running {args.cheap_model} zero-shot baseline...")
        zs_output = f"patch_zeroshot_{args.cheap_model.replace('-', '_')}.json"
        all_tasks = _run_zero_shot_baseline(
            args.cheap_model, args.data_dir, zs_output, args.max_per_class
        )

    if args.pair:
        all_tasks = [t for t in all_tasks if t.get("pair_id") == args.pair]

    failures = [t for t in all_tasks if not t["correct"]]
    total    = len(all_tasks)
    console.print(f"\nBaseline: {total - len(failures)}/{total} correct | "
                  f"[red]{len(failures)} failure(s) to patch[/red]")

    # --zero-shot-only: save baseline and exit
    if args.zero_shot_only:
        out_path = Path(_HERE) / args.output
        with open(out_path, "w") as f:
            json.dump({"tasks": all_tasks, "total": total, "failures": len(failures)}, f, indent=2)
        console.print(f"Zero-shot baseline saved to [cyan]{args.output}[/cyan]")
        return

    if not failures:
        console.print("[green]No failures — nothing to patch.[/green]")
        return

    # Step 2: Initialise an isolated, empty patch rules file
    console.print(f"\n[bold]Step 2[/bold]: Initialising patch rules file...")
    rule_engine = _init_patch_rules(patch_rules_path)
    console.print(f"  Created empty: {patch_rules_path.name}")

    # Steps 3–6: Process failures ONE AT A TIME.
    # For each failure: author + validate + register rule, then immediately
    # test the cheap model on that specific image with the cumulative patch
    # rules so far.  This gives per-failure feedback before moving on and
    # avoids queuing up all expert calls before any test is run.
    console.print(f"\n[bold]Steps 3–6[/bold]: Per-failure patch + test loop...\n")

    patch_records:     list[dict] = []
    all_rerun_tasks:   list[dict] = []
    cross_pair_events: list[dict] = []
    before_correct = total - len(failures)

    rerun_output = f"patch_rerun_{args.cheap_model.replace('-', '_')}.json"

    for i, failure in enumerate(failures, 1):
        console.rule(f"[bold]Failure {i}/{len(failures)}: {failure['task_id']}[/bold]")

        # --- Author + validate + register rule for this single failure ---
        records = await run_patch_loop(
            failures=[failure],
            ds=ds,
            cheap_model=args.cheap_model,
            expert_model=args.expert_model,
            rule_engine=rule_engine,
            max_val_per_class=args.max_val_per_class,
            max_authoring_per_class=args.max_authoring_per_class,
            max_confirm_per_class=args.max_confirm_per_class,
            min_precision=args.min_precision,
            dry_run=args.dry_run,
            validator_model=args.validator_model,
        )
        patch_records.extend(records)

        rec = records[0] if records else {}
        registered_now = rec.get("registered", False)

        # --- Immediately test: run the cheap model on this failure image ---
        # with all patch rules registered so far.
        if not args.skip_rerun and not args.dry_run:
            console.print(
                f"\n  [bold]Test[/bold]: Re-running {args.cheap_model} on "
                f"[cyan]{failure['task_id']}[/cyan] with cumulative patch rules..."
            )
            step_tasks = _run_pipeline_with_patch_rules(
                cheap_model=args.cheap_model,
                data_dir=args.data_dir,
                output=rerun_output,
                max_per_class=args.max_per_class,
                patch_rules_path=patch_rules_path,
                failure_task_ids=[failure["task_id"]],
            )
            all_rerun_tasks.extend(step_tasks)

            fixed_now = sum(1 for t in step_tasks if t["correct"])
            if fixed_now:
                console.print(
                    f"  [green]✓ FIXED[/green] — {args.cheap_model} now correctly "
                    f"classifies {failure['task_id']}"
                )
            else:
                reason = "rule not registered" if not registered_now else "rule registered but cheap model still failed"
                console.print(
                    f"  [red]✗ STILL FAILING[/red] — {reason}"
                )
                if i < len(failures):
                    console.print(
                        f"  [yellow]Patch did not fix this failure. "
                        f"Stopping before remaining {len(failures) - i} failure(s). "
                        f"Review the rule and re-run with --failures-from to resume.[/yellow]"
                    )
                    # Save state and exit the loop — no point continuing if
                    # the patch strategy is not working.
                    n_accepted   = sum(1 for r in patch_records if r["accepted"])
                    n_registered = sum(1 for r in patch_records if r["registered"])
                    session = {
                        "cheap_model":        args.cheap_model,
                        "expert_model":       args.expert_model,
                        "validator_model":    effective_validator,
                        "patch_rules_file":   str(patch_rules_path),
                        "dry_run":            args.dry_run,
                        "total_tasks":        total,
                        "failures_before":    len(failures),
                        "failures_processed": i,
                        "stopped_early":      True,
                        "stop_reason":        reason,
                        "rules_authored":     len(patch_records),
                        "rules_accepted":     n_accepted,
                        "rules_registered":   n_registered,
                        "cross_pair_events":  cross_pair_events,
                        "patch_records":      patch_records,
                    }
                    with open(args.output, "w") as f:
                        json.dump(session, f, indent=2)
                    console.print(f"\nSession saved to [cyan]{args.output}[/cyan]")
                    return

            # Cross-pair check for rules registered in this step
            step_events = _detect_cross_pair_firing(step_tasks, records)
            cross_pair_events.extend(step_events)
            if step_events:
                console.print(f"  [yellow]⚠ Cross-pair firing on {len(step_events)} task(s)[/yellow]")

        # Save incremental session state after each failure so progress is
        # preserved even if a later failure causes an error.
        n_accepted   = sum(1 for r in patch_records if r["accepted"])
        n_registered = sum(1 for r in patch_records if r["registered"])
        session = {
            "cheap_model":       args.cheap_model,
            "expert_model":      args.expert_model,
            "validator_model":   effective_validator,
            "patch_rules_file":  str(patch_rules_path),
            "dry_run":           args.dry_run,
            "total_tasks":       total,
            "failures_before":   len(failures),
            "failures_processed": i,
            "rules_authored":    len(patch_records),
            "rules_accepted":    n_accepted,
            "rules_registered":  n_registered,
            "cross_pair_events": cross_pair_events,
            "patch_records":     patch_records,
        }
        with open(args.output, "w") as f:
            json.dump(session, f, indent=2)

    # --- Final summary ---
    accepted   = [r for r in patch_records if r["accepted"]]
    registered = [r for r in patch_records if r["registered"]]

    rerun_fixed   = sum(1 for t in all_rerun_tasks if t["correct"])
    after_correct = before_correct + rerun_fixed

    if cross_pair_events:
        console.print("\n[bold yellow]⚠ Cross-pair rule firing detected:[/bold yellow]")
        for ev in cross_pair_events:
            console.print(
                f"  Rule intended for [cyan]{ev['intended_pair']}[/cyan] "
                f"fired on [magenta]{ev['fired_on_pair']}[/magenta] "
                f"(task {ev['task_id']}) → {ev['flag']}"
            )
        console.print(
            "  [dim]→ Surface to human expert: the pupil has been enlightened, "
            "the master must confirm or refine.[/dim]"
        )

    console.rule("[bold]Patch Summary[/bold]")
    tbl = Table(show_header=True, header_style="bold")
    tbl.add_column("Phase"); tbl.add_column("Correct"); tbl.add_column("Accuracy")
    tbl.add_row("Before patching (zero-shot)",
                f"{before_correct}/{total}", f"{before_correct/total*100:.1f}%")
    if all_rerun_tasks:
        tbl.add_row("After patching (KF on failures only)",
                    f"{after_correct}/{total}", f"{after_correct/total*100:.1f}%")
        tbl.add_row("Delta", f"{rerun_fixed:+d}", f"{rerun_fixed/total*100:+.1f}pp")
        tbl.add_row("Failures patched",
                    f"{rerun_fixed}/{len(failures)}", "")
    tbl.add_row("Rules authored/accepted/registered",
                f"{len(patch_records)}/{len(accepted)}/{len(registered)}", "")
    console.print(tbl)

    if args.dry_run:
        console.print("\n[yellow]Dry-run: no rules registered.[/yellow]")

    console.print(f"\nSession saved to [cyan]{args.output}[/cyan]")


if __name__ == "__main__":
    asyncio.run(main())
