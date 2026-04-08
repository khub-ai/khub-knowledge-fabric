"""
patch.py — Dialogic patching loop for the KF bird (CUB-200-2011) pipeline.

Parallel to usecases/image-classification/dermatology/python/patch.py but for
bird species identification.  Uses CUB-200-2011 data and ornithological prompts.

Workflow:
  1. Run a cheap VLM (e.g. Qwen3-VL-8B) zero-shot to identify failures.
  2. For each failure, call an expert VLM (or claude-tutor) to author a rule.
  3. Validate the candidate rule against the labeled training pool.
  4. Register accepted rules.
  5. Re-run the cheap VLM with the rules and report improvement.

Usage:
  # Cheap model + API expert + cheap validator
  python patch.py \\
      --cheap-model qwen/qwen3-vl-8b-instruct \\
      --expert-model claude-sonnet-4-6 \\
      --validator-model claude-sonnet-4-6 \\
      --pair bronzed_cowbird_vs_shiny_cowbird

  # Human-in-the-loop tutor
  python patch.py \\
      --cheap-model qwen/qwen3-vl-8b-instruct \\
      --expert-model claude-tutor \\
      --validator-model claude-sonnet-4-6 \\
      --pair bronzed_cowbird_vs_shiny_cowbird

  # Load failures from an existing baseline file
  python patch.py \\
      --failures-from results_baseline_qwen3vl8b.json \\
      --cheap-model qwen/qwen3-vl-8b-instruct \\
      --expert-model claude-tutor \\
      --validator-model claude-sonnet-4-6
"""

from __future__ import annotations
import argparse
import asyncio
import io
import json
import os
import sys
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_HERE    = Path(__file__).resolve().parent   # birds/python/
_UC_DIR  = _HERE.parents[1]                # usecases/image-classification/
_KF_ROOT = _HERE.parents[3]               # repo root (khub-knowledge-fabric/)
# Insert in reverse priority order (last insert = highest priority)
for _p in (str(_KF_ROOT), str(_UC_DIR / "src"), str(_UC_DIR / "python"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import agents
from dataset import load as load_cub, DEFAULT_DATA_DIR, CUBDataset

from confusable_pairs import CONFUSABLE_PAIRS, ConfusablePair

from rich.console import Console
from rich.table import Table

console = Console()

DATASET_TAG = "bird-uc200"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_api_keys() -> None:
    key_file = Path("P:/_access/Security/api_keys.env")
    if key_file.exists():
        for line in key_file.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
                    if not os.environ.get(k):
                        os.environ[k] = v


def _pair_by_id(pair_id: str) -> ConfusablePair | None:
    for cp in CONFUSABLE_PAIRS:
        cid = (
            f"{cp.class_name_a.lower().replace(' ', '_')}"
            f"_vs_"
            f"{cp.class_name_b.lower().replace(' ', '_')}"
        )
        if cid == pair_id:
            return cp
    return None


def _pair_id(cp: ConfusablePair) -> str:
    return (
        f"{cp.class_name_a.lower().replace(' ', '_')}"
        f"_vs_"
        f"{cp.class_name_b.lower().replace(' ', '_')}"
    )


def _pair_info(cp: ConfusablePair) -> dict:
    return {
        "class_a": cp.class_name_a,
        "class_b": cp.class_name_b,
        "class_id_a": cp.class_id_a,
        "class_id_b": cp.class_id_b,
        "pair_id": _pair_id(cp),
    }


# ---------------------------------------------------------------------------
# Zero-shot baseline (direct, no harness subprocess)
# ---------------------------------------------------------------------------

_ZERO_SHOT_SYSTEM = """\
You are a bird identification expert. You will be shown a photograph of a bird.
Identify the species from the provided choices.

Output ONLY a JSON object:
{
  "prediction": "<species name — exactly as given in the choices>",
  "confidence": "high" | "medium" | "low",
  "reasoning": "Brief note on the key field marks you used."
}
"""


async def _classify_zero_shot(
    image_path: str,
    class_a: str,
    class_b: str,
    model: str,
) -> tuple[dict, int]:
    """Classify one bird image as class_a or class_b, zero-shot."""
    content = [
        agents._image_block(image_path),
        {
            "type": "text",
            "text": (
                f"Choices: {class_a} OR {class_b}\n\n"
                f"Which species is in this photograph? "
                f"Choose EXACTLY one of the two options above."
            ),
        },
    ]
    text, ms = await agents.call_agent(
        "BASELINE_CLASSIFIER",
        content,
        system_prompt=_ZERO_SHOT_SYSTEM,
        model=model,
        max_tokens=512,
    )
    result = agents._parse_json_block(text)
    if result and "prediction" in result:
        # Normalise: snap prediction to the closest class name
        pred = result["prediction"].strip()
        if class_a.lower() in pred.lower():
            result["prediction"] = class_a
        elif class_b.lower() in pred.lower():
            result["prediction"] = class_b
        return result, ms

    # Fallback: look for class name in raw text
    text_lower = text.lower()
    if class_a.lower() in text_lower and class_b.lower() not in text_lower:
        return {"prediction": class_a, "confidence": "low", "reasoning": text}, ms
    if class_b.lower() in text_lower and class_a.lower() not in text_lower:
        return {"prediction": class_b, "confidence": "low", "reasoning": text}, ms
    return {"prediction": class_a, "confidence": "low", "reasoning": text}, ms


async def run_zero_shot_baseline(
    ds: CUBDataset,
    pairs: list[ConfusablePair],
    cheap_model: str,
    max_per_class: int,
    split: str = "test",
) -> list[dict]:
    """Run cheap model zero-shot on all pairs and return task result dicts."""
    tasks = []
    for cp in pairs:
        pi = _pair_info(cp)
        class_a, class_b = pi["class_a"], pi["class_b"]
        id_a, id_b = pi["class_id_a"], pi["class_id_b"]
        pair_id = pi["pair_id"]

        images_a = ds.sample_images(id_a, max_per_class, split=split, seed=0)
        images_b = ds.sample_images(id_b, max_per_class, split=split, seed=0)

        for img in images_a:
            tasks.append((pair_id, pi, str(img.file_path), class_a))
        for img in images_b:
            tasks.append((pair_id, pi, str(img.file_path), class_b))

    results = []
    for pair_id, pi, img_path, ground_truth in tasks:
        class_a, class_b = pi["class_a"], pi["class_b"]
        result, ms = await _classify_zero_shot(img_path, class_a, class_b, cheap_model)
        prediction = result.get("prediction", "")
        correct = prediction == ground_truth
        img_id = Path(img_path).stem
        results.append({
            "task_id": f"{pair_id}_{img_id}",
            "pair_id": pair_id,
            "image_path": img_path,
            "correct_label": ground_truth,
            "predicted_label": prediction,
            "correct": correct,
            "confidence": result.get("confidence", ""),
            "reasoning": result.get("reasoning", ""),
            "duration_ms": ms,
        })
        status = "[green]OK[/green]" if correct else "[red]WRONG[/red]"
        console.print(
            f"  {status} [{pair_id}] {Path(img_path).name}: "
            f"{ground_truth} -> {prediction}"
        )
    return results


# ---------------------------------------------------------------------------
# Rerun with patch rules
# ---------------------------------------------------------------------------

_PATCHED_CLASSIFIER_SYSTEM = """\
You are a bird identification expert. You will be shown a photograph of a bird.

Identify the species from the provided choices.

IMPORTANT: You have been given a set of MANDATORY identification rules below.
For each rule, check whether ALL of its pre-conditions are met in the photograph.
If a rule's pre-conditions are ALL met, you MUST apply that rule's classification.
If no rule fires, use your own expert judgment.

Output ONLY a JSON object:
{
  "prediction": "<species name — exactly as given in the choices>",
  "rule_fired": "<rule_id that fired, or null>",
  "confidence": "high" | "medium" | "low",
  "reasoning": "Brief note on what led to your answer."
}
"""


def _format_patch_rules(rules: list[dict]) -> str:
    """Format registered patch rules for injection into classifier prompt."""
    if not rules:
        return "(no patch rules registered)"
    lines = []
    for i, r in enumerate(rules, 1):
        rule_id = r.get("id", f"r_{i:03d}")
        condition = r.get("condition", "")
        action = r.get("action", "")
        # Extract pre-conditions from condition string
        # Condition format: "[Patch rule — pair_id] precond1; precond2; ..."
        import re
        m = re.match(r"\[Patch rule — [^\]]+\]\s*(.+)", condition, re.DOTALL)
        precond_str = m.group(1) if m else condition
        preconditions = [p.strip() for p in precond_str.split(";") if p.strip()]
        lines.append(f"Rule {rule_id}:")
        lines.append(f"  Action: {action}")
        lines.append(f"  Pre-conditions (ALL must be met):")
        for pc in preconditions:
            lines.append(f"    - {pc}")
    return "\n".join(lines)


async def _classify_with_rules(
    image_path: str,
    class_a: str,
    class_b: str,
    rules: list[dict],
    model: str,
) -> tuple[dict, int]:
    """Classify a bird image with patch rules injected as hard gates."""
    rules_text = _format_patch_rules(rules)
    content = [
        agents._image_block(image_path),
        {
            "type": "text",
            "text": (
                f"Choices: {class_a} OR {class_b}\n\n"
                f"PATCH RULES:\n{rules_text}\n\n"
                f"Check each rule's pre-conditions against what you can observe "
                f"in this photograph. If a rule fires, apply it. "
                f"Otherwise, use your own judgment."
            ),
        },
    ]
    text, ms = await agents.call_agent(
        "PATCHED_CLASSIFIER",
        content,
        system_prompt=_PATCHED_CLASSIFIER_SYSTEM,
        model=model,
        max_tokens=512,
    )
    result = agents._parse_json_block(text)
    if result and "prediction" in result:
        pred = result["prediction"].strip()
        if class_a.lower() in pred.lower():
            result["prediction"] = class_a
        elif class_b.lower() in pred.lower():
            result["prediction"] = class_b
        return result, ms

    text_lower = text.lower()
    if class_a.lower() in text_lower and class_b.lower() not in text_lower:
        return {"prediction": class_a, "rule_fired": None, "confidence": "low", "reasoning": text}, ms
    if class_b.lower() in text_lower and class_a.lower() not in text_lower:
        return {"prediction": class_b, "rule_fired": None, "confidence": "low", "reasoning": text}, ms
    return {"prediction": class_a, "rule_fired": None, "confidence": "low", "reasoning": text}, ms


async def rerun_with_patch_rules(
    failure: dict,
    all_registered_rules: list[dict],
    cheap_model: str,
) -> dict:
    """Re-classify a single failure image with registered patch rules active."""
    pair_id = failure["pair_id"]
    img_path = failure["image_path"]
    ground_truth = failure["correct_label"]

    cp = _pair_by_id(pair_id)
    if not cp:
        return {"task_id": failure["task_id"], "correct": False,
                "predicted_label": "?", "error": f"pair not found: {pair_id}"}

    class_a, class_b = cp.class_name_a, cp.class_name_b

    # Only use rules relevant to this pair
    pair_rules = [
        r for r in all_registered_rules
        if pair_id in r.get("condition", "")
    ]

    result, ms = await _classify_with_rules(
        img_path, class_a, class_b, pair_rules, cheap_model
    )
    prediction = result.get("prediction", "")
    correct = prediction == ground_truth
    return {
        "task_id": failure["task_id"],
        "pair_id": pair_id,
        "image_path": img_path,
        "correct_label": ground_truth,
        "predicted_label": prediction,
        "correct": correct,
        "rule_fired": result.get("rule_fired"),
        "confidence": result.get("confidence", ""),
        "reasoning": result.get("reasoning", ""),
        "duration_ms": ms,
    }


# ---------------------------------------------------------------------------
# Rule registration
# ---------------------------------------------------------------------------

_registered_rules: list[dict] = []   # in-memory registry for this session
_rule_counter = 0


def _register_rule(
    candidate_rule: dict,
    pair_id: str,
    cheap_model: str,
    expert_model: str,
) -> dict | None:
    """Register an accepted patch rule and return its entry."""
    global _rule_counter
    _rule_counter += 1
    rule_id = f"r_{_rule_counter:03d}"

    rule_text = candidate_rule["rule"]
    favors = candidate_rule["favors"]
    confidence = candidate_rule.get("confidence", "medium")
    preconditions = candidate_rule.get("preconditions", [])

    precond_str = "; ".join(preconditions) if preconditions else rule_text
    condition = f"[Patch rule — {pair_id}] {precond_str}"
    action = f"Classify as {favors}. Confidence: {confidence}. Rule: {rule_text}"

    entry = {
        "id": rule_id,
        "condition": condition,
        "action": action,
        "favors": favors,
        "pair_id": pair_id,
        "source": f"expert:{expert_model}",
        "triggered_by": pair_id,
        "preconditions": preconditions,
        "rule_text": rule_text,
    }
    _registered_rules.append(entry)
    return entry


def _save_patch_rules(path: Path) -> None:
    path.write_text(json.dumps(_registered_rules, indent=2))


# ---------------------------------------------------------------------------
# Core patching loop (mirrors dermatology/python/patch.py)
# ---------------------------------------------------------------------------

async def run_patch_loop(
    failures: list[dict],
    ds: CUBDataset,
    cheap_model: str,
    expert_model: str,
    max_val_per_class: int,
    max_authoring_per_class: int,
    max_confirm_per_class: int,
    min_precision: float,
    dry_run: bool,
    validator_model: str = "",
) -> list[dict]:
    """Author, validate, and register rules for each failure.

    validator_model: model for pool validation. Defaults to expert_model.
    Returns list of patch records.
    """
    _val_model = validator_model or expert_model
    patch_records = []

    for i, failure in enumerate(failures, 1):
        pair_id = failure["pair_id"]
        task_id = failure["task_id"]
        wrong   = failure["predicted_label"]
        correct = failure["correct_label"]
        img_path = failure["image_path"]

        console.rule(f"[{i}/{len(failures)}] {task_id}")
        console.print(f"  Wrong: [red]{wrong}[/red]  →  Correct: [green]{correct}[/green]")

        cp = _pair_by_id(pair_id)
        if not cp:
            console.print(f"  [yellow]Pair not found for {pair_id}, skipping[/yellow]")
            continue

        pi = _pair_info(cp)
        task = {
            "class_a": pi["class_a"],
            "class_b": pi["class_b"],
            "pair_id": pair_id,
            "test_image_path": img_path,
            "test_label": correct,
        }

        # --- Step 1a: Expert rule authoring ---
        console.print(f"  Calling expert VLM ({expert_model}) to author rule...")
        candidate_rule, _ = await agents.run_expert_rule_author(
            task=task,
            wrong_prediction=wrong,
            correct_label=correct,
            model_reasoning=failure.get("reasoning", ""),
            model=expert_model,
        )
        console.print(f"  Rule: [italic]{candidate_rule.get('rule', '')[:120]}[/italic]")
        console.print(
            f"  Favors: {candidate_rule.get('favors')} | "
            f"Confidence: {candidate_rule.get('confidence')}"
        )
        for pc in candidate_rule.get("preconditions", []):
            console.print(f"    Pre-condition: {pc}")

        # --- Step 1b: Rule completion ---
        pre_completion_rule = {**candidate_rule, "label": "pre_completion"}
        console.print("  Completing rule (filling implicit background conditions)...")
        candidate_rule, _ = await agents.run_rule_completer(
            candidate_rule=candidate_rule,
            pair_info=pi,
            model=expert_model,
        )
        added = candidate_rule.get("added_preconditions", [])
        if added:
            console.print(f"  [cyan]{len(added)} background condition(s) added:[/cyan]")
            for pc in added:
                console.print(f"    + {pc}")
        else:
            console.print("  Rule already complete — no additions needed.")

        # --- Step 1c: Semantic validation ---
        console.print(f"  Running semantic validation ({expert_model})...")
        semantic_result, _ = await agents.run_semantic_rule_validator(
            candidate_rule=candidate_rule,
            pair_info=pi,
            model=expert_model,
        )
        semantic_overall = semantic_result.get("overall", "accept")
        console.print(
            f"  Semantic: [bold]{semantic_overall.upper()}[/bold] — "
            f"{semantic_result.get('rationale', '')[:120]}"
        )
        for pr in semantic_result.get("precondition_ratings", []):
            rating = pr.get("rating", "?")
            color = "green" if rating == "reliable" else (
                "red" if rating == "unreliable" else "yellow"
            )
            console.print(f"    [{color}]{rating}[/{color}]: {pr.get('precondition', '')[:80]}")

        if semantic_overall == "reject":
            console.print(
                "  [red]SKIPPING image validation — semantic validator rejected.[/red]"
            )
            patch_records.append({
                "task_id": task_id, "pair_id": pair_id,
                "wrong_prediction": wrong, "correct_label": correct,
                "candidate_rule": {k: v for k, v in candidate_rule.items()
                                   if k != "raw_response"},
                "semantic_validation": semantic_result,
                "validation": {"fires_on_trigger": False, "accepted": False,
                               "rejection_reason": "semantic_validation_rejected"},
                "spectrum_history": [], "accepted": False, "registered": False,
            })
            continue

        # --- Step 2: Sample validation pools ---
        id_a, id_b = pi["class_id_a"], pi["class_id_b"]

        held_imgs_a = [
            (str(img.file_path), pi["class_a"])
            for img in ds.sample_images(id_a, max_val_per_class, split="train", seed=42)
        ]
        held_imgs_b = [
            (str(img.file_path), pi["class_b"])
            for img in ds.sample_images(id_b, max_val_per_class, split="train", seed=42)
        ]
        held_out_images = held_imgs_a + held_imgs_b
        console.print(f"  Held-out pool: {len(held_out_images)} images")

        # Authoring pool (lazily sampled for contrastive analysis only)
        authoring_pool_cache: dict = {}

        async def _get_authoring_cases(rule_to_check):
            cache_key = id(rule_to_check)
            if cache_key in authoring_pool_cache:
                return authoring_pool_cache[cache_key]
            auth_a = [(str(img.file_path), pi["class_a"])
                      for img in ds.sample_images(id_a, max_authoring_per_class,
                                                  split="train", seed=0)]
            auth_b = [(str(img.file_path), pi["class_b"])
                      for img in ds.sample_images(id_b, max_authoring_per_class,
                                                  split="train", seed=0)]
            auth_val = await agents.validate_candidate_rule(
                candidate_rule=rule_to_check,
                validation_images=auth_a + auth_b,
                trigger_image_path=img_path,
                trigger_correct_label=correct,
                model=_val_model,
            )
            tp_c = auth_val.get("tp_cases", [])
            fp_c = auth_val.get("fp_cases", [])
            authoring_pool_cache[cache_key] = (tp_c, fp_c)
            return tp_c, fp_c

        # Confirmation pool (lazily sampled after held-out gate passes)
        confirm_pool_cache: list = []

        async def _get_confirmation_pool() -> list:
            if confirm_pool_cache:
                return confirm_pool_cache
            cf_a = [(str(img.file_path), pi["class_a"])
                    for img in ds.sample_images(id_a, max_confirm_per_class,
                                                split="train", seed=123)]
            cf_b = [(str(img.file_path), pi["class_b"])
                    for img in ds.sample_images(id_b, max_confirm_per_class,
                                                split="train", seed=123)]
            seen = {p for p, _ in held_out_images}
            confirm_pool_cache.extend(
                [(p, lbl) for p, lbl in (cf_a + cf_b) if p not in seen]
            )
            return confirm_pool_cache

        # --- Step 3: Held-out gate ---
        console.print("  Running held-out gate...")
        validation = await agents.validate_candidate_rule(
            candidate_rule=candidate_rule,
            validation_images=held_out_images,
            trigger_image_path=img_path,
            trigger_correct_label=correct,
            model=_val_model,
        )

        fires_on_trigger = validation["fires_on_trigger"]
        precision = validation["precision"]
        fp = validation["fp"]
        accepted = validation["accepted"] and precision >= min_precision

        console.print(
            f"  Held-out gate: "
            f"TP={validation['tp']} FP={fp} "
            f"TN={validation['tn']} FN={validation['fn']} | "
            f"precision={precision:.2f} recall={validation['recall']:.2f} | "
            f"fires_on_trigger={fires_on_trigger}"
        )

        spectrum_history: list[dict] = []
        active_rule = candidate_rule
        best_level = None
        confirmation_validation = None

        if accepted:
            best_level = {"level": 3, "label": "original", **candidate_rule}
        elif not fires_on_trigger:
            # Check if completion over-tightened
            console.print(
                "  [yellow]fires_on_trigger=False[/yellow] — "
                "checking pre-completion rule..."
            )
            pre_val = await agents.validate_candidate_rule(
                candidate_rule=pre_completion_rule,
                validation_images=held_out_images,
                trigger_image_path=img_path,
                trigger_correct_label=correct,
                model=_val_model,
            )
            if pre_val["fires_on_trigger"]:
                console.print(
                    f"  Pre-completion rule fires (precision={pre_val['precision']:.2f}) "
                    f"— completion over-tightened. Running spectrum..."
                )
                tp_cases, fp_cases = await _get_authoring_cases(pre_completion_rule)
                contrastive, _ = await agents.run_contrastive_feature_analysis(
                    tp_cases=tp_cases, fp_cases=fp_cases,
                    candidate_rule=pre_completion_rule,
                    pair_info=pi, model=expert_model,
                )
                spectrum_levels, _ = await agents.run_rule_spectrum_generator(
                    candidate_rule=pre_completion_rule,
                    tp_cases=tp_cases, fp_cases=fp_cases,
                    contrastive_result=contrastive,
                    pair_info=pi, model=expert_model,
                )
                level_0 = {**pre_completion_rule, "level": 0, "label": "pre_completion"}
                spectrum_levels = [level_0] + spectrum_levels
                validations = await agents.validate_candidate_rules_batch(
                    candidate_rules=spectrum_levels,
                    validation_images=held_out_images,
                    trigger_image_path=img_path,
                    trigger_correct_label=correct,
                    model=_val_model,
                )
                for lv, vr in zip(spectrum_levels, validations):
                    lv_acc = vr["accepted"] and vr["precision"] >= min_precision
                    n_pc = len(lv.get("preconditions", []))
                    console.print(
                        f"    Level {lv['level']} ({lv.get('label','?')}, {n_pc} pre-cond): "
                        f"TP={vr['tp']} FP={vr['fp']} "
                        f"precision={vr['precision']:.2f} "
                        f"fires={vr['fires_on_trigger']} "
                        f"→ {'[green]PASS[/green]' if lv_acc else '[red]FAIL[/red]'}"
                    )
                    spectrum_history.append({
                        "level": lv["level"], "label": lv.get("label", ""),
                        "n_preconditions": n_pc,
                        "rule": {k: v for k, v in lv.items() if k != "raw_response"},
                        "validation": {k: v for k, v in vr.items()
                                       if k not in ("tp_cases", "fp_cases")},
                        "accepted": lv_acc,
                    })
                    if lv_acc and best_level is None:
                        best_level = lv
                        active_rule = lv
                        validation = vr
                        accepted = True

                if best_level is None:
                    console.print("  [red]No level passed[/red] — escalating to expert.")
                    best_vr = max(zip(spectrum_levels, validations),
                                  key=lambda x: x[1]["precision"])
                    _surface_to_expert(best_vr[0], best_vr[1], pi, task_id)
            else:
                console.print(
                    "  Pre-completion also fails on trigger — not a completion artifact. "
                    "[red]REJECTED[/red]."
                )
        elif fp == 0:
            console.print(
                f"  [red]REJECTED[/red] — precision {precision:.2f} < {min_precision:.2f} "
                f"(no FP to analyze)"
            )
        else:
            # Spectrum search
            tp_cases = validation.get("tp_cases", [])
            fp_cases = validation.get("fp_cases", [])
            console.print(
                f"  [yellow]REJECTED[/yellow] — FP={fp}, precision={precision:.2f}. "
                f"Running contrastive analysis + spectrum..."
            )
            contrastive, _ = await agents.run_contrastive_feature_analysis(
                tp_cases=tp_cases, fp_cases=fp_cases,
                candidate_rule=candidate_rule,
                pair_info=pi, model=expert_model,
            )
            disc_feature = contrastive.get("discriminating_feature")
            console.print(
                f"  Discriminating feature: "
                f"[italic]{contrastive['description'][:100]}[/italic] "
                f"(present in {contrastive.get('present_in','?')} cases, "
                f"confidence={contrastive.get('confidence','?')})"
            )

            if not disc_feature:
                console.print("  No feature identified — escalating to expert.")
                _surface_to_expert(candidate_rule, validation, pi, task_id)
            else:
                console.print("  Generating 4-level specificity spectrum...")
                spectrum_levels, _ = await agents.run_rule_spectrum_generator(
                    candidate_rule=candidate_rule,
                    tp_cases=tp_cases, fp_cases=fp_cases,
                    contrastive_result=contrastive,
                    pair_info=pi, model=expert_model,
                )
                console.print(f"  Spectrum: {len(spectrum_levels)} level(s) generated")

                validations = await agents.validate_candidate_rules_batch(
                    candidate_rules=spectrum_levels,
                    validation_images=held_out_images,
                    trigger_image_path=img_path,
                    trigger_correct_label=correct,
                    model=_val_model,
                )

                for lv, vr in zip(spectrum_levels, validations):
                    lv_acc = vr["accepted"] and vr["precision"] >= min_precision
                    n_pc = len(lv.get("preconditions", []))
                    console.print(
                        f"    Level {lv['level']} ({lv.get('label','?')}, {n_pc} pre-cond): "
                        f"TP={vr['tp']} FP={vr['fp']} "
                        f"precision={vr['precision']:.2f} "
                        f"fires={vr['fires_on_trigger']} "
                        f"→ {'[green]PASS[/green]' if lv_acc else '[red]FAIL[/red]'}"
                    )
                    spectrum_history.append({
                        "level": lv["level"], "label": lv.get("label", ""),
                        "n_preconditions": n_pc,
                        "rule": {k: v for k, v in lv.items() if k != "raw_response"},
                        "validation": {k: v for k, v in vr.items()
                                       if k not in ("tp_cases", "fp_cases")},
                        "accepted": lv_acc,
                    })
                    if lv_acc and best_level is None:
                        best_level = lv
                        active_rule = lv
                        validation = vr
                        accepted = True

                if best_level is None:
                    console.print("  [red]No level passed[/red] — escalating to expert.")
                    best_vr = max(zip(spectrum_levels, validations),
                                  key=lambda x: x[1]["precision"])
                    _surface_to_expert(best_vr[0], best_vr[1], pi, task_id)
                else:
                    console.print(
                        f"  [green]ACCEPTED[/green] level {best_level['level']} "
                        f"({best_level.get('label','?')}) — "
                        f"{len(best_level.get('preconditions',[]))} pre-condition(s)"
                    )

        # --- Step 3d: Confirmation pool ---
        if accepted and max_confirm_per_class > 0:
            confirm_images = await _get_confirmation_pool()
            console.print(
                f"  Confirmation pool: {len(confirm_images)} fresh images "
                f"(seed=123, {max_confirm_per_class}/class)..."
            )
            confirmation_validation = await agents.validate_candidate_rule(
                candidate_rule=active_rule,
                validation_images=confirm_images,
                trigger_image_path=img_path,
                trigger_correct_label=correct,
                model=_val_model,
            )
            cf_p = confirmation_validation["precision"]
            console.print(
                f"  Confirmation: TP={confirmation_validation['tp']} "
                f"FP={confirmation_validation['fp']} | precision={cf_p:.2f}"
            )
            if cf_p < min_precision:
                console.print(
                    f"  [red]CONFIRMATION FAILED[/red] — precision {cf_p:.2f} "
                    f"< {min_precision:.2f}. Rejecting."
                )
                accepted = False
            else:
                console.print(f"  [green]CONFIRMED[/green] ({cf_p:.2f} ≥ {min_precision:.2f})")

        record = {
            "task_id": task_id, "pair_id": pair_id,
            "wrong_prediction": wrong, "correct_label": correct,
            "candidate_rule": {k: v for k, v in active_rule.items()
                               if k != "raw_response"},
            "semantic_validation": semantic_result,
            "validation": {k: v for k, v in validation.items()
                           if k not in ("tp_cases", "fp_cases")},
            "confirmation_validation": (
                {k: v for k, v in confirmation_validation.items()
                 if k not in ("tp_cases", "fp_cases")}
                if confirmation_validation else None
            ),
            "spectrum_history": spectrum_history,
            "accepted": accepted,
            "registered": False,
        }

        # --- Step 4: Register rule ---
        if accepted and not dry_run:
            entry = _register_rule(
                candidate_rule=active_rule,
                pair_id=pair_id,
                cheap_model=cheap_model,
                expert_model=expert_model,
            )
            record["registered"] = entry is not None
            record["rule_id"] = entry["id"] if entry else None
            if entry:
                console.print(f"  Rule registered as {entry['id']}.")
        elif not accepted:
            console.print(
                f"  [red]REJECTED[/red] — precision={validation['precision']:.2f} "
                f"FP={validation['fp']} "
                f"({'spectrum tried' if spectrum_history else 'no spectrum'})"
            )

        patch_records.append(record)

    return patch_records


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _surface_to_expert(active_rule, validation, pair_info, task_id):
    fp_cases = validation.get("fp_cases", [])
    tp_cases = validation.get("tp_cases", [])
    favors = active_rule.get("favors", "?")
    class_a = pair_info.get("class_a", "?")
    class_b = pair_info.get("class_b", "?")
    pair_id = pair_info.get("pair_id", "?")

    console.print("\n  [bold yellow]⚑ EXPERT REVIEW REQUIRED[/bold yellow]")
    console.print(f"  Task:   {task_id}")
    console.print(f"  Rule favors: {favors} (authored for {pair_id})")
    console.print(f"  Problem: rule fired on {len(fp_cases)} wrong-class image(s):")
    for c in fp_cases:
        console.print(f"    • {c['ground_truth']}: {c.get('observations','')[:120]}")
    if tp_cases:
        console.print(f"  Correct firings ({len(tp_cases)} TP):")
        for c in tp_cases[:3]:
            console.print(f"    • {c['ground_truth']}: {c.get('observations','')[:80]}")
    console.print(
        f"\n  [bold]Question for expert[/bold]: Does the rule pattern genuinely "
        f"distinguish {favors} from {class_a if favors == class_b else class_b}?"
    )


def parse_args():
    p = argparse.ArgumentParser(description="KF bird dialogic patching loop")
    p.add_argument("--cheap-model",      dest="cheap_model",
                   default="qwen/qwen3-vl-8b-instruct")
    p.add_argument("--expert-model",     dest="expert_model",
                   default="claude-sonnet-4-6")
    p.add_argument("--validator-model",  dest="validator_model",  default="",
                   help="Model for pool validation. Defaults to expert-model.")
    p.add_argument("--failures-from",    dest="failures_from",    default="",
                   help="Load failures from an existing baseline JSON")
    p.add_argument("--pair",             default="",
                   help="Limit to a single pair ID "
                        "(e.g. bronzed_cowbird_vs_shiny_cowbird)")
    p.add_argument("--max-per-class",    dest="max_per_class",    type=int, default=3)
    p.add_argument("--max-val-per-class",        dest="max_val_per_class",        type=int, default=8)
    p.add_argument("--max-authoring-per-class",  dest="max_authoring_per_class",  type=int, default=8)
    p.add_argument("--max-confirm-per-class",    dest="max_confirm_per_class",    type=int, default=16)
    p.add_argument("--min-precision",    dest="min_precision",    type=float, default=0.75)
    p.add_argument("--data-dir",         dest="data_dir",
                   default=str(DEFAULT_DATA_DIR))
    p.add_argument("--patch-rules",      dest="patch_rules",
                   default="patch_rules_birds.json")
    p.add_argument("--dry-run",          action="store_true")
    p.add_argument("--output",           default="patch_session_birds.json")
    p.add_argument("--skip-rerun",       dest="skip_rerun", action="store_true")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    _load_api_keys()

    effective_validator = args.validator_model or args.expert_model

    console.rule("[bold]KF Bird Dialogic Patching Loop[/bold]")
    console.print(f"  Cheap model:      [cyan]{args.cheap_model}[/cyan]")
    console.print(f"  Expert model:     [cyan]{args.expert_model}[/cyan]")
    console.print(f"  Validator model:  [cyan]{effective_validator}[/cyan]")
    console.print(f"  Held-out pool/class: {args.max_val_per_class}")
    console.print(f"  Dry-run:          {args.dry_run}")

    console.print(f"\n[dim]Loading CUB-200-2011 from {args.data_dir}...[/dim]")
    ds = load_cub(args.data_dir)

    # Select pairs to run
    if args.pair:
        cp = _pair_by_id(args.pair)
        if not cp:
            console.print(f"[red]Pair not found: {args.pair}[/red]")
            available = [_pair_id(c) for c in CONFUSABLE_PAIRS]
            console.print("Available pairs:")
            for pid in available:
                console.print(f"  {pid}")
            sys.exit(1)
        pairs = [cp]
    else:
        pairs = list(CONFUSABLE_PAIRS)

    # Step 1: Get failures
    if args.failures_from:
        console.print(f"\nLoading failures from [cyan]{args.failures_from}[/cyan]...")
        with open(args.failures_from) as f:
            prev = json.load(f)
        all_tasks = prev.get("tasks", [])
        if args.pair:
            all_tasks = [t for t in all_tasks if t.get("pair_id") == args.pair]
    else:
        console.print(
            f"\n[bold]Step 1[/bold]: Running {args.cheap_model} zero-shot "
            f"on {len(pairs)} pair(s)..."
        )
        all_tasks = await run_zero_shot_baseline(
            ds=ds,
            pairs=pairs,
            cheap_model=args.cheap_model,
            max_per_class=args.max_per_class,
        )

    failures = [t for t in all_tasks if not t["correct"]]
    total = len(all_tasks)
    console.print(
        f"\nBaseline: {total - len(failures)}/{total} correct | "
        f"[red]{len(failures)} failure(s) to patch[/red]"
    )

    if not failures:
        console.print("[green]No failures — nothing to patch.[/green]")
        return

    # Step 2: Init patch rules
    patch_rules_path = Path(_HERE) / args.patch_rules
    global _registered_rules, _rule_counter
    _registered_rules = []
    _rule_counter = 0
    console.print(f"\n[bold]Step 2[/bold]: Patch rules initialised (in-memory).")

    # Steps 3–6: Per-failure loop
    console.print(f"\n[bold]Steps 3–6[/bold]: Per-failure patch + test loop...\n")
    patch_records: list[dict] = []
    all_rerun_tasks: list[dict] = []
    before_correct = total - len(failures)

    for i, failure in enumerate(failures, 1):
        console.rule(f"[bold]Failure {i}/{len(failures)}: {failure['task_id']}[/bold]")

        records = await run_patch_loop(
            failures=[failure],
            ds=ds,
            cheap_model=args.cheap_model,
            expert_model=args.expert_model,
            max_val_per_class=args.max_val_per_class,
            max_authoring_per_class=args.max_authoring_per_class,
            max_confirm_per_class=args.max_confirm_per_class,
            min_precision=args.min_precision,
            dry_run=args.dry_run,
            validator_model=args.validator_model,
        )
        patch_records.extend(records)

        rec = records[0] if records else {}

        # Immediate test: re-classify this failure with all rules so far
        if not args.skip_rerun and not args.dry_run:
            console.print(
                f"\n  [bold]Test[/bold]: Re-running {args.cheap_model} on "
                f"[cyan]{failure['task_id']}[/cyan] with patch rules..."
            )
            rerun = await rerun_with_patch_rules(
                failure=failure,
                all_registered_rules=_registered_rules,
                cheap_model=args.cheap_model,
            )
            all_rerun_tasks.append(rerun)

            if rerun["correct"]:
                console.print(
                    f"  [green]FIXED[/green] -- {args.cheap_model} now correctly "
                    f"identifies {failure['task_id']}"
                )
            else:
                registered_now = rec.get("registered", False)
                reason = (
                    "rule not registered"
                    if not registered_now
                    else "rule registered but cheap model still failed"
                )
                console.print(f"  [red]✗ STILL FAILING[/red] — {reason}")
                if i < len(failures):
                    console.print(
                        f"  [yellow]Stopping before remaining {len(failures) - i} "
                        f"failure(s). Review rule and re-run with --failures-from.[/yellow]"
                    )
                    # Save and exit
                    _save_patch_rules(patch_rules_path)
                    _save_session(args, patch_records, total, failures,
                                  effective_validator, stopped_early=True,
                                  stop_reason=reason)
                    return

        # Save incremental state
        _save_patch_rules(patch_rules_path)
        _save_session(args, patch_records, total, failures, effective_validator)

    # Final summary
    n_accepted   = sum(1 for r in patch_records if r["accepted"])
    n_registered = sum(1 for r in patch_records if r["registered"])
    after_correct = before_correct + sum(1 for t in all_rerun_tasks if t["correct"])

    console.rule("[bold]Patch Summary[/bold]")
    t = Table(show_header=True)
    t.add_column("Phase"); t.add_column("Correct"); t.add_column("Accuracy")
    t.add_row("Before patching (zero-shot)", f"{before_correct}/{total}",
              f"{before_correct/total*100:.1f}%")
    if all_rerun_tasks:
        t.add_row("After patching (KF on failures only)",
                  f"{after_correct}/{total}",
                  f"{after_correct/total*100:.1f}%")
        t.add_row("Delta", f"+{after_correct-before_correct}", "")
    t.add_row("Failures patched", f"{len(failures)}/{len(failures)}", "")
    t.add_row("Rules authored/accepted/registered",
              f"{len(patch_records)}/{n_accepted}/{n_registered}", "")
    console.print(t)

    _save_patch_rules(patch_rules_path)
    _save_session(args, patch_records, total, failures, effective_validator)
    console.print(f"\nSession saved to [cyan]{args.output}[/cyan]")


def _save_session(args, patch_records, total, failures, effective_validator,
                  stopped_early=False, stop_reason=""):
    n_accepted   = sum(1 for r in patch_records if r["accepted"])
    n_registered = sum(1 for r in patch_records if r["registered"])
    session = {
        "cheap_model":       args.cheap_model,
        "expert_model":      args.expert_model,
        "validator_model":   effective_validator,
        "patch_rules_file":  args.patch_rules,
        "dry_run":           args.dry_run,
        "total_tasks":       total,
        "failures_before":   len(failures),
        "rules_authored":    len(patch_records),
        "rules_accepted":    n_accepted,
        "rules_registered":  n_registered,
        "stopped_early":     stopped_early,
        "stop_reason":       stop_reason,
        "patch_records":     patch_records,
    }
    out_path = Path(_HERE) / args.output
    with open(out_path, "w") as f:
        json.dump(session, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
