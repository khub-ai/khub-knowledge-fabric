"""
experiment_dialogic_vs_singleshot.py

Controlled comparison: single-shot elicitation vs three-party dialogic
distillation, using the same triggers, same pool, same test set.

Independent variable: whether KF steers a multi-round dialog.
Both methods use Sonnet as tutor and validator, same pool gate.

Phases:
  1. baseline   — Qwen zero-shot on 100 mel + 100 nev (test split)
  2. triggers   — select N failures as trigger images (fixed seed)
  3. singleshot — single-turn Opus elicitation + grounding + pool gate
  4. dialogic   — multi-round dialogic distillation + grounding + pool gate
  5. eval       — rerun Qwen on failures with each method's accepted rules
  6. report     — summary table + per-image comparison

Usage:
  python experiment_dialogic_vs_singleshot.py                # full run
  python experiment_dialogic_vs_singleshot.py --phase baseline
  python experiment_dialogic_vs_singleshot.py --phase triggers
  python experiment_dialogic_vs_singleshot.py --phase singleshot
  python experiment_dialogic_vs_singleshot.py --phase dialogic
  python experiment_dialogic_vs_singleshot.py --phase eval
  python experiment_dialogic_vs_singleshot.py --phase report
  python experiment_dialogic_vs_singleshot.py --resume       # skip completed phases
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_KF_ROOT = _HERE.parents[4]
if str(_KF_ROOT) not in sys.path:
    sys.path.insert(0, str(_KF_ROOT))

import agents
from dataset import load as load_ham10000, CONFUSABLE_PAIRS
from domain_config import DERM_CONFIG

from core.dialogic_distillation import run_dialogic_distillation
from core.dialogic_distillation.prompts import dialogic_tutor_system, ROUND1_PROMPT

from rich.console import Console
from rich.table import Table

console = Console()

# ── Config ─────────────────────────────────────────────────────────────────
DATA_DIR       = r"C:\_backup\ml\data\DermaMNIST_HAM10000"
PAIR_ID        = "melanoma_vs_melanocytic_nevus"
IMAGES_PER_CLASS = 100     # test set size per class
N_TRIGGERS     = 10        # number of failure images to seed from
TRIGGER_SEED   = 7         # fixed seed for trigger selection
POOL_PER_CLASS = 10        # pool gate size per class (train split, seed=42)
TUTOR_MODEL    = "claude-sonnet-4-6"
VALIDATOR_MODEL = "claude-sonnet-4-6"
CHEAP_MODEL    = "qwen/qwen3-vl-8b-instruct"
MAX_ROUNDS     = 5         # max dialog rounds for dialogic method
OUTPUT_DIR     = _HERE / ".tmp" / "experiment_dvs"

# ── API keys ───────────────────────────────────────────────────────────────
def _load_api_keys():
    kf = Path("P:/_access/Security/api_keys.env")
    if kf.exists():
        for line in kf.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip(); v = v.strip()
                if k in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY") \
                        and not os.environ.get(k):
                    os.environ[k] = v


# ── Phase 1: Baseline ─────────────────────────────────────────────────────
async def phase_baseline():
    """Run Qwen zero-shot on IMAGES_PER_CLASS mel + nev from test split."""
    console.rule("[bold]Phase 1: Zero-Shot Baseline[/bold]")

    ds = load_ham10000(DATA_DIR)
    pair = next(p for p in CONFUSABLE_PAIRS if p["pair_id"] == PAIR_ID)

    mel_imgs = ds.sample_images("mel", IMAGES_PER_CLASS, split="test", seed=42)
    nev_imgs = ds.sample_images("nv", IMAGES_PER_CLASS, split="test", seed=42)
    console.print(f"  Test set: {len(mel_imgs)} mel + {len(nev_imgs)} nev "
                  f"= {len(mel_imgs)+len(nev_imgs)} images")

    agents.ACTIVE_MODEL = CHEAP_MODEL
    agents.DEFAULT_MODEL = CHEAP_MODEL

    tasks = []
    for img in mel_imgs:
        tasks.append({
            "image_id": img.image_id,
            "file_path": str(img.file_path),
            "ground_truth": pair["class_a"],  # Melanoma
        })
    for img in nev_imgs:
        tasks.append({
            "image_id": img.image_id,
            "file_path": str(img.file_path),
            "ground_truth": pair["class_b"],  # Melanocytic Nevus
        })

    results = []
    correct = 0
    for i, task in enumerate(tasks, 1):
        t0 = time.time()
        decision, ms = await agents.run_baseline(
            {
                "class_a": pair["class_a"],
                "class_b": pair["class_b"],
                "test_image_path": task["file_path"],
            },
            mode="zero_shot",
        )
        predicted = decision.get("label", "uncertain")
        is_correct = predicted == task["ground_truth"]
        if is_correct:
            correct += 1

        results.append({
            "image_id": task["image_id"],
            "ground_truth": task["ground_truth"],
            "predicted": predicted,
            "correct": is_correct,
            "ms": ms,
        })

        status = "[green]OK[/green]" if is_correct else "[red]WRONG[/red]"
        console.print(f"  [{i:3d}/{len(tasks)}] {task['image_id']}  "
                      f"gt={task['ground_truth'][:3]}  pred={predicted[:3]}  {status}")

    total = len(tasks)
    failures = [r for r in results if not r["correct"]]

    out = {
        "phase": "baseline",
        "model": CHEAP_MODEL,
        "images_per_class": IMAGES_PER_CLASS,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "n_failures": len(failures),
        "tasks": results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "baseline.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    console.print(f"\n  Baseline: {correct}/{total} = {correct/total*100:.1f}%")
    console.print(f"  Failures: {len(failures)}")
    console.print(f"  Saved to {OUTPUT_DIR / 'baseline.json'}")
    return out


# ── Phase 2: Select Triggers ──────────────────────────────────────────────
def phase_triggers():
    """Select N_TRIGGERS melanoma failures as trigger images."""
    console.rule("[bold]Phase 2: Select Triggers[/bold]")

    with open(OUTPUT_DIR / "baseline.json", encoding="utf-8") as f:
        baseline = json.load(f)

    # Only melanoma failures (nevus misclassified as melanoma is a different error)
    mel_failures = [t for t in baseline["tasks"]
                    if not t["correct"] and t["ground_truth"] == "Melanoma"]
    console.print(f"  Melanoma failures: {len(mel_failures)}")

    import random
    rng = random.Random(TRIGGER_SEED)
    triggers = rng.sample(mel_failures, min(N_TRIGGERS, len(mel_failures)))

    trigger_ids = [t["image_id"] for t in triggers]
    console.print(f"  Selected {len(trigger_ids)} triggers (seed={TRIGGER_SEED}):")
    for tid in trigger_ids:
        console.print(f"    {tid}")

    out = {
        "phase": "triggers",
        "seed": TRIGGER_SEED,
        "n_requested": N_TRIGGERS,
        "n_selected": len(trigger_ids),
        "trigger_ids": trigger_ids,
        "mel_failures_total": len(mel_failures),
    }
    with open(OUTPUT_DIR / "triggers.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    console.print(f"  Saved to {OUTPUT_DIR / 'triggers.json'}")
    return out


# ── Phase 3: Single-Shot Elicitation ──────────────────────────────────────
async def phase_singleshot():
    """Single-turn Opus elicitation with grounding check + pool gate."""
    console.rule("[bold]Phase 3: Single-Shot Elicitation[/bold]")

    with open(OUTPUT_DIR / "triggers.json", encoding="utf-8") as f:
        triggers = json.load(f)
    trigger_ids = triggers["trigger_ids"]

    ds = load_ham10000(DATA_DIR)
    pair = next(p for p in CONFUSABLE_PAIRS if p["pair_id"] == PAIR_ID)
    label_mel = pair["class_a"]
    label_nev = pair["class_b"]

    # Build image map
    all_mel = (ds.sample_images("mel", 500, split="test", seed=0)
               + ds.sample_images("mel", 500, split="train", seed=0))
    all_nv = (ds.sample_images("nv", 500, split="test", seed=0)
              + ds.sample_images("nv", 500, split="train", seed=0))
    img_map = {img.image_id: str(img.file_path) for img in all_mel + all_nv}

    # Pool gate images (train split, seed=42)
    pool_mel = [(str(img.file_path), label_mel)
                for img in ds.sample_images("mel", POOL_PER_CLASS,
                                            split="train", seed=42)]
    pool_nv = [(str(img.file_path), label_nev)
               for img in ds.sample_images("nv", POOL_PER_CLASS,
                                           split="train", seed=42)]
    pool_images = pool_mel + pool_nv
    console.print(f"  Pool: {len(pool_mel)} mel + {len(pool_nv)} nev = {len(pool_images)}")

    transcripts = []
    for i, fid in enumerate(trigger_ids, 1):
        path = img_map.get(fid)
        if not path:
            console.print(f"  [red]Not found: {fid}[/red]")
            continue

        console.rule(f"[{i}/{len(trigger_ids)}] {fid}")

        # Single-turn call
        vocab_lines = ""
        if DERM_CONFIG.good_vocabulary_examples or DERM_CONFIG.bad_vocabulary_examples:
            parts = []
            for ex in DERM_CONFIG.good_vocabulary_examples:
                parts.append(f'   - GOOD: "{ex}"')
            for ex in DERM_CONFIG.bad_vocabulary_examples:
                parts.append(f'   - BAD: "{ex}"')
            vocab_lines = "\n".join(parts) + "\n"

        prompt_text = ROUND1_PROMPT.format(
            class_a=label_mel,
            class_b=label_nev,
            correct_label=label_mel,
            wrong_prediction=label_nev,
            pupil_reasoning="(cheap model predicted Melanocytic Nevus)",
            item_noun=DERM_CONFIG.item_noun,
            class_noun=DERM_CONFIG.class_noun,
            vocab_examples=vocab_lines,
        )
        content = [
            agents._image_block(path),
            {"type": "text", "text": prompt_text},
        ]
        raw_text, ms = await agents.call_agent(
            "SINGLESHOT_TUTOR",
            content,
            system_prompt=dialogic_tutor_system(DERM_CONFIG),
            model=TUTOR_MODEL,
            max_tokens=2048,
        )

        rule = agents._parse_json_block(raw_text)
        if not rule or "preconditions" not in rule:
            rule = {"rule": raw_text, "feature": "unknown",
                    "favors": label_mel, "confidence": "low",
                    "preconditions": [], "rationale": ""}

        console.print(f"  Rule: [italic]{rule.get('rule', '')[:120]}[/italic]")

        # Grounding check
        val_result, _ = await agents.run_rule_validator_on_image(
            image_path=path,
            ground_truth=label_mel,
            candidate_rule=rule,
            model=VALIDATOR_MODEL,
        )
        fires = val_result.get("precondition_met", False)
        observations = val_result.get("observations", "")
        status = "[green]FIRES[/green]" if fires else "[red]DOES NOT FIRE[/red]"
        console.print(f"  Grounding: {status}")

        # Pool gate (only if grounded)
        pool_result = None
        accepted = False
        if fires:
            pool_result = await agents.validate_candidate_rule(
                candidate_rule=rule,
                validation_images=pool_images,
                trigger_image_path=path,
                trigger_correct_label=label_mel,
                model=VALIDATOR_MODEL,
            )
            accepted = pool_result.get("accepted", False)
            console.print(f"  Pool: TP={pool_result['tp']} FP={pool_result['fp']} "
                          f"prec={pool_result['precision']:.2f} "
                          f"{'[green]PASS[/green]' if accepted else '[red]FAIL[/red]'}")

        transcript = {
            "image_id": fid,
            "method": "singleshot",
            "rule": {k: v for k, v in rule.items() if k != "raw_response"},
            "grounded": fires,
            "validator_observations": observations,
            "pool_result": ({k: v for k, v in pool_result.items()
                            if k not in ("tp_cases", "fp_cases")}
                           if pool_result else None),
            "accepted": accepted,
        }
        transcripts.append(transcript)

    n_grounded = sum(1 for t in transcripts if t["grounded"])
    n_accepted = sum(1 for t in transcripts if t["accepted"])
    console.print(f"\n  Single-shot: {n_grounded}/{len(transcripts)} grounded, "
                  f"{n_accepted}/{len(transcripts)} accepted")

    # Extract accepted rules
    accepted_rules = []
    for t in transcripts:
        if t["accepted"]:
            r = t["rule"].copy()
            r["id"] = f"r_ss_{t['image_id']}"
            r["triggered_by"] = t["image_id"]
            r["rule_text"] = r.get("rule", "")
            accepted_rules.append(r)

    out = {
        "phase": "singleshot",
        "tutor_model": TUTOR_MODEL,
        "validator_model": VALIDATOR_MODEL,
        "n_triggers": len(transcripts),
        "n_grounded": n_grounded,
        "n_accepted": n_accepted,
        "transcripts": transcripts,
        "accepted_rules": accepted_rules,
    }
    with open(OUTPUT_DIR / "singleshot.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Write accepted rules in patch format for rerun
    patch = {"version": 2, "rules": accepted_rules}
    with open(OUTPUT_DIR / "rules_singleshot.json", "w", encoding="utf-8") as f:
        json.dump(patch, f, indent=2, ensure_ascii=False)

    console.print(f"  Saved to {OUTPUT_DIR / 'singleshot.json'}")
    return out


# ── Phase 4: Dialogic Distillation ────────────────────────────────────────
async def phase_dialogic():
    """Multi-round dialogic distillation with KF steering."""
    console.rule("[bold]Phase 4: Dialogic Distillation[/bold]")

    with open(OUTPUT_DIR / "triggers.json", encoding="utf-8") as f:
        triggers = json.load(f)
    trigger_ids = triggers["trigger_ids"]

    ds = load_ham10000(DATA_DIR)
    pair = next(p for p in CONFUSABLE_PAIRS if p["pair_id"] == PAIR_ID)
    label_mel = pair["class_a"]
    label_nev = pair["class_b"]

    # Build image map
    all_mel = (ds.sample_images("mel", 500, split="test", seed=0)
               + ds.sample_images("mel", 500, split="train", seed=0))
    all_nv = (ds.sample_images("nv", 500, split="test", seed=0)
              + ds.sample_images("nv", 500, split="train", seed=0))
    img_map = {img.image_id: str(img.file_path) for img in all_mel + all_nv}

    # Pool gate images (same pool as single-shot — train split, seed=42)
    pool_mel = [(str(img.file_path), label_mel)
                for img in ds.sample_images("mel", POOL_PER_CLASS,
                                            split="train", seed=42)]
    pool_nv = [(str(img.file_path), label_nev)
               for img in ds.sample_images("nv", POOL_PER_CLASS,
                                           split="train", seed=42)]
    pool_images = pool_mel + pool_nv
    console.print(f"  Pool: {len(pool_mel)} mel + {len(pool_nv)} nev = {len(pool_images)}")

    transcripts = []
    for i, fid in enumerate(trigger_ids, 1):
        path = img_map.get(fid)
        if not path:
            console.print(f"  [red]Not found: {fid}[/red]")
            continue

        console.rule(f"[{i}/{len(trigger_ids)}] {fid}")

        transcript = await run_dialogic_distillation(
            image_path=path,
            image_id=fid,
            correct_label=label_mel,
            wrong_prediction=label_nev,
            pupil_reasoning="(cheap model predicted Melanocytic Nevus)",
            pair_info=pair,
            config=DERM_CONFIG,
            tutor_model=TUTOR_MODEL,
            validator_model=VALIDATOR_MODEL,
            max_rounds=MAX_ROUNDS,
            pool_images=pool_images,
            call_agent_fn=agents.call_agent,
            console=console,
        )
        transcripts.append(transcript)

    n_grounded = sum(1 for t in transcripts
                     if t.get("grounded_at_round") is not None)
    n_accepted = sum(1 for t in transcripts if t.get("outcome") == "accepted")
    console.print(f"\n  Dialogic: {n_grounded}/{len(transcripts)} grounded, "
                  f"{n_accepted}/{len(transcripts)} accepted")

    # Extract accepted rules
    accepted_rules = []
    for t in transcripts:
        if t.get("outcome") == "accepted" and t.get("final_rule"):
            r = t["final_rule"].copy()
            r["id"] = f"r_dl_{t['image_id']}"
            r["triggered_by"] = t["image_id"]
            r["rule_text"] = r.get("rule", "")
            accepted_rules.append(r)

    out = {
        "phase": "dialogic",
        "tutor_model": TUTOR_MODEL,
        "validator_model": VALIDATOR_MODEL,
        "max_rounds": MAX_ROUNDS,
        "n_triggers": len(transcripts),
        "n_grounded": n_grounded,
        "n_accepted": n_accepted,
        "transcripts": transcripts,
        "accepted_rules": accepted_rules,
    }
    with open(OUTPUT_DIR / "dialogic.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Write accepted rules in patch format for rerun
    patch = {"version": 2, "rules": accepted_rules}
    with open(OUTPUT_DIR / "rules_dialogic.json", "w", encoding="utf-8") as f:
        json.dump(patch, f, indent=2, ensure_ascii=False)

    console.print(f"  Saved to {OUTPUT_DIR / 'dialogic.json'}")
    return out


# ── Phase 5: Evaluate ─────────────────────────────────────────────────────

_RERUN_SYSTEM_WITH_RULES = """\
You are an expert dermatologist. You will be shown a dermoscopic image.
Classify it as one of the two specified lesion types based solely on the image.

IMPORTANT: Before classifying, check each of these expert-validated rules.
If ANY rule's preconditions match what you see in the image, you MUST follow
that rule's classification recommendation.

{rules_block}

Output ONLY a JSON object:
{{
  "label": "<class_a_name>" | "<class_b_name>",
  "confidence": 0.0,
  "reasoning": "Brief dermoscopic rationale."
}}
"""


def _format_rules_block(rules: list[dict]) -> str:
    """Format accepted rules for injection into the system prompt."""
    lines = []
    for i, r in enumerate(rules, 1):
        rule_text = r.get("rule_text") or r.get("rule", "")
        preconditions = r.get("preconditions", [])
        favors = r.get("favors", "")
        lines.append(f"Rule {i}: {rule_text}")
        for pc in preconditions:
            lines.append(f"  - {pc}")
        lines.append(f"  → Classify as: {favors}")
        lines.append("")
    return "\n".join(lines)


async def phase_eval():
    """Rerun Qwen on failure images with each method's accepted rules."""
    console.rule("[bold]Phase 5: Evaluation[/bold]")

    with open(OUTPUT_DIR / "baseline.json", encoding="utf-8") as f:
        baseline = json.load(f)

    ds = load_ham10000(DATA_DIR)
    pair = next(p for p in CONFUSABLE_PAIRS if p["pair_id"] == PAIR_ID)

    # Build image map (same seeds as baseline)
    all_mel = (ds.sample_images("mel", 500, split="test", seed=0)
               + ds.sample_images("mel", 500, split="train", seed=0))
    all_nv = (ds.sample_images("nv", 500, split="test", seed=0)
              + ds.sample_images("nv", 500, split="train", seed=0))
    img_map = {img.image_id: str(img.file_path) for img in all_mel + all_nv}

    failures = [t for t in baseline["tasks"] if not t["correct"]]

    agents.ACTIVE_MODEL = CHEAP_MODEL
    agents.DEFAULT_MODEL = CHEAP_MODEL

    for method in ("singleshot", "dialogic"):
        rules_file = OUTPUT_DIR / f"rules_{method}.json"
        if not rules_file.exists():
            console.print(f"  [yellow]Skipping {method}: no rules file[/yellow]")
            continue

        with open(rules_file, encoding="utf-8") as f:
            rules_data = json.load(f)
        rules = rules_data.get("rules", [])

        n_rules = len(rules)
        if n_rules == 0:
            console.print(f"  [yellow]{method}: 0 rules — writing baseline copy[/yellow]")
            out = {
                "phase": f"eval_{method}",
                "n_rules": 0,
                "tasks": baseline["tasks"],
                "correct": baseline["correct"],
                "total": baseline["total"],
                "accuracy": baseline["accuracy"],
                "fixed": 0,
            }
            with open(OUTPUT_DIR / f"eval_{method}.json", "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            continue

        console.rule(f"Evaluating [bold]{method}[/bold] ({n_rules} rules)")

        rules_block = _format_rules_block(rules)
        system = _RERUN_SYSTEM_WITH_RULES.format(rules_block=rules_block)

        # Rerun only failures — successes are kept from baseline
        fixed = 0
        rerun_results = {}
        for i, task in enumerate(failures, 1):
            iid = task["image_id"]
            path = img_map.get(iid)
            if not path:
                console.print(f"  [red]Not found: {iid}[/red]")
                continue

            content = [
                agents._image_block(path),
                {"type": "text", "text": (
                    f"Classify this dermoscopic image as either "
                    f"'{pair['class_a']}' or '{pair['class_b']}'.\n"
                    "First check each expert rule. If any rule's preconditions "
                    "match, follow that rule. Otherwise classify normally.\n"
                    "Return the JSON object specified in the system prompt."
                )},
            ]

            text, ms = await agents.call_agent(
                f"EVAL_{method.upper()}",
                content,
                system_prompt=system,
                max_tokens=512,
            )

            result = agents._parse_json_block(text)
            if result and "label" in result:
                predicted = result["label"]
            else:
                predicted = "uncertain"
                for cls in (pair["class_a"], pair["class_b"]):
                    if cls.lower() in text.lower():
                        predicted = cls
                        break

            is_correct = predicted == task["ground_truth"]
            if is_correct:
                fixed += 1

            status = "[green]FIXED[/green]" if is_correct else "[red]still wrong[/red]"
            console.print(f"  [{i:2d}/{len(failures)}] {iid}  {status}  "
                          f"pred={predicted[:3]}")
            rerun_results[iid] = {
                "predicted": predicted,
                "correct": is_correct,
            }

        # Merge: baseline successes + rerun results
        merged = []
        correct = 0
        for bt in baseline["tasks"]:
            iid = bt["image_id"]
            if bt["correct"]:
                merged.append({**bt, "source": "baseline"})
                correct += 1
            elif iid in rerun_results:
                rr = rerun_results[iid]
                merged.append({
                    "image_id": iid,
                    "ground_truth": bt["ground_truth"],
                    "predicted": rr["predicted"],
                    "correct": rr["correct"],
                    "source": "rerun",
                })
                if rr["correct"]:
                    correct += 1
            else:
                merged.append({**bt, "source": "baseline_kept"})

        total = len(merged)
        out = {
            "phase": f"eval_{method}",
            "n_rules": n_rules,
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4),
            "baseline_correct": baseline["correct"],
            "failures_tested": len(failures),
            "fixed": fixed,
            "tasks": merged,
        }
        with open(OUTPUT_DIR / f"eval_{method}.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        console.print(f"\n  {method}: {correct}/{total} = {correct/total*100:.1f}%  "
                      f"(+{fixed} fixed from {len(failures)} failures)")


# ── Phase 6: Report ───────────────────────────────────────────────────────
def phase_report():
    """Print summary comparison table."""
    console.rule("[bold]Phase 6: Results[/bold]")

    with open(OUTPUT_DIR / "baseline.json", encoding="utf-8") as f:
        baseline = json.load(f)

    # Load distillation results
    ss_data = dl_data = None
    ss_file = OUTPUT_DIR / "singleshot.json"
    dl_file = OUTPUT_DIR / "dialogic.json"
    if ss_file.exists():
        with open(ss_file, encoding="utf-8") as f:
            ss_data = json.load(f)
    if dl_file.exists():
        with open(dl_file, encoding="utf-8") as f:
            dl_data = json.load(f)

    # Load eval results
    eval_ss = eval_dl = None
    eval_ss_file = OUTPUT_DIR / "eval_singleshot.json"
    eval_dl_file = OUTPUT_DIR / "eval_dialogic.json"
    if eval_ss_file.exists():
        with open(eval_ss_file, encoding="utf-8") as f:
            eval_ss = json.load(f)
    if eval_dl_file.exists():
        with open(eval_dl_file, encoding="utf-8") as f:
            eval_dl = json.load(f)

    # ── Table 1: Rule Conversion ──────────────────────────────────────
    tbl1 = Table(title="Rule Conversion Rate", show_header=True)
    tbl1.add_column("Method")
    tbl1.add_column("Triggers")
    tbl1.add_column("Grounded")
    tbl1.add_column("Accepted")
    tbl1.add_column("Conversion")

    if ss_data:
        n = ss_data["n_triggers"]
        ng = ss_data["n_grounded"]
        na = ss_data["n_accepted"]
        tbl1.add_row("Single-shot", str(n), f"{ng}/{n}",
                      f"{na}/{n}", f"{na/n*100:.0f}%")
    if dl_data:
        n = dl_data["n_triggers"]
        ng = dl_data["n_grounded"]
        na = dl_data["n_accepted"]
        tbl1.add_row("Dialogic", str(n), f"{ng}/{n}",
                      f"{na}/{n}", f"{na/n*100:.0f}%")
    console.print(tbl1)

    # ── Table 2: Classification Accuracy ──────────────────────────────
    tbl2 = Table(title="Classification Accuracy (Qwen Mel/Nev)", show_header=True)
    tbl2.add_column("Phase")
    tbl2.add_column("Correct")
    tbl2.add_column("Total")
    tbl2.add_column("Accuracy")
    tbl2.add_column("Delta")

    bc = baseline["correct"]
    bt = baseline["total"]
    tbl2.add_row("Zero-shot baseline", str(bc), str(bt),
                 f"{bc/bt*100:.1f}%", "—")

    if eval_ss:
        sc = eval_ss["correct"]
        st = eval_ss["total"]
        delta = sc - bc
        tbl2.add_row(f"+ Single-shot ({eval_ss['n_rules']} rules)",
                     str(sc), str(st),
                     f"{sc/st*100:.1f}%",
                     f"+{delta} (+{delta/st*100:.1f}pp)")
    if eval_dl:
        dc = eval_dl["correct"]
        dt = eval_dl["total"]
        delta = dc - bc
        tbl2.add_row(f"+ Dialogic ({eval_dl['n_rules']} rules)",
                     str(dc), str(dt),
                     f"{dc/dt*100:.1f}%",
                     f"+{delta} (+{delta/dt*100:.1f}pp)")
    console.print(tbl2)

    # ── Per-image comparison ──────────────────────────────────────────
    if eval_ss and eval_dl:
        console.print("\n[bold]Per-image comparison (failures only):[/bold]")
        ss_map = {t["image_id"]: t for t in eval_ss["tasks"]}
        dl_map = {t["image_id"]: t for t in eval_dl["tasks"]}

        both_fixed = ss_only = dl_only = neither = 0
        for bt_task in baseline["tasks"]:
            if bt_task["correct"]:
                continue
            iid = bt_task["image_id"]
            ss_ok = ss_map.get(iid, {}).get("correct", False)
            dl_ok = dl_map.get(iid, {}).get("correct", False)
            if ss_ok and dl_ok:
                both_fixed += 1
            elif ss_ok:
                ss_only += 1
            elif dl_ok:
                dl_only += 1
            else:
                neither += 1

        n_fail = baseline["n_failures"]
        console.print(f"  Both fixed:      {both_fixed}")
        console.print(f"  Single-shot only: {ss_only}")
        console.print(f"  Dialogic only:    {dl_only}")
        console.print(f"  Neither:          {neither}")
        console.print(f"  Total failures:   {n_fail}")

        # McNemar's test (paired comparison)
        # b = ss correct & dl wrong, c = ss wrong & dl correct
        b = ss_only   # single-shot fixed but dialogic didn't
        c = dl_only   # dialogic fixed but single-shot didn't
        if b + c > 0:
            # McNemar chi-squared (with continuity correction)
            chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0
            console.print(f"\n  McNemar's test: b={b}, c={c}, "
                          f"chi2={chi2:.2f}")
            # Rough p-value interpretation
            if chi2 > 6.63:
                console.print(f"  [green]p < 0.01 — highly significant[/green]")
            elif chi2 > 3.84:
                console.print(f"  [green]p < 0.05 — significant[/green]")
            else:
                console.print(f"  [yellow]p >= 0.05 — not significant[/yellow]")
        else:
            console.print("\n  McNemar's test: no discordant pairs")

    # Save report
    report = {
        "baseline": {"correct": bc, "total": bt, "accuracy": baseline["accuracy"]},
    }
    if ss_data:
        report["singleshot_conversion"] = {
            "triggers": ss_data["n_triggers"],
            "grounded": ss_data["n_grounded"],
            "accepted": ss_data["n_accepted"],
        }
    if dl_data:
        report["dialogic_conversion"] = {
            "triggers": dl_data["n_triggers"],
            "grounded": dl_data["n_grounded"],
            "accepted": dl_data["n_accepted"],
        }
    if eval_ss:
        report["singleshot_eval"] = {
            "correct": eval_ss["correct"],
            "total": eval_ss["total"],
            "accuracy": eval_ss["accuracy"],
            "fixed": eval_ss["fixed"],
        }
    if eval_dl:
        report["dialogic_eval"] = {
            "correct": eval_dl["correct"],
            "total": eval_dl["total"],
            "accuracy": eval_dl["accuracy"],
            "fixed": eval_dl["fixed"],
        }
    with open(OUTPUT_DIR / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    console.print(f"\n  Saved report to {OUTPUT_DIR / 'report.json'}")


# ── CLI ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Controlled experiment: single-shot vs dialogic distillation")
    p.add_argument("--phase",
                   choices=["baseline", "triggers", "singleshot", "dialogic",
                            "eval", "report"],
                   default="",
                   help="Run a single phase (default: all phases)")
    p.add_argument("--resume", action="store_true",
                   help="Skip phases whose output files already exist")
    return p.parse_args()


async def main():
    args = parse_args()
    _load_api_keys()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    phases = [
        ("baseline",   phase_baseline,   OUTPUT_DIR / "baseline.json"),
        ("triggers",   phase_triggers,   OUTPUT_DIR / "triggers.json"),
        ("singleshot", phase_singleshot, OUTPUT_DIR / "singleshot.json"),
        ("dialogic",   phase_dialogic,   OUTPUT_DIR / "dialogic.json"),
        ("eval",       phase_eval,       OUTPUT_DIR / "eval_dialogic.json"),
        ("report",     phase_report,     OUTPUT_DIR / "report.json"),
    ]

    for name, fn, out_file in phases:
        if args.phase and args.phase != name:
            continue
        if args.resume and out_file.exists() and name not in ("eval", "report"):
            console.print(f"[dim]  Skipping {name} (output exists)[/dim]")
            continue

        if asyncio.iscoroutinefunction(fn):
            await fn()
        else:
            fn()

    console.print("\n[bold green]Experiment complete.[/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
