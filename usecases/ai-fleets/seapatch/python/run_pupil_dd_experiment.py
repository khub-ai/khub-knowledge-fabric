"""
run_pupil_dd_experiment.py — End-to-end PUPIL + DD experiment runner.

Measures how much a single maritime DD session improves Qwen's person_in_water
classification accuracy on SeaDronesSee validation frames.

Phases:
  A — Load SeaDronesSee val frames (COCO annotations)
  B — Baseline PUPIL evaluation (Qwen, no rules)
  C — Select the most confident miss as the DD failure frame
  D — Run a full maritime DD session (cross-modal TUTOR + pool + tier grounding)
  E — Re-evaluate with DD rules injected into the PUPIL prompt
  F — Save full JSON report

Usage:
    python run_pupil_dd_experiment.py \\
        --dataset-root V:/_mlArchive/_models/SeaDronesSee/compressed \\
        --pool-dir .tmp/pool \\
        --output .tmp/pupil_dd_experiment.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo-root and local path setup  (mirrors run_dd_session.py)
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR / "simulation"))

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from agents import run_maritime_dd_session
from domain_config import MARITIME_SAR_CONFIG, TIER_OBSERVABILITY, CONFUSABLE_PAIRS
from run_dd_session import load_pool, get_pair_info
from pool_builder import (
    load_coco,
    build_image_index,
    get_person_category_ids,
    build_annotation_index,
    build_file_index,
    select_positive_frames,
    select_negative_frames,
)
from qwen_pupil_eval import (
    PUPIL_MODEL,
    run_baseline_eval,
    run_eval_with_rules,
    find_confident_misses,
    eval_summary,
)
from simulation.thermal_oracle import oracle_for_frame


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a full PUPIL + DD improvement experiment on SeaDronesSee.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset-root", required=True,
        help="SeaDronesSee root directory (must contain images/ and annotations/).",
    )
    p.add_argument(
        "--annotation-file", default=None,
        help="COCO annotation file (default: <root>/annotations/instances_val.json).",
    )
    p.add_argument(
        "--pool-dir", required=True,
        help="Pre-built pool directory (output of pool_builder.py).",
    )
    p.add_argument(
        "--n-eval-positive", type=int, default=30,
        help="Number of person_in_water frames to evaluate (default: 30).",
    )
    p.add_argument(
        "--n-eval-negative", type=int, default=30,
        help="Number of no-person frames to evaluate (default: 30).",
    )
    p.add_argument(
        "--confidence-threshold", type=float, default=0.70,
        help="Min confidence to count as a confident miss (default: 0.70).",
    )
    p.add_argument(
        "--pupil-model", default=PUPIL_MODEL,
        help=f"PUPIL model on OpenRouter (default: {PUPIL_MODEL}).",
    )
    p.add_argument(
        "--tutor-model", default="claude-opus-4-6",
        help="TUTOR model for DD session (default: claude-opus-4-6).",
    )
    p.add_argument(
        "--validator-model", default="claude-sonnet-4-6",
        help="Validator model for pool validation and grounding (default: claude-sonnet-4-6).",
    )
    p.add_argument(
        "--output", default=".tmp/pupil_dd_experiment.json",
        help="Output JSON report path (default: .tmp/pupil_dd_experiment.json).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for frame sampling (default: 42).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _header(text: str) -> None:
    print(f"\n{'='*3} {text} {'='*(max(0, 60 - len(text)))}")


def _print_summary(summary: dict, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    print(
        f"{prefix}Total={summary['total']} "
        f"Correct={summary['correct']} "
        f"Accuracy={summary['accuracy']:.2%}"
    )
    print(
        f"{prefix}TP={summary['tp']} FP={summary['fp']} "
        f"FN={summary['fn']} TN={summary['tn']} "
        f"Precision={summary['precision']:.2%} "
        f"Recall={summary['recall']:.2%}"
    )
    print(f"{prefix}Confident misses (conf>=0.70): {summary['confident_misses']}")
    dist_str = "  ".join(f"{k}:{v}" for k, v in summary["class_distribution"].items())
    print(f"{prefix}Class distribution: {dist_str}")


def _print_comparison(before: dict, after: dict) -> None:
    """Print a before/after comparison table."""
    metrics = ["accuracy", "precision", "recall", "confident_misses"]
    fmt = "{:<22} {:>12} {:>12} {:>10}"
    print()
    print(fmt.format("Metric", "Before DD", "After DD", "Delta"))
    print("-" * 60)
    for m in metrics:
        bv = before[m]
        av = after[m]
        if isinstance(bv, float):
            delta = av - bv
            print(fmt.format(m, f"{bv:.4f}", f"{av:.4f}", f"{delta:+.4f}"))
        else:
            delta = av - bv
            print(fmt.format(m, str(bv), str(av), f"{delta:+d}"))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

async def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    report: dict = {
        "args": vars(args),
        "phases": {},
    }
    t_global = time.monotonic()

    # -----------------------------------------------------------------------
    # Phase A: Load frames
    # -----------------------------------------------------------------------
    _header("Phase A: Load frames")

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"ERROR: dataset root not found: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    # Resolve annotation file
    if args.annotation_file:
        ann_path = Path(args.annotation_file)
    else:
        ann_path = dataset_root / "annotations" / "instances_val.json"
        if not ann_path.exists():
            for alt in ["val.json", "annotations_val.json"]:
                alt_path = dataset_root / "annotations" / alt
                if alt_path.exists():
                    ann_path = alt_path
                    break

    if not ann_path.exists():
        print(f"ERROR: annotation file not found: {ann_path}", file=sys.stderr)
        print("Use --annotation-file to specify the path explicitly.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading annotations: {ann_path}")
    coco = load_coco(ann_path)
    image_index = build_image_index(coco)
    person_cat_ids = get_person_category_ids(coco)
    person_ann_index = build_annotation_index(coco, person_cat_ids)

    print(f"Building file index...")
    file_index = build_file_index(dataset_root)
    print(f"File index: {len(file_index)} images")

    # Sample positive (person) and negative (no-person) frames
    positives = select_positive_frames(
        image_index=image_index,
        person_ann_index=person_ann_index,
        dataset_root=dataset_root,
        n=args.n_eval_positive,
        hardest=False,
        file_index=file_index,
    )
    negatives = select_negative_frames(
        image_index=image_index,
        person_ann_index=person_ann_index,
        dataset_root=dataset_root,
        n=args.n_eval_negative,
        file_index=file_index,
    )

    if not positives:
        print("ERROR: no person_in_water frames found — check annotation file / "
              "PERSON_CATEGORY_NAMES in pool_builder.py", file=sys.stderr)
        sys.exit(1)

    # Build labeled_frames list: (path, label)
    labeled_frames: list[tuple[str, str]] = []
    for f in positives:
        labeled_frames.append((f["image_path"], "person_in_water"))
    for f in negatives:
        labeled_frames.append((f["image_path"], "no_person"))

    # Shuffle to avoid positives-then-negatives ordering effects
    random.shuffle(labeled_frames)

    print(f"Sampled: {len(positives)} person_in_water, {len(negatives)} no-person frames")
    print(f"Total eval set: {len(labeled_frames)} frames")

    report["phases"]["A"] = {
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "total": len(labeled_frames),
        "annotation_file": str(ann_path),
    }

    # -----------------------------------------------------------------------
    # Phase B: Baseline PUPIL evaluation
    # -----------------------------------------------------------------------
    _header("Phase B: Baseline PUPIL evaluation")

    print(f"Running classify_frame on {len(labeled_frames)} frames "
          f"(model: {args.pupil_model}, n_concurrent=5)...")
    t_b = time.monotonic()

    baseline_results = await run_baseline_eval(
        labeled_frames=labeled_frames,
        model=args.pupil_model,
        n_concurrent=5,
    )

    elapsed_b = int((time.monotonic() - t_b) * 1000)

    # For binary metrics we use person_in_water as positive, but negatives
    # are labeled "no_person" — map to treat any non-person prediction as TN
    # Remap: treat any frame whose ground_truth is not "person_in_water" as negative
    baseline_summary = eval_summary(baseline_results, ground_truth_class="person_in_water")
    _print_summary(baseline_summary, "baseline")

    confident_misses = find_confident_misses(
        baseline_results,
        ground_truth="person_in_water",
        confidence_min=args.confidence_threshold,
    )
    print(f"\nConfident misses at threshold {args.confidence_threshold}: "
          f"{len(confident_misses)}")
    for i, miss in enumerate(confident_misses[:5]):
        print(f"  [{i+1}] conf={miss['confidence']:.3f} "
              f"predicted={miss['predicted_class']} "
              f"  {Path(miss['image_path']).name}")

    report["phases"]["B"] = {
        "summary": baseline_summary,
        "confident_misses_count": len(confident_misses),
        "duration_ms": elapsed_b,
    }

    # -----------------------------------------------------------------------
    # Phase C: Select failure frame for DD session
    # -----------------------------------------------------------------------
    _header("Phase C: Select failure frame for DD session")

    failure_result = None
    effective_threshold = args.confidence_threshold

    if confident_misses:
        failure_result = confident_misses[0]
    else:
        # Lower threshold and retry
        lowered = 0.50
        print(f"WARNING: no confident misses at {args.confidence_threshold}. "
              f"Lowering threshold to {lowered}.")
        effective_threshold = lowered
        confident_misses_low = find_confident_misses(
            baseline_results,
            ground_truth="person_in_water",
            confidence_min=lowered,
        )
        if confident_misses_low:
            failure_result = confident_misses_low[0]
        else:
            # Last resort: pick any misclassified person frame
            any_miss = [
                r for r in baseline_results
                if r.get("ground_truth") == "person_in_water"
                and r.get("predicted_class") != "person_in_water"
            ]
            if any_miss:
                failure_result = sorted(
                    any_miss, key=lambda x: x.get("confidence", 0.0), reverse=True
                )[0]
                print("WARNING: using lowest-confidence miss as fallback failure frame.")

    if failure_result is None:
        print("ERROR: no misclassified person_in_water frames found — "
              "baseline may already be 100% on positives. "
              "Cannot proceed with DD session.", file=sys.stderr)
        report["phases"]["C"] = {"outcome": "no_failure_found"}
        _save_report(report, args.output)
        sys.exit(0)

    failure_image_path = failure_result["image_path"]
    pupil_classification = failure_result["predicted_class"]
    pupil_confidence = failure_result["confidence"]

    print(f"Selected failure frame: {Path(failure_image_path).name}")
    print(f"  Qwen predicted:  {pupil_classification} (conf={pupil_confidence:.3f})")
    print(f"  Ground truth:    person_in_water")

    # Get thermal oracle confirmation
    thermal_conf = oracle_for_frame(
        frame_path=failure_image_path,
        ground_truth_label="person_in_water",
    )
    print(f"  Oracle confirmation: {thermal_conf.confirmation_details[:120]}...")

    report["phases"]["C"] = {
        "failure_image": failure_image_path,
        "pupil_classification": pupil_classification,
        "pupil_confidence": pupil_confidence,
        "effective_threshold": effective_threshold,
        "thermal_confirmation": {
            "modality": thermal_conf.confirmation_modality,
            "details": thermal_conf.confirmation_details,
        },
    }

    # -----------------------------------------------------------------------
    # Phase D: Run DD session
    # -----------------------------------------------------------------------
    _header("Phase D: Run DD session")

    pool_dir = Path(args.pool_dir)
    if not pool_dir.is_dir():
        print(f"ERROR: pool directory not found: {pool_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        pool_images = load_pool(pool_dir)
    except ValueError as e:
        print(f"ERROR loading pool: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Pool loaded: {len(pool_images)} frames from {pool_dir}")
    pool_counts: dict[str, int] = {}
    for _, lbl in pool_images:
        pool_counts[lbl] = pool_counts.get(lbl, 0) + 1
    for lbl, cnt in sorted(pool_counts.items()):
        print(f"  {lbl}: {cnt}")

    pair_info = get_pair_info("person_in_water", pupil_classification)

    t_d = time.monotonic()
    transcript = await run_maritime_dd_session(
        failure_image_path=failure_image_path,
        confirmation_modality=thermal_conf.confirmation_modality,
        confirmation_details=thermal_conf.confirmation_details,
        ground_truth_class="person_in_water",
        pupil_classification=pupil_classification,
        pupil_confidence=pupil_confidence,
        pool_images=pool_images,
        pair_info=pair_info,
        config=MARITIME_SAR_CONFIG,
        tier_observability=TIER_OBSERVABILITY,
        tutor_model=args.tutor_model,
        validator_model=args.validator_model,
        tiers=["scout", "commander"],
    )
    elapsed_d = int((time.monotonic() - t_d) * 1000)

    dd_outcome = transcript.get("outcome", "unknown")
    print(f"\nDD outcome: {dd_outcome.upper()}")
    if dd_outcome == "accepted":
        for tier, rule in transcript.get("final_rules", {}).items():
            print(f"  [{tier}] {rule.get('rule', '')[:120]}")

    report["phases"]["D"] = {
        "outcome": dd_outcome,
        "duration_ms": elapsed_d,
        "transcript": transcript,
    }

    # -----------------------------------------------------------------------
    # Phase E: Re-evaluate with DD rules injected
    # -----------------------------------------------------------------------
    _header("Phase E: Re-evaluate with DD rules injected")

    after_summary = None
    if dd_outcome == "accepted":
        final_rules = transcript.get("final_rules", {})
        scout_rule = final_rules.get("scout")

        if scout_rule:
            # Wrap in list — run_eval_with_rules expects list[dict]
            rules_to_inject = [scout_rule]
            rule_source = "scout"
        elif final_rules:
            # Fall back to first available tier
            rule_source = next(iter(final_rules))
            rules_to_inject = [final_rules[rule_source]]
        else:
            rules_to_inject = []
            rule_source = "none"

        if rules_to_inject:
            print(f"Injecting rules from tier: {rule_source}")
            print(f"Rule: {rules_to_inject[0].get('rule', '')[:120]}")
            preconditions = rules_to_inject[0].get("preconditions", [])
            for pc in preconditions:
                print(f"  pre: {pc[:100]}")

            t_e = time.monotonic()
            after_results = await run_eval_with_rules(
                labeled_frames=labeled_frames,
                rules=rules_to_inject,
                model=args.pupil_model,
                n_concurrent=5,
            )
            elapsed_e = int((time.monotonic() - t_e) * 1000)

            after_summary = eval_summary(after_results, ground_truth_class="person_in_water")
            _print_summary(after_summary, "after_dd")

            print("\nBefore/After comparison:")
            _print_comparison(baseline_summary, after_summary)

            report["phases"]["E"] = {
                "rule_source_tier": rule_source,
                "rules_injected": rules_to_inject,
                "summary": after_summary,
                "duration_ms": elapsed_e,
            }
        else:
            print("No rules available to inject (empty final_rules).")
            report["phases"]["E"] = {"outcome": "no_rules_to_inject"}
    else:
        print(f"DD session did not produce rules (outcome: {dd_outcome}). "
              f"Skipping post-DD evaluation.")
        report["phases"]["E"] = {"outcome": f"dd_{dd_outcome}_skipped"}

    # -----------------------------------------------------------------------
    # Phase F: Save report
    # -----------------------------------------------------------------------
    _header("Phase F: Save report")

    report["summary"] = {
        "baseline": baseline_summary,
        "after_dd": after_summary,
        "dd_outcome": dd_outcome,
        "total_duration_ms": int((time.monotonic() - t_global) * 1000),
    }

    _save_report(report, args.output)


def _save_report(report: dict, output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"Report written to: {out.resolve()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
