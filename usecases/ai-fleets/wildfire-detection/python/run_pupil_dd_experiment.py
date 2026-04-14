"""
run_pupil_dd_experiment.py — End-to-end PUPIL + DD experiment runner for PyroWatch.

Measures how much a single wildfire DD session improves Claude Haiku's
early_smoke_signature classification accuracy on FIgLib frames.

Phases:
  A — Extract FIgLib tarball sequences and label frames by offset
  B — Baseline PUPIL evaluation (Haiku, no rules)
  C — Select the most confident miss as the DD failure frame
  D — Run a full wildfire DD session (MWIR oracle + TUTOR + pool + tier grounding)
  E — Re-evaluate with DD rules injected into the PUPIL prompt
  F — Save full JSON report

Usage:
    python run_pupil_dd_experiment.py \\
        --dataset-root "V:/_mlArchive/data/images/FIgLib" \\
        --output .tmp/pyrowatch_dd_experiment.json

    # With RAWS conditions (Santa Ana event):
    python run_pupil_dd_experiment.py \\
        --dataset-root "V:/_mlArchive/data/images/FIgLib" \\
        --raws-conditions '{"wind_speed_mph": 28, "relative_humidity_pct": 8, "temperature_f": 95}' \\
        --output .tmp/pyrowatch_dd_experiment_redFlag.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import tarfile
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Repo-root and local path setup
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[4]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR / "simulation"))

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from agents import run_wildfire_dd_session
from domain_config import WILDFIRE_DETECTION_CONFIG, TIER_OBSERVABILITY, CONFUSABLE_PAIRS
from pupil_eval import (
    PUPIL_MODEL,
    run_baseline_eval,
    run_eval_with_rules,
    find_confident_misses,
    eval_summary,
)
from simulation.mwir_oracle import oracle_for_frame


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Offset ranges (seconds from ignition) for frame labeling
POSITIVE_OFFSET_MIN = 0       # Ignition moment (offset = 0 or positive)
POSITIVE_OFFSET_MAX = 600     # 10 minutes post-ignition: early smoke, often ambiguous
NEGATIVE_OFFSET_MAX = -600    # Well before ignition: clean terrain (offset < -600)

# FIgLib filename pattern: <unix_timestamp>_<+/-NNNNN>.jpg
_OFFSET_RE = re.compile(r"_([+-]\d+)\.jpg$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a full PUPIL + DD improvement experiment on FIgLib.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset-root", required=True,
        help="Directory containing FIgLib .tgz tarball files.",
    )
    p.add_argument(
        "--extract-dir", default=".tmp/figlib_frames",
        help="Directory for extracted frames (default: .tmp/figlib_frames).",
    )
    p.add_argument(
        "--n-eval-sequences", type=int, default=4,
        help="Number of FIgLib sequences to use for evaluation (default: 4).",
    )
    p.add_argument(
        "--n-pool-sequences", type=int, default=2,
        help="Number of FIgLib sequences to use for DD pool (default: 2).",
    )
    p.add_argument(
        "--n-eval-positive", type=int, default=20,
        help="Max positive frames (offset 0..+600s) to evaluate (default: 20).",
    )
    p.add_argument(
        "--n-eval-negative", type=int, default=20,
        help="Max negative frames (offset < -600s) to evaluate (default: 20).",
    )
    p.add_argument(
        "--n-pool-positive", type=int, default=8,
        help="Positive frames in DD pool (default: 8).",
    )
    p.add_argument(
        "--n-pool-negative", type=int, default=10,
        help="Negative frames in DD pool (default: 10).",
    )
    p.add_argument(
        "--confidence-threshold", type=float, default=0.70,
        help="Min confidence to count as a confident miss (default: 0.70).",
    )
    p.add_argument(
        "--pupil-model", default=PUPIL_MODEL,
        help=f"PUPIL model (default: {PUPIL_MODEL}).",
    )
    p.add_argument(
        "--tutor-model", default="claude-opus-4-6",
        help="TUTOR model for DD session (default: claude-opus-4-6).",
    )
    p.add_argument(
        "--validator-model", default="claude-sonnet-4-6",
        help="Validator model for pool validation (default: claude-sonnet-4-6).",
    )
    p.add_argument(
        "--raws-conditions", default=None,
        help=(
            "JSON string with RAWS meteorological conditions, e.g. "
            "'{\"wind_speed_mph\": 28, \"relative_humidity_pct\": 8, "
            "\"temperature_f\": 95}'. "
            "Triggers environmental context injection in DD session."
        ),
    )
    p.add_argument(
        "--primary-tier", default="ground_sentinel",
        choices=list(TIER_OBSERVABILITY.keys()),
        help="Primary evaluation tier for rule injection (default: ground_sentinel).",
    )
    p.add_argument(
        "--tiers", default="ground_sentinel,scout_drone",
        help="Comma-separated tiers for DD session (default: ground_sentinel,scout_drone).",
    )
    p.add_argument(
        "--output", default=".tmp/pyrowatch_dd_experiment.json",
        help="Output JSON report path (default: .tmp/pyrowatch_dd_experiment.json).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for frame sampling (default: 42).",
    )
    p.add_argument(
        "--no-extract", action="store_true",
        help="Skip extraction if frames already exist in --extract-dir.",
    )
    p.add_argument(
        "--skip-pool-validation", action="store_true",
        help=(
            "Accept the TUTOR's initial rule without pool validation "
            "(use when the validator model is rate-limited)."
        ),
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
        f"Recall={summary['recall']:.2%} "
        f"F1={summary['f1']:.2%}"
    )
    print(f"{prefix}Confident misses (conf>=0.70): {summary['confident_misses']}")
    if summary.get("confusion"):
        conf_str = "  ".join(
            f"{k}:{v}" for k, v in sorted(summary["confusion"].items())
        )
        print(f"{prefix}Confusion on positives: {conf_str}")
    dist_str = "  ".join(f"{k}:{v}" for k, v in summary["class_distribution"].items())
    print(f"{prefix}Class distribution: {dist_str}")


def _print_comparison(before: dict, after: dict) -> None:
    """Print a before/after comparison table."""
    metrics = ["accuracy", "precision", "recall", "f1", "confident_misses"]
    fmt = "{:<24} {:>12} {:>12} {:>10}"
    print()
    print(fmt.format("Metric", "Before DD", "After DD", "Delta"))
    print("-" * 62)
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
# FIgLib frame extraction
# ---------------------------------------------------------------------------

def _parse_offset(filename: str) -> Optional[int]:
    """Parse the signed offset in seconds from a FIgLib filename.

    FIgLib naming: <unix_timestamp>_<+/-NNNNN>.jpg
    Examples: 1465065720_+00120.jpg → +120
              1465063380_-02220.jpg → -2220
    """
    m = _OFFSET_RE.search(filename)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return None


def extract_sequences(
    tarballs_dir: Path,
    extract_dir: Path,
    sequence_names: list[str],
    skip_if_exists: bool = False,
) -> dict[str, Path]:
    """Extract named FIgLib sequences to extract_dir.

    Parameters
    ----------
    tarballs_dir:
        Directory containing .tgz files.
    extract_dir:
        Destination for extracted sequence directories.
    sequence_names:
        List of tarball stem names (e.g. "20160604_FIRE_rm-n-mobo-c").
    skip_if_exists:
        If True, skip extraction when the sequence directory already exists.

    Returns
    -------
    dict mapping sequence_name → Path to the extracted sequence directory.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}

    for name in sequence_names:
        tarball = tarballs_dir / f"{name}.tgz"
        if not tarball.exists():
            print(f"  WARNING: tarball not found: {tarball}", file=sys.stderr)
            continue

        seq_dir = extract_dir / name
        if skip_if_exists and seq_dir.exists():
            print(f"  [skip] {name} (already extracted)")
            result[name] = seq_dir
            continue

        print(f"  Extracting {name}...")
        with tarfile.open(str(tarball), "r:gz") as tf:
            tf.extractall(str(extract_dir))

        if seq_dir.exists():
            result[name] = seq_dir
        else:
            print(f"  WARNING: expected dir {seq_dir} not found after extraction")

    return result


def label_frames_from_sequences(
    seq_dirs: dict[str, Path],
    positive_label: str = "early_smoke_signature",
    negative_label: str = "no_fire",
    pos_min: int = POSITIVE_OFFSET_MIN,
    pos_max: int = POSITIVE_OFFSET_MAX,
    neg_max: int = NEGATIVE_OFFSET_MAX,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Scan extracted sequences and label frames by signed offset.

    Returns
    -------
    (positives, negatives) — each is a list of (image_path, label) tuples.
    """
    positives: list[tuple[str, str]] = []
    negatives: list[tuple[str, str]] = []

    for seq_name, seq_dir in seq_dirs.items():
        frames = sorted(seq_dir.glob("*.jpg"))
        for frame in frames:
            offset = _parse_offset(frame.name)
            if offset is None:
                continue
            if pos_min <= offset <= pos_max:
                positives.append((str(frame), positive_label))
            elif offset < neg_max:
                negatives.append((str(frame), negative_label))
            # Frames with offset in (neg_max..pos_min) are excluded (ambiguous window)

    return positives, negatives


# ---------------------------------------------------------------------------
# Pool loading (for DD session)
# ---------------------------------------------------------------------------

def build_pool_images(
    seq_dirs: dict[str, Path],
    n_positive: int,
    n_negative: int,
    positive_label: str = "early_smoke_signature",
    negative_label: str = "no_fire",
) -> list[tuple[str, str]]:
    """Build a pool of (image_path, label) tuples for the DD session.

    Selects the earliest post-ignition frames as positives (most likely to
    be confused with shimmer/haze) and late pre-ignition frames as negatives.
    """
    positives, negatives = label_frames_from_sequences(
        seq_dirs,
        positive_label=positive_label,
        negative_label=negative_label,
    )
    # Pool positives: prefer offset 0..+120 (earliest, hardest)
    pool_pos = positives[:n_positive] if positives else []
    pool_neg = random.sample(negatives, min(n_negative, len(negatives))) if negatives else []

    return pool_pos + pool_neg


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

async def main() -> None:  # noqa: C901
    args = parse_args()
    random.seed(args.seed)

    report: dict = {
        "args": vars(args),
        "phases": {},
    }
    t_global = time.monotonic()

    # -----------------------------------------------------------------------
    # Phase A: Extract FIgLib sequences and label frames
    # -----------------------------------------------------------------------
    _header("Phase A: Extract FIgLib sequences and label frames")

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"ERROR: dataset root not found: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    # Discover all tarballs and shuffle
    all_tarballs = sorted(dataset_root.glob("*.tgz"))
    if not all_tarballs:
        print(f"ERROR: no .tgz files found in {dataset_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(all_tarballs)} FIgLib sequences")

    rng = random.Random(args.seed)
    shuffled = list(all_tarballs)
    rng.shuffle(shuffled)

    n_total = args.n_pool_sequences + args.n_eval_sequences
    if n_total > len(shuffled):
        print(
            f"WARNING: requested {n_total} sequences but only {len(shuffled)} available. "
            f"Using all.",
            file=sys.stderr,
        )
        n_total = len(shuffled)

    pool_tarballs = shuffled[:args.n_pool_sequences]
    eval_tarballs = shuffled[args.n_pool_sequences:n_total]

    pool_seq_names = [t.stem for t in pool_tarballs]
    eval_seq_names = [t.stem for t in eval_tarballs]

    print(f"\nPool sequences ({len(pool_seq_names)}):")
    for name in pool_seq_names:
        print(f"  {name}")
    print(f"\nEval sequences ({len(eval_seq_names)}):")
    for name in eval_seq_names:
        print(f"  {name}")

    extract_dir = Path(args.extract_dir)
    pool_extract = extract_dir / "pool"
    eval_extract = extract_dir / "eval"

    # Extract pool sequences
    print("\nExtracting pool sequences...")
    pool_seq_dirs = extract_sequences(
        tarballs_dir=dataset_root,
        extract_dir=pool_extract,
        sequence_names=pool_seq_names,
        skip_if_exists=args.no_extract,
    )

    # Extract eval sequences
    print("Extracting eval sequences...")
    eval_seq_dirs = extract_sequences(
        tarballs_dir=dataset_root,
        extract_dir=eval_extract,
        sequence_names=eval_seq_names,
        skip_if_exists=args.no_extract,
    )

    if not eval_seq_dirs:
        print("ERROR: no eval sequences extracted.", file=sys.stderr)
        sys.exit(1)

    # Label eval frames
    all_positives, all_negatives = label_frames_from_sequences(eval_seq_dirs)

    print(f"\nEval frames found:")
    print(f"  positive (offset 0..+600s):   {len(all_positives)}")
    print(f"  negative (offset < -600s):     {len(all_negatives)}")

    if not all_positives:
        print(
            "ERROR: no positive frames found in eval sequences. "
            "Offset labeling may have failed.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Sample and shuffle eval set
    sampled_pos = all_positives[:args.n_eval_positive]
    sampled_neg = (
        random.sample(all_negatives, min(args.n_eval_negative, len(all_negatives)))
    )
    labeled_frames: list[tuple[str, str]] = sampled_pos + sampled_neg
    random.shuffle(labeled_frames)

    print(f"Eval set: {len(sampled_pos)} positive + {len(sampled_neg)} negative = "
          f"{len(labeled_frames)} total")

    # Build pool images
    pool_images = build_pool_images(
        seq_dirs=pool_seq_dirs,
        n_positive=args.n_pool_positive,
        n_negative=args.n_pool_negative,
    )
    pool_counts: dict[str, int] = {}
    for _, lbl in pool_images:
        pool_counts[lbl] = pool_counts.get(lbl, 0) + 1
    print(f"Pool: {pool_counts}")

    report["phases"]["A"] = {
        "pool_sequences": pool_seq_names,
        "eval_sequences": eval_seq_names,
        "n_eval_positive_found": len(all_positives),
        "n_eval_negative_found": len(all_negatives),
        "n_eval_positive_sampled": len(sampled_pos),
        "n_eval_negative_sampled": len(sampled_neg),
        "total_eval_frames": len(labeled_frames),
        "pool_composition": pool_counts,
        "extract_dir": str(extract_dir),
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

    baseline_summary = eval_summary(
        baseline_results, ground_truth_class="early_smoke_signature"
    )
    _print_summary(baseline_summary, "baseline")

    confident_misses = find_confident_misses(
        baseline_results,
        ground_truth="early_smoke_signature",
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
        lowered = 0.50
        print(f"WARNING: no confident misses at {args.confidence_threshold}. "
              f"Lowering threshold to {lowered}.")
        effective_threshold = lowered
        misses_low = find_confident_misses(
            baseline_results,
            ground_truth="early_smoke_signature",
            confidence_min=lowered,
        )
        if misses_low:
            failure_result = misses_low[0]
        else:
            any_miss = [
                r for r in baseline_results
                if r.get("ground_truth") == "early_smoke_signature"
                and r.get("predicted_class") != "early_smoke_signature"
            ]
            if any_miss:
                failure_result = sorted(
                    any_miss, key=lambda x: x.get("confidence", 0.0), reverse=True
                )[0]
                print("WARNING: using lowest-confidence miss as fallback failure frame.")

    if failure_result is None:
        print(
            "ERROR: no misclassified early_smoke_signature frames found — "
            "baseline may already be 100% on positives. "
            "Cannot proceed with DD session.",
            file=sys.stderr,
        )
        report["phases"]["C"] = {"outcome": "no_failure_found"}
        _save_report(report, args.output)
        sys.exit(0)

    failure_image_path = failure_result["image_path"]
    pupil_classification = failure_result["predicted_class"]
    pupil_confidence = failure_result["confidence"]

    print(f"Selected failure frame: {Path(failure_image_path).name}")
    print(f"  PUPIL predicted:  {pupil_classification} (conf={pupil_confidence:.3f})")
    print(f"  Ground truth:     early_smoke_signature")
    print(f"  Reasoning: {failure_result.get('reasoning', '')[:120]}")

    # MWIR oracle confirmation (simulated ground truth)
    mwir_conf = oracle_for_frame(
        frame_path=failure_image_path,
        ground_truth_label="early_smoke_signature",
        agent_id="CA-1",
        coordinates=None,
    )
    print(f"\n  MWIR oracle: {mwir_conf.confirmation_details[:160]}...")

    report["phases"]["C"] = {
        "failure_image": failure_image_path,
        "pupil_classification": pupil_classification,
        "pupil_confidence": pupil_confidence,
        "pupil_reasoning": failure_result.get("reasoning", ""),
        "effective_threshold": effective_threshold,
        "mwir_confirmation": {
            "modality": mwir_conf.confirmation_modality,
            "details": mwir_conf.confirmation_details,
            "confidence": mwir_conf.confidence,
        },
    }

    # -----------------------------------------------------------------------
    # Phase D: Run DD session
    # -----------------------------------------------------------------------
    _header("Phase D: Run DD session")

    if not pool_images:
        print("ERROR: pool is empty — cannot run DD session.", file=sys.stderr)
        sys.exit(1)

    print(f"Pool: {len(pool_images)} frames ({pool_counts})")
    print(f"TUTOR model:     {args.tutor_model}")
    print(f"Validator model: {args.validator_model}")
    print(f"Tiers:           {args.tiers}")

    raws_conditions: Optional[dict] = None
    if args.raws_conditions:
        try:
            raws_conditions = json.loads(args.raws_conditions)
            print(f"RAWS conditions: {raws_conditions}")
        except json.JSONDecodeError as e:
            print(f"WARNING: could not parse --raws-conditions JSON: {e}")

    # Build pair_info for the confusable pair
    pair_info = {
        "positive": "early_smoke_signature",
        "confusable": pupil_classification,
        "pair": f"early_smoke_signature_vs_{pupil_classification}",
        "description": (
            f"PUPIL predicted '{pupil_classification}' for a frame that MWIR "
            f"confirms is 'early_smoke_signature'. "
            f"Explore what distinguishes early combustion smoke from {pupil_classification}."
        ),
    }

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]

    t_d = time.monotonic()
    transcript = await run_wildfire_dd_session(
        failure_image_path=failure_image_path,
        confirmation_modality=mwir_conf.confirmation_modality,
        confirmation_details=mwir_conf.confirmation_details,
        ground_truth_class="early_smoke_signature",
        pupil_classification=pupil_classification,
        pupil_confidence=pupil_confidence,
        pool_images=pool_images,
        pair_info=pair_info,
        config=WILDFIRE_DETECTION_CONFIG,
        tier_observability=TIER_OBSERVABILITY,
        raws_conditions=raws_conditions,
        tutor_model=args.tutor_model,
        validator_model=args.validator_model,
        tiers=tiers,
        skip_pool_validation=args.skip_pool_validation,
    )
    elapsed_d = int((time.monotonic() - t_d) * 1000)

    dd_outcome = transcript.get("outcome", "unknown")
    print(f"\nDD outcome: {dd_outcome.upper()}")
    if dd_outcome == "accepted":
        for tier, rule in transcript.get("final_rules", {}).items():
            rule_text = rule.get("rule", "") if isinstance(rule, dict) else str(rule)
            print(f"  [{tier}] {rule_text[:140]}")
    else:
        reason = transcript.get("rejection_reason", "")
        if reason:
            print(f"  Rejection reason: {reason[:200]}")

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

        # Prefer primary tier; fall back to any available tier
        primary_tier = args.primary_tier
        if primary_tier in final_rules:
            rule_entry = final_rules[primary_tier]
            rule_source = primary_tier
        elif final_rules:
            rule_source = next(iter(final_rules))
            rule_entry = final_rules[rule_source]
        else:
            rule_entry = None
            rule_source = "none"

        if rule_entry is not None:
            # Normalise rule entry to a list[dict] for run_eval_with_rules
            if isinstance(rule_entry, dict):
                rules_to_inject = [rule_entry]
            elif isinstance(rule_entry, list):
                rules_to_inject = rule_entry
            else:
                rules_to_inject = [{"rule": str(rule_entry)}]

            print(f"Injecting rules from tier: {rule_source}")
            rule_text = rules_to_inject[0].get("rule", "") if isinstance(rules_to_inject[0], dict) else str(rules_to_inject[0])
            print(f"Rule: {rule_text[:160]}")
            preconditions = rules_to_inject[0].get("preconditions", []) if isinstance(rules_to_inject[0], dict) else []
            for pc in preconditions:
                print(f"  pre: {pc[:120]}")

            t_e = time.monotonic()
            after_results = await run_eval_with_rules(
                labeled_frames=labeled_frames,
                rules=rules_to_inject,
                model=args.pupil_model,
                n_concurrent=5,
            )
            elapsed_e = int((time.monotonic() - t_e) * 1000)

            after_summary = eval_summary(
                after_results, ground_truth_class="early_smoke_signature"
            )
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
    _print_measured_results(baseline_summary, after_summary, dd_outcome)


def _save_report(report: dict, output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"Report written to: {out.resolve()}")


def _print_measured_results(
    before: dict,
    after: Optional[dict],
    dd_outcome: str,
) -> None:
    """Print §10-style measured results summary for README inclusion."""
    print()
    print("=" * 70)
    print("MEASURED RESULTS — paste into README §10")
    print("=" * 70)
    if before:
        print(f"Baseline (no DD rules):")
        print(f"  Recall (early smoke): {before['recall']:.1%}")
        print(f"  Precision:            {before['precision']:.1%}")
        print(f"  Accuracy:             {before['accuracy']:.1%}")
        print(f"  Confident misses:     {before['confident_misses']}")
    if after and dd_outcome == "accepted":
        recall_delta = after["recall"] - before["recall"]
        miss_delta = after["confident_misses"] - before["confident_misses"]
        print(f"\nAfter single DD session:")
        print(f"  Recall (early smoke): {after['recall']:.1%}  ({recall_delta:+.1%})")
        print(f"  Precision:            {after['precision']:.1%}")
        print(f"  Accuracy:             {after['accuracy']:.1%}")
        print(f"  Confident misses:     {after['confident_misses']}  ({miss_delta:+d})")
    else:
        print(f"\nDD session outcome: {dd_outcome} — no after-DD results.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
