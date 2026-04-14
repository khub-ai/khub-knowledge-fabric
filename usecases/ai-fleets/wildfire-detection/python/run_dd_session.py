"""
run_dd_session.py — Standalone PyroWatch wildfire DD session runner.

Runs a complete wildfire DD session given a failure image (sentinel or scout
camera frame), MWIR confirmation text, labeled pool directory, and optional
RAWS weather conditions.

No simulator required — works directly with FIgLib frames or any labeled
image directory.

Usage:
    python run_dd_session.py \\
        --failure-image path/to/sentinel_frame_gs047.jpg \\
        --confirmation "MWIR camera confirmed 380°C surface anomaly at sector 7" \\
        --raws-conditions '{"wind_speed_mph": 28, "relative_humidity_pct": 8, "temperature_f": 95}' \\
        --pool-dir data/pool/ \\
        --tiers ground_sentinel,scout_drone \\
        --output .tmp/session_001.json

Pool directory structure:
    pool/
      early_smoke_signature/   <- positive class frames
      heat_shimmer_artifact/   <- negative class frames
      pool_manifest.json       <- optional; if absent, walks subdirectories

Output JSON contains:
    - initial_rule: cross-modal TUTOR's first candidate rule
    - pool_result: precision/recall on pool
    - grounding_reports: per-tier grounding check details
    - final_rules: per-tier adapted rules with context_preconditions
    - active_condition_set: which RAWS condition set was active at session time
    - outcome: "accepted" | "pool_failed"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_THIS_DIR))

from agents import run_wildfire_dd_session
from domain_config import (
    WILDFIRE_DETECTION_CONFIG,
    TIER_OBSERVABILITY,
    CONFUSABLE_PAIRS,
)


# ---------------------------------------------------------------------------
# Pool loader
# ---------------------------------------------------------------------------

_POSITIVE_CLASS = "early_smoke_signature"
_NEGATIVE_CLASSES = {"heat_shimmer_artifact", "atmospheric_haze", "dust_plume",
                     "fog_patch", "no_fire"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load_pool(pool_dir: Path) -> list[tuple[str, str]]:
    """Load labeled pool images from a directory.

    Reads pool_manifest.json if present; otherwise walks subdirectories
    where the subdirectory name is the class label.
    """
    manifest_path = pool_dir / "pool_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        return [(str(pool_dir / entry["path"]), entry["label"]) for entry in manifest]

    pool: list[tuple[str, str]] = []
    for class_dir in sorted(pool_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in _IMAGE_EXTS:
                pool.append((str(img_path), label))

    if not pool:
        raise ValueError(
            f"No images found in {pool_dir}. Expected subdirectories named "
            f"after class labels (e.g. early_smoke_signature/, heat_shimmer_artifact/)."
        )
    return pool


def pool_summary(pool: list[tuple[str, str]]) -> str:
    counts: dict[str, int] = {}
    for _, label in pool:
        counts[label] = counts.get(label, 0) + 1
    return "  " + "\n  ".join(f"{label}: {n}" for label, n in sorted(counts.items()))


# ---------------------------------------------------------------------------
# Pair info
# ---------------------------------------------------------------------------

def get_pair_info(ground_truth_class: str, pupil_class: str) -> dict:
    for pair in CONFUSABLE_PAIRS:
        classes = {pair["class_a"], pair["class_b"]}
        if ground_truth_class in classes and pupil_class in classes:
            return pair
    return {
        "class_a": ground_truth_class,
        "class_b": pupil_class,
        "pair_id": f"{ground_truth_class}_vs_{pupil_class}",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a PyroWatch wildfire DD session on a failure frame + labeled pool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--failure-image", required=True,
        help="Path to the failure frame (sentinel or scout RGB image classified as wrong class).",
    )
    p.add_argument(
        "--confirmation", required=True,
        help="Confirmation details from the MWIR sensor or ground truth source.",
    )
    p.add_argument(
        "--confirmation-modality", default="MWIR_calibrated_thermal",
        help="Label for the confirmation sensor (default: MWIR_calibrated_thermal).",
    )
    p.add_argument(
        "--ground-truth", default=_POSITIVE_CLASS,
        help=f"Correct class for the failure frame (default: {_POSITIVE_CLASS}).",
    )
    p.add_argument(
        "--pupil-class", default="heat_shimmer_artifact",
        help="Class predicted by the failing classifier (default: heat_shimmer_artifact).",
    )
    p.add_argument(
        "--pupil-confidence", type=float, default=0.94,
        help="Classifier confidence for the wrong prediction (default: 0.94).",
    )
    p.add_argument(
        "--pool-dir", required=True,
        help="Directory containing labeled pool images (subdirs = class labels).",
    )
    p.add_argument(
        "--raws-conditions", default=None,
        help=(
            "JSON string of live RAWS weather station conditions. "
            "Keys: wind_speed_mph, relative_humidity_pct, temperature_f, "
            "fire_weather_watch (bool). "
            'Example: \'{"wind_speed_mph": 28, "relative_humidity_pct": 8, "temperature_f": 95}\''
        ),
    )
    p.add_argument(
        "--primary-tier", default="ground_sentinel",
        choices=list(TIER_OBSERVABILITY.keys()),
        help="Tier whose sensor captured the failure image (default: ground_sentinel).",
    )
    p.add_argument(
        "--tiers", default="ground_sentinel,scout_drone",
        help=(
            "Comma-separated tiers to adapt the rule for. "
            "commander_aircraft is informational only. "
            "(default: ground_sentinel,scout_drone)"
        ),
    )
    p.add_argument(
        "--tutor-model", default="claude-opus-4-6",
        help="Model for TUTOR and contrastive analysis (default: claude-opus-4-6).",
    )
    p.add_argument(
        "--validator-model", default="claude-sonnet-4-6",
        help="Model for pool validation and grounding check (default: claude-sonnet-4-6).",
    )
    p.add_argument(
        "--output", default=None,
        help="Path to write the session transcript JSON (default: stdout).",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output.",
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    failure_image = Path(args.failure_image)
    if not failure_image.exists():
        print(f"ERROR: failure image not found: {failure_image}", file=sys.stderr)
        sys.exit(1)

    pool_dir = Path(args.pool_dir)
    if not pool_dir.is_dir():
        print(f"ERROR: pool directory not found: {pool_dir}", file=sys.stderr)
        sys.exit(1)

    tiers = [t.strip() for t in args.tiers.split(",")]
    unknown_tiers = [t for t in tiers if t not in TIER_OBSERVABILITY]
    if unknown_tiers:
        print(f"ERROR: unknown tiers: {unknown_tiers}. "
              f"Available: {list(TIER_OBSERVABILITY.keys())}", file=sys.stderr)
        sys.exit(1)

    # Parse RAWS conditions
    raws_conditions: dict | None = None
    if args.raws_conditions:
        try:
            raws_conditions = json.loads(args.raws_conditions)
        except json.JSONDecodeError as e:
            print(f"ERROR: --raws-conditions is not valid JSON: {e}", file=sys.stderr)
            sys.exit(1)

    # Load pool
    try:
        pool_images = load_pool(pool_dir)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    pair_info = get_pair_info(args.ground_truth, args.pupil_class)

    # Optional rich console
    console = None
    if not args.quiet:
        try:
            from rich.console import Console
            console = Console()
            console.print(f"\n[bold]PyroWatch DD Session[/bold]")
            console.print(f"  Failure image : {failure_image}")
            console.print(f"  Confirmation  : {args.confirmation[:80]}")
            console.print(f"  Ground truth  : {args.ground_truth}")
            console.print(f"  Pupil class   : {args.pupil_class} ({args.pupil_confidence:.2f})")
            console.print(f"  Primary tier  : {args.primary_tier}")
            console.print(f"  Pool ({len(pool_images)} frames):\n{pool_summary(pool_images)}")
            console.print(f"  Tiers         : {tiers}")
            if raws_conditions:
                console.print(f"  RAWS          : {raws_conditions}")
        except ImportError:
            pass

    transcript = await run_wildfire_dd_session(
        failure_image_path=str(failure_image),
        confirmation_modality=args.confirmation_modality,
        confirmation_details=args.confirmation,
        ground_truth_class=args.ground_truth,
        pupil_classification=args.pupil_class,
        pupil_confidence=args.pupil_confidence,
        pool_images=pool_images,
        pair_info=pair_info,
        raws_conditions=raws_conditions,
        config=WILDFIRE_DETECTION_CONFIG,
        tier_observability=TIER_OBSERVABILITY,
        tutor_model=args.tutor_model,
        validator_model=args.validator_model,
        tiers=tiers,
        primary_tier=args.primary_tier,
        console=console,
    )

    output_json = json.dumps(transcript, indent=2, default=str)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_json)
        if not args.quiet:
            print(f"\nTranscript written to {out_path}")
    else:
        print(output_json)

    outcome = transcript.get("outcome", "unknown")
    if not args.quiet:
        if outcome == "accepted":
            active_set = transcript.get("active_condition_set", "unknown")
            print(f"\nOutcome: ACCEPTED  (condition set: {active_set})")
            for tier, rule in transcript.get("final_rules", {}).items():
                print(f"  [{tier}] {rule.get('rule', '')[:120]}")
                ctx = rule.get("context_preconditions", {})
                if ctx:
                    print(f"    context sets: {list(ctx.keys())}")
        else:
            print(f"\nOutcome: {outcome.upper()}")

    sys.exit(0 if outcome == "accepted" else 1)


if __name__ == "__main__":
    asyncio.run(main())
