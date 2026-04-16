"""
run_dd_session.py — Standalone maritime DD session runner.

Runs a complete SeaPatch DD session given a failure image, thermal confirmation,
and a labeled pool directory. No simulator required — works directly with
SeaDronesSee frames.

Usage:
    python run_dd_session.py \\
        --failure-image path/to/scout_frame.jpg \\
        --confirmation "Thermal camera confirmed 37°C heat signature at sector 4" \\
        --ground-truth person_in_water \\
        --pupil-class whitecap \\
        --pupil-confidence 0.91 \\
        --pool-dir data/pool/ \\
        --output results/session_001.json

Pool directory structure:
    pool/
      person_in_water/   <- positive class frames
      whitecap/          <- negative class frames
      floating_debris/   <- negative class frames (optional)
      pool_manifest.json <- optional; if absent, walks subdirectories

Output JSON contains:
    - initial_rule: cross-modal TUTOR's first rule
    - pool_result: precision/recall on pool
    - final_rules: per-tier adapted rules (scout, commander)
    - grounding_reports: per-tier grounding check details
    - outcome: "accepted" | "pool_failed"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Resolve repo root
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_THIS_DIR))

from agents import run_maritime_dd_session
from domain_config import (
    MARITIME_SAR_CONFIG,
    TIER_OBSERVABILITY,
    CONFUSABLE_PAIRS,
)


# ---------------------------------------------------------------------------
# Pool loader
# ---------------------------------------------------------------------------

_POSITIVE_CLASS = "person_in_water"
_NEGATIVE_CLASSES = {"whitecap", "floating_debris", "life_ring_unoccupied", "no_person"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load_pool(pool_dir: Path) -> list[tuple[str, str]]:
    """Load labeled pool images from a directory.

    Looks for pool_manifest.json first; falls back to walking subdirectories
    where the subdirectory name is the class label.

    Returns list of (image_path, label) tuples.
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
            f"after class labels (e.g. person_in_water/, whitecap/)."
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
    """Find the matching confusable pair or construct a minimal one."""
    for pair in CONFUSABLE_PAIRS:
        classes = {pair["class_a"], pair["class_b"]}
        if ground_truth_class in classes and pupil_class in classes:
            return pair
    # Fallback for unlisted pairs
    return {
        "class_a": ground_truth_class,
        "class_b": pupil_class,
        "pair_id": f"{ground_truth_class}_vs_{pupil_class}",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a maritime DD session on a failure frame + labeled pool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--failure-image", required=True,
        help="Path to the failure frame (scout RGB image classified as wrong class).",
    )
    p.add_argument(
        "--confirmation", required=True,
        help="Confirmation details from the cross-modal sensor (thermal / ground truth).",
    )
    p.add_argument(
        "--confirmation-modality", default="thermal_FLIR",
        help="Label for the confirmation sensor (default: thermal_FLIR).",
    )
    p.add_argument(
        "--ground-truth", default=_POSITIVE_CLASS,
        help=f"Correct class for the failure frame (default: {_POSITIVE_CLASS}).",
    )
    p.add_argument(
        "--pupil-class", default="whitecap",
        help="Class predicted by the scout classifier (default: whitecap).",
    )
    p.add_argument(
        "--pupil-confidence", type=float, default=0.91,
        help="Scout classifier confidence for the wrong prediction (default: 0.91).",
    )
    p.add_argument(
        "--pool-dir", required=True,
        help="Directory containing labeled pool images (subdirs = class labels).",
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
        "--tiers", default="scout,commander",
        help="Comma-separated list of tiers to adapt for (default: scout,commander).",
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

    # Load pool
    try:
        pool_images = load_pool(pool_dir)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    pair_info = get_pair_info(args.ground_truth, args.pupil_class)

    # Optional rich console for progress
    console = None
    if not args.quiet:
        try:
            from rich.console import Console
            console = Console()
            console.print(f"\n[bold]SeaPatch DD Session[/bold]")
            console.print(f"  Failure image : {failure_image}")
            console.print(f"  Confirmation  : {args.confirmation[:80]}")
            console.print(f"  Ground truth  : {args.ground_truth}")
            console.print(f"  Pupil class   : {args.pupil_class} ({args.pupil_confidence:.2f})")
            console.print(f"  Pool ({len(pool_images)} frames):\n{pool_summary(pool_images)}")
            console.print(f"  Tiers         : {tiers}")
        except ImportError:
            pass

    transcript = await run_maritime_dd_session(
        failure_image_path=str(failure_image),
        confirmation_modality=args.confirmation_modality,
        confirmation_details=args.confirmation,
        ground_truth_class=args.ground_truth,
        pupil_classification=args.pupil_class,
        pupil_confidence=args.pupil_confidence,
        pool_images=pool_images,
        pair_info=pair_info,
        config=MARITIME_SAR_CONFIG,
        tier_observability=TIER_OBSERVABILITY,
        tutor_model=args.tutor_model,
        validator_model=args.validator_model,
        tiers=tiers,
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
            print(f"\nOutcome: ACCEPTED")
            for tier, rule in transcript.get("final_rules", {}).items():
                print(f"  [{tier}] {rule.get('rule', '')[:120]}")
        else:
            print(f"\nOutcome: {outcome.upper()}")

    sys.exit(0 if outcome == "accepted" else 1)


if __name__ == "__main__":
    asyncio.run(main())
