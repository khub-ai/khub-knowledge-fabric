"""
pool_builder.py — FIgLib labeled pool construction for PyroWatch.

Processes the FIgLib (Fire Ignition image Library) dataset to extract:
  - Early smoke frames (positive class: "early_smoke_signature")
  - Non-fire frames (negative class: "heat_shimmer_artifact" or "no_fire")

FIgLib dataset: https://github.com/brain-facens/FIgLib
Expected structure (two common layouts — both supported):

  Layout A (flat binary):
    <dataset_root>/
      fire/
        img_0001.jpg
        ...
      non-fire/
        img_0001.jpg
        ...

  Layout B (sequence-based, with timestamps):
    <dataset_root>/
      sequences/
        <camera_id>_<date>/
          <timestamp>_fire.jpg
          <timestamp>_non-fire.jpg
          ...

Pool output is a directory ready for run_dd_session.py.

Usage:
    # Build pool from FIgLib flat layout
    python pool_builder.py \\
        --dataset-root data/figlib/ \\
        --pool-dir .tmp/pool/ \\
        --n-positive 10 \\
        --n-negative 16 \\
        --select-earliest

    # Also extract candidate failure frames
    python pool_builder.py \\
        --dataset-root data/figlib/ \\
        --pool-dir .tmp/pool/ \\
        --failure-dir .tmp/failure_frames/ \\
        --n-failure 5

    # Use ALERTWildfire non-fire frames as negatives instead of FIgLib
    python pool_builder.py \\
        --dataset-root data/figlib/ \\
        --negative-dir data/alertwildfire_nonfires/ \\
        --pool-dir .tmp/pool/
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POSITIVE_LABEL = "early_smoke_signature"
NEGATIVE_LABEL = "heat_shimmer_artifact"

# FIgLib uses these directory names (may vary by release)
FIGLIB_FIRE_DIRS = {"fire", "Fire", "FIRE", "smoke", "Smoke", "positive"}
FIGLIB_NONFIRE_DIRS = {"non-fire", "Non-Fire", "NON-FIRE", "non_fire", "no-fire",
                       "nofire", "negative", "normal", "background"}

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Regex to detect "fire" timestamp suffix in Layout B sequence dirs
_FIRE_SUFFIX_RE = re.compile(r"_fire\b", re.IGNORECASE)
_NONFIRE_SUFFIX_RE = re.compile(r"_(non.?fire|no.?fire|normal|background)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Dataset layout detection
# ---------------------------------------------------------------------------

def detect_layout(dataset_root: Path) -> str:
    """Return 'flat', 'sequence', or 'unknown'."""
    children = [p.name for p in dataset_root.iterdir() if p.is_dir()]
    if any(c in FIGLIB_FIRE_DIRS for c in children):
        return "flat"
    sequences_dir = dataset_root / "sequences"
    if sequences_dir.exists():
        return "sequence"
    # Recurse one level
    for child in dataset_root.iterdir():
        if child.is_dir():
            sub = [p.name for p in child.iterdir() if p.is_dir()]
            if any(s in FIGLIB_FIRE_DIRS for s in sub):
                return "flat_nested"
    return "unknown"


# ---------------------------------------------------------------------------
# Frame discovery
# ---------------------------------------------------------------------------

def find_fire_frames(dataset_root: Path) -> list[Path]:
    """Return all fire/smoke frame paths from any supported FIgLib layout."""
    layout = detect_layout(dataset_root)
    frames: list[Path] = []

    if layout in ("flat", "flat_nested"):
        search_roots = [dataset_root]
        if layout == "flat_nested":
            search_roots = [p for p in dataset_root.iterdir() if p.is_dir()]
        for root in search_roots:
            for d in root.iterdir():
                if d.is_dir() and d.name in FIGLIB_FIRE_DIRS:
                    frames.extend(
                        p for p in sorted(d.rglob("*"))
                        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
                    )

    elif layout == "sequence":
        for seq_dir in sorted((dataset_root / "sequences").iterdir()):
            if seq_dir.is_dir():
                for img in sorted(seq_dir.iterdir()):
                    if img.is_file() and img.suffix.lower() in _IMAGE_EXTS:
                        if _FIRE_SUFFIX_RE.search(img.stem):
                            frames.append(img)

    else:
        # Generic fallback: walk entire tree for files in fire-named dirs
        for p in dataset_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                if any(part in FIGLIB_FIRE_DIRS for part in p.parts):
                    frames.append(p)

    return frames


def find_nonfire_frames(dataset_root: Path,
                        extra_dirs: list[Path] | None = None) -> list[Path]:
    """Return all non-fire frame paths from FIgLib or supplementary directories."""
    layout = detect_layout(dataset_root)
    frames: list[Path] = []

    if layout in ("flat", "flat_nested"):
        search_roots = [dataset_root]
        if layout == "flat_nested":
            search_roots = [p for p in dataset_root.iterdir() if p.is_dir()]
        for root in search_roots:
            for d in root.iterdir():
                if d.is_dir() and d.name in FIGLIB_NONFIRE_DIRS:
                    frames.extend(
                        p for p in sorted(d.rglob("*"))
                        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
                    )

    elif layout == "sequence":
        for seq_dir in sorted((dataset_root / "sequences").iterdir()):
            if seq_dir.is_dir():
                for img in sorted(seq_dir.iterdir()):
                    if img.is_file() and img.suffix.lower() in _IMAGE_EXTS:
                        if _NONFIRE_SUFFIX_RE.search(img.stem):
                            frames.append(img)

    else:
        for p in dataset_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                if any(part in FIGLIB_NONFIRE_DIRS for part in p.parts):
                    frames.append(p)

    # Supplementary non-fire directories (ALERTWildfire, HPWREN, etc.)
    if extra_dirs:
        for extra in extra_dirs:
            if extra.is_dir():
                frames.extend(
                    p for p in sorted(extra.rglob("*"))
                    if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
                )

    return frames


# ---------------------------------------------------------------------------
# Frame metadata — FIgLib sequence information
# ---------------------------------------------------------------------------

def extract_sequence_metadata(path: Path) -> dict:
    """Extract camera ID, date, and sequence position from a FIgLib path.

    Layout B paths contain camera and date in the parent directory name.
    Layout A paths have no sequence structure.
    """
    meta: dict = {"path": str(path), "sequence": None, "sequence_position": None}

    # Layout B: parent dir is <camera_id>_<date>
    parent = path.parent
    if re.match(r"[A-Za-z0-9]+_\d{4}", parent.name):
        meta["sequence"] = parent.name
        # Sequence position: sort frames in the dir, find index
        siblings = sorted(
            p for p in parent.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        )
        try:
            meta["sequence_position"] = siblings.index(path)
            meta["sequence_length"] = len(siblings)
        except ValueError:
            pass

    return meta


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------

def select_fire_frames(
    frames: list[Path],
    n: int,
    select_earliest: bool = False,
) -> list[dict]:
    """Select n fire frames, optionally preferring earliest frames in each sequence.

    Earliest frames in an ignition sequence represent the hardest cases:
    smallest, faintest smoke, most similar to heat shimmer. These are the
    direct analogue of SeaDronesSee's 'smallest bounding box' selection.
    """
    candidates = [extract_sequence_metadata(f) for f in frames]

    if select_earliest:
        # Sort by sequence position (earliest first), then by path for stability
        candidates.sort(key=lambda m: (m.get("sequence_position") or 999, str(m["path"])))
    else:
        random.shuffle(candidates)

    return candidates[:n]


def select_nonfire_frames(frames: list[Path], n: int) -> list[dict]:
    """Select n non-fire frames at random."""
    candidates = [{"path": str(f)} for f in frames]
    random.shuffle(candidates)
    return candidates[:n]


# ---------------------------------------------------------------------------
# Pool output
# ---------------------------------------------------------------------------

def copy_frames_to_pool(
    positives: list[dict],
    negatives: list[dict],
    pool_dir: Path,
    positive_label: str = POSITIVE_LABEL,
    negative_label: str = NEGATIVE_LABEL,
    failure_frames: list[dict] | None = None,
    failure_dir: Path | None = None,
) -> list[dict]:
    """Copy frames into pool directory structure and write manifest."""
    pool_dir.mkdir(parents=True, exist_ok=True)
    (pool_dir / positive_label).mkdir(exist_ok=True)
    (pool_dir / negative_label).mkdir(exist_ok=True)

    manifest = []

    for i, frame in enumerate(positives):
        src = Path(frame["path"])
        dest_name = f"pos_{i:04d}{src.suffix}"
        dest = pool_dir / positive_label / dest_name
        shutil.copy2(src, dest)
        entry = {
            "path": f"{positive_label}/{dest_name}",
            "label": positive_label,
            "source": str(src),
        }
        if frame.get("sequence"):
            entry["sequence"] = frame["sequence"]
            entry["sequence_position"] = frame.get("sequence_position")
        manifest.append(entry)

    for i, frame in enumerate(negatives):
        src = Path(frame["path"])
        dest_name = f"neg_{i:04d}{src.suffix}"
        dest = pool_dir / negative_label / dest_name
        shutil.copy2(src, dest)
        manifest.append({
            "path": f"{negative_label}/{dest_name}",
            "label": negative_label,
            "source": str(src),
        })

    (pool_dir / "pool_manifest.json").write_text(json.dumps(manifest, indent=2))

    if failure_frames and failure_dir:
        failure_dir.mkdir(parents=True, exist_ok=True)
        failure_manifest = []
        for i, frame in enumerate(failure_frames):
            src = Path(frame["path"])
            dest_name = f"failure_{i:04d}{src.suffix}"
            dest = failure_dir / dest_name
            shutil.copy2(src, dest)
            entry = {
                "path": dest_name,
                "label": positive_label,
                "source": str(src),
                "note": "Earliest ignition frame — most similar to heat shimmer",
            }
            if frame.get("sequence"):
                entry["sequence"] = frame["sequence"]
                entry["sequence_position"] = frame.get("sequence_position")
            failure_manifest.append(entry)
        (failure_dir / "failure_manifest.json").write_text(
            json.dumps(failure_manifest, indent=2)
        )
        print(f"Failure frames: {len(failure_frames)} written to {failure_dir}")

    return manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a labeled pool from FIgLib for wildfire DD sessions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset-root", required=True,
        help="Root directory of the FIgLib dataset.",
    )
    p.add_argument(
        "--pool-dir", required=True,
        help="Output directory for the labeled pool.",
    )
    p.add_argument(
        "--n-positive", type=int, default=10,
        help="Number of fire/smoke frames to include (default: 10).",
    )
    p.add_argument(
        "--n-negative", type=int, default=16,
        help="Number of non-fire frames to include (default: 16).",
    )
    p.add_argument(
        "--negative-label", default=NEGATIVE_LABEL,
        help=f"Label for non-fire frames (default: {NEGATIVE_LABEL}).",
    )
    p.add_argument(
        "--select-earliest", action="store_true",
        help=(
            "For positive frames, prefer the earliest frames in each ignition "
            "sequence (smallest, faintest smoke — hardest for classifier)."
        ),
    )
    p.add_argument(
        "--negative-dir", default=None, nargs="+",
        help=(
            "Additional directories of non-fire frames to supplement FIgLib "
            "negatives (e.g. ALERTWildfire archive, HPWREN non-fire periods)."
        ),
    )
    p.add_argument(
        "--failure-dir", default=None,
        help="If set, also extract candidate failure frames here.",
    )
    p.add_argument(
        "--n-failure", type=int, default=5,
        help="Number of failure frame candidates to extract (default: 5).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for frame selection (default: 42).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"ERROR: dataset root not found: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    layout = detect_layout(dataset_root)
    print(f"Dataset layout detected: {layout}")

    print("Scanning fire frames...")
    fire_frames = find_fire_frames(dataset_root)
    print(f"  Found {len(fire_frames)} fire frames")

    extra_neg_dirs = [Path(d) for d in args.negative_dir] if args.negative_dir else None
    print("Scanning non-fire frames...")
    nonfire_frames = find_nonfire_frames(dataset_root, extra_dirs=extra_neg_dirs)
    print(f"  Found {len(nonfire_frames)} non-fire frames")

    if not fire_frames:
        print(
            "ERROR: no fire frames found. Check that the dataset is in a "
            "supported layout (fire/ or non-fire/ subdirectories, or "
            "sequences/ with _fire/_non-fire filename suffixes).",
            file=sys.stderr,
        )
        sys.exit(1)

    if not nonfire_frames:
        print(
            "WARNING: no non-fire frames found in dataset. Pool will contain "
            "only positive frames. Add --negative-dir to supply negatives.",
            file=sys.stderr,
        )

    pool_dir = Path(args.pool_dir)
    failure_dir = Path(args.failure_dir) if args.failure_dir else None

    positives = select_fire_frames(
        frames=fire_frames,
        n=args.n_positive,
        select_earliest=args.select_earliest,
    )
    negatives = select_nonfire_frames(frames=nonfire_frames, n=args.n_negative)

    failure_frames_selected = None
    if failure_dir:
        # Earliest frames not already in the pool
        all_early = select_fire_frames(
            frames=fire_frames,
            n=args.n_positive + args.n_failure,
            select_earliest=True,
        )
        pool_paths = {m["path"] for m in positives}
        failure_frames_selected = [
            m for m in all_early if m["path"] not in pool_paths
        ][:args.n_failure]

    print(f"\nSelected: {len(positives)} positive, {len(negatives)} negative frames")

    manifest = copy_frames_to_pool(
        positives=positives,
        negatives=negatives,
        pool_dir=pool_dir,
        positive_label=POSITIVE_LABEL,
        negative_label=args.negative_label,
        failure_frames=failure_frames_selected,
        failure_dir=failure_dir,
    )

    print(f"\nPool written to {pool_dir}")
    print(f"  {len([m for m in manifest if m['label'] == POSITIVE_LABEL])} {POSITIVE_LABEL}")
    print(f"  {len([m for m in manifest if m['label'] != POSITIVE_LABEL])} {args.negative_label}")
    print(f"  manifest: {pool_dir}/pool_manifest.json")
    print(f"\nNext step:")
    print(f"  python run_dd_session.py \\")
    if failure_dir:
        print(f"      --failure-image {failure_dir}/failure_0000.jpg \\")
    print(f"      --confirmation 'MWIR camera confirmed 380°C surface anomaly' \\")
    print(f"      --raws-conditions '{{\"wind_speed_mph\": 28, \"relative_humidity_pct\": 8, \"temperature_f\": 95}}' \\")
    print(f"      --pool-dir {pool_dir}")


if __name__ == "__main__":
    main()
