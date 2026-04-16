"""
pool_builder.py — SeaDronesSee labeled pool construction.

Processes the SeaDronesSee COCO-format dataset to extract:
  - Person frames (positive class: "person_in_water")
  - No-person frames (negative classes: "whitecap", "floating_debris", etc.)

Outputs a pool directory ready for run_dd_session.py and a pool_manifest.json.

SeaDronesSee dataset: https://github.com/Ben93kie/SeaDronesSee
Expected structure:
    <dataset_root>/
      images/
        <sequence_dir>/
          frame_000001.jpg
          ...
      annotations/
        instances_train.json
        instances_val.json
        instances_test.json

Usage:
    # Build pool from validation split (recommended — no training data needed)
    python pool_builder.py \\
        --dataset-root data/seadronessee/ \\
        --split val \\
        --pool-dir data/pool/ \\
        --n-positive 10 \\
        --n-negative 20 \\
        --select-hardest

    # Also extract candidate failure frames (smallest visible persons)
    python pool_builder.py \\
        --dataset-root data/seadronessee/ \\
        --split val \\
        --pool-dir data/pool/ \\
        --failure-dir data/failure_frames/ \\
        --n-failure 5
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# SeaDronesSee category mapping
# ---------------------------------------------------------------------------
# SeaDronesSee categories (may vary by release):
#   1: swimmer  2: boat  3: jetski  4: life_saving_appliance  5: buoy
# We treat category 1 ("swimmer") as person_in_water.

PERSON_CATEGORY_NAMES = {"swimmer", "person", "human"}
NEGATIVE_LABEL = "whitecap"        # default label for no-person frames
POSITIVE_LABEL = "person_in_water"

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------------------
# COCO annotation loader
# ---------------------------------------------------------------------------

def load_coco(annotation_path: Path) -> dict:
    return json.loads(annotation_path.read_text())


def get_person_category_ids(coco: dict) -> set[int]:
    """Return category IDs corresponding to person/swimmer."""
    return {
        cat["id"]
        for cat in coco.get("categories", [])
        if cat["name"].lower() in PERSON_CATEGORY_NAMES
    }


def build_image_index(coco: dict) -> dict[int, dict]:
    """image_id -> image_info"""
    return {img["id"]: img for img in coco.get("images", [])}


def build_annotation_index(coco: dict, person_cat_ids: set[int]) -> dict[int, list[dict]]:
    """image_id -> list of person annotations"""
    index: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        if ann.get("category_id") in person_cat_ids:
            index.setdefault(ann["image_id"], []).append(ann)
    return index


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------

def bbox_area(ann: dict) -> float:
    """Return bounding box area from COCO annotation."""
    bbox = ann.get("bbox", [0, 0, 0, 0])  # [x, y, w, h]
    return bbox[2] * bbox[3]


def build_file_index(dataset_root: Path) -> dict[str, Path]:
    """Build a filename → full path index by scanning the dataset images directory.

    Handles datasets where images are in subdirectories (e.g. images/val/, images/train/).
    """
    index: dict[str, Path] = {}
    images_dir = dataset_root / "images"
    search_root = images_dir if images_dir.exists() else dataset_root
    for p in search_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
            index[p.name] = p
    return index


def _resolve_image_path(img_info: dict, dataset_root: Path,
                        file_index: dict[str, Path]) -> Path | None:
    """Resolve the full path for an image, trying multiple locations."""
    fname = img_info["file_name"]
    # 1. Index lookup by basename (handles images/val/3464.jpg etc.)
    basename = Path(fname).name
    if basename in file_index:
        return file_index[basename]
    # 2. Direct join
    for candidate in [
        dataset_root / "images" / fname,
        dataset_root / fname,
    ]:
        if candidate.exists():
            return candidate
    return None


def select_positive_frames(
    image_index: dict[int, dict],
    person_ann_index: dict[int, list[dict]],
    dataset_root: Path,
    n: int,
    hardest: bool = False,
    file_index: dict[str, Path] | None = None,
) -> list[dict]:
    """Select n person frames from the dataset.

    If hardest=True, prefer frames where the person occupies the smallest
    fraction of the image (hardest for a classifier to detect).
    """
    if file_index is None:
        file_index = build_file_index(dataset_root)
    candidates = []
    for image_id, anns in person_ann_index.items():
        img_info = image_index.get(image_id)
        if img_info is None:
            continue
        img_path = _resolve_image_path(img_info, dataset_root, file_index)
        if img_path is None:
            continue

        min_bbox = min(bbox_area(a) for a in anns)
        img_area = img_info.get("width", 1920) * img_info.get("height", 1080)
        person_fraction = min_bbox / img_area if img_area > 0 else 0.0

        candidates.append({
            "image_id": image_id,
            "image_path": str(img_path),
            "file_name": img_info["file_name"],
            "n_persons": len(anns),
            "min_bbox_area": min_bbox,
            "person_fraction": person_fraction,
        })

    if hardest:
        candidates.sort(key=lambda x: x["person_fraction"])
    else:
        random.shuffle(candidates)

    return candidates[:n]


def select_negative_frames(
    image_index: dict[int, dict],
    person_ann_index: dict[int, list[dict]],
    dataset_root: Path,
    n: int,
    file_index: dict[str, Path] | None = None,
) -> list[dict]:
    """Select n frames that contain NO person annotations."""
    if file_index is None:
        file_index = build_file_index(dataset_root)
    person_image_ids = set(person_ann_index.keys())
    candidates = []

    for image_id, img_info in image_index.items():
        if image_id in person_image_ids:
            continue
        img_path = _resolve_image_path(img_info, dataset_root, file_index)
        if img_path is None:
            continue
        candidates.append({
            "image_id": image_id,
            "image_path": str(img_path),
            "file_name": img_info["file_name"],
        })

    random.shuffle(candidates)
    return candidates[:n]


# ---------------------------------------------------------------------------
# Pool output
# ---------------------------------------------------------------------------

def copy_frames_to_pool(
    positives: list[dict],
    negatives: list[dict],
    pool_dir: Path,
    negative_label: str,
    failure_frames: list[dict] | None = None,
    failure_dir: Path | None = None,
) -> list[dict]:
    """Copy frames into pool directory structure and write manifest."""
    pool_dir.mkdir(parents=True, exist_ok=True)
    (pool_dir / POSITIVE_LABEL).mkdir(exist_ok=True)
    (pool_dir / negative_label).mkdir(exist_ok=True)

    manifest = []

    for i, frame in enumerate(positives):
        src = Path(frame["image_path"])
        dest_name = f"pos_{i:04d}{src.suffix}"
        dest = pool_dir / POSITIVE_LABEL / dest_name
        shutil.copy2(src, dest)
        manifest.append({
            "path": f"{POSITIVE_LABEL}/{dest_name}",
            "label": POSITIVE_LABEL,
            "source_image_id": frame["image_id"],
            "person_fraction": frame.get("person_fraction"),
        })

    for i, frame in enumerate(negatives):
        src = Path(frame["image_path"])
        dest_name = f"neg_{i:04d}{src.suffix}"
        dest = pool_dir / negative_label / dest_name
        shutil.copy2(src, dest)
        manifest.append({
            "path": f"{negative_label}/{dest_name}",
            "label": negative_label,
            "source_image_id": frame["image_id"],
        })

    # Write manifest
    (pool_dir / "pool_manifest.json").write_text(json.dumps(manifest, indent=2))

    # Copy failure frames if requested
    if failure_frames and failure_dir:
        failure_dir.mkdir(parents=True, exist_ok=True)
        failure_manifest = []
        for i, frame in enumerate(failure_frames):
            src = Path(frame["image_path"])
            dest_name = f"failure_{i:04d}{src.suffix}"
            dest = failure_dir / dest_name
            shutil.copy2(src, dest)
            failure_manifest.append({
                "path": dest_name,
                "label": POSITIVE_LABEL,
                "source_image_id": frame["image_id"],
                "person_fraction": frame.get("person_fraction"),
                "note": "Small person — likely to be confidently missed by baseline classifier",
            })
        (failure_dir / "failure_manifest.json").write_text(json.dumps(failure_manifest, indent=2))
        print(f"Failure frames: {len(failure_frames)} written to {failure_dir}")

    return manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a labeled pool from SeaDronesSee for maritime DD sessions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset-root", required=True,
        help="Root directory of the SeaDronesSee dataset.",
    )
    p.add_argument(
        "--split", default="val", choices=["train", "val", "test"],
        help="Dataset split to use (default: val).",
    )
    p.add_argument(
        "--annotation-file", default=None,
        help="Override annotation file path (default: <root>/annotations/instances_<split>.json).",
    )
    p.add_argument(
        "--pool-dir", required=True,
        help="Output directory for the labeled pool.",
    )
    p.add_argument(
        "--n-positive", type=int, default=10,
        help="Number of person-in-water frames to include (default: 10).",
    )
    p.add_argument(
        "--n-negative", type=int, default=20,
        help="Number of no-person frames to include (default: 20).",
    )
    p.add_argument(
        "--negative-label", default=NEGATIVE_LABEL,
        help=f"Label for no-person frames (default: {NEGATIVE_LABEL}).",
    )
    p.add_argument(
        "--select-hardest", action="store_true",
        help="For positive frames, prefer those with the smallest (hardest to detect) persons.",
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

    # Find annotation file
    if args.annotation_file:
        ann_path = Path(args.annotation_file)
    else:
        ann_path = dataset_root / "annotations" / f"instances_{args.split}.json"
        if not ann_path.exists():
            # Try alternate naming conventions
            for alt in [f"{args.split}.json", f"annotations_{args.split}.json"]:
                alt_path = dataset_root / "annotations" / alt
                if alt_path.exists():
                    ann_path = alt_path
                    break

    if not ann_path.exists():
        print(f"ERROR: annotation file not found: {ann_path}", file=sys.stderr)
        print("Use --annotation-file to specify the path explicitly.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading annotations from {ann_path}...")
    coco = load_coco(ann_path)
    image_index = build_image_index(coco)
    person_cat_ids = get_person_category_ids(coco)

    if not person_cat_ids:
        print(f"WARNING: no person category found. Available categories:")
        for cat in coco.get("categories", []):
            print(f"  {cat['id']}: {cat['name']}")
        print("Use PERSON_CATEGORY_NAMES in pool_builder.py to add the correct name.")

    person_ann_index = build_annotation_index(coco, person_cat_ids)
    n_person_images = len(person_ann_index)
    n_total_images = len(image_index)
    print(f"Images: {n_total_images} total, {n_person_images} with person annotations")

    # Build file index once — shared across all selection calls
    print("Building file index...")
    file_index = build_file_index(dataset_root)
    print(f"File index: {len(file_index)} images found")

    # Select frames
    pool_dir = Path(args.pool_dir)
    failure_dir = Path(args.failure_dir) if args.failure_dir else None

    positives = select_positive_frames(
        image_index=image_index,
        person_ann_index=person_ann_index,
        dataset_root=dataset_root,
        n=args.n_positive,
        hardest=args.select_hardest,
        file_index=file_index,
    )
    negatives = select_negative_frames(
        image_index=image_index,
        person_ann_index=person_ann_index,
        dataset_root=dataset_root,
        n=args.n_negative,
        file_index=file_index,
    )
    failure_frames = None
    if failure_dir:
        # Hardest positives (smallest person, pool-excluded)
        all_hard = select_positive_frames(
            image_index=image_index,
            person_ann_index=person_ann_index,
            dataset_root=dataset_root,
            n=args.n_positive + args.n_failure,
            hardest=True,
            file_index=file_index,
        )
        # Take the hardest ones not already in positives
        pool_ids = {f["image_id"] for f in positives}
        failure_frames = [f for f in all_hard if f["image_id"] not in pool_ids][:args.n_failure]

    print(f"\nSelected: {len(positives)} positive, {len(negatives)} negative frames")
    if positives and args.select_hardest:
        fracs = [f["person_fraction"] for f in positives]
        print(f"Person fraction: min={min(fracs):.4f} max={max(fracs):.4f} "
              f"mean={sum(fracs)/len(fracs):.4f}")

    manifest = copy_frames_to_pool(
        positives=positives,
        negatives=negatives,
        pool_dir=pool_dir,
        negative_label=args.negative_label,
        failure_frames=failure_frames,
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
    print(f"      --confirmation 'Thermal camera confirmed heat signature' \\")
    print(f"      --pool-dir {pool_dir}")


if __name__ == "__main__":
    main()
