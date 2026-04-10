"""
dataset.py — RSCD (Road Surface Classification Dataset) loader for the KF road surface use case.

Provides:
  load(data_dir)         Load RSCD from a directory.
  RSCDDataset            Container with per-class split access and sampling.
  RoadImage              Single image record (image_id, friction, material, unevenness, file_path).
  CONFUSABLE_PAIRS       List of confusable pair dicts ranked by safety priority.
  DEFAULT_DATA_DIR       Default path to RSCD folder.

Dataset source:
  RSCD — Road Surface Classification Dataset (Tsinghua University)
  ~1M images (240×360px), 27 classes: friction × material × roughness
  Download: https://thu-rsxd.com/dxhdiefb/  (~14 GB, CC BY-NC)
  Kaggle:   https://www.kaggle.com/datasets/cristvollerei/rscd-dataset-1million
  GitHub:   https://github.com/ztsrxh/RSCD-Road_Surface_Classification_Dataset

RSCD directory structure expected:
  <data_dir>/
    train.csv   (columns: filename, friction, material, unevenness)
    val.csv
    test.csv
    images/
      00000001.jpg
      00000002.jpg
      ...

Split logic:
  Uses RSCD's built-in train/val/test splits from the CSV files.
  If only a single CSV is present, falls back to 80/20 per-class split by filename order.
"""

from __future__ import annotations
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_DATA_DIR = Path(r"C:\_backup\ml\data\RSCD")

_SEED = 42

# ---------------------------------------------------------------------------
# Class name mappings
# ---------------------------------------------------------------------------

FRICTION_NAMES = {
    "dry":          "Dry",
    "wet":          "Wet",
    "water":        "Standing Water",
    "fresh_snow":   "Fresh Snow",
    "melted_snow":  "Melted Snow",
    "ice":          "Ice",
}

MATERIAL_NAMES = {
    "asphalt":  "Asphalt",
    "concrete": "Concrete",
    "dirt":     "Dirt/Mud",
    "gravel":   "Gravel",
}

ROUGHNESS_NAMES = {
    "smooth": "Smooth",
    "slight": "Slight Unevenness",
    "severe": "Severe Unevenness",
}

# ---------------------------------------------------------------------------
# Confusable pairs — ranked by safety priority
# ---------------------------------------------------------------------------
# Each pair focuses on the FRICTION dimension since that is safety-critical.
# class_a / class_b are human-readable labels used in prompts.
# friction_a / friction_b are the raw CSV labels for filtering.
# material_filter: if set, restrict both classes to this material for cleaner pairs.

CONFUSABLE_PAIRS = [
    # Priority 1 — Safety-critical friction confusions
    {
        "pair_id":       "wet_vs_ice",
        "friction_a":    "wet",
        "class_a":       "Wet Road",
        "friction_b":    "ice",
        "class_b":       "Icy Road",
        "material_filter": "asphalt",   # Restrict to asphalt for cleaner comparison
        "description":   "Both appear as dark reflective surfaces; ice obscures surface texture completely",
    },
    {
        "pair_id":       "dry_vs_wet",
        "friction_a":    "dry",
        "class_a":       "Dry Road",
        "friction_b":    "wet",
        "class_b":       "Wet Road",
        "material_filter": "asphalt",
        "description":   "Early-stage wetting is ambiguous; surface begins to darken unevenly",
    },
    # Priority 2 — Snow state confusions
    {
        "pair_id":       "fresh_snow_vs_melted_snow",
        "friction_a":    "fresh_snow",
        "class_a":       "Fresh Snow",
        "friction_b":    "melted_snow",
        "class_b":       "Melted Snow / Slush",
        "material_filter": None,
        "description":   "Partial melt creates mixed appearance; slush varies widely",
    },
    # Priority 3 — Water depth confusion
    {
        "pair_id":       "wet_vs_standing_water",
        "friction_a":    "wet",
        "class_a":       "Wet Road",
        "friction_b":    "water",
        "class_b":       "Standing Water",
        "material_filter": "asphalt",
        "description":   "Thin film vs pooled water; aquaplaning risk differs substantially",
    },
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RoadImage:
    image_id:    str         # e.g. "00000001"
    friction:    str         # e.g. "wet", "ice", "dry"
    material:    str         # e.g. "asphalt", "concrete", "" (unlabeled for snow/ice)
    unevenness:  str         # e.g. "smooth", "slight", "severe", "" (unlabeled)
    file_path:   Path        # absolute path to JPEG
    split:       str         # "train", "val", "test"

    @property
    def friction_label(self) -> str:
        return FRICTION_NAMES.get(self.friction, self.friction)

    @property
    def material_label(self) -> str:
        return MATERIAL_NAMES.get(self.material, self.material) if self.material else ""


@dataclass
class RSCDDataset:
    _images: List[RoadImage]
    # Nested dict: friction -> split -> list[RoadImage]
    _split_index: Dict[str, Dict[str, List[RoadImage]]]

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def images_for_friction(self, friction: str, split: str = "test") -> List[RoadImage]:
        """Return all images for a friction class in the given split."""
        return self._split_index.get(friction, {}).get(split, [])

    def images_for_pair(
        self,
        friction: str,
        split: str = "test",
        material_filter: Optional[str] = None,
    ) -> List[RoadImage]:
        """Return images for a friction class, optionally filtered by material."""
        imgs = self.images_for_friction(friction, split=split)
        if material_filter:
            imgs = [img for img in imgs if img.material == material_filter]
        return imgs

    def sample_images(
        self,
        friction: str,
        n: int,
        split: str = "train",
        seed: int = _SEED,
        material_filter: Optional[str] = None,
    ) -> List[RoadImage]:
        """Return up to n images from the given split, deterministically sampled."""
        imgs = self.images_for_pair(friction, split=split, material_filter=material_filter)
        rng = random.Random(seed)
        return rng.sample(imgs, min(n, len(imgs)))

    def class_stats(self) -> Dict[str, Dict[str, int]]:
        """Return {friction: {split: count}} for all classes."""
        stats: Dict[str, Dict[str, int]] = {}
        for friction, splits in self._split_index.items():
            stats[friction] = {sp: len(imgs) for sp, imgs in splits.items()}
        return stats


# ---------------------------------------------------------------------------
# CSV reader helpers
# ---------------------------------------------------------------------------

def _read_csv(csv_path: Path, images_dir: Path, split: str) -> List[RoadImage]:
    """Parse one RSCD CSV file and return a list of RoadImage records."""
    images: List[RoadImage] = []
    if not csv_path.exists():
        return images

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Normalize column names (RSCD CSVs may vary: 'filename' or 'image' etc.)
        for row in reader:
            fname = (row.get("filename") or row.get("image") or row.get("file") or "").strip()
            friction = (row.get("friction") or row.get("label_friction") or "").strip().lower()
            material = (row.get("material") or row.get("label_material") or "").strip().lower()
            unevenness = (row.get("unevenness") or row.get("roughness")
                         or row.get("label_unevenness") or "").strip().lower()

            if not fname or not friction:
                continue

            image_id = Path(fname).stem
            file_path = images_dir / fname
            # Try without subdir too
            if not file_path.exists():
                file_path = images_dir / Path(fname).name

            images.append(RoadImage(
                image_id=image_id,
                friction=friction,
                material=material,
                unevenness=unevenness,
                file_path=file_path,
                split=split,
            ))

    return images


def _build_split_index(
    all_images: List[RoadImage],
) -> Dict[str, Dict[str, List[RoadImage]]]:
    """Build {friction: {split: [images]}} index."""
    index: Dict[str, Dict[str, List[RoadImage]]] = {}
    for img in all_images:
        index.setdefault(img.friction, {}).setdefault(img.split, []).append(img)
    # Sort each list by image_id for reproducibility
    for friction in index:
        for split in index[friction]:
            index[friction][split].sort(key=lambda x: x.image_id)
    return index


def _fallback_split(images: List[RoadImage]) -> List[RoadImage]:
    """Assign train/test splits when no pre-split CSVs are found.

    Groups by friction, sorts by image_id, takes first 80% as train and last 20% as test.
    """
    by_friction: Dict[str, List[RoadImage]] = {}
    for img in images:
        by_friction.setdefault(img.friction, []).append(img)

    result: List[RoadImage] = []
    for friction, imgs in by_friction.items():
        sorted_imgs = sorted(imgs, key=lambda x: x.image_id)
        n = len(sorted_imgs)
        split_point = int(n * 0.8)
        for i, img in enumerate(sorted_imgs):
            split = "train" if i < split_point else "test"
            result.append(RoadImage(
                image_id=img.image_id,
                friction=img.friction,
                material=img.material,
                unevenness=img.unevenness,
                file_path=img.file_path,
                split=split,
            ))
    return result


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load(data_dir: Optional[str | Path] = None) -> RSCDDataset:
    """Load RSCD from disk and build per-friction split index.

    Supports two directory layouts:
      Layout A (with pre-split CSVs):
        <data_dir>/train.csv, val.csv, test.csv
        <data_dir>/images/  (or images directly in data_dir)

      Layout B (single CSV):
        <data_dir>/labels.csv  (or metadata.csv, rscd.csv)
        <data_dir>/images/

    Falls back to 80/20 per-friction split if no CSV splits exist.

    Args:
        data_dir: Path to the RSCD root folder. Defaults to DEFAULT_DATA_DIR.

    Returns:
        RSCDDataset with pre-built split index.

    Raises:
        FileNotFoundError: if data_dir does not exist.
        ValueError: if no images could be loaded (empty CSVs / wrong path).
    """
    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    if not root.exists():
        raise FileNotFoundError(
            f"RSCD data directory not found: {root}\n"
            f"Download from https://thu-rsxd.com/dxhdiefb/ and extract here."
        )

    images_dir = root / "images" if (root / "images").exists() else root

    all_images: List[RoadImage] = []

    # ------------------------------------------------------------------
    # Try Layout A: separate train/val/test CSVs
    # ------------------------------------------------------------------
    train_csv = root / "train.csv"
    val_csv   = root / "val.csv"
    test_csv  = root / "test.csv"

    if train_csv.exists():
        all_images.extend(_read_csv(train_csv, images_dir, "train"))
        all_images.extend(_read_csv(val_csv,   images_dir, "val"))
        all_images.extend(_read_csv(test_csv,  images_dir, "test"))
    else:
        # ------------------------------------------------------------------
        # Layout B: single CSV with all images, then split manually
        # ------------------------------------------------------------------
        for candidate in ("labels.csv", "metadata.csv", "rscd.csv", "dataset.csv",
                          "annotations.csv"):
            csv_path = root / candidate
            if csv_path.exists():
                raw = _read_csv(csv_path, images_dir, "all")
                all_images = _fallback_split(raw)
                break

    if not all_images:
        raise ValueError(
            f"No images loaded from {root}.\n"
            f"Expected CSV files (train.csv/val.csv/test.csv or labels.csv) "
            f"with columns: filename, friction, material, unevenness.\n"
            f"Check dataset layout against README in road-surface/README.md."
        )

    split_index = _build_split_index(all_images)
    return RSCDDataset(_images=all_images, _split_index=split_index)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ds = load()
    print("RSCD dataset loaded.")
    stats = ds.class_stats()
    print(f"\n{'Friction':<16} {'Train':>8} {'Val':>8} {'Test':>8}")
    print("-" * 44)
    for friction, splits in sorted(stats.items()):
        label = FRICTION_NAMES.get(friction, friction)
        train = splits.get("train", 0)
        val   = splits.get("val", 0)
        test  = splits.get("test", 0)
        print(f"  {label:<14} {train:>8} {val:>8} {test:>8}")

    print(f"\nTotal: {len(ds._images)} images")

    print(f"\nConfusable pairs:")
    for cp in CONFUSABLE_PAIRS:
        a_imgs = ds.images_for_pair(cp["friction_a"], split="test",
                                    material_filter=cp.get("material_filter"))
        b_imgs = ds.images_for_pair(cp["friction_b"], split="test",
                                    material_filter=cp.get("material_filter"))
        print(f"  {cp['pair_id']:40s}  {cp['class_a']}: {len(a_imgs):4d}  "
              f"{cp['class_b']}: {len(b_imgs):4d} (test)")
