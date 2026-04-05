"""
dataset.py — HAM10000 dataset loader for the KF dermatology use case.

Provides:
  load(data_dir)         Load HAM10000 from a directory.
  HAM10000Dataset        Container with per-class split access and sampling.
  DermImage              Single image record (image_id, lesion_id, dx, file_path).
  CONFUSABLE_PAIRS       List of confusable pair dicts for ensemble runs.
  DEFAULT_DATA_DIR       Default path to DermaMNIST_HAM10000 folder.

Split logic:
  Per-dx class: sort unique lesion_ids, take first 80% as train, last 20% as test.
  Test set: one image per test lesion_id (first alphabetically by image_id).
  Train set: all images from train lesion_ids.
"""

from __future__ import annotations
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_DATA_DIR = Path(r"C:\_backup\ml\data\DermaMNIST_HAM10000")

_SEED = 42

_DX_NAMES = {
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevus",
    "bkl":   "Benign Keratosis",
    "bcc":   "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis",
    "vasc":  "Vascular Lesion",
    "df":    "Dermatofibroma",
}

CONFUSABLE_PAIRS = [
    {
        "pair_id": "melanoma_vs_melanocytic_nevus",
        "dx_a": "mel",
        "class_a": "Melanoma",
        "dx_b": "nv",
        "class_b": "Melanocytic Nevus",
    },
    {
        "pair_id": "basal_cell_carcinoma_vs_benign_keratosis",
        "dx_a": "bcc",
        "class_a": "Basal Cell Carcinoma",
        "dx_b": "bkl",
        "class_b": "Benign Keratosis",
    },
    {
        "pair_id": "actinic_keratosis_vs_benign_keratosis",
        "dx_a": "akiec",
        "class_a": "Actinic Keratosis",
        "dx_b": "bkl",
        "class_b": "Benign Keratosis",
    },
]


@dataclass
class DermImage:
    image_id:  str       # e.g. "ISIC_0024306"
    lesion_id: str       # e.g. "HAM_0002761"
    dx:        str       # e.g. "mel", "nv", "bkl", "bcc", "akiec"
    file_path: Path      # absolute path to JPEG
    dx_type:   str       # e.g. "histo", "follow_up", "consensus", "confocal"


@dataclass
class HAM10000Dataset:
    _images: List[DermImage]
    # Nested dict: dx -> split -> list[DermImage]
    _split_index: Dict[str, Dict[str, List[DermImage]]]

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def images_for_class(self, dx: str, split: str = "test") -> List[DermImage]:
        """Return all images for a diagnosis code in the given split."""
        return self._split_index.get(dx, {}).get(split, [])

    def images_for_class_sk_only(self, dx: str, split: str = "test") -> List[DermImage]:
        """Like images_for_class but for bkl, only return histo-confirmed images
        (proxy for seborrheic keratosis; excludes lichenoid keratosis variants)."""
        imgs = self.images_for_class(dx, split=split)
        if dx == "bkl":
            return [img for img in imgs if img.dx_type == "histo"]
        return imgs

    def sample_images(
        self,
        dx: str,
        n: int,
        split: str = "train",
        seed: int = _SEED,
    ) -> List[DermImage]:
        """Return up to n images, one per unique lesion_id, from the given split."""
        imgs = self.images_for_class(dx, split=split)
        # One image per lesion_id (first by image_id)
        seen_lesions: set[str] = set()
        unique: List[DermImage] = []
        for img in sorted(imgs, key=lambda x: x.image_id):
            if img.lesion_id not in seen_lesions:
                seen_lesions.add(img.lesion_id)
                unique.append(img)
        rng = random.Random(seed)
        return rng.sample(unique, min(n, len(unique)))

    def class_name_clean(self, dx: str) -> str:
        """Return the human-readable class name for a dx code."""
        return _DX_NAMES.get(dx, dx)


def _find_image_path(data_dir: Path, image_id: str) -> Optional[Path]:
    """Search part_1 then part_2 for the JPEG file."""
    fname = f"{image_id}.jpg"
    for part in ("HAM10000_images_part_1", "HAM10000_images_part_2"):
        p = data_dir / part / fname
        if p.exists():
            return p
    return None


def load(data_dir: Optional[str | Path] = None) -> HAM10000Dataset:
    """Load HAM10000 from disk and build train/test splits per dx class.

    Args:
        data_dir: Path to the DermaMNIST_HAM10000 folder.
                  Defaults to DEFAULT_DATA_DIR.

    Returns:
        HAM10000Dataset with pre-built split index.
    """
    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    metadata_path = root / "HAM10000_metadata.csv"

    # ------------------------------------------------------------------
    # 1. Read metadata
    # ------------------------------------------------------------------
    rows: List[dict] = []
    with metadata_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # ------------------------------------------------------------------
    # 2. Build per-dx groups of (lesion_id, image_id) pairs
    # ------------------------------------------------------------------
    # dx -> lesion_id -> list[(image_id, dx_type)]
    dx_lesion_images: Dict[str, Dict[str, List[tuple]]] = {}
    for row in rows:
        dx = row["dx"].strip()
        lesion_id = row["lesion_id"].strip()
        image_id = row["image_id"].strip()
        dx_type = row.get("dx_type", "").strip()
        dx_lesion_images.setdefault(dx, {}).setdefault(lesion_id, []).append((image_id, dx_type))

    # ------------------------------------------------------------------
    # 3. Split lesion_ids: sort, 80% train / 20% test
    # ------------------------------------------------------------------
    # dx -> split -> list[DermImage]
    split_index: Dict[str, Dict[str, List[DermImage]]] = {}
    all_images: List[DermImage] = []

    for dx, lesion_map in dx_lesion_images.items():
        sorted_lesion_ids = sorted(lesion_map.keys())
        n = len(sorted_lesion_ids)
        split_point = int(n * 0.8)
        train_lesions = set(sorted_lesion_ids[:split_point])
        test_lesions  = set(sorted_lesion_ids[split_point:])

        train_imgs: List[DermImage] = []
        test_imgs:  List[DermImage] = []

        for lesion_id, image_entries in lesion_map.items():
            sorted_entries = sorted(image_entries, key=lambda e: e[0])
            if lesion_id in train_lesions:
                # All images from train lesion
                for iid, dxt in sorted_entries:
                    fp = _find_image_path(root, iid)
                    if fp is None:
                        fp = root / "HAM10000_images_part_1" / f"{iid}.jpg"
                    img = DermImage(
                        image_id=iid,
                        lesion_id=lesion_id,
                        dx=dx,
                        file_path=fp,
                        dx_type=dxt,
                    )
                    train_imgs.append(img)
                    all_images.append(img)
            elif lesion_id in test_lesions:
                # One image per test lesion (first alphabetically)
                iid, dxt = sorted_entries[0]
                fp = _find_image_path(root, iid)
                if fp is None:
                    fp = root / "HAM10000_images_part_1" / f"{iid}.jpg"
                img = DermImage(
                    image_id=iid,
                    lesion_id=lesion_id,
                    dx=dx,
                    file_path=fp,
                    dx_type=dxt,
                )
                test_imgs.append(img)
                all_images.append(img)

        # Sort each split list by image_id for reproducibility
        train_imgs.sort(key=lambda x: x.image_id)
        test_imgs.sort(key=lambda x: x.image_id)

        split_index[dx] = {"train": train_imgs, "test": test_imgs}

    return HAM10000Dataset(_images=all_images, _split_index=split_index)


if __name__ == "__main__":
    ds = load()
    print("HAM10000 dataset loaded.")
    for dx, name in _DX_NAMES.items():
        train = ds.images_for_class(dx, split="train")
        test  = ds.images_for_class(dx, split="test")
        if train or test:
            print(f"  {dx:6s} ({name:25s})  train={len(train):4d}  test={len(test):4d}")
    print()
    for cp in CONFUSABLE_PAIRS:
        sample = ds.sample_images(cp["dx_a"], n=3, split="train")
        print(f"  Sample {cp['class_a']} train: {[s.image_id for s in sample]}")
