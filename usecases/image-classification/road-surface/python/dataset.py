"""
dataset.py — RSCD (Road Surface Classification Dataset) loader for the KF road surface use case.

Provides:
  load(data_dir)         Load RSCD from a directory (or zip file).
  RSCDDataset            Container with per-class split access and sampling.
  RoadImage              Single image record.
  CONFUSABLE_PAIRS       List of confusable pair dicts ranked by safety priority.
  DEFAULT_DATA_DIR       Default path to RSCD zip or extracted folder.

Dataset source:
  RSCD — Road Surface Classification Dataset (Tsinghua University)
  ~1M images (240×360px), labels encoded in filenames
  Kaggle:   https://www.kaggle.com/datasets/cristvollerei/rscd-dataset-1million
  GitHub:   https://github.com/ztsrxh/RSCD-Road_Surface_Classification_Dataset

Actual layout (confirmed from zip inspection):
  rscd-dataset-1million.zip
    RSCD dataset-1million/
      train/               ~557,600 images
      test_50k/            ~29,900 images
      vali_20k/            ~12,500 images

  Filenames:  <timestamp>-<friction>-<material>-<roughness>.jpg
  Example:    2022012523413511-wet-asphalt-smooth.jpg

  friction classes (3):  dry, wet, water
  material classes (2):  asphalt, concrete
  roughness classes (3): smooth, slight, severe

Split mapping:
  "train"  ← train/
  "test"   ← test_50k/
  "val"    ← vali_20k/

Note on class scope:
  This release contains only dry/wet/water friction classes.
  No ice, snow, or slush classes exist in this version.
  The primary DD confusable pair is dry vs wet (subtle, hard).
  The secondary pair is wet vs water (moderate, still valuable).
"""

from __future__ import annotations

import re
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path(r"C:\_backup\ml\data")
DEFAULT_ZIP_NAME = "rscd-dataset-1million.zip"
ZIP_ROOT         = "RSCD dataset-1million"   # top-level folder inside the zip

_SEED = 42

# Regex to parse RSCD filenames
# Matches: <timestamp>-<friction>-<material>-<roughness>.jpg
# friction and material can contain underscores (e.g. fresh_snow, dirt_mud)
_FNAME_RE = re.compile(
    r"^(\d+)-"
    r"(dry|wet|water|fresh_snow|melted_snow|ice)"
    r"-(asphalt|concrete|dirt|gravel|dirt_mud)"
    r"-(smooth|slight|severe)"
    r"\.jpg$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Human-readable label maps
# ---------------------------------------------------------------------------

FRICTION_NAMES = {
    "dry":          "Dry",
    "wet":          "Wet",
    "water":        "Standing Water",
    "fresh_snow":   "Fresh Snow",
    "melted_snow":  "Melted Snow / Slush",
    "ice":          "Ice",
}

MATERIAL_NAMES = {
    "asphalt":   "Asphalt",
    "concrete":  "Concrete",
    "dirt":      "Dirt/Mud",
    "dirt_mud":  "Dirt/Mud",
    "gravel":    "Gravel",
}

ROUGHNESS_NAMES = {
    "smooth": "Smooth",
    "slight": "Slight Unevenness",
    "severe": "Severe Unevenness",
}

# Folder name → canonical split name
_SPLIT_DIRS = {
    "train":    "train",
    "test_50k": "test",
    "vali_20k": "val",
}

# ---------------------------------------------------------------------------
# Confusable pairs — ranked by DD interest
# ---------------------------------------------------------------------------
# Only pairs present in this RSCD release (dry/wet/water).
# Ice and snow pairs removed — not available in this dataset version.

CONFUSABLE_PAIRS = [
    {
        "pair_id":       "dry_vs_wet",
        "friction_a":    "dry",
        "class_a":       "Dry Road",
        "friction_b":    "wet",
        "class_b":       "Wet Road",
        "material_filter": "asphalt",
        "description": (
            "Primary DD pair. Subtle tonal darkening and reduced inter-aggregate "
            "contrast are the main visual cues. Early-stage wetting is genuinely "
            "ambiguous. Strong vocabulary gap: Qwen describes both as 'grey road'."
        ),
        "dd_priority": "high",
    },
    {
        "pair_id":       "wet_vs_water",
        "friction_a":    "wet",
        "class_a":       "Wet Road",
        "friction_b":    "water",
        "class_b":       "Standing Water",
        "material_filter": "asphalt",
        "description": (
            "Secondary pair. Standing water produces dramatic specular reflections "
            "from surrounding light sources. More visually obvious than dry vs wet, "
            "but the boundary between thin water film and standing water remains "
            "genuinely ambiguous. Aquaplaning risk differs substantially."
        ),
        "dd_priority": "medium",
    },
    {
        "pair_id":       "dry_vs_wet_concrete",
        "friction_a":    "dry",
        "class_a":       "Dry Concrete",
        "friction_b":    "wet",
        "class_b":       "Wet Concrete",
        "material_filter": "concrete",
        "description": (
            "Concrete variant of the primary pair. Lighter base colour makes "
            "moisture detection harder — wet concrete contrast is lower than wet "
            "asphalt. Useful for testing rule transfer across materials."
        ),
        "dd_priority": "medium",
    },
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RoadImage:
    image_id:   str     # timestamp portion of filename, e.g. "2022012523413511"
    friction:   str     # "dry" | "wet" | "water"
    material:   str     # "asphalt" | "concrete"
    roughness:  str     # "smooth" | "slight" | "severe"
    split:      str     # "train" | "val" | "test"
    # Exactly one of file_path / zip_path is set
    file_path:  Optional[Path] = None   # set when loaded from extracted folder
    zip_path:   Optional[str]  = None   # set when loaded from zip (internal path)
    _zip_ref:   Optional[zipfile.ZipFile] = None  # shared zip handle (not serialised)

    @property
    def friction_label(self) -> str:
        return FRICTION_NAMES.get(self.friction, self.friction)

    @property
    def material_label(self) -> str:
        return MATERIAL_NAMES.get(self.material, self.material)

    @property
    def roughness_label(self) -> str:
        return ROUGHNESS_NAMES.get(self.roughness, self.roughness)

    def read_bytes(self) -> bytes:
        """Return raw JPEG bytes regardless of source (file or zip)."""
        if self.file_path is not None:
            return self.file_path.read_bytes()
        if self._zip_ref is not None and self.zip_path is not None:
            return self._zip_ref.read(self.zip_path)
        raise RuntimeError(
            f"RoadImage {self.image_id} has no file_path and no zip reference."
        )

    def __repr__(self) -> str:
        src = str(self.file_path) if self.file_path else self.zip_path
        return (
            f"RoadImage(id={self.image_id}, "
            f"{self.friction}/{self.material}/{self.roughness}, "
            f"split={self.split}, src={src})"
        )


@dataclass
class RSCDDataset:
    """Container providing split-aware access to RSCD images."""

    _images: List[RoadImage]
    # {friction: {split: [RoadImage]}}
    _index: Dict[str, Dict[str, List[RoadImage]]]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def images_for_friction(
        self,
        friction: str,
        split: str = "test",
    ) -> List[RoadImage]:
        """All images for a friction class in the given split."""
        return self._index.get(friction, {}).get(split, [])

    def images_for_pair(
        self,
        friction: str,
        split: str = "test",
        material_filter: Optional[str] = None,
        roughness_filter: Optional[str] = None,
    ) -> List[RoadImage]:
        """Images for a friction class with optional material/roughness filters."""
        imgs = self.images_for_friction(friction, split=split)
        if material_filter:
            imgs = [i for i in imgs if i.material == material_filter]
        if roughness_filter:
            imgs = [i for i in imgs if i.roughness == roughness_filter]
        return imgs

    def sample_images(
        self,
        friction: str,
        n: int,
        split: str = "test",
        seed: int = _SEED,
        material_filter: Optional[str] = None,
        roughness_filter: Optional[str] = None,
    ) -> List[RoadImage]:
        """Return up to n images, deterministically sampled."""
        imgs = self.images_for_pair(
            friction,
            split=split,
            material_filter=material_filter,
            roughness_filter=roughness_filter,
        )
        rng = random.Random(seed)
        return rng.sample(imgs, min(n, len(imgs)))

    def sample_pair(
        self,
        pair_id: str,
        n_per_class: int = 20,
        split: str = "test",
        seed: int = _SEED,
    ) -> tuple[List[RoadImage], List[RoadImage]]:
        """Return (class_a_images, class_b_images) for a named confusable pair."""
        cp = next((p for p in CONFUSABLE_PAIRS if p["pair_id"] == pair_id), None)
        if cp is None:
            raise ValueError(
                f"Unknown pair_id '{pair_id}'. "
                f"Available: {[p['pair_id'] for p in CONFUSABLE_PAIRS]}"
            )
        mf = cp.get("material_filter")
        a = self.sample_images(cp["friction_a"], n_per_class, split=split,
                               seed=seed, material_filter=mf)
        b = self.sample_images(cp["friction_b"], n_per_class, split=split,
                               seed=seed + 1, material_filter=mf)
        return a, b

    def class_stats(self) -> Dict[str, Dict[str, int]]:
        """{friction: {split: count}} for all classes."""
        return {
            f: {sp: len(imgs) for sp, imgs in splits.items()}
            for f, splits in self._index.items()
        }

    def __len__(self) -> int:
        return len(self._images)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_filename(fname: str) -> Optional[tuple[str, str, str, str]]:
    """Parse RSCD filename → (image_id, friction, material, roughness) or None."""
    m = _FNAME_RE.match(Path(fname).name)
    if not m:
        return None
    return m.group(1), m.group(2).lower(), m.group(3).lower(), m.group(4).lower()


def _build_index(images: List[RoadImage]) -> Dict[str, Dict[str, List[RoadImage]]]:
    index: Dict[str, Dict[str, List[RoadImage]]] = {}
    for img in images:
        index.setdefault(img.friction, {}).setdefault(img.split, []).append(img)
    # Sort for reproducibility
    for f in index:
        for sp in index[f]:
            index[f][sp].sort(key=lambda x: x.image_id)
    return index


def _iter_extracted(root: Path) -> Iterator[RoadImage]:
    """Yield RoadImage records from an extracted RSCD folder."""
    for folder_name, split in _SPLIT_DIRS.items():
        split_dir = root / folder_name
        if not split_dir.exists():
            continue
        for fpath in sorted(split_dir.glob("*.jpg")):
            parsed = _parse_filename(fpath.name)
            if parsed is None:
                continue
            image_id, friction, material, roughness = parsed
            yield RoadImage(
                image_id=image_id,
                friction=friction,
                material=material,
                roughness=roughness,
                split=split,
                file_path=fpath,
            )


def _iter_zip(zf: zipfile.ZipFile) -> Iterator[RoadImage]:
    """Yield RoadImage records from RSCD zip without extracting."""
    for entry in zf.namelist():
        parts = entry.split("/")
        # Expected: ZIP_ROOT / <split_folder> / <filename>.jpg
        if len(parts) != 3:
            continue
        _, folder_name, fname = parts
        if not fname.lower().endswith(".jpg"):
            continue
        split = _SPLIT_DIRS.get(folder_name)
        if split is None:
            continue
        parsed = _parse_filename(fname)
        if parsed is None:
            continue
        image_id, friction, material, roughness = parsed
        yield RoadImage(
            image_id=image_id,
            friction=friction,
            material=material,
            roughness=roughness,
            split=split,
            zip_path=entry,
            _zip_ref=zf,
        )


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load(
    data_dir: Optional[str | Path] = None,
    *,
    from_zip: Optional[str | Path] = None,
    max_per_class: Optional[int] = None,
    splits: Optional[List[str]] = None,
) -> RSCDDataset:
    """Load RSCD and return an RSCDDataset.

    Two modes:
      1. Extracted folder  (data_dir points to folder containing train/, test_50k/, vali_20k/)
      2. Direct from zip   (from_zip points to rscd-dataset-1million.zip)

    Args:
        data_dir:      Path to extracted RSCD root folder. Defaults to DEFAULT_DATA_DIR.
                       Loader checks for train/ subfolder; if not found and a zip is
                       present in data_dir, automatically switches to zip mode.
        from_zip:      Explicit path to the RSCD zip file. Overrides data_dir.
        max_per_class: Cap images per (friction, split) combination — useful for
                       quick development runs (e.g. max_per_class=500).
        splits:        Restrict to these splits, e.g. ["test"]. Default: all splits.

    Returns:
        RSCDDataset with built index.

    Raises:
        FileNotFoundError: if neither a folder nor a zip can be found.
        ValueError: if no images could be loaded.
    """
    allowed_splits = set(splits) if splits else {"train", "val", "test"}

    # ------------------------------------------------------------------
    # Resolve source
    # ------------------------------------------------------------------
    if from_zip is not None:
        zip_path = Path(from_zip)
        if not zip_path.exists():
            raise FileNotFoundError(f"RSCD zip not found: {zip_path}")
        zf = zipfile.ZipFile(zip_path)
        all_images = list(_iter_zip(zf))

    else:
        root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR

        # Check if this is the folder containing train/ subdirs
        if (root / "train").exists() or (root / "test_50k").exists():
            all_images = list(_iter_extracted(root))

        # Check if data_dir itself is the zip
        elif root.suffix == ".zip" and root.exists():
            zf = zipfile.ZipFile(root)
            all_images = list(_iter_zip(zf))

        # Auto-detect zip in the data_dir folder
        else:
            zip_candidate = root / DEFAULT_ZIP_NAME
            if zip_candidate.exists():
                zf = zipfile.ZipFile(zip_candidate)
                all_images = list(_iter_zip(zf))
            else:
                raise FileNotFoundError(
                    f"Could not find RSCD data at: {root}\n"
                    f"Expected either:\n"
                    f"  - Extracted folder with train/, test_50k/, vali_20k/ subfolders\n"
                    f"  - Zip file at {zip_candidate}\n"
                    f"Download: https://www.kaggle.com/datasets/cristvollerei/rscd-dataset-1million"
                )

    # ------------------------------------------------------------------
    # Filter splits
    # ------------------------------------------------------------------
    all_images = [img for img in all_images if img.split in allowed_splits]

    if not all_images:
        raise ValueError(
            "No images loaded. Check data_dir path and that the zip is not corrupted."
        )

    # ------------------------------------------------------------------
    # Optional per-class cap (for development)
    # ------------------------------------------------------------------
    if max_per_class is not None:
        from collections import defaultdict
        counters: Dict[tuple, int] = defaultdict(int)
        capped: List[RoadImage] = []
        for img in all_images:
            key = (img.friction, img.split)
            if counters[key] < max_per_class:
                capped.append(img)
                counters[key] += 1
        all_images = capped

    index = _build_index(all_images)
    return RSCDDataset(_images=all_images, _index=index)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    zip_path = Path(r"C:\_backup\ml\data\rscd-dataset-1million.zip")

    print("Loading RSCD from zip (test split only, max 1000/class for speed)...")
    ds = load(from_zip=zip_path, splits=["test"], max_per_class=1000)

    print(f"\nLoaded {len(ds)} images.\n")
    print(f"{'Friction':<18} {'Test':>8}")
    print("-" * 30)
    stats = ds.class_stats()
    for friction in sorted(stats):
        label = FRICTION_NAMES.get(friction, friction)
        test_n = stats[friction].get("test", 0)
        print(f"  {label:<16} {test_n:>8,}")

    print(f"\nConfusable pairs (test split):")
    for cp in CONFUSABLE_PAIRS:
        a, b = ds.sample_pair(cp["pair_id"], n_per_class=20, split="test")
        print(f"  {cp['pair_id']}")
        print(f"    {cp['class_a']}: {len(a)} images")
        print(f"    {cp['class_b']}: {len(b)} images")
        if a:
            print(f"    Sample A: {a[0]}")
        if b:
            print(f"    Sample B: {b[0]}")
        print()
