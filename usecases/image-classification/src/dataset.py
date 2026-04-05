"""
CUB-200-2011 dataset loader.
Provides train/test splits, class metadata, and image paths.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

from config import (
    IMAGES_DIR, CLASSES_FILE, IMAGES_FILE,
    LABELS_FILE, SPLIT_FILE, BBOXES_FILE, SEED
)


@dataclass
class CUBImage:
    image_id: int
    file_path: Path          # absolute path to .jpg
    class_id: int            # 1-indexed
    class_name: str          # e.g. "029.American_Crow"
    is_train: bool
    bbox: Optional[Tuple[float, float, float, float]] = None  # x, y, w, h


@dataclass
class CUBDataset:
    classes: Dict[int, str]          # class_id -> "NNN.Species_Name"
    images: List[CUBImage]
    _by_class: Dict[int, List[CUBImage]] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        for img in self.images:
            self._by_class.setdefault(img.class_id, []).append(img)

    # ------------------------------------------------------------------
    def train_images(self) -> List[CUBImage]:
        return [i for i in self.images if i.is_train]

    def test_images(self) -> List[CUBImage]:
        return [i for i in self.images if not i.is_train]

    def images_for_class(self, class_id: int, split: str = "test") -> List[CUBImage]:
        imgs = self._by_class.get(class_id, [])
        if split == "train":
            return [i for i in imgs if i.is_train]
        elif split == "test":
            return [i for i in imgs if not i.is_train]
        return imgs

    def class_name_clean(self, class_id: int) -> str:
        """Return plain species name, e.g. 'American Crow'."""
        raw = self.classes[class_id]           # "029.American_Crow"
        name = raw.split(".", 1)[1]            # "American_Crow"
        return name.replace("_", " ")

    def sample_test_images(
        self,
        class_id: int,
        n: int,
        seed: int = SEED,
    ) -> List[CUBImage]:
        imgs = self.images_for_class(class_id, split="test")
        rng = random.Random(seed)
        return rng.sample(imgs, min(n, len(imgs)))


def load() -> CUBDataset:
    """Load the full CUB-200-2011 dataset from disk."""

    # 1. Classes
    classes: Dict[int, str] = {}
    with open(CLASSES_FILE) as f:
        for line in f:
            cid, name = line.strip().split(" ", 1)
            classes[int(cid)] = name

    # 2. Image filenames
    filenames: Dict[int, Path] = {}
    with open(IMAGES_FILE) as f:
        for line in f:
            iid, rel_path = line.strip().split(" ", 1)
            filenames[int(iid)] = IMAGES_DIR / rel_path

    # 3. Labels
    labels: Dict[int, int] = {}
    with open(LABELS_FILE) as f:
        for line in f:
            iid, cid = line.strip().split()
            labels[int(iid)] = int(cid)

    # 4. Train/test split  (1 = train, 0 = test)
    splits: Dict[int, bool] = {}
    with open(SPLIT_FILE) as f:
        for line in f:
            iid, flag = line.strip().split()
            splits[int(iid)] = flag == "1"

    # 5. Bounding boxes
    bboxes: Dict[int, Tuple] = {}
    with open(BBOXES_FILE) as f:
        for line in f:
            parts = line.strip().split()
            iid = int(parts[0])
            bboxes[iid] = tuple(float(v) for v in parts[1:])

    # 6. Assemble
    images = []
    for iid, path in filenames.items():
        cid = labels[iid]
        images.append(CUBImage(
            image_id=iid,
            file_path=path,
            class_id=cid,
            class_name=classes[cid],
            is_train=splits[iid],
            bbox=bboxes.get(iid),
        ))

    return CUBDataset(classes=classes, images=images)


if __name__ == "__main__":
    ds = load()
    train = ds.train_images()
    test  = ds.test_images()
    print(f"Classes : {len(ds.classes)}")
    print(f"Total   : {len(ds.images)}")
    print(f"Train   : {len(train)}")
    print(f"Test    : {len(test)}")
    # Spot-check
    sample = ds.sample_test_images(class_id=29, n=3)
    print(f"\nSample test images for class 29 ({ds.class_name_clean(29)}):")
    for img in sample:
        print(f"  [{img.image_id}] {img.file_path.name}  bbox={img.bbox}")
