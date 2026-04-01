"""
dataset.py — Standalone CUB-200-2011 loader for the KF UC200 use case.

Takes data_dir as a parameter — no hardcoded paths.  The harness passes
--data-dir on the CLI; defaults to the same path used by src/config.py.

Provides:
  load(data_dir)         Load full CUB-200-2011 dataset from a directory.
  CUBDataset             Container with per-class split access and sampling.
  CUBImage               Single image record (path, class, split, bbox).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random


DEFAULT_DATA_DIR = Path(r"C:\_backup\ml\data\_tmp\CUB_200_2011\CUB_200_2011")

_SEED = 42


@dataclass
class CUBImage:
    image_id: int
    file_path: Path                                     # absolute path to .jpg
    class_id: int                                       # 1-indexed CUB class ID
    class_name: str                                     # e.g. "029.American_Crow"
    is_train: bool
    bbox: Optional[Tuple[float, float, float, float]] = None  # x, y, w, h


@dataclass
class CUBDataset:
    classes: Dict[int, str]         # class_id -> "NNN.Species_Name"
    images: List[CUBImage]
    _by_class: Dict[int, List[CUBImage]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for img in self.images:
            self._by_class.setdefault(img.class_id, []).append(img)

    def images_for_class(self, class_id: int, split: str = "test") -> List[CUBImage]:
        imgs = self._by_class.get(class_id, [])
        if split == "train":
            return [i for i in imgs if i.is_train]
        elif split == "test":
            return [i for i in imgs if not i.is_train]
        return imgs

    def class_name_clean(self, class_id: int) -> str:
        """Return plain species name, e.g. 'American Crow'."""
        raw = self.classes[class_id]    # "029.American_Crow"
        return raw.split(".", 1)[1].replace("_", " ")

    def sample_images(
        self,
        class_id: int,
        n: int,
        split: str = "train",
        seed: int = _SEED,
    ) -> List[CUBImage]:
        imgs = self.images_for_class(class_id, split=split)
        rng = random.Random(seed)
        return rng.sample(imgs, min(n, len(imgs)))


def load(data_dir: str | Path = DEFAULT_DATA_DIR) -> CUBDataset:
    """Load the full CUB-200-2011 dataset from disk."""
    root = Path(data_dir)

    def _read(rel: str) -> list[str]:
        return (root / rel).read_text().splitlines()

    # 1. Classes
    classes: Dict[int, str] = {}
    for line in _read("classes.txt"):
        cid, name = line.strip().split(" ", 1)
        classes[int(cid)] = name

    # 2. Image filenames
    filenames: Dict[int, Path] = {}
    images_dir = root / "images"
    for line in _read("images.txt"):
        iid, rel_path = line.strip().split(" ", 1)
        filenames[int(iid)] = images_dir / rel_path

    # 3. Labels
    labels: Dict[int, int] = {}
    for line in _read("image_class_labels.txt"):
        iid, cid = line.strip().split()
        labels[int(iid)] = int(cid)

    # 4. Train/test split  (1 = train, 0 = test)
    splits: Dict[int, bool] = {}
    for line in _read("train_test_split.txt"):
        iid, flag = line.strip().split()
        splits[int(iid)] = flag == "1"

    # 5. Bounding boxes
    bboxes: Dict[int, Tuple] = {}
    for line in _read("bounding_boxes.txt"):
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
    train = [i for i in ds.images if i.is_train]
    test  = [i for i in ds.images if not i.is_train]
    print(f"Classes : {len(ds.classes)}")
    print(f"Total   : {len(ds.images)}")
    print(f"Train   : {len(train)}")
    print(f"Test    : {len(test)}")
    sample = ds.sample_images(class_id=29, n=3, split="test")
    print(f"\nSample test images for class 29 ({ds.class_name_clean(29)}):")
    for img in sample:
        print(f"  [{img.image_id}] {img.file_path.name}  bbox={img.bbox}")
