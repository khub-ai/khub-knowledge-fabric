"""
test_pool_builder.py — Unit tests for pool_builder.py COCO parsing logic.

No LLM calls, no dataset required. Tests COCO annotation parsing, frame
selection, and pool directory output.

Run from repo root:
    python -m pytest usecases/ai-fleets/drone-swarm/python/tests/test_pool_builder.py -v
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

_PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PYTHON_DIR))

from pool_builder import (
    get_person_category_ids,
    build_image_index,
    build_annotation_index,
    select_positive_frames,
    select_negative_frames,
    copy_frames_to_pool,
    POSITIVE_LABEL,
)


# ---------------------------------------------------------------------------
# COCO fixture builders
# ---------------------------------------------------------------------------

def _make_coco(
    n_images: int = 10,
    n_person_images: int = 3,
    person_cat_id: int = 1,
    person_cat_name: str = "swimmer",
) -> dict:
    """Build a minimal COCO annotation dict for testing."""
    categories = [{"id": person_cat_id, "name": person_cat_name}]
    images = [
        {"id": i, "file_name": f"seq/frame_{i:04d}.jpg", "width": 1920, "height": 1080}
        for i in range(n_images)
    ]
    annotations = []
    for i in range(n_person_images):
        annotations.append({
            "id": i,
            "image_id": i,
            "category_id": person_cat_id,
            "bbox": [800, 400, 20, 30],  # small bbox — person
        })
    return {"categories": categories, "images": images, "annotations": annotations}


# ---------------------------------------------------------------------------
# Category ID parsing
# ---------------------------------------------------------------------------

class TestGetPersonCategoryIds:

    def test_finds_swimmer(self):
        coco = _make_coco(person_cat_name="swimmer")
        ids = get_person_category_ids(coco)
        assert ids == {1}

    def test_finds_person(self):
        coco = {"categories": [{"id": 5, "name": "person"}]}
        ids = get_person_category_ids(coco)
        assert ids == {5}

    def test_case_insensitive(self):
        coco = {"categories": [{"id": 2, "name": "SWIMMER"}]}
        ids = get_person_category_ids(coco)
        assert ids == {2}

    def test_unknown_category(self):
        coco = {"categories": [{"id": 1, "name": "boat"}]}
        ids = get_person_category_ids(coco)
        assert ids == set()

    def test_empty_categories(self):
        ids = get_person_category_ids({})
        assert ids == set()

    def test_multiple_categories(self):
        coco = {
            "categories": [
                {"id": 1, "name": "swimmer"},
                {"id": 2, "name": "boat"},
                {"id": 3, "name": "buoy"},
            ]
        }
        ids = get_person_category_ids(coco)
        assert ids == {1}


# ---------------------------------------------------------------------------
# Image index
# ---------------------------------------------------------------------------

class TestBuildImageIndex:

    def test_keys_are_image_ids(self):
        coco = _make_coco(n_images=5)
        index = build_image_index(coco)
        assert set(index.keys()) == {0, 1, 2, 3, 4}

    def test_values_are_image_dicts(self):
        coco = _make_coco(n_images=2)
        index = build_image_index(coco)
        assert index[0]["file_name"] == "seq/frame_0000.jpg"

    def test_empty_images(self):
        assert build_image_index({}) == {}


# ---------------------------------------------------------------------------
# Annotation index
# ---------------------------------------------------------------------------

class TestBuildAnnotationIndex:

    def test_groups_by_image_id(self):
        coco = _make_coco(n_images=5, n_person_images=3)
        person_cat_ids = get_person_category_ids(coco)
        index = build_annotation_index(coco, person_cat_ids)
        assert set(index.keys()) == {0, 1, 2}

    def test_ignores_non_person_annotations(self):
        coco = {
            "categories": [{"id": 1, "name": "swimmer"}, {"id": 2, "name": "boat"}],
            "images": [{"id": 0, "file_name": "f.jpg", "width": 1920, "height": 1080}],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 1, "bbox": [0, 0, 20, 30]},  # person
                {"id": 1, "image_id": 0, "category_id": 2, "bbox": [0, 0, 200, 300]},  # boat
            ],
        }
        index = build_annotation_index(coco, {1})
        assert len(index[0]) == 1
        assert index[0][0]["category_id"] == 1

    def test_empty(self):
        assert build_annotation_index({}, {1}) == {}


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------

class TestSelectPositiveFrames:

    def test_returns_n_frames(self, tmp_path):
        # Create fake image files
        seq_dir = tmp_path / "seq"
        seq_dir.mkdir()
        for i in range(5):
            (seq_dir / f"frame_{i:04d}.jpg").write_bytes(b"JPEG")

        coco = _make_coco(n_images=5, n_person_images=5)
        image_index = build_image_index(coco)
        person_cat_ids = get_person_category_ids(coco)
        person_ann_index = build_annotation_index(coco, person_cat_ids)

        positives = select_positive_frames(image_index, person_ann_index, tmp_path, n=3)
        assert len(positives) == 3

    def test_respects_n_limit(self, tmp_path):
        seq_dir = tmp_path / "seq"
        seq_dir.mkdir()
        for i in range(2):
            (seq_dir / f"frame_{i:04d}.jpg").write_bytes(b"JPEG")

        coco = _make_coco(n_images=2, n_person_images=2)
        image_index = build_image_index(coco)
        person_cat_ids = get_person_category_ids(coco)
        person_ann_index = build_annotation_index(coco, person_cat_ids)

        # Request more than available
        positives = select_positive_frames(image_index, person_ann_index, tmp_path, n=100)
        assert len(positives) == 2

    def test_hardest_sorted_by_fraction(self, tmp_path):
        seq_dir = tmp_path / "seq"
        seq_dir.mkdir()
        for i in range(3):
            (seq_dir / f"frame_{i:04d}.jpg").write_bytes(b"JPEG")

        coco = {
            "categories": [{"id": 1, "name": "swimmer"}],
            "images": [
                {"id": i, "file_name": f"seq/frame_{i:04d}.jpg", "width": 1920, "height": 1080}
                for i in range(3)
            ],
            "annotations": [
                # Different bbox sizes → different person_fraction
                {"id": 0, "image_id": 0, "category_id": 1, "bbox": [800, 400, 5, 7]},    # tiny
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [800, 400, 50, 70]},  # medium
                {"id": 2, "image_id": 2, "category_id": 1, "bbox": [800, 400, 200, 300]}, # large
            ],
        }
        image_index = build_image_index(coco)
        person_cat_ids = get_person_category_ids(coco)
        person_ann_index = build_annotation_index(coco, person_cat_ids)

        positives = select_positive_frames(
            image_index, person_ann_index, tmp_path, n=3, hardest=True
        )
        # First should be smallest person_fraction (hardest = image 0)
        fractions = [p["person_fraction"] for p in positives]
        assert fractions == sorted(fractions)

    def test_skips_missing_files(self, tmp_path):
        # Only create some of the image files
        seq_dir = tmp_path / "seq"
        seq_dir.mkdir()
        (seq_dir / "frame_0000.jpg").write_bytes(b"JPEG")
        # frame_0001.jpg and frame_0002.jpg deliberately not created

        coco = _make_coco(n_images=3, n_person_images=3)
        image_index = build_image_index(coco)
        person_cat_ids = get_person_category_ids(coco)
        person_ann_index = build_annotation_index(coco, person_cat_ids)

        positives = select_positive_frames(image_index, person_ann_index, tmp_path, n=10)
        assert len(positives) == 1  # Only 1 file exists


class TestSelectNegativeFrames:

    def test_excludes_person_images(self, tmp_path):
        seq_dir = tmp_path / "seq"
        seq_dir.mkdir()
        for i in range(5):
            (seq_dir / f"frame_{i:04d}.jpg").write_bytes(b"JPEG")

        # Images 0,1,2 have persons; 3,4 don't
        coco = _make_coco(n_images=5, n_person_images=3)
        image_index = build_image_index(coco)
        person_cat_ids = get_person_category_ids(coco)
        person_ann_index = build_annotation_index(coco, person_cat_ids)

        negatives = select_negative_frames(image_index, person_ann_index, tmp_path, n=10)
        negative_ids = {f["image_id"] for f in negatives}
        assert negative_ids.isdisjoint({0, 1, 2})

    def test_returns_n_frames(self, tmp_path):
        seq_dir = tmp_path / "seq"
        seq_dir.mkdir()
        for i in range(10):
            (seq_dir / f"frame_{i:04d}.jpg").write_bytes(b"JPEG")

        coco = _make_coco(n_images=10, n_person_images=3)
        image_index = build_image_index(coco)
        person_cat_ids = get_person_category_ids(coco)
        person_ann_index = build_annotation_index(coco, person_cat_ids)

        negatives = select_negative_frames(image_index, person_ann_index, tmp_path, n=4)
        assert len(negatives) == 4


# ---------------------------------------------------------------------------
# Pool output
# ---------------------------------------------------------------------------

class TestCopyFramesToPool:

    def test_creates_directory_structure(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        pool = tmp_path / "pool"

        # Create source images
        positives = []
        for i in range(2):
            p = src / f"person_{i}.jpg"
            p.write_bytes(b"JPEG")
            positives.append({"image_id": i, "image_path": str(p),
                               "person_fraction": 0.001 * i})

        negatives = []
        for i in range(3):
            p = src / f"whitecap_{i}.jpg"
            p.write_bytes(b"JPEG")
            negatives.append({"image_id": i + 100, "image_path": str(p)})

        manifest = copy_frames_to_pool(positives, negatives, pool, negative_label="whitecap")

        assert (pool / POSITIVE_LABEL).is_dir()
        assert (pool / "whitecap").is_dir()
        assert (pool / "pool_manifest.json").exists()

        pos_files = list((pool / POSITIVE_LABEL).iterdir())
        neg_files = list((pool / "whitecap").iterdir())
        assert len(pos_files) == 2
        assert len(neg_files) == 3

    def test_manifest_structure(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        pool = tmp_path / "pool"

        p = src / "person.jpg"
        p.write_bytes(b"JPEG")
        positives = [{"image_id": 1, "image_path": str(p), "person_fraction": 0.0005}]

        n = src / "whitecap.jpg"
        n.write_bytes(b"JPEG")
        negatives = [{"image_id": 2, "image_path": str(n)}]

        manifest = copy_frames_to_pool(positives, negatives, pool, negative_label="whitecap")

        assert len(manifest) == 2
        labels = {e["label"] for e in manifest}
        assert labels == {POSITIVE_LABEL, "whitecap"}

        # Manifest on disk should match returned manifest
        disk_manifest = json.loads((pool / "pool_manifest.json").read_text())
        assert disk_manifest == manifest

    def test_failure_dir_output(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        pool = tmp_path / "pool"
        failure_dir = tmp_path / "failures"

        p = src / "person.jpg"
        p.write_bytes(b"JPEG")
        positives = [{"image_id": 1, "image_path": str(p), "person_fraction": 0.0003}]
        failure_frames = [{"image_id": 1, "image_path": str(p), "person_fraction": 0.0003}]

        copy_frames_to_pool(
            positives, [], pool, negative_label="whitecap",
            failure_frames=failure_frames, failure_dir=failure_dir,
        )

        assert failure_dir.is_dir()
        assert (failure_dir / "failure_manifest.json").exists()
        failure_manifest = json.loads((failure_dir / "failure_manifest.json").read_text())
        assert len(failure_manifest) == 1
        assert failure_manifest[0]["label"] == POSITIVE_LABEL
