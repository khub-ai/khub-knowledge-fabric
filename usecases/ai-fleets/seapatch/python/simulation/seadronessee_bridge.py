"""
seadronessee_bridge.py — SeaDronesSee frame injection for swarm simulation.

Maps SeaDronesSee dataset frames to simulated per-drone camera feeds, enabling
the Phase 2 fleet broadcast demo without a rendering engine (AirSim/Gazebo
camera plugin).

Instead of live camera frames from simulated drones, each drone "captures"
a frame from a pre-assigned sequence of SeaDronesSee images. The bridge
assigns sequences to drone IDs and serves frames on demand.

Usage in simulation:

    from simulation.seadronessee_bridge import SeaDronesSeebridge

    bridge = SeaDronesSeebridge(
        dataset_root=Path("data/seadronessee"),
        coco_annotation=Path("data/seadronessee/annotations/instances_val.json"),
    )

    # Assign sequences to drone IDs
    scout_ids = [f"S{i+1:02d}" for i in range(38)]
    commander_ids = ["C1", "C2"]
    bridge.assign_sequences(scout_ids + commander_ids)

    # Populate archive from historical frames
    from archive import FrameBuffer
    buffer = FrameBuffer()
    n = bridge.inject_to_archive(buffer, lookback_seconds=2700, tier="scout")
    print(f"Injected {n} historical frames into archive")

    # Get a specific frame
    frame_path = bridge.get_frame("S22", timestamp="-18m")

See DESIGN.md §5.2 for the simulation architecture.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterator

import sys
_PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PYTHON_DIR))

from archive import (
    ArchivedFrame,
    FrameBuffer,
)


# ---------------------------------------------------------------------------
# COCO annotation support (mirrors pool_builder.py)
# ---------------------------------------------------------------------------

PERSON_CATEGORY_NAMES = {"swimmer", "person", "human"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _load_coco(path: Path) -> dict:
    return json.loads(path.read_text())


def _get_person_cat_ids(coco: dict) -> set[int]:
    return {
        cat["id"]
        for cat in coco.get("categories", [])
        if cat["name"].lower() in PERSON_CATEGORY_NAMES
    }


def _build_image_index(coco: dict) -> dict[int, dict]:
    return {img["id"]: img for img in coco.get("images", [])}


def _build_person_set(coco: dict, person_cat_ids: set[int]) -> set[int]:
    """Return set of image_ids that contain at least one person annotation."""
    return {
        ann["image_id"]
        for ann in coco.get("annotations", [])
        if ann.get("category_id") in person_cat_ids
    }


# ---------------------------------------------------------------------------
# Drone sequence assignment
# ---------------------------------------------------------------------------

@dataclass
class DroneSequence:
    drone_id: str
    tier: str
    frames: list[Path]           # ordered list of image paths
    current_index: int = 0
    ground_truth: list[str] = field(default_factory=list)  # label per frame


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class SeaDronesSeebridge:
    """Assigns SeaDronesSee dataset frames to simulated drone camera feeds.

    Each drone is assigned a shuffled subset of dataset frames that it
    "captures" in sequence. Frames can be retrieved by drone ID and
    relative timestamp.
    """

    def __init__(
        self,
        dataset_root: Path,
        coco_annotation: Path | None = None,
        seed: int = 42,
    ) -> None:
        self._root = dataset_root
        self._seed = seed
        self._rng = random.Random(seed)
        self._sequences: dict[str, DroneSequence] = {}

        # Load COCO annotations if provided — needed for ground truth labels
        self._person_image_ids: set[int] = set()
        self._image_index: dict[int, dict] = {}
        if coco_annotation and coco_annotation.exists():
            coco = _load_coco(coco_annotation)
            person_cat_ids = _get_person_cat_ids(coco)
            self._image_index = _build_image_index(coco)
            self._person_image_ids = _build_person_set(coco, person_cat_ids)

        # Build flat list of all available image paths
        self._all_frames = self._discover_frames()

    def _discover_frames(self) -> list[Path]:
        """Walk the dataset root and collect all image file paths."""
        frames = []
        images_dir = self._root / "images"
        if images_dir.exists():
            for p in sorted(images_dir.rglob("*")):
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                    frames.append(p)
        else:
            # Flat structure
            for p in sorted(self._root.rglob("*")):
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                    frames.append(p)
        return frames

    def _label_for_frame(self, frame_path: Path) -> str:
        """Return ground truth label for a frame path (if COCO available)."""
        # Try to match by filename to COCO image record
        for img_info in self._image_index.values():
            if Path(img_info["file_name"]).name == frame_path.name:
                if img_info["id"] in self._person_image_ids:
                    return "person_in_water"
                return "whitecap"
        # No COCO match — use directory name as heuristic
        parent = frame_path.parent.name.lower()
        if any(kw in parent for kw in ("person", "swimmer", "human")):
            return "person_in_water"
        return "whitecap"

    def assign_sequences(
        self,
        drone_ids: list[str],
        tier_map: dict[str, str] | None = None,
        frames_per_drone: int = 200,
    ) -> None:
        """Assign a shuffled subset of dataset frames to each drone ID.

        tier_map: {drone_id -> "scout" | "commander"}. If None, IDs starting
        with "C" are commanders; all others are scouts.
        """
        if not self._all_frames:
            raise RuntimeError(
                f"No image frames found under {self._root}. "
                "Check dataset_root path."
            )

        for drone_id in drone_ids:
            tier = "scout"
            if tier_map:
                tier = tier_map.get(drone_id, "scout")
            elif drone_id.startswith("C"):
                tier = "commander"

            # Each drone gets its own shuffled slice of the dataset
            shuffled = self._all_frames.copy()
            self._rng.shuffle(shuffled)
            assigned = shuffled[:frames_per_drone]

            ground_truth = [self._label_for_frame(p) for p in assigned]

            self._sequences[drone_id] = DroneSequence(
                drone_id=drone_id,
                tier=tier,
                frames=assigned,
                ground_truth=ground_truth,
            )

    def get_frame(
        self,
        drone_id: str,
        timestamp: str = "latest",
    ) -> Path | None:
        """Return the image path for a drone at a given timestamp.

        timestamp:
          "latest" — most recently captured frame
          "-18m"   — frame captured approximately 18 minutes ago
          int index — direct frame index

        Returns None if no frame is available.
        """
        if drone_id not in self._sequences:
            return None
        seq = self._sequences[drone_id]
        if not seq.frames:
            return None

        if timestamp == "latest":
            return seq.frames[seq.current_index % len(seq.frames)]

        if timestamp.startswith("-") and timestamp.endswith("m"):
            # Approximate: 1 frame every 2 seconds → 30 frames/min
            try:
                minutes_ago = float(timestamp[1:-1])
                frames_ago = int(minutes_ago * 30)
                idx = max(0, seq.current_index - frames_ago) % len(seq.frames)
                return seq.frames[idx]
            except ValueError:
                pass

        # Try integer index
        try:
            idx = int(timestamp) % len(seq.frames)
            return seq.frames[idx]
        except (ValueError, TypeError):
            return seq.frames[0]

    def advance(self, drone_id: str, n_frames: int = 1) -> None:
        """Advance a drone's frame pointer (simulate passage of time)."""
        if drone_id in self._sequences:
            seq = self._sequences[drone_id]
            seq.current_index = (seq.current_index + n_frames) % max(1, len(seq.frames))

    def advance_all(self, n_frames: int = 1) -> None:
        """Advance all drone frame pointers."""
        for drone_id in self._sequences:
            self.advance(drone_id, n_frames)

    def iter_confident_misses(
        self,
        tier: str = "scout",
        confidence: float = 0.91,
    ) -> Iterator[tuple[str, Path, str, float]]:
        """Yield (drone_id, frame_path, ground_truth_label, confidence) for
        frames where the ground truth is person_in_water but the original
        classifier would score them as whitecap (confident miss).

        In the absence of a running classifier, this yields person frames
        from the dataset as candidates — these are what the scout classifier
        was wrong about.
        """
        for drone_id, seq in self._sequences.items():
            if seq.tier != tier:
                continue
            for frame_path, label in zip(seq.frames, seq.ground_truth):
                if label == "person_in_water":
                    # These are the confident misses the scout classified as whitecap
                    yield drone_id, frame_path, label, confidence

    def inject_to_archive(
        self,
        frame_buffer: FrameBuffer,
        lookback_seconds: int = 2700,
        tier: str = "scout",
        original_class: str = "whitecap",
        original_confidence: float = 0.91,
        seconds_between_frames: float = 2.0,
    ) -> int:
        """Populate a FrameBuffer from the drone sequences.

        Creates ArchivedFrame entries with timestamps going backwards from now,
        simulating the drone fleet's historical capture archive.

        Only injects person_in_water frames as "whitecap" (confident misses)
        to represent the archive of frames that were incorrectly classified
        and not queued for human review.

        Returns number of frames injected.
        """
        base_time = datetime.now(timezone.utc)
        cutoff = base_time - timedelta(seconds=lookback_seconds)

        injected = 0
        for drone_id, seq in self._sequences.items():
            if seq.tier != tier:
                continue

            for i, (frame_path, label) in enumerate(zip(seq.frames, seq.ground_truth)):
                # Simulate capture time going backwards
                captured_at = base_time - timedelta(seconds=i * seconds_between_frames)
                if captured_at < cutoff:
                    break

                # Only inject person frames as whitecap (the confident misses)
                if label != "person_in_water":
                    continue

                frame = ArchivedFrame(
                    frame_id=f"{drone_id}_{frame_path.stem}",
                    drone_id=drone_id,
                    tier=tier,
                    captured_at=captured_at,
                    image_path=str(frame_path),
                    original_class=original_class,
                    original_confidence=original_confidence,
                )
                frame_buffer.add_frame(frame)
                injected += 1

        return injected

    def summary(self) -> dict:
        total_frames = sum(len(s.frames) for s in self._sequences.values())
        return {
            "dataset_root": str(self._root),
            "total_available_frames": len(self._all_frames),
            "n_drones_assigned": len(self._sequences),
            "total_assigned_frames": total_frames,
            "has_coco_labels": bool(self._image_index),
            "n_person_images_in_coco": len(self._person_image_ids),
            "by_tier": {
                tier: sum(1 for s in self._sequences.values() if s.tier == tier)
                for tier in ("scout", "commander")
            },
        }
