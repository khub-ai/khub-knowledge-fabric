"""
archive.py — Frame buffer and retroactive reprocessing for SeaPatch.

Implements:
  - FrameBuffer: time-indexed store of archived drone frames
  - reprocess_archive(): apply a newly minted rule to historical confident negatives
  - ReclassifiedFrame: output type for retroactive detections

In Phase 1 (standalone DD), the frame buffer is populated from the
SeaDronesSee dataset via seadronessee_bridge.py. In Phase 2 (simulation),
frames are ingested from live Gazebo drone feeds.

See DESIGN.md §3.4 for the retroactive reprocessing protocol.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from core.dialogic_distillation import agents as _core_agents
from core.dialogic_distillation.protocols import DomainConfig

_PYTHON_DIR = Path(__file__).resolve().parent
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from domain_config import MARITIME_SAR_CONFIG


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ArchivedFrame:
    frame_id: str
    drone_id: str
    tier: str                          # "scout" | "commander"
    captured_at: datetime
    image_path: str
    original_class: str                # class predicted at capture time
    original_confidence: float
    coordinates: tuple[float, float] | None = None  # (lat, lon) — None in simulation

    @property
    def age_seconds(self) -> float:
        now = datetime.now(timezone.utc)
        delta = now - self.captured_at
        return delta.total_seconds()


@dataclass
class ReclassifiedFrame:
    frame: ArchivedFrame
    new_class: str
    rule_id: str
    rule_text: str
    observations: str
    reclassified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame.frame_id,
            "drone_id": self.frame.drone_id,
            "tier": self.frame.tier,
            "captured_at": self.frame.captured_at.isoformat(),
            "image_path": self.frame.image_path,
            "original_class": self.frame.original_class,
            "original_confidence": self.frame.original_confidence,
            "new_class": self.new_class,
            "rule_id": self.rule_id,
            "rule_text": self.rule_text,
            "observations": self.observations,
            "reclassified_at": self.reclassified_at.isoformat(),
            "coordinates": self.frame.coordinates,
        }


# ---------------------------------------------------------------------------
# Frame buffer
# ---------------------------------------------------------------------------

class FrameBuffer:
    """Time-indexed store of archived drone frames.

    Frames are stored in memory (simulation) or on disk (production).
    For production use, replace the in-memory list with a database query.
    """

    def __init__(self) -> None:
        self._frames: list[ArchivedFrame] = []

    def add_frame(self, frame: ArchivedFrame) -> None:
        """Add a frame to the buffer."""
        self._frames.append(frame)

    def add_frames(self, frames: list[ArchivedFrame]) -> None:
        """Add multiple frames."""
        self._frames.extend(frames)

    def query(
        self,
        max_age_seconds: int = 2700,
        tier: str | None = None,
        original_class: str | None = None,
        confidence_min: float = 0.0,
        drone_id: str | None = None,
    ) -> list[ArchivedFrame]:
        """Return frames matching the query criteria.

        Args:
            max_age_seconds: only include frames captured within this window
            tier: filter by drone tier ("scout" | "commander")
            original_class: filter by the class predicted at capture time
            confidence_min: only include frames where original_confidence >= this
            drone_id: filter to a specific drone
        """
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
        results = []
        for frame in self._frames:
            if frame.captured_at < cutoff:
                continue
            if tier and frame.tier != tier:
                continue
            if original_class and frame.original_class != original_class:
                continue
            if frame.original_confidence < confidence_min:
                continue
            if drone_id and frame.drone_id != drone_id:
                continue
            results.append(frame)
        return results

    def __len__(self) -> int:
        return len(self._frames)

    def summary(self) -> dict:
        by_tier: dict[str, int] = {}
        by_class: dict[str, int] = {}
        for frame in self._frames:
            by_tier[frame.tier] = by_tier.get(frame.tier, 0) + 1
            by_class[frame.original_class] = by_class.get(frame.original_class, 0) + 1
        return {
            "total_frames": len(self._frames),
            "by_tier": by_tier,
            "by_class": by_class,
        }


# ---------------------------------------------------------------------------
# Apply a rule to a single archived frame
# ---------------------------------------------------------------------------

async def _apply_rule_to_frame(
    rule: dict,
    frame: ArchivedFrame,
    rule_id: str,
    config: DomainConfig,
    validator_model: str = "",
    call_agent_fn: Callable | None = None,
) -> ReclassifiedFrame | None:
    """Apply a rule to one archived frame.

    Returns ReclassifiedFrame if the rule fires (reclassification happened),
    None if the rule does not fire.
    """
    if not Path(frame.image_path).exists():
        return None

    result, _ = await _core_agents.run_rule_validator_on_image(
        image_path=frame.image_path,
        ground_truth=frame.original_class,  # original label (may be wrong)
        candidate_rule=rule,
        config=config,
        model=validator_model,
        call_agent_fn=call_agent_fn,
    )

    if result.get("precondition_met") and result.get("would_predict") == rule.get("favors"):
        return ReclassifiedFrame(
            frame=frame,
            new_class=rule["favors"],
            rule_id=rule_id,
            rule_text=rule.get("rule", ""),
            observations=result.get("observations", ""),
        )
    return None


# ---------------------------------------------------------------------------
# Retroactive reprocessing
# ---------------------------------------------------------------------------

async def reprocess_archive(
    rule: dict,
    rule_id: str,
    frame_buffer: FrameBuffer,
    config: DomainConfig = MARITIME_SAR_CONFIG,
    lookback_seconds: int = 2700,         # 45 minutes
    tier: str = "scout",
    reprocess_class: str = "life_ring_unoccupied",  # only re-examine confident negatives
    confidence_min: float = 0.70,
    validator_model: str = "claude-sonnet-4-6",
    call_agent_fn: Callable | None = None,
    console=None,
) -> list[ReclassifiedFrame]:
    """Retroactively apply a newly minted rule to the archive.

    Only re-examines frames where:
      - tier matches
      - original_class matches reprocess_class (confident negatives)
      - original_confidence >= confidence_min

    This targets the frames that were confidently wrong — the ones that
    would never have been queued for human review.

    Returns list of reclassified frames (only those the rule fires on).
    """
    _print = console.print if console else lambda *a, **kw: None

    frames = frame_buffer.query(
        max_age_seconds=lookback_seconds,
        tier=tier,
        original_class=reprocess_class,
        confidence_min=confidence_min,
    )

    _print(f"  Archive query: {len(frames)} {reprocess_class} frames "
           f"from {tier} tier in last {lookback_seconds//60}m "
           f"(confidence >= {confidence_min})")

    if not frames:
        return []

    t0 = time.monotonic()

    # Run rule application concurrently
    tasks = [
        _apply_rule_to_frame(
            rule=rule,
            frame=frame,
            rule_id=rule_id,
            config=config,
            validator_model=validator_model,
            call_agent_fn=call_agent_fn,
        )
        for frame in frames
    ]
    results = await asyncio.gather(*tasks)

    reclassified = [r for r in results if r is not None]
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    _print(f"  Reprocessed {len(frames)} frames in {elapsed_ms}ms — "
           f"{len(reclassified)} reclassified as {rule.get('favors', '?')}")

    return reclassified


# ---------------------------------------------------------------------------
# Convenience: load frames from a directory (for simulation / testing)
# ---------------------------------------------------------------------------

def load_frames_from_directory(
    directory: Path,
    drone_id: str,
    tier: str,
    original_class: str,
    original_confidence: float = 0.91,
    base_time: datetime | None = None,
    seconds_between_frames: float = 2.0,
) -> list[ArchivedFrame]:
    """Create ArchivedFrame entries from a directory of images.

    Useful for simulating a drone's historical archive from a SeaDronesSee
    split without running a live simulation.

    Frames are timestamped backwards from base_time (or now) at
    seconds_between_frames intervals.
    """
    if base_time is None:
        base_time = datetime.now(timezone.utc)

    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )

    frames = []
    for i, img_path in enumerate(image_paths):
        # Assign timestamps going backwards from base_time
        offset = timedelta(seconds=i * seconds_between_frames)
        captured_at = base_time - offset

        frame = ArchivedFrame(
            frame_id=f"{drone_id}_{img_path.stem}",
            drone_id=drone_id,
            tier=tier,
            captured_at=captured_at,
            image_path=str(img_path),
            original_class=original_class,
            original_confidence=original_confidence,
        )
        frames.append(frame)

    return frames
