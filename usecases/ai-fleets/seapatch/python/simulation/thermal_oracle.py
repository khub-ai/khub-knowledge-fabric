"""
thermal_oracle.py — Ground truth oracle replacing live thermal FLIR input.

In Phase 1/2, we don't have a real commander drone with a thermal camera.
This module substitutes SeaDronesSee ground truth labels for thermal
confirmation, so the TUTOR dialogue can run without physical hardware.

In production (Phase 4), replace oracle_for_frame() with a real thermal
image reader that calls the FLIR API or reads from a Gazebo thermal plugin.

See DESIGN.md §5.3 for the full thermal simulation options.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple


class ThermalConfirmation(NamedTuple):
    ground_truth_class: str
    confirmation_modality: str
    confirmation_details: str
    confidence: float


# ---------------------------------------------------------------------------
# Ground truth oracle (simulation substitute for thermal FLIR)
# ---------------------------------------------------------------------------

def oracle_for_frame(
    frame_path: str | Path,
    ground_truth_label: str,
    drone_id: str = "C1",
    coordinates: tuple[float, float] | None = None,
) -> ThermalConfirmation:
    """Return a simulated thermal confirmation for a frame.

    Parameters
    ----------
    frame_path:
        Path to the RGB frame that needs confirmation. Not used in
        simulation — the label is provided directly. Kept as parameter
        so the calling signature matches the production interface.
    ground_truth_label:
        The actual ground truth class (from SeaDronesSee annotations
        or manual labeling).
    drone_id:
        ID of the commander drone providing confirmation.
    coordinates:
        Optional (lat, lon) of the detection location.

    Returns
    -------
    ThermalConfirmation with details formatted for the TUTOR prompt.
    """
    coord_str = ""
    if coordinates:
        coord_str = f" at coordinates ({coordinates[0]:.4f}, {coordinates[1]:.4f})"

    if ground_truth_label == "person_in_water":
        details = (
            f"Commander drone {drone_id} thermal camera (FLIR Tau 2) detected "
            f"a spatially stable 36–37°C oval heat signature "
            f"approximately 20×15 cm{coord_str}. "
            f"The signature persisted across 8 consecutive thermal frames "
            f"(~16 seconds), ruling out transient sea surface heating. "
            f"Signature geometry is consistent with a human head at the waterline. "
            f"[ORACLE: SeaDronesSee ground truth = person_in_water]"
        )
    else:
        details = (
            f"Commander drone {drone_id} thermal camera detected no persistent "
            f"heat signature above ambient sea surface temperature{coord_str}. "
            f"Sea surface temperature: 14°C. No thermal anomaly detected. "
            f"[ORACLE: SeaDronesSee ground truth = {ground_truth_label}]"
        )

    return ThermalConfirmation(
        ground_truth_class=ground_truth_label,
        confirmation_modality="thermal_FLIR (oracle)",
        confirmation_details=details,
        confidence=1.0,  # oracle is always correct
    )


# ---------------------------------------------------------------------------
# Manifest-based oracle (for pool_builder.py output with failure_manifest.json)
# ---------------------------------------------------------------------------

def oracle_from_manifest(
    frame_path: str | Path,
    failure_dir: Path,
    drone_id: str = "C1",
) -> ThermalConfirmation | None:
    """Look up ground truth from a failure_manifest.json.

    Returns None if the frame is not in the manifest.
    """
    manifest_path = failure_dir / "failure_manifest.json"
    if not manifest_path.exists():
        return None

    frame_name = Path(frame_path).name
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest:
        if Path(entry["path"]).name == frame_name:
            return oracle_for_frame(
                frame_path=frame_path,
                ground_truth_label=entry["label"],
                drone_id=drone_id,
            )
    return None


# ---------------------------------------------------------------------------
# FLIR reader stub (production replacement)
# ---------------------------------------------------------------------------

def thermal_from_flir_image(
    thermal_image_path: str | Path,
    rgb_frame_path: str | Path,
    drone_id: str = "C1",
    coordinates: tuple[float, float] | None = None,
) -> ThermalConfirmation:
    """Production interface: extract confirmation from a real FLIR thermal image.

    NOT IMPLEMENTED — placeholder for Phase 4 real-data integration.

    In production:
    - Read the thermal image (FLIR radiometric JPEG or TIFF)
    - Detect any heat signature above sea surface baseline
    - If detected: extract centroid temperature and geometry
    - Return ThermalConfirmation with measured values

    For now raises NotImplementedError to make the stub explicit.
    """
    raise NotImplementedError(
        "thermal_from_flir_image() is a Phase 4 stub. "
        "Use oracle_for_frame() for simulation (Phase 1/2)."
    )
