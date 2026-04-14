"""
mwir_oracle.py — Simulated MWIR commander aircraft oracle for PyroWatch.

Provides ground truth confirmation in simulation by returning the label
from the dataset annotation, formatted as it would appear from a real
calibrated MWIR sensor system.

In Phase 1 (FIgLib), ground truth comes from dataset labels (fire / non-fire).
In Phase 4 (production), this is replaced by the actual MWIR camera API.

Usage:
    from simulation.mwir_oracle import oracle_for_frame

    result = oracle_for_frame(
        frame_path="data/figlib/fire/img_0042.jpg",
        ground_truth_label="early_smoke_signature",
        agent_id="CA-1",
        coordinates=(34.052, -117.318),
    )
    # result.confirmation_details → "MWIR sensor CA-1 confirmed 380°C surface
    #   anomaly at (34.052, -117.318). Absolute temperature above 280°C
    #   chaparral combustion threshold..."
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Oracle output
# ---------------------------------------------------------------------------

@dataclass
class MWIROracleResult:
    ground_truth_class: str
    confirmation_modality: str
    confirmation_details: str
    agent_id: str
    confidence: float          # 1.0 in simulation (ground truth)
    coordinates: tuple[float, float] | None = None


# ---------------------------------------------------------------------------
# Temperature profiles per class
# ---------------------------------------------------------------------------

# Typical MWIR absolute temperature readings (°C) by class.
# Used to generate realistic confirmation text.
MWIR_TEMPERATURE_BY_CLASS: dict[str, dict] = {
    "early_smoke_signature": {
        "surface_temp_c": 380,
        "background_temp_c": 45,   # hot rock on a 95°F day
        "description": "distinct surface temperature anomaly above 280°C chaparral combustion threshold",
    },
    "heat_shimmer_artifact": {
        "surface_temp_c": 58,
        "background_temp_c": 45,
        "description": "surface temperature consistent with sun-heated terrain (58°C); below combustion threshold",
    },
    "atmospheric_haze": {
        "surface_temp_c": 46,
        "background_temp_c": 45,
        "description": "no surface temperature anomaly; atmospheric column only",
    },
    "dust_plume": {
        "surface_temp_c": 52,
        "background_temp_c": 45,
        "description": "minor surface disturbance; no persistent thermal source above background",
    },
    "fog_patch": {
        "surface_temp_c": 35,
        "background_temp_c": 45,
        "description": "low surface temperature consistent with moisture evaporation; no ignition source",
    },
    "no_fire": {
        "surface_temp_c": 47,
        "background_temp_c": 45,
        "description": "no surface temperature anomaly; terrain within normal diurnal range",
    },
}

_FIRE_CLASSES = {"early_smoke_signature"}
_COMBUSTION_THRESHOLD_C = 280


def oracle_for_frame(
    frame_path: str,
    ground_truth_label: str,
    agent_id: str,
    coordinates: tuple[float, float] | None = None,
) -> MWIROracleResult:
    """Return a simulated MWIR oracle result for a frame.

    Parameters
    ----------
    frame_path:
        Path to the RGB frame (used for provenance; not read here).
    ground_truth_label:
        The true class of the frame from dataset annotation.
    agent_id:
        Commander aircraft ID making the MWIR observation.
    coordinates:
        (lat, lon) of the anomaly, if known.
    """
    profile = MWIR_TEMPERATURE_BY_CLASS.get(
        ground_truth_label,
        MWIR_TEMPERATURE_BY_CLASS["no_fire"],
    )

    surface_temp = profile["surface_temp_c"]
    bg_temp = profile["background_temp_c"]
    desc = profile["description"]

    coord_str = (
        f" at ({coordinates[0]:.4f}°, {coordinates[1]:.4f}°)"
        if coordinates else ""
    )

    if ground_truth_label in _FIRE_CLASSES:
        details = (
            f"MWIR sensor {agent_id} confirmed {surface_temp}°C surface anomaly{coord_str}. "
            f"{desc.capitalize()}. "
            f"Delta above background terrain: +{surface_temp - bg_temp}°C. "
            f"Anomaly spatially stable across 3 consecutive MWIR frames (2-second interval). "
            f"Absolute temperature exceeds {_COMBUSTION_THRESHOLD_C}°C combustion threshold — "
            f"confirmed ignition."
        )
    else:
        details = (
            f"MWIR sensor {agent_id} measured {surface_temp}°C peak surface temperature{coord_str}. "
            f"{desc.capitalize()}. "
            f"Delta above background terrain: +{surface_temp - bg_temp}°C. "
            f"No persistent thermal source above {_COMBUSTION_THRESHOLD_C}°C combustion threshold. "
            f"Not confirmed as ignition."
        )

    return MWIROracleResult(
        ground_truth_class=ground_truth_label,
        confirmation_modality="MWIR_calibrated_thermal",
        confirmation_details=details,
        agent_id=agent_id,
        confidence=1.0,
        coordinates=coordinates,
    )


def mwir_from_real_sensor(
    mwir_image_path: str,
    rgb_image_path: str,
) -> MWIROracleResult:
    """Placeholder for Phase 4 real MWIR sensor integration.

    In production, this reads calibrated MWIR frames from the aircraft
    camera API and returns an absolute temperature map for threshold comparison.

    Not implemented — Phase 4 only.
    """
    raise NotImplementedError(
        "mwir_from_real_sensor() requires Phase 4 hardware integration. "
        "Use oracle_for_frame() for simulation."
    )
