"""
Domain configuration for the PyroWatch wildfire early-ignition detection use case.

Three-tier fleet: ground sentinels (RGB PTZ + uncalibrated LWIR),
scout drones (RGB + LWIR 640×480), commander aircraft (calibrated MWIR + RGB).

Key adaptations over SeaPatch:
  1. Three tiers instead of two — grounding check produces three rule variants
     from a single DD session.
  2. Environmental context injection — accepted rules include RAWS
     meteorological preconditions (wind, humidity, temperature) evaluated at
     inference time. The same optical rule activates aggressively under red
     flag conditions and conservatively in spring, without retraining.
  3. Temporal feature reformulation — "smoke drift consistent across frames"
     (available on fixed sentinel PTZ) is reformulated to a within-frame proxy
     (wind-axis elongation) for single-pass scout drones.

See README.md §7–8 for the design rationale.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from core.dialogic_distillation.protocols import DomainConfig

# ---------------------------------------------------------------------------
# Primary domain configuration
# ---------------------------------------------------------------------------

WILDFIRE_DETECTION_CONFIG = DomainConfig(
    expert_role=(
        "Cal Fire lookout, aerial observer, and wildfire behavior analyst "
        "with 19 years of field experience in Southern California chaparral "
        "fire environments"
    ),
    item_noun="wildfire surveillance camera frame",
    item_noun_plural="wildfire surveillance camera frames",
    classification_noun="early ignition assessment",
    class_noun="ignition class",
    feature_noun=(
        "smoke plume characteristic, terrain heating pattern, "
        "or atmospheric optical signature"
    ),
    observation_guidance=(
        "smoke column color (blue/blue-gray vs gray vs transparent shimmer), "
        "presence of a well-defined point source at the base of the haze, "
        "spatial pattern of haze distribution (concentrated origin vs diffuse "
        "hillside shimmer), elongation axis of haze column relative to visible "
        "terrain topography, color saturation gradient from base to dispersal "
        "zone, contrast of haze against sky and terrain background, "
        "opacity at the origin point vs distal dispersal zone"
    ),
    non_visual_exclusions=(
        "MWIR calibrated temperature readings (not available on sentinel or "
        "scout tiers), RAWS weather station measurements (not in camera frame), "
        "AGL altitude measurements, GPS coordinates, wind speed numerical values, "
        "relative humidity percentage, acoustic fire sound, "
        "satellite FIRMS hotspot data, radio dispatch communications"
    ),
    good_vocabulary_examples=[
        (
            "a blue or blue-gray haze visible in the frame, distinctly different "
            "in color from the surrounding terrain and sky — chaparral terpene "
            "smoke has this cooler color at early stages, unlike mature gray smoke "
            "or the transparent wavering of heat shimmer"
        ),
        (
            "the haze has a clearly identifiable point of origin: a concentrated "
            "spot of higher opacity from which the haze fans outward — unlike heat "
            "shimmer, which rises simultaneously from a broad hillside surface "
            "with no single source point"
        ),
        (
            "the haze column is more opaque and saturated at its base than at "
            "its distal end, indicating a source that is emitting material rather "
            "than a surface reflection effect — shimmer shows uniform transparency "
            "across its full extent"
        ),
    ],
    bad_vocabulary_examples=[
        (
            "MWIR camera confirms 380°C surface temperature at these coordinates "
            "(MWIR sensor not present on sentinel or scout tiers)"
        ),
        (
            "RAWS reports 28 mph wind and 8% relative humidity "
            "(weather station data not available in the camera frame)"
        ),
        (
            "the haze drifts southeast consistently across 12 consecutive frames "
            "(temporal feature — single-pass scout drones process one frame per "
            "location; reformulate as a within-frame spatial proxy)"
        ),
        (
            "fire has been confirmed by ground crew radio "
            "(external confirmation not available at inference time)"
        ),
        (
            "this is clearly early-stage chaparral combustion "
            "(direct identification — must be derived from observable optical "
            "features, not stated as a premise)"
        ),
    ],
    precision_gate=0.90,   # false positives divert fire resources; gate is firm
    max_fp=0,              # zero tolerance during red flag periods
)

# ---------------------------------------------------------------------------
# Tier-specific observability contexts for the KF grounding check
# ---------------------------------------------------------------------------

TIER_OBSERVABILITY: dict[str, str] = {
    "ground_sentinel": (
        "Camera: RGB PTZ 4K (3840×2160), motorised pan-tilt-zoom. "
        "Deployment: fixed mountaintop or ridgeline position, 350–2000 m from "
        "typical detection range. "
        "Approximate pixel footprint at 1 km range: 25–40 cm per pixel at 1× zoom; "
        "4–6 cm per pixel at full zoom on a PTZ system. "
        "Temporal access: the camera covers the same terrain continuously; "
        "consecutive frames of the same area ARE available — multi-frame "
        "temporal features (drift direction, oscillation vs consistent motion) "
        "are OBSERVABLE on this tier. "
        "Limitations: backlit afternoon conditions degrade blue color detection; "
        "heat shimmer at very high temperatures (> 60°C ground surface) can "
        "create wavering artifacts indistinguishable from faint smoke at low zoom."
    ),
    "scout_drone": (
        "Camera: RGB 20MP (5472×3648) on a stabilised 2-axis gimbal. "
        "Operational altitude: 80–150 m AGL over terrain. "
        "Approximate pixel footprint: 2–5 cm per pixel at nadir. "
        "Flight pattern: single-pass patrol route; each terrain location is "
        "visited once per 12-minute repeat cycle. "
        "IMPORTANT: This tier processes single frames only. The drone passes "
        "over each location once before moving on; features requiring comparison "
        "across consecutive frames (drift direction, oscillation pattern) are "
        "NOT observable and must be reformulated as within-frame spatial proxies. "
        "LWIR sensor (640×480, uncalibrated) available but provides only relative "
        "thermal contrast — cannot distinguish 60°C hot rock from 280°C ignition."
    ),
    "commander_aircraft": (
        "Camera: RGB 60MP (9504×6336) on a stabilised 3-axis gimbal, plus "
        "calibrated MWIR sensor (640×512, 3–5 μm band, absolute temperature "
        "output in °C). "
        "Operational altitude: 500–1500 m AGL, long-loiter capable. "
        "Approximate RGB pixel footprint: 5–15 cm per pixel. "
        "Approximate MWIR pixel footprint: 80–240 cm per pixel. "
        "MWIR provides absolute temperature: ignition threshold 280°C is "
        "detectable at high confidence above background terrain. "
        "Temporal features available via 60-second onboard frame buffer. "
        "NOTE: Commander aircraft confirm ignitions via MWIR — the optical RGB "
        "rule is typically not needed at this tier. Rule adaptation for this "
        "tier is informational only."
    ),
}

# ---------------------------------------------------------------------------
# Confusable pairs
# ---------------------------------------------------------------------------

CONFUSABLE_PAIRS = [
    {
        "class_a": "early_smoke_signature",
        "class_b": "heat_shimmer_artifact",
        "description": (
            "Primary pair. Both present as faint translucent distortions above "
            "terrain surfaces in hot, dry conditions. "
            "Discriminating features: smoke has blue/blue-gray color, point source, "
            "asymmetric opacity gradient (dense at base, faint at dispersal). "
            "Shimmer is transparent, distributed across broad surface area, "
            "has no single origin point."
        ),
        "priority": "critical",
    },
    {
        "class_a": "early_smoke_signature",
        "class_b": "atmospheric_haze",
        "description": (
            "Distant atmospheric haze (pollution, marine layer intrusion). "
            "Discriminating features: atmospheric haze is uniform across the full "
            "horizon band with no local origin point; smoke has a concentrated "
            "source that fans outward. "
            "Most relevant at greater detection ranges (> 5 km) or coastal areas."
        ),
        "priority": "high",
    },
    {
        "class_a": "early_smoke_signature",
        "class_b": "dust_plume",
        "description": (
            "Dust raised by vehicles or wind on unpaved roads. "
            "Discriminating features: dust is tan/brown, rises from road "
            "or disturbed soil surface, typically has a linear trajectory "
            "following vehicle path; smoke is blue-gray and rises from a "
            "stationary point."
        ),
        "priority": "medium",
    },
    {
        "class_a": "early_smoke_signature",
        "class_b": "fog_patch",
        "description": (
            "Localised morning fog or marine layer remnant in a canyon or "
            "shaded area. Distinguishing features: fog is white/gray, "
            "fills topographic depressions uniformly, has no point source; "
            "smoke rises from a specific ignition point. "
            "Relevant in early morning or coastal terrain."
        ),
        "priority": "low",
    },
]

# ---------------------------------------------------------------------------
# Environmental context preconditions
# ---------------------------------------------------------------------------
# These are injected into accepted rules as runtime-evaluated preconditions
# checked against live RAWS weather station data. The same optical feature
# rule fires more aggressively under red flag conditions.
#
# The dict maps condition_set name → list of RAWS threshold strings.
# Thresholds are evaluated at rule inference time by get_raws_context().

RED_FLAG_PRECONDITIONS: dict[str, str] = {
    "raws_wind_speed_mph": "> 20",
    "raws_relative_humidity_pct": "< 15",
    "raws_temperature_f": "> 85",
    "fire_weather_watch": "true",
}

ELEVATED_RISK_PRECONDITIONS: dict[str, str] = {
    "raws_wind_speed_mph": "> 10",
    "raws_relative_humidity_pct": "< 25",
    "raws_temperature_f": "> 75",
}

# urgency_base scalar applied per condition set
CONDITION_SET_URGENCY: dict[str, float] = {
    "red_flag": 0.85,
    "elevated_risk": 0.60,
    "normal": 0.35,
}

# ---------------------------------------------------------------------------
# Cross-modal TUTOR prompt template (MWIR → optical RGB)
# ---------------------------------------------------------------------------

CROSS_MODAL_TUTOR_PROMPT = """
You are a {expert_role}.

The camera classifier assessed the image below as: **{pupil_classification}**
(confidence: {pupil_confidence:.2f}).

Ground truth confirmation from {confirmation_modality}: **{ground_truth_class}**
Confirmation details: {confirmation_details}

The image was captured by a {tier_description}.

Important constraint: describe only features that are VISIBLE IN THIS SINGLE
OPTICAL RGB IMAGE. The MWIR thermal reading confirms ground truth but is not
available to the classifier at inference time.

IMPORTANT for scout drone tier: This camera processes single frames only — it
cannot compare across frames to detect drift direction or temporal patterns.
Describe only features observable within this single image.

Your task: describe what features ARE VISIBLE IN THIS SINGLE OPTICAL IMAGE
that should have led to a correct classification of **{ground_truth_class}**.

Focus only on: smoke color, haze spatial distribution, presence of a point
source, opacity gradient from base to dispersal, contrast against background
terrain and sky.

Precondition quality rules — CRITICAL:
1. Write 3–4 preconditions maximum. Fewer, stronger preconditions beat many weak ones.
2. Every precondition must describe a POSITIVE feature that IS visibly present
   in the image. Do NOT write absence conditions ("lacks X", "no X", "without X",
   "does not have X") — a validator checking a new image cannot reliably confirm
   the absence of a feature.
3. Avoid ALL measurements (pixel size, distance, percentage). Use qualitative
   terms: "concentrated", "faint", "blue-gray", not "3% of frame area".
4. Describe features that are TRUE FOR THE CLASS in general, not just this one
   instance. This frame is an example; the preconditions must generalise.
5. Only include a precondition if you are CERTAIN a third-party observer could
   confirm it just by looking at this image. When in doubt, leave it out.
""".strip()
