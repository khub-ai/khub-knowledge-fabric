"""
Domain configuration for the SeaPatch maritime SAR use case.

Cross-modal DD: TUTOR knowledge originates in thermal FLIR imaging (commander
tier); PUPIL classifies RGB sea-surface frames from scout drones.

Key adaptation over standard DD: temporal features (spatial stability across
frames) must be reformulated as within-frame proxies for single-frame
classifiers on the scout tier. See DESIGN.md §3.2.

See DESIGN.md §4 for the full domain configuration rationale.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from core.dialogic_distillation.protocols import DomainConfig

# ---------------------------------------------------------------------------
# Primary domain configuration
# ---------------------------------------------------------------------------

MARITIME_SAR_CONFIG = DomainConfig(
    expert_role=(
        "coast guard rescue swimmer and maritime search-and-rescue coordinator "
        "with operational experience in open-water person-recovery across "
        "multiple sea states"
    ),
    item_noun="UAV sea-surface surveillance frame",
    item_noun_plural="UAV sea-surface surveillance frames",
    classification_noun="person-in-water assessment",
    class_noun="presence class",
    feature_noun=(
        "optical sea-surface feature, floating silhouette characteristic, "
        "or water-disturbance pattern"
    ),
    observation_guidance=(
        "floating object silhouette geometry and bilateral symmetry, "
        "head-and-shoulder oval profile above the waterline, "
        "arm-induced V-shaped water disturbance flanking the floating object, "
        "brightness uniformity within a floating object compared with whitecap "
        "diffusion pattern, colour contrast between skin tone or life vest and "
        "surrounding water, object size relative to wave features, "
        "shadow geometry cast by an upright head at low sun angles"
    ),
    non_visual_exclusions=(
        "thermal infrared temperature readings (not available on scout tier), "
        "AIS transponder position data, acoustic pinger signals, GPS beacon "
        "coordinates, radio communication content, water temperature, "
        "time since person entered water, current or tide forecast data"
    ),
    good_vocabulary_examples=[
        (
            "oval region approximately 20–25 cm across, centred above the wave "
            "crests, with approximately uniform brightness across its full "
            "extent — unlike a whitecap, which is bright at the crest and "
            "fades radially to match surrounding water within the same "
            "object boundary"
        ),
        (
            "slight bilateral darkening visible at the 3 o'clock and 9 o'clock "
            "positions of the oval, just at the waterline — consistent with "
            "submerged arms pressing laterally outward and downward; absent in "
            "floating debris of comparable size"
        ),
        (
            "oval silhouette is geometrically stable: clean ellipse with "
            "aspect ratio approximately 1.5:1, maintained above the wave "
            "crests, with no irregular fringing or foam dispersal at the "
            "boundary — the irregular, asymmetric profile of whitecap foam "
            "is absent"
        ),
    ],
    bad_vocabulary_examples=[
        (
            "thermal camera shows a 37°C heat signature at this position "
            "(thermal sensor not present on scout tier)"
        ),
        (
            "AIS data confirms person entered water at these coordinates "
            "(AIS not available to scout classifier at inference time)"
        ),
        (
            "the oval region maintains its position across 15 consecutive "
            "frames (temporal feature — scout processes single frames only; "
            "reformulate as a within-frame proxy)"
        ),
        (
            "survivor confirmed by rescue team radio contact "
            "(external confirmation not available at inference time)"
        ),
        (
            "the oval is a human head "
            "(direct identification — must be derived from observable optical "
            "features, not stated as a premise)"
        ),
    ],
    precision_gate=0.90,   # safety-critical: false negatives cost lives
    max_fp=0,              # zero tolerance: every FP diverts rescue resources
)

# ---------------------------------------------------------------------------
# Tier-specific observability contexts for the KF grounding check
# ---------------------------------------------------------------------------
# Injected into the grounding check prompt to validate rules against actual
# sensor capabilities per fleet tier.
#
# NOTE: scouts are single-frame classifiers. Any temporal criterion must be
# reformulated as a within-frame proxy before it can be included in the
# scout-tier rule. The grounding check flags temporal features automatically;
# see DESIGN.md §3.2 for the reformulation protocol.

TIER_OBSERVABILITY: dict[str, str] = {
    "scout": (
        "Camera: 12MP RGB, fixed wide-angle mount, no stabilisation. "
        "Operational altitude: 20–40 m AGL. "
        "Approximate pixel footprint: 1–2 cm per pixel at nadir. "
        "Sea spray and wave motion introduce intermittent motion blur. "
        "Features requiring sub-10 cm resolution or precise colour calibration "
        "are NOT reliably observable. "
        "IMPORTANT: This tier processes single frames only. Features that "
        "require temporal comparison across multiple frames (e.g., stability, "
        "motion, change) are NOT observable and must be reformulated as "
        "within-frame proxies."
    ),
    "commander": (
        "Camera: 20MP RGB, stabilised 3-axis gimbal, plus FLIR Tau 2 thermal "
        "(336×256, 13 mm lens). "
        "Operational altitude: 10–20 m AGL with hover capability. "
        "Approximate RGB pixel footprint: 0.5–1 cm per pixel at nadir. "
        "Approximate thermal pixel footprint: 5–8 cm at operational altitude. "
        "Most floating object geometry, shadow details, and surface texture "
        "features are observable. "
        "Temporal features are available via a 30-second onboard frame buffer."
    ),
}

# ---------------------------------------------------------------------------
# Confusable pairs
# ---------------------------------------------------------------------------

CONFUSABLE_PAIRS = [
    {
        "class_a": "person_in_water",
        "class_b": "whitecap",
        "description": (
            "Primary pair. Both present as small pale ovals at 20–40 m altitude. "
            "Ground truth via thermal camera or manual recovery. "
            "Discriminating features: brightness uniformity, oval geometry "
            "regularity, flanking V-shaped disturbance."
        ),
        "priority": "critical",
    },
    {
        "class_a": "person_in_water",
        "class_b": "floating_debris",
        "description": (
            "Floating debris (packaging, clothing, wood) at similar size scale. "
            "Discriminating features: bilateral symmetry, oval silhouette vs. "
            "irregular debris outline, absence of V-shaped flanking pattern."
        ),
        "priority": "high",
    },
    {
        "class_a": "person_in_water",
        "class_b": "life_ring_unoccupied",
        "description": (
            "Life ring thrown overboard but not reached by the person. "
            "Discriminating features: life ring has a distinct torus geometry "
            "(bright ring, dark centre); person-in-water has solid oval with "
            "no central void."
        ),
        "priority": "medium",
    },
    {
        "class_a": "person_in_water",
        "class_b": "seabird_on_water",
        "description": (
            "At lower altitudes, seabirds sitting on the water present as small "
            "pale ovals. Discriminating features: seabirds show an elongated "
            "horizontal profile; person-in-water shows a near-vertical aspect "
            "ratio (head-up posture). Relevant primarily for commander tier."
        ),
        "priority": "low",
    },
]

# ---------------------------------------------------------------------------
# Cross-modal TUTOR prompt template
# ---------------------------------------------------------------------------
# Replaces the standard DD TUTOR prompt when the confirmation comes from a
# different sensor modality than the PUPIL's perception.
#
# Includes an explicit instruction to avoid temporal criteria and instead
# articulate within-frame proxies. This is injected directly; the grounding
# check then validates and may request further reformulation.

CROSS_MODAL_TUTOR_PROMPT = """
You are a {expert_role}.

The optical scout drone classified the image below as: **{pupil_classification}**
(confidence: {pupil_confidence:.2f}).

Ground truth confirmation from {confirmation_modality}: **{ground_truth_class}**
Confirmation details: {confirmation_details}

The optical image is shown below. The scout drone's camera is a 12MP RGB
sensor at 20–40 m altitude with no stabilisation. It cannot see thermal
signatures, AIS data, or any sensor output other than RGB colour.

Important constraint: the scout classifier processes single frames only — it
cannot compare across frames to detect stability or motion. Describe only
features observable within this single image.

Your task: describe what features are VISIBLE IN THIS SINGLE OPTICAL IMAGE
that should have led to a correct classification of **{ground_truth_class}**.

Focus only on: silhouette geometry, brightness distribution, bilateral symmetry,
colour contrast, shadow patterns, and within-frame water-disturbance geometry.
Do not reference thermal readings, AIS data, or any cross-frame temporal
comparison.

Precondition quality rules — CRITICAL:
1. Write 3–4 preconditions maximum. Fewer, stronger preconditions beat many weak ones.
2. Every precondition must describe a POSITIVE feature that IS visibly present
   in the image. Do NOT write absence conditions ("lacks X", "no X", "without X",
   "does not have X") — a validator checking a new image cannot reliably confirm
   the absence of a feature.
3. Avoid ALL measurements (pixel size, aspect ratio numbers, distances, angles).
   Use qualitative terms: "small", "compact", "roughly circular", not "1:1.5 ratio".
4. Describe features that are TRUE FOR THE CLASS in general, not just this one
   instance. This frame is an example; the preconditions must generalise.
5. Only include a precondition if you are CERTAIN a third-party observer could
   confirm it just by looking at this image. When in doubt, leave it out.
""".strip()
