"""
domain_config.py — Road surface condition domain configuration for dialogic distillation.

Road surface classification for autonomous driving and ADAS applications.
The PUPIL must distinguish surface friction states (dry, wet, ice, snow),
materials (asphalt, concrete, gravel, dirt), and roughness levels from
forward-facing camera images — distinctions that are safety-critical
and require pavement engineering expertise to get right.
"""
from core.dialogic_distillation import DomainConfig

ROAD_SURFACE_CONFIG = DomainConfig(
    expert_role="senior pavement engineer and road safety specialist",
    item_noun="road surface image",
    item_noun_plural="road surface images",
    classification_noun="surface condition assessment",
    class_noun="condition class",
    feature_noun="surface condition indicator",
    observation_guidance=(
        "surface reflectivity and sheen pattern, texture granularity, "
        "color uniformity, presence of standing water or puddles, "
        "visible aggregate or binder, crack patterns and their geometry, "
        "surface deposit appearance (white/gray/translucent), "
        "edge sharpness of wet-dry boundaries, tire track visibility"
    ),
    non_visual_exclusions=(
        "ambient temperature, weather forecast, time of day, season, "
        "geographic location, recent maintenance history, traffic volume, "
        "de-icing treatment schedule"
    ),
    good_vocabulary_examples=[
        "uniform dark surface with mirror-like reflective sheen and no visible texture",
        "coarse gray aggregate clearly visible with dry matte finish between stones",
        "thin translucent film on surface with faint tire tracks still visible through it",
        "white crystalline deposit in a patchy pattern with exposed dark pavement between patches",
    ],
    bad_vocabulary_examples=[
        "black ice conditions present (engineering assessment, not a visual description)",
        "critically wet pavement near freezing point (requires temperature knowledge)",
        "alligator cracking indicating subbase failure (structural diagnosis, not visual)",
        "chemically wet surface from de-icing treatment (requires treatment history)",
    ],
    # Safety-critical domain: require higher precision and zero false positives.
    # A rule that misclassifies ice as wet (FP) in production could cause accidents.
    precision_gate=0.90,
    max_fp=0,
)
