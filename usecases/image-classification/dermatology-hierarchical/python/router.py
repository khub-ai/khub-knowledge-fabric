"""
router.py — Direct visual-similarity Level-1 group router.

Bypasses the full Observer → Schema → Mediator → Verifier pipeline for L1
group routing.  Instead, makes a single multi-image VLM call:

    test image  +  5 curated canonical references  →  predicted group name

The VLM is shown one prototypical canonical image per group (pre-curated by an
expert) and asked which group the test image most resembles.  No feature
extraction schema is involved — pure visual pattern matching against known
exemplars.

Motivation:
    The full KF pipeline fails at L1 because Qwen2.5-VL (and other VLMs) report
    all discriminating features (pigment network, lacunae, arborizing vessels,
    milia cysts) as *absent* with high confidence, regardless of actual image
    content.  Visual similarity to a canonical reference bypasses this
    feature-denial problem: the model answers a simpler question ("does this
    look like THAT?") rather than the schema-filling question ("is feature X
    present?").
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from dataset import LEVEL1_GROUPS, LEVEL1_GROUP_NAMES   # hierarchical dataset


# ---------------------------------------------------------------------------
# Router system prompt
# ---------------------------------------------------------------------------

_ROUTER_SYSTEM = """\
You are a board-certified dermatologist performing coarse group classification
of dermoscopic skin lesion images.

You will see:
  • TEST IMAGE      — the unknown lesion to classify
  • REFERENCES      — one or more canonical dermoscopic exemplars per group.
                      Some groups show 2 exemplars because the group contains
                      subtypes with distinct visual patterns (e.g., warty vs.
                      erythematous keratosis).  Match the test image to ANY
                      exemplar within the group.

Your task:
  Determine which group the TEST IMAGE belongs to by comparing its overall
  dermoscopic pattern to the reference images.

Group overview (use this to interpret the references, not as a substitute for them):
  Melanocytic        — organised pigmented architecture: brown pigment network (meshwork of lines),
                       symmetric dots/globules; may show blue-white veil or regression in melanoma.
  Keratosis-type     — surface keratin structures: milia-like cysts (white round dots),
                       comedo-like openings (dark plugged pores), warty/stuck-on texture;
                       OR erythematous (pink/red) background with diffuse redness (actinic keratosis).
  Basal Cell Carcinoma — bright red arborising (tree-like branching) vessels on pearly/white
                       background; blue-gray ovoid nests; no true pigment network.
  Vascular Lesion    — sharply defined red or purple round/oval lacunae (blood-filled compartments);
                       overall reddish/purplish lesion colour; no pigment network.
  Dermatofibroma     — central white or ivory scar-like structureless patch surrounded by a
                       peripheral light-brown pigment ring; symmetric, benign appearance.

Instructions:
  1. Study each REFERENCE image carefully to anchor the visual archetype for each group.
     When a group has 2 exemplars, the test image need only resemble EITHER one.
  2. Compare the TEST IMAGE holistically against all reference images.
  3. Select the group whose exemplar(s) the test image most closely resembles
     in terms of colour palette, texture, structure, and dermoscopic pattern.
  4. Do NOT rely on the text descriptions alone — ground your answer in visual
     similarity to the actual reference images shown.
  5. You MUST pick exactly one of the five group names.

Output ONLY a JSON object:
{
  "predicted_group": "<exact group name — one of the 5 listed above>",
  "confidence": 0.0,
  "visual_match": "Which specific visual features of the test image match the chosen reference?",
  "reasoning": "Why does this group fit better than the runner-up?"
}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_agents():
    """Lazy-load the agents module that was registered by ensemble.py."""
    mod = sys.modules.get("agents")
    if mod is None:
        raise RuntimeError(
            "agents module not found in sys.modules — "
            "ensure ensemble.py was imported before calling run_l1_router"
        )
    return mod


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_l1_router(
    test_image_path: str,
    curated_refs: dict[str, str],
    verbose: bool = False,
) -> tuple[str, float, int]:
    """Route the test image to one of 5 Level-1 groups via visual similarity.

    Args:
        test_image_path: Absolute path to the test dermoscopic image.
        curated_refs:    {group_name: path_to_canonical_reference_image}
                         Must include at least one entry; missing groups get
                         a warning but the call proceeds with available refs.
        verbose:         Print intermediate output if True.

    Returns:
        (predicted_group_name, confidence, duration_ms)
        predicted_group_name is always one of LEVEL1_GROUP_NAMES.
    """
    agents       = _get_agents()
    call_agent   = agents.call_agent
    _image_block = agents._image_block
    _parse_json  = agents._parse_json_block

    # Build the content block list: test image then references per group.
    # curated_refs values may be a string (single path) or a list of paths
    # (multiple canonical examples for groups with visually diverse subtypes).
    content_blocks: list[dict] = [
        {"type": "text", "text": "TEST IMAGE (classify this lesion):"},
        _image_block(test_image_path),
    ]

    refs_shown = 0
    for grp_name in LEVEL1_GROUP_NAMES:
        raw = curated_refs.get(grp_name, "")
        # Normalise to a list of paths
        if isinstance(raw, str):
            paths = [raw] if raw else []
        else:
            paths = [p for p in raw if p]

        valid_paths = [p for p in paths if Path(p).exists()]
        if not valid_paths:
            if verbose:
                print(f"  [router] WARNING: no valid reference for {grp_name!r}")
            continue

        if len(valid_paths) == 1:
            content_blocks.append(
                {"type": "text", "text": f"\nCANONICAL REFERENCE — {grp_name}:"}
            )
            content_blocks.append(_image_block(valid_paths[0]))
        else:
            # Multiple exemplars for a diverse group (e.g., Keratosis-type: BKL + AK)
            content_blocks.append(
                {"type": "text",
                 "text": f"\nCANONICAL REFERENCES — {grp_name} ({len(valid_paths)} exemplars):"}
            )
            for k, p in enumerate(valid_paths, 1):
                content_blocks.append(
                    {"type": "text", "text": f"  {grp_name} exemplar {k}:"}
                )
                content_blocks.append(_image_block(p))
        refs_shown += 1

        if verbose and len(valid_paths) > len(paths):
            missing = len(paths) - len(valid_paths)
            print(f"  [router] {grp_name}: {missing} reference path(s) not found on disk")

    if refs_shown == 0:
        raise ValueError("run_l1_router: curated_refs contains no valid image paths")

    content_blocks.append({
        "type": "text",
        "text": (
            f"\nGroup candidates (choose exactly one): {', '.join(LEVEL1_GROUP_NAMES)}\n\n"
            "Compare the TEST IMAGE to each CANONICAL REFERENCE shown above. "
            "Output JSON with 'predicted_group', 'confidence', 'visual_match', 'reasoning'."
        ),
    })

    text, ms = await call_agent(
        "L1_ROUTER",
        content_blocks,
        system_prompt=_ROUTER_SYSTEM,
        max_tokens=512,
    )

    # Parse response
    result = _parse_json(text)
    if result and "predicted_group" in result:
        grp  = result["predicted_group"].strip()
        conf = float(result.get("confidence", 0.5))

        # Exact match first
        if grp in LEVEL1_GROUP_NAMES:
            if verbose:
                print(f"  [router] → {grp!r}  conf={conf:.2f}  ({ms}ms)")
                print(f"    match: {result.get('visual_match', '')[:120]}")
            return grp, conf, ms

        # Fuzzy match (case-insensitive substring)
        grp_lower = grp.lower()
        for gn in LEVEL1_GROUP_NAMES:
            if gn.lower() in grp_lower or grp_lower in gn.lower():
                if verbose:
                    print(f"  [router] (fuzzy) → {gn!r}  conf={conf:.2f}  ({ms}ms)")
                return gn, conf, ms

    # Fallback: scan raw text for any group name
    text_lower = text.lower()
    for gn in LEVEL1_GROUP_NAMES:
        if gn.lower() in text_lower:
            if verbose:
                print(f"  [router] (text-scan fallback) → {gn!r}  ({ms}ms)")
            return gn, 0.2, ms

    # Last resort
    if verbose:
        print(f"  [router] PARSE FAILURE — defaulting to {LEVEL1_GROUP_NAMES[0]!r}")
        print(f"    raw: {text[:300]!r}")
    return LEVEL1_GROUP_NAMES[0], 0.1, ms
