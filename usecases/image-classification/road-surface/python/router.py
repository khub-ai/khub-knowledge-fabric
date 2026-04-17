"""
router.py — Visual similarity L1 group router for road surface classification.

Bypasses the schema-based Observer pipeline for coarse L1 routing.
Makes a single multi-image VLM call:

    test image  +  curated canonical references  →  predicted L1 group

The VLM is shown a prototypical canonical image per friction group
(pre-curated by curate_references.py) and asked which group the test image
most resembles.  No feature extraction schema is involved — pure visual
pattern matching against known exemplars.

Motivation (lesson from dermatology hierarchical experiments):
    The full KF schema pipeline fails at L1 coarse routing because VLMs
    report all discriminating features (surface reflectivity, aggregate
    visibility, texture presence) as ABSENT with high confidence, regardless
    of actual image content.  Visual similarity to a canonical reference
    bypasses this feature-denial problem: the model answers the simpler
    question "does this look like THAT?" rather than "is feature X present?".

Usage:
    Imported and called from a harness or evaluation script.
    curated_refs is loaded from curated_references.json produced by
    curate_references.py.

    from router import run_l1_router
    group, confidence, ms = await run_l1_router(test_path, curated_refs)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# L1 group definitions (must match curate_references.py)
# ---------------------------------------------------------------------------

L1_GROUP_NAMES = ["Dry", "Wet", "Water"]

# When ice / snow classes are available in the RSCD release, extend here:
# L1_GROUP_NAMES = ["Dry", "Wet", "Water", "Ice / Snow"]

_DEFAULT_CURATED_REFS_PATH = (
    Path(__file__).resolve().parent.parent / "curated_references.json"
)


# ---------------------------------------------------------------------------
# Router system prompt
# ---------------------------------------------------------------------------

_ROUTER_SYSTEM = """\
You are a senior pavement engineer and road safety specialist assessing
road surface friction states from vehicle camera images.

You will see:
  • TEST IMAGE   — the unknown road surface image to classify
  • REFERENCES   — one or more canonical exemplars per friction group.
                   Some groups may show 2 exemplars to cover visually
                   distinct subtypes. Match the test image to ANY
                   exemplar within the group.

Your task:
  Determine which L1 friction group the TEST IMAGE belongs to by comparing
  its overall surface appearance to the reference images.

Group overview (use this to interpret the references, not as a substitute
for them):
  Dry       — fully matte surface, no sheen, aggregate texture fully visible,
               uniform dark-gray color, no reflections.
  Wet       — thin moisture film causing subtle darkening; surface texture
               still partially visible through the film; possible faint
               specular sheen but NOT mirror-like; no standing puddles.
  Water     — standing water or deep continuous film; mirror-like sky
               reflection; surface texture completely hidden beneath water;
               distinct puddle boundaries or ripple patterns visible.
  (Ice/Snow — not in current RSCD release; reserved for future expansion)

Instructions:
  1. Study each REFERENCE image carefully to anchor the visual archetype
     for each group. When a group has 2 exemplars, the test image need
     only resemble EITHER one.
  2. Compare the TEST IMAGE holistically against all reference images.
  3. Focus on:
       — surface reflectivity (matte vs. sheen vs. mirror reflection)
       — texture visibility (aggregate clearly visible vs. partially
         visible through film vs. completely hidden)
       — color uniformity (even dark-gray vs. darker with wet patches
         vs. distinct bright reflective areas)
  4. Select the group whose exemplar(s) the test image most closely
     resembles. Do NOT rely on text descriptions alone.
  5. You MUST pick exactly one group name from the candidates list.

Output ONLY a JSON object:
{
  "predicted_group": "<exact group name>",
  "confidence": 0.0,
  "visual_match": "Which specific surface features match the chosen reference?",
  "reasoning": "Why does this group fit better than the runner-up?"
}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_agents():
    """Lazy-load agents from sys.modules (populated by the calling script)."""
    mod = sys.modules.get("agents")
    if mod is None:
        # Try direct import if called standalone
        _HERE = Path(__file__).resolve().parent
        if str(_HERE) not in sys.path:
            sys.path.insert(0, str(_HERE))
        import agents as _agents_mod
        return _agents_mod
    return mod


def load_curated_refs(path: Optional[str] = None) -> dict[str, str | list[str]]:
    """Load curated_references.json from disk.

    Returns {group_name: path_or_list_of_paths}.
    Raises FileNotFoundError with a helpful message if file is missing.
    """
    ref_path = Path(path) if path else _DEFAULT_CURATED_REFS_PATH
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Curated references not found at {ref_path}.\n"
            f"Run curate_references.py first:\n"
            f"  python curate_references.py --n-candidates 8"
        )
    return json.loads(ref_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_l1_router(
    test_image_path: str,
    curated_refs: dict[str, str | list[str]],
    verbose: bool = False,
) -> tuple[str, float, int]:
    """Route the test image to one of the L1 friction groups via visual similarity.

    Args:
        test_image_path: Absolute path to the test road surface image.
        curated_refs:    {group_name: path_or_list_of_paths}
                         Must include at least one entry; missing groups get
                         a warning but the call proceeds with available refs.
        verbose:         Print intermediate output if True.

    Returns:
        (predicted_group_name, confidence, duration_ms)
        predicted_group_name is always one of L1_GROUP_NAMES.
    """
    agents   = _get_agents()
    call_agent = agents.call_agent

    # Import image_block and parse_json_block from core via agents module
    # (road surface agents re-exports from dermatology which re-exports from core)
    try:
        from core.dialogic_distillation.agents import image_block as _image_block
        from core.dialogic_distillation.agents import parse_json_block as _parse_json
    except ImportError:
        # Fallback: try the dermatology agents directly
        _derm = agents._derm_agents
        _image_block = _derm._image_block
        _parse_json  = _derm._parse_json_block

    # Build content: test image followed by references per group
    content: list[dict] = [
        {"type": "text", "text": "TEST IMAGE (classify this road surface):"},
        _image_block(test_image_path),
    ]

    refs_shown = 0
    for grp_name in L1_GROUP_NAMES:
        raw = curated_refs.get(grp_name, "")
        paths = [raw] if isinstance(raw, str) else list(raw)
        valid = [p for p in paths if p and Path(p).exists()]

        if not valid:
            if verbose:
                print(f"  [router] WARNING: no valid reference for {grp_name!r}")
            continue

        if len(valid) == 1:
            content.append({"type": "text", "text": f"\nCANONICAL REFERENCE — {grp_name}:"})
            content.append(_image_block(valid[0]))
        else:
            content.append({
                "type": "text",
                "text": f"\nCANONICAL REFERENCES — {grp_name} ({len(valid)} exemplars):",
            })
            for k, p in enumerate(valid, 1):
                content.append({"type": "text", "text": f"  {grp_name} exemplar {k}:"})
                content.append(_image_block(p))

        refs_shown += 1

    if refs_shown == 0:
        raise ValueError("run_l1_router: curated_refs contains no valid image paths")

    content.append({
        "type": "text",
        "text": (
            f"\nGroup candidates (choose exactly one): {', '.join(L1_GROUP_NAMES)}\n\n"
            "Compare the TEST IMAGE to each CANONICAL REFERENCE shown above. "
            "Output JSON with 'predicted_group', 'confidence', "
            "'visual_match', 'reasoning'."
        ),
    })

    text, ms = await call_agent(
        "L1_ROUTER",
        content,
        system_prompt=_ROUTER_SYSTEM,
        max_tokens=512,
    )

    # Parse JSON response
    result = _parse_json(text)
    if result and "predicted_group" in result:
        grp  = result["predicted_group"].strip()
        conf = float(result.get("confidence", 0.5))

        # Exact match
        if grp in L1_GROUP_NAMES:
            if verbose:
                print(f"  [router] → {grp!r}  conf={conf:.2f}  ({ms}ms)")
                print(f"    match: {result.get('visual_match', '')[:120]}")
            return grp, conf, ms

        # Fuzzy match (case-insensitive substring)
        grp_lower = grp.lower()
        for gn in L1_GROUP_NAMES:
            if gn.lower() in grp_lower or grp_lower in gn.lower():
                if verbose:
                    print(f"  [router] (fuzzy) → {gn!r}  conf={conf:.2f}  ({ms}ms)")
                return gn, conf, ms

    # Fallback: scan raw text for any group name
    text_lower = text.lower()
    for gn in L1_GROUP_NAMES:
        if gn.lower() in text_lower:
            if verbose:
                print(f"  [router] (text-scan fallback) → {gn!r}  ({ms}ms)")
            return gn, 0.2, ms

    # Last resort
    if verbose:
        print(f"  [router] PARSE FAILURE — defaulting to {L1_GROUP_NAMES[0]!r}")
        print(f"    raw: {text[:300]!r}")
    return L1_GROUP_NAMES[0], 0.1, ms
