"""
curate_references.py — Expert curation of canonical reference images per L1 group.

Presents N candidate RSCD training images to a VLM acting as a senior pavement
engineer and asks it to select the single most visually prototypical,
unambiguous representative for each L1 friction group.

The selected image becomes the canonical reference for the group in the L1
visual-similarity router (router.py).  The router bypasses the schema-based
Observer pipeline for coarse routing — a key lesson from the dermatology
hierarchical experiments where VLMs systematically denied seeing discriminating
features when asked directly.

Output: curated_references.json  {group_name: absolute_image_path}

Usage:
    python curate_references.py                          # 6 candidates/group
    python curate_references.py --n-candidates 10
    python curate_references.py --out my_refs.json
    python curate_references.py --model claude-opus-4-6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

_HERE    = Path(__file__).resolve().parent
_KF_ROOT = _HERE.parents[4]
if str(_KF_ROOT) not in sys.path:
    sys.path.insert(0, str(_KF_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=_KF_ROOT / ".env", override=False)

# Load API keys from local file if present
_ACCESS_KEYS = Path("P:/_access/Security/api_keys.env")
if _ACCESS_KEYS.exists():
    for _line in _ACCESS_KEYS.read_text().splitlines():
        if "=" in _line:
            _k, _v = _line.split("=", 1)
            _k, _v = _k.strip(), _v.strip()
            if _k in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY") and not os.environ.get(_k):
                os.environ[_k] = _v

import agents
from dataset import load as load_rscd, DEFAULT_DATA_DIR

_TMP_DIR = (_HERE / ".." / ".." / ".." / ".." / ".tmp" / "rscd_curate").resolve()
_TMP_DIR.mkdir(parents=True, exist_ok=True)

_DEFAULT_OUT = _HERE.parent / "curated_references.json"


# ---------------------------------------------------------------------------
# L1 friction groups
# ---------------------------------------------------------------------------

# Three L1 groups covering all RSCD friction classes in this release.
# Ice / snow / melted-snow are placeholders for when those classes become
# available in the dataset.
L1_GROUPS = [
    {
        "name":    "Dry",
        "classes": ["dry"],   # RSCD friction labels
    },
    {
        "name":    "Wet",
        "classes": ["wet"],
    },
    {
        "name":    "Water",
        "classes": ["water"],
    },
    # Uncomment when ice/snow classes are available in the RSCD release:
    # {
    #     "name":    "Ice / Snow",
    #     "classes": ["ice", "snow", "melted_snow"],
    # },
]

L1_GROUP_NAMES = [g["name"] for g in L1_GROUPS]


# ---------------------------------------------------------------------------
# Curator system prompt
# ---------------------------------------------------------------------------

_CURATOR_SYSTEM = """\
You are a senior pavement engineer and road safety specialist with 20+ years
of experience assessing road surface conditions from vehicle camera imagery.

Task: from a set of candidate road surface images all belonging to the SAME
friction class, select the SINGLE BEST canonical representative.

Ideal canonical representative criteria (in order of priority):
  1. CLARITY     — the defining hallmark features of the friction class are
                   prominently visible and unmistakable.
  2. PURITY      — the image could NOT be mistaken for any adjacent friction
                   class; no borderline cases.
  3. IMAGE QUALITY — good exposure, typical vehicle camera perspective,
                   representative lighting conditions.
  4. PROTOTYPICALITY — the image would be an excellent training or reference
                   example of the friction class.

Output ONLY a JSON object:
{
  "selected_index": 1,
  "reasoning": "What specific visual features make this the ideal canonical example?",
  "runner_up_index": 2,
  "disqualified": [3, 4]
}
(Index 1 = first candidate image shown, index 2 = second, etc.)
"""

_GROUP_DESCRIPTIONS: dict[str, str] = {
    "Dry": (
        "Dry road surface — NO moisture present.\n"
        "The ideal representative shows:\n"
        "  • Clearly MATTE surface finish — no sheen, no specular highlights\n"
        "  • Surface texture (aggregate stones, asphalt grain, pavement markings)"
        " FULLY visible with high contrast\n"
        "  • Uniform dark-gray or black color throughout\n"
        "  • Sharp, clear visibility of any road markings or lane lines\n"
        "CRITICAL: The selected image must be unambiguously dry — no subtle"
        " darkening, no hint of moisture at edges or in shadows."
    ),
    "Wet": (
        "Wet road surface — THIN moisture FILM present, but NOT standing water.\n"
        "The ideal representative shows:\n"
        "  • Subtle overall darkening of the surface compared to dry\n"
        "  • Surface texture (aggregate, grain) still PARTIALLY VISIBLE through"
        " the thin film\n"
        "  • Possible faint specular sheen in well-lit areas, but NOT a full"
        " mirror reflection\n"
        "  • Tire tracks may be visible as slightly lighter trails through the"
        " moisture\n"
        "  • No distinct puddles or pooled water — moisture is a film, not"
        " accumulated water\n"
        "CRITICAL: The selected image must be clearly wetter than dry but clearly"
        " NOT standing water. The texture must still be partially visible."
    ),
    "Water": (
        "Water — STANDING WATER or continuous water film deep enough to cover"
        " surface texture.\n"
        "The ideal representative shows:\n"
        "  • MIRROR-LIKE reflective surface — sky, surroundings, or headlights"
        " visibly reflected\n"
        "  • Surface texture (aggregate, grain) COMPLETELY HIDDEN beneath the"
        " water\n"
        "  • Possible distinct puddle boundaries (wet-to-standing-water edge)\n"
        "  • Strong, high-contrast specular highlights\n"
        "  • Possible ripple patterns or distortion in the reflection\n"
        "CRITICAL: The selected image must show unambiguous standing water, not"
        " merely wet pavement. The reflection of sky or surroundings should be"
        " clearly visible."
    ),
}


# ---------------------------------------------------------------------------
# Curation of a single group
# ---------------------------------------------------------------------------

async def curate_group(
    group_name: str,
    candidate_paths: list[str],
    verbose: bool = True,
) -> str:
    """Present candidates to the VLM curator and return the selected path."""
    from core.dialogic_distillation.agents import image_block, parse_json_block

    desc = _GROUP_DESCRIPTIONS.get(group_name, group_name)

    content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"GROUP: {group_name}\n"
                f"Number of candidates: {len(candidate_paths)}\n\n"
                f"Group description:\n{desc}\n\n"
                "Study each candidate image below:"
            ),
        }
    ]
    for i, path in enumerate(candidate_paths, 1):
        content.append({"type": "text", "text": f"\nCandidate {i} — {Path(path).name}:"})
        content.append(image_block(path))

    content.append({
        "type": "text",
        "text": (
            "\nNow select the SINGLE BEST canonical representative for this group.\n"
            "Output JSON with 'selected_index', 'reasoning', "
            "'runner_up_index', 'disqualified'."
        ),
    })

    text, ms = await agents.call_agent(
        "CURATOR",
        content,
        system_prompt=_CURATOR_SYSTEM,
        max_tokens=768,
    )

    if verbose:
        print(f"  [curator] raw response ({ms}ms): {text[:300]!r}")

    result = parse_json_block(text)
    if result and "selected_index" in result:
        idx = int(result["selected_index"]) - 1
        if 0 <= idx < len(candidate_paths):
            selected = candidate_paths[idx]
            if verbose:
                print(f"  [curator] → Selected candidate {idx+1}: {Path(selected).name}")
                print(f"    Reasoning: {result.get('reasoning', '')[:200]}")
            return selected
        elif verbose:
            print(f"  [curator] Index {idx+1} out of range ({len(candidate_paths)} candidates)")

    # Fallback: text scan for "Candidate N"
    for i, p in enumerate(candidate_paths, 1):
        for needle in [f"candidate {i}", f"image {i}", f"#{i}", f"option {i}"]:
            if needle in text.lower():
                if verbose:
                    print(f"  [curator] (text fallback) → candidate {i}: {Path(p).name}")
                return p

    if verbose:
        print("  [curator] PARSE FAILURE — defaulting to first candidate")
    return candidate_paths[0]


# ---------------------------------------------------------------------------
# Full curation run
# ---------------------------------------------------------------------------

async def run_curation(
    data_dir: str,
    out_path: Path,
    n_candidates: int = 6,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, str]:
    """Curate canonical reference images for all L1 groups.

    Returns {group_name: selected_image_path}.
    Writes the result to out_path (JSON).
    """
    ds = load_rscd(data_dir)
    curated: dict[str, str] = {}

    for group in L1_GROUPS:
        grp_name    = group["name"]
        grp_classes = group["classes"]

        print(f"\n{'='*60}")
        print(f"Curating group: {grp_name}")
        print(f"  Friction classes: {', '.join(grp_classes)}")

        # Sample candidates across all friction classes in the group
        n_per_cls = max(2, n_candidates // len(grp_classes))
        candidates: list[str] = []
        for friction in grp_classes:
            imgs = ds.sample_images(friction, n=n_per_cls, split="train", seed=seed)
            for img in imgs:
                path = img.resolve_path(_TMP_DIR)
                candidates.append(str(path))

        candidates = candidates[:n_candidates]
        if not candidates:
            print(f"  [warn] No valid candidates for {grp_name} — skipping")
            continue

        print(f"  Candidate count: {len(candidates)}")
        for i, p in enumerate(candidates, 1):
            print(f"    [{i}] {Path(p).name}")

        selected = await curate_group(grp_name, candidates, verbose=verbose)
        curated[grp_name] = selected
        print(f"  SELECTED: {selected}")

    # Persist
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(curated, indent=2), encoding="utf-8")
    tmp.replace(out_path)

    print(f"\n{'='*60}")
    print(f"Curated references saved to: {out_path}")
    for grp, path in curated.items():
        print(f"  {grp:<20}: {Path(path).name}")

    return curated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Curate canonical reference images per L1 road surface group"
    )
    p.add_argument("--data-dir",     default=str(DEFAULT_DATA_DIR))
    p.add_argument("--n-candidates", dest="n_candidates", type=int, default=6)
    p.add_argument("--out",          default=str(_DEFAULT_OUT))
    p.add_argument("--model",        default="claude-opus-4-6")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--quiet",        action="store_true")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    agents._derm_agents._set_active_model(args.model)
    agents._derm_agents._set_default_model(args.model)

    await run_curation(
        data_dir=args.data_dir,
        out_path=Path(args.out),
        n_candidates=args.n_candidates,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    asyncio.run(main())
