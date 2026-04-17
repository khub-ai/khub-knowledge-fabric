"""
curate_references.py — Expert curation of canonical reference images per L1 group.

For each of the 5 Level-1 groups, presents N candidate training images to a VLM
(acting as a dermoscopy expert curator) and asks it to select the single most
visually prototypical and unambiguous representative.

The selected image becomes the canonical reference for that group in the L1
visual-similarity router (router.py).

Output:  curated_references.json  {group_name: absolute_image_path}

Usage:
    python curate_references.py                         # 6 candidates/group, qwen72b
    python curate_references.py --n-candidates 8        # more candidates per group
    python curate_references.py --model qwen/qwen2.5-vl-72b-instruct
    python curate_references.py --out my_refs.json
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
from pathlib import Path

# UTF-8 stdout on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_HERE    = Path(__file__).resolve().parent
_KF_ROOT = _HERE.parents[4]
_MC_PY   = _HERE.parents[1] / "dermatology-multiclass" / "python"
_D2_PY   = _HERE.parents[1] / "dermatology" / "python"

for _p in (str(_KF_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_D2_PY) not in sys.path:
    sys.path.append(str(_D2_PY))

import importlib.util as _ilu


def _load_mod(name: str, path: Path):
    """Load a Python module from an absolute path, registering it in sys.modules."""
    spec = _ilu.spec_from_file_location(name, path)
    mod  = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------
# Load the HIERARCHICAL dataset FIRST (before multiclass agents can
# register sys.modules["dataset"] as the multiclass version).
# We store it under a private name; attributes are accessed directly.
# ------------------------------------------------------------------
_h_ds_mod = _load_mod("_h_dataset_curation", _HERE / "dataset.py")
LEVEL1_GROUPS = _h_ds_mod.LEVEL1_GROUPS
_NAME_TO_DX   = _h_ds_mod.NAME_TO_DX   # same mapping in both datasets

# ------------------------------------------------------------------
# Now load multiclass dataset and agents (they may overwrite
# sys.modules["dataset"] but we've already captured what we need).
# ------------------------------------------------------------------
_mc_ds  = _load_mod("_mc_dataset_curation", _MC_PY / "dataset.py")
_mc_ags = _load_mod("_mc_agents_curation",  _MC_PY / "agents.py")

load_ham10000    = _mc_ds.load
DEFAULT_DATA_DIR = _mc_ds.DEFAULT_DATA_DIR


# ---------------------------------------------------------------------------
# Curator prompts
# ---------------------------------------------------------------------------

_CURATOR_SYSTEM = """\
You are a board-certified dermatologist and dermoscopy educator with 20+ years of experience.

Task: from a set of candidate dermoscopic images all belonging to the SAME lesion group,
select the SINGLE BEST canonical representative for that group.

Ideal canonical representative criteria (in order of priority):
  1. CLARITY     — the defining hallmark features of the group are clearly and prominently visible.
  2. PURITY      — the image could NOT be mistaken for any other group; no confounding overlap.
  3. IMAGE QUALITY — sharp focus, appropriate magnification, good colour balance.
  4. PROTOTYPICALITY — the image would be an excellent textbook/teaching example of the group.

Output ONLY a JSON object:
{
  "selected_index": 1,
  "reasoning": "What specific dermoscopic features make this the ideal canonical example?",
  "runner_up_index": 2,
  "disqualified": [3, 4]
}
(Index 1 = first candidate image shown, index 2 = second, etc.)
"""

_GROUP_DESCRIPTIONS: dict[str, str] = {
    "Melanocytic": (
        "Melanocytic group (contains Melanoma AND Melanocytic Nevus).\n"
        "The ideal representative shows organised MELANOCYTIC architecture:\n"
        "  • A clear pigment network (regular or irregular brown meshwork of lines)\n"
        "  • OR prominent dots / globules symmetrically or asymmetrically distributed\n"
        "  • The image should unmistakably scream 'pigmented melanocytic lesion'.\n"
        "Prefer an image WITHOUT blue-white veil or regression (those are fine for the group\n"
        "but a 'clean' melanocytic pattern is easier to recognise as the archetype)."
    ),
    "Keratosis-type": (
        "Keratosis-type group (contains Benign Keratosis AND Actinic Keratosis).\n"
        "The ideal representative shows unmistakable KERATIN surface structures:\n"
        "  • Milia-like cysts: bright white/yellowish sharply defined round dots\n"
        "  • Comedo-like openings: dark brown/black circular plugged pore-like depressions\n"
        "  • Cerebriform / fingerprint ridge-and-furrow surface texture\n"
        "  • A 'stuck-on' verrucous appearance\n"
        "Alternatively (for AK): clearly erythematous (pink/red) lesion background.\n"
        "AVOID images that could be mistaken for melanocytic or BCC."
    ),
    "Basal Cell Carcinoma": (
        "Basal Cell Carcinoma group (single class).\n"
        "The ideal representative shows unmistakable BCC features:\n"
        "  • Bright red ARBORISING (tree-like branching) vessels in sharp focus\n"
        "  • On a pearly white / translucent background\n"
        "  • Possibly blue-gray ovoid nests or leaf-like areas\n"
        "  • NO true pigment network\n"
        "This should be the clearest, most textbook-perfect BCC image you can find."
    ),
    "Vascular Lesion": (
        "Vascular Lesion group (Haemangioma / Angiokeratoma).\n"
        "The ideal representative shows unmistakable VASCULAR LACUNAE:\n"
        "  • Multiple sharply demarcated red, dark-red, or purple round/oval spaces\n"
        "  • Like blood-filled compartments or bubbles\n"
        "  • Overall reddish/purplish lesion colour\n"
        "  • NO pigment network\n"
        "AVOID images where the lacunae are subtle or could be confused with globules."
    ),
    "Dermatofibroma": (
        "Dermatofibroma group (single class).\n"
        "The ideal representative shows the classic DF pattern:\n"
        "  • Central WHITE or IVORY scar-like structureless patch (depressed/fibrous)\n"
        "  • Surrounded by a peripheral light-brown pigment ring or fine network\n"
        "  • Symmetric, benign overall appearance\n"
        "This should be immediately recognisable as DF, not confused with any other group."
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
    """Present candidates to the VLM expert and get the best representative.

    Returns the absolute path of the selected image.
    """
    call_agent   = _mc_ags.call_agent
    _image_block = _mc_ags._image_block
    _parse_json  = _mc_ags._parse_json_block

    desc = _GROUP_DESCRIPTIONS.get(group_name, group_name)

    # Build content blocks: header + numbered candidates
    content_blocks: list[dict] = [
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
        content_blocks.append({"type": "text", "text": f"\nCandidate {i} — {Path(path).name}:"})
        content_blocks.append(_image_block(path))

    content_blocks.append({
        "type": "text",
        "text": (
            "\nNow select the SINGLE BEST canonical representative for this group.\n"
            "Output JSON with 'selected_index', 'reasoning', 'runner_up_index', 'disqualified'."
        ),
    })

    text, ms = await call_agent(
        "CURATOR",
        content_blocks,
        system_prompt=_CURATOR_SYSTEM,
        max_tokens=768,
    )

    if verbose:
        print(f"  [curator] raw response ({ms}ms): {text[:300]!r}")

    # Parse response
    result = _parse_json(text)
    if result and "selected_index" in result:
        idx = int(result["selected_index"]) - 1
        if 0 <= idx < len(candidate_paths):
            selected = candidate_paths[idx]
            if verbose:
                print(f"  [curator] → Selected candidate {idx+1}: {Path(selected).name}")
                print(f"    Reasoning: {result.get('reasoning', '')[:200]}")
            return selected
        else:
            if verbose:
                print(f"  [curator] Index {idx+1} out of range ({len(candidate_paths)} candidates)")

    # Fallback: scan text for "Candidate N"
    for i, p in enumerate(candidate_paths, 1):
        for needle in [f"candidate {i}", f"image {i}", f"#{i}", f"option {i}"]:
            if needle in text.lower():
                if verbose:
                    print(f"  [curator] (text fallback) → candidate {i}: {Path(p).name}")
                return p

    # Last resort: first candidate
    if verbose:
        print(f"  [curator] PARSE FAILURE — defaulting to first candidate")
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
    """Curate canonical reference images for all 5 Level-1 groups.

    Returns {group_name: selected_image_path}.
    Also writes the result to out_path (JSON).
    """
    ds = load_ham10000(data_dir)
    curated: dict[str, str] = {}

    for group in LEVEL1_GROUPS:
        grp_name    = group["name"]
        grp_classes = group["classes"]

        print(f"\n{'='*60}")
        print(f"Curating group: {grp_name}")
        print(f"  Classes: {', '.join(grp_classes)}")

        # Sample candidates evenly from each class in the group
        n_per_cls = max(2, n_candidates // len(grp_classes))
        candidates: list[str] = []
        for cls_name in grp_classes:
            dx   = _NAME_TO_DX.get(cls_name, "")
            imgs = ds.sample_images(dx, n=n_per_cls, split="train", seed=seed)
            for img in imgs:
                if Path(img.file_path).exists():
                    candidates.append(str(img.file_path))
                else:
                    print(f"  [warn] missing file: {img.file_path}")

        # Trim to n_candidates
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
        print(f"  {grp:<28}: {Path(path).name}")

    return curated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Curate canonical reference images per Level-1 dermoscopic group"
    )
    p.add_argument("--data-dir",      default=str(DEFAULT_DATA_DIR),
                   help="Root directory of the HAM10000 dataset")
    p.add_argument("--n-candidates",  dest="n_candidates", type=int, default=6,
                   help="Number of candidate images to show the VLM per group (default: 6)")
    p.add_argument("--out",           default=str(_HERE / "curated_references.json"),
                   help="Output JSON path (default: curated_references.json)")
    p.add_argument("--model",         default="qwen/qwen2.5-vl-72b-instruct",
                   help="VLM model to use as expert curator")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--quiet",         action="store_true")
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_KF_ROOT / ".env", override=False)

    _mc_ags._set_active_model(args.model)
    _mc_ags._set_default_model(args.model)

    await run_curation(
        data_dir=args.data_dir,
        out_path=Path(args.out),
        n_candidates=args.n_candidates,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    asyncio.run(main())
