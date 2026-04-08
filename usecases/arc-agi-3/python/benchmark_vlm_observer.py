"""
benchmark_vlm_observer.py — Compare an open VLM against the recorded Claude
Sonnet OBSERVER on real ARC-AGI-3 game frames.

Goal: answer two empirical questions before committing to an offline VLM stack.
  Q1. Does a strong open VLM (e.g. Qwen2.5-VL) produce OBSERVER-style JSON
      that is recognizably the same shape of analysis as Sonnet's?
  Q2. Does dual-input prompting (image + structured object list) close the
      gap vs. image-only prompting?

The script:
  1. Samples N playlog steps that have a stored `observer_analysis`.
  2. Renders each frame as an upscaled PNG with a thin grid overlay.
  3. Builds two prompts per frame: image-only and image+structured-objects.
  4. Calls a configurable VLM endpoint (Together.ai by default).
  5. Saves a side-by-side markdown report and the raw responses.

Usage (PowerShell / bash):
    python benchmark_vlm_observer.py --n 10 \
        --model "Qwen/Qwen2.5-VL-72B-Instruct" \
        --out ../.tmp/vlm_benchmark

Env vars:
    TOGETHER_API_KEY   required for the default backend
    VLM_BASE_URL       optional override (default: https://api.together.xyz/v1)

This script is intentionally self-contained — it does not import the existing
agents.py pipeline, so it can be run, edited, and deleted without touching
production code.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ----- Optional deps ---------------------------------------------------------
try:
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit("Pillow is required: pip install pillow")

try:
    from openai import AsyncOpenAI  # OpenAI-compatible client (works for Together)
except ImportError:
    sys.exit("openai client is required: pip install openai")


# ----- Constants -------------------------------------------------------------

# Standard ARC-AGI palette for color indices 0-9.
ARC_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (0,   0,   0),    # black
    1: (30,  147, 255),  # blue
    2: (249, 60,  49),   # red
    3: (79,  204, 48),   # green
    4: (255, 220, 0),    # yellow
    5: (153, 153, 153),  # grey
    6: (229, 58,  163),  # magenta
    7: (255, 133, 27),   # orange
    8: (135, 216, 241),  # azure
    9: (255, 255, 255),  # white
}

# Distinct fallback colors for game-specific extended indices 10-19.
EXT_PALETTE: dict[int, tuple[int, int, int]] = {
    10: (139, 69,  19),   11: (75,  0,   130),
    12: (255, 20,  147),  13: (0,   100, 0),
    14: (210, 180, 140),  15: (47,  79,  79),
    16: (218, 112, 214),  17: (240, 230, 140),
    18: (95,  158, 160),  19: (188, 143, 143),
}


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLAYLOG_DIR = REPO_ROOT / "usecases" / "arc-agi-3" / ".tmp" / "playlogs"
DEFAULT_OUT_DIR     = REPO_ROOT / "usecases" / "arc-agi-3" / ".tmp" / "vlm_benchmark"


# ----- Data ------------------------------------------------------------------

@dataclass
class Sample:
    episode: str
    step_file: Path
    frame: list[list[int]]
    sonnet_observer_text: str
    action_name: str
    step_number: int


# ----- Sampling --------------------------------------------------------------

def collect_samples(playlog_dir: Path, n: int, seed: int = 0) -> list[Sample]:
    """Pick N step JSONs that contain a non-empty observer_analysis."""
    rng = random.Random(seed)
    candidates: list[Path] = []
    for ep_dir in sorted(playlog_dir.iterdir()):
        if not ep_dir.is_dir():
            continue
        for f in ep_dir.glob("[0-9]*.json"):
            # Bias toward larger files — they tend to have full observer output.
            if f.stat().st_size > 5_000:
                candidates.append(f)
    rng.shuffle(candidates)

    samples: list[Sample] = []
    for path in candidates:
        if len(samples) >= n:
            break
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        observer = data.get("observer_analysis") or ""
        frame = (data.get("returned") or {}).get("frame")
        if not observer or not frame:
            continue
        if not isinstance(frame, list) or not frame or not isinstance(frame[0], list):
            continue
        samples.append(Sample(
            episode=path.parent.name,
            step_file=path,
            frame=frame,
            sonnet_observer_text=observer,
            action_name=data.get("action_name", "?"),
            step_number=int(data.get("step_number", 0)),
        ))
    return samples


# ----- Rendering -------------------------------------------------------------

def render_frame_png(frame: list[list[int]], cell_px: int = 16,
                     grid: bool = True) -> bytes:
    """Render a 2D color-index grid as a high-contrast PNG with grid overlay."""
    rows = len(frame)
    cols = len(frame[0])
    img = Image.new("RGB", (cols * cell_px, rows * cell_px), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for r in range(rows):
        for c in range(cols):
            v = int(frame[r][c])
            color = ARC_PALETTE.get(v) or EXT_PALETTE.get(v, (200, 0, 200))
            x0, y0 = c * cell_px, r * cell_px
            draw.rectangle([x0, y0, x0 + cell_px - 1, y0 + cell_px - 1], fill=color)
    if grid and cell_px >= 8:
        line = (60, 60, 60)
        for r in range(rows + 1):
            y = min(r * cell_px, rows * cell_px - 1)
            draw.line([(0, y), (cols * cell_px - 1, y)], fill=line, width=1)
        for c in range(cols + 1):
            x = min(c * cell_px, cols * cell_px - 1)
            draw.line([(x, 0), (x, rows * cell_px - 1)], fill=line, width=1)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ----- Lightweight object summary (no object_tracker dependency) -------------

def summarize_objects_quick(frame: list[list[int]]) -> str:
    """A cheap structural summary: per-color cell count, bbox, centroid.
    Used as the 'structured object list' half of the dual-input prompt.
    """
    rows = len(frame)
    cols = len(frame[0])
    stats: dict[int, dict[str, Any]] = {}
    for r in range(rows):
        for c in range(cols):
            v = int(frame[r][c])
            s = stats.setdefault(v, {"n": 0, "rmin": rows, "rmax": -1,
                                     "cmin": cols, "cmax": -1, "rs": 0, "cs": 0})
            s["n"] += 1
            if r < s["rmin"]: s["rmin"] = r
            if r > s["rmax"]: s["rmax"] = r
            if c < s["cmin"]: s["cmin"] = c
            if c > s["cmax"]: s["cmax"] = c
            s["rs"] += r
            s["cs"] += c
    # Background = most-frequent color.
    bg = max(stats.items(), key=lambda kv: kv[1]["n"])[0]
    lines = [f"Frame: {rows}x{cols}, background color={bg} ({stats[bg]['n']} cells)"]
    for v, s in sorted(stats.items(), key=lambda kv: -kv[1]["n"]):
        if v == bg:
            continue
        cy = s["rs"] / s["n"]
        cx = s["cs"] / s["n"]
        lines.append(
            f"- color {v}: {s['n']} cells, "
            f"bbox=(r{s['rmin']}-{s['rmax']}, c{s['cmin']}-{s['cmax']}), "
            f"centroid=({cy:.1f},{cx:.1f})"
        )
    return "\n".join(lines)


# ----- Prompt assembly -------------------------------------------------------

OBSERVER_SCHEMA_HINT = """Respond with a single JSON block (inside ```json fences) with this schema:
{
  "level_description": "one sentence",
  "visual_observations": ["...", "..."],
  "identified_objects": ["[GUESS|CONFIRMED] description", "..."],
  "concept_bindings": {
    "<color_value>": {"role": "player_piece|step_counter|goal_region|reference_pattern|wall|other",
                       "confidence": "high|medium|low", "label": "[GUESS]|[CONFIRMED]"}
  },
  "hypothesized_goal": "[GUESS] one sentence",
  "promising_actions": ["ACTION1", "..."],
  "reasoning": "brief"
}"""

SYSTEM_PROMPT = (
    "You are OBSERVER, the visual analyst for an interactive ARC-AGI-3 game. "
    "You receive a rendered game frame and must describe what you see, identify "
    "the agent and objects, and hypothesize the goal. Be concise and specific. "
    "Always label hypotheses as [GUESS] unless directly confirmed. "
    "Use generic role names. A wrong binding is worse than no binding.\n\n"
    + OBSERVER_SCHEMA_HINT
)


def build_user_blocks(image_b64: str, structured: str | None) -> list[dict]:
    text = (
        "## Current frame\n\n"
        "An ARC-AGI-3 game frame is shown in the image. Cells are colored using "
        "the standard ARC palette (0=black, 1=blue, 2=red, 3=green, 4=yellow, "
        "5=grey, 6=magenta, 7=orange, 8=azure, 9=white). Extended colors 10+ are "
        "game-specific and their roles are unknown until inferred from behavior.\n"
    )
    if structured:
        text += "\n## Mechanically detected objects\n\n" + structured + "\n"
    text += (
        "\n## Task\n\n"
        "Produce the OBSERVER JSON for this frame. Identify the likely agent, "
        "the likely goal region or reference pattern, and any structural landmarks."
    )
    return [
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        {"type": "text", "text": text},
    ]


# ----- VLM call --------------------------------------------------------------

async def call_vlm(client: AsyncOpenAI, model: str, blocks: list[dict],
                   max_tokens: int = 1024) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": blocks},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


# ----- Parsing & scoring -----------------------------------------------------

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def extract_json(text: str) -> dict | None:
    if not text:
        return None
    m = JSON_BLOCK_RE.search(text)
    raw = m.group(1) if m else text
    try:
        return json.loads(raw)
    except Exception:
        # Try to find first {...} balanced span
        start = raw.find("{")
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "{": depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start:i+1])
                    except Exception:
                        return None
        return None


def compare_outputs(baseline: dict | None, candidate: dict | None) -> dict:
    """Coarse comparison metrics. Not a quality judgement — just structure overlap."""
    if not baseline or not candidate:
        return {"valid_json": bool(candidate), "schema_fields": 0,
                "binding_overlap": 0.0, "binding_role_agreement": 0.0}
    expected_fields = {"level_description", "visual_observations",
                       "identified_objects", "concept_bindings",
                       "hypothesized_goal", "promising_actions"}
    have = expected_fields & set(candidate.keys())
    base_b = baseline.get("concept_bindings") or {}
    cand_b = candidate.get("concept_bindings") or {}
    base_keys = set(str(k) for k in base_b.keys())
    cand_keys = set(str(k) for k in cand_b.keys())
    overlap = (len(base_keys & cand_keys) / max(1, len(base_keys))) if base_keys else 0.0

    agree = 0
    common = base_keys & cand_keys
    for k in common:
        br = (base_b.get(k) or base_b.get(int(k) if k.isdigit() else k) or {})
        cr = cand_b.get(k) or {}
        if isinstance(br, dict) and isinstance(cr, dict):
            if (br.get("role") or "").lower() == (cr.get("role") or "").lower():
                agree += 1
    role_agreement = agree / max(1, len(common))

    return {
        "valid_json": True,
        "schema_fields": len(have),
        "binding_overlap": round(overlap, 3),
        "binding_role_agreement": round(role_agreement, 3),
    }


# ----- Main ------------------------------------------------------------------

async def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frames").mkdir(exist_ok=True)
    (out_dir / "responses").mkdir(exist_ok=True)

    samples = collect_samples(Path(args.playlogs), args.n, seed=args.seed)
    if not samples:
        sys.exit(f"No usable samples found in {args.playlogs}")
    print(f"Loaded {len(samples)} samples from {args.playlogs}", file=sys.stderr)

    base_url = os.environ.get("VLM_BASE_URL", "https://api.together.xyz/v1")
    api_key = os.environ.get("TOGETHER_API_KEY") or os.environ.get("VLM_API_KEY")
    if not api_key:
        sys.exit("Set TOGETHER_API_KEY (or VLM_API_KEY) before running.")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    report_lines: list[str] = [
        f"# VLM OBSERVER benchmark — {args.model}",
        f"Samples: {len(samples)}  •  base_url: `{base_url}`\n",
        "| # | episode | step | mode | json | fields | binding_overlap | role_agree |",
        "|---|---------|------|------|------|--------|-----------------|------------|",
    ]
    aggregate: dict[str, list[float]] = {"image_only": [], "dual_input": []}

    for i, s in enumerate(samples):
        png = render_frame_png(s.frame, cell_px=args.cell_px, grid=True)
        png_path = out_dir / "frames" / f"{i:02d}_{s.episode}_{s.step_number:04d}.png"
        png_path.write_bytes(png)
        b64 = base64.b64encode(png).decode("ascii")
        structured = summarize_objects_quick(s.frame)

        baseline_json = extract_json(s.sonnet_observer_text)

        for mode, blocks in [
            ("image_only", build_user_blocks(b64, None)),
            ("dual_input", build_user_blocks(b64, structured)),
        ]:
            try:
                text = await call_vlm(client, args.model, blocks, max_tokens=args.max_tokens)
            except Exception as e:
                text = f"[ERROR] {type(e).__name__}: {e}"
            (out_dir / "responses" / f"{i:02d}_{mode}.txt").write_text(text, encoding="utf-8")
            cand = extract_json(text)
            metrics = compare_outputs(baseline_json, cand)
            score = metrics["schema_fields"] / 6 * 0.4 + metrics["binding_overlap"] * 0.3 \
                    + metrics["binding_role_agreement"] * 0.3
            aggregate[mode].append(score)
            report_lines.append(
                f"| {i} | {s.episode} | {s.step_number} | {mode} | "
                f"{'Y' if metrics['valid_json'] else 'N'} | "
                f"{metrics['schema_fields']}/6 | {metrics['binding_overlap']} | "
                f"{metrics['binding_role_agreement']} |"
            )
            print(f"[{i:02d}/{mode}] score={score:.2f} {metrics}", file=sys.stderr)

        # Save baseline alongside for inspection
        (out_dir / "responses" / f"{i:02d}_sonnet_baseline.txt").write_text(
            s.sonnet_observer_text, encoding="utf-8")

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    report_lines += [
        "",
        "## Aggregate",
        f"- image_only mean composite score: **{avg(aggregate['image_only']):.3f}**",
        f"- dual_input mean composite score: **{avg(aggregate['dual_input']):.3f}**",
        "",
        "Composite = 0.4·schema + 0.3·binding_overlap + 0.3·role_agreement.",
        "This is a structural sanity check, not a quality judgement. Read the",
        "side-by-side text files in `responses/` for actual reasoning quality.",
    ]
    (out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nReport written to {out_dir / 'report.md'}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=10, help="Number of samples")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct",
                   help="VLM model id (default: Qwen/Qwen2.5-VL-72B-Instruct on Together)")
    p.add_argument("--playlogs", default=str(DEFAULT_PLAYLOG_DIR))
    p.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--cell-px", type=int, default=16,
                   help="Pixels per grid cell when rendering (default 16)")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
