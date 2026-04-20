"""Capture the initial frame of an ARC-AGI-3 game via the local SDK.

Writes a JSON file with the frame grid + all metadata the prompt needs.
Also renders an upscaled PNG for optional image-input experiments.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Ensure arc-agi-3 repo is importable.
ARC_REPO = Path(os.environ.get(
    "ARC_AGI_3_REPO", r"C:\_backup\github\arc-agi-3"
))
sys.path.insert(0, str(ARC_REPO))

from arc_agi import Arcade, OperationMode
from arc_agi.rendering import COLOR_MAP


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


PALETTE = {k: _hex_to_rgb(v) for k, v in COLOR_MAP.items()}


def render_png(grid: np.ndarray, out_path: Path, cell_px: int = 8) -> None:
    h, w = grid.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for k, c in PALETTE.items():
        rgb[grid == k] = c
    img = Image.fromarray(rgb, "RGB").resize(
        (w * cell_px, h * cell_px), Image.NEAREST
    )
    img.save(out_path)


def _normalise_frame(raw_frame) -> np.ndarray:
    """Real SDK hands list[ndarray]; extract the 2-D int grid."""
    if isinstance(raw_frame, list) and len(raw_frame) == 1:
        inner = raw_frame[0]
        if isinstance(inner, np.ndarray):
            return inner.astype(int)
        return np.array(inner, dtype=int)
    return np.array(raw_frame, dtype=int)


def capture(game_id: str, out_dir: Path) -> dict:
    arc = Arcade(
        operation_mode   = OperationMode.OFFLINE,
        environments_dir = str(ARC_REPO / "environment_files"),
    )
    env = arc.make(game_id)
    obs = env.reset()

    grid = _normalise_frame(obs.frame)
    action_labels = [f"ACTION{int(a)}" for a in obs.available_actions]

    info = env.environment_info
    info_d = info.model_dump() if hasattr(info, "model_dump") else dict(info.__dict__)

    payload = {
        "game_id":          str(obs.game_id),
        "title":            str(info_d.get("title")),
        "tags":             list(info_d.get("tags") or []),
        "state":            getattr(obs.state, "name", str(obs.state)),
        "levels_completed": int(obs.levels_completed),
        "win_levels":       int(obs.win_levels),
        "available_actions": action_labels,
        "frame_shape":      [int(grid.shape[0]), int(grid.shape[1])],
        "unique_colors":    sorted(int(x) for x in np.unique(grid).tolist()),
        "grid":             grid.tolist(),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{game_id}_L1_init.json"
    png_path  = out_dir / f"{game_id}_L1_init.png"
    json_path.write_text(json.dumps(payload, indent=2))
    render_png(grid, png_path)
    _write_index_html(out_dir, payload, png_path.name)
    print(f"wrote {json_path}")
    print(f"wrote {png_path}")
    return payload


def _write_index_html(out_dir: Path, payload: dict, png_name: str) -> None:
    """Write index.html so the preview panel displays a usable page."""
    legend_rows = "".join(
        f'<tr><td style="background:{COLOR_MAP[k]};width:24px"></td>'
        f'<td>{k}</td></tr>'
        for k in payload["unique_colors"]
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{payload['game_id']} L1 init</title>
<style>
body {{ font-family: system-ui, sans-serif; padding: 16px; background:#111; color:#eee; }}
h1 {{ font-size: 16px; margin: 0 0 8px; }}
.meta {{ font-size: 12px; color: #bbb; margin-bottom: 12px; }}
img {{ image-rendering: pixelated; border: 1px solid #444; }}
table {{ border-collapse: collapse; margin-top: 12px; font-size: 12px; }}
td {{ border: 1px solid #444; padding: 2px 6px; }}
</style></head><body>
<h1>{payload['game_id']} — level {payload['levels_completed']+1} / {payload['win_levels']}</h1>
<div class="meta">
  state={payload['state']} · available_actions={payload['available_actions']} ·
  palette={payload['unique_colors']} · shape={payload['frame_shape']}
</div>
<img src="{png_name}" width="512" height="512">
<table>
<tr><th>colour</th><th>palette id</th></tr>
{legend_rows}
</table>
</body></html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ls20-9607627b")
    ap.add_argument("--out-dir", default="../benchmarks/frames")
    a = ap.parse_args()
    capture(a.game, Path(a.out_dir))


if __name__ == "__main__":
    main()
