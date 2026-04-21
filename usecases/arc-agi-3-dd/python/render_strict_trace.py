"""Render a strict-mode session's training-data directory as a readable
HTML page.

Given a directory of turn_NNN.json records (the distillation corpus
format written by discovery_play.py), produce a single HTML file that
shows, for each turn:
  - The rendered frame (from frame_b64) at full visual size
  - TUTOR's rationale + hypotheses + predict + revise
  - The parsed command and target cell
  - Outcome: reached/not, agent_end_cell, lc change, frame delta summary
  - Per-turn cost + tokens

Plus a top-of-page summary from metadata.json: outcome, turns, total
cost, initial/final lc, action_effects, cell size.

Usage:
  python render_strict_trace.py <training_data_dir>            # explicit
  python render_strict_trace.py                                 # defaults to latest_strict

Writes:
  <dir>/trace.html  (stable filename next to the turn_NNN.json files)
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

DEFAULT_DIR = Path(r"C:\_backup\github\khub-knowledge-fabric\.tmp\training_data\ls20-9607627b\latest_strict")


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Strict-mode trace: {game_id} / {trial_id}</title>
<style>
  :root {{
    --bg: #0f1115; --card: #181b22; --muted: #8a93a6; --accent: #66d9ef;
    --good: #a0e57a; --bad: #ff6b6b; --warn: #ffd36b; --text: #e6e8ee;
  }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
          background: var(--bg); color: var(--text); margin: 0; padding: 24px; }}
  h1 {{ margin: 0 0 6px 0; font-size: 22px; }}
  .sub {{ color: var(--muted); font-size: 13px; margin-bottom: 16px; }}
  .summary {{ background: var(--card); padding: 14px 18px; border-radius: 8px;
              margin-bottom: 24px; display: grid;
              grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
              gap: 10px 22px; font-size: 13px; }}
  .summary .k {{ color: var(--muted); text-transform: uppercase;
                 letter-spacing: 0.04em; font-size: 11px; }}
  .summary .v {{ color: var(--text); font-variant-numeric: tabular-nums; }}
  .turn {{ background: var(--card); border-radius: 8px; padding: 18px;
           margin-bottom: 18px; display: grid;
           grid-template-columns: 520px 1fr; gap: 22px; }}
  .turn.win {{ border-left: 4px solid var(--good); padding-left: 14px; }}
  .turn.miss {{ border-left: 4px solid var(--warn); padding-left: 14px; }}
  .turn.error {{ border-left: 4px solid var(--bad); padding-left: 14px; }}
  .turn img {{ width: 100%; height: auto; display: block; image-rendering: pixelated;
               border-radius: 4px; }}
  .turn h3 {{ margin: 0 0 8px 0; font-size: 16px; }}
  .turn .tag {{ display: inline-block; padding: 2px 8px; border-radius: 10px;
                font-size: 11px; margin-right: 6px; background: #242833; }}
  .tag.good {{ background: rgba(160,229,122,0.2); color: var(--good); }}
  .tag.bad  {{ background: rgba(255,107,107,0.2); color: var(--bad); }}
  .tag.warn {{ background: rgba(255,211,107,0.2); color: var(--warn); }}
  .field {{ margin-top: 10px; font-size: 13px; }}
  .field .label {{ color: var(--muted); text-transform: uppercase;
                   font-size: 10px; letter-spacing: 0.06em; margin-bottom: 2px; }}
  pre {{ background: #0b0d11; padding: 8px; border-radius: 4px; overflow-x: auto;
         font-size: 12px; margin: 4px 0; white-space: pre-wrap; color: #d5d8e2; }}
  .kv {{ font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; }}
  .costbar {{ font-size: 11px; color: var(--muted); margin-top: 8px; }}
</style>
</head>
<body>
<h1>Strict-mode trace: {game_id}</h1>
<div class="sub">Trial: {trial_id} &middot; Rendered from {dir}</div>

<div class="summary">
  {summary_cells}
</div>

{turns_html}

</body>
</html>
"""


def _summary_cells(meta: dict) -> str:
    def kv(label, val):
        return (f'<div><div class="k">{html.escape(label)}</div>'
                f'<div class="v">{html.escape(str(val))}</div></div>')
    cells = []
    cells.append(kv("Outcome", meta.get("outcome", "?")))
    cells.append(kv("Final state", meta.get("final_state", "?")))
    cells.append(kv("Initial LC", meta.get("initial_lc", "?")))
    cells.append(kv("Final LC", meta.get("levels_completed", "?")))
    cells.append(kv("Turns", meta.get("turns", "?")))
    cells.append(kv("Advancing", meta.get("advancing_turns", "?")))
    cells.append(kv("Total cost",
                    f"${meta.get('total_cost_usd', 0.0):.4f}"))
    cells.append(kv("Mode", meta.get("mode", "?")))
    ae = meta.get("action_effects") or {}
    if ae:
        cells.append(kv("Action effects",
                        " ".join(f"{k}={tuple(v)}" for k, v in ae.items())))
    return "\n  ".join(cells)


def _turn_html(rec: dict) -> str:
    turn = rec.get("turn", "?")
    md = rec.get("metadata") or {}
    advanced = md.get("advanced_level", False)
    reached = md.get("target_reached", False)
    klass = "win" if advanced else ("miss" if reached else "error")

    tags = []
    if advanced:
        tags.append('<span class="tag good">LEVEL ADVANCED</span>')
    elif reached:
        tags.append('<span class="tag warn">reached target, no advance</span>')
    else:
        tags.append('<span class="tag bad">did not reach target</span>')

    state = md.get("state", "?")
    lc = md.get("levels_completed", "?")
    agent_cell = md.get("agent_pos")
    target_cell = md.get("target_cell") or md.get("parsed_target_cell")
    agent_end = md.get("agent_cell_end")
    diff_cells = md.get("frame_diff_cells", "-")
    delta = md.get("delta_summary") or {}

    # Parse TUTOR's assistant JSON for rationale/hypotheses/predict/revise.
    assistant_raw = rec.get("assistant", "") or ""
    try:
        import re
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", assistant_raw, re.DOTALL)
        jtext = fence.group(1) if fence else assistant_raw[assistant_raw.find("{"):assistant_raw.rfind("}")+1]
        cmd = json.loads(jtext)
    except Exception:
        cmd = {}
    rationale  = cmd.get("rationale", "")
    hypotheses = cmd.get("hypotheses", "")
    predict    = cmd.get("predict", "")
    revise     = cmd.get("revise", "")

    frame_b64 = rec.get("frame_b64") or ""
    img_tag = (f'<img src="data:image/png;base64,{frame_b64}" alt="frame turn {turn}">'
               if frame_b64 else '<div style="color:#888">(no frame image)</div>')

    def field(label, value):
        if value is None or value == "":
            return ""
        if isinstance(value, (dict, list)):
            value = json.dumps(value, indent=2)
        return (f'<div class="field"><div class="label">{html.escape(label)}</div>'
                f'<pre>{html.escape(str(value))}</pre></div>')

    body = []
    body.append(f'<div><h3>Turn {turn} &middot; agent={agent_cell} &rarr; target={target_cell}</h3>'
                f'<div>{" ".join(tags)}</div>'
                f'<div class="kv field">'
                f'state: <b>{html.escape(str(state))}</b> &middot; '
                f'lc={lc} &middot; '
                f'end_cell={agent_end} &middot; '
                f'frame_diff={diff_cells} &middot; '
                f'delta={json.dumps(delta)}'
                f'</div>')
    body.append(field("Rationale", rationale))
    body.append(field("Hypotheses", hypotheses))
    body.append(field("Predict", predict))
    body.append(field("Revise", revise))
    body.append(f'<div class="costbar">cost ${md.get("cost_usd", 0):.4f} &middot; '
                f'{md.get("input_tokens", 0)} in + {md.get("output_tokens", 0)} out tokens &middot; '
                f'{md.get("latency_ms", 0)} ms</div>')
    body.append('</div>')

    return (f'<div class="turn {klass}">'
            f'<div>{img_tag}</div>'
            f'<div>{"".join(body)}</div>'
            f'</div>')


def render(dir_path: Path) -> Path:
    meta_path = dir_path / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no metadata.json in {dir_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    turn_files = sorted(
        [p for p in dir_path.iterdir() if p.name.startswith("turn_") and p.suffix == ".json"],
        key=lambda p: int(p.stem.split("_")[1]),
    )
    turns_html = "\n".join(
        _turn_html(json.loads(p.read_text(encoding="utf-8"))) for p in turn_files
    )

    out = PAGE_TEMPLATE.format(
        game_id       = html.escape(str(meta.get("game_id", "?"))),
        trial_id      = html.escape(str(meta.get("trial_id", "?"))),
        dir           = html.escape(str(dir_path)),
        summary_cells = _summary_cells(meta),
        turns_html    = turns_html,
    )

    out_path = dir_path / "trace.html"
    out_path.write_text(out, encoding="utf-8")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default=str(DEFAULT_DIR),
                    help="Training-data directory containing turn_NNN.json "
                         "(default: latest_strict)")
    a = ap.parse_args()
    p = render(Path(a.dir))
    print(f"Wrote: {p}")
    print(f"Open:  file:///{p.as_posix()}")


if __name__ == "__main__":
    main()
