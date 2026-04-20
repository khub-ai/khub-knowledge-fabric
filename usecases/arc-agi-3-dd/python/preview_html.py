"""Render a single index.html for the preview panel (port 8753).

Shows the current frame image plus the latest TUTOR ↔ harness exchanges
(Round 1 probes + observations, and Round 2 revision notes if present).

Invoked from run_trial.py and run_round2.py after a session writes its
artefacts.  Overwrites frames/index.html each time.
"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path


def _json_pretty(obj) -> str:
    return escape(json.dumps(obj, indent=2, ensure_ascii=False))


def _render_round1_exchanges(session_dir: Path) -> str:
    """Turn tutor_reply + tutor_probe_results into a compact HTML section."""
    try:
        tutor_reply = json.loads((session_dir / "tutor_reply.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ""
    try:
        probe_results = json.loads((session_dir / "tutor_probe_results.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        probe_results = []

    assessment = (tutor_reply or {}).get("assessment") or {}
    probes = assessment.get("probes", []) or []
    probe_results_by_id = {pr.get("probe_id"): pr for pr in probe_results}

    parts: list[str] = []
    parts.append('<div class="section">')
    parts.append(f'<h2>Round 1 — TUTOR ↔ harness exchanges '
                 f'<span class="sub">({session_dir.name})</span></h2>')
    parts.append('<div class="exchanges">')
    for p in probes:
        pid = p.get("probe_id", "?")
        exec_r = probe_results_by_id.get(pid) or {}
        parts.append('<div class="exchange">')
        parts.append(f'<div class="exchange-header">{escape(pid)} — '
                     f'<span class="hypo">{escape(str(p.get("hypothesis","?")))}</span></div>')

        # TUTOR directive
        parts.append('<div class="row"><div class="who tutor">TUTOR</div>'
                     '<div class="msg">')
        instr = p.get("instructions") or []
        obs   = p.get("observe") or []
        parts.append('<div class="label">instructions</div>')
        parts.append('<pre>' + escape("\n".join(instr)) + '</pre>')
        parts.append('<div class="label">observe</div>')
        parts.append('<pre>' + escape("\n".join(obs)) + '</pre>')
        outcome = p.get("outcome_map") or {}
        if outcome:
            parts.append('<div class="label">outcome_map</div>')
            parts.append('<pre>' + _json_pretty(outcome) + '</pre>')
        parts.append('</div></div>')  # end row

        # Harness reply
        parts.append('<div class="row"><div class="who harness">HARNESS</div>'
                     '<div class="msg">')
        if exec_r.get("executed"):
            parts.append('<div class="label">trace</div>')
            parts.append('<pre>' + escape(
                "\n".join(t.get("instr", "") +
                          (f"  → {t['state_after']}" if "state_after" in t else "")
                          for t in exec_r.get("trace", []))
            ) + '</pre>')
            parts.append('<div class="label">observations</div>')
            parts.append('<pre>' + _json_pretty(exec_r.get("observations", [])) + '</pre>')
            parts.append(f'<div class="label">final_state</div>'
                         f'<pre>{escape(str(exec_r.get("final_state","?")))}</pre>')
        else:
            reason = exec_r.get("parse_errors") or exec_r.get("exec_error") or "(not executed — no record)"
            parts.append('<div class="label error">not executed</div>')
            parts.append('<pre>' + escape(str(reason)) + '</pre>')
        parts.append('</div></div>')  # end row

        parts.append('</div>')  # end exchange
    parts.append('</div></div>')  # end exchanges / section
    return "\n".join(parts)


def _render_round2_notes(round2_dir: Path) -> str:
    try:
        tutor_r2 = json.loads((round2_dir / "tutor_round2_reply.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ""
    assess = (tutor_r2 or {}).get("assessment") or {}
    notes  = assess.get("revision_notes") or []
    variant = "?"
    try:
        manifest = json.loads((round2_dir / "manifest.json").read_text(encoding="utf-8"))
        variant = manifest.get("variant", "?")
    except FileNotFoundError:
        pass
    if not notes and not assess.get("no_changes_reason"):
        return ""

    parts = ['<div class="section">']
    parts.append(f'<h2>Round {escape(variant)} — TUTOR revision notes '
                 f'<span class="sub">({round2_dir.name})</span></h2>')
    parts.append('<ul class="revnotes">')
    if assess.get("no_changes_reason"):
        parts.append(f'<li><em>no_changes_reason:</em> '
                     f'{escape(assess.get("no_changes_reason",""))}</li>')
    for n in notes:
        if isinstance(n, dict):
            parts.append(
                f'<li><b>{escape(str(n.get("round1_ref","?")))}</b>: '
                f'{escape(str(n.get("change","?")))}'
                f'<br><span class="reason">because: '
                f'{escape(str(n.get("reason","?")))}</span></li>'
            )
        else:
            parts.append(f'<li>{escape(str(n))}</li>')
    parts.append('</ul></div>')
    return "\n".join(parts)


STYLE = """
body { font-family: system-ui, sans-serif; padding: 16px; background:#111; color:#eee; max-width: 1100px; }
h1 { font-size: 16px; margin: 0 0 8px; }
h2 { font-size: 14px; margin: 24px 0 8px; border-bottom: 1px solid #333; padding-bottom: 4px; }
h2 .sub { color: #888; font-weight: normal; font-size: 11px; margin-left: 8px; }
.meta { font-size: 12px; color: #bbb; margin-bottom: 12px; }
img { image-rendering: pixelated; border: 1px solid #444; display: block; }
table.palette { border-collapse: collapse; margin-top: 12px; font-size: 12px; }
table.palette td { border: 1px solid #444; padding: 2px 6px; }
.section { margin-top: 16px; }
.exchange { border: 1px solid #2a2a2a; border-radius: 4px; margin: 12px 0; padding: 8px; background: #181818; }
.exchange-header { font-size: 12px; font-weight: bold; color: #ddd; margin-bottom: 6px; }
.exchange-header .hypo { color: #bbb; font-weight: normal; }
.row { display: flex; gap: 8px; margin: 4px 0; }
.who { flex: 0 0 68px; font-size: 11px; font-weight: bold; padding: 2px 4px; border-radius: 3px; text-align: center; height: fit-content; }
.who.tutor   { background:#264653; color:#fff; }
.who.harness { background:#2a9d8f; color:#fff; }
.msg { flex: 1; background: #0e0e0e; padding: 6px; border-radius: 3px; font-size: 11px; }
.label { color: #888; font-size: 10px; text-transform: uppercase; margin-top: 4px; }
.label.error { color: #ff8b6d; }
pre { margin: 2px 0 6px; white-space: pre-wrap; word-break: break-word; background: transparent; color: #ddd; font-size: 11px; }
.revnotes { font-size: 12px; color: #ddd; }
.revnotes li { margin: 6px 0; }
.revnotes .reason { color: #aaa; font-size: 11px; }
"""


def render_index(
    *,
    frames_dir:      Path,
    png_name:        str,
    frame_payload:   dict,
    round1_session:  Path | None = None,
    round2_session:  Path | None = None,
) -> None:
    """Write index.html with frame + any available exchanges and revision notes."""
    from arc_agi.rendering import COLOR_MAP
    legend_rows = "".join(
        f'<tr><td style="background:{COLOR_MAP[k]};width:24px"></td>'
        f'<td>{k}</td></tr>'
        for k in frame_payload["unique_colors"]
    )
    frame_header = (
        f"<h1>{escape(str(frame_payload['game_id']))} — level "
        f"{int(frame_payload['levels_completed'])+1} / "
        f"{int(frame_payload['win_levels'])}</h1>"
        f'<div class="meta">'
        f"state={escape(str(frame_payload['state']))} · "
        f"available_actions={escape(str(frame_payload['available_actions']))} · "
        f"palette={escape(str(frame_payload['unique_colors']))} · "
        f"shape={escape(str(frame_payload['frame_shape']))}"
        "</div>"
    )

    exchanges_html = ""
    if round1_session is not None:
        exchanges_html += _render_round1_exchanges(round1_session)
    revnotes_html = ""
    if round2_session is not None:
        revnotes_html += _render_round2_notes(round2_session)

    palette_table = (
        '<div class="section"><h2>Palette</h2>'
        '<table class="palette">'
        '<tr><th>colour</th><th>palette id</th></tr>'
        f'{legend_rows}'
        '</table></div>'
    )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>{escape(str(frame_payload['game_id']))} preview</title>
<style>{STYLE}</style>
</head><body>
{frame_header}
<img src="{escape(png_name)}" width="512" height="512">
{exchanges_html}
{revnotes_html}
{palette_table}
</body></html>
"""
    (frames_dir / "index.html").write_text(html, encoding="utf-8")
