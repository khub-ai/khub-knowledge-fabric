"""Render a single index.html for the preview panel (port 8753).

Shows the current frame image plus the latest TUTOR ↔ harness exchanges
(Round 1 probes + observations, and Round 2 revision notes if present).
Also renders live play-session transcripts via render_play_session().

Invoked from run_trial.py, run_round2.py, and run_play.py.
Overwrites frames/index.html each time.
"""
from __future__ import annotations

import base64
import io
import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path


def _json_pretty(obj) -> str:
    return escape(json.dumps(obj, indent=2, ensure_ascii=False))


def grid_to_png_b64(grid) -> str:
    """Render a 2-D numpy grid to a base64-encoded PNG using the ARC palette."""
    from arc_agi.rendering import COLOR_MAP
    from PIL import Image
    import numpy as np
    arr = np.asarray(grid)
    h, w = arr.shape[:2]
    img = Image.new("RGB", (w, h))
    pixels = img.load()
    for r in range(h):
        for c in range(w):
            v = int(arr[r, c])
            hx = COLOR_MAP.get(v, "#000000FF")
            pixels[c, r] = (int(hx[1:3], 16), int(hx[3:5], 16), int(hx[5:7], 16))
    img = img.resize((256, 256), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _render_initial_prompt(session_dir: Path) -> str:
    """Render the initial system+user prompt, if saved, as a collapsible block."""
    path = session_dir / "initial_prompt.txt"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    parts = ['<div class="section">']
    parts.append(f'<h2>Round 1 — initial prompt '
                 f'<span class="sub">({session_dir.name}/initial_prompt.txt)</span></h2>')
    parts.append('<details class="prompt-block"><summary>show prompt</summary>')
    parts.append('<pre class="prompt-pre">' + escape(text) + '</pre>')
    parts.append('</details></div>')
    return "\n".join(parts)


def _render_change_report(cr: dict) -> str:
    """Render a CHANGE_REPORT observation compactly as HTML."""
    parts: list[str] = ['<div class="change-report">']
    motions = cr.get("element_motions") or []
    if motions:
        parts.append('<div class="cr-label">element_motions</div><ul class="cr-list">')
        for m in motions:
            sign_r = "+" if m.get("dr", 0) >= 0 else ""
            sign_c = "+" if m.get("dc", 0) >= 0 else ""
            mv = "moved" if m.get("moved") else "static"
            parts.append(
                f'<li>#{escape(str(m.get("element_id","?")))} '
                f'<b>{escape(str(m.get("name","?")))}</b> '
                f'({mv}) Δ=({sign_r}{m.get("dr",0)}, {sign_c}{m.get("dc",0)}) '
                f'pre={escape(str(m.get("pre_bbox")))} → post={escape(str(m.get("post_bbox")))}</li>'
            )
        parts.append('</ul>')
    for key, label in (("appearances", "appearances"),
                       ("disappearances", "disappearances"),
                       ("counter_changes", "counter_changes")):
        items = cr.get(key) or []
        if items:
            parts.append(f'<div class="cr-label">{label}</div>')
            parts.append('<pre>' + _json_pretty(items) + '</pre>')
    unex = cr.get("unexplained_regions") or []
    if unex:
        parts.append('<div class="cr-label">unexplained_regions</div>')
        parts.append('<pre>' + _json_pretty(unex) + '</pre>')
    if cr.get("full_frame_fallback") is not None:
        parts.append('<div class="cr-label error">full_frame_fallback</div>')
        parts.append('<pre>(diff exceeded 30% of frame — full post-frame attached)</pre>')
    totals = cr.get("totals") or {}
    if totals:
        parts.append(
            f'<div class="cr-totals">diff={totals.get("diff_cells","?")} '
            f'unexplained={totals.get("unexplained_cells","?")} '
            f'/{totals.get("frame_area","?")} cells</div>'
        )
    parts.append('</div>')
    return "\n".join(parts)


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
            obs_list = exec_r.get("observations", []) or []
            other_obs = []
            for obs in obs_list:
                if obs.get("kind") == "CHANGE_REPORT":
                    parts.append('<div class="label">CHANGE_REPORT</div>')
                    parts.append(_render_change_report(obs))
                else:
                    other_obs.append(obs)
            if other_obs:
                parts.append('<pre>' + _json_pretty(other_obs) + '</pre>')
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
.prompt-block { background: #0e0e0e; border: 1px solid #2a2a2a; border-radius: 4px; padding: 6px; font-size: 11px; }
.prompt-block summary { cursor: pointer; color: #bbb; }
.prompt-pre { max-height: 400px; overflow: auto; color: #ccc; }
.change-report { background: #0a0a0a; border: 1px solid #1d3036; border-radius: 3px; padding: 6px; margin: 4px 0; }
.cr-label { color: #8ec0c8; font-size: 10px; text-transform: uppercase; margin-top: 4px; font-weight: bold; }
.cr-label.error { color: #ff8b6d; }
.cr-list { margin: 2px 0 4px 16px; padding: 0; font-size: 11px; color: #ddd; }
.cr-list li { margin: 1px 0; }
.cr-totals { color: #888; font-size: 10px; margin-top: 4px; }
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

    initial_prompt_html = ""
    if round1_session is not None:
        initial_prompt_html = _render_initial_prompt(round1_session)

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
{initial_prompt_html}
{exchanges_html}
{revnotes_html}
{palette_table}
</body></html>
"""
    (frames_dir / "index.html").write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Play-session live preview
# ---------------------------------------------------------------------------

PLAY_STYLE = STYLE + """
.live-banner { font-size:13px; font-weight:bold; padding:6px 12px; border-radius:4px;
               margin-bottom:8px; display:flex; gap:16px; align-items:center; }
.live-banner.running { background:#0d2b1a; color:#52b788; border:1px solid #2d6a4f; }
.live-banner.done    { background:#1a1a1a; color:#aaa;    border:1px solid #444; }
.live-banner .blink  { animation:blink 1s step-end infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
.play-header { display:flex; gap:24px; align-items:baseline; flex-wrap:wrap;
               background:#1a1a2e; padding:10px 14px; border-radius:6px; margin-bottom:12px; }
.play-header .game-id { font-size:14px; font-weight:bold; color:#e0e0ff; }
.play-header .stat    { font-size:12px; color:#aaa; }
.play-header .stat b  { color:#ccc; }
.turn-card { border:1px solid #2a2a3a; border-radius:6px; margin:10px 0;
             background:#161622; overflow:hidden; }
.turn-card.win      { border-color:#2a9d8f; }
.turn-card.game_over{ border-color:#e76f51; }
.turn-head { display:flex; align-items:center; gap:12px; padding:6px 10px;
             background:#1e1e30; font-size:12px; flex-wrap:wrap; }
.turn-num  { font-weight:bold; color:#e0e0ff; min-width:52px; }
.turn-action { background:#264653; color:#fff; border-radius:3px;
               padding:1px 7px; font-size:11px; font-weight:bold; }
.turn-state  { color:#888; font-size:11px; }
.turn-ts     { color:#666; font-size:11px; margin-left:auto; }
.turn-cost   { color:#a8dadc; font-size:11px; }
.turn-lat    { color:#888; font-size:11px; }
.turn-body   { display:flex; gap:0; }
.turn-frame  { flex:0 0 auto; padding:8px; background:#0e0e18;
               border-right:1px solid #2a2a3a; }
.turn-frame img { image-rendering:pixelated; display:block; }
.turn-info   { flex:1; padding:8px 10px; font-size:11px; }
.rationale   { color:#ddd; margin-bottom:6px; }
.revise      { background:#2a1a0a; border-left:3px solid #e9c46a;
               padding:4px 8px; margin:4px 0; color:#f4d58d; font-size:11px; }
.cr-compact  { color:#888; font-size:10px; margin-top:4px; }
.wk-block    { background:#0e0e18; border:1px solid #2a2a3a; border-radius:4px;
               padding:8px; margin-bottom:12px; font-size:11px; }
.wk-block summary { cursor:pointer; color:#aaa; font-size:12px; padding:2px 0; }
.wk-block pre { max-height:300px; overflow:auto; color:#ccc; margin:6px 0 0; }
.cost-total  { color:#a8dadc; font-weight:bold; }
.pgk-block   { background:#0e1a0e; border:1px solid #1d3d1d; border-radius:4px;
               padding:10px; margin-top:12px; font-size:12px; color:#b7e4c7; }
.pgk-block h3 { margin:0 0 8px; font-size:13px; color:#52b788; }
.src-label   { font-size:9px; font-weight:bold; border-radius:2px; padding:1px 4px;
               margin-right:5px; vertical-align:middle; letter-spacing:.03em; }
.src-tutor   { background:#264653; color:#a8dadc; }
.src-harness { background:#1d3d1d; color:#74c69d; }
.src-cc      { background:#3d2020; color:#e07070; }
.harness-note  { background:#0d2215; border-left:3px solid #52b788;
                 padding:4px 8px; margin:4px 0; color:#b7e4c7; font-size:11px; }
.harness-error { background:#2a0e0e; border-left:3px solid #e76f51;
                 padding:4px 8px; margin:4px 0; color:#ff8b6d; font-size:11px; }
.harness-info  { background:#0a1a2a; border-left:3px solid #457b9d;
                 padding:4px 8px; margin:4px 0; color:#a8dadc; font-size:11px; }
"""


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"T+{int(seconds)}s"
    return f"T+{int(seconds//60)}m{int(seconds%60):02d}s"


def _fmt_cost(usd: float) -> str:
    if usd < 0.001:
        return f"${usd*1000:.3f}m"
    return f"${usd:.4f}"


def render_play_session(
    session_dir: Path,
    frames_dir:  Path,
    *,
    live:        bool = True,
) -> None:
    """Write (or overwrite) frames/index.html with the full play-session transcript.

    Call after every turn while `live=True` (adds auto-refresh meta tag).
    Call with `live=False` when the session ends.
    """
    log_path = session_dir / "play_log.jsonl"
    if not log_path.exists():
        return

    entries = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass

    wk_text = ""
    wk_path = session_dir / "working_knowledge.md"
    if wk_path.exists():
        wk_text = wk_path.read_text(encoding="utf-8")

    pgk_text = ""
    pgk_path = session_dir / "post_game_knowledge.md"
    if pgk_path.exists():
        pgk_text = pgk_path.read_text(encoding="utf-8")

    # Aggregate stats — support both old (action) and new (command) schema
    turn_entries  = [e for e in entries if "action" in e or "command" in e]
    total_cost    = sum(e.get("cost_usd", 0) for e in turn_entries)
    total_in_tok  = sum(e.get("input_tokens", 0) for e in turn_entries)
    total_out_tok = sum(e.get("output_tokens", 0) for e in turn_entries)
    final_state   = "…"
    game_id       = ""
    levels_completed_now = 0
    win_levels_now = 1
    session_start: datetime | None = None
    for e in entries:
        if "final_state" in e:
            final_state = e["final_state"]
        if e.get("turn_start_iso") and session_start is None:
            try:
                session_start = datetime.fromisoformat(e["turn_start_iso"])
            except Exception:  # noqa: BLE001
                pass
        if e.get("game_id"):
            game_id = e["game_id"]
        if "levels_completed" in e:
            levels_completed_now = int(e["levels_completed"])
        if "win_levels" in e:
            win_levels_now = int(e["win_levels"])
    if not live and turn_entries:
        last = turn_entries[-1]
        final_state = last.get("state", final_state)

    try:
        manifest = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
        game_id = game_id or manifest.get("game_id", "")
        if not live:
            final_state = manifest.get("final_state", final_state)
    except FileNotFoundError:
        pass

    # Extract run ID from session_dir name (e.g. trial_20260420T123456Z_play → 20260420T123456Z)
    run_id = session_dir.name
    for prefix in ("trial_",):
        if run_id.startswith(prefix):
            run_id = run_id[len(prefix):]
    for suffix in ("_play", "_round2", "_round1"):
        if run_id.endswith(suffix):
            run_id = run_id[:-len(suffix)]

    # Parse run_id timestamp for JS relative-time display
    run_iso = ""
    try:
        run_dt = datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        run_iso = run_dt.isoformat()
    except Exception:
        pass

    # Human-readable level label: series + "Level N / M"
    series = game_id.split("-")[0] if "-" in game_id else game_id
    level_label = f"Level {levels_completed_now + 1}"
    if win_levels_now > 1:
        level_label += f" / {win_levels_now}"
    page_title = f"{series} {level_label}"

    refresh_tag = '<meta http-equiv="refresh" content="5; url=#latest">' if live else ""
    now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    # ---- header
    status_color = {"WIN": "#2a9d8f", "GAME_OVER": "#e76f51"}.get(final_state, "#888")
    run_ago_span = (f'<b id="run-ago" data-run-ts="{escape(run_iso)}">'
                   f'{escape(run_id)}</b>') if run_iso else f'<b>{escape(run_id)}</b>'
    if live:
        banner = (f'<div class="live-banner running">'
                  f'<span class="blink">&#9679;</span> LIVE &nbsp;|&nbsp; '
                  f'last update <b>{now_str}</b> &nbsp;|&nbsp; '
                  f'auto-refresh every 5 s &nbsp;|&nbsp; '
                  f'run {run_ago_span}</div>')
    else:
        banner = (f'<div class="live-banner done">'
                  f'&#9632; DONE &nbsp;|&nbsp; '
                  f'finished at <b>{now_str}</b> &nbsp;|&nbsp; '
                  f'state=<b style="color:{status_color}">{escape(final_state)}</b> &nbsp;|&nbsp; '
                  f'run {run_ago_span}</div>')

    parts = [f"""<!doctype html>
<html><head><meta charset="utf-8">{refresh_tag}
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<title>{escape(page_title)} play</title>
<style>{PLAY_STYLE}</style></head><body>
{banner}
<div class="play-header">
  <span class="game-id">{escape(page_title)}</span>
  <span class="stat">TUTOR calls <b>{len(turn_entries)}</b></span>
  <span class="stat">state <b style="color:{status_color}">{escape(final_state)}</b></span>
  <span class="stat cost-total">cost <b>{_fmt_cost(total_cost)}</b></span>
  <span class="stat">tokens in <b>{total_in_tok:,}</b> / out <b>{total_out_tok:,}</b></span>
  <span class="stat" title="{escape(session_dir.name)}">run {run_ago_span}</span>
</div>"""]

    # ---- working knowledge
    parts.append('<details class="wk-block"><summary>WORKING_KNOWLEDGE</summary>')
    parts.append(f'<pre>{escape(wk_text)}</pre></details>')

    # ---- turn cards (newest = last; mark last card id="latest" for auto-scroll)
    for e_idx, e in enumerate(turn_entries):
        turn      = e.get("turn", "?")
        # support both old (action) and new (command) schema
        action    = e.get("command") or e.get("action", "?")
        args      = e.get("args") or {}
        state     = e.get("state", "?")
        rationale = e.get("rationale", "")
        revise    = e.get("revise_knowledge", "")
        lat_ms    = e.get("latency_ms")
        cost      = e.get("cost_usd", 0)
        in_tok    = e.get("input_tokens", 0)
        out_tok   = e.get("output_tokens", 0)
        frame_b64     = e.get("frame_b64", "")
        seq           = e.get("action_sequence") or []
        steps         = e.get("steps_taken", 0)
        cursor        = e.get("cursor_pos")
        cr_dict       = e.get("command_result") or {}
        harness_note  = cr_dict.get("harness_note") or e.get("harness_note", "")
        exec_error    = cr_dict.get("error") or e.get("exec_error") or e.get("error", "")
        elem_overlaps = cr_dict.get("element_overlaps") or e.get("element_overlaps") or []
        glyph_info    = cr_dict.get("glyph_interior") or {}

        ts_str = ""
        elapsed_str = ""
        if e.get("turn_start_iso"):
            try:
                ts = datetime.fromisoformat(e["turn_start_iso"])
                ts_str = ts.strftime("%H:%M:%S")
                if session_start:
                    elapsed_str = _fmt_elapsed((ts - session_start).total_seconds())
            except Exception:  # noqa: BLE001
                pass

        # change_report: check top-level first, then last step of motion_log
        cr = e.get("change_report") or {}
        if not cr:
            motion_log = cr_dict.get("motion_log") or []
            if motion_log:
                cr = motion_log[-1].get("change_report") or {}
        cr_pm = (cr.get("primary_motion") or {})
        cr_line = ""
        if cr:
            totals = cr.get("totals") or {}
            pm_name = cr_pm.get("name", "") if cr_pm else ""
            pm_dr   = cr_pm.get("dr") if cr_pm else None
            pm_dc   = cr_pm.get("dc") if cr_pm else None
            counters = cr.get("counter_changes") or []
            ctr_str = "; ".join(
                f"{c.get('name','?')}:{c.get('before_fill','?')}->{c.get('after_fill','?')}"
                for c in counters
            )
            pm_str = f"primary={escape(pm_name)} dr={pm_dr} dc={pm_dc}" if pm_name else "no primary_motion"
            cr_line = (f"diff={totals.get('diff_cells','?')} "
                       f"| {pm_str}"
                       + (f" | counters: {escape(ctr_str)}" if ctr_str else ""))

        card_cls = "turn-card"
        if state == "WIN":
            card_cls += " win"
        elif state == "GAME_OVER":
            card_cls += " game_over"

        args_brief = ""
        if args:
            try:
                args_brief = " " + json.dumps(args, separators=(",", ":"))
            except Exception:  # noqa: BLE001
                pass

        cursor_html = ""
        if cursor:
            cursor_html = f'<span class="turn-state"> cursor={escape(str(cursor))}</span>'
        steps_html = ""
        if steps:
            steps_html = f'<span class="turn-state"> game-steps={steps}</span>'

        is_last = (e_idx == len(turn_entries) - 1)
        card_id = ' id="latest"' if is_last else ""
        parts.append(f'<div class="{card_cls}"{card_id}>')
        parts.append(
            f'<div class="turn-head">'
            f'<span class="turn-num">#{turn}</span>'
            f'<span class="turn-action">{escape(action)}</span>'
            f'<span class="turn-state">{escape(args_brief)}</span>'
            f'<span class="turn-state"> state={escape(state)}</span>'
            f'{cursor_html}{steps_html}'
            f'<span class="turn-cost">{_fmt_cost(cost)} ({in_tok}in/{out_tok}out)</span>'
            f'<span class="turn-lat">{lat_ms}ms</span>'
            f'<span class="turn-ts" title="{escape(ts_str)}">{escape(elapsed_str)}</span>'
            f'</div>'
        )
        parts.append('<div class="turn-body">')
        if frame_b64:
            parts.append(
                f'<div class="turn-frame">'
                f'<img src="data:image/png;base64,{frame_b64}" width="128" height="128">'
                f'</div>'
            )
        parts.append('<div class="turn-info">')
        if rationale:
            parts.append(
                f'<div class="rationale">'
                f'<span class="src-label src-tutor">TUTOR</span>'
                f'{escape(rationale)}</div>'
            )
        if revise:
            parts.append(
                f'<div class="revise">'
                f'<span class="src-label src-tutor">TUTOR REVISE</span>'
                f'{escape(revise)}</div>'
            )
        if harness_note:
            parts.append(
                f'<div class="harness-note">'
                f'<span class="src-label src-harness">HARNESS NOTE</span>'
                f'{escape(harness_note)}</div>'
            )
        if exec_error:
            parts.append(
                f'<div class="harness-error">'
                f'<span class="src-label src-harness">HARNESS ERROR</span>'
                f'{escape(str(exec_error))}</div>'
            )
        if elem_overlaps:
            parts.append(
                f'<div class="harness-info">'
                f'<span class="src-label src-harness">HARNESS OVERLAPS</span>'
                f'{escape(str(elem_overlaps))}</div>'
            )
        if glyph_info:
            changed = glyph_info.get("changed")
            glyph_cls = "harness-note" if changed else "cr-compact"
            glyph_txt = ("GLYPH ROTATED" if changed else "glyph unchanged") + \
                        f" (rows 11-13 cols 35-37)"
            parts.append(
                f'<div class="{glyph_cls}">'
                f'<span class="src-label src-harness">HARNESS GLYPH</span>'
                f'{escape(glyph_txt)}</div>'
            )
        if cr_line:
            parts.append(
                f'<div class="cr-compact">'
                f'<span class="src-label src-harness">HARNESS</span>'
                f'{cr_line}</div>'
            )
        parts.append('</div></div></div>')  # turn-info / turn-body / turn-card

    # ---- post-game note
    if pgk_text:
        parts.append('<div class="pgk-block">')
        parts.append('<h3>Post-game knowledge note</h3>')
        parts.append(f'<pre>{escape(pgk_text)}</pre>')
        parts.append('</div>')

    if live:
        parts.append('<script>var el=document.getElementById("latest");'
                     'if(el)el.scrollIntoView({behavior:"smooth",block:"start"});</script>')
    parts.append("""<script>
(function(){
  function fmtAgo(isoTs){
    var d=new Date(isoTs), s=Math.round((Date.now()-d.getTime())/1000);
    if(s<60) return s+'s ago';
    var m=Math.floor(s/60), r=s%60;
    if(m<60) return m+'m '+r+'s ago';
    var h=Math.floor(m/60); m=m%60;
    return h+'h '+m+'m ago';
  }
  function tick(){
    var el=document.querySelectorAll('#run-ago[data-run-ts]');
    el.forEach(function(e){
      var ts=e.getAttribute('data-run-ts');
      if(ts) e.textContent='run '+fmtAgo(ts);
    });
  }
  tick(); setInterval(tick,1000);
})();
</script>""")
    parts.append('</body></html>')
    (frames_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")
