"""Strict-mode play loop.

Runs under the prime directive: no env._game access, no privileged
signals, no hardcoded function tags, no authored KB.  Only obs fields
and pixel analysis.

Architecture:
  1. Reset env; extract initial frame components.
  2. Bootstrap: mechanically probe each available direction action to
     learn (dr, dc) effects.  Zero LLM calls.
  3. Main loop: for each turn, send TUTOR the untagged components +
     known effects + recent outcomes; TUTOR picks a MOVE_TO target.
  4. Harness executes MOVE_TO via BFS over an EMPIRICALLY-DISCOVERED
     passable grid (every cell tentatively passable; failed moves
     accumulate (r, c, action) walls).
  5. After each TUTOR-decided command, observe obs.levels_completed
     and obs.state; stop on level advance or game over or max turns.

Budget cost per session:
  - Bootstrap: ~4 env.step calls (one per direction), 0 LLM.
  - Each TUTOR turn: ~$0.08-0.12 depending on prompt size.
  - BFS navigation costs env.step calls but no LLM.

Writes:
  - session_dir/play_log.jsonl     (per-turn records)
  - session_dir/working_kb.md      (postgame accumulated)
  - session_dir/manifest.json       (outcome summary)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ARC_REPO = Path(os.environ.get("ARC_AGI_3_REPO", r"C:\_backup\github\arc-agi-3"))
sys.path.insert(0, str(ARC_REPO))

from arc_agi import Arcade, OperationMode                           # noqa: E402

from dsl_executor import _normalise_frame                           # noqa: E402
from pixel_elements import extract_components, summarize_frame, narrate_frame_delta  # noqa: E402
from cell_system import CellSystem, infer_cell_system, components_in_cells  # noqa: E402
from discovery_bootstrap import bootstrap_action_effects             # noqa: E402
from discovery_prompts import SYSTEM_DISCOVERY, USER_DISCOVERY_TEMPLATE  # noqa: E402
import backends                                                      # noqa: E402

TUTOR_MODEL = "claude-sonnet-4-6"
TRAINING_DATA_DIR = HERE.parents[2] / ".tmp" / "training_data"


# ---------------------------------------------------------------------------
# Strict-mode solutions (for silent replay of already-solved levels)
# ---------------------------------------------------------------------------
# A recorded solution is the exact sequence of action integers the agent
# took from spawn to level completion, plus a frame-hash fingerprint of
# the level's entry frame so replay can detect when the recorded solve
# doesn't apply (e.g. layout changed between sessions).

def _strict_solutions_path(game_id: str) -> Path:
    return HERE.parent / "benchmarks" / "knowledge_base" / f"{game_id}_strict_solutions.json"


def _load_strict_solutions(game_id: str) -> dict:
    p = _strict_solutions_path(game_id)
    if not p.exists():
        return {"game_id": game_id, "levels": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"game_id": game_id, "levels": {}}


def _save_strict_solutions(game_id: str, data: dict) -> None:
    p = _strict_solutions_path(game_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    data.setdefault("game_id", game_id)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _frame_hash(grid: np.ndarray) -> str:
    """16-char SHA-256 fingerprint of a palette frame."""
    import hashlib
    return hashlib.sha256(
        np.asarray(grid, dtype=np.int8).tobytes()
    ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _format_frame_text(grid: np.ndarray) -> str:
    rows = [", ".join(f"{int(v):2d}" for v in row) for row in grid]
    return "[\n" + ",\n".join(f"  [{r}]" for r in rows) + "\n]"


# Palette -> RGB mapping.  The PALETTE values 0-15 are abstract indices in
# the game; the actual screen colors are not part of the public obs (only
# integer palette numbers).  We pick a DISTINCT, HIGH-CONTRAST palette
# here so a vision model can tell them apart.  Each palette gets a
# visually distinct color; this mapping is arbitrary but stable.
_VIEW_PALETTE_RGB = [
    (  0,   0,   0),   # 0 black
    (255, 255, 255),   # 1 white
    (128, 128, 128),   # 2 mid-gray
    ( 60,  60,  90),   # 3 dark blue-gray (background)
    (100,  50,  30),   # 4 dark brown (wall)
    (180, 180, 180),   # 5 light gray
    (255,   0,   0),   # 6 red
    (  0, 200,   0),   # 7 green
    (  0,   0, 255),   # 8 blue
    (255, 140,   0),   # 9 orange
    (255,   0, 255),   # 10 magenta
    (255, 230,   0),   # 11 yellow
    (  0, 200, 255),   # 12 cyan
    (170,   0, 170),   # 13 purple
    (100, 255, 100),   # 14 light green
    (200, 150, 100),   # 15 tan
]


def _frame_to_png_b64(
    grid:       np.ndarray,
    upscale:    int = 10,
    cell_size:  int | None = None,
    origin:     tuple[int, int] | None = None,
    agent_cell: tuple[int, int] | None = None,
    annotations: list[dict] | None = None,
) -> str:
    """Render a palette frame as an upscaled color PNG (base64).

    Optional annotations: list of {cell: (cr, cc), label: str, color: (R,G,B)}.
    Each annotation draws a small marker at the cell center with a text
    label ("#id") so TUTOR's vision can match the image to the text
    COMPONENTS list unambiguously.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io, base64

        H, W = grid.shape
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        for p in range(len(_VIEW_PALETTE_RGB)):
            mask = (grid == p)
            if mask.any():
                rgb[mask] = _VIEW_PALETTE_RGB[p]
        img = Image.fromarray(rgb, mode="RGB").resize(
            (W * upscale, H * upscale), Image.NEAREST,
        )

        if cell_size and origin is not None:
            draw = ImageDraw.Draw(img)
            grid_color = (80, 80, 80)
            half = cell_size / 2.0

            # Cell grid lines.
            c0 = origin[1] - half
            c = c0 - int((c0 // cell_size) * cell_size)
            while c < W:
                x = int(c * upscale)
                draw.line([(x, 0), (x, H * upscale - 1)], fill=grid_color, width=1)
                c += cell_size
            r0 = origin[0] - half
            r = r0 - int((r0 // cell_size) * cell_size)
            while r < H:
                y = int(r * upscale)
                draw.line([(0, y), (W * upscale - 1, y)], fill=grid_color, width=1)
                r += cell_size

            # Try to load a font for labels; fall back to default if missing.
            try:
                font = ImageFont.truetype("arial.ttf", size=max(10, upscale + 2))
            except Exception:
                font = ImageFont.load_default()

            # Annotate distinctive components with their ids + cell addresses.
            for ann in (annotations or []):
                ac_r, ac_c = ann["cell"]
                pix_r = origin[0] + ac_r * cell_size
                pix_c = origin[1] + ac_c * cell_size
                x_center = int(pix_c * upscale + upscale / 2)
                y_center = int(pix_r * upscale + upscale / 2)
                color = ann.get("color", (255, 255, 0))
                # Small circle at cell center.
                ring_r = max(4, upscale // 2)
                draw.ellipse([x_center - ring_r, y_center - ring_r,
                              x_center + ring_r, y_center + ring_r],
                             outline=color, width=2)
                # Label: #id above the circle.
                label = ann.get("label", "")
                if label:
                    # White text with black stroke for legibility.
                    tx, ty = x_center + ring_r + 2, y_center - ring_r - 2
                    # Background pad to make label visible on any background.
                    try:
                        bbox = draw.textbbox((tx, ty), label, font=font)
                        draw.rectangle(bbox, fill=(0, 0, 0))
                    except Exception:
                        pass
                    draw.text((tx, ty), label, fill=color, font=font)

            # Agent cell outline in bright green, drawn LAST so it's on top.
            if agent_cell is not None:
                ar = origin[0] + agent_cell[0] * cell_size
                ac = origin[1] + agent_cell[1] * cell_size
                x0 = int((ac - half) * upscale)
                y0 = int((ar - half) * upscale)
                x1 = int((ac + half) * upscale)
                y1 = int((ar + half) * upscale)
                for off in range(3):
                    draw.rectangle([x0-off, y0-off, x1+off, y1+off],
                                   outline=(0, 255, 0), width=1)
                # Label the agent cell with "YOU".
                try:
                    bbox = draw.textbbox((x0 + 2, y0 + 2), "YOU", font=font)
                    draw.rectangle(bbox, fill=(0, 0, 0))
                except Exception:
                    pass
                draw.text((x0 + 2, y0 + 2), "YOU", fill=(0, 255, 0), font=font)

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        print(f"[render] frame-to-png failed: {e}")
        return ""


def _extract_json(text: str) -> dict:
    import re
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    first = text.find("{")
    if first < 0:
        raise ValueError(f"no JSON in reply: {text[:200]!r}")
    depth = 0
    for i in range(first, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[first:i + 1])
    raise ValueError("unterminated JSON")


def _components_summary(
    comps:      list[dict],
    agent_fp:   tuple[int, int, tuple] | None,
    cs:         CellSystem,
    max_n:      int = 15,
) -> str:
    """Render components in CELL COORDS, sorted by distinctiveness.

    All positions are reported as cell addresses, not raw pixels.  Each
    component also lists cells_covered so TUTOR knows sprites that span
    multiple cells.
    """
    if not comps:
        return "  (none)"
    palette_total: dict[int, int] = {}
    for c in comps:
        palette_total[c["palette"]] = palette_total.get(c["palette"], 0) + c["size"]

    # Enrich with cell coords first, then filter out the agent.
    enriched = components_in_cells(comps, cs)
    filtered = []
    for c in enriched:
        if agent_fp is not None and (
            c["palette"] == agent_fp[0] and c["size"] == agent_fp[1]
            and tuple(c["extent"]) == tuple(agent_fp[2])
        ):
            continue
        filtered.append(c)

    filtered.sort(key=lambda c: (palette_total[c["palette"]], c["size"]))

    out = []
    for c in filtered[:max_n]:
        rare_note = f"pal_total={palette_total[c['palette']]}"
        covered = c["cells_covered"]
        covered_str = f"covers={covered}" if len(covered) > 1 else ""
        out.append(
            f"  id={c['id']:2d} pal={c['palette']:2d} sz={c['size']:3d} "
            f"cell={c['cell']} ext={c['extent']} {rare_note} {covered_str}"
        )
    if len(filtered) > max_n:
        out.append(f"  ... ({len(filtered) - max_n} more)")
    return "\n".join(out)


def _action_effects_text(
    effects: dict[str, tuple[int, int]],
    cs:      CellSystem | None = None,
) -> str:
    if not effects:
        return "  (none learned yet)"
    lines = []
    for a, (dr, dc) in effects.items():
        if cs is not None:
            cdr, cdc = cs.action_to_cell_delta(dr, dc)
            lines.append(f"  {a}: cell_delta=({cdr:+d}, {cdc:+d})")
        else:
            lines.append(f"  {a}: dr={dr:+d}, dc={dc:+d}")
    return "\n".join(lines)


def _history_text(hist: list[dict], n: int = 3) -> str:
    if not hist:
        return "  (none)"
    lines = []
    for h in hist[-n:]:
        lines.append(
            f"  turn {h['turn']}: MOVE_TO {h.get('target')} "
            f"reached={h.get('reached')} "
            f"agent_end={h.get('cur_pos')} "
            f"lc={h.get('lc_before')}->{h.get('lc_after')} "
            f"frame_diff_cells={h.get('frame_diff_cells')}"
        )
        # Narrate any observed component changes.  Brief but concrete.
        delta = h.get("delta") or {}
        if delta.get("disappeared"):
            dstr = ", ".join(
                f"pal={d['palette']} sz={d['size']} @{d['centroid']}"
                for d in delta["disappeared"][:3]
            )
            lines.append(f"    CHANGES disappeared: {dstr}")
        if delta.get("appeared"):
            astr = ", ".join(
                f"pal={d['palette']} sz={d['size']} @{d['centroid']}"
                for d in delta["appeared"][:3]
            )
            lines.append(f"    CHANGES appeared: {astr}")
        if delta.get("moved"):
            mstr = ", ".join(
                f"pal={d['palette']} sz={d['size']} {d['from']}->{d['to']}"
                for d in delta["moved"][:3]
            )
            lines.append(f"    CHANGES moved (non-agent): {mstr}")
        rat = h.get("rationale") or ""
        if rat:
            lines.append(f"    rationale: '{rat[:80]}'")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Harness-level BFS navigator for STRICT mode
# ---------------------------------------------------------------------------
#
# No hardcoded wall palettes: walls are accumulated EMPIRICALLY -- every
# (r, c, action) for which the agent failed to move (frame changed less
# than a movement-threshold).  Passable grid is "every cell tentatively
# passable".  If BFS can't find a path, we return the closest reachable.

def _probe_phase(
    env,
    obs,
    cs:             CellSystem,
    agent_fp:       tuple[int, int, tuple],
    agent_cell:     tuple[int, int],
    comps:          list[dict],
    cell_actions:   dict[str, tuple[int, int]],
    walls_cells:    set[tuple[int, int, str]],
    cell_bounds:    tuple[int, int, int, int],
    max_probes:     int = 3,
    max_steps_per_probe: int = 10,
) -> tuple[list[dict], tuple[int, int], object]:
    """Harness-autonomous probe of distinctive unknown objects.

    Ranks non-agent components by distinctiveness (rare palette + small
    size), then for each of the top N, routes the agent to its cell via
    BFS and records the full before->after frame delta.  This gives
    TUTOR a baseline for what happens when stepping on unknown icons,
    WITHOUT requiring TUTOR to burn calls on that.

    Returns:
      (probe_results, final_agent_cell, final_frame)
    where each probe_result is a dict with component metadata plus
    observed effects (reached, diff_cells, disappeared/appeared/moved
    summaries, obs fields delta).
    """
    # Rank non-agent components by distinctiveness.
    palette_total: dict[int, int] = {}
    for c in comps:
        palette_total[c["palette"]] = palette_total.get(c["palette"], 0) + c["size"]
    candidates = [
        c for c in components_in_cells(comps, cs)
        if not (c["palette"] == agent_fp[0] and c["size"] == agent_fp[1]
                and tuple(c["extent"]) == tuple(agent_fp[2]))
    ]
    candidates.sort(key=lambda c: (palette_total[c["palette"]], c["size"]))
    candidates = candidates[:max_probes]

    results: list[dict] = []
    cur_cell = agent_cell

    for cand in candidates:
        target = tuple(cand["cell"])
        if target == cur_cell:
            results.append({
                "component_id":  cand["id"],
                "palette":       cand["palette"],
                "size":          cand["size"],
                "target_cell":   list(target),
                "reached":       True,
                "note":          "already at this cell",
            })
            continue

        frame_before = _normalise_frame(obs.frame)

        path = _bfs_plan_cells(cur_cell, target, cell_actions, walls_cells,
                                cell_bounds=cell_bounds, max_steps=20)
        if not path:
            results.append({
                "component_id":  cand["id"],
                "palette":       cand["palette"],
                "size":          cand["size"],
                "target_cell":   list(target),
                "reached":       False,
                "note":          "BFS no path",
            })
            continue

        lc_before = int(obs.levels_completed)
        steps_run = 0
        for action_name in path[:max_steps_per_probe]:
            act_int = int(action_name.replace("ACTION", ""))
            obs = env.step(act_int)
            steps_run += 1
            cur_frame = _normalise_frame(obs.frame)
            new_cell = _agent_cell_from_frame(cur_frame, agent_fp, cs)
            if new_cell is None:
                break
            if new_cell == cur_cell:
                # Blocked -- record wall for future BFS, stop this probe.
                walls_cells.add((cur_cell[0], cur_cell[1], action_name))
                break
            cur_cell = new_cell
            if obs.state.name != "NOT_FINISHED" or int(obs.levels_completed) > lc_before:
                break
            if new_cell == target:
                break

        frame_after = _normalise_frame(obs.frame)
        delta = narrate_frame_delta(frame_before, frame_after, agent_fp)
        diff_cells = int((frame_before != frame_after).sum())
        lc_after = int(obs.levels_completed)
        reached = (cur_cell == target)
        results.append({
            "component_id":   cand["id"],
            "palette":        cand["palette"],
            "size":           cand["size"],
            "target_cell":    list(target),
            "reached":        reached,
            "steps_run":      steps_run,
            "diff_cells":     diff_cells,
            "lc_before":      lc_before,
            "lc_after":       lc_after,
            "disappeared_n":  len(delta.get("disappeared") or []),
            "appeared_n":     len(delta.get("appeared")    or []),
            "moved_n":        len(delta.get("moved")       or []),
            "disappeared":    delta.get("disappeared") or [],
            "appeared":       delta.get("appeared")    or [],
            "moved":          delta.get("moved")       or [],
        })

        # Early-exit if game ended (WIN / GAME_OVER).
        if obs.state.name != "NOT_FINISHED":
            break

    return results, cur_cell, obs


def _bfs_plan_cells(
    start_cell:    tuple[int, int],
    target_cell:   tuple[int, int],
    cell_actions:  dict[str, tuple[int, int]],
    walls_cells:   set[tuple[int, int, str]],
    cell_bounds:   tuple[int, int, int, int] | None = None,
    max_steps:     int = 60,
) -> Optional[list[str]]:
    """BFS over CELL coordinates.  `cell_actions` is the action -> cell_delta
    map (e.g. ACTION1 -> (-1, 0)).  Walls are (cell_r, cell_c, action)
    triples meaning: from this cell, this action is known-blocked.

    If cell_bounds is given as (cr_min, cc_min, cr_max, cc_max), cells
    outside the bounds are treated as impassable (prevents BFS from
    wandering off-grid).  Otherwise BFS trusts only walls + visited.
    """
    if start_cell == target_cell:
        return []
    if not cell_actions:
        return None

    queue: deque = deque([(start_cell, [])])
    visited: dict[tuple[int, int], int] = {start_cell: 0}

    best_pos = start_cell
    best_dist = (start_cell[0] - target_cell[0]) ** 2 + (start_cell[1] - target_cell[1]) ** 2
    best_path: list[str] = []

    while queue:
        (cr, cc), path = queue.popleft()
        d = (cr - target_cell[0]) ** 2 + (cc - target_cell[1]) ** 2
        if d < best_dist:
            best_dist = d
            best_pos = (cr, cc)
            best_path = path
        if len(path) >= max_steps:
            continue
        for action, (dcr, dcc) in cell_actions.items():
            if (cr, cc, action) in walls_cells:
                continue
            nr, nc = cr + dcr, cc + dcc
            if cell_bounds is not None:
                if not (cell_bounds[0] <= nr <= cell_bounds[2]
                        and cell_bounds[1] <= nc <= cell_bounds[3]):
                    continue
            steps = len(path) + 1
            if (nr, nc) in visited and visited[(nr, nc)] <= steps:
                continue
            visited[(nr, nc)] = steps
            new_path = path + [action]
            if (nr, nc) == target_cell:
                return new_path
            queue.append(((nr, nc), new_path))

    return best_path if best_path else None


# ---------------------------------------------------------------------------
# Pixel-derived cursor tracking (strict-mode replacement for _agent_cursor_from_game)
# ---------------------------------------------------------------------------

def _agent_centroid_from_frame(
    frame: np.ndarray,
    agent_fingerprint: tuple[int, int, tuple] | None,
) -> tuple[int, int] | None:
    """Find the agent's PIXEL centroid by matching the known fingerprint."""
    if agent_fingerprint is None:
        return None
    pal, size, extent = agent_fingerprint
    comps = extract_components(frame, min_size=2)
    matches = [c for c in comps
               if c["palette"] == pal and c["size"] == size
               and tuple(c["extent"]) == tuple(extent)]
    if len(matches) != 1:
        return None
    return (matches[0]["centroid"][0], matches[0]["centroid"][1])


def _agent_cell_from_frame(
    frame: np.ndarray,
    agent_fingerprint: tuple[int, int, tuple] | None,
    cs: CellSystem,
) -> tuple[int, int] | None:
    """Find the agent's CELL address.  Returns None if unresolvable."""
    pix = _agent_centroid_from_frame(frame, agent_fingerprint)
    if pix is None:
        return None
    return cs.pix_to_cell(pix[0], pix[1])


# ---------------------------------------------------------------------------
# Main play loop
# ---------------------------------------------------------------------------

def _kb_runtime_path(game_id: str) -> Path:
    return HERE.parent / "benchmarks" / "knowledge_base" / f"{game_id}_runtime.json"


def _load_kb(game_id: str) -> dict:
    """Load accumulated cross-session knowledge.

    Strict-mode KB fields (all fields pixel/obs-derived, no env._game):
      action_effects_learned:  {"ACTION1": [dr, dc], ...}
      walls_by_level:          {"0": [[r, c, action], ...], ...}
      agent_fingerprint:       [palette, size, [h, w]]   (inferred from motion)
      blocked_targets_by_level:{"0": [[r, c], ...], ...} (MOVE_TO failed to advance lc)
      last_updated:            ISO timestamp
    """
    p = _kb_runtime_path(game_id)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_kb(game_id: str, kb: dict) -> None:
    p = _kb_runtime_path(game_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    kb["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    p.write_text(json.dumps(kb, indent=2), encoding="utf-8")


def run_session(
    game_id:        str,
    max_turns:      int = 8,
    session_dir:    Path | None = None,
    max_tokens:     int = 1500,
    model:          str = TUTOR_MODEL,
    record_solution: bool = True,
    replay_solved:   bool = True,
) -> dict:
    arc = Arcade(
        operation_mode   = OperationMode.OFFLINE,
        environments_dir = str(ARC_REPO / "environment_files"),
    )
    env = arc.make(game_id)
    obs = env.reset()
    frame0 = _normalise_frame(obs.frame)

    if session_dir:
        session_dir.mkdir(parents=True, exist_ok=True)
        log_path     = session_dir / "play_log.jsonl"
        manifest_path = session_dir / "manifest.json"
        log_path.write_text("")
    else:
        log_path = None
        manifest_path = None

    # ---- LOAD ACCUMULATED KB ----
    kb = _load_kb(game_id)
    kb.setdefault("action_effects_learned", {})
    kb.setdefault("walls_by_level", {})
    kb.setdefault("blocked_targets_by_level", {})
    prior_effects = {
        a: tuple(e) for a, e in kb.get("action_effects_learned", {}).items()
    }
    prior_fp_raw = kb.get("agent_fingerprint")
    prior_agent_fp = (
        (int(prior_fp_raw[0]), int(prior_fp_raw[1]), tuple(prior_fp_raw[2]))
        if prior_fp_raw else None
    )
    print(f"[kb] loaded action_effects={prior_effects} agent_fp={prior_agent_fp}")

    # ---- BOOTSTRAP (no LLM) ----
    # Skip bootstrap entirely if we already know all 4 directional actions.
    available = [int(a) for a in obs.available_actions if int(a) != 0]
    need_bootstrap = any(f"ACTION{a}" not in prior_effects for a in available)

    if need_bootstrap:
        print(f"[strict] bootstrap probing actions {available}...")
        boot = bootstrap_action_effects(env, _normalise_frame, available)
        learned = {a: tuple(e) for a, e in boot["action_effects_learned"].items()}
        action_effects = {**prior_effects, **learned}
        agent_fp = prior_agent_fp
        if not agent_fp and boot["agent_candidates"]:
            agent_fp = boot["agent_candidates"][0][0]
        print(f"[strict] action_effects now: {action_effects}")
        if agent_fp:
            print(f"[strict] agent fingerprint: palette={agent_fp[0]} size={agent_fp[1]} extent={agent_fp[2]}")
    else:
        print(f"[kb] skipping bootstrap -- all actions already known")
        action_effects = dict(prior_effects)
        agent_fp = prior_agent_fp

    # Reset env so turn 1 starts at spawn (bootstrap may have moved us).
    obs = env.reset()
    frame = _normalise_frame(obs.frame)

    # ---- INFER CELL SYSTEM ----
    # Requires both action_effects (from bootstrap) and the agent sprite's
    # centroid at spawn.  Fails gracefully if either is unknown.
    spawn_centroid = _agent_centroid_from_frame(frame, agent_fp)
    if not action_effects or spawn_centroid is None:
        print("[strict] FATAL: cannot infer cell system (missing action_effects or agent sprite)")
        return {"ok": False, "outcome": "NO_CELL_SYSTEM"}
    cs = infer_cell_system(action_effects, spawn_centroid)
    print(f"[cell] cell_size={cs.cell_size} origin=({cs.origin_r},{cs.origin_c}) "
          f"spawn_cell=(0,0)")

    # Cell-space action deltas.
    cell_actions: dict[str, tuple[int, int]] = {
        a: cs.action_to_cell_delta(dr, dc) for a, (dr, dc) in action_effects.items()
    }

    # ---- WALLS & BLOCKED-TARGETS FROM KB (cell coords) ----
    level_key = str(int(obs.levels_completed))
    walls_cells: set[tuple[int, int, str]] = {
        (int(cr), int(cc), str(a))
        for cr, cc, a in kb.get("walls_by_level_cells", {}).get(level_key, [])
    }
    # Blocked cells are session-local (not persisted).  "Unreachable with
    # current walls" often changes as walls invalidate across sessions.
    # Persisting them just accumulates a long "do not try" list that the
    # agent has to re-validate anyway.
    blocked_cells: set[tuple[int, int]] = set()
    # Load prior hypotheses accumulated by previous sessions (free-form text TUTOR wrote).
    prior_hypotheses_list: list[str] = list(
        kb.get("hypotheses_by_level", {}).get(level_key, [])
    )
    if prior_hypotheses_list:
        print(f"[kb] L{level_key}: loaded {len(prior_hypotheses_list)} prior hypotheses")

    # Wall observation bookkeeping (session-local, persisted at end):
    #   wall_hit_counts: how many times each (cr,cc,action) failed this session.
    #   new_walls_this_session: walls first observed this session (count = 1+).
    #   successful_moves: (cr,cc,action) that DID move the agent this session.
    # Rules:
    #   - BFS treats walls_cells as impassable (hard).
    #   - On hit: increment hit count.  Cap: after MAX_NEW_WALLS per session,
    #     no more hits add walls (prevents poisoning from a single bad run).
    #   - On success: remove (cr,cc,action) from walls_cells; mark in successful_moves.
    # At session end:
    #   - For walls hit 2+ times this session, confirm into KB.
    #   - For walls hit once: keep in KB only if already there (inherited confirmation).
    #   - For (cr,cc,action) in successful_moves: remove from KB (it's passable).
    wall_hit_counts: dict[tuple[int, int, str], int] = {}
    successful_moves: set[tuple[int, int, str]] = set()
    MAX_NEW_WALLS_PER_SESSION = 18
    new_walls_recorded = 0
    print(f"[kb] L{level_key}: {len(walls_cells)} cell-walls, "
          f"{len(blocked_cells)} blocked-cells loaded "
          f"(session cap on new walls: {MAX_NEW_WALLS_PER_SESSION})")

    # ---- LOAD STRICT-MODE SOLUTIONS ----
    solutions_data = _load_strict_solutions(game_id) if (replay_solved or record_solution) else None
    if solutions_data:
        known_levels = sorted(int(k) for k in solutions_data.get("levels", {}))
        print(f"[sol] {len(known_levels)} recorded strict-mode solutions: {known_levels}")

    # ---- TURN LOOP ----
    history: list[dict] = []
    training_records: list[dict] = []
    cost_usd_total = 0.0
    turns_used = 0
    level_advanced = False
    final_state = "NOT_FINISHED"
    initial_lc = int(obs.levels_completed)

    cur_level_steps: list[str] = []
    level_entry_hashes: dict[int, str] = {}
    last_seen_lc = initial_lc   # triggers per-level re-init on change
    probe_results_by_level: dict[int, list[dict]] = {}
    probed_levels: set[int] = set()

    for turn in range(1, max_turns + 1):
        turns_used = turn
        if obs.state.name != "NOT_FINISHED":
            final_state = obs.state.name
            break

        frame = _normalise_frame(obs.frame)
        cur_lc = int(obs.levels_completed)
        if cur_lc not in level_entry_hashes:
            level_entry_hashes[cur_lc] = _frame_hash(frame)

        # On level change, re-initialize cell system + walls from KB.
        if cur_lc != last_seen_lc:
            # Persist current level's walls back to KB before switching.
            existing_walls = {
                (int(cr), int(cc), str(a))
                for cr, cc, a in kb.setdefault("walls_by_level_cells", {}).get(level_key, [])
            }
            kb["walls_by_level_cells"][level_key] = sorted(
                [list(w) for w in (existing_walls | walls_cells)]
            )
            # Re-infer cell system for the new level (new spawn).
            new_centroid = _agent_centroid_from_frame(frame, agent_fp)
            if new_centroid is None:
                print(f"[level-switch] cannot locate agent in L{cur_lc} frame; stopping")
                break
            cs = infer_cell_system(action_effects, new_centroid)
            cell_actions = {
                a: cs.action_to_cell_delta(dr, dc)
                for a, (dr, dc) in action_effects.items()
            }
            level_key = str(cur_lc)
            walls_cells = {
                (int(cr), int(cc), str(a))
                for cr, cc, a in kb.setdefault("walls_by_level_cells", {}).get(level_key, [])
            }
            blocked_cells = {
                (int(cr), int(cc))
                for cr, cc in kb.setdefault("blocked_targets_by_level_cells", {}).get(level_key, [])
            }
            print(f"[level-switch] L{cur_lc}: cell_size={cs.cell_size} "
                  f"origin=({cs.origin_r},{cs.origin_c}) "
                  f"walls={len(walls_cells)} blocked={len(blocked_cells)}")
            last_seen_lc = cur_lc

        # ---- AUTO-PROBE UNKNOWN DISTINCTIVE COMPONENTS ----
        # Once per level (first entry in this session), try to touch the top-N
        # distinctive non-agent components to observe their effects.  Zero
        # LLM cost; saves TUTOR from burning calls on basic exploration.
        if cur_lc not in probed_levels:
            probed_levels.add(cur_lc)
            cell_bounds = (
                cs.pix_to_cell(0, 0)[0], cs.pix_to_cell(0, 0)[1],
                cs.pix_to_cell(frame.shape[0] - 1, frame.shape[1] - 1)[0],
                cs.pix_to_cell(frame.shape[0] - 1, frame.shape[1] - 1)[1],
            )
            probe_comps = extract_components(frame, min_size=2)
            probe_agent_cell = _agent_cell_from_frame(frame, agent_fp, cs)
            if probe_agent_cell is not None:
                print(f"[probe] L{cur_lc}: auto-probing top distinctive components...")
                probe_results, probe_end_cell, probe_end_obs = _probe_phase(
                    env, obs, cs, agent_fp, probe_agent_cell, probe_comps,
                    cell_actions, walls_cells, cell_bounds,
                    max_probes=3, max_steps_per_probe=10,
                )
                for r in probe_results:
                    note = r.get("note", "")
                    print(f"[probe]   cmp#{r['component_id']} (pal={r['palette']} sz={r['size']}) "
                          f"-> target={r['target_cell']} reached={r.get('reached')} "
                          f"diff={r.get('diff_cells','-')} "
                          f"lc={r.get('lc_before','-')}->{r.get('lc_after','-')} {note}")
                probe_results_by_level[cur_lc] = probe_results
                if probe_end_obs is not None:
                    obs = probe_end_obs
                    frame = _normalise_frame(obs.frame)
                    cur_lc = int(obs.levels_completed)
                # Re-query cur_lc / agent cell etc. below in normal flow.
                # If probes advanced the level, the next iteration will pick up.
                if obs.state.name != "NOT_FINISHED":
                    final_state = obs.state.name
                    break
                if cur_lc != last_seen_lc:
                    # Probes caused a level transition -- loop will re-init.
                    continue

        # ---- REPLAY CHECK ----
        if replay_solved and solutions_data:
            recorded = (solutions_data.get("levels") or {}).get(str(cur_lc))
            if recorded:
                stored_hash = recorded.get("frame_hash_on_entry")
                current_hash = level_entry_hashes[cur_lc]
                if stored_hash is None or stored_hash == current_hash:
                    steps = recorded.get("game_steps") or []
                    print(f"[replay] L{cur_lc}: replaying {len(steps)} recorded steps "
                          f"(hash match: {stored_hash == current_hash})...")
                    replay_lc_before = cur_lc
                    for step in steps:
                        obs = env.step(int(step.replace("ACTION", "")))
                        if obs.state.name != "NOT_FINISHED":
                            break
                        if int(obs.levels_completed) > replay_lc_before:
                            break
                    new_lc = int(obs.levels_completed)
                    if new_lc > replay_lc_before:
                        print(f"[replay] L{cur_lc} -> L{new_lc} in {len(steps)} steps (no TUTOR)")
                        level_advanced = True  # session counts as success
                        cur_level_steps = []
                        turns_used = turn - 1   # replay doesn't count against TUTOR budget
                        # continue the outer for-loop to attempt the next level
                        continue
                    else:
                        print(f"[replay] L{cur_lc} replay did not advance -- falling through to TUTOR")

        # Build turn prompt ingredients (cell-coord primary).  Re-read frame
        # in case replay modified it (replay continues; fall-through reads fresh).
        frame = _normalise_frame(obs.frame)
        comps = extract_components(frame, min_size=2)
        agent_cell = _agent_cell_from_frame(frame, agent_fp, cs)
        agent_pos = _agent_centroid_from_frame(frame, agent_fp)
        agent_comp = next((c for c in comps
                          if agent_fp and c["palette"] == agent_fp[0]
                          and c["size"] == agent_fp[1]
                          and tuple(c["extent"]) == tuple(agent_fp[2])), None)

        # "Tried" = targets we should not naively retry.  We persist
        # UNREACHABLE targets as blocked_cells (cross-session), because
        # "BFS couldn't get there" is enduring information.  We do NOT
        # persist "reached but lc didn't advance" -- that class includes
        # TRIGGERS whose effect is on the NEXT goal visit, not on lc.
        # Still, within this session, we show TUTOR recent reached-inert
        # targets as a hint (not a hard ban).
        tried_cell_set: set[tuple[int, int]] = set(blocked_cells)
        reached_inert: set[tuple[int, int]] = set()
        for h in history:
            tc = h.get("target_cell")
            if not tc:
                continue
            if not h.get("reached"):
                tried_cell_set.add((int(tc[0]), int(tc[1])))
            elif h.get("lc_after", 0) <= h.get("lc_before", 0):
                # Reached but no lc change.  Might be a trigger (effect
                # deferred) -- keep visible but not hard-blocked.
                reached_inert.add((int(tc[0]), int(tc[1])))
        tried = sorted(tried_cell_set)
        inert = sorted(reached_inert)
        tried_text_parts = []
        if tried:
            tried_text_parts.append("  UNREACHABLE (don't retry unless walls change):")
            tried_text_parts.extend(f"    - cell {list(t)}" for t in tried)
        if inert:
            tried_text_parts.append("  REACHED-but-no-lc-advance (may be triggers or scenery):")
            tried_text_parts.extend(f"    - cell {list(t)}" for t in inert)
        tried_text = "\n".join(tried_text_parts) if tried_text_parts else "  (none yet)"

        prior_hyp_text = (
            "\n".join(f"  - {h}" for h in prior_hypotheses_list[-8:])
            if prior_hypotheses_list else "  (none yet -- this is the first session on this level)"
        )

        # Format probe results for this level (if any).
        pr = probe_results_by_level.get(cur_lc, [])
        if pr:
            probe_lines = []
            for r in pr:
                if not r.get("reached", False):
                    probe_lines.append(
                        f"  cmp#{r['component_id']} pal={r['palette']} sz={r['size']} "
                        f"at cell {r['target_cell']}: UNREACHABLE ({r.get('note','')})"
                    )
                    continue
                parts = [
                    f"  cmp#{r['component_id']} pal={r['palette']} sz={r['size']} "
                    f"at cell {r['target_cell']}: REACHED",
                    f"    lc {r.get('lc_before')}->{r.get('lc_after')}",
                    f"    diff_cells={r.get('diff_cells')} "
                    f"(disappeared={r.get('disappeared_n',0)} "
                    f"appeared={r.get('appeared_n',0)} "
                    f"moved={r.get('moved_n',0)})"
                ]
                # Annotate what disappeared if useful.
                for d in (r.get("disappeared") or [])[:2]:
                    parts.append(f"    DISAPPEARED: pal={d['palette']} sz={d['size']} "
                                 f"centroid={d['centroid']}")
                for d in (r.get("appeared") or [])[:2]:
                    parts.append(f"    APPEARED: pal={d['palette']} sz={d['size']} "
                                 f"centroid={d['centroid']}")
                probe_lines.append("\n".join(parts))
            probe_text = "\n".join(probe_lines)
        else:
            probe_text = "  (no probes run this level)"

        user_msg = USER_DISCOVERY_TEMPLATE.format(
            turn            = turn,
            state           = obs.state.name,
            lc              = int(obs.levels_completed),
            win_levels      = int(obs.win_levels),
            actions         = [int(a) for a in obs.available_actions],
            cell_size       = cs.cell_size,
            agent_cell      = list(agent_cell) if agent_cell else "?",
            agent_pal       = agent_comp["palette"] if agent_comp else "?",
            agent_size      = agent_comp["size"] if agent_comp else "?",
            agent_extent    = agent_comp["extent"] if agent_comp else "?",
            action_effects  = _action_effects_text(action_effects, cs),
            components      = _components_summary(comps, agent_fp, cs),
            probe_results   = probe_text,
            prior_hypotheses = prior_hyp_text,
            hist_n          = min(3, len(history)),
            history         = _history_text(history),
            tried_targets   = tried_text,
        )

        # Build annotation list for the rendered image: the TOP-N distinctive
        # non-agent components, labeled with their id so TUTOR can match the
        # picture to the textual COMPONENTS list unambiguously.
        palette_total: dict[int, int] = {}
        for c in comps:
            palette_total[c["palette"]] = palette_total.get(c["palette"], 0) + c["size"]
        ann_enriched = components_in_cells(comps, cs)
        ann_enriched = [
            c for c in ann_enriched
            if not (agent_fp is not None and c["palette"] == agent_fp[0]
                    and c["size"] == agent_fp[1]
                    and tuple(c["extent"]) == tuple(agent_fp[2]))
        ]
        ann_enriched.sort(key=lambda c: (palette_total[c["palette"]], c["size"]))
        # Annotate the top 8 most distinctive components.
        annotations = [
            {"cell": tuple(c["cell"]),
             "label": f"#{c['id']}",
             "color": (255, 255, 0) if palette_total[c["palette"]] <= 12 else (0, 220, 220)}
            for c in ann_enriched[:8]
        ]

        img_b64 = _frame_to_png_b64(
            frame,
            upscale    = 10,
            cell_size  = cs.cell_size,
            origin     = (cs.origin_r, cs.origin_c),
            agent_cell = agent_cell,
            annotations = annotations,
        )

        print(f"\n[turn {turn}] calling TUTOR (agent cell={agent_cell})...")
        t0 = time.time()
        rsp = backends.call_anthropic(
            model      = model,
            system     = SYSTEM_DISCOVERY,
            user       = user_msg,
            image_b64  = img_b64 or None,
            max_tokens = max_tokens,
        )
        latency_ms = int((time.time() - t0) * 1000)
        reply_text = rsp.get("reply", "") or ""
        in_tok  = rsp.get("input_tokens",  0)
        out_tok = rsp.get("output_tokens", 0)
        cost    = rsp.get("cost_usd", 0.0)
        cost_usd_total += cost

        # Capture pre-execution snapshot for the distillation record.
        # advanced_level/reached will be filled AFTER execution so the
        # distillation job can filter by outcome.
        turn_record = {
            "turn":       turn,
            "system":     SYSTEM_DISCOVERY,
            "user":       user_msg,
            "assistant":  reply_text,
            "frame_b64":  _frame_to_png_b64(frame, upscale=2),   # small for storage
            "metadata": {
                "state":                obs.state.name,
                "levels_completed":     int(obs.levels_completed),
                "win_levels":           int(obs.win_levels),
                "agent_pos":            list(agent_pos) if agent_pos else None,
                "action_effects_known": {a: list(e) for a, e in action_effects.items()},
                "walls_known":          len(walls_cells),
                "cell_size":            cs.cell_size,
                "cell_origin":          [cs.origin_r, cs.origin_c],
                "cost_usd":             cost,
                "latency_ms":           latency_ms,
                "input_tokens":         in_tok,
                "output_tokens":        out_tok,
                "turn_start_iso":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
                "strict_mode":          True,
            },
        }
        training_records.append(turn_record)

        try:
            cmd = _extract_json(reply_text)
        except Exception as e:
            print(f"[turn {turn}] bad JSON from TUTOR: {e}")
            break

        # TUTOR is expected to output target_cell (primary).  Keep
        # target_pos accepted as a fallback for robustness.
        args = cmd.get("args", {}) or {}
        target_cell_raw = args.get("target_cell") or args.get("target_pos")
        rationale = cmd.get("rationale", "") or ""
        revise = cmd.get("revise", "") or ""
        print(f"[turn {turn}] rationale: {rationale[:100]}")
        print(f"[turn {turn}] target_cell: {target_cell_raw}  "
              f"(${cost:.3f}, {in_tok}+{out_tok} tok, {latency_ms}ms)")

        if not target_cell_raw or len(target_cell_raw) != 2:
            print(f"[turn {turn}] missing target_cell; stopping")
            break

        target_cell = (int(target_cell_raw[0]), int(target_cell_raw[1]))
        if agent_cell is None:
            print(f"[turn {turn}] agent_cell unknown; stopping")
            break

        # Plan via cell-space BFS.  Bound BFS roughly to the pixel frame.
        cell_bounds = (
            cs.pix_to_cell(0, 0)[0],
            cs.pix_to_cell(0, 0)[1],
            cs.pix_to_cell(frame.shape[0] - 1, frame.shape[1] - 1)[0],
            cs.pix_to_cell(frame.shape[0] - 1, frame.shape[1] - 1)[1],
        )
        path = _bfs_plan_cells(agent_cell, target_cell, cell_actions,
                                walls_cells, cell_bounds=cell_bounds)
        if not path:
            print(f"[turn {turn}] BFS found no path from cell {agent_cell} to {target_cell}")
            history.append({
                "turn":             turn,
                "target_cell":      list(target_cell),
                "reached":          False,
                "cur_cell":         list(agent_cell) if agent_cell else None,
                "rationale":        rationale,
                "revise":           revise,
                "lc_before":        int(obs.levels_completed),
                "lc_after":         int(obs.levels_completed),
                "frame_diff_cells": 0,
                "cost_usd":         cost,
                "note":             "no path",
            })
            continue

        # Execute path; re-plan on wall hits up to _MAX_REROUTES times.
        lc_before = int(obs.levels_completed)
        cur_cell = agent_cell
        exec_frame_before = frame.copy()
        _MAX_REROUTES = 3
        # Per-command step cap.  Keep conservative so a single misguided
        # MOVE_TO doesn't drain budget and trigger a level reset.  At
        # cell_size=5 pixels/step and ~2 budget/action, 15 actions = 30
        # budget consumed per turn max -- leaves headroom for turn budget
        # of 42 in ls20.
        _MAX_TOTAL_STEPS = 15
        total_steps_this_cmd = 0
        reroutes_done = 0
        current_plan = list(path)
        while current_plan and total_steps_this_cmd < _MAX_TOTAL_STEPS:
            action_name = current_plan.pop(0)
            act_int = int(action_name.replace("ACTION", ""))
            obs = env.step(act_int)
            cur_level_steps.append(action_name)   # accumulate for solution recording
            total_steps_this_cmd += 1
            cur_frame = _normalise_frame(obs.frame)
            new_cell = _agent_cell_from_frame(cur_frame, agent_fp, cs)
            if new_cell is None:
                break
            if new_cell == cur_cell:
                # Move failed -- agent didn't budge.
                wall_key = (cur_cell[0], cur_cell[1], action_name)
                wall_hit_counts[wall_key] = wall_hit_counts.get(wall_key, 0) + 1
                # Add to wall set (BFS avoidance) unless we're past the
                # session cap on NEW walls (prevents poisoning).
                if wall_key not in walls_cells and new_walls_recorded < MAX_NEW_WALLS_PER_SESSION:
                    walls_cells.add(wall_key)
                    new_walls_recorded += 1
                elif wall_key not in walls_cells:
                    # Capped: don't re-learn as wall.  BFS may keep trying it.
                    pass
                if reroutes_done < _MAX_REROUTES:
                    reroutes_done += 1
                    current_plan = _bfs_plan_cells(
                        cur_cell, target_cell, cell_actions, walls_cells,
                        cell_bounds=cell_bounds,
                    ) or []
                    continue
                break
            # Move succeeded: (cur_cell, action) is passable.  Invalidate any
            # stale wall at that location (this is how we recover from
            # false-positive walls recorded in earlier sessions).
            success_key = (cur_cell[0], cur_cell[1], action_name)
            successful_moves.add(success_key)
            walls_cells.discard(success_key)
            cur_cell = new_cell
            if obs.state.name != "NOT_FINISHED" or int(obs.levels_completed) > lc_before:
                break
            if not current_plan:
                if cur_cell != target_cell and reroutes_done < _MAX_REROUTES:
                    reroutes_done += 1
                    current_plan = _bfs_plan_cells(
                        cur_cell, target_cell, cell_actions, walls_cells,
                        cell_bounds=cell_bounds,
                    ) or []

        lc_after = int(obs.levels_completed)
        reached = (cur_cell == target_cell)
        frame_after = _normalise_frame(obs.frame)
        diff_cells = int(np.sum(exec_frame_before != frame_after))
        delta = narrate_frame_delta(exec_frame_before, frame_after, agent_fp)
        history.append({
            "turn":             turn,
            "target_cell":      list(target_cell),
            "reached":          reached,
            "cur_cell":         list(cur_cell) if cur_cell else None,
            "rationale":        rationale,
            "revise":           revise,
            "lc_before":        lc_before,
            "lc_after":         lc_after,
            "frame_diff_cells": diff_cells,
            "delta":            delta,
            "cost_usd":         cost,
        })

        # Accumulate TUTOR's hypotheses for cross-session persistence.
        hyp_text = cmd.get("hypotheses")
        if isinstance(hyp_text, str) and hyp_text.strip():
            # Tag with turn for traceability; dedupe against existing list.
            hyp_entry = f"(turn {turn}, L{lc_before}): {hyp_text.strip()[:300]}"
            if hyp_entry not in prior_hypotheses_list:
                prior_hypotheses_list.append(hyp_entry)

        turn_record["metadata"].update({
            "advanced_level":     lc_after > lc_before,
            "target_reached":     reached,
            "frame_diff_cells":   diff_cells,
            "target_cell":        list(target_cell),
            "agent_cell_end":     list(cur_cell) if cur_cell else None,
            "delta_summary": {
                "disappeared_n": len(delta.get("disappeared") or []),
                "appeared_n":    len(delta.get("appeared")    or []),
                "moved_n":       len(delta.get("moved")       or []),
            },
            "parsed_command":  cmd.get("command"),
            "parsed_target_cell": (cmd.get("args") or {}).get("target_cell"),
            "parsed_hypotheses": cmd.get("hypotheses"),
        })
        print(f"[turn {turn}] executed {len(path)}-step path; "
              f"reached={reached} cur_cell={cur_cell} lc={lc_before}->{lc_after} "
              f"diff_cells={diff_cells} walls_cells={len(walls_cells)}")

        if log_path:
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "turn":         turn,
                    "target_cell":  list(target_cell),
                    "rationale":    rationale,
                    "revise":       revise,
                    "path":         path,
                    "reached":      reached,
                    "cur_cell":     list(cur_cell) if cur_cell else None,
                    "lc_before":    lc_before,
                    "lc_after":     lc_after,
                    "diff_cells":   diff_cells,
                    "cost_usd":     cost,
                }) + "\n")

        if lc_after > lc_before:
            # Record this level's solution if not already recorded or if
            # we found a shorter path than before.
            if record_solution and solutions_data is not None and cur_level_steps:
                prev = (solutions_data.get("levels") or {}).get(str(lc_before))
                prev_len = len(prev.get("game_steps", [])) if prev else None
                if prev_len is None or len(cur_level_steps) < prev_len:
                    entry_hash = level_entry_hashes.get(lc_before)
                    solutions_data.setdefault("levels", {})[str(lc_before)] = {
                        "game_steps":        list(cur_level_steps),
                        "step_count":        len(cur_level_steps),
                        "frame_hash_on_entry": entry_hash,
                        "solver":            "strict_mode_tutor",
                        "solved_at":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "session":           session_dir.name if session_dir else None,
                    }
                    _save_strict_solutions(game_id, solutions_data)
                    print(f"[sol] recorded L{lc_before} solution "
                          f"({len(cur_level_steps)} steps, prev={prev_len})")
                else:
                    print(f"[sol] L{lc_before} solve ({len(cur_level_steps)} steps) "
                          f"not shorter than existing ({prev_len}); kept existing")
            cur_level_steps = []   # start fresh for next level's recording

        if lc_after > lc_before:
            level_advanced = True
            print(f"[turn {turn}] *** LEVEL {lc_before} -> {lc_after} ADVANCED ***")
            # Don't break -- continue to attempt the NEXT level with
            # remaining turn budget.  Loop terminates on state != NOT_FINISHED
            # or when max_turns is reached.

    # ---- WRAP-UP ----
    # Only persist UNREACHABLE targets (BFS + execution could not get the
    # agent there).  "Reached but lc didn't advance" is NOT persisted --
    # those are commonly triggers that fire on the next goal visit.
    session_blocked_cells = set(blocked_cells)
    for h in history:
        if not h.get("reached"):   # only unreachable failures
            tc = h.get("target_cell")
            if tc:
                session_blocked_cells.add((int(tc[0]), int(tc[1])))

    # ---- PERSIST KB ----
    kb["action_effects_learned"] = {a: list(e) for a, e in action_effects.items()}
    if agent_fp is not None:
        kb["agent_fingerprint"] = [agent_fp[0], agent_fp[1], list(agent_fp[2])]
    kb["cell_system"] = {
        "cell_size": cs.cell_size,
        "origin":    [cs.origin_r, cs.origin_c],
        "level_key": level_key,   # origin is level-specific; re-infer per new level
    }
    kb.setdefault("walls_by_level_cells", {})
    kb.setdefault("blocked_targets_by_level_cells", {})
    # Merge walls into KB with a 2-strike confirmation rule:
    #   - Remove any walls we OBSERVED the agent move through successfully.
    #   - Add walls that were hit 2+ times this session (confirmed).
    #   - Walls already in KB that were hit 1x stay (already confirmed).
    #   - Walls hit only 1x for the first time this session are NOT persisted
    #     (too uncertain; the failure may have been circumstantial).
    existing_walls = {
        (int(cr), int(cc), str(a))
        for cr, cc, a in kb["walls_by_level_cells"].get(level_key, [])
    }
    # Remove invalidated walls.
    merged_walls = existing_walls - successful_moves
    # Add confirmed walls from this session.
    confirmed_this_session = {k for k, n in wall_hit_counts.items() if n >= 2}
    # For walls hit 1x: only keep if they were in the KB already (implicit re-confirmation).
    once_hit_this_session = {k for k, n in wall_hit_counts.items() if n == 1}
    reconfirmed = once_hit_this_session & existing_walls
    merged_walls |= confirmed_this_session
    # (reconfirmed is already in merged_walls via existing_walls - successful_moves
    #  minus nothing; the 1x hits add no new info but don't reduce either.)
    kb["walls_by_level_cells"][level_key] = sorted([list(w) for w in merged_walls])
    # NOTE: blocked_cells are NOT persisted across sessions (they are
    # "unreachable with current walls", which changes).  Walls persist.
    kb["blocked_targets_by_level_cells"][level_key] = []   # always clear
    # Persist TUTOR's hypotheses across sessions (cap at last 40 to bound growth).
    kb.setdefault("hypotheses_by_level", {})[level_key] = prior_hypotheses_list[-40:]
    n_invalidated = len(existing_walls & successful_moves)
    n_new_confirmed = len(confirmed_this_session - existing_walls)
    print(f"[kb] wall changes: +{n_new_confirmed} confirmed, "
          f"-{n_invalidated} invalidated by successful moves, "
          f"{len(once_hit_this_session)} single hits ignored")
    _save_kb(game_id, kb)
    print(f"[kb] saved: {len(merged_walls)} cell-walls, "
          f"{len(session_blocked_cells)} blocked-cells for L{level_key}")

    result = {
        "game_id":             game_id,
        "outcome":             "LEVEL_ADVANCED" if level_advanced else "NO_ADVANCE",
        "turns_used":          turns_used,
        "initial_lc":          initial_lc,
        "final_lc":            int(obs.levels_completed),
        "final_state":         final_state,
        "cost_usd_total":      round(cost_usd_total, 4),
        "action_effects":      {a: list(e) for a, e in action_effects.items()},
        "cell_size":           cs.cell_size,
        "walls_learned_cells": [list(w) for w in walls_cells],
        "walls_total_in_kb":   len(merged_walls),
        "blocked_cells":       sorted([list(t) for t in session_blocked_cells]),
        "bootstrap_skipped":   not need_bootstrap,
        "history":             history,
    }
    print("\n" + "=" * 60)
    print(f"RESULT: {result['outcome']}  turns={result['turns_used']} "
          f"lc={result['initial_lc']}->{result['final_lc']} "
          f"cost=${result['cost_usd_total']:.4f}")
    print(f"KB: cell_walls={result['walls_total_in_kb']} blocked_cells={len(result['blocked_cells'])}")
    print("=" * 60)

    if manifest_path:
        manifest_path.write_text(json.dumps(result, indent=2))

    # ---- DUMP TRAINING DATA (for distillation of smaller model) ----
    # Matches the legacy run_play.py format so both strict and legacy
    # sessions feed a unified training corpus.  Each record is one
    # (system, user, assistant) triple with outcome metadata.
    if training_records:
        trial_id = session_dir.name if session_dir else time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        td_dir = TRAINING_DATA_DIR / game_id / trial_id
        td_dir.mkdir(parents=True, exist_ok=True)
        (td_dir / "metadata.json").write_text(json.dumps({
            "game_id":            game_id,
            "trial_id":           trial_id,
            "mode":               "strict",
            "outcome":            result["outcome"],
            "final_state":        result["final_state"],
            "levels_completed":   result["final_lc"],
            "initial_lc":         result["initial_lc"],
            "turns":              len(training_records),
            "advancing_turns":    sum(1 for r in training_records
                                     if r["metadata"].get("advanced_level")),
            "total_cost_usd":     round(cost_usd_total, 6),
            "action_effects":     result["action_effects"],
            "session_dir":        str(session_dir) if session_dir else None,
            "created_at":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2), encoding="utf-8")
        for r in training_records:
            (td_dir / f"turn_{r['turn']:03d}.json").write_text(
                json.dumps(r, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        print(f"[distill] wrote {len(training_records)} training records to {td_dir}")

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game",        default="ls20-9607627b")
    ap.add_argument("--max-turns",   type=int, default=8)
    ap.add_argument("--session-dir",
                    default=None,
                    help="Where to write log + manifest; defaults to benchmarks/sessions/trial_<ts>_strict")
    ap.add_argument("--max-tokens",  type=int, default=1500)
    ap.add_argument("--no-record-solution", action="store_true",
                    help="Disable writing strict-mode solutions to JSON")
    ap.add_argument("--no-replay-solved",   action="store_true",
                    help="Disable silent replay of already-solved levels")
    a = ap.parse_args()

    if a.session_dir:
        sd = Path(a.session_dir)
    else:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        sd = HERE.parent / "benchmarks" / "sessions" / f"trial_{ts}_strict"

    result = run_session(
        game_id         = a.game,
        max_turns       = a.max_turns,
        session_dir     = sd,
        max_tokens      = a.max_tokens,
        record_solution = not a.no_record_solution,
        replay_solved   = not a.no_replay_solved,
    )
    sys.exit(0 if result["outcome"] == "LEVEL_ADVANCED" else 1)


if __name__ == "__main__":
    main()
