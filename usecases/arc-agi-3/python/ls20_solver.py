"""
ls20_solver.py — generic BFS solver for LS20 levels.

Takes a `level_meta` dict in the format produced by `record_run.py` (and the
upcoming visual extractor) and returns an action sequence that wins the level.

Schema (level_meta):
    walls:           list[{"x": int, "y": int}]
    targets:         list[{"x": int, "y": int}]
    resets:          list[{"x": int, "y": int}]
    rot_changers:    list[{"x": int, "y": int}]
    color_changers:  list[{"x": int, "y": int}]
    shape_changers:  list[{"x": int, "y": int}]
    push_pads:       list[{"name": str, "x": int, "y": int}]   # name suffix _l/_r/_u/_d/_b
    step_counter:    int      # max counter value
    steps_dec:       int      # decrement per move
    start_shape, start_color, start_rot:  initial state values
    goal_shape,  goal_color,  goal_rot:   target state values

Schema (start_state, optional override):
    {"x": int, "y": int, "shape_idx": int, "color_idx": int, "rot_idx": int, "counter": int}

Actions:
    A1=up    (dy=-5)
    A2=down  (dy=+5)
    A3=left  (dx=-5)
    A4=right (dx=+5)

Coordinate convention:
    Player cells are on the 5-grid: x∈{4,9,...,59}, y∈{0,5,...,55}.
    A sprite at (sx, sy) lives in the player cell (px, py) where
        px = ((sx - 4) // 5) * 5 + 4
        py = (sy // 5) * 5
    Touching a cell that contains a changer/reset/target sprite fires the effect.

Returns:
    list[str] of action names (["ACTION3", "ACTION1", ...]) or None on failure.
"""

from __future__ import annotations
from collections import deque
from typing import Any

VALID_X = set(range(4, 60, 5))   # {4, 9, ..., 59}
VALID_Y = set(range(0, 60, 5))   # {0, 5, ..., 55}

ACTIONS: list[tuple[str, int, int]] = [
    ("ACTION1",  0, -5),  # up
    ("ACTION2",  0, +5),  # down
    ("ACTION3", -5,  0),  # left
    ("ACTION4", +5,  0),  # right
]

# Color index mapping: indices used by the game (color_idx) cycle through
# a 4-element palette per-level. The metadata's goal_color is the actual
# arc palette index (e.g. 9=white). We need to compare via index. The
# meta also stores `color_names` mapping idx→palette_id (per-run, in meta).
# Solver compares color_idx directly when goal is given as an index, otherwise
# we resolve via the palette mapping passed in.


def sprite_to_cell(sx: int, sy: int) -> tuple[int, int]:
    """Map a sprite pixel coordinate to the player cell whose bbox contains it."""
    px = ((sx - 4) // 5) * 5 + 4
    py = (sy // 5) * 5
    return px, py


def _build_static(level_meta: dict, color_palette_map: dict[int, int] | None = None) -> dict:
    """Pre-compute fast lookup tables from level_meta."""
    walls = {sprite_to_cell(s["x"], s["y"]) for s in level_meta.get("walls", [])}

    targets = [sprite_to_cell(s["x"], s["y"]) for s in level_meta.get("targets", [])]

    # Push-pad slides stop one cell short of the target as well as walls.
    slide_stops = walls | set(targets)

    resets = [sprite_to_cell(s["x"], s["y"]) for s in level_meta.get("resets", [])]

    rot_changers   = {sprite_to_cell(s["x"], s["y"]) for s in level_meta.get("rot_changers", [])}
    color_changers = {sprite_to_cell(s["x"], s["y"]) for s in level_meta.get("color_changers", [])}
    shape_changers = {sprite_to_cell(s["x"], s["y"]) for s in level_meta.get("shape_changers", [])}

    # Push pads: each pad sprite (5x5) has its colored row/col on a single edge,
    # which determines its visible pixel offset within the sprite anchor:
    #   _b (push down): colored row 0       offset (0, 0)
    #   _t (push up):   colored row 4       offset (0, 4)
    #   _r (push right):colored col 0       offset (0, 0)
    #   _l (push left): colored col 4       offset (4, 0)
    # The pad CELL is the player cell containing the visible pixel; the trigger
    # cell is pad_cell + push_direction. Slide stops just before wall/target.
    PAD_INFO = {
        "_r": (( 5,  0), (0, 0)),
        "_l": ((-5,  0), (4, 0)),
        "_b": (( 0,  5), (0, 0)),
        "_d": (( 0,  5), (0, 0)),
        "_t": (( 0, -5), (0, 4)),
        "_u": (( 0, -5), (0, 4)),
    }
    pads: dict[tuple[int, int], tuple[int, int]] = {}
    for p in level_meta.get("push_pads", []):
        sx, sy = p["x"], p["y"]
        info = PAD_INFO.get(p.get("name", "")[-2:])
        if info is None:
            continue
        (dx, dy), (ox, oy) = info
        pad_cell = sprite_to_cell(sx + ox, sy + oy)
        trigger = (pad_cell[0] + dx, pad_cell[1] + dy)
        if trigger[0] not in VALID_X or trigger[1] not in VALID_Y:
            continue
        pads[trigger] = _slide(trigger, dx, dy, slide_stops)

    # Carriers: simulate each, store per-step changer cells + period.
    # Exclude each carrier's static initial cell from the static changer set.
    sim_carriers: list[dict] = []
    for c in level_meta.get("carriers", []):
        cells, period = _simulate_carrier(c)
        sim_carriers.append({"kind": c["kind"], "cells": cells, "period": period})
        init_cell = sprite_to_cell(c["init_x"], c["init_y"])
        if c["kind"] == "rot":     rot_changers.discard(init_cell)
        elif c["kind"] == "color": color_changers.discard(init_cell)
        elif c["kind"] == "shape": shape_changers.discard(init_cell)

    from math import gcd
    global_period = 1
    for c in sim_carriers:
        global_period = global_period * c["period"] // gcd(global_period, c["period"])

    return {
        "walls":          walls,
        "targets":        targets,
        "resets":         resets,
        "rot_changers":   rot_changers,
        "color_changers": color_changers,
        "shape_changers": shape_changers,
        "pads":           pads,
        "ctr_max":        level_meta.get("step_counter", 42),
        "steps_dec":      level_meta.get("steps_dec", 2),
        "carriers":       sim_carriers,
        "global_period":  global_period,
    }


# Carrier direction encoding (matches game's nakogfhyus): 0=down, 1=right, 2=up, 3=left
_CARRIER_DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def _simulate_carrier(c: dict, max_steps: int = 400) -> tuple[list[tuple[int, int]], int]:
    """Simulate carrier+changer trajectory deterministically. Returns
    (cells_per_step, period) where cells_per_step[i] = (cell_x, cell_y) of the
    changer AFTER i carrier-steps (index 0 = initial position before any move).
    Period is detected by state (sx,sy,dir) repetition; if no period found,
    returns trajectory as-is with period == len(trajectory)."""
    cx, cy = c["carrier_x"], c["carrier_y"]
    cw, ch = c["carrier_w"], c["carrier_h"]
    pixels = c["pixels"]  # 2D list, indexed [py][px]
    sx, sy = c["init_x"], c["init_y"]
    d = c["init_dir"]
    cell = c["cell"]

    def valid(nx: int, ny: int) -> bool:
        if not (cx <= nx < cx + cw and cy <= ny < cy + ch):
            return False
        py, px = ny - cy, nx - cx
        return pixels[py][px] >= 0

    states: list[tuple[int, int, int]] = [(sx, sy, d)]
    cells: list[tuple[int, int]] = [sprite_to_cell(sx, sy)]
    seen: dict[tuple[int, int, int], int] = {(sx, sy, d): 0}
    period = 0
    for _ in range(max_steps):
        moved = False
        for delta in (0, -1, 1, 2):
            nd = (d + delta) % 4
            dxc, dyc = _CARRIER_DIRS[nd]
            nx, ny = sx + dxc * cell, sy + dyc * cell
            if valid(nx, ny):
                sx, sy, d = nx, ny, nd
                moved = True
                break
        # If !moved, carrier stays put (still advances step index).
        cells.append(sprite_to_cell(sx, sy))
        key = (sx, sy, d)
        if key in seen:
            period = len(states) - seen[key]
            break
        seen[key] = len(states)
        states.append(key)
    if period == 0:
        period = len(cells)
    return cells, period


def _slide(start: tuple[int, int], dx: int, dy: int,
           walls: set[tuple[int, int]]) -> tuple[int, int]:
    """Slide from `start` by (dx,dy) until just before a wall or grid edge."""
    x, y = start
    while True:
        nx, ny = x + dx, y + dy
        if nx not in VALID_X or ny not in VALID_Y or (nx, ny) in walls:
            return (x, y)
        x, y = nx, ny


def solve_ls20(level_meta: dict, start_state: dict | None = None,
               *, color_idx_map: dict[int, int] | None = None,
               rot_values: list[int] | None = None,
               max_states: int = 10_000_000) -> list[str] | None:
    """
    BFS over (x, y, shape_idx, color_idx, rot_idx, resets_bitmask, counter).

    Returns the winning action sequence or None.
    """
    static = _build_static(level_meta)

    walls   = static["walls"]
    targets = static["targets"]
    resets  = static["resets"]
    rot_ch  = static["rot_changers"]
    col_ch  = static["color_changers"]
    sh_ch   = static["shape_changers"]
    pads    = static["pads"]
    ctr_max = static["ctr_max"]
    dec     = static["steps_dec"]
    sim_carriers = static["carriers"]
    global_period = static["global_period"]

    def dyn_changers(sidx: int) -> tuple[set, set, set]:
        """Dynamic changer cells contributed by carriers at carrier-step index sidx."""
        dr: set = set(); dc: set = set(); ds: set = set()
        for c in sim_carriers:
            cell = c["cells"][sidx % c["period"]]
            if c["kind"] == "rot":     dr.add(cell)
            elif c["kind"] == "color": dc.add(cell)
            elif c["kind"] == "shape": ds.add(cell)
        return dr, dc, ds

    if not targets:
        return None

    # Resolve per-target goals. Single-target levels store scalars; multi-target
    # store lists of equal length to `targets`. Normalize to lists.
    if rot_values is None:
        rot_values = [0, 90, 180, 270]

    def _as_list(v, n):
        return list(v) if isinstance(v, (list, tuple)) else [v] * n

    n_t = len(targets)
    goal_rot_raw = _as_list(level_meta["goal_rot"], n_t)
    goal_col_raw = _as_list(level_meta["goal_color"], n_t)
    goal_sh_raw  = _as_list(level_meta["goal_shape"], n_t)

    goal_rot_idxs = [rot_values.index(r) if r in rot_values else 0 for r in goal_rot_raw]
    if color_idx_map is not None:
        goal_col_idxs = [next((i for i, v in color_idx_map.items() if v == c), 0)
                         for c in goal_col_raw]
    else:
        goal_col_idxs = goal_col_raw[:]
    goal_sh_idxs  = goal_sh_raw[:]

    # Map cell -> target index for fast lookup.
    target_idx_of = {t: i for i, t in enumerate(targets)}
    all_done_mask = (1 << n_t) - 1

    # Initial state
    if start_state is not None:
        x0 = start_state["player_x"]
        y0 = start_state["player_y"]
        sh0 = start_state["shape_idx"]
        col0 = start_state["color_idx"]
        rot0 = start_state["rot_idx"]
        ctr0 = start_state.get("counter", ctr_max)
    else:
        # Fall back to meta — but meta does not store start_x/start_y, only
        # start_shape/color/rot. Caller must supply position via start_state.
        return None

    n_resets = len(resets)
    start_key = (x0, y0, sh0, col0, rot0, 0, ctr0, 0, 0)  # +tmask

    # BFS
    parent: dict[tuple, tuple | None] = {start_key: None}
    action_taken: dict[tuple, str | None] = {start_key: None}
    queue: deque = deque([start_key])
    explored = 0

    goal_state: tuple | None = None

    while queue:
        state = queue.popleft()
        explored += 1
        if explored > max_states:
            return None
        x, y, sh, col, rot, rmask, ctr, sidx, tmask = state

        # Carriers advance once per attempted action (even if blocked the
        # game undoes the move; here we just compute the post-advance cells
        # used for changer collision and only commit step_idx on accepted moves).
        next_sidx = (sidx + 1) % global_period if global_period > 1 else 0
        dyn_rot, dyn_col, dyn_sh = dyn_changers(next_sidx) if sim_carriers else (set(), set(), set())
        eff_rot_ch = rot_ch | dyn_rot
        eff_col_ch = col_ch | dyn_col
        eff_sh_ch  = sh_ch  | dyn_sh

        for aname, dx, dy in ACTIONS:
            nx, ny = x + dx, y + dy
            if nx not in VALID_X or ny not in VALID_Y:
                continue
            if (nx, ny) in walls:
                continue

            # Counter / reset handling
            new_rmask = rmask
            new_ctr = ctr
            counter_was_reset = False
            for i, rcell in enumerate(resets):
                if not (rmask & (1 << i)) and (nx, ny) == rcell:
                    new_rmask |= (1 << i)
                    new_ctr = ctr_max
                    counter_was_reset = True
                    break
            if not counter_was_reset:
                new_ctr = ctr - dec
                if new_ctr <= 0:
                    continue

            new_rot = (rot + 1) % 4 if (nx, ny) in eff_rot_ch else rot
            new_col = (col + 1) % 4 if (nx, ny) in eff_col_ch else col
            new_sh  = (sh  + 1) % 6 if (nx, ny) in eff_sh_ch  else sh

            # Push pads — teleport, then changers fire at the slide destination
            # too (game runs txnfzvzetn after the pad slide animation completes).
            if (nx, ny) in pads:
                nx, ny = pads[(nx, ny)]
                if (nx, ny) in eff_rot_ch: new_rot = (new_rot + 1) % 4
                if (nx, ny) in eff_col_ch: new_col = (new_col + 1) % 4
                if (nx, ny) in eff_sh_ch:  new_sh  = (new_sh  + 1) % 6
                for i, rcell in enumerate(resets):
                    if not (new_rmask & (1 << i)) and (nx, ny) == rcell:
                        new_rmask |= (1 << i)
                        new_ctr = ctr_max
                        break

            # Target handling: if (nx,ny) is an uncleared target, must match its
            # goal — otherwise blocked. Cleared targets are free to walk through.
            new_tmask = tmask
            ti = target_idx_of.get((nx, ny))
            if ti is not None and not (tmask & (1 << ti)):
                if (new_rot == goal_rot_idxs[ti] and new_col == goal_col_idxs[ti]
                        and new_sh == goal_sh_idxs[ti]):
                    new_tmask |= (1 << ti)
                else:
                    continue  # blocked: wrong state on uncleared target

            # Win check (all targets cleared)
            if new_tmask == all_done_mask:
                key = (nx, ny, new_sh, new_col, new_rot, new_rmask, new_ctr, next_sidx, new_tmask)
                parent[key] = state
                action_taken[key] = aname
                goal_state = key
                break

            key = (nx, ny, new_sh, new_col, new_rot, new_rmask, new_ctr, next_sidx, new_tmask)
            if key not in parent:
                parent[key] = state
                action_taken[key] = aname
                queue.append(key)

        if goal_state is not None:
            break

    if goal_state is None:
        return None

    # Reconstruct path
    path: list[str] = []
    cur: tuple | None = goal_state
    while cur is not None and action_taken.get(cur) is not None:
        path.append(action_taken[cur])  # type: ignore[arg-type]
        cur = parent[cur]
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Game-internals adapters (used at runtime by ensemble.py).
#
# These mirror record_run.py:_level_meta / _game_state but are colocated here
# so the solver has no upstream dependency. The dict format is the contract;
# a future visual extractor must produce dicts with the same shape.
# ---------------------------------------------------------------------------

def level_meta_from_game(game, level_idx: int) -> dict:
    """Extract structured level metadata from an arc-agi-3 game object."""
    lv   = game._levels[level_idx]
    data = lv._data if hasattr(lv, "_data") else {}
    walls, targets, resets = [], [], []
    sh_ch, col_ch, rot_ch, pads = [], [], [], []
    carriers_raw = []  # xfmluydglp sprites: (x,y,w,h,pixels)
    changer_sprites = []  # (x, y, kind) for ttfwljgohq/soyhouuebz/rhsxkxzdjz
    for s in lv._sprites:
        tags = s.tags or []
        pos  = {"x": s.x, "y": s.y}
        if "ihdgageizm" in tags:
            walls.append(pos)
        elif "rjlbuycveu" in tags:
            targets.append(pos)
        elif "npxgalaybz" in tags:
            resets.append(pos)
        elif "ttfwljgohq" in tags:
            sh_ch.append(pos); changer_sprites.append((s.x, s.y, "shape"))
        elif "soyhouuebz" in tags:
            col_ch.append(pos); changer_sprites.append((s.x, s.y, "color"))
        elif "rhsxkxzdjz" in tags:
            rot_ch.append(pos); changer_sprites.append((s.x, s.y, "rot"))
        elif "gbvqrjtaqo" in tags:
            pads.append({"name": s.name, "x": s.x, "y": s.y})
        elif "xfmluydglp" in tags:
            carriers_raw.append(s)
    # Pair carriers with the changers whose initial pos sits inside the carrier bbox.
    carriers: list[dict] = []
    for car in carriers_raw:
        for (cx, cy, kind) in changer_sprites:
            if car.x <= cx < car.x + car.width and car.y <= cy < car.y + car.height:
                carriers.append({
                    "carrier_x": car.x, "carrier_y": car.y,
                    "carrier_w": car.width, "carrier_h": car.height,
                    "pixels": car.pixels.tolist(),
                    "init_x": cx, "init_y": cy,
                    "init_dir": 0, "cell": 5,
                    "kind": kind,
                })
                break
    return {
        "level":          level_idx + 1,
        "step_counter":   data.get("StepCounter", 42),
        "steps_dec":      data.get("StepsDecrement", 2),
        "goal_shape":     data.get("kvynsvxbpi"),
        "goal_color":     data.get("GoalColor"),
        "goal_rot":       data.get("GoalRotation"),
        "start_shape":    data.get("StartShape"),
        "start_color":    data.get("StartColor"),
        "start_rot":      data.get("StartRotation"),
        "fog":            data.get("Fog", False),
        "walls":          walls,
        "targets":        targets,
        "resets":         resets,
        "shape_changers": sh_ch,
        "color_changers": col_ch,
        "rot_changers":   rot_ch,
        "push_pads":      pads,
        "carriers":       carriers,
    }


def state_from_game(game) -> dict:
    return {
        "player_x":   game.gudziatsk.x,
        "player_y":   game.gudziatsk.y,
        "shape_idx":  game.fwckfzsyc,
        "color_idx":  game.hiaauhahz,
        "rot_idx":    game.cklxociuu,
        "counter":    game._step_counter_ui.current_steps,
        "counter_max":game._step_counter_ui.osgviligwp,
        "lives":      game.aqygnziho,
    }


def color_idx_map_from_game(game) -> dict[int, int]:
    """Map color_idx (0..n-1) → arc palette id used in goal_color metadata."""
    return {i: v for i, v in enumerate(game.tnkekoeuk)}


def plan_ls20_level(game, level_idx: int) -> list[str] | None:
    """Convenience: extract meta+state from game and return BFS action plan."""
    meta  = level_meta_from_game(game, level_idx)
    state = state_from_game(game)
    cmap  = color_idx_map_from_game(game)
    return solve_ls20(meta, state, color_idx_map=cmap)


# ---------------------------------------------------------------------------
# Offline validation harness
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json, sys
    from pathlib import Path

    rec_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "../.tmp/playlogs/presolve_record.json")
    rec = json.loads(rec_path.read_text())

    color_idx_map = {int(k): v for k, v in rec["meta"].get("color_names", {}).items()}
    rot_values = rec["meta"].get("rot_values", [0, 90, 180, 270])

    # Build a list of (level_idx, start_state_at_level_entry).
    # An entry state for level N is either: the state_before of the first step
    # whose level == N, or the state_after of a step where level_advanced fired
    # taking us into N.
    level_entry_states: dict[int, dict] = {}
    for s in rec["steps"]:
        lvl = s["level"]
        if lvl not in level_entry_states:
            level_entry_states[lvl] = s["state_before"]
        if s.get("level_advanced"):
            level_entry_states[lvl + 1] = s["state_after"]

    for lv in rec["levels"]:
        n = lv["level"]
        if n not in level_entry_states:
            print(f"L{n}: skip (no entry state in steps)")
            continue
        start = level_entry_states[n]
        path = solve_ls20(lv, start, color_idx_map=color_idx_map, rot_values=rot_values)
        if path is None:
            print(f"L{n}: NO SOLUTION")
        else:
            print(f"L{n}: solved in {len(path)} actions")
