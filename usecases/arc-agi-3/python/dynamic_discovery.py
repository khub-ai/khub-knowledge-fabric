"""
dynamic_discovery.py — observation-based game object discovery for ARC-AGI-3.

Discovers interactive object positions and roles without reading source code by
analysing:

  1. Frame data    — identifies non-walkable, non-player cells as candidates.
  2. Action history — classifies each candidate from observed behaviour:
       diff > 80 at position, same level  → rotation changer (RC)
       levels_after > levels_before       → win gate
       candidate disappeared + was visited → consumable/ring
  3. Concept bindings — pre-classifies unvisited candidates by color role
       learned from prior play (e.g. "color 11 = step_counter_ring").

This replaces the hardcoded positional entries in game_knowledge.json, so the
system can handle a "slightly altered" game (same mechanics, different layout)
without re-reading source code.

Public API
----------
  find_nonwalkable_candidates(frame, walkable_colors, player_colors)
      → dict {(col, row): color}

  build_level_model(initial_frame, current_frame, action_history, ...)
      → LevelModel

  compute_dynamic_waypoints(model, player_pos, ...)
      → (waypoints: list[(col,row)], extra_passable: set[(col,row)])

  load_discovered_knowledge(path) → dict
  save_discovered_knowledge(path, data) → None
  update_discovered_knowledge(path, game_id, level, rc_visits, color_roles) → None
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Curiosity / Hypothesis system
# ---------------------------------------------------------------------------

class HypothesisStore:
    """Tracks curiosity-driven hypotheses arising from visual similarities.

    On initialisation (or reset), scans the initial frame for pairs of objects
    that look similar but differ in size/state — these are "unusual coincidences"
    that may signal a match-to-win sub-goal.

    Each hypothesis tracks how the Jaccard distance between the pair evolves
    as the agent acts.  Decreasing distance = the pair is converging.
    Distance == 0 = confirmed match.

    Usage:
        store = HypothesisStore()
        store.seed(initial_frame)           # call once per level
        new_confirmed = store.update(frame, diff)  # call after each step with diff > baseline
        goals = store.active_subgoals()     # human-readable strings for prompt injection
    """

    # Walk-baseline diff.  Only update hypotheses on steps that exceed this,
    # so normal movement doesn't trigger expensive rescans.
    DIFF_THRESHOLD = 60

    def __init__(self) -> None:
        # List of live hypotheses.  Each entry:
        # {
        #   "color_small": int, "color_large": int,
        #   "distance_initial": float, "distance_current": float,
        #   "best_transform": str,
        #   "centroid_small": (row, col), "centroid_large": (row, col),
        #   "confirmed": bool,
        #   "history": [(step, distance), ...]   # last N observations
        # }
        self._hypotheses: list[dict] = []
        self._step: int = 0

    def reset(self) -> None:
        """Clear all hypotheses (call at the start of each level)."""
        self._hypotheses.clear()
        self._step = 0

    def seed(self, frame: list[list[int]]) -> None:
        """Scan frame for similar object pairs and seed hypotheses.

        Safe to call multiple times — existing hypotheses with the same
        (color_small, color_large) key are not duplicated.
        """
        from object_tracker import find_similar_object_pairs
        pairs = find_similar_object_pairs(frame)
        # Key includes rounded centroid to distinguish same-color pairs at
        # different positions (e.g. two blue objects of different sizes).
        def _hyp_key(p: dict) -> tuple:
            cs = tuple(round(x) for x in p["centroid_small"])
            cl = tuple(round(x) for x in p["centroid_large"])
            return (p["color_small"], p["color_large"], cs, cl)

        existing_keys = {
            _hyp_key({
                "color_small": h["color_small"], "color_large": h["color_large"],
                "centroid_small": h["centroid_small"], "centroid_large": h["centroid_large"],
            })
            for h in self._hypotheses
        }
        for p in pairs:
            key = _hyp_key(p)
            if key in existing_keys:
                continue
            self._hypotheses.append({
                "color_small":       p["color_small"],
                "color_large":       p["color_large"],
                "distance_initial":  p["distance"],
                "distance_current":  p["distance"],
                "best_transform":    p["best_transform"],
                "centroid_small":    p["centroid_small"],
                "centroid_large":    p["centroid_large"],
                "confirmed":         p["distance"] == 0.0,
                "history":           [(self._step, p["distance"])],
            })
            existing_keys.add(key)
        if pairs:
            print(
                f"  [CURIOSITY] Seeded {len(pairs)} similar-object hypothesis(es): "
                + ", ".join(
                    f"color{p['color_small']}<->color{p['color_large']} "
                    f"dist={p['distance']:.3f} via {p['best_transform']}"
                    for p in pairs
                ),
                flush=True,
            )

    def update(self, frame: list[list[int]], diff: int) -> list[dict]:
        """Re-measure all active hypotheses against the current frame.

        Called after any step where diff > DIFF_THRESHOLD.
        Returns list of newly confirmed hypotheses (distance just reached 0).
        """
        if diff <= self.DIFF_THRESHOLD or not self._hypotheses:
            return []
        self._step += 1
        from object_tracker import detect_objects, extract_subgrid, compare_shapes
        objs_by_color: dict[int, list] = {}
        for o in detect_objects(frame):
            objs_by_color.setdefault(o.color, []).append(o)

        newly_confirmed: list[dict] = []
        for hyp in self._hypotheses:
            if hyp["confirmed"]:
                continue
            s_objs = objs_by_color.get(hyp["color_small"], [])
            l_objs = objs_by_color.get(hyp["color_large"], [])
            if not s_objs or not l_objs:
                continue
            # When colors differ: pick largest of each color.
            # When same color: pick the object closest in position to each seeded centroid
            # (to avoid mixing up the large reference with the small RC indicator).
            if hyp["color_small"] != hyp["color_large"]:
                s_obj = max(s_objs, key=lambda o: o.size)
                l_obj = max(l_objs, key=lambda o: o.size)
            else:
                # Same color — distinguish by proximity to seeded centroids
                def _closest(objs, target_centroid):
                    tr, tc = target_centroid
                    return min(objs, key=lambda o: (o.centroid[0]-tr)**2+(o.centroid[1]-tc)**2)
                s_obj = _closest(s_objs, hyp["centroid_small"])
                l_obj = _closest(l_objs, hyp["centroid_large"])
                if s_obj is l_obj:
                    continue  # can't distinguish — skip this update
            mask_s = extract_subgrid(frame, s_obj.bbox, foreground_color=hyp["color_small"])
            mask_l = extract_subgrid(frame, l_obj.bbox, foreground_color=hyp["color_large"])
            result = compare_shapes(mask_s, mask_l)
            prev_dist = hyp["distance_current"]
            hyp["distance_current"] = result["best_distance"]
            hyp["best_transform"] = result["best_transform"]
            hyp["centroid_small"] = s_obj.centroid
            hyp["centroid_large"] = l_obj.centroid
            hyp["history"].append((self._step, result["best_distance"]))
            if len(hyp["history"]) > 20:
                hyp["history"] = hyp["history"][-20:]
            print(
                f"  [CURIOSITY] Hypothesis color{hyp['color_small']}<->color{hyp['color_large']}: "
                f"dist {prev_dist:.3f} -> {result['best_distance']:.3f} "
                f"(transform={result['best_transform']})",
                flush=True,
            )
            if result["best_distance"] == 0.0:
                hyp["confirmed"] = True
                newly_confirmed.append(hyp)
                print(
                    f"  [CURIOSITY] *** CONFIRMED MATCH: "
                    f"color{hyp['color_small']} matches color{hyp['color_large']} "
                    f"via {result['best_transform']} — probing blocked positions",
                    flush=True,
                )
        return newly_confirmed

    def active_subgoals(self) -> list[str]:
        """Return human-readable sub-goal strings for prompt injection.

        Only returns goals for hypotheses that show evidence of convergence
        (distance decreased from initial, or already confirmed).
        """
        goals: list[str] = []
        for hyp in self._hypotheses:
            if hyp["confirmed"]:
                goals.append(
                    f"CONFIRMED: The small object (color{hyp['color_small']} near "
                    f"row={hyp['centroid_small'][0]:.0f},col={hyp['centroid_small'][1]:.0f}) "
                    f"now matches the large reference (color{hyp['color_large']} near "
                    f"row={hyp['centroid_large'][0]:.0f},col={hyp['centroid_large'][1]:.0f}). "
                    f"Check whether blocked positions are now accessible."
                )
            elif hyp["distance_current"] < hyp["distance_initial"] - 0.05:
                goals.append(
                    f"HYPOTHESIS (converging): Visiting the small object "
                    f"(color{hyp['color_small']} near "
                    f"row={hyp['centroid_small'][0]:.0f},col={hyp['centroid_small'][1]:.0f}) "
                    f"is bringing it closer to matching the large reference "
                    f"(color{hyp['color_large']}). "
                    f"Current distance={hyp['distance_current']:.3f} "
                    f"(started at {hyp['distance_initial']:.3f}). "
                    f"Continue visiting to reach full match."
                )
            elif hyp["distance_initial"] <= 0.35:
                goals.append(
                    f"HYPOTHESIS (untested): Small object (color{hyp['color_small']} near "
                    f"row={hyp['centroid_small'][0]:.0f},col={hyp['centroid_small'][1]:.0f}) "
                    f"resembles large object (color{hyp['color_large']}) — "
                    f"may be a rotatable/matchable element. "
                    f"Try visiting it to see if it changes."
                )
        return goals

    def has_confirmed(self) -> bool:
        return any(h["confirmed"] for h in self._hypotheses)

    def all_hypotheses(self) -> list[dict]:
        return list(self._hypotheses)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LevelModel:
    """
    Discovered knowledge about a single level, built incrementally.

    Attributes
    ----------
    candidates : {(col,row): color}  — all non-walkable, non-player cells from
                  the initial frame.
    rc_positions : list of confirmed rotation-changer positions.
    ring_positions : unvisited step-counter rings still present in current frame.
    consumed_rings : rings already collected (disappeared from frame).
    win_gate : confirmed win-target position, or None if not yet discovered.
    ring_refuels : dict mapping ring position -> effective budget (in moves)
                   remaining AFTER visiting that ring.  Populated by the caller
                   once step_budget is known; defaults to 0 (unknown).
                   Different rings may restore different amounts (partial refuels,
                   multi-tier rings, etc.).
    """
    candidates:     dict  = field(default_factory=dict)
    rc_positions:   list  = field(default_factory=list)
    ring_positions: list  = field(default_factory=list)
    consumed_rings: list  = field(default_factory=list)
    win_gate: Optional[tuple] = None
    ring_refuels:   dict  = field(default_factory=dict)  # pos -> effective budget after visit

    @property
    def has_rc(self) -> bool:
        return bool(self.rc_positions)

    @property
    def has_win_gate(self) -> bool:
        return self.win_gate is not None

    @property
    def is_navigable(self) -> bool:
        """Enough info to plan a win route (at minimum an RC and a win gate)."""
        return self.has_rc and self.has_win_gate

    def unvisited_candidates(self, visited: set) -> list:
        """Return candidates not yet visited and not yet classified."""
        classified = (
            set(map(tuple, self.rc_positions))
            | set(map(tuple, self.ring_positions))
            | set(map(tuple, self.consumed_rings))
            | ({tuple(self.win_gate)} if self.win_gate else set())
        )
        return [pos for pos in self.candidates
                if pos not in classified and pos not in visited]

    def __repr__(self) -> str:
        return (
            f"LevelModel(rc={self.rc_positions}, "
            f"rings={self.ring_positions}+{self.consumed_rings}consumed, "
            f"win={self.win_gate}, "
            f"candidates={len(self.candidates)})"
        )


# ---------------------------------------------------------------------------
# Game hypothesis — dynamic inference of game mechanics
# ---------------------------------------------------------------------------

@dataclass
class GameHypothesis:
    """
    Prior + observed values for fundamental game mechanics.

    The ``prior_*`` fields are seeded from game_knowledge.json (human-supplied
    or from a previous run on this game).  The ``obs_*`` fields are filled in
    at runtime by inference functions that observe frame data and action
    history.  The ``effective_*`` properties return the observed value when
    available, falling back to the prior.

    This design means game_knowledge.json values are *hypotheses*, not facts:
    they work for the known version of the game but will be overridden if
    observation disagrees.
    """
    prior_player_colors:   set
    prior_walkable_colors: set
    prior_step_size:       int
    prior_action_map:      dict

    obs_player_colors:   Optional[set]  = None
    obs_walkable_colors: Optional[set]  = None
    obs_step_size:       Optional[int]  = None
    obs_action_map:      Optional[dict] = None

    @property
    def effective_player_colors(self) -> set:
        return self.obs_player_colors if self.obs_player_colors is not None \
               else self.prior_player_colors

    @property
    def effective_walkable_colors(self) -> set:
        if self.obs_walkable_colors is not None:
            # P6: merge observed with prior.  Observation confirms some
            # colors are walkable but doesn't contradict unobserved ones
            # (the player simply hasn't visited them yet).  Wall detection
            # is the mechanism that REMOVES a color from walkable.
            return self.prior_walkable_colors | self.obs_walkable_colors
        return self.prior_walkable_colors

    @property
    def effective_step_size(self) -> int:
        return self.obs_step_size if self.obs_step_size is not None \
               else self.prior_step_size

    @property
    def effective_action_map(self) -> dict:
        if self.obs_action_map is not None:
            # P6: merge observed directions with prior so that unobserved
            # directions keep their prior mapping (partial inference is
            # common in early steps when only 2 of 4 directions are used).
            merged = dict(self.prior_action_map)
            merged.update(self.obs_action_map)
            return merged
        return self.prior_action_map

    def __repr__(self) -> str:
        _src = lambda obs: "OBS" if obs is not None else "PRI"
        return (
            f"GameHypothesis("
            f"player_colors={self.effective_player_colors}"
            f"[{_src(self.obs_player_colors)}], "
            f"walkable={self.effective_walkable_colors}"
            f"[{_src(self.obs_walkable_colors)}], "
            f"step_size={self.effective_step_size}"
            f"[{_src(self.obs_step_size)}], "
            f"actions={list(self.effective_action_map.keys())}"
            f"[{_src(self.obs_action_map)}])"
        )


def infer_step_size_from_positions(
    history: list[dict],
    min_samples: int = 3,
) -> Optional[int]:
    """
    Infer the movement grid step size from the GCD of observed displacements.

    Scans action_history for consecutive steps where player_pos changed and
    computes the GCD of all non-zero displacement magnitudes.  Returns None
    when fewer than ``min_samples`` valid moves are available.
    """
    from math import gcd

    displacements: list[int] = []
    prev_pos: Optional[tuple] = None

    for step in history:
        pos = step.get("player_pos")
        if pos is None:
            continue
        pos = tuple(pos)
        if prev_pos is not None and pos != prev_pos:
            dx = abs(pos[0] - prev_pos[0])
            dy = abs(pos[1] - prev_pos[1])
            if dx > 0:
                displacements.append(dx)
            if dy > 0:
                displacements.append(dy)
        prev_pos = pos

    if len(displacements) < min_samples:
        return None

    result = displacements[0]
    for d in displacements[1:]:
        result = gcd(result, d)
    return result if result > 0 else None


def infer_action_directions_from_history(
    history: list[dict],
    step_size_hint: int = 5,
) -> dict[str, str]:
    """
    Infer the action_map {direction: action_name} from observed displacements.

    For each action taken, collects the resulting displacement (prev_pos →
    current_pos).  Maps the dominant displacement to a canonical direction
    (UP/DOWN/LEFT/RIGHT).  Returns an empty dict when fewer than 2 directions
    can be confirmed.
    """
    from collections import Counter, defaultdict

    action_deltas: dict[str, list] = defaultdict(list)
    prev_pos: Optional[tuple] = None

    for step in history:
        pos = step.get("player_pos")
        act = step.get("action")
        if pos is not None and act is not None and prev_pos is not None:
            pos = tuple(pos)
            if pos != prev_pos:
                action_deltas[act].append(
                    (pos[0] - prev_pos[0], pos[1] - prev_pos[1])
                )
        if pos is not None:
            prev_pos = tuple(pos)

    canonical: dict[tuple, str] = {
        (0, -step_size_hint): "UP",
        (0,  step_size_hint): "DOWN",
        (-step_size_hint, 0): "LEFT",
        ( step_size_hint, 0): "RIGHT",
    }

    action_map: dict[str, str] = {}
    for act, deltas in action_deltas.items():
        if not deltas:
            continue
        dominant = Counter(deltas).most_common(1)[0][0]
        direction = canonical.get(dominant)
        if direction:
            action_map[direction] = act

    return action_map if len(action_map) >= 2 else {}


def infer_walkable_from_visits(
    current_frame: list[list[int]],
    visited_positions: set[tuple],
    player_colors: set[int],
    step_size: int,
    min_visits: int = 3,
) -> Optional[set[int]]:
    """
    Infer walkable colors from the colors observed at visited grid positions.

    Any color that appears at >= ``min_visits`` distinct visited positions
    (excluding player colors and color 0) is classified as walkable.
    Returns None when insufficient data is available.
    """
    color_count: dict[int, int] = {}
    rows = len(current_frame)
    cols = len(current_frame[0]) if rows else 0

    for pos in visited_positions:
        col, row = pos
        if not (0 <= row < rows and 0 <= col < cols):
            continue
        color = current_frame[row][col]
        if color in player_colors or color == 0:
            continue
        color_count[color] = color_count.get(color, 0) + 1

    walkable = {c for c, cnt in color_count.items() if cnt >= min_visits}
    return walkable if walkable else None


def infer_player_colors_from_diff(
    frame_before: list[list[int]],
    frame_after: list[list[int]],
    step_size_hint: int = 5,
    max_cluster_pixels: int = 200,
) -> Optional[set[int]]:
    """
    Infer player sprite colors from the pixel diff between two frames.

    The player sprite is typically the largest small cluster of changed pixels.
    Returns None when the diff is empty or too large to interpret.
    """
    from collections import Counter

    changed: dict[tuple, int] = {}
    for r in range(min(len(frame_before), len(frame_after))):
        rb = frame_before[r]
        ra = frame_after[r]
        for c in range(min(len(rb), len(ra))):
            if rb[c] != ra[c] and ra[c] != 0:
                changed[(c, r)] = ra[c]

    if not changed or len(changed) > max_cluster_pixels * 4:
        return None

    # Connected-component analysis — find the largest small cluster
    visited_px: set[tuple] = set()
    clusters: list[tuple] = []  # (pixels_list, dominant_color)

    for start in list(changed.keys()):
        if start in visited_px:
            continue
        component: list[tuple] = []
        queue = [start]
        seen: set[tuple] = {start}
        while queue:
            px = queue.pop()
            component.append(px)
            cx, cy = px
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nb = (cx + dx, cy + dy)
                if nb not in seen and nb in changed:
                    seen.add(nb)
                    queue.append(nb)
        visited_px.update(component)
        if len(component) <= max_cluster_pixels:
            dom = Counter(changed[p] for p in component).most_common(1)[0][0]
            clusters.append((component, dom))

    if not clusters:
        return None

    # Largest qualifying cluster is most likely the player sprite
    clusters.sort(key=lambda x: len(x[0]), reverse=True)
    largest = clusters[0][0]
    player_colors = {changed[p] for p in largest if changed[p] != 0}
    return player_colors if player_colors else None


# ---------------------------------------------------------------------------
# Frame analysis
# ---------------------------------------------------------------------------

def find_border_candidates(
    frame: list[list[int]],
    walkable_colors: set[int],
    player_colors: set[int],
    step_size: int,
    wall_colors: set[int] | None = None,
    blocked_positions: set[tuple] | None = None,
) -> dict[tuple[int, int], int]:
    """
    Find step-aligned positions that are NOT walkable but ARE exactly one
    step_size away from a walkable cell.

    These are positions the player could reach via extra_passable (interactive
    objects like RCs, rings, win gates), as opposed to walls which block
    movement even when added to extra_passable.

    Returns
    -------
    {(col, row): color}  keyed by step-aligned game positions.
    """
    ignore = walkable_colors | player_colors | {0} | (wall_colors or set())
    blocked = blocked_positions or set()

    rows = len(frame)
    cols = len(frame[0]) if rows else 0

    # Find walkable pixel positions
    walkable_set: set[tuple[int, int]] = set()
    for r in range(rows):
        for c in range(cols):
            if frame[r][c] in walkable_colors:
                walkable_set.add((c, r))

    # First pass: gather all step-aligned border candidates
    raw: dict[tuple[int, int], int] = {}
    for (wc, wr) in walkable_set:
        for dc, dr in [(step_size, 0), (-step_size, 0),
                       (0, step_size), (0, -step_size)]:
            nc, nr = wc + dc, wr + dr
            pos = (nc, nr)
            if pos in blocked:
                continue
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            color = frame[nr][nc]
            if color not in ignore:
                raw[pos] = color

    if not raw:
        return {}

    # P2 tension: when wall_colors are unknown, the dominant border color is
    # almost always the wall texture.  Excluding it here is a heuristic that
    # should eventually be replaced by OBSERVER classification.  For now we
    # keep it as a pragmatic filter when concept_bindings has not yet supplied
    # wall_colors — but only for this fallback path.
    if not wall_colors:
        color_counts: dict[int, int] = {}
        for color in raw.values():
            color_counts[color] = color_counts.get(color, 0) + 1
        if color_counts:
            dominant = max(color_counts, key=lambda c: color_counts[c])
            if color_counts[dominant] > len(raw) * 0.5:
                return {pos: c for pos, c in raw.items() if c != dominant}

    return raw


def find_sprite_cells(
    frame: list[list[int]],
    walkable_colors: set[int],
    player_colors: set[int],
    step_size: int,
    player_pos: tuple[int, int],
    wall_colors: set[int] | None = None,
    blocked_positions: set[tuple] | None = None,
    max_sprite_pixels: int = 12,
) -> dict[tuple[int, int], int]:
    """
    Find player grid cells that contain small non-walkable sprites.

    Unlike find_border_candidates (which looks for non-walkable cells one
    step_size away from walkable cells), this function maps SPRITE PIXEL
    coordinates to the player grid cell that contains them.  It correctly
    handles games (like ls20) where the RC or ring sprite pixel may sit
    inside a cell whose background color IS walkable — the background color
    is ignored; what matters is the sprite color.

    Grid-cell mapping uses the player's current position to infer the grid
    offset:
        x_offset = player_pos[0] % step_size
        y_offset = player_pos[1] % step_size
        cell_x   = ((px - x_offset) // step_size) * step_size + x_offset
        cell_y   = ((py - y_offset) // step_size) * step_size + y_offset

    Filtering:
        1. Skip ignored colors (walkable, player, zero, known wall colors).
        2. Only keep cells with ≤ max_sprite_pixels sprite pixels to exclude
           large UI elements.  P2 note: ideally the OBSERVER should classify
           what is a sprite vs. UI element; this size filter is a pragmatic
           stand-in.

    Returns
    -------
    {(col, row): dominant_sprite_color}  — player-grid cell → sprite color.
    """
    from collections import Counter

    ignore = walkable_colors | player_colors | {0} | (wall_colors or set())
    blocked = blocked_positions or set()

    x_offset = player_pos[0] % step_size
    y_offset = player_pos[1] % step_size

    rows = len(frame)
    cols = len(frame[0]) if rows else 0

    # Infer playfield Y extent from the DOMINANT walkable color (floor).
    # Border colors may extend into the HUD area, so we use only the most
    # common walkable color (which is the floor, not the borders) to
    # determine where the playfield ends.
    _wc_counts: dict[int, int] = {}
    for _r in range(rows):
        for _c in range(cols):
            _fc = frame[_r][_c]
            if _fc in walkable_colors:
                _wc_counts[_fc] = _wc_counts.get(_fc, 0) + 1
    _floor_color = max(_wc_counts, key=lambda c: _wc_counts[c]) if _wc_counts else 0
    _max_walkable_y = 0
    for _r in range(rows):
        for _c in range(cols):
            if frame[_r][_c] == _floor_color:
                _max_walkable_y = max(_max_walkable_y, _r)
    _playfield_limit_y = _max_walkable_y + 1

    # Map non-walkable sprite pixels → player grid cell → list of sprite colors
    cell_colors: dict[tuple[int, int], list[int]] = {}
    for r in range(rows):
        for c in range(cols):
            color = frame[r][c]
            if color in ignore:
                continue
            cell_x = ((c - x_offset) // step_size) * step_size + x_offset
            cell_y = ((r - y_offset) // step_size) * step_size + y_offset
            if cell_x < 0 or cell_y < 0 or cell_y >= _playfield_limit_y:
                continue
            cell = (cell_x, cell_y)
            if cell not in blocked:
                cell_colors.setdefault(cell, []).append(color)

    if not cell_colors:
        return {}

    # Keep only cells with small sprite footprints (≤ max_sprite_pixels pixels)
    small: dict[tuple[int, int], int] = {
        cell: Counter(colors).most_common(1)[0][0]
        for cell, colors in cell_colors.items()
        if len(colors) <= max_sprite_pixels
    }

    if not small:
        return {}

    return small


def find_nonwalkable_candidates(
    frame: list[list[int]],
    walkable_colors: set[int],
    player_colors: set[int],
    wall_colors: set[int] | None = None,
    max_cluster_pixels: int = 60,
) -> dict[tuple[int, int], int]:
    """
    Scan the frame and return non-walkable cells that are likely interactive
    objects (rotation changers, rings, win gates), filtering out walls and
    large UI display elements.

    Filtering strategy
    ------------------
    1. Exclude known wall colors (from concept_bindings["wall_colors"]).
    2. Exclude the single most common non-walkable color — it is almost always
       the dominant wall color (e.g. color 4 fills 60%+ of ls20 frames).
    3. Find connected components of each remaining color; only keep components
       with ≤ max_cluster_pixels pixels (interactive sprites are small).

    Returns
    -------
    {(col, row): color}  — pixel-level candidate positions.
    """
    ignore = walkable_colors | player_colors | {0} | (wall_colors or set())

    # Count pixels per non-ignored color
    counts: dict[int, int] = {}
    rows = len(frame)
    cols = len(frame[0]) if rows else 0
    for r in range(rows):
        for c in range(cols):
            color = frame[r][c]
            if color not in ignore:
                counts[color] = counts.get(color, 0) + 1

    if not counts:
        return {}

    # P2 tension: removing the dominant non-walkable color is a heuristic —
    # the OBSERVER should classify wall vs. interactive objects.  Kept as a
    # pragmatic filter until OBSERVER-driven classification is in place.
    dominant_wall = max(counts, key=lambda c: counts[c])
    ignore = ignore | {dominant_wall}

    # Remaining candidate pixels
    raw: dict[tuple[int, int], int] = {}
    for r in range(rows):
        for c in range(cols):
            color = frame[r][c]
            if color not in ignore:
                raw[(c, r)] = color

    if not raw:
        return {}

    # Connected-component filtering (4-connectivity): only keep small clusters
    visited: set[tuple[int, int]] = set()
    candidates: dict[tuple[int, int], int] = {}

    for start in list(raw.keys()):
        if start in visited:
            continue
        color = raw[start]
        # BFS flood fill
        component: list[tuple[int, int]] = []
        queue = [start]
        seen_local: set[tuple[int, int]] = {start}
        while queue:
            pos = queue.pop()
            component.append(pos)
            cx, cy = pos
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nb = (cx + dx, cy + dy)
                if nb not in seen_local and raw.get(nb) == color:
                    seen_local.add(nb)
                    queue.append(nb)
        visited.update(component)
        # Only include small clusters (large = wall decoration / UI element)
        if len(component) <= max_cluster_pixels:
            for pos in component:
                candidates[pos] = color

    return candidates


# ---------------------------------------------------------------------------
# Build level model from frame + history + concept bindings
# ---------------------------------------------------------------------------

def build_level_model(
    initial_frame: list[list[int]],
    current_frame: list[list[int]],
    action_history: list[dict],
    walkable_colors: set[int],
    player_colors: set[int],
    step_size: int,
    player_pos: tuple[int, int] | None = None,
    concept_bindings: dict | None = None,
    history_start_idx: int = 0,
    last_reset_idx: int = -1,
    blocked_positions: set[tuple] | None = None,
    start_levels: int = 0,
) -> LevelModel:
    """
    Build a LevelModel from frame analysis, action history, and concept bindings.

    Classification order (highest → lowest confidence):
      1. Action history:
           level advance at player_pos   → win_gate
           diff > 80 at any visited pos  → rc  (no longer restricted to candidates)
      2. Frame disappearance + visited:
           in initial_cands, not in current_cands, was visited → consumed_ring
      3. Concept bindings by color:
           pre-classify unvisited candidates via known color→role mappings.

    Parameters
    ----------
    initial_frame    : frame captured at the START of the current level.
    current_frame    : the live frame right now.
    action_history   : full episode history list.
    walkable_colors  : set of floor/walkable color ints.
    player_colors    : set of player-sprite color ints.
    step_size        : BFS grid step size (pixels).
    player_pos       : current player (col, row).  Used to infer grid offset
                       for sprite-to-cell mapping.
    concept_bindings : dict {color_int: {"role": str, ...}} or {color_int: str}.
    history_start_idx: index in action_history where this level started.
    last_reset_idx   : index of last game-reset (life lost) in action_history.
    start_levels     : levels_completed count at the start of this level.
                       Prevents false win_gate detection when presolve steps
                       are not recorded in action_history.
    """
    model = LevelModel()

    # Extract wall colors from concept_bindings (accumulated by wall-contact logic)
    _wall_colors: set[int] = set()
    if concept_bindings:
        _wc_list = concept_bindings.get("wall_colors") or []
        _wall_colors = set(_wc_list) if isinstance(_wc_list, (list, set)) else set()

    # --- Frame candidates --------------------------------------------------
    # Primary method: find_sprite_cells maps non-walkable sprite pixels to
    # their player grid cells using the grid offset inferred from player_pos.
    # This correctly handles games (like ls20) where the RC is a tiny sprite
    # overlaid on a walkable floor cell — the cell itself is walkable, but
    # the sprite color marks it as interesting.
    #
    # Fallback: if player_pos is unknown, use find_border_candidates instead.
    if player_pos is not None:
        # NOTE: wall_colors intentionally NOT passed here.  find_sprite_cells
        # already uses max_sprite_pixels=12 to exclude large wall structures.
        # Passing wall_colors would also exclude tiny 1-pixel PUSH_PAD sprites
        # whose color (e.g. 1 or 4) happens to match wall border pixels — which
        # would make them invisible to candidate detection after OBSERVER runs.
        initial_cands = find_sprite_cells(
            initial_frame, walkable_colors, player_colors, step_size, player_pos,
            wall_colors=None, blocked_positions=blocked_positions)
        current_cands = find_sprite_cells(
            current_frame, walkable_colors, player_colors, step_size, player_pos,
            wall_colors=None, blocked_positions=blocked_positions)
    else:
        initial_cands = find_border_candidates(
            initial_frame, walkable_colors, player_colors, step_size,
            wall_colors=_wall_colors, blocked_positions=blocked_positions)
        current_cands = find_border_candidates(
            current_frame, walkable_colors, player_colors, step_size,
            wall_colors=_wall_colors, blocked_positions=blocked_positions)

    model.candidates = dict(initial_cands)
    # Also merge current-frame candidates so animated sprites (e.g. moving
    # rotation changers) are targeted at their NOW position, not their stale
    # initial-frame position.
    for _cp, _cc in current_cands.items():
        if _cp not in model.candidates:
            model.candidates[_cp] = _cc

    # Belt-and-suspenders: also remove any blocked positions that slipped through
    if blocked_positions:
        for bp in blocked_positions:
            model.candidates.pop(bp, None)

    # --- Minority-walkable candidates -------------------------------------
    # In many games one walkable color appears only at special locations (exit
    # tiles, win gates, goal zones).  Identify the walkable color that appears
    # fewest times at grid-aligned positions and add those cells as candidates.
    # On first visit they look like any floor tile, but history analysis will
    # classify them as win_gate if level advances there.
    if player_pos is not None and len(walkable_colors) > 1:
        _x_off = player_pos[0] % step_size
        _y_off = player_pos[1] % step_size
        # Compute playfield Y limit using dominant floor color (same as find_sprite_cells)
        _frame_rows = len(initial_frame)
        _frame_cols = len(initial_frame[0]) if _frame_rows else 0
        _mw_counts: dict[int, int] = {}
        for _r in range(_frame_rows):
            for _c in range(_frame_cols):
                _fc = initial_frame[_r][_c]
                if _fc in walkable_colors:
                    _mw_counts[_fc] = _mw_counts.get(_fc, 0) + 1
        _mw_floor = max(_mw_counts, key=lambda c: _mw_counts[c]) if _mw_counts else 0
        _mw_max_y = 0
        for _r in range(_frame_rows):
            for _c in range(_frame_cols):
                if initial_frame[_r][_c] == _mw_floor:
                    _mw_max_y = max(_mw_max_y, _r)
        _mw_limit_y = _mw_max_y + 1

        _wc_grid: dict[int, list[tuple[int, int]]] = {}
        for _r in range(_frame_rows):
            for _c in range(_frame_cols):
                _col = initial_frame[_r][_c]
                if _col in walkable_colors:
                    # Check if pixel is at a grid-aligned position within playfield
                    if ((_c - _x_off) % step_size == 0
                            and (_r - _y_off) % step_size == 0
                            and _r < _mw_limit_y):
                        _wc_grid.setdefault(_col, []).append((_c, _r))
        if _wc_grid:
            _minority_color = min(_wc_grid, key=lambda c: len(_wc_grid[c]))
            _majority_count = max(len(v) for k, v in _wc_grid.items()
                                  if k != _minority_color) if len(_wc_grid) > 1 else 0
            # Only treat as minority if noticeably fewer than the majority floor
            if _majority_count == 0 or len(_wc_grid[_minority_color]) < _majority_count * 0.5:
                for _cell in _wc_grid[_minority_color]:
                    bp_set = blocked_positions or set()
                    if _cell not in model.candidates and _cell not in bp_set:
                        model.candidates[_cell] = _minority_color

    # Consumed = present in initial but absent now (rings get eaten on visit)
    consumed_positions: set[tuple] = {
        pos for pos in initial_cands if pos not in current_cands
    }

    # --- History analysis --------------------------------------------------
    visited_set: set[tuple] = set()
    # Initialize prev_levels to start_levels (not 0) so that the first step in
    # action_history doesn't falsely trigger win_gate detection.  When presolve
    # steps are executed outside the main loop they are not appended to
    # action_history, so history_start_idx=0 and the first entry already has
    # levels=start_levels — without this init, curr_levels > prev_levels would
    # fire a false positive on the first entry.
    prev_levels = start_levels

    prev_diff = 0  # diff of the immediately preceding step
    # prev_pos: player position from the previous step.  Because action_history
    # records state_after (where the player IS after teleport), not state_before
    # (where they stood when they triggered the PUSH_PAD), the actual trigger
    # position is always one step back — i.e. prev_pos.
    prev_pos: Optional[tuple] = None
    # visit_diffs: maximum diff observed when the player was at each position.
    # Used to distinguish "sprite animated away" (diff~54) from "object consumed"
    # (diff >= 80 for ring refills, >> 300 for RC triggers).
    visit_diffs: dict[tuple, int] = {}
    for i, step in enumerate(action_history):
        if i < history_start_idx:
            prev_levels = step.get("levels", 0)
            prev_diff   = step.get("diff", 0)
            _pp = step.get("player_pos")
            if _pp is not None:
                prev_pos = tuple(_pp)
            continue
        if i <= last_reset_idx:
            prev_levels = step.get("levels", 0)
            prev_diff   = 0  # invalidate diff across a reset boundary
            prev_pos    = None  # don't carry stale position across a reset
            continue

        ppos = step.get("player_pos")
        curr_levels = step.get("levels", 0)
        diff = step.get("diff", 0)

        if ppos is not None:
            pos = tuple(ppos)
            visited_set.add(pos)
            # Only record visit diffs for *actual movements* (pos != prev_pos).
            # A blocked action leaves pos==prev_pos and the diff comes from
            # unrelated background animations (e.g. the ls20 top-right
            # white/black sprite swap that produces diff≈76 per tick).
            # Recording blocked-action diffs here causes the small-indicator
            # RC classifier below to learn phantom RCs at cells the player
            # never actually entered — the observed bug on ls20 L2 that
            # spawned a phantom RC at (14, 35) and sent the agent chasing
            # the wrong target for ~20 cycles.
            _moved_this_step = (prev_pos is None) or (pos != prev_pos)
            if _moved_this_step and diff > visit_diffs.get(pos, 0):
                visit_diffs[pos] = diff

            # Win gate: level advanced — use prev_pos (the PUSH_PAD trigger
            # position) rather than pos (the teleport landing position).
            # action_history stores state_after, so pos is where the player
            # landed after the trigger; prev_pos is where they stood before.
            if curr_levels > prev_levels:
                trigger = prev_pos if prev_pos is not None else pos
                if model.win_gate is None:
                    model.win_gate = trigger

            # RC: elevated diff at ANY visited position, no level advance.
            # Same reasoning: use prev_pos as the trigger position.
            # Only classify if the PREVIOUS step's diff was NOT already elevated —
            # rotation animations span 2 frames, so the step after the trigger
            # also shows elevated diff (lagging residual). We only want the first step.
            #
            # Two RC categories:
            #   Small-indicator RC (e.g. ls20 level 2): diff in (55, 80).
            #     Only a tiny sprite changes — the indicator rotates.
            #     Normal walk baseline ≈ 54; ring refills start at ~80.
            #   Full-maze RC (classic): diff in (300, 3000].
            #     Many objects move simultaneously.
            #
            # Ring refills (diff ~80-300) are excluded from both categories.
            # Upper bound 3000: game-reset restore frames have diff ~4000 and
            # must NOT be classified as RC visits.
            # Extra guard: the very first step of a new level (i == history_start_idx
            # and start_levels > 0) compares a frame from the *previous* level to the
            # first frame of this level — the resulting diff is a level-transition
            # artifact (~1462 for ls20), not an RC interaction.  Suppress RC detection
            # for that single step to avoid a false positive.
            elif (
                (
                    (55 < diff < 80 and prev_diff <= 55)       # small-indicator RC
                    or (300 < diff <= 3000 and prev_diff <= 300)  # full-maze RC
                )
                and not (i == history_start_idx and start_levels > 0)
            ):
                # For small-indicator RCs (diff < 80): the player is physically ON
                # the RC cell when the indicator fires — use pos (current position).
                # For full-maze RCs (diff 300+): the player steps onto a push-pad
                # that teleports them; use prev_pos (the pad cell, not the landing).
                if 55 < diff < 80:
                    # Require actual movement.  A small-indicator RC only fires
                    # when the player STEPS ONTO it.  If the action was blocked
                    # (pos == prev_pos) the agent bounced off a wall — the
                    # diff in (55, 80) came from unrelated background animation
                    # (e.g. the ls20 top-right white/black sprite swap produces
                    # diff≈76 without any player interaction) and must not be
                    # attributed to the blocked cell as a phantom RC.
                    if prev_pos is not None and pos == prev_pos:
                        trigger = None
                    else:
                        trigger = pos
                else:
                    trigger = prev_pos if prev_pos is not None else pos
                if trigger is not None and trigger not in model.rc_positions:
                    model.rc_positions.append(trigger)

        prev_diff   = diff
        prev_levels = curr_levels
        if ppos is not None:
            prev_pos = tuple(ppos)

    # --- Ring detection from frame disappearance ---------------------------
    rc_pos_set = set(map(tuple, model.rc_positions))
    win_pos    = tuple(model.win_gate) if model.win_gate else None
    for pos in consumed_positions:
        if pos in visited_set:
            # Guard: if already classified as RC or win gate, don't re-classify
            # as a consumed ring.  (Player standing ON an RC makes it look like
            # it disappeared from the current frame — same for win gate.)
            if pos in rc_pos_set or pos == win_pos:
                continue
            # Only classify as consumed ring if the visit had a notable diff
            # (>= 80).  A normal walk (diff~54) does NOT indicate the object was
            # consumed — animated sprites (e.g. rotation changers) move away on
            # their own, making their old position look "consumed" without any
            # player interaction.
            if visit_diffs.get(pos, 0) < 80:
                continue
            if pos not in model.consumed_rings:
                model.consumed_rings.append(pos)

    # --- Low-diff RC detection via initial_cands membership ------------------
    # Some levels have RCs that only rotate a tiny indicator sprite rather than
    # the full maze, producing diff ~55-79 (above normal-walk baseline ~54 but
    # below ring-visit range ~80+).  The standard history-loop RC detection
    # (300 < diff ≤ 3000) misses these.
    #
    # Detection: if a visited position that was a sprite candidate in the
    # initial frame (initial_cands) shows a slightly-elevated diff (55-79),
    # it is almost certainly a low-diff RC.  This check uses initial_cands
    # membership instead of consumed_positions so the result is STABLE across
    # cycles — the RC is retained in model.rc_positions even when the sprite
    # has reappeared (after the player left the RC tile).
    #
    # Background walk diff ≈ 54.  Ring visits ≥ 80.  Low-diff RC ≈ 55-79.
    # These ranges are empirically separated with no overlap in ls20 levels.
    rc_pos_set = set(map(tuple, model.rc_positions))
    win_pos    = tuple(model.win_gate) if model.win_gate else None
    _classified = (
        rc_pos_set
        | set(map(tuple, model.ring_positions))
        | set(map(tuple, model.consumed_rings))
        | ({win_pos} if win_pos else set())
    )
    for pos, d in visit_diffs.items():
        if pos in _classified:
            continue
        # Diff slightly above background (55-79): candidate for low-diff RC
        if 55 <= d < 80:
            # The position must have been a sprite candidate in the initial
            # frame — this ensures it is an interactive object, not noise.
            if pos in initial_cands:
                if pos not in model.rc_positions:
                    model.rc_positions.append(pos)
                _classified.add(pos)

    # --- Pre-classify unvisited candidates via concept bindings ------------
    if concept_bindings:
        ring_colors: set[int] = set()
        rc_colors:   set[int] = set()
        win_colors:  set[int] = set()

        for color, binding in concept_bindings.items():
            if not isinstance(color, int):
                continue
            role = (
                binding.get("role", "") if isinstance(binding, dict)
                else str(binding)
            ).lower()
            # P2: "step_counter" alone means the HUD bar — NOT a ring.
            # Only "ring", "consumable", or compound roles like
            # "step_counter_ring" should classify as ring.
            if any(k in role for k in ("ring", "consumable", "npx")):
                ring_colors.add(color)
            elif any(k in role for k in ("rotation", "rot_changer",
                                          "state_changer", "changer")):
                rc_colors.add(color)
            elif any(k in role for k in ("win", "target", "goal", "objective",
                                          "finish")):
                win_colors.add(color)

        for pos, color in initial_cands.items():
            # Skip positions the player has empirically failed to enter —
            # they are walls in the current maze state.
            if blocked_positions and pos in blocked_positions:
                continue
            if color in win_colors:
                # Win gates are persistent (not consumed) — classify even if
                # the player has already visited the position.  The player may
                # need to return after additional RC visits.
                if model.win_gate is None:
                    model.win_gate = pos
                continue
            if pos in visited_set:
                continue  # already classified from history
            if color in ring_colors:
                if pos not in model.ring_positions and pos not in model.consumed_rings:
                    model.ring_positions.append(pos)
            elif color in rc_colors:
                if pos not in model.rc_positions:
                    model.rc_positions.append(pos)

    return model


# ---------------------------------------------------------------------------
# Budget-aware dynamic waypoint planner
# ---------------------------------------------------------------------------

def compute_dynamic_waypoints(
    model: LevelModel,
    player_pos: tuple[int, int],
    rc_visits_done: int,
    n_rc_visits_needed: int,
    sc_resets_done: set[tuple],
    walkable_set: set[tuple],
    step_size: int,
    step_budget: int,
    steps_since_reset: int,
    extra_passable: set[tuple],
) -> tuple[list[tuple], set[tuple]]:
    """
    Build an ordered waypoint list respecting the step-budget constraint.

    Strategy
    --------
    Greedy simulation:
      - Start from player_pos with `steps_since_reset` steps already used.
      - For each remaining RC visit (plus a bounce cell between consecutive same-pos
        visits), check if budget allows reaching it; if not, insert the nearest
        unvisited ring first.
      - After all RC visits, navigate to the win gate (with ring insertion if needed).

    Parameters
    ----------
    model              : LevelModel with discovered object positions.
    player_pos         : current player position (col, row).
    rc_visits_done     : confirmed RC visits from action_history.
    n_rc_visits_needed : total RC visits required to win this level.
                         Use 0 to trigger "try win gate now" (for post-discovery).
    sc_resets_done     : positions of rings already collected in this level cycle.
    walkable_set       : set of walkable (col, row) pixel positions.
    step_size          : BFS movement grid size.
    step_budget        : step counter reset value (max steps between rings).
    steps_since_reset  : steps taken since last ring (or level start).
    extra_passable     : additional BFS-passable cells (interactive objects, etc.).

    Returns
    -------
    (waypoints, extra_passable)
      waypoints[0] is player_pos; subsequent entries are the ordered goals.
    """
    try:
        from nav_bfs import bfs_path as _bfs_path
    except ImportError:
        # Fallback: return simple direct path
        wp = [player_pos]
        if model.win_gate:
            wp.append(tuple(model.win_gate))
        return wp, set(extra_passable)

    wp_extra = set(extra_passable)

    # Build a combined passable set for distance estimation
    all_objects: list[tuple] = (
        [tuple(p) for p in model.rc_positions]
        + [tuple(p) for p in model.ring_positions]
        + ([tuple(model.win_gate)] if model.win_gate else [])
    )
    for p in all_objects:
        wp_extra.add(p)
        for dc, dr in [(0, step_size), (0, -step_size),
                       (step_size, 0), (-step_size, 0)]:
            wp_extra.add((p[0] + dc, p[1] + dr))

    passable = walkable_set | wp_extra

    def dist(a: tuple, b: tuple) -> int:
        if a == b:
            return 0
        path = _bfs_path(a, b, walkable_set, step_size, wp_extra)
        return len(path) if path is not None else 9999

    # Rings available (unvisited, still in current frame)
    unvisited_rings = [
        tuple(r) for r in model.ring_positions
        if tuple(r) not in sc_resets_done
    ]

    def nearest_ring(pos: tuple) -> tuple[tuple | None, int]:
        best_r, best_d = None, 9999
        for r in unvisited_rings:
            if r in sc_resets_done:
                continue
            d = dist(pos, r)
            if d < best_d:
                best_r, best_d = r, d
        return best_r, best_d

    waypoints: list[tuple] = [player_pos]
    current_pos = player_pos
    budget_left = step_budget - steps_since_reset

    def _add_wp(pos: tuple) -> None:
        """Add waypoint, inserting a ring if budget would be exceeded."""
        nonlocal current_pos, budget_left
        d = dist(current_pos, pos)
        # If this leg would exhaust the budget, insert the nearest ring first.
        if d >= budget_left and unvisited_rings:
            ring, ring_d = nearest_ring(current_pos)
            if ring is not None and ring not in sc_resets_done:
                ring_to_pos = dist(ring, pos)
                if ring_d + ring_to_pos < budget_left + step_budget:
                    waypoints.append(ring)
                    wp_extra.add(ring)
                    budget_left = step_budget - dist(ring, pos)
                    current_pos = ring
                    if ring in unvisited_rings:
                        unvisited_rings.remove(ring)
                    sc_resets_done.add(ring)
                    d = dist(current_pos, pos)  # recompute after ring
        waypoints.append(pos)
        wp_extra.add(pos)
        budget_left = max(1, budget_left - d)
        current_pos = pos

    # ---------- RC visits --------------------------------------------------
    # Strategy: distribute visits_remaining across all discovered RC positions
    # in round-robin order.  Each position gets at least one visit per round
    # until the total is met.  This handles levels where multiple distinct
    # changers are present (e.g. level 3: color_changer + rot_changer × 2).
    rc_positions = [tuple(p) for p in model.rc_positions]
    visits_remaining = max(0, n_rc_visits_needed - rc_visits_done)

    if rc_positions and visits_remaining > 0:
        # Pre-compute bounce neighbours for each RC position
        bounces: dict[tuple, tuple | None] = {}
        for rp in rc_positions:
            bounce = None
            for dc, dr in [(0, step_size), (0, -step_size),
                           (step_size, 0), (-step_size, 0)]:
                cand = (rp[0] + dc, rp[1] + dr)
                if cand in walkable_set:
                    bounce = cand
                    break
            bounces[rp] = bounce

        # Visit strategy: one pass through rc_positions in discovery order,
        # then repeat the last-discovered position for any extra visits.
        # Rationale: color-changers typically need exactly 1 visit (discrete
        # state change), while rotation-changers are discovered last and may
        # need multiple visits to reach the goal rotation.
        n_rcs = len(rc_positions)
        rc_visit_idx = rc_visits_done  # index into rc_positions for next visit
        last_was_rc_pos: tuple | None = current_pos if current_pos in set(rc_positions) else None

        for _ in range(visits_remaining):
            if rc_visit_idx < n_rcs:
                rp = rc_positions[rc_visit_idx]
            else:
                # Past first full round — keep visiting last-discovered RC
                rp = rc_positions[-1]
            bounce = bounces[rp]
            # Insert bounce if we'd land on the same RC twice in a row
            if last_was_rc_pos == rp and bounce is not None and waypoints[-1] != bounce:
                _add_wp(bounce)
            _add_wp(rp)
            last_was_rc_pos = rp
            rc_visit_idx += 1

    # ---------- Win gate ---------------------------------------------------
    if model.win_gate:
        win = tuple(model.win_gate)
        _add_wp(win)
        # Extra passable: cells adjacent to win gate (for approach / border tiles)
        for dc, dr in [(0, step_size), (0, -step_size),
                       (step_size, 0), (-step_size, 0)]:
            wp_extra.add((win[0] + dc, win[1] + dr))

    return waypoints, wp_extra


# ---------------------------------------------------------------------------
# Optimal BFS planner with refueling state
# ---------------------------------------------------------------------------

def plan_to_win(
    current_pos: tuple,
    current_budget: int,
    ring_refuels: dict,
    rc_positions: list,
    rc_visits_done: int,
    n_rc_needed: int,
    win_gate: Optional[tuple],
    walkable_set: set,
    step_size: int,
) -> Optional[list]:
    """
    Find the minimum-steps path from current_pos to win_gate that completes
    all required RC visits, refueling at ring tiles as needed.

    Uses BFS over augmented state (position, budget, rc_visits_done) so the
    plan is optimal and requires no hardcoded thresholds.

    Parameters
    ----------
    current_pos   : player's current grid position (col, row).
    current_budget: effective moves remaining before step-counter hits 0.
    ring_refuels  : dict mapping ring position -> new effective budget after
                    visiting that ring (color/shape agnostic — purely positional).
                    Different rings may restore different amounts.
    rc_positions  : list of confirmed RC positions.
    rc_visits_done: RC visits already confirmed from action history.
    n_rc_needed   : total RC visits required to unlock win gate.
    win_gate      : win gate position, or None (returns None immediately).
    walkable_set  : set of BFS-passable (col, row) positions.
    step_size     : grid step size in pixels.

    Returns
    -------
    List of positions [current_pos, p1, ..., win_gate] or None if unreachable.
    """
    if win_gate is None or current_budget <= 0:
        return None

    from collections import deque

    rc_set   = frozenset(tuple(p) for p in rc_positions)
    ring_set = frozenset(ring_refuels.keys())
    win      = tuple(win_gate)

    # Maximum possible budget (cap state budget to keep state space bounded)
    max_budget = max(ring_refuels.values(), default=current_budget)
    max_budget = max(max_budget, current_budget)

    # Build passable set: walkable ∪ all special positions ∪ their neighbours
    special   = rc_set | ring_set | {win}
    passable  = walkable_set | special
    dirs      = [(0, step_size), (0, -step_size), (step_size, 0), (-step_size, 0)]
    for sp in special:
        for dc, dr in dirs:
            nb = (sp[0] + dc, sp[1] + dr)
            if nb in walkable_set:
                passable.add(nb)

    init_rc    = min(rc_visits_done, n_rc_needed)
    init_state = (current_pos, min(current_budget, max_budget), init_rc)

    # BFS: state = (position, budget, rc_visits)
    parent: dict = {init_state: None}   # state -> predecessor state
    queue  = deque([init_state])
    goal   = None

    while queue:
        state = queue.popleft()
        pos, budget, rc_visits = state

        if pos == win and rc_visits >= n_rc_needed:
            goal = state
            break

        if budget <= 0:
            continue

        for dc, dr in dirs:
            npos = (pos[0] + dc, pos[1] + dr)
            if npos not in passable:
                continue

            nbud = budget - 1
            if nbud < 0:
                continue

            # Refuel: update budget to ring's known amount
            if npos in ring_refuels:
                nbud = ring_refuels[npos]

            # RC visit: only count entry from a non-RC cell
            nrc = rc_visits
            if npos in rc_set and pos not in rc_set:
                nrc = min(n_rc_needed, rc_visits + 1)

            nstate = (npos, min(nbud, max_budget), nrc)
            if nstate not in parent:
                parent[nstate] = state
                queue.append(nstate)

    if goal is None:
        return None

    # Reconstruct position list from goal back to start
    path: list = []
    state = goal
    while state is not None:
        path.append(state[0])
        state = parent[state]
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Nearest-candidate exploration (when model is incomplete)
# ---------------------------------------------------------------------------

def nearest_exploration_waypoint(
    model: LevelModel,
    player_pos: tuple[int, int],
    visited: set[tuple],
    walkable_set: set[tuple],
    step_size: int,
    extra_passable: set[tuple],
) -> tuple[list[tuple], set[tuple]]:
    """
    Return a waypoint list that navigates to the nearest unclassified candidate.

    Used during the discovery phase when the level model is incomplete.
    """
    try:
        from nav_bfs import bfs_path as _bfs_path
    except ImportError:
        return [player_pos], set(extra_passable)

    wp_extra = set(extra_passable)
    unvisited = model.unvisited_candidates(visited)
    if not unvisited:
        # All candidates visited — nowhere new to explore
        return [player_pos], wp_extra

    best_pos, best_len = None, 9999
    for cand in unvisited:
        wp_extra.add(cand)
        path = _bfs_path(player_pos, cand, walkable_set, step_size, wp_extra)
        if path is not None and len(path) < best_len:
            best_pos, best_len = cand, len(path)

    if best_pos is None:
        return [player_pos], wp_extra

    return [player_pos, best_pos], wp_extra


# ---------------------------------------------------------------------------
# Discovered-knowledge persistence
# ---------------------------------------------------------------------------
# A lightweight JSON sidecar that persists facts learned from successful plays:
#   - how many RC visits each level requires
#   - which colors map to which object roles (game-specific color→role)
#
# Complementary to (not replacing) the LLM-driven concept_bindings system.
# ---------------------------------------------------------------------------

def load_discovered_knowledge(path: str | Path) -> dict:
    """Load discovered_knowledge.json; return {} if file not found."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_discovered_knowledge(path: str | Path, data: dict) -> None:
    """Write discovered_knowledge.json atomically."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def update_discovered_knowledge(
    path: str | Path,
    game_id: str,
    level: int,
    rc_visits: int | None = None,
    color_roles: dict[int, str] | None = None,
) -> None:
    """
    Merge new facts into discovered_knowledge.json.

    Parameters
    ----------
    game_id     : e.g. "ls20"
    level       : 1-indexed level number
    rc_visits   : how many RC visits were needed to win this level
    color_roles : {color_int: role_str} pairs learned from this play
    """
    data = load_discovered_knowledge(path)
    game = data.setdefault(game_id, {})

    if rc_visits is not None:
        lrc = game.setdefault("level_rc_visits", {})
        lrc[str(level)] = rc_visits
        print(f"  [DISCOVER] saved rc_visits={rc_visits} for {game_id} level {level}",
              flush=True)

    if color_roles:
        cr = game.setdefault("color_roles", {})
        for c, r in color_roles.items():
            cr[str(c)] = r
            print(f"  [DISCOVER] saved color_role: {c} → {r}", flush=True)

    save_discovered_knowledge(path, data)


def get_n_rc_visits(
    discovered: dict,
    game_id: str,
    level: int,
) -> int | None:
    """Return the stored RC visit count for a level, or None if unknown."""
    try:
        return int(discovered.get(game_id, {})
                              .get("level_rc_visits", {})
                              .get(str(level), None))
    except (TypeError, ValueError):
        return None


def get_color_roles(
    discovered: dict,
    game_id: str,
) -> dict[int, str]:
    """Return stored color→role dict for a game (keys are ints)."""
    raw = discovered.get(game_id, {}).get("color_roles", {})
    return {int(k): v for k, v in raw.items() if k.isdigit()}


# ---------------------------------------------------------------------------
# Convenience: merge discovered color_roles into concept_bindings format
# ---------------------------------------------------------------------------

def enrich_concept_bindings_from_discovered(
    concept_bindings: dict,
    discovered: dict,
    game_id: str,
) -> dict:
    """
    Return a copy of concept_bindings supplemented with color_roles from
    discovered_knowledge.json.  Existing high-confidence bindings are not
    overridden.
    """
    merged = dict(concept_bindings)
    color_roles = get_color_roles(discovered, game_id)
    for color, role in color_roles.items():
        existing = merged.get(color)
        if existing is None:
            merged[color] = {"role": role, "confidence": 0.7,
                              "level_obs": 0, "total_obs": 0}
        elif isinstance(existing, dict) and existing.get("confidence", 0) < 0.7:
            merged[color] = {**existing, "role": role, "confidence": 0.7}
    return merged
