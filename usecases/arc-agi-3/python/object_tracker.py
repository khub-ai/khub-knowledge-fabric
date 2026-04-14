"""
object_tracker.py — Object-level frame analysis for ARC-AGI-3.

Provides zero-LLM-cost object detection and tracking across frames so that
the ensemble can reason about "the azure rectangle moved right 1 step" rather
than "52 cells changed color."

Public API
----------
detect_objects(frame)
    -> list[ObjectRecord]
    Find all connected components (4-connectivity) in a 2-D color grid.

diff_objects(frame_before, frame_after)
    -> ObjectDiff
    Match objects across two frames and classify each change as moved,
    appeared, or disappeared.

format_object_diff(diff, max_entries=8)
    -> str
    Produce a compact human-readable summary for prompt injection.

summarize_action_effects(action_effects)
    -> str
    Render the full accumulated action-effect table (object-level) for
    injection into OBSERVER / MEDIATOR prompts.

extract_subgrid(frame, bbox, foreground_color=None)
    -> list[list[int]]
    Crop a rectangular region from a frame.  Optionally binarize to a
    foreground mask.

compare_shapes(mask_a, mask_b)
    -> dict
    Compare two binary masks under all 8 D4 symmetry transforms (4 rotations
    x 2 reflections).  Returns whether they match, which transform, and
    Jaccard distances for each.

find_transformation(pairs)
    -> dict
    Given multiple (input, output) mask pairs, find a single consistent D4
    transform that maps every input to its output.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Color names — matches the SDK rendering palette (arc_agi/rendering.py)
# ---------------------------------------------------------------------------

_COLOR_NAMES = {
    0: "white",         # #FFFFFF
    1: "off-white",     # #CCCCCC
    2: "light-grey",    # #999999
    3: "dark-grey",     # #666666
    4: "off-black",     # #333333
    5: "black",         # #000000
    6: "magenta",       # #E53AA3
    7: "pink",          # #FF7BCC
    8: "red",           # #F93C31
    9: "blue",          # #1E93FF
    10: "cyan",         # #88D8F1
    11: "yellow",       # #FFDC00
    12: "orange",       # #FF851B
    13: "maroon",       # #921231
    14: "green",        # #4FCC30
    15: "purple",       # #A356D6
}


def color_name(c: int) -> str:
    return _COLOR_NAMES.get(int(c), f"color{c}")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ObjectRecord:
    """One connected component of the same color."""
    color: int
    size: int                        # number of cells
    centroid: tuple[float, float]    # (row, col)
    bbox: dict                       # r_min/r_max/c_min/c_max
    is_background: bool = False      # True if very large relative to frame

    # Derived geometry (populated by detect_objects)
    width: int = 0                   # cols spanned (c_max - c_min + 1)
    height: int = 0                  # rows spanned (r_max - r_min + 1)

    @property
    def aspect_ratio(self) -> float:
        """width / height; >1 = horizontal, <1 = vertical, 1 = square."""
        return self.width / self.height if self.height else 1.0

    @property
    def orientation(self) -> str:
        """'horizontal', 'vertical', or 'square' based on bounding-box shape."""
        r = self.aspect_ratio
        if r > 1.3:
            return "horizontal"
        if r < 0.77:
            return "vertical"
        return "square"

    @property
    def label(self) -> str:
        return (
            f"{color_name(self.color)}"
            f"(size={self.size}, {self.width}w x {self.height}h, {self.orientation})"
        )

    def distance_to(self, other: "ObjectRecord") -> float:
        return math.hypot(
            self.centroid[0] - other.centroid[0],
            self.centroid[1] - other.centroid[1],
        )


@dataclass
class MovedObject:
    obj: ObjectRecord
    delta_r: float
    delta_c: float
    from_centroid: tuple[float, float]
    to_centroid: tuple[float, float]

    @property
    def direction(self) -> str:
        dr, dc = self.delta_r, self.delta_c
        parts = []
        if abs(dr) > 0.5:
            parts.append("down" if dr > 0 else "up")
        if abs(dc) > 0.5:
            parts.append("right" if dc > 0 else "left")
        return "+".join(parts) if parts else "stationary"

    @property
    def magnitude(self) -> float:
        return math.hypot(self.delta_r, self.delta_c)


@dataclass
class AttributeChange:
    """Records which attributes changed between the before and after versions of a matched object."""
    color: int                              # color of the object (key for lookup)
    before: ObjectRecord
    after: ObjectRecord
    changed: list[str] = field(default_factory=list)  # e.g. ["size", "orientation", "width"]

    @property
    def summary(self) -> str:
        parts = []
        for attr in self.changed:
            bv = getattr(self.before, attr, None)
            av = getattr(self.after,  attr, None)
            if attr == "orientation":
                parts.append(f"orientation {bv}->{av}")
            elif attr == "size":
                parts.append(f"size {bv}->{av}")
            elif attr == "width":
                parts.append(f"width {bv}->{av}")
            elif attr == "height":
                parts.append(f"height {bv}->{av}")
            else:
                parts.append(f"{attr} {bv}->{av}")
        label = color_name(self.color)
        return f"{label}: " + ", ".join(parts) if parts else f"{label}: no attribute change"


@dataclass
class ObjectDiff:
    """Result of comparing two frames at the object level."""
    moved: list[MovedObject] = field(default_factory=list)
    appeared: list[ObjectRecord] = field(default_factory=list)
    disappeared: list[ObjectRecord] = field(default_factory=list)
    stationary: list[ObjectRecord] = field(default_factory=list)
    attribute_changes: list[AttributeChange] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.moved or self.appeared or self.disappeared or self.attribute_changes)

    @property
    def total_moved_cells(self) -> int:
        return sum(m.obj.size for m in self.moved)


# ---------------------------------------------------------------------------
# Connected-component detection
# ---------------------------------------------------------------------------

def detect_objects(
    frame: list,
    background_threshold: float = 0.40,
) -> list[ObjectRecord]:
    """
    Find all connected components (4-connectivity) in a 2-D color grid.

    Parameters
    ----------
    frame : list[list[int]]
        2-D grid of integer color values.
    background_threshold : float
        Objects whose size exceeds this fraction of total cells are marked
        as background (large uniform regions like the game floor).
        Default 0.40 = 40%.  Raised from 0.25 so that mid-size structural
        regions (play arenas, platforms) are not silently excluded.

    Returns
    -------
    List of ObjectRecord, sorted by size descending (largest first).
    """
    if not frame or not frame[0]:
        return []

    rows = len(frame)
    cols = len(frame[0])
    total_cells = rows * cols
    visited = [[False] * cols for _ in range(rows)]
    objects: list[ObjectRecord] = []

    for r0 in range(rows):
        for c0 in range(cols):
            if visited[r0][c0]:
                continue
            color = int(frame[r0][c0])
            # BFS
            cells: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque()
            q.append((r0, c0))
            visited[r0][c0] = True
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols
                            and not visited[nr][nc]
                            and int(frame[nr][nc]) == color):
                        visited[nr][nc] = True
                        q.append((nr, nc))

            size = len(cells)
            if size == 0:
                continue

            rs = [r for r, c in cells]
            cs = [c for r, c in cells]
            centroid = (sum(rs) / size, sum(cs) / size)
            r_min, r_max = min(rs), max(rs)
            c_min, c_max = min(cs), max(cs)
            bbox = {"r_min": r_min, "r_max": r_max, "c_min": c_min, "c_max": c_max}
            is_bg = (size / total_cells) >= background_threshold

            objects.append(ObjectRecord(
                color=color,
                size=size,
                centroid=centroid,
                bbox=bbox,
                is_background=is_bg,
                width=c_max - c_min + 1,
                height=r_max - r_min + 1,
            ))

    objects.sort(key=lambda o: o.size, reverse=True)
    return objects


# ---------------------------------------------------------------------------
# Object tracking (frame diff)
# ---------------------------------------------------------------------------

def diff_objects(
    frame_before: list,
    frame_after: list,
    match_radius: float = 10.0,
) -> ObjectDiff:
    """
    Compare two frames at the object level.

    For each color present in either frame, this function matches objects
    by minimum centroid distance (greedy nearest-neighbour, same color only).

    Parameters
    ----------
    match_radius : float
        Maximum centroid distance to consider two objects the same entity.
        Objects further apart are treated as disappeared + appeared.
    """
    if not frame_before or not frame_after:
        return ObjectDiff()

    before_objs = detect_objects(frame_before)
    after_objs  = detect_objects(frame_after)

    # Group by color
    def by_color(objs: list[ObjectRecord]) -> dict[int, list[ObjectRecord]]:
        d: dict[int, list[ObjectRecord]] = {}
        for o in objs:
            d.setdefault(o.color, []).append(o)
        return d

    before_by_color = by_color(before_objs)
    after_by_color  = by_color(after_objs)

    result = ObjectDiff()
    all_colors = set(before_by_color) | set(after_by_color)

    for color in all_colors:
        b_list = before_by_color.get(color, [])
        a_list = list(after_by_color.get(color, []))

        matched_a: set[int] = set()

        for b_obj in b_list:
            # Find closest unmatched after-object of same color
            best_idx: Optional[int] = None
            best_dist = float("inf")
            for i, a_obj in enumerate(a_list):
                if i in matched_a:
                    continue
                d = b_obj.distance_to(a_obj)
                if d < best_dist:
                    best_dist = d
                    best_idx = i

            if best_idx is not None and best_dist <= match_radius:
                matched_a.add(best_idx)
                a_obj = a_list[best_idx]
                delta_r = a_obj.centroid[0] - b_obj.centroid[0]
                delta_c = a_obj.centroid[1] - b_obj.centroid[1]

                if abs(delta_r) < 0.5 and abs(delta_c) < 0.5:
                    result.stationary.append(b_obj)
                else:
                    result.moved.append(MovedObject(
                        obj=a_obj,
                        delta_r=round(delta_r, 1),
                        delta_c=round(delta_c, 1),
                        from_centroid=b_obj.centroid,
                        to_centroid=a_obj.centroid,
                    ))

                # Check for attribute changes between matched before/after objects
                changed_attrs: list[str] = []
                if b_obj.size != a_obj.size:
                    changed_attrs.append("size")
                if b_obj.width != a_obj.width:
                    changed_attrs.append("width")
                if b_obj.height != a_obj.height:
                    changed_attrs.append("height")
                if b_obj.orientation != a_obj.orientation:
                    changed_attrs.append("orientation")
                if changed_attrs:
                    result.attribute_changes.append(AttributeChange(
                        color=color,
                        before=b_obj,
                        after=a_obj,
                        changed=changed_attrs,
                    ))
            else:
                # No match found — object disappeared
                result.disappeared.append(b_obj)

        # Unmatched after-objects appeared
        for i, a_obj in enumerate(a_list):
            if i not in matched_a:
                result.appeared.append(a_obj)

    return result


# ---------------------------------------------------------------------------
# Formatting for prompts
# ---------------------------------------------------------------------------

def format_object_diff(diff: ObjectDiff, max_entries: int = 8) -> str:
    """Compact human-readable summary of an ObjectDiff for prompt injection."""
    if not diff.has_changes:
        return "no object-level changes detected"

    lines: list[str] = []
    count = 0

    for m in diff.moved:
        if count >= max_entries:
            break
        if m.obj.is_background:
            continue
        dr = f"{m.delta_r:+.0f}r" if abs(m.delta_r) >= 0.5 else ""
        dc = f"{m.delta_c:+.0f}c" if abs(m.delta_c) >= 0.5 else ""
        delta_str = " ".join(filter(None, [dr, dc])) or "~0"
        lines.append(
            f"  MOVED   {m.obj.label} "
            f"{m.direction} by ({delta_str})  "
            f"centroid {_fmt_pt(m.from_centroid)} -> {_fmt_pt(m.to_centroid)}"
        )
        count += 1

    for o in diff.appeared:
        if count >= max_entries:
            break
        if o.is_background:
            continue
        lines.append(f"  APPEARED  {o.label} at centroid {_fmt_pt(o.centroid)}")
        count += 1

    for o in diff.disappeared:
        if count >= max_entries:
            break
        if o.is_background:
            continue
        lines.append(f"  GONE      {o.label} from centroid {_fmt_pt(o.centroid)}")
        count += 1

    for ac in diff.attribute_changes:
        if count >= max_entries:
            break
        if ac.before.is_background:
            continue
        lines.append(f"  CHANGED   {ac.summary}")
        count += 1

    if not lines:
        # Only background changes — report briefly
        bg_moved = [m for m in diff.moved if m.obj.is_background]
        if bg_moved:
            lines.append(
                f"  only background region(s) shifted "
                f"(e.g. {bg_moved[0].obj.label} {bg_moved[0].direction})"
            )
        else:
            lines.append("  only background-level changes")

    return "\n".join(lines)


def summarize_action_effects(action_effects: dict) -> str:
    """
    Render the full accumulated action-effect table for prompt injection.

    Uses the `object_observations` field if present (richer), otherwise
    falls back to pixel-level diff stats.
    """
    if not action_effects:
        return "  (no action effects recorded yet)"

    lines: list[str] = []
    for action, e in sorted(action_effects.items()):
        total    = e.get("total_calls", 0)
        nonzero  = e.get("nonzero_calls", 0)
        advances = e.get("level_advances", 0)

        obj_obs: list[dict] = e.get("object_observations", [])
        if obj_obs:
            # Summarize the most consistent object-level effect
            summary = _consensus_object_summary(obj_obs)
        else:
            # Fallback: pixel-level
            obs_list = e.get("observations", [{}])
            last = obs_list[-1] if obs_list else {}
            summary = f"pixel diff={last.get('diff', 0)}"

        lines.append(
            f"  {action}: called={total} changed={nonzero} "
            f"level_advances={advances}\n"
            f"    {summary}"
        )

    return "\n".join(lines)


def _consensus_object_summary(obj_obs: list[dict]) -> str:
    """
    Given a list of per-step object observations, produce a consensus summary.

    Reports both the most-consistent movement pattern AND any attribute changes
    (orientation, color, shape) observed — these are critical for detecting
    CHANGER cells in state-transformation games.
    """
    if not obj_obs:
        return "(no object observations)"

    # Count moves per (color, direction)
    move_counts: dict[tuple[int, str], list[float]] = {}
    # Count attribute changes per (color, attr, before_val, after_val)
    attr_counts: dict[tuple, int] = {}
    no_change_count = 0

    for obs in obj_obs:
        moved = obs.get("moved", [])
        attrs = obs.get("attribute_changes", [])
        appeared = obs.get("appeared", [])
        disappeared = obs.get("disappeared", [])

        if not moved and not attrs and not appeared and not disappeared:
            no_change_count += 1
            continue

        for m in moved:
            if m.get("is_background"):
                continue
            key = (m["color"], m["direction"])
            move_counts.setdefault(key, []).append(m.get("magnitude", 0))

        for ac in attrs:
            color = ac.get("color", -1)
            changed = ac.get("changed", [])
            before = ac.get("before", {})
            after  = ac.get("after", {})
            for attr in changed:
                bv = before.get(attr, "?")
                av = after.get(attr, "?")
                key = (color, attr, str(bv), str(av))
                attr_counts[key] = attr_counts.get(key, 0) + 1

    parts: list[str] = []

    if move_counts:
        best_key = max(move_counts, key=lambda k: len(move_counts[k]))
        color, direction = best_key
        magnitudes = move_counts[best_key]
        avg_mag = sum(magnitudes) / len(magnitudes)
        freq = len(magnitudes)
        total = len(obj_obs)
        parts.append(
            f"{color_name(color)} object moves {direction} "
            f"~{avg_mag:.1f} cells/call  "
            f"(observed {freq}/{total} calls)"
        )
    elif no_change_count == len(obj_obs):
        parts.append("no object movement observed (all calls had no effect)")
    else:
        parts.append("changes detected but no clear object movement pattern")

    # Surface attribute changes — these reveal CHANGER cells
    for (color, attr, bv, av), count in sorted(attr_counts.items(),
                                                key=lambda x: -x[1]):
        total = len(obj_obs)
        parts.append(
            f"*** ATTRIBUTE CHANGE: {color_name(color)} {attr} "
            f"{bv}->{av}  (observed {count}/{total} calls) "
            f"-- possible CHANGER cell visited ***"
        )

    return "\n    ".join(parts) if parts else "(no object observations)"


def summarize_current_objects(frame: list, concept_bindings: dict | None = None) -> str:
    """
    One-line summary of all non-background objects in the current frame.

    concept_bindings supports two schemas:
      {color_int: "role_name"}    -- color-keyed roles (player_piece, step_counter)
      {"wall_colors": [c, ...]}   -- concept-keyed list (wall is game-local)
    Colors in wall_colors are labeled "wall?" to signal game-local uncertainty.
    """
    objs = detect_objects(frame)
    bindings = concept_bindings or {}
    wall_colors: set = set(bindings.get("wall_colors", []))
    lines = []
    for o in objs:
        if o.is_background:
            continue
        if o.color in wall_colors:
            concept = "wall?"
        else:
            raw = bindings.get(o.color)
            if isinstance(raw, dict):
                role = raw.get("role", "")
                conf = raw.get("confidence", 0.0)
                concept = f"{role}({conf:.0%})" if role else None
            else:
                concept = raw  # plain string or None
        name = f"{concept}:{color_name(o.color)}" if concept else color_name(o.color)
        lines.append(
            f"  {name}: size={o.size} centroid={_fmt_pt(o.centroid)} "
            f"{o.width}w x {o.height}h {o.orientation}"
        )
    return "\n".join(lines) if lines else "  (no foreground objects detected)"


def compute_trend_predictions(action_effects: dict, steps_remaining: int) -> list[str]:
    """
    Scan accumulated object_observations for monotonic trends across ANY
    tracked attribute and project future states.

    Works entirely from raw numeric before/after values stored in
    attribute_changes records — no string parsing, no hardcoded attribute names.
    Any attribute stored by the object tracker (size, width, height, and any
    future additions) is automatically eligible for trend analysis.

    Detects:
    - Attribute trend: any numeric attribute consistently increasing or
      decreasing each action → predicts depletion (→0) or unbounded growth.
    - Position drift: object consistently moving in one direction across
      all observed moves → predicts it will hit a boundary.

    Requires ≥ 2 observations with consistent sign to fire (avoids noise).
    """
    if not action_effects:
        return []

    predictions: list[str] = []

    # -----------------------------------------------------------------------
    # Collect per-(color, attribute) time series from raw numeric records.
    # We accumulate deltas (after - before) across all stored observations.
    # The last seen "after" value for each attribute gives the current state.
    # -----------------------------------------------------------------------
    attr_deltas:   dict[tuple, list[float]] = {}  # (color, attr) -> [Δ per obs]
    attr_last_val: dict[tuple, float]       = {}  # (color, attr) -> last after value

    move_deltas_r: dict[int, list[float]] = {}  # color -> [Δrow per move obs]
    move_deltas_c: dict[int, list[float]] = {}  # color -> [Δcol per move obs]

    for _action, e in sorted(action_effects.items()):
        for obs in e.get("object_observations", []):
            for ac in obs.get("attribute_changes", []):
                color  = ac.get("color")
                before = ac.get("before", {})
                after  = ac.get("after",  {})
                if color is None or not before or not after:
                    continue
                for attr in ac.get("changed", []):
                    bv = before.get(attr)
                    av = after.get(attr)
                    if bv is None or av is None:
                        continue
                    try:
                        delta = float(av) - float(bv)
                    except (TypeError, ValueError):
                        continue
                    key = (color, attr)
                    attr_deltas.setdefault(key, []).append(delta)
                    attr_last_val[key] = float(av)

            for mv in obs.get("moved", []):
                color = mv.get("color")
                if color is None or mv.get("is_background"):
                    continue
                move_deltas_r.setdefault(color, []).append(float(mv.get("delta_r", 0)))
                move_deltas_c.setdefault(color, []).append(float(mv.get("delta_c", 0)))

    # -----------------------------------------------------------------------
    # Attribute trend analysis (generic — works for size, width, height, etc.)
    # -----------------------------------------------------------------------
    for (color, attr), deltas in attr_deltas.items():
        if len(deltas) < 2:
            continue
        nonzero = [d for d in deltas if abs(d) > 0.1]
        if len(nonzero) < 2:
            continue
        signs = set(1 if d > 0 else -1 for d in nonzero)
        if len(signs) != 1:
            continue  # mixed — no reliable trend
        avg_delta = sum(nonzero) / len(nonzero)
        if abs(avg_delta) < 0.5:
            continue

        cname = color_name(color)
        last_val = attr_last_val.get((color, attr))
        if avg_delta < 0:
            if last_val is not None and last_val > 0:
                steps_to_zero = int(last_val / abs(avg_delta))
                urgency = " [URGENT]" if steps_to_zero <= steps_remaining else ""
                predictions.append(
                    f"PREDICTION: {cname}(color{color}) {attr} shrinking "
                    f"~{abs(avg_delta):.1f}/action, currently ~{last_val:.0f}. "
                    f"Reaches 0 in ~{steps_to_zero} more actions.{urgency}"
                )
        else:
            if last_val is not None:
                predictions.append(
                    f"PREDICTION: {cname}(color{color}) {attr} growing "
                    f"~{avg_delta:.1f}/action, currently ~{last_val:.0f} — "
                    f"unbounded growth."
                )

    # -----------------------------------------------------------------------
    # Position drift trend analysis
    # -----------------------------------------------------------------------
    for color in set(move_deltas_r) | set(move_deltas_c):
        dr_list = move_deltas_r.get(color, [])
        dc_list = move_deltas_c.get(color, [])
        cname   = color_name(color)
        drift_parts: list[str] = []

        for axis, deltas, pos_label, neg_label in (
            ("row", dr_list, "down", "up"),
            ("col", dc_list, "right", "left"),
        ):
            nonzero = [d for d in deltas if abs(d) > 0.1]
            if len(nonzero) < 2:
                continue
            signs = set(1 if d > 0 else -1 for d in nonzero)
            if len(signs) != 1:
                continue  # reverses — bounded oscillation, not a drift
            avg = sum(nonzero) / len(nonzero)
            direction = pos_label if avg > 0 else neg_label
            drift_parts.append(f"{direction} ~{abs(avg):.1f} cells/{axis}/action")

        if drift_parts:
            predictions.append(
                f"PREDICTION: {cname}(color{color}) consistently drifting "
                + ", ".join(drift_parts)
                + " — will reach a boundary if uncorrected."
            )

    return predictions


def detect_wall_contacts(
    frame: list,
    obj: ObjectRecord,
    direction: str,
) -> dict:
    """
    When an object fails to move in a direction, examine what is immediately
    adjacent to it in that direction to identify what stopped it.

    Parameters
    ----------
    frame : current game frame (list of lists)
    obj   : the object that did not move
    direction : "up", "down", "left", "right" (the direction that was blocked)

    Returns
    -------
    dict with:
      "adjacent_colors": Counter of colors found in the cells just beyond the
                         object's bounding box in the blocked direction.
      "sample_cells":    A few (row, col) positions that were inspected.
      "direction":       The blocked direction.
    """
    from collections import Counter

    if not frame:
        return {}

    rows = len(frame)
    cols = len(frame[0]) if rows else 0
    bbox = obj.bbox

    # Determine which edge to scan and in which direction to step
    if direction == "up":
        edge_r = bbox["r_min"] - 1
        scan   = [(edge_r, c) for c in range(bbox["c_min"], bbox["c_max"] + 1)
                  if 0 <= edge_r < rows]
    elif direction == "down":
        edge_r = bbox["r_max"] + 1
        scan   = [(edge_r, c) for c in range(bbox["c_min"], bbox["c_max"] + 1)
                  if 0 <= edge_r < rows]
    elif direction == "left":
        edge_c = bbox["c_min"] - 1
        scan   = [(r, edge_c) for r in range(bbox["r_min"], bbox["r_max"] + 1)
                  if 0 <= edge_c < cols]
    elif direction == "right":
        edge_c = bbox["c_max"] + 1
        scan   = [(r, edge_c) for r in range(bbox["r_min"], bbox["r_max"] + 1)
                  if 0 <= edge_c < cols]
    else:
        return {}

    color_counts: Counter = Counter()
    for r, c in scan:
        color_counts[int(frame[r][c])] += 1

    return {
        "direction":       direction,
        "adjacent_colors": dict(color_counts),
        "sample_cells":    scan[:5],
    }


def infer_typical_direction(action_effects: dict, action_name: str) -> Optional[str]:
    """
    From accumulated move observations for a given action, return the most
    common non-stationary direction the player piece moved, or None if unknown.
    """
    obs_list = action_effects.get(action_name, {}).get("object_observations", [])
    dir_counts: dict[str, int] = {}
    for obs in obs_list:
        for mv in obs.get("moved", []):
            if mv.get("is_background"):
                continue
            d = mv.get("direction", "stationary")
            if d != "stationary":
                dir_counts[d] = dir_counts.get(d, 0) + 1
    return max(dir_counts, key=lambda k: dir_counts[k]) if dir_counts else None


def _fmt_pt(pt: tuple) -> str:
    return f"({pt[0]:.0f},{pt[1]:.0f})"


# ---------------------------------------------------------------------------
# Structural analysis — containment, spatial relations, landmark context
# ---------------------------------------------------------------------------

@dataclass
class ContainmentRelation:
    """One object whose bounding box is fully enclosed within another's."""
    container: ObjectRecord   # outer object (e.g. a bordered box)
    content:   ObjectRecord   # inner object (e.g. a pattern inside the box)


@dataclass
class SpatialRelation:
    """A directional or alignment relationship between two objects."""
    obj_a:    ObjectRecord
    obj_b:    ObjectRecord
    relation: str    # "above", "below", "left", "right"
    col_aligned: bool   # centroids within COL_ALIGN_TOL columns
    row_aligned: bool   # centroids within ROW_ALIGN_TOL rows
    distance: float


_COL_ALIGN_TOL = 4   # columns — centroids this close share a column
_ROW_ALIGN_TOL = 4   # rows


def detect_containment(objects: list[ObjectRecord]) -> list[ContainmentRelation]:
    """Return all (container, content) pairs where content.bbox ⊆ container.bbox.

    Only considers foreground objects (is_background=False).
    Different colors only — an object cannot contain a same-color piece.
    """
    fg = [o for o in objects if not o.is_background]
    relations: list[ContainmentRelation] = []
    for i, inner in enumerate(fg):
        for j, outer in enumerate(fg):
            if i == j or inner.color == outer.color:
                continue
            b_in  = inner.bbox
            b_out = outer.bbox
            if (b_out["r_min"] <= b_in["r_min"]
                    and b_out["r_max"] >= b_in["r_max"]
                    and b_out["c_min"] <= b_in["c_min"]
                    and b_out["c_max"] >= b_in["c_max"]
                    and outer.size > inner.size):
                relations.append(ContainmentRelation(container=outer, content=inner))
    return relations


def detect_spatial_relations(
    objects: list[ObjectRecord],
    max_distance: float = 40.0,
) -> list[SpatialRelation]:
    """Return pairwise directional relationships for all foreground object pairs
    within max_distance of each other (centroid-to-centroid).

    Each pair (A, B) appears once with A above/left of B.
    """
    fg = [o for o in objects if not o.is_background]
    relations: list[SpatialRelation] = []
    for i in range(len(fg)):
        for j in range(i + 1, len(fg)):
            a, b = fg[i], fg[j]
            dr = b.centroid[0] - a.centroid[0]   # positive = b is below a
            dc = b.centroid[1] - a.centroid[1]   # positive = b is right of a
            dist = math.hypot(dr, dc)
            if dist > max_distance:
                continue
            # Primary direction: describes where A is relative to B.
            # dr > 0 means b is lower (higher row) → a is above b.
            # dc > 0 means b is to the right → a is to the left of b.
            rel = ("above" if dr > 0 else "below") if abs(dr) >= abs(dc) \
                else ("left" if dc > 0 else "right")
            col_aligned = abs(dc) <= _COL_ALIGN_TOL
            row_aligned = abs(dr) <= _ROW_ALIGN_TOL
            relations.append(SpatialRelation(
                obj_a=a, obj_b=b,
                relation=rel,
                col_aligned=col_aligned,
                row_aligned=row_aligned,
                distance=round(dist, 1),
            ))
    return relations


def auto_detect_concepts(
    action_effects: dict,
    objects: list[ObjectRecord],
    concept_bindings: dict,
) -> dict[int, dict]:
    """Zero-cost heuristic concept detection that runs after every step.

    Scans action_effects for behavioral signatures and returns suggested
    concept_bindings updates — structured identically to OBSERVER output so
    they flow through the same merge path.

    Rules (all produce [GUESS] unless otherwise noted):
    - step_counter  : bar-shaped object (aspect ≥ 3 or ≤ 0.33) whose size
                      decreases by a consistent non-zero delta on every
                      observed action.  Confidence:
                        1 observation  → 0.35  (one shrink seen)
                        2 observations, consistent delta → 0.60
                        3+ observations, consistent delta → 0.85

    New signatures can be added here as new games are explored.
    Already-bound roles with confidence ≥ 0.7 are not downgraded.
    """
    obj_by_color: dict[int, ObjectRecord] = {o.color: o for o in objects}
    suggestions: dict[int, dict] = {}

    # --- Collect per-color size-change deltas across all actions/observations ---
    deltas_by_color: dict[int, list[int]] = {}
    for effects in action_effects.values():
        for obs in effects.get("object_observations", []):
            for ac in obs.get("attribute_changes", []):
                if "size" not in ac.get("changed", []):
                    continue
                color = ac.get("color")
                before = ac.get("before", {}).get("size")
                after  = ac.get("after",  {}).get("size")
                if color is None or before is None or after is None:
                    continue
                deltas_by_color.setdefault(color, []).append(after - before)

    # --- Step-counter signature ---
    for color, deltas in deltas_by_color.items():
        obj = obj_by_color.get(color)
        if obj is None:
            continue

        # Don't downgrade a role already bound with high confidence.
        existing = concept_bindings.get(color)
        if isinstance(existing, dict):
            if existing.get("role") not in (None, "step_counter"):
                continue          # already bound to something else
            if existing.get("confidence", 0) >= 0.7 and existing.get("role") == "step_counter":
                continue          # already confirmed, nothing to add

        # All observed deltas must be negative (strictly shrinking every action).
        if not deltas or not all(d < 0 for d in deltas):
            continue

        # Consistency: all deltas equal (constant depletion rate).
        consistent = len(set(deltas)) == 1
        n = len(deltas)

        if n == 1:
            conf = 0.35
        elif consistent:
            conf = 0.85 if n >= 3 else 0.60
        else:
            conf = 0.40   # shrinking but variable rate — weaker guess

        # Bar-shaped objects are the canonical step-counter form; penalise others.
        ar = obj.aspect_ratio
        if not (ar >= 3.0 or ar <= 0.33):
            conf *= 0.5   # not bar-shaped — still possible but less likely

        if conf < 0.25:
            continue

        suggestions[color] = {
            "role":          "step_counter",
            "confidence":    round(conf, 2),
            "observations":  n,
            "auto_detected": True,
        }

    return suggestions


def detect_contacts(
    player_colors: set[int],
    objects: list[ObjectRecord],
    margin: int = 5,
) -> dict[int, int]:
    """Return {color: min_gap} for non-player foreground objects within *margin* cells.

    min_gap is the minimum axis-aligned bounding-box distance in cells:
      0  — bboxes share at least one cell (player is physically on top of the object)
      1  — bboxes are adjacent (touching edges, no shared cells — side-by-side)
      N  — bboxes are N cells apart (near-contact, proximity trigger)

    Callers choose their own threshold:
      on_top_of = {c for c, g in contacts.items() if g == 0}
      touching  = {c for c, g in contacts.items() if g <= 1}
      nearby    = {c for c, g in contacts.items() if g <= 5}
      all_seen  = set(contacts)          # everything within margin

    margin=5 keeps one action-step worth of proximity in view.
    """
    player_objs = [o for o in objects if o.color in player_colors]
    target_objs = [o for o in objects
                   if o.color not in player_colors and not o.is_background]
    result: dict[int, int] = {}
    for po in player_objs:
        pb = po.bbox
        for to in target_objs:
            tb = to.bbox
            # Gap on each axis: 0 means bboxes share cells (actual overlap),
            # 1 means adjacent (touching edges but no shared cells).
            row_gap = max(0, max(pb["r_min"], tb["r_min"]) - min(pb["r_max"], tb["r_max"]))
            col_gap = max(0, max(pb["c_min"], tb["c_min"]) - min(pb["c_max"], tb["c_max"]))
            gap = max(row_gap, col_gap)   # Chebyshev-style: worst-axis gap
            if gap <= margin:
                prev = result.get(to.color)
                if prev is None or gap < prev:
                    result[to.color] = gap
    return result


def infer_action_directions(
    action_effects: dict,
) -> dict[str, tuple[int, int]]:
    """Derive (dr, dc) per action from accumulated action_effects observations.

    Scans all object_observations for non-background moved objects and returns
    the most commonly observed non-zero (dr, dc) per action name.

    Returns {} for actions with no non-stationary observations yet.
    """
    result: dict[str, tuple[int, int]] = {}
    for action_name, effects in action_effects.items():
        dr_votes: dict[int, int] = {}
        dc_votes: dict[int, int] = {}
        for obs in effects.get("object_observations", []):
            for mv in obs.get("moved", []):
                if mv.get("is_background"):
                    continue
                dr = mv.get("delta_r", 0)
                dc = mv.get("delta_c", 0)
                if dr != 0 or dc != 0:
                    dr_votes[dr] = dr_votes.get(dr, 0) + 1
                    dc_votes[dc] = dc_votes.get(dc, 0) + 1
        if dr_votes or dc_votes:
            best_dr = max(dr_votes, key=lambda k: dr_votes[k]) if dr_votes else 0
            best_dc = max(dc_votes, key=lambda k: dc_votes[k]) if dc_votes else 0
            best_dr = int(round(best_dr))
            best_dc = int(round(best_dc))
            if best_dr != 0 or best_dc != 0:
                result[action_name] = (best_dr, best_dc)
    return result


def detect_arena_delta(
    frame_before: list,
    frame_after: list,
    player_colors: set[int],
    min_size_change: int = 3,
) -> dict:
    """Compare two frames — ignoring player-colored pixels — to identify world-state changes.

    Filters out the player object so that only changes in the environment
    (not the player's own movement) are reported.  Useful for inferring causal
    effects of contact events: "what changed in the world when the player
    touched this object?"

    Parameters
    ----------
    frame_before, frame_after : list[list[int]]
        Frames immediately before and after the action that caused contact.
    player_colors : set[int]
        Colors belonging to the player (excluded from comparison).
    min_size_change : int
        Minimum absolute pixel-count change to report a size change (noise filter).

    Returns
    -------
    dict with keys:
        "appeared"     : list of {color, size, centroid, bbox} — new non-player objects
        "disappeared"  : list of {color, size, centroid, bbox} — objects that vanished
        "changed_size" : list of {color, before_size, after_size, delta} — resized objects
        "any_change"   : bool — True if any of the above lists is non-empty
    """
    def _non_player_objs(frame: list) -> dict[int, "ObjectRecord"]:
        return {
            o.color: o
            for o in detect_objects(frame)
            if o.color not in player_colors and not o.is_background
        }

    before_objs = _non_player_objs(frame_before)
    after_objs  = _non_player_objs(frame_after)
    all_colors  = set(before_objs) | set(after_objs)

    appeared:     list[dict] = []
    disappeared:  list[dict] = []
    changed_size: list[dict] = []

    for color in all_colors:
        bef = before_objs.get(color)
        aft = after_objs.get(color)
        if bef is None and aft is not None:
            appeared.append({
                "color": color,
                "size": aft.size,
                "centroid": aft.centroid,
                "bbox": aft.bbox,
            })
        elif bef is not None and aft is None:
            disappeared.append({
                "color": color,
                "size": bef.size,
                "centroid": bef.centroid,
                "bbox": bef.bbox,
            })
        elif bef is not None and aft is not None:
            delta = aft.size - bef.size
            if abs(delta) >= min_size_change:
                changed_size.append({
                    "color": color,
                    "before_size": bef.size,
                    "after_size":  aft.size,
                    "delta":       delta,
                })

    any_change = bool(appeared or disappeared or changed_size)
    return {
        "appeared":     appeared,
        "disappeared":  disappeared,
        "changed_size": changed_size,
        "any_change":   any_change,
    }


def _compress_route(route: list[str]) -> str:
    """Compress ['ACTION1','ACTION1','ACTION3'] -> '2xACTION1 + 1xACTION3'."""
    if not route:
        return "(no moves)"
    parts: list[str] = []
    cur, count = route[0], 1
    for act in route[1:]:
        if act == cur:
            count += 1
        else:
            parts.append(f"{count}x{cur}")
            cur, count = act, 1
    parts.append(f"{count}x{cur}")
    return " + ".join(parts)


def plan_route(
    frame: list,
    player_objects: list[ObjectRecord],
    target_centroid: tuple[float, float],
    action_directions: dict[str, tuple[int, int]],
    max_steps: int = 20,
    target_colors: set[int] | None = None,
) -> list[str] | None:
    """BFS path planner in action-space.

    Finds the shortest sequence of actions that moves the player so its
    bounding box overlaps or is adjacent to the target centroid.  Returns
    None if the target is unreachable within max_steps, or if inputs are
    insufficient (no player, no action directions).

    Parameters
    ----------
    frame : list[list[int]]
        Current game frame — used to build the passability map.
    player_objects : list[ObjectRecord]
        All objects identified as the player (dynamic colors).
    target_centroid : (row, col)
        Goal centroid to reach.
    action_directions : {action_name: (dr, dc)}
        Confirmed movement deltas per action.  Must have at least one entry.
    max_steps : int
        BFS depth limit (default 20).
    target_colors : set[int], optional
        Colors of all exploration target objects.  Their cells are removed
        from the blocked set so the player can navigate into them.  Pass all
        uncontacted object colors together so co-located objects of different
        colors (e.g., a cross made of two colors) don't block each other.
    """
    from collections import deque

    if not player_objects or not action_directions:
        return None

    frame_h = len(frame)
    frame_w = len(frame[0]) if frame else 0
    if frame_h == 0 or frame_w == 0:
        return None

    player_colors = {o.color for o in player_objects}

    # Passability map.  Only SMALL-TO-MEDIUM objects are treated as walls.
    # Large non-background objects (> 10% of frame area) are arena surfaces
    # that the player navigates ON — they must remain passable.  This avoids
    # erroneously blocking the arena interior when it uses the same detection
    # threshold as foreground walls but is structurally different.
    #
    # Rule: block a cell only if it belongs to an object that is:
    #   - not background (is_background=False)
    #   - not the player
    #   - not an exploration target
    #   - smaller than 10% of the total frame area
    #
    # This correctly passes arena surfaces (typically 15-30% of frame) while
    # blocking walls and UI elements (typically 0.1-6% of frame).
    _passable_targets = target_colors or set()
    objects = detect_objects(frame)
    total_cells = frame_h * frame_w
    wall_size_limit = total_cells * 0.10   # anything larger is arena, not wall
    blocked: set[tuple[int, int]] = set()
    for o in objects:
        if o.is_background or o.color in player_colors or o.color in _passable_targets:
            continue
        if o.size > wall_size_limit:
            continue  # large object — treat as arena surface (passable)
        for r in range(o.bbox["r_min"], o.bbox["r_max"] + 1):
            for c in range(o.bbox["c_min"], o.bbox["c_max"] + 1):
                if r < frame_h and c < frame_w and frame[r][c] == o.color:
                    blocked.add((r, c))

    tr, tc = target_centroid

    def _bfs_from(candidate: "ObjectRecord") -> list[str] | None:
        """Run BFS starting from a specific candidate player object."""
        p_h = int(candidate.bbox["r_max"]) - int(candidate.bbox["r_min"]) + 1
        p_w = int(candidate.bbox["c_max"]) - int(candidate.bbox["c_min"]) + 1
        start_r = int(candidate.bbox["r_min"])
        start_c = int(candidate.bbox["c_min"])

        def centroid_of(r_min: int, c_min: int) -> tuple[float, float]:
            return (r_min + p_h / 2, c_min + p_w / 2)

        def is_valid(r_min: int, c_min: int) -> bool:
            if r_min < 0 or c_min < 0:
                return False
            if r_min + p_h > frame_h or c_min + p_w > frame_w:
                return False
            for r in range(r_min, r_min + p_h):
                for c in range(c_min, c_min + p_w):
                    if (r, c) in blocked:
                        return False
            return True

        def is_goal(r_min: int, c_min: int) -> bool:
            # Require the player bbox to actually contain the target centroid
            # (same semantics as detect_contacts gap=0: actual cell overlap).
            tr_int = int(round(tr))
            tc_int = int(round(tc))
            return (r_min <= tr_int < r_min + p_h and
                    c_min <= tc_int < c_min + p_w)

        start = (start_r, start_c)
        if not is_valid(start_r, start_c):
            return None

        visited: set[tuple[int, int]] = {start}
        queue: deque = deque([(start, [])])

        while queue:
            (r, c), path = queue.popleft()
            if len(path) >= max_steps:
                continue
            for action_name, (dr, dc) in action_directions.items():
                nr, nc = r + int(dr), c + int(dc)
                if (nr, nc) in visited:
                    continue
                if not is_valid(nr, nc):
                    continue
                new_path = path + [action_name]
                if is_goal(nr, nc):
                    return new_path
                visited.add((nr, nc))
                queue.append(((nr, nc), new_path))
        return None

    # Try each candidate player object as the starting position; return the
    # shortest successful route.  This handles cases where the same color
    # appears in both the UI (static) and the playing area (dynamic).
    best_route: list[str] | None = None
    for candidate in player_objects:
        route = _bfs_from(candidate)
        if route is not None:
            if best_route is None or len(route) < len(best_route):
                best_route = route

    return best_route


def _nav_hint(player_centroid: tuple, target_centroid: tuple) -> str:
    """Straight-line nav hint ignoring walls (fallback when BFS unavailable)."""
    dr = target_centroid[0] - player_centroid[0]
    dc = target_centroid[1] - player_centroid[1]
    parts = []
    if abs(dr) >= 2.5:
        n = round(abs(dr) / 5)
        parts.append(f"{n}x{'ACTION2' if dr > 0 else 'ACTION1'}")
    if abs(dc) >= 2.5:
        n = round(abs(dc) / 5)
        parts.append(f"{n}x{'ACTION4' if dc > 0 else 'ACTION3'}")
    if not parts:
        return "already adjacent"
    return " + ".join(parts)


def format_structural_context(
    frame: list,
    concept_bindings: dict | None = None,
    known_dynamic_colors: set[int] | None = None,
    explored_colors: set[int] | None = None,
    action_directions: dict[str, tuple[int, int]] | None = None,
    contact_events: list[dict] | None = None,
) -> str:
    """Produce a human-readable structural context string for the OBSERVER prompt.

    Highlights:
    - Confirmed action directions (skip re-characterization)
    - Container/content pairs (bordered boxes with patterns inside)
    - Column and row alignments between objects (especially player ↔ goal)
    - Objects whose color has been seen moving (dynamic) vs never moved (static)
    - Contact events: causal world-state changes observed when player touched objects
    - Exploration manifest with BFS-computed routes (or straight-line fallback)

    Parameters
    ----------
    frame : list[list[int]]
        Current frame.
    concept_bindings : dict, optional
        {color_int: role | {"role":..., ...}} for labeling objects by role.
    known_dynamic_colors : set[int], optional
        Colors observed moving in previous steps.
    explored_colors : set[int], optional
        Colors the player has already made full contact with.
    action_directions : dict, optional
        {action_name: (dr, dc)} confirmed from observations.  When provided,
        the exploration manifest shows BFS-computed exact routes instead of
        straight-line approximations, and a directions summary is prepended
        so the MEDIATOR does not waste steps re-characterizing actions.
    contact_events : list[dict], optional
        Records of world-state changes observed when the player touched specific
        objects.  Each entry: {"touched_color": int, "step": int, "delta": dict}
        where delta has "appeared", "disappeared", "changed_size" sub-lists.
        Rendered as a "Contact history" section so agents can reason causally
        about which objects unlock other objects or trigger state transitions.
    """
    objects = detect_objects(frame)
    containment = detect_containment(objects)
    spatial = detect_spatial_relations(objects)
    bindings = concept_bindings or {}
    dynamic = known_dynamic_colors or set()
    explored = explored_colors or set()
    directions = action_directions or {}
    events = contact_events or []

    def _role(color: int) -> str:
        raw = bindings.get(color)
        if isinstance(raw, dict):
            return raw.get("role", "")
        return raw or ""

    def _label(o: ObjectRecord) -> str:
        role = _role(o.color)
        name = f"{color_name(o.color)}"
        if role:
            name = f"{role}({name})"
        nature = "dynamic" if o.color in dynamic else "static"
        return f"{name} size={o.size} {o.width}w×{o.height}h @{_fmt_pt(o.centroid)} [{nature}]"

    lines: list[str] = []

    # --- Slot strip puzzle detection (TR87-type) ---
    # This is a per-game artifact: it encodes hardcoded knowledge of the TR87
    # rule grammar (slot semantics, alter mode, etc.) and would not generalize
    # to an unseen game. In competition mode it must be excluded so KF relies
    # only on game-agnostic primitives + concepts learned from training.
    import ensemble as _ens  # local import: avoid circular at module load
    if not getattr(_ens, "COMPETITION_MODE", False):
        _slot_data = detect_slot_strips(frame)
        if _slot_data:
            lines.append(format_slot_strip_context(_slot_data))

    # Pre-compute groups to detect puzzle levels (groups are reused at the end).
    _groups = detect_groups(frame, objects, containment) if objects and containment else []
    _is_puzzle = len(_groups) >= 6  # puzzle levels have many grouped slots

    # --- Confirmed action directions (suppresses re-characterization) ---
    # Skip for puzzle levels: navigation-era directions don't apply to puzzles.
    if directions and not _is_puzzle:
        dir_labels = {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}
        dir_parts = []
        for act in sorted(directions):
            dr, dc = directions[act]
            # Normalise to unit direction for label
            label = dir_labels.get((0 if dr == 0 else (-1 if dr < 0 else 1),
                                    0 if dc == 0 else (-1 if dc < 0 else 1)), "?")
            dist = abs(dr) if dr != 0 else abs(dc)
            dir_parts.append(f"{act}={label}{dist}")
        lines.append(
            "Action directions (confirmed — do NOT waste steps re-characterizing): "
            + "  ".join(dir_parts)
        )

    # --- Containers ---
    if containment:
        lines.append("Containers (bordered regions enclosing other objects):")
        # Group by container
        by_container: dict[int, list[ContainmentRelation]] = {}
        for rel in containment:
            key = id(rel.container)
            by_container.setdefault(key, []).append(rel)
        for rels in by_container.values():
            c = rels[0].container
            lines.append(f"  BOX  {_label(c)}")
            for rel in rels:
                lines.append(f"    └─ {_label(rel.content)}")

    # --- Column/row alignments (most useful for navigation) ---
    # Skip container↔content pairs — already shown in Containers section.
    container_content_pairs: set[frozenset] = {
        frozenset({id(r.container), id(r.content)}) for r in containment
    }
    align_lines: list[str] = []
    for rel in spatial:
        if not (rel.col_aligned or rel.row_aligned):
            continue
        if frozenset({id(rel.obj_a), id(rel.obj_b)}) in container_content_pairs:
            continue
        kind = []
        if rel.col_aligned:
            kind.append("same-col")
        if rel.row_aligned:
            kind.append("same-row")
        align_lines.append(
            f"  {_label(rel.obj_a)}  {rel.relation}  {_label(rel.obj_b)}"
            f"  [{', '.join(kind)}, dist={rel.distance}]"
        )
    if align_lines:
        lines.append("Spatial alignments:")
        lines.extend(align_lines)

    # --- Uncontained static foreground objects not yet role-bound ---
    contained_ids = {id(r.content) for r in containment}
    contained_ids |= {id(r.container) for r in containment}
    orphan_static = [
        o for o in objects
        if not o.is_background
        and o.color not in dynamic
        and id(o) not in contained_ids
    ]
    if orphan_static:
        lines.append("Other static foreground objects (not yet role-bound):")
        for o in orphan_static:
            lines.append(f"  {_label(o)}")

    # --- Exploration manifest -------------------------------------------
    # All non-background, non-player foreground objects, grouped by whether
    # the player has already made contact with them.  Gives MEDIATOR a clear
    # checklist: what is still unknown and how to reach it.

    # Build exclusion sets from concept_bindings
    _non_interactive_roles_set = {"wall", "background", "border", "reference_pattern", "reference_box"}
    _step_counter_colors_set: set[int] = set()
    _non_interactive_colors_set: set[int] = set()
    for _k, _v in bindings.items():
        try:
            _color_int = int(_k)
        except (ValueError, TypeError):
            continue
        if isinstance(_v, dict):
            _bound_role = _v.get("role", "")
        elif isinstance(_v, str):
            _bound_role = _v
        else:
            continue
        if _bound_role == "step_counter":
            _step_counter_colors_set.add(_color_int)
        elif _bound_role in _non_interactive_roles_set:
            _non_interactive_colors_set.add(_color_int)
    # Also pick up explicit wall_colors list if present
    for _wc in bindings.get("wall_colors", []):
        try:
            _non_interactive_colors_set.add(int(_wc))
        except (ValueError, TypeError):
            pass

    # Bar-shaped step_counter instances (excluded — not interaction targets)
    _step_counter_bar_ids: set[int] = set()
    for _o in objects:
        if _o.color in _step_counter_colors_set:
            if _o.aspect_ratio >= 3.0 or _o.aspect_ratio <= 0.33:
                _step_counter_bar_ids.add(id(_o))

    # Container objects (already shown in containers section; not navigation targets)
    _container_obj_ids_manifest = {id(r.container) for r in containment}

    # Size cap: objects > 10× total player size are structural (arena floor/walls)
    _player_total_size = sum(o.size for o in objects if o.color in dynamic) or 1
    _max_manifest_size = _player_total_size * 10

    fg_non_player = [
        o for o in objects
        if not o.is_background
        and o.color not in dynamic
        and o.color not in _non_interactive_colors_set
        and id(o) not in _container_obj_ids_manifest
        and id(o) not in _step_counter_bar_ids
        and o.size <= _max_manifest_size
    ]

    # Determine player centroid for navigation hints
    player_objs = [o for o in objects if o.color in dynamic]
    if player_objs and fg_non_player:
        # Use centroid of the largest dynamic object as the player reference
        player_ref = max(player_objs, key=lambda o: o.size)
        p_centroid = player_ref.centroid

        untouched = [o for o in fg_non_player if o.color not in explored]
        touched   = [o for o in fg_non_player if o.color in explored]

        lines.append("")
        lines.append(
            f"Exploration manifest  "
            f"({len(touched)} contacted, {len(untouched)} not yet contacted):"
        )
        if touched:
            lines.append("  Contacted:")
            for o in touched:
                lines.append(f"    [DONE] {_label(o)}")
        if untouched:
            lines.append("  NOT YET CONTACTED — behavior unknown:")
            player_objs_list = [o for o in objects if o.color in dynamic]
            # All untouched colors are made passable together so co-located
            # objects of different colors (e.g., a cross of two colors) don't
            # block each other during BFS routing.
            untouched_colors = {o.color for o in untouched}
            for o in untouched:
                if directions:
                    route = plan_route(
                        frame, player_objs_list, o.centroid, directions,
                        max_steps=30,
                        target_colors=untouched_colors,
                    )
                    if route:
                        nav_str = f"route: {_compress_route(route)}"
                    else:
                        # BFS couldn't find a path — walls or arena shape block the
                        # straight approach.  Flag for LLM-assisted route planning.
                        nav_str = (
                            f"nav≈{_nav_hint(p_centroid, o.centroid)} "
                            f"[NEEDS-PLANNING: BFS found no direct route — "
                            f"reason likely a wall or arena gap; plan a detour]"
                        )
                else:
                    nav_str = f"nav≈{_nav_hint(p_centroid, o.centroid)}"
                lines.append(f"    [TODO] {_label(o)}  {nav_str}")

    # --- Contact history (causal world-state changes from touch events) ---
    # Each entry records what changed in the environment when the player
    # made full contact with a specific object color.  This lets agents
    # reason: "touching colorX caused colorY to appear — I should go touch Y."
    if events:
        lines.append("")
        lines.append("Contact history (world changes observed when player touched an object):")
        for ev in events:
            color = ev.get("touched_color")
            step  = ev.get("step", "?")
            delta = ev.get("delta", {})
            effects: list[str] = []
            for a in delta.get("appeared", []):
                cr, cc = a["centroid"]
                effects.append(
                    f"color{a['color']}({color_name(a['color'])}) APPEARED "
                    f"size={a['size']} @({cr:.0f},{cc:.0f})"
                )
            for d in delta.get("disappeared", []):
                cr, cc = d["centroid"]
                effects.append(
                    f"color{d['color']}({color_name(d['color'])}) DISAPPEARED "
                    f"(was size={d['size']} @({cr:.0f},{cc:.0f}))"
                )
            for ch in delta.get("changed_size", []):
                sign = "+" if ch["delta"] > 0 else ""
                effects.append(
                    f"color{ch['color']}({color_name(ch['color'])}) size changed "
                    f"{ch['before_size']}->{ch['after_size']} ({sign}{ch['delta']})"
                )
            if effects:
                lines.append(
                    f"  [step {step}] touching color{color}({color_name(color)}) caused:"
                )
                for eff in effects:
                    lines.append(f"    -> {eff}")
            else:
                lines.append(
                    f"  [step {step}] touching color{color}({color_name(color)}): "
                    f"no detected world change (may be a passive/inert object)"
                )

    # --- Visual reasoning: grouping, types, focus, mismatches ---------------
    if _groups:
        groups = _groups
        if groups:
            types = classify_group_types(groups)
            # Partition into reference (top half) and target (bottom half)
            rows = len(frame)
            midpoint = rows / 2
            ref_groups = [g for g in groups if g.centroid[0] < midpoint]
            tgt_groups = [g for g in groups if g.centroid[0] >= midpoint]

            # Build a position label for each group: "g3@(row,col)"
            def _pos_label(g: ObjectGroup) -> str:
                r, c = g.centroid
                return f"{g.id}@({int(r)},{int(c)})"

            if len(types) > 1:
                lines.append("\nVisual groups:")
                for t in types:
                    ids = [_pos_label(g) for g in t.instances]
                    lines.append(f"  Type {t.type_id} ({len(t.instances)}x): {t.description}")
                    lines.append(f"    instances: {ids}")

            if ref_groups and tgt_groups:
                # Sort target groups by column for stable slot ordering
                tgt_sorted = sorted(tgt_groups, key=lambda g: g.centroid[1])
                slot_index = {id(g): i for i, g in enumerate(tgt_sorted)}

                def _slot_label(g: ObjectGroup) -> str:
                    idx = slot_index.get(id(g))
                    if idx is not None:
                        return f"{g.id}@({int(g.centroid[0])},{int(g.centroid[1])}) [slot {idx+1}/{len(tgt_sorted)}]"
                    return _pos_label(g)

                # Focus detection
                focus = detect_focus(objects, groups, indicator_colors={0})
                if focus and focus.target_group:
                    fg = focus.target_group
                    lines.append(
                        f"\nFocus/cursor: {color_name(focus.color)} markers "
                        f"-> {_slot_label(fg)}"
                    )

                # Pairwise mismatches (cyan strip -> reference pairs)
                pw = detect_pairwise_mismatches(ref_groups, tgt_groups, frame)
                if pw:
                    lines.append(f"\nReference slot mapping ({len(pw)} matched):")
                    for mm in pw:
                        lines.append(
                            f"  {_slot_label(mm.group_b)} -> ref {_pos_label(mm.group_a)}: {mm.detail}"
                        )

                # Simple mismatches for target groups NOT already covered
                # by pairwise mapping (to avoid conflicting signals).
                pw_covered = {id(mm.group_b) for mm in pw} if pw else set()
                remaining_tgt = [g for g in tgt_groups if id(g) not in pw_covered]
                sm = detect_mismatches(ref_groups, remaining_tgt, frame)
                # Include pairwise results as matches for the count
                total = len(sm) + len(pw_covered)
                pw_match_count = sum(1 for mm in (pw or []) if mm.match)
                matched = [mm for mm in sm if mm.match]
                mismatched = [mm for mm in sm if not mm.match]
                all_matched = len(matched) + pw_match_count
                # Applicability gate: only emit mismatches if there's positive
                # evidence this is actually a pattern-matching puzzle (any
                # confirmed matches or pairwise mappings). Otherwise the
                # "closest to" signal is noise on navigation games like ls20.
                _puzzle_evidence = bool(matched) or bool(pw)
                if matched:
                    lines.append(f"\nContent matches ({all_matched}/{total}):")
                    for mm in matched:
                        lines.append(f"  {_slot_label(mm.group_b)}: MATCH via {mm.best_transform}")
                if mismatched and _puzzle_evidence:
                    lines.append(f"\nContent mismatches ({len(mismatched)}/{total}):")
                    for mm in mismatched:
                        lines.append(f"  {_slot_label(mm.group_b)}: closest to {_pos_label(mm.group_a)} ({mm.detail})")

    return "\n".join(lines) if lines else "  (no structural context detected)"


# ---------------------------------------------------------------------------
# Visual reasoning primitives: Grouping, Type Equivalence, Focus, Mismatch
# ---------------------------------------------------------------------------

@dataclass
class ObjectGroup:
    """A logical group of spatially related objects treated as one unit.

    E.g., a cyan border-box containing a black shape = one group.
    A (cyan-box, connector, pink-box) triple = one pair-group.
    """
    id: str                              # unique group id, e.g. "g0", "g1"
    members: list[ObjectRecord]          # constituent objects
    role: str = ""                       # e.g. "reference_pair", "editable_slot"
    group_type: str = ""                 # type tag for equivalence, e.g. "box_pair"
    bbox: dict = field(default_factory=dict)  # combined bounding box
    anchor: ObjectRecord | None = None   # primary member (e.g. the border box)
    content_mask: list[list[int]] | None = None  # extracted foreground mask

    @property
    def centroid(self) -> tuple[float, float]:
        if not self.members:
            return (0.0, 0.0)
        r = sum(m.centroid[0] for m in self.members) / len(self.members)
        c = sum(m.centroid[1] for m in self.members) / len(self.members)
        return (r, c)

    @property
    def label(self) -> str:
        colors = sorted(set(m.color for m in self.members))
        color_str = "+".join(color_name(c) for c in colors)
        return f"group[{self.id}]({color_str}, {self.role or self.group_type or '?'})"


@dataclass
class GroupType:
    """A type shared by multiple ObjectGroups.

    Groups of the same type have the same structural signature: same member
    color set, same spatial arrangement, same size class.
    """
    type_id: str                         # e.g. "t0", "t1"
    signature: str                       # structural fingerprint
    instances: list[ObjectGroup] = field(default_factory=list)
    description: str = ""                # human-readable, e.g. "cyan+black box pair"


@dataclass
class FocusIndicator:
    """A visual cursor/bracket selecting one element from a set."""
    indicator_objects: list[ObjectRecord]  # the cursor objects (e.g. white brackets)
    target_group: ObjectGroup | None       # which group is currently selected
    target_index: int = -1                 # index within the group type's instances
    color: int = -1                        # color of the indicator


@dataclass
class Mismatch:
    """A difference between two same-type groups."""
    group_a: ObjectGroup                  # reference group
    group_b: ObjectGroup                  # group being compared
    match: bool                           # True if content masks are identical
    best_transform: str = ""              # closest D4 transform name
    distance: float = 1.0                 # Jaccard distance (0 = identical)
    detail: str = ""                      # human-readable description


def _combined_bbox(objects: list[ObjectRecord]) -> dict:
    """Compute the bounding box that encloses all given objects."""
    r_min = min(o.bbox["r_min"] for o in objects)
    r_max = max(o.bbox["r_max"] for o in objects)
    c_min = min(o.bbox["c_min"] for o in objects)
    c_max = max(o.bbox["c_max"] for o in objects)
    return {"r_min": r_min, "r_max": r_max, "c_min": c_min, "c_max": c_max}


def _bbox_adjacent(a: dict, b: dict, gap: int = 3) -> bool:
    """True if two bounding boxes are within *gap* cells of each other
    horizontally or vertically (but not overlapping)."""
    # Horizontal adjacency (same row band)
    row_overlap = a["r_min"] <= b["r_max"] and b["r_min"] <= a["r_max"]
    if row_overlap:
        h_gap = max(b["c_min"] - a["c_max"], a["c_min"] - b["c_max"])
        if 0 < h_gap <= gap:
            return True
    # Vertical adjacency (same column band)
    col_overlap = a["c_min"] <= b["c_max"] and b["c_min"] <= a["c_max"]
    if col_overlap:
        v_gap = max(b["r_min"] - a["r_max"], a["r_min"] - b["r_max"])
        if 0 < v_gap <= gap:
            return True
    return False


def _bbox_overlaps(a: dict, b: dict) -> bool:
    """True if two bounding boxes overlap at all."""
    return (a["r_min"] <= b["r_max"] and b["r_min"] <= a["r_max"]
            and a["c_min"] <= b["c_max"] and b["c_min"] <= a["c_max"])


def _group_signature(members: list[ObjectRecord]) -> str:
    """Structural fingerprint: sorted color set + size class + arrangement.

    Two groups with the same signature are the same type.
    Objects smaller than 3 cells are treated as border artifacts and excluded
    from the size/arrangement parts of the signature (but their colors are
    still included).
    """
    colors = tuple(sorted(set(m.color for m in members)))
    # Filter out border artifacts for size/arrangement
    significant = [m for m in members if m.size >= 3]
    if not significant:
        significant = members  # fallback
    sizes = tuple(sorted(_size_bucket(m.size) for m in significant))
    # Arrangement: relative positions (quantized)
    if len(significant) >= 2:
        cx = sum(m.centroid[1] for m in significant) / len(significant)
        cy = sum(m.centroid[0] for m in significant) / len(significant)
        arrangement = tuple(sorted(
            (round((m.centroid[0] - cy) / 5), round((m.centroid[1] - cx) / 5))
            for m in significant
        ))
    else:
        arrangement = ()
    return f"colors={colors}|sizes={sizes}|arr={arrangement}"


def _size_bucket(size: int) -> str:
    if size <= 5:
        return "tiny"
    if size <= 20:
        return "small"
    if size <= 60:
        return "medium"
    if size <= 200:
        return "large"
    return "xlarge"


def _cluster_objects(
    objects: list[ObjectRecord], gap: int = 3,
) -> list[list[ObjectRecord]]:
    """Cluster objects by spatial proximity using single-linkage.

    Two objects are linked if their bounding boxes are within *gap* cells.
    Returns a list of clusters (each a list of objects).
    """
    n = len(objects)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = objects[i].bbox, objects[j].bbox
            # Check horizontal gap (row-overlap + column gap)
            row_overlap = bi["r_min"] <= bj["r_max"] and bj["r_min"] <= bi["r_max"]
            col_overlap = bi["c_min"] <= bj["c_max"] and bj["c_min"] <= bi["c_max"]
            h_gap = max(bj["c_min"] - bi["c_max"], bi["c_min"] - bj["c_max"])
            v_gap = max(bj["r_min"] - bi["r_max"], bi["r_min"] - bj["r_max"])
            if (row_overlap and h_gap <= gap) or (col_overlap and v_gap <= gap):
                union(i, j)

    from collections import defaultdict
    clusters_map: dict[int, list[ObjectRecord]] = defaultdict(list)
    for i in range(n):
        clusters_map[find(i)].append(objects[i])
    return list(clusters_map.values())


# ---- 1. GROUPING ----------------------------------------------------------

def detect_groups(
    frame: list[list[int]],
    objects: list[ObjectRecord],
    containment: list[ContainmentRelation] | None = None,
    *,
    adjacency_gap: int = 5,
    min_group_size: int = 2,
    max_container_fraction: float = 0.10,
) -> list[ObjectGroup]:
    """Detect logical groups of objects based on containment and adjacency.

    Strategy:
    1. Filter containment to use only *tightest* (smallest) container per
       content object, excluding region-scale containers.
    2. Each tight container + its contents = one atomic "box group."
    3. Adjacent box groups of different colors are merged into "pair groups."
    4. Each group gets a content mask extracted from the inner bbox.

    Parameters
    ----------
    frame : the full game frame (needed for mask extraction)
    objects : all detected objects
    containment : pre-computed containment relations (or None to compute)
    adjacency_gap : max pixel gap between boxes to consider them paired
    min_group_size : minimum members to form a group (default 2)
    max_container_fraction : containers whose size exceeds this fraction of
        the frame are excluded (too large = background region, not a box)
    """
    if containment is None:
        containment = detect_containment(objects)

    total_cells = len(frame) * len(frame[0]) if frame else 1
    max_container_size = int(total_cells * max_container_fraction)

    # Step 1: For each content object, find its TIGHTEST (smallest) container.
    # Exclude region-scale containers.
    tightest: dict[int, tuple[ObjectRecord, ObjectRecord]] = {}  # id(content) -> (container, content)
    for rel in containment:
        if rel.container.size > max_container_size:
            continue  # skip region-scale containers
        cid = id(rel.content)
        if cid not in tightest or rel.container.size < tightest[cid][0].size:
            tightest[cid] = (rel.container, rel.content)

    # Step 2: Build atomic box groups: each container + its tight contents.
    container_to_contents: dict[int, list[ObjectRecord]] = {}
    container_objs: dict[int, ObjectRecord] = {}
    for container, content in tightest.values():
        key = id(container)
        container_objs[key] = container
        container_to_contents.setdefault(key, []).append(content)

    # Each box group = container + its contents.
    # If a container has multiple spatially separated contents, split into
    # individual slot groups (e.g., a strip with 5 sub-shapes).
    box_groups: dict[int, list[ObjectRecord]] = {}  # key = id(container) or synthetic
    split_slot_keys: set[int] = set()  # keys from strip-split groups
    _next_synthetic = -1  # synthetic keys for split groups

    for key, contents in container_to_contents.items():
        container = container_objs[key]
        if len(contents) <= 2:
            box_groups[key] = [container] + contents
            continue

        # Cluster spatially separated contents via single-linkage clustering.
        clusters = _cluster_objects(contents, gap=1)
        if len(clusters) <= 1:
            box_groups[key] = [container] + contents
        else:
            # Split: each cluster + shared container = one slot group
            for cluster in clusters:
                _next_synthetic -= 1
                syn_key = _next_synthetic
                box_groups[syn_key] = [container] + list(cluster)
                container_objs[syn_key] = container
                split_slot_keys.add(syn_key)

    # Step 2b: Exclude nested box groups entirely.
    # If a container is itself a content object (has its own tightest container),
    # it's a nested level (e.g., black 5×5 inside a colored 7×7 box containing
    # pink pattern content).  These create duplicate groups that interfere with
    # pairing and deduplication.
    content_obj_ids = set(id(content) for _, content in tightest.values())
    nested_keys = {k for k in box_groups if id(container_objs[k]) in content_obj_ids}
    for nk in nested_keys:
        del box_groups[nk]
        # Keep container_objs entries (harmless, may be shared)

    # Step 3: Merge adjacent box groups of different container colors into pairs.
    keys = list(box_groups.keys())
    # Union-find: each key starts in its own set
    parent: dict[int, int] = {k: k for k in keys}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Helper: content-effective bbox for split slots (use content position,
    # not the full strip bbox), falling back to container bbox otherwise.
    def _effective_bb(key: int) -> dict:
        if key in split_slot_keys:
            container = container_objs[key]
            content = [m for m in box_groups[key] if m is not container]
            if content:
                return _combined_bbox(content)
        return container_objs[key].bbox

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a_box = container_objs[keys[i]]
            b_box = container_objs[keys[j]]
            # Only pair different-color containers of similar size
            if a_box.color == b_box.color:
                continue
            size_ratio = max(a_box.size, b_box.size) / max(min(a_box.size, b_box.size), 1)
            if size_ratio > 3.0:
                continue  # too different in size to be a pair

            # Use content-effective bbox for spatial checks
            a_bb = _effective_bb(keys[i])
            b_bb = _effective_bb(keys[j])

            # Horizontal adjacency (boxes share a row band)
            row_overlap = a_bb["r_min"] <= b_bb["r_max"] and b_bb["r_min"] <= a_bb["r_max"]
            if row_overlap:
                h_gap = max(b_bb["c_min"] - a_bb["c_max"], a_bb["c_min"] - b_bb["c_max"])
                if 0 < h_gap <= adjacency_gap:
                    union(keys[i], keys[j])
                    continue

            # Vertical adjacency for split slot groups (stacked strips).
            # Only applies when both groups are from strip-split containers,
            # preventing reference 7×7 boxes from merging with target strips.
            if keys[i] in split_slot_keys and keys[j] in split_slot_keys:
                col_overlap = a_bb["c_min"] <= b_bb["c_max"] and b_bb["c_min"] <= a_bb["c_max"]
                if col_overlap:
                    v_gap = max(b_bb["r_min"] - a_bb["r_max"], a_bb["r_min"] - b_bb["r_max"])
                    if 0 < v_gap <= adjacency_gap * 3:
                        union(keys[i], keys[j])

    # Collect merged groups
    from collections import defaultdict
    root_to_keys: dict[int, list[int]] = defaultdict(list)
    for k in keys:
        root_to_keys[find(k)].append(k)

    groups: list[ObjectGroup] = []
    gid = 0

    for root, member_keys in root_to_keys.items():
        all_members: list[ObjectRecord] = []
        anchors: list[ObjectRecord] = []
        seen_ids: set[int] = set()
        for mk in member_keys:
            anchors.append(container_objs[mk])
            for m in box_groups[mk]:
                if id(m) not in seen_ids:
                    seen_ids.add(id(m))
                    all_members.append(m)

        if len(all_members) < min_group_size:
            continue

        combined = _combined_bbox(all_members)

        # Extract content mask from each anchor's inner bbox (skip 1-cell border).
        # For split slot groups (anchor much larger than content), use the
        # content objects' bbox instead of the anchor's full inner bbox.
        anchor_colors = set(a.color for a in anchors)
        content_objs = [m for m in all_members if m not in anchors]
        masks: list[tuple[ObjectRecord, list[list[int]]]] = []
        for anchor in anchors:
            # Scope content to objects within this anchor's bbox.
            # Critical for vertically-paired groups where two anchors (strips)
            # share the content list — each anchor should only use its own content.
            a_bb = anchor.bbox
            anchor_content = [
                m for m in content_objs
                if (a_bb["r_min"] <= m.centroid[0] <= a_bb["r_max"]
                    and a_bb["c_min"] <= m.centroid[1] <= a_bb["c_max"])
            ]
            if not anchor_content:
                anchor_content = content_objs  # fallback
            # Check if this is a split slot (anchor >> content)
            content_size = sum(o.size for o in anchor_content)
            is_slot = anchor_content and anchor.size > content_size * 5
            if is_slot and anchor_content:
                # Use anchor-scoped content objects' bbox
                inner = _combined_bbox(anchor_content)
            else:
                # Normal: use anchor's inner bbox (skip 1-cell border)
                inner = {
                    "r_min": anchor.bbox["r_min"] + 1,
                    "r_max": anchor.bbox["r_max"] - 1,
                    "c_min": anchor.bbox["c_min"] + 1,
                    "c_max": anchor.bbox["c_max"] - 1,
                }
            shape_color = _find_shape_color(frame, inner, {anchor.color})
            if shape_color is not None:
                mask = extract_subgrid(frame, inner, foreground_color=shape_color)
                masks.append((anchor, mask))

        # Use the first mask as the group's content_mask
        primary_mask = masks[0][1] if masks else None

        # For split slot groups, use content-only bbox and members to avoid
        # the shared container inflating the signature.
        anchor_ids = {id(a) for a in anchors}
        content_objs_only = [m for m in all_members if id(m) not in anchor_ids]
        is_slot = (
            len(anchors) == 1
            and content_objs_only
            and anchors[0].size > 50  # only large containers can be strips
            and anchors[0].size > sum(o.size for o in content_objs_only) * 5
        )
        if is_slot:
            effective_bbox = _combined_bbox(content_objs_only)
        else:
            effective_bbox = combined

        group = ObjectGroup(
            id=f"g{gid}",
            members=all_members,
            bbox=effective_bbox,
            anchor=anchors[0],
            content_mask=primary_mask,
        )
        group._is_slot = is_slot  # type: ignore[attr-defined]
        # Store all sub-masks as a private attribute for pairwise comparison
        group._sub_masks = masks  # type: ignore[attr-defined]
        groups.append(group)
        gid += 1

    # Deduplicate: if two groups overlap spatially and share non-anchor content,
    # keep the more specific one (smaller anchor).
    # Build content-object -> group mapping (excluding all anchors).
    # For paired groups with multiple anchors (e.g., vertically-paired strip
    # slots sharing cyan + pink strip containers), exclude ALL anchors, not
    # just g.anchor — otherwise shared strips trigger false dedup.
    all_anchor_ids: set[int] = set()
    for root, member_keys in root_to_keys.items():
        for mk in member_keys:
            all_anchor_ids.add(id(container_objs[mk]))
    content_to_groups: dict[int, list[ObjectGroup]] = {}
    for g in groups:
        for m in g.members:
            if id(m) not in all_anchor_ids:
                content_to_groups.setdefault(id(m), []).append(g)

    # Also check: if a content object in group A is the anchor of group B
    # and their bboxes overlap, they're duplicates.
    anchor_to_group: dict[int, ObjectGroup] = {}
    for g in groups:
        if g.anchor:
            anchor_to_group[id(g.anchor)] = g

    drop_groups: set[str] = set()
    for obj_id, grps in content_to_groups.items():
        # Check if this content object is also an anchor in another group
        all_groups = list(grps)
        if obj_id in anchor_to_group:
            anchor_grp = anchor_to_group[obj_id]
            if anchor_grp not in all_groups:
                # Check bbox overlap with any content group
                for cg in grps:
                    if _bbox_overlaps(cg.bbox, anchor_grp.bbox):
                        all_groups.append(anchor_grp)
                        break

        if len(all_groups) <= 1:
            continue
        # Keep the group with the smallest anchor (most specific container).
        best = min(all_groups, key=lambda g: g.anchor.size if g.anchor else 0)
        for g in all_groups:
            if g is not best:
                drop_groups.add(g.id)

    groups = [g for g in groups if g.id not in drop_groups]

    # Sort groups by spatial position (row, col) for stable IDs across cycles.
    # Same physical slot will always get the same ID regardless of detection order.
    groups.sort(key=lambda g: (round(g.centroid[0], 1), round(g.centroid[1], 1)))

    # Re-number groups sequentially
    for i, g in enumerate(groups):
        g.id = f"g{i}"

    return groups


def _find_root(merged: dict[int, set[int]], key: int) -> int:
    """Simple union-find root lookup."""
    while True:
        parent_set = merged[key]
        # The root is the key whose set we're in
        for candidate in parent_set:
            if merged[candidate] is parent_set:
                return candidate
        return key


def _find_shape_color(
    frame: list[list[int]],
    bbox: dict,
    exclude_colors: set[int],
) -> int | None:
    """Find the most common non-excluded, non-background color inside a bbox."""
    from collections import Counter
    counts: Counter = Counter()
    bg_colors = set()
    # Gather all background-like objects (the frame's dominant colors)
    r_min, r_max = bbox["r_min"], bbox["r_max"]
    c_min, c_max = bbox["c_min"], bbox["c_max"]
    for r in range(r_min, r_max + 1):
        for c in range(c_min, c_max + 1):
            val = frame[r][c]
            if val not in exclude_colors:
                counts[val] += 1

    if not counts:
        return None
    # Return the most common non-excluded color
    for color, _count in counts.most_common():
        return color
    return None


# ---- 2. TYPE EQUIVALENCE --------------------------------------------------

def classify_group_types(groups: list[ObjectGroup]) -> list[GroupType]:
    """Assign each group to a type based on structural signature.

    Groups with identical signatures (same colors, sizes, arrangement)
    are instances of the same type.
    """
    type_map: dict[str, GroupType] = {}
    tid = 0

    for g in groups:
        # For slot groups, use content-only members for signature to avoid
        # the shared container inflating the fingerprint.
        is_slot = getattr(g, "_is_slot", False)
        if is_slot and g.anchor:
            sig_members = [m for m in g.members if m is not g.anchor]
        else:
            sig_members = g.members
        sig = _group_signature(sig_members) if sig_members else _group_signature(g.members)
        if sig not in type_map:
            type_map[sig] = GroupType(
                type_id=f"t{tid}",
                signature=sig,
                description=_describe_type(g),
            )
            tid += 1
        gt = type_map[sig]
        gt.instances.append(g)
        g.group_type = gt.type_id

    return list(type_map.values())


def _describe_type(g: ObjectGroup) -> str:
    """Human-readable description of a group type."""
    colors = sorted(set(color_name(m.color) for m in g.members))
    sizes = [m.size for m in g.members]
    return f"{'+'.join(colors)} group ({len(g.members)} members, sizes {sizes})"


# ---- 3. FOCUS DETECTION ---------------------------------------------------

def detect_focus(
    objects: list[ObjectRecord],
    groups: list[ObjectGroup],
    *,
    indicator_colors: set[int] | None = None,
    indicator_max_size: int = 15,
) -> FocusIndicator | None:
    """Detect a visual focus indicator (cursor/bracket) and determine which
    group it's pointing at.

    Strategy:
    1. Find small objects of a distinct color (not used by any group) — these
       are candidate indicators (e.g., white brackets in TR87).
    2. If indicator_colors is given, use those directly.
    3. For each candidate set, find the group whose bbox they overlap or are
       closest to — that's the focused group.
    """
    # Colors used by groups
    group_colors: set[int] = set()
    for g in groups:
        for m in g.members:
            group_colors.add(m.color)

    # Find candidate indicator objects: small, non-background, non-group-color
    candidates: list[ObjectRecord] = []
    for o in objects:
        if o.is_background:
            continue
        if o.size > indicator_max_size:
            continue
        if indicator_colors is not None:
            if o.color in indicator_colors:
                candidates.append(o)
        else:
            if o.color not in group_colors:
                candidates.append(o)

    if not candidates:
        return None

    # Group candidates by color (the indicator is usually one color)
    from collections import defaultdict
    by_color: dict[int, list[ObjectRecord]] = defaultdict(list)
    for c in candidates:
        by_color[c.color].append(c)

    # Pick the color with the most candidates (brackets come in pairs)
    best_color = max(by_color, key=lambda c: len(by_color[c]))
    indicators = by_color[best_color]

    # Find which group the indicators are closest to
    indicator_centroid_r = sum(o.centroid[0] for o in indicators) / len(indicators)
    indicator_centroid_c = sum(o.centroid[1] for o in indicators) / len(indicators)

    best_group = None
    best_dist = float("inf")
    best_idx = -1

    # Sort groups by type so we can compute index within type
    type_instances: dict[str, list[ObjectGroup]] = {}
    for g in groups:
        type_instances.setdefault(g.group_type, []).append(g)
    # Sort each type's instances by column position (left to right)
    for instances in type_instances.values():
        instances.sort(key=lambda g: g.centroid[1])

    for g in groups:
        gr, gc = g.centroid
        dist = math.hypot(indicator_centroid_r - gr, indicator_centroid_c - gc)
        if dist < best_dist:
            best_dist = dist
            best_group = g
            # Find index within its type
            instances = type_instances.get(g.group_type, [])
            best_idx = instances.index(g) if g in instances else -1

    return FocusIndicator(
        indicator_objects=indicators,
        target_group=best_group,
        target_index=best_idx,
        color=best_color,
    )


# ---- 4. MISMATCH DETECTION -----------------------------------------------

def detect_mismatches(
    reference_groups: list[ObjectGroup],
    target_groups: list[ObjectGroup],
    frame: list[list[int]],
) -> list[Mismatch]:
    """Compare target groups against reference groups to find mismatches.

    For each target group, finds the reference group with the best-matching
    content mask and reports whether it's an exact match or a mismatch.

    If the groups contain paired boxes (two different anchor colors), the
    comparison is done on each box separately and the *relationship* between
    the pair members is compared.
    """
    mismatches: list[Mismatch] = []

    for tg in target_groups:
        if tg.content_mask is None:
            continue
        t_h, t_w = len(tg.content_mask), len(tg.content_mask[0]) if tg.content_mask else 0

        best_match: Mismatch | None = None
        best_dist = float("inf")

        for rg in reference_groups:
            if rg.content_mask is None:
                continue

            # Collect candidate reference masks.
            # Primary mask first; only try sub-masks when the target is
            # much smaller than the primary (e.g. 3x3 pink slot vs 5x5
            # primary), to avoid false matches across different sub-box
            # types of the same size.
            r_h0 = len(rg.content_mask)
            r_w0 = len(rg.content_mask[0]) if rg.content_mask else 0
            primary_too_big = (r_h0 - t_h > 1) or (r_w0 - t_w > 1)

            r_masks = [rg.content_mask]
            if primary_too_big:
                sub_masks = getattr(rg, "_sub_masks", None)
                if sub_masks:
                    for _anchor, sm in sub_masks:
                        sm_h = len(sm)
                        sm_w = len(sm[0]) if sm else 0
                        # Only add sub-masks that are closer in size to target
                        if abs(sm_h - t_h) <= 1 and abs(sm_w - t_w) <= 1:
                            r_masks.append(sm)

            for r_mask in r_masks:
                result = compare_shapes(r_mask, tg.content_mask)
                dist = result["best_distance"]

                if dist < best_dist:
                    best_dist = dist
                    best_match = Mismatch(
                        group_a=rg,
                        group_b=tg,
                        match=result["match"],
                        best_transform=result["best_transform"] or "",
                        distance=dist,
                        detail=(f"exact match via {result['transform']}"
                                if result["match"]
                                else f"closest: {result['best_transform']} "
                                     f"(dist={dist:.3f})"),
                    )

        if best_match is not None:
            mismatches.append(best_match)

    return mismatches


def detect_pairwise_mismatches(
    reference_groups: list[ObjectGroup],
    target_groups: list[ObjectGroup],
    frame: list[list[int]],
) -> list[Mismatch]:
    """Compare paired groups by their internal relationship, not absolute content.

    For groups that contain two sub-boxes of different colors (e.g., cyan + pink),
    extract the content mask from each sub-box separately, then compare:
    - The relationship (input_mask -> output_mask) in reference pairs
    - Against the relationship in target pairs

    This detects whether the *transformation* matches, not just the shapes.
    """
    def _split_pair(group: ObjectGroup, frame: list[list[int]]) -> tuple[list[list[int]] | None, list[list[int]] | None]:
        """Split a pair-group into (input_mask, output_mask) using stored sub_masks.

        Sub-masks are sorted by column (leftmost = input, rightmost = output).
        """
        sub_masks = getattr(group, "_sub_masks", None)
        if not sub_masks:
            return group.content_mask, None
        if len(sub_masks) < 2:
            return sub_masks[0][1] if sub_masks else None, None
        # Sort by column position of the anchor
        sorted_masks = sorted(sub_masks, key=lambda x: x[0].bbox["c_min"])
        return sorted_masks[0][1], sorted_masks[1][1]

    mismatches: list[Mismatch] = []

    # Extract (input, output) from each reference pair
    ref_pairs: list[tuple[list[list[int]] | None, list[list[int]] | None, ObjectGroup]] = []
    for rg in reference_groups:
        inp, out = _split_pair(rg, frame)
        ref_pairs.append((inp, out, rg))

    for tg in target_groups:
        t_inp, t_out = _split_pair(tg, frame)
        if t_inp is None:
            continue

        # If target has no output side, compare input only
        if t_out is None:
            # Compare target's single mask against reference inputs
            for r_inp, r_out, rg in ref_pairs:
                if r_inp is None:
                    continue
                result = compare_shapes(t_inp, r_inp)
                if result["match"]:
                    # Found which reference pair this target corresponds to
                    mismatches.append(Mismatch(
                        group_a=rg,
                        group_b=tg,
                        match=True,
                        best_transform=result["transform"] or "",
                        distance=0.0,
                        detail=f"input matches ref {rg.id} via {result['transform']}",
                    ))
                    break
            continue

        # Both target and references have (input, output) pairs.
        # For each reference pair, check if the same relationship holds.
        best: Mismatch | None = None
        best_dist = float("inf")

        for r_inp, r_out, rg in ref_pairs:
            if r_inp is None or r_out is None:
                continue
            # Check: does target_input match ref_input under some transform?
            input_cmp = compare_shapes(t_inp, r_inp)
            if not input_cmp["match"]:
                continue  # Different input shape — not the same reference pair

            # Inputs match → check if outputs also match
            output_cmp = compare_shapes(t_out, r_out)
            mm = Mismatch(
                group_a=rg,
                group_b=tg,
                match=output_cmp["match"],
                best_transform=output_cmp["best_transform"] or "",
                distance=output_cmp["best_distance"],
                detail=(f"pair matches ref {rg.id}"
                        if output_cmp["match"]
                        else f"input matches ref {rg.id} but output differs "
                             f"(dist={output_cmp['best_distance']:.3f})"),
            )
            if output_cmp["best_distance"] < best_dist:
                best_dist = output_cmp["best_distance"]
                best = mm

        if best is not None:
            mismatches.append(best)

    return mismatches

# The 8 symmetry operations of the dihedral group D4 (square symmetry).
# Each is a function: (row, col, height, width) -> (new_row, new_col).
# After applying the function, the caller must recompute the bounding size.
TRANSFORM_NAMES = [
    "identity",       # 0°
    "rot90",           # 90° clockwise
    "rot180",          # 180°
    "rot270",          # 270° clockwise (= 90° counter-clockwise)
    "flip_h",          # horizontal mirror (left-right)
    "flip_v",          # vertical mirror (top-bottom)
    "flip_main_diag",  # transpose (reflect over main diagonal)
    "flip_anti_diag",  # reflect over anti-diagonal
]


def extract_subgrid(
    frame: list[list[int]],
    bbox: dict,
    *,
    foreground_color: int | None = None,
) -> list[list[int]]:
    """Extract the rectangular sub-grid from *frame* bounded by *bbox*.

    Parameters
    ----------
    frame : 2-D grid of ints
    bbox : dict with keys r_min, r_max, c_min, c_max
    foreground_color : if given, return a binary mask (1 = foreground, 0 = bg)
                       where foreground = cells matching this color.

    Returns
    -------
    2-D list[list[int]] — the cropped region (or binary mask).
    """
    r_min, r_max = bbox["r_min"], bbox["r_max"]
    c_min, c_max = bbox["c_min"], bbox["c_max"]
    sub = []
    for r in range(r_min, r_max + 1):
        row = []
        for c in range(c_min, c_max + 1):
            val = frame[r][c]
            if foreground_color is not None:
                row.append(1 if val == foreground_color else 0)
            else:
                row.append(val)
        sub.append(row)
    return sub


def _to_pixel_set(mask: list[list[int]]) -> frozenset[tuple[int, int]]:
    """Convert a 2-D binary mask to a frozenset of (row, col) coordinates."""
    return frozenset(
        (r, c)
        for r, row in enumerate(mask)
        for c, val in enumerate(row)
        if val
    )


def _normalize_pixel_set(
    pixels: frozenset[tuple[int, int]],
) -> frozenset[tuple[int, int]]:
    """Translate a pixel set so its bounding box starts at (0, 0)."""
    if not pixels:
        return pixels
    min_r = min(r for r, _ in pixels)
    min_c = min(c for _, c in pixels)
    return frozenset((r - min_r, c - min_c) for r, c in pixels)


def _apply_transform(
    pixels: frozenset[tuple[int, int]],
    transform_index: int,
) -> frozenset[tuple[int, int]]:
    """Apply one of the 8 D4 symmetry transforms to a pixel set.

    The pixel set is assumed to be origin-normalized (min row/col = 0).
    The result is re-normalized to origin after transformation.
    """
    if not pixels:
        return pixels
    max_r = max(r for r, _ in pixels)
    max_c = max(c for _, c in pixels)

    if transform_index == 0:    # identity
        return pixels
    elif transform_index == 1:  # rot90 CW: (r, c) -> (c, max_r - r)
        out = frozenset((c, max_r - r) for r, c in pixels)
    elif transform_index == 2:  # rot180: (r, c) -> (max_r - r, max_c - c)
        out = frozenset((max_r - r, max_c - c) for r, c in pixels)
    elif transform_index == 3:  # rot270 CW: (r, c) -> (max_c - c, r)
        out = frozenset((max_c - c, r) for r, c in pixels)
    elif transform_index == 4:  # flip_h: (r, c) -> (r, max_c - c)
        out = frozenset((r, max_c - c) for r, c in pixels)
    elif transform_index == 5:  # flip_v: (r, c) -> (max_r - r, c)
        out = frozenset((max_r - r, c) for r, c in pixels)
    elif transform_index == 6:  # flip_main_diag: (r, c) -> (c, r)
        out = frozenset((c, r) for r, c in pixels)
    elif transform_index == 7:  # flip_anti_diag: (r, c) -> (max_c - c, max_r - r)
        out = frozenset((max_c - c, max_r - r) for r, c in pixels)
    else:
        raise ValueError(f"Invalid transform_index: {transform_index}")

    return _normalize_pixel_set(out)


def all_transforms(
    mask: list[list[int]],
) -> list[tuple[str, frozenset[tuple[int, int]]]]:
    """Generate all 8 D4 transforms of a binary mask.

    Returns a list of (transform_name, normalized_pixel_set) tuples.
    """
    base = _normalize_pixel_set(_to_pixel_set(mask))
    return [
        (TRANSFORM_NAMES[i], _apply_transform(base, i))
        for i in range(8)
    ]


def compare_shapes(
    mask_a: list[list[int]],
    mask_b: list[list[int]],
) -> dict:
    """Compare two binary masks under all D4 symmetry transforms.

    Returns
    -------
    dict with keys:
        "match": bool — True if any transform of A equals B
        "transform": str | None — name of the matching transform (A -> B)
        "transform_index": int | None
        "all_distances": list[tuple[str, float]] — Jaccard distance for each
            transform (0.0 = perfect match, 1.0 = no overlap)
    """
    pixels_b = _normalize_pixel_set(_to_pixel_set(mask_b))
    transforms_a = all_transforms(mask_a)

    best_dist = 1.0
    best_name = None
    best_idx = None
    distances = []

    for i, (name, pixels_a_t) in enumerate(transforms_a):
        if not pixels_a_t and not pixels_b:
            dist = 0.0
        elif not pixels_a_t or not pixels_b:
            dist = 1.0
        else:
            union = pixels_a_t | pixels_b
            inter = pixels_a_t & pixels_b
            dist = 1.0 - len(inter) / len(union) if union else 0.0
        distances.append((name, round(dist, 4)))
        if dist < best_dist:
            best_dist = dist
            best_name = name
            best_idx = i

    return {
        "match": best_dist == 0.0,
        "transform": best_name if best_dist == 0.0 else None,
        "transform_index": best_idx if best_dist == 0.0 else None,
        "best_transform": best_name,
        "best_distance": best_dist,
        "all_distances": distances,
    }


def find_similar_object_pairs(
    frame: list[list[int]],
    *,
    distance_threshold: float = 0.35,
    same_color_distance_threshold: float = 0.70,
    size_ratio_min: float = 1.2,
    min_size: int = 8,
) -> list[dict]:
    """Scan a frame for pairs of objects that look similar but differ in size or state.

    Two objects are "curious" when:
      • Their shapes are similar (best Jaccard distance < distance_threshold)
      • One is noticeably larger than the other (size ratio >= size_ratio_min)
      • Both are big enough to be meaningful (size >= min_size)

    This is the seed for the curiosity / hypothesis system: such pairs suggest
    a "small version" and "large reference version" of the same element, which
    in many games indicates a match-to-win condition.

    Returns a list of dicts, each with:
        color_small  : int    — color index of the smaller object
        color_large  : int    — color index of the larger object
        size_small   : int
        size_large   : int
        centroid_small : tuple[float, float]  — (row, col)
        centroid_large : tuple[float, float]
        bbox_small   : dict
        bbox_large   : dict
        distance     : float  — Jaccard distance (0.0 = identical shape)
        best_transform : str  — transform that maps small -> large
    """
    objs = detect_objects(frame)
    objs = [o for o in objs if o.size >= min_size]
    pairs: list[dict] = []
    for i, a in enumerate(objs):
        for b in objs[i + 1:]:
            small, large = (a, b) if a.size <= b.size else (b, a)
            if small.size == 0:
                continue
            ratio = large.size / small.size
            if ratio < size_ratio_min:
                continue
            # Extract binary masks and compare shapes
            mask_s = extract_subgrid(frame, small.bbox, foreground_color=small.color)
            mask_l = extract_subgrid(frame, large.bbox, foreground_color=large.color)
            result = compare_shapes(mask_s, mask_l)
            # Same-color pairs with sufficient size ratio are always interesting:
            # they may be the same element in different rotation/transformation states.
            # Use a looser threshold for same-color pairs.
            effective_threshold = (
                same_color_distance_threshold if a.color == b.color
                else distance_threshold
            )
            if result["best_distance"] <= effective_threshold:
                pairs.append({
                    "color_small":    small.color,
                    "color_large":    large.color,
                    "size_small":     small.size,
                    "size_large":     large.size,
                    "centroid_small": small.centroid,
                    "centroid_large": large.centroid,
                    "bbox_small":     small.bbox,
                    "bbox_large":     large.bbox,
                    "distance":       result["best_distance"],
                    "best_transform": result["best_transform"],
                })
    # Sort by ascending distance (best matches first)
    pairs.sort(key=lambda p: p["distance"])
    return pairs


def find_transformation(
    pairs: list[tuple[list[list[int]], list[list[int]]]],
) -> dict:
    """Given a list of (input_mask, output_mask) pairs, find a consistent
    D4 transform that maps every input to its output.

    Parameters
    ----------
    pairs : list of (mask_a, mask_b) where each mask is a 2-D binary grid.

    Returns
    -------
    dict with keys:
        "consistent": bool — True if one transform works for ALL pairs
        "transform": str | None — the consistent transform name
        "transform_index": int | None
        "per_pair": list[dict] — compare_shapes result for each pair
        "candidate_transforms": list[str] — transforms that matched at least
            one pair (useful when no single transform is consistent)
    """
    if not pairs:
        return {
            "consistent": False,
            "transform": None,
            "transform_index": None,
            "per_pair": [],
            "candidate_transforms": [],
        }

    per_pair = [compare_shapes(a, b) for a, b in pairs]

    # Find transforms that are exact matches for ALL pairs
    # Start with all 8 candidates, intersect down
    valid_indices: set[int] | None = None
    for pp in per_pair:
        exact_this = set()
        for i, (name, dist) in enumerate(pp["all_distances"]):
            if dist == 0.0:
                exact_this.add(i)
        if valid_indices is None:
            valid_indices = exact_this
        else:
            valid_indices &= exact_this

    if valid_indices is None:
        valid_indices = set()

    # Pick the simplest consistent transform (lowest index = identity first)
    consistent = len(valid_indices) > 0
    best_idx = min(valid_indices) if valid_indices else None
    best_name = TRANSFORM_NAMES[best_idx] if best_idx is not None else None

    # Candidate transforms: matched at least one pair
    candidate_set: set[int] = set()
    for pp in per_pair:
        for i, (name, dist) in enumerate(pp["all_distances"]):
            if dist == 0.0:
                candidate_set.add(i)

    return {
        "consistent": consistent,
        "transform": best_name,
        "transform_index": best_idx,
        "per_pair": per_pair,
        "candidate_transforms": [TRANSFORM_NAMES[i] for i in sorted(candidate_set)],
    }


# ---------------------------------------------------------------------------
# Slot strip detection (TR87-type slot puzzles)
# ---------------------------------------------------------------------------
#
# A "slot strip puzzle" has:
#   - A reference strip: a horizontal row of N bordered 7×7 slots (cyan or
#     yellow frame), each containing a 5×5 inner pattern (fixed).
#   - An editable strip: same layout with a different frame color (pink),
#     where the inner pattern cycles via ACTION1/ACTION2.
#   - A rule area: pairs of (LHS-frame, RHS-frame) slots connected by a
#     horizontal dark-grey bar — each pair is one rewrite rule.
#   - A cursor: a 5×2 white band above/below the active editable slot.
#
# This section provides detect_slot_strips(frame) which reads all of the
# above from raw pixel data and returns a structured summary including the
# computed action plan.
#
# Sprite patterns are from TR87's nxkictbbvzt sprite family (A1–A7, B1–B7,
# C1–C7).  All 4 rotations of each are pre-computed so rotated sprites are
# matched correctly.

# Canonical normalized 5×5 patterns (1=black pixel, 0=transparent/frame).
# Row-major order: index i*5+j = row i, col j.
_SLOT_CANONICAL: dict[tuple, str] = {
    # A series
    (0,0,1,0,0, 0,0,1,0,0, 0,1,1,1,0, 0,0,1,0,0, 1,1,1,1,1): "A1",
    (0,0,0,0,1, 0,0,1,0,1, 1,1,1,1,1, 1,0,1,0,0, 1,0,0,0,0): "A2",
    (1,1,1,1,1, 1,0,1,0,0, 1,0,0,0,0, 1,0,1,0,0, 1,1,1,1,1): "A3",
    (0,0,1,0,0, 1,1,1,1,1, 1,0,1,0,1, 1,0,1,0,1, 0,0,1,0,0): "A4",
    (1,0,0,0,1, 1,1,1,1,1, 0,0,1,0,0, 1,1,1,1,1, 1,0,0,0,1): "A5",
    (1,1,1,1,1, 0,0,1,0,0, 1,1,1,0,0, 1,0,0,0,0, 1,1,1,1,1): "A6",
    (1,0,0,0,1, 1,1,1,1,1, 1,0,0,0,1, 1,0,0,0,1, 1,1,0,1,1): "A7",
    # B series
    (1,0,0,0,0, 1,1,1,1,0, 1,0,0,1,0, 1,0,0,1,0, 1,1,1,1,1): "B1",
    (1,1,1,1,1, 1,0,0,0,1, 1,1,1,0,1, 1,0,1,0,1, 1,1,1,1,1): "B2",
    (0,0,1,1,1, 0,0,1,0,1, 1,1,1,1,1, 1,0,1,0,0, 1,1,1,0,0): "B3",
    (0,1,1,1,0, 0,1,0,1,0, 1,1,1,1,1, 1,0,0,0,1, 1,1,1,1,1): "B4",
    (1,1,1,1,1, 1,0,0,0,1, 1,0,0,0,1, 1,1,1,1,1, 0,0,1,0,0): "B5",
    (1,1,1,1,0, 1,0,0,1,1, 1,0,0,0,1, 1,1,0,0,1, 0,1,1,1,1): "B6",
    (0,0,1,0,0, 1,1,1,1,1, 1,0,1,0,1, 1,1,1,1,1, 0,0,1,0,0): "B7",
    # C series
    (1,0,1,0,1, 1,0,0,0,0, 1,0,1,0,1, 1,0,0,0,0, 1,1,1,1,1): "C1",
    (1,1,1,0,1, 1,0,0,0,0, 0,0,0,0,0, 0,0,0,0,1, 1,0,1,1,1): "C2",
    (1,0,1,1,1, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 1,1,1,0,1): "C3",
    (0,0,1,0,0, 1,0,1,0,1, 0,0,1,0,0, 1,0,1,0,1, 0,0,1,0,0): "C4",
    (1,1,0,1,1, 1,0,0,0,1, 1,0,1,0,1, 1,0,0,0,1, 1,1,0,1,1): "C5",
    (1,0,1,1,1, 1,0,1,0,1, 1,1,1,0,1, 0,0,0,0,0, 1,0,1,0,1): "C6",
    (1,0,0,0,1, 0,0,0,0,0, 1,0,0,0,1, 1,1,0,1,1, 0,1,0,1,0): "C7",
}

# Build full lookup including all 4 rotations of each canonical pattern.
def _build_slot_lookup() -> dict[tuple, str]:
    def _rot90(p: tuple) -> tuple:
        """Rotate 5×5 tuple 90° clockwise: new[r,c] = old[4-c, r]."""
        return tuple(p[(4 - c) * 5 + r] for r in range(5) for c in range(5))

    lookup: dict[tuple, str] = {}
    for base_pat, name in _SLOT_CANONICAL.items():
        p = base_pat
        for rot in range(4):
            lookup.setdefault(p, name)  # first match wins (prefer 0°)
            p = _rot90(p)
    return lookup


_SLOT_LOOKUP: dict[tuple, str] = _build_slot_lookup()

# Sprite number from name suffix, e.g. "A3" -> 3, "B7" -> 7
def _slot_num(name: str) -> int:
    return int(name[-1])

# Sprite series from name, e.g. "A3" -> "A", "B7" -> "B"
def _slot_series(name: str) -> str:
    return name[-2]


def _identify_slot(frame: list, r: int, c: int) -> str | None:
    """Identify the TR87 sprite in the inner 5×5 of a 7×7 slot at (r, c).

    Returns a sprite name like "A3", "B6", or None if unrecognized.
    """
    raw = tuple(
        1 if frame[r + 1 + i][c + 1 + j] == 5 else 0
        for i in range(5)
        for j in range(5)
    )
    return _SLOT_LOOKUP.get(raw)


def _slot_raw_pattern(frame: list, r: int, c: int) -> tuple:
    """Return the raw 25-bit inner pattern of a 7×7 slot at (r, c).

    Unlike _identify_slot, this preserves rotation — two slots with the same
    sprite at different rotations return different tuples.  Used as rule keys
    in alter_rules mode where different rotations map to different targets.
    """
    return tuple(
        1 if frame[r + 1 + i][c + 1 + j] == 5 else 0
        for i in range(5)
        for j in range(5)
    )


def _find_horizontal_strips(
    frame: list,
    frame_colors: set[int],
    min_slots: int = 2,
) -> list[dict]:
    """Find horizontal rows of identical-colored 7×7 framed slots.

    Returns list of strip dicts:
        frame_color, top_row, left_cols (sorted), n_slots, slot_names
    """
    rows = len(frame)
    cols = len(frame[0]) if rows else 0
    SLOT = 7

    def _slot_border_ok(r: int, c: int, fc: int) -> bool:
        """Check that a 7×7 window at (r, c) has an all-fc border."""
        if r + SLOT > rows or c + SLOT > cols:
            return False
        if not all(frame[r][c + j] == fc for j in range(SLOT)):
            return False
        if not all(frame[r + SLOT - 1][c + j] == fc for j in range(SLOT)):
            return False
        if not all(frame[r + i][c] == fc for i in range(SLOT)):
            return False
        if not all(frame[r + i][c + SLOT - 1] == fc for i in range(SLOT)):
            return False
        return True

    strips = []

    for r in range(rows - SLOT + 1):
        # Only process strip START positions to avoid false positives from
        # large continuous same-color regions.  A strip start is a column where
        # the pixel immediately to the left is NOT the current frame color.
        for c in range(cols - SLOT + 1):
            fc = int(frame[r][c])
            if fc not in frame_colors:
                continue
            # Must be a strip start: neighbor to the left is not fc
            if c > 0 and int(frame[r][c - 1]) == fc:
                continue
            if not _slot_border_ok(r, c, fc):
                continue
            # Extend to consecutive slots spaced exactly SLOT apart
            left_cols = [c]
            next_c = c + SLOT
            while next_c + SLOT <= cols + 1 and _slot_border_ok(r, next_c, fc):
                left_cols.append(next_c)
                next_c += SLOT
            if len(left_cols) >= min_slots:
                names = [_identify_slot(frame, r, lc) for lc in left_cols]
                raw_patterns = [_slot_raw_pattern(frame, r, lc) for lc in left_cols]
                strips.append({
                    "frame_color": fc,
                    "top_row": r,
                    "left_cols": left_cols,
                    "n_slots": len(left_cols),
                    "slot_names": names,
                    "slot_raw_patterns": raw_patterns,
                })

    return strips


def _find_rule_pairs(
    frame: list,
    background_row: int = 34,
) -> list[tuple[list[str | None], list[str | None], list[tuple], list[tuple]]]:
    """Detect rule pairs (including multi-sprite) from the rule area.

    The rule area is all rows with top_row < background_row.  At each row,
    slot groups are sorted left-to-right and paired as (group0, group1),
    (group2, group3), … where each group is a consecutive run of same-color
    slots.  This handles both 1:1 and multi-sprite (1:N, N:1, N:M) rules.

    Returns list of (lhs_names, rhs_names, lhs_raw, rhs_raw):
        lhs_names / rhs_names — sprite names (may contain None for unidentified)
        lhs_raw / rhs_raw    — raw 25-bit pattern tuples (rotation-preserving)

    The raw fields are used by detect_slot_strips in alter_rules mode where
    different rotations of the same sprite represent different rules.
    """
    FRAME_COLORS = {10, 7, 11}

    # Collect all slot groups in the rule area using the existing strip finder
    rule_groups = [
        g for g in _find_horizontal_strips(frame, FRAME_COLORS, min_slots=1)
        if g["top_row"] < background_row
    ]
    if not rule_groups:
        return []

    # Group by row
    from collections import defaultdict
    by_row: dict = defaultdict(list)
    for g in rule_groups:
        by_row[g["top_row"]].append(g)

    rules: list[tuple[list, list, list, list]] = []
    for _row in sorted(by_row):
        groups = sorted(by_row[_row], key=lambda g: g["left_cols"][0])
        # Pair consecutive groups: (0→1), (2→3), …
        for i in range(0, len(groups) - 1, 2):
            lhs_names = groups[i]["slot_names"]
            rhs_names = groups[i + 1]["slot_names"]
            lhs_raw = groups[i]["slot_raw_patterns"]
            rhs_raw = groups[i + 1]["slot_raw_patterns"]
            rules.append((list(lhs_names), list(rhs_names), list(lhs_raw), list(rhs_raw)))

    return rules


def _apply_rules_to_seq(
    seq: list[str | None],
    rule_pairs: list[tuple[list, list]],
    n_passes: int = 1,
) -> list[str | None] | None:
    """Apply sequence-rewriting rules to seq n_passes times.

    rule_pairs: list of (lhs_names, rhs_names) tuples.
    Returns the transformed sequence, or None if any element couldn't be
    matched.  Tries longest LHS patterns first at each position.
    """
    rule_map: dict[tuple, list] = {}
    for lhs, rhs in rule_pairs:
        key = tuple(lhs)
        if None not in key:
            rule_map[key] = list(rhs)

    result: list = list(seq)
    for _ in range(n_passes):
        out: list = []
        i = 0
        while i < len(result):
            matched = False
            # Try longest keys first
            for size in sorted({len(k) for k in rule_map}, reverse=True):
                if i + size > len(result):
                    continue
                key = tuple(result[i: i + size])
                if key in rule_map:
                    out.extend(rule_map[key])
                    i += size
                    matched = True
                    break
            if not matched:
                return None
        result = out
    return result


def _solve_alter_visual(
    rule_area_groups: list[dict],
    reference: list[str | None],
    target: list[str | None],
) -> list[int] | None:
    """Compute per-group rotation deltas for alter_rules mode.

    In alter_rules mode, each rule group in the rule area is a "dial" that can
    be rotated.  Rotating a group by k means all sprites in it advance k steps
    (sprite_num increments by k, mod 7).  The goal is to choose rotations such
    that applying the (rotated) rules to `reference` yields `target`.

    rule_area_groups — list of strip dicts from _find_horizontal_strips, sorted
        by (top_row, left_col).  Groups alternate LHS/RHS: index 0=LHS of rule
        0, index 1=RHS of rule 0, index 2=LHS of rule 1, ...

    Returns a list of integer deltas (one per group, same ordering) or None if
    no consistent rotation assignment exists.
    """
    if len(rule_area_groups) % 2 != 0:
        return None
    n_rules = len(rule_area_groups) // 2
    rotations: dict[int, int] = {}

    def search(ref_pos: int, tgt_pos: int) -> bool:
        if ref_pos == len(reference) and tgt_pos == len(target):
            return True
        if ref_pos > len(reference) or tgt_pos > len(target):
            return False
        for rule_i in range(n_rules):
            lhs_idx = rule_i * 2
            rhs_idx = rule_i * 2 + 1
            lhs_names = rule_area_groups[lhs_idx]["slot_names"]
            rhs_names = rule_area_groups[rhs_idx]["slot_names"]
            lhs_size = len(lhs_names)
            rhs_size = len(rhs_names)
            if ref_pos + lhs_size > len(reference):
                continue
            if tgt_pos + rhs_size > len(target):
                continue
            ref_sub = reference[ref_pos: ref_pos + lhs_size]
            tgt_sub = target[tgt_pos: tgt_pos + rhs_size]
            if None in ref_sub or None in tgt_sub:
                continue
            if None in lhs_names or None in rhs_names:
                continue

            # Find rotation k_lhs that maps lhs_names → ref_sub (sprite-number shift)
            k_lhs = (_slot_num(ref_sub[0]) - _slot_num(lhs_names[0]) + 7) % 7
            if not all(
                (_slot_num(lhs_names[j]) - 1 + k_lhs) % 7 + 1 == _slot_num(ref_sub[j])
                for j in range(lhs_size)
            ):
                continue
            if lhs_idx in rotations and rotations[lhs_idx] != k_lhs:
                continue

            # Find rotation k_rhs that maps rhs_names → tgt_sub
            k_rhs = (_slot_num(tgt_sub[0]) - _slot_num(rhs_names[0]) + 7) % 7
            if not all(
                (_slot_num(rhs_names[j]) - 1 + k_rhs) % 7 + 1 == _slot_num(tgt_sub[j])
                for j in range(rhs_size)
            ):
                continue
            if rhs_idx in rotations and rotations[rhs_idx] != k_rhs:
                continue

            old_lhs = rotations.get(lhs_idx)
            old_rhs = rotations.get(rhs_idx)
            rotations[lhs_idx] = k_lhs
            rotations[rhs_idx] = k_rhs

            if search(ref_pos + lhs_size, tgt_pos + rhs_size):
                return True

            if old_lhs is None:
                del rotations[lhs_idx]
            else:
                rotations[lhs_idx] = old_lhs
            if old_rhs is None:
                del rotations[rhs_idx]
            else:
                rotations[rhs_idx] = old_rhs

        return False

    if not search(0, 0):
        return None
    return [rotations.get(i, 0) for i in range(len(rule_area_groups))]


def _detect_cursor_in_rule_area(frame: list, rule_area_groups: list[dict]) -> int | None:
    """Return 0-indexed position of the cursor among rule area groups.

    The cursor is a 5-wide white (color 0) row just above or below the active
    group's first slot inner area.
    """
    CURSOR_COLOR = 0
    SLOT = 7
    rows = len(frame)
    for group_idx, group in enumerate(rule_area_groups):
        top_r = group["top_row"]
        left_c = group["left_cols"][0]
        inner_c = left_c + 1
        for check_r in [top_r - 4, top_r - 3, top_r - 2,
                        top_r + SLOT + 1, top_r + SLOT + 2, top_r + SLOT + 3]:
            if 0 <= check_r < rows:
                if all(
                    0 <= inner_c + j < len(frame[check_r]) and
                    frame[check_r][inner_c + j] == CURSOR_COLOR
                    for j in range(5)
                ):
                    return group_idx
    return None


def _solve_alter_double_visual(
    rule_area_groups: list[dict],
    reference: list[str | None],
    target: list[str | None],
) -> list[int] | None:
    """Compute per-group rotation deltas for alter_rules + double_translation mode.

    Rules interleave stage-1 (A-series LHS → B-series RHS) and stage-2
    (B-series LHS → C-series RHS) pairs in visual order.  Stage-1 rules are
    applied to reference to produce an intermediate B-series sequence; stage-2
    rules then map that intermediate to the target.

    Both stages require independent LHS and RHS rotations (delta per group).
    Stage-1 RHS deltas are searched (0-6) because they determine the unknown
    intermediate; stage-2 deltas are determined by matching intermediate→target.

    Returns a list of integer deltas (one per group) or None on failure.
    """
    if len(rule_area_groups) % 2 != 0:
        return None
    n_rules = len(rule_area_groups) // 2

    # Separate rules into stage1 (LHS matches reference series) and stage2
    ref_series = _slot_series(reference[0]) if reference and reference[0] else "A"
    stage1 = [
        i for i in range(n_rules)
        if (rule_area_groups[i * 2]["slot_names"]
            and rule_area_groups[i * 2]["slot_names"][0]
            and _slot_series(rule_area_groups[i * 2]["slot_names"][0]) == ref_series)
    ]
    stage2 = [i for i in range(n_rules) if i not in stage1]

    rotations: dict[int, int] = {}
    intermediate: list[str] = []

    def search2(inter_pos: int, tgt_pos: int) -> bool:
        if inter_pos == len(intermediate) and tgt_pos == len(target):
            return True
        if inter_pos > len(intermediate) or tgt_pos > len(target):
            return False
        for rule_i in stage2:
            lhs_idx = rule_i * 2
            rhs_idx = rule_i * 2 + 1
            lhs_names = rule_area_groups[lhs_idx]["slot_names"]
            rhs_names = rule_area_groups[rhs_idx]["slot_names"]
            lhs_size = len(lhs_names)
            rhs_size = len(rhs_names)
            if inter_pos + lhs_size > len(intermediate):
                continue
            if tgt_pos + rhs_size > len(target):
                continue
            inter_sub = intermediate[inter_pos: inter_pos + lhs_size]
            tgt_sub = target[tgt_pos: tgt_pos + rhs_size]
            if None in inter_sub or None in tgt_sub:
                continue
            if None in lhs_names or None in rhs_names:
                continue
            k_lhs = (_slot_num(inter_sub[0]) - _slot_num(lhs_names[0]) + 7) % 7
            if not all(
                (_slot_num(lhs_names[j]) - 1 + k_lhs) % 7 + 1 == _slot_num(inter_sub[j])
                for j in range(lhs_size)
            ):
                continue
            if lhs_idx in rotations and rotations[lhs_idx] != k_lhs:
                continue
            k_rhs = (_slot_num(tgt_sub[0]) - _slot_num(rhs_names[0]) + 7) % 7
            if not all(
                (_slot_num(rhs_names[j]) - 1 + k_rhs) % 7 + 1 == _slot_num(tgt_sub[j])
                for j in range(rhs_size)
            ):
                continue
            if rhs_idx in rotations and rotations[rhs_idx] != k_rhs:
                continue
            old_l, old_r = rotations.get(lhs_idx), rotations.get(rhs_idx)
            rotations[lhs_idx] = k_lhs
            rotations[rhs_idx] = k_rhs
            if search2(inter_pos + lhs_size, tgt_pos + rhs_size):
                return True
            if old_l is None:
                del rotations[lhs_idx]
            else:
                rotations[lhs_idx] = old_l
            if old_r is None:
                del rotations[rhs_idx]
            else:
                rotations[rhs_idx] = old_r
        return False

    def search1(ref_pos: int) -> bool:
        if ref_pos == len(reference):
            return search2(0, 0)
        for rule_i in stage1:
            lhs_idx = rule_i * 2
            rhs_idx = rule_i * 2 + 1
            lhs_names = rule_area_groups[lhs_idx]["slot_names"]
            rhs_names = rule_area_groups[rhs_idx]["slot_names"]
            lhs_size = len(lhs_names)
            rhs_size = len(rhs_names)
            if ref_pos + lhs_size > len(reference):
                continue
            ref_sub = reference[ref_pos: ref_pos + lhs_size]
            if None in ref_sub or None in lhs_names:
                continue
            k_lhs = (_slot_num(ref_sub[0]) - _slot_num(lhs_names[0]) + 7) % 7
            if not all(
                (_slot_num(lhs_names[j]) - 1 + k_lhs) % 7 + 1 == _slot_num(ref_sub[j])
                for j in range(lhs_size)
            ):
                continue
            if lhs_idx in rotations and rotations[lhs_idx] != k_lhs:
                continue
            # Try all RHS rotations — they determine the intermediate (unknown)
            for k_rhs in range(7):
                if rhs_idx in rotations and rotations[rhs_idx] != k_rhs:
                    continue
                rotated_rhs = [
                    rhs_names[j][:-1] + str((_slot_num(rhs_names[j]) - 1 + k_rhs) % 7 + 1)
                    for j in range(rhs_size)
                ]
                old_l, old_r = rotations.get(lhs_idx), rotations.get(rhs_idx)
                rotations[lhs_idx] = k_lhs
                rotations[rhs_idx] = k_rhs
                intermediate.extend(rotated_rhs)
                if search1(ref_pos + lhs_size):
                    return True
                del intermediate[-rhs_size:]
                if old_l is None:
                    del rotations[lhs_idx]
                else:
                    rotations[lhs_idx] = old_l
                if old_r is None:
                    del rotations[rhs_idx]
                else:
                    rotations[rhs_idx] = old_r
        return False

    if not search1(0):
        return None
    return [rotations.get(i, 0) for i in range(len(rule_area_groups))]


def _detect_cursor_slot(frame: list, strip: dict) -> int | None:
    """Return the 0-indexed slot position of the cursor in the given strip.

    The cursor is a 5-wide white (color 0) row appearing just above or below
    the strip's inner area (with a 4-row offset per the game engine).
    """
    CURSOR_COLOR = 0  # white
    rows = len(frame)
    top_r = strip["top_row"]
    SLOT = 7

    for slot_i, left_c in enumerate(strip["left_cols"]):
        inner_c = left_c + 1  # inner area starts 1 col inside the frame
        # Above: cursor top at top_r - 4 (offset from pjqbnqnbsq: y_offset=4)
        # Below: cursor top at top_r + SLOT + 2 (approx)
        for check_r in [top_r - 4, top_r - 3, top_r - 2, top_r + SLOT + 1,
                        top_r + SLOT + 2, top_r + SLOT + 3]:
            if 0 <= check_r < rows:
                if all(0 <= inner_c + j < len(frame[check_r]) and
                       frame[check_r][inner_c + j] == CURSOR_COLOR
                       for j in range(5)):
                    return slot_i
    return None


def detect_slot_strips(frame: list) -> dict | None:
    """Detect a TR87-type slot puzzle from raw pixel data.

    Scans the frame for horizontal slot strips and rule pairs, then computes
    the target state and action plan for the editable strip.

    The reference strip is the topmost (smallest top_row) strip; the editable
    strip is the one below it.  Frame color is NOT used to distinguish them —
    different levels use different color schemes (e.g., L2 has pink=reference,
    yellow=editable).

    Returns a dict with keys:
        reference     list[str|None]   variant names in reference strip
        editable      list[str|None]   variant names in editable strip
        cursor_slot   int|None         0-indexed active slot in editable strip
        rules         list[(lhs,rhs)]  detected rule pairs (names or None)
        rule_map      dict             {lhs_name: rhs_name} for single-sprite rules
        target        list[str|None]   target names for each editable slot
        deltas        list[int|None]   ACTION2 presses needed per slot
        action_plan   list[int]        action sequence (2=ACTION2, 4=ACTION4)
        mode_hint     str              "standard", "double", or "unknown"
    Returns None if no slot strip structure is found.
    """
    FRAME_COLORS = {10, 7, 11}  # cyan, pink, yellow
    # TR87 background (dark-grey) starts at row 34 in all levels.
    # Puzzle strips are within the background; rule area is above it.
    BACKGROUND_ROW = 34

    all_strips = _find_horizontal_strips(frame, FRAME_COLORS, min_slots=2)
    if not all_strips:
        return None

    # Puzzle strips: within the background region (top_row >= BACKGROUND_ROW)
    puzzle_strips = [s for s in all_strips if s["top_row"] >= BACKGROUND_ROW]
    if not puzzle_strips:
        return None

    # Need at least two puzzle strips at different rows
    unique_rows = sorted({s["top_row"] for s in puzzle_strips})
    if len(unique_rows) < 2:
        return None

    # Reference = strips at topmost puzzle row (mirrors game: min y = zvojhrjxxm)
    # Editable = strips at second puzzle row (ztgmtnnufb)
    ref_row = unique_rows[0]
    edit_row = unique_rows[1]

    ref_candidates = [s for s in puzzle_strips if s["top_row"] == ref_row]
    edit_candidates = [s for s in puzzle_strips if s["top_row"] == edit_row]

    ref_strip = max(ref_candidates, key=lambda s: s["n_slots"])
    edit_strip = max(edit_candidates, key=lambda s: s["n_slots"])

    reference = ref_strip["slot_names"]
    editable = edit_strip["slot_names"]
    cursor_slot = _detect_cursor_slot(frame, edit_strip)

    # Find rule pairs from the rule area (rows < BACKGROUND_ROW)
    # Each entry: (lhs_names, rhs_names, lhs_raw, rhs_raw)
    rules_full = _find_rule_pairs(frame, background_row=BACKGROUND_ROW)
    # Name-only view for display and standard matching
    rules = [(lhs, rhs) for lhs, rhs, _lr, _rr in rules_full]

    # Build rule_map for format_slot_strip_context (flat 1:1 view for display)
    rule_map: dict[str, str] = {}
    for lhs_names, rhs_names in rules:
        if len(lhs_names) == 1 and len(rhs_names) == 1:
            ln, rn = lhs_names[0], rhs_names[0]
            if ln and rn:
                rule_map[ln] = rn

    # Detect alter_rules mode: the SAME sprite name appears as LHS in two or more
    # DISTINCT rules (same sprite at different rotations → different rules).
    # This is different from a multi-sprite LHS (e.g. [C3, C3] → ...) where the
    # same name repeats within one rule's LHS — that's a normal multi-slot pattern.
    # We check by collecting one representative name per rule and looking for repeats.
    lhs_first_per_rule = [lhs[0] for lhs, _ in rules if lhs and lhs[0]]
    lhs_names_flat = [n for lhs, _ in rules for n in lhs if n]
    alter_mode = len(set(lhs_first_per_rule)) < len(lhs_first_per_rule)

    mode_hint = "standard"
    target: list[str | None]

    if alter_mode:
        # In alter_rules mode the interactive elements are the rule area slot
        # groups (rows < BACKGROUND_ROW), not the puzzle strips.  Each group is
        # a "dial" rotated via ACTION2.  The puzzle strips display the reference
        # (input A-series) and the goal output (target B-series, read-only).
        # We compute the required rotation delta for each group so that
        # rules(reference) == target (editable strip IS the target here).

        # Collect rule area groups in cursor-traversal order (top then left)
        rule_area_groups_sorted = sorted(
            [g for g in _find_horizontal_strips(frame, FRAME_COLORS, min_slots=1)
             if g["top_row"] < BACKGROUND_ROW],
            key=lambda g: (g["top_row"], g["left_cols"][0]),
        )

        # editable (second puzzle strip) is the goal/target in alter_rules mode
        target: list[str | None] = list(editable)
        alter_deltas = _solve_alter_visual(rule_area_groups_sorted, reference, editable)

        if alter_deltas is not None:
            alter_cursor = _detect_cursor_in_rule_area(frame, rule_area_groups_sorted)
            n_groups = len(rule_area_groups_sorted)
            start = alter_cursor if alter_cursor is not None else 0
            order = list(range(start, n_groups)) + list(range(0, start))
            action_plan: list[int] = []
            for idx, group_idx in enumerate(order):
                d = alter_deltas[group_idx]
                action_plan.extend([2] * d)
                if idx < n_groups - 1:
                    action_plan.append(4)
            deltas: list[int | None] = [None] * len(editable)  # per-puzzle-slot N/A
            mode_hint = "alter"
        else:
            target = [None] * len(editable)
            deltas = [None] * len(editable)
            action_plan = []
            mode_hint = "alter-failed"

        return {
            "reference": reference,
            "editable": editable,
            "cursor_slot": cursor_slot,
            "rules": rules,
            "rule_map": rule_map,
            "target": target,
            "deltas": deltas,
            "action_plan": action_plan,
            "mode_hint": mode_hint,
            "alter_group_deltas": alter_deltas,
        }
    else:
        # Compute target via multi-sprite sequence rewriting (single pass first)
        target_single = _apply_rules_to_seq(reference, rules, n_passes=1)

        if target_single is not None:
            target = target_single
            # Check if target series matches editable series; if not, try double
            edit_series_set = {_slot_series(n) for n in editable if n}
            tgt_series_set  = {_slot_series(t) for t in target if t}
            if tgt_series_set and edit_series_set and not tgt_series_set & edit_series_set:
                target_double = _apply_rules_to_seq(reference, rules, n_passes=2)
                if target_double is not None:
                    tgt2_series = {_slot_series(t) for t in target_double if t}
                    if tgt2_series & edit_series_set:
                        target = target_double
                        mode_hint = "double"
        else:
            # Single pass failed; try double pass directly
            target_double = _apply_rules_to_seq(reference, rules, n_passes=2)
            if target_double is not None:
                target = target_double
                mode_hint = "double"
            else:
                # Both standard passes failed.  If rules mix A-series and B-series
                # LHS sprites, this is alter_rules + double_translation mode (L6):
                # rule groups in the rule area are rotatable dials, two-stage.
                lhs_series_set = {_slot_series(n) for n in lhs_names_flat}
                if len(lhs_series_set) > 1:
                    rule_area_groups_sorted = sorted(
                        [g for g in _find_horizontal_strips(frame, FRAME_COLORS, min_slots=1)
                         if g["top_row"] < BACKGROUND_ROW],
                        key=lambda g: (g["top_row"], g["left_cols"][0]),
                    )
                    alter_double_deltas = _solve_alter_double_visual(
                        rule_area_groups_sorted, reference, editable
                    )
                    if alter_double_deltas is not None:
                        alter_cursor = _detect_cursor_in_rule_area(frame, rule_area_groups_sorted)
                        n_groups = len(rule_area_groups_sorted)
                        start = alter_cursor if alter_cursor is not None else 0
                        order = list(range(start, n_groups)) + list(range(0, start))
                        action_plan_ad: list[int] = []
                        for idx, group_idx in enumerate(order):
                            d = alter_double_deltas[group_idx]
                            action_plan_ad.extend([2] * d)
                            if idx < n_groups - 1:
                                action_plan_ad.append(4)
                        return {
                            "reference": reference,
                            "editable": editable,
                            "cursor_slot": cursor_slot,
                            "rules": rules,
                            "rule_map": rule_map,
                            "target": list(editable),  # editable IS the goal
                            "deltas": [None] * len(editable),
                            "action_plan": action_plan_ad,
                            "mode_hint": "alter-double",
                            "alter_group_deltas": alter_double_deltas,
                        }
                target = [None] * len(reference)

    # Compute deltas: target must have same length as editable
    if len(target) != len(editable):
        # Size mismatch — cannot align; produce partial deltas
        deltas_raw: list[int | None] = []
        for tgt, cur in zip(target, editable):
            if tgt is None or cur is None:
                deltas_raw.append(None)
            elif _slot_series(tgt) != _slot_series(cur):
                deltas_raw.append(None)
            else:
                deltas_raw.append((_slot_num(tgt) - _slot_num(cur)) % 7)
        deltas: list[int | None] = deltas_raw
    else:
        deltas = []
        for tgt, cur in zip(target, editable):
            if tgt is None or cur is None:
                deltas.append(None)
            elif _slot_series(tgt) != _slot_series(cur):
                deltas.append(None)
            else:
                deltas.append((_slot_num(tgt) - _slot_num(cur)) % 7)

    # Build action plan starting from the detected cursor position so the plan
    # works correctly after a transition-peek ACTION4 (cursor at slot 1).
    n_edit = len(editable)
    action_plan: list[int] = []
    if None not in deltas and len(deltas) == n_edit:
        # Determine starting slot from cursor detection
        start = cursor_slot if cursor_slot is not None else 0
        # Visit slots in order: start, start+1, ..., n_edit-1, 0, ..., start-1
        order = list(range(start, n_edit)) + list(range(0, start))
        for idx, slot_idx in enumerate(order):
            d = deltas[slot_idx]
            action_plan.extend([2] * d)  # type: ignore[arg-type]
            if idx < n_edit - 1:
                action_plan.append(4)  # cursor right

    return {
        "reference": reference,
        "editable": editable,
        "cursor_slot": cursor_slot,
        "rules": rules,
        "rule_map": rule_map,
        "target": target,
        "deltas": deltas,
        "action_plan": action_plan,
        "mode_hint": mode_hint,
    }


def format_slot_strip_context(slot_data: dict) -> str:
    """Format detect_slot_strips output as a structural context string.

    Returns a multi-line string ready for injection into the MEDIATOR prompt.
    """
    lines: list[str] = []
    lines.append("Slot strips:")

    ref = slot_data["reference"]
    edit = slot_data["editable"]
    cursor = slot_data["cursor_slot"]
    rules = slot_data["rules"]
    target = slot_data["target"]
    deltas = slot_data["deltas"]
    plan = slot_data["action_plan"]
    mode = slot_data["mode_hint"]

    def _fmt_names(names: list) -> str:
        return "  ".join(n if n else "?" for n in names)

    n_ref = len(ref)
    n_edit = len(edit)
    lines.append(f"  Reference ({n_ref} slots): {_fmt_names(ref)}")
    cursor_mark = ""
    edit_slots_str = []
    for i, n in enumerate(edit):
        mark = " [CURSOR]" if i == cursor else ""
        edit_slots_str.append(f"{n if n else '?'}{mark}")
    lines.append(f"  Editable ({n_edit} slots): {'  '.join(edit_slots_str)}")

    if rules:
        def _fmt_rule(lhs_names, rhs_names):
            l = "+".join(n if n else "?" for n in lhs_names)
            r = "+".join(n if n else "?" for n in rhs_names)
            return f"{l}->{r}"
        rule_strs = [_fmt_rule(lhs, rhs) for lhs, rhs in rules]
        lines.append(f"  Rules ({len(rules)}): {', '.join(rule_strs)}")

    if any(t is not None for t in target):
        tgt_str = _fmt_names(target)
        lines.append(f"  Target (rules applied{' twice' if mode == 'double' else ''}): {tgt_str}")

    if None not in deltas:
        delta_parts = [f"slot{i+1}:+{d}" for i, d in enumerate(deltas)]  # type: ignore
        lines.append(f"  Deltas (ACTION2 presses): {', '.join(delta_parts)}")
    else:
        solved = [(i, d) for i, d in enumerate(deltas) if d is not None]
        if solved:
            parts = [f"slot{i+1}:+{d}" for i, d in solved]
            lines.append(f"  Deltas (partial): {', '.join(parts)}")

    if plan:
        # Summarize the plan as groups of ACTION2s and ACTION4s
        plan_str = " ".join(
            f"2×{sum(1 for x in g if x == 2)}" if 2 in g else "4"
            for g in _split_actions(plan)
        )
        lines.append(f"  Action plan: {plan_str}")
        lines.append(
            f"  Raw actions: [{', '.join(str(a) for a in plan)}]"
        )
        lines.append(
            "  IMPORTANT: Execute this action plan directly using ACTION2 and ACTION4. "
            "Do NOT waste steps on exploration."
        )
    elif None not in deltas:
        lines.append("  Action plan: already solved (all deltas = 0)")
    else:
        lines.append(
            "  Action plan: incomplete — some rule mappings unresolved. "
            "Use ACTION2 to cycle the current slot and ACTION4 to move right."
        )

    return "\n".join(lines)


def _split_actions(plan: list[int]) -> list[list[int]]:
    """Split action plan into groups separated by ACTION4."""
    groups: list[list[int]] = []
    current: list[int] = []
    for a in plan:
        if a == 4:
            if current:
                groups.append(current)
            groups.append([4])
            current = []
        else:
            current.append(a)
    if current:
        groups.append(current)
    return groups
