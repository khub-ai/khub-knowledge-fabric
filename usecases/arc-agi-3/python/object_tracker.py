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
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Color names (same as ARC-AGI palette)
# ---------------------------------------------------------------------------

_COLOR_NAMES = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "grey",
    6: "magenta",
    7: "orange",
    8: "azure",
    9: "white",
    # Colors beyond 9 are game-specific extensions — named generically so the
    # system discovers their roles through exploration rather than hardcoding.
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
    background_threshold: float = 0.25,
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
        Default 0.25 = 25%.

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

    Looks for the most frequently-moved non-background object and describes
    the typical direction and magnitude.
    """
    if not obj_obs:
        return "(no object observations)"

    # Count moves per (color, direction)
    move_counts: dict[tuple[int, str], list[float]] = {}
    no_change_count = 0

    for obs in obj_obs:
        moved = obs.get("moved", [])
        if not moved and not obs.get("appeared") and not obs.get("disappeared"):
            no_change_count += 1
            continue
        for m in moved:
            if m.get("is_background"):
                continue
            key = (m["color"], m["direction"])
            move_counts.setdefault(key, []).append(m.get("magnitude", 0))

    if not move_counts:
        if no_change_count == len(obj_obs):
            return "no object movement observed (all calls had no effect)"
        return "changes detected but no clear object movement pattern"

    # Find the most frequent move pattern
    best_key = max(move_counts, key=lambda k: len(move_counts[k]))
    color, direction = best_key
    magnitudes = move_counts[best_key]
    avg_mag = sum(magnitudes) / len(magnitudes)
    freq = len(magnitudes)
    total = len(obj_obs)

    return (
        f"{color_name(color)} object moves {direction} "
        f"~{avg_mag:.1f} cells/call  "
        f"(observed {freq}/{total} calls)"
    )


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
            concept = bindings.get(o.color)
        name = f"{concept}({color_name(o.color)})" if concept else color_name(o.color)
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
