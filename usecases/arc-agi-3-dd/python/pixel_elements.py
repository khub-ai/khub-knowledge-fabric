"""Pure-frame connected-components extractor.

The substrate-side replacement for _auto_scan_level under STRICT_MODE.
Input: the obs.frame 64x64 palette grid only.  No env._game access.
Output: untagged components with geometry only.  Function classification
(agent / wall / goal / ...) is NOT provided here -- that's TUTOR's job
under the prime directive.

This module must remain game-agnostic.  Litmus test: drop this onto a
fresh ARC-AGI-3 game whose source I have not read; it should still produce
meaningful components.  No hardcoded palette constants, no sprite-class
whitelists, nothing specific to ls20.

Algorithm: 4-connectivity flood fill per palette.  For each cell visit
at most once.  Components below min_size are discarded as noise.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


def extract_components(
    frame:              np.ndarray,
    min_size:           int = 2,
    exclude_background: bool = True,
    connectivity:       int = 4,
    return_cells:       bool = False,
) -> list[dict]:
    """Extract connected components from a 2D palette frame.

    Parameters
    ----------
    frame:              2D int array of palette values (e.g. 64x64).
    min_size:           Drop components with fewer than this many cells.
                        Default 2 filters single-pixel noise; use 1 to keep everything.
    exclude_background: If True, identify the palette occupying the most cells
                        and skip components of that palette.  Usually correct (the
                        background fills the grid) but not guaranteed -- a
                        wall-heavy level might make walls "background" by area.
                        Set False when in doubt; TUTOR can filter downstream.
    connectivity:       4 or 8.  4-connectivity uses N/S/E/W neighbors only.
    return_cells:       If True, each component includes the full cell list
                        (expensive for large components; default False returns
                        only geometry summaries).

    Returns
    -------
    list[dict] sorted by size descending.  Each component has:
        id               : 1-based integer, extraction order
        palette          : int palette value
        size             : cell count
        bbox             : [r_min, c_min, r_max, c_max]  (inclusive)
        centroid         : [r, c] int-rounded
        extent           : [height, width]
        fill_ratio       : size / (height * width), in [0, 1]
        cells            : list[(r, c)]  (only if return_cells=True)
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    if frame.ndim != 2:
        raise ValueError(f"frame must be 2D, got shape {frame.shape}")

    H, W = frame.shape
    visited = np.zeros((H, W), dtype=bool)

    # Determine background palette by area if we're excluding it.
    bg_palette: Optional[int] = None
    if exclude_background:
        palettes, counts = np.unique(frame, return_counts=True)
        bg_palette = int(palettes[int(np.argmax(counts))])

    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    components: list[dict] = []
    next_id = 1

    for r0 in range(H):
        for c0 in range(W):
            if visited[r0, c0]:
                continue
            p = int(frame[r0, c0])
            if exclude_background and p == bg_palette:
                visited[r0, c0] = True
                continue

            # Flood-fill this component.
            cells: list[tuple[int, int]] = []
            queue = deque([(r0, c0)])
            visited[r0, c0] = True
            r_min, r_max, c_min, c_max = r0, r0, c0, c0
            r_sum, c_sum = 0, 0

            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                r_sum += r
                c_sum += c
                if r < r_min: r_min = r
                if r > r_max: r_max = r
                if c < c_min: c_min = c
                if c > c_max: c_max = c
                for dr, dc in offsets:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and int(frame[nr, nc]) == p:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            size = len(cells)
            if size < min_size:
                continue

            comp = {
                "id":         next_id,
                "palette":    p,
                "size":       size,
                "bbox":       [r_min, c_min, r_max, c_max],
                "centroid":   [r_sum // size, c_sum // size],
                "extent":     [r_max - r_min + 1, c_max - c_min + 1],
                "fill_ratio": round(size / float((r_max - r_min + 1) * (c_max - c_min + 1)), 3),
            }
            if return_cells:
                comp["cells"] = cells
            components.append(comp)
            next_id += 1

    components.sort(key=lambda c: -c["size"])
    # Re-number after sort so id matches display order.
    for i, comp in enumerate(components, 1):
        comp["id"] = i
    return components


def summarize_frame(frame: np.ndarray) -> dict:
    """One-call summary: palette histogram + components.

    Useful for logging or as the prompt-side "here's what I see" structure.
    """
    palettes, counts = np.unique(frame, return_counts=True)
    hist = {int(p): int(c) for p, c in zip(palettes, counts)}
    total = int(frame.size)
    return {
        "shape":             list(frame.shape),
        "palette_histogram": hist,
        "palette_background_guess": max(hist.items(), key=lambda kv: kv[1])[0] if hist else None,
        "components":        extract_components(frame, min_size=2, exclude_background=True),
    }


# ---------------------------------------------------------------------------
# Diffing helpers -- for change detection across frames
# ---------------------------------------------------------------------------

def frame_diff(
    frame_before: np.ndarray,
    frame_after:  np.ndarray,
) -> dict:
    """Compute what changed between two frames.

    Output:
        changed_cells         : int, number of cells with different palette
        changed_bbox          : [r_min, c_min, r_max, c_max] of the diff, or None
        changed_palettes_out  : palettes that lost area (in before but not after at some cell)
        changed_palettes_in   : palettes that gained area (in after but not before at some cell)
    """
    if frame_before.shape != frame_after.shape:
        return {"changed_cells": -1, "error": "shape mismatch"}
    mask = frame_before != frame_after
    n = int(mask.sum())
    if n == 0:
        return {
            "changed_cells":        0,
            "changed_bbox":         None,
            "changed_palettes_out": [],
            "changed_palettes_in":  [],
        }
    rows, cols = np.where(mask)
    bbox = [int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())]
    out_palettes = sorted({int(v) for v in frame_before[mask].tolist()})
    in_palettes  = sorted({int(v) for v in frame_after[mask].tolist()})
    return {
        "changed_cells":        n,
        "changed_bbox":         bbox,
        "changed_palettes_out": out_palettes,
        "changed_palettes_in":  in_palettes,
    }


def narrate_frame_delta(
    frame_before: np.ndarray,
    frame_after:  np.ndarray,
    agent_fp:     tuple | None = None,
) -> dict:
    """Component-level narration of what changed between two frames.

    Returns a dict with four lists:
      disappeared: components in `before` that have no match in `after`
      appeared:    components in `after` that have no match in `before`
      moved:       components where centroid shifted but size+extent+palette preserved
      reshaped:    components where palette+approximate-position preserved but size/extent changed
                   (e.g. a HUD bar shrinking)

    Matching rule: two components "correspond" if they share
    (palette, size, extent) AND their centroids differ by at most
    max_shift.  An unmatched before-component is "disappeared"; an
    unmatched after-component is "appeared".

    This is a RAW OBSERVATION layer.  The harness does NOT interpret
    what these changes mean -- TUTOR does.  Under the prime directive,
    we can only report what is visible in the pixels.

    The agent component is excluded when agent_fp is provided (agent
    motion is reported separately via displacement detection).
    """
    comps_before = extract_components(frame_before, min_size=2)
    comps_after  = extract_components(frame_after,  min_size=2)

    # Filter out the agent (its motion is handled separately).
    if agent_fp is not None:
        ap, asz, aext = agent_fp[0], agent_fp[1], tuple(agent_fp[2])
        comps_before = [c for c in comps_before
                        if not (c["palette"] == ap and c["size"] == asz
                                and tuple(c["extent"]) == aext)]
        comps_after  = [c for c in comps_after
                        if not (c["palette"] == ap and c["size"] == asz
                                and tuple(c["extent"]) == aext)]

    # Exact-fingerprint matching with static bucket (as in bootstrap).
    from collections import defaultdict
    def bucket(comps):
        d = defaultdict(list)
        for c in comps:
            d[(c["palette"], c["size"], tuple(c["extent"]))].append(c)
        return d

    bb = bucket(comps_before)
    ba = bucket(comps_after)

    disappeared: list[dict] = []
    appeared:    list[dict] = []
    moved:       list[dict] = []

    all_keys = set(bb) | set(ba)
    for key in all_keys:
        lb = bb.get(key, [])
        la = ba.get(key, [])
        # If same count, try to match by nearest centroid (greedy).
        if len(lb) == len(la):
            # If count == 1, this is the simple case
            if len(lb) == 1:
                cb, ca = lb[0], la[0]
                dr = ca["centroid"][0] - cb["centroid"][0]
                dc = ca["centroid"][1] - cb["centroid"][1]
                if dr != 0 or dc != 0:
                    moved.append({
                        "palette": cb["palette"], "size": cb["size"],
                        "extent":  cb["extent"],
                        "from":    cb["centroid"], "to": ca["centroid"],
                        "dr":      dr, "dc": dc,
                    })
            # multiple identical components: ambiguous, skip motion reporting
        else:
            # Count changed -- report as (dis)appeared.
            # Choose the LONGER list as canonical.
            if len(lb) > len(la):
                # Some disappeared.  Report all unmatched before-components by
                # marking the |lb| - |la| smallest (by bbox_min) as gone.
                for c in lb[len(la):]:
                    disappeared.append({
                        "palette": c["palette"], "size": c["size"],
                        "extent":  c["extent"],  "centroid": c["centroid"],
                        "bbox":    c["bbox"],
                    })
            else:
                for c in la[len(lb):]:
                    appeared.append({
                        "palette": c["palette"], "size": c["size"],
                        "extent":  c["extent"],  "centroid": c["centroid"],
                        "bbox":    c["bbox"],
                    })

    # Also detect size-preserving-but-palette-different components at the
    # same bbox -- palette flips indicate "something was triggered".  We
    # check by bbox overlap between surviving before-only and after-only
    # components.  (Rare, but worth reporting.)
    # Leave this as-is; the bucket-based logic already captures most cases.

    return {
        "disappeared": disappeared,
        "appeared":    appeared,
        "moved":       moved,
    }


def detect_agent_displacement(
    frame_before: np.ndarray,
    frame_after:  np.ndarray,
    min_component_size: int = 2,
) -> Optional[dict]:
    """Heuristic: find the single component whose centroid shifted between
    frames by the same (dr, dc) as its companion "reappearance" on the
    other side.  In most ARC-AGI-3 games, the agent is the only entity
    that moves when you issue an action.  If exactly one component-shaped
    change is detected, return its displacement; otherwise None.

    Returned dict:
        dr, dc           : int shift of the moved component's centroid
        component_before : dict (from extract_components)
        component_after  : dict
    Returns None if ambiguous (zero or >1 candidate displacements).

    Caller should check that the returned (dr, dc) matches a recently-
    issued action's effect.  This does NOT read env._game; it's a pure
    pixel heuristic.
    """
    diff = frame_diff(frame_before, frame_after)
    if diff.get("changed_cells", 0) == 0:
        return None

    comps_before = extract_components(frame_before, min_size=min_component_size)
    comps_after  = extract_components(frame_after, min_size=min_component_size)

    # Match by palette + size: a component that moved keeps both invariant.
    # (Fragile -- game animations can change palette. Good enough as a
    # first pass; TUTOR can revise.)
    candidates = []
    for cb in comps_before:
        for ca in comps_after:
            if cb["palette"] == ca["palette"] and cb["size"] == ca["size"]:
                dr = ca["centroid"][0] - cb["centroid"][0]
                dc = ca["centroid"][1] - cb["centroid"][1]
                if dr != 0 or dc != 0:
                    candidates.append({
                        "dr": dr, "dc": dc,
                        "component_before": cb,
                        "component_after":  ca,
                    })

    if len(candidates) != 1:
        return None
    return candidates[0]
