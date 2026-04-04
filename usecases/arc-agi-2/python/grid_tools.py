"""
grid_tools.py — Numpy-based grid operations for ARC-AGI puzzles.

All functions accept/return plain Python lists (List[List[int]]) so they
interoperate cleanly with JSON. Numpy is used only internally.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Grid = List[List[int]]


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def to_np(grid: Grid) -> np.ndarray:
    return np.array(grid, dtype=np.int32)

def to_list(arr: np.ndarray) -> Grid:
    return arr.tolist()


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------

def shape(grid: Grid) -> Tuple[int, int]:
    arr = to_np(grid)
    return arr.shape  # (rows, cols)

def unique_colors(grid: Grid) -> List[int]:
    return sorted(set(np.unique(to_np(grid)).tolist()))

def color_count(grid: Grid, color: int) -> int:
    return int(np.sum(to_np(grid) == color))

def bounding_box(grid: Grid, color: int) -> Optional[Tuple[int, int, int, int]]:
    """Return (row_min, col_min, row_max, col_max) for all cells of `color`."""
    arr = to_np(grid)
    rows, cols = np.where(arr == color)
    if len(rows) == 0:
        return None
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------

def rotate_90(grid: Grid, k: int = 1) -> Grid:
    """Rotate counter-clockwise by 90*k degrees."""
    return to_list(np.rot90(to_np(grid), k))

def flip_horizontal(grid: Grid) -> Grid:
    return to_list(np.fliplr(to_np(grid)))

def flip_vertical(grid: Grid) -> Grid:
    return to_list(np.flipud(to_np(grid)))

def transpose(grid: Grid) -> Grid:
    return to_list(to_np(grid).T)

def crop(grid: Grid, row_start: int, col_start: int, row_end: int, col_end: int) -> Grid:
    """Crop to [row_start:row_end, col_start:col_end] (exclusive end)."""
    return to_list(to_np(grid)[row_start:row_end, col_start:col_end])

def pad(grid: Grid, top: int = 0, bottom: int = 0, left: int = 0, right: int = 0, fill: int = 0) -> Grid:
    return to_list(np.pad(to_np(grid), ((top, bottom), (left, right)), constant_values=fill))

def replace_color(grid: Grid, from_color: int, to_color: int) -> Grid:
    arr = to_np(grid).copy()
    arr[arr == from_color] = to_color
    return to_list(arr)

def apply_gravity(grid: Grid, direction: str = "down", background: int = 0) -> Grid:
    """
    Slide non-background cells in `direction` ('down', 'up', 'left', 'right').
    Works column-wise for down/up, row-wise for left/right.
    """
    arr = to_np(grid).copy()
    rows, cols = arr.shape

    if direction in ("down", "up"):
        for c in range(cols):
            col = arr[:, c]
            non_bg = col[col != background]
            n = len(non_bg)
            if direction == "down":
                arr[:, c] = np.array([background] * (rows - n) + list(non_bg))
            else:
                arr[:, c] = np.array(list(non_bg) + [background] * (rows - n))
    else:
        for r in range(rows):
            row = arr[r, :]
            non_bg = row[row != background]
            n = len(non_bg)
            if direction == "right":
                arr[r, :] = np.array([background] * (cols - n) + list(non_bg))
            else:
                arr[r, :] = np.array(list(non_bg) + [background] * (cols - n))

    return to_list(arr)

def flood_fill(grid: Grid, start_row: int, start_col: int, fill_color: int) -> Grid:
    """BFS flood fill from (start_row, start_col)."""
    arr = to_np(grid).copy()
    target = int(arr[start_row, start_col])
    if target == fill_color:
        return to_list(arr)
    rows, cols = arr.shape
    stack = [(start_row, start_col)]
    while stack:
        r, c = stack.pop()
        if r < 0 or r >= rows or c < 0 or c >= cols:
            continue
        if arr[r, c] != target:
            continue
        arr[r, c] = fill_color
        stack.extend([(r+1,c),(r-1,c),(r,c+1),(r,c-1)])
    return to_list(arr)

def _find_components(grid: Grid, background: int = 0) -> list[dict]:
    """
    Return all 4-connected components of non-background cells.
    Each component dict has: cells (list of (r,c)), color, top, bottom, left, right.
    """
    from collections import deque
    arr = to_np(grid)
    rows, cols = arr.shape
    visited = np.zeros((rows, cols), dtype=bool)
    components: list[dict] = []

    for r in range(rows):
        for c in range(cols):
            v = int(arr[r, c])
            if v != background and not visited[r, c]:
                cells: list[tuple[int, int]] = []
                q: deque[tuple[int, int]] = deque([(r, c)])
                visited[r, c] = True
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for nr, nc in ((cr+1,cc),(cr-1,cc),(cr,cc+1),(cr,cc-1)):
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr,nc] and int(arr[nr,nc]) == v:
                            visited[nr,nc] = True
                            q.append((nr, nc))
                rs = [x[0] for x in cells]
                cs = [x[1] for x in cells]
                components.append({
                    "cells": cells, "color": v,
                    "top": min(rs), "bottom": max(rs),
                    "left": min(cs), "right": max(cs),
                })
    return components


def _is_closed_hollow_rect(comp: dict) -> bool:
    """
    True if the component is a closed hollow rectangle:
    - Bounding box at least 3×3
    - All perimeter cells of bounding box filled, all interior cells empty
    """
    t, b, l, r = comp["top"], comp["bottom"], comp["left"], comp["right"]
    h, w = b - t + 1, r - l + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * (h + w) - 4  # perimeter cell count
    if len(comp["cells"]) != expected:
        return False
    cells_set = set(comp["cells"])
    for row in range(t, b + 1):
        for col in range(l, r + 1):
            on_perim = (row == t or row == b or col == l or col == r)
            in_set = (row, col) in cells_set
            if on_perim != in_set:
                return False
    return True


def gravity_by_type(
    grid: Grid,
    background: int = 0,
    **kwargs,
) -> Grid:
    """
    Type-classified gravity: closed hollow rectangles float UP, open/cross shapes sink DOWN.

    Classification:
      - Closed hollow rectangle: bounding box ≥ 3×3, all perimeter cells filled,
        all interior cells background.
      - Open/cross shape: everything else (plus, T, L, incomplete frame, solid, etc.)

    Movement rules:
      - Each object slides as a rigid unit (all cells shift by the same row delta).
      - Object color is preserved.
      - Closed rects stack from row 0 downward (sorted by original top row).
      - Open shapes stack from last row upward (sorted by original bottom row).
      - Different-type objects pass through each other freely.
      - Same-type objects maintain original relative vertical order (no overlap).
    """
    arr = to_np(grid)
    rows, cols = arr.shape

    components = _find_components(grid, background=background)
    for comp in components:
        comp["type"] = "closed_rect" if _is_closed_hollow_rect(comp) else "open_shape"

    result = np.zeros_like(arr)

    # --- Stack closed rects from top ---
    closed = sorted(
        [c for c in components if c["type"] == "closed_rect"],
        key=lambda x: x["top"],
    )
    col_top = [0] * cols  # next free row from top, per column

    for comp in closed:
        # new_top = highest row index such that no cell of the placed object
        # overlaps with already-placed cells (respects col_top per column).
        # For cell (cr, cc): placed row = new_top + (cr - comp["top"])
        # Constraint: new_top + (cr - comp["top"]) >= col_top[cc]
        # => new_top >= col_top[cc] - (cr - comp["top"])
        new_top = max(
            (col_top[cc] - (cr - comp["top"]) for (cr, cc) in comp["cells"]),
            default=0,
        )
        new_top = max(new_top, 0)
        delta = new_top - comp["top"]
        for (cr, cc) in comp["cells"]:
            result[cr + delta, cc] = comp["color"]
        for (cr, cc) in comp["cells"]:
            col_top[cc] = max(col_top[cc], cr + delta + 1)

    # --- Stack open shapes from bottom ---
    open_shapes = sorted(
        [c for c in components if c["type"] == "open_shape"],
        key=lambda x: -x["bottom"],
    )
    col_bot = [rows - 1] * cols  # next free row from bottom, per column

    for comp in open_shapes:
        # new_bottom = lowest row index such that no cell overlaps already-placed.
        # For cell (cr, cc): placed row = new_bottom - (comp["bottom"] - cr)
        # Constraint: new_bottom - (comp["bottom"] - cr) <= col_bot[cc]
        # => new_bottom <= col_bot[cc] + (comp["bottom"] - cr)
        new_bottom = min(
            (col_bot[cc] + (comp["bottom"] - cr) for (cr, cc) in comp["cells"]),
            default=rows - 1,
        )
        new_bottom = min(new_bottom, rows - 1)
        delta = new_bottom - comp["bottom"]
        for (cr, cc) in comp["cells"]:
            result[cr + delta, cc] = comp["color"]
        for (cr, cc) in comp["cells"]:
            col_bot[cc] = min(col_bot[cc], cr + delta - 1)

    return to_list(result)


def count_connected_components(grid: Grid, color: int) -> int:
    """Count 4-connected components of the given color."""
    arr = to_np(grid).copy()
    rows, cols = arr.shape
    visited = np.zeros_like(arr, dtype=bool)
    count = 0
    for r in range(rows):
        for c in range(cols):
            if arr[r, c] == color and not visited[r, c]:
                count += 1
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                        continue
                    if visited[cr, cc] or arr[cr, cc] != color:
                        continue
                    visited[cr, cc] = True
                    stack.extend([(cr+1,cc),(cr-1,cc),(cr,cc+1),(cr,cc-1)])
    return count


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def grids_equal(a: Grid, b: Grid) -> bool:
    if not a or not b:
        return False
    try:
        return np.array_equal(to_np(a), to_np(b))
    except Exception:
        return False

def diff_cells(a: Grid, b: Grid) -> List[Tuple[int, int, int, int]]:
    """Return list of (row, col, val_a, val_b) for cells that differ."""
    arr_a, arr_b = to_np(a), to_np(b)
    if arr_a.shape != arr_b.shape:
        return [(-1, -1, -1, -1)]
    rows, cols = np.where(arr_a != arr_b)
    result = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        result.append((r, c, int(arr_a[r, c]), int(arr_b[r, c])))
    return result

def cell_accuracy(predicted: Grid, expected: Grid) -> float:
    """Fraction of cells that are correct (0.0–1.0). Returns 0.0 on shape mismatch."""
    try:
        arr_p, arr_e = to_np(predicted), to_np(expected)
        if arr_p.shape != arr_e.shape:
            return 0.0
        total = arr_p.size
        correct = int(np.sum(arr_p == arr_e))
        return correct / total if total > 0 else 1.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

# ARC-AGI color palette (index → name)
COLOR_NAMES = {
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "gray", 6: "magenta", 7: "orange", 8: "azure", 9: "maroon"
}


def barrier_beam(grid: Grid, background: int = 0, **kwargs) -> Grid:
    """
    8-barrier (full row or column) divides the grid into a 4-side and a 2-side.
    For each row/col that contains 4-cells:
      1. Original 4 positions → 3 (shadow marker)
      2. Fill between 4s and barrier with 4 (shoot toward barrier)
      3. Pack the 2-cells for that row/col flush against the FAR EDGE,
         fill between barrier and packed 2s with 8
    Rows/cols with no 4s are unchanged.
    Works for both vertical and horizontal barriers, and for 4s on either side.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    result = [row[:] for row in grid]

    full_col_8 = [c for c in range(cols) if all(grid[r][c] == 8 for r in range(rows))]
    full_row_8 = [r for r in range(rows) if all(grid[r][c] == 8 for c in range(cols))]

    if full_col_8:
        barrier = full_col_8[0]
        four_left = any(grid[r][c] == 4 for r in range(rows) for c in range(barrier))
        for r in range(rows):
            if four_left:
                fours = [c for c in range(barrier) if grid[r][c] == 4]
                twos  = [c for c in range(barrier + 1, cols) if grid[r][c] == 2]
                if not fours:
                    continue
                for c in fours:
                    result[r][c] = 3
                for c in range(max(fours) + 1, barrier):
                    result[r][c] = 4
                pack = cols - len(twos)
                for c in range(barrier + 1, pack):
                    result[r][c] = 8
                for i in range(len(twos)):
                    result[r][pack + i] = 2
            else:
                fours = [c for c in range(barrier + 1, cols) if grid[r][c] == 4]
                twos  = [c for c in range(barrier) if grid[r][c] == 2]
                if not fours:
                    continue
                for c in fours:
                    result[r][c] = 3
                for c in range(barrier + 1, min(fours)):
                    result[r][c] = 4
                pack = len(twos)
                for c in range(pack, barrier):
                    result[r][c] = 8
                for i in range(len(twos)):
                    result[r][i] = 2

    elif full_row_8:
        barrier = full_row_8[0]
        four_above = any(grid[r][c] == 4 for r in range(barrier) for c in range(cols))
        for c in range(cols):
            if four_above:
                fours = [r for r in range(barrier) if grid[r][c] == 4]
                twos  = [r for r in range(barrier + 1, rows) if grid[r][c] == 2]
                if not fours:
                    continue
                for r in fours:
                    result[r][c] = 3
                for r in range(max(fours) + 1, barrier):
                    result[r][c] = 4
                pack = rows - len(twos)
                for r in range(barrier + 1, pack):
                    result[r][c] = 8
                for i in range(len(twos)):
                    result[pack + i][c] = 2
            else:
                fours = [r for r in range(barrier + 1, rows) if grid[r][c] == 4]
                twos  = [r for r in range(barrier) if grid[r][c] == 2]
                if not fours:
                    continue
                for r in fours:
                    result[r][c] = 3
                for r in range(barrier + 1, min(fours)):
                    result[r][c] = 4
                pack = len(twos)
                for r in range(pack, barrier):
                    result[r][c] = 8
                for i in range(len(twos)):
                    result[i][c] = 2

    return result


def draw_lines_and_replace_intersecting_rects(
    grid: Grid, line_color: int = 1, rect_color: int = 2, **kwargs
) -> Grid:
    """
    1-markers on opposite edges define lines; convert adjacent/intersecting
    2-rectangles to 1s and draw the full lines.

    Line detection (opposite-edge pairing only):
    - Horizontal line at row R: markers at (R, 0) AND (R, ncols-1)
    - Vertical line at col C:   markers at (0, C) AND (nrows-1, C)

    Conversion rule: a connected rect_color component is converted entirely
    to line_color if ANY line row r satisfies r0-1 <= r <= r1+1, OR any
    line col c satisfies c0-1 <= c <= c1+1 (adjacent counts, not just
    passing through).
    """
    from collections import deque
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if not rows or not cols:
        return [row[:] for row in grid]

    lc = kwargs.get('line_color', line_color)
    rc = kwargs.get('rect_color', rect_color)

    # Collect marker positions
    by_row = {}
    by_col = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == lc:
                by_row.setdefault(r, set()).add(c)
                by_col.setdefault(c, set()).add(r)

    # Opposite-edge pairing
    h_lines = {r for r, cs in by_row.items() if 0 in cs and cols - 1 in cs}
    v_lines = {c for c, rs in by_col.items() if 0 in rs and rows - 1 in rs}

    # Find connected components of rect_color
    visited = [[False] * cols for _ in range(rows)]
    components = []
    for r0 in range(rows):
        for c0 in range(cols):
            if grid[r0][c0] == rc and not visited[r0][c0]:
                comp = []
                q = deque([(r0, c0)])
                visited[r0][c0] = True
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc]==rc:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                rs = [x for x, _ in comp]; cs2 = [x for _, x in comp]
                components.append((comp, min(rs), max(rs), min(cs2), max(cs2)))

    # Determine which components to convert (within 1 cell of any line)
    convert = set()
    for comp, rmin, rmax, cmin, cmax in components:
        if (any(rmin-1 <= lr <= rmax+1 for lr in h_lines) or
                any(cmin-1 <= lc2 <= cmax+1 for lc2 in v_lines)):
            for cell in comp:
                convert.add(cell)

    # Build output
    result = [row[:] for row in grid]
    for r in h_lines:
        for c in range(cols):
            result[r][c] = lc
    for c in v_lines:
        for r in range(rows):
            result[r][c] = lc
    for r, c in convert:
        result[r][c] = lc
    return result


def recolor_by_hole_count(
    grid: Grid,
    color_map: dict | None = None,
    object_color: int = 8,
    background: int = 0,
    **kwargs,
) -> Grid:
    """
    Recolor each connected component of object_color cells based on the number
    of topological holes (isolated enclosed background regions) it contains.

    color_map: {hole_count: output_color}
        MEDIATOR should infer this from the demo pairs and pass it explicitly.
        Example: color_map={0: 5, 1: 2, 2: 9}
        If omitted, falls back to: 0 holes → keep object_color, N holes → color N.

    How holes are counted:
      1. Flood-fill background from all border cells → marks "exterior" cells.
      2. Any background cell NOT reachable from the border is "interior" (a hole).
      3. Each contiguous interior region is one hole.
      4. Each hole is attributed to the object component whose cells are most
         adjacent to that hole region.
    """
    from collections import deque

    if not grid or not grid[0]:
        return [row[:] for row in grid]

    rows, cols = len(grid), len(grid[0])

    # --- Step 1: flood-fill exterior background ---
    exterior = [[False] * cols for _ in range(rows)]
    q: deque = deque()
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1) and grid[r][c] == background:
                exterior[r][c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not exterior[nr][nc] and grid[nr][nc] == background:
                exterior[nr][nc] = True
                q.append((nr, nc))

    # --- Step 2: label object components ---
    comp_label = [[0] * cols for _ in range(rows)]
    comp_cells: dict[int, list] = {}
    cid = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == object_color and comp_label[r][c] == 0:
                cid += 1
                comp_cells[cid] = []
                q = deque([(r, c)])
                comp_label[r][c] = cid
                while q:
                    cr, cc = q.popleft()
                    comp_cells[cid].append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == object_color and comp_label[nr][nc] == 0:
                            comp_label[nr][nc] = cid
                            q.append((nr, nc))

    # --- Step 3: count holes per component ---
    hole_count: dict[int, int] = {k: 0 for k in comp_cells}
    interior_visited = [[False] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == background and not exterior[r][c] and not interior_visited[r][c]:
                # BFS this interior region
                region: list = []
                q = deque([(r, c)])
                interior_visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    region.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == background and not exterior[nr][nc] and not interior_visited[nr][nc]:
                            interior_visited[nr][nc] = True
                            q.append((nr, nc))
                # Attribute hole to adjacent component with most boundary contact
                adj: dict[int, int] = {}
                for hr, hc in region:
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = hr + dr, hc + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            k = comp_label[nr][nc]
                            if k:
                                adj[k] = adj.get(k, 0) + 1
                if adj:
                    best = max(adj, key=lambda x: adj[x])
                    hole_count[best] += 1

    # --- Step 4: recolor ---
    # Normalize color_map keys to int (LLMs sometimes generate string keys like {"1": 1})
    if color_map is not None:
        color_map = {int(k): int(v) for k, v in color_map.items()}
    result = [row[:] for row in grid]
    for k, cells in comp_cells.items():
        cnt = hole_count[k]
        if color_map is not None:
            new_color = color_map.get(cnt, object_color)
        else:
            new_color = object_color if cnt == 0 else cnt
        for r, c in cells:
            result[r][c] = new_color
    return result


def radiate_sequences(grid: Grid, background: int = 0, **kwargs) -> Grid:
    """
    Two-phase transformation for puzzles that contain multiple linear non-zero
    sequences (all cells of a single color in a straight horizontal or vertical
    line):

    Phase 1 — longest sequence radiates diagonally:
      Each cell in the longest sequence (by cell count) radiates its own color
      along all 4 diagonal directions (NW, NE, SW, SE), filling background
      cells until blocked by a non-background cell or the grid boundary.
      Cells are processed from the tip (topmost then leftmost) to the end so
      earlier elements' radiation acts as a barrier for later ones.

    Phase 2 — shorter sequences expand via BFS:
      Each shorter sequence expands outward in all 8 directions via BFS,
      filling only background cells.  Cells already claimed by Phase 1 act as
      natural barriers, so each group expands only into its reachable region.

    This models tasks where one "spine" sequence radiates diagonal stripes
    across the grid while peripheral sequences flood-fill the remaining space.
    """
    from collections import deque

    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if rows == 0 or cols == 0:
        return [row[:] for row in grid]

    # ── Step 1: find all 4-connected non-zero groups ─────────────────────────
    visited = [[False] * cols for _ in range(rows)]
    groups: list[list[tuple[int, int, int]]] = []   # each entry: (r, c, color)

    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != background and not visited[r][c]:
                group: list[tuple[int, int, int]] = []
                q: deque = deque([(r, c)])
                visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    group.append((cr, cc, grid[cr][cc]))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < rows and 0 <= nc < cols
                                and grid[nr][nc] != background
                                and not visited[nr][nc]):
                            visited[nr][nc] = True
                            q.append((nr, nc))
                groups.append(group)

    if not groups:
        return [row[:] for row in grid]

    # ── Step 2: identify longest group ────────────────────────────────────────
    groups.sort(key=lambda g: -len(g))
    longest = groups[0]
    shorter = groups[1:]

    # ── Step 3: build working result grid ────────────────────────────────────
    result = [row[:] for row in grid]

    # ── Step 4: Phase 1 — radiate longest sequence diagonally ────────────────
    # Process from tip: sort by (row, col) so topmost-then-leftmost goes first.
    for r, c, color in sorted(longest, key=lambda x: (x[0], x[1])):
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = r + dr, c + dc
            while 0 <= nr < rows and 0 <= nc < cols:
                if result[nr][nc] != background:
                    break          # stop at any already-filled cell
                result[nr][nc] = color
                nr += dr
                nc += dc

    # ── Step 5: Phase 2 — BFS expand each shorter sequence ───────────────────
    for group in shorter:
        bfs_q: deque = deque()
        local_seen: set = set()

        # Seed: group cells themselves are already filled; queue their empty neighbours
        for r, c, color in group:
            local_seen.add((r, c))

        for r, c, color in group:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols
                            and result[nr][nc] == background
                            and (nr, nc) not in local_seen):
                        local_seen.add((nr, nc))
                        result[nr][nc] = color
                        bfs_q.append((nr, nc, color))

        while bfs_q:
            r, c, color = bfs_q.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols
                            and result[nr][nc] == background
                            and (nr, nc) not in local_seen):
                        local_seen.add((nr, nc))
                        result[nr][nc] = color
                        bfs_q.append((nr, nc, color))

    return result


def recolor_small_components(
    grid: Grid,
    background: int = 0,
    max_size: int = 2,
    new_color: int = 3,
    **kwargs,
) -> Grid:
    """
    Find all connected components of same-colored non-background cells
    (4-connectivity). Any component with <= max_size cells is recolored to
    new_color. Larger components are left unchanged.

    Default parameters match task 12eac192: components of size 1-2 → color 3.
    MEDIATOR should infer max_size and new_color from the demos.
    """
    from collections import deque
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    result = [row[:] for row in grid]
    visited = [[False] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v == background or visited[r][c]:
                continue
            comp: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque([(r, c)])
            visited[r][c] = True
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < rows and 0 <= nc < cols
                            and not visited[nr][nc]
                            and grid[nr][nc] == v):
                        visited[nr][nc] = True
                        q.append((nr, nc))
            if len(comp) <= max_size:
                for cr, cc in comp:
                    result[cr][cc] = new_color
    return result


def fill_blocks_from_key(grid: Grid, map_color: int = 8, background: int = 0, **kwargs) -> Grid:
    """
    Two-region puzzle: a small colored "key" pattern and a large "map" of map_color blocks.
    The map's block layout matches the key's non-zero pattern rotated 90° CW.
    Replace each map_color block with the solid color from the matching rotated-key position.

    Algorithm:
    1. Separate key cells (non-zero, non-map_color) and map cells (map_color).
    2. Find bounding box of key → key_grid (2D array of colors).
    3. Find bounding box of map → determine block size.
    4. Try all 4 rotations of key (0°, 90° CW, 180°, 270° CW); find the one where
       non-zero positions match the map's filled blocks.
    5. Fill each map block with the color from the matching rotated key.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    # --- 1. Find bounding boxes ---
    key_rows_all = [r for r in range(rows) for c in range(cols)
                    if grid[r][c] != background and grid[r][c] != map_color]
    key_cols_all = [c for r in range(rows) for c in range(cols)
                    if grid[r][c] != background and grid[r][c] != map_color]
    map_rows_all = [r for r in range(rows) for c in range(cols) if grid[r][c] == map_color]
    map_cols_all = [c for r in range(rows) for c in range(cols) if grid[r][c] == map_color]

    if not key_rows_all or not map_rows_all:
        return [row[:] for row in grid]

    kr0, kr1 = min(key_rows_all), max(key_rows_all)
    kc0, kc1 = min(key_cols_all), max(key_cols_all)
    mr0, mr1 = min(map_rows_all), max(map_rows_all)
    mc0, mc1 = min(map_cols_all), max(map_cols_all)

    key_h = kr1 - kr0 + 1
    key_w = kc1 - kc0 + 1
    map_h = mr1 - mr0 + 1
    map_w = mc1 - mc0 + 1

    # --- 2. Build key grid ---
    key_grid = [[grid[kr0 + i][kc0 + j] for j in range(key_w)] for i in range(key_h)]

    # --- 3. Determine block size ---
    block_r = map_h // key_h if key_h > 0 else 1
    block_c = map_w // key_w if key_w > 0 else 1
    # If block sizes differ, try the other key dimension order
    if block_r == 0 or block_c == 0:
        return [row[:] for row in grid]

    # --- 4. Build map block pattern ---
    n_block_rows = map_h // block_r
    n_block_cols = map_w // block_c

    def block_is_filled(bi, bj):
        r_start = mr0 + bi * block_r
        c_start = mc0 + bj * block_c
        return any(grid[r_start + dr][c_start + dc] == map_color
                   for dr in range(block_r) for dc in range(block_c))

    map_pattern = [[1 if block_is_filled(i, j) else 0
                    for j in range(n_block_cols)]
                   for i in range(n_block_rows)]

    # --- 5. Try all 4 rotations of key ---
    def rotate_90cw(g):
        r, c = len(g), len(g[0])
        return [[g[r - 1 - j][i] for j in range(r)] for i in range(c)]

    def nonzero_pattern(g):
        return [[1 if g[i][j] != 0 else 0 for j in range(len(g[0]))] for i in range(len(g))]

    def matches(kg, mp):
        if len(kg) != len(mp) or (kg and len(kg[0]) != len(mp[0])):
            return False
        return all(((kg[i][j] != 0) == (mp[i][j] == 1))
                   for i in range(len(mp)) for j in range(len(mp[0])))

    rotated = key_grid
    matched_key = None
    for _ in range(4):
        if matches(rotated, map_pattern):
            matched_key = rotated
            break
        rotated = rotate_90cw(rotated)

    if matched_key is None:
        return [row[:] for row in grid]

    # --- 6. Fill blocks ---
    result = [row[:] for row in grid]
    for bi in range(n_block_rows):
        for bj in range(n_block_cols):
            color = matched_key[bi][bj]
            if color == 0:
                continue
            r_start = mr0 + bi * block_r
            c_start = mc0 + bj * block_c
            for dr in range(block_r):
                for dc in range(block_c):
                    result[r_start + dr][c_start + dc] = color

    return result


def unshear_right(grid: Grid, background: int = 0, **kwargs) -> Grid:
    """
    One-step de-shear: for each color group, keep the bottom row fixed
    and shift every cell in every other row right by 1, capped at the
    group's max right column (min(col + 1, max_right)).

    Groups by color (not 4-connectivity) because sheared shapes are
    diagonally connected, not orthogonally connected.
    """
    from collections import defaultdict
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    color_cells: dict = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != background:
                color_cells[v].append((r, c))

    result = [[background] * cols for _ in range(rows)]

    for color, cells in color_cells.items():
        bottom_row = max(r for r, _ in cells)
        max_right = max(c for _, c in cells)
        for r, c in cells:
            if r == bottom_row:
                result[r][c] = color
            else:
                result[r][min(c + 1, max_right)] = color

    return result


# ANSI color codes for terminal rendering
_ANSI = {
    0: "\033[40m",   # black bg
    1: "\033[44m",   # blue bg
    2: "\033[41m",   # red bg
    3: "\033[42m",   # green bg
    4: "\033[43m",   # yellow bg
    5: "\033[47m",   # white/gray bg
    6: "\033[45m",   # magenta bg
    7: "\033[43m",   # orange → yellow
    8: "\033[46m",   # azure/cyan bg
    9: "\033[41m",   # maroon → red
}
_RESET = "\033[0m"

def border_gravity(grid: Grid, background: int = 7, **kwargs) -> Grid:
    """Border-color gravity: row 0 color floats UP, last-row color sinks DOWN.

    Rule:
      - The top border color is grid[0][0]; the bottom border color is grid[-1][0].
      - Interior floating cells (rows 1..rows-2) of each border color are grouped
        into 8-connected components.
      - Each top-color component shifts up so its topmost row becomes row 1.
      - Each bottom-color component shifts down so its bottommost row becomes rows-2.
      - Border rows (0 and rows-1) are unchanged.
      - All other interior cells become `background`.
    """
    arr = to_np(grid)
    rows, cols = arr.shape
    top_color = int(arr[0, 0])
    bot_color = int(arr[rows - 1, 0])

    def get_components(color: int):
        pos_set = {(r, c) for r in range(1, rows - 1) for c in range(cols) if arr[r, c] == color}
        visited: set = set()
        comps = []
        for start in sorted(pos_set):
            if start in visited:
                continue
            comp = []
            stack = [start]
            while stack:
                rc = stack.pop()
                if rc in visited:
                    continue
                visited.add(rc)
                comp.append(rc)
                r0, c0 = rc
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nb = (r0 + dr, c0 + dc)
                        if nb in pos_set and nb not in visited:
                            stack.append(nb)
            comps.append(comp)
        return comps

    out = np.full_like(arr, background)
    out[0] = arr[0]
    out[rows - 1] = arr[rows - 1]

    for comp in get_components(top_color):
        min_row = min(r for r, _ in comp)
        shift = min_row - 1
        for r, c in comp:
            out[r - shift, c] = top_color

    for comp in get_components(bot_color):
        max_row = max(r for r, _ in comp)
        shift = (rows - 2) - max_row
        for r, c in comp:
            out[r + shift, c] = bot_color

    return to_list(out)


def grid_to_str(grid: Grid, use_ansi: bool = False) -> str:
    """Compact string representation of a grid."""
    lines = []
    for row in grid:
        if use_ansi:
            lines.append("".join(f"{_ANSI.get(c, '')} {c} {_RESET}" for c in row))
        else:
            lines.append(" ".join(str(c) for c in row))
    return "\n".join(lines)

def summarize(grid: Grid) -> str:
    """One-line summary: shape + color distribution."""
    if not grid:
        return "(empty)"
    r, c = shape(grid)
    dist = {col: color_count(grid, col) for col in unique_colors(grid)}
    dist_str = ", ".join(f"{COLOR_NAMES.get(k, k)}={v}" for k, v in sorted(dist.items()))
    return f"{r}×{c}  [{dist_str}]"
