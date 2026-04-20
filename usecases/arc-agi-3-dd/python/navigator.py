"""Harness-side BFS navigator for ARC-AGI-3 play loop.

Given a set of known action effects (action_name -> (dr, dc)) and a
start position, plans a path to a target position using BFS over the
reachable discrete grid.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

GRID_H = 64
GRID_W = 64

# Palette values the agent CANNOT enter (read from the game frame).
# Palette 4 = wall/gap tiles confirmed by pixel analysis.
WALL_PALETTES: frozenset[int] = frozenset({4})


def _passable(nr: int, nc: int, passable_grid) -> bool:
    """Return True if cell (nr,nc) is navigable according to passable_grid."""
    if passable_grid is None:
        return True
    return bool(passable_grid[nr, nc])


def build_passable_grid(
    frame_grid,
    wall_palettes: frozenset[int] = WALL_PALETTES,
) -> np.ndarray:
    """Return a 64×64 bool array: True = agent can enter this cell.

    Reads palette values directly from the game frame so BFS never needs to
    discover walls by bumping into them.
    """
    arr = np.asarray(frame_grid, dtype=np.int32)
    passable = np.ones(arr.shape[:2], dtype=bool)
    for p in wall_palettes:
        passable[arr == p] = False
    return passable


def bfs_navigate(
    start:          tuple[int, int],
    target:         tuple[int, int],
    action_effects: dict[str, tuple[int, int]],
    max_steps:      int = 60,
    walls:          set | None = None,
    passable_grid           = None,
) -> Optional[list[str]]:
    """Return the shortest action sequence to move from `start` to `target`.

    action_effects maps action name -> (dr, dc).  Only actions with non-zero
    displacement are used for navigation; zero-displacement actions are ignored.

    Returns a list of action names (possibly empty if already at target), or
    None if the target is unreachable within max_steps.
    """
    if start == target:
        return []

    move_actions = {a: (dr, dc) for a, (dr, dc) in action_effects.items()
                    if dr != 0 or dc != 0}
    if not move_actions:
        return None

    sr, sc = start
    tr, tc = target

    # BFS: state = (row, col), track best path
    queue: deque[tuple[tuple[int, int], list[str]]] = deque([((sr, sc), [])])
    visited: dict[tuple[int, int], int] = {(sr, sc): 0}

    while queue:
        (r, c), path = queue.popleft()
        if len(path) >= max_steps:
            continue
        for action, (dr, dc) in move_actions.items():
            nr = max(0, min(GRID_H - 1, r + dr))
            nc = max(0, min(GRID_W - 1, c + dc))
            if walls and (r, c, action) in walls:
                # Recorded wall: trust it only if passable_grid also confirms
                # the destination is impassable.  If passable_grid says it's
                # open, the wall was likely a cursor-drift false positive.
                if not _passable(nr, nc, passable_grid):
                    continue  # real wall confirmed by palette
                # else: passable_grid overrides — allow this move
            if not _passable(nr, nc, passable_grid):
                continue  # palette-level wall — skip
            steps = len(path) + 1
            if (nr, nc) in visited and visited[(nr, nc)] <= steps:
                continue
            new_path = path + [action]
            if (nr, nc) == (tr, tc):
                return new_path
            visited[(nr, nc)] = steps
            queue.append(((nr, nc), new_path))

    return None  # unreachable within max_steps


def nearest_reachable(
    start:          tuple[int, int],
    target:         tuple[int, int],
    action_effects: dict[str, tuple[int, int]],
    max_steps:      int = 60,
    walls:          set | None = None,
    passable_grid           = None,
) -> Optional[tuple[tuple[int, int], list[str]]]:
    """Return (closest_reachable_pos, path) to get as close as possible to target.

    Used when the exact target cell is not on the reachable grid.
    """
    move_actions = {a: (dr, dc) for a, (dr, dc) in action_effects.items()
                    if dr != 0 or dc != 0}
    if not move_actions:
        return None

    sr, sc = start
    tr, tc = target
    best_pos = (sr, sc)
    best_path: list[str] = []
    best_dist = (sr - tr) ** 2 + (sc - tc) ** 2

    queue: deque[tuple[tuple[int, int], list[str]]] = deque([((sr, sc), [])])
    visited: dict[tuple[int, int], int] = {(sr, sc): 0}

    while queue:
        (r, c), path = queue.popleft()
        dist = (r - tr) ** 2 + (c - tc) ** 2
        if dist < best_dist:
            best_dist = dist
            best_pos = (r, c)
            best_path = path
        if len(path) >= max_steps:
            continue
        for action, (dr, dc) in move_actions.items():
            nr = max(0, min(GRID_H - 1, r + dr))
            nc = max(0, min(GRID_W - 1, c + dc))
            if walls and (r, c, action) in walls:
                if not _passable(nr, nc, passable_grid):
                    continue  # real wall confirmed by palette
            if not _passable(nr, nc, passable_grid):
                continue
            steps = len(path) + 1
            if (nr, nc) in visited and visited[(nr, nc)] <= steps:
                continue
            visited[(nr, nc)] = steps
            queue.append(((nr, nc), path + [action]))

    if best_pos == (sr, sc):
        return None
    return best_pos, best_path
