"""Harness-side BFS navigator for ARC-AGI-3 play loop.

Given a set of known action effects (action_name -> (dr, dc)) and a
start position, plans a path to a target position using BFS over the
reachable discrete grid.
"""
from __future__ import annotations

from collections import deque
from typing import Optional


GRID_H = 64
GRID_W = 64


def bfs_navigate(
    start:          tuple[int, int],
    target:         tuple[int, int],
    action_effects: dict[str, tuple[int, int]],
    max_steps:      int = 60,
    walls:          set | None = None,
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
            if walls and (r, c, action) in walls:
                continue  # known wall — skip
            nr = max(0, min(GRID_H - 1, r + dr))
            nc = max(0, min(GRID_W - 1, c + dc))
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
            if walls and (r, c, action) in walls:
                continue
            nr = max(0, min(GRID_H - 1, r + dr))
            nc = max(0, min(GRID_W - 1, c + dc))
            steps = len(path) + 1
            if (nr, nc) in visited and visited[(nr, nc)] <= steps:
                continue
            visited[(nr, nc)] = steps
            queue.append(((nr, nc), path + [action]))

    if best_pos == (sr, sc):
        return None
    return best_pos, best_path
