"""
nav_bfs.py — game-agnostic BFS navigation for ARC-AGI-3.

Computes an optimal action sequence from a start position through
one or more waypoints to a final goal, given a pixel grid and
walkability constraints.
"""
from collections import deque
from typing import Optional

# Direction → (delta_col, delta_row)
_DIR_DELTAS = {
    "UP":    (0, -1),
    "DOWN":  (0,  1),
    "LEFT":  (-1, 0),
    "RIGHT": ( 1, 0),
}


def extract_walkable_grid(frame: list[list[int]], walkable_colors: set[int],
                           step_size: int = 5) -> set[tuple[int, int]]:
    """
    Return the set of pixel positions (col, row) that are marked as walkable
    in the frame.

    We sample every pixel rather than only multiples of step_size because
    the actual step-aligned positions in ARC-AGI-3 games often start at a
    non-zero offset (e.g. col=4, 9, 14, 19 … rather than 0, 5, 10, 15 …).
    Storing all walkable pixels is still cheap for the ~64×64 frames used by
    ARC-AGI-3, and it ensures that BFS nodes at any alignment can find their
    neighbours in the walkable set.
    """
    if not frame:
        return set()
    rows = len(frame)
    cols = len(frame[0]) if rows > 0 else 0
    walkable = set()
    for r in range(rows):
        row = frame[r]
        for c in range(cols):
            if row[c] in walkable_colors:
                walkable.add((c, r))
    return walkable


def bfs_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    walkable: set[tuple[int, int]],
    step_size: int = 5,
    extra_passable: Optional[set[tuple[int, int]]] = None,
) -> Optional[list[str]]:
    """
    BFS from start to goal in coarse-grid coordinates (col, row).
    Each step moves step_size pixels in one of 4 directions.

    Returns list of direction strings ("UP", "DOWN", "LEFT", "RIGHT"),
    or None if no path exists.

    extra_passable: additional cells (like ROT_CHANGER, WIN_TARGET) that
    are not color3 but the player can still occupy.
    """
    passable = walkable | (extra_passable or set())
    passable.add(start)
    passable.add(goal)

    if start == goal:
        return []

    queue = deque([(start, [])])
    visited = {start}

    while queue:
        (col, row), path = queue.popleft()
        for direction, (dc, dr) in _DIR_DELTAS.items():
            nc = col + dc * step_size
            nr = row + dr * step_size
            npos = (nc, nr)
            if npos in visited:
                continue
            if npos not in passable:
                continue
            new_path = path + [direction]
            if npos == goal:
                return new_path
            visited.add(npos)
            queue.append((npos, new_path))

    return None  # no path found


def directions_to_actions(directions: list[str],
                           action_map: dict[str, str]) -> list[str]:
    """
    Convert direction strings to ACTION names using action_map.
    action_map: {"UP": "ACTION1", "DOWN": "ACTION2", ...}
    """
    # action_map maps direction -> action name
    return [action_map[d] for d in directions]


def compute_navigation_plan(
    frame: list[list[int]],
    waypoints: list[tuple[int, int]],  # [start, intermediate..., goal]
    walkable_colors: set[int],
    step_size: int,
    action_map: dict[str, str],  # {"UP": "ACTION1", ...}
    extra_passable: Optional[set[tuple[int, int]]] = None,
    blocked_positions: Optional[set[tuple[int, int]]] = None,
) -> Optional[list[str]]:
    """
    Compute the full action sequence to navigate through all waypoints in order.
    Returns list of ACTION names, or None if any segment has no path.

    blocked_positions: cells the player has empirically failed to enter
    (walls between grid cells).  These are subtracted from the walkable set
    so BFS routes around them.
    """
    walkable = extract_walkable_grid(frame, walkable_colors, step_size)
    if blocked_positions:
        walkable -= blocked_positions
    full_actions = []
    for i in range(len(waypoints) - 1):
        seg_path = bfs_path(waypoints[i], waypoints[i + 1], walkable,
                            step_size, extra_passable)
        if seg_path is None:
            return None
        full_actions.extend(directions_to_actions(seg_path, action_map))
    return full_actions


def format_nav_plan(actions: list[str]) -> str:
    """Compact representation: ACTION1×3, ACTION3×2, ..."""
    if not actions:
        return "(no moves needed)"
    result = []
    i = 0
    while i < len(actions):
        act = actions[i]
        count = 1
        while i + count < len(actions) and actions[i + count] == act:
            count += 1
        result.append(f"{act}×{count}" if count > 1 else act)
        i += count
    return ", ".join(result)


def find_player_position(frame: list[list[int]],
                          player_colors: set[int]) -> Optional[tuple[int, int]]:
    """
    Locate the player in the frame by finding all pixels that belong to any
    player color, then returning the top-left corner of their bounding box.

    The BFS coarse grid treats this top-left corner as the player's position
    (col, row).  Returns None if no player pixels are found.
    """
    min_r, min_c = None, None
    for r, row in enumerate(frame):
        for c, px in enumerate(row):
            if px in player_colors:
                if min_r is None or r < min_r:
                    min_r = r
                    min_c = c
                elif r == min_r and c < min_c:
                    min_c = c
    if min_r is None:
        return None
    return (min_c, min_r)  # (col, row)
