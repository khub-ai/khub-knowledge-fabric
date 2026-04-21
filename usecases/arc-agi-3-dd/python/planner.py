"""State-augmented BFS planner (BFS+) for ARC-AGI-3 levels.

Solves the "reach win cell with N rotation advances and non-zero budget,
using at most M pickups along the way" constrained shortest-path problem.

Classical shortest-path navigators (navigator.py bfs_navigate) only handle
start -> target with a boolean passable grid.  They cannot reason about:
  - required intermediate stops (cross cell must be entered K times)
  - consumable resources (pickups reset budget to max)
  - rotation-gated goals (win cell is impassable until advances_done == K)
  - budget depletion (running below 0 kills the level)

This module solves it mechanically via BFS on the augmented state space
  (position, advances_done, pickups_remaining, budget)

For typical ARC-AGI-3 levels the state space is small (~30K states for 2/7
= 50 passable cells * 4 advances * 4 pickup subsets * 43 budget values),
so plain BFS with unit edge weights returns the provably optimal sequence
in well under 100 ms.

Dependency boundary: only navigator.build_passable_grid is shared.  No
imports from run_play.py or anthropic SDK here -- the planner is
standalone and unit-testable.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


# Game mechanics:
#   - Entering a cross cell increments advances_done (capped at required).
#     The game fires "entry" only when the move's destination IS the cross
#     cell (KB B13). Standing on it does not re-fire; you must step off
#     and step back.
#   - Entering a pickup cell consumes it (removed from the level) and
#     resets budget to budget_max (KB B14).
#   - Entering the win cell with advances_done < required is IMPASSABLE
#     (KB trap: "mismatch causes the cell to return bwdzgjttjp=True").
#     We model this by gating transitions into the win cell on alignment.
#   - Budget decrements by 1 per env.step. Budget at 0 = level reset +
#     life lost.  We prune any state with budget <= 0 (cannot continue).


def solve_align_phase(
    *,
    spawn:              tuple[int, int],
    cross_positions:    list[tuple[int, int]],
    pickup_positions:   list[tuple[int, int]],
    advances_required:  int,
    budget_max:         int,
    action_effects:     dict[str, tuple[int, int]],
    passable_grid:      np.ndarray,
    walls:              set | None = None,
    budget_current:     int | None = None,
    budget_per_action:  int = 1,
    min_final_budget:   int = 0,
) -> Optional[tuple[list[str], tuple[int, int], int, frozenset]]:
    """Phase 1 of a two-phase solve: reach advances_done == advances_required
    with at least `min_final_budget` budget remaining.

    Returns (sequence, final_pos, final_budget, final_pickups_remaining)
    or None if unreachable.  We exhaust the BFS frontier and return the
    alignment-terminal state with MAX budget remaining (tie-break: fewer
    steps = earlier discovery).  This typically requires detouring through
    a pickup when the direct align-path would leave budget too low.
    """
    H, W = passable_grid.shape
    cross_set = frozenset(tuple(c) for c in cross_positions)
    walls = walls or set()
    budget0 = budget_current if budget_current is not None else budget_max
    pickups0 = frozenset(tuple(p) for p in pickup_positions)
    move_actions = {a: (dr, dc) for a, (dr, dc) in action_effects.items()
                    if dr != 0 or dc != 0}
    if not move_actions:
        return None

    init_state = (tuple(spawn), 0, pickups0, budget0)
    if advances_required == 0:
        return ([], tuple(spawn), budget0, pickups0)

    parent: dict = {init_state: None}
    queue: deque = deque([init_state])

    # Collect all alignment-terminal states; caller picks among them.
    terminals: list = []   # list of (score, state)

    while queue:
        state = queue.popleft()
        pos, adv, pickups, budget = state
        if budget <= 0:
            continue
        for action_name, (dr, dc) in move_actions.items():
            if (pos[0], pos[1], action_name) in walls:
                continue
            nr = max(0, min(H - 1, pos[0] + dr))
            nc = max(0, min(W - 1, pos[1] + dc))
            new_pos = (nr, nc)
            if not passable_grid[nr, nc] or new_pos == pos:
                continue
            new_adv = min(adv + 1, advances_required) if new_pos in cross_set else adv
            new_pickups = pickups
            new_budget = budget - budget_per_action
            if new_pos in pickups:
                new_pickups = pickups - {new_pos}
                new_budget = budget_max
            new_state = (new_pos, new_adv, new_pickups, new_budget)

            if new_state in parent:
                continue
            parent[new_state] = (state, action_name)

            if new_adv == advances_required and new_budget >= min_final_budget:
                # Score = current_budget + (unused_pickups * budget_max).
                score = new_budget + len(new_pickups) * budget_max
                terminals.append((score, new_state))
                # Don't enqueue beyond alignment -- more wandering only
                # wastes budget.
                continue

            if new_budget <= 0:
                continue
            queue.append(new_state)

    if not terminals:
        return None

    # Sort terminals by score descending; caller will try each.
    terminals.sort(key=lambda t: -t[0])
    # Legacy single-candidate API: return the TOP scoring terminal.
    _, terminal_state = terminals[0]
    path_rev: list[str] = []
    cur = terminal_state
    while parent[cur] is not None:
        prev, act = parent[cur]
        path_rev.append(act)
        cur = prev
    path_rev.reverse()
    t_pos, _, t_pickups, t_budget = terminal_state
    return (path_rev, t_pos, t_budget, t_pickups)


def solve_align_phase_all(
    **kw
) -> list:
    """Same as solve_align_phase but returns ALL alignment-terminal states,
    sorted by descending score (current_budget + unused_pickups * budget_max).
    Each element is (path, final_pos, final_budget, pickups_remaining)."""
    # Duplicate of solve_align_phase but collects all terminals with paths.
    passable_grid     = kw["passable_grid"]
    spawn             = kw["spawn"]
    cross_positions   = kw["cross_positions"]
    pickup_positions  = kw["pickup_positions"]
    advances_required = kw["advances_required"]
    budget_max        = kw["budget_max"]
    action_effects    = kw["action_effects"]
    walls             = kw.get("walls") or set()
    budget_current    = kw.get("budget_current")
    budget_per_action = kw.get("budget_per_action", 1)
    min_final_budget  = kw.get("min_final_budget", 0)

    H, W = passable_grid.shape
    cross_set = frozenset(tuple(c) for c in cross_positions)
    budget0   = budget_current if budget_current is not None else budget_max
    pickups0  = frozenset(tuple(p) for p in pickup_positions)
    move_actions = {a: (dr, dc) for a, (dr, dc) in action_effects.items()
                    if dr != 0 or dc != 0}
    if not move_actions:
        return []

    init_state = (tuple(spawn), 0, pickups0, budget0)
    if advances_required == 0:
        return [([], tuple(spawn), budget0, pickups0)]

    parent: dict = {init_state: None}
    queue: deque = deque([init_state])
    terminals: list = []

    while queue:
        state = queue.popleft()
        pos, adv, pickups, budget = state
        if budget <= 0:
            continue
        for action_name, (dr, dc) in move_actions.items():
            if (pos[0], pos[1], action_name) in walls:
                continue
            nr = max(0, min(H - 1, pos[0] + dr))
            nc = max(0, min(W - 1, pos[1] + dc))
            new_pos = (nr, nc)
            if not passable_grid[nr, nc] or new_pos == pos:
                continue
            new_adv = min(adv + 1, advances_required) if new_pos in cross_set else adv
            new_pickups = pickups
            new_budget = budget - budget_per_action
            if new_pos in pickups:
                new_pickups = pickups - {new_pos}
                new_budget = budget_max
            new_state = (new_pos, new_adv, new_pickups, new_budget)

            if new_state in parent:
                continue
            parent[new_state] = (state, action_name)

            if new_adv == advances_required and new_budget >= min_final_budget:
                score = new_budget + len(new_pickups) * budget_max
                terminals.append((score, new_state))
                continue

            if new_budget <= 0:
                continue
            queue.append(new_state)

    # Reconstruct each terminal's path and return sorted.
    candidates = []
    for score, terminal_state in terminals:
        path_rev: list[str] = []
        cur = terminal_state
        while parent[cur] is not None:
            prev, act = parent[cur]
            path_rev.append(act)
            cur = prev
        path_rev.reverse()
        t_pos, _, t_pickups, t_budget = terminal_state
        candidates.append((score, path_rev, t_pos, t_budget, t_pickups))
    candidates.sort(key=lambda c: -c[0])
    return [(p, pos, b, pk) for (_, p, pos, b, pk) in candidates]


def solve_win_phase(
    *,
    start_pos:          tuple[int, int],
    pickup_positions:   list[tuple[int, int]],   # remaining (not consumed)
    win_position:       tuple[int, int],
    budget_current:     int,
    budget_max:         int,
    action_effects:     dict[str, tuple[int, int]],
    passable_grid:      np.ndarray,              # post-alignment geometry
    walls:              set | None = None,
    budget_per_action:  int = 1,
) -> Optional[list[str]]:
    """Phase 2 of a two-phase solve: navigate from start_pos to win_position
    with alignment already achieved.  Uses the post-alignment passable grid
    (which the caller obtained by snapshotting the env frame after phase 1).

    Pickups may still be collected for extra budget.  Rotation is assumed
    fixed (no cross cells on the way).
    """
    H, W = passable_grid.shape
    walls = walls or set()
    pickups0 = frozenset(tuple(p) for p in pickup_positions)
    move_actions = {a: (dr, dc) for a, (dr, dc) in action_effects.items()
                    if dr != 0 or dc != 0}
    if not move_actions:
        return None

    init_state = (tuple(start_pos), pickups0, budget_current)
    if init_state[0] == tuple(win_position):
        return []

    parent: dict = {init_state: None}
    queue: deque = deque([init_state])

    while queue:
        state = queue.popleft()
        pos, pickups, budget = state
        if budget <= 0:
            continue
        for action_name, (dr, dc) in move_actions.items():
            if (pos[0], pos[1], action_name) in walls:
                continue
            nr = max(0, min(H - 1, pos[0] + dr))
            nc = max(0, min(W - 1, pos[1] + dc))
            new_pos = (nr, nc)
            if not passable_grid[nr, nc] or new_pos == pos:
                continue
            new_pickups = pickups
            new_budget = budget - budget_per_action
            if new_pos in pickups:
                new_pickups = pickups - {new_pos}
                new_budget = budget_max
            if new_pos == tuple(win_position):
                path_rev = [action_name]
                cur = state
                while parent[cur] is not None:
                    prev, act = parent[cur]
                    path_rev.append(act)
                    cur = prev
                path_rev.reverse()
                return path_rev
            if new_budget <= 0:
                continue
            new_state = (new_pos, new_pickups, new_budget)
            if new_state in parent:
                continue
            parent[new_state] = (state, action_name)
            queue.append(new_state)

    return None


def solve_level(
    *,
    spawn:              tuple[int, int],
    cross_positions:    list[tuple[int, int]],
    pickup_positions:   list[tuple[int, int]],
    win_position:       tuple[int, int],
    advances_required:  int,
    budget_max:         int,
    action_effects:     dict[str, tuple[int, int]],
    passable_grid:      np.ndarray,           # (H, W) bool
    walls:              set | None = None,    # {(r, c, action_name)}
    budget_current:     int | None = None,    # default: start at budget_max
    budget_per_action:  int = 1,
) -> Optional[list[str]]:
    """Return the shortest action sequence that solves this level, or None
    if unsolvable from the given start state.

    The returned list contains action names like "ACTION1", "ACTION4", ...
    in order.  Execute via env.step(int(name[-1])) to replay deterministically.
    """
    H, W = passable_grid.shape
    cross_set   = frozenset(tuple(c) for c in cross_positions)
    win_cell    = tuple(win_position)
    walls       = walls or set()
    budget0     = budget_current if budget_current is not None else budget_max
    pickups0    = frozenset(tuple(p) for p in pickup_positions)

    # Zero-displacement actions (if any) are unusable for navigation.
    move_actions = {
        a: (dr, dc) for a, (dr, dc) in action_effects.items()
        if dr != 0 or dc != 0
    }
    if not move_actions:
        return None

    init_state = (tuple(spawn), 0, pickups0, budget0)

    # Degenerate: already at goal (spawn is win cell, no advances needed).
    if init_state[0] == win_cell and advances_required == 0:
        return []

    # BFS with parent-pointer path reconstruction.
    # parent: state -> (prev_state, action_taken_to_reach_state) | None
    parent: dict = {init_state: None}
    queue: deque = deque([init_state])

    while queue:
        state = queue.popleft()
        pos, adv, pickups, budget = state

        # budget <= 0 means next move would kill us; treat as dead-end.
        # (We still check budget pre-step in the expansion below.)
        if budget <= 0:
            continue

        for action_name, (dr, dc) in move_actions.items():
            # Runtime wall: "from pos, this action is blocked."
            if (pos[0], pos[1], action_name) in walls:
                continue

            nr = max(0, min(H - 1, pos[0] + dr))
            nc = max(0, min(W - 1, pos[1] + dc))
            new_pos = (nr, nc)

            # Blocked by palette-level wall.
            if not passable_grid[nr, nc]:
                continue

            # Boundary clamp produced a no-op (move into edge of grid).
            if new_pos == pos:
                continue

            # Compute new augmented state.
            new_adv     = adv
            new_pickups = pickups
            new_budget  = budget - budget_per_action

            if new_pos in cross_set:
                new_adv = min(adv + 1, advances_required)

            if new_pos in pickups:
                # Consumable: pickup disappears, budget resets.
                new_pickups = pickups - {new_pos}
                new_budget  = budget_max

            # Win cell is rotation-gated.  Entering while not aligned is
            # physically impossible (game treats cell as a wall).
            if new_pos == win_cell and new_adv < advances_required:
                continue

            # Budget dead-end: if this move exhausts budget and we're NOT
            # at the goal, we cannot continue from there.  (Staying is OK
            # only if we're done.)
            new_state = (new_pos, new_adv, new_pickups, new_budget)

            # Goal test: at win cell AND rotation aligned.
            if new_pos == win_cell and new_adv == advances_required:
                # Reconstruct the path.
                path_rev: list[str] = [action_name]
                cur = state
                while parent[cur] is not None:
                    prev, act = parent[cur]
                    path_rev.append(act)
                    cur = prev
                path_rev.reverse()
                return path_rev

            # Dead if budget hits 0 here (we already know we're not at the
            # goal because the goal-check above returned).
            if new_budget <= 0:
                continue

            if new_state in parent:
                continue
            parent[new_state] = (state, action_name)
            queue.append(new_state)

    return None   # unsolvable


# -----------------------------------------------------------------------------
# Optional convenience: build a level problem from an env observation.
# -----------------------------------------------------------------------------

def problem_from_env_query(
    *,
    query_level_state: dict,
    passable_grid: np.ndarray,
    action_effects: dict[str, tuple[int, int]],
    walls: set | None = None,
) -> dict:
    """Translate a `_query_level_state(env)` dict into `solve_level(**kw)`.

    Returns a kwargs dict ready to splat into solve_level.  Returns None if
    the query is missing essential fields.
    """
    needed = (
        "agent_cursor", "cross_positions", "pickup_positions",
        "win_positions", "advances_remaining", "budget_max",
    )
    if not all(k in query_level_state for k in needed):
        return None

    win_list = query_level_state["win_positions"] or []
    if not win_list:
        return None

    # advances_remaining counts the remaining entries needed; the total
    # required is that plus whatever's already been done.  But the planner
    # starts fresh (advances_done=0), so we pass advances_remaining as
    # "required from here."
    advances_req = int(query_level_state["advances_remaining"])

    return dict(
        spawn              = tuple(query_level_state["agent_cursor"]),
        cross_positions    = [tuple(c) for c in query_level_state["cross_positions"]],
        pickup_positions   = [tuple(p) for p in query_level_state["pickup_positions"]],
        win_position       = tuple(win_list[0]),   # use first if multiple
        advances_required  = advances_req,
        budget_max         = int(query_level_state["budget_max"]),
        budget_current     = int(query_level_state.get("budget_current", query_level_state["budget_max"])),
        action_effects     = action_effects,
        passable_grid      = passable_grid,
        walls              = walls,
    )
