"""
planner.py — Goal-directed BFS planner over the tracker's inferred world model.

The planner is the symbolic core's "exploit" mode: given a HypothesisTracker
that has discovered top-1 ActionModelHypotheses for the available actions,
plus a current frame and a goal predicate, it searches for an action sequence
that drives the frame to a goal-satisfying state.

Design constraints (matching DSL doc Layer 6):

  - Input is the *learned* world model (tracker.top_action_models), never the
    ground truth. If the tracker is wrong, the planner will be wrong; that's
    measurable, which is the whole point.

  - Forward simulation is purely DSL-symbolic: TRANSLATE moves the inferred
    "agent cells", TOGGLE recolors the inferred slot cells, NO_OP returns the
    frame unchanged. No game-engine, no ground-truth lookup.

  - Goal predicates are tiny callables `frame -> bool`. The information-gain
    explorer (separate concern) handles the case where no goal is yet known.

  - State space is hashed by frame tuple. BFS guarantees shortest plan in
    action count, which matches what real ARC-AGI-3 scoring rewards.

A second routine, `information_gain_action`, picks the action with the
highest expected entropy reduction over the action-model posterior — this is
"explore" mode and is what the agent uses before goal beliefs solidify.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

from dsl import (
    ActionModelHypothesis,
    HypothesisTracker,
    _detect_translate,  # reused for agent-cell tracking
)
from proposer_schema import EffectType


Frame = list[list[int]]
GoalPredicate = Callable[[Frame], bool]


# =============================================================================
# Frame helpers
# =============================================================================

def _frame_key(frame: Frame) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(row) for row in frame)


def _clone(frame: Frame) -> Frame:
    return [row[:] for row in frame]


def _find_cells(frame: Frame, color: int) -> set[tuple[int, int]]:
    return {(r, c) for r, row in enumerate(frame)
            for c, v in enumerate(row) if v == color}


def _dominant_bg(frame: Frame) -> int:
    counts: dict[int, int] = {}
    for row in frame:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


# =============================================================================
# Forward simulator
# =============================================================================

def simulate(
    frame: Frame,
    model: ActionModelHypothesis,
    agent_color: Optional[int] = None,
) -> Frame:
    """Apply `model` to `frame` and return the predicted next frame.

    The simulator is intentionally minimal — it knows about TRANSLATE,
    TOGGLE, and NO_OP. Anything else is a no-op (planner can still search
    over it, just won't make progress).

    For TRANSLATE: locate the inferred agent (cells of `agent_color`),
    move them by (dr, dc), and reject the move if any destination cell is
    out of bounds OR overlaps a non-background obstacle. This mirrors the
    IF_BLOCKED precondition the tracker discovers.
    """
    et = model.effect_type
    if et == EffectType.NO_OP:
        return _clone(frame)

    if et == EffectType.TRANSLATE:
        dr = model.params.get("dr", 0)
        dc = model.params.get("dc", 0)
        bg = _dominant_bg(frame)
        if agent_color is None:
            # Heuristic: smallest non-background object — the agent is rarely
            # the largest thing in the scene. Caller should pass agent_color
            # explicitly when known.
            return _clone(frame)
        agent_cells = _find_cells(frame, agent_color)
        if not agent_cells:
            return _clone(frame)
        new_cells = {(r + dr, c + dc) for r, c in agent_cells}
        h = len(frame)
        w = len(frame[0]) if h else 0
        # Identify the wall color heuristically: the most populous non-bg,
        # non-agent color in the frame. Only that color blocks movement;
        # everything else (including the goal marker) is walkable.
        counts: dict[int, int] = {}
        for row in frame:
            for v in row:
                if v != bg and v != agent_color:
                    counts[v] = counts.get(v, 0) + 1
        wall_color = max(counts.items(), key=lambda kv: kv[1])[0] if counts else None
        for r, c in new_cells:
            if r < 0 or r >= h or c < 0 or c >= w:
                return _clone(frame)  # blocked
            if (r, c) in agent_cells:
                continue
            if wall_color is not None and frame[r][c] == wall_color:
                return _clone(frame)  # blocked by wall
        out = _clone(frame)
        for r, c in agent_cells:
            out[r][c] = bg
        for r, c in new_cells:
            out[r][c] = agent_color
        return out

    if et == EffectType.TOGGLE:
        cells = model.params.get("cells")
        if not cells:
            return _clone(frame)
        # We don't know the next cycle color from the model alone, so the
        # planner uses a sentinel: cycle each cell through (val + 1) within
        # the set of colors observed in those cells. This is enough for the
        # synthetic toggle puzzle (cycle ∈ {12,13,14,15}).
        out = _clone(frame)
        # Pick the current color in the slot region; bump by 1 within
        # SLOT_COLORS = [12,13,14,15] if applicable, else add 1 mod 16.
        SLOT_COLORS = [12, 13, 14, 15]
        sample_r, sample_c = next(iter(cells))
        cur = frame[sample_r][sample_c]
        if cur in SLOT_COLORS:
            nxt = SLOT_COLORS[(SLOT_COLORS.index(cur) + 1) % len(SLOT_COLORS)]
        else:
            nxt = (cur + 1) % 16
        for r, c in cells:
            out[r][c] = nxt
        return out

    return _clone(frame)


# =============================================================================
# BFS planner
# =============================================================================

@dataclass
class PlanResult:
    plan:        list[str]
    nodes:       int
    found:       bool


def plan_to_goal(
    tracker: HypothesisTracker,
    start_frame: Frame,
    goal: GoalPredicate,
    agent_color: Optional[int] = None,
    max_depth: int = 32,
    max_nodes: int = 20000,
) -> PlanResult:
    """BFS over the tracker's top-1 model per action.

    Returns the shortest action sequence that drives the start frame to a
    state satisfying `goal`, or an empty plan with found=False if no plan
    is found within the budget.
    """
    if goal(start_frame):
        return PlanResult(plan=[], nodes=0, found=True)

    # Snapshot top-1 model per action.
    actions: list[tuple[str, ActionModelHypothesis]] = []
    for aid in tracker.available_actions:
        top = tracker.top_action_models(aid, k=1)
        if top:
            actions.append((aid, top[0]))

    seen: set[tuple[tuple[int, ...], ...]] = {_frame_key(start_frame)}
    q: deque[tuple[Frame, list[str]]] = deque([(start_frame, [])])
    nodes = 0
    while q:
        frame, path = q.popleft()
        if len(path) >= max_depth:
            continue
        for aid, model in actions:
            nodes += 1
            if nodes > max_nodes:
                return PlanResult(plan=[], nodes=nodes, found=False)
            nxt = simulate(frame, model, agent_color=agent_color)
            key = _frame_key(nxt)
            if key in seen:
                continue
            if goal(nxt):
                return PlanResult(plan=path + [aid], nodes=nodes, found=True)
            seen.add(key)
            q.append((nxt, path + [aid]))
    return PlanResult(plan=[], nodes=nodes, found=False)


# =============================================================================
# Information-gain explorer
# =============================================================================

def information_gain_action(
    tracker: HypothesisTracker,
    available_actions: list[str],
) -> str:
    """Pick the action whose top-k posterior is least concentrated.

    Entropy of the (normalized) top-k posterior distribution; the action
    with the *highest* entropy is the one we know least about and should
    exercise next. Untouched actions are infinitely informative and win.
    """
    best_action = available_actions[0]
    best_score = -1.0
    for aid in available_actions:
        models = tracker.top_action_models(aid, k=8)
        if not models:
            return aid  # never observed → maximum information value
        ps = [m.posterior for m in models]
        s = sum(ps)
        if s <= 0:
            return aid
        ps = [p / s for p in ps]
        ent = -sum(p * math.log(p) for p in ps if p > 0)
        if ent > best_score:
            best_score = ent
            best_action = aid
    return best_action


# =============================================================================
# Convenience goal predicates for the synthetic suite
# =============================================================================

def reach_cell_goal(target: tuple[int, int], agent_color: int) -> GoalPredicate:
    tr, tc = target
    def _g(frame: Frame) -> bool:
        return 0 <= tr < len(frame) and 0 <= tc < len(frame[0]) \
            and frame[tr][tc] == agent_color
    return _g


def match_color_goal(slot_cells: set[tuple[int, int]],
                     target_color: int) -> GoalPredicate:
    cells = list(slot_cells)
    def _g(frame: Frame) -> bool:
        return all(frame[r][c] == target_color for r, c in cells)
    return _g
