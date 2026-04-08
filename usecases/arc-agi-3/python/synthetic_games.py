"""
synthetic_games.py — Procedural generator for in-DSL test games.

The point of this file: generate playable mini-games whose hidden mechanics
are sampled from the DSL primitive set, so the symbolic core (hypothesis
tracker, planner, rule library, Proposer integration) can be developed and
benchmarked *without ever touching real ARC-AGI-3 frames*.

If a layer can't solve random in-DSL games, no amount of real-game tuning
will save it. Conversely, a layer that solves a wide diversity of synthetic
games has retired most of the architectural risk before the first real frame
is loaded.

What's covered in v0:
  - Navigation games: agent + walls + goal cell. Each action is a hidden
    Translate(dr,dc) with IF_BLOCKED precondition. Goal is Reach(agent, goal).
  - Toggle puzzles: a single slot whose color cycles through K options.
    One action advances the cycle, others are no-ops or decoys. Goal is
    Match(slot, target_color).

What v0 leaves out (deferred — all expressible in the DSL when needed):
  - Multi-object interactions (push, swap, merge)
  - Counters and progress bars
  - Pixel-mouse / coordinate actions
  - Multi-slot puzzles, reference pairs
  - Sequence goals, equal-count goals

Each game exposes:
  - reset() / step(action_id) -> (frame, won, advanced)
  - render() -> list[list[int]] (the same shape real ARC-AGI-3 frames have)
  - .ground_truth — the hidden mechanics, used by tests to verify discovery
  - .solve() -> list[str]  optimal action sequence using the ground truth
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from proposer_schema import EffectType, GoalPredicateType, PreconditionType


# =============================================================================
# Color conventions for synthetic frames
# =============================================================================

BG    = 0
WALL  = 5
AGENT = 8
GOAL  = 4
SLOT_COLORS = [12, 13, 14, 15]   # cycle for toggle puzzles


# =============================================================================
# Ground truth (hidden mechanics)
# =============================================================================

@dataclass
class HiddenEffect:
    """One per action_id in the ground truth."""
    effect_type:  EffectType
    precondition: PreconditionType = PreconditionType.ALWAYS
    # Free-form parameters; the meaning depends on effect_type.
    # For TRANSLATE: {"dr": int, "dc": int}
    # For TOGGLE:    {"step": int}  (cycle increment)
    # For NO_OP:     {}
    params:       dict = field(default_factory=dict)


@dataclass
class GroundTruth:
    action_models: dict[str, HiddenEffect]
    goal_predicate: GoalPredicateType
    goal_params:    dict          # e.g. {"target_color": 14}
    description:    str = ""


# =============================================================================
# Game base class
# =============================================================================

class SyntheticGame:
    """Common interface. Subclasses implement step() and render()."""

    grid_h: int
    grid_w: int
    available_actions: list[str]
    ground_truth: GroundTruth

    def reset(self) -> list[list[int]]:
        raise NotImplementedError

    def step(self, action_id: str) -> tuple[list[list[int]], bool, bool]:
        """Return (new_frame, won, level_advanced).

        For v0, won == level_advanced (single-level games).
        """
        raise NotImplementedError

    def render(self) -> list[list[int]]:
        raise NotImplementedError

    @property
    def won(self) -> bool:
        raise NotImplementedError

    def solve(self, max_depth: int = 64) -> list[str] | None:
        """Optimal solver using the ground truth, for test verification.
        Returns the action-id sequence or None if unsolvable within budget.
        """
        raise NotImplementedError


# =============================================================================
# NavigationGame
# =============================================================================

class NavigationGame(SyntheticGame):
    """Agent + walls + goal cell. Hidden translate-per-action, IF_BLOCKED."""

    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        walls: set[tuple[int, int]],
        agent_start: tuple[int, int],
        goal_pos: tuple[int, int],
        ground_truth: GroundTruth,
    ) -> None:
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.walls = set(walls)
        self.agent_start = agent_start
        self.goal_pos = goal_pos
        self.ground_truth = ground_truth
        self.available_actions = list(ground_truth.action_models.keys())
        self.agent_pos = agent_start
        self._won = False

    def reset(self) -> list[list[int]]:
        self.agent_pos = self.agent_start
        self._won = False
        return self.render()

    @property
    def won(self) -> bool:
        return self._won

    def render(self) -> list[list[int]]:
        frame = [[BG] * self.grid_w for _ in range(self.grid_h)]
        for r, c in self.walls:
            frame[r][c] = WALL
        gr, gc = self.goal_pos
        frame[gr][gc] = GOAL
        ar, ac = self.agent_pos
        frame[ar][ac] = AGENT
        return frame

    def step(self, action_id: str) -> tuple[list[list[int]], bool, bool]:
        if self._won:
            return self.render(), True, False
        eff = self.ground_truth.action_models.get(action_id)
        advanced = False
        if eff and eff.effect_type == EffectType.TRANSLATE:
            dr, dc = eff.params.get("dr", 0), eff.params.get("dc", 0)
            nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            blocked = (
                nr < 0 or nr >= self.grid_h
                or nc < 0 or nc >= self.grid_w
                or (nr, nc) in self.walls
            )
            if not blocked:
                self.agent_pos = (nr, nc)
                if self.agent_pos == self.goal_pos:
                    self._won = True
                    advanced = True
        # NO_OP and unknown effects do nothing.
        return self.render(), self._won, advanced

    def solve(self, max_depth: int = 64) -> list[str] | None:
        """BFS over hidden translates. Used by tests, never by the agent."""
        translates: list[tuple[str, int, int]] = []
        for aid, eff in self.ground_truth.action_models.items():
            if eff.effect_type == EffectType.TRANSLATE:
                translates.append((aid, eff.params["dr"], eff.params["dc"]))
        start = self.agent_start
        if start == self.goal_pos:
            return []
        seen = {start}
        q: deque[tuple[tuple[int, int], list[str]]] = deque([(start, [])])
        while q:
            pos, path = q.popleft()
            if len(path) >= max_depth:
                continue
            for aid, dr, dc in translates:
                nr, nc = pos[0] + dr, pos[1] + dc
                if (nr < 0 or nr >= self.grid_h or nc < 0 or nc >= self.grid_w
                        or (nr, nc) in self.walls):
                    continue
                if (nr, nc) in seen:
                    continue
                if (nr, nc) == self.goal_pos:
                    return path + [aid]
                seen.add((nr, nc))
                q.append(((nr, nc), path + [aid]))
        return None


# =============================================================================
# TogglePuzzle
# =============================================================================

class TogglePuzzle(SyntheticGame):
    """Single slot whose color cycles through K options.

    One hidden action advances the cycle; the others are NO_OP decoys.
    Goal: match slot color to a fixed target color.
    """

    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        slot_pos: tuple[int, int],
        slot_radius: int,
        cycle: list[int],
        start_index: int,
        target_index: int,
        ground_truth: GroundTruth,
    ) -> None:
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.slot_pos = slot_pos
        self.slot_radius = slot_radius
        self.cycle = cycle
        self.start_index = start_index
        self.target_index = target_index
        self.ground_truth = ground_truth
        self.available_actions = list(ground_truth.action_models.keys())
        self.index = start_index
        self._won = False

    def reset(self) -> list[list[int]]:
        self.index = self.start_index
        self._won = False
        return self.render()

    @property
    def won(self) -> bool:
        return self._won

    def render(self) -> list[list[int]]:
        frame = [[BG] * self.grid_w for _ in range(self.grid_h)]
        # Show the target as a small reference square in the corner.
        for r in range(self.slot_radius):
            for c in range(self.slot_radius):
                frame[r][c] = self.cycle[self.target_index]
        # Show the slot in the center.
        cr, cc = self.slot_pos
        for r in range(cr - self.slot_radius + 1, cr + self.slot_radius):
            for c in range(cc - self.slot_radius + 1, cc + self.slot_radius):
                if 0 <= r < self.grid_h and 0 <= c < self.grid_w:
                    frame[r][c] = self.cycle[self.index]
        return frame

    def step(self, action_id: str) -> tuple[list[list[int]], bool, bool]:
        if self._won:
            return self.render(), True, False
        eff = self.ground_truth.action_models.get(action_id)
        advanced = False
        if eff and eff.effect_type == EffectType.TOGGLE:
            step = eff.params.get("step", 1)
            self.index = (self.index + step) % len(self.cycle)
            if self.index == self.target_index:
                self._won = True
                advanced = True
        return self.render(), self._won, advanced

    def solve(self, max_depth: int = 64) -> list[str] | None:
        # Find the toggle action and apply it the right number of times.
        toggle_aid = None
        step = 1
        for aid, eff in self.ground_truth.action_models.items():
            if eff.effect_type == EffectType.TOGGLE:
                toggle_aid = aid
                step = eff.params.get("step", 1)
                break
        if toggle_aid is None:
            return None
        n = len(self.cycle)
        diff = (self.target_index - self.start_index) % n
        # How many applications of step gets us there?
        for k in range(n + 1):
            if (k * step) % n == diff:
                if k > max_depth:
                    return None
                return [toggle_aid] * k
        return None


# =============================================================================
# Generators
# =============================================================================

def _action_ids(n: int) -> list[str]:
    return [f"ACTION{i+1}" for i in range(n)]


def generate_navigation_game(
    seed: int,
    grid_h: int = 10,
    grid_w: int = 10,
    n_walls: int = 8,
    n_actions: int = 4,
) -> NavigationGame:
    rng = random.Random(seed)
    actions = _action_ids(n_actions)

    # Hidden translates: pick distinct (dr,dc) per action from the cardinal set,
    # padded with NO_OP if n_actions > 4.
    candidate_dirs = [(0, 1), (0, -1), (1, 0), (-1, 0),
                      (1, 1), (-1, -1), (1, -1), (-1, 1)]
    rng.shuffle(candidate_dirs)
    action_models: dict[str, HiddenEffect] = {}
    for i, aid in enumerate(actions):
        if i < len(candidate_dirs):
            dr, dc = candidate_dirs[i]
            action_models[aid] = HiddenEffect(
                effect_type=EffectType.TRANSLATE,
                precondition=PreconditionType.IF_BLOCKED,
                params={"dr": dr, "dc": dc},
            )
        else:
            action_models[aid] = HiddenEffect(effect_type=EffectType.NO_OP)

    # Place walls, agent, goal — retry until solvable.
    for _ in range(50):
        walls: set[tuple[int, int]] = set()
        while len(walls) < n_walls:
            walls.add((rng.randrange(grid_h), rng.randrange(grid_w)))
        free = [(r, c) for r in range(grid_h) for c in range(grid_w)
                if (r, c) not in walls]
        if len(free) < 2:
            continue
        rng.shuffle(free)
        agent_start = free[0]
        goal_pos = free[1]
        gt = GroundTruth(
            action_models=action_models,
            goal_predicate=GoalPredicateType.REACH,
            goal_params={"target_pos": goal_pos},
            description=f"Navigate agent at {agent_start} to goal at {goal_pos}",
        )
        game = NavigationGame(grid_h, grid_w, walls, agent_start, goal_pos, gt)
        if game.solve(max_depth=64) is not None:
            return game
    raise RuntimeError(f"Failed to generate solvable navigation game (seed={seed})")


def generate_toggle_puzzle(
    seed: int,
    grid_h: int = 10,
    grid_w: int = 10,
    n_actions: int = 3,
    cycle_len: int = 3,
) -> TogglePuzzle:
    rng = random.Random(seed)
    actions = _action_ids(n_actions)

    # Pick which action is the real toggle; others are NO_OPs.
    toggle_idx = rng.randrange(n_actions)
    step = rng.choice([1, 2])
    cycle = SLOT_COLORS[:cycle_len]
    action_models: dict[str, HiddenEffect] = {}
    for i, aid in enumerate(actions):
        if i == toggle_idx:
            action_models[aid] = HiddenEffect(
                effect_type=EffectType.TOGGLE,
                params={"step": step},
            )
        else:
            action_models[aid] = HiddenEffect(effect_type=EffectType.NO_OP)

    start_index = 0
    target_index = rng.randrange(1, cycle_len)   # never trivially won
    gt = GroundTruth(
        action_models=action_models,
        goal_predicate=GoalPredicateType.MATCH,
        goal_params={"target_color": cycle[target_index]},
        description=f"Toggle slot to color {cycle[target_index]}",
    )
    return TogglePuzzle(
        grid_h=grid_h, grid_w=grid_w,
        slot_pos=(grid_h // 2, grid_w // 2),
        slot_radius=2,
        cycle=cycle,
        start_index=start_index,
        target_index=target_index,
        ground_truth=gt,
    )


# =============================================================================
# Game catalog — used by tests and (later) batch evaluation
# =============================================================================

GAME_GENERATORS: dict[str, Callable[[int], SyntheticGame]] = {
    "navigation": generate_navigation_game,
    "toggle":     generate_toggle_puzzle,
}


def generate_batch(family: str, n: int, base_seed: int = 0
                   ) -> list[SyntheticGame]:
    """Generate n distinct games from a family for batch evaluation."""
    gen = GAME_GENERATORS[family]
    return [gen(base_seed + i) for i in range(n)]
