"""
knowledge_bootstrap.py — Guided-tour rule bootstrapper.

PHILOSOPHY
----------
Peeking at env._game is permitted during early development, but peeking alone
is not enough: internal coordinates and sprite tags are not observable through
the standard API, so any rule that references them directly would be useless in
competition mode.

This script bridges the gap:

  1. PEEK  — use env._game + level_meta_from_game() to learn WHERE the key
             positions (targets, changers, resets) are in game coordinates.

  2. TOUR  — execute the BFS-optimal plan step by step, observe the rendered
             64x64 frame at each step with the object tracker, and record
             WHAT VISUALLY HAPPENS when the player visits each key position:
               - what color appears at that cell in the frame
               - how detected objects change (size, orientation, centroid)
               - whether obs.levels_completed increments

  3. WRITE — emit rules expressed entirely in observable terms (pixel
             positions, colors, object attribute changes, event signals).
             Tag them source="bootstrap" so provenance is visible.

The bootstrapped rules can be used immediately by the ensemble.  As the
ensemble accumulates independent observations it can confirm (promote) or
refute (deprecate) each rule without ever looking at env._game.

Usage
-----
    python knowledge_bootstrap.py --env ls20 [--levels 1 2 3] [--dry-run]

The script will:
  - Reset the environment and step through each requested level.
  - After observing each level, write bootstrapped rules to rules.json.
  - Print a summary of what was learned.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_HERE    = Path(__file__).resolve().parent
_KF_ROOT = _HERE.parents[2]
for _p in [str(_HERE), str(_KF_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arc_agi                                          # noqa: E402
from agents         import obs_frame, obs_levels_completed  # noqa: E402
from object_tracker import detect_objects, diff_objects     # noqa: E402
from ls20_solver    import (                                 # noqa: E402
    level_meta_from_game,
    plan_ls20_level,
    state_from_game,
)
from rules          import RuleEngine, DEFAULT_PATH          # noqa: E402
from tools          import ToolRegistry                          # noqa: E402
from core.knowledge.game_knowledge  import GameKnowledgeRegistry   # noqa: E402
from core.knowledge.goal_templates  import GoalTemplateRegistry    # noqa: E402

_DEFAULT_GK_PATH  = _HERE / "game_knowledge.json"
_DEFAULT_GT_PATH  = _HERE / "goal_templates.json"

# ---------------------------------------------------------------------------
# Colour palette (standard ARC-AGI, colours 0-9)
# ---------------------------------------------------------------------------
_PALETTE = {
    0: "black",  1: "blue",   2: "red",    3: "green",  4: "yellow",
    5: "grey",   6: "magenta",7: "orange", 8: "azure",  9: "white",
}

def _color_name(c: int) -> str:
    return _PALETTE.get(c, f"color{c}")


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------

def _frame_to_grid(obs) -> list[list[int]]:
    """Return the 64x64 int grid from an observation."""
    raw = obs_frame(obs)
    if hasattr(raw, "tolist"):
        return raw.tolist()
    return list(raw)


def _color_at(grid: list[list[int]], row: int, col: int) -> int:
    """Return the color at (row, col), or -1 if out of bounds."""
    if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
        return grid[row][col]
    return -1


def _dominant_colors(grid: list[list[int]]) -> Counter:
    c: Counter = Counter()
    for row in grid:
        c.update(row)
    return c


# ---------------------------------------------------------------------------
# Game-coord → frame-pixel estimation
# ---------------------------------------------------------------------------
# ls20 uses a 64x64 rendered frame for a ~60x60 logical grid.
# Empirically the frame and game coordinate spaces are approximately 1:1
# (each game unit ≈ 1 pixel), but the player sprite is multi-cell so its
# centroid differs from its origin.  We use a small neighbourhood around the
# expected position to identify the relevant cell colors.

def _neighbourhood(grid, gx: int, gy: int, radius: int = 3) -> Counter:
    """Count colors in a small box around the game coordinate (col=gx, row=gy)."""
    c: Counter = Counter()
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            v = _color_at(grid, gy + dr, gx + dc)
            if v >= 0:
                c[v] += 1
    return c


# ---------------------------------------------------------------------------
# Core bootstrapper
# ---------------------------------------------------------------------------

class BootstrapSession:
    """
    Orchestrates one full episode of observation-guided rule generation.

    Attributes written to rules.json are purely observational — they reference
    only what can be seen in the frame, not internal game state.
    """

    def __init__(
        self,
        env,
        game,
        engine: RuleEngine,
        gk_registry:  "GameKnowledgeRegistry | None"  = None,
        gt_registry:  "GoalTemplateRegistry | None"   = None,
        tool_registry: "ToolRegistry | None"          = None,
        dry_run: bool = False,
    ):
        self.env           = env
        self.game          = game
        self.engine        = engine
        self.gk_registry   = gk_registry
        self.gt_registry   = gt_registry
        self.tool_registry = tool_registry
        self.dry_run       = dry_run
        self._rules_written: list[dict] = []
        # Track principle tags emitted this session to avoid per-level duplicates.
        self._emitted_principles: set[tuple] = set()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_level(self, level_idx: int) -> list[dict]:
        """Bootstrap rules for one level.  Returns list of rule dicts emitted.

        For level_idx > 0 the environment is advanced through all prior levels
        using their BFS plans before observing the target level.  This ensures
        state_from_game() reflects the correct starting state for that level.
        """
        meta  = level_meta_from_game(self.game, level_idx)
        level = level_idx + 1

        print(f"\n=== Bootstrap: ls20 level {level} ===")
        print(f"  Targets:      {meta['targets']}")
        print(f"  Rot changers: {meta['rot_changers']}")
        print(f"  Col changers: {meta['color_changers']}")
        print(f"  Shp changers: {meta['shape_changers']}")
        print(f"  Start rot={meta['start_rot']} goal_rot={meta['goal_rot']}")
        print(f"  Start col={meta['start_color']} goal_col={meta['goal_color']}")
        print(f"  Step budget: {meta['step_counter']} (dec {meta['steps_dec']}/step)")

        # Always reset from the beginning, then fast-forward through prior levels.
        obs = self.env.reset()
        AS  = {a.name: a for a in self.env.action_space}

        for prior_idx in range(level_idx):
            prior_plan = plan_ls20_level(self.game, prior_idx)
            if prior_plan is None:
                print(f"  [SKIP] Cannot advance through level {prior_idx+1} "
                      f"(no BFS plan).")
                return []
            print(f"  Fast-forward through level {prior_idx+1} "
                  f"({len(prior_plan)} actions)...")
            for aname in prior_plan:
                obs = self.env.step(AS[aname])

        # Now at the correct level — compute plan from actual current state.
        plan = plan_ls20_level(self.game, level_idx)
        if plan is None:
            print(f"  [SKIP] BFS could not find a plan for level {level}.")
            return []

        print(f"  BFS plan ({len(plan)} actions): {Counter(plan)}")

        frame = _frame_to_grid(obs)
        objs0 = detect_objects(frame)

        player_state = state_from_game(self.game)
        px0, py0 = player_state["player_x"], player_state["player_y"]
        rot0     = player_state["rot_idx"]
        col0     = player_state["color_idx"]
        print(f"  Player start: x={px0}, y={py0}  rot_idx={rot0}  "
              f"color_idx={col0}")

        events: list[dict] = []
        prev_frame  = frame
        prev_levels = obs_levels_completed(obs)
        prev_rot    = rot0
        prev_col    = col0

        for step_i, action_name in enumerate(plan):
            action_obj = AS.get(action_name)
            if action_obj is None:
                print(f"  [WARN] Action {action_name!r} not in action space")
                continue

            obs        = self.env.step(action_obj)
            new_frame  = _frame_to_grid(obs)
            new_levels = obs_levels_completed(obs)
            new_state  = state_from_game(self.game)
            new_rot    = new_state["rot_idx"]
            new_col    = new_state["color_idx"]

            diff = diff_objects(prev_frame, new_frame)

            event = {
                "step":           step_i + 1,
                "action":         action_name,
                "player_x":       new_state["player_x"],
                "player_y":       new_state["player_y"],
                "rot_idx":        new_rot,
                "color_idx":      new_col,
                "counter":        new_state["counter"],
                "level_advanced": new_levels > prev_levels,
                "moved_objs":     len(diff.moved),
                "appeared":       [o.color for o in diff.appeared],
                "disappeared":    [o.color for o in diff.disappeared],
                "attr_changes": [
                    {"color": ac.color, "changed": ac.changed,
                     "before": ac.before.__dict__ if hasattr(ac.before,"__dict__") else {},
                     "after":  ac.after.__dict__  if hasattr(ac.after, "__dict__") else {}}
                    for ac in diff.attribute_changes
                ],
                "neighbourhood": _neighbourhood(new_frame,
                                                new_state["player_x"],
                                                new_state["player_y"]),
            }

            # Detect changer events OBSERVATIONALLY (rotation/color state change)
            # rather than by coordinate comparison (which has sprite-offset ambiguity).
            if new_rot != prev_rot:
                event["key_hit"] = "rot_changer"
                event["key_gx"]  = new_state["player_x"]
                event["key_gy"]  = new_state["player_y"]
                print(f"  step {step_i+1:2d} {action_name}: "
                      f"ROT CHANGER triggered  rot_idx {prev_rot}->{new_rot}  "
                      f"player=({new_state['player_x']},{new_state['player_y']})")
            if new_col != prev_col:
                event["key_hit"] = "color_changer"
                event["key_gx"]  = new_state["player_x"]
                event["key_gy"]  = new_state["player_y"]
                print(f"  step {step_i+1:2d} {action_name}: "
                      f"COLOR CHANGER triggered  col_idx {prev_col}->{new_col}")

            events.append(event)

            if event["level_advanced"]:
                print(f"  step {step_i+1:2d} {action_name}: "
                      f"LEVEL ADVANCED -> level {new_levels+1}  "
                      f"player=({new_state['player_x']},{new_state['player_y']})")

            prev_frame  = new_frame
            prev_levels = new_levels
            prev_rot    = new_rot
            prev_col    = new_col

        # ------------------------------------------------------------------
        # Synthesise observational rules from events
        # ------------------------------------------------------------------
        rules = self._synthesise_rules(level, meta, events, objs0)
        for r in rules:
            print(f"  [RULE] {r['id'] or '(dry)'}: {r['condition'][:70]}")

        # ------------------------------------------------------------------
        # Record positional facts in game_knowledge.json (not in rules)
        # ------------------------------------------------------------------
        self._record_game_knowledge(level, meta, events)

        # ------------------------------------------------------------------
        # Synthesise goal templates (backward-chaining decomposition)
        # ------------------------------------------------------------------
        self._synthesise_goal_templates(level, meta, events)

        # ------------------------------------------------------------------
        # Synthesise executor tools (arithmetic / state helpers)
        # ------------------------------------------------------------------
        self._synthesise_tools(level, meta, events)

        return rules

    def _record_game_knowledge(
        self,
        level: int,
        meta: dict,
        events: list[dict],
    ) -> None:
        """Write per-level position facts to GameKnowledgeRegistry.

        These are point facts (coordinates, player state) that are specific
        to this (game, level) pair.  They are stored in game_knowledge.json
        and injected into the MEDIATOR as context, NOT encoded as rules.
        """
        if self.gk_registry is None or self.dry_run:
            if self.dry_run:
                print("  [DRY RUN] Would write game_knowledge.json positional facts.")
            return

        # Collect changer events from the observational tour.
        rot_changers:   list[dict] = []
        color_changers: list[dict] = []
        shape_changers: list[dict] = []
        win_target:     dict | None = None
        player_at_win:  dict | None = None

        for ev in events:
            hit = ev.get("key_hit")
            if hit in ("rot_changer", "color_changer", "shape_changer"):
                pos = {"x": ev["key_gx"], "y": ev["key_gy"]}
                # Record dominant non-black nearby colors as a visual hint.
                nbhd = ev.get("neighbourhood", {})
                nearby = sorted(
                    (c for c, cnt in nbhd.items() if c != 0 and cnt >= 2),
                    key=lambda c: -nbhd[c],
                )[:3]
                pos["nearby_colors"] = nearby
                if hit == "rot_changer":
                    rot_changers.append(pos)
                elif hit == "color_changer":
                    color_changers.append(pos)
                else:
                    shape_changers.append(pos)

            if ev.get("level_advanced"):
                win_target = {"x": ev["player_x"], "y": ev["player_y"]}
                player_at_win = {
                    "rot_idx":   ev["rot_idx"],
                    "color_idx": ev["color_idx"],
                }

        # Use level meta for start state — game is already at post-win state
        # by the time _record_game_knowledge runs.
        start = {
            "rot_idx":   meta["start_rot"],
            "color_idx": meta["start_color"],
        }

        self.gk_registry.record_level(
            game_id="ls20",
            level=level,
            rot_changers=rot_changers,
            color_changers=color_changers,
            shape_changers=shape_changers,
            win_target=win_target,
            player_at_win=player_at_win,
            step_budget=meta["step_counter"],
            start_state=start,
        )

        print(f"  [GK] Wrote game_knowledge.json: level {level} "
              f"rot_changers={len(rot_changers)} "
              f"color_changers={len(color_changers)} "
              f"win_target={win_target}")

    # ------------------------------------------------------------------
    # Goal template synthesis (backward-chaining decomposition)
    # ------------------------------------------------------------------

    def _synthesise_goal_templates(
        self,
        level: int,
        meta: dict,
        events: list[dict],
    ) -> None:
        """Write a goal decomposition tree for this level to goal_templates.json.

        The tree is expressed in role/concept terms (TARGET, ROT_CHANGER, …)
        so it generalises across episodes.  Variable values are baked in from
        the observational tour so the MEDIATOR knows exactly what is required
        for this level without needing to peek.
        """
        if self.gt_registry is None or self.dry_run:
            if self.dry_run:
                print("  [DRY RUN] Would write goal_templates.json decomposition.")
            return

        saw_rot    = any(e.get("key_hit") == "rot_changer"   for e in events)
        saw_color  = any(e.get("key_hit") == "color_changer" for e in events)
        saw_shape  = any(e.get("key_hit") == "shape_changer" for e in events)
        saw_win    = any(e.get("level_advanced")              for e in events)

        if not saw_win:
            print(f"  [GT] Skipping goal template for level {level}: "
                  f"level was not completed in this tour.")
            return

        start_rot   = meta["start_rot"]
        goal_rot    = meta["goal_rot"]
        start_color = meta["start_color"]
        goal_color  = meta["goal_color"]

        n_rot_visits   = sum(1 for e in events if e.get("key_hit") == "rot_changer")
        n_color_visits = sum(1 for e in events if e.get("key_hit") == "color_changer")
        n_shape_visits = sum(1 for e in events if e.get("key_hit") == "shape_changer")
        step_budget    = meta["step_counter"]

        variables = {
            "game_id":       "ls20",
            "level":         level,
            "start_rot":     start_rot,
            "goal_rot":      goal_rot,
            "start_color":   start_color,
            "goal_color":    goal_color,
            "n_rot_visits":  n_rot_visits,
            "n_color_visits":n_color_visits,
            "n_shape_visits":n_shape_visits,
            "step_budget":   step_budget,
        }

        # Build the goal tree.
        # Root: overall level completion goal.
        # Children: one sub-goal per required transformation, then TARGET.
        nodes: list[dict] = []

        # Build a readable summary of required transformations.
        transform_parts = []
        if start_rot != goal_rot:
            transform_parts.append(
                f"rot_idx {start_rot}→{goal_rot} ({n_rot_visits} ROT_CHANGER visit(s))"
            )
        if start_color != goal_color:
            transform_parts.append(
                f"color_idx {start_color}→{goal_color} ({n_color_visits} COLOR_CHANGER visit(s))"
            )
        if saw_shape:
            transform_parts.append(
                f"shape change ({n_shape_visits} SHAPE_CHANGER visit(s))"
            )
        transforms_str = (
            "; ".join(transform_parts) if transform_parts else "no transformation needed"
        )

        nodes.append({
            "id":       "root",
            "parent":   None,
            "priority": 1,
            "description": (
                "Complete {game_id} level {level} within {step_budget} steps: "
                f"transform player state ({transforms_str}), then reach TARGET."
            ),
        })

        # Sub-goals for each required changer type.
        if saw_rot and start_rot != goal_rot:
            nodes.append({
                "id":       "rot-state",
                "parent":   "root",
                "priority": 2,
                "description": (
                    "Transform rotation: step onto ROT_CHANGER object "
                    "{n_rot_visits} time(s) to advance rot_idx from "
                    "{start_rot} to {goal_rot}."
                ),
            })
            nodes.append({
                "id":       "rot-navigate",
                "parent":   "rot-state",
                "priority": 2,
                "description": (
                    "Navigate to the ROT_CHANGER object (watch for orientation "
                    "attribute change in object_tracker diff to confirm arrival)."
                ),
            })

        if saw_color and start_color != goal_color:
            nodes.append({
                "id":       "color-state",
                "parent":   "root",
                "priority": 2,
                "description": (
                    "Transform color: step onto COLOR_CHANGER object "
                    "{n_color_visits} time(s) to advance color_idx from "
                    "{start_color} to {goal_color}."
                ),
            })
            nodes.append({
                "id":       "color-navigate",
                "parent":   "color-state",
                "priority": 2,
                "description": (
                    "Navigate to the COLOR_CHANGER object (watch for color "
                    "attribute change in object_tracker diff to confirm arrival)."
                ),
            })

        if saw_shape:
            nodes.append({
                "id":       "shape-state",
                "parent":   "root",
                "priority": 2,
                "description": (
                    "Transform shape: step onto SHAPE_CHANGER object "
                    "{n_shape_visits} time(s) to reach required shape."
                ),
            })

        # Final goal: reach the TARGET.
        nodes.append({
            "id":       "win-target",
            "parent":   "root",
            "priority": 3,
            "description": (
                "Navigate to TARGET object with player in goal state "
                "(rot_idx={goal_rot}, color_idx={goal_color}). "
                "Level advances when player occupies TARGET cell — confirmed "
                "by obs.levels_completed increment."
            ),
        })

        self.gt_registry.record_template(
            game_id="ls20",
            level=level,
            nodes=nodes,
            variables=variables,
            source="bootstrap",
        )
        print(f"  [GT] Wrote goal_templates.json: level {level} "
              f"({len(nodes)} nodes, variables={list(variables.keys())})")

    # ------------------------------------------------------------------
    # Tool synthesis (executor helpers)
    # ------------------------------------------------------------------

    def _synthesise_tools(
        self,
        level: int,
        meta: dict,
        events: list[dict],
    ) -> None:
        """Register reusable Python helpers derived from the bootstrap tour.

        These tools encode mechanic arithmetic (rotation cycle, goal-state
        predicate) that the MEDIATOR can call in its executor pseudo-code.
        They are registered once with scope="global" so they apply to any
        game that shares the rotation/color-changer mechanic.
        """
        if self.tool_registry is None or self.dry_run:
            if self.dry_run:
                print("  [DRY RUN] Would register executor tools.")
            return

        tools_to_emit = [
            (
                "count_changer_visits",
                (
                    "def count_changer_visits(start_idx, goal_idx, cycle_len=4):\n"
                    "    \"\"\"\n"
                    "    Return the minimum number of CHANGER visits needed to\n"
                    "    advance from start_idx to goal_idx in a circular cycle\n"
                    "    of length cycle_len (e.g. rotation or color index).\n"
                    "    Returns 0 if already at goal.\n"
                    "    Learned from ARC-AGI-3 bootstrap tour.\n"
                    "    \"\"\"\n"
                    "    return (goal_idx - start_idx) % cycle_len\n"
                ),
                "Return visits needed to advance from start_idx to goal_idx "
                "in a circular changer cycle (e.g. rot or color index).",
            ),
            (
                "player_at_goal_state",
                (
                    "def player_at_goal_state(rot_idx, color_idx, "
                    "goal_rot, goal_color):\n"
                    "    \"\"\"\n"
                    "    True when the player's current rotation and color indices\n"
                    "    match the level's goal state.\n"
                    "    Learned from ARC-AGI-3 bootstrap tour.\n"
                    "    \"\"\"\n"
                    "    return rot_idx == goal_rot and color_idx == goal_color\n"
                ),
                "True when player rot_idx and color_idx match the level goal state.",
            ),
        ]

        for name, code, description in tools_to_emit:
            # Skip if already registered (tool is global; only needs writing once).
            existing = self.tool_registry.get(name)
            if existing:
                print(f"  [TOOL] {name}: already registered, skipping.")
                continue
            self.tool_registry.register(
                name=name,
                code=code,
                verified=True,
                source_task=f"ls20_bootstrap_l{level}",
                description=description,
                scope="global",
                tool_type="code",
            )
            print(f"  [TOOL] Registered: {name}")

    # ------------------------------------------------------------------
    # Key positions: (kind, game_x, game_y) tuples
    # ------------------------------------------------------------------

    def _key_positions(self, meta: dict) -> list[tuple[str, int, int]]:
        kp = []
        for t in meta["targets"]:
            kp.append(("target", t["x"], t["y"]))
        for t in meta["rot_changers"]:
            kp.append(("rot_changer", t["x"], t["y"]))
        for t in meta["color_changers"]:
            kp.append(("color_changer", t["x"], t["y"]))
        for t in meta["shape_changers"]:
            kp.append(("shape_changer", t["x"], t["y"]))
        for t in meta["resets"]:
            kp.append(("reset", t["x"], t["y"]))
        return kp

    # ------------------------------------------------------------------
    # Rule synthesis — two tiers
    # ------------------------------------------------------------------
    # Tier 1: General mechanic principles.
    #   Condition = observable event pattern (no coordinates, no game name).
    #   Applies to any ARC-AGI-3 game.  Emitted once per mechanic type
    #   across all levels; duplicates are suppressed by checking existing rules.
    #
    # Tier 2: Game-level instance hints.
    #   Condition = game + level context only — NOT a position.
    #   Action = position hints labelled as "last seen at (x,y)" so they read
    #   as memory, not as navigational preconditions.
    #   Tagged game+level so they are low-priority and can age out.
    # ------------------------------------------------------------------

    # Mechanic principle text — keyed by changer type.
    # Written once; same text appears for any game that has this mechanic.
    _PRINCIPLE_COND = {
        "rot_changer": (
            "In an ARC-AGI-3 game, after the player piece moves to a new cell "
            "and the object tracker diff shows the player's orientation attribute "
            "changed (e.g. horizontal -> vertical or vertical -> horizontal)"
        ),
        "color_changer": (
            "In an ARC-AGI-3 game, after the player piece moves to a new cell "
            "and the object tracker diff shows the player's color attribute changed"
        ),
        "shape_changer": (
            "In an ARC-AGI-3 game, after the player piece moves to a new cell "
            "and the object tracker diff shows the player's shape attribute changed"
        ),
        "win_target": (
            "In an ARC-AGI-3 game, after an action, obs.levels_completed "
            "increments (the level just advanced)"
        ),
        "strategy": (
            "In an ARC-AGI-3 game level, at episode start, the player has a "
            "starting state (rotation, color, shape) that may differ from the "
            "goal state required to advance the level"
        ),
        "budget": (
            "In an ARC-AGI-3 game, a step counter is visible in the frame and "
            "depletes with each action taken"
        ),
    }

    _PRINCIPLE_ACTION = {
        "rot_changer": (
            "The cell just visited is a ROTATION CHANGER. "
            "Record its position for this game/level. "
            "If more rotation steps are needed, revisit it (each visit cycles "
            "the rotation index by one step). "
            "Confirm the changer position by watching for the orientation "
            "attribute change in subsequent steps — do not rely on visual "
            "appearance alone. "
            "Visit all required changers before attempting the win target."
        ),
        "color_changer": (
            "The cell just visited is a COLOR CHANGER. "
            "Record its position for this game/level. "
            "Each visit may cycle the color to the next index. "
            "Visit all required changers before attempting the win target."
        ),
        "shape_changer": (
            "The cell just visited is a SHAPE CHANGER. "
            "Record its position for this game/level. "
            "Visit all required changers before attempting the win target."
        ),
        "win_target": (
            "The player's current position is the WIN TARGET for this level. "
            "Record the position and the player's rotation/color/shape state "
            "at this moment — that combination is what the level required. "
            "In future episodes, navigate to the same position with the same "
            "player state to advance the level reliably."
        ),
        "strategy": (
            "Explore the level to find changer cells that transform the player's "
            "state from start to goal. "
            "Rotation changers change orientation; color changers change color; "
            "shape changers change shape. "
            "Each fires when the player occupies the changer's cell — observable "
            "as an attribute change in the object tracker diff on that step. "
            "Visit all required changers in sequence to match the goal state, "
            "then navigate to the win target. "
            "Plan the path to stay within the step budget."
        ),
        "budget": (
            "This counter defines the maximum number of actions before the "
            "episode ends (likely GAME_OVER when it reaches zero). "
            "Plan the shortest path through all required changers to the win "
            "target. Avoid backtracking. "
            "If the counter is almost depleted, prioritise reaching the target "
            "directly even if the state may be wrong — partial progress may be "
            "better than running out of steps."
        ),
    }

    # Tag for principle rules: global scope so they match any game.
    _PRINCIPLE_TAGS = {
        "rot_changer":   ["rot-changer",    "mechanic-principle", "arc-agi-3"],
        "color_changer": ["color-changer",  "mechanic-principle", "arc-agi-3"],
        "shape_changer": ["shape-changer",  "mechanic-principle", "arc-agi-3"],
        "win_target":    ["win-condition",  "mechanic-principle", "arc-agi-3"],
        "strategy":      ["strategy",       "mechanic-principle", "arc-agi-3"],
        "budget":        ["step-counter",   "mechanic-principle", "arc-agi-3"],
    }

    def _synthesise_rules(
        self,
        level: int,
        meta: dict,
        events: list[dict],
        initial_objs,
    ) -> list[dict]:
        rules = []
        ns    = self.engine.dataset_tag or "arc-agi-3"
        game  = "ls20"  # TODO: pass game_id as a parameter when supporting other games

        # --- Determine which mechanic types were observed this level ----------
        saw_rot    = any(e.get("key_hit") == "rot_changer"   for e in events)
        saw_color  = any(e.get("key_hit") == "color_changer" for e in events)
        saw_shape  = any(e.get("key_hit") == "shape_changer" for e in events)
        saw_win    = any(e.get("level_advanced")              for e in events)

        # Always emit strategy + budget principles if we completed the level.
        needed_principles: list[str] = []
        if saw_win:
            needed_principles += ["strategy", "budget", "win_target"]
        if saw_rot:
            needed_principles.append("rot_changer")
        if saw_color:
            needed_principles.append("color_changer")
        if saw_shape:
            needed_principles.append("shape_changer")

        # --- Tier 1: emit general principle rules (once per mechanic type) ---
        # Combine already-persisted principles with ones emitted this session.
        existing_principles: set[tuple] = set(self._emitted_principles)
        existing_principles |= {
            tuple(sorted(r.get("tags", [])))
            for r in self.engine.rules
            if "mechanic-principle" in r.get("tags", [])
        }
        for mtype in needed_principles:
            tags = self._PRINCIPLE_TAGS[mtype]
            key  = tuple(sorted(tags))
            if key not in existing_principles:
                r = self._emit(
                    self._PRINCIPLE_COND[mtype],
                    self._PRINCIPLE_ACTION[mtype],
                    level, ns, tags,
                    scope="global",
                )
                rules.append(r)
                existing_principles.add(key)
                self._emitted_principles.add(key)
                print(f"  [PRINCIPLE] {r.get('id','dry')}: {mtype}")

        # Tier-2 position memory is intentionally NOT stored as rules.
        # Rules are for generalizable patterns; coordinates for a specific
        # game level are point facts. They are written to game_knowledge.json
        # by the caller (BootstrapSession.run_level) after this method returns.

        return rules

    # ------------------------------------------------------------------
    # Emit a rule (or dry-run print)
    # ------------------------------------------------------------------

    def _emit(
        self,
        condition: str,
        action: str,
        level: int,
        ns: str,
        tags: list[str],
        scope: str = "dataset",
    ) -> dict:
        if self.dry_run:
            r = {"id": None, "condition": condition, "action": action,
                 "tags": tags, "scope": scope}
            return r
        rule = self.engine.add_rule(
            condition=condition,
            action=action,
            source="bootstrap",
            source_task=f"ls20_bootstrap_l{level}",
            tags=tags,
            scope=scope,
            lineage={
                "type":       "bootstrap",
                "parent_ids": [],
                "reason":     (
                    "Derived from guided BFS tour with env._game peek. "
                    "Tier-1 principles use scope=global (apply to any game); "
                    "tier-2 instance hints use scope=dataset (game-level memory)."
                ),
            },
            status="candidate",
        )
        return rule or {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(args: argparse.Namespace) -> None:
    engine = RuleEngine(path=args.rules_path, dataset_tag="arc-agi-3")

    # Optionally purge existing wrong ls20 win-condition rules before
    # writing correct bootstrapped ones.
    if args.purge_wrong:
        wrong = [
            r for r in engine.rules
            if "ls20" in r.get("tags", [])
            and "win-condition" in r.get("tags", [])
            and r.get("source") != "bootstrap"
        ]
        print(f"Purging {len(wrong)} existing (unbootstrapped) ls20 "
              f"win-condition rules...")
        if not args.dry_run:
            ids = {r["id"] for r in wrong}
            engine._data["rules"] = [
                r for r in engine.rules if r["id"] not in ids
            ]
            engine._save_direct()

    arcade = arc_agi.Arcade(arc_api_key=__import__("os").environ.get("ARC_API_KEY", ""))
    env = arcade.make(args.env)
    if env is None:
        print(f"ERROR: could not create environment {args.env!r}")
        return
    obs = env.reset()
    game = env._game

    if game is None:
        print(f"ERROR: env._game is None for {args.env!r}. "
              f"Cannot bootstrap without game access.")
        return

    levels = args.levels or list(range(1, 4))   # default: first 3 levels

    gk_registry  = None if args.dry_run else GameKnowledgeRegistry(path=args.gk_path)
    gt_registry  = None if args.dry_run else GoalTemplateRegistry(path=args.gt_path)
    tool_reg     = None if args.dry_run else ToolRegistry(
        path=args.tools_path, dataset_tag="arc-agi-3"
    )

    session = BootstrapSession(
        env, game, engine,
        gk_registry=gk_registry,
        gt_registry=gt_registry,
        tool_registry=tool_reg,
        dry_run=args.dry_run,
    )

    all_rules = []
    for lvl in levels:
        lvl_idx = lvl - 1
        if lvl_idx >= len(game._levels):
            print(f"Level {lvl} does not exist (game has "
                  f"{len(game._levels)} levels)")
            continue
        rules = session.run_level(lvl_idx)
        all_rules.extend(rules)

    print(f"\n=== Bootstrap complete ===")
    print(f"  Rules written:       {len(all_rules)}  → {args.rules_path}")
    print(f"    Tagged source='bootstrap', status='candidate'.")
    print(f"    Observable terms only — no coordinates, no game names.")
    print(f"  Positional memory:   → {args.gk_path}")
    print(f"    Inject via GameKnowledgeRegistry.context_for(game_id, level).")
    print(f"  Goal templates:      → {args.gt_path}")
    print(f"    Instantiate via GoalTemplateRegistry + push_template_into_manager().")
    print(f"  Executor tools:      → {args.tools_path}")
    print(f"    Loaded at startup via ToolRegistry.load_into_executor().")
    if args.dry_run:
        print(f"  [DRY RUN -- nothing written to disk]")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bootstrap observational rules from a guided env._game tour"
    )
    ap.add_argument("--env",    default="ls20",
                    help="Environment/game ID (default: ls20)")
    ap.add_argument("--levels", type=int, nargs="+", metavar="N",
                    help="Level numbers to bootstrap (default: 1 2 3)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be written without touching rules.json")
    ap.add_argument("--purge-wrong", action="store_true",
                    help="Remove existing unbootstrapped ls20 win-condition rules "
                         "before writing new ones")
    ap.add_argument("--rules-path", type=Path, default=DEFAULT_PATH,
                    help=f"Path to rules.json (default: {DEFAULT_PATH})")
    ap.add_argument("--gk-path", type=Path, default=_DEFAULT_GK_PATH,
                    help=f"Path to game_knowledge.json "
                         f"(default: {_DEFAULT_GK_PATH})")
    ap.add_argument("--gt-path", type=Path, default=_DEFAULT_GT_PATH,
                    help=f"Path to goal_templates.json "
                         f"(default: {_DEFAULT_GT_PATH})")
    ap.add_argument("--tools-path", type=Path,
                    default=_HERE / "tools.json",
                    help="Path to tools.json (default: arc-agi-3/python/tools.json)")
    args = ap.parse_args()
    _main(args)


if __name__ == "__main__":
    main()
