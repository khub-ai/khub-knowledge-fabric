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

    def __init__(self, env, game, engine: RuleEngine, dry_run: bool = False):
        self.env      = env
        self.game     = game
        self.engine   = engine
        self.dry_run  = dry_run
        self._rules_written: list[dict] = []

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

        return rules

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
    # Rule synthesis
    # ------------------------------------------------------------------

    def _synthesise_rules(
        self,
        level: int,
        meta: dict,
        events: list[dict],
        initial_objs,
    ) -> list[dict]:
        rules = []
        ns    = self.engine.dataset_tag or "arc-agi-3"

        # --- Rule: rotation changer observation ----------------------------
        rot_events = [e for e in events if e.get("key_hit") == "rot_changer"]
        if rot_events:
            re = rot_events[0]
            gx, gy   = re["key_gx"], re["key_gy"]
            nb       = re["neighbourhood"]
            dominant = [_color_name(c) for c, _ in nb.most_common(3) if c != 3]  # skip green bg
            rot_before = None
            for e in events:
                if e["step"] < re["step"]:
                    rot_before = e["rot_idx"]
            rot_after = re["rot_idx"]
            attr_desc = ""
            if re["attr_changes"]:
                ac = re["attr_changes"][0]
                before_orient = ac["before"].get("orientation", "?")
                after_orient  = ac["after"].get("orientation",  "?")
                attr_desc = (f"; player orientation changed from "
                             f"{before_orient} to {after_orient}")

            cond = (
                f"In ls20 level {level}, the player piece approaches a visually "
                f"distinct cell at approximately game-coord (col {gx}, row {gy}) "
                f"[frame colours nearby: {', '.join(dominant[:2])}]"
            )
            action = (
                f"This cell is a ROTATION CHANGER — visit it before the target. "
                f"After stepping on it the player rotation index changes from "
                f"{rot_before} to {rot_after}{attr_desc}. "
                f"[OBSERVABLE: watch for player orientation attribute change in "
                f"the object tracker diff when this cell is visited]"
            )
            rules.append(self._emit(cond, action, level, ns,
                                    ["rot-changer", "mechanic-discovery",
                                     f"ls20", f"level-{level}"]))

        # --- Rule: target observation + win condition ----------------------
        target_events = [e for e in events if e.get("key_hit") == "target"]
        level_advance = [e for e in events if e.get("level_advanced")]
        if level_advance:
            adv = level_advance[0]
            gx = adv.get("key_gx") or (meta["targets"][0]["x"] if meta["targets"] else "?")
            gy = adv.get("key_gy") or (meta["targets"][0]["y"] if meta["targets"] else "?")
            nb       = adv["neighbourhood"]
            dominant = [_color_name(c) for c, _ in nb.most_common(3) if c != 3]

            cond = (
                f"In ls20 level {level}, the player piece is positioned at "
                f"approximately game-coord (col {gx}, row {gy}) with the correct "
                f"rotation/color/shape state (rot_idx={adv['rot_idx']}, "
                f"color_idx={adv['color_idx']})"
            )
            action = (
                f"The level advances — this is the WIN TARGET for level {level}. "
                f"[OBSERVABLE: obs.levels_completed increments; "
                f"frame colours near target: {', '.join(dominant[:2])}; "
                f"visit rot_changer(s) first if start_rot != goal_rot]"
            )
            rules.append(self._emit(cond, action, level, ns,
                                    ["win-condition", "target",
                                     f"ls20", f"level-{level}"]))

        # --- Rule: full navigation strategy --------------------------------
        plan_step_types = Counter()
        seq_parts = []
        changers_visited = []
        for e in events:
            kh = e.get("key_hit")
            if kh:
                changers_visited.append(
                    f"visit {kh} at ({e['key_gx']},{e['key_gy']})"
                )
            plan_step_types[e["action"]] += 1

        changer_str = (
            f"First visit: {'; then '.join(changers_visited)}. "
            if changers_visited else ""
        )
        action_summary = ", ".join(
            f"{cnt}x{a[-1]}" for a, cnt in
            sorted(plan_step_types.items(), key=lambda x: -x[1])
        )
        budget = meta["step_counter"]
        dec    = meta["steps_dec"]

        cond = (
            f"In ls20 level {level}, the episode starts and the goal is to "
            f"advance to the next level (step budget: {budget}, "
            f"decrements {dec}/step = {budget//dec} moves max)"
        )
        action = (
            f"{changer_str}"
            f"Optimal action sequence uses {len(events)} steps "
            f"({action_summary}). "
            f"[BOOTSTRAPPED from BFS — confirm through observation: "
            f"orientation change at changer, levels_completed increment at target]"
        )
        rules.append(self._emit(cond, action, level, ns,
                                ["strategy", "navigation",
                                 f"ls20", f"level-{level}"]))

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
    ) -> dict:
        if self.dry_run:
            r = {"id": None, "condition": condition, "action": action, "tags": tags}
            return r
        rule = self.engine.add_rule(
            condition=condition,
            action=action,
            source="bootstrap",
            source_task=f"ls20_bootstrap_l{level}",
            tags=tags,
            lineage={
                "type":       "bootstrap",
                "parent_ids": [],
                "reason":     (
                    "Derived from guided BFS tour with env._game peek. "
                    "Expressed in observable terms; confirm through independent "
                    "observation to promote to active."
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

    session = BootstrapSession(env, game, engine, dry_run=args.dry_run)

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
    print(f"  Rules written: {len(all_rules)}")
    print(f"  Rules are tagged source='bootstrap' and status='candidate'.")
    print(f"  They are expressed in observable terms and can be confirmed")
    print(f"  independently by the OBSERVER without peeking at env._game.")
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
    args = ap.parse_args()
    _main(args)


if __name__ == "__main__":
    main()
