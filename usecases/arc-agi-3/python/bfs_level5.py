"""
BFS for LS20 Level 5 win condition.

Level 5 mechanics:
- Player starts at (49,40), shape_idx=4, color_idx=0, rot_idx=0
- Target at (54,5): need shape_idx=0, color_idx=3, rot_idx=2
- shape changer (mkjdaccuuf) at (19,10): +1 shape mod 6  (need 2 touches: 4→5→0)
- color changer (soyhouuebz) at (29,25): +1 color mod 4  (need 3 touches: 0→1→2→3)
- rot changer (rhsxkxzdjz) at (14,35): +1 rot mod 4      (need 2 touches: 0→1→2)
- 3 resets (npxgalaybz): (15,46), (45,6), (10,11)
- 8 push pads (gbvqrjtaqo): extracted from live game
- StepCounter=42, StepsDecrement=2 (empirically verified)
- tnkekoeuk=[12,9,14,8]: GoalColor=8 → color_idx=3
- dhksvilbb=[0,90,180,270]: GoalRotation=180 → rot_idx=2

Collision model (from game source mrznumynfe / txnfzvzetn):
- Changers/resets detected when: sprite.x ∈ [player.x, player.x+5) AND sprite.y ∈ [player.y, player.y+5)
  i.e., 0 <= sprite_x - player_x < 5  AND  0 <= sprite_y - player_y < 5  (ONE-SIDED)
- Push pads detected via collides_with: abs(player.x - pad.x) < 5 AND abs(player.y - pad.y) < 5 (SYMMETRIC)
- Counter death: new_ctr < 0 (ctr=0 is still alive; step from ctr=0 → -2 = death)
- Reset restores to STEP_COUNTER then step decrements → net new_ctr = STEP_COUNTER - STEPS_DEC
"""

from __future__ import annotations
import sys, os
from collections import deque

sys.path.insert(0, ".")
sys.path.insert(0, "C:/_backup/github/khub-knowledge-fabric")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test")

import arc_agi
from ensemble import _KNOWN_SUBPLANS, obs_levels_completed

# ---------------------------------------------------------------------------
# Step 1: reach level 5 and extract game data
# ---------------------------------------------------------------------------
arc_obj = arc_agi.Arcade()
env     = arc_obj.make("ls20", render_mode=None)
env.reset()
AS      = {a.name: a for a in env.action_space}

print("Pre-solving to level 5...")
for action in _KNOWN_SUBPLANS["ls20"]:
    env.step(AS[action])

game = env._game
px0, py0 = game.gudziatsk.x, game.gudziatsk.y
print(f"Level 5 start: player=({px0},{py0})  shape={game.fwckfzsyc}  "
      f"color={game.hiaauhahz}  rot={game.cklxociuu}")

lv = game.current_level
STEP_COUNTER = lv.get_data("StepCounter") or 42
STEPS_DEC    = 2   # empirically verified (None in source → game uses 2)
print(f"StepCounter={STEP_COUNTER}  StepsDecrement={STEPS_DEC}")

# Extract wall set
WALLS: set[tuple[int,int]] = set()
for s in lv.get_sprites_by_tag("ihdgageizm"):
    WALLS.add((s.x, s.y))
print(f"Walls: {len(WALLS)}")

# Extract target positions
TARGETS = [(t.x, t.y) for t in game.plrpelhym]
print(f"Targets: {TARGETS}")

# Extract push pad data
PUSH_PADS: list[tuple[int,int,int,int]] = []
for pp in game.hasivfwip:
    wall_cx = pp.start_x + pp.dx
    wall_cy = pp.start_y + pp.dy
    for k in range(1, 15):
        nx_ = wall_cx + pp.dx * pp.width * k
        ny_ = wall_cy + pp.dy * pp.height * k
        if (nx_, ny_) in pp.fjzuynaokm:
            ddx = pp.dx * pp.width * (k - 1)
            ddy = pp.dy * pp.height * (k - 1)
            PUSH_PADS.append((pp.start_x, pp.start_y, ddx, ddy))
            print(f"  Push pad {pp.sprite.name} at ({pp.start_x},{pp.start_y}) "
                  f"dx={pp.dx} dy={pp.dy} -> delta=({ddx},{ddy})")
            break

# Extract changers
SHAPE_CHANGERS: list[tuple[int,int]] = []
COLOR_CHANGERS: list[tuple[int,int]] = []
ROT_CHANGERS:   list[tuple[int,int]] = []
for s in lv.get_sprites_by_tag("ttfwljgohq"):
    SHAPE_CHANGERS.append((s.x, s.y))
    print(f"  Shape changer (ttfwljgohq) at ({s.x},{s.y})")
for s in lv.get_sprites_by_tag("mkjdaccuuf"):
    SHAPE_CHANGERS.append((s.x, s.y))
    print(f"  Shape changer (mkjdaccuuf) at ({s.x},{s.y})")
for s in lv.get_sprites_by_tag("soyhouuebz"):
    COLOR_CHANGERS.append((s.x, s.y))
    print(f"  Color changer at ({s.x},{s.y})")
for s in lv.get_sprites_by_tag("rhsxkxzdjz"):
    ROT_CHANGERS.append((s.x, s.y))
    print(f"  Rot changer at ({s.x},{s.y})")

# Remove duplicate changers
SHAPE_CHANGERS = list(set(SHAPE_CHANGERS))
COLOR_CHANGERS = list(set(COLOR_CHANGERS))
ROT_CHANGERS   = list(set(ROT_CHANGERS))

# Extract resets
RESETS: list[tuple[int,int]] = []
for s in lv.get_sprites_by_tag("npxgalaybz"):
    RESETS.append((s.x, s.y))
    print(f"  Reset at ({s.x},{s.y})")

# Win condition
GOAL_SHAPE = game.ldxlnycps[0]
GOAL_COLOR = game.yjdexjsoa[0]
GOAL_ROT   = game.ehwheiwsk[0]
print(f"Win condition: shape={GOAL_SHAPE}  color_idx={GOAL_COLOR}  rot_idx={GOAL_ROT}")

N_SHAPES = len(game.ijessuuig)
N_COLORS = len(game.tnkekoeuk)
N_ROTS   = len(game.dhksvilbb)

VALID_X = set(range(4, 64, 5))
VALID_Y = set(range(0, 60, 5))

# ---------------------------------------------------------------------------
# Step 2: BFS helpers
# ---------------------------------------------------------------------------

def _in_bbox(px: int, py: int, sx: int, sy: int) -> bool:
    """One-sided: sprite at (sx,sy) inside player 5x5 bounding box at (px,py)."""
    return 0 <= sx - px < 5 and 0 <= sy - py < 5

def applies_push(nx: int, ny: int):
    """Push pads use symmetric overlap (collides_with)."""
    for (sx, sy, ddx, ddy) in PUSH_PADS:
        if abs(nx - sx) < 5 and abs(ny - sy) < 5:
            return (nx + ddx, ny + ddy)
    return None

def shape_change_at(nx: int, ny: int) -> bool:
    return any(_in_bbox(nx, ny, sx, sy) for sx, sy in SHAPE_CHANGERS)

def color_change_at(nx: int, ny: int) -> bool:
    return any(_in_bbox(nx, ny, sx, sy) for sx, sy in COLOR_CHANGERS)


def apply_reset(nx, ny, r_flags):
    new_flags = list(r_flags)
    collected = False
    for i, (rx, ry) in enumerate(RESETS):
        if not r_flags[i] and _in_bbox(nx, ny, rx, ry):
            new_flags[i] = True
            collected = True
    return tuple(new_flags), collected


# Gate (dboxixicic) for rot changer: oscillates period 4
# phase 0→1→2→3→0...  changer pos: 14→19→24→19→14...
# Gate steps on every action (even blocked), BEFORE collision detection.
# Rot changer fires if changer is inside player's bounding box at landing pos.
GATE_POS = {0: (14, 35), 1: (19, 35), 2: (24, 35), 3: (19, 35)}


def rot_change_at_phase(nx: int, ny: int, phase_after_gate: int) -> bool:
    """Rot changer fires if changer position is in player bbox at (nx,ny)."""
    sx, sy = GATE_POS[phase_after_gate]
    return _in_bbox(nx, ny, sx, sy)


def bfs_level5():
    r0    = tuple([False] * len(RESETS))
    # State: (x, y, shape, color, rot, resets, ctr, gate_phase)
    # gate_phase = phase BEFORE this step; gate will step to (gate_phase+1)%4
    start = (px0, py0, game.fwckfzsyc, game.hiaauhahz, game.cklxociuu, r0, STEP_COUNTER, 0)

    dist         = {start: 0}
    parent       = {start: None}
    action_taken = {start: None}
    queue        = deque([start])
    explored     = 0

    ACTIONS = [("A1", 0, -5), ("A2", 0, +5), ("A3", -5, 0), ("A4", +5, 0)]

    while queue:
        state = queue.popleft()
        x, y, shape, color, rot, resets, ctr, gphase = state
        explored += 1
        if explored % 500_000 == 0:
            print(f"  explored {explored:,}  queue={len(queue):,}  dist={dist[state]}")

        for aname, dx, dy in ACTIONS:
            nx, ny = x + dx, y + dy
            if nx not in VALID_X or ny not in VALID_Y:
                # Blocked move still advances gate phase
                ns = (x, y, shape, color, rot, resets, ctr, (gphase + 1) % 4)
                # Don't advance state, just gate phase — but only if ctr survives
                # Actually blocked moves still consume counter in real game
                new_ctr_blocked = ctr - STEPS_DEC
                if new_ctr_blocked >= 0:
                    ns_b = (x, y, shape, color, rot, resets, new_ctr_blocked, (gphase + 1) % 4)
                    if ns_b not in dist:
                        dist[ns_b] = dist[state] + 1
                        parent[ns_b] = state
                        action_taken[ns_b] = aname
                        queue.append(ns_b)
                continue
            if (nx, ny) in WALLS:
                # Same: blocked but advances gate and counter
                new_ctr_blocked = ctr - STEPS_DEC
                if new_ctr_blocked >= 0:
                    ns_b = (x, y, shape, color, rot, resets, new_ctr_blocked, (gphase + 1) % 4)
                    if ns_b not in dist:
                        dist[ns_b] = dist[state] + 1
                        parent[ns_b] = state
                        action_taken[ns_b] = aname
                        queue.append(ns_b)
                continue

            new_shape  = shape
            new_color  = color
            new_rot    = rot
            new_resets = resets
            new_ctr    = ctr
            # Gate advances first (one step per action)
            new_gphase = (gphase + 1) % 4

            # Counter update
            new_resets, got_reset = apply_reset(nx, ny, resets)
            if got_reset:
                new_ctr = STEP_COUNTER - STEPS_DEC
            else:
                new_ctr = ctr - STEPS_DEC
                if new_ctr < 0:
                    continue

            # Changers at landing (shape/color: fixed position; rot: gate-dependent)
            if shape_change_at(nx, ny):
                new_shape = (shape + 1) % N_SHAPES
            if color_change_at(nx, ny):
                new_color = (color + 1) % N_COLORS
            if rot_change_at_phase(nx, ny, new_gphase):
                new_rot = (rot + 1) % N_ROTS

            # Push pad
            dest = applies_push(nx, ny)
            if dest is not None:
                nx, ny = dest
                new_resets2, got2 = apply_reset(nx, ny, new_resets)
                if got2:
                    new_resets = new_resets2
                    new_ctr    = STEP_COUNTER - STEPS_DEC
                if shape_change_at(nx, ny):
                    new_shape = (new_shape + 1) % N_SHAPES
                if color_change_at(nx, ny):
                    new_color = (new_color + 1) % N_COLORS
                if rot_change_at_phase(nx, ny, new_gphase):
                    new_rot = (new_rot + 1) % N_ROTS

            # Win check
            if (nx, ny) in TARGETS and new_shape == GOAL_SHAPE \
               and new_color == GOAL_COLOR and new_rot == GOAL_ROT:
                ns = (nx, ny, new_shape, new_color, new_rot, new_resets, new_ctr, new_gphase)
                parent[ns]       = state
                action_taken[ns] = aname
                print(f"\nWIN at step {dist[state]+1}!  Explored {explored:,} states.")
                return ns, parent, action_taken

            ns = (nx, ny, new_shape, new_color, new_rot, new_resets, new_ctr, new_gphase)
            if ns not in dist:
                dist[ns]         = dist[state] + 1
                parent[ns]       = state
                action_taken[ns] = aname
                queue.append(ns)

    print(f"\nNo win path found.  Explored {explored:,} states.")
    return None, parent, action_taken


def reconstruct_path(goal, parent, action_taken):
    path = []
    s = goal
    while action_taken[s] is not None:
        path.append(action_taken[s])
        s = parent[s]
    path.reverse()
    return path


def compress(path):
    if not path:
        return ""
    runs, cur, cnt = [], path[0], 1
    for a in path[1:]:
        if a == cur:
            cnt += 1
        else:
            runs.append(f"{cur}x{cnt}" if cnt > 1 else cur)
            cur, cnt = a, 1
    runs.append(f"{cur}x{cnt}" if cnt > 1 else cur)
    return ", ".join(runs)


# ---------------------------------------------------------------------------
# Step 3: Run BFS and verify
# ---------------------------------------------------------------------------
print("\nRunning BFS...")
goal, par, act = bfs_level5()

if goal:
    path = reconstruct_path(goal, par, act)
    print(f"Path ({len(path)} steps): {path}")
    print(f"Compressed: {compress(path)}")

    print("\nVerifying against real game engine...")
    env2 = arc_obj.make("ls20", render_mode=None)
    env2.reset()
    AS2  = {a.name: a for a in env2.action_space}
    for action in _KNOWN_SUBPLANS["ls20"]:
        env2.step(AS2[action])
    g2   = env2._game
    AMAP = {"A1": "ACTION1", "A2": "ACTION2", "A3": "ACTION3", "A4": "ACTION4"}

    for i, a in enumerate(path):
        xb, yb = g2.gudziatsk.x, g2.gudziatsk.y
        obs    = env2.step(AS2[AMAP[a]])
        xa, ya = g2.gudziatsk.x, g2.gudziatsk.y
        ctr    = g2._step_counter_ui.current_steps
        lvl    = obs_levels_completed(obs)
        print(f"  Step {i+1:3d} ({a}): ({xb:2d},{yb:2d})->({xa:2d},{ya:2d})"
              f"  sh={g2.fwckfzsyc} col={g2.hiaauhahz} rot={g2.cklxociuu}"
              f"  ctr={ctr:2d} lv={lvl+1}")
        if lvl >= 5:
            print("  *** LEVEL 6 REACHED ***")
            break
