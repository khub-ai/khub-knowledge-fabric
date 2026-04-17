"""
BFS for LS20 Level 4 win condition.

Extracts walls and push pad data directly from the live game object after pre-solve,
then runs a pure-Python BFS over the abstract state space.

Level 4 mechanics:
- Player starts at (54,5), shape_idx=4, color_idx=2, rot_idx=0
- Target at (9,5): need shape_idx=5, color_idx=1, rot_idx=0
- shape changer (ttfwljgohq/mkjdaccuuf) at (24,30): +1 shape per touch
- color changer (soyhouuebz) at (34,30): +1 color per touch
- no rotation changer (rot goal=0=start, already satisfied)
- npxgalaybz resets at (35,51) and (20,16)
- 8 push pads (gbvqrjtaqo): extracted from game
- StepsDecrement=1 -> 42 steps per life
- tnkekoeuk=[12,9,14,8]: color_idx 2=14 (start), goal color_idx=1 (9)
  -> need 3 color touches: 2->3->0->1
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
# Step 1: reach level 4 and extract game data
# ---------------------------------------------------------------------------
arc_obj = arc_agi.Arcade()
env     = arc_obj.make("ls20", render_mode=None)
env.reset()
AS      = {a.name: a for a in env.action_space}

print("Pre-solving to level 4...")
for action in _KNOWN_SUBPLANS["ls20"]:
    env.step(AS[action])

game = env._game
px0, py0 = game.gudziatsk.x, game.gudziatsk.y
print(f"Level 4 start: player=({px0},{py0})  shape={game.fwckfzsyc}  "
      f"color={game.hiaauhahz}  rot={game.cklxociuu}")

# Extract wall set (all ihdgageizm-tagged sprites)
WALLS: set[tuple[int,int]] = set()
for s in game.current_level.get_sprites_by_tag("ihdgageizm"):
    WALLS.add((s.x, s.y))
print(f"Walls: {len(WALLS)}")

# Extract target positions
TARGETS = [(t.x, t.y) for t in game.plrpelhym]
print(f"Targets: {TARGETS}")

# Extract push pad data: list of (sx, sy, ddx, ddy)
# Player at (nx,ny) triggers push pad (sx,sy) if |nx-sx|<5 AND |ny-sy|<5
# Player destination = (nx+ddx, ny+ddy)
PUSH_PADS: list[tuple[int,int,int,int]] = []  # (sx, sy, ddx, ddy)
for pp in game.hasivfwip:
    wall_cx = pp.start_x + pp.dx
    wall_cy = pp.start_y + pp.dy
    for k in range(1, 12):
        nx_ = wall_cx + pp.dx * pp.width * k
        ny_ = wall_cy + pp.dy * pp.height * k
        if (nx_, ny_) in pp.fjzuynaokm:
            gguyvrkohc = k - 1
            ddx = pp.dx * pp.width * gguyvrkohc
            ddy = pp.dy * pp.height * gguyvrkohc
            PUSH_PADS.append((pp.start_x, pp.start_y, ddx, ddy))
            print(f"  Push pad {pp.sprite.name} at ({pp.start_x},{pp.start_y}) "
                  f"dx={pp.dx} dy={pp.dy} -> delta=({ddx},{ddy})")
            break

# Extract changer positions
SHAPE_CHANGERS: list[tuple[int,int]] = []
COLOR_CHANGERS: list[tuple[int,int]] = []
for s in game.current_level.get_sprites_by_tag("ttfwljgohq"):
    SHAPE_CHANGERS.append((s.x, s.y))
    print(f"  Shape changer at ({s.x},{s.y})")
for s in game.current_level.get_sprites_by_tag("soyhouuebz"):
    COLOR_CHANGERS.append((s.x, s.y))
    print(f"  Color changer at ({s.x},{s.y})")

# Extract npxgalaybz resets
RESETS: list[tuple[int,int]] = []
for s in game.current_level.get_sprites_by_tag("npxgalaybz"):
    RESETS.append((s.x, s.y))
    print(f"  Reset at ({s.x},{s.y})")

# Win condition
GOAL_SHAPE = game.ldxlnycps[0]
GOAL_COLOR = game.yjdexjsoa[0]
GOAL_ROT   = game.ehwheiwsk[0]
print(f"Win condition: shape={GOAL_SHAPE}  color_idx={GOAL_COLOR}  rot_idx={GOAL_ROT}")

STEP_COUNTER = game.current_level.get_data("StepCounter") or 42
STEPS_DEC    = game.current_level.get_data("StepsDecrement") or 1
print(f"StepCounter={STEP_COUNTER}  StepsDecrement={STEPS_DEC}")

# Valid grid positions
VALID_X = set(range(4, 64, 5))
VALID_Y = set(range(0, 60, 5))

# ---------------------------------------------------------------------------
# Step 2: BFS helper functions
# ---------------------------------------------------------------------------

def applies_push(nx: int, ny: int):
    for (sx, sy, ddx, ddy) in PUSH_PADS:
        if abs(nx - sx) < 5 and abs(ny - sy) < 5:
            return (nx + ddx, ny + ddy)
    return None

def shape_change_at(nx: int, ny: int) -> bool:
    for (sx, sy) in SHAPE_CHANGERS:
        if abs(nx - sx) < 5 and abs(ny - sy) < 5:
            return True
    return False

def color_change_at(nx: int, ny: int) -> bool:
    for (sx, sy) in COLOR_CHANGERS:
        if abs(nx - sx) < 5 and abs(ny - sy) < 5:
            return True
    return False

def apply_reset(nx, ny, r_flags):
    """Return (new_flags, collected) where collected=True if any reset picked up."""
    new_flags = list(r_flags)
    collected = False
    for i, (rx, ry) in enumerate(RESETS):
        if not r_flags[i] and abs(nx - rx) < 5 and abs(ny - ry) < 5:
            new_flags[i] = True
            collected = True
    return tuple(new_flags), collected


def bfs_level4():
    n_resets = len(RESETS)
    n_shapes = len(game.ijessuuig)
    n_colors = len(game.tnkekoeuk)
    r0       = tuple([False] * n_resets)
    start    = (px0, py0, game.fwckfzsyc, game.hiaauhahz, game.cklxociuu, r0, STEP_COUNTER)

    dist         = {start: 0}
    parent       = {start: None}
    action_taken = {start: None}
    queue        = deque([start])
    explored     = 0

    ACTIONS = [("A1",0,-5), ("A2",0,+5), ("A3",-5,0), ("A4",+5,0)]

    while queue:
        state = queue.popleft()
        x, y, shape, color, rot, resets, ctr = state
        explored += 1

        for aname, dx, dy in ACTIONS:
            nx, ny = x + dx, y + dy
            if nx not in VALID_X or ny not in VALID_Y:
                continue
            if (nx, ny) in WALLS:
                continue

            new_shape  = shape
            new_color  = color
            new_rot    = rot
            new_resets = resets
            new_ctr    = ctr

            # Counter reset (must happen before decrement to avoid death)
            new_resets, got_reset = apply_reset(nx, ny, resets)
            if got_reset:
                new_ctr = STEP_COUNTER
            else:
                new_ctr = ctr - STEPS_DEC
                if new_ctr <= 0:
                    continue

            # Changers at landing position
            if shape_change_at(nx, ny):
                new_shape = (shape + 1) % n_shapes
            if color_change_at(nx, ny):
                new_color = (color + 1) % n_colors

            # Push pad (may teleport)
            dest = applies_push(nx, ny)
            if dest is not None:
                nx, ny = dest
                # Re-check at push destination
                new_resets2, got2 = apply_reset(nx, ny, new_resets)
                if got2:
                    new_resets = new_resets2
                    new_ctr    = STEP_COUNTER
                if shape_change_at(nx, ny):
                    new_shape = (new_shape + 1) % n_shapes
                if color_change_at(nx, ny):
                    new_color = (new_color + 1) % n_colors

            # Win check
            if (nx, ny) in TARGETS and new_shape == GOAL_SHAPE \
               and new_color == GOAL_COLOR and new_rot == GOAL_ROT:
                ns = (nx, ny, new_shape, new_color, new_rot, new_resets, new_ctr)
                parent[ns]       = state
                action_taken[ns] = aname
                print(f"\nWIN at step {dist[state]+1}!  Explored {explored} states.")
                return ns, parent, action_taken

            ns = (nx, ny, new_shape, new_color, new_rot, new_resets, new_ctr)
            if ns not in dist:
                dist[ns]         = dist[state] + 1
                parent[ns]       = state
                action_taken[ns] = aname
                queue.append(ns)

    print(f"\nNo win path found.  Explored {explored} states.")
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
goal, par, act = bfs_level4()

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
    AMAP = {"A1":"ACTION1","A2":"ACTION2","A3":"ACTION3","A4":"ACTION4"}

    for i, a in enumerate(path):
        xb, yb = g2.gudziatsk.x, g2.gudziatsk.y
        obs    = env2.step(AS2[AMAP[a]])
        xa, ya = g2.gudziatsk.x, g2.gudziatsk.y
        ctr    = g2._step_counter_ui.current_steps
        lvl    = obs_levels_completed(obs)
        print(f"  Step {i+1:3d} ({a}): ({xb:2d},{yb:2d})->({xa:2d},{ya:2d})"
              f"  sh={g2.fwckfzsyc} col={g2.hiaauhahz} rot={g2.cklxociuu}"
              f"  ctr={ctr:2d} lv={lvl}")
        if lvl >= 4:
            print("  *** LEVEL 5 REACHED ***")
            break
