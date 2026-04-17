"""
BFS for LS20 Level 7 win condition.

Level 7 mechanics:
- Player starts at (19,15), shape_idx=1, color_idx=0, rot_idx=0
- StartColor=12 → color_idx=0 (tnkekoeuk=[12,9,14,8]; idx 0=12)
- Single target at (29,50): need shape=0, color_idx=3, rot_idx=2
- shape changer (mkjdaccuuf) at (19,40): +1 shape mod 6  (need 5 touches: 1→2→3→4→5→0)
- color changer (soyhouuebz) at  (9,40): +1 color mod 4  (need 3 touches: 0→1→2→3)
- rot   changer (rhsxkxzdjz): SLIDING GATE period 8, undo on blocked moves
    Phase → changer position (verified empirically):
    0:(54,10)  1:(54,15)  2:(54,20)  3:(54,25)
    4:(54,30)  5:(54,25)  6:(54,20)  7:(54,15)
    Rail: (54,5) w=1 h=29, cell=5, start=(54,10)
- 6 resets (npxgalaybz): (30,21),(50,6),(15,46),(40,6),(55,51),(10,6)
- 3 push pads (gbvqrjtaqo):
    kapcaakvb_b at (39,19): delta=(0,+20) [down]
    lujfinsby_t at (34,31): delta=(0,-20) [up]
    tihiodtoj_l at (40,30): delta=(-10,0) [left]
- StepCounter=42, StepsDecrement=2 (None in source → default 2)

Collision rules (same as all LS20 levels):
- Changers/resets: ONE-SIDED bbox (0 <= sx-px < 5 AND 0 <= sy-py < 5)
- Push pads: SYMMETRIC abs<5
- Gate advances only on successful moves (undone when blocked)
- Counter death: new_ctr < 0
- Reset: new_ctr = STEP_COUNTER - STEPS_DEC
- Wrong-target collision: akoadfsur freeze, treat as impassable in BFS
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
# Step 1: pre-solve to level 7 and extract game data
# ---------------------------------------------------------------------------
arc_obj = arc_agi.Arcade()
env     = arc_obj.make("ls20", render_mode=None)
env.reset()
AS      = {a.name: a for a in env.action_space}

print("Pre-solving to level 7...")
for action in _KNOWN_SUBPLANS["ls20"]:
    env.step(AS[action])

game = env._game
px0, py0 = game.gudziatsk.x, game.gudziatsk.y
print(f"Level 7 start: player=({px0},{py0})  shape={game.fwckfzsyc}  "
      f"color={game.hiaauhahz}  rot={game.cklxociuu}")

lv           = game.current_level
STEP_COUNTER = lv.get_data("StepCounter") or 42
STEPS_DEC    = lv.get_data("StepsDecrement") or 2
print(f"StepCounter={STEP_COUNTER}  StepsDecrement={STEPS_DEC}")

# Walls
WALLS: set[tuple[int,int]] = set()
for s in lv.get_sprites_by_tag("ihdgageizm"):
    WALLS.add((s.x, s.y))
print(f"Walls: {len(WALLS)}")

# Targets (single)
TARGETS     = [(t.x, t.y) for t in game.plrpelhym]
GOAL_SHAPES = list(game.ldxlnycps)
GOAL_COLORS = list(game.yjdexjsoa)
GOAL_ROTS   = list(game.ehwheiwsk)
print(f"Targets: {TARGETS}")
print(f"  Goals: shapes={GOAL_SHAPES}  colors={GOAL_COLORS}  rots={GOAL_ROTS}")

# Push pads
PUSH_PADS: list[tuple[int,int,int,int]] = []
for pp in game.hasivfwip:
    wall_cx = pp.start_x + pp.dx
    wall_cy = pp.start_y + pp.dy
    for k in range(1, 20):
        nx_ = wall_cx + pp.dx * pp.width * k
        ny_ = wall_cy + pp.dy * pp.height * k
        if (nx_, ny_) in pp.fjzuynaokm:
            ddx = pp.dx * pp.width * (k - 1)
            ddy = pp.dy * pp.height * (k - 1)
            PUSH_PADS.append((pp.start_x, pp.start_y, ddx, ddy))
            print(f"  Push pad {pp.sprite.name} at ({pp.start_x},{pp.start_y}) "
                  f"-> delta=({ddx},{ddy})")
            break

# Shape/color changers (fixed positions)
SHAPE_CHANGERS: list[tuple[int,int]] = []
COLOR_CHANGERS: list[tuple[int,int]] = []
for tag in ("ttfwljgohq", "mkjdaccuuf"):
    for s in lv.get_sprites_by_tag(tag):
        SHAPE_CHANGERS.append((s.x, s.y))
        print(f"  Shape changer at ({s.x},{s.y})")
for s in lv.get_sprites_by_tag("soyhouuebz"):
    COLOR_CHANGERS.append((s.x, s.y))
    print(f"  Color changer at ({s.x},{s.y})")
SHAPE_CHANGERS = list(set(SHAPE_CHANGERS))
COLOR_CHANGERS = list(set(COLOR_CHANGERS))

# Rot changer gate — measure period with SUCCESSFUL moves
# From (19,15): A3→(14,15) is clear, bounce A3/A4 to advance gate without going anywhere harmful
print("Measuring rot changer gate oscillation (need successful moves)...")
arc2 = arc_agi.Arcade()
env2 = arc2.make("ls20", render_mode=None)
env2.reset()
AS2  = {a.name: a for a in env2.action_space}
for action in _KNOWN_SUBPLANS["ls20"]:
    env2.step(AS2[action])
g2 = env2._game

ROT_GATE_POS: list[tuple[int,int]] = []
for i in range(13):
    ch = g2.wsoslqeku[0]._sprite
    ROT_GATE_POS.append((ch.x, ch.y))
    # Bounce A3/A4 between (19,15) and (14,15) — both clear
    env2.step(AS2["ACTION3"] if i % 2 == 0 else AS2["ACTION4"])

# Find period
period = None
for p in range(1, 13):
    if ROT_GATE_POS[p] == ROT_GATE_POS[0]:
        period = p
        break
assert period is not None, f"Period not found in {ROT_GATE_POS}"
print(f"  Gate positions (phases 0-{period}): {ROT_GATE_POS[:period+1]}")
print(f"  Period = {period}")
GATE_PERIOD = period
ROT_GATE_POS = ROT_GATE_POS[:period]

# Resets
RESETS: list[tuple[int,int]] = []
for s in lv.get_sprites_by_tag("npxgalaybz"):
    RESETS.append((s.x, s.y))
    print(f"  Reset at ({s.x},{s.y})")

N_SHAPES = len(game.ijessuuig)
N_COLORS = len(game.tnkekoeuk)
N_ROTS   = len(game.dhksvilbb)

VALID_X = set(range(4, 64, 5))
VALID_Y = set(range(0, 60, 5))

# GATE_PERIOD and ROT_POS already set above from empirical measurement
ROT_POS = ROT_GATE_POS   # phases 0..(GATE_PERIOD-1)

# ---------------------------------------------------------------------------
# Step 2: BFS helpers
# ---------------------------------------------------------------------------

def _in_bbox(px: int, py: int, sx: int, sy: int) -> bool:
    return 0 <= sx - px < 5 and 0 <= sy - py < 5


def apply_push_chain(nx: int, ny: int) -> tuple[int, int]:
    for _ in range(len(PUSH_PADS) + 1):
        moved = False
        for (sx, sy, ddx, ddy) in PUSH_PADS:
            if abs(nx - sx) < 5 and abs(ny - sy) < 5:
                nx, ny = nx + ddx, ny + ddy
                moved = True
                break
        if not moved:
            break
    return nx, ny


def apply_changers_at(nx: int, ny: int, shape: int, color: int, rot: int,
                      next_phase: int) -> tuple[int, int, int]:
    if any(_in_bbox(nx, ny, sx, sy) for sx, sy in SHAPE_CHANGERS):
        shape = (shape + 1) % N_SHAPES
    if any(_in_bbox(nx, ny, sx, sy) for sx, sy in COLOR_CHANGERS):
        color = (color + 1) % N_COLORS
    rx, ry = ROT_POS[next_phase]
    if _in_bbox(nx, ny, rx, ry):
        rot = (rot + 1) % N_ROTS
    return shape, color, rot


def apply_reset(nx: int, ny: int, r_flags: tuple) -> tuple[tuple, bool]:
    new_flags = list(r_flags)
    collected = False
    for i, (rx, ry) in enumerate(RESETS):
        if not r_flags[i] and _in_bbox(nx, ny, rx, ry):
            new_flags[i] = True
            collected = True
    return tuple(new_flags), collected


def is_target_blocking(nx: int, ny: int, shape: int, color: int, rot: int,
                        thit: int) -> bool:
    for i, (tx, ty) in enumerate(TARGETS):
        if (thit >> i) & 1:
            continue
        if nx == tx and ny == ty:
            if not (shape == GOAL_SHAPES[i] and color == GOAL_COLORS[i] and rot == GOAL_ROTS[i]):
                return True
    return False


def check_target_hit(nx: int, ny: int, shape: int, color: int, rot: int,
                     thit: int) -> int:
    for i, (tx, ty) in enumerate(TARGETS):
        if (thit >> i) & 1:
            continue
        if nx == tx and ny == ty:
            if shape == GOAL_SHAPES[i] and color == GOAL_COLORS[i] and rot == GOAL_ROTS[i]:
                thit |= (1 << i)
    return thit


# ---------------------------------------------------------------------------
# Step 3: BFS
# State: (x, y, shape, color, rot, resets_tuple, ctr, gate_phase, targets_hit)
# ---------------------------------------------------------------------------

def bfs_level7():
    r0    = tuple([False] * len(RESETS))
    start = (px0, py0, game.fwckfzsyc, game.hiaauhahz, game.cklxociuu,
             r0, STEP_COUNTER, 0, 0)
    WIN_MASK = (1 << len(TARGETS)) - 1

    dist         = {start: 0}
    parent       = {start: None}
    action_taken = {start: None}
    queue        = deque([start])
    explored     = 0

    ACTIONS = [("A1", 0, -5), ("A2", 0, +5), ("A3", -5, 0), ("A4", +5, 0)]

    while queue:
        state = queue.popleft()
        x, y, shape, color, rot, resets, ctr, gphase, thit = state
        explored += 1
        if explored % 500_000 == 0:
            print(f"  explored {explored:,}  queue={len(queue):,}  dist={dist[state]}")

        for aname, dx, dy in ACTIONS:
            nx, ny = x + dx, y + dy

            # Wall blocked: gate stays, counter decrements
            if nx not in VALID_X or ny not in VALID_Y or (nx, ny) in WALLS:
                new_ctr_b = ctr - STEPS_DEC
                if new_ctr_b >= 0:
                    ns_b = (x, y, shape, color, rot, resets, new_ctr_b, gphase, thit)
                    if ns_b not in dist:
                        dist[ns_b]         = dist[state] + 1
                        parent[ns_b]       = state
                        action_taken[ns_b] = aname
                        queue.append(ns_b)
                continue

            next_phase = (gphase + 1) % GATE_PERIOD

            # Wrong-target: treat as impassable
            if is_target_blocking(nx, ny, shape, color, rot, thit):
                continue

            # Successful move
            new_resets, got_reset = apply_reset(nx, ny, resets)
            if got_reset:
                new_ctr = STEP_COUNTER - STEPS_DEC
            else:
                new_ctr = ctr - STEPS_DEC
                if new_ctr < 0:
                    continue

            new_shape, new_color, new_rot = apply_changers_at(
                nx, ny, shape, color, rot, next_phase)

            px2, py2 = apply_push_chain(nx, ny)
            if (px2, py2) != (nx, ny):
                nx, ny = px2, py2
                new_resets2, got2 = apply_reset(nx, ny, new_resets)
                if got2:
                    new_resets = new_resets2
                    new_ctr    = STEP_COUNTER - STEPS_DEC
                new_shape, new_color, new_rot = apply_changers_at(
                    nx, ny, new_shape, new_color, new_rot, next_phase)

            new_thit = check_target_hit(nx, ny, new_shape, new_color, new_rot, thit)

            if new_thit == WIN_MASK:
                ns = (nx, ny, new_shape, new_color, new_rot,
                      new_resets, new_ctr, next_phase, new_thit)
                parent[ns]       = state
                action_taken[ns] = aname
                print(f"\nWIN at step {dist[state]+1}!  Explored {explored:,} states.")
                return ns, parent, action_taken

            ns = (nx, ny, new_shape, new_color, new_rot,
                  new_resets, new_ctr, next_phase, new_thit)
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
# Step 4: Run BFS and verify
# ---------------------------------------------------------------------------
print("\nRunning BFS...")
goal, par, act = bfs_level7()

if goal:
    path = reconstruct_path(goal, par, act)
    print(f"Path ({len(path)} steps): {path}")
    print(f"Compressed: {compress(path)}")

    print("\nVerifying against real game engine...")
    env3 = arc_obj.make("ls20", render_mode=None)
    env3.reset()
    AS3  = {a.name: a for a in env3.action_space}
    for action in _KNOWN_SUBPLANS["ls20"]:
        env3.step(AS3[action])
    g3   = env3._game
    AMAP = {"A1": "ACTION1", "A2": "ACTION2", "A3": "ACTION3", "A4": "ACTION4"}

    for i, a in enumerate(path):
        xb, yb = g3.gudziatsk.x, g3.gudziatsk.y
        obs    = env3.step(AS3[AMAP[a]])
        xa, ya = g3.gudziatsk.x, g3.gudziatsk.y
        ctr    = g3._step_counter_ui.current_steps
        lvl    = obs_levels_completed(obs)
        print(f"  Step {i+1:3d} ({a}): ({xb:2d},{yb:2d})->({xa:2d},{ya:2d})"
              f"  sh={g3.fwckfzsyc} col={g3.hiaauhahz} rot={g3.cklxociuu}"
              f"  ctr={ctr:2d}  lv={lvl+1}")
        if lvl >= 7:
            print("  *** GAME WON (all 7 levels) ***")
            break
