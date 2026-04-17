"""
BFS for LS20 Level 6 win condition.

Level 6 mechanics:
- Player starts at (24,50), shape_idx=0, color_idx=2, rot_idx=0
- StartColor=14 → color_idx=2 (tnkekoeuk=[12,9,14,8]; idx 0=12,1=9,2=14,3=8)
- TWO targets (must hit both in any order):
    Target 0 at (54,50): need shape=5, color_idx=1, rot_idx=1
    Target 1 at (54,35): need shape=0, color_idx=3, rot_idx=2
- THREE sliding gates (dboxixicic), period 8 (advance only on successful moves):
    rot   changer (rhsxkxzdjz): phase → position
        0:(34,40)  1:(29,40)  2:(24,40)  3:(19,40)
        4:(14,40)  5:(19,40)  6:(24,40)  7:(29,40)
    shape changer (mkjdaccuuf/ttfwljgohq): phase → position
        0:(14,10)  1:(19,10)  2:(24,10)  3:(29,10)
        4:(34,10)  5:(29,10)  6:(24,10)  7:(19,10)
    color changer (soyhouuebz): phase → position
        0:(24,30)  1:(19,30)  2:(19,25)  3:(19,20)
        4:(24,20)  5:(29,20)  6:(29,25)  7:(29,30)
  A changer fires when its position == player landing position
  (both on 5-pixel grid, so one-sided bbox is equivalent to exact match).
- 3 resets (npxgalaybz): (40,6), (10,46), (10,6)
- 2 push pads (gbvqrjtaqo):
    kapcaakvb_b at (49,4): delta=(0,20)  [down, player_y + 20]
    tihiodtoj_l at (50,20): delta=(-10,0) [left, player_x - 10]
  Chained: landing on pad1 at y≈4 → (49,ny+20); if that triggers pad2 → (-10,0)
- StepCounter=42, StepsDecrement=1
- Win: both targets consumed (targets_hit == 0b11)

Collision rules:
- Changers/resets: ONE-SIDED bbox (= exact grid match for grid-aligned sprites)
- Push pads: SYMMETRIC abs<5 (= symmetric; player at (49,0..8) triggers pad1)
- Wall blocked: gate undone, counter DECREMENTS
- Wrong-target blocked: gate undone, counter does NOT decrement (akoadfsur freeze, 5 wasted
  env-steps but never optimal → treat as impassable in BFS)
- Gate advances only when player successfully moves
- Counter death: new_ctr < 0
- Reset: new_ctr = STEP_COUNTER - STEPS_DEC
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
# Step 1: pre-solve to level 6 and extract game data
# ---------------------------------------------------------------------------
arc_obj = arc_agi.Arcade()
env     = arc_obj.make("ls20", render_mode=None)
env.reset()
AS      = {a.name: a for a in env.action_space}

print("Pre-solving to level 6...")
for action in _KNOWN_SUBPLANS["ls20"]:
    env.step(AS[action])

game = env._game
px0, py0 = game.gudziatsk.x, game.gudziatsk.y
print(f"Level 6 start: player=({px0},{py0})  shape={game.fwckfzsyc}  "
      f"color={game.hiaauhahz}  rot={game.cklxociuu}")

lv           = game.current_level
STEP_COUNTER = lv.get_data("StepCounter") or 42
STEPS_DEC    = lv.get_data("StepsDecrement") or 1
print(f"StepCounter={STEP_COUNTER}  StepsDecrement={STEPS_DEC}")

# Walls
WALLS: set[tuple[int,int]] = set()
for s in lv.get_sprites_by_tag("ihdgageizm"):
    WALLS.add((s.x, s.y))
print(f"Walls: {len(WALLS)}")

# Targets
TARGETS     = [(t.x, t.y) for t in game.plrpelhym]
GOAL_SHAPES = list(game.ldxlnycps)
GOAL_COLORS = list(game.yjdexjsoa)
GOAL_ROTS   = list(game.ehwheiwsk)
print(f"Targets: {TARGETS}")
print(f"  Goals: shapes={GOAL_SHAPES}  colors={GOAL_COLORS}  rots={GOAL_ROTS}")

# Push pads (pre-computed deltas)
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

# ---------------------------------------------------------------------------
# Precomputed gate positions per phase (verified empirically, period=8)
# Each tuple: (rot_changer_pos, shape_changer_pos, color_changer_pos)
# A changer fires when its position equals the player's landing position.
# ---------------------------------------------------------------------------
GATE_PERIOD = 8
ROT_POS   = [(34,40),(29,40),(24,40),(19,40),(14,40),(19,40),(24,40),(29,40)]
SHAPE_POS = [(14,10),(19,10),(24,10),(29,10),(34,10),(29,10),(24,10),(19,10)]
COLOR_POS = [(24,30),(19,30),(19,25),(19,20),(24,20),(29,20),(29,25),(29,30)]

# ---------------------------------------------------------------------------
# Step 2: BFS helpers
# ---------------------------------------------------------------------------

def apply_push_chain(nx: int, ny: int) -> tuple[int, int]:
    """Apply push pads iteratively (handles chained pads)."""
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
    """Apply changers firing at (nx,ny) given the gate has already stepped to next_phase."""
    if (nx, ny) == ROT_POS[next_phase]:
        rot = (rot + 1) % N_ROTS
    if (nx, ny) == SHAPE_POS[next_phase]:
        shape = (shape + 1) % N_SHAPES
    if (nx, ny) == COLOR_POS[next_phase]:
        color = (color + 1) % N_COLORS
    return shape, color, rot


def _in_bbox(px: int, py: int, sx: int, sy: int) -> bool:
    """One-sided: sprite at (sx,sy) inside player 5x5 bbox at (px,py)."""
    return 0 <= sx - px < 5 and 0 <= sy - py < 5


def apply_reset(nx: int, ny: int, r_flags: tuple) -> tuple[tuple, bool]:
    """Resets use one-sided bbox (reset sprites are NOT always on the 5-pixel grid)."""
    new_flags = list(r_flags)
    collected = False
    for i, (rx, ry) in enumerate(RESETS):
        if not r_flags[i] and _in_bbox(nx, ny, rx, ry):
            new_flags[i] = True
            collected = True
    return tuple(new_flags), collected


def is_target_blocking(nx: int, ny: int, shape: int, color: int, rot: int,
                        thit: int) -> bool:
    """True if (nx,ny) is an unconsumed target whose attributes don't match."""
    for i, (tx, ty) in enumerate(TARGETS):
        if (thit >> i) & 1:
            continue   # already consumed
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

def bfs_level6():
    r0    = tuple([False] * len(RESETS))
    start = (px0, py0, game.fwckfzsyc, game.hiaauhahz, game.cklxociuu,
             r0, STEP_COUNTER, 0, 0)
    WIN_MASK = (1 << len(TARGETS)) - 1   # 0b11

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

            # ---- WALL blocked: gate stays, counter decrements ----
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

            # Gate steps first (advance to next_phase)
            next_phase = (gphase + 1) % GATE_PERIOD

            # ---- WRONG-TARGET blocked: gate stays, counter unchanged ----
            # (akoadfsur freeze — always suboptimal, never on optimal path)
            if is_target_blocking(nx, ny, shape, color, rot, thit):
                continue   # skip entirely — no useful state change

            # ---- Successful move ----

            # Reset check at landing
            new_resets, got_reset = apply_reset(nx, ny, resets)
            if got_reset:
                new_ctr = STEP_COUNTER - STEPS_DEC
            else:
                new_ctr = ctr - STEPS_DEC
                if new_ctr < 0:
                    continue

            # Changers at landing (gate is now at next_phase)
            new_shape, new_color, new_rot = apply_changers_at(
                nx, ny, shape, color, rot, next_phase)

            # Push pad (chained)
            px2, py2 = apply_push_chain(nx, ny)
            if (px2, py2) != (nx, ny):
                nx, ny = px2, py2
                new_resets2, got2 = apply_reset(nx, ny, new_resets)
                if got2:
                    new_resets = new_resets2
                    new_ctr    = STEP_COUNTER - STEPS_DEC
                new_shape, new_color, new_rot = apply_changers_at(
                    nx, ny, new_shape, new_color, new_rot, next_phase)

            # Target hit check
            new_thit = check_target_hit(nx, ny, new_shape, new_color, new_rot, thit)

            # Win check
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
goal, par, act = bfs_level6()

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

    targets_left = len(TARGETS)
    for i, a in enumerate(path):
        xb, yb = g2.gudziatsk.x, g2.gudziatsk.y
        obs    = env2.step(AS2[AMAP[a]])
        xa, ya = g2.gudziatsk.x, g2.gudziatsk.y
        ctr    = g2._step_counter_ui.current_steps
        lvl    = obs_levels_completed(obs)
        remaining = len(g2.plrpelhym)
        if remaining < targets_left:
            targets_left = remaining
            print(f"  *** TARGET HIT at step {i+1} (targets remaining: {remaining}) ***")
        print(f"  Step {i+1:3d} ({a}): ({xb:2d},{yb:2d})->({xa:2d},{ya:2d})"
              f"  sh={g2.fwckfzsyc} col={g2.hiaauhahz} rot={g2.cklxociuu}"
              f"  ctr={ctr:2d}  lv={lvl+1}")
        if lvl >= 6:
            print("  *** LEVEL 7 REACHED ***")
            break
