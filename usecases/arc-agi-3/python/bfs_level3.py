"""
BFS for LS20 Level 3 win condition.

Level 3 mechanics:
- Player starts at (9,45) rot_idx=0, color_idx=0 (orange/12)
- Target at (54,50): need rot_idx=2 (180°), color_idx=1 (white/9)
- Rotation changer (rhsxkxzdjz) at (49,10): player bbox contains (49,10) when player at (49,10)
- Color changer (soyhouuebz) at (29,45): player bbox contains (29,45) when player at (29,45)
- Counter resets (npxgalaybz) at (35,16) and (20,31):
    -> (35,16) collected when player at (34,15) [bbox [34,39)×[15,20) contains (35,16)]
    -> (20,31) collected when player at (19,30) [bbox [19,24)×[30,35) contains (20,31)]
- Push pad (kapcaakvb_b, gbvqrjtaqo) at (54,4):
    -> Player at (54,5) triggers push: player moved to (54,45) in same step
    -> (yjgargdic_r at (8,5) does NOT push: target not in right direction)
- StepCounter=42, StepsDecrement=2 -> 21 steps per life
- Actions: A1=up(y-5), A2=down(y+5), A3=left(x-5), A4=right(x+5)
- Grid: x in {4,9,14,...,59}, y in {0,5,10,...,55}

Wall sprites (ihdgageizm tag): ihdgageizm, fesygzfqui, mxfhnkdzvf, krdypjjivz
"""

from collections import deque

# Valid grid positions
VALID_X = set(range(4, 64, 5))   # {4, 9, 14, ..., 59}
VALID_Y = set(range(0, 60, 5))   # {0, 5, 10, ..., 55}

# Wall positions (all ihdgageizm-tagged sprites)
# ihdgageizm sprites + fesygzfqui(tag=ihdgageizm) at (4,5) + mxfhnkdzvf(tag=ihdgageizm) at (54,0)
WALLS = {
    # Top row y=0
    (4,0),(9,0),(14,0),(19,0),(24,0),(29,0),(34,0),(39,0),(44,0),(49,0),(59,0),
    # Left wall x=4 (y=0..55, including y=5 via fesygzfqui)
    (4,5),(4,10),(4,15),(4,20),(4,25),(4,30),(4,35),(4,40),(4,45),(4,50),(4,55),
    # Right wall x=59 (y=0..55)
    (59,5),(59,10),(59,15),(59,20),(59,25),(59,30),(59,35),(59,40),(59,45),(59,50),(59,55),
    # Bottom cluster
    (9,50),(9,55),(14,55),(19,55),(24,55),(29,55),(34,55),(39,55),(44,55),(49,55),(54,55),
    (4,55),
    # mxfhnkdzvf at (54,0) — ihdgageizm-tagged
    (54,0),
    # Interior maze walls
    (39,5),(39,10),(39,15),(39,20),(39,35),(39,40),(39,45),(39,50),
    (14,10),(14,15),(14,20),(14,25),(14,30),(14,35),(14,50),
    (19,10),(19,35),(19,40),(19,45),(19,50),
    (24,10),(24,35),
    (34,10),(34,35),
    (44,20),(44,35),(44,40),(44,45),(44,50),
    (49,20),(49,35),(49,40),(49,45),(49,50),
}

# Push pads (gbvqrjtaqo sprites):
#   yjgargdic_r at (8,5):  dx=1, pushes player from (9,5)  to (34,5)  [stops before wall at (39,5)]
#   kapcaakvb_b at (54,4): dy=1, pushes player from (54,5) to (54,45) [stops before target at (54,50)]
PUSH_PADS = {
    (9, 5):  (34, 5),    # yjgargdic_r: push right
    (54, 5): (54, 45),   # kapcaakvb_b: push down
}

# Special positions (activated when player lands here)
ROT_CHANGER   = (49, 10)   # rot_idx = (rot_idx+1) % 4
COLOR_CHANGER = (29, 45)   # color_idx = (color_idx+1) % 4
RESET1_POS    = (34, 15)   # npxgalaybz at (35,16)
RESET2_POS    = (19, 30)   # npxgalaybz at (20,31)
TARGET_POS    = (54, 50)   # win if rot_idx==2 and color_idx==1

# Goal
GOAL_ROT   = 2   # 180 degrees
GOAL_COLOR = 1   # index 1 = white/9

STEP_COUNTER_START = 42
STEPS_DECREMENT    = 2


def apply_position_effects(nx, ny, rot, col, r1, r2, ctr):
    """Apply all effects at position (nx,ny). Returns (nx,ny,rot,col,r1,r2,ctr)."""
    new_rot = rot
    new_col = col
    new_r1  = r1
    new_r2  = r2
    new_ctr = ctr
    counter_reset = False

    # Counter resets (prevent counter decrement)
    if not r1 and (nx, ny) == RESET1_POS:
        new_r1 = True
        new_ctr = STEP_COUNTER_START
        counter_reset = True
    if not r2 and (nx, ny) == RESET2_POS:
        new_r2 = True
        new_ctr = STEP_COUNTER_START
        counter_reset = True

    if not counter_reset:
        new_ctr = ctr - STEPS_DECREMENT
        if new_ctr <= 0:
            return None  # game over

    # Rotation changer
    if (nx, ny) == ROT_CHANGER:
        new_rot = (rot + 1) % 4

    # Color changer
    if (nx, ny) == COLOR_CHANGER:
        new_col = (col + 1) % 4

    # Push pads: teleport to exit cell
    if (nx, ny) in PUSH_PADS:
        nx, ny = PUSH_PADS[(nx, ny)]

    return (nx, ny, new_rot, new_col, new_r1, new_r2, new_ctr)


def bfs_level3():
    # State: (x, y, rot_idx, color_idx, r1, r2, counter)
    start = (9, 45, 0, 0, False, False, 42)

    dist   = {start: 0}
    parent = {start: None}
    action_taken = {start: None}
    queue  = deque([start])

    actions = [
        ("A1", 0, -5),   # up: y -= 5
        ("A2", 0, +5),   # down: y += 5
        ("A3", -5, 0),   # left: x -= 5
        ("A4", +5, 0),   # right: x += 5
    ]

    goal_state = None
    states_explored = 0

    while queue:
        state = queue.popleft()
        x, y, rot, col, r1, r2, ctr = state
        states_explored += 1

        for aname, dx, dy in actions:
            nx, ny = x + dx, y + dy

            # Bounds check
            if nx < 4 or nx > 59 or ny < 0 or ny > 55:
                continue
            # Must be on valid grid
            if nx not in VALID_X or ny not in VALID_Y:
                continue
            # Wall check
            if (nx, ny) in WALLS:
                continue

            # Apply effects at new position
            result = apply_position_effects(nx, ny, rot, col, r1, r2, ctr)
            if result is None:
                continue  # game over

            nx, ny, new_rot, new_col, new_r1, new_r2, new_ctr = result

            # Check win
            if (nx, ny) == TARGET_POS and new_rot == GOAL_ROT and new_col == GOAL_COLOR:
                new_state = (nx, ny, new_rot, new_col, new_r1, new_r2, new_ctr)
                parent[new_state]       = state
                action_taken[new_state] = aname
                goal_state = new_state
                print(f"WIN found at step {dist[state] + 1}! States explored: {states_explored}")
                return goal_state, parent, action_taken

            new_state = (nx, ny, new_rot, new_col, new_r1, new_r2, new_ctr)
            if new_state not in dist:
                dist[new_state]         = dist[state] + 1
                parent[new_state]       = state
                action_taken[new_state] = aname
                queue.append(new_state)

    print(f"No win path found! States explored: {states_explored}")
    # Debug: print reachable positions
    positions = set((s[0], s[1]) for s in dist.keys())
    xs = sorted(set(p[0] for p in positions))
    print(f"\nReachable x values: {xs}")
    # Print grid
    print("\nReachable position map (y=0 top, y=55 bottom):")
    for y in range(0, 60, 5):
        row = ""
        for x in range(4, 64, 5):
            if (x,y) in WALLS:
                row += " W"
            elif (x,y) in positions:
                row += " ."
            else:
                row += " _"
        print(f"y={y:2d}: {row}")
    return None, parent, action_taken


def reconstruct_path(goal_state, parent, action_taken):
    path = []
    state = goal_state
    while action_taken[state] is not None:
        path.append(action_taken[state])
        state = parent[state]
    path.reverse()
    return path


if __name__ == "__main__":
    print("BFS for level 3...")
    goal, parent, action_taken = bfs_level3()

    if goal:
        path = reconstruct_path(goal, parent, action_taken)
        print(f"\nPath ({len(path)} steps):")
        print(path)

        # Compress into runs
        compressed = []
        if path:
            cur = path[0]
            cnt = 1
            for a in path[1:]:
                if a == cur:
                    cnt += 1
                else:
                    compressed.append(f"{cur}x{cnt}" if cnt > 1 else cur)
                    cur = a
                    cnt = 1
            compressed.append(f"{cur}x{cnt}" if cnt > 1 else cur)
        print("Compressed:", ", ".join(compressed))

        # Trace through state for verification
        print("\nStep-by-step trace:")
        x, y, rot, col, r1, r2, ctr = 9, 45, 0, 0, False, False, 42
        actions_map = {"A1": (0,-5), "A2": (0,+5), "A3": (-5,0), "A4": (+5,0)}
        for i, a in enumerate(path):
            dx, dy = actions_map[a]
            x += dx; y += dy
            ctr_reset = False
            if not r1 and (x,y) == RESET1_POS: r1=True; ctr=42; ctr_reset=True
            if not r2 and (x,y) == RESET2_POS: r2=True; ctr=42; ctr_reset=True
            if not ctr_reset: ctr -= 2
            if (x,y) == ROT_CHANGER: rot=(rot+1)%4
            if (x,y) == COLOR_CHANGER: col=(col+1)%4
            push = ""
            if (x,y) in PUSH_PADS: x,y=PUSH_PADS[(x,y)]; push=f" [PUSH->({x},{y})]"
            print(f"  Step {i+1:3d}: {a} -> ({x:2d},{y:2d}) rot={rot} col={col} ctr={ctr:2d}"
                  f"{' [RESET]' if ctr_reset else ''}"
                  f"{' [ROT]' if (x,y)==ROT_CHANGER else ''}"
                  f"{' [COL]' if (x,y)==COLOR_CHANGER else ''}"
                  f"{push.replace(chr(8594), '->')}")
