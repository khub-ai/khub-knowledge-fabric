"""
Deep probe: track what changes per action, multiple iterations.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import arc_agi
import numpy as np

api_key = ""
for line in open("P:/_access/Security/api_keys.env"):
    if line.startswith("arc_api_key="):
        api_key = line.split("=", 1)[1].strip()

arc = arc_agi.Arcade(arc_api_key=api_key)
env = arc.make("tr87", render_mode=None)
frame0 = env.reset()
grid0 = np.array(frame0.frame[0])

_CC = {0: '.', 1: 'W', 2: 'R', 3: 'G', 4: 'B', 5: 'Y', 6: 'M', 7: 'O', 8: 'C', 9: 'P', 10: '?'}

def to_str(grid, row):
    return "".join(_CC.get(int(v), str(int(v))) for v in grid[row])

def show_strip_slot(grid, row_slice, col_slice, name):
    rows = list(range(*row_slice.indices(64)))
    vals = []
    for r in rows:
        inner = grid[r, col_slice]
        binary = "".join('Y' if v == 5 else ('?' if v == 10 else 'O') for v in inner)
        vals.append(binary)
    print(f"    {name}: {' | '.join(vals)}")

CYAN_ROWS = slice(41, 46)
ORANGE_ROWS = slice(52, 57)
SLOT_INNER_COLS = [
    slice(15, 20),  # slot 1: cols 15-19
    slice(22, 27),  # slot 2: cols 22-26
    slice(29, 34),  # slot 3: cols 29-33
    slice(36, 41),  # slot 4: cols 36-40
    slice(43, 48),  # slot 5: cols 43-47
]

def show_state(grid, label):
    print(f"\n--- {label} ---")
    for i, col_sl in enumerate(SLOT_INNER_COLS):
        cyan_inner = grid[CYAN_ROWS, col_sl]
        orange_inner = grid[ORANGE_ROWS, col_sl]
        cyan_bin = (cyan_inner == 5).astype(int).tolist()
        orange_bin = (orange_inner == 5).astype(int).tolist()
        match = cyan_bin == orange_bin
        print(f"  Slot {i+1}: cyan={cyan_bin} orange={orange_bin} MATCH={match}")
    # Show cursor location
    cursor_rows = [48, 49, 59, 60]
    for r in cursor_rows:
        row = grid[r]
        black_cols = list(np.where(row == 0)[0])
        if black_cols:
            print(f"  Cursor black @ row {r}: cols {black_cols}")

show_state(grid0, "INITIAL")

# Step through 6 ACTION1 presses (cycle of 4 should bring back to start)
actions = [1, 1, 1, 1,   # 4x ACTION1 (full rotation cycle)
           2,            # ACTION2 (cursor right?)
           1, 1, 1, 1]   # 4x ACTION1 again

grid_prev = grid0
for step_i, action in enumerate(actions):
    frame = env.step(action)
    grid = np.array(frame.frame[0])

    diff = grid != grid_prev
    changed_rows = list(np.where(diff.any(axis=1))[0])

    print(f"\n=== STEP {step_i+1}: ACTION{action} === changed rows: {changed_rows}")

    # Show the changed rows
    for r in changed_rows:
        if r in range(40, 64):
            print(f"  row {r:2d}: {to_str(grid, r)}")

    show_state(grid, f"After step {step_i+1} (ACTION{action})")
    grid_prev = grid
