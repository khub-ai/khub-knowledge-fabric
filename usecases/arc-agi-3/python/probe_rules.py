"""
Extract the 6 rule pairs from the red reference area.
Left pattern (color 10 frame) -> Right pattern (color 7 frame)
Then find which orange state matches each rule RHS.
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

# Reference area (red bg, rows 0-33) has 3×2 grid of rule pairs
# Row groups: rows 4-10, 13-19, 22-28
# Col groups: (cols 12-18 left + 22-28 right), (cols 35-41 left + 45-51 right)
# Left = color 10 frame (inner 5×5 at rows+1 to rows+5, cols+1 to cols+5)
# Right = color 7 frame (inner 5×5)

RED_ROW_GROUPS = [(4, 10), (13, 19), (22, 28)]   # (start_row, end_row inclusive)
RED_COL_GROUPS = [
    (12, 18, 22, 28),   # left cols 12-18, right cols 22-28
    (35, 41, 45, 51),   # left cols 35-41, right cols 45-51
]

def extract_rule_pattern(grid, row_start, col_start, col_end):
    """Extract 5×5 binary pattern from a 7×7 box.
    Inner region = rows (row_start+1) to (row_start+5),
                   cols (col_start+1) to (col_start+5)
    """
    inner_rows = slice(row_start + 1, row_start + 6)  # 5 rows
    inner_cols = slice(col_start + 1, col_start + 6)  # 5 cols
    patch = grid[inner_rows, inner_cols]
    return (patch == 5).astype(int)  # yellow pixels = 1

print("=== RED AREA RULE PAIRS ===")
rules = []
for r_start, r_end in RED_ROW_GROUPS:
    for l_start, l_end, right_start, right_end in RED_COL_GROUPS:
        lhs = extract_rule_pattern(grid0, r_start, l_start, l_end)
        rhs = extract_rule_pattern(grid0, r_start, right_start, right_end)
        rules.append((lhs, rhs))
        print(f"  Rule (rows {r_start}-{r_end}, left@{l_start}-{l_end} -> right@{right_start}-{right_end}):")
        print(f"    LHS: {lhs.tolist()}")
        print(f"    RHS: {rhs.tolist()}")

print()

# Cyan strip patterns (5 slots, fixed)
CYAN_INNER_ROWS = slice(41, 46)   # rows 41-45
SLOT_INNER_COLS = [
    slice(15, 20),  # slot 1
    slice(22, 27),  # slot 2
    slice(29, 34),  # slot 3
    slice(36, 41),  # slot 4
    slice(43, 48),  # slot 5
]

print("=== CYAN STRIP PATTERNS (LHS reference) ===")
cyan_patterns = []
for i, col_sl in enumerate(SLOT_INNER_COLS):
    pat = (grid0[CYAN_INNER_ROWS, col_sl] == 5).astype(int)
    cyan_patterns.append(pat)
    print(f"  Cyan slot {i+1}: {pat.tolist()}")

# Match each cyan slot to a rule LHS
print()
print("=== MATCH CYAN SLOTS TO RULES ===")
for i, cpat in enumerate(cyan_patterns):
    matched = False
    for j, (lhs, rhs) in enumerate(rules):
        if np.array_equal(cpat, lhs):
            print(f"  Cyan slot {i+1} -> Rule {j+1} (exact LHS match)")
            print(f"    Target RHS: {rhs.tolist()}")
            matched = True
        # Also try rotations of LHS
        for k in range(1, 4):
            lhs_rot = np.rot90(lhs, k)
            if np.array_equal(cpat, lhs_rot):
                print(f"  Cyan slot {i+1} -> Rule {j+1} (LHS rotated {k*90}°)")
                print(f"    Target RHS (rotated {k*90}°): {np.rot90(rhs, k).tolist()}")
                matched = True
    if not matched:
        print(f"  Cyan slot {i+1}: NO RULE MATCH FOUND")

# Now enumerate all 7 orange states for slot 1 (cursor at slot 1)
print()
print("=== ORANGE SLOT 1: ALL 7 STATES ===")
ORANGE_INNER_ROWS = slice(52, 57)  # rows 52-56
orange_states = []

# Current state (initial)
orange_now = (grid0[ORANGE_INNER_ROWS, SLOT_INNER_COLS[0]] == 5).astype(int)
orange_states.append(orange_now.copy())
print(f"  State 0 (initial): {orange_now.tolist()}")

# Get remaining 6 states by pressing ACTION2 (forward)
grid_prev = grid0.copy()
for k in range(1, 7):
    frame = env.step(2)  # ACTION2 = forward
    grid = np.array(frame.frame[0])
    orange_k = (grid[ORANGE_INNER_ROWS, SLOT_INNER_COLS[0]] == 5).astype(int)
    orange_states.append(orange_k.copy())
    print(f"  State {k} (after {k}× A2): {orange_k.tolist()}")
    grid_prev = grid

print()
print("=== CHECK WHICH ORANGE STATE MATCHES RULE RHS FOR SLOT 1 ===")
# Find target RHS for cyan slot 1
cpat = cyan_patterns[0]
for j, (lhs, rhs) in enumerate(rules):
    for k in range(4):
        lhs_rot = np.rot90(lhs, k)
        if np.array_equal(cpat, lhs_rot):
            target_rhs = np.rot90(rhs, k)
            print(f"  Target for slot 1 = Rule {j+1} rotated {k*90}°")
            print(f"  Target RHS: {target_rhs.tolist()}")
            for s, ostate in enumerate(orange_states):
                if np.array_equal(ostate, target_rhs):
                    print(f"  *** MATCH: Orange state {s} matches target RHS! ***")
