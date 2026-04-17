"""
Probe TR87 actions: show what changes when ACTION1 and ACTION2 are applied.
Focuses on the cyan (color10) and orange (color7) strips in the green area.
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

def show_rows(grid, rows, label=""):
    if label:
        print(label)
    for r in rows:
        print(f"  {r:2d}: " + "".join(_CC.get(int(v), str(int(v))) for v in grid[r]))

def extract_inner(grid, row_slice, col_slice):
    patch = grid[row_slice, col_slice]
    return patch

# Show initial state of key strips
print("=== INITIAL STATE ===")
print("Cursor area (rows 47-50, 58-61):")
show_rows(grid0, range(47, 62))
print()

print("Cyan strip (rows 40-46):")
show_rows(grid0, range(40, 47))
print()

print("Orange strip (rows 51-57):")
show_rows(grid0, range(51, 58))
print()

# Inner patterns for each slot
CYAN_INNER_ROWS = slice(41, 46)   # rows 41-45
ORANGE_INNER_ROWS = slice(52, 57)  # rows 52-56
SLOT_COLS = [
    (slice(14, 21), slice(15, 20)),  # slot 1: (full, inner)
    (slice(21, 28), slice(22, 27)),  # slot 2
    (slice(28, 35), slice(29, 34)),  # slot 3
    (slice(35, 42), slice(36, 41)),  # slot 4
    (slice(42, 49), slice(43, 48)),  # slot 5
]

def show_patterns(grid, label):
    print(f"\n{label}")
    for i, (full_c, inner_c) in enumerate(SLOT_COLS):
        cyan_inner = grid[CYAN_INNER_ROWS, inner_c]
        orange_inner = grid[ORANGE_INNER_ROWS, inner_c]
        cyan_bin = (cyan_inner == 5).astype(int)
        orange_bin = (orange_inner == 5).astype(int)
        match = np.array_equal(cyan_bin, orange_bin)
        print(f"  Slot {i+1}: cyan_inner={list(cyan_inner.flatten())} "
              f"orange_inner={list(orange_inner.flatten())} MATCH={match}")

show_patterns(grid0, "Initial patterns (cyan vs orange):")

# Apply ACTION1 (rotate)
print("\n\n=== AFTER ACTION1 ===")
frame1 = env.step(1)
grid1 = np.array(frame1.frame[0])

# Show changes
diff = grid0 != grid1
changed_rows = np.where(diff.any(axis=1))[0]
print(f"Changed rows: {changed_rows}")
for r in changed_rows:
    print(f"  row {r:2d}: " + "".join(_CC.get(int(v), str(int(v))) for v in grid1[r]))

show_patterns(grid1, "After ACTION1 patterns:")

# Apply ACTION2 (cursor right)
print("\n\n=== AFTER ACTION2 ===")
frame2 = env.step(2)
grid2 = np.array(frame2.frame[0])

diff2 = grid1 != grid2
changed_rows2 = np.where(diff2.any(axis=1))[0]
print(f"Changed rows: {changed_rows2}")
print("Cursor area after ACTION2:")
show_rows(grid2, range(47, 62))
show_patterns(grid2, "After ACTION2 patterns:")
