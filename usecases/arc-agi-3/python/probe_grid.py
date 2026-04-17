"""
Probe TR87 grid structure: show unique colors, find white markers, slot regions.
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
frame_data = env.reset()

grid = np.array(frame_data.frame[0])  # (64, 64) int8
print(f"Grid shape: {grid.shape}")
print(f"Unique colors: {np.unique(grid)}")
print()

# Color map from agents.py
COLOR_NAMES = {0: "black", 1: "white", 2: "red", 3: "green", 4: "blue",
               5: "yellow", 6: "magenta", 7: "orange", 8: "cyan", 9: "pink"}

for color in np.unique(grid):
    rows, cols = np.where(grid == color)
    print(f"Color {color} ({COLOR_NAMES.get(int(color), '?')}): "
          f"rows {rows.min()}-{rows.max()}, cols {cols.min()}-{cols.max()}, "
          f"count={len(rows)}")

print()
print("Row sums (number of non-black pixels per row):")
for r in range(64):
    row = grid[r]
    nz = np.sum(row != 0)  # non-black (0 is not black here, 2 is red which might be bg)
    dominant = np.bincount(row.astype(int) + 0).argmax()
    if r % 4 == 0 or nz > 0:
        print(f"  row {r:2d}: dominant={dominant}({COLOR_NAMES.get(dominant,'?')}), "
              f"unique={list(np.unique(row))}")

print()
# Print raw grid as color chars
_CC = {0: '.', 1: 'W', 2: 'R', 3: 'G', 4: 'B', 5: 'Y', 6: 'M', 7: 'O', 8: 'C', 9: 'P'}
print("Grid (R=red/bg, W=white/cursor, C=cyan, P=pink, M=magenta, G=green, B=blue, .=black):")
for r in range(64):
    print(f"{r:2d}: " + "".join(_CC.get(int(v), '?') for v in grid[r]))
