"""Quick test: extract shapes from TR87 level 1 and run shape comparison.

Layout from the char-encoded frame:
  Top half (rows 0-33): Red background with 3 rows of paired boxes
    Each row has 4 boxes: X-box, O-box, X-box, O-box
    X = color10 border, O = orange border, # = grey foreground shape
    Green arrows (3-cell horizontal) connect X-box -> O-box pairs

  Bottom half (rows 34-63):
    Row 40-46: X-strip (color10 border) with 5 grey shapes (reference?)
    Row 48-49: Black cursor markers
    Row 51-57: O-strip (orange border) with 5 grey shapes (editable?)
    Row 59-60: Black cursor markers
    Row 63: Blue step counter bar

  So the X-boxes and X-strip are the INPUT/REFERENCE side.
  The O-boxes and O-strip are the OUTPUT/EDITABLE side.
"""
import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parents[2]))

_key_file = Path("P:/_access/Security/api_keys.env")
if _key_file.exists():
    for _line in _key_file.read_text().splitlines():
        for _prefix in ("ANTHROPIC_API_KEY=", "arc_api_key=", "TOGETHER_API_KEY="):
            if _line.startswith(_prefix):
                _var = _prefix.rstrip("=").upper()
                if not os.environ.get(_var):
                    os.environ[_var] = _line.split("=", 1)[1].strip()

import arc_agi
from agents import obs_frame
from object_tracker import (
    extract_subgrid, compare_shapes, find_transformation,
)

# Get the TR87 initial frame
arc = arc_agi.Arcade(arc_api_key=os.environ.get("ARC_API_KEY", ""))
env = arc.make("tr87")
obs = env.reset()
frame = obs_frame(obs)

# The frame layout (from visual inspection):
#
# Top: 3 rows of paired boxes, each pair = (X-box at left, O-box at right)
# The pairs share the same row band and are separated by a green arrow.
#
# Row 1 (rows 4-10):  X-box cols 12-18, O-box cols 22-28  |  X-box cols 35-41, O-box cols 45-51
# Row 2 (rows 13-19): X-box cols 12-18, O-box cols 22-28  |  X-box cols 35-41, O-box cols 45-51
# Row 3 (rows 22-28): X-box cols 12-18, O-box cols 22-28  |  X-box cols 35-41, O-box cols 45-51
#
# Each box is 7x7 with a 1-cell border (X or O), inner shape is 5x5 grey on border-color bg.
# The "shape" inside each box is drawn in grey (#/color5) on the box-color background.
#
# Bottom strips:
# X-strip (rows 40-46, cols 14-48): 5 sub-boxes of 5x5 grey shapes, reference
# O-strip (rows 51-57, cols 14-48): 5 sub-boxes of 5x5 grey shapes, editable

# Define the inner regions (skip the 1-cell border)
# Inner = rows ±1, cols ±1 from the 7x7 box
PAIRS = [
    # (X-box inner bbox, O-box inner bbox) — input -> output reference pairs
    # Row 1, left pair
    ({"r_min": 5, "r_max": 9, "c_min": 13, "c_max": 17},
     {"r_min": 5, "r_max": 9, "c_min": 23, "c_max": 27}),
    # Row 1, right pair
    ({"r_min": 5, "r_max": 9, "c_min": 36, "c_max": 40},
     {"r_min": 5, "r_max": 9, "c_min": 46, "c_max": 50}),
    # Row 2, left pair
    ({"r_min": 14, "r_max": 18, "c_min": 13, "c_max": 17},
     {"r_min": 14, "r_max": 18, "c_min": 23, "c_max": 27}),
    # Row 2, right pair
    ({"r_min": 14, "r_max": 18, "c_min": 36, "c_max": 40},
     {"r_min": 14, "r_max": 18, "c_min": 46, "c_max": 50}),
    # Row 3, left pair
    ({"r_min": 23, "r_max": 27, "c_min": 13, "c_max": 17},
     {"r_min": 23, "r_max": 27, "c_min": 23, "c_max": 27}),
    # Row 3, right pair
    ({"r_min": 23, "r_max": 27, "c_min": 36, "c_max": 40},
     {"r_min": 23, "r_max": 27, "c_min": 46, "c_max": 50}),
]

# Bottom strips: 5 sub-boxes each, 5x5 inner, spaced 7 apart starting at col 15
STRIP_X_BOXES = []  # reference strip (rows 41-45)
STRIP_O_BOXES = []  # editable strip (rows 52-56)
for i in range(5):
    c_start = 15 + i * 7
    STRIP_X_BOXES.append({"r_min": 41, "r_max": 45, "c_min": c_start, "c_max": c_start + 4})
    STRIP_O_BOXES.append({"r_min": 52, "r_max": 56, "c_min": c_start, "c_max": c_start + 4})


def print_mask(mask, label=""):
    if label:
        print(f"  {label}:")
    for row in mask:
        print("    ", ''.join('#' if v else '.' for v in row))


print("=" * 60)
print("REFERENCE PAIRS (top section)")
print("=" * 60)

# Grey (color 5) is the foreground shape inside each box
mask_pairs = []
for idx, (x_bbox, o_bbox) in enumerate(PAIRS):
    mask_x = extract_subgrid(frame, x_bbox, foreground_color=5)
    mask_o = extract_subgrid(frame, o_bbox, foreground_color=5)
    mask_pairs.append((mask_x, mask_o))

    print(f"\nPair {idx + 1}:")
    print_mask(mask_x, "Input (X-box)")
    print_mask(mask_o, "Output (O-box)")

    result = compare_shapes(mask_x, mask_o)
    if result["match"]:
        print(f"  -> MATCH via: {result['transform']}")
    else:
        print(f"  -> NO exact match. Best: {result['best_transform']} (dist={result['best_distance']:.4f})")
        # Show top 3 closest
        sorted_dists = sorted(result["all_distances"], key=lambda x: x[1])
        for name, dist in sorted_dists[:3]:
            print(f"     {name:20s}: {dist:.4f}")

print("\n" + "=" * 60)
print("FIND CONSISTENT TRANSFORMATION")
print("=" * 60)

tf_result = find_transformation(mask_pairs)
print(f"\nConsistent transform found: {tf_result['consistent']}")
print(f"Transform: {tf_result['transform']}")
print(f"Candidate transforms: {tf_result['candidate_transforms']}")

print("\n" + "=" * 60)
print("BOTTOM STRIPS")
print("=" * 60)

print("\nX-strip (reference):")
for i, bbox in enumerate(STRIP_X_BOXES):
    mask = extract_subgrid(frame, bbox, foreground_color=5)
    print_mask(mask, f"Position {i + 1}")

print("\nO-strip (current/editable):")
for i, bbox in enumerate(STRIP_O_BOXES):
    mask = extract_subgrid(frame, bbox, foreground_color=5)
    print_mask(mask, f"Position {i + 1}")

# Compare each X-strip shape to its O-strip counterpart
print("\n--- X vs O strip comparison ---")
for i in range(5):
    mask_x = extract_subgrid(frame, STRIP_X_BOXES[i], foreground_color=5)
    mask_o = extract_subgrid(frame, STRIP_O_BOXES[i], foreground_color=5)
    result = compare_shapes(mask_x, mask_o)
    if result["match"]:
        print(f"  Position {i + 1}: MATCH via {result['transform']}")
    else:
        print(f"  Position {i + 1}: NO match. Best: {result['best_transform']} (dist={result['best_distance']:.4f})")

# If we found a consistent transformation, apply it to see what the O-strip SHOULD be
if tf_result["consistent"] and tf_result["transform_index"] is not None:
    from object_tracker import _to_pixel_set, _normalize_pixel_set, _apply_transform

    print(f"\n--- Expected O-strip (applying '{tf_result['transform']}' to X-strip) ---")
    for i in range(5):
        mask_x = extract_subgrid(frame, STRIP_X_BOXES[i], foreground_color=5)
        pixels = _normalize_pixel_set(_to_pixel_set(mask_x))
        transformed = _apply_transform(pixels, tf_result["transform_index"])

        # Convert back to mask for display
        if transformed:
            max_r = max(r for r, _ in transformed)
            max_c = max(c for _, c in transformed)
            out = [[0] * (max_c + 1) for _ in range(max_r + 1)]
            for r, c in transformed:
                out[r][c] = 1
            print_mask(out, f"Position {i + 1} (expected)")
        else:
            print(f"  Position {i + 1}: empty")
