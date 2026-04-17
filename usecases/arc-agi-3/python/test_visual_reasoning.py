"""Test the visual reasoning primitives against TR87 level 1."""
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
    detect_objects, detect_containment, color_name,
    detect_groups, classify_group_types, detect_focus,
    detect_mismatches, detect_pairwise_mismatches,
)

# Get the TR87 initial frame
arc = arc_agi.Arcade(arc_api_key=os.environ.get("ARC_API_KEY", ""))
env = arc.make("tr87")
obs = env.reset()
frame = obs_frame(obs)
objects = detect_objects(frame)

print("=" * 60)
print("1. CONTAINMENT")
print("=" * 60)
containment = detect_containment(objects)
print(f"Found {len(containment)} containment relations:")
for rel in containment[:20]:
    print(f"  {color_name(rel.container.color)}(size={rel.container.size}) "
          f"contains {color_name(rel.content.color)}(size={rel.content.size}) "
          f"at bbox {rel.content.bbox}")

print(f"\n{'=' * 60}")
print("2. GROUPING")
print("=" * 60)
groups = detect_groups(frame, objects, containment, adjacency_gap=5)
print(f"Found {len(groups)} groups:")
for g in groups:
    member_desc = ", ".join(
        f"{color_name(m.color)}(size={m.size})" for m in g.members
    )
    has_mask = "yes" if g.content_mask is not None else "no"
    print(f"  {g.id}: members=[{member_desc}]  bbox={g.bbox}  mask={has_mask}")

print(f"\n{'=' * 60}")
print("3. TYPE EQUIVALENCE")
print("=" * 60)
types = classify_group_types(groups)
print(f"Found {len(types)} group types:")
for t in types:
    instance_ids = [g.id for g in t.instances]
    print(f"  {t.type_id}: {t.description}")
    print(f"    signature: {t.signature}")
    print(f"    instances: {instance_ids} ({len(t.instances)} groups)")

print(f"\n{'=' * 60}")
print("4. FOCUS DETECTION")
print("=" * 60)
# In TR87, the cursor is white (color 0 in SDK = white)
focus = detect_focus(objects, groups, indicator_colors={0})
if focus:
    target_label = focus.target_group.label if focus.target_group else "none"
    print(f"Focus indicator: color={color_name(focus.color)}, "
          f"{len(focus.indicator_objects)} markers")
    for ind in focus.indicator_objects:
        print(f"  marker at centroid={ind.centroid} size={ind.size}")
    print(f"  Target group: {target_label}")
    print(f"  Target index: {focus.target_index}")
else:
    print("No focus indicator detected")

print(f"\n{'=' * 60}")
print("5. MISMATCH DETECTION")
print("=" * 60)

# Find reference and target groups by type
# Reference pairs = groups in the top half (centroid row < 34)
# Target groups = groups in the bottom half (centroid row >= 34)
ref_groups = [g for g in groups if g.centroid[0] < 34]
tgt_groups = [g for g in groups if g.centroid[0] >= 34]

print(f"Reference groups ({len(ref_groups)}):")
for g in ref_groups:
    print(f"  {g.id} at centroid {g.centroid}")
print(f"Target groups ({len(tgt_groups)}):")
for g in tgt_groups:
    print(f"  {g.id} at centroid {g.centroid}")

if ref_groups and tgt_groups:
    # Simple content comparison
    mismatches = detect_mismatches(ref_groups, tgt_groups, frame)
    print(f"\nSimple mismatches ({len(mismatches)}):")
    for mm in mismatches:
        status = "MATCH" if mm.match else "MISMATCH"
        print(f"  {mm.group_b.id} vs {mm.group_a.id}: {status} — {mm.detail}")

    # Pairwise relationship comparison
    pw_mismatches = detect_pairwise_mismatches(ref_groups, tgt_groups, frame)
    print(f"\nPairwise mismatches ({len(pw_mismatches)}):")
    for mm in pw_mismatches:
        status = "MATCH" if mm.match else "MISMATCH"
        print(f"  {mm.group_b.id} vs {mm.group_a.id}: {status} — {mm.detail}")

# Show content masks for some groups
print(f"\n{'=' * 60}")
print("6. CONTENT MASKS (sample)")
print("=" * 60)
for g in groups[:6]:
    if g.content_mask:
        print(f"\n  {g.id} ({g.group_type}) mask:")
        for row in g.content_mask:
            print("    ", ''.join('#' if v else '.' for v in row))
