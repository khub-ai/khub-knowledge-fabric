"""
Probe TR87 internal game state to find sprite names and win conditions.
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

# Access internal game state
game = getattr(env, "_game", None) or getattr(env, "_env", None)
if game is None:
    for attr in dir(env):
        if not attr.startswith("_"):
            val = getattr(env, attr)
            if hasattr(val, "current_level"):
                game = val
                break
    print("game:", game)
    print("env attrs:", [a for a in dir(env) if not a.startswith("__")])
else:
    print("Found game:", type(game).__name__)

if game is None:
    # Try wrapping
    print("Trying to find game via env attributes...")
    for attr in dir(env):
        obj = getattr(env, attr, None)
        print(f"  {attr}: {type(obj).__name__}")
    sys.exit(1)

# Inspect zvojhrjxxm (cyan strip sprites) and ztgmtnnufb (orange strip sprites)
print()
print("=== CYAN STRIP SPRITES (zvojhrjxxm) ===")
for i, sp in enumerate(game.zvojhrjxxm):
    print(f"  Slot {i+1}: name={sp.name!r}, pos=({sp.x},{sp.y})")

print()
print("=== ORANGE STRIP SPRITES (ztgmtnnufb) ===")
for i, sp in enumerate(game.ztgmtnnufb):
    print(f"  Slot {i+1}: name={sp.name!r}, pos=({sp.x},{sp.y})")

print()
print("=== RULE PAIRS (cifzvbcuwqe) ===")
for i, (lhs, rhs) in enumerate(game.cifzvbcuwqe):
    lhs_names = [s.name for s in lhs]
    rhs_names = [s.name for s in rhs]
    print(f"  Rule {i+1}: LHS={lhs_names} -> RHS={rhs_names}")

print()
print("=== WIN CHECK: bsqsshqpox() ===")
result = game.bsqsshqpox()
print(f"  Win condition met: {result}")

print()
print("=== ACTION4 (cursor right) ===")
frame = env.step(4)  # ACTION4 = cursor right
game_after = getattr(env, "_game", game)
print("Cursor index after ACTION4:", game.qvtymdcqear_index)
print("Orange sprites after ACTION4:")
for i, sp in enumerate(game.ztgmtnnufb):
    print(f"  Slot {i+1}: name={sp.name!r}")
