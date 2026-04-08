"""Probe ls20 action directions per shape & rotation."""
import os, sys
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test")
sys.path.insert(0, "."); sys.path.insert(0, "C:/_backup/github/khub-knowledge-fabric")

import arc_agi
from ls20_solver import plan_ls20_level, state_from_game

arc = arc_agi.Arcade()
env = arc.make("ls20", render_mode=None)
obs = env.reset()
g = env._game
AS = {a.name: a for a in env.action_space}

# List candidate game attributes that might encode movement
print("game attrs containing 'shape', 'dir', 'move':")
for n in dir(g):
    if any(s in n.lower() for s in ('shape','dir','move','action','rot')):
        try: v = getattr(g, n)
        except Exception: continue
        if not callable(v):
            print(f"  {n} = {repr(v)[:200]}")

print()
# Pre-solve to L5 with planner
for lvl in (0,1,2,3):
    plan = plan_ls20_level(g, lvl)
    for a in plan:
        from ensemble import obs_levels_completed
        before = obs_levels_completed(obs)
        obs = env.step(AS[a])
        if obs_levels_completed(obs) > before: break

print("=== At L5 ===")
print("State:", state_from_game(g))

# Probe each action: store before/after delta for shape=4
print("\n--- Probing ACTION1..4 from current state (shape=4) ---")
for an in ("ACTION1","ACTION2","ACTION3","ACTION4"):
    s0 = state_from_game(g)
    obs = env.step(AS[an])
    s1 = state_from_game(g)
    dx = s1["player_x"] - s0["player_x"]
    dy = s1["player_y"] - s0["player_y"]
    print(f"  {an}: ({s0['player_x']},{s0['player_y']}) -> ({s1['player_x']},{s1['player_y']})  delta=({dx},{dy})  shape={s1['shape_idx']} rot={s1['rot_idx']}")
    # undo by applying inverse if possible — actually just continue
