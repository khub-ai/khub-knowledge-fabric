"""End-to-end test: solve all LS20 levels via planner."""
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

from ensemble import obs_levels_completed

n_levels = len(g._levels)
total_steps = 0
for lvl in range(n_levels):
    before_lvl = obs_levels_completed(obs)
    plan = plan_ls20_level(g, before_lvl)
    if plan is None:
        print(f"L{before_lvl+1}: NO PLAN (BFS failed)")
        break
    print(f"L{before_lvl+1}: plan={len(plan)} actions")
    advanced = False
    for i, a in enumerate(plan):
        obs = env.step(AS[a])
        total_steps += 1
        if obs_levels_completed(obs) > before_lvl:
            advanced = True
            print(f"  -> advanced after {i+1}/{len(plan)} actions; total_steps={total_steps}")
            break
    if not advanced:
        print(f"  -> FAILED to advance. Final state: {state_from_game(g)}")
        break

print(f"\nTotal levels completed: {obs_levels_completed(obs)}/{n_levels}")
print(f"Total steps: {total_steps}")
