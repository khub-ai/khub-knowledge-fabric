"""End-to-end test: solve all LS20 levels via the visual (no game internals) path."""
import os, sys
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test")
sys.path.insert(0, "."); sys.path.insert(0, "C:/_backup/github/khub-knowledge-fabric")

import arc_agi
from ls20_visual import plan_ls20_level_visual

arc = arc_agi.Arcade()
env = arc.make("ls20", render_mode=None)
obs = env.reset()
AS = {a.name: a for a in env.action_space}

n_levels = int(getattr(obs, "win_levels", 7))
total_steps = 0
for _ in range(n_levels):
    before = int(obs.levels_completed)
    plan = plan_ls20_level_visual(obs)
    if plan is None:
        print(f"L{before+1}: NO PLAN")
        break
    print(f"L{before+1}: plan={len(plan)} actions")
    advanced = False
    for i, a in enumerate(plan):
        obs = env.step(AS[a])
        total_steps += 1
        if int(obs.levels_completed) > before:
            advanced = True
            print(f"  -> advanced after {i+1}/{len(plan)} actions; total={total_steps}")
            break
    if not advanced:
        print(f"  -> FAILED to advance")
        break

print(f"\nTotal levels completed: {int(obs.levels_completed)}/{n_levels}")
print(f"Total steps: {total_steps}")
