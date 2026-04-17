"""Debug LS20 L3: run live, print per-step state divergence vs BFS prediction."""
import os, sys
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test")
sys.path.insert(0, "."); sys.path.insert(0, "C:/_backup/github/khub-knowledge-fabric")

import arc_agi
from ls20_solver import (plan_ls20_level, level_meta_from_game, state_from_game,
                         _build_static, sprite_to_cell, VALID_X, VALID_Y)

arc = arc_agi.Arcade()
env = arc.make("ls20", render_mode=None)
obs = env.reset()
g = env._game
AS = {a.name: a for a in env.action_space}

def adv():
    from ensemble import obs_levels_completed
    return obs_levels_completed(obs)

# Run L1 & L2
for lvl in (0, 1, 2, 3):
    plan = plan_ls20_level(g, lvl)
    print(f"L{lvl+1}: plan={len(plan)} actions")
    for a in plan:
        before = adv()
        globals()['obs'] = env.step(AS[a])
        if adv() > before:
            break

print("\n=== Now at level", adv() + 1, "===")
print("Live state:", state_from_game(g))

meta = level_meta_from_game(g, adv())
state = state_from_game(g)
print(f"L3 meta: walls={len(meta['walls'])} pads={len(meta['push_pads'])} "
      f"resets={len(meta['resets'])} target={meta['targets']}")
print(f"  rot_changers={meta['rot_changers']} color_changers={meta['color_changers']}")
print(f"  push_pads={meta['push_pads']}")
print(f"  goal: sh={meta['goal_shape']} col={meta['goal_color']} rot={meta['goal_rot']}")

static = _build_static(meta)
print(f"\nStatic pads (cell -> dest): {static['pads']}")
print(f"Static rot_ch cells: {static['rot_changers']}")
print(f"Static col_ch cells: {static['color_changers']}")
print(f"Target cell: {static['target']}")

plan = plan_ls20_level(g, adv())
print(f"\nL3 plan: {len(plan)} actions: {plan}")
print()

# Step through L3 plan and print live vs predicted
import sys as _s
walls = static["walls"]
pads = static["pads"]
resets = static["resets"]
rot_ch = static["rot_changers"]
col_ch = static["color_changers"]
ctr_max = static["ctr_max"]; dec = static["steps_dec"]
# Simulated state (mirror BFS effects)
sx, sy = state["player_x"], state["player_y"]
srot = state["rot_idx"]; scol = state["color_idx"]; sshp = state["shape_idx"]
sctr = state["counter"]
srmask = 0
ACTS = {"ACTION1":(0,-5),"ACTION2":(0,5),"ACTION3":(-5,0),"ACTION4":(5,0)}
for i, a in enumerate(plan):
    dx, dy = ACTS[a]
    # Predict
    pnx, pny = sx+dx, sy+dy
    psr, psc, psh = srot, scol, sshp
    pctr = sctr
    pmsk = srmask
    blocked = (pnx,pny) in walls or pnx not in VALID_X or pny not in VALID_Y
    if not blocked:
        cr = False
        for j, rc in enumerate(resets):
            if not (pmsk & (1<<j)) and (pnx,pny)==rc:
                pmsk |= 1<<j; pctr = ctr_max; cr=True; break
        if not cr: pctr -= dec
        if (pnx,pny) in rot_ch: psr=(psr+1)%4
        if (pnx,pny) in col_ch: psc=(psc+1)%4
        if (pnx,pny) in pads: pnx, pny = pads[(pnx,pny)]
    sx, sy, srot, scol, sshp, sctr, srmask = pnx, pny, psr, psc, psh, pctr, pmsk

    # Live step
    before_lvl = adv()
    before_state = state_from_game(g)
    globals()['obs'] = env.step(AS[a])
    after_state = state_from_game(g)
    after_lvl = adv()
    advanced = after_lvl > before_lvl

    match = (after_state["player_x"]==sx and after_state["player_y"]==sy
             and after_state["rot_idx"]==srot and after_state["color_idx"]==scol)
    tag = "OK" if match else "DIVERGE"
    print(f"  {i+1:3d}. {a} pred=({sx},{sy},rot={srot},col={scol},ctr={sctr}) "
          f"live=({after_state['player_x']},{after_state['player_y']},rot={after_state['rot_idx']},"
          f"col={after_state['color_idx']},ctr={after_state['counter']}) {tag}"
          f"{' ADVANCED' if advanced else ''}")
    if not match:
        print(f"      blocked_pred={blocked}  before_live=({before_state['player_x']},{before_state['player_y']})")
        # Continue to see if it diverges further
    if advanced:
        print("ADVANCED to L", after_lvl+1)
        break
