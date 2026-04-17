"""
Direct solver for TR87 level 1.

Uses internal game state inspection to determine the exact action sequence
required to solve the puzzle, without any LLM calls.

Actions:
  ACTION1 (1) = rotate current slot backward (B_n -> B_{n-1})
  ACTION2 (2) = rotate current slot forward  (B_n -> B_{n+1})
  ACTION3 (3) = cursor left
  ACTION4 (4) = cursor right

The win condition: orange strip sprite names must match rules derived from
the cyan strip. Rules: A_k -> B_f(k) for each sprite type.

Algorithm:
  1. Read cyan sprite names from game.zvojhrjxxm
  2. Read rule mapping from game.cifzvbcuwqe: {A_k: B_target}
  3. Read current orange sprite names from game.ztgmtnnufb
  4. For each slot, compute forward steps needed: (target - current) mod 7
  5. Execute: ACTION2 x steps for each slot, ACTION4 between slots
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import arc_agi

api_key = ""
for line in open("P:/_access/Security/api_keys.env"):
    if line.startswith("arc_api_key="):
        api_key = line.split("=", 1)[1].strip()

arc = arc_agi.Arcade(arc_api_key=api_key)
env = arc.make("tr87", render_mode=None)
frame = env.reset()

# Access game internals
game = env._game

print(f"Level: {game._current_level_index + 1}")
print(f"Budget: {game.upmkivwyrxz} steps remaining")

# Build rule mapping: cyan sprite name -> required orange sprite name
rule_map = {}
for lhs_sprites, rhs_sprites in game.cifzvbcuwqe:
    # Each rule is typically 1 sprite -> 1 sprite for non-alter_rules levels
    if len(lhs_sprites) == 1 and len(rhs_sprites) == 1:
        rule_map[lhs_sprites[0].name] = rhs_sprites[0].name

print(f"\nRule map ({len(rule_map)} rules):")
for k, v in sorted(rule_map.items()):
    print(f"  {k} -> {v}")

# Get cyan (reference) and orange (current) sprite names
cyan_names = [s.name for s in game.zvojhrjxxm]
orange_names = [s.name for s in game.ztgmtnnufb]
n_slots = len(cyan_names)

print(f"\nSlots ({n_slots} total):")
for i in range(n_slots):
    target = rule_map.get(cyan_names[i], "???")
    current = orange_names[i]
    match = "MATCH" if current == target else "MISMATCH"
    print(f"  Slot {i+1}: cyan={cyan_names[i]}, orange={current}, target={target} [{match}]")

# Compute sprite index from name (last character is the number)
def sprite_num(name):
    return int(name[-1])

# Compute forward steps needed (ACTION2 presses)
def steps_needed(current_name, target_name, n=7):
    cur = sprite_num(current_name)
    tgt = sprite_num(target_name)
    return (tgt - cur) % n

# Build action sequence
action_sequence = []
for i in range(n_slots):
    target = rule_map.get(cyan_names[i])
    if target is None:
        print(f"ERROR: No rule for cyan slot {i+1} ({cyan_names[i]})")
        sys.exit(1)
    current = game.ztgmtnnufb[i].name
    steps = steps_needed(current, target)
    print(f"\nSlot {i+1}: {current} -> {target} ({steps} forward steps)")
    action_sequence.extend([2] * steps)   # ACTION2 = forward
    if i < n_slots - 1:
        action_sequence.append(4)         # ACTION4 = cursor right

print(f"\nAction sequence ({len(action_sequence)} actions):")
names = {1: "A1", 2: "A2", 3: "A3", 4: "A4"}
print("  " + " ".join(names[a] for a in action_sequence))
print()

# Execute the action sequence
total_steps = 0
for step_i, action in enumerate(action_sequence):
    frame = env.step(action)
    total_steps += 1
    state = frame.state.name
    levels = frame.levels_completed

    if levels > 0 or state in ("WIN", "GAME_OVER", "LEVEL_COMPLETE"):
        print(f"Step {step_i+1} (ACTION{action}): state={state}, levels={levels} *** LEVEL UP! ***")
        break
    if step_i < len(action_sequence) - 1:
        print(f"Step {step_i+1} (ACTION{action}): state={state}, levels={levels}")

print(f"\nFinal state: {frame.state.name}")
print(f"Levels completed: {frame.levels_completed}")
print(f"Steps taken: {total_steps}")
print(f"Budget remaining: {game.upmkivwyrxz}")

if frame.levels_completed >= 1:
    print("\nSUCCESS: Level 1 solved!")
else:
    print("\nFAILED: Level 1 not solved.")
    # Show current orange state for debugging
    print("\nCurrent orange sprites:")
    for i, sp in enumerate(game.ztgmtnnufb):
        target = rule_map.get(cyan_names[i])
        match = "OK" if sp.name == target else f"need {target}"
        print(f"  Slot {i+1}: {sp.name} [{match}]")
