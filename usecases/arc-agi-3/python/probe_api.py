"""Quick probe of arc_agi API."""
import arc_agi, os, sys

api_key = ""
for line in open("P:/_access/Security/api_keys.env"):
    if line.startswith("arc_api_key="):
        api_key = line.split("=", 1)[1].strip()

arc = arc_agi.Arcade(arc_api_key=api_key)
env = arc.make("tr87", render_mode=None)
print("env:", env)
print("action_space:", getattr(env, "action_space", None))
print("observation_space:", getattr(env, "observation_space", None))
frame = env.reset()
print("frame type:", type(frame))
print("frame.state:", frame.state)
print("frame.levels_completed:", frame.levels_completed)
print("frame.available_actions:", frame.available_actions)
print("frame.frame type:", type(frame.frame))
print("frame.frame:", repr(frame.frame)[:500] if frame.frame is not None else None)

# Try a step with action=2 (ACTION2 = cursor right)
frame2 = env.step(2)
print("\nAfter step(2):")
print("frame2 type:", type(frame2))
print("frame2.state:", frame2.state)
print("frame2.frame type:", type(frame2.frame))
# Check grid shape
grid = frame2.frame[0]
print("grid shape:", grid.shape)
print("grid[0]:", grid[0])

# Try action_space values
print("\naction_space values:", env.action_space)
print("action_space[0] type:", type(env.action_space[0]))
print("action_space[0] value:", env.action_space[0].value)
