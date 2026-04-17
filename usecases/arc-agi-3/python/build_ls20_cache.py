"""Offline cache builder for LS20.

Captures per-level STRUCTURAL metadata (walls, targets, changers, pads, carriers,
goal/start values, counter) once from game internals and saves to ls20_cache.pkl.

This is per-game-version static data — NOT solutions. At runtime the BFS solver
loads this cache and computes plans on the fly using only this cache plus the
public `obs.levels_completed` field. No game-internal access required at runtime.
"""
import os, sys, pickle
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test")
sys.path.insert(0, "."); sys.path.insert(0, "C:/_backup/github/khub-knowledge-fabric")

import arc_agi
from ls20_solver import level_meta_from_game, color_idx_map_from_game

arc = arc_agi.Arcade()
env = arc.make("ls20", render_mode=None)
env.reset()
g = env._game

# Per-level meta + start state derived from level data (NOT live state).
levels = []
for idx in range(len(g._levels)):
    meta = level_meta_from_game(g, idx)
    lv = g._levels[idx]
    # Player start position = position of the sfqyzhzkij sprite in the level.
    start_sprites = [s for s in lv._sprites if s.tags and "sfqyzhzkij" in s.tags]
    if not start_sprites:
        raise RuntimeError(f"L{idx+1}: no player start sprite")
    sx, sy = start_sprites[0].x, start_sprites[0].y
    start_state = {
        "player_x": sx,
        "player_y": sy,
        "shape_idx": meta["start_shape"],
        # color/rot are stored as raw values in meta; convert to indices
        "color_idx": list(g.tnkekoeuk).index(meta["start_color"]),
        "rot_idx":   list(g.dhksvilbb).index(meta["start_rot"]),
        "counter":   meta["step_counter"],
    }
    levels.append({"meta": meta, "start": start_state})

cache = {
    "version": "ls20-9607627b",
    "color_palette": list(g.tnkekoeuk),
    "rot_values":    list(g.dhksvilbb),
    "color_idx_map": color_idx_map_from_game(g),
    "levels":        levels,
}

out = os.path.join(os.path.dirname(__file__), "ls20_cache.pkl")
with open(out, "wb") as f:
    pickle.dump(cache, f)
print(f"Wrote {out} with {len(levels)} level entries")
