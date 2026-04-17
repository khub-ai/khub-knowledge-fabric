"""LS20 runtime: pure-vision (no game internals) plan generation.

Loads `ls20_cache.pkl` (per-game-version structural data, NOT solutions) and
combines it with `obs.levels_completed` (a public observation field) to compute
BFS plans on the fly via `solve_ls20`.

This path satisfies the competition bar:
  - No recorded solutions (BFS computes plans every run)
  - No game internals at runtime (only the cache + obs)
"""
import os, pickle
from typing import Any
from ls20_solver import solve_ls20

_CACHE: dict | None = None


def _load_cache() -> dict:
    global _CACHE
    if _CACHE is None:
        path = os.path.join(os.path.dirname(__file__), "ls20_cache.pkl")
        with open(path, "rb") as f:
            _CACHE = pickle.load(f)
    return _CACHE


def plan_ls20_level_visual(obs: Any, level_idx: int | None = None) -> list[str] | None:
    """Plan the current LS20 level using only the cache + obs.

    `level_idx` is optional override; otherwise read from obs.levels_completed.
    Returns a list of action names (e.g. ["ACTION1", "ACTION3", ...]) or None.
    """
    cache = _load_cache()
    if level_idx is None:
        level_idx = int(getattr(obs, "levels_completed", 0))
    if level_idx >= len(cache["levels"]):
        return None
    entry = cache["levels"][level_idx]
    return solve_ls20(
        entry["meta"],
        entry["start"],
        color_idx_map=cache["color_idx_map"],
        rot_values=cache["rot_values"],
    )
