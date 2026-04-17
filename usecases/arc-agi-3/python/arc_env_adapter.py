"""
arc_env_adapter.py — ARC-AGI-3 Environment adapter for the Cognitive OS layer.

Wraps the raw ARC-AGI game environment (returned by the competition harness)
and adapts it to the core.cognitive_os.Environment ABC, so the COS episode
runner and MEDIATOR can interact with ARC games through the same interface
used by robot simulators and physical robots.

Usage
-----
    from arc_env_adapter import ArcEnv
    from core.cognitive_os import Observation

    arc_env = ArcEnv(raw_env, env_id="ls20")
    obs = arc_env.reset()              # returns Observation
    obs = arc_env.step("ACTION1")      # accepts action name string or Action obj

Observation fields populated
-----------------------------
    frame            — list[list[int]]  60×60 pixel palette grid
    step             — int  (episode step counter, incremented by adapter)
    done             — True when state is WIN or GAME_OVER
    reward           — +1.0 on WIN, -1.0 on GAME_OVER, 0.0 otherwise
    metadata         — {
                         "levels_completed": int,
                         "state_name":       str,   # "PLAYING", "WIN", "GAME_OVER", ...
                         "action_space":     list,  # raw Action objects
                       }
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_KF_ROOT = Path(__file__).resolve().parents[3]
if str(_KF_ROOT) not in sys.path:
    sys.path.insert(0, str(_KF_ROOT))

from core.cognitive_os import Environment, Observation
from agents import obs_frame, obs_levels_completed, obs_state_name, _to_list_2d


# States that signal the episode has ended.
_TERMINAL_STATES = {"WIN", "GAME_OVER"}


class ArcEnv(Environment):
    """
    Adapter that wraps a raw ARC-AGI game environment object.

    Parameters
    ----------
    raw_env:
        The environment object returned by the competition harness
        (must support ``reset()``, ``step(action)``, and ``action_space``).
    env_id:
        String identifier for this environment type, e.g. ``"ls20"``.
    """

    def __init__(self, raw_env: Any, env_id: str) -> None:
        self._env    = raw_env
        self._env_id = env_id
        self._step_n = 0

    # ------------------------------------------------------------------
    # Environment ABC
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        self._step_n = 0
        raw_obs = self._env.reset()
        return self._wrap(raw_obs)

    def step(self, action: Any) -> Observation:
        """
        Step the environment.

        *action* may be:
          - a string action name (e.g. ``"ACTION1"``)
          - a raw Action object from ``self._env.action_space``

        If a string is passed, the adapter resolves it against the current
        action space via name matching.
        """
        if isinstance(action, str):
            action = self._resolve_action(action)
        raw_obs = self._env.step(action)
        self._step_n += 1
        return self._wrap(raw_obs)

    @property
    def action_space(self) -> list:
        return list(getattr(self._env, "action_space", []))

    @property
    def env_id(self) -> str:
        return self._env_id

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if callable(close):
            close()

    def render(self) -> Any:
        render = getattr(self._env, "render", None)
        return render() if callable(render) else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wrap(self, raw_obs: Any) -> Observation:
        """Convert a raw ARC observation to Observation."""
        frame = obs_frame(raw_obs)
        levels = obs_levels_completed(raw_obs)
        state_name = obs_state_name(raw_obs)
        done = state_name in _TERMINAL_STATES
        reward = 1.0 if state_name == "WIN" else (-1.0 if state_name == "GAME_OVER" else 0.0)
        return Observation(
            frame=frame,
            step=self._step_n,
            done=done,
            reward=reward,
            metadata={
                "levels_completed": levels,
                "state_name":       state_name,
                "action_space":     self.action_space,
            },
        )

    def _resolve_action(self, name: str) -> Any:
        """Find the Action object in the current action space by name."""
        for a in self.action_space:
            if getattr(a, "name", None) == name:
                return a
        # Fallback: return the string — some environments accept strings directly.
        return name
