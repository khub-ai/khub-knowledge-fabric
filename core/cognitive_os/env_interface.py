"""
env_interface.py — Environment ABC and Observation dataclass.

Every domain-specific environment (ARC-AGI game wrapper, AI2-THOR adapter,
Unitree robot interface, ...) must implement the Environment ABC.  The
episode runner and MEDIATOR interact only with this interface — they never
call game-specific or robot-specific APIs directly.

Observation
-----------
Carries the agent's current perception snapshot.  ``frame`` is intentionally
typed as Any: ARC uses a 2-D pixel grid (list[list[int]]), AI2-THOR returns
an RGB image dict, a physical robot returns a dict of sensor readings.

The ``metadata`` dict is the escape hatch for env-specific extras (levels
completed, lives remaining, step counter, joint states, ...) that the
MEDIATOR or domain rules may need but the core runner does not.

Environment
-----------
Minimal contract: reset(), step(), action_space, env_id.  Adapters are
responsible for translating between the generic Action values returned by
the planner and whatever the underlying API expects.

Example
-------
    class ArcEnv(Environment):
        def __init__(self, game, env_id): ...
        def reset(self) -> Observation: ...
        def step(self, action: Any) -> Observation: ...
        @property
        def action_space(self) -> list: ...
        @property
        def env_id(self) -> str: ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Observation:
    """
    Canonical sensor snapshot: what the agent currently perceives.

    Attributes
    ----------
    frame:
        Raw sensor data.  Type depends on the domain:
          - ARC-AGI:  list[list[int]]  (60×60 pixel palette indices)
          - AI2-THOR: dict             (RGB image + depth + instance segmentation)
          - Robot:    dict             (camera + proprioception + LiDAR, ...)
    step:
        Episode step counter (incremented by the environment on each step).
    done:
        True when the episode has ended (win, loss, or time limit reached).
    reward:
        Scalar feedback signal.  Environments that do not expose rewards
        should leave this at 0.0.
    metadata:
        Domain-specific extras that don't fit the generic fields.
        Examples: ``{"levels_completed": 2, "lives": 3, "step_counter": 38}``
    """
    frame:    Any
    step:     int   = 0
    done:     bool  = False
    reward:   float = 0.0
    metadata: dict  = field(default_factory=dict)


class Environment(ABC):
    """
    Abstract base class for all environment adapters.

    Implementations wrap a game engine, simulator, or physical robot API
    and expose a uniform interface to the episode runner.

    The adapter is responsible for:
      - Translating planner actions (strings or Action objects) to API calls.
      - Packing raw sensor data and episode state into Observation.
      - Managing episode lifecycle (lives, level transitions, time limits).
    """

    # ------------------------------------------------------------------
    # Core interface — must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def reset(self) -> Observation:
        """
        Reset the environment and return the initial Observation.

        May restore a saved checkpoint (e.g. ARC scorecard with N levels
        already completed) or start from scratch depending on configuration.
        """

    @abstractmethod
    def step(self, action: Any) -> Observation:
        """
        Apply *action* and return the resulting Observation.

        ``action`` is whatever the planner produces — an Action enum value,
        a string name, a joint velocity vector, etc.  The adapter converts it.
        """

    @property
    @abstractmethod
    def action_space(self) -> list:
        """
        List of valid actions for the current episode.

        Elements may be strings, enum values, or any domain-appropriate type.
        The planner receives this list and selects from it.
        """

    @property
    @abstractmethod
    def env_id(self) -> str:
        """
        Stable string identifier for this environment type (e.g. "ls20", "ai2thor").

        Used by the episode runner for logging and adapter selection.
        """

    # ------------------------------------------------------------------
    # Optional hooks — override as needed
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release resources (simulator, network connection, hardware).  No-op by default."""

    def seed(self, s: int) -> None:
        """Set the random seed.  No-op if the environment is deterministic."""

    def render(self) -> Any:
        """Return a visual representation for debugging.  None by default."""
        return None
