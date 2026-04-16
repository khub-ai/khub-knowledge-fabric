"""
perception.py — Perception ABC for the Cognitive OS layer.

Perception modules convert raw sensor data (a frame from the environment)
into structured percepts that the OBSERVER, MEDIATOR, and StateStore can
reason about.

Implementations
---------------
  ArcPerception  (usecases/arc-agi-3/python/)
      Reads a 60×60 pixel palette grid.  Uses color histograms, diff analysis,
      and VLM calls to classify objects, detect movement, and track state.

  RobotPerception  (usecases/robotics/adapters/  — Phase 1)
      Reads RGB-D camera frames + proprioception.  Uses CV, depth estimation,
      and object-detection models to build a scene graph.

Design notes
------------
- ``perceive`` is intentionally stateless: same frame → same result.  Callers
  maintain history in StateStore, not inside the Perception module.
- ``track_changes`` accepts two consecutive frames and returns a diff dict.
  This mirrors the pixel-diff approach in arc-agi-3 and will map to optical
  flow / change detection in robotics.
- The VLM call path (OBSERVER) is not part of this ABC; perception here is
  the lower-level, more deterministic layer.  VLM integration happens in the
  OBSERVER agent which *uses* a Perception implementation as one of its tools.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Perception(ABC):
    """
    Convert raw sensor frames into structured percepts.

    All methods accept domain-specific frame types (pixel grids, image dicts,
    sensor bundles) and return plain dicts — the common currency of the
    OBSERVER → StateStore pipeline.
    """

    # ------------------------------------------------------------------
    # Core interface — must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def perceive(self, frame: Any) -> dict:
        """
        Extract percepts from a single sensor frame.

        Returns a dict of percept observations.  Keys are domain-agnostic
        where possible (e.g. ``"objects"``, ``"player_pos"``, ``"passable"``).
        Domain-specific extras go in a ``"_domain"`` sub-dict.

        The returned dict is meant to be written into StateStore by the caller:
            percepts = perception.perceive(obs.frame)
            for key, value in percepts.items():
                store.set(("percept", key), value, scope="step")
        """

    @abstractmethod
    def track_changes(self, prev_frame: Any, curr_frame: Any) -> dict:
        """
        Compute a diff between two consecutive frames.

        Returns a dict describing what changed:
            {
                "pixel_diff": int,          # aggregate change magnitude
                "changed_regions": [...],   # bounding boxes or cell lists
                "appeared": [...],          # new objects / entities
                "disappeared": [...],       # removed objects / entities
                "moved": [...],             # (entity, from, to) tuples
            }
        Implementations fill only the fields they can compute; absent keys
        signal "not available" rather than "no change".
        """

    # ------------------------------------------------------------------
    # Optional helpers — override for richer perception
    # ------------------------------------------------------------------

    def classify_object(self, frame: Any, region: Any) -> dict:
        """
        Classify a single object/region.

        Returns a dict: ``{"category": str, "confidence": float, ...}``.
        Default implementation returns an empty dict (not implemented).
        """
        return {}

    def estimate_player_pos(self, frame: Any) -> Any:
        """
        Return the agent/player position in the frame coordinate system.

        Return type is domain-dependent (pixel cell tuple, (x,y,z) world
        coordinates, ...).  Returns None if not determinable.
        """
        return None

    def summarise_for_prompt(self, frame: Any, prev_frame: Any = None) -> str:
        """
        Produce a concise natural-language description of the frame for LLM
        injection.  Called by the OBSERVER before its VLM analysis.

        Default: returns an empty string (caller falls back to raw frame).
        """
        return ""
