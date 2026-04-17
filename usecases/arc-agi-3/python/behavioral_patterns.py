"""Behavioral-pattern hypothesis registry.

Records (pre_pos, action, post_pos) tuples from the agent's trajectory and
converts repeated observations into *predictive hypotheses* about future
outcomes — e.g.:

  * "(pos P, action A) never moves me" -> futile / blocked.
  * "(pos P, action A) moves me by delta" -> productive, trust for planning.

This is a generic mechanism. It is not game-specific. It complements (and
survives) the transient `_blocked` set computed from the current life's
history — behavioral hypotheses persist across the unblock-after-RC logic
so the agent cannot repeatedly bang its head on a wall even when some
other code path has cleared the wall from the blocked set.

The registry is INVALIDATED by maze-mutation events (an RC visit rotates
the maze). It is NOT invalidated by life-loss resets (walls in the maze
do not move when the budget runs out).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class _DeltaCount:
    delta: tuple[int, int]
    count: int


@dataclass
class BehavioralPatternRegistry:
    """(pos, action) -> observed delta histogram.

    Typical lifecycle per cycle:
      reg = BehavioralPatternRegistry()
      reg.rebuild_from_history(action_history, last_maze_mutation_idx)
      if reg.is_futile(player_pos, candidate_action): skip
      predicted_delta = reg.predict(pos, action)  # may be None
    """

    # key = (pos_tuple, action_name) -> { delta_tuple: count }
    _obs: dict[tuple, dict[tuple[int, int], int]] = field(default_factory=dict)
    # index (inclusive) up to which history was consumed; patterns before
    # this index are discarded at rebuild time.
    last_mutation_idx: int = -1

    # --------------------------------------------------------------------- build
    def rebuild_from_history(
        self,
        action_history: list,
        last_maze_mutation_idx: int = -1,
    ) -> None:
        """Reconstruct the registry from `action_history`.

        `last_maze_mutation_idx` should be the history index of the most
        recent RC visit / maze-layout change. All steps up to and including
        this index are ignored — after a rotation, prior (pos, action)
        outcomes no longer apply.

        The function is idempotent: calling it again with a longer history
        produces the registry that reflects the new full history.
        """
        self._obs.clear()
        self.last_mutation_idx = last_maze_mutation_idx

        prev_pos: Optional[tuple[int, int]] = None
        for idx, step in enumerate(action_history):
            if idx <= last_maze_mutation_idx:
                prev_pos = None
                continue
            diff = step.get("diff", 0)
            # Life-loss reset (diff>=3000, pos=None): resets player position
            # but leaves walls intact. Drop prev_pos so we don't build a
            # spurious pattern across the teleport, but keep accumulated
            # observations — walls learned in the previous life are still
            # walls.
            if diff >= 3000 or step.get("player_pos") is None:
                prev_pos = None
                continue
            cur_pos_raw = step.get("player_pos")
            cur_pos = tuple(cur_pos_raw) if cur_pos_raw is not None else None
            action = step.get("action")
            if prev_pos is not None and cur_pos is not None and action:
                delta = (cur_pos[0] - prev_pos[0], cur_pos[1] - prev_pos[1])
                key = (prev_pos, action)
                bucket = self._obs.setdefault(key, {})
                bucket[delta] = bucket.get(delta, 0) + 1
            prev_pos = cur_pos

    # ------------------------------------------------------------------- queries
    def observations(self, pos: tuple[int, int], action: str) -> dict:
        """Return a copy of the delta histogram for (pos, action)."""
        return dict(self._obs.get((tuple(pos), action), {}))

    def predict(
        self,
        pos: tuple[int, int],
        action: str,
    ) -> Optional[tuple[int, int]]:
        """Return the most-frequently-observed delta for (pos, action).

        None if the pair has never been observed.
        """
        bucket = self._obs.get((tuple(pos), action))
        if not bucket:
            return None
        return max(bucket, key=bucket.get)

    def is_futile(
        self,
        pos: tuple[int, int],
        action: str,
        min_count: int = 2,
    ) -> bool:
        """True if (pos, action) has produced only no-op outcomes ≥ min_count.

        A no-op is delta == (0, 0) — the player tried to move but didn't.
        Ignores mixed evidence: if any movement has ever been observed from
        this (pos, action), the pair is considered productive (perhaps the
        wall is intermittent, or this was a walkable cell that later became
        a wall after rotation — we don't want to over-constrain).
        """
        bucket = self._obs.get((tuple(pos), action), {})
        if not bucket:
            return False
        noop = bucket.get((0, 0), 0)
        moved = sum(c for d, c in bucket.items() if d != (0, 0))
        return noop >= min_count and moved == 0

    def futile_actions(
        self,
        pos: tuple[int, int],
        actions: list[str],
        min_count: int = 2,
    ) -> list[str]:
        """Return the subset of `actions` that are predicted futile at `pos`."""
        return [a for a in actions if self.is_futile(pos, a, min_count=min_count)]

    def __bool__(self) -> bool:
        return bool(self._obs)

    def summary(self, top_n: int = 5) -> str:
        """Human-readable summary: top-N entries by total observation count."""
        if not self._obs:
            return "BehavioralPatternRegistry(empty)"
        ranked = sorted(
            self._obs.items(),
            key=lambda kv: sum(kv[1].values()),
            reverse=True,
        )[:top_n]
        parts = []
        for (pos, action), bucket in ranked:
            best_delta = max(bucket, key=bucket.get)
            parts.append(
                f"{pos}/{action}→{best_delta}×{bucket[best_delta]}"
            )
        return "BehavioralPatternRegistry(" + ", ".join(parts) + ")"
