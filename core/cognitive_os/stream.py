"""
stream.py — Sensory-action-decision stream for the Cognitive OS.

A single append-only record of *everything the agent observes, decides, and
does*, tick by tick. The stream is the substrate that pattern miners run
over to derive:

  * behavioural rules  ("action A at pos P is a no-op" — walls)
  * positive hypotheses ("stepping on tile T refills a counter" — rings)
  * co-occurrence rules  (already handled by core.knowledge.co_occurrence)

Domain-agnostic by design. ARC frames and robot sensor bundles both land
here as ``observation`` dicts; the miners don't know which is which.

Relationship to other modules
-----------------------------
* ``core.cognitive_os.state_store.StateStore`` holds the current *world
  model* — a queryable fact base. The stream here is the *trajectory log*
  from which world-model updates are derived. They coexist.
* ``core.cognitive_os.hypothesis.HypothesisRegistry`` holds testable claims.
  Miners are the bridge: stream in -> hypotheses / rules out.
* ``action_history`` (in ensemble.py) is the current ARC trajectory log.
  The stream recorder subsumes it: domain adapters convert an
  ``action_history`` entry into a ``Tick`` and feed it here.

Design choices
--------------
* Ticks are mutable-but-append-only: once appended, miners attach
  derived annotations (notable events, etc.) in place rather than copying.
* Bounded memory. The stream keeps a rolling window by default so it can
  be used inside long-running loops without pathological growth. Callers
  can persist snapshots via ``save``.
* Miners are pluggable and run incrementally: when ``record(tick)`` is
  called, registered miners see the new tick only and may also look back
  at the window. Miners mutate an externally-supplied HypothesisRegistry.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional


# --------------------------------------------------------------------- Tick


@dataclass
class Tick:
    """One sensory-action-decision-outcome record.

    Fields
    ------
    step_idx
        Monotonic integer; usually the environment step counter.
    wall_time
        ``time.time()`` when this tick was recorded.
    observation
        Sensory snapshot *before* the action. Domain-agnostic dict.
        For ARC: ``{"frame": 2-d-grid, "player_pos": (c,r), "level": n}``.
        For robotics: ``{"joint_state": [...], "camera": handle, ...}``.
    action
        Action name chosen by the agent.  None for the initial tick
        (reset observation) where no action was taken yet.
    action_params
        Optional structured parameters that accompanied the action.
    decision
        Context explaining *why* this action was picked.  Typical keys:
        ``"plan"`` (list of planned actions), ``"rationale"`` (string),
        ``"active_goals"`` (list), ``"cycle_id"``.  Empty when unavailable.
    outcome
        Observable result after execution.  Typical keys:
        ``"post_pos"``, ``"diff_magnitude"``, ``"levels_delta"``,
        ``"reward"``, ``"counter_value"``.
    notable_events
        Event labels derived from the outcome — e.g. ``"maze_rotation"``,
        ``"counter_jump"``, ``"level_advance"``, ``"stuck_noop"``.
        Domain adapters and miners append to this.
    """

    step_idx:       int
    observation:    dict         = field(default_factory=dict)
    action:         Optional[str]= None
    action_params:  dict         = field(default_factory=dict)
    decision:       dict         = field(default_factory=dict)
    outcome:        dict         = field(default_factory=dict)
    notable_events: list[str]    = field(default_factory=list)
    wall_time:      float        = field(default_factory=time.time)

    # ---- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        return _to_jsonable(asdict(self))


def _to_jsonable(x: Any) -> Any:
    """Recursively convert tuples to lists for JSON output."""
    if isinstance(x, tuple):
        return [_to_jsonable(e) for e in x]
    if isinstance(x, list):
        return [_to_jsonable(e) for e in x]
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    return x


# -------------------------------------------------------------- StreamRecorder


MinerFn = Callable[["StreamRecorder", Tick], None]


class StreamRecorder:
    """Bounded, append-only tick log that runs pluggable miners on each record.

    Typical usage::

        stream = StreamRecorder(max_len=2048)
        stream.register_miner(futile_miner)
        stream.register_miner(surprise_miner)
        ...
        tick = adapter.tick_from(action_history[-1], step_idx=...)
        stream.record(tick)     # miners run automatically
        refill_subjects = hypothesis_registry.subjects("refill_at_pos")

    Parameters
    ----------
    max_len
        Maximum number of ticks to keep in memory.  Older ticks are
        dropped when the cap is hit.  Set to ``0`` for unlimited (only do
        this for short episodes).
    reset_marker
        A callable that receives a tick and returns True if that tick is
        a hard reset boundary.  When called, ``reset_marker`` optionally
        lets callers clear the window at a detected reset — see
        ``reset_window``.
    """

    def __init__(
        self,
        max_len: int = 4096,
        reset_marker: Optional[Callable[[Tick], bool]] = None,
    ):
        self._ticks: list[Tick] = []
        self.max_len      = max_len
        self._miners: list[MinerFn] = []
        self._reset_marker = reset_marker

    # ---- configuration -----------------------------------------------------

    def register_miner(self, miner: MinerFn) -> None:
        self._miners.append(miner)

    # ---- mutation ----------------------------------------------------------

    def record(self, tick: Tick) -> Tick:
        """Append ``tick`` and run all registered miners on it."""
        self._ticks.append(tick)
        if self.max_len and len(self._ticks) > self.max_len:
            # Drop the oldest; preserve recency window.
            self._ticks = self._ticks[-self.max_len:]
        for m in self._miners:
            try:
                m(self, tick)
            except Exception as exc:  # miners must never break the loop
                tick.notable_events.append(f"miner_error:{type(exc).__name__}")
        return tick

    def reset_window(self) -> None:
        """Drop all retained ticks.  Use on level boundaries or episode resets
        if the caller wants per-level miner behaviour independent of history."""
        self._ticks.clear()

    # ---- query -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._ticks)

    def __iter__(self):
        return iter(self._ticks)

    def last(self, n: int) -> list[Tick]:
        if n <= 0:
            return []
        return self._ticks[-n:]

    def since(self, step_idx_exclusive: int) -> list[Tick]:
        return [t for t in self._ticks if t.step_idx > step_idx_exclusive]

    def find(self, predicate: Callable[[Tick], bool]) -> list[Tick]:
        return [t for t in self._ticks if predicate(t)]

    def ticks(self) -> list[Tick]:
        """Return a *copy* of the current tick list — safe for iteration
        from miners that also mutate."""
        return list(self._ticks)

    # ---- persistence -------------------------------------------------------

    def save(self, path: Path, tail: Optional[int] = None) -> int:
        """Write ticks to ``path`` as a JSON array. ``tail`` limits to the
        last N ticks; None writes all retained ticks."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        items = self._ticks if tail is None else self._ticks[-tail:]
        data = [t.to_dict() for t in items]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return len(data)
