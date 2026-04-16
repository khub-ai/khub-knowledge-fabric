"""
arc_stream_adapter.py — Bridge from ARC ``action_history`` to core Ticks.

Purpose
-------
Translate an ARC-AGI-3 ``action_history`` entry (the ensemble's
per-step record) into a domain-agnostic
:class:`core.cognitive_os.stream.Tick` annotated with generic
``notable_events`` that the core miners understand.

This is the *only* place where ARC-specific diff thresholds live.
Downstream code (miners, hypothesis registry, planner consumers) sees
the cross-domain vocabulary and knows nothing about ARC pixel diffs.

Input schema (from ensemble.py line ~3501)
------------------------------------------
    action_history[i] = {
        "action":      "ACTION1",
        "data":        {...},
        "levels":      L_after,
        "state":       "PLAYING",
        "diff":        int,            # pixel diff between frames
        "player_pos":  (c, r) | None,  # AFTER the action
        "frame_before": 2-d grid,
    }

Notable events we emit (cross-domain vocabulary)
-----------------------------------------------
``stuck_noop``
    ``diff <= STUCK_DIFF_THRESHOLD``.  The commanded action produced no
    observable change; the FutilePatternMiner uses this to detect walls.
``life_loss_reset``
    ``diff >= LIFE_LOSS_DIFF`` (~3000+). Hard reset boundary; the
    FutilePatternMiner treats this as a domain reset via
    ``invalidating_events``.  Robotics analogue: e-stop / episode reset.
``level_advance``
    ``levels`` incremented. Terminal success event for a level.
``maze_rotation``
    ``300 < diff < LIFE_LOSS_DIFF`` with no level advance. Classic
    full-maze RC visit: world layout has changed, previously-mined
    wall knowledge may be stale.  The FutilePatternMiner treats this
    as an invalidating event.
``counter_jump``
    ``80 <= diff <= 300`` with no level advance. Ring-refill-style
    surprise: something notable happened at this position that is
    consistent with a budget refill. The SurpriseMiner's default
    classifier converts this into a provisional ``refill_at_pos``
    hypothesis that the planner can probe.
``anomalous_diff``
    ``54 < diff < 80`` with no level advance.  Low-confidence surprise
    (small-indicator RC, sprite tweens, etc.). Not currently consumed
    by a built-in miner but available for classifier plugins.

Robotics analogue
-----------------
A robotics adapter (not yet written — see
``feedback_cognitive_os_dual_domain`` memory) would produce the same
``notable_events`` from its own sensor deltas:

    stuck_noop        ~  commanded motion produced no pose delta
    life_loss_reset   ~  e-stop / episode reset
    level_advance     ~  goal sensor fires
    maze_rotation     ~  environment state change beyond normal local
                         deltas (e.g., door opened, tray moved)
    counter_jump      ~  recharge / resource-gain event
    anomalous_diff    ~  low-confidence surprise (predictor residual)

The core miners are identical; only this adapter changes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

# Make ``core`` importable when this file is loaded from the arc-agi-3
# usecase directory (ensemble.py lives alongside this module).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.cognitive_os.stream import Tick                       # noqa: E402


# ARC diff-band thresholds. Keep these in one place — any change to the
# game dynamics only needs editing here.
STUCK_DIFF_THRESHOLD   = 10    # diff ≤ this = no-op (wall, mis-click)
WALK_DIFF_CEIL         = 54    # normal 1-step walk diff baseline
ANOMALOUS_DIFF_CEIL    = 80    # small-indicator changes end here
COUNTER_JUMP_CEIL      = 300   # ring refills end here
MAZE_ROTATION_CEIL     = 3000  # classic RC end here; >= this is a reset
LIFE_LOSS_DIFF         = 3000

# DistalEffectMiner inputs. Cells within LOCAL_RADIUS of player_pos are
# excluded from distal_changes (they reflect the player's own motion
# footprint, not a distal device-like effect). Caps guard against
# emitting thousands of points for global rotations / life-loss resets.
DISTAL_LOCAL_RADIUS     = 8
DISTAL_MAX_CELLS        = 12
DISTAL_GLOBAL_CAP       = 400   # more than this = maze rotation, drop all
DISTAL_DIFF_MIN         = 54    # only mine when step diff exceeds a walk
DISTAL_DIFF_MAX         = 300   # classic RC / life-loss diffs are not mined


# ---------------------------------------------------------------- translation


def tick_from_history_entry(
    entry: dict,
    prev_entry: Optional[dict],
    step_idx: int,
    *,
    next_entry: Optional[dict] = None,
) -> Tick:
    """Translate one ``action_history`` entry into a generic Tick.

    Parameters
    ----------
    entry
        The entry just appended to ``action_history``.
    prev_entry
        The immediately preceding entry, or None at the start of a run
        or immediately after a life-loss reset.  Used to recover the
        pre-action position (``action_history`` stores post-action pos).
    step_idx
        Monotonic integer, typically the history index.  Persisted on
        the Tick so miners can cite evidence by step.
    next_entry
        The *following* history entry, if any.  Its ``frame_before`` is
        the post-frame of the current entry.  Required for
        ``distal_changes`` computation; absent during live ``record_latest``
        calls where the next step has not yet been taken.

    Notes
    -----
    ``prev_entry`` is intentionally optional — callers that don't want
    to track it can pass None and the resulting Tick will have
    ``observation["player_pos"] = None``.  Miners that require a pos
    for the pre-state (e.g., FutilePatternMiner) will skip such ticks.
    """
    diff          = int(entry.get("diff") or 0)
    post_pos_raw  = entry.get("player_pos")
    post_pos      = _as_tuple(post_pos_raw)
    prev_pos      = _as_tuple((prev_entry or {}).get("player_pos"))
    levels_before = int((prev_entry or {}).get("levels") or 0)
    levels_after  = int(entry.get("levels") or 0)
    state_after   = entry.get("state")

    notable = _classify_events(
        diff          = diff,
        levels_before = levels_before,
        levels_after  = levels_after,
    )

    observation = {
        "player_pos": prev_pos,
        "level":      levels_before,
    }
    outcome: dict = {
        # Keep both names so miners that key on either find it.
        "post_pos":        post_pos,
        "pos":             post_pos,
        "diff_magnitude":  diff,
        "levels_delta":    levels_after - levels_before,
        "level":           levels_after,
        "state":           state_after,
    }

    # Distal-effect observation: diff the pre- and post-frames and
    # report cells that changed outside a local radius of the player's
    # pre-action position.  Only emitted when the diff band is
    # consistent with an anomalous/counter-jump event (not trivial
    # walks, not life-loss resets) — the miner's corroboration
    # threshold handles false positives from the rare distal ripple.
    if (
        next_entry is not None
        and DISTAL_DIFF_MIN < diff <= DISTAL_DIFF_MAX
        and "level_advance" not in notable
        and "life_loss_reset" not in notable
    ):
        frame_before = entry.get("frame_before")
        frame_after  = next_entry.get("frame_before")
        distal = _compute_distal_changes(
            frame_before = frame_before,
            frame_after  = frame_after,
            player_pos   = prev_pos,
            local_radius = DISTAL_LOCAL_RADIUS,
            global_cap   = DISTAL_GLOBAL_CAP,
            max_cells    = DISTAL_MAX_CELLS,
        )
        if distal:
            outcome["distal_changes"] = distal

    return Tick(
        step_idx       = step_idx,
        observation    = observation,
        action         = entry.get("action"),
        action_params  = dict(entry.get("data") or {}),
        outcome        = outcome,
        notable_events = notable,
    )


def record_latest(
    action_history: list[dict],
    recorder,                      # StreamRecorder (avoid circular import)
    *,
    step_idx: Optional[int] = None,
) -> Tick:
    """Build a Tick from the last entry in ``action_history`` and feed
    it to ``recorder``.  Convenience for ensemble.py, which appends to
    ``action_history`` in one place and wants one line to keep the
    stream in sync.

    ``step_idx`` defaults to ``len(action_history) - 1``.

    Note: the post-frame for the last entry is not yet known (the next
    step has not been taken), so ``record_latest`` cannot compute
    distal_changes.  ``rebuild_stream`` — called each cycle in the
    ensemble — does the look-ahead diff pass and populates them.
    """
    if not action_history:
        raise ValueError("action_history is empty")
    idx = len(action_history) - 1 if step_idx is None else step_idx
    entry = action_history[-1]
    prev  = action_history[-2] if len(action_history) >= 2 else None
    tick  = tick_from_history_entry(entry, prev, step_idx=idx)
    recorder.record(tick)
    return tick


def rebuild_stream(
    action_history: list[dict],
    recorder,                      # StreamRecorder
    *,
    history_start_idx: int = 0,
) -> int:
    """Replay ``action_history`` through ``recorder`` from scratch.

    Use when the per-cycle orchestrator holds a fresh StreamRecorder (as
    ensemble.py does).  Feeds every entry as a Tick, running all
    registered miners.  Returns the number of ticks recorded.

    ``history_start_idx`` limits replay to entries from that index on
    — useful to replay only the current level's history instead of the
    whole episode.

    Look-ahead: each Tick's ``next_entry`` is the following history
    entry (if any), whose ``frame_before`` carries the current step's
    post-frame.  This is what lets DistalEffectMiner see both sides of
    the diff in one pass without blowing up storage with a duplicated
    ``frame_after`` field.
    """
    recorder.reset_window()
    n = 0
    total = len(action_history)
    for i, entry in enumerate(action_history):
        if i < history_start_idx:
            continue
        prev = action_history[i - 1] if i > history_start_idx else None
        nxt  = action_history[i + 1] if i + 1 < total else None
        tick = tick_from_history_entry(
            entry, prev, step_idx=i, next_entry=nxt,
        )
        recorder.record(tick)
        n += 1
    return n


# ------------------------------------------------------------------- helpers


def _classify_events(
    *,
    diff: int,
    levels_before: int,
    levels_after: int,
) -> list[str]:
    """ARC-specific diff-band classification producing generic event
    labels. See module docstring for the vocabulary."""
    events: list[str] = []

    # Hard reset trumps everything: in a life-loss the diff is cosmetic
    # (player teleported) and diff-band mechanics don't apply.
    if diff >= LIFE_LOSS_DIFF:
        events.append("life_loss_reset")
        return events

    if levels_after > levels_before:
        events.append("level_advance")
        return events

    if diff <= STUCK_DIFF_THRESHOLD:
        events.append("stuck_noop")
    elif diff <= WALK_DIFF_CEIL:
        # Ordinary walk — not notable.
        pass
    elif diff < ANOMALOUS_DIFF_CEIL:
        events.append("anomalous_diff")
    elif diff <= COUNTER_JUMP_CEIL:
        events.append("counter_jump")
    elif diff <= MAZE_ROTATION_CEIL:
        events.append("maze_rotation")
    # diff >= LIFE_LOSS_DIFF already handled above

    return events


def _as_tuple(v: Any) -> Optional[tuple]:
    """Coerce ``(c, r)`` list/tuple to a tuple, or return None."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return (int(v[0]), int(v[1]))
    return None


def _compute_distal_changes(
    *,
    frame_before: Any,
    frame_after: Any,
    player_pos: Optional[Tuple[int, int]],
    local_radius: int,
    global_cap: int,
    max_cells: int,
) -> List[Tuple[int, int]]:
    """Return (r, c) cells changed between pre- and post-frames that
    lie farther than ``local_radius`` Manhattan steps from
    ``player_pos``.

    Returns an empty list when:
      * either frame is missing or empty,
      * ``player_pos`` is None (no reference for locality),
      * the total changed-cell count exceeds ``global_cap`` (indicates
        a full-frame rotation / reset, which is not a distal device
        effect).

    Down-samples to ``max_cells`` cells by quantising to a coarse grid
    so the miner sees one point per affected region, not one per pixel.
    """
    if not frame_before or not frame_after or player_pos is None:
        return []
    pc, pr = int(player_pos[0]), int(player_pos[1])
    rows = min(len(frame_before), len(frame_after))
    changed: List[Tuple[int, int]] = []
    for r in range(rows):
        row_b = frame_before[r]
        row_a = frame_after[r]
        cols = min(len(row_b), len(row_a))
        for c in range(cols):
            try:
                if row_b[c] == row_a[c]:
                    continue
            except Exception:
                continue
            if abs(c - pc) + abs(r - pr) <= local_radius:
                continue
            changed.append((r, c))
            # Cheap escape hatch: don't run to completion on a full
            # global re-paint.
            if len(changed) > global_cap:
                return []

    if not changed:
        return []

    # Coarsen: 5×5 quantisation groups pixels of the same sprite into a
    # single cell, giving the miner one "affected region" per device
    # effect rather than one per dirty pixel.
    bucket = 5
    seen: set = set()
    coarse: List[Tuple[int, int]] = []
    for (r, c) in changed:
        key = (r // bucket, c // bucket)
        if key in seen:
            continue
        seen.add(key)
        # Report bucket centre so downstream "cell" coordinates are
        # stable across minor sprite jitter.
        coarse.append((key[0] * bucket + bucket // 2,
                       key[1] * bucket + bucket // 2))
        if len(coarse) >= max_cells:
            break
    return coarse
