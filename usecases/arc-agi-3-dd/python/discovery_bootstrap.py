"""Mechanical probing phase -- learn action effects without calling TUTOR.

Executes each available action once from the spawn and infers its (dr, dc)
by diffing frames.  Pure-pixel analysis, no env._game access.  Costs only
env.step calls (a few game-budget ticks), no LLM tokens.

Gotchas handled here:
  - Multi-palette agent sprites: the agent's pixels may span several
    palette values (e.g. palette 12 centre + palette 9 accent).  We
    identify the "agent" by: it's the smallest connected-components
    cluster that moves consistently across the probe sweep.
  - HUD changes (budget counter, lives): these are static in location
    but change in extent.  They show up in frame diffs too.  We filter
    by requiring the matching candidate's EXTENT (h, w) to be preserved
    -- shrinking/growing components (HUDs) don't pass.
  - Blocked actions: if an action doesn't move the agent (wall at spawn
    in that direction), we leave that action's effect UNKNOWN and the
    caller can try again from a different position.

This module returns a dict, never writes files directly.  The caller
decides what to persist.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from pixel_elements import extract_components, frame_diff


def _match_displaced_component(
    comps_before: list[dict],
    comps_after:  list[dict],
) -> Optional[tuple[int, int, dict, dict]]:
    """Find the displacement shared by all components whose
    palette+size+extent were preserved between frames.

    ARC-AGI-3 agent sprites are frequently multi-palette (e.g. a core
    palette plus an accent palette).  When the agent moves, EVERY part
    of its sprite shifts by the same (dr, dc).  So "agent motion" is
    represented by one or more candidate components that all agree on
    the displacement vector.

    Returns (dr, dc, sample_before, sample_after) -- the sample is any
    one of the agreeing components, useful for fingerprinting.  Returns
    None if candidates disagree on the shift (ambiguous) or there are
    none (no sprite moved in a preservable way).
    """
    # Bucket by (palette, size, extent).  For a bucket where BOTH before
    # and after have exactly one component, their shift is unambiguous.
    # For multi-component buckets (e.g. a row of three identical HUD dots),
    # any before-after pairing is ambiguous -- skip.  This eliminates
    # cross-match false positives from static identical objects.
    from collections import defaultdict
    buckets_before: dict[tuple, list[dict]] = defaultdict(list)
    buckets_after:  dict[tuple, list[dict]] = defaultdict(list)
    for c in comps_before:
        buckets_before[(c["palette"], c["size"], tuple(c["extent"]))].append(c)
    for c in comps_after:
        buckets_after[(c["palette"], c["size"], tuple(c["extent"]))].append(c)

    candidates: list[tuple[int, int, dict, dict]] = []
    for key in set(buckets_before) | set(buckets_after):
        b_list = buckets_before.get(key, [])
        a_list = buckets_after.get(key, [])
        if len(b_list) != 1 or len(a_list) != 1:
            continue   # ambiguous or appeared/disappeared -- skip
        cb, ca = b_list[0], a_list[0]
        dr = ca["centroid"][0] - cb["centroid"][0]
        dc = ca["centroid"][1] - cb["centroid"][1]
        if dr != 0 or dc != 0:
            candidates.append((dr, dc, cb, ca))

    if not candidates:
        return None

    # All candidates must agree on (dr, dc) for the shift to be unambiguous.
    unique_shifts = {(dr, dc) for (dr, dc, _, _) in candidates}
    if len(unique_shifts) != 1:
        return None
    dr, dc = next(iter(unique_shifts))
    # Return the first agreeing candidate as a representative sample.
    return candidates[0]


def bootstrap_action_effects(
    env,
    normalise_frame,
    available_actions: list[int],
) -> dict:
    """Execute each action once from the current state; record its effect.

    Parameters
    ----------
    env:                an ARC-AGI-3 env (reset state recommended).
    normalise_frame:    callable frame->2D numpy array.
    available_actions:  list of action integers (e.g. [1, 2, 3, 4]).

    Returns
    -------
    {
      "action_effects_learned":  {"ACTION1": (dr, dc), ...},   # only actions we could infer
      "action_effects_unknown":  [1, 3],                         # actions that didn't produce a clean displacement
      "agent_candidates":        [(palette, size, extent), ...], # consistent agent fingerprints across probes
      "probe_log":               [ {action, frame_diff, match} , ...],
    }

    Cost: len(available_actions) env.step calls (typically 4).  Zero LLM.
    """
    effects: dict[str, tuple[int, int]] = {}
    unknown: list[int] = []
    agent_fingerprints: list[tuple[int, int, tuple]] = []
    probe_log: list[dict] = []

    obs = getattr(env, "_last_obs", None)
    if obs is None:
        # Need a starting frame.  The env should already have an obs from
        # the preceding reset/step; if not, calling reset() is the only
        # way to guarantee a frame, and that's the caller's responsibility.
        try:
            obs = env.reset()
        except Exception:
            return {"action_effects_learned": {}, "action_effects_unknown": list(available_actions),
                    "agent_candidates": [], "probe_log": []}

    frame_before = normalise_frame(obs.frame)

    for act_int in available_actions:
        obs = env.step(act_int)
        frame_after = normalise_frame(obs.frame)
        diff = frame_diff(frame_before, frame_after)
        comps_before = extract_components(frame_before, min_size=2)
        comps_after  = extract_components(frame_after,  min_size=2)
        matched = _match_displaced_component(comps_before, comps_after)

        entry = {
            "action":        f"ACTION{act_int}",
            "changed_cells": diff["changed_cells"],
            "changed_bbox":  diff["changed_bbox"],
        }

        if matched is not None:
            dr, dc, cb, ca = matched
            effects[f"ACTION{act_int}"] = (dr, dc)
            agent_fingerprints.append((cb["palette"], cb["size"], tuple(cb["extent"])))
            entry["inferred_effect"]     = [dr, dc]
            entry["agent_component"]     = {
                "palette": cb["palette"],
                "size":    cb["size"],
                "extent":  cb["extent"],
            }
        else:
            unknown.append(act_int)
            entry["inferred_effect"] = None

        probe_log.append(entry)
        frame_before = frame_after   # chain: next action measured from here

    # Consensus agent fingerprint: most common (palette, size, extent).
    from collections import Counter
    consensus = Counter(agent_fingerprints).most_common()
    return {
        "action_effects_learned": effects,
        "action_effects_unknown": unknown,
        "agent_candidates":       consensus,
        "probe_log":              probe_log,
    }
