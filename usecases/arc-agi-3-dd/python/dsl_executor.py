"""Execute parsed probes against a live ARC-AGI-3 env.

Each probe is self-contained: reset the env, apply instructions in
order, then evaluate every observation.  Results are a plain dict that
can be JSON-serialised and compared against the model's outcome_map.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from dsl import (
    DoOne, DoSeq, RepeatDo, Reset,
    ObsRegionDelta, ObsElementMoved, ObsState,
    ObsAvailableActions, ObsScoreDelta,
    ProbeParseResult,
)

ARC_REPO = Path(os.environ.get(
    "ARC_AGI_3_REPO", r"C:\_backup\github\arc-agi-3"
))
sys.path.insert(0, str(ARC_REPO))

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402


def _normalise_frame(raw_frame) -> np.ndarray:
    if isinstance(raw_frame, list) and len(raw_frame) == 1:
        inner = raw_frame[0]
        if isinstance(inner, np.ndarray):
            return inner.astype(int)
        return np.array(inner, dtype=int)
    return np.array(raw_frame, dtype=int)


def _action_for_label(label: str) -> GameAction:
    # "ACTION1" -> GameAction.ACTION1
    return GameAction[label]


def _signature_colour(start_grid: np.ndarray, target_bbox) -> int | None:
    """Pick the colour most distinctive to the element's pre-bbox.

    Score each colour inside the bbox by (inside_frac / (outside_frac + eps))
    — i.e. how over-represented it is vs the rest of the grid.  This avoids
    picking the floor colour when the element sits on a large uniform floor.
    """
    r0, c0, r1, c1 = target_bbox
    patch = start_grid[r0:r1 + 1, c0:c1 + 1]
    if patch.size == 0:
        return None
    total_cells    = start_grid.size
    outside_cells  = total_cells - patch.size
    if outside_cells <= 0:
        return int(np.bincount(patch.ravel()).argmax())
    best_c: int | None = None
    best_score = -1.0
    for c in np.unique(patch):
        c = int(c)
        inside  = int(np.sum(patch == c))
        outside = int(np.sum(start_grid == c)) - inside
        in_frac  = inside / patch.size
        out_frac = outside / outside_cells
        score = in_frac / (out_frac + 1e-6)
        if score > best_score:
            best_score = score
            best_c = c
    return best_c


def _bbox_of_value(grid: np.ndarray, target_bbox, tracked_colour: int | None = None):
    """Return (bbox, colour) of the tracked-colour connected region nearest
    to target_bbox.  If tracked_colour is None, derive it from the pre-bbox
    via `_signature_colour` (computed on the pre-probe grid).

    Returns None if the colour has vanished.
    """
    if tracked_colour is None:
        tracked_colour = _signature_colour(grid, target_bbox)
        if tracked_colour is None:
            return None

    mask = grid == tracked_colour
    if not mask.any():
        return None

    # Flood-fill (BFS) connected components of `tracked_colour`, keep the
    # one nearest the target bbox centre.  Avoids returning a huge bbox when
    # the colour also appears in unrelated regions.
    r0, c0, r1, c1 = target_bbox
    cr = (r0 + r1) / 2.0
    cc = (c0 + c1) / 2.0
    h, w = grid.shape
    visited = np.zeros_like(mask, dtype=bool)
    best: list[int] | None = None
    best_d = float("inf")
    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or visited[sr, sc]:
                continue
            # BFS
            stack = [(sr, sc)]
            rmin, cmin, rmax, cmax = sr, sc, sr, sc
            while stack:
                rr, cc_ = stack.pop()
                if rr < 0 or rr >= h or cc_ < 0 or cc_ >= w:
                    continue
                if visited[rr, cc_] or not mask[rr, cc_]:
                    continue
                visited[rr, cc_] = True
                if rr < rmin: rmin = rr
                if cc_ < cmin: cmin = cc_
                if rr > rmax: rmax = rr
                if cc_ > cmax: cmax = cc_
                stack.extend([(rr+1, cc_), (rr-1, cc_), (rr, cc_+1), (rr, cc_-1)])
            mcr = (rmin + rmax) / 2.0
            mcc = (cmin + cmax) / 2.0
            d = (mcr - cr) ** 2 + (mcc - cc) ** 2
            if d < best_d:
                best_d = d
                best = [int(rmin), int(cmin), int(rmax), int(cmax)]
    if best is None:
        return None
    return best, tracked_colour


def run_probe(
    probe:             ProbeParseResult,
    game_id:           str,
    element_bboxes:    Dict[int, list],   # from the model's ELEMENTS section
) -> Dict[str, Any]:
    """Execute one probe.  Returns a dict with per-observation results
    and a top-level error field if execution blew up."""
    if probe.errors:
        return {
            "probe_id":     probe.probe_id,
            "executed":     False,
            "parse_errors": probe.errors,
        }

    arc = Arcade(
        operation_mode   = OperationMode.OFFLINE,
        environments_dir = str(ARC_REPO / "environment_files"),
    )
    env = arc.make(game_id)
    obs0 = env.reset()
    start_grid  = _normalise_frame(obs0.frame)
    start_score = int(obs0.levels_completed)

    # Apply instructions.
    cur_obs = obs0
    trace: List[Dict[str, Any]] = []
    try:
        for instr in probe.instructions:
            if isinstance(instr, Reset):
                cur_obs = env.reset()
                trace.append({"instr": "RESET"})
            elif isinstance(instr, DoOne):
                cur_obs = env.step(_action_for_label(instr.action))
                trace.append({"instr": f"DO {instr.action}",
                              "state_after": getattr(cur_obs.state, "name", str(cur_obs.state))})
            elif isinstance(instr, DoSeq):
                for a in instr.actions:
                    cur_obs = env.step(_action_for_label(a))
                    trace.append({"instr": f"DO {a} (seq)",
                                  "state_after": getattr(cur_obs.state, "name", str(cur_obs.state))})
            elif isinstance(instr, RepeatDo):
                for _ in range(instr.n):
                    cur_obs = env.step(_action_for_label(instr.action))
                trace.append({"instr": f"REPEAT DO {instr.action} {instr.n}",
                              "state_after": getattr(cur_obs.state, "name", str(cur_obs.state))})
    except Exception as e:  # noqa: BLE001
        return {
            "probe_id":   probe.probe_id,
            "executed":   False,
            "exec_error": f"{type(e).__name__}: {e}",
            "trace":      trace,
        }

    end_grid  = _normalise_frame(cur_obs.frame)
    end_state = getattr(cur_obs.state, "name", str(cur_obs.state))
    end_score = int(cur_obs.levels_completed)
    end_actions = [f"ACTION{int(a)}" for a in cur_obs.available_actions]

    # Evaluate observations.
    obs_results: List[Dict[str, Any]] = []
    for o in probe.observations:
        if isinstance(o, ObsState):
            obs_results.append({"kind": "STATE", "value": end_state})
        elif isinstance(o, ObsAvailableActions):
            obs_results.append({"kind": "AVAILABLE_ACTIONS", "value": end_actions})
        elif isinstance(o, ObsScoreDelta):
            obs_results.append({"kind": "SCORE_DELTA", "value": end_score - start_score})
        elif isinstance(o, ObsRegionDelta):
            r0, c0, r1, c1 = o.bbox
            pre  = start_grid[r0:r1+1, c0:c1+1]
            post = end_grid[r0:r1+1, c0:c1+1]
            delta = int(np.sum(pre != post))
            obs_results.append({
                "kind": "REGION_DELTA",
                "bbox": list(o.bbox),
                "value": delta,
            })
        elif isinstance(o, ObsElementMoved):
            target = element_bboxes.get(o.element_id)
            if target is None:
                obs_results.append({
                    "kind": "ELEMENT_MOVED",
                    "element_id": o.element_id,
                    "error": "element id not in ELEMENTS",
                })
                continue
            sig_col = _signature_colour(start_grid, target)
            before  = _bbox_of_value(start_grid, target, sig_col) if sig_col is not None else None
            before_bbox = before[0] if before is not None else None
            after   = _bbox_of_value(end_grid, target, sig_col) if sig_col is not None else None
            after_bbox = after[0] if after is not None else None
            moved = (after_bbox != before_bbox) if (before_bbox and after_bbox) else None
            obs_results.append({
                "kind": "ELEMENT_MOVED",
                "element_id": o.element_id,
                "pre_bbox":  before_bbox,
                "post_bbox": after_bbox,
                "moved":     bool(moved) if moved is not None else None,
                "tracked_colour": sig_col,
            })

    return {
        "probe_id":     probe.probe_id,
        "hypothesis":   probe.hypothesis,
        "executed":     True,
        "trace":        trace,
        "observations": obs_results,
        "final_state":  end_state,
        "final_score":  end_score,
        "final_actions": end_actions,
    }
