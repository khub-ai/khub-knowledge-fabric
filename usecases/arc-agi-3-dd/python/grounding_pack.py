"""Build a facts-only grounding pack from Round-1 probe results.

Inputs:
  * the Round-1 session dir (must contain tutor_probe_results.json and
    tutor_reply.json -- we need the ELEMENTS list to say what each
    element_id refers to by pre_bbox).
Output:
  * a JSON-serialisable dict: empirical observations only, no hypotheses,
  no outcome-maps, no prose.  Models see this in Round 2 as evidence.
"""
from __future__ import annotations

import json
from pathlib import Path


def build_grounding_pack(session_dir: Path) -> dict:
    tutor_reply = json.loads((session_dir / "tutor_reply.json").read_text(encoding="utf-8"))
    probe_results = json.loads((session_dir / "tutor_probe_results.json").read_text(encoding="utf-8"))

    tutor_assessment = tutor_reply.get("assessment") or {}
    element_lookup = {
        int(e["id"]): {
            "name":     e.get("name"),
            "pre_bbox": e.get("bbox"),
            "tutor_function_guess": e.get("function"),
        }
        for e in tutor_assessment.get("elements", [])
    }

    facts = []
    for pr in probe_results:
        if not pr.get("executed"):
            facts.append({
                "probe_id":     pr.get("probe_id"),
                "status":       "not_executed",
                "parse_errors": pr.get("parse_errors"),
                "exec_error":   pr.get("exec_error"),
            })
            continue

        clean_obs = []
        for o in pr.get("observations", []):
            kind = o.get("kind")
            if kind == "ELEMENT_MOVED":
                eid = o.get("element_id")
                el = element_lookup.get(eid, {})
                clean_obs.append({
                    "kind":            "ELEMENT_MOVED",
                    "element_id":      eid,
                    "element_name":    el.get("name"),
                    "referenced_pre_bbox": el.get("pre_bbox"),
                    "tracked_colour":  o.get("tracked_colour"),
                    "pre_bbox_found":  o.get("pre_bbox"),
                    "post_bbox_found": o.get("post_bbox"),
                    "moved":           o.get("moved"),
                    "tracker_note":    _tracker_reliability_note(o, el),
                })
            elif kind in ("REGION_DELTA", "STATE", "SCORE_DELTA", "AVAILABLE_ACTIONS"):
                clean_obs.append({k: v for k, v in o.items() if k != "raw"})
            else:
                clean_obs.append(o)

        facts.append({
            "probe_id":         pr.get("probe_id"),
            "instructions":     [t.get("instr") for t in pr.get("trace", [])],
            "final_state":      pr.get("final_state"),
            "final_score":      pr.get("final_score"),
            "observations":     clean_obs,
        })

    return {
        "elements_referenced": element_lookup,
        "probe_facts":         facts,
        "executor_caveats":    [
            "ELEMENT_MOVED tracks a 'signature colour' (colour most "
            "over-represented in the element's pre-bbox vs the rest of the "
            "grid), then returns the bbox of the nearest connected component "
            "of that colour after the instructions run.",
            "If the element's distinctive colour also appears elsewhere, the "
            "nearest-component heuristic may still produce a misleading bbox. "
            "Treat a post_bbox whose area is >>10x the element's pre-bbox as "
            "a tracking failure, not a real movement.",
        ],
    }


def _tracker_reliability_note(obs: dict, el: dict) -> str | None:
    pre   = obs.get("pre_bbox")
    post  = obs.get("post_bbox")
    ref   = el.get("pre_bbox")
    if not (pre and post and ref):
        return None
    ref_area  = max(1, (ref[2] - ref[0] + 1) * (ref[3] - ref[1] + 1))
    post_area = max(1, (post[2] - post[0] + 1) * (post[3] - post[1] + 1))
    if post_area > 10 * ref_area:
        return (f"post_bbox area {post_area} is >>10x element pre-bbox area "
                f"{ref_area} — treat as tracking failure, not real movement")
    return None


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    a = ap.parse_args()
    pack = build_grounding_pack(Path(a.session_dir))
    print(json.dumps(pack, indent=2))


if __name__ == "__main__":
    main()
