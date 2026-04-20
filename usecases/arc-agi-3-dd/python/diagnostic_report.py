"""Build a diagnostic markdown report for a Round-2 session.

The report is intentionally qualitative and leads with what went wrong
or remained uncertain, not with agreement metrics.  Sections:

  1. TIER-A findings (coordinates + visual descriptions)
     — for each model, list claims that look physically wrong
       (out-of-bounds bboxes, implausible areas, pixel-space instead
       of grid-space, etc.)
  2. Evidence contradictions
     — Round-1 claims the probes directly disproved, per model.
  3. Tier-B guess changes
     — function/role revisions between Round 1 and Round 2 (informational).
  4. Disagreements after evidence
     — Round-2 points where TUTOR and PUPIL still disagree; candidates
       for human arbitration.
  5. Unresolved open_questions
     — Round-1 open_questions not touched by any executed probe.
  6. Harness reliability notes
     — executor_caveats + any tracker-failure signals in the facts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


FRAME_SHAPE = (64, 64)


def _is_out_of_bounds(bbox, shape=FRAME_SHAPE) -> bool:
    if not bbox or len(bbox) != 4:
        return True
    r0, c0, r1, c1 = bbox
    return (r0 < 0 or c0 < 0 or r1 >= shape[0] or c1 >= shape[1]
            or r0 > r1 or c0 > c1)


def _looks_like_pixel_space(bbox, png_side: int = 512) -> bool:
    if not bbox or len(bbox) != 4:
        return False
    return max(bbox) > FRAME_SHAPE[0] and max(bbox) <= png_side * 2


def _bbox_area(bbox) -> int:
    if not bbox or len(bbox) != 4:
        return 0
    r0, c0, r1, c1 = bbox
    return max(0, (r1 - r0 + 1) * (c1 - c0 + 1))


def tier_a_findings(assessment: dict) -> list[dict]:
    """List physically suspect coordinate/visual claims."""
    findings = []
    if not assessment:
        return findings
    for e in assessment.get("elements", []):
        bbox = e.get("bbox")
        if _is_out_of_bounds(bbox):
            findings.append({
                "element_id": e.get("id"),
                "name":       e.get("name"),
                "bbox":       bbox,
                "issue":      "bbox out of bounds for 64x64 grid"
                              + (" (looks like pixel-space)"
                                 if _looks_like_pixel_space(bbox) else ""),
            })
            continue
        area = _bbox_area(bbox)
        if area > (FRAME_SHAPE[0] * FRAME_SHAPE[1]) * 0.7:
            findings.append({
                "element_id": e.get("id"),
                "name":       e.get("name"),
                "bbox":       bbox,
                "issue":      f"bbox covers {area}/{FRAME_SHAPE[0]*FRAME_SHAPE[1]} cells "
                              f"— implausibly large for a discrete element",
            })
    return findings


def evidence_contradictions(round1: dict, facts: dict) -> list[dict]:
    """Claims from a Round-1 assessment the executed probes directly contradicted.

    Heuristics (not exhaustive):
      * If a probe's ELEMENT_MOVED shows moved=True for an element that
        Round-1 labelled as a wall / decor / readout, that's a
        contradiction.
      * If a probe's final_state is NOT_FINISHED but Round-1 predicted
        GAME_OVER (rough string match against outcome_map keys), note it.
    """
    out = []
    if not round1 or not facts:
        return out
    element_fn = {int(e["id"]): e.get("function", "unknown")
                  for e in round1.get("elements", [])}
    for f in facts.get("probe_facts", []):
        for o in f.get("observations", []):
            if o.get("kind") == "ELEMENT_MOVED" and o.get("moved") is True:
                eid = o.get("element_id")
                fn = element_fn.get(eid)
                if fn in {"wall", "decor", "readout", "counter"}:
                    out.append({
                        "probe_id":   f.get("probe_id"),
                        "element_id": eid,
                        "element_name": o.get("element_name"),
                        "round1_function": fn,
                        "contradiction": (f"Round 1 said element {eid} is a "
                                          f"'{fn}' but probe moved it"),
                    })
    return out


def tier_b_changes(r1: dict, r2: dict) -> list[dict]:
    """Function/role revisions between Round 1 and Round 2."""
    changes = []
    if not (r1 and r2):
        return changes
    r1_fn = {int(e["id"]): e.get("function") for e in r1.get("elements", [])}
    r2_fn = {int(e["id"]): e.get("function") for e in r2.get("elements", [])}
    retracted = set(r1_fn) - set(r2_fn)
    added     = set(r2_fn) - set(r1_fn)
    for eid in sorted(set(r1_fn) & set(r2_fn)):
        if r1_fn[eid] != r2_fn[eid]:
            changes.append({
                "element_id": eid,
                "from":       r1_fn[eid],
                "to":         r2_fn[eid],
                "kind":       "function_revised",
            })
    for eid in sorted(retracted):
        changes.append({"element_id": eid, "kind": "retracted",
                        "from": r1_fn[eid]})
    for eid in sorted(added):
        changes.append({"element_id": eid, "kind": "added",
                        "to": r2_fn[eid]})
    return changes


def post_evidence_disagreements(tutor_r2: dict, pupil_r2: dict) -> list[dict]:
    """Round-2 claims that TUTOR and PUPIL still disagree on.

    Specifically: first_action, primary_goal topic agreement, and per-element
    function where bboxes overlap.
    """
    out = []
    if not (tutor_r2 and pupil_r2):
        return out
    t_s = tutor_r2.get("initial_strategy") or {}
    p_s = pupil_r2.get("initial_strategy") or {}
    if t_s.get("first_action") != p_s.get("first_action"):
        out.append({
            "kind": "first_action_disagreement",
            "tutor": t_s.get("first_action"),
            "pupil": p_s.get("first_action"),
        })
    # Element-function disagreement where bboxes overlap (IoU > 0.3)
    for te in tutor_r2.get("elements", []):
        for pe in pupil_r2.get("elements", []):
            iou = _iou(te.get("bbox"), pe.get("bbox"))
            if iou > 0.3 and te.get("function") != pe.get("function"):
                out.append({
                    "kind": "element_function_disagreement",
                    "tutor_id":   te.get("id"),
                    "pupil_id":   pe.get("id"),
                    "tutor_name": te.get("name"),
                    "pupil_name": pe.get("name"),
                    "bbox_iou":   round(iou, 2),
                    "tutor_function": te.get("function"),
                    "pupil_function": pe.get("function"),
                })
    return out


def _iou(a, b) -> float:
    if not (a and b and len(a) == 4 and len(b) == 4):
        return 0.0
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    ir0 = max(ar0, br0); ic0 = max(ac0, bc0)
    ir1 = min(ar1, br1); ic1 = min(ac1, bc1)
    if ir1 < ir0 or ic1 < ic0:
        return 0.0
    inter = (ir1 - ir0 + 1) * (ic1 - ic0 + 1)
    area_a = max(0, (ar1 - ar0 + 1) * (ac1 - ac0 + 1))
    area_b = max(0, (br1 - br0 + 1) * (bc1 - bc0 + 1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def unresolved_open_questions(round1: dict, facts: dict) -> list[str]:
    """Round-1 open_questions whose keywords don't appear in any probe's
    instructions or observations.  Rough keyword overlap."""
    out = []
    qs = ((round1 or {}).get("initial_strategy") or {}).get("open_questions") or []
    all_probe_text = ""
    for f in (facts or {}).get("probe_facts", []) or []:
        all_probe_text += " ".join(f.get("instructions") or [])
        for o in f.get("observations") or []:
            all_probe_text += " " + json.dumps(o)
    all_probe_text = all_probe_text.lower()
    for q in qs:
        ql = q.lower()
        # rough: if none of the salient tokens appear, mark unresolved
        tokens = [t for t in ql.replace("?", "").split()
                  if len(t) > 3 and t not in {"what", "which", "does", "that",
                                               "with", "from", "into", "this",
                                               "there", "they", "between",
                                               "happens", "happen"}]
        if not any(t in all_probe_text for t in tokens):
            out.append(q)
    return out


def harness_reliability(facts: dict) -> list[str]:
    notes = list(facts.get("executor_caveats") or [])
    for f in facts.get("probe_facts") or []:
        for o in f.get("observations") or []:
            if o.get("tracker_note"):
                notes.append(
                    f"{f.get('probe_id')}: element {o.get('element_id')}: "
                    f"{o['tracker_note']}"
                )
    return notes


def build_markdown(
    *,
    trial_id:     str,
    variant:      str,                      # "2A" or "2B" or other
    prior_knowledge: Optional[str],
    round1_tutor: dict,
    round1_pupil: dict,
    round2_tutor: dict,
    round2_pupil: dict,
    facts:        dict,
    tutor_model:  str,
    pupil_model:  str,
) -> str:
    md: list[str] = []
    md.append(f"# Trial {trial_id} — Round-{variant} diagnostic\n")
    md.append(f"- TUTOR: `{tutor_model}`")
    md.append(f"- PUPIL: `{pupil_model}`")
    if prior_knowledge:
        md.append("\n**Prior knowledge injected:**\n")
        md.append("```")
        md.append(prior_knowledge)
        md.append("```\n")
    else:
        md.append("\n_No prior knowledge injected (2A baseline)._\n")

    md.append("## 1. Tier-A findings (coordinates & visual grounding)\n")
    for role, r2 in (("TUTOR", round2_tutor), ("PUPIL", round2_pupil)):
        ta = tier_a_findings(r2)
        md.append(f"### {role}")
        if not ta:
            md.append("- _none flagged_")
        else:
            for f in ta:
                md.append(f"- element {f.get('element_id')} "
                          f"`{f.get('name')}` bbox={f.get('bbox')}: "
                          f"{f.get('issue')}")
        md.append("")

    md.append("## 2. Evidence contradictions (Round 1 → probe facts)\n")
    for role, r1 in (("TUTOR", round1_tutor), ("PUPIL", round1_pupil)):
        ec = evidence_contradictions(r1, facts)
        md.append(f"### {role}")
        if not ec:
            md.append("- _none detected by current heuristics_")
        else:
            for c in ec:
                md.append(f"- {c['probe_id']}: {c['contradiction']}")
        md.append("")

    md.append("## 3. Tier-B guess changes (Round 1 → Round 2)\n")
    for role, r1, r2 in (("TUTOR", round1_tutor, round2_tutor),
                          ("PUPIL", round1_pupil, round2_pupil)):
        tb = tier_b_changes(r1, r2)
        md.append(f"### {role}")
        if not tb:
            md.append("- _no function revisions_")
        else:
            for c in tb:
                if c["kind"] == "function_revised":
                    md.append(f"- element {c['element_id']}: "
                              f"{c['from']} → {c['to']}")
                elif c["kind"] == "retracted":
                    md.append(f"- element {c['element_id']}: "
                              f"retracted (was {c['from']})")
                elif c["kind"] == "added":
                    md.append(f"- element {c['element_id']}: "
                              f"added (now {c['to']})")
        md.append("")

    md.append("## 4. Disagreements after evidence (arbitration candidates)\n")
    disagreements = post_evidence_disagreements(round2_tutor, round2_pupil)
    if not disagreements:
        md.append("- _no disagreements detected by current heuristics_")
    else:
        for d in disagreements:
            if d["kind"] == "first_action_disagreement":
                md.append(f"- **first_action**: TUTOR={d['tutor']} vs "
                          f"PUPIL={d['pupil']}")
            elif d["kind"] == "element_function_disagreement":
                md.append(f"- **element function** at IoU={d['bbox_iou']}: "
                          f"TUTOR({d['tutor_id']} `{d['tutor_name']}`)="
                          f"{d['tutor_function']} vs "
                          f"PUPIL({d['pupil_id']} `{d['pupil_name']}`)="
                          f"{d['pupil_function']}")
    md.append("")

    md.append("## 5. Round-1 open_questions not touched by probes\n")
    for role, r1 in (("TUTOR", round1_tutor), ("PUPIL", round1_pupil)):
        un = unresolved_open_questions(r1, facts)
        md.append(f"### {role}")
        if not un:
            md.append("- _all covered (or no open questions)_")
        else:
            for q in un:
                md.append(f"- {q}")
        md.append("")

    md.append("## 6. Harness reliability notes\n")
    hn = harness_reliability(facts)
    for n in hn:
        md.append(f"- {n}")
    md.append("")

    md.append("## 7. Revision notes (what each model said changed)\n")
    for role, r2 in (("TUTOR", round2_tutor), ("PUPIL", round2_pupil)):
        md.append(f"### {role}")
        notes = (r2 or {}).get("revision_notes") or []
        if not notes and (r2 or {}).get("no_changes_reason"):
            md.append(f"- _no_changes_reason_: "
                      f"{(r2 or {}).get('no_changes_reason')}")
        elif not notes:
            md.append("- _none_")
        else:
            for n in notes:
                if isinstance(n, dict):
                    md.append(f"- **{n.get('round1_ref','?')}**: "
                              f"{n.get('change','?')} — "
                              f"_{n.get('reason','?')}_")
                else:
                    md.append(f"- {n}")
        md.append("")

    return "\n".join(md)
