"""Produce a PUPIL-vs-TUTOR comparison report for one INITIAL_ASSESSMENT trial.

Metrics in v0.1:
  - parse_ok_both        : both replies parsed as JSON
  - element_count_tutor / element_count_pupil
  - element_bbox_iou_avg : best-match IoU for each pupil element against
                           any tutor element (0.0 if no match)
  - element_function_agreement_rate : fraction of IoU-matched pairs
                                      where function labels agree
  - similar_group_member_overlap    : Jaccard of member sets, after
                                      matching groups by best-element-IoU
  - strategy_first_action_match     : bool
  - strategy_primary_goal_len_ratio : len(pupil) / len(tutor)
  - probe_count_tutor / probe_count_pupil
  - probe_valid_rate_tutor / _pupil : fraction of probes that pass DSL
                                      validation
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _iou(a: List[int], b: List[int]) -> float:
    r0 = max(a[0], b[0]); c0 = max(a[1], b[1])
    r1 = min(a[2], b[2]); c1 = min(a[3], b[3])
    if r1 < r0 or c1 < c0:
        return 0.0
    inter = (r1 - r0 + 1) * (c1 - c0 + 1)
    area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _best_match(el, pool) -> Tuple[Optional[int], float]:
    """Return (pool_index, iou) of the best IoU match, or (None, 0)."""
    best_i, best = None, 0.0
    bbox = el.get("bbox")
    if not bbox or len(bbox) != 4:
        return None, 0.0
    for i, other in enumerate(pool):
        ob = other.get("bbox")
        if not ob or len(ob) != 4:
            continue
        v = _iou(bbox, ob)
        if v > best:
            best_i, best = i, v
    return best_i, best


def _element_metrics(tutor_els: List[dict], pupil_els: List[dict]) -> Dict[str, Any]:
    ious = []
    func_agree = 0
    func_total = 0
    matches = []
    for pe in pupil_els:
        idx, iou = _best_match(pe, tutor_els)
        ious.append(iou)
        if idx is not None and iou > 0.3:
            te = tutor_els[idx]
            func_total += 1
            if (pe.get("function") or "").lower() == (te.get("function") or "").lower():
                func_agree += 1
            matches.append({
                "pupil_id": pe.get("id"),
                "tutor_id": te.get("id"),
                "iou":      round(iou, 3),
                "pupil_fn": pe.get("function"),
                "tutor_fn": te.get("function"),
                "pupil_name": pe.get("name"),
                "tutor_name": te.get("name"),
            })
    avg_iou = (sum(ious) / len(ious)) if ious else 0.0
    return {
        "element_count_tutor": len(tutor_els),
        "element_count_pupil": len(pupil_els),
        "element_bbox_iou_avg": round(avg_iou, 3),
        "element_function_agreement_rate": (
            round(func_agree / func_total, 3) if func_total else None
        ),
        "element_matches": matches,
    }


def _group_metrics(tutor_a, pupil_a, el_matches: List[dict]) -> Dict[str, Any]:
    # map pupil id -> tutor id (best IoU match)
    p2t = {m["pupil_id"]: m["tutor_id"] for m in el_matches if m.get("iou", 0) >= 0.3}
    tutor_groups = tutor_a.get("similar_groups") or []
    pupil_groups = pupil_a.get("similar_groups") or []
    best_overlap = 0.0
    if tutor_groups and pupil_groups:
        overlaps = []
        for pg in pupil_groups:
            pg_as_tutor = {p2t.get(i) for i in pg.get("member_ids", []) if i in p2t}
            pg_as_tutor.discard(None)
            if not pg_as_tutor:
                continue
            best = 0.0
            for tg in tutor_groups:
                tg_members = set(tg.get("member_ids", []))
                if not tg_members:
                    continue
                inter = len(pg_as_tutor & tg_members)
                union = len(pg_as_tutor | tg_members)
                if union and inter / union > best:
                    best = inter / union
            overlaps.append(best)
        if overlaps:
            best_overlap = sum(overlaps) / len(overlaps)
    return {
        "group_count_tutor": len(tutor_groups),
        "group_count_pupil": len(pupil_groups),
        "similar_group_member_overlap": round(best_overlap, 3),
    }


def _strategy_metrics(tutor_a, pupil_a) -> Dict[str, Any]:
    ts = tutor_a.get("initial_strategy") or {}
    ps = pupil_a.get("initial_strategy") or {}
    t_goal = ts.get("primary_goal") or ""
    p_goal = ps.get("primary_goal") or ""
    return {
        "strategy_first_action_match": ts.get("first_action") == ps.get("first_action"),
        "strategy_first_action_tutor": ts.get("first_action"),
        "strategy_first_action_pupil": ps.get("first_action"),
        "strategy_primary_goal_len_ratio": (
            round(len(p_goal) / max(1, len(t_goal)), 3)
        ),
        "strategy_open_questions_count_tutor": len(ts.get("open_questions") or []),
        "strategy_open_questions_count_pupil": len(ps.get("open_questions") or []),
    }


def _probe_metrics(tutor_result: dict, pupil_result: dict) -> Dict[str, Any]:
    def _stat(res):
        probes = res.get("probe_results", []) or []
        n = len(probes)
        ok = sum(1 for p in probes if p.get("executed"))
        return n, (round(ok / n, 3) if n else None)
    t_n, t_ok = _stat(tutor_result)
    p_n, p_ok = _stat(pupil_result)
    return {
        "probe_count_tutor":       t_n,
        "probe_count_pupil":       p_n,
        "probe_valid_rate_tutor":  t_ok,
        "probe_valid_rate_pupil":  p_ok,
    }


def _section_summaries(a: dict) -> Dict[str, str]:
    if not a:
        return {"elements": "—", "similar_groups": "—",
                "initial_strategy": "—", "probes": "—"}
    els = a.get("elements") or []
    gps = a.get("similar_groups") or []
    st  = a.get("initial_strategy") or {}
    prs = a.get("probes") or []
    fn_counts: Dict[str, int] = {}
    for e in els:
        fn = e.get("function", "unknown")
        fn_counts[fn] = fn_counts.get(fn, 0) + 1
    fn_summary = ", ".join(f"{k}:{v}" for k, v in sorted(fn_counts.items()))
    return {
        "elements":         f"{len(els)} ({fn_summary})",
        "similar_groups":   f"{len(gps)} groups",
        "initial_strategy": f"first_action={st.get('first_action')!r}, goal={(st.get('primary_goal') or '')[:60]}",
        "probes":           f"{len(prs)} probes",
    }


def build_report(tutor_entry: dict, pupil_entry: dict) -> Dict[str, Any]:
    ta = tutor_entry.get("assessment")
    pa = pupil_entry.get("assessment")
    out: Dict[str, Any] = {
        "parse_ok_both": bool(ta) and bool(pa),
        "parse_ok_tutor": bool(ta),
        "parse_ok_pupil": bool(pa),
        "section_summaries": {
            "elements":         {"TUTOR": _section_summaries(ta)["elements"],
                                 "PUPIL": _section_summaries(pa)["elements"]},
            "similar_groups":   {"TUTOR": _section_summaries(ta)["similar_groups"],
                                 "PUPIL": _section_summaries(pa)["similar_groups"]},
            "initial_strategy": {"TUTOR": _section_summaries(ta)["initial_strategy"],
                                 "PUPIL": _section_summaries(pa)["initial_strategy"]},
            "probes":           {"TUTOR": _section_summaries(ta)["probes"],
                                 "PUPIL": _section_summaries(pa)["probes"]},
        },
    }
    if not (ta and pa):
        return out

    metrics: Dict[str, Any] = {}
    el = _element_metrics(ta.get("elements") or [], pa.get("elements") or [])
    metrics.update({k: v for k, v in el.items() if k != "element_matches"})
    metrics.update(_group_metrics(ta, pa, el["element_matches"]))
    metrics.update(_strategy_metrics(ta, pa))
    metrics.update(_probe_metrics(tutor_entry, pupil_entry))
    out["metrics"] = metrics
    out["element_matches"] = el["element_matches"]
    return out
