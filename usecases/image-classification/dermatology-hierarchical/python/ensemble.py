"""
ensemble.py — Hierarchical (coarse-to-fine) ensemble orchestrator.

Two-level pipeline:
  Level 1  Classify into one of 5 coarse groups using the full KF pipeline
           (Observer -> Mediator -> Verifier).
  Level 2  Classify into the fine class within the predicted group, again
           using the full KF pipeline — but only when the group contains
           more than one class.  Solo groups skip Level 2.

Groups:
  Melanocytic       -> Melanoma | Melanocytic Nevus        (Level 2)
  Keratosis-type    -> Benign Keratosis | Actinic Keratosis (Level 2)
  Basal Cell Carcinoma  (solo)
  Vascular Lesion       (solo)
  Dermatofibroma        (solo)

Rationale:
  Flat 7-way classification collapses because Melanoma/Nevus dominate the
  model's prior, starving BKL/AK/BCC of correct predictions.  Coarse-to-fine
  first asks "melanocytic vs keratosis vs ..." — a much easier signal — then
  refines within the winning group where the competing classes are genuinely
  similar and the relevant features are class-specific.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path setup — insert hierarchical dir first, 2-way last (for rules/tools)
# ---------------------------------------------------------------------------

_HERE    = Path(__file__).resolve().parent
_KF_ROOT = _HERE.parents[4]
_MC_PY   = _HERE.parents[1] / "dermatology-multiclass" / "python"
_D2_PY   = _HERE.parents[1] / "dermatology" / "python"

for _p in (str(_KF_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_D2_PY) not in sys.path:
    sys.path.append(str(_D2_PY))

# ---------------------------------------------------------------------------
# Import hierarchical dataset FIRST (before multiclass can shadow it)
# ---------------------------------------------------------------------------

from dataset import (                           # noqa: E402 — must be early
    LEVEL1_GROUPS,
    LEVEL1_CATEGORY_SET_ID,
    LEVEL1_GROUP_NAMES,
    LEVEL2_SUBPROBLEMS,
    CLASS_TO_GROUP_ID,
    CLASS_TO_GROUP_NAME,
    GROUP_NAME_TO_ID,
    GROUP_TO_SOLO_CLASS,
)

# ---------------------------------------------------------------------------
# Import multiclass ensemble's run_ensemble under a unique module name
# so it uses its own agents/dataset without polluting sys.modules['ensemble']
# ---------------------------------------------------------------------------

import importlib.util as _ilu

def _load_mc_module(name: str, path: Path):
    spec = _ilu.spec_from_file_location(name, path)
    mod  = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Multiclass ensemble — uses multiclass agents/dataset internally
_mc_ens = _load_mc_module("_mc_ensemble_hierarchical", _MC_PY / "ensemble.py")
_run_ensemble_base = _mc_ens.run_ensemble

# Rules / tools from 2-way (appended to sys.path, so safe to import normally)
from rules import RuleEngine     # noqa: E402
from tools import ToolRegistry   # noqa: E402

DATASET_TAG   = "derm-ham10000"
MAX_REVISIONS = 1

_DEFAULT_RULES = {
    "l1":         str(_HERE / "rules_h_l1.json"),
    "melanocytic": str(_HERE / "rules_h_l2_melanocytic.json"),
    "keratosis":   str(_HERE / "rules_h_l2_keratosis.json"),
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_hierarchical_ensemble(
    task: dict,
    task_id: str = "unknown",
    rule_engine_l1: Optional[RuleEngine] = None,
    rule_engines_l2: Optional[dict[str, RuleEngine]] = None,
    tool_registry: Optional[ToolRegistry] = None,
    verbose: bool = True,
    dataset: str = "derm-ham10000",
    dataset_tag: str = DATASET_TAG,
    max_revisions: int = MAX_REVISIONS,
    test_mode: bool = False,
) -> dict:
    """Run the 2-level hierarchical ensemble on a single fine-grained task.

    Args:
        task: Dict with keys:
            test_image_path  str
            test_label       str   fine class name (e.g. "Melanoma")
            few_shot         dict  {class_name: [paths]}  — all 7 classes

    Returns:
        Result dict including per-level detail, aggregated cost, and a
        ``failure_level`` field ("l1", "l2", or None) for diagnosis.
    """
    if rule_engine_l1 is None:
        rule_engine_l1 = RuleEngine(path=_DEFAULT_RULES["l1"], dataset_tag=dataset_tag)
    if rule_engines_l2 is None:
        rule_engines_l2 = {
            gid: RuleEngine(path=_DEFAULT_RULES[gid], dataset_tag=dataset_tag)
            for gid in LEVEL2_SUBPROBLEMS
        }
    if tool_registry is None:
        tool_registry = ToolRegistry(dataset_tag=dataset_tag)

    correct_label   = task.get("test_label", "")
    true_group_id   = CLASS_TO_GROUP_ID.get(correct_label, "")
    true_group_name = CLASS_TO_GROUP_NAME.get(correct_label, "")

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    t_start = time.time()

    # ------------------------------------------------------------------
    # Build Level-1 task — group-level classification
    # One representative few-shot image per group (from any class in group)
    # ------------------------------------------------------------------
    l1_few_shot: dict[str, list[str]] = {}
    for grp in LEVEL1_GROUPS:
        for cls in grp["classes"]:
            paths = task.get("few_shot", {}).get(cls, [])
            if paths:
                l1_few_shot[grp["name"]] = paths[:1]
                break

    l1_task = {
        "category_set_id": LEVEL1_CATEGORY_SET_ID,
        "categories":      LEVEL1_GROUP_NAMES,
        "test_image_path": task["test_image_path"],
        "test_label":      true_group_name,
        "few_shot":        l1_few_shot,
    }

    log(f"\n{'='*60}")
    log(f"Task:       {task_id}")
    log(f"True class: {correct_label!r}  (group: {true_group_name!r})")
    log(f"\n--- LEVEL 1: group classification ---")
    log(f"    Groups: {', '.join(LEVEL1_GROUP_NAMES)}")

    l1_result = await _run_ensemble_base(
        l1_task,
        task_id=f"{task_id}_l1",
        rule_engine=rule_engine_l1,
        tool_registry=tool_registry,
        verbose=verbose,
        dataset=dataset,
        dataset_tag=dataset_tag,
        max_revisions=max_revisions,
        test_mode=test_mode,
    )

    l1_predicted_group = l1_result["predicted_label"]
    l1_correct         = (l1_predicted_group == true_group_name)
    predicted_group_id = GROUP_NAME_TO_ID.get(l1_predicted_group, "unknown")

    log(f"\nLevel 1: {l1_predicted_group!r}  (true: {true_group_name!r})  "
        f"{'CORRECT' if l1_correct else 'WRONG'}")

    # ------------------------------------------------------------------
    # Level 2 — fine classification within predicted group
    # ------------------------------------------------------------------
    l2_result       = None
    l2_correct      = None
    predicted_class = None
    failure_level   = None

    if predicted_group_id in GROUP_TO_SOLO_CLASS:
        # Solo group — map directly to the single class
        predicted_class = GROUP_TO_SOLO_CLASS[predicted_group_id]
        log(f"Level 2: skipped (solo group -> {predicted_class!r})")
        if not l1_correct:
            failure_level = "l1"

    elif predicted_group_id in LEVEL2_SUBPROBLEMS:
        sub         = LEVEL2_SUBPROBLEMS[predicted_group_id]
        l2_cats     = sub["categories"]
        l2_few_shot = {c: task.get("few_shot", {}).get(c, []) for c in l2_cats}

        l2_task = {
            "category_set_id": sub["category_set_id"],
            "categories":      l2_cats,
            "test_image_path": task["test_image_path"],
            "test_label":      correct_label,
            "few_shot":        l2_few_shot,
        }

        log(f"\n--- LEVEL 2: fine classification ---")
        log(f"    Classes: {', '.join(l2_cats)}")

        l2_result = await _run_ensemble_base(
            l2_task,
            task_id=f"{task_id}_l2",
            rule_engine=rule_engines_l2.get(predicted_group_id),
            tool_registry=tool_registry,
            verbose=verbose,
            dataset=dataset,
            dataset_tag=dataset_tag,
            max_revisions=max_revisions,
            test_mode=test_mode,
        )

        predicted_class = l2_result["predicted_label"]
        l2_correct      = (predicted_class == correct_label) if correct_label else None

        log(f"\nLevel 2: {predicted_class!r}  (true: {correct_label!r})  "
            f"{'CORRECT' if l2_correct else 'WRONG'}")

        if not l1_correct:
            failure_level = "l1"                # wrong group -> L1 fault
        elif not l2_correct:
            failure_level = "l2"                # right group, wrong class -> L2 fault

    else:
        # Fallback: unknown group — use group name as prediction
        predicted_class = l1_predicted_group
        log(f"Level 2: unknown group {predicted_group_id!r} — using group as prediction")
        failure_level = "l1"

    is_correct  = (predicted_class == correct_label) if correct_label else None
    duration_ms = int((time.time() - t_start) * 1000)

    # Aggregate costs
    l1_cost    = l1_result.get("cost_usd", 0.0)
    l2_cost    = l2_result.get("cost_usd", 0.0) if l2_result else 0.0
    total_cost = round(l1_cost + l2_cost, 6)
    total_calls = (l1_result.get("api_calls", 0) +
                   (l2_result.get("api_calls", 0) if l2_result else 0))

    log(f"\n{'='*60}")
    log(f"FINAL: {'CORRECT' if is_correct else 'WRONG'}  "
        f"predicted={predicted_class!r}  actual={correct_label!r}")
    log(f"Cost: ${total_cost:.4f}  |  API calls: {total_calls}  |  "
        f"Duration: {duration_ms/1000:.1f}s")

    return {
        "task_id":           task_id,
        "correct_label":     correct_label,
        "predicted_label":   predicted_class,
        "correct":           is_correct,
        "failure_level":     failure_level,
        # Level 1
        "true_group":        true_group_name,
        "predicted_group":   l1_predicted_group,
        "l1_correct":        l1_correct,
        "l1_confidence":     l1_result.get("confidence", 0.0),
        "l1_cost_usd":       l1_cost,
        "l1_api_calls":      l1_result.get("api_calls", 0),
        # Level 2
        "l2_run":            l2_result is not None,
        "l2_correct":        l2_correct,
        "l2_confidence":     l2_result.get("confidence", 0.0) if l2_result else None,
        "l2_cost_usd":       l2_cost,
        "l2_api_calls":      l2_result.get("api_calls", 0) if l2_result else 0,
        # Totals
        "cost_usd":          total_cost,
        "api_calls":         total_calls,
        "duration_ms":       duration_ms,
        "model":             l1_result.get("model", ""),
        "dataset":           dataset,
        # Slim sub-results (no raw feature records — keep output compact)
        "l1_detail":         _slim(l1_result),
        "l2_detail":         _slim(l2_result) if l2_result else None,
    }


def _slim(r: dict) -> dict:
    """Strip large fields from a sub-result to keep the output file manageable."""
    drop = {"feature_record", "decision", "verification", "rule_ids_fired"}
    return {k: v for k, v in r.items() if k not in drop}
