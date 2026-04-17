"""
harness.py — CLI test runner for the hierarchical (coarse-to-fine) KF ensemble.

Two-level pipeline:
  Level 1  Classify image into one of 5 coarse groups.
  Level 2  Classify into fine class within the predicted group (if multi-class).

Usage:
  python harness.py                              # 3 images per class (21 total)
  python harness.py --max-per-class 10           # 10 per class (70 total)
  python harness.py --all                        # all test images
  python harness.py --class Melanoma             # single class only
  python harness.py --mode test                  # read-only, no learning
  python harness.py --resume                     # continue interrupted run
  python harness.py --model qwen/qwen2.5-vl-72b-instruct

Output:  results_hierarchical.json  (override: --output)
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import io
import json
import os
import sys
import time
from pathlib import Path

# UTF-8 stdout/stderr on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_HERE    = Path(__file__).resolve().parent
_KF_ROOT = _HERE.parents[4]
_MC_PY   = _HERE.parents[1] / "dermatology-multiclass" / "python"
_D2_PY   = _HERE.parents[1] / "dermatology" / "python"

for _p in (str(_KF_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_D2_PY) not in sys.path:
    sys.path.append(str(_D2_PY))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Local modules (hierarchical dir wins for dataset.py)
from dataset import (
    ALL_CLASSES,
    LEVEL1_GROUPS,
    LEVEL1_GROUP_NAMES,
    LEVEL2_SUBPROBLEMS,
    CLASS_TO_GROUP_NAME,
    CLASS_TO_GROUP_ID,
    GROUP_TO_SOLO_CLASS,
)
from ensemble import run_hierarchical_ensemble, DATASET_TAG
from rules import RuleEngine
from tools import ToolRegistry

# Multiclass dataset loader (loaded via importlib to avoid dataset.py name clash)
import importlib.util as _ilu

def _load_mc(name, relpath):
    spec = _ilu.spec_from_file_location(name, _MC_PY / relpath)
    mod  = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_mc_ds = _load_mc("_mc_dataset_h", "dataset.py")

load_ham10000    = _mc_ds.load
DEFAULT_DATA_DIR = _mc_ds.DEFAULT_DATA_DIR
HAM10000Dataset  = _mc_ds.HAM10000Dataset

# NOTE: do NOT separately load multiclass agents here.
# ensemble.py (imported above) already loaded multiclass agents into
# sys.modules['agents'] when it loaded the mc ensemble via importlib.
# We retrieve that exact instance in main() to call _set_active_model on it,
# ensuring ACTIVE_MODEL is set on the same module object the ensemble uses.

console = Console()

DEFAULT_OUTPUT_PATH = "results_hierarchical.json"
DEFAULT_N_FEW_SHOT  = 2    # per fine class — L1 verifier uses 1/group, L2 uses all per class


# ---------------------------------------------------------------------------
# Task builder
# ---------------------------------------------------------------------------

def build_tasks(
    ds: HAM10000Dataset,
    n_few_shot: int = DEFAULT_N_FEW_SHOT,
    max_per_class: int | None = None,
    filter_class: str = "",
) -> list[dict]:
    """Build hierarchical task dicts for all test images.

    few_shot is keyed by FINE class name (all 7).
    The ensemble builds group-level few_shot internally from this.
    """
    few_shot: dict[str, list[str]] = {}
    for c in ALL_CLASSES:
        imgs = ds.sample_images(c["dx"], n=n_few_shot, split="train", seed=42)
        few_shot[c["name"]] = [str(img.file_path) for img in imgs]

    tasks: list[dict] = []
    for c in ALL_CLASSES:
        if filter_class and c["name"] != filter_class:
            continue
        test_imgs = ds.images_for_class(c["dx"], split="test")
        if max_per_class is not None:
            test_imgs = test_imgs[:max_per_class]
        for img in test_imgs:
            tasks.append({
                "test_image_path": str(img.file_path),
                "test_label":      c["name"],
                "few_shot":        few_shot,
                "_image_id":       img.image_id,
                "_dx":             c["dx"],
            })
    return tasks


def task_id_for(task: dict) -> str:
    dx  = task.get("_dx", "unk")
    iid = task.get("_image_id", "?")
    return f"derm_h_{dx}_{iid}"


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def _save_results(
    output_path: Path,
    all_results: list[dict],
    model: str,
    dataset: str,
    rule_engine_l1: RuleEngine,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total    = len(all_results)
    correct  = sum(1 for r in all_results if r.get("correct"))
    accuracy = correct / total if total else 0.0

    # Per-class breakdown (fine class)
    class_stats: dict[str, dict] = {}
    for r in all_results:
        lbl = r.get("correct_label", "unknown")
        s   = class_stats.setdefault(lbl, {"correct": 0, "total": 0})
        s["total"] += 1
        if r.get("correct"):
            s["correct"] += 1
    for s in class_stats.values():
        s["accuracy"] = s["correct"] / s["total"] if s["total"] else 0.0

    # Per-group L1 breakdown
    group_stats: dict[str, dict] = {}
    for r in all_results:
        grp = r.get("true_group", "unknown")
        s   = group_stats.setdefault(grp, {"correct": 0, "total": 0})
        s["total"] += 1
        if r.get("l1_correct"):
            s["correct"] += 1
    for s in group_stats.values():
        s["accuracy"] = s["correct"] / s["total"] if s["total"] else 0.0

    # Fine confusion matrix
    confusion: dict[str, dict[str, int]] = {}
    for r in all_results:
        pred  = r.get("predicted_label", "?")
        truth = r.get("correct_label", "?")
        confusion.setdefault(truth, {}).setdefault(pred, 0)
        confusion[truth][pred] += 1

    # Failure attribution
    l1_failures = sum(1 for r in all_results if r.get("failure_level") == "l1")
    l2_failures = sum(1 for r in all_results if r.get("failure_level") == "l2")

    total_cost  = sum(r.get("cost_usd",    0.0) for r in all_results)
    l1_cost     = sum(r.get("l1_cost_usd", 0.0) for r in all_results)
    l2_cost     = sum(r.get("l2_cost_usd", 0.0) for r in all_results)
    avg_ms      = sum(r.get("duration_ms",  0)   for r in all_results) / max(total, 1)
    total_calls = sum(r.get("api_calls",    0)   for r in all_results)
    run_ts      = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    payload = {
        "summary": {
            "correct":          correct,
            "total":            total,
            "accuracy":         accuracy,
            "l1_failures":      l1_failures,
            "l2_failures":      l2_failures,
            "avg_ms":           avg_ms,
            "total_cost_usd":   round(total_cost, 6),
            "l1_cost_usd":      round(l1_cost, 6),
            "l2_cost_usd":      round(l2_cost, 6),
            "avg_cost_usd":     round(total_cost / max(total, 1), 6),
            "total_api_calls":  total_calls,
            "model":            model,
            "dataset":          dataset,
            "timestamp":        run_ts,
            "l1_rules":         rule_engine_l1.stats_summary(),
        },
        "per_class":  class_stats,
        "per_group":  group_stats,
        "confusion":  confusion,
        "tasks":      all_results,
    }

    tmp = output_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, output_path)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KF Dermatology Hierarchical (2-level) Ensemble Harness"
    )
    p.add_argument("--data-dir",      default=str(DEFAULT_DATA_DIR))
    p.add_argument("--all",           action="store_true",
                   help="Run all test images (overrides --max-per-class only when both given)")
    p.add_argument("--class",         dest="filter_class", default="",
                   help="Run only this fine class name")
    p.add_argument("--max-per-class", dest="max_per_class", type=int, default=3)
    p.add_argument("--n-few-shot",    dest="n_few_shot", type=int, default=DEFAULT_N_FEW_SHOT)
    p.add_argument("--output",        default=DEFAULT_OUTPUT_PATH)
    p.add_argument("--resume",        action="store_true")
    p.add_argument("--mode",          choices=["train", "test"], default="train")
    p.add_argument("--max-revisions", dest="max_revisions", type=int, default=None)
    p.add_argument("--dataset",       default="derm-ham10000")
    p.add_argument("--dataset-tag",   dest="dataset_tag", default=DATASET_TAG)
    p.add_argument("--rules-l1",      dest="rules_l1",
                   default=str(_HERE / "rules_h_l1.json"),
                   help="Rules file for Level-1 group classification")
    p.add_argument("--rules-l2-melanocytic", dest="rules_l2_melanocytic",
                   default=str(_HERE / "rules_h_l2_melanocytic.json"))
    p.add_argument("--rules-l2-keratosis",   dest="rules_l2_keratosis",
                   default=str(_HERE / "rules_h_l2_keratosis.json"))
    p.add_argument("--model",         default="claude-sonnet-4-6")
    p.add_argument("--quiet",         action="store_true")
    p.add_argument("--prompts",       action="store_true")
    p.add_argument(
        "--curated-refs",
        dest="curated_refs",
        default=str(_HERE / "curated_references.json"),
        help=(
            "JSON file mapping group names to canonical reference image paths "
            "(produced by curate_references.py).  When present and non-empty, "
            "Level-1 routing uses the visual-similarity router instead of the full "
            "KF Observer→Mediator→Verifier pipeline.  Pass --curated-refs='' to "
            "force the full KF L1 pipeline regardless."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    args      = parse_args()
    verbose   = not args.quiet
    test_mode = args.mode == "test"

    # Set model on the exact agents module instance the ensemble uses.
    # ensemble.py loads multiclass agents into sys.modules['agents']; we must
    # call _set_active_model on THAT instance, not a separately loaded copy.
    _agents = sys.modules.get("agents")
    if _agents is None:
        raise RuntimeError("multiclass agents module not found in sys.modules — "
                           "ensure ensemble.py was imported before main() runs")
    _agents._set_show_prompts(args.prompts)
    _agents._set_active_model(args.model)
    _agents._set_default_model(args.model)

    from ensemble import MAX_REVISIONS as _MR
    import ensemble as _ens_mod
    if args.max_revisions is not None:
        _ens_mod.MAX_REVISIONS = args.max_revisions

    # API keys
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_KF_ROOT / ".env", override=False)

    # Curated canonical references for L1 visual-similarity router
    curated_refs: dict[str, str] = {}
    curated_refs_path = Path(args.curated_refs) if args.curated_refs else None
    if curated_refs_path and curated_refs_path.exists():
        try:
            curated_refs = json.loads(curated_refs_path.read_text(encoding="utf-8"))
            console.print(
                f"[dim]Curated references loaded: {len(curated_refs)} groups "
                f"from {curated_refs_path.name}[/dim]"
            )
        except Exception as exc:
            console.print(f"[yellow]Warning: could not load curated refs: {exc}[/yellow]")
    elif args.curated_refs:
        console.print(
            f"[yellow]Curated refs file not found: {args.curated_refs}\n"
            f"  → Run curate_references.py first to generate it.\n"
            f"  → Falling back to full KF pipeline for Level-1 routing.[/yellow]"
        )

    # Data
    ds = load_ham10000(args.data_dir)
    max_pc = None if args.all else args.max_per_class
    all_tasks = build_tasks(ds, n_few_shot=args.n_few_shot,
                            max_per_class=max_pc, filter_class=args.filter_class)
    total_tasks = len(all_tasks)

    # Rule engines
    rule_engine_l1 = RuleEngine(path=args.rules_l1, dataset_tag=args.dataset_tag)
    rule_engines_l2 = {
        "melanocytic": RuleEngine(path=args.rules_l2_melanocytic,
                                  dataset_tag=args.dataset_tag),
        "keratosis":   RuleEngine(path=args.rules_l2_keratosis,
                                  dataset_tag=args.dataset_tag),
    }

    tool_registry = ToolRegistry(read_only=test_mode, dataset_tag=args.dataset)

    output_path  = Path(args.output)
    all_results: list[dict] = []
    completed_ids: set[str] = set()
    correct_count = 0

    if args.resume and output_path.exists():
        existing      = json.loads(output_path.read_text(encoding="utf-8"))
        all_results   = existing.get("tasks", [])
        completed_ids = {r["task_id"] for r in all_results}
        correct_count = sum(1 for r in all_results if r.get("correct"))
        console.print(f"[dim]Resuming: {len(completed_ids)} tasks already done.[/dim]")

    _scope   = args.filter_class or "all 7 classes"
    _l1_mode = (
        f"visual-similarity router ({len(curated_refs)} curated refs)"
        if curated_refs
        else "full KF pipeline (Observer→Mediator→Verifier)"
    )
    console.print(Panel(
        f"[bold]KF Dermatology Hierarchical — 2-Level Ensemble[/bold]\n"
        f"Model:       [cyan]{args.model}[/cyan]\n"
        f"Tasks:       {total_tasks}  (scope={_scope})\n"
        f"Mode:        {'test (read-only)' if test_mode else 'train (learning)'}\n"
        f"Few-shot:    {args.n_few_shot} images/fine-class\n"
        f"Level 1:     {len(LEVEL1_GROUP_NAMES)} groups — {', '.join(LEVEL1_GROUP_NAMES)}\n"
        f"L1 routing:  {_l1_mode}\n"
        + "Level 2:     " + ", ".join(f"{k}: {v['categories']}" for k,v in LEVEL2_SUBPROBLEMS.items()),
        title="Hierarchical Ensemble", border_style="blue"
    ))

    t_run_start = time.time()

    for i, task in enumerate(all_tasks, 1):
        tid = task_id_for(task)
        if tid in completed_ids:
            continue

        console.rule(f"[{i}/{total_tasks}] {tid}")

        try:
            meta = await run_hierarchical_ensemble(
                task,
                task_id=tid,
                rule_engine_l1=rule_engine_l1,
                rule_engines_l2=rule_engines_l2,
                tool_registry=tool_registry,
                verbose=verbose,
                dataset=args.dataset,
                dataset_tag=args.dataset_tag,
                test_mode=test_mode,
                curated_refs=curated_refs,
            )
        except Exception as exc:
            console.print(f"[red]ERROR on {tid}: {exc}[/red]")
            import traceback; traceback.print_exc()
            continue

        if meta.get("correct"):
            correct_count += 1

        all_results.append(meta)
        completed = len(all_results)
        acc = correct_count / completed if completed else 0
        console.print(
            f"  {'[green]CORRECT[/green]' if meta.get('correct') else '[red]WRONG[/red]'}  "
            f"predicted={meta.get('predicted_label','?')!r}  "
            f"actual={meta.get('correct_label','?')!r}  "
            f"(L1={'OK' if meta.get('l1_correct') else 'FAIL'}  "
            f"L2={'OK' if meta.get('l2_correct') else ('SKIP' if not meta.get('l2_run') else 'FAIL')})  "
            f"acc={acc:.1%}"
        )

        _save_results(output_path, all_results, args.model, args.dataset, rule_engine_l1)

    # Final summary table
    elapsed = time.time() - t_run_start
    total   = len(all_results)
    correct = sum(1 for r in all_results if r.get("correct"))

    t = Table(title="Hierarchical Ensemble — Final Results", show_header=True)
    t.add_column("Metric", style="bold")
    t.add_column("Value")
    t.add_row("Overall accuracy",
              f"{correct}/{total} ({correct/total:.1%})" if total else "—")
    t.add_row("L1 failures",
              str(sum(1 for r in all_results if r.get("failure_level") == "l1")))
    t.add_row("L2 failures",
              str(sum(1 for r in all_results if r.get("failure_level") == "l2")))
    t.add_row("Total cost",
              f"${sum(r.get('cost_usd',0) for r in all_results):.4f}")
    t.add_row("Total API calls",
              str(sum(r.get("api_calls", 0) for r in all_results)))
    t.add_row("Model", args.model)
    t.add_row("Elapsed", f"{elapsed:.0f}s")

    # Per-class accuracy
    from collections import defaultdict
    pc: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in all_results:
        lbl = r.get("correct_label", "?")
        pc[lbl]["total"] += 1
        if r.get("correct"):
            pc[lbl]["correct"] += 1

    console.print(t)
    console.print("\n[bold]Per-class accuracy:[/bold]")
    for cls in sorted(pc):
        d = pc[cls]
        pct = 100 * d["correct"] / d["total"] if d["total"] else 0
        grp = CLASS_TO_GROUP_NAME.get(cls, "?")
        bar = "█" * d["correct"] + "░" * (d["total"] - d["correct"])
        console.print(f"  {cls:<28} [{grp:<20}]  {d['correct']}/{d['total']} ({pct:.0f}%)  {bar}")

    console.print(f"\nResults saved to: [cyan]{output_path}[/cyan]")


if __name__ == "__main__":
    asyncio.run(main())
