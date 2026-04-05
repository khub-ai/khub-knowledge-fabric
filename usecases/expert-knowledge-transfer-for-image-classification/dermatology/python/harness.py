"""
harness.py — CLI test runner for the KF derm-ham10000 dermoscopy ensemble.

Usage:
  python harness.py                                                   # run first pair, first image
  python harness.py --pair melanoma_vs_melanocytic_nevus              # single pair, all test images
  python harness.py --all                                             # all 3 pairs, all test images
  python harness.py --all --limit 30                                  # first 30 images across all pairs
  python harness.py --all --resume                                    # continue an interrupted run
  python harness.py --task-list ids.json                              # JSON list of task_ids to run
  python harness.py --all --mode test                                 # read-only, no learning
  python harness.py --prune                                           # maintenance audit, no task execution
  python harness.py --migrate                                         # import knowledge_base rules -> RuleEngine

Data directory default: C:\\_backup\\ml\\data\\DermaMNIST_HAM10000
Override with --data-dir.

Dataset tag: derm-ham10000  (override with --dataset-tag)
Output:      results.json   (override with --output)
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

# Force UTF-8 stdout/stderr on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Path setup — must happen before local imports
# ---------------------------------------------------------------------------
_HERE    = Path(__file__).parent
_KF_ROOT = _HERE.resolve().parents[4]
for _p in (str(_KF_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

import ensemble as _ensemble_mod
from ensemble import run_ensemble, DATASET_TAG
from rules import RuleEngine
from tools import ToolRegistry
from dataset import (
    load as load_ham10000,
    DEFAULT_DATA_DIR,
    HAM10000Dataset,
    CONFUSABLE_PAIRS,
)
import agents

console = Console()


# ---------------------------------------------------------------------------
# Task builder — creates one task dict per test image
# ---------------------------------------------------------------------------

def build_tasks(
    ds: HAM10000Dataset,
    pairs: list[dict],
    n_few_shot: int = 3,
    max_per_class: int | None = None,
    sk_only: bool = False,
) -> list[dict]:
    """Build task dicts for all test images in the given confusable pairs."""
    tasks = []
    for cp in pairs:
        pid = cp["pair_id"]
        class_a, class_b = cp["class_a"], cp["class_b"]
        dx_a, dx_b = cp["dx_a"], cp["dx_b"]

        few_shot_a = [str(img.file_path) for img in ds.sample_images(dx_a, n=n_few_shot, split="train")]
        few_shot_b = [str(img.file_path) for img in ds.sample_images(dx_b, n=n_few_shot, split="train")]

        if sk_only and dx_a == "bkl":
            test_a = ds.images_for_class_sk_only(dx_a, split="test")
        else:
            test_a = ds.images_for_class(dx_a, split="test")
        if sk_only and dx_b == "bkl":
            test_b = ds.images_for_class_sk_only(dx_b, split="test")
        else:
            test_b = ds.images_for_class(dx_b, split="test")

        if max_per_class is not None:
            test_a = test_a[:max_per_class]
            test_b = test_b[:max_per_class]

        for img in test_a:
            tasks.append({
                "pair_id": pid, "class_a": class_a, "class_b": class_b,
                "dx_a": dx_a, "dx_b": dx_b,
                "test_image_path": str(img.file_path),
                "test_label": class_a,
                "few_shot_a": few_shot_a, "few_shot_b": few_shot_b,
                "_image_id": img.image_id,
            })
        for img in test_b:
            tasks.append({
                "pair_id": pid, "class_a": class_a, "class_b": class_b,
                "dx_a": dx_a, "dx_b": dx_b,
                "test_image_path": str(img.file_path),
                "test_label": class_b,
                "few_shot_a": few_shot_a, "few_shot_b": few_shot_b,
                "_image_id": img.image_id,
            })
    return tasks


def task_id_for(task: dict) -> str:
    return f"{task['pair_id']}_{task['_image_id']}"


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def _save_results(
    output_path: Path,
    all_results: list[dict],
    model: str,
    dataset: str,
    rules: RuleEngine,
) -> None:
    """Atomically write results with per-pair and aggregate accuracy stats."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total   = len(all_results)
    correct = sum(1 for r in all_results if r.get("correct"))
    accuracy = correct / total if total else 0.0

    # Per-pair breakdown
    pair_stats: dict[str, dict] = {}
    for r in all_results:
        pid = r.get("pair_id", "unknown")
        s = pair_stats.setdefault(pid, {"correct": 0, "total": 0})
        s["total"] += 1
        if r.get("correct"):
            s["correct"] += 1
    for pid, s in pair_stats.items():
        s["accuracy"] = s["correct"] / s["total"] if s["total"] else 0.0

    total_cost  = sum(r.get("cost_usd", 0.0) for r in all_results)
    avg_ms      = sum(r.get("duration_ms", 0) for r in all_results) / max(total, 1)
    total_input = sum(r.get("input_tokens", 0) for r in all_results)
    total_out   = sum(r.get("output_tokens", 0) for r in all_results)
    total_cc    = sum(r.get("cache_creation_tokens", 0) for r in all_results)
    total_cr    = sum(r.get("cache_read_tokens", 0) for r in all_results)
    total_calls = sum(r.get("api_calls", 0) for r in all_results)
    run_ts      = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    payload = {
        "summary": {
            "correct":               correct,
            "total":                 total,
            "accuracy":              accuracy,
            "avg_ms":                avg_ms,
            "total_cost_usd":        round(total_cost, 6),
            "avg_cost_usd":          round(total_cost / max(total, 1), 6),
            "total_input_tokens":    total_input,
            "total_output_tokens":   total_out,
            "total_cache_creation":  total_cc,
            "total_cache_read":      total_cr,
            "total_api_calls":       total_calls,
            "model":                 model,
            "dataset":               dataset,
            "timestamp":             run_ts,
            "rules":                 rules.stats_summary(),
        },
        "per_pair": pair_stats,
        "tasks": all_results,
    }
    tmp = output_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, output_path)


def _save_failed(failed_path: Path, all_results: list[dict]) -> None:
    failed_ids = [r["task_id"] for r in all_results if not r.get("correct")]
    tmp = failed_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(failed_ids, indent=2), encoding="utf-8")
    os.replace(tmp, failed_path)


# ---------------------------------------------------------------------------
# Prune audit
# ---------------------------------------------------------------------------

def run_prune_audit(rules: RuleEngine, min_fired: int = 10) -> None:
    console.print(Panel(
        f"[bold]Prune Audit[/bold]\n"
        f"Namespace: [cyan]{rules.dataset_tag}[/cyan]\n"
        f"Performance threshold: fires >= {min_fired}, 0 successes -> deprecate",
        title="--prune"
    ))
    before = rules.stats_summary()
    changed = rules.auto_deprecate(min_fired=min_fired)
    after = rules.stats_summary()

    t = Table(title="Pruning Results")
    t.add_column("Metric"); t.add_column("Before"); t.add_column("After")
    t.add_row("Active",     str(before["active"]),     str(after["active"]))
    t.add_row("Flagged",    str(before["flagged"]),    str(after["flagged"]))
    t.add_row("Deprecated", str(before["deprecated"]), str(after["deprecated"]))
    console.print(t)

    if changed:
        console.print(f"[yellow]Changed: {changed}[/yellow]")
    else:
        console.print("[green]No rules changed.[/green]")
    console.print(f"Rules file: [bold]{rules.path}[/bold]")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KF Dermatology (HAM10000) Ensemble Harness")
    p.add_argument("--data-dir",    default=str(DEFAULT_DATA_DIR),
                   help="Path to DermaMNIST_HAM10000 folder")
    p.add_argument("--pair",        default="",
                   help="Run a single pair by ID (e.g. melanoma_vs_melanocytic_nevus)")
    p.add_argument("--task-id",     dest="task_id", default="",
                   help="Run a single task by task_id (pair_id + image_id)")
    p.add_argument("--task-list",   dest="task_list", default="",
                   help="JSON list of task_ids to run")
    p.add_argument("--all",         action="store_true",
                   help="Run all 3 confusable pairs")
    p.add_argument("--limit",       type=int, default=None,
                   help="Cap total tasks this run")
    p.add_argument("--max-per-class", dest="max_per_class", type=int, default=None,
                   help="Max test images per class per pair (default: all)")
    p.add_argument("--n-few-shot",  dest="n_few_shot", type=int, default=3,
                   help="Few-shot images per class for verifier (default: 3)")
    p.add_argument("--sk-only", dest="sk_only", action="store_true",
               help="For bkl class: restrict to histology-confirmed images only "
                    "(proxy for seborrheic keratosis, excludes lichenoid keratosis variants)")
    p.add_argument("--output",      default="results.json")
    p.add_argument("--failed-output", dest="failed_output", default="",
                   help="Write failed task_ids to this JSON file")
    p.add_argument("--resume",      action="store_true",
                   help="Skip task_ids already in --output")
    p.add_argument("--mode",        choices=["train", "test"], default="train",
                   help="train: learn and persist rules. test: read-only.")
    p.add_argument("--baseline",     choices=["zero_shot", "few_shot"], default="",
               help="Run a baseline instead of the full pipeline (zero_shot or few_shot)")
    p.add_argument("--max-revisions", dest="max_revisions", type=int, default=None,
                   help="Override MAX_REVISIONS (default: 1)")
    p.add_argument("--dataset",     default="derm-ham10000")
    p.add_argument("--dataset-tag", dest="dataset_tag", default=DATASET_TAG)
    p.add_argument("--rules",       default="",
                   help="Path to rules.json (default: auto)")
    p.add_argument("--quiet",       action="store_true")
    p.add_argument("--prompts",     action="store_true",
                   help="Print full prompts sent to each agent")
    p.add_argument("--prune",       action="store_true",
                   help="Run maintenance prune audit and exit")
    p.add_argument("--prune-threshold", dest="prune_threshold", type=int, default=10)
    p.add_argument("--migrate",     action="store_true",
                   help="Import knowledge_base/*.json rules into RuleEngine, then exit. "
                        "Runs migrate_rules.py. Pass --dry-run to preview without saving.")
    p.add_argument("--dry-run",     action="store_true",
                   help="With --migrate: preview rules without saving")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    args = parse_args()
    verbose = not args.quiet
    test_mode = args.mode == "test"
    baseline_mode = args.baseline  # "" | "zero_shot" | "few_shot"
    agents.SHOW_PROMPTS = args.prompts

    if args.max_revisions is not None:
        _ensemble_mod.MAX_REVISIONS = args.max_revisions

    # API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        key_file = Path("P:/_access/Security/api_keys.env")
        if key_file.exists():
            for line in key_file.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()
                    break
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY not set[/red]")
        sys.exit(1)

    # Rule engine
    rules_path = args.rules or None
    rules = RuleEngine(rules_path, dataset_tag=args.dataset_tag)

    # --migrate: import knowledge_base rules
    if args.migrate:
        import migrate_rules
        migrate_rules.run(rules, dry_run=args.dry_run)
        return

    # --prune: maintenance audit
    if args.prune:
        run_prune_audit(rules, min_fired=args.prune_threshold)
        return

    # Tool registry
    tool_reg = ToolRegistry(
        read_only=test_mode,
        dataset_tag=args.dataset_tag,
    )

    # Load HAM10000 dataset
    console.print(f"[dim]Loading HAM10000 from {args.data_dir}...[/dim]")
    try:
        ds = load_ham10000(args.data_dir)
    except Exception as e:
        console.print(f"[red]Failed to load dataset: {e}[/red]")
        console.print(f"[red]Check --data-dir, should point to DermaMNIST_HAM10000 folder[/red]")
        sys.exit(1)

    # Select pairs
    all_pairs = CONFUSABLE_PAIRS
    if args.pair:
        selected_pairs = [cp for cp in all_pairs if cp["pair_id"] == args.pair]
        if not selected_pairs:
            console.print(f"[red]Pair not found: {args.pair!r}[/red]")
            console.print(f"Available: {', '.join(cp['pair_id'] for cp in all_pairs)}")
            sys.exit(1)
    elif args.all:
        selected_pairs = all_pairs
    else:
        selected_pairs = [all_pairs[0]]

    # Build all candidate tasks
    all_tasks = build_tasks(
        ds,
        selected_pairs,
        n_few_shot=args.n_few_shot,
        max_per_class=args.max_per_class,
        sk_only=args.sk_only,
    )

    # --task-id: single image
    if args.task_id:
        all_tasks = [t for t in all_tasks if task_id_for(t) == args.task_id]
        if not all_tasks:
            console.print(f"[red]task_id not found: {args.task_id!r}[/red]")
            sys.exit(1)

    # --task-list: explicit list of task_ids
    if args.task_list:
        tl_path = Path(args.task_list)
        if not tl_path.exists():
            console.print(f"[red]--task-list file not found: {tl_path}[/red]")
            sys.exit(1)
        id_set = set(json.loads(tl_path.read_text(encoding="utf-8")))
        all_tasks = [t for t in all_tasks if task_id_for(t) in id_set]
        console.print(f"[dim]  --task-list: {len(all_tasks)} task(s) selected[/dim]")

    # Resume: skip already-done tasks
    output_path = Path(args.output)
    all_results: list[dict] = []
    completed_ids: set[str] = set()
    correct_count = 0

    if args.resume and output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        all_results = existing.get("tasks", [])
        completed_ids = {r["task_id"] for r in all_results}
        correct_count = sum(1 for r in all_results if r.get("correct"))
        console.print(
            f"[dim]  Resuming: {len(completed_ids)} tasks already done "
            f"({correct_count} correct)[/dim]"
        )
        all_tasks = [t for t in all_tasks if task_id_for(t) not in completed_ids]

    # Apply --limit
    if args.limit is not None:
        all_tasks = all_tasks[:args.limit]

    total_tasks = len(all_tasks)
    if total_tasks == 0:
        console.print("[yellow]No tasks to run.[/yellow]")
        return

    from agents import DEFAULT_MODEL, get_cost_tracker
    _pair_label = "all" if args.all else (args.pair or selected_pairs[0]["pair_id"])
    _mode_label = "[red]test[/red] (read-only)" if test_mode else "[green]train[/green] (learning)"
    console.print(Panel(
        f"[bold]KF Dermatology (HAM10000)[/bold]\n"
        f"Model:     [cyan]{DEFAULT_MODEL}[/cyan]\n"
        f"Tasks:     {total_tasks}  (pair={_pair_label})\n"
        f"Mode:      {_mode_label}\n"
        + (f"Baseline:  {baseline_mode}\n" if baseline_mode else "")
        + f"Few-shot:  {args.n_few_shot} images/class\n"
        f"NS tag:    [cyan]{args.dataset_tag}[/cyan]\n"
        f"Rules:     {rules.path}  {rules.stats_summary()}\n"
        f"Output:    {args.output}",
        title="Harness"
    ))

    run_results: list[dict] = []

    for i, task in enumerate(all_tasks, 1):
        tid = task_id_for(task)
        console.rule(f"[{i}/{total_tasks}] {tid}")

        if baseline_mode:
            bl_decision, bl_ms = await agents.run_baseline(task, mode=baseline_mode)
            bl_label = bl_decision.get("label", "uncertain")
            bl_correct_label = task["test_label"]
            bl_correct = bl_label == bl_correct_label
            if bl_correct:
                correct_count += 1
            meta = {
                "task_id":        tid,
                "pair_id":        task["pair_id"],
                "predicted_label": bl_label,
                "correct_label":   bl_correct_label,
                "correct":         bl_correct,
                "confidence":      bl_decision.get("confidence", 0.0),
                "reasoning":       bl_decision.get("reasoning", ""),
                "duration_ms":     bl_ms,
                "cost_usd":        get_cost_tracker().total_cost_usd - sum(r.get("cost_usd", 0) for r in all_results + run_results),
                "api_calls":       1,
                "model":           DEFAULT_MODEL,
                "baseline_mode":   baseline_mode,
            }
        else:
            meta = await run_ensemble(
                task=task,
                task_id=tid,
                rule_engine=rules,
                tool_registry=tool_reg,
                verbose=verbose,
                dataset=args.dataset,
                dataset_tag=args.dataset_tag,
                max_revisions=_ensemble_mod.MAX_REVISIONS,
                test_mode=test_mode,
            )

        if meta.get("correct") and not baseline_mode:
            correct_count += 1

        all_results.append(meta)
        run_results.append(meta)

        _save_results(output_path, all_results, meta.get("model", "?"), args.dataset, rules)
        if args.failed_output:
            _save_failed(Path(args.failed_output), all_results)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    rs = run_results
    total      = len(rs)
    n_correct  = sum(1 for r in rs if r.get("correct"))
    accuracy   = n_correct / total if total else 0.0
    avg_ms     = sum(r.get("duration_ms", 0) for r in rs) / max(total, 1)
    total_cost = sum(r.get("cost_usd", 0.0) for r in rs)
    total_in   = sum(r.get("input_tokens", 0) for r in rs)
    total_out  = sum(r.get("output_tokens", 0) for r in rs)
    total_cc   = sum(r.get("cache_creation_tokens", 0) for r in rs)
    total_cr   = sum(r.get("cache_read_tokens", 0) for r in rs)
    total_calls= sum(r.get("api_calls", 0) for r in rs)

    # Per-pair accuracy
    pair_acc: dict[str, dict] = {}
    for r in rs:
        pid = r.get("pair_id", "?")
        s = pair_acc.setdefault(pid, {"correct": 0, "total": 0})
        s["total"] += 1
        if r.get("correct"):
            s["correct"] += 1

    t = Table(title="Run Summary (this invocation)")
    t.add_column("Metric"); t.add_column("Value")
    t.add_row("Tasks run",         str(total))
    t.add_row("Correct",           f"{n_correct}/{total}  ({accuracy*100:.1f}%)")
    t.add_row("Avg duration",      f"{avg_ms/1000:.1f}s")
    t.add_row("Total cost (USD)",  f"${total_cost:.4f}")
    t.add_row("Avg cost / task",   f"${total_cost/max(total,1):.4f}")
    t.add_row("Input tokens",      f"{total_in:,}")
    t.add_row("Output tokens",     f"{total_out:,}")
    t.add_row("Cache create",      f"{total_cc:,}")
    t.add_row("Cache read",        f"{total_cr:,}")
    t.add_row("Total API calls",   f"{total_calls:,}")
    t.add_row("Model",             DEFAULT_MODEL)
    t.add_row("Dataset",           args.dataset)
    rs_summary = rules.stats_summary()
    t.add_row("Rules (active)",    str(rs_summary["active"]))
    t.add_row("Rules (total)",     str(rs_summary["total"]))
    console.print(t)

    if pair_acc:
        pt = Table(title="Per-Pair Accuracy")
        pt.add_column("Pair"); pt.add_column("Correct"); pt.add_column("Total"); pt.add_column("Accuracy")
        for pid, s in sorted(pair_acc.items()):
            pct = s["correct"] / s["total"] * 100 if s["total"] else 0
            pt.add_row(pid, str(s["correct"]), str(s["total"]), f"{pct:.1f}%")
        console.print(pt)

    console.print(f"Results written to [bold]{args.output}[/bold]")
    if args.failed_output:
        n_failed = sum(1 for r in all_results if not r.get("correct"))
        console.print(f"Failed tasks ({n_failed}) written to [bold]{args.failed_output}[/bold]")


if __name__ == "__main__":
    asyncio.run(main())
