"""
harness.py — CLI test runner for the Python ARC-AGI ensemble.

Usage:
  python harness.py                          # run first task
  python harness.py --task-id 1e0a9b12       # specific task
  python harness.py --limit 10               # first 10 tasks
  python harness.py --limit 5 --offset 20   # tasks 21-25
  python harness.py --all                    # entire dataset
  python harness.py --all --resume           # resume interrupted run
  python harness.py --all --skip-ids v1_ids.json   # v2-only tasks
  python harness.py --all --failed-output failed.json  # track failures
  python harness.py --task-list failed.json --human    # retry failures
  python harness.py --human                  # enable human-in-the-loop
  python harness.py --charts                 # save charts per task
  python harness.py --output results.json    # custom output file

Data directory default: C:/_backup/arctest2025/data/training
Override with --data-dir.  For ARC-AGI-v2, point --data-dir at the v2
training folder (same file naming convention as v1).
"""

from __future__ import annotations
import argparse
import asyncio
import datetime
import json
import os
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# ---------------------------------------------------------------------------
# Resolve paths before importing local modules
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import ensemble
from ensemble import run_ensemble
from rules import RuleEngine
from tools import ToolRegistry
from metadata import TaskMetadata, compute_outcome
from visualize import save_all_charts
import agents
from agents import DEFAULT_MODEL

console = Console()

DEFAULT_DATA_DIR = "C:/_backup/arctest2025/data/training"
DEFAULT_OUTPUT   = "results.json"


# ---------------------------------------------------------------------------
# Incremental save (called after every task)
# ---------------------------------------------------------------------------

def _save_results(
    output_path: Path,
    all_results: list[dict],
    model: str,
    dataset: str,
    rules: "RuleEngine",
) -> None:
    """Atomically write results with up-to-date summary stats."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total        = len(all_results)
    correct      = sum(1 for r in all_results if r.get("correct"))
    accuracy     = correct / total if total > 0 else 0.0
    avg_ms       = sum(r["duration_ms"] for r in all_results) / max(total, 1)
    conv_rate    = sum(1 for r in all_results if r.get("converged")) / max(total, 1)
    total_cost         = sum(r.get("cost_usd", 0.0) for r in all_results)
    total_tokens       = sum(r.get("input_tokens", 0) + r.get("output_tokens", 0) for r in all_results)
    total_cache_creation = sum(r.get("cache_creation_tokens", 0) for r in all_results)
    total_cache_read     = sum(r.get("cache_read_tokens", 0) for r in all_results)
    hints_used         = sum(1 for r in all_results if r.get("human_hints"))
    run_ts             = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    payload = {
        "summary": {
            "correct":               correct,
            "total":                 total,
            "accuracy":              accuracy,
            "avg_ms":                avg_ms,
            "conv_rate":             conv_rate,
            "total_cost_usd":        round(total_cost, 6),
            "avg_cost_usd":          round(total_cost / max(total, 1), 6),
            "total_tokens":          total_tokens,
            "total_cache_creation":  total_cache_creation,
            "total_cache_read":      total_cache_read,
            "hints_used":            hints_used,
            "model":          model,
            "dataset":        dataset,
            "timestamp":      run_ts,
            "rules":          rules.stats_summary(),
        },
        "tasks": all_results,
    }
    tmp = output_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, output_path)


def run_prune_audit(
    rules: "RuleEngine",
    min_fired: int = 10,
    redundancy_threshold: float = 0.5,
) -> None:
    """Run maintenance pruning and print a detailed report. Does not run tasks."""
    console.print(Panel(
        f"[bold]Prune Audit[/bold]\n"
        f"Namespace: [cyan]{rules.dataset_tag}[/cyan]\n"
        f"Performance threshold: fires >= {min_fired}, 0 successes → deprecate\n"
        f"Staleness: 50 tasks seen, 0 fires → flag  |  100 → deprecate\n"
        f"Redundancy Jaccard >= {redundancy_threshold:.0%}",
        title="--prune"
    ))

    before = rules.stats_summary()

    # --- Performance / staleness pruning ---
    changed = rules.auto_deprecate(min_fired=min_fired)

    after = rules.stats_summary()
    n_deprecated = after["deprecated"] - before["deprecated"]
    n_flagged    = after["flagged"]    - before["flagged"]

    table = Table(title="Pruning Results")
    table.add_column("Metric")
    table.add_column("Before")
    table.add_column("After")
    table.add_row("Active",     str(before["active"]),     str(after["active"]))
    table.add_row("Flagged",    str(before["flagged"]),    str(after["flagged"]))
    table.add_row("Deprecated", str(before["deprecated"]), str(after["deprecated"]))
    table.add_row("Archived",   str(before["archived"]),   str(after["archived"]))
    console.print(table)

    if changed:
        console.print(f"\n[yellow]Changed rules ({len(changed)}):[/yellow]")
        for rid in changed:
            r = rules.get(rid)
            if r:
                reason = r.get("deprecated_reason") or r.get("flagged_reason", "")
                console.print(f"  [{r['status'].upper():12}] {rid}  — {reason}")
    else:
        console.print("[green]No rules changed.[/green]")

    # --- Redundancy detection ---
    pairs = rules.find_redundant_pairs(threshold=redundancy_threshold)
    if pairs:
        console.print(f"\n[yellow]Redundant rule pairs (Jaccard >= {redundancy_threshold:.0%}):[/yellow]")
        rtable = Table()
        rtable.add_column("Rule A")
        rtable.add_column("Rule B")
        rtable.add_column("Jaccard")
        rtable.add_column("Shared / Union")
        rtable.add_column("Suggestion")
        for p in pairs:
            rtable.add_row(
                p["rule_a"], p["rule_b"],
                f"{p['jaccard']:.2f}",
                f"{p['shared']} / {p['union']}",
                p["suggestion"],
            )
        console.print(rtable)
    else:
        console.print("[green]No redundant rule pairs found.[/green]")

    console.print(f"\nRules file: [bold]{rules.path}[/bold]")


def _save_failed(failed_path: Path, all_results: list[dict]) -> None:
    """Atomically write/update a JSON list of task IDs where correct=False."""
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    failed_ids = [r["task_id"] for r in all_results if not r.get("correct")]
    tmp = failed_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(failed_ids, indent=2), encoding="utf-8")
    os.replace(tmp, failed_path)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ARC-AGI Python Ensemble Test Harness")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--limit",    type=int, default=None,
                   help="Max tasks to run this invocation (default: 1 without --all, unlimited with --all)")
    p.add_argument("--offset",   type=int, default=0)
    p.add_argument("--task-id", "--task", dest="task_id", default="")
    p.add_argument("--output",   default=DEFAULT_OUTPUT)
    p.add_argument("--mode",     choices=["train", "test"], default="train",
                   help="train (default): learn and persist new rules/tools. "
                        "test: read-only knowledge, no persistence, max 1 revision.")
    p.add_argument("--human",    action="store_true", help="Enable human-in-the-loop checkpoints (train mode only)")
    p.add_argument("--hypothesis",     default="", metavar="TEXT",
                   help="Pre-fill your hypothesis (shown before solvers run)")
    p.add_argument("--insight",        default="", metavar="TEXT",
                   help="Pre-fill your insight (shown after solver hypotheses, before MEDIATOR)")
    p.add_argument("--revision-hint",  default="", metavar="TEXT",
                   help="Pre-fill your revision hint (shown each time EXECUTOR fails)")
    p.add_argument("--prompts",  action="store_true", help="Print full prompts sent to each agent")
    p.add_argument("--charts",   action="store_true", help="Save charts per task")
    p.add_argument("--charts-dir", default="charts")
    p.add_argument("--rules",        default="", help="Path to rules.json (default: auto)")
    p.add_argument("--max-revisions", type=int, default=None, help="Override MAX_REVISIONS (default: 5)")
    p.add_argument("--all",      action="store_true", help="Run entire dataset (ignores --limit/--offset)")
    p.add_argument("--resume",   action="store_true", help="Skip tasks already recorded in --output file")
    p.add_argument("--skip-ids", dest="skip_ids", default="", metavar="FILE",
                   help="Path to a JSON challenges file or list of task IDs to exclude")
    p.add_argument("--task-list", dest="task_list", default="", metavar="FILE",
                   help="Path to a JSON list of task IDs to run (e.g. a failed.json from a previous run)")
    p.add_argument("--failed-output", dest="failed_output", default="", metavar="FILE",
                   help="Path to write/update a JSON list of failed task IDs after each task")
    p.add_argument("--quiet",    action="store_true", help="Minimal output")
    p.add_argument("--dataset",  default="training",
                   help="Dataset name for leaderboard tracking (training/eval/test)")
    p.add_argument("--dataset-tag", dest="dataset_tag", default="arc-agi-legacy",
                   help="Namespace tag for rule/tool filtering "
                        "(default: arc-agi-legacy; use arc-agi-3 for v3 tasks)")
    p.add_argument("--prune",    action="store_true",
                   help="Run maintenance pruning audit and exit (no task execution). "
                        "Flags/deprecates stale or low-performing rules and reports "
                        "redundant pairs. Respects --dataset-tag.")
    p.add_argument("--prune-threshold", dest="prune_threshold", type=int, default=10,
                   help="min_fired threshold for performance deprecation (default: 10)")
    p.add_argument("--redundancy-threshold", dest="redundancy_threshold",
                   type=float, default=0.5,
                   help="Jaccard threshold for redundancy detection (default: 0.5)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    args = parse_args()
    verbose = not args.quiet
    agents.SHOW_PROMPTS = args.prompts
    test_mode = (args.mode == "test")
    if test_mode:
        ensemble.MAX_REVISIONS = 1
        if args.human:
            console.print("[yellow]Warning: --human is ignored in test mode[/yellow]")
            args.human = False
    if args.max_revisions is not None:
        ensemble.MAX_REVISIONS = args.max_revisions

    # Load API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        key_file = Path("P:/_access/Security/api_keys.env")
        if key_file.exists():
            for line in key_file.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()
                    break
        if not os.environ.get("ANTHROPIC_API_KEY"):
            console.print("[red]ANTHROPIC_API_KEY not set[/red]")
            sys.exit(1)

    # Load data
    data_dir = Path(args.data_dir)
    challenges_path = data_dir / "arc-agi_training_challenges.json"
    solutions_path  = data_dir / "arc-agi_training_solutions.json"

    if not challenges_path.exists():
        console.print(f"[red]Challenges file not found: {challenges_path}[/red]")
        sys.exit(1)

    challenges = json.loads(challenges_path.read_text(encoding="utf-8"))
    solutions  = json.loads(solutions_path.read_text(encoding="utf-8")) if solutions_path.exists() else {}

    # Select tasks
    all_ids = list(challenges.keys())
    if args.task_id:
        task_ids = [args.task_id]
    elif args.task_list:
        tl_path = Path(args.task_list)
        if not tl_path.exists():
            console.print(f"[red]--task-list file not found: {tl_path}[/red]")
            sys.exit(1)
        task_ids = json.loads(tl_path.read_text(encoding="utf-8"))
        if not isinstance(task_ids, list):
            console.print("[red]--task-list file must be a JSON list of task IDs[/red]")
            sys.exit(1)
        console.print(f"[dim]  --task-list: {len(task_ids)} task(s) loaded from {tl_path}[/dim]")
    elif args.all:
        task_ids = all_ids  # --limit applied after filtering (below)
    else:
        limit = args.limit if args.limit is not None else 1
        task_ids = all_ids[args.offset : args.offset + limit]

    # Skip-IDs filter (e.g. to exclude v1 tasks and run v2-only)
    skip_ids: set[str] = set()
    if args.skip_ids:
        skip_path = Path(args.skip_ids)
        if not skip_path.exists():
            console.print(f"[red]--skip-ids file not found: {skip_path}[/red]")
            sys.exit(1)
        raw = json.loads(skip_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            skip_ids = set(raw.keys())   # challenges JSON — use keys as IDs
        elif isinstance(raw, list):
            skip_ids = set(raw)          # plain list of IDs
        else:
            console.print("[red]--skip-ids file must be a JSON dict or list[/red]")
            sys.exit(1)
        before = len(task_ids)
        task_ids = [tid for tid in task_ids if tid not in skip_ids]
        console.print(f"[dim]  --skip-ids: excluded {before - len(task_ids)} tasks "
                      f"({len(task_ids)} remaining)[/dim]")

    # Rule engine
    rules_path = args.rules or None
    rules = RuleEngine(rules_path, dataset_tag=args.dataset_tag)

    # --prune: maintenance audit — no task execution
    if args.prune:
        run_prune_audit(
            rules,
            min_fired=args.prune_threshold,
            redundancy_threshold=args.redundancy_threshold,
        )
        return

    # Tool registry — load and re-register all previously verified tools
    tool_reg = ToolRegistry(read_only=test_mode, dataset_tag=args.dataset_tag)
    loaded_tools = tool_reg.load_into_executor()
    if loaded_tools:
        console.print(f"[dim]  Restored {len(loaded_tools)} tool(s) from registry: {loaded_tools}[/dim]")

    # Resume: load previously completed results and skip those task IDs
    output_path = Path(args.output)
    all_results: list[dict] = []
    run_results: list[dict] = []   # only tasks run THIS invocation
    correct_count = 0
    completed_ids: set[str] = set()
    if args.resume and output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        all_results = existing.get("tasks", [])
        completed_ids = {r["task_id"] for r in all_results}
        correct_count = sum(1 for r in all_results if r.get("correct"))
        console.print(
            f"[dim]  Resuming: {len(completed_ids)} tasks already done "
            f"({correct_count} correct)[/dim]"
        )
        task_ids = [tid for tid in task_ids if tid not in completed_ids]

    # Apply --limit as a per-run cap after all filtering
    if args.all and args.limit is not None:
        task_ids = task_ids[:args.limit]

    limit_display = args.limit if args.limit is not None else 1
    scope = "all" if args.all else f"offset={args.offset}, limit={limit_display}"
    if args.all and args.limit is not None:
        scope = f"all, capped at {args.limit} this run"
    resume_note = f"  [dim]{len(completed_ids)} already done[/dim]" if completed_ids else ""
    console.print(Panel(
        f"[bold]ARC-AGI Python Ensemble[/bold]\n"
        f"Model:  [cyan]{DEFAULT_MODEL}[/cyan]\n"
        f"Tasks:  {len(task_ids)} remaining ({scope}){resume_note}\n"
        f"Mode:   {'[red]test[/red] (read-only, max 1 revision)' if test_mode else '[green]train[/green] (learning enabled)'}\n"
        f"Flags:  human={'on' if args.human else 'off'}  "
        f"prompts={'on' if args.prompts else 'off'}  "
        f"charts={'on' if args.charts else 'off'}  "
        f"resume={'on' if args.resume else 'off'}\n"
        f"NS tag: [cyan]{args.dataset_tag}[/cyan]\n"
        f"Rules:  {rules.path}  {rules.stats_summary()}\n"
        f"Tools:  {tool_reg.path}  {tool_reg.stats_summary()}",
        title="Harness"
    ))

    for i, task_id in enumerate(task_ids, 1):
        task = challenges.get(task_id)
        if task is None:
            console.print(f"[yellow]Task {task_id} not found, skipping[/yellow]")
            continue

        expected = solutions.get(task_id, [None])[0]

        console.rule(f"[{i}/{len(task_ids)}] {task_id}")

        t0 = time.time()
        meta: TaskMetadata = await run_ensemble(
            task=task,
            task_id=task_id,
            expected=expected,
            rule_engine=rules,
            tool_registry=tool_reg,
            human_in_loop=args.human,
            human_hypothesis=args.hypothesis,
            human_insight=args.insight,
            human_revision_hint=args.revision_hint,
            verbose=verbose,
            dataset=args.dataset,
            test_mode=test_mode,
        )

        if meta.correct:
            correct_count += 1

        row = {
            "task_id":         task_id,
            "correct":         meta.correct,
            "cell_accuracy":   meta.cell_accuracy,
            "converged":       meta.mediator.converged if meta.mediator else False,
            "rounds":          meta.rounds_completed,
            "duration_ms":     meta.total_duration_ms,
            "cost_usd":              meta.cost_usd,
            "input_tokens":          meta.input_tokens,
            "cache_creation_tokens": meta.cache_creation_tokens,
            "cache_read_tokens":     meta.cache_read_tokens,
            "output_tokens":         meta.output_tokens,
            "api_calls":             meta.api_calls,
            "human_hints":     meta.human_hints_used,
            "tools_generated":  meta.tools_generated,
            "matched_rule_ids": meta.matched_rule_ids,
            "model":           meta.model,
            "dataset":         meta.dataset,
        }
        all_results.append(row)
        run_results.append(row)

        if args.charts and expected:
            saved = save_all_charts(meta, expected=expected, out_dir=args.charts_dir)
            console.print(f"  Charts: {', '.join(saved)}")

        # Incremental save — survives crashes; loses at most the current task
        _save_results(output_path, all_results, DEFAULT_MODEL, args.dataset, rules)
        if args.failed_output:
            _save_failed(Path(args.failed_output), all_results)

    # ------------------------------------------------------------------
    # Final summary table  (stats are for THIS run only, not resumed tasks)
    # ------------------------------------------------------------------
    rs           = run_results   # alias: only tasks run this invocation
    total        = len(rs)
    run_correct  = sum(1 for r in rs if r.get("correct"))
    accuracy     = run_correct / total if total > 0 else 0.0
    avg_ms       = sum(r["duration_ms"] for r in rs) / max(total, 1)
    conv_rate    = sum(1 for r in rs if r.get("converged")) / max(total, 1)
    total_cost         = sum(r.get("cost_usd", 0.0) for r in rs)
    avg_cost           = total_cost / max(total, 1)
    total_input_tok    = sum(r.get("input_tokens", 0) for r in rs)
    total_output_tok   = sum(r.get("output_tokens", 0) for r in rs)
    total_cache_create = sum(r.get("cache_creation_tokens", 0) for r in rs)
    total_cache_read   = sum(r.get("cache_read_tokens", 0) for r in rs)
    total_api_calls    = sum(r.get("api_calls", 0) for r in rs)
    hints_count        = sum(1 for r in rs if r.get("human_hints"))

    table = Table(title="Run Summary (this invocation)")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Tasks run",            str(total))
    table.add_row("Correct",              f"{run_correct}/{total}  ({accuracy*100:.1f}%)")
    table.add_row("Avg duration",         f"{avg_ms/1000:.1f}s")
    table.add_row("Convergence rate",     f"{conv_rate*100:.1f}%")
    table.add_row("Total cost (USD)",     f"${total_cost:.4f}")
    table.add_row("Avg cost / task",      f"${avg_cost:.4f}")
    table.add_row("Input tokens",         f"{total_input_tok:,}")
    table.add_row("Output tokens",        f"{total_output_tok:,}")
    table.add_row("Cache create tokens",  f"{total_cache_create:,}")
    table.add_row("Cache read tokens",    f"{total_cache_read:,}")
    table.add_row("Total API calls",      f"{total_api_calls:,}")
    table.add_row("Human hints used",     f"{hints_count}/{total} tasks")
    table.add_row("Model",            DEFAULT_MODEL)
    table.add_row("Dataset",          args.dataset)
    rs = rules.stats_summary()
    table.add_row("Rules (active)",    str(rs["active"]))
    table.add_row("Rules (candidates)", str(rs.get("candidates", 0)))
    table.add_row("Rules (total)",     str(rs["total"]))
    console.print(table)
    console.print(f"Results written to [bold]{args.output}[/bold]")
    if args.failed_output:
        failed_count = sum(1 for r in all_results if not r.get("correct"))
        console.print(f"Failed tasks ({failed_count}) written to [bold]{args.failed_output}[/bold]")


if __name__ == "__main__":
    asyncio.run(main())
