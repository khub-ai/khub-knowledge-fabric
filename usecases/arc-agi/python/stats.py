"""
stats.py — ARC-AGI ensemble performance reporter.

Usage:
  python stats.py                          # aggregate all known result files
  python stats.py results_v2_21_30.json    # specific file(s)
  python stats.py --results-dir .          # scan directory for result JSONs
  python stats.py --rules-only             # rule stats only, no task results needed

Sections printed:
  1. Overall accuracy and task breakdown
  2. Solve-method breakdown (rule / dynamic-tool / solver-only)
  3. Rule effectiveness (lineage, fire/success rates)
  4. Generalization regime stats (candidates, promotions, merged rules)
  5. Dynamically generated tools and their outcomes
  6. Cost and token efficiency
  7. Failed task list
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

console = Console()

# ---------------------------------------------------------------------------
# Default result files (relative to _HERE)
# ---------------------------------------------------------------------------
DEFAULT_RESULT_FILES = [
    "results.json",
    "results_v2_21_30.json",
    "results_v2_31_35.json",
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_tasks(paths: list[Path]) -> dict[str, dict]:
    """Load and deduplicate task rows from result files."""
    all_tasks: dict[str, dict] = {}
    for p in paths:
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            tasks = data.get("tasks", []) if isinstance(data, dict) else data
            for t in tasks:
                tid = t.get("task_id")
                if tid and tid not in all_tasks:
                    all_tasks[tid] = t
        except Exception as e:
            console.print(f"[yellow]Warning: could not read {p}: {e}[/yellow]")
    return all_tasks


def load_rules(rules_path: Path) -> list[dict]:
    if not rules_path.exists():
        return []
    data = json.loads(rules_path.read_text(encoding="utf-8"))
    return data.get("rules", data) if isinstance(data, dict) else data


def load_tools(tools_path: Path) -> list[dict]:
    if not tools_path.exists():
        return []
    data = json.loads(tools_path.read_text(encoding="utf-8"))
    raw = data.get("tools", data) if isinstance(data, dict) else data
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return [{"name": k, **v} for k, v in raw.items()]
    return []


# ---------------------------------------------------------------------------
# Classify tasks
# ---------------------------------------------------------------------------

def classify_tasks(tasks: dict[str, dict]) -> dict:
    correct   = {tid: t for tid, t in tasks.items() if t.get("correct")}
    failed    = {tid: t for tid, t in tasks.items() if not t.get("correct")}

    rule_solved    = {}
    tool_solved    = {}
    solver_solved  = {}

    for tid, t in correct.items():
        has_rule = bool(t.get("matched_rule_ids"))
        # Fallback heuristic for old results without matched_rule_ids
        if not has_rule and t.get("api_calls", 99) <= 3 and not t.get("tools_generated"):
            has_rule = True
        if has_rule:
            rule_solved[tid] = t
        elif t.get("tools_generated"):
            tool_solved[tid] = t
        else:
            solver_solved[tid] = t

    return {
        "total":   tasks,
        "correct": correct,
        "failed":  failed,
        "rule_solved":   rule_solved,
        "tool_solved":   tool_solved,
        "solver_solved": solver_solved,
    }


# ---------------------------------------------------------------------------
# Section printers
# ---------------------------------------------------------------------------

def print_overview(cls: dict) -> None:
    total   = len(cls["total"])
    correct = len(cls["correct"])
    failed  = len(cls["failed"])
    acc     = 100 * correct / total if total else 0

    t = Table(title="Overall Accuracy", box=box.SIMPLE_HEAVY)
    t.add_column("Metric", style="bold")
    t.add_column("Value")
    t.add_row("Tasks tracked",   str(total))
    t.add_row("Correct",         f"{correct} / {total}  ({acc:.1f}%)")
    t.add_row("Failed",          str(failed))
    console.print(t)


def print_solve_method(cls: dict) -> None:
    correct = len(cls["correct"])
    rs = len(cls["rule_solved"])
    ts = len(cls["tool_solved"])
    ss = len(cls["solver_solved"])

    def pct(n):
        return f"{100*n/correct:.0f}%" if correct else "—"

    t = Table(title="How Tasks Were Solved", box=box.SIMPLE_HEAVY)
    t.add_column("Method", style="bold")
    t.add_column("Count")
    t.add_column("% of correct")
    t.add_column("Task IDs")
    t.add_row("Rule matched (Round 0)", str(rs), pct(rs),
              ", ".join(sorted(cls["rule_solved"])) or "—")
    t.add_row("Dynamic tool generated", str(ts), pct(ts),
              ", ".join(sorted(cls["tool_solved"])) or "—")
    t.add_row("Solver+MEDIATOR only",   str(ss), pct(ss),
              ", ".join(sorted(cls["solver_solved"])) or "—")
    console.print(t)

    # Tool details
    if cls["tool_solved"]:
        t2 = Table(title="Dynamic Tools That Solved Tasks", box=box.SIMPLE)
        t2.add_column("Task ID")
        t2.add_column("Tools generated")
        for tid, task in sorted(cls["tool_solved"].items()):
            t2.add_row(tid, ", ".join(task.get("tools_generated", [])))
        console.print(t2)


def print_rule_stats(rules: list[dict]) -> None:
    active     = [r for r in rules if r.get("status") == "active"]
    candidates = [r for r in rules if r.get("status") == "candidate"]
    deprecated = [r for r in rules if r.get("status") == "deprecated"]

    # Lineage breakdown
    lineage_counts: dict[str, int] = defaultdict(int)
    for r in active:
        lin = r.get("lineage", {})
        lt  = lin.get("type", "unknown") if isinstance(lin, dict) else str(lin)
        lineage_counts[lt] += 1

    t = Table(title="Rule Base Summary", box=box.SIMPLE_HEAVY)
    t.add_column("Metric", style="bold")
    t.add_column("Value")
    t.add_row("Total rules",     str(len(rules)))
    t.add_row("Active",          str(len(active)))
    t.add_row("Candidates",      str(len(candidates)))
    t.add_row("Deprecated",      str(len(deprecated)))
    for lt, cnt in sorted(lineage_counts.items()):
        t.add_row(f"  Lineage: {lt}", str(cnt))
    total_fired     = sum(r.get("stats", {}).get("fired", 0)     for r in active)
    total_succeeded = sum(r.get("stats", {}).get("succeeded", 0) for r in active)
    fire_rate = f"{100*total_succeeded/total_fired:.0f}%" if total_fired else "—"
    t.add_row("Total firings",   str(total_fired))
    t.add_row("Total successes", str(total_succeeded))
    t.add_row("Overall fire→success rate", fire_rate)
    console.print(t)

    # Rules with successes
    succeeded_rules = [r for r in active if r.get("stats", {}).get("succeeded", 0) > 0]
    succeeded_rules.sort(key=lambda x: -x.get("stats", {}).get("succeeded", 0))

    t2 = Table(title="Active Rules by Success Count", box=box.SIMPLE)
    t2.add_column("ID")
    t2.add_column("Source task")
    t2.add_column("Lineage")
    t2.add_column("Fired")
    t2.add_column("Succeeded")
    t2.add_column("Tags")
    for r in succeeded_rules:
        s   = r["stats"]["succeeded"]
        f   = r["stats"]["fired"]
        lin = r.get("lineage", {})
        lt  = lin.get("type", "?") if isinstance(lin, dict) else str(lin)
        t2.add_row(r["id"], r.get("source_task", "?"), lt,
                   str(f), str(s), ", ".join(r.get("tags", [])))
    console.print(t2)


def print_generalization_stats(rules: list[dict]) -> None:
    active = [r for r in rules if r.get("status") == "active"]
    cands  = [r for r in rules if r.get("status") == "candidate"]

    generalized = [r for r in active if _lineage_type(r) == "generalized"]
    merged      = [r for r in active if _lineage_type(r) in ("merged", "consolidated")]

    gen_succeeded = [r for r in generalized if r.get("stats", {}).get("succeeded", 0) > 0]
    mrg_succeeded = [r for r in merged      if r.get("stats", {}).get("succeeded", 0) > 0]

    t = Table(title="Generalization Regime", box=box.SIMPLE_HEAVY)
    t.add_column("Category", style="bold")
    t.add_column("Count")
    t.add_column("With successes")
    t.add_row("Generalized rules (active)",    str(len(generalized)), str(len(gen_succeeded)))
    t.add_row("Merged/consolidated (active)",  str(len(merged)),      str(len(mrg_succeeded)))
    t.add_row("Candidate rules (unconfirmed)", str(len(cands)),       "—")
    console.print(t)

    if generalized:
        t2 = Table(title="Generalized Rules Detail", box=box.SIMPLE)
        t2.add_column("ID")
        t2.add_column("Source task")
        t2.add_column("Parent IDs")
        t2.add_column("Fired")
        t2.add_column("Succeeded")
        t2.add_column("Tags")
        for r in sorted(generalized, key=lambda x: -x.get("stats", {}).get("succeeded", 0)):
            lin = r.get("lineage", {})
            parents = ", ".join(lin.get("parent_ids", [])) if isinstance(lin, dict) else "?"
            s = r.get("stats", {}).get("succeeded", 0)
            f = r.get("stats", {}).get("fired", 0)
            t2.add_row(r["id"], r.get("source_task", "?"), parents,
                       str(f), str(s), ", ".join(r.get("tags", [])[:3]))
        console.print(t2)

    if cands:
        t3 = Table(title="Candidate Rules (awaiting confirmation)", box=box.SIMPLE)
        t3.add_column("ID")
        t3.add_column("Source task")
        t3.add_column("Condition (truncated)")
        for r in cands:
            t3.add_row(r["id"], r.get("source_task", "?"),
                       r.get("condition", "")[:80])
        console.print(t3)


def print_tool_stats(tools: list[dict], tasks: dict[str, dict]) -> None:
    # Map tool name → tasks it appeared in (from result rows)
    tool_tasks: dict[str, list[str]] = defaultdict(list)
    for tid, t in tasks.items():
        for tool in t.get("tools_generated", []):
            label = f"{tid}({'SOLVED' if t.get('correct') else 'FAILED'})"
            tool_tasks[tool].append(label)

    # Registry-level stats from tools.json
    tool_registry: dict[str, dict] = {t.get("name", f"tool_{i}"): t for i, t in enumerate(tools)}

    all_tool_names = sorted(set(list(tool_tasks.keys()) + [
        n for n, t in tool_registry.items() if not t.get("builtin", False)
        and t.get("times_used", 0) > 0
    ]))

    if not all_tool_names and not tool_tasks:
        console.print("[dim]No dynamic tools recorded in results.[/dim]")
        return

    t = Table(title="Dynamically Generated Tools", box=box.SIMPLE_HEAVY)
    t.add_column("Tool name", style="bold")
    t.add_column("Outcome")
    t.add_column("Tasks")

    solved_count  = 0
    failed_count  = 0
    for name in sorted(tool_tasks.keys()):
        entries = tool_tasks[name]
        solved  = any("SOLVED" in e for e in entries)
        outcome = "[green]SOLVED[/green]" if solved else "[red]FAILED[/red]"
        if solved:
            solved_count += 1
        else:
            failed_count += 1
        t.add_row(name, outcome, ", ".join(entries))

    console.print(t)

    summary = Table(box=box.SIMPLE)
    summary.add_column("Metric")
    summary.add_column("Value")
    summary.add_row("Total dynamic tools created", str(len(tool_tasks)))
    summary.add_row("Contributed to a solution",   str(solved_count))
    summary.add_row("Generated but task still failed", str(failed_count))
    total_builtin = sum(1 for t in tools if t.get("builtin", False))
    if total_builtin:
        summary.add_row("Builtin tools (total)", str(total_builtin))
    console.print(summary)


def print_cost_stats(tasks: dict[str, dict]) -> None:
    rows = list(tasks.values())
    if not rows:
        return
    total_cost    = sum(r.get("cost_usd", 0)       for r in rows)
    total_input   = sum(r.get("input_tokens", 0)   for r in rows)
    total_output  = sum(r.get("output_tokens", 0)  for r in rows)
    total_api     = sum(r.get("api_calls", 0)       for r in rows)
    correct_rows  = [r for r in rows if r.get("correct")]
    avg_dur_s     = sum(r.get("duration_ms", 0) for r in rows) / max(len(rows), 1) / 1000
    avg_dur_ok    = sum(r.get("duration_ms", 0) for r in correct_rows) / max(len(correct_rows), 1) / 1000
    avg_cost      = total_cost / max(len(rows), 1)
    avg_cost_ok   = sum(r.get("cost_usd", 0) for r in correct_rows) / max(len(correct_rows), 1)
    avg_calls     = total_api / max(len(rows), 1)

    t = Table(title="Cost & Efficiency", box=box.SIMPLE_HEAVY)
    t.add_column("Metric", style="bold")
    t.add_column("Value")
    t.add_row("Total cost (tracked tasks)", f"${total_cost:.4f}")
    t.add_row("Avg cost / task",            f"${avg_cost:.4f}")
    t.add_row("Avg cost / correct task",    f"${avg_cost_ok:.4f}")
    t.add_row("Avg duration / task",        f"{avg_dur_s:.1f}s")
    t.add_row("Avg duration / correct",     f"{avg_dur_ok:.1f}s")
    t.add_row("Total input tokens",         f"{total_input:,}")
    t.add_row("Total output tokens",        f"{total_output:,}")
    t.add_row("Total API calls",            f"{total_api:,}")
    t.add_row("Avg API calls / task",       f"{avg_calls:.1f}")
    console.print(t)


def print_failed_tasks(cls: dict) -> None:
    failed = cls["failed"]
    if not failed:
        console.print("[green]No failed tasks.[/green]")
        return
    t = Table(title=f"Failed Tasks ({len(failed)})", box=box.SIMPLE_HEAVY)
    t.add_column("Task ID", style="bold")
    t.add_column("Cell acc")
    t.add_column("Rounds")
    t.add_column("Cost")
    t.add_column("Tools tried")
    t.add_column("Rules matched")
    for tid, task in sorted(failed.items()):
        t.add_row(
            tid,
            f"{task.get('cell_accuracy', 0)*100:.0f}%",
            str(task.get("rounds", "?")),
            f"${task.get('cost_usd', 0):.3f}",
            ", ".join(task.get("tools_generated", [])) or "—",
            ", ".join(task.get("matched_rule_ids", [])) or "—",
        )
    console.print(t)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lineage_type(r: dict) -> str:
    lin = r.get("lineage", {})
    if isinstance(lin, dict):
        return lin.get("type", "unknown")
    return str(lin)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="ARC-AGI ensemble stats reporter")
    p.add_argument("files", nargs="*",
                   help="Result JSON files to aggregate (default: auto-discover)")
    p.add_argument("--results-dir", default=str(_HERE),
                   help="Directory to scan for result JSONs")
    p.add_argument("--rules",  default=str(_HERE / "rules.json"))
    p.add_argument("--tools",  default=str(_HERE / "tools.json"))
    p.add_argument("--rules-only", action="store_true",
                   help="Print rule/generalization stats only")
    p.add_argument("--section", choices=[
        "overview", "methods", "rules", "generalization", "tools", "cost", "failed"
    ], help="Print a single section only")
    args = p.parse_args()

    # Resolve result files
    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        scan_dir = Path(args.results_dir)
        paths = [scan_dir / f for f in DEFAULT_RESULT_FILES]
        # Also pick up any other results_*.json in the dir
        for extra in scan_dir.glob("results_*.json"):
            if extra not in paths:
                paths.append(extra)

    tasks = load_tasks(paths) if not args.rules_only else {}
    rules = load_rules(Path(args.rules))
    tools = load_tools(Path(args.tools))
    cls   = classify_tasks(tasks)

    console.print()
    console.print(Panel.fit(
        f"[bold]ARC-AGI Ensemble — Performance Report[/bold]\n"
        f"Tasks from {len(paths)} file(s)  |  Rules: {len(rules)}  |  Tools: {len(tools)}",
        border_style="blue",
    ))
    console.print()

    sections = {
        "overview":       lambda: print_overview(cls),
        "methods":        lambda: print_solve_method(cls),
        "rules":          lambda: print_rule_stats(rules),
        "generalization": lambda: print_generalization_stats(rules),
        "tools":          lambda: print_tool_stats(tools, tasks),
        "cost":           lambda: print_cost_stats(tasks),
        "failed":         lambda: print_failed_tasks(cls),
    }

    if args.section:
        sections[args.section]()
    elif args.rules_only:
        sections["rules"]()
        sections["generalization"]()
    else:
        for name, fn in sections.items():
            console.print(Rule(name.title()))
            fn()
            console.print()


if __name__ == "__main__":
    main()
