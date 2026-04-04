"""
metadata.py — Structured metadata capture for each debate round.

Records solver hypotheses, CRITIC verdicts, convergence signals, and
per-cell accuracy so we can analyze ensemble behavior and build
visualizations after the fact.
"""

from __future__ import annotations
import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from grid_tools import Grid, grids_equal, cell_accuracy, diff_cells


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SolverEntry:
    agent: str
    round: int
    rule: str
    confidence: str                   # "high" | "medium" | "low"
    grid: Optional[Grid]
    raw_response: str
    duration_ms: int

@dataclass
class CriticVerdict:
    agent: str = "CRITIC"
    round: int = 2
    verdicts: dict[str, str] = field(default_factory=dict)   # solver → "PASS"|"FAIL"
    notes: str = ""
    raw_response: str = ""
    duration_ms: int = 0

@dataclass
class MediatorDecision:
    agent: str = "MEDIATOR"
    round: int = 4
    answer: Optional[Grid] = None
    rationale: str = ""
    converged: bool = False
    raw_response: str = ""
    duration_ms: int = 0
    kb_updates: dict[str, int] = field(default_factory=dict)  # counts

@dataclass
class TaskMetadata:
    task_id: str
    train_pairs: int
    test_shape: tuple[int, int]
    expected_shape: Optional[tuple[int, int]]
    start_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    solvers_r1: list[SolverEntry] = field(default_factory=list)
    solvers_r3: list[SolverEntry] = field(default_factory=list)
    critic: Optional[CriticVerdict] = None
    mediator: Optional[MediatorDecision] = None

    # Outcome
    correct: Optional[bool] = None
    cell_accuracy: Optional[float] = None
    total_duration_ms: int = 0
    rounds_completed: int = 0

    # Leaderboard stats
    model: str = ""
    dataset: str = ""
    human_hints_used: bool = False
    tools_generated: list = field(default_factory=list)   # names of dynamically created tools
    matched_rule_ids: list = field(default_factory=list)  # rule IDs matched in Round 0
    input_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Convert grids to lists (already lists, but dataclasses may wrap them)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Grid extraction
# ---------------------------------------------------------------------------

def extract_json_grid(text: str) -> Optional[Grid]:
    """
    Scan ALL fenced code blocks for a JSON object with a "grid" key.
    Returns the last matching grid (first match is often the quoted input).
    """
    block_re = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    last_grid = None
    for raw in block_re.findall(text):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "grid" in obj:
                grid = obj["grid"]
                if isinstance(grid, list) and all(isinstance(r, list) for r in grid):
                    last_grid = grid
        except (json.JSONDecodeError, Exception):
            pass
    return last_grid

def extract_solver_fields(text: str) -> tuple[Optional[Grid], str, str]:
    """
    Extract (grid, rule, confidence) from a solver response.
    Falls back to empty strings if fields are missing.
    """
    block_re = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    last_grid = None
    rule = ""
    confidence = "medium"
    for raw in block_re.findall(text):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "grid" in obj:
                grid = obj.get("grid")
                if isinstance(grid, list) and all(isinstance(r, list) for r in grid):
                    last_grid = grid
                rule = obj.get("rule", rule)
                confidence = obj.get("confidence", confidence)
        except Exception:
            pass
    return last_grid, rule, confidence

def extract_critic_verdicts(text: str) -> dict[str, str]:
    """
    Parse CRITIC output for PASS/FAIL per solver.
    Looks for lines like: "SOLVER-SPATIAL: PASS" or "SOLVER-PROCEDURAL: FAIL"
    Also tries a JSON block with a "verdicts" key.
    """
    # Try JSON block first
    block_re = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    for raw in block_re.findall(text):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "verdicts" in obj:
                return {k: str(v).upper() for k, v in obj["verdicts"].items()}
        except Exception:
            pass

    # Fallback: line scan — try full name and short name
    verdicts: dict[str, str] = {}
    solver_names = {
        "SOLVER-SPATIAL": ["SOLVER-SPATIAL", "SPATIAL"],
        "SOLVER-PROCEDURAL": ["SOLVER-PROCEDURAL", "PROCEDURAL"],
        "SOLVER-ANALOGICAL": ["SOLVER-ANALOGICAL", "ANALOGICAL"],
    }
    for canonical, aliases in solver_names.items():
        for name in aliases:
            pattern = re.compile(rf"{name}[:\s*\-]*(PASS|FAIL)", re.IGNORECASE)
            m = pattern.search(text)
            if m:
                verdicts[canonical] = m.group(1).upper()
                break

    # Last resort: look for any PASS/FAIL lines and map by order
    if not verdicts:
        pf_matches = re.findall(r"\b(PASS|FAIL)\b", text, re.IGNORECASE)
        solvers = ["SOLVER-SPATIAL", "SOLVER-PROCEDURAL", "SOLVER-ANALOGICAL"]
        for i, v in enumerate(pf_matches[:3]):
            verdicts[solvers[i]] = v.upper()

    return verdicts


# ---------------------------------------------------------------------------
# Accuracy reporting
# ---------------------------------------------------------------------------

def compute_outcome(meta: TaskMetadata, expected: Optional[Grid]) -> None:
    """Mutate meta in-place with outcome fields given the expected solution."""
    if meta.mediator is None or meta.mediator.answer is None or expected is None:
        meta.correct = False
        meta.cell_accuracy = 0.0
        return
    meta.correct = grids_equal(meta.mediator.answer, expected)
    meta.cell_accuracy = cell_accuracy(meta.mediator.answer, expected)


# ---------------------------------------------------------------------------
# Summary display (rich)
# ---------------------------------------------------------------------------

def print_task_summary(meta: TaskMetadata, expected: Optional[Grid] = None,
                        use_rich: bool = True) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        console = Console()
    except ImportError:
        use_rich = False

    if not use_rich:
        _print_plain(meta, expected)
        return

    # Header
    status = "+ CORRECT" if meta.correct else "X WRONG"
    color = "green" if meta.correct else "red"
    console.print(Panel(
        f"[{color}]{status}[/{color}]  |  "
        f"Task [bold]{meta.task_id}[/bold]  |  "
        f"Cell accuracy: {(meta.cell_accuracy or 0.0)*100:.1f}%  |  "
        f"{meta.total_duration_ms/1000:.1f}s  |  "
        f"{meta.rounds_completed} rounds",
        title="Task Result"
    ))

    # Solver table
    table = Table(title="Solver Proposals")
    table.add_column("Round", style="dim")
    table.add_column("Agent")
    table.add_column("Confidence")
    table.add_column("Hypothesis", no_wrap=False)
    table.add_column("Grid shape")
    table.add_column("Correct?" if expected else "")

    for entry in meta.solvers_r1 + meta.solvers_r3:
        g = entry.grid
        shape_str = f"{len(g)}×{len(g[0])}" if g and g[0] else "?"
        if expected and g:
            acc = f"{cell_accuracy(g, expected)*100:.0f}%"
        else:
            acc = ""
        table.add_row(
            str(entry.round),
            entry.agent,
            entry.confidence,
            entry.rule,
            shape_str,
            acc,
        )
    console.print(table)

    # CRITIC
    if meta.critic:
        v = meta.critic.verdicts
        verdict_str = "  ".join(f"{k}: {'[green]PASS[/green]' if v == 'PASS' else '[red]FAIL[/red]'}"
                                 for k, v in v.items())
        console.print(f"[bold]CRITIC verdicts:[/bold] {verdict_str}")

    # Diff (if wrong)
    if not meta.correct and expected and meta.mediator and meta.mediator.answer:
        diffs = diff_cells(meta.mediator.answer, expected)
        console.print(f"[yellow]{len(diffs)} cells differ[/yellow]")
        for r, c, got, want in diffs[:10]:
            console.print(f"  ({r},{c}): got {got}, expected {want}")
        if len(diffs) > 10:
            console.print(f"  ... and {len(diffs)-10} more")


def _print_plain(meta: TaskMetadata, expected: Optional[Grid] = None) -> None:
    status = "CORRECT" if meta.correct else "WRONG"
    print(f"[{status}] {meta.task_id}  acc={((meta.cell_accuracy or 0)*100):.1f}%  "
          f"{meta.total_duration_ms/1000:.1f}s  {meta.rounds_completed} rounds")
    if meta.critic:
        print(f"  CRITIC: {meta.critic.verdicts}")
