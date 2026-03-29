"""
visualize.py — Charts and visualizations for ARC-AGI ensemble analysis.

Charts produced:
  1. Hypothesis Grid Evolution  — per-task, shows each solver's proposed grid
                                   next to the expected output
  2. Debate Flow Diagram        — rounds × agents timeline with PASS/FAIL coloring
  3. Learning Curve             — accuracy over tasks, with KB size overlay
  4. Ensemble vs Solo           — comparison bar chart (when solo baseline available)
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from grid_tools import Grid, to_np
from metadata import TaskMetadata, SolverEntry


# ---------------------------------------------------------------------------
# ARC color palette
# ---------------------------------------------------------------------------

ARC_COLORS = [
    "#000000",  # 0 black
    "#1E93FF",  # 1 blue
    "#F93800",  # 2 red
    "#4FCC30",  # 3 green
    "#FFDC00",  # 4 yellow
    "#999999",  # 5 gray
    "#E53AA3",  # 6 magenta
    "#FF851B",  # 7 orange
    "#87D8F1",  # 8 azure/cyan
    "#921231",  # 9 maroon
]
_CMAP = ListedColormap(ARC_COLORS)


def _plot_grid(ax: plt.Axes, grid: Optional[Grid], title: str = "") -> None:
    """Render a single ARC grid onto a matplotlib Axes."""
    ax.set_title(title, fontsize=8, pad=3)
    if grid is None or not grid:
        ax.text(0.5, 0.5, "(none)", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="gray")
        ax.axis("off")
        return

    arr = to_np(grid)
    ax.imshow(arr, cmap=_CMAP, vmin=0, vmax=9, interpolation="nearest",
              aspect="equal")

    rows, cols = arr.shape
    for r in range(rows + 1):
        ax.axhline(r - 0.5, color="white", linewidth=0.5)
    for c in range(cols + 1):
        ax.axvline(c - 0.5, color="white", linewidth=0.5)

    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# 1. Hypothesis Grid Evolution
# ---------------------------------------------------------------------------

def plot_hypothesis_evolution(
    meta: TaskMetadata,
    expected: Optional[Grid] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side grids for all solver proposals across rounds,
    plus the expected output.

    Layout:  [R1: S-SPATIAL] [R1: S-PROC] [R1: S-ANAL]
             [R3: S-SPATIAL] [R3: S-PROC] [R3: S-ANAL]
             [MEDIATOR answer]             [Expected]
    """
    all_entries: list[tuple[str, SolverEntry]] = (
        [("R1", e) for e in meta.solvers_r1] +
        [("R3", e) for e in meta.solvers_r3]
    )

    n_solver_rows = 0
    if meta.solvers_r1:
        n_solver_rows += 1
    if meta.solvers_r3:
        n_solver_rows += 1

    ncols = 3
    nrows = n_solver_rows + 1  # last row: mediator + expected

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5 + 0.5))
    fig.suptitle(f"Task {meta.task_id} — Hypothesis Grid Evolution", fontsize=10)

    axes = np.array(axes).reshape(nrows, ncols)

    row = 0
    for round_label, entries in [("R1", meta.solvers_r1), ("R3", meta.solvers_r3)]:
        if not entries:
            continue
        for col, e in enumerate(entries[:3]):
            verdict = ""
            if round_label == "R1" and meta.critic:
                v = meta.critic.verdicts.get(e.agent, "")
                verdict = f" [{v}]" if v else ""
            _plot_grid(axes[row][col], e.grid,
                       f"{round_label} {e.agent.replace('SOLVER-', '')}{verdict}")
        row += 1

    # Last row: mediator answer + expected
    mediator_grid = meta.mediator.answer if meta.mediator else None
    correct_str = ""
    if meta.correct is not None:
        correct_str = " ✓" if meta.correct else " ✗"
    _plot_grid(axes[nrows - 1][0], mediator_grid, f"MEDIATOR answer{correct_str}")
    _plot_grid(axes[nrows - 1][1], expected, "Expected")
    axes[nrows - 1][2].axis("off")  # empty slot

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# 2. Debate Flow Diagram
# ---------------------------------------------------------------------------

def plot_debate_flow(
    meta: TaskMetadata,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Timeline diagram: columns = agents, rows = rounds.
    Cells are colored green (PASS), red (FAIL), or blue (active).
    """
    agents = ["SOLVER-SPATIAL", "SOLVER-PROCEDURAL", "SOLVER-ANALOGICAL", "CRITIC", "MEDIATOR"]
    max_round = meta.rounds_completed or 4

    fig, ax = plt.subplots(figsize=(len(agents) * 1.8, max_round * 1.2 + 1))
    fig.suptitle(f"Task {meta.task_id} — Debate Flow", fontsize=10)

    ax.set_xlim(-0.5, len(agents) - 0.5)
    ax.set_ylim(-0.5, max_round - 0.5)
    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels([a.replace("SOLVER-", "S-") for a in agents], fontsize=7, rotation=20)
    ax.set_yticks(range(max_round))
    ax.set_yticklabels([f"R{r+1}" for r in range(max_round)], fontsize=8)
    ax.invert_yaxis()
    ax.grid(False)

    # SOLVER rounds 1 and 3
    solver_rounds = {e.agent: e for e in meta.solvers_r1}
    solver_r3 = {e.agent: e for e in meta.solvers_r3}

    for col, agent in enumerate(agents):
        if agent in ("SOLVER-SPATIAL", "SOLVER-PROCEDURAL", "SOLVER-ANALOGICAL"):
            # Round 1
            e = solver_rounds.get(agent)
            if e:
                color = {"high": "#22c55e", "medium": "#f59e0b", "low": "#ef4444"}.get(
                    e.confidence, "#93c5fd"
                )
                rect = mpatches.FancyBboxPatch(
                    (col - 0.4, -0.35), 0.8, 0.7,
                    boxstyle="round,pad=0.05", facecolor=color, edgecolor="white", linewidth=1
                )
                ax.add_patch(rect)
                ax.text(col, 0, e.confidence[0].upper(), ha="center", va="center",
                        fontsize=7, color="white", weight="bold")
            # Round 3
            e3 = solver_r3.get(agent)
            if e3:
                color = {"high": "#22c55e", "medium": "#f59e0b", "low": "#ef4444"}.get(
                    e3.confidence, "#93c5fd"
                )
                rect = mpatches.FancyBboxPatch(
                    (col - 0.4, 1.65), 0.8, 0.7,
                    boxstyle="round,pad=0.05", facecolor=color, edgecolor="white", linewidth=1
                )
                ax.add_patch(rect)
                ax.text(col, 2, e3.confidence[0].upper(), ha="center", va="center",
                        fontsize=7, color="white", weight="bold")

        elif agent == "CRITIC" and meta.critic:
            row = 1
            for solver, verdict in meta.critic.verdicts.items():
                s_col = agents.index(solver) if solver in agents else -1
                if s_col >= 0:
                    # Arrow from solver R1 to CRITIC
                    ax.annotate("", xy=(col, row), xytext=(s_col, row),
                                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
                color = "#22c55e" if verdict == "PASS" else "#ef4444"
                ax.text(col, row, verdict[:4], ha="center", va="center",
                        fontsize=7, color=color, weight="bold")

        elif agent == "MEDIATOR" and meta.mediator:
            row = meta.mediator.round - 1
            color = "#22c55e" if meta.correct else "#ef4444"
            rect = mpatches.FancyBboxPatch(
                (col - 0.4, row - 0.35), 0.8, 0.7,
                boxstyle="round,pad=0.05", facecolor=color, edgecolor="white", linewidth=1
            )
            ax.add_patch(rect)
            ax.text(col, row, "✓" if meta.correct else "✗", ha="center", va="center",
                    fontsize=10, color="white", weight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# 3. Learning Curve
# ---------------------------------------------------------------------------

def plot_learning_curve(
    results: list[dict],
    kb_sizes: Optional[list[int]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    results: list of {"task_id": ..., "correct": bool, "cell_accuracy": float}
    kb_sizes: optional list of KB pattern counts at each task
    """
    n = len(results)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No results", ha="center", va="center")
        return fig

    x = list(range(1, n + 1))
    cumulative_acc = []
    running_correct = 0
    cell_accs = []

    for i, r in enumerate(results):
        running_correct += 1 if r.get("correct") else 0
        cumulative_acc.append(running_correct / (i + 1))
        cell_accs.append(r.get("cell_accuracy", 0.0) or 0.0)

    fig, ax1 = plt.subplots(figsize=(max(6, n * 0.5), 4))
    fig.suptitle("Ensemble Learning Curve", fontsize=10)

    ax1.plot(x, cumulative_acc, "b-o", markersize=4, label="Cumulative accuracy")
    ax1.bar(x, cell_accs, alpha=0.3, color="steelblue", label="Cell accuracy")
    ax1.set_xlabel("Task #")
    ax1.set_ylabel("Accuracy", color="b")
    ax1.set_ylim(0, 1.1)
    ax1.tick_params(axis="y", labelcolor="b")

    if kb_sizes and len(kb_sizes) == n:
        ax2 = ax1.twinx()
        ax2.plot(x, kb_sizes, "g--s", markersize=4, label="KB patterns")
        ax2.set_ylabel("KB patterns", color="g")
        ax2.tick_params(axis="y", labelcolor="g")

    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc="upper left", fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 4. Ensemble vs Solo comparison
# ---------------------------------------------------------------------------

def plot_ensemble_vs_solo(
    ensemble_results: list[dict],
    solo_results: Optional[dict[str, list[dict]]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart comparing ensemble accuracy vs individual solver accuracy.
    solo_results: {"SOLVER-SPATIAL": [...], "SOLVER-PROCEDURAL": [...], ...}
    """
    labels = ["Ensemble"]
    accuracies = [
        sum(r.get("correct", False) for r in ensemble_results) / max(len(ensemble_results), 1)
    ]

    if solo_results:
        for name, res in solo_results.items():
            labels.append(name.replace("SOLVER-", ""))
            accuracies.append(
                sum(r.get("correct", False) for r in res) / max(len(res), 1)
            )

    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.4), 4))
    fig.suptitle("Ensemble vs Solo Accuracy", fontsize=10)
    colors = ["#22c55e"] + ["#93c5fd"] * (len(labels) - 1)
    bars = ax.bar(labels, accuracies, color=colors, edgecolor="white")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc*100:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Convenience: save all charts for a completed run
# ---------------------------------------------------------------------------

def save_all_charts(
    meta: TaskMetadata,
    expected: Optional[Grid] = None,
    out_dir: str = ".",
) -> list[str]:
    """Generate and save hypothesis evolution + debate flow for a single task."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved = []

    hyp_path = str(out / f"{meta.task_id}_hypothesis.png")
    plot_hypothesis_evolution(meta, expected=expected, save_path=hyp_path)
    saved.append(hyp_path)

    flow_path = str(out / f"{meta.task_id}_debate_flow.png")
    plot_debate_flow(meta, save_path=flow_path)
    saved.append(flow_path)

    return saved
