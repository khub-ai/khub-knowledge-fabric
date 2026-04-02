"""
ensemble_monitor.py — Real-time developer GUI for the ARC-AGI-3 ensemble.

Reads playlog JSON files written by ensemble.py (one file per env.step() call)
and displays a live dashboard showing:
  - Current game frame (color-rendered grid)
  - Episode / step / cycle / level progress
  - Active goals tree (goal hierarchy from GoalManager)
  - State key-value store
  - OBSERVER analysis (full text)
  - MEDIATOR action plan and reasoning
  - Matched knowledge rules
  - Rolling cost / token stats
  - Live auto-refresh (polls the playlog directory every second)

Usage:
  python ensemble_monitor.py                         # latest run
  python ensemble_monitor.py --playlog-dir playlogs/20260402-123456_ep01
  python ensemble_monitor.py --refresh-ms 2000       # slower polling
  python ensemble_monitor.py --no-live               # static, no auto-refresh

Based on the playlog_viewer.py in tests/arc-agi-3/ (original kept intact).
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font as tkfont, ttk
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# ARC-AGI color palette (values 0-12)
# ---------------------------------------------------------------------------

PALETTE = {
    0:  "#000000",   # black
    1:  "#0074D9",   # blue
    2:  "#FF4136",   # red
    3:  "#2ECC40",   # green
    4:  "#AAAAAA",   # grey
    5:  "#FFDC00",   # yellow
    6:  "#AA00FF",   # magenta
    7:  "#FF851B",   # orange
    8:  "#7FDBFF",   # azure
    9:  "#F012BE",   # fuchsia
    10: "#7B7B7B",
    11: "#85144B",
    12: "#39CCCC",
}

# Goal status colors
GOAL_COLORS = {
    "active":   "#2ECC40",
    "pending":  "#FFDC00",
    "resolved": "#7FDBFF",
    "failed":   "#FF4136",
    "abandoned":"#AAAAAA",
}

DEFAULT_PLAYLOG_ROOT = Path(__file__).parent / "playlogs"
REFRESH_MS_DEFAULT   = 1000


# ---------------------------------------------------------------------------
# Playlog loading helpers
# ---------------------------------------------------------------------------

def find_latest_episode_dir(root: Path) -> Optional[Path]:
    """Return the most recently modified episode directory under root."""
    if not root.exists():
        return None
    candidates = [d for d in root.iterdir() if d.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def load_steps(playlog_dir: Path) -> List[Dict]:
    """Load all step JSON files from a playlog directory, sorted by step number."""
    step_files = sorted(playlog_dir.glob("[0-9][0-9][0-9][0-9]-*.json"))
    steps = []
    for path in step_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        payload["_frame"] = _decode_frame(payload.get("returned", {}).get("frame"))
        payload["_path"]  = str(path)
        steps.append(payload)
    return steps


def _decode_frame(frame_value) -> Optional[List[List[int]]]:
    """Decode a frame field to a list-of-lists of ints.

    Handles these formats:
    - list[list[int]]    — 2D grid stored directly (ensemble.py format)
    - [list[list[int]]]  — 2D grid wrapped in one extra list (original viewer format)
    - str / list[str]    — serialized as a string (legacy)
    """
    if not frame_value:
        return None

    if isinstance(frame_value, str):
        return _parse_frame_string(frame_value)

    if not isinstance(frame_value, list):
        return None

    first = frame_value[0]

    # List of strings -> parse the first string
    if isinstance(first, str):
        return _parse_frame_string(first)

    if isinstance(first, list):
        # If first row contains ints/floats -> frame_value is the 2D grid directly
        if first and isinstance(first[0], (int, float)):
            return [[int(v) for v in row] for row in frame_value]
        # Otherwise first is itself a 2D grid (wrapped format: [grid2d])
        if first and isinstance(first[0], list):
            return [[int(v) for v in row] for row in first]

    # frame_value is a flat list of ints (shouldn't happen but handle gracefully)
    if isinstance(first, (int, float)):
        return [frame_value]

    return None


def _parse_frame_string(text: str) -> List[List[int]]:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        numbers = [int(t) for t in re.findall(r"-?\d+", line)]
        if numbers:
            rows.append(numbers)
    return rows


# ---------------------------------------------------------------------------
# Main Monitor Application
# ---------------------------------------------------------------------------

class EnsembleMonitor:

    def __init__(
        self,
        root: tk.Tk,
        playlog_root: Path,
        playlog_dir: Optional[Path],
        refresh_ms: int,
        live: bool,
    ) -> None:
        self.root          = root
        self.playlog_root  = playlog_root
        self.playlog_dir   = playlog_dir
        self.refresh_ms    = refresh_ms
        self.live          = live
        self.steps: List[Dict] = []
        self.index         = 0
        self.cell_size     = 6
        self._live_paused  = False
        self._after_id     = None
        self._last_step_count = 0

        root.title("ARC-AGI-3 Ensemble Monitor")
        self._build_ui()
        self._bind_keys()

        if self.playlog_dir and self.playlog_dir.exists():
            self._load_dir(self.playlog_dir)
        else:
            self._try_latest()

        if self.live:
            self._schedule_refresh()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=0)   # left: frame
        self.root.columnconfigure(1, weight=1)   # right: panels
        self.root.rowconfigure(0, weight=1)

        # ---- Left column: frame + controls ---------------------------------
        left = ttk.Frame(self.root, padding=8)
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(1, weight=1)

        # Status bar at top
        self.status_var = tk.StringVar(value="No playlog loaded")
        ttk.Label(left, textvariable=self.status_var,
                  font=("Courier New", 9)).grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 4))

        # Canvas inside a scrollable frame
        canvas_outer = ttk.Frame(left)
        canvas_outer.grid(row=1, column=0, columnspan=5, sticky="nsew")
        canvas_outer.rowconfigure(0, weight=1)
        canvas_outer.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_outer, background="black",
                                highlightthickness=1, highlightbackground="#555")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        vsb = ttk.Scrollbar(canvas_outer, orient="vertical",   command=self.canvas.yview)
        hsb = ttk.Scrollbar(canvas_outer, orient="horizontal", command=self.canvas.xview)
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # Nav buttons
        nav = ttk.Frame(left)
        nav.grid(row=2, column=0, columnspan=5, sticky="ew", pady=(6, 0))
        ttk.Button(nav, text="|<",     command=lambda: self._go(0)).pack(side="left", padx=2)
        ttk.Button(nav, text="<",      command=self._prev).pack(side="left", padx=2)
        ttk.Button(nav, text=">",      command=self._next).pack(side="left", padx=2)
        ttk.Button(nav, text=">|",     command=lambda: self._go(max(0, len(self.steps) - 1))).pack(side="left", padx=2)
        ttk.Button(nav, text="Open...", command=self._choose_folder).pack(side="left", padx=8)
        ttk.Button(nav, text="Zoom+",  command=lambda: self._zoom(1)).pack(side="left", padx=2)
        ttk.Button(nav, text="Zoom-",  command=lambda: self._zoom(-1)).pack(side="left", padx=2)
        ttk.Button(nav, text="Fit",    command=self._fit_zoom).pack(side="left", padx=2)

        self._live_btn = ttk.Button(nav, text="Pause Live",
                                    command=self._toggle_live)
        self._live_btn.pack(side="left", padx=8)
        if not self.live:
            self._live_btn.configure(state="disabled")

        # ---- Right column: notebook of panels ------------------------------
        right = ttk.Frame(self.root, padding=(0, 8, 8, 8))
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        nb = ttk.Notebook(right)
        nb.grid(row=0, column=0, sticky="nsew")

        # Tab 0: Progress (always visible)
        t_prog = ttk.Frame(nb, padding=8)
        nb.add(t_prog, text="Progress")
        self._build_progress_tab(t_prog)

        # Tab 1: Goals
        t_goals = ttk.Frame(nb, padding=8)
        nb.add(t_goals, text="Goals")
        self._build_goals_tab(t_goals)

        # Tab 2: State
        t_state = ttk.Frame(nb, padding=8)
        nb.add(t_state, text="State")
        self._build_state_tab(t_state)

        # Tab 3: OBSERVER
        t_obs = ttk.Frame(nb, padding=8)
        nb.add(t_obs, text="OBSERVER")
        self._build_text_tab(t_obs, "observer_text")

        # Tab 4: MEDIATOR
        t_med = ttk.Frame(nb, padding=8)
        nb.add(t_med, text="MEDIATOR")
        self._build_mediator_tab(t_med)

        # Tab 5: Rules
        t_rules = ttk.Frame(nb, padding=8)
        nb.add(t_rules, text="Rules")
        self._build_text_tab(t_rules, "rules_text")

        # Tab 6: Costs
        t_cost = ttk.Frame(nb, padding=8)
        nb.add(t_cost, text="Costs")
        self._build_progress_cost_tab(t_cost)

        # Help bar
        ttk.Label(right,
                  text="Keys: Space/Right=next  Left=prev  Home=first  End=last  L=live toggle  +/-/F=zoom",
                  font=("Courier New", 8), foreground="#888"
                  ).grid(row=1, column=0, sticky="w", pady=(4, 0))

    def _build_progress_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        self._prog_vars: Dict[str, tk.StringVar] = {}
        rows_spec = [
            ("Episode",       "episode"),
            ("Cycle",         "cycle"),
            ("Plan step",     "plan_index"),
            ("Step",          "step"),
            ("State",         "state"),
            ("Levels",        "levels"),
            ("Action",        "action"),
            ("Data",          "data"),
            ("Diff cells",    "diff_count"),
            ("Change box",    "bbox"),
        ]
        for i, (label, key) in enumerate(rows_spec):
            self._prog_vars[key] = tk.StringVar(value="—")
            ttk.Label(parent, text=f"{label}:", width=12, anchor="e").grid(
                row=i, column=0, sticky="e", padx=(0, 6), pady=2)
            ttk.Label(parent, textvariable=self._prog_vars[key],
                      wraplength=320, justify="left").grid(
                row=i, column=1, sticky="w", pady=2)

    def _build_goals_tab(self, parent: ttk.Frame) -> None:
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        self.goals_text = self._make_scrolled_text(parent, height=25)
        self.goals_text.grid(row=0, column=0, sticky="nsew")

    def _build_state_tab(self, parent: ttk.Frame) -> None:
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        self.state_text = self._make_scrolled_text(parent, height=25)
        self.state_text.grid(row=0, column=0, sticky="nsew")

    def _build_text_tab(self, parent: ttk.Frame, attr: str) -> None:
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        widget = self._make_scrolled_text(parent, height=25)
        widget.grid(row=0, column=0, sticky="nsew")
        setattr(self, attr, widget)

    def _build_mediator_tab(self, parent: ttk.Frame) -> None:
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        plan_frame = ttk.LabelFrame(parent, text="Action Plan", padding=6)
        plan_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        plan_frame.columnconfigure(0, weight=1)
        self.mediator_plan_text = self._make_scrolled_text(plan_frame, height=6)
        self.mediator_plan_text.grid(row=0, column=0, sticky="ew")

        reason_frame = ttk.LabelFrame(parent, text="Reasoning", padding=6)
        reason_frame.grid(row=1, column=0, sticky="nsew")
        reason_frame.rowconfigure(0, weight=1)
        reason_frame.columnconfigure(0, weight=1)
        self.mediator_reason_text = self._make_scrolled_text(reason_frame, height=18)
        self.mediator_reason_text.grid(row=0, column=0, sticky="nsew")

    def _build_progress_cost_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        self._cost_vars: Dict[str, tk.StringVar] = {}
        rows_spec = [
            ("Cost (episode)",  "cost_episode"),
            ("Input tokens",    "tokens_input"),
            ("Output tokens",   "tokens_output"),
            ("API calls",       "api_calls"),
        ]
        for i, (label, key) in enumerate(rows_spec):
            self._cost_vars[key] = tk.StringVar(value="—")
            ttk.Label(parent, text=f"{label}:", width=16, anchor="e").grid(
                row=i, column=0, sticky="e", padx=(0, 6), pady=3)
            ttk.Label(parent, textvariable=self._cost_vars[key], justify="left").grid(
                row=i, column=1, sticky="w", pady=3)

    @staticmethod
    def _make_scrolled_text(parent: tk.Widget, height: int = 20) -> tk.Text:
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=0, sticky="nsew")
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        t = tk.Text(frame, wrap="word", width=60, height=height,
                    font=("Courier New", 9))
        sb = ttk.Scrollbar(frame, orient="vertical", command=t.yview)
        t.configure(yscrollcommand=sb.set)
        t.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        # Tag for colored goal status
        t.tag_configure("active",   foreground=GOAL_COLORS["active"])
        t.tag_configure("pending",  foreground=GOAL_COLORS["pending"])
        t.tag_configure("resolved", foreground=GOAL_COLORS["resolved"])
        t.tag_configure("failed",   foreground=GOAL_COLORS["failed"])
        t.tag_configure("abandoned",foreground=GOAL_COLORS["abandoned"])
        t.tag_configure("header",   font=("Courier New", 9, "bold"))
        t.tag_configure("dim",      foreground="#888888")

        return t

    # -----------------------------------------------------------------------
    # Key bindings
    # -----------------------------------------------------------------------

    def _bind_keys(self) -> None:
        self.root.bind("<space>",  lambda e: self._next())
        self.root.bind("<Right>",  lambda e: self._next())
        self.root.bind("<Left>",   lambda e: self._prev())
        self.root.bind("<Home>",   lambda e: self._go(0))
        self.root.bind("<End>",    lambda e: self._go(max(0, len(self.steps) - 1)))
        self.root.bind("+",        lambda e: self._zoom(1))
        self.root.bind("=",        lambda e: self._zoom(1))
        self.root.bind("-",        lambda e: self._zoom(-1))
        self.root.bind("f",        lambda e: self._fit_zoom())
        self.root.bind("F",        lambda e: self._fit_zoom())
        self.root.bind("l",        lambda e: self._toggle_live())
        self.root.bind("L",        lambda e: self._toggle_live())

    # -----------------------------------------------------------------------
    # Navigation
    # -----------------------------------------------------------------------

    def _go(self, index: int) -> None:
        if not self.steps:
            return
        self.index = max(0, min(index, len(self.steps) - 1))
        self._show_step(self.index)

    def _next(self) -> None:
        self._go(self.index + 1)

    def _prev(self) -> None:
        self._go(self.index - 1)

    def _zoom(self, delta: int) -> None:
        self.cell_size = max(1, min(20, self.cell_size + delta))
        self._redraw_frame()

    def _fit_zoom(self) -> None:
        if not self.steps:
            return
        frame = self.steps[min(self.index, len(self.steps) - 1)].get("_frame")
        if not frame:
            return
        h = len(frame)
        w = len(frame[0]) if frame else 1
        sw = max(self.root.winfo_screenwidth(), 800)
        sh = max(self.root.winfo_screenheight(), 600)
        max_w = max(200, int(sw * 0.45))
        max_h = max(200, int(sh * 0.80))
        self.cell_size = max(1, min(max_w // w, max_h // h))
        self._redraw_frame()

    # -----------------------------------------------------------------------
    # Directory loading
    # -----------------------------------------------------------------------

    def _choose_folder(self) -> None:
        selected = filedialog.askdirectory(
            title="Choose playlog directory",
            initialdir=str(self.playlog_root),
            mustexist=True,
        )
        if selected:
            self._load_dir(Path(selected))

    def _load_dir(self, d: Path) -> None:
        self.playlog_dir = d
        self.steps       = load_steps(d)
        self._last_step_count = len(self.steps)
        self.index       = max(0, len(self.steps) - 1)
        self.root.title(f"ARC-AGI-3 Monitor — {d.name}")
        if self.steps:
            self._show_step(self.index)
        else:
            self.status_var.set(f"No step files found in {d}")

    def _try_latest(self) -> None:
        latest = find_latest_episode_dir(self.playlog_root)
        if latest:
            self._load_dir(latest)
        else:
            self.status_var.set(f"No episodes found under {self.playlog_root}")

    # -----------------------------------------------------------------------
    # Live refresh
    # -----------------------------------------------------------------------

    def _toggle_live(self) -> None:
        if not self.live:
            return
        self._live_paused = not self._live_paused
        self._live_btn.configure(text="Resume Live" if self._live_paused else "Pause Live")

    def _schedule_refresh(self) -> None:
        self._after_id = self.root.after(self.refresh_ms, self._live_tick)

    def _live_tick(self) -> None:
        if not self._live_paused:
            self._live_refresh()
        self._schedule_refresh()

    def _live_refresh(self) -> None:
        """Re-scan the playlog directory for new step files."""
        # First check if a new episode dir appeared
        latest = find_latest_episode_dir(self.playlog_root)
        if latest and latest != self.playlog_dir:
            self._load_dir(latest)
            return

        if not self.playlog_dir:
            return

        new_steps = load_steps(self.playlog_dir)
        if len(new_steps) > len(self.steps):
            self.steps = new_steps
            # Only auto-advance to latest if we were already at the last step
            if self.index >= self._last_step_count - 1:
                self.index = len(self.steps) - 1
            self._last_step_count = len(self.steps)
            self._show_step(self.index)

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------

    def _show_step(self, index: int) -> None:
        if not self.steps:
            return
        self.index = max(0, min(index, len(self.steps) - 1))
        step = self.steps[self.index]

        self._render_frame(step.get("_frame"))
        self._update_progress(step)
        self._update_goals(step)
        self._update_state(step)
        self._update_observer(step)
        self._update_mediator(step)
        self._update_rules(step)
        self._update_costs(step)

        ep   = step.get("episode", "?")
        cyc  = step.get("cycle", "?")
        n    = self.index + 1
        tot  = len(self.steps)
        act  = step.get("action_name", "?")
        lvl  = step.get("levels_completed", "?")
        self.status_var.set(
            f"{self.playlog_dir.name if self.playlog_dir else ''}  |  "
            f"ep={ep} cy={cyc}  |  step {n}/{tot}  |  {act}  |  levels={lvl}"
            + ("  [LIVE]" if self.live and not self._live_paused else "")
        )

    def _render_frame(self, frame: Optional[List[List[int]]]) -> None:
        self.canvas.delete("all")
        if not frame:
            return
        h = len(frame)
        w = len(frame[0]) if frame else 0
        if w == 0:
            return
        cw = w * self.cell_size
        ch = h * self.cell_size
        self.canvas.configure(width=min(cw, 800), height=min(ch, 700))
        self.canvas.configure(scrollregion=(0, 0, cw, ch))
        for y, row in enumerate(frame):
            y0 = y * self.cell_size
            y1 = y0 + self.cell_size
            for x, val in enumerate(row):
                x0 = x * self.cell_size
                x1 = x0 + self.cell_size
                color = PALETTE.get(int(val), "#FFFFFF")
                self.canvas.create_rectangle(x0, y0, x1, y1,
                                             fill=color, outline=color)

    def _redraw_frame(self) -> None:
        if self.steps:
            self._render_frame(self.steps[self.index].get("_frame"))

    def _update_progress(self, step: dict) -> None:
        cs = step.get("change_summary", {})
        bbox = cs.get("bbox")
        bbox_str = "none" if not bbox else \
            f"x={bbox['x_min']}..{bbox['x_max']}, y={bbox['y_min']}..{bbox['y_max']}"

        self._prog_vars["episode"].set(str(step.get("episode", "?")))
        self._prog_vars["cycle"].set(str(step.get("cycle", "?")))
        self._prog_vars["plan_index"].set(str(step.get("plan_index", "?")))
        self._prog_vars["step"].set(
            f"{step.get('step_number', '?')} / {len(self.steps)}")
        self._prog_vars["state"].set(str(step.get("observation_state", "?")))
        self._prog_vars["levels"].set(
            f"{step.get('levels_completed', '?')} / {step.get('win_levels', '?')}")
        self._prog_vars["action"].set(str(step.get("action_name", "?")))
        data = step.get("action_data", {})
        self._prog_vars["data"].set(json.dumps(data) if data else "{}")
        self._prog_vars["diff_count"].set(str(cs.get("diff_count", 0)))
        self._prog_vars["bbox"].set(bbox_str)

    def _update_goals(self, step: dict) -> None:
        goals = step.get("active_goals") or []
        w = self.goals_text
        w.configure(state="normal")
        w.delete("1.0", "end")
        if not goals:
            w.insert("end", "(no goals recorded for this step)")
        else:
            w.insert("end", "Active & Pending Goals\n", "header")
            w.insert("end", "-" * 50 + "\n", "dim")
            for g in goals:
                status   = g.get("status", "?")
                gid      = g.get("id", "?")
                desc     = g.get("description", "?")
                priority = g.get("priority", "?")
                parent   = g.get("parent_id")
                indent   = "  " if parent else ""
                line = f"{indent}[{status:8}] (p={priority}) {gid}: {desc}\n"
                tag = status if status in GOAL_COLORS else ""
                w.insert("end", line, tag)
        w.configure(state="disabled")

    def _update_state(self, step: dict) -> None:
        snap = step.get("state_snapshot") or {}
        w = self.state_text
        w.configure(state="normal")
        w.delete("1.0", "end")
        if not snap:
            w.insert("end", "(no state recorded for this step)")
        else:
            w.insert("end", "State Store\n", "header")
            w.insert("end", "-" * 50 + "\n", "dim")
            for k, v in sorted(snap.items()):
                v_str = json.dumps(v) if not isinstance(v, str) else v
                wrapped = textwrap.fill(v_str, width=56, subsequent_indent="          ")
                w.insert("end", f"  {k}:\n    {wrapped}\n\n")
        w.configure(state="disabled")

    def _update_observer(self, step: dict) -> None:
        text = step.get("observer_analysis") or step.get("decision_note") or ""
        _set_text(self.observer_text, text or "(no OBSERVER analysis recorded)")

    def _update_mediator(self, step: dict) -> None:
        plan = step.get("mediator_plan") or []
        reasoning = step.get("mediator_reasoning") or ""

        plan_lines = []
        for i, s in enumerate(plan, 1):
            action = s.get("action", "?")
            data   = s.get("data") or {}
            data_s = json.dumps(data) if data else ""
            plan_lines.append(f"  {i}. {action}  {data_s}")
        plan_str = "\n".join(plan_lines) if plan_lines else "(no plan)"

        _set_text(self.mediator_plan_text, plan_str)
        _set_text(self.mediator_reason_text,
                  reasoning or "(no reasoning recorded)")

    def _update_rules(self, step: dict) -> None:
        rules = step.get("matched_rules") or []
        if rules:
            text = "Matched rule IDs:\n" + "\n".join(f"  - {r}" for r in rules)
        else:
            text = "(no rules matched this cycle)"
        _set_text(self.rules_text, text)

    def _update_costs(self, step: dict) -> None:
        self._cost_vars["cost_episode"].set(
            f"${step.get('cost_episode', 0):.6f}")
        self._cost_vars["tokens_input"].set(
            f"{step.get('tokens_input', 0):,}")
        self._cost_vars["tokens_output"].set(
            f"{step.get('tokens_output', 0):,}")
        self._cost_vars["api_calls"].set(str(step.get("api_calls", 0)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_text(widget: tk.Text, content: str) -> None:
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget.insert("1.0", content)
    widget.configure(state="disabled")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time developer monitor for the ARC-AGI-3 ensemble.")
    p.add_argument("--playlog-dir", dest="playlog_dir", type=Path, default=None,
                   help="Specific playlog episode directory to open. "
                        "Default: latest under ./playlogs/")
    p.add_argument("--playlog-root", dest="playlog_root", type=Path,
                   default=DEFAULT_PLAYLOG_ROOT,
                   help="Root folder that contains per-episode subdirs.")
    p.add_argument("--refresh-ms",  dest="refresh_ms", type=int,
                   default=REFRESH_MS_DEFAULT,
                   help=f"Live refresh interval in ms (default: {REFRESH_MS_DEFAULT})")
    p.add_argument("--no-live",     dest="no_live", action="store_true",
                   help="Disable auto-refresh (static mode).")
    p.add_argument("--cell-size",   dest="cell_size", type=int, default=6,
                   help="Initial pixel size per frame cell (default: 6).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    root = tk.Tk()
    style = ttk.Style(root)
    for theme in ("vista", "clam", "alt", "default"):
        if theme in style.theme_names():
            style.theme_use(theme)
            break

    monitor = EnsembleMonitor(
        root         = root,
        playlog_root = args.playlog_root,
        playlog_dir  = args.playlog_dir,
        refresh_ms   = args.refresh_ms,
        live         = not args.no_live,
    )
    monitor.cell_size = args.cell_size

    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    w  = min(1400, max(1100, sw - 80))
    h  = min(900,  max(700,  sh - 80))
    root.geometry(f"{w}x{h}+0+0")
    root.minsize(900, 640)
    root.mainloop()


if __name__ == "__main__":
    main()
