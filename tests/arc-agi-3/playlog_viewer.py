import argparse
import json
import re
import textwrap
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Dict, List, Optional


PALETTE = {
    0:  "#000000",
    1:  "#0074D9",
    2:  "#FF4136",
    3:  "#2ECC40",
    4:  "#AAAAAA",
    5:  "#FFDC00",
    6:  "#AA00FF",
    7:  "#FF851B",
    8:  "#7FDBFF",
    9:  "#F012BE",
    10: "#7B7B7B",
    11: "#85144B",
    12: "#39CCCC",
}

COLOR_NAMES = {
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "grey",
    5: "yellow", 6: "magenta", 7: "orange", 8: "azure", 9: "white",
    10: "color10", 11: "color11", 12: "color12",
}

DEFAULT_ROOT = Path(r"C:\_backup\github\khub-knowledge-fabric\usecases\arc-agi-3\python\playlogs")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_steps(playlog_dir: Path) -> List[Dict]:
    step_files = sorted(playlog_dir.glob("[0-9][0-9][0-9][0-9]-*.json"))
    if not step_files:
        # Fallback: older 3-digit naming
        step_files = sorted(playlog_dir.glob("[0-9][0-9][0-9]-*.json"))
    steps = []
    for path in step_files:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        frame = payload.get("returned", {}).get("frame")
        payload["_frame"] = decode_frame(frame)
        payload["_path"] = path
        steps.append(payload)
    if not steps:
        raise FileNotFoundError(f"No step JSON files found in {playlog_dir}")
    return steps


def latest_playlog(root: Path) -> Path:
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No playlog folders found under {root}")
    return sorted(candidates)[-1]


def decode_frame(frame_value) -> Optional[List[List[int]]]:
    if not frame_value:
        return None
    if isinstance(frame_value, list):
        first = frame_value[0]
        if isinstance(first, list):
            return first
        if isinstance(first, str):
            return parse_frame_string(first)
    if isinstance(frame_value, str):
        return parse_frame_string(frame_value)
    return None


def parse_frame_string(frame_text: str) -> List[List[int]]:
    rows = []
    for raw_line in frame_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        numbers = [int(token) for token in re.findall(r"-?\d+", line)]
        if numbers:
            rows.append(numbers)
    return rows


def parse_observer_json(text: str) -> dict:
    """Extract and parse the JSON block inside observer_analysis.

    Falls back to per-field regex extraction when the JSON is truncated
    (which happens when the LLM response was cut off before finishing).
    """
    if not text:
        return {}
    # Strip ```json ... ``` fences
    stripped = re.sub(r"^```json\s*", "", text.strip(), flags=re.MULTILINE)
    stripped = re.sub(r"```\s*$", "", stripped.strip(), flags=re.MULTILINE)

    # Try full parse first
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Try finding outermost {...} block (handles trailing prose after the JSON)
    m = re.search(r"\{.*\}", stripped, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # Fallback: regex-extract individual string and list fields from truncated JSON.
    result: dict = {}

    def _grab_str(field: str) -> Optional[str]:
        pat = rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"'
        m2 = re.search(pat, stripped, re.DOTALL)
        return m2.group(1).replace("\\n", "\n").replace('\\"', '"') if m2 else None

    def _grab_list(field: str) -> list:
        pat = rf'"{field}"\s*:\s*\[(.*?)\]'
        m2 = re.search(pat, stripped, re.DOTALL)
        if not m2:
            return []
        inner = m2.group(1).strip()
        if not inner:
            return []
        # Each item may be a string or object — collect quoted strings and {...} objects
        items = []
        for s in re.findall(r'"((?:[^"\\]|\\.)*)"', inner):
            items.append(s.replace("\\n", "\n").replace('\\"', '"'))
        return items

    for field in ("level_description", "hypothesized_goal", "reasoning"):
        val = _grab_str(field)
        if val:
            result[field] = val + ("  [TRUNCATED]" if field == "reasoning" and not stripped.rstrip().endswith("}") else "")

    for field in ("visual_observations", "action_characterizations", "promising_actions"):
        items = _grab_list(field)
        if items:
            result[field] = items

    # identified_objects: list of {color, role, confidence, label, description}
    objs_m = re.search(r'"identified_objects"\s*:\s*\[(.*?)\]', stripped, re.DOTALL)
    if objs_m:
        objs_text = objs_m.group(1)
        # Split on top-level {...} blocks
        objs = []
        for obj_m in re.finditer(r'\{([^{}]*)\}', objs_text):
            try:
                objs.append(json.loads("{" + obj_m.group(1) + "}"))
            except Exception:
                pass
        if objs:
            result["identified_objects"] = objs

    return result


def fmt_wrap(text: str, width: int = 72, indent: str = "") -> str:
    if not text:
        return ""
    return textwrap.fill(str(text), width=width, subsequent_indent=indent)


# ---------------------------------------------------------------------------
# Main viewer
# ---------------------------------------------------------------------------

class PlaylogViewer:
    def __init__(self, root: tk.Tk, playlog_dir: Path, cell_size: int, autoplay_ms: int):
        self.root = root
        self.playlog_dir = playlog_dir
        self.steps = load_steps(playlog_dir)
        self.index = 0
        self.cell_size = cell_size
        self.autoplay_ms = autoplay_ms
        self.autoplay = False
        self.after_id = None

        first_frame = self.steps[0]["_frame"]
        if not first_frame:
            raise ValueError("First step does not contain frame data")
        self.grid_height = len(first_frame)
        self.grid_width = len(first_frame[0])
        self.cell_size = self._fit_cell_size(cell_size)

        self.root.title(f"ARC Playlog Viewer — {playlog_dir.name}")
        self._build_ui()
        self._bind_keys()
        self.show_step(0)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _fit_cell_size(self, requested: int) -> int:
        screen_w = max(self.root.winfo_screenwidth(), 800)
        screen_h = max(self.root.winfo_screenheight(), 600)
        max_canvas_w = max(320, min(int(screen_w * 0.42), screen_w - 620))
        max_canvas_h = max(320, int(screen_h * 0.72))
        cell_w = max_canvas_w // self.grid_width
        cell_h = max_canvas_h // self.grid_height
        return max(2, min(requested, cell_w, cell_h))

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # ── Left: frame canvas + controls ──────────────────────────────
        left = ttk.Frame(self.root, padding=10)
        left.grid(row=0, column=0, sticky="nsew")

        self.canvas_width = self.grid_width * self.cell_size
        self.canvas_height = self.grid_height * self.cell_size

        canvas_frame = ttk.Frame(left)
        canvas_frame.grid(row=0, column=0, columnspan=4, sticky="nsew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.canvas_width, height=self.canvas_height,
            background="black", highlightthickness=1,
            highlightbackground="#666666",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        y_scroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.canvas.configure(scrollregion=(0, 0, self.canvas_width, self.canvas_height))

        # Status bar below canvas
        self.status_var = tk.StringVar()
        ttk.Label(left, textvariable=self.status_var, padding=(0, 6, 0, 6)).grid(
            row=1, column=0, columnspan=4, sticky="w"
        )

        # Navigation row
        ttk.Button(left, text="Prev",  command=self.prev_step).grid(row=2, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(left, text="Next",  command=self.next_step).grid(row=2, column=1, sticky="ew", padx=(0, 4))
        self.play_button = ttk.Button(left, text="Play", command=self.toggle_autoplay)
        self.play_button.grid(row=2, column=2, sticky="ew", padx=(0, 4))
        ttk.Button(left, text="Open…", command=self.choose_folder).grid(row=2, column=3, sticky="ew")

        # Zoom row
        ttk.Button(left, text="Zoom +", command=lambda: self.adjust_zoom(1)).grid(row=3, column=0, sticky="ew", pady=(6, 0), padx=(0, 4))
        ttk.Button(left, text="Zoom -", command=lambda: self.adjust_zoom(-1)).grid(row=3, column=1, sticky="ew", pady=(6, 0), padx=(0, 4))
        ttk.Button(left, text="Fit",    command=self.fit_zoom).grid(row=3, column=2, sticky="ew", pady=(6, 0), padx=(0, 4))
        ttk.Label(left, text="Space=next  ←→=step  P=play  +/-=zoom", padding=(0, 6, 0, 0)).grid(
            row=3, column=3, sticky="w"
        )

        # ── Right: tabbed notebook ──────────────────────────────────────
        right = ttk.Frame(self.root, padding=(0, 10, 10, 10))
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.notebook = ttk.Notebook(right)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self._tab_overview  = self._make_text_tab("Overview")
        self._tab_observer  = self._make_text_tab("OBSERVER")
        self._tab_mediator  = self._make_text_tab("MEDIATOR")
        self._tab_state     = self._make_text_tab("State")

    def _make_text_tab(self, title: str) -> tk.Text:
        """Add a scrollable Text tab to the notebook and return the Text widget."""
        frame = ttk.Frame(self.notebook, padding=6)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self.notebook.add(frame, text=title)

        text = tk.Text(
            frame,
            wrap="word",
            width=72,
            font=("Consolas", 9),
            state="disabled",
            relief="flat",
            padx=6, pady=6,
        )
        text.grid(row=0, column=0, sticky="nsew")

        vsb = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        text.configure(yscrollcommand=vsb.set)

        # Tag styles
        text.tag_configure("h1",    font=("Consolas", 10, "bold"), spacing1=6)
        text.tag_configure("h2",    font=("Consolas", 9,  "bold"), spacing1=4)
        text.tag_configure("label", font=("Consolas", 9,  "bold"))
        text.tag_configure("ok",    foreground="#007700")
        text.tag_configure("warn",  foreground="#BB5500")
        text.tag_configure("dim",   foreground="#888888")
        text.tag_configure("rule",  foreground="#0055AA")
        text.tag_configure("goal_active",  foreground="#007700")
        text.tag_configure("goal_pending", foreground="#BB5500")
        text.tag_configure("goal_done",    foreground="#888888")

        return text

    # ------------------------------------------------------------------
    # Key bindings
    # ------------------------------------------------------------------

    def _bind_keys(self) -> None:
        self.root.bind("<space>",  lambda e: self.next_step())
        self.root.bind("<Right>",  lambda e: self.next_step())
        self.root.bind("<Left>",   lambda e: self.prev_step())
        self.root.bind("<Home>",   lambda e: self.show_step(0))
        self.root.bind("<End>",    lambda e: self.show_step(len(self.steps) - 1))
        self.root.bind("p",        lambda e: self.toggle_autoplay())
        self.root.bind("P",        lambda e: self.toggle_autoplay())
        self.root.bind("+",        lambda e: self.adjust_zoom(1))
        self.root.bind("-",        lambda e: self.adjust_zoom(-1))
        self.root.bind("=",        lambda e: self.adjust_zoom(1))
        self.root.bind("f",        lambda e: self.fit_zoom())
        self.root.bind("F",        lambda e: self.fit_zoom())

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def show_step(self, index: int) -> None:
        self.index = max(0, min(index, len(self.steps) - 1))
        step = self.steps[self.index]
        self._render_frame(step["_frame"])
        self.canvas.xview_moveto(0.0)
        self.canvas.yview_moveto(0.0)
        self._update_status(step)
        self._update_overview(step)
        self._update_observer(step)
        self._update_mediator(step)
        self._update_state(step)

    def next_step(self) -> None:
        if self.index < len(self.steps) - 1:
            self.show_step(self.index + 1)
        else:
            self.stop_autoplay()

    def prev_step(self) -> None:
        self.show_step(self.index - 1)

    def choose_folder(self) -> None:
        selected = filedialog.askdirectory(
            title="Choose playlog directory",
            initialdir=str(self.playlog_dir.parent),
            mustexist=True,
        )
        if not selected:
            return
        self.stop_autoplay()
        self.playlog_dir = Path(selected)
        self.steps = load_steps(self.playlog_dir)
        self.index = 0
        self.root.title(f"ARC Playlog Viewer — {self.playlog_dir.name}")
        self.show_step(0)

    # ------------------------------------------------------------------
    # Frame rendering
    # ------------------------------------------------------------------

    def _render_frame(self, frame: Optional[List[List[int]]]) -> None:
        self.canvas.delete("all")
        if not frame:
            return
        self.canvas_width  = self.grid_width  * self.cell_size
        self.canvas_height = self.grid_height * self.cell_size
        self.canvas.configure(width=self.canvas_width, height=self.canvas_height)
        self.canvas.configure(scrollregion=(0, 0, self.canvas_width, self.canvas_height))
        for y, row in enumerate(frame):
            y0, y1 = y * self.cell_size, (y + 1) * self.cell_size
            for x, value in enumerate(row):
                x0, x1 = x * self.cell_size, (x + 1) * self.cell_size
                color = PALETTE.get(value, "#FFFFFF")
                self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, fill=color)

    def adjust_zoom(self, delta: int) -> None:
        self.cell_size = max(1, min(16, self.cell_size + delta))
        self.show_step(self.index)

    def fit_zoom(self) -> None:
        self.cell_size = self._fit_cell_size(1)
        self.show_step(self.index)

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _update_status(self, step: dict) -> None:
        cs = step.get("change_summary", {})
        self.status_var.set(
            f"{self.playlog_dir.name}  |  step {step.get('step_number', '?')}/"
            f"{len(self.steps)}  |  cycle {step.get('cycle', '?')}  |  "
            f"{step.get('action_name', '?')}  |  "
            f"levels {step.get('levels_completed', '?')}/{step.get('win_levels', '?')}  |  "
            f"diff={cs.get('diff_count', 0)}"
        )

    # ------------------------------------------------------------------
    # Tab 1: Overview  (goals, concept bindings, cost)
    # ------------------------------------------------------------------

    def _update_overview(self, step: dict) -> None:
        t = self._tab_overview
        t.configure(state="normal")
        t.delete("1.0", "end")

        # ── Step header ───────────────────────────────────────────────
        t.insert("end", f"Step {step.get('step_number','?')}  "
                        f"cycle={step.get('cycle','?')}  "
                        f"plan_idx={step.get('plan_index','?')}  "
                        f"action={step.get('action_name','?')}\n", "h1")
        cs = step.get("change_summary", {})
        bbox = cs.get("bbox")
        bbox_str = ("none" if not bbox
                    else f"rows {bbox.get('y_min')}–{bbox.get('y_max')}, "
                         f"cols {bbox.get('x_min')}–{bbox.get('x_max')}")
        t.insert("end", f"state={step.get('observation_state','')}  "
                        f"diff={cs.get('diff_count',0)}  bbox={bbox_str}\n\n", "dim")

        # ── Goals ─────────────────────────────────────────────────────
        t.insert("end", "GOALS\n", "h2")
        goals: list = step.get("active_goals", [])
        if goals:
            # Build parent map for indentation
            by_id = {g["id"]: g for g in goals}
            for g in goals:
                indent = "    " if g.get("parent_id") else ""
                status = g.get("status", "?").upper()
                pri    = g.get("priority", "?")
                desc   = g.get("description", "")
                line   = f"{indent}[{pri}] {status:8s}  {desc}\n"
                tag = ("goal_active"  if status == "ACTIVE"
                       else "goal_done" if status in ("RESOLVED", "FAILED")
                       else "goal_pending")
                t.insert("end", line, tag)
        else:
            t.insert("end", "  (none)\n", "dim")
        t.insert("end", "\n")

        # ── Concept bindings ──────────────────────────────────────────
        t.insert("end", "CONCEPT BINDINGS\n", "h2")
        snap = step.get("state_snapshot", {})
        cb   = snap.get("concept_bindings", {})
        if cb:
            for k, v in sorted(cb.items(), key=lambda x: str(x[0])):
                if k == "wall_colors":
                    names = ", ".join(
                        f"{COLOR_NAMES.get(c, f'color{c}')}({c})" for c in sorted(v)
                    )
                    t.insert("end", f"  wall (game-local): [{names}]\n", "warn")
                elif isinstance(v, dict):
                    role  = v.get("role", "?")
                    conf  = v.get("confidence", 0)
                    lobs  = v.get("level_obs", v.get("observations", "?"))
                    tobs  = v.get("total_obs",  v.get("observations", "?"))
                    cname = COLOR_NAMES.get(int(k), f"color{k}") if str(k).isdigit() else k
                    tag   = "ok" if conf >= 0.7 else ("warn" if conf >= 0.4 else "dim")
                    t.insert("end", f"  color{k} ({cname}): ", "label")
                    t.insert("end", f"{role}  conf={conf:.0%}  "
                                    f"lv={lobs}obs  total={tobs}obs\n", tag)
                else:
                    cname = COLOR_NAMES.get(int(k), f"color{k}") if str(k).isdigit() else k
                    t.insert("end", f"  color{k} ({cname}): {v}\n")
        else:
            t.insert("end", "  (none yet)\n", "dim")
        t.insert("end", "\n")

        # ── Cost / tokens ─────────────────────────────────────────────
        t.insert("end", "COST (episode cumulative)\n", "h2")
        t.insert("end",
            f"  ${step.get('cost_episode', 0):.4f}  |  "
            f"in={step.get('tokens_input', 0):,}tok  "
            f"out={step.get('tokens_output', 0):,}tok  "
            f"calls={step.get('api_calls', 0)}\n"
        )

        t.configure(state="disabled")

    # ------------------------------------------------------------------
    # Tab 2: OBSERVER
    # ------------------------------------------------------------------

    def _update_observer(self, step: dict) -> None:
        t = self._tab_observer
        t.configure(state="normal")
        t.delete("1.0", "end")

        raw  = step.get("observer_analysis", "")
        obs  = parse_observer_json(raw)

        if not obs:
            t.insert("end", "OBSERVER OUTPUT\n", "h1")
            t.insert("end", raw or "(empty)\n")
            t.configure(state="disabled")
            return

        # Level description
        t.insert("end", "LEVEL DESCRIPTION\n", "h2")
        t.insert("end", fmt_wrap(obs.get("level_description", "—"), indent="  ") + "\n\n")

        # Hypothesized goal
        t.insert("end", "HYPOTHESIZED GOAL\n", "h2")
        t.insert("end", fmt_wrap(obs.get("hypothesized_goal", "—"), indent="  ") + "\n\n")

        # Identified objects
        objects = obs.get("identified_objects", [])
        if objects:
            t.insert("end", f"IDENTIFIED OBJECTS  ({len(objects)})\n", "h2")
            for obj in objects:
                if isinstance(obj, dict):
                    line = (f"  {obj.get('color','?')}  {obj.get('role','?')}  "
                            f"conf={obj.get('confidence','?')}  "
                            f"{obj.get('label','')}  {obj.get('description','')}")
                else:
                    line = f"  {obj}"
                tag = "ok" if "[CONFIRMED]" in str(obj) else ("warn" if "[GUESS]" in str(obj) else "")
                t.insert("end", fmt_wrap(line, indent="    ") + "\n", tag)
            t.insert("end", "\n")

        # Visual observations
        vis = obs.get("visual_observations", [])
        if vis:
            t.insert("end", f"VISUAL OBSERVATIONS  ({len(vis)})\n", "h2")
            for item in vis:
                t.insert("end", fmt_wrap(f"  • {item}", indent="    ") + "\n")
            t.insert("end", "\n")

        # Action characterizations
        chars = obs.get("action_characterizations", [])
        if chars:
            t.insert("end", f"ACTION CHARACTERIZATIONS  ({len(chars)})\n", "h2")
            for item in chars:
                t.insert("end", fmt_wrap(f"  • {item}", indent="    ") + "\n")
            t.insert("end", "\n")

        # Promising actions
        promising = obs.get("promising_actions", [])
        if promising:
            t.insert("end", "PROMISING ACTIONS\n", "h2")
            for item in promising:
                t.insert("end", f"  • {item}\n")
            t.insert("end", "\n")

        # Reasoning
        reasoning = obs.get("reasoning", "")
        if reasoning:
            t.insert("end", "REASONING\n", "h2")
            t.insert("end", fmt_wrap(reasoning, indent="  ") + "\n")

        t.configure(state="disabled")

    # ------------------------------------------------------------------
    # Tab 3: MEDIATOR
    # ------------------------------------------------------------------

    def _update_mediator(self, step: dict) -> None:
        t = self._tab_mediator
        t.configure(state="normal")
        t.delete("1.0", "end")

        # Action plan
        plan: list = step.get("mediator_plan", [])
        t.insert("end", f"ACTION PLAN  ({len(plan)} action(s))\n", "h2")
        if plan:
            for i, act in enumerate(plan, 1):
                name = act.get("action", "?") if isinstance(act, dict) else str(act)
                data = act.get("data", {}) if isinstance(act, dict) else {}
                data_str = f"  {data}" if data else ""
                t.insert("end", f"  {i}. {name}{data_str}\n")
        else:
            t.insert("end", "  (none)\n", "dim")
        t.insert("end", "\n")

        # Matched rules
        rules: list = step.get("matched_rules", [])
        t.insert("end", f"MATCHED RULES  ({len(rules)})\n", "h2")
        if rules:
            line = "  " + "  ".join(str(r) for r in rules)
            t.insert("end", fmt_wrap(line, indent="  ") + "\n", "rule")
        else:
            t.insert("end", "  (none)\n", "dim")
        t.insert("end", "\n")

        # Mediator reasoning
        reasoning = step.get("mediator_reasoning", "")
        t.insert("end", "REASONING\n", "h2")
        t.insert("end", fmt_wrap(reasoning or "(empty)", indent="  ") + "\n\n")

        # Decision note (raw from step JSON)
        note = step.get("decision_note", "")
        if note and note != reasoning:
            t.insert("end", "DECISION NOTE (raw)\n", "h2")
            t.insert("end", fmt_wrap(note, indent="  ") + "\n")

        t.configure(state="disabled")

    # ------------------------------------------------------------------
    # Tab 4: State
    # ------------------------------------------------------------------

    def _update_state(self, step: dict) -> None:
        t = self._tab_state
        t.configure(state="normal")
        t.delete("1.0", "end")

        snap: dict = step.get("state_snapshot", {})
        if not snap:
            t.insert("end", "(empty state snapshot)\n", "dim")
            t.configure(state="disabled")
            return

        for key, val in snap.items():
            if key == "concept_bindings":
                # Already shown in Overview tab — skip here
                continue
            if key == "action_effects":
                # Summarise only — too verbose to show raw
                t.insert("end", "action_effects\n", "h2")
                for act_name, act_data in val.items():
                    total   = act_data.get("total_calls", 0)
                    nonzero = act_data.get("nonzero_calls", 0)
                    lv_adv  = act_data.get("level_advances", 0)
                    obj_obs = act_data.get("object_observations", [])
                    last_summary = ""
                    if obj_obs:
                        last_summary = obj_obs[-1].get("summary", "")
                    t.insert("end", f"  {act_name}: ", "label")
                    t.insert("end",
                        f"calls={total} nonzero={nonzero} lv_advances={lv_adv}\n")
                    if last_summary:
                        t.insert("end",
                            f"    last: {fmt_wrap(last_summary, width=68, indent='    ')}\n",
                            "dim")
                t.insert("end", "\n")
            elif isinstance(val, dict):
                t.insert("end", f"{key}\n", "h2")
                for k2, v2 in val.items():
                    t.insert("end", f"  {k2}: ", "label")
                    t.insert("end", f"{v2}\n")
                t.insert("end", "\n")
            elif isinstance(val, list):
                t.insert("end", f"{key}\n", "h2")
                for item in val:
                    t.insert("end", f"  • {item}\n")
                t.insert("end", "\n")
            else:
                t.insert("end", f"{key}: ", "label")
                t.insert("end", f"{val}\n")

        t.configure(state="disabled")

    # ------------------------------------------------------------------
    # Autoplay
    # ------------------------------------------------------------------

    def toggle_autoplay(self) -> None:
        self.autoplay = not self.autoplay
        self.play_button.configure(text="Pause" if self.autoplay else "Play")
        if self.autoplay:
            self._schedule_next()
        else:
            self.stop_autoplay()

    def _schedule_next(self) -> None:
        self.after_id = self.root.after(self.autoplay_ms, self._autoplay_tick)

    def _autoplay_tick(self) -> None:
        if not self.autoplay:
            return
        if self.index >= len(self.steps) - 1:
            self.stop_autoplay()
            return
        self.next_step()
        self._schedule_next()

    def stop_autoplay(self) -> None:
        self.autoplay = False
        self.play_button.configure(text="Play")
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ARC-AGI-3 playlogs.")
    parser.add_argument(
        "--playlog-dir", type=Path, default=None,
        help="Specific playlog directory to open. Defaults to the latest under the default root.",
    )
    parser.add_argument("--cell-size",   type=int, default=1,   help="Pixel size per grid cell.")
    parser.add_argument("--autoplay-ms", type=int, default=600, help="Autoplay delay in ms.")
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    playlog_dir = args.playlog_dir or latest_playlog(DEFAULT_ROOT)

    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")

    viewer = PlaylogViewer(root, playlog_dir, cell_size=args.cell_size, autoplay_ms=args.autoplay_ms)

    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    width  = min(1280, max(1000, screen_w - 60))
    height = min(820,  max(680,  screen_h - 60))
    root.geometry(f"{width}x{height}+0+0")
    root.minsize(900, 600)
    root.mainloop()


if __name__ == "__main__":
    main()
