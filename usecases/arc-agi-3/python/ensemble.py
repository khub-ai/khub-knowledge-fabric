"""
ensemble.py — ARC-AGI-3 episode orchestrator.

One episode = one attempt to complete an ARC-AGI-3 environment (e.g. LS20).

Architecture (per OBSERVER-MEDIATOR cycle within an episode):
  Round 0  — Rule matching: which patterns apply to the current state?
  Round 1  — OBSERVER: analyze current frame -> structured observation
  Round 2  — MEDIATOR: observation + rules + goals -> action plan + goal updates
  Round 3  — Execute: call env.step() for each planned action; stop if a level
             advances or WIN/GAME_OVER is detected; loop back to Round 1.

Post-episode: fire rule success/failure based on levels completed.
"""

from __future__ import annotations
import asyncio
import datetime
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import sys
_KF_ROOT = Path(__file__).resolve().parents[3]
if str(_KF_ROOT) not in sys.path:
    sys.path.insert(0, str(_KF_ROOT))

from core.knowledge.state          import StateManager
from core.knowledge.goals          import GoalManager
from core.knowledge.goal_templates import GoalTemplateRegistry, push_template_into_manager
from core.knowledge.game_knowledge import GameKnowledgeRegistry
from core.pipeline.agents import (
    reset_cost_tracker,
    get_cost_tracker,
    DEFAULT_MODEL,
    call_agent as _core_call_agent,
)

from rules import RuleEngine
from tools import ToolRegistry
from object_tracker import (
    diff_objects, format_object_diff, summarize_current_objects,
    detect_wall_contacts, infer_typical_direction,
    compute_trend_predictions, detect_objects, color_name,
    detect_containment, detect_contacts, auto_detect_concepts,
    infer_action_directions, detect_arena_delta, format_structural_context,
)
from agents import (
    run_observer,
    run_mediator,
    parse_concept_bindings,
    obs_state_name,
    obs_levels_completed,
    obs_frame,
    frame_to_str,
    format_action_space,
    _format_action_effects,
)
from core.knowledge.co_occurrence import CoOccurrenceRegistry, events_from_step
from nav_bfs import (
    compute_navigation_plan,
    find_player_position,
    format_nav_plan,
)


# ---------------------------------------------------------------------------
# Frame export helper
# ---------------------------------------------------------------------------

def save_level_frame(
    frame: list,
    env_id: str,
    level: int,
    out_dir: Path,
    scale: int = 8,
) -> None:
    """Save a 64×64 game frame as a PNG in out_dir.

    Filename: {env_id}_level{level+1}.png  (level is 0-indexed internally,
    so level 0 = displayed "level 1").
    Silently skipped if Pillow is not installed.
    """
    try:
        from render_replay import frame_to_image, add_label
        from PIL import Image  # noqa: F401 — confirm PIL available
    except ImportError:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    img = frame_to_image(frame, scale=scale)
    label = (
        f"{env_id} — level {level + 1}  "
        f"({len(frame[0])}x{len(frame)} grid)"
    )
    img = add_label(img, label)
    fname = out_dir / f"{env_id}_level{level + 1}.png"
    img.save(fname)


# ---------------------------------------------------------------------------
# Tunable limits (overridable by harness)
# ---------------------------------------------------------------------------

MAX_STEPS  = 200   # hard cap on total env.step() calls per episode
MAX_CYCLES = 40    # hard cap on OBSERVER-MEDIATOR cycles per episode

# Feature flag: skip the OBSERVER LLM call on puzzle levels (≥6 visual groups)
# and pass the structural context directly to the MEDIATOR.
# Set to False to revert to full OBSERVER calls on every cycle.
SKIP_OBSERVER_FOR_PUZZLES = True

# Competition mode: when True, ALL per-game artifacts (LS20 BFS planner, TR87
# slot-strip detector, hardcoded subplans, puzzle bypass paths) are excluded.
# KF must solve every game using only game-agnostic primitives and concepts
# learned from training. Training mode (False) keeps the bespoke solvers
# enabled so they can be used as ground-truth references for the generalized
# layer under development.
COMPETITION_MODE = True

# In competition mode, execute at most this many actions per MEDIATOR plan
# before breaking and letting the MEDIATOR re-evaluate with fresh observations.
# Keeps the MEDIATOR from committing to a long wrong plan without feedback.
COMPETITION_MAX_PLAN_CHUNK = 3

# Hard USD cap on LLM spending per episode in competition mode.
# Once the accumulated cost (OBSERVER + MEDIATOR calls) exceeds this threshold,
# all LLM calls are suppressed for the rest of the episode — only BFS-direct
# execution continues.  Set to 0 to disable the cap.
# Rationale: each OBSERVER agent turn costs ~$0.05-0.15; a 40-cycle run with
# the 4-turn tool loop can reach $10-20 before the BFS-direct bypass kicks in.
COMPETITION_MAX_LLM_USD = 1.0

# Maximum number of LLM cycles (OBSERVER+MEDIATOR) per episode in competition
# mode.  After this many cycles the LLM path is silently skipped; BFS-direct
# continues.  Set to 0 to disable.
COMPETITION_MAX_LLM_CYCLES = 8

# Known action sequences to pre-solve (skip) levels that are already mastered.
# Keys are env_id strings.  Values are ordered action names.
# LS20: ACTION3×3 + ACTION1×4 + ACTION4×3 + ACTION1×3 = 13 steps to complete level 1.
_KNOWN_SUBPLANS: dict[str, list[str]] = {
    # ls20 level 1→2 (13 steps): navigate cross at rows 47-48, cols 50-52
    # ls20 level 2→3 (45 steps): shape-match puzzle
    #   - Navigate to rotation changer at (49,45), touch 3× to reach 270°
    #   - Collect step-counter resets at (40,51)→step20 and (15,16)→step40
    #   - Navigate to target at (14,40) with rotation=270°
    # ls20 level 3→4 (41 steps): color+rotation+push-pad puzzle
    #   - Go up col 9 to y=5 (8×A1): yjgargdic_r push pad fires at (9,5) → pushed to (34,5)
    #   - Left to (29,5) (A3), down to (29,15) (2×A2)
    #   - Right to (34,15): collect reset1 (A4)
    #   - Down to (34,30) (3×A2), left to (29,30) (A3)
    #   - Down to (29,45): touch color changer col→1 (3×A2)
    #   - Up to (29,30) (3×A1), left to (19,30): collect reset2 (2×A3)
    #   - Up to (19,25) (A1), right across maze to (54,25) (7×A4)
    #   - Up to (54,10) (3×A1), left to (49,10): touch rot changer rot→1 (A3)
    #   - Up to (49,5) (A1), down to (49,10): touch rot changer rot→2 (A2)
    #   - Up to (49,5) (A1), right to (54,5): kapcaakvb_b push pad → pushed to (54,45) (A4)
    #   - Down to (54,50): target with rot=2, col=1 → WIN (A2)
    # ls20 level 4→5 (43 steps): shape+color+push-pad puzzle
    #   - Start (54,5): shape=4→need 5 (mkjdaccuuf at 24,30), color=2→need 1 (3× soyhouuebz at 34,30)
    #   - A3x3: left to (39,5)
    #   - A2x3: down to (39,20)
    #   - A3: step to (34,20) → yjgargdic_r pad at (33,20) fires → pushed right to (54,20)
    #   - A2x2: down to (54,30)
    #   - A3x2: left to (44,30)
    #   - A1: step to (44,25) → tihiodtoj_l pad at (45,25) fires → pushed left to (34,25)
    #   - A2,A1,A2,A1,A2: bounce on color changer at (34,30): col 2→3→0→1
    #   - A1x2: up from (34,25) → (34,20) → yjgargdic_r pad → pushed to (54,20)
    #   - A3x2: left to (49,20) → kapcaakvb_b pad at (44,19) fires → pushed down to (44,45)
    #   - A1: step to (44,40) → tihiodtoj_l pad at (45,40) fires → pushed left to (34,40)
    #   - A2,A3x2: down and left to (24,45)
    #   - A1: step to (24,40) → tihiodtoj_l pad at (25,40) fires → pushed left to (9,40)
    #   - A1: step to (9,35) → yjgargdic_r pad at (8,35) fires → pushed right to (24,35)
    #   - A1: step to (24,30) → mkjdaccuuf shape changer fires: shape 4→5
    #   - A2x2,A4,A1x4: down, right, up navigating to (19,15) → reset1 collected (ctr→42)
    #   - A4,A1,A4,A1x2: navigate right and up to (24,5)
    #   - A3x3: left to (9,5) → target with shape=5, col=1, rot=0 → WIN
    "ls20": (
        ["ACTION3"] * 3 + ["ACTION1"] * 4 + ["ACTION4"] * 3 + ["ACTION1"] * 3
        + ["ACTION1", "ACTION4", "ACTION1", "ACTION1", "ACTION1", "ACTION1", "ACTION1",
           "ACTION4", "ACTION4", "ACTION2", "ACTION4", "ACTION2", "ACTION2", "ACTION2",
           "ACTION2", "ACTION2", "ACTION2", "ACTION2", "ACTION3", "ACTION3",
           "ACTION4", "ACTION4", "ACTION1", "ACTION3", "ACTION4"]
        + ["ACTION1"] * 7
        + ["ACTION3"] * 7
        + ["ACTION2"] * 6
        + ["ACTION1"] * 8
        + ["ACTION3"]
        + ["ACTION2"] * 2
        + ["ACTION4"]
        + ["ACTION2"] * 3
        + ["ACTION3"]
        + ["ACTION2"] * 3
        + ["ACTION1"] * 3
        + ["ACTION3"] * 2
        + ["ACTION1"]
        + ["ACTION4"] * 7
        + ["ACTION1"] * 3
        + ["ACTION3"]
        + ["ACTION1"]
        + ["ACTION2"]
        + ["ACTION1"]
        + ["ACTION4"]
        + ["ACTION2"]
        + ["ACTION3"] * 3
        + ["ACTION2"] * 3
        + ["ACTION3"]
        + ["ACTION2"] * 2
        + ["ACTION3"] * 2
        + ["ACTION1"]
        + ["ACTION2", "ACTION1", "ACTION2", "ACTION1", "ACTION2"]
        + ["ACTION1"] * 2
        + ["ACTION3"] * 2
        + ["ACTION1"]
        + ["ACTION2"]
        + ["ACTION3"] * 2
        + ["ACTION1"] * 3
        + ["ACTION2"] * 2
        + ["ACTION4"]
        + ["ACTION1"] * 4
        + ["ACTION4"]
        + ["ACTION1"]
        + ["ACTION4"]
        + ["ACTION1"] * 2
        + ["ACTION3"] * 3
        # ls20 level 5→6 (44 steps): shape+color+rot+reset+push-pad puzzle
        #   - Start (49,40): shape=4→need 0 (mkjdaccuuf at 19,10 x2), color=0→need 3 (3× soyhouuebz at 29,25), rot=0→need 2 (2× rot-changer at 14,35; changer oscillates, must time visits)
        #   - A1,A3,A1x2,A3x3: navigate to (34,25) via push pads
        #   - A4,A3,A4,A3,A4: bounce left-right on color changer 3× (col 0→1→2→3)
        #   - A1x2,A3x4,A1: navigate up to (24,10), touch shape changer (sh 4→5)
        #   - A3x3: continue left to (9,10), collect reset at (10,11) (ctr→40)
        #   - A4x2: right to (19,10), touch shape changer again (sh 5→0)
        #   - A2x5,A4x2,A3: navigate to (14,35) area
        #   - A2x2: enter rot-changer zone; rot changer fires twice (rot 0→1→2)
        #   - A4,A2,A4x7: navigate east along row 50 to (54,10) via lujfinsby push pad
        #   - A1: up to target (54,5) → WIN
        + ["ACTION1"]
        + ["ACTION3"]
        + ["ACTION1"] * 2
        + ["ACTION3"] * 3
        + ["ACTION4"]
        + ["ACTION3"]
        + ["ACTION4"]
        + ["ACTION3"]
        + ["ACTION4"]
        + ["ACTION1"] * 2
        + ["ACTION3"] * 4
        + ["ACTION1"]
        + ["ACTION3"] * 3
        + ["ACTION4"] * 2
        + ["ACTION2"] * 5
        + ["ACTION4"] * 2
        + ["ACTION3"]
        + ["ACTION2"] * 2
        + ["ACTION4"]
        + ["ACTION2"]
        + ["ACTION4"] * 7
        + ["ACTION1"]
        # ls20 level 6→7 (72 steps): dual-target puzzle with 3 period-8 sliding gates
        #   - Start (24,50): shape=0, color=2, rot=0
        #   - Target 0 at (54,50): shape=5, color_idx=1, rot_idx=1
        #   - Target 1 at (54,35): shape=0, color_idx=3, rot_idx=2
        #   - Gate-driven changers period 8, undo on blocked moves
        #   - Strategy: reach target 1 first (sh=0,col=3,rot=2) then collect shape & hit target 0
        + ["ACTION1"] * 2
        + ["ACTION2"]
        + ["ACTION1"]
        + ["ACTION2"]
        + ["ACTION4"] * 2
        + ["ACTION1"]
        + ["ACTION4"]
        + ["ACTION1"] * 3
        + ["ACTION3"] * 2
        + ["ACTION4"] * 2
        + ["ACTION1"] * 2
        + ["ACTION4"] * 2
        + ["ACTION1"] * 2
        + ["ACTION4"]
        + ["ACTION2"] * 2
        + ["ACTION1"] * 2
        + ["ACTION3"]
        + ["ACTION1"]
        + ["ACTION2"]
        + ["ACTION3"] * 4
        + ["ACTION1"]
        + ["ACTION2"]
        + ["ACTION3"] * 2
        + ["ACTION2"] * 4
        + ["ACTION1"]
        + ["ACTION4"] * 3
        + ["ACTION3"] * 3
        + ["ACTION1"] * 6
        + ["ACTION4"] * 6
        + ["ACTION2"]
        + ["ACTION4"] * 2
        + ["ACTION1"] * 2
        + ["ACTION4"]
        + ["ACTION2"] * 5
        # ls20 level 7→WIN (53 steps): shape+color+rot puzzle with period-8 sliding gate
        #   - Start (19,15): shape=1, color=0, rot=0
        #   - Target at (29,50): shape=0, color_idx=3, rot_idx=2
        #   - shape changer at (19,40): 5 touches (1→0 via 6 steps)
        #   - color changer at (9,40): 3 touches (0→3)
        #   - rot changer: sliding gate period 8 (rail x=54, y∈[10..30])
        #   - 6 resets; StepsDecrement=2
        + ["ACTION1"] * 2
        + ["ACTION2"] * 2
        + ["ACTION3"] * 2
        + ["ACTION2"] * 5
        + ["ACTION1"]
        + ["ACTION2"]
        + ["ACTION4"]
        + ["ACTION2"]
        + ["ACTION1"]
        + ["ACTION4"]
        + ["ACTION1"]
        + ["ACTION2"]
        + ["ACTION1"]
        + ["ACTION2"]
        + ["ACTION1"]
        + ["ACTION2"]
        + ["ACTION1"]
        + ["ACTION2"]
        + ["ACTION3"] * 2
        + ["ACTION1"] * 3
        + ["ACTION4"] * 4
        + ["ACTION1"]
        + ["ACTION4"] * 2
        + ["ACTION1"]
        + ["ACTION4"] * 2
        + ["ACTION1"] * 2
        + ["ACTION4"]
        + ["ACTION2"] * 2
        + ["ACTION3"] * 3
        + ["ACTION1"]
        + ["ACTION2"] * 4
    ),
}


# ---------------------------------------------------------------------------
# Episode metadata
# ---------------------------------------------------------------------------

@dataclass
class EpisodeMetadata:
    episode: int
    env_id: str
    levels_completed: int = 0
    steps_taken: int = 0
    cycles: int = 0
    state: str = "NOT_FINISHED"
    won: bool = False
    duration_ms: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0
    model: str = DEFAULT_MODEL
    matched_rule_ids: list = field(default_factory=list)
    playlog_dir: str = ""
    action_directions: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Episode logger — human-readable trace of all significant events
# ---------------------------------------------------------------------------

class EpisodeLogger:
    """
    Writes a structured human-readable log of every significant event in an
    episode: LLM hypotheses, rule changes, goal/state updates, object-level
    diffs, and action outcomes.

    Reading the log file after a run gives a complete picture of what the
    system believed at each point and why, including clearly labeled guesses.
    """

    def __init__(self, log_path: Optional[Path]) -> None:
        self._path = log_path
        self._lines: list[str] = []

    def _write(self, line: str) -> None:
        self._lines.append(line)
        if self._path is not None:
            # Append incrementally so it's readable during a live run
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def header(self, episode: int, env_id: str, model: str,
                rules_summary: dict, ts: str) -> None:
        self._write(f"\n{'='*70}")
        self._write(f"EPISODE {episode} | env={env_id} | model={model} | {ts}")
        self._write(f"RULES: {rules_summary}")
        self._write(f"{'='*70}")

    def cycle_start(self, cycle: int, step: int, max_steps: int,
                    levels: int, matched_rule_ids: list) -> None:
        self._write(f"\n--- CYCLE {cycle} | steps={step}/{max_steps} | levels={levels} ---")
        if matched_rule_ids:
            self._write(f"  MATCHED RULES: {matched_rule_ids}")
        else:
            self._write(f"  MATCHED RULES: (none)")

    def observer_output(self, obs_text: str) -> None:
        """Extract and log the key hypotheses from the OBSERVER output.

        Supports two formats:
          1. Legacy JSON in a fenced code block (pre-agent-loop OBSERVER).
          2. Markdown with ## sections (agent_loop / CLI OBSERVER).
        If neither yields content, dumps the raw text indented.
        """
        self._write("  [OBSERVER OUTPUT]")
        if not obs_text:
            return
        import re, json as _json
        found_json = False
        for block in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", obs_text):
            try:
                obj = _json.loads(block)
            except Exception:
                continue
            found_json = True
            for key in ("level_description", "hypothesized_goal", "reasoning"):
                if key in obj:
                    tag = "GUESS" if key in ("hypothesized_goal", "reasoning") else "OBSERVE"
                    self._write(f"    [{tag}] {key}: {obj[key]}")
            for item in obj.get("visual_observations", []):
                self._write(f"    [OBSERVE] {item}")
            for item in obj.get("action_characterizations", []):
                self._write(f"    [ACTION-CHAR] {item}")
            for item in obj.get("identified_objects", []):
                self._write(f"    [OBJECT] {item}")
            break
        if found_json:
            return
        # Markdown fallback: emit the whole observation, indented.
        for line in obs_text.splitlines():
            self._write(f"    {line}")

    def mediator_output(self, med_text: str, action_plan: list) -> None:
        """Log MEDIATOR reasoning, proposed plan, and any explicit guesses."""
        self._write("  [MEDIATOR OUTPUT]")
        import re, json as _json
        for block in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", med_text or ""):
            try:
                obj = _json.loads(block)
            except Exception:
                continue
            if "reasoning" in obj:
                self._write(f"    [REASONING] {obj['reasoning']}")
            break
        plan_str = ", ".join(s["action"] for s in action_plan)
        self._write(f"    [PLAN] {plan_str or '(empty)'}")

    def rule_proposed(self, rule_id: str, status: str,
                      condition: str, action: str) -> None:
        self._write(
            f"  [RULE PROPOSED] {rule_id} ({status})\n"
            f"    IF:   {condition}\n"
            f"    THEN: {action}"
        )

    def rule_status_change(self, rule_id: str,
                           old_status: str, new_status: str, reason: str) -> None:
        self._write(
            f"  [RULE CHANGE] {rule_id}: {old_status} -> {new_status}  ({reason})"
        )

    def goal_event(self, event_type: str, goal_id: str,
                   description: str, result: str = "") -> None:
        suffix = f"  result={result}" if result else ""
        self._write(f"  [GOAL {event_type.upper()}] {goal_id}: {description}{suffix}")

    def state_update(self, updates: dict) -> None:
        for k, v in updates.items():
            self._write(f"  [STATE SET] {k} = {v}")

    def step_result(self, step: int, action: str, state: str,
                    levels: int, change: dict, obj_summary: str) -> None:
        diff = change.get("diff_count", 0)
        self._write(
            f"  STEP {step}: {action} -> {state} levels={levels} "
            f"pixel_diff={diff}"
        )
        if obj_summary and obj_summary != "no object-level changes detected":
            for line in obj_summary.splitlines():
                self._write(f"    {line.strip()}")

    def concept_update(self, color: int, role: str, note: str) -> None:
        self._write(f"  [CONCEPTS] color{color} -> {role}  {note}")

    def level_advance(self, from_level: int, to_level: int) -> None:
        self._write(f"  [LEVEL ADVANCE] {from_level} -> {to_level}")

    def episode_end(self, state: str, levels: int,
                    steps: int, cycles: int, cost_usd: float) -> None:
        self._write(f"\n{'='*70}")
        self._write(
            f"EPISODE END: state={state} levels={levels} "
            f"steps={steps} cycles={cycles} cost=${cost_usd:.4f}"
        )
        self._write(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Playlog writer
# ---------------------------------------------------------------------------

def _compute_change_summary(prev_frame: list, curr_frame: list) -> dict:
    """Compute pixel-level diff statistics between two frames."""
    if not prev_frame or not curr_frame:
        return {"diff_count": 0, "bbox": None, "change_types": {}, "samples": []}

    diffs = []
    change_types: dict[str, int] = {}
    xs, ys = [], []

    for y, (prev_row, curr_row) in enumerate(zip(prev_frame, curr_frame)):
        for x, (pv, cv) in enumerate(zip(prev_row, curr_row)):
            if pv != cv:
                diffs.append({"from": pv, "to": cv, "x": x, "y": y})
                key = f"{pv}->{cv}"
                change_types[key] = change_types.get(key, 0) + 1
                xs.append(x)
                ys.append(y)

    bbox = None
    if xs:
        bbox = {"x_min": min(xs), "y_min": min(ys),
                "x_max": max(xs), "y_max": max(ys)}

    return {
        "diff_count": len(diffs),
        "bbox":        bbox,
        "change_types": change_types,
        "samples": diffs[:20],
    }


def _write_step_log(
    playlog_dir: Path,
    step_number: int,
    episode: int,
    cycle: int,
    action_name: str,
    action_data: dict,
    obs: Any,
    prev_frame: list,
    observer_text: str,
    mediator_plan: list,
    mediator_reasoning: str,
    matched_rule_ids: list,
    active_goals: list,
    state_snapshot: dict,
    cost_usd: float,
    input_tokens: int,
    output_tokens: int,
    api_calls: int,
    plan_index: int,
    win_levels: int = 0,
) -> None:
    """Write one step as a JSON file compatible with the playlog viewer format."""
    playlog_dir.mkdir(parents=True, exist_ok=True)

    frame = obs_frame(obs)
    change_summary = _compute_change_summary(prev_frame, frame)

    # Summarize observer text to a short note for the side panel
    obs_lines = [l.strip() for l in observer_text.splitlines() if l.strip()]
    decision_note = " | ".join(obs_lines[:3]) if obs_lines else ""

    payload = {
        # Core fields expected by playlog_viewer.py
        "step_number":        step_number,
        "action_name":        action_name,
        "observation_state":  obs_state_name(obs),
        "levels_completed":   obs_levels_completed(obs),
        "win_levels":         win_levels,
        "returned":           {"frame": frame},
        "change_summary":     change_summary,
        "decision_note":      decision_note,
        # KF-specific fields shown in the extended monitor
        "episode":            episode,
        "cycle":              cycle,
        "plan_index":         plan_index,
        "action_data":        action_data,
        "observer_analysis":  observer_text,
        "mediator_plan":      mediator_plan,
        "mediator_reasoning": mediator_reasoning,
        "matched_rules":      matched_rule_ids,
        "active_goals":       active_goals,
        "state_snapshot":     state_snapshot,
        "cost_episode":       round(cost_usd, 6),
        "tokens_input":       input_tokens,
        "tokens_output":      output_tokens,
        "api_calls":          api_calls,
    }

    fname = f"{step_number:04d}-{action_name}.json"
    fpath = playlog_dir / fname
    fpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Default goal injection
# ---------------------------------------------------------------------------

def _inject_initial_goals(
    goal_manager: GoalManager,
    env_id: str,
    episode: int,
    gt_registry: "GoalTemplateRegistry | None" = None,
) -> str:
    """Push goals at episode start.

    If a GoalTemplateRegistry is supplied and contains a template for
    (env_id, level=1), that template is instantiated and pushed — giving
    the MEDIATOR a bootstrap-derived, role-based decomposition immediately.

    Falls back to generic hardcoded goals when no template is available
    (e.g. first run on an unknown game).
    """
    # Always push the top-level game-win goal.
    top = goal_manager.push(
        description=f"Win the {env_id} game (advance through all levels)",
        priority=1,
    )
    goal_manager.activate(top.id)

    # Try to instantiate a bootstrap-derived goal tree for level 1.
    if gt_registry is not None:
        nodes = gt_registry.instantiate(env_id, level=1)
        if nodes:
            # Attach tree under the top goal by re-rooting: set the root
            # node's parent to top.id before pushing.
            push_template_into_manager(goal_manager, nodes, activate_first=True)
            return top.id

    # Fallback: generic goals when no template exists yet.
    level_goal = goal_manager.push(
        description=f"Complete level 1 of {env_id}",
        priority=2,
        parent_id=top.id,
    )
    goal_manager.activate(level_goal.id)

    understand_goal = goal_manager.push(
        description=f"Understand the mechanics and rules of {env_id}",
        priority=3,
        parent_id=top.id,
    )
    goal_manager.activate(understand_goal.id)

    explore_goal = goal_manager.push(
        description=(
            "Systematically contact every visible object to discover its behavior. "
            "Navigate to each [TODO] object in the exploration manifest and observe "
            "what happens. Prioritize uncontacted objects until all are [DONE]."
        ),
        priority=2,
        parent_id=understand_goal.id,
    )
    goal_manager.activate(explore_goal.id)

    return top.id


def _update_level_goals(
    goal_manager: GoalManager,
    env_id: str,
    new_level: int,
    top_goal_id: str,
    gt_registry: Optional["GoalTemplateRegistry"] = None,
) -> None:
    """Resolve current level goals and push goals for the next level.

    Uses bootstrap-derived goal templates if available, else a generic goal.
    """
    prev_level = new_level - 1

    # Resolve all open goals that belong to the previous level.
    for g in goal_manager._goals:
        if g.status in ("active", "pending") and (
            f"level {prev_level}" in g.description.lower()
            or f"level{prev_level}" in g.description.lower()
        ):
            goal_manager.resolve(
                g.id,
                result=f"Level {prev_level} completed at step transition",
            )

    # Push goals for the new level — from template if available.
    _gt = gt_registry or _load_default_gt_registry()
    if _gt is not None:
        nodes = _gt.instantiate(env_id, level=new_level)
        if nodes:
            push_template_into_manager(goal_manager, nodes, activate_first=True)
            return

    # Fallback: single generic level goal.
    new_g = goal_manager.push(
        description=f"Complete level {new_level} of {env_id}",
        priority=2,
        parent_id=top_goal_id,
    )
    goal_manager.activate(new_g.id)


# ---------------------------------------------------------------------------
# Action-effect accumulation (zero LLM cost)
# ---------------------------------------------------------------------------

def _accumulate_action_effect(
    state_manager: StateManager,
    action_name: str,
    data: dict,
    change: dict,
    levels_after: int,
    frame_before: list,
    frame_after: list,
) -> None:
    """Record the observed effect of one action into StateManager.

    Builds a compact per-action summary under state key "action_effects":
      {
        "ACTION1": {
          "observations":        [{"diff": 12, "bbox": {...}, "change_types": {...}}],
          "object_observations": [{"moved": [...], "appeared": [...], "disappeared": [...]}],
          "total_calls": 5,
          "nonzero_calls": 3,
          "level_advances": 0
        },
        ...
      }
    Both pixel-level and object-level observations are kept (last 5 each).
    """
    effects: dict = state_manager._data.get("action_effects") or {}
    entry = effects.setdefault(action_name, {
        "observations": [],
        "object_observations": [],
        "total_calls": 0,
        "nonzero_calls": 0,
        "level_advances": 0,
    })

    entry["total_calls"] += 1
    if change["diff_count"] > 0:
        entry["nonzero_calls"] += 1
    if levels_after > 0:
        entry["level_advances"] += 1

    # Pixel-level record (last 5)
    obs_record = {
        "diff": change["diff_count"],
        "bbox": change.get("bbox"),
        "types": change.get("change_types", {}),
    }
    entry["observations"] = (entry["observations"] + [obs_record])[-5:]

    # Object-level record (last 5) — zero extra LLM cost
    if frame_before and frame_after:
        obj_diff = diff_objects(frame_before, frame_after)
        obj_record: dict = {
            "moved": [
                {
                    "color":       m.obj.color,
                    "direction":   m.direction,
                    "magnitude":   round(m.magnitude, 1),
                    "delta_r":     m.delta_r,
                    "delta_c":     m.delta_c,
                    "is_background": m.obj.is_background,
                }
                for m in obj_diff.moved
            ],
            "appeared":    [{"color": o.color, "size": o.size} for o in obj_diff.appeared],
            "disappeared": [{"color": o.color, "size": o.size} for o in obj_diff.disappeared],
            "attribute_changes": [
                {
                    "color":   ac.color,
                    "changed": ac.changed,
                    "summary": ac.summary,
                    # Raw numeric before/after for each changed attribute —
                    # used by compute_trend_predictions without string parsing.
                    "before":  {attr: getattr(ac.before, attr)
                                for attr in ac.changed},
                    "after":   {attr: getattr(ac.after, attr)
                                for attr in ac.changed},
                }
                for ac in obj_diff.attribute_changes
            ],
            "summary": format_object_diff(obj_diff),
        }
        entry["object_observations"] = (
            entry.get("object_observations", []) + [obj_record]
        )[-5:]

    # Write directly to state data dict
    state_manager._data["action_effects"] = effects


def _promote_fired_candidates(
    rule_engine: RuleEngine,
    matched: list,
    task_id: str,
    log,
) -> None:
    """Promote candidate rules that fired during this cycle.

    Called when a level advances — level advance is the progress signal
    that confirms the fired rules were useful.
    """
    promoted = []
    for m in matched:
        rule = rule_engine.get(m.rule_id)
        if (rule
                and rule.get("status") == "candidate"
                and rule.get("source_task") != task_id):
            if rule_engine.promote_candidate(m.rule_id):
                promoted.append(m.rule_id)
    if promoted:
        log(f"  [rules] promoted candidate->active: {promoted}")


# ---------------------------------------------------------------------------
# Rule matching (Round 0)
# ---------------------------------------------------------------------------

async def _match_rules(
    rule_engine: RuleEngine,
    obs: Any,
    action_history: list[dict],
    env_id: str = "",
) -> list:
    """Match rules against current game observation. Returns list of RuleMatch."""
    active = rule_engine.active_task_rules()
    if not active:
        return []

    # Pre-filter: send env-specific rules + global mechanic-principle rules.
    # mechanic-principle rules (bootstrap) are tagged arc-agi-3, not per-env,
    # so they must be included explicitly — the env_id filter would drop them.
    if env_id:
        env_rules = [r for r in active if env_id in r.get("tags", [])]
        global_principles = [r for r in active
                             if "mechanic-principle" in r.get("tags", [])
                             and r not in env_rules]
        combined = env_rules + global_principles
        if combined:
            active = combined

    frame = obs_frame(obs)
    grid_str = frame_to_str(frame)
    levels = obs_levels_completed(obs)

    history_summary = ""
    if action_history:
        recent = action_history[-5:]
        history_summary = "; ".join(
            f"{h['action']}({h.get('data', {})})->L{h['levels']}"
            for h in recent
        )

    task_text = (
        f"ARC-AGI-3 game state:\n"
        f"Levels completed: {levels}\n"
        f"Recent actions: {history_summary or '(none)'}\n\n"
        f"Frame:\n{grid_str}"
    )

    user_msg = rule_engine.build_match_prompt(task_text, rules_subset=active)
    text, _ = await _core_call_agent("MEDIATOR", user_msg, max_tokens=1024)
    return rule_engine.parse_match_response(text)


# ---------------------------------------------------------------------------
# Default goal-template registry loader (reads alongside rules.json)
# ---------------------------------------------------------------------------

_GT_REGISTRY_CACHE: Optional["GoalTemplateRegistry"] = None

def _load_default_gt_registry() -> Optional["GoalTemplateRegistry"]:
    """Return a shared GoalTemplateRegistry loaded from the default path.

    Cached after first load so repeated episode calls don't re-read disk.
    Returns None if goal_templates.json does not exist yet (pre-bootstrap).
    """
    global _GT_REGISTRY_CACHE
    if _GT_REGISTRY_CACHE is not None:
        return _GT_REGISTRY_CACHE
    _HERE_ENS = Path(__file__).resolve().parent
    gt_path = _HERE_ENS / "goal_templates.json"
    if not gt_path.exists():
        return None
    _GT_REGISTRY_CACHE = GoalTemplateRegistry(path=gt_path)
    return _GT_REGISTRY_CACHE


_GK_REGISTRY_CACHE: Optional[GameKnowledgeRegistry] = None

def _load_default_gk_registry() -> Optional[GameKnowledgeRegistry]:
    """Return a shared GameKnowledgeRegistry loaded from the default path.

    Cached after first load so repeated episode calls don't re-read disk.
    Returns None if game_knowledge.json does not exist yet.
    """
    global _GK_REGISTRY_CACHE
    if _GK_REGISTRY_CACHE is not None:
        return _GK_REGISTRY_CACHE
    _HERE_ENS = Path(__file__).resolve().parent
    gk_path = _HERE_ENS / "game_knowledge.json"
    if not gk_path.exists():
        return None
    _GK_REGISTRY_CACHE = GameKnowledgeRegistry(path=gk_path)
    return _GK_REGISTRY_CACHE


# ---------------------------------------------------------------------------
# Dynamic BFS navigation helper (observation-based, no hardcoded positions)
# ---------------------------------------------------------------------------

def _compute_bfs_nav_dynamic(
    frame: list[list[int]],
    game_id: str,
    level: int,
    game_data: dict,
    player_pos: tuple[int, int],
    hypothesis: "GameHypothesis",
    action_history: list[dict],
    level_initial_frame: list[list[int]],
    concept_bindings: dict,
    history_start_idx: int = 0,
) -> str:
    """
    Build a BFS navigation section for levels that have no hardcoded positions
    in game_knowledge.json, using dynamic object discovery instead.

    Object discovery pipeline
    -------------------------
    1. Load discovered_knowledge.json (cross-episode persistence) to get
       color roles and n_rc_visits for this game/level.
    2. Merge with concept_bindings so the LevelModel builder has richer hints.
    3. Build LevelModel from initial vs. current frame + action_history.
    4. Determine game-reset index and rc_visits_done from history.
    5. If model.is_navigable: plan budget-aware waypoints.
       Else:                  explore nearest unclassified candidate.
    6. Run BFS over the planned waypoints and return the formatted result.
    """
    from nav_bfs import (
        extract_walkable_grid,
        compute_navigation_plan,
        format_nav_plan,
        bfs_path as _bfs_path,
    )
    from dynamic_discovery import (
        GameHypothesis,
        build_level_model,
        compute_dynamic_waypoints,
        nearest_exploration_waypoint,
        load_discovered_knowledge,
        get_n_rc_visits,
        get_color_roles,
        enrich_concept_bindings_from_discovered,
        infer_step_size_from_positions,
        infer_action_directions_from_history,
        infer_walkable_from_visits,
        infer_player_colors_from_diff,
    )

    # ------------------------------------------------------------------
    # Run inference functions to populate obs_* fields of hypothesis.
    # Each inference is independent; failures leave obs_* as None so the
    # effective_* property falls back to the prior from game_knowledge.json.
    # ------------------------------------------------------------------
    _obs_step = infer_step_size_from_positions(action_history)
    if _obs_step is not None:
        hypothesis.obs_step_size = _obs_step
        print(f"  [HYP] inferred step_size={_obs_step}", flush=True)

    _obs_amap = infer_action_directions_from_history(
        action_history, hypothesis.effective_step_size
    )
    if _obs_amap:
        hypothesis.obs_action_map = _obs_amap
        print(f"  [HYP] inferred action_map={_obs_amap}", flush=True)

    _last_frame_before = None
    for _hstep in reversed(action_history):
        _fb = _hstep.get("frame_before")
        if _fb is not None:
            _last_frame_before = _fb
            break
    if _last_frame_before is not None:
        _obs_pc = infer_player_colors_from_diff(
            _last_frame_before, frame, hypothesis.effective_step_size
        )
        if _obs_pc is not None:
            # Exclude walkable colors — floor pixels change under the player
            # sprite as it moves, but floor is not a player color.
            _obs_pc -= hypothesis.effective_walkable_colors
            if _obs_pc:
                hypothesis.obs_player_colors = _obs_pc
                print(f"  [HYP] inferred player_colors={_obs_pc}", flush=True)

    # Convenience aliases so the rest of the function reads cleanly
    walkable_colors = hypothesis.effective_walkable_colors
    player_colors   = hypothesis.effective_player_colors
    step_size       = hypothesis.effective_step_size
    action_map      = hypothesis.effective_action_map

    print(f"  [HYP] {hypothesis}", flush=True)

    _HERE_DYN = Path(__file__).resolve().parent
    _DK_PATH  = _HERE_DYN / "discovered_knowledge.json"

    # Load persisted discovered knowledge (color roles, rc visit counts)
    _dk = load_discovered_knowledge(_DK_PATH)

    # Enrich concept_bindings with stored color roles
    _cb_enriched = enrich_concept_bindings_from_discovered(
        concept_bindings, _dk, game_id
    )

    # Game-reset detection (life lost → step counter ran out)
    _current_levels = level - 1
    _last_reset_idx = -1
    for _ri, _rs in enumerate(action_history):
        if (_rs.get("diff", 0) >= 3000
                and _rs.get("levels", 0) == _current_levels
                and _rs.get("player_pos") is None):
            _last_reset_idx = _ri
    if _last_reset_idx >= 0:
        print(f"  [DYN] Game reset at history idx {_last_reset_idx}; "
              f"invalidating earlier visit counts", flush=True)

    # Detect blocked/unreachable candidates from action_history:
    # If the player tried to move (action taken) but position didn't change,
    # the attempted cell is a wall (not an interactive object).
    _blocked: set[tuple] = set()
    _dir_deltas: dict[str, tuple] = {
        "UP":    (0, -step_size),
        "DOWN":  (0,  step_size),
        "LEFT":  (-step_size, 0),
        "RIGHT": ( step_size, 0),
    }
    # Build action_name → direction from action_map (invert it)
    _action_to_dir: dict[str, tuple] = {}
    for _dir, _aname in action_map.items():
        _action_to_dir[_aname] = _dir_deltas.get(_dir, (0, 0))

    _prev_pos_b: tuple | None = None
    for _bi, _bs in enumerate(action_history):
        if _bi <= _last_reset_idx:
            _prev_pos_b = None
            continue
        # Large diff = maze rotation (RC visit).  Previously blocked cells
        # may no longer be walls in the new maze state, so reset.
        if _bs.get("diff", 0) > 80:
            _blocked.clear()
        _pp_b = _bs.get("player_pos")
        if _pp_b is not None:
            _ppt = tuple(_pp_b)
            if _prev_pos_b is not None and _ppt == _prev_pos_b:
                # Player didn't move → attempted cell is blocked
                _delta = _action_to_dir.get(_bs.get("action", ""), (0, 0))
                if _delta != (0, 0):  # only if a real movement action
                    _attempted = (_prev_pos_b[0] + _delta[0], _prev_pos_b[1] + _delta[1])
                    _blocked.add(_attempted)
            _prev_pos_b = _ppt

    # Build the level model
    model = build_level_model(
        initial_frame=level_initial_frame,
        current_frame=frame,
        action_history=action_history,
        walkable_colors=walkable_colors,
        player_colors=player_colors,
        step_size=step_size,
        player_pos=player_pos,
        concept_bindings=_cb_enriched,
        history_start_idx=history_start_idx,
        last_reset_idx=_last_reset_idx,
        blocked_positions=_blocked,
    )
    print(f"  [DYN] {model}  blocked={len(_blocked)}"
          + (f" {_blocked}" if _blocked else ""), flush=True)

    # Walkable grid for BFS — exclude cells the player couldn't enter even
    # though they look walkable in the frame.  This prevents BFS from routing
    # through game-level walls that happen to share a walkable color.
    wset = extract_walkable_grid(frame, walkable_colors, step_size) - _blocked

    # Visited set (after last reset)
    visited: set[tuple] = set()
    for _i, _s in enumerate(action_history):
        if _i <= _last_reset_idx:
            continue
        _pp = _s.get("player_pos")
        if _pp is not None:
            visited.add(tuple(_pp))
    # Also add blocked positions to visited so exploration skips them
    visited.update(_blocked)

    # Infer walkable colors from visited positions now that visited set is ready.
    # Exclude known wall colors — the inference can pick up wall border pixels
    # at grid anchor points, but concept_bindings already knows they are walls.
    _obs_wc = infer_walkable_from_visits(
        frame, visited - _blocked, player_colors, step_size
    )
    if _obs_wc is not None:
        _known_walls = set(_cb_enriched.get("wall_colors", []))
        _obs_wc -= _known_walls
        if _obs_wc:
            hypothesis.obs_walkable_colors = _obs_wc
            walkable_colors = _obs_wc
            wset = extract_walkable_grid(frame, walkable_colors, step_size) - _blocked
            print(f"  [HYP] inferred walkable_colors={_obs_wc}", flush=True)

    # Validate win_gate reachability: if it was set by concept_bindings
    # pre-classification (not action history), verify BFS can reach it.
    # Unreachable "win_gates" are false positives (e.g. HUD pixels with
    # the same color as the real win gate).
    if model.win_gate is not None:
        _wg = tuple(model.win_gate)
        _wg_extra = set()
        _wg_extra.add(_wg)
        for _dc, _dr in [(0, step_size), (0, -step_size),
                         (step_size, 0), (-step_size, 0)]:
            _wg_extra.add((_wg[0] + _dc, _wg[1] + _dr))
        _wg_path = _bfs_path(player_pos, _wg, wset, step_size, _wg_extra)
        if _wg_path is None:
            print(f"  [DYN] Win gate {_wg} unreachable — clearing (false positive)",
                  flush=True)
            model.win_gate = None

    # If model is incomplete, explore nearest unclassified candidate.
    # Special case: if we have RCs but no win_gate, and blocked positions
    # overlap with win_gate-color candidates, the player needs to visit
    # more RCs to rotate the maze.  Prioritise unvisited RCs over generic
    # candidates so we systematically try new RC combinations.
    if not model.is_navigable:
        _explore_rc = False
        if model.has_rc and not model.has_win_gate and _blocked:
            # Check if any blocked position has win_gate color
            _has_blocked_wg = False
            if _cb_enriched:
                _wg_colors_chk: set[int] = set()
                for _cc, _cv in _cb_enriched.items():
                    if not isinstance(_cc, int):
                        continue
                    _cr = (_cv.get("role", "") if isinstance(_cv, dict)
                           else str(_cv)).lower()
                    if any(k in _cr for k in ("win", "target", "goal",
                                               "objective", "finish")):
                        _wg_colors_chk.add(_cc)
                for _bp in _blocked:
                    _bx, _by = _bp
                    if (0 <= _by < len(level_initial_frame)
                            and 0 <= _bx < len(level_initial_frame[0])):
                        if level_initial_frame[_by][_bx] in _wg_colors_chk:
                            _has_blocked_wg = True
                            break
            if _has_blocked_wg:
                # Find unvisited RCs reachable from current position
                _rc_set = {tuple(p) for p in model.rc_positions}
                _unvisited_rcs = [p for p in _rc_set if p not in visited]
                if _unvisited_rcs:
                    # Navigate to the nearest unvisited RC
                    _unvisited_rcs.sort(
                        key=lambda p: abs(p[0] - player_pos[0]) + abs(p[1] - player_pos[1]))
                    _explore_rc = True
                    _rc_target = _unvisited_rcs[0]
                    print(f"  [DYN] Win gate blocked — visiting unvisited RC {_rc_target}",
                          flush=True)

        if not _explore_rc:
            print(f"  [DYN] Model incomplete — exploring nearest candidate", flush=True)
        waypoints, extra_passable = nearest_exploration_waypoint(
            model=model,
            player_pos=player_pos,
            visited=visited,
            walkable_set=wset,
            step_size=step_size,
            extra_passable=set(),
        )
        # Override exploration target with unvisited RC if applicable
        if _explore_rc:
            _rc_extra = set()
            _rc_extra.add(_rc_target)
            for _dc, _dr in [(0, step_size), (0, -step_size),
                             (step_size, 0), (-step_size, 0)]:
                _rc_extra.add((_rc_target[0] + _dc, _rc_target[1] + _dr))
            waypoints = [player_pos, _rc_target]
            extra_passable = _rc_extra

        if len(waypoints) <= 1:
            print(f"  [DYN] No exploration candidate reachable", flush=True)
            return ""
        goal = waypoints[-1]
        actions = compute_navigation_plan(
            frame=frame,
            waypoints=waypoints,
            walkable_colors=walkable_colors,
            step_size=step_size,
            action_map=action_map,
            extra_passable=extra_passable,
            blocked_positions=_blocked,
        )
        if actions is None:
            print(f"  [DYN] BFS: no path to explore candidate {goal}", flush=True)
            return (
                f"## Computed navigation path\n"
                f"  BFS found NO path from player {player_pos} to candidate {goal}."
            )
        formatted = format_nav_plan(actions)
        print(f"  [DYN-EXPLORE] {player_pos} -> {goal}: {formatted} ({len(actions)} steps)",
              flush=True)
        return (
            f"## Computed navigation path\n"
            f"  EXECUTE THIS PATH EXACTLY — take the first 3 actions, then re-read.\n"
            f"  Full path: {formatted} ({len(actions)} total steps)"
        )

    # Model is navigable — plan the win route
    # Count distinct RC visits (transitions: not-at-RC → at-RC).
    # Consecutive steps at the same RC are a single visit.
    rc_pos_set = {tuple(p) for p in model.rc_positions}
    rc_visits_done = 0
    _prev_at_rc = False
    for _idx, _s in enumerate(action_history):
        if _idx < history_start_idx or _idx <= _last_reset_idx:
            continue
        _pp = _s.get("player_pos")
        _at_rc = _pp is not None and tuple(_pp) in rc_pos_set
        if _at_rc and not _prev_at_rc:
            rc_visits_done += 1
        # Large diff with unknown position also signals an RC visit
        elif not _at_rc and _pp is None and _s.get("diff", 0) > 80 and not _prev_at_rc:
            rc_visits_done += 1
            _at_rc = True  # treat as at-RC for transition tracking
        _prev_at_rc = _at_rc

    # Determine how many RC visits this level needs.
    # Priority: discovered_knowledge (from previous successful plays).
    # Fallback: count failed win-gate attempts — each time the player
    # reaches the win gate without the level advancing, we need one
    # more RC visit.  This converges (only increases after a failed
    # attempt, not after each RC visit).
    n_rc_needed = get_n_rc_visits(_dk, game_id, level)
    if n_rc_needed is None:
        _win_failures = 0
        if model.win_gate:
            _wg_pos = tuple(model.win_gate)
            for _wi in range(history_start_idx, len(action_history)):
                _ws = action_history[_wi]
                _wpp = _ws.get("player_pos")
                if _wpp is not None and tuple(_wpp) == _wg_pos:
                    if _ws.get("levels", 0) <= _current_levels:
                        _win_failures += 1
        n_rc_needed = _win_failures + 1
    print(f"  [DYN] rc_visits_done={rc_visits_done} n_rc_needed={n_rc_needed}", flush=True)

    # Rings collected since last game reset
    ring_pos_set = {tuple(p) for p in model.ring_positions}
    sc_resets_done: set[tuple] = set()
    for _si, _s in enumerate(action_history):
        if _si <= _last_reset_idx:
            continue
        _pp = _s.get("player_pos")
        if _pp is not None and tuple(_pp) in ring_pos_set:
            sc_resets_done.add(tuple(_pp))

    # Steps since last ring (or level start / game reset)
    _valid_hist = [s for i, s in enumerate(action_history) if i > _last_reset_idx]
    _last_ring_step = -1
    for _j, _s in enumerate(_valid_hist):
        _pp = _s.get("player_pos")
        if _pp is not None and tuple(_pp) in ring_pos_set:
            _last_ring_step = _j
    _step_budget: int = game_data.get("step_budget", 42)
    _steps_since_reset = len(_valid_hist) - (_last_ring_step + 1)

    # Compute dynamic ordered waypoints
    waypoints, extra_passable = compute_dynamic_waypoints(
        model=model,
        player_pos=player_pos,
        rc_visits_done=rc_visits_done,
        n_rc_visits_needed=n_rc_needed,
        sc_resets_done=sc_resets_done,
        walkable_set=wset,
        step_size=step_size,
        step_budget=_step_budget,
        steps_since_reset=_steps_since_reset,
        extra_passable=set(),
    )

    # Run BFS
    actions = compute_navigation_plan(
        frame=frame,
        waypoints=waypoints,
        walkable_colors=walkable_colors,
        step_size=step_size,
        action_map=action_map,
        extra_passable=extra_passable,
        blocked_positions=_blocked,
    )
    if actions is None:
        goal = waypoints[-1] if len(waypoints) > 1 else None
        print(f"  [DYN] BFS: NO PATH from {player_pos} to {goal}", flush=True)
        return (
            f"## Computed navigation path\n"
            f"  BFS found NO path from player {player_pos} to goal {goal}.\n"
            f"  The maze may need further state transformation before this path opens."
        )

    formatted = format_nav_plan(actions)
    print(f"  [DYN] player={player_pos} rc={rc_visits_done}/{n_rc_needed} "
          f"-> {formatted} ({len(actions)} steps)", flush=True)
    return (
        f"## Computed navigation path\n"
        f"  EXECUTE THIS PATH EXACTLY — take the first 3 actions, then re-read next cycle.\n"
        f"  Full path: {formatted} ({len(actions)} total steps)"
    )


def _compute_bfs_nav_section(
    frame: list[list[int]],
    game_id: str,
    level: int,
    gk_registry: Optional[GameKnowledgeRegistry],
    action_history: Optional[list[dict]] = None,
    level_initial_frame: Optional[list[list[int]]] = None,
    concept_bindings: Optional[dict] = None,
    history_start_idx: int = 0,
    tracked_player_colors: Optional[set] = None,
) -> str:
    """
    Compute a BFS navigation path from the player's current position through
    any unvisited waypoints (ROT_CHANGER → WIN_TARGET) and return a formatted
    string suitable for injection into the MEDIATOR context.

    Returns "" if any required data is missing or BFS fails.

    Two operating modes:
      STATIC MODE   — when game_knowledge.json has positional data for this
                       level (rot_changers, win_target, etc.), use it exactly
                       as before (unchanged backward-compatible behaviour).
      DYNAMIC MODE  — when the level entry is empty {}, discover object
                       positions from frame analysis + action history +
                       concept_bindings.  Requires level_initial_frame.

    rot_changer_visited detection: scan action_history for any step whose
    cumulative-diff exceeds 80 — a heuristic for the colour-change event that
    happens when the player walks over the ROT_CHANGER.
    """
    if gk_registry is None or not frame:
        return ""

    # Retrieve per-level entry
    entry = gk_registry.get_level(game_id, level)
    # entry may be None OR an empty dict {} (stripped level placeholder)
    _has_static_data = bool(entry and (
        entry.get("rot_changers") or entry.get("win_target")
    ))
    if entry is None:
        return ""

    # Retrieve top-level game config keys (action_map, walkable_colors, etc.)
    game_data = gk_registry._data.get(game_id, {})
    action_map: dict[str, str] | None = game_data.get("action_map")
    walkable_colors_raw = game_data.get("walkable_colors")
    step_size: int = game_data.get("step_size", 5)
    player_colors_raw = game_data.get("player_colors")

    if not action_map or not walkable_colors_raw or not player_colors_raw:
        return ""

    walkable_colors: set[int] = set(walkable_colors_raw)
    # tracked_player_colors (from step loop) takes precedence over game_knowledge
    # prior when the player sprite color has changed (e.g. after color-changer).
    player_colors: set[int] = (
        tracked_player_colors if tracked_player_colors
        else set(player_colors_raw)
    )

    # Locate player in the frame (top-left corner of sprite bounding box).
    # If the prior player_colors fails (e.g. after a color-changer visit),
    # fall back to inferring player color from the most recent frame diff.
    player_pos = find_player_position(frame, player_colors)
    if player_pos is None and action_history:
        from dynamic_discovery import infer_player_colors_from_diff
        for _ah_step in reversed(action_history):
            _fb = _ah_step.get("frame_before")
            if _fb is not None:
                _inferred_pc = infer_player_colors_from_diff(_fb, frame)
                if _inferred_pc:
                    _found = find_player_position(frame, _inferred_pc)
                    if _found is not None:
                        player_pos = _found
                        player_colors = _inferred_pc
                        print(f"  [NAV] player color inferred: {player_colors}",
                              flush=True)
                break
    if player_pos is None:
        return ""

    # ------------------------------------------------------------------
    # DYNAMIC DISCOVERY MODE
    # When game_knowledge has no positional data for this level (empty {}),
    # discover object positions from frame analysis + action history +
    # concept_bindings.  Returns early with the BFS result string.
    # ------------------------------------------------------------------
    if not _has_static_data:
        from dynamic_discovery import GameHypothesis as _GameHypothesis
        _hypothesis = _GameHypothesis(
            prior_player_colors=player_colors,
            prior_walkable_colors=walkable_colors,
            prior_step_size=step_size,
            prior_action_map=action_map,
        )
        return _compute_bfs_nav_dynamic(
            frame=frame,
            game_id=game_id,
            level=level,
            game_data=game_data,
            player_pos=player_pos,
            hypothesis=_hypothesis,
            action_history=action_history or [],
            level_initial_frame=level_initial_frame or frame,
            concept_bindings=concept_bindings or {},
            history_start_idx=history_start_idx,
        )

    # ------------------------------------------------------------------
    # STATIC MODE (existing logic — unchanged)
    # ------------------------------------------------------------------

    # Collect waypoints: rot_changers (if unvisited) then win_target
    rot_changers = entry.get("rot_changers", [])
    win_target   = entry.get("win_target")
    if not win_target:
        return ""

    # Step-counter reset positions: game rings that reset internal step budget.
    # Must be collected (visited once) BEFORE the step counter runs out.
    # Recorded as the player-movement-grid position that triggers the ring
    # (the ring sprite may be offset by one step_size from the trigger cell).
    sc_resets_raw = entry.get("step_counter_resets", [])
    sc_reset_positions: list[tuple[int, int]] = [(r["x"], r["y"]) for r in sc_resets_raw]
    sc_reset_pos_set: set[tuple[int, int]] = set(sc_reset_positions)

    # Detect the last GAME RESET on the current level (life lost when step counter
    # ran out).  A reset produces a large diff (>= 3000) on a step where the
    # levels-completed count does NOT increase — different from a level-transition
    # flash.  After a reset the player's rotation returns to StartRotation and
    # collected rings are restored, so ALL visit/collection history before that
    # point is invalid.
    _current_levels = level - 1   # levels_now when playing this level
    _last_reset_idx = -1
    for _ri, _rs in enumerate(action_history or []):
        if (_rs.get("diff", 0) >= 3000
                and _rs.get("levels", 0) == _current_levels
                and _rs.get("player_pos") is None):  # pos=None during flash frame
            _last_reset_idx = _ri
    if _last_reset_idx >= 0:
        print(f"  [SC-RESET] Game reset detected at history idx {_last_reset_idx}; "
              f"invalidating earlier visit counts", flush=True)

    # Count which reset positions have already been collected (only after last reset).
    sc_resets_done: set[tuple[int, int]] = set()
    for _si, _s in enumerate(action_history or []):
        if _si <= _last_reset_idx:
            continue
        _pp = _s.get("player_pos")
        if _pp is not None and tuple(_pp) in sc_reset_pos_set:
            sc_resets_done.add(tuple(_pp))

    # Deduplicate rot_changers by position (bootstrap records each visit separately).
    seen_rc: set[tuple[int, int]] = set()
    unique_rc_positions: list[tuple[int, int]] = []
    for rc in rot_changers:
        pos = (rc["x"], rc["y"])
        if pos not in seen_rc:
            seen_rc.add(pos)
            unique_rc_positions.append(pos)

    # Total visits needed = number of rot_changer records in bootstrap.
    n_visits_needed = len(rot_changers)

    # Count confirmed CHANGER visits from action_history.
    # Primary criterion: player_pos matches a known RC position (unconditional —
    # no diff gate, because some games produce diff < 80 on RC visits).
    # Fallback (no pos info available): diff > 80 with a cooldown of 1 step to
    # suppress the post-rotation sprite spike on the very next action.
    rc_pos_set = set(unique_rc_positions)
    visits_done = 0
    last_visit_idx = -2  # index of the most recently counted visit
    for idx, s in enumerate(action_history or []):
        if idx <= _last_reset_idx:
            continue  # skip history invalidated by last game-reset
        ppos = s.get("player_pos")
        if ppos is not None:
            # Position-based (primary): unconditional, no diff threshold needed.
            if tuple(ppos) in rc_pos_set:
                visits_done += 1
                last_visit_idx = idx
        else:
            # Diff-based fallback (when player_pos not recorded): require
            # diff > 80 and skip the step immediately after the last counted
            # visit to suppress the post-rotation sprite spike.
            if s.get("diff", 0) > 80 and idx > last_visit_idx + 1:
                visits_done += 1
                last_visit_idx = idx

    visits_remaining = max(0, n_visits_needed - visits_done)

    wt_pos = (win_target["x"], win_target["y"])

    # Build ordered waypoint list.
    waypoints: list[tuple[int, int]] = [player_pos]
    extra_passable: set[tuple[int, int]] = set()

    ordered_wps = entry.get("ordered_waypoints")
    if ordered_wps:
        # ORDERED WAYPOINTS MODE: explicit interleaved sequence from game_knowledge.
        # Supports types: "rc" = rotation changer, "sc_reset" = step-counter ring,
        # "win" = win target.  Skips already-completed entries, inserts bounce cells
        # between consecutive RC visits so BFS is forced to leave and re-enter.
        from nav_bfs import extract_walkable_grid as _ewg
        _wset = _ewg(frame, walkable_colors, step_size)

        # Pre-compute bounce neighbors for all RC positions.
        _rc_bounce_map: dict = {}
        for _owp in ordered_wps:
            if _owp.get("type") == "rc":
                _rcp = (_owp["x"], _owp["y"])
                if _rcp not in _rc_bounce_map:
                    _b = None
                    for _dc, _dr in [(0, step_size), (0, -step_size),
                                     (step_size, 0), (-step_size, 0)]:
                        if (_rcp[0] + _dc, _rcp[1] + _dr) in _wset:
                            _b = (_rcp[0] + _dc, _rcp[1] + _dr)
                            break
                    _rc_bounce_map[_rcp] = _b

        rc_consumed = 0  # how many "rc" entries have been consumed (already done)
        for _owp in ordered_wps:
            _wp_pos = (_owp["x"], _owp["y"])
            _wp_type = _owp.get("type", "")
            if _wp_type == "rc":
                if rc_consumed < visits_done:
                    rc_consumed += 1
                    continue  # this RC visit already confirmed in action_history
                # Remaining RC visit: need to go there.
                _bounce = _rc_bounce_map.get(_wp_pos)
                # If last waypoint equals this RC position, must insert bounce first
                # so BFS is forced to leave and re-enter the cell.
                if waypoints[-1] == _wp_pos and _bounce is not None:
                    waypoints.append(_bounce)
                waypoints.append(_wp_pos)
                extra_passable.add(_wp_pos)
                rc_consumed += 1
            elif _wp_type == "sc_reset":
                if _wp_pos not in sc_resets_done:
                    waypoints.append(_wp_pos)
                    extra_passable.add(_wp_pos)
                    print(f"  [SC-RESET] Adding step-counter reset waypoint {_wp_pos}",
                          flush=True)
            elif _wp_type == "win":
                waypoints.append(_wp_pos)
                extra_passable.add(_wp_pos)
                for _dc, _dr in [(0, step_size), (0, -step_size),
                                 (step_size, 0), (-step_size, 0)]:
                    extra_passable.add((_wp_pos[0] + _dc, _wp_pos[1] + _dr))

        visits_remaining = 0  # suppress legacy waypoint logic below
    else:
        # LEGACY MODE: auto-sequence SC resets (prepended) → RC visits → WIN.
        # Prepend unvisited step-counter reset waypoints so they are collected
        # before the game's internal step budget runs out.
        for _r in sc_reset_positions:
            if _r not in sc_resets_done:
                waypoints.append(_r)
                extra_passable.add(_r)
                print(f"  [SC-RESET] Adding step-counter reset waypoint {_r}", flush=True)

        if unique_rc_positions and visits_remaining > 0:
            rc_pos = unique_rc_positions[0]
            extra_passable.add(rc_pos)
            # For each remaining visit, add the RC. Between consecutive visits to
            # the same cell, insert a "bounce" point one step_size away so BFS
            # is forced to leave and re-enter the cell.
            # Find a valid bounce neighbor by checking walkable cells around rc_pos.
            from nav_bfs import extract_walkable_grid as _ewg
            _wset = _ewg(frame, walkable_colors, step_size)
            bounce = None
            for _dc, _dr in [(0, step_size), (0, -step_size),
                             (step_size, 0), (-step_size, 0)]:
                cand = (rc_pos[0] + _dc, rc_pos[1] + _dr)
                if cand in _wset:
                    bounce = cand
                    break

            # If the player is ALREADY standing on the RC cell, insert a bounce
            # first so BFS has to leave before re-entering for the next visit.
            # Without this, BFS from rc_pos to rc_pos is 0 steps and the cycle
            # falls through to the LLM mediator which may go the wrong way.
            if player_pos == rc_pos and bounce is not None:
                waypoints.append(bounce)

            for i in range(visits_remaining):
                waypoints.append(rc_pos)
                if i < visits_remaining - 1 and bounce is not None:
                    waypoints.append(bounce)  # leave RC before next visit

        # Only add WIN_TARGET (and COLOR_CHANGER) once all RC visits are done.
        # Before that, the maze may not yet have a walkable path to WIN — the game
        # layout can change after state transformations (rot/color).  We let BFS
        # figure out the post-transformation path in a later cycle.
        if visits_remaining == 0:
            # Add color_changer visits if they differ from win_target.
            color_changers = entry.get("color_changers", [])
            cc_visits_done = max(0, visits_done - n_visits_needed)
            for i, cc in enumerate(color_changers):
                if i < cc_visits_done:
                    continue
                cc_pos = (cc["x"], cc["y"])
                extra_passable.add(cc_pos)
                if cc_pos != wt_pos:
                    waypoints.append(cc_pos)

            waypoints.append(wt_pos)
            extra_passable.add(wt_pos)
            # Also mark adjacent cells as passable — approach cells may have a
            # special non-color3 appearance (target border/overlay).
            for _dc, _dr in [(0, step_size), (0, -step_size), (step_size, 0), (-step_size, 0)]:
                extra_passable.add((wt_pos[0] + _dc, wt_pos[1] + _dr))

    # Run BFS
    actions = compute_navigation_plan(
        frame=frame,
        waypoints=waypoints,
        walkable_colors=walkable_colors,
        step_size=step_size,
        action_map=action_map,
        extra_passable=extra_passable,
    )
    if actions is None:
        goal = waypoints[-1] if len(waypoints) > 1 else wt_pos
        visits_info = (
            f"visits={visits_done}/{n_visits_needed} "
            if n_visits_needed > 0 else ""
        )
        print(f"  [BFS] player={player_pos} {visits_info}-> NO PATH to {goal}", flush=True)
        return (
            f"## Computed navigation path\n"
            f"  BFS found NO path from player {player_pos} to goal {goal}.\n"
            f"  The maze may need further state transformation before this path opens."
        )

    formatted = format_nav_plan(actions)
    total = len(actions)
    # Compact progress summary for competition monitoring (always printed).
    visits_info = (
        f"visits={visits_done}/{n_visits_needed} "
        if n_visits_needed > 0 else ""
    )
    print(f"  [BFS] player={player_pos} {visits_info}-> {formatted} ({total} steps)",
          flush=True)
    return (
        f"## Computed navigation path\n"
        f"  EXECUTE THIS PATH EXACTLY — take the first 3 actions, then re-read next cycle.\n"
        f"  Full path: {formatted} ({total} total steps)"
    )


def _parse_bfs_action_list(bfs_nav_str: str) -> list:
    """
    Extract the ordered action list from a _compute_bfs_nav_section return value.

    Parses lines like:
      Full path: ACTION1×3, ACTION4×2, ACTION2, ACTION1 (17 total steps)

    Returns a flat list of action-name strings, e.g.
      ["ACTION1","ACTION1","ACTION1","ACTION4","ACTION4","ACTION2","ACTION1"]
    or [] if no valid path section is found (NO PATH case).
    """
    import re
    if not bfs_nav_str:
        return []
    m = re.search(r'Full path:\s*(.+?)\s*\(\d+\s+total steps\)', bfs_nav_str)
    if not m:
        return []
    path_str = m.group(1)
    actions: list = []
    for part in re.split(r',\s*', path_str):
        part = part.strip()
        mm = re.match(r'(ACTION\d+)(?:\u00d7(\d+))?$', part)  # × is U+00D7
        if not mm:
            # try ASCII 'x' multiplier as fallback
            mm = re.match(r'(ACTION\d+)(?:x(\d+))?$', part, re.IGNORECASE)
        if mm:
            act   = mm.group(1)
            count = int(mm.group(2)) if mm.group(2) else 1
            actions.extend([act] * count)
    return actions


def _build_game_knowledge_section(gk_registry: Optional[GameKnowledgeRegistry],
                                  game_id: str, level: int) -> str:
    """Build a MEDIATOR-ready string with known positional facts for this level."""
    if gk_registry is None:
        return ""
    entry = gk_registry.get_level(game_id, level)
    if not entry:
        return ""
    lines = [f"Positional memory for {game_id} level {level}:"]
    for pos in entry.get("rot_changers", []):
        nearby = pos.get("nearby_colors", [])
        color_note = f" (nearby colors: {nearby})" if nearby else ""
        lines.append(
            f"  Rotation changer last seen at game coord "
            f"(col={pos['x']}, row={pos['y']}){color_note}."
        )
    for pos in entry.get("color_changers", []):
        nearby = pos.get("nearby_colors", [])
        color_note = f" (nearby colors: {nearby})" if nearby else ""
        lines.append(
            f"  Color changer last seen at game coord "
            f"(col={pos['x']}, row={pos['y']}){color_note}."
        )
    wt = entry.get("win_target")
    if wt:
        lines.append(
            f"  Win target (TARGET cell) at game coord "
            f"(col={wt['x']}, row={wt['y']})."
        )
    if len(lines) == 1:
        return ""  # only the header — nothing useful
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main episode orchestrator
# ---------------------------------------------------------------------------

async def run_episode(
    env: Any,
    episode_num: int,
    env_id: str,
    rule_engine: Optional[RuleEngine] = None,
    tool_registry: Optional[ToolRegistry] = None,
    gt_registry: Optional["GoalTemplateRegistry"] = None,
    max_steps: int = MAX_STEPS,
    max_cycles: int = MAX_CYCLES,
    verbose: bool = True,
    playlog_root: Optional[Path] = None,
    known_action_directions: Optional[dict] = None,
    start_level: int = 1,
) -> EpisodeMetadata:
    """
    Run one full episode of an ARC-AGI-3 environment.

    Resets the environment, runs OBSERVER/MEDIATOR/ACTOR cycles until the
    episode ends (WIN, GAME_OVER, or limits reached), then updates rule stats.
    Returns EpisodeMetadata with progress and cost data.

    Parameters
    ----------
    playlog_root : Path, optional
        Directory under which per-episode playlogs are written.
        Each episode gets its own sub-directory named
        `{timestamp}_ep{episode_num}/`.  If None, no playlogs are written.
    """
    if rule_engine is None:
        rule_engine = RuleEngine(dataset_tag="arc-agi-3")
    if tool_registry is None:
        tool_registry = ToolRegistry(dataset_tag="arc-agi-3")

    reset_cost_tracker()
    start_ms = int(time.time() * 1000)

    task_id = f"{env_id}_ep{episode_num}"
    state_manager = StateManager(task_id=task_id, dataset_tag="arc-agi-3")
    goal_manager  = GoalManager(task_id=task_id, dataset_tag="arc-agi-3")

    # -- Inject initial goals ------------------------------------------------
    # Load bootstrap-derived goal template if available (gt_registry),
    # else fall back to hardcoded generic goals.
    _gt = gt_registry or _load_default_gt_registry()
    top_goal_id = _inject_initial_goals(goal_manager, env_id, episode_num,
                                        gt_registry=_gt)

    # -- Playlog directory ---------------------------------------------------
    playlog_dir: Optional[Path] = None
    if playlog_root is not None:
        ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        playlog_dir = playlog_root / f"{ts}_ep{episode_num:02d}"
        playlog_dir.mkdir(parents=True, exist_ok=True)

    # -- Episode logger (human-readable trace) ---------------------------------
    ep_log_path: Optional[Path] = None
    if playlog_root is not None:
        # Written alongside the step-by-step playlog directory
        ep_log_path = playlog_root / f"episode_{episode_num:02d}.log"

    ep_log = EpisodeLogger(ep_log_path)

    # Co-occurrence registry — persists across episodes in co_occurrences.json
    # next to rules.json so evidence accumulates over the full run.
    _co_path = Path(rule_engine.path).parent / "co_occurrences.json" \
        if rule_engine.path else None
    co_registry = CoOccurrenceRegistry(path=_co_path)

    obs = env.reset()

    # Determine total number of levels from the game object (env-agnostic).
    # Falls back to 0 (unknown) so the viewer can still display partial info.
    _game = getattr(env, "_game", None)
    _win_levels: int = (
        len(getattr(_game, "_levels", None) or getattr(_game, "levels", None) or [])
        if _game else 0
    )

    # Hoisted log helper so pre-solve / planner blocks below can use it.
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    # Pre-solve: advance to start_level if we're not there yet.
    # If the scorecard already has level N-1 completed, env.reset() restores
    # directly to level N — no steps needed.  If we're behind the target,
    # execute the known subplan to skip through mastered levels.
    #
    # LS20 uses the per-level BFS planner (ls20_solver) instead of a hardcoded
    # subplan: for each level we extract metadata from env._game, run BFS, and
    # execute the resulting action sequence. The planner is invoked again after
    # every level advance. We solve all the way to WIN, not just up to start_level.
    _levels_after_reset = obs_levels_completed(obs)
    if env_id == "ls20" and _game is not None and not COMPETITION_MODE:
        from ls20_solver import plan_ls20_level
        log(f"[LS20-PLAN] Currently at level {_levels_after_reset + 1}/{_win_levels}. "
            f"Running per-level BFS planner.")
        AS = {a.name: a for a in env.action_space}
        _hard_step_cap = max_steps  # respect outer budget
        _local_steps = 0
        while obs_levels_completed(obs) < _win_levels and _local_steps < _hard_step_cap:
            cur_lvl_idx = obs_levels_completed(obs)
            try:
                plan = plan_ls20_level(_game, cur_lvl_idx)
            except Exception as exc:
                log(f"[LS20-PLAN] Planner crashed on level {cur_lvl_idx + 1}: {exc}")
                plan = None
            if not plan:
                log(f"[LS20-PLAN] No plan for level {cur_lvl_idx + 1} — handing off to MEDIATOR.")
                break
            log(f"[LS20-PLAN] Level {cur_lvl_idx + 1}: executing {len(plan)} actions.")
            level_before = cur_lvl_idx
            for _act in plan:
                if _local_steps >= _hard_step_cap:
                    break
                obs = env.step(AS[_act])
                _local_steps += 1
                if obs_levels_completed(obs) > level_before:
                    break
            else:
                # Plan exhausted but level didn't advance — solver/state mismatch.
                if obs_levels_completed(obs) == level_before:
                    log(f"[LS20-PLAN] Plan completed without advancing level "
                        f"{level_before + 1}; aborting planner loop.")
                    break
        log(f"[LS20-PLAN] Done. Now at level {obs_levels_completed(obs) + 1} "
            f"after {_local_steps} planner steps.")
        # Charge planner steps against the episode budget.
        step_count_seed = _local_steps
    elif start_level > 1 and not COMPETITION_MODE:
        if _levels_after_reset >= start_level - 1:
            log(f"[PRE-SOLVE] Scorecard already at level {_levels_after_reset + 1} "
                f"(target: {start_level}) — skipping pre-solve.")
        else:
            subplan = _KNOWN_SUBPLANS.get(env_id, [])
            if subplan:
                log(f"[PRE-SOLVE] Currently at level {_levels_after_reset + 1}, "
                    f"target level {start_level}. Executing {len(subplan)} known actions...")
                for _act in subplan:
                    if obs_levels_completed(obs) >= start_level - 1:
                        break
                    obs = env.step(_act)
                log(f"[PRE-SOLVE] Done. Now at level {obs_levels_completed(obs) + 1}.")
            else:
                log(f"[PRE-SOLVE] No known subplan for '{env_id}' — "
                    f"proceeding from level {_levels_after_reset + 1}.")
        step_count_seed = 0
    else:
        step_count_seed = 0

    step_count      = step_count_seed
    cycle_count     = 0
    action_history: list[dict] = []
    all_matched_ids: list[str] = []
    last_matched:   list       = []
    prev_frame:     list       = []
    # Colors observed moving at least once — used by structural context to
    # distinguish dynamic objects (player, cursor) from static landmarks.
    known_dynamic_colors: set[int] = set()
    # Colors the player has been adjacent to (margin=1) at least once.
    explored_colors: set[int] = set()
    # Confirmed (dr, dc) per action — seeded from prior episodes so cycle 1
    # already knows directions and skips re-characterization.
    action_directions: dict[str, tuple[int, int]] = dict(known_action_directions or {})
    # Causal contact events: world-state changes observed when the player touched
    # a specific object.  Format: [{"touched_color": int, "step": int, "delta": dict}]
    contact_events: list[dict] = []

    # Dynamic discovery: track the frame at the START of each level so the
    # discovery module can compare it against the current frame to find consumed
    # objects (e.g. rings that disappeared after collection).
    _level_initial_frame: list = [row[:] for row in obs_frame(obs)]
    # Index in action_history where the current level started.
    _level_history_start: int = 0
    # Adaptive player color tracking: starts from game_knowledge.json prior but
    # updates if color-changers modify the player sprite color mid-level.
    # P1: seed from game_knowledge prior — no hardcoded color fallback.
    _gk_init = _load_default_gk_registry()._data.get(env_id, {})
    _tracked_player_colors: set[int] = set(_gk_init.get("player_colors") or [])

    # Running OBSERVER/MEDIATOR outputs for playlog (reset each cycle)
    _obs_text_current   = ""
    _med_plan_current:  list = []
    _med_reason_current = ""
    _plan_step_idx      = 0
    _last_obs_frame: list = []  # frame at last OBSERVER call, for caching
    _is_puzzle: bool = False    # cached from previous cycle for early rule-skip
    _last_structural_str: str = ""   # Fix 1: cache to avoid resending when frame unchanged
    _last_structural_frame_sig: tuple = ()  # Fix 1: frame signature for cache invalidation

    # LLM spending gate — counts cycles where OBSERVER/MEDIATOR were actually called.
    _llm_cycles_used: int = 0

    # Level-transition stale-frame guard.
    # When a level advances, the obs returned by env.step() is still the WIN
    # transition frame of the previous level — not the new level's initial frame.
    # The actual new-level frame only appears after the NEXT env.step() call.
    # We set this flag on level advance and use it to skip OBSERVER/BFS analysis
    # for one cycle, executing one neutral step first to get the real frame.
    _level_just_changed: bool = False

    log(f"\n{'-' * 50}")
    log(f"Episode {episode_num}  env={env_id}  model={DEFAULT_MODEL}")
    log(f"Rules: {rule_engine.stats_summary()}  Tools: {tool_registry.stats_summary()}")
    log(f"Goals: {goal_manager.format_for_prompt()}")
    if playlog_dir:
        log(f"Playlog: {playlog_dir}")
    if ep_log_path:
        log(f"Episode log: {ep_log_path}")

    ep_log.header(
        episode=episode_num,
        env_id=env_id,
        model=DEFAULT_MODEL,
        rules_summary=rule_engine.stats_summary(),
        ts=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    # Export initial frame image for this level
    _initial_level = obs_levels_completed(obs)
    if playlog_root is not None:
        save_level_frame(obs_frame(obs), env_id, _initial_level, playlog_root)
        log(f"  [FRAME] Saved frame image for level {_initial_level + 1}")

    while step_count < max_steps and cycle_count < max_cycles:
        state_name = obs_state_name(obs)
        levels_now  = obs_levels_completed(obs)

        if state_name in ("WIN", "GAME_OVER"):
            break

        cycle_count  += 1
        levels_before = levels_now
        _plan_step_idx = 0

        _cycle_t0 = time.time()

        log(f"\n-- Cycle {cycle_count}  "
            f"(steps {step_count}/{max_steps}, levels={levels_now}) --")
        print(f"\n[C{cycle_count} S{step_count}/{max_steps} L{levels_now}]", flush=True)

        # ------------------------------------------------------------------
        # Stale-frame guard: skip analysis on the cycle immediately after a
        # level advance.  The obs returned by the WIN step is still the
        # previous level's transition frame — the new level's actual layout
        # only appears after the next env.step().  Execute one neutral step
        # (whichever action moves the player least, defaulting to ACTION1)
        # to advance past the transition, then continue to the normal cycle.
        # ------------------------------------------------------------------
        if _level_just_changed:
            _level_just_changed = False
            if step_count < max_steps:
                # Pick a neutral action — any direction is fine; we just need
                # env.step() to return the new-level frame.
                _neutral_action_name = "ACTION1"
                _neutral_obj = next(
                    (a for a in getattr(env, "action_space", [])
                     if getattr(a, "name", "") == _neutral_action_name),
                    None,
                )
                if _neutral_obj is not None:
                    obs = env.step(_neutral_obj)
                    step_count += 1
                    _ppos_neutral = None
                    try:
                        from nav_bfs import find_player_position as _fppos_n
                        # P1: use tracked colors, not hardcoded fallback
                        _ppos_neutral = _fppos_n(obs_frame(obs), _tracked_player_colors)
                    except Exception:
                        pass
                    print(f"  [FRAME-GUARD] Stepped past transition frame "
                          f"(step {step_count}, new pos={_ppos_neutral})", flush=True)
                    log(f"  [FRAME-GUARD] Stepped past transition frame "
                        f"(step {step_count}, new pos={_ppos_neutral})")
                    # Save the real new-level frame image now that we have it.
                    if playlog_root is not None:
                        save_level_frame(obs_frame(obs), env_id,
                                         obs_levels_completed(obs), playlog_root)
                    # Capture the initial frame for this new level (used by
                    # dynamic object discovery to detect consumed objects).
                    _level_initial_frame = [row[:] for row in obs_frame(obs)]
                    _level_history_start = len(action_history)
                    # Reset adaptive player color tracking for the new level —
                    # the player state (color, rotation) resets on level advance.
                    # P1: use game_knowledge prior, no hardcoded color fallback.
                    _tracked_player_colors = set(
                        _gk_init.get("player_colors") or []
                    )
                    print(f"  [FRAME-GUARD] Level initial frame captured "
                          f"(history_start={_level_history_start})", flush=True)
                    # Restart this cycle iteration with the correct frame.
                    continue

        # ------------------------------------------------------------------
        # Round 0: Rule matching
        # ------------------------------------------------------------------
        # Skip rule matching after cycle 2 on the same level —
        # saves ~12s per cycle. Re-match only on cycle 1, 2, or level change.
        # For puzzle levels, match only on cycle 1 (action semantics are stable).
        _t0 = time.time()
        _skip_rules = (cycle_count > 2 and last_matched) or (
            _is_puzzle and cycle_count > 1 and last_matched
        )
        if _skip_rules:
            matched = last_matched
            log("  [rules] reusing previous matches (cycle > 2, same level)")
        else:
            matched = await _match_rules(rule_engine, obs, action_history, env_id=env_id)
        last_matched = matched
        if matched:
            ids = [m.rule_id for m in matched]
            all_matched_ids.extend(ids)
            log(f"  [rules] matched: {ids}")

        ep_log.cycle_start(cycle_count, step_count, max_steps, levels_now,
                           [m.rule_id for m in matched])

        _rules_ms = int((time.time() - _t0) * 1000)

        _t0 = time.time()
        rules_section = rule_engine.format_fired_rules_for_prompt(matched, current_level=levels_now + 1)
        tools_section = tool_registry.build_tool_section_for_prompt()

        # State / goal context
        _gc = goal_manager.format_for_prompt()
        _sc = state_manager.format_for_prompt()
        _gs = ""
        if _gc and _gc != "Goals: (none)":
            _gs += f"\n{_gc}"
        if _sc and _sc not in ("Current state: (empty)", ""):
            _gs += f"\n{_sc}"

        # ------------------------------------------------------------------
        # Round 1: OBSERVER
        # ------------------------------------------------------------------
        available_actions = list(getattr(env, "action_space", []))
        concept_bindings: dict = state_manager._data.get("concept_bindings") or {}

        _prep_ms = int((time.time() - _t0) * 1000)

        # Pre-compute structural context (zero LLM cost) so it can be passed
        # directly to the MEDIATOR — the OBSERVER may summarize/drop BFS routes.
        _t0 = time.time()
        _curr_frame = obs_frame(obs)

        # Fix 1: cache structural_str — only rebuild when frame or bindings changed.
        # Use a lightweight frame signature (sample of cells) for fast comparison.
        _h_f = len(_curr_frame)
        _w_f = len(_curr_frame[0]) if _curr_frame else 0
        _frame_sig = (
            _h_f, _w_f,
            tuple(_curr_frame[0]) if _curr_frame else (),
            tuple(_curr_frame[_h_f // 2]) if _curr_frame else (),
            tuple(_curr_frame[-1]) if _curr_frame else (),
            tuple(sorted(str(k) for k in concept_bindings.keys())) if concept_bindings else (),
        )
        if _frame_sig != _last_structural_frame_sig:
            structural_str = format_structural_context(
                _curr_frame,
                concept_bindings=concept_bindings,
                known_dynamic_colors=known_dynamic_colors,
                explored_colors=explored_colors,
                action_directions=action_directions,
                contact_events=contact_events,
            )
            _last_structural_str = structural_str
            _last_structural_frame_sig = _frame_sig
        else:
            structural_str = _last_structural_str

        _struct_ms = int((time.time() - _t0) * 1000)

        # Detect puzzle levels from structural context (≥6 visual groups or slot strips).
        _is_puzzle = "Visual groups:" in structural_str or "Slot strips:" in structural_str

        # Check if the frame changed enough to warrant a new OBSERVER call.
        _frame_diff = 0
        if _last_obs_frame:
            for r_idx in range(min(len(_curr_frame), len(_last_obs_frame))):
                for c_idx in range(min(len(_curr_frame[r_idx]), len(_last_obs_frame[r_idx]))):
                    if _curr_frame[r_idx][c_idx] != _last_obs_frame[r_idx][c_idx]:
                        _frame_diff += 1
        else:
            _frame_diff = 999  # first cycle — always observe

        # ------------------------------------------------------------------
        # BFS-DIRECT mode (COMPETITION_MODE only)
        # Pre-compute the BFS navigation path here, before OBSERVER/MEDIATOR.
        # If BFS has a valid action sequence, we bypass both OBSERVER and
        # MEDIATOR entirely and execute those actions directly — eliminating
        # the LLM deviation problem that caused levels to be missed.
        # ------------------------------------------------------------------
        _bfs_direct_actions: list = []
        _bfs_direct_nav_str: str  = ""
        _bfs_direct_gk_str:  str  = ""
        if COMPETITION_MODE:
            _gk_pre = _load_default_gk_registry()
            _bfs_direct_nav_str = _compute_bfs_nav_section(
                frame=_curr_frame,
                game_id=env_id,
                level=levels_now + 1,
                gk_registry=_gk_pre,
                action_history=action_history,
                level_initial_frame=_level_initial_frame,
                concept_bindings=state_manager._data.get("concept_bindings"),
                history_start_idx=_level_history_start,
                tracked_player_colors=_tracked_player_colors,
            )
            _bfs_direct_actions = _parse_bfs_action_list(_bfs_direct_nav_str)
            _bfs_direct_gk_str  = _build_game_knowledge_section(
                _gk_pre, env_id, levels_now + 1
            )

        _t0 = time.time()

        # --- Spending gate: check USD cap and LLM cycle cap ---
        # Once either limit is hit in competition mode, suppress all LLM calls
        # for the remainder of this episode (BFS-direct will still run).
        _llm_allowed = True
        if COMPETITION_MODE:
            _current_cost = get_cost_tracker().cost_usd()
            _over_usd  = COMPETITION_MAX_LLM_USD  > 0 and _current_cost  >= COMPETITION_MAX_LLM_USD
            _over_cyc  = COMPETITION_MAX_LLM_CYCLES > 0 and _llm_cycles_used >= COMPETITION_MAX_LLM_CYCLES
            if _over_usd or _over_cyc:
                _llm_allowed = False
                reason = f"cost=${_current_cost:.3f}>=${COMPETITION_MAX_LLM_USD}" if _over_usd \
                    else f"llm_cycles={_llm_cycles_used}>={COMPETITION_MAX_LLM_CYCLES}"
                print(f"  [GATE] LLM suppressed ({reason}) — BFS-direct only", flush=True)

        # --- BFS-DIRECT: skip OBSERVER+MEDIATOR when BFS has a valid path ---
        _bfs_direct_mode = COMPETITION_MODE and bool(_bfs_direct_actions)
        if _bfs_direct_mode:
            obs_text            = _obs_text_current or ""
            _obs_text_current   = obs_text
            action_plan         = [{"action": a, "data": {}} for a in _bfs_direct_actions[:COMPETITION_MAX_PLAN_CHUNK]]
            med_text            = ""
            _med_plan_current   = action_plan
            _med_reason_current = ""
            _t0 = time.time()   # reset timer so _observer_ms ≈ 0 below
            log(f"  [BFS-DIRECT] Executing BFS path: {[p['action'] for p in action_plan]}")
        # --- LLM gate: if spending cap hit and no BFS path, skip the whole cycle ---
        elif COMPETITION_MODE and not _llm_allowed:
            log("  [GATE] No BFS path and LLM suppressed — skipping cycle")
            print("  [GATE] No BFS path and LLM suppressed — skipping cycle", flush=True)
            continue
        # --- Puzzle-level OBSERVER bypass (SKIP_OBSERVER_FOR_PUZZLES) ---
        # On puzzle levels the structural context already contains groups,
        # focus/cursor, mismatch info, and reference slot mapping — everything
        # the MEDIATOR needs. We build a lightweight synthetic OBSERVER output
        # from locally-computed data, saving ~$0.15 per skipped call.
        elif COMPETITION_MODE and (_frame_diff > 3 or not _obs_text_current):
            # --- COMPETITION MODE OBSERVER ---
            # Use the cheap single-call run_observer (same as non-competition).
            # The 4-turn tool-loop (run_observer_agent) costs up to 4x more per
            # cycle and provides no meaningful benefit when BFS-direct handles
            # all navigation — the OBSERVER output is only used in the rare
            # NO-PATH fallback cycles.
            obs_text, _observer_ms = await run_observer(
                obs,
                available_actions,
                action_history,
                rules_section=rules_section,
                action_effects=state_manager._data.get("action_effects"),
                concept_bindings=concept_bindings,
                steps_remaining=max_steps - step_count,
                known_dynamic_colors=known_dynamic_colors,
                explored_colors=explored_colors,
                action_directions=action_directions,
                contact_events=contact_events,
                verbose=verbose,
            )
            _last_obs_frame = [row[:] for row in _curr_frame]
            _llm_cycles_used += 1
            log(f"  [OBSERVER] competition single-call (llm_cycles={_llm_cycles_used})")
            # Inject top recalled concepts into the MEDIATOR's rules_section.
            # This closes the producer->consumer loop: concepts the agent has
            # learned (in this episode or previous ones) are surfaced to the
            # planner so it can act on them.
            try:
                from agent_loop import get_registry as _get_reg, DOMAIN as _DOMAIN
                _reg = _get_reg()
                _hits = _reg.recall(
                    domain=_DOMAIN,
                    limit=5,
                    include_cross_domain=True,
                    cross_domain_kinds=["compositional", "mechanic"],
                )
                if _hits:
                    _lines = ["", "## Learned mechanics (from ConceptRegistry)", ""]
                    for _c in _hits:
                        _tag = "" if _c.domain == _DOMAIN else f" [from {_c.domain}]"
                        _lines.append(
                            f"- **{_c.name}** ({_c.kind}, conf={_c.confidence:.2f}){_tag}"
                        )
                        _lines.append(f"  - {_c.abstraction.get('summary', '')}")
                        _hint = _c.abstraction.get("hint", "")
                        if _hint:
                            _lines.append(f"  - hint: {_hint}")
                    rules_section = (rules_section or "") + "\n".join(_lines)
                    log(f"  [CONCEPTS] injected {len(_hits)} learned concept(s) into MEDIATOR")
            except Exception as _e:
                log(f"  [CONCEPTS] recall failed: {_e}")
        elif SKIP_OBSERVER_FOR_PUZZLES and _is_puzzle and not COMPETITION_MODE:
            _action_effects = state_manager._data.get("action_effects") or {}
            _effects_str = _format_action_effects(_action_effects)
            _preds = compute_trend_predictions(_action_effects, max_steps - step_count)
            _preds_str = ("\n".join(f"  {p}" for p in _preds)
                          if _preds else "  (none)")
            obs_text = (
                "## Structural context (puzzle level — OBSERVER bypassed)\n\n"
                f"{structural_str}\n\n"
                "## Observed action effects\n\n"
                f"{_effects_str}\n\n"
                "## Trend predictions\n\n"
                f"{_preds_str}\n"
            )
            log("  [OBSERVER] Puzzle level — bypassed LLM (structural context only)")
        elif _frame_diff > 3 or not _obs_text_current:
            obs_text, _obs_ms = await run_observer(
                obs,
                available_actions,
                action_history,
                rules_section=rules_section,
                action_effects=state_manager._data.get("action_effects"),
                concept_bindings=concept_bindings,
                steps_remaining=max_steps - step_count,
                known_dynamic_colors=known_dynamic_colors,
                explored_colors=explored_colors,
                action_directions=action_directions,
                contact_events=contact_events,
                verbose=verbose,
            )
            _last_obs_frame = [row[:] for row in _curr_frame]
        else:
            obs_text = _obs_text_current  # reuse cached OBSERVER output
            log("  [OBSERVER] Frame unchanged (<4 pixels) — reusing cached analysis")
        _observer_ms = int((time.time() - _t0) * 1000)

        _obs_text_current = obs_text
        ep_log.observer_output(obs_text)

        # Merge any new concept bindings the OBSERVER proposed.
        # For integer (color) keys, accumulate confidence and observation count
        # rather than overwriting, so repeated observations strengthen bindings.
        new_bindings = parse_concept_bindings(obs_text)
        if new_bindings:
            existing = state_manager._data.get("concept_bindings") or {}
            merged = dict(existing)
            changed = False
            for k, v in new_bindings.items():
                if not isinstance(k, int):
                    # Pass-through (e.g. wall_colors)
                    if merged.get(k) != v:
                        merged[k] = v
                        changed = True
                    continue
                # Build the incoming binding as a normalised dict
                if isinstance(v, dict):
                    new_conf = v.get("confidence", 0.6)
                    new_role = str(v.get("role", ""))
                else:
                    new_conf = 0.6
                    new_role = str(v)

                if k not in merged:
                    merged[k] = {
                        "role":           new_role,
                        "confidence":     round(new_conf, 3),
                        # level_obs: observations in the current level/episode
                        # total_obs: lifetime observations across all levels/games
                        "level_obs":      1,
                        "total_obs":      1,
                    }
                    changed = True
                else:
                    prev = merged[k]
                    # Upgrade plain string bindings to rich dict on first merge
                    if isinstance(prev, str):
                        prev = {"role": prev, "confidence": 0.5,
                                "level_obs": 1, "total_obs": 1}
                    # Keep backward compat with old dicts that used "observations"
                    if "observations" in prev and "total_obs" not in prev:
                        prev["total_obs"] = prev.pop("observations")
                        prev.setdefault("level_obs", prev["total_obs"])
                    level_obs  = prev.get("level_obs", 1)
                    total_obs  = prev.get("total_obs", 1)
                    # Weighted running average on total observations —
                    # more observations → slower drift, i.e. harder to un-bind
                    blended = (prev["confidence"] * total_obs + new_conf) / (total_obs + 1)
                    updated = {
                        "role":       new_role,
                        "confidence": round(min(blended, 1.0), 3),
                        "level_obs":  level_obs + 1,
                        "total_obs":  total_obs + 1,
                    }
                    if updated != prev:
                        merged[k] = updated
                        changed = True
            if changed:
                state_manager._data["concept_bindings"] = merged
                # Log only color-keyed bindings in a readable form
                summary = {
                    f"color{k}": (
                        f"{v['role']}({v['confidence']:.0%}, {v.get('total_obs', '?')}obs)"
                        if isinstance(v, dict) else v
                    )
                    for k, v in merged.items() if isinstance(k, int)
                }
                log(f"  [CONCEPTS] {summary}")
                ep_log._write(f"  [CONCEPTS] {summary}")

        # ------------------------------------------------------------------
        # Round 2: MEDIATOR (skipped in BFS-direct mode)
        # ------------------------------------------------------------------
        _mediator_ms = 0  # default; overwritten below when mediator runs
        if not _bfs_direct_mode:
          _t0 = time.time()
          # Fix 2: cap action history at 6 entries for MEDIATOR — the
          # `## Recent action outcomes` already sent to OBSERVER covers last 8
          # steps compactly; the full 15-entry history is redundant.
          _mediator_history = action_history[-6:] if action_history else []

          # Fix 3: in COMPETITION_MODE the OBSERVER already summarises the
          # frame; sending structural_str to MEDIATOR again is double-billing.
          _mediator_struct = "" if COMPETITION_MODE else structural_str

          _gk = _load_default_gk_registry()
          _game_knowledge_str = _build_game_knowledge_section(
              _gk, env_id, levels_now + 1
          )
          _bfs_nav_str = _compute_bfs_nav_section(
              frame=_curr_frame,
              game_id=env_id,
              level=levels_now + 1,
              gk_registry=_gk,
              action_history=action_history,
              level_initial_frame=_level_initial_frame,
              concept_bindings=state_manager._data.get("concept_bindings"),
              history_start_idx=_level_history_start,
              tracked_player_colors=_tracked_player_colors,
          )
          if _bfs_nav_str:
              _game_knowledge_str = (
                  (_game_knowledge_str + "\n" if _game_knowledge_str else "")
                  + _bfs_nav_str
              )

          action_plan, med_text, _med_ms = await run_mediator(
              obs_text,
              rules_section=rules_section,
              tools_section=tools_section,
              action_history=_mediator_history,
              available_actions=available_actions,
              state_section=_gs,
              action_directions=action_directions if action_directions else None,
              structural_context_str=_mediator_struct,
              game_knowledge_section=_game_knowledge_str,
              verbose=verbose,
          )
          _mediator_ms = int((time.time() - _t0) * 1000)
          # Count this as an LLM cycle (observer+mediator both ran)
          if COMPETITION_MODE:
              _cost_now = get_cost_tracker().cost_usd()
              print(f"  [GATE] LLM cycle {_llm_cycles_used} complete "
                    f"(cumulative cost=${_cost_now:.3f})", flush=True)

        _med_plan_current  = action_plan
        _med_reason_current = _extract_reasoning(med_text)
        ep_log.mediator_output(med_text, action_plan)

        # Parse goal + state updates from MEDIATOR response
        _updates = GoalManager.parse_agent_updates(med_text or "")
        if _updates:
            if "goal_updates" in _updates:
                try:
                    _glog = goal_manager.apply_updates(_updates)
                    if _glog:
                        log(f"  [goals] {'; '.join(_glog)}")
                        # Verbose: one console line per goal change for visibility.
                        for entry in _glog:
                            log(f"    [GOAL+] {entry}")
                            ep_log.goal_event("update", "", entry)
                except Exception:
                    pass  # malformed goal_updates — ignore gracefully
            if "state_updates" in _updates:
                state_manager.apply_agent_updates(_updates["state_updates"])
                sets = _updates["state_updates"].get("set", {})
                log(f"  [state] updated: {list(sets.keys())}")
                # Verbose: show each key=value so user can see what MEDIATOR set.
                for _k, _v in sets.items():
                    _vs = str(_v)
                    if len(_vs) > 120:
                        _vs = _vs[:117] + "..."
                    log(f"    [STATE+] {_k} = {_vs}")
                ep_log.state_update(sets)

        # Extract candidate rules proposed by MEDIATOR (exploration + planning).
        # All arc-agi-3 rules start as "candidate" — they must be independently
        # confirmed (level advance or win) before being promoted to "active".
        # In COMPETITION_MODE, skip rule proposals to prevent feedback loops
        # from accumulating incorrect game-specific coordinate rules.
        if COMPETITION_MODE:
            pass
        elif med_text:
            rule_changes = rule_engine.parse_mediator_rule_updates(
                med_text, task_id, source_level=levels_now + 1
            )
            if rule_changes:
                new_ids = []
                for r in rule_changes:
                    _is_new = r.get("lineage", {}).get("type") == "new"
                    if _is_new:
                        # Downgrade active → candidate so it needs confirmation
                        r["status"] = "candidate"
                        rule_engine.save()
                    new_ids.append(r["id"])
                    # Verbose: print each proposed rule with its IF/THEN so
                    # the user can see what's being learned in real time.
                    _tag = "NEW" if _is_new else "UPD"
                    _cond = (r.get("condition", "") or "").strip()
                    _act  = (r.get("action", "") or "").strip()
                    if len(_cond) > 140: _cond = _cond[:137] + "..."
                    if len(_act)  > 140: _act  = _act[:137] + "..."
                    log(f"    [RULE {_tag}] {r['id']} ({r.get('status','candidate')})")
                    log(f"      IF:   {_cond}")
                    log(f"      THEN: {_act}")
                    ep_log.rule_proposed(
                        r["id"], r.get("status", "candidate"), _cond, _act,
                    )
                log(f"  [rules] {len(rule_changes)} rule(s) proposed: {new_ids}")

        # Print per-cycle cost + goal summary after MEDIATOR completes
        if verbose:
            ct_cycle = get_cost_tracker()
            log(f"  [COST] calls={ct_cycle.api_calls} "
                f"in={ct_cycle.input_tokens}tok out={ct_cycle.output_tokens}tok "
                f"cost=${ct_cycle.cost_usd():.4f}")
            active_goals = [g for g in goal_manager._goals
                            if g.status in ("active", "pending")]
            if active_goals:
                log("  [GOALS]")
                for g in active_goals[:6]:
                    indent = "    " if g.parent_id else "  "
                    log(f"  {indent}[{g.priority}] {g.description[:70]}")
            concepts = state_manager._data.get("concept_bindings") or {}
            if concepts:
                cb_str = "  ".join(f"color{c}={n}" for c, n in sorted(concepts.items(), key=lambda x: str(x[0])))
                log(f"  [CONCEPTS] {cb_str}")

        # Timing summary for cycle
        _cycle_total_ms = int((time.time() - _cycle_t0) * 1000)
        log(f"  [TIMING] cycle={_cycle_total_ms}ms  "
            f"rules={_rules_ms}ms  struct={_struct_ms}ms  "
            f"observer={_observer_ms}ms  mediator={_mediator_ms}ms")
        ep_log._write(
            f"  [TIMING] cycle={_cycle_total_ms}ms  "
            f"rules={_rules_ms}ms  struct={_struct_ms}ms  "
            f"observer={_observer_ms}ms  mediator={_mediator_ms}ms"
        )

        if not action_plan:
            log("  [MEDIATOR] No action plan produced — cycle skipped")
            continue

        # ------------------------------------------------------------------
        # Round 3: Execute action plan
        # ------------------------------------------------------------------
        log(f"  [ACTOR] Executing {len(action_plan)} action(s)...")

        matched_ids_now = [m.rule_id for m in matched]
        active_goals_snap = [
            {"id": g.id, "description": g.description, "status": g.status,
             "priority": g.priority, "parent_id": g.parent_id}
            for g in goal_manager._goals
            if g.status in ("active", "pending")
        ]
        state_snap = dict(state_manager._data)

        _plan_chunk_count = 0
        for step_spec in action_plan:
            if step_count >= max_steps:
                log(f"  [ACTOR] Step limit ({max_steps}) reached, stopping")
                break
            # In competition mode, limit plan execution to a small chunk so the
            # MEDIATOR re-evaluates with fresh observations rather than blindly
            # executing a long stale plan.
            if COMPETITION_MODE and _plan_chunk_count >= COMPETITION_MAX_PLAN_CHUNK:
                log(f"  [ACTOR] Competition plan-chunk limit "
                    f"({COMPETITION_MAX_PLAN_CHUNK}) reached — re-evaluating")
                break

            action_name = step_spec["action"]
            data        = step_spec.get("data") or {}

            # Look up action object from env.action_space by name
            action_obj = next(
                (a for a in getattr(env, "action_space", [])
                 if getattr(a, "name", str(a)) == action_name),
                None,
            )
            if action_obj is None:
                log(f"    Unknown action '{action_name}', skipping")
                continue

            frame_before  = obs_frame(obs)
            levels_before = obs_levels_completed(obs)
            obs = env.step(action_obj, data=data)
            step_count      += 1
            _plan_step_idx  += 1
            _plan_chunk_count += 1

            levels_after = obs_levels_completed(obs)
            state_after  = obs_state_name(obs)

            # Compute per-step pixel diff so the agent loop can detect
            # wall-hits / dead actions on subsequent cycles.
            _frame_after = obs_frame(obs)
            _step_diff = 0
            try:
                for _r in range(min(len(_frame_after), len(frame_before))):
                    _ra = _frame_after[_r]
                    _rb = frame_before[_r]
                    for _c in range(min(len(_ra), len(_rb))):
                        if _ra[_c] != _rb[_c]:
                            _step_diff += 1
            except Exception:
                _step_diff = -1

            # Record player position alongside diff so CHANGER detection
            # can use position (not just diff) to avoid false positives from
            # the rotated-sprite diff spike that follows a real CHANGER visit.
            # _tracked_player_colors adapts when a color-changer alters the
            # player sprite color (e.g. level 3: color 12 → 9 after visit).
            try:
                from nav_bfs import find_player_position as _fppos2
                _ppos2 = _fppos2(obs_frame(obs), _tracked_player_colors)
                if _ppos2 is None and _step_diff > 50:
                    # Player color may have changed — infer from frame diff
                    from dynamic_discovery import infer_player_colors_from_diff
                    _inferred_pc = infer_player_colors_from_diff(
                        frame_before, obs_frame(obs)
                    )
                    if _inferred_pc:
                        # Exclude walkable colors from inferred set
                        _wc_set = set(_gk_init.get("walkable_colors") or [])
                        _inferred_pc -= _wc_set
                    if _inferred_pc:
                        _ppos2 = _fppos2(obs_frame(obs), _inferred_pc)
                        if _ppos2 is not None:
                            _tracked_player_colors = _inferred_pc
                            print(f"  [HYP] player color updated: "
                                  f"{_tracked_player_colors}", flush=True)
            except Exception:
                _ppos2 = None
            action_history.append({
                "action":      action_name,
                "data":        data,
                "levels":      levels_after,
                "state":       state_after,
                "diff":        _step_diff,
                "player_pos":  _ppos2,
                "frame_before": frame_before,
            })

            # Always-on progress line: step / action / diff / player pos / levels.
            # P1: use _tracked_player_colors (adaptive), not re-read from gk.
            try:
                from nav_bfs import find_player_position as _fppos
                _ppos = _fppos(obs_frame(obs), _tracked_player_colors)
            except Exception:
                _ppos = "?"
            _changer_flag = " *** LARGE DIFF ***" if _step_diff > 80 else ""
            _level_flag   = f" -> LEVEL {levels_after}!" if levels_after > levels_before else ""
            print(
                f"  step {step_count:3d} {action_name:8s} "
                f"diff={_step_diff:4d} pos={str(_ppos):12s} "
                f"L={levels_after}{_changer_flag}{_level_flag}",
                flush=True,
            )

            # Write playlog for this step (use live state so action_effects show up)
            if playlog_dir is not None:
                ct_now = get_cost_tracker()
                _write_step_log(
                    playlog_dir   = playlog_dir,
                    step_number   = step_count,
                    episode       = episode_num,
                    cycle         = cycle_count,
                    action_name   = action_name,
                    action_data   = data,
                    obs           = obs,
                    prev_frame    = frame_before,
                    observer_text = _obs_text_current,
                    mediator_plan = _med_plan_current,
                    mediator_reasoning = _med_reason_current,
                    matched_rule_ids   = matched_ids_now,
                    active_goals       = [
                        {"id": g.id, "description": g.description,
                         "status": g.status, "priority": g.priority,
                         "parent_id": g.parent_id}
                        for g in goal_manager._goals
                        if g.status in ("active", "pending")
                    ],
                    state_snapshot     = dict(state_manager._data),
                    cost_usd     = ct_now.cost_usd(),
                    input_tokens = ct_now.input_tokens,
                    output_tokens= ct_now.output_tokens,
                    api_calls    = ct_now.api_calls,
                    plan_index   = _plan_step_idx,
                    win_levels   = _win_levels,
                )

            _step_t0 = time.time()
            curr_frame = obs_frame(obs)
            change = _compute_change_summary(frame_before, curr_frame)
            prev_frame = curr_frame

            # -- Accumulate action-effect observations into state (zero cost) --
            _accumulate_action_effect(
                state_manager, action_name, data, change, levels_after,
                frame_before=frame_before,
                frame_after=curr_frame,
            )

            # Object-level summary for the log
            obj_diff = diff_objects(frame_before, curr_frame)
            obj_summary = format_object_diff(obj_diff)

            # Track which colors have ever moved (used by structural context)
            known_dynamic_colors.update(
                m.obj.color for m in obj_diff.moved if not m.obj.is_background
            )

            # Update confirmed action directions from latest observations.
            new_dirs = infer_action_directions(
                state_manager._data.get("action_effects") or {}
            )
            action_directions.update(new_dirs)

            # Track which object colors the player has contacted (adjacent to).
            # Uses known_dynamic_colors as the player color set — after the first
            # move, dynamic colors are the player/cursor.
            if known_dynamic_colors:
                contacts = detect_contacts(
                    known_dynamic_colors, detect_objects(curr_frame)
                )
                # "explored" = player bbox overlaps the object (gap = 0, full contact).
                # Near-contact (gap 1–5) is captured in contacts but does NOT mark
                # an object explored — reserved for future proximity-trigger use cases.
                newly_touched = {
                    c for c, gap in contacts.items()
                    if gap == 0 and c not in explored_colors
                }
                if newly_touched:
                    # Record causal world changes for each newly touched object.
                    # We compare the frame just before this action vs. the frame
                    # after, filtering out the player's own pixels, to isolate
                    # what changed in the environment due to the contact.
                    delta = detect_arena_delta(
                        frame_before, curr_frame, known_dynamic_colors
                    )
                    step_num = state_manager._data.get("total_steps_taken", 0)
                    for touched_color in newly_touched:
                        event = {
                            "touched_color": touched_color,
                            "step":          step_num,
                            "delta":         delta,
                        }
                        contact_events.append(event)
                        if delta["any_change"]:
                            appeared_desc = ", ".join(
                                f"color{a['color']} appeared"
                                for a in delta["appeared"]
                            )
                            disappeared_desc = ", ".join(
                                f"color{d['color']} disappeared"
                                for d in delta["disappeared"]
                            )
                            changed_desc = ", ".join(
                                f"color{ch['color']} size {ch['delta']:+d}"
                                for ch in delta["changed_size"]
                            )
                            parts = [p for p in [appeared_desc, disappeared_desc, changed_desc] if p]
                            ep_log._write(
                                f"  [CONTACT] color{touched_color} touched at step {step_num}"
                                f" -> world changed: {'; '.join(parts)}"
                            )
                        else:
                            ep_log._write(
                                f"  [CONTACT] color{touched_color} touched at step {step_num}"
                                f" -> no world change detected"
                            )
                    explored_colors.update(newly_touched)

            # --- Zero-cost concept auto-detection ----------------------------
            # Run heuristic signatures over action_effects and emit guesses into
            # concept_bindings immediately — before the next LLM cycle runs.
            # The OBSERVER can confirm, raise confidence, or override these.
            effects_for_detection = state_manager._data.get("action_effects") or {}
            cb_current = state_manager._data.get("concept_bindings") or {}
            auto_suggestions = auto_detect_concepts(
                effects_for_detection, detect_objects(curr_frame), cb_current
            )
            if auto_suggestions:
                cb_merged = dict(cb_current)
                for color, suggestion in auto_suggestions.items():
                    existing = cb_merged.get(color)
                    if isinstance(existing, dict):
                        # Blend: keep whichever confidence is higher.
                        if suggestion["confidence"] > existing.get("confidence", 0):
                            prev_conf = existing.get("confidence", 0)
                            cb_merged[color] = {**existing, **suggestion}
                            ep_log.concept_update(
                                color, suggestion["role"],
                                f"[GUESS] auto-detected (conf {prev_conf:.2f}->{suggestion['confidence']:.2f},"
                                f" n={suggestion['observations']})",
                            )
                    else:
                        cb_merged[color] = suggestion
                        ep_log.concept_update(
                            color, suggestion["role"],
                            f"[GUESS] auto-detected (conf {suggestion['confidence']:.2f},"
                            f" n={suggestion['observations']})",
                        )
                state_manager.update({"concept_bindings": cb_merged})

            # --- Co-occurrence observation -----------------------------------
            concept_bindings_now = state_manager._data.get("concept_bindings") or {}
            co_events = events_from_step(
                obj_diff, concept_bindings_now,
                levels_delta=levels_after - levels_before,
            )
            if co_events:
                co_registry.observe_step(co_events)

            # --- Wall detection: if an object that normally moves didn't ----
            effects_now = state_manager._data.get("action_effects") or {}
            typical_dir = infer_typical_direction(effects_now, action_name)
            if typical_dir:
                # Find non-background objects that didn't move this step
                moved_colors = {m.obj.color for m in obj_diff.moved}
                for candidate in detect_objects(frame_before):
                    if candidate.is_background or candidate.color in moved_colors:
                        continue
                    # Only care about objects that have moved before under this action
                    past_movers = {
                        mv.get("color")
                        for obs in effects_now.get(action_name, {})
                                              .get("object_observations", [])
                        for mv in obs.get("moved", [])
                    }
                    if candidate.color not in past_movers:
                        continue
                    contact = detect_wall_contacts(frame_before, candidate, typical_dir)
                    if contact and contact.get("adjacent_colors"):
                        wall_colors = contact["adjacent_colors"]
                        wall_str = ", ".join(
                            f"color{c}({color_name(c)}) x{n}"
                            for c, n in sorted(wall_colors.items())
                        )
                        msg = (f"wall contact detected: "
                               f"{color_name(candidate.color)}(color{candidate.color}) "
                               f"blocked {typical_dir} by [{wall_str}]")
                        log(f"    [WALL] {msg}")
                        ep_log._write(f"  [WALL] {msg}")
                        # Record wall candidates under the concept name "wall",
                        # not under the color. Colors are game-specific; the
                        # concept "wall" is not. A different game may bind a
                        # completely different color to "wall".
                        bindings = state_manager._data.get("concept_bindings") or {}
                        known_walls: set = set(bindings.get("wall_colors", []))
                        # Exclude colors already bound to a non-wall role (e.g.
                        # player_piece, step_counter) — those are game actors,
                        # not walls.  Also exclude every color that has moved at
                        # any point this episode (moving objects can't be walls).
                        non_wall_concepts: set = {
                            k for k, v in bindings.items()
                            if isinstance(k, int) and v != "wall"
                        }
                        ever_moved: set = {
                            mv.get("color")
                            for _act, act_data in effects_now.items()
                            for obs in act_data.get("object_observations", [])
                            for mv in obs.get("moved", [])
                        }
                        # Containers hold other objects inside them (e.g. goal
                        # boxes) — never label a container color as a wall.
                        container_colors: set = {
                            rel.container.color
                            for rel in detect_containment(
                                detect_objects(frame_before)
                            )
                        }
                        excluded = non_wall_concepts | ever_moved | container_colors | {0}
                        new_walls = set(wall_colors.keys()) - known_walls - excluded
                        if new_walls:
                            known_walls.update(new_walls)
                            bindings["wall_colors"] = sorted(known_walls)
                            state_manager._data["concept_bindings"] = bindings
                            for wc in new_walls:
                                ep_log._write(
                                    f"  [CONCEPTS] wall candidate added: "
                                    f"color{wc} ({color_name(wc)}) "
                                    f"[game-local, not permanent]"
                                )

            # --- Urgency goals from trend predictions -----------------------
            preds = compute_trend_predictions(
                state_manager._data.get("action_effects") or {},
                max_steps - step_count,
            )
            for pred in preds:
                if "[URGENT]" in pred:
                    # Check if we already have a goal for this prediction
                    pred_short = pred[:60]
                    already = any(
                        pred_short[:40] in g.description
                        for g in goal_manager._goals
                        if g.status in ("active", "pending")
                    )
                    if not already:
                        ug = goal_manager.push(
                            description=f"URGENT: {pred[:120]}",
                            priority=1,
                            parent_id=top_goal_id,
                        )
                        goal_manager.activate(ug.id)
                        log(f"    [GOAL] urgency goal pushed: {pred[:80]}")
                        ep_log._write(f"  [GOAL URGENT] {pred}")

            _step_proc_ms = int((time.time() - _step_t0) * 1000)

            data_str = f" {data}" if data else ""
            log(f"    step {step_count}: {action_name}{data_str}"
                f" -> {state_after} levels={levels_after}"
                f" diff={change['diff_count']}"
                f" proc={_step_proc_ms}ms")
            if verbose and obj_summary != "no object-level changes detected":
                for line in obj_summary.splitlines():
                    log(f"      {line.strip()}")
            ep_log.step_result(
                step_count, action_name, state_after,
                levels_after, change, obj_summary,
            )

            if state_after in ("WIN", "GAME_OVER"):
                break

            if levels_after > levels_before:
                last_matched = []  # force rule re-match on new level
                _last_obs_frame = []  # force OBSERVER re-run on new level
                _level_just_changed = True  # next cycle: skip stale transition frame
                log(f"  [ACTOR] Level advanced: {levels_before} -> {levels_after}")
                ep_log.level_advance(levels_before, levels_after)
                # Persist discovered knowledge for this level (RC visits, color roles).
                # This allows future episodes to navigate without hardcoded data.
                try:
                    from dynamic_discovery import (
                        update_discovered_knowledge,
                        build_level_model,
                        load_discovered_knowledge,
                        get_n_rc_visits,
                    )
                    _HERE_UC = Path(__file__).resolve().parent
                    _DK_PATH_UC = _HERE_UC / "discovered_knowledge.json"
                    _dk_now = load_discovered_knowledge(_DK_PATH_UC)
                    _completed_level = levels_before + 1  # 1-indexed
                    # Count RC visits for this level (from history since level start)
                    # P1: use game_knowledge priors, no hardcoded fallbacks.
                    _pc_uc = set(_tracked_player_colors)
                    _wc_uc = set(_gk_init.get("walkable_colors") or [])
                    _ss_uc = _gk_init.get("step_size", 1)
                    _lif_uc = _level_initial_frame
                    _cb_uc = state_manager._data.get("concept_bindings") or {}
                    _curr_f_uc = obs_frame(obs)
                    _ppos_uc = find_player_position(_curr_f_uc, _pc_uc)
                    _model_uc = build_level_model(
                        initial_frame=_lif_uc,
                        current_frame=_curr_f_uc,
                        action_history=action_history,
                        walkable_colors=_wc_uc,
                        player_colors=_pc_uc,
                        step_size=_ss_uc,
                        player_pos=_ppos_uc,
                        concept_bindings=_cb_uc,
                        history_start_idx=_level_history_start,
                        last_reset_idx=-1,
                    )
                    # Count confirmed RC visits in this level cycle
                    _rc_set_uc = {tuple(p) for p in _model_uc.rc_positions}
                    _rc_v_uc = sum(
                        1 for _si in range(_level_history_start, len(action_history))
                        if tuple(action_history[_si].get("player_pos") or ()) in _rc_set_uc
                    )
                    if _rc_set_uc and get_n_rc_visits(_dk_now, env_id, _completed_level) is None:
                        # Save rc_visits and color roles for this level.
                        # Use the SPRITE color from model.candidates (the
                        # color of the non-walkable overlay), not the floor
                        # pixel at the grid anchor which is walkable.
                        _color_roles_uc: dict[int, str] = {}
                        _cands_uc = _model_uc.candidates
                        for _rp in _model_uc.rc_positions:
                            _rcolor = _cands_uc.get(tuple(_rp), -1)
                            if _rcolor > 0:
                                _color_roles_uc[_rcolor] = "rotation_changer"
                        for _rng in _model_uc.ring_positions + _model_uc.consumed_rings:
                            _rng_color = _cands_uc.get(tuple(_rng), -1)
                            if _rng_color > 0:
                                _color_roles_uc[_rng_color] = "step_counter_ring"
                        if _model_uc.win_gate:
                            _wg_color = _cands_uc.get(tuple(_model_uc.win_gate), -1)
                            if _wg_color > 0:
                                _color_roles_uc[_wg_color] = "win_gate"
                        update_discovered_knowledge(
                            _DK_PATH_UC, env_id, _completed_level,
                            rc_visits=_rc_v_uc if _rc_v_uc > 0 else None,
                            color_roles=_color_roles_uc or None,
                        )
                except Exception as _dk_err:
                    log(f"  [DISCOVER] persistence error (non-fatal): {_dk_err}")
                # NOTE: frame image is saved by the stale-frame guard in the
                # next cycle (after one neutral step), so we get the real new-
                # level layout rather than the WIN transition frame.
                # Reset per-level observation counts in concept bindings so
                # short-term (level) and long-term (lifetime) stats stay distinct
                bindings_now = state_manager._data.get("concept_bindings") or {}
                for ck, cv in bindings_now.items():
                    if isinstance(ck, int) and isinstance(cv, dict):
                        cv["level_obs"] = 0
                _update_level_goals(
                    goal_manager, env_id, levels_after, top_goal_id,
                    gt_registry=_gt,
                )
                log(f"  [goals] {goal_manager.format_for_prompt()}")
                # Promote candidate rules that fired this cycle — level advance
                # is the progress signal confirming they were useful
                _promote_fired_candidates(
                    rule_engine, matched, task_id, log
                )
                # Slot-strip puzzles (TR87-type): the frame returned on level
                # completion still shows the previous level's solved state.
                # Take one ACTION4 "peek" step so the next OBSERVER cycle
                # sees the new level's initial visual content.
                if _is_puzzle and step_count < max_steps:
                    _peek_action = next(
                        (a for a in getattr(env, "action_space", [])
                         if getattr(a, "name", str(a)) == "ACTION4"),
                        None,
                    )
                    if _peek_action is not None:
                        obs = env.step(_peek_action, data={})
                        step_count += 1
                        log(f"  [ACTOR] Transition peek: ACTION4 to load "
                            f"next level frame (levels={obs_levels_completed(obs)})")
                break  # re-enter OBSERVER with fresh state

    # -----------------------------------------------------------------------
    # Episode complete
    # -----------------------------------------------------------------------
    final_state  = obs_state_name(obs)
    final_levels = obs_levels_completed(obs)
    won = final_state == "WIN"

    if won:
        for g in goal_manager._goals:
            if g.status == "active":
                goal_manager.resolve(g.id, result="Won the game")

    ct = get_cost_tracker()
    duration_ms = int(time.time() * 1000) - start_ms

    meta = EpisodeMetadata(
        episode              = episode_num,
        env_id               = env_id,
        levels_completed     = final_levels,
        steps_taken          = step_count,
        cycles               = cycle_count,
        state                = final_state,
        won                  = won,
        duration_ms          = duration_ms,
        cost_usd             = round(ct.cost_usd(), 6),
        input_tokens         = ct.input_tokens,
        cache_creation_tokens= ct.cache_creation_tokens,
        cache_read_tokens    = ct.cache_read_tokens,
        output_tokens        = ct.output_tokens,
        api_calls            = ct.api_calls,
        model                = DEFAULT_MODEL,
        matched_rule_ids     = list(dict.fromkeys(all_matched_ids)),
        playlog_dir          = str(playlog_dir) if playlog_dir else "",
        action_directions    = dict(action_directions),
    )

    log(f"\nEpisode {episode_num} complete: "
        f"levels={final_levels} state={final_state} "
        f"steps={step_count} cycles={cycle_count} "
        f"cost=${meta.cost_usd:.4f}")

    # Update rule stats based on episode outcome
    for m in last_matched:
        if won:
            rule_engine.record_success(m.rule_id, task_id)
        else:
            rule_engine.record_failure(m.rule_id, task_id)
    rule_engine.increment_tasks_seen(fired_ids=set(all_matched_ids))
    # arc-agi-3 exploration rules need multiple episodes to be confirmed;
    # don't kill a candidate just because one episode ended without a win.
    deprecated_ids = rule_engine.auto_deprecate(min_candidate_fired=5)
    for rid in deprecated_ids:
        r = rule_engine.get(rid)
        ep_log.rule_status_change(
            rid, "candidate", "deprecated",
            r.get("deprecated_reason", "") if r else "",
        )

    # -- Hard-prune: physically remove provably useless rules -----------------
    # Runs after auto_deprecate() so newly deprecated rules are included.
    # co_occ_stale_tasks=20: co-occurrence candidates that have been evaluated
    # 20+ times without ever firing are clearly not matching anything useful.
    prune_counts = rule_engine.hard_prune(
        remove_deprecated=True,
        remove_stale_orphans=True,
        co_occ_stale_tasks=20,
    )
    if prune_counts["total"] > 0:
        ep_log._write(
            f"[PRUNE] Hard-deleted {prune_counts['total']} rules: "
            f"{prune_counts['deprecated']} deprecated, "
            f"{prune_counts['orphans']} stale-orphan, "
            f"{prune_counts['co_occ_stale']} stale co-occurrence"
        )
        log(
            f"  [PRUNE] {prune_counts['total']} rules hard-deleted "
            f"({prune_counts['deprecated']} depr, "
            f"{prune_counts['orphans']} orphan, "
            f"{prune_counts['co_occ_stale']} co-occ stale)"
        )

    # -- Co-occurrence promotion: emit candidate rules for strong pairs -------
    # Thresholds are intentionally strict:
    #   min_count=8   — need 8 observations before calling it a pattern
    #   min_consistency=0.90 — must co-occur in 90% of steps where subject fires
    #   max_rules=20  — hard cap per episode to prevent burst bloat
    # These pairs are still only candidates — they need independent confirmation
    # across episodes before becoming active.
    ns_tag = rule_engine.dataset_tag or "arc-agi-3"
    co_new = co_registry.promote_to_rules(
        rule_engine,
        min_count=8,
        min_consistency=0.90,
        ns_tag=ns_tag,
        source_task=f"ep{episode_num:02d}",
        max_rules=20,
    )
    if co_new:
        ids = [r["id"] for r in co_new]
        log(f"  [CO-OCC] {len(co_new)} new co-occurrence rule(s): {ids}")
        for r in co_new:
            ep_log._write(
                f"  [CO-OCC] {r['id']} (candidate)\n"
                f"    IF:   {r['condition']}\n"
                f"    THEN: {r['action']}"
            )

    # -- Concept binding confidence summary -----------------------------------
    bindings = state_manager._data.get("concept_bindings") or {}
    conf_lines = []
    for k, v in sorted(
        ((k, v) for k, v in bindings.items() if isinstance(k, int)),
        key=lambda x: x[0],
    ):
        if isinstance(v, dict):
            conf_lines.append(
                f"    color{k} -> {v.get('role','?')} "
                f"(confidence={v.get('confidence',0):.0%}  "
                f"this-level={v.get('level_obs', v.get('observations',0))}obs  "
                f"lifetime={v.get('total_obs', v.get('observations',0))}obs)"
            )
        else:
            conf_lines.append(f"    color{k} -> {v} (confidence=unknown)")
    if conf_lines:
        log("  [CONCEPTS] bindings at episode end:")
        for line in conf_lines:
            log(line)
        ep_log._write("  [CONCEPTS] bindings at episode end:\n" + "\n".join(conf_lines))

    perf_report = rule_engine.format_performance_report()
    log(perf_report)
    ep_log._write(perf_report)

    ep_log.episode_end(
        state=final_state,
        levels=final_levels,
        steps=step_count,
        cycles=cycle_count,
        cost_usd=round(ct.cost_usd(), 4),
    )

    return meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_reasoning(text: str) -> str:
    """Pull the 'reasoning' field from MEDIATOR's JSON response."""
    import re
    for block in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text or ""):
        try:
            obj = json.loads(block)
            if isinstance(obj, dict) and "reasoning" in obj:
                return str(obj["reasoning"])
        except (json.JSONDecodeError, ValueError):
            pass
    return ""
