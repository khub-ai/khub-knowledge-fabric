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

from core.knowledge.state import StateManager
from core.knowledge.goals import GoalManager
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
COMPETITION_MODE = False

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
) -> str:
    """Push the standard top-level goals at the start of an episode."""
    top = goal_manager.push(
        description=f"Win the {env_id} game (advance through all levels)",
        priority=1,
    )
    goal_manager.activate(top.id)

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

    return top.id  # caller may use to create further subgoals


def _update_level_goals(
    goal_manager: GoalManager,
    env_id: str,
    new_level: int,
    top_goal_id: str,
) -> None:
    """Resolve current level goal and push the next one when a level advances."""
    # Resolve any active goal whose description mentions the previous level
    prev_level = new_level - 1
    for g in goal_manager._goals:
        if (g.status == "active"
                and f"level {prev_level}" in g.description.lower()):
            goal_manager.resolve(
                g.id,
                result=f"Level {prev_level} completed at step transition",
            )

    # Push new level goal
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

    # Pre-filter: only send rules tagged with the current environment to the LLM.
    # This cuts the rule set from ~3000 to ~900 for a typical environment.
    if env_id:
        env_rules = [r for r in active if env_id in r.get("tags", [])]
        if env_rules:
            active = env_rules

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
# Main episode orchestrator
# ---------------------------------------------------------------------------

async def run_episode(
    env: Any,
    episode_num: int,
    env_id: str,
    rule_engine: Optional[RuleEngine] = None,
    tool_registry: Optional[ToolRegistry] = None,
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
    top_goal_id = _inject_initial_goals(goal_manager, env_id, episode_num)

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

    # Running OBSERVER/MEDIATOR outputs for playlog (reset each cycle)
    _obs_text_current   = ""
    _med_plan_current:  list = []
    _med_reason_current = ""
    _plan_step_idx      = 0
    _last_obs_frame: list = []  # frame at last OBSERVER call, for caching
    _is_puzzle: bool = False    # cached from previous cycle for early rule-skip
    _last_structural_str: str = ""   # Fix 1: cache to avoid resending when frame unchanged
    _last_structural_frame_sig: tuple = ()  # Fix 1: frame signature for cache invalidation

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
            tuple(sorted(concept_bindings.keys())) if concept_bindings else (),
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

        _t0 = time.time()

        # --- Puzzle-level OBSERVER bypass (SKIP_OBSERVER_FOR_PUZZLES) ---
        # On puzzle levels the structural context already contains groups,
        # focus/cursor, mismatch info, and reference slot mapping — everything
        # the MEDIATOR needs. We build a lightweight synthetic OBSERVER output
        # from locally-computed data, saving ~$0.15 per skipped call.
        if COMPETITION_MODE and (_frame_diff > 3 or not _obs_text_current):
            # --- COMPETITION MODE: bounded tool-loop OBSERVER ---
            # Replaces the single-shot OBSERVER call with an Anthropic
            # tool-use loop that has hard caps and access only to a small
            # custom toolset (no Bash/Read/Write). The agent can recall
            # learned concepts from the persistent ConceptRegistry and
            # record new ones. Per-game artifacts (LS20 BFS, TR87 slot-
            # strip detector) are excluded by design here.
            from agent_loop import run_observer_agent
            from object_tracker import summarize_current_objects
            _objects_summary = summarize_current_objects(_curr_frame, concept_bindings)
            # Minimal context: dimensions, level, budget, action names tried,
            # PRIOR HYPOTHESIS (so the agent refines instead of starting cold),
            # and a compact recent-action-outcomes trace (so the agent sees
            # wall hits and dead actions without spending a tool call). The
            # agent fetches the frame/objects/effects via tools as needed.
            _h = len(_curr_frame)
            _w = len(_curr_frame[0]) if _curr_frame else 0
            _ae = state_manager._data.get("action_effects") or {}
            _tried = ", ".join(sorted(_ae.keys())) or "(none yet)"
            _untried_actions = [
                a.name for a in available_actions
                if not getattr(a, "is_complex", lambda: False)() and a.name not in _ae
            ]
            # Prior hypothesis carried from MEDIATOR's last cycle (if any).
            _prior_hypo = state_manager._data.get("current_hypothesis") or "(none yet)"
            # Compact recent action outcomes trace (last 8 steps), with
            # explicit WALL markers when an action produced no movement.
            _recent: list[str] = []
            for _h_entry in (action_history[-8:] if action_history else []):
                _act = _h_entry.get("action", "?")
                _diff = _h_entry.get("diff", _h_entry.get("frame_diff"))
                _moved = _h_entry.get("moved")
                if _diff is not None and _diff <= 4:
                    _recent.append(f"{_act} -> WALL/no-movement (diff={_diff})")
                elif _moved:
                    _recent.append(f"{_act} -> {_moved}")
                else:
                    _recent.append(f"{_act} -> diff={_diff}")
            _recent_str = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(_recent)) or "  (no actions yet)"

            # Stuck detector: count consecutive recent steps with diff<=4.
            # If the last N steps all produced no meaningful change, the
            # current hypothesis is almost certainly wrong — force a reset.
            _stuck_n = 0
            for _h_entry in reversed(action_history[-12:] if action_history else []):
                _d = _h_entry.get("diff")
                if _d is not None and _d <= 4:
                    _stuck_n += 1
                else:
                    break
            _stuck_block = ""
            if _stuck_n >= 6:
                _stuck_block = (
                    f"## STAGNATION ALERT\n"
                    f"The last {_stuck_n} steps produced essentially no state change. "
                    f"Your prior hypothesis is almost certainly WRONG or the situation "
                    f"has changed. You MUST:\n"
                    f"  1. Discard the prior hypothesis and form a NEW one.\n"
                    f"  2. Try an action or strategy you have NOT recently tried.\n"
                    f"  3. Do NOT propose a 'submit' or 'puzzle solved' plan unless "
                    f"levels_completed has actually increased.\n\n"
                )
                # Also wipe MEDIATOR's stale hypothesis so the next cycle starts fresh.
                try:
                    state_manager._data["current_hypothesis"] = None
                    _prior_hypo = "(reset due to stagnation)"
                except Exception:
                    pass

            _user_ctx = (
                f"## Current game state\n"
                f"env_id: {env_id}\n"
                f"level: {_levels_after_reset + 1}\n"
                f"frame: {_h}x{_w}\n"
                f"steps_remaining: {max_steps - step_count}\n"
                f"actions_tried: {_tried}\n"
                f"actions_untried: {', '.join(_untried_actions) or '(none)'}\n\n"
                f"{_stuck_block}"
                f"## Prior hypothesis (from last cycle — refine, do not discard)\n"
                f"{_prior_hypo}\n\n"
                f"## Recent action outcomes (most recent last)\n"
                f"{_recent_str}\n\n"
                f"Use your tools to verify or refine the hypothesis. "
                f"If a recent action hit a wall, do NOT plan to repeat it. "
                f"Start with recall_concepts."
            )
            obs_text, _observer_ms, _agent_stats = await run_observer_agent(
                current_frame=_curr_frame,
                previous_frame=_last_obs_frame or None,
                action_effects=state_manager._data.get("action_effects") or {},
                objects_summary=_objects_summary,
                concept_bindings=concept_bindings,
                episode_meta={
                    "env_id": env_id,
                    "level": _levels_after_reset + 1,
                    "step": step_count,
                },
                user_context=_user_ctx,
                verbose=verbose,
            )
            _last_obs_frame = [row[:] for row in _curr_frame]
            log(
                f"  [OBSERVER] competition agent: turns={_agent_stats['turns']} "
                f"tool_calls={_agent_stats['tool_calls']} "
                f"recorded={len(_agent_stats['recorded'])} "
                f"confirmed={len(_agent_stats['confirmed'])}"
            )
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
        # Round 2: MEDIATOR
        # ------------------------------------------------------------------
        _t0 = time.time()
        # Fix 2: cap action history at 6 entries for MEDIATOR — the
        # `## Recent action outcomes` already sent to OBSERVER covers last 8
        # steps compactly; the full 15-entry history is redundant.
        _mediator_history = action_history[-6:] if action_history else []

        # Fix 3: in COMPETITION_MODE the OBSERVER already summarises the
        # frame; sending structural_str to MEDIATOR again is double-billing.
        _mediator_struct = "" if COMPETITION_MODE else structural_str

        action_plan, med_text, _med_ms = await run_mediator(
            obs_text,
            rules_section=rules_section,
            tools_section=tools_section,
            action_history=_mediator_history,
            available_actions=available_actions,
            state_section=_gs,
            action_directions=action_directions if action_directions else None,
            structural_context_str=_mediator_struct,
            verbose=verbose,
        )
        _mediator_ms = int((time.time() - _t0) * 1000)

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
        if med_text:
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

        for step_spec in action_plan:
            if step_count >= max_steps:
                log(f"  [ACTOR] Step limit ({max_steps}) reached, stopping")
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
            step_count    += 1
            _plan_step_idx += 1

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

            action_history.append({
                "action": action_name,
                "data":   data,
                "levels": levels_after,
                "state":  state_after,
                "diff":   _step_diff,
            })

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
                log(f"  [ACTOR] Level advanced: {levels_before} -> {levels_after}")
                ep_log.level_advance(levels_before, levels_after)
                if playlog_root is not None:
                    save_level_frame(obs_frame(obs), env_id, levels_after, playlog_root)
                    log(f"  [FRAME] Saved frame image for level {levels_after + 1}")
                # Reset per-level observation counts in concept bindings so
                # short-term (level) and long-term (lifetime) stats stay distinct
                bindings_now = state_manager._data.get("concept_bindings") or {}
                for ck, cv in bindings_now.items():
                    if isinstance(ck, int) and isinstance(cv, dict):
                        cv["level_obs"] = 0
                _update_level_goals(
                    goal_manager, env_id, levels_after, top_goal_id
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
