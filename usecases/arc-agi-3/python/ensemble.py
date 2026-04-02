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
)


# ---------------------------------------------------------------------------
# Tunable limits (overridable by harness)
# ---------------------------------------------------------------------------

MAX_STEPS  = 200   # hard cap on total env.step() calls per episode
MAX_CYCLES = 40    # hard cap on OBSERVER-MEDIATOR cycles per episode


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
        """Extract and log the key hypotheses from the OBSERVER JSON output."""
        self._write("  [OBSERVER OUTPUT]")
        import re, json as _json
        for block in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", obs_text or ""):
            try:
                obj = _json.loads(block)
            except Exception:
                continue
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
            break  # only first JSON block

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
        "win_levels":         7,                        # LS20 has 7 levels; TODO: read from env
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
) -> list:
    """Match rules against current game observation. Returns list of RuleMatch."""
    active = rule_engine.active_task_rules()
    if not active:
        return []

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

    user_msg = rule_engine.build_match_prompt(task_text)
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

    obs = env.reset()
    step_count      = 0
    cycle_count     = 0
    action_history: list[dict] = []
    all_matched_ids: list[str] = []
    last_matched:   list       = []
    prev_frame:     list       = []

    # Running OBSERVER/MEDIATOR outputs for playlog (reset each cycle)
    _obs_text_current   = ""
    _med_plan_current:  list = []
    _med_reason_current = ""
    _plan_step_idx      = 0

    def log(msg: str) -> None:
        if verbose:
            print(msg)

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

    while step_count < max_steps and cycle_count < max_cycles:
        state_name = obs_state_name(obs)
        levels_now  = obs_levels_completed(obs)

        if state_name in ("WIN", "GAME_OVER"):
            break

        cycle_count  += 1
        levels_before = levels_now
        _plan_step_idx = 0

        log(f"\n-- Cycle {cycle_count}  "
            f"(steps {step_count}/{max_steps}, levels={levels_now}) --")

        # ------------------------------------------------------------------
        # Round 0: Rule matching
        # ------------------------------------------------------------------
        matched = await _match_rules(rule_engine, obs, action_history)
        last_matched = matched
        if matched:
            ids = [m.rule_id for m in matched]
            all_matched_ids.extend(ids)
            log(f"  [rules] matched: {ids}")

        ep_log.cycle_start(cycle_count, step_count, max_steps, levels_now,
                           [m.rule_id for m in matched])

        rules_section = rule_engine.format_fired_rules_for_prompt(matched)
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
        obs_text, _obs_ms = await run_observer(
            obs,
            available_actions,
            action_history,
            rules_section=rules_section,
            action_effects=state_manager._data.get("action_effects"),
            concept_bindings=concept_bindings,
            steps_remaining=max_steps - step_count,
            verbose=verbose,
        )
        _obs_text_current = obs_text
        ep_log.observer_output(obs_text)

        # Merge any new concept bindings the OBSERVER proposed
        new_bindings = parse_concept_bindings(obs_text)
        if new_bindings:
            existing = state_manager._data.get("concept_bindings") or {}
            merged = {**existing, **new_bindings}
            if merged != existing:
                state_manager._data["concept_bindings"] = merged
                log(f"  [concepts] bindings updated: {merged}")
                ep_log._write(f"  [CONCEPTS] bindings: {merged}")

        # ------------------------------------------------------------------
        # Round 2: MEDIATOR
        # ------------------------------------------------------------------
        action_plan, med_text, _med_ms = await run_mediator(
            obs_text,
            rules_section=rules_section,
            tools_section=tools_section,
            action_history=action_history,
            available_actions=available_actions,
            state_section=_gs,
            verbose=verbose,
        )
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
                        for entry in _glog:
                            ep_log.goal_event("update", "", entry)
                except Exception:
                    pass  # malformed goal_updates — ignore gracefully
            if "state_updates" in _updates:
                state_manager.apply_agent_updates(_updates["state_updates"])
                sets = _updates["state_updates"].get("set", {})
                log(f"  [state] updated: {list(sets.keys())}")
                ep_log.state_update(sets)

        # Extract candidate rules proposed by MEDIATOR (exploration + planning).
        # All arc-agi-3 rules start as "candidate" — they must be independently
        # confirmed (level advance or win) before being promoted to "active".
        if med_text:
            rule_changes = rule_engine.parse_mediator_rule_updates(
                med_text, task_id
            )
            if rule_changes:
                new_ids = []
                for r in rule_changes:
                    if r.get("lineage", {}).get("type") == "new":
                        # Downgrade active → candidate so it needs confirmation
                        r["status"] = "candidate"
                        rule_engine.save()
                    new_ids.append(r["id"])
                    ep_log.rule_proposed(
                        r["id"], r.get("status", "candidate"),
                        r.get("condition", ""), r.get("action", ""),
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
                cb_str = "  ".join(f"color{c}={n}" for c, n in sorted(concepts.items()))
                log(f"  [CONCEPTS] {cb_str}")

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

            frame_before = obs_frame(obs)
            obs = env.step(action_obj, data=data)
            step_count    += 1
            _plan_step_idx += 1

            levels_after = obs_levels_completed(obs)
            state_after  = obs_state_name(obs)

            action_history.append({
                "action": action_name,
                "data":   data,
                "levels": levels_after,
                "state":  state_after,
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
                )

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
                        new_walls = set(wall_colors.keys()) - known_walls
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

            data_str = f" {data}" if data else ""
            log(f"    step {step_count}: {action_name}{data_str}"
                f" -> {state_after} levels={levels_after}"
                f" diff={change['diff_count']}")
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
                log(f"  [ACTOR] Level advanced: {levels_before} -> {levels_after}")
                ep_log.level_advance(levels_before, levels_after)
                _update_level_goals(
                    goal_manager, env_id, levels_after, top_goal_id
                )
                log(f"  [goals] {goal_manager.format_for_prompt()}")
                # Promote candidate rules that fired this cycle — level advance
                # is the progress signal confirming they were useful
                _promote_fired_candidates(
                    rule_engine, matched, task_id, log
                )
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
