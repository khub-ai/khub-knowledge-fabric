"""
agents.py — ARC-AGI-3 agent runner functions.

Loads system prompts from the prompts/ directory, formats game observations
into structured prompts, and parses MEDIATOR action plans.

Generic LLM infrastructure lives in core/pipeline/agents.py and is imported
here. Domain-specific: OBSERVER (analyzes game frame) and MEDIATOR (plans actions).
"""

from __future__ import annotations
import sys
import json
import re
import time
from pathlib import Path
from typing import Any, Optional

_KF_ROOT = Path(__file__).resolve().parents[3]
if str(_KF_ROOT) not in sys.path:
    sys.path.insert(0, str(_KF_ROOT))

from core.pipeline.agents import (          # noqa: E402
    get_client,
    CostTracker,
    reset_cost_tracker,
    get_cost_tracker,
    DEFAULT_MODEL,
    DEFAULT_MAX_TOKENS,
    SHOW_PROMPTS,
    _print_prompt,
    call_agent as _core_call_agent,
)
import core.pipeline.agents as _core_agents
from object_tracker import (
    summarize_action_effects,
    summarize_current_objects,
    compute_trend_predictions,
    format_structural_context,
    color_name,
)

# ---------------------------------------------------------------------------
# Prompt files
# ---------------------------------------------------------------------------

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

PROMPT_FILES = {
    "OBSERVER": PROMPTS_DIR / "observer.md",
    "MEDIATOR": PROMPTS_DIR / "mediator.md",
}

_prompt_cache: dict[str, str] = {}


def load_prompt(agent_id: str) -> str:
    if agent_id not in _prompt_cache:
        path = PROMPT_FILES[agent_id]
        _prompt_cache[agent_id] = path.read_text(encoding="utf-8")
    return _prompt_cache[agent_id]


# Re-export for convenience
def call_agent(agent_id: str, user_message: Any, **kwargs):
    return _core_call_agent(agent_id, user_message, **kwargs)


# ---------------------------------------------------------------------------
# Frame formatting
# ---------------------------------------------------------------------------

# ARC-AGI color codes (same 10 colors used across v1/v2/v3)
_COLOR_CHARS = {
    0: ".", 1: "B", 2: "R", 3: "G", 4: "Y",
    5: "b", 6: "M", 7: "O", 8: "A", 9: "W",
    # Colors 10-19 rendered with distinct lowercase letters so the LLM can
    # see them clearly without us pre-assigning semantic meaning.
    10: "j", 11: "k", 12: "x", 13: "v", 14: "u",
    15: "t", 16: "s", 17: "q", 18: "p", 19: "n",
}


def _to_list_2d(frame: Any) -> list:
    """Convert a game frame (list, numpy array, or nested) to a plain list-of-lists 2D grid.

    Handles:
    - plain list[list[int]]       — returned as-is
    - numpy ndarray (2-D)         — converted via .tolist()
    - list[numpy ndarray]         — unwrap outer list, convert inner array
    - nested containers           — drilled recursively until 2D found
    """
    # Numpy array → convert in one shot
    if hasattr(frame, "tolist"):
        result = frame.tolist()
        if isinstance(result, list) and result and isinstance(result[0], list):
            return result
        return result if isinstance(result, list) else []

    if not isinstance(frame, list) or not frame:
        return []

    first = frame[0]

    # Outer wrapper contains a numpy array → convert it
    if hasattr(first, "tolist"):
        inner = first.tolist()
        if isinstance(inner, list) and inner and isinstance(inner[0], list):
            return inner
        return inner if isinstance(inner, list) else []

    # Already a flat list-of-lists
    if isinstance(first, list) and first and not isinstance(first[0], list):
        return frame

    # Recurse into first element
    inner = _to_list_2d(first)
    return inner if inner else frame


def frame_to_str(frame: Any) -> str:
    """Convert a game frame to a compact grid string (same format as ARC-AGI-2)."""
    grid = _to_list_2d(frame)
    if not grid:
        return "(empty frame)"
    lines = []
    for row in grid:
        if isinstance(row, list):
            lines.append(" ".join(_COLOR_CHARS.get(int(v), "?") for v in row))
        else:
            lines.append(str(row))
    return "\n".join(lines)


def frame_shape(frame: Any) -> str:
    grid = _to_list_2d(frame)
    if not grid:
        return "?"
    rows = len(grid)
    cols = len(grid[0]) if isinstance(grid[0], list) else 0
    return f"{rows}x{cols}"


# ---------------------------------------------------------------------------
# Observation field helpers
# ---------------------------------------------------------------------------

def _obs_field(obs: Any, name: str, default: Any = None) -> Any:
    if obs is None:
        return default
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


def obs_state_name(obs: Any) -> str:
    state = _obs_field(obs, "state")
    if state is None:
        return "UNKNOWN"
    if hasattr(state, "name"):
        return state.name
    return str(state)


def obs_levels_completed(obs: Any) -> int:
    return int(_obs_field(obs, "levels_completed", 0) or 0)


def obs_frame(obs: Any) -> Any:
    """Return the frame from an observation as a plain list-of-lists 2D grid."""
    raw = _obs_field(obs, "frame")
    if raw is None:
        return []
    return _to_list_2d(raw) if raw is not None else []


def format_action_space(actions: list[Any]) -> str:
    """Format available actions for prompt injection."""
    lines = []
    for a in actions:
        name = getattr(a, "name", str(a))
        is_complex = bool(getattr(a, "is_complex", lambda: False)())
        kind = 'complex -- requires {"x": int, "y": int}' if is_complex else "simple -- use {}"
        lines.append(f"  - {name}: {kind}")
    return "\n".join(lines) if lines else "  (none available)"


def format_action_history(history: list[dict], last_n: int = 12) -> str:
    """Format recent action history for prompt injection."""
    if not history:
        return "  (no actions taken yet)"
    recent = history[-last_n:]
    offset = len(history) - len(recent)
    lines = []
    for i, h in enumerate(recent):
        data_str = f" {h['data']}" if h.get("data") else ""
        diff = h.get("diff", -1)
        diff_str = f" diff={diff}" if diff >= 0 else ""
        # Flag abnormally large diffs — may indicate a CHANGER cell was visited
        changer_flag = " [*** LARGE DIFF — possible CHANGER visited ***]" if diff > 80 else ""
        lines.append(
            f"  Step {offset + i + 1}: "
            f"{h['action']}{data_str}"
            f" → levels={h['levels']} state={h['state']}{diff_str}{changer_flag}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OBSERVER
# ---------------------------------------------------------------------------

def _format_action_effects(action_effects: dict) -> str:
    """Format the accumulated action-effect table for prompt injection.

    Uses object-level movement summaries where available (richer signal),
    with pixel-level diff stats as a fallback.
    """
    if not action_effects:
        return "  (no action effects recorded yet -- all actions are unexplored)"
    # Use object_tracker's consensus summary (uses object_observations if present)
    return summarize_action_effects(action_effects)


_CONFIDENCE_MAP = {"high": 0.9, "medium": 0.6, "low": 0.3}


def parse_concept_bindings(observer_text: str) -> dict[int | str, Any]:
    """
    Extract concept_bindings from OBSERVER JSON output.

    Accepts two input formats:
      Simple:   {"concept_bindings": {"12": "player_piece"}}
      Rich:     {"concept_bindings": {"12": {"role": "player_piece",
                                             "confidence": "high",
                                             "label": "[GUESS]"}}}

    Returns a dict where integer keys → binding dicts with fields:
      role (str), confidence (float 0–1), observations (int, starts at 1).
    Non-integer keys (e.g. "wall_colors") are passed through unchanged.
    """
    bindings: dict[int | str, Any] = {}
    for block in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", observer_text or ""):
        try:
            obj = json.loads(block)
        except (json.JSONDecodeError, ValueError):
            continue
        raw = obj.get("concept_bindings", {})
        if isinstance(raw, dict):
            for k, v in raw.items():
                # Pass-through non-color keys (e.g. wall_colors list)
                try:
                    color_int = int(k)
                except (ValueError, TypeError):
                    bindings[k] = v
                    continue
                # Normalise value to rich format
                if isinstance(v, dict):
                    role = str(v.get("role", v.get("name", "")))
                    conf_raw = v.get("confidence", "medium")
                    label    = str(v.get("label", ""))
                else:
                    role     = str(v)
                    conf_raw = "medium"
                    label    = ""
                # [CONFIRMED] label boosts confidence; [GUESS] lowers it
                if "[CONFIRMED]" in label.upper():
                    conf_raw = "high"
                elif "[GUESS]" in label.upper():
                    conf_raw = "low"
                conf_float = _CONFIDENCE_MAP.get(
                    str(conf_raw).lower(),
                    float(conf_raw) if str(conf_raw).replace(".", "").isdigit() else 0.6,
                )
                bindings[color_int] = {
                    "role":         role,
                    "confidence":   conf_float,
                    "observations": 1,
                }
        break
    return bindings


async def run_observer(
    obs: Any,
    available_actions: list[Any],
    action_history: list[dict],
    rules_section: str = "",
    action_effects: Optional[dict] = None,
    concept_bindings: Optional[dict] = None,
    steps_remaining: int = 0,
    known_dynamic_colors: Optional[set] = None,
    explored_colors: Optional[set] = None,
    action_directions: Optional[dict] = None,
    contact_events: Optional[list] = None,
    verbose: bool = True,
) -> tuple[str, int]:
    """
    OBSERVER: analyze the current game frame and produce a structured observation.

    Returns (observation_text, duration_ms).
    """
    system_prompt = load_prompt("OBSERVER")

    frame = obs_frame(obs)
    grid_str = frame_to_str(frame)
    levels = obs_levels_completed(obs)
    state = obs_state_name(obs)
    shape = frame_shape(frame)
    actions_str = format_action_space(available_actions)
    history_str = format_action_history(action_history)
    effects_str = _format_action_effects(action_effects or {})

    # Flag any available simple actions that have never been tried
    tried_actions = set((action_effects or {}).keys())
    untried = [
        a.name for a in available_actions
        if not getattr(a, "is_complex", lambda: False)()
        and a.name not in tried_actions
    ]
    if untried:
        effects_str += (
            "\n\n  **UNTRIED ACTIONS (zero observations — must characterize before repeating known actions):**\n"
            + "\n".join(f"  - {a}: unknown — not yet called" for a in untried)
        )

    objects_str = summarize_current_objects(frame, concept_bindings)
    structural_str = format_structural_context(
        frame,
        concept_bindings=concept_bindings,
        known_dynamic_colors=known_dynamic_colors,
        explored_colors=explored_colors,
        action_directions=action_directions,
        contact_events=contact_events,
    )
    predictions = compute_trend_predictions(action_effects or {}, steps_remaining)
    predictions_str = (
        "\n".join(f"  {p}" for p in predictions)
        if predictions else "  (none yet — not enough data)"
    )

    # Format known concept bindings for the prompt.
    # concept_bindings uses two schemas:
    #   {color_int: "role_name"}   — color-keyed roles (player_piece, step_counter, ...)
    #   {"wall_colors": [c, ...]}  — concept-keyed list (wall colors for THIS game only)
    if concept_bindings:
        cb_lines = []
        for k, v in sorted(concept_bindings.items(), key=lambda x: str(x[0])):
            if k == "wall_colors":
                colors_str = ", ".join(
                    f"color{c}({color_name(c)})" for c in sorted(v)
                )
                cb_lines.append(
                    f"  wall (game-local) = [{colors_str}]  "
                    f"-- NOTE: wall colors differ between games"
                )
            else:
                try:
                    cb_lines.append(f"  color{k} ({color_name(k)}) = {v}")
                except Exception:
                    pass
        concepts_str = "\n".join(cb_lines) if cb_lines else "  (none identified yet)"
    else:
        concepts_str = "  (none identified yet)"

    user_message = (
        f"## Current game state\n\n"
        f"State: {state}\n"
        f"Levels completed: {levels}\n"
        f"Steps remaining this episode: {steps_remaining}\n\n"
        f"## Current frame ({shape})\n\n"
        f"{grid_str}\n\n"
        f"## Current objects (non-background)\n\n"
        f"{objects_str}\n\n"
        f"## Structural context (containment & spatial alignment — zero-cost)\n\n"
        f"{structural_str}\n\n"
        f"## Known concept bindings\n\n"
        f"{concepts_str}\n\n"
        f"## Trend predictions (zero-cost projections)\n\n"
        f"{predictions_str}\n\n"
        f"## Observed action effects (accumulated this episode)\n\n"
        f"{effects_str}\n\n"
        f"## Available actions\n\n"
        f"{actions_str}\n\n"
        f"## Recent action history\n\n"
        f"{history_str}\n"
    )

    if rules_section:
        user_message += f"\n## Matched knowledge rules\n\n{rules_section}\n"

    if verbose:
        print(f"  [OBSERVER] Analyzing frame ({shape}, levels={levels})...")

    t0 = time.time()
    text, ms = await _core_call_agent(
        "OBSERVER",
        user_message,
        system_prompt=system_prompt,
        max_tokens=1024,
    )

    if verbose:
        print(f"  [OBSERVER] Done in {ms}ms")

    return text, ms


# ---------------------------------------------------------------------------
# MEDIATOR
# ---------------------------------------------------------------------------

async def run_mediator(
    observer_text: str,
    rules_section: str = "",
    tools_section: str = "",
    action_history: Optional[list[dict]] = None,
    available_actions: Optional[list[Any]] = None,
    state_section: str = "",
    action_directions: Optional[dict] = None,
    structural_context_str: str = "",
    game_knowledge_section: str = "",
    verbose: bool = True,
) -> tuple[list[dict], str, int]:
    """
    MEDIATOR: produce a concrete action plan from OBSERVER analysis.

    Returns (action_plan, raw_mediator_text, duration_ms).
    action_plan is a list of {"action": str, "data": dict}.
    """
    system_prompt = load_prompt("MEDIATOR")

    user_message = ""

    # If action directions are already confirmed, inject a hard constraint at the
    # very top of the MEDIATOR input so it cannot be overridden by matched rules.
    if action_directions:
        _dir_labels = {
            (-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"
        }
        _parts = []
        for act in sorted(action_directions):
            dr, dc = action_directions[act]
            label = _dir_labels.get(
                (0 if dr == 0 else (-1 if dr < 0 else 1),
                 0 if dc == 0 else (-1 if dc < 0 else 1)), "?"
            )
            dist = abs(dr) if dr != 0 else abs(dc)
            _parts.append(f"{act}={label}{dist}")
        user_message += (
            "## ACTION DIRECTIONS — CONFIRMED FROM PRIOR EPISODES\n\n"
            "THESE ARE ALREADY KNOWN. DO NOT SPEND ANY STEPS RE-CHARACTERIZING THEM.\n"
            "Ignore any rules that suggest re-testing these actions.\n"
            f"  {',  '.join(_parts)}\n\n"
        )

    if game_knowledge_section:
        user_message += (
            "## Bootstrapped game knowledge\n\n"
            f"{game_knowledge_section}\n\n"
        )

    if structural_context_str:
        user_message += (
            "## Structural context (exploration manifest & BFS routes)\n\n"
            f"{structural_context_str}\n\n"
        )

    user_message += f"## OBSERVER analysis\n\n{observer_text}\n"

    if rules_section:
        user_message += f"\n## Prior knowledge rules\n\n{rules_section}\n"

    if tools_section:
        user_message += f"\n## Available action sequences (tools)\n\n{tools_section}\n"

    if available_actions is not None:
        user_message += f"\n## Available actions\n\n{format_action_space(available_actions)}\n"

    if action_history:
        history_str = format_action_history(action_history, last_n=6)
        user_message += f"\n## Full action history\n\n{history_str}\n"

    if state_section:
        user_message += f"\n## State and goals\n\n{state_section}\n"

    if verbose:
        print("  [MEDIATOR] Planning actions...")

    text, ms = await _core_call_agent(
        "MEDIATOR",
        user_message,
        system_prompt=system_prompt,
        max_tokens=2048,
    )

    action_plan = parse_action_plan(text)

    if verbose:
        plan_summary = ", ".join(
            f"{s['action']}({s['data'] or ''})" for s in action_plan[:6]
        )
        if len(action_plan) > 6:
            plan_summary += f" ... (+{len(action_plan) - 6} more)"
        print(f"  [MEDIATOR] Done in {ms}ms — {len(action_plan)} action(s): {plan_summary}")

    return action_plan, text, ms


# ---------------------------------------------------------------------------
# Action plan parsing
# ---------------------------------------------------------------------------

def parse_action_plan(text: str) -> list[dict]:
    """
    Parse MEDIATOR response to extract an ordered action plan.

    Accepts JSON in these forms:
      - {"action_plan": [...]}
      - {"actions": [...]}
      - [{"action": ..., "data": ...}, ...]
    JSON may appear inside a ```json ... ``` fence or inline.
    """
    # Try fenced JSON blocks first (most reliable)
    for block in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text):
        result = _try_parse_plan(block)
        if result is not None:
            return result

    # Try finding the largest JSON object/array in the text
    for match in re.finditer(r"(\{[\s\S]*\}|\[[\s\S]*\])", text):
        result = _try_parse_plan(match.group())
        if result is not None:
            return result

    return []


def _try_parse_plan(raw: str) -> Optional[list[dict]]:
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None

    if isinstance(obj, list):
        plan = _normalize_plan(obj)
        return plan if plan else None

    if isinstance(obj, dict):
        for key in ("action_plan", "actions", "steps", "plan"):
            if key in obj and isinstance(obj[key], list):
                plan = _normalize_plan(obj[key])
                if plan:
                    return plan

    return None


def _normalize_plan(raw: list) -> list[dict]:
    """Normalize action plan entries to {"action": str, "data": dict}."""
    result = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        action = (
            item.get("action")
            or item.get("name")
            or item.get("type")
            or item.get("action_name")
        )
        if not action:
            continue
        data = (
            item.get("data")
            or item.get("args")
            or item.get("params")
            or item.get("coordinates")
            or {}
        )
        if not isinstance(data, dict):
            data = {}
        result.append({"action": str(action), "data": data})
    return result
