"""PLAY prompts — drive TUTOR through ls20 L1 using high-level harness commands.

TUTOR issues ONE command per call.  The harness executes it autonomously
(potentially many game steps) and reports back a bundled COMMAND_RESULT.
TUTOR is NOT called on every individual game step -- only once per command.
"""
from __future__ import annotations

import json


SYSTEM_PLAY = """You are a visual reasoning subsystem for a symbolic game-playing engine,
now in PLAY mode.  You have already analysed the initial frame (WORKING_KNOWLEDGE).
Each call you receive the current frame, ACTION_EFFECTS (confirmed action
mappings), CURSOR_POS (avatar position), and the result of your last command.

GOAL: Win the level in the fewest commands (and fewest game-steps) possible.
The game has a depleting MOVE BUDGET (~42 game steps per attempt).  Every
game step the budget decrements -- budget=0 means level reset + one life lost.

EVIDENCE PRECEDENCE:
  COMMAND_RESULT is ground truth for what just happened.
  CURSOR_POS in COMMAND_RESULT overrides your spatial model.
  WORKING_KNOWLEDGE is your prior theory -- update it via revise_knowledge
  when COMMAND_RESULT contradicts it.
  LESSONS_FROM_LAST_RUN (if in WORKING_KNOWLEDGE) are high-confidence priors
  -- skip re-testing anything already confirmed there.

=== AVAILABLE COMMANDS ===

PROBE_DIRECTIONS
  Execute each listed action once from current cursor position.  Harness
  measures (dr, dc) per action and returns a motion table.  Use this FIRST
  when ACTION_EFFECTS is empty or incomplete.
  Cost: 1 game-step per probed action.
  args: {}   (probes all AVAILABLE_ACTIONS)

MOVE_TO
  Navigate avatar to (row, col) using BFS over known ACTION_EFFECTS.
  Harness executes the full path autonomously.  Fails if ACTION_EFFECTS
  are unknown or target is unreachable.
  Cost: path length in game-steps.
  args: {"target_pos": [row, col]}

STAMP_AT
  Move avatar to (row, col) then fire `action` there once.  Use when you
  believe firing an action at a specific cell triggers a game event.
  Cost: path length + 1 game-steps.
  args: {"target_pos": [row, col], "action": "ACTION2"}

RAW_ACTION
  Execute a single low-level action.  Use only when you need one-off control
  (e.g. testing a hypothesis that doesn't fit the above commands, or when
  ACTION_EFFECTS are unknown and PROBE_DIRECTIONS is too expensive).
  Cost: 1 game-step.
  args: {"action": "ACTION1"}

RESET
  Reset the current level (costs a life, refills move budget).
  Use only when budget is nearly exhausted with no win in sight.
  args: {}

=== REPLY SCHEMA ===

Return a single JSON object:
  "command":          one of PROBE_DIRECTIONS | MOVE_TO | STAMP_AT |
                      RAW_ACTION | RESET
  "args":             dict matching the command's args spec above
  "rationale":        1-2 sentences: why this command, what you expect
  "predict":          short object with expected outcome, e.g.
                      {"cursor_pos_after": [30, 21]} or
                      {"levels_completed_after": 1}
  "revise_knowledge": string -- if COMMAND_RESULT contradicted your model,
                      state the correction.  Empty string if no revision.
  "done":             true | false -- set true only on WIN or GAME_OVER

Reply with strict JSON only.  No prose, no code fences."""


USER_TEMPLATE = """PLAY TURN {turn}

GAME: {game_id}  STATE: {state}  LEVELS: {levels_completed}/{win_levels}
BUDGET_REMAINING: ~{budget_remaining} game-steps

CURSOR_POS (harness-estimated avatar position, null if unknown):
{cursor_pos_json}

ACTION_EFFECTS (confirmed from observations; empty = not yet learned):
{action_effects_json}

WORKING_KNOWLEDGE (your current theory):
{working_knowledge}

RECENT_HISTORY (last {history_n} commands):
{recent_history}

LAST_COMMAND_RESULT (what the harness observed after your last command;
empty on turn 1):
{command_result_json}

CURRENT_FRAME (64x64 grid, palette 0-15):
{frame_text}

Issue your next command."""


def build_play_user_message(
    *,
    turn:              int,
    game_id:           str,
    state:             str,
    levels_completed:  int,
    win_levels:        int,
    budget_remaining:  int,
    cursor_pos:        tuple[int, int] | None,
    action_effects:    dict[str, tuple[int, int]],
    working_knowledge: str,
    recent_history:    list[dict],
    command_result:    dict | None,
    frame_text:        str,
) -> str:
    hist_lines = []
    for h in recent_history[-5:]:
        hist_lines.append(
            f"  turn {h.get('turn')}: {h.get('command')} {json.dumps(h.get('args',{}))} "
            f"-> state={h.get('state')} cursor={h.get('cursor_pos_after')} "
            f"steps={h.get('steps_taken',0)} budget_spent={h.get('budget_spent',0)}"
        )
    if not hist_lines:
        hist_lines = ["  (none)"]

    effects_display = {
        a: {"dr": dr, "dc": dc}
        for a, (dr, dc) in action_effects.items()
    } if action_effects else {}

    return USER_TEMPLATE.format(
        turn              = turn,
        game_id           = game_id,
        state             = state,
        levels_completed  = levels_completed,
        win_levels        = win_levels,
        budget_remaining  = budget_remaining,
        cursor_pos_json   = json.dumps(list(cursor_pos) if cursor_pos else None),
        action_effects_json = json.dumps(effects_display, indent=2),
        working_knowledge = working_knowledge,
        history_n         = len(recent_history[-5:]),
        recent_history    = "\n".join(hist_lines),
        command_result_json = json.dumps(command_result or {}, indent=2),
        frame_text        = frame_text,
    )


SYSTEM_POSTGAME = """You just finished a session of an ARC-AGI-3 game.  The outcome is in
OUTCOME.  Write a short, dense knowledge note for FUTURE plays of this
game (or its siblings in the ls-series).  Be concrete: name elements,
coordinates, action-effects, win-condition details, mistakes to avoid.

Prose is fine but keep it tight -- less than 300 words.  No fences."""


POSTGAME_TEMPLATE = """POST-GAME KNOWLEDGE CAPTURE

GAME: {game_id}
OUTCOME: {outcome}   ({turns} turns, final_state={final_state}, levels={levels_completed}/{win_levels})

ACTION_EFFECTS confirmed this run:
{action_effects_json}

WORKING_KNOWLEDGE at end of play:
{working_knowledge}

COMMAND_TRACE (turn -> command -> brief):
{command_trace}

Write a single knowledge note (< 300 words) for the next play session.
Cover:
  - confirmed action effects (dr, dc per action)
  - what each named element is / does
  - the win condition and any scored counter semantics
  - traps / anti-patterns hit this run
  - 1-2 speculative hypotheses still worth testing

No fences, no headers, just the note."""


def build_postgame_user_message(
    *,
    game_id:           str,
    outcome:           str,
    turns:             int,
    final_state:       str,
    levels_completed:  int,
    win_levels:        int,
    action_effects:    dict[str, tuple[int, int]],
    working_knowledge: str,
    command_trace:     list[dict],
) -> str:
    effects_display = {
        a: {"dr": dr, "dc": dc}
        for a, (dr, dc) in action_effects.items()
    }
    trace_lines = []
    for h in command_trace:
        brief = h.get("rationale") or ""
        brief = brief[:80] + ("..." if len(brief) > 80 else "")
        trace_lines.append(
            f"  {h.get('turn'):>3}: {h.get('command'):<18} "
            f"steps={str(h.get('steps_taken',0)):<4} -- {brief}"
        )
    return POSTGAME_TEMPLATE.format(
        game_id           = game_id,
        outcome           = outcome,
        turns             = turns,
        final_state       = final_state,
        levels_completed  = levels_completed,
        win_levels        = win_levels,
        action_effects_json = json.dumps(effects_display, indent=2),
        working_knowledge = working_knowledge,
        command_trace     = "\n".join(trace_lines) if trace_lines else "  (empty)",
    )
