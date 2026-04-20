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

ROTATION TRACKER OBLIGATION (mandatory every turn — highest priority):
  rotation_tracker is harness-computed from authoritative game state.
  Read it FIRST every turn, before anything else.  Values are valid for
  the CURRENT level — the harness re-derives them after every auto-
  advance to a new sub-level.

  rotation_tracker fields:
    aligned             – True iff cur_rot_idx == goal_rot_idx for the
                          current level's first unfulfilled win marker.
    advances_remaining  – how many more cross visits to reach goal
                          (already accounts for any advances made).
                          0 means aligned.  Do NOT visit the cross
                          when this is 0.
    win_position        – [row, col] of the current level's open win
                          marker (CHANGES per level; never hardcode).
    cross_position      – [row, col] of the current level's rotation
                          trigger (CHANGES per level; never hardcode).
    cur_rot_idx, goal_rot_idx – rotation indices (0..3).
    level_index         – which sub-level you are currently on.
    rotation_advanced   – True iff THIS command advanced rotation.
    level_completed     – True iff THIS command completed a sub-level.

  DECISION TREE (apply in order):
    1. If aligned=True:
         Issue MOVE_TO rotation_tracker.win_position immediately.  Do
         NOT visit the cross again — each extra visit ROTATES PAST goal
         (advances_remaining would jump to 3, not 0).
    2. Else if advances_remaining > 0:
         Navigate to rotation_tracker.cross_position and enter its cell
         ONCE per needed advance.  Each entry advances cur_rot_idx by
         +1 mod 4.  For levels that need multiple advances, you must
         enter the cross cell separately for each advance (leave, come
         back).  Confirm advances_remaining decreased before leaving
         the cross area.

  After level_completed=True: the game has auto-advanced to the next
  sub-level; cursor_pos, win_position, cross_position, advances_remaining
  are ALL refreshed automatically.  Read them again and restart the
  decision tree for the new level.

  glyph_summary.pattern_match_report is a secondary visual corroboration
  only; rotation_tracker is the authoritative signal.

EVIDENCE PRECEDENCE:
  COMMAND_RESULT is ground truth for what just happened.
  CURSOR_POS in COMMAND_RESULT overrides your spatial model.
  harness_note: if non-null, the harness detected a coordinate error in your
    command.  Example: "HARNESS CORRECTION: target [row,col] is empty space.
    Nearest element: 'cross' actual center=[r,c], bbox=[...]."
    TREAT THIS AS GROUND TRUTH -- update WORKING_KNOWLEDGE immediately via
    revise_knowledge and use the corrected coordinates in future commands.
  element_overlaps: which named elements the avatar is currently ON.
    Use this to confirm you reached the right element -- it is authoritative.
  rotation_tracker: harness-computed rotation state (authoritative; read
    the ROTATION TRACKER OBLIGATION section for semantics).  Valid per
    current sub-level only — refreshes on level auto-advance.
  glyph_summary: dict with two parts:
    - "<element_name>_3x3": 3×3 palette grid for each small named element
    - pattern_match_report: list of {pair, match_deg, result} for every
      element pair.  match_deg=0 corroborates aligned=True but is secondary
      to rotation_tracker.
  target_analysis.nearby_elements: elements near your intended target with
    their ACTUAL bboxes and centers (supplementary detail behind harness_note).
  target_analysis.walls_hit: which moves were blocked and where.
    The harness records walls and routes future BFS around them automatically.
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
                      {"cursor_pos_after": [row, col]} or
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

PREV_LEVEL_NOTES (summary of the previous sub-level if one was completed
during this session; use as a prior only — per-level positions, rotation
counts, and element geometry may differ):
{prev_level_notes}

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
    prev_level_notes:  str = "",
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
        prev_level_notes  = prev_level_notes.strip() or "(no prior sub-level completed this session)",
        history_n         = len(recent_history[-5:]),
        recent_history    = "\n".join(hist_lines),
        command_result_json = json.dumps(command_result or {}, indent=2),
        frame_text        = frame_text,
    )


SYSTEM_POSTGAME = """You just finished a session of an ARC-AGI-3 game.  Your job now is to
produce an UPDATED cumulative knowledge note for this game that will be
loaded as GAME_KNOWLEDGE_BASE on every future session.

You will be given:
  PRIOR_KNOWLEDGE_BASE  -- the KB you (or prior runs) wrote last time.
                           Preserve anything still true; refine what the
                           current run contradicts; add what's new.
  HARNESS_OBSERVED_FACTS -- ground-truth events the harness recorded
                           (levels_completed deltas, per-turn level
                           completion events, action_effects, walls).
                           These are AUTHORITATIVE.  If your narrative
                           contradicts them, trust the facts and revise.
  OUTCOME + COMMAND_TRACE -- what you did and how it ended.

Write a single dense note (< 400 words, prose OK, no headers/fences):
  - confirmed action effects (dr, dc per action)
  - level structure: how many sub-levels exist (win_levels), what
    triggers levels_completed advancement, what varies between levels
    vs. stays constant (positions per level, rotations, etc.)
  - win condition and counter semantics (e.g. rotation index, goal
    state check, one-trigger vs. multi-trigger, state-dependent walls)
  - traps / false-lesson risks (cursor drift on glyph repaint, auto-
    advance between sub-levels, win_position being state-dependent)
  - confirmed element roles; coordinates ONLY if stable across levels
    (otherwise mark them as level-specific and describe how to find
    them on a new level)
  - 1-2 hypotheses still worth testing

Do NOT duplicate PRIOR_KNOWLEDGE_BASE verbatim -- build on it.  Do NOT
claim failure if HARNESS_OBSERVED_FACTS shows a level was completed.
Do NOT hardcode single-level coordinates as if they apply globally."""


POSTGAME_TEMPLATE = """POST-GAME KNOWLEDGE CAPTURE

GAME: {game_id}
OUTCOME: {outcome}   ({turns} turns, final_state={final_state}, levels={levels_completed}/{win_levels})

PRIOR_KNOWLEDGE_BASE (cumulative knowledge from earlier sessions — blank if first run):
{prior_kb}

HARNESS_OBSERVED_FACTS (ground truth — authoritative over your narrative):
  initial_levels_completed: {initial_lc}
  final_levels_completed:   {final_lc}
  win_levels_required:      {win_levels}
  level_completion_events:  {level_events_json}
  action_effects_confirmed: {action_effects_json}
  walls_learned:            {walls_json}

WORKING_KNOWLEDGE at end of play:
{working_knowledge}

COMMAND_TRACE (turn -> command -> brief):
{command_trace}

Write the updated cumulative knowledge note now (< 400 words, no fences,
no headers)."""


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
    prior_kb:                str = "",
    initial_lc:              int = 0,
    final_lc:                int = 0,
    level_completion_events: list[dict] | None = None,
    walls_learned:           list[tuple[int, int, str]] | None = None,
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
        prior_kb          = prior_kb.strip() if prior_kb else "(none — this is the first recorded session for this game)",
        initial_lc        = initial_lc,
        final_lc          = final_lc,
        level_events_json = json.dumps(level_completion_events or [], indent=2),
        action_effects_json = json.dumps(effects_display, indent=2),
        walls_json        = json.dumps([list(w) for w in (walls_learned or [])], indent=2),
        working_knowledge = working_knowledge,
        command_trace     = "\n".join(trace_lines) if trace_lines else "  (empty)",
    )
