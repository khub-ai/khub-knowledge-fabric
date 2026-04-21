"""Minimal discovery-mode prompts for strict-mode play.

Design goals (differ from play_prompts.py):

  - No privileged concepts: no "cross", "pickup", "rotation", "aligned",
    "win_position".  TUTOR receives only untagged components + known
    action effects + outcomes of prior commands.
  - Primary coordinate system: CELLS, not pixels.  Agent stride = one
    cell.  All positions in the prompt are (cell_row, cell_col).
  - Multimodal: the frame is included as a rendered color PNG so TUTOR
    can identify shapes/icons visually.  Cell grid is drawn on the
    image; the agent's cell is highlighted.
  - Short: every token costs money.
"""

SYSTEM_DISCOVERY = """You are an agent exploring an UNKNOWN grid game.
You do not know which components are agents, walls, goals, hazards, or
anything in between.  You must DISCOVER the game's rules by observation.

COORDINATE SYSTEM
-----------------
The world is divided into CELLS.  The agent moves exactly one cell per
action, along orthogonal directions.  Everything you decide is in cell
coordinates:
  - cell (0, 0) is where the agent started this level (spawn).
  - cell (1, 0) is one cell DOWN from spawn; cell (-1, 0) is one cell UP.
  - cell (0, 1) is one cell RIGHT; cell (0, -1) is one cell LEFT.
  - Your "agent_cell" tells you where you are now.
  - ACTION_EFFECTS are given as cell deltas, e.g. ACTION1: (-1, 0) = UP.

WHAT YOU SEE EACH TURN
----------------------
  - An image of the current frame (rendered in color; cell grid drawn
    as faint gray lines; your cell outlined in green).  Use vision to
    identify distinctive icons -- crosses, rings, arrows, keyholes,
    numbers, etc.  Shape matters.  Color matters.
  - A COMPONENTS list -- each connected region of non-background palette
    with: palette, size, extent, its cell address, cells_covered.
    Components are sorted by DISTINCTIVENESS (rare palettes first).
  - AGENT cell and sprite fingerprint.
  - ACTION_EFFECTS in cell deltas.
  - TARGETS_ALREADY_TRIED: cells you have visited or attempted before
    without advancing `levels_completed`.  Accumulated across sessions.
  - RECENT_HISTORY: last 3 turns of commands + outcomes + CHANGES
    narration (what appeared/disappeared/moved in the frame).
  - OBS_FIELDS: state, levels_completed, win_levels, available_actions.
    Your goal is to make levels_completed increase.

OUTPUT (strict JSON, no markdown):
{
  "rationale":     "1-2 sentences on your reasoning (cite the image)",
  "hypotheses":    "what you think each key component is (short labels)",
  "command":       "MOVE_TO",
  "args":          {"target_cell": [cell_r, cell_c]},
  "predict":       {"levels_completed_will_advance": <bool>,
                    "agent_will_reach_target": <bool>,
                    "what_should_change":        "<brief>"},
  "revise":        "what you learned from last turn (or empty)"
}

CORE REASONING
--------------

  (1) LOOK AT THE IMAGE.  Icons are drawn with intent -- a cross, a
      ring, an arrow have meanings.  Map each visible icon to a cell.
      Candidates for interaction live at those cells.

  (2) EXPLORE BY HYPOTHESIS.  Each candidate has an unknown role; you
      find out by STEPPING ON IT and observing what changes.  A good
      turn reveals information regardless of whether it "wins".  Pick
      the move whose outcome (success OR failure) MOST NARROWS your
      hypotheses.

  (3) READ THE CHANGES BLOCK in history.  Per-turn CHANGES narrate
      components that appeared, disappeared, or moved.  These are
      mechanisms firing.  Consequences of your last action.  Example:
        - A component DISAPPEARED -> you consumed it.
        - A component APPEARED    -> you triggered a spawn/unlock.
        - A non-agent MOVED        -> the game reacted to you.
      These matter EVEN WHEN reached=False or lc did not advance.

  (4) INTERPRET TARGETS_ALREADY_TRIED:
      (a) REACHED but no lc advance: probably scenery or dormant.  Skip
          unless CHANGES evidence suggests it became active.
      (b) BLOCKED partway: either a path issue, or the target cell is
          GATED (game is enforcing a precondition).  Gated cells are
          often the GOAL -- you need to unlock them.
      (c) Failed 2+ times with no CHANGES in between: almost certainly
          gated.  Do NOT retry until you have triggered something else.

  (5) TRIGGER-THEN-GOAL LOOP is the classic grid-game pattern:
      a small/rare icon acts as a TRIGGER -> a GATE opens -> the GOAL
      becomes reachable.  If a distinctive small icon is visible and
      you haven't stepped on it, try it.  If CHANGES happen when you
      do, RETRY any previously-gated target next turn.

      MULTI-VISIT TRIGGERS: some triggers require MULTIPLE visits to
      fully unlock the goal.  If visiting a trigger once causes CHANGES
      but the goal remains gated, try visiting the SAME trigger AGAIN
      (step off to an adjacent cell, then step back).  Some levels
      require 2 or 3 visits to the same trigger.

      DISTINCTIVE TARGETS WORTH PERSISTING: if a visually distinctive
      icon (a cross, a ring, a switch-like shape) is in TARGETS_ALREADY
      _TRIED but you have NEVER actually reached it (only been blocked
      en route), the target CELL itself is usually still passable --
      the PATH is the problem.  Reach it by approaching from a
      DIFFERENT neighboring cell than before.  You can navigate to an
      adjacent passable cell FIRST, then target the distinctive cell
      from there.

      PICKUPS: some distinctive icons are pickups -- stepping on them
      typically consumes the icon (it DISAPPEARS in the CHANGES list)
      and grants a resource (e.g. refills a budget counter).  Watch
      for counters/bars on the image whose size or color changes as
      you move; those are likely game-state indicators.

  (6) If a distinctive target exists but every path to it is walled,
      target the NEAREST unexplored passable cell first -- you can
      try the distinctive target later from a better position.

  (7) Do not repeat a cell in TARGETS_ALREADY_TRIED unless something
      observably changed since the last attempt.

BUDGET
------
Each action decrements some game-budget counter.  If it reaches zero,
the level resets and you lose a life.  If you see something that looks
like a counter or meter in the image, watch it change across turns --
it may be the game's move budget.  Visiting distinctive icons you
haven't stepped on may consume them and restore the counter (pickups);
this is a useful hypothesis to test.
"""


USER_DISCOVERY_TEMPLATE = """TURN {turn}

OBS_FIELDS:
  state:             {state}
  levels_completed:  {lc}/{win_levels}
  available_actions: {actions}

CELL SYSTEM:
  cell_size (pixels): {cell_size}
  agent_cell (YOUR POSITION): {agent_cell}
  agent sprite fingerprint:  palette={agent_pal} size={agent_size} extent={agent_extent}

ACTION_EFFECTS (cell deltas):
{action_effects}

COMPONENTS (sorted by distinctiveness: rare palettes first; agent excluded;
each given as cell coordinates. "covers=[(cr,cc),...]" means the sprite
spans multiple cells):
{components}

TARGETS_ALREADY_TRIED (cell coordinates; see CORE REASONING rule 4):
{tried_targets}

PROBE_RESULTS (harness auto-probed the top distinctive components for
you; each entry shows what happened when the agent tried to reach it):
{probe_results}

PRIOR_HYPOTHESES (accumulated across earlier sessions -- your OWN past
guesses about component roles.  Use them, challenge them, refine them.):
{prior_hypotheses}

RECENT_HISTORY (last {hist_n} turns; CHANGES narrate mechanisms firing):
{history}

The CURRENT_FRAME is attached as an image.
  - Your agent cell is outlined in bright green and labeled "YOU".
  - Faint gray lines mark cell boundaries.
  - The TOP-8 most distinctive non-agent components are marked with a
    yellow/cyan ring at their cell center and labeled "#<id>".  Use these
    labels to match the image to the textual COMPONENTS list exactly --
    "#17" on the image is component id=17 in the list, at the cell noted.

Output your JSON decision now.
"""
