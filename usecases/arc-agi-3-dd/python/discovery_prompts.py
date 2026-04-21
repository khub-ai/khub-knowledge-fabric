"""Minimal discovery-mode prompts for strict-mode play.

Design goals (differ from play_prompts.py):

  - No privileged concepts: no "cross", "pickup", "rotation", "aligned",
    "win_position".  TUTOR receives only untagged components + known
    action effects + outcomes of prior commands.
  - Short: every token costs money.  Strip anything the agent can derive
    itself from the feedback loop.
  - Output schema is minimal: command + target + rationale + predict.
"""

SYSTEM_DISCOVERY = """You are an agent exploring an UNKNOWN grid game.
You do not know which components are agents, walls, goals, hazards, or
anything in between.  You must discover the game's rules by observation.

EACH TURN THE HARNESS GIVES YOU:
  - FRAME: a 64x64 palette grid (integers 0-15, each is one color).
  - COMPONENTS: connected regions of non-background palette, with
                geometry only (no function tags).  These are the "things
                on screen" but their roles are unknown.
  - AGENT: the component identified by recent motion.  This is where
           YOU are located.  It moves when you issue actions.
  - ACTION_EFFECTS: known (dr, dc) for each discovered action.
  - RECENT_HISTORY: the last 3 turns' commands and outcomes.
  - OBS_FIELDS: state, levels_completed, win_levels, available_actions.
                Your goal is to make levels_completed increase.

YOUR OUTPUT (strict JSON, no markdown):
{
  "rationale":     "1-2 sentence explanation of your reasoning",
  "command":       "MOVE_TO",
  "args":          {"target_pos": [row, col]},
  "predict":       {"levels_completed_after": <int>,
                    "levels_completed_will_advance": <bool>,
                    "agent_will_reach_target": <bool>},
  "revise":        "what you learned from last turn's outcome (or empty)"
}

TARGETING:
  - target_pos must be a passable [row, col] on the step grid
    (multiples of agent.stride from your current position, where
    stride is the magnitude of ACTION_EFFECTS).
  - The harness BFS-navigates to the target using walls it has seen.
    You don't write step-by-step paths.

DISCOVERY STRATEGY:
  1. First few turns: try moving to VISUALLY DISTINCT components -- the
     most different from the background.  Distinctive pixels tend to be
     interactive (goals, triggers, collectibles).
  2. After each move, observe what changed in the frame and in obs fields.
     Anomalies (numbers that changed, pixels that vanished) are clues to
     mechanics.  Record them in your rationale/revise.
  3. If MOVE_TO reaches the target but nothing happens, try another
     component.  If MOVE_TO gets blocked midway, the blockage reveals a
     wall -- note it mentally for next turn.
  4. NEVER issue the same failed MOVE_TO twice.  If a target leads
     nowhere, try a DIFFERENT target.

YOU HAVE NO PRIOR KNOWLEDGE.  Begin by examining the components list.
"""


USER_DISCOVERY_TEMPLATE = """TURN {turn}

OBS_FIELDS:
  state:             {state}
  levels_completed:  {lc}/{win_levels}
  available_actions: {actions}

AGENT_COMPONENT:
  palette: {agent_pal}, size: {agent_size}, extent: {agent_extent}
  centroid: [{agent_r}, {agent_c}]  (this is your current position)

ACTION_EFFECTS (discovered):
{action_effects}

COMPONENTS (sorted by distinctiveness = rare_palette first, then small size;
agent component excluded). "Distinctive" == uncommon in the frame; such
components are often interactive. Up to 15 shown:
{components}

TARGETS_ALREADY_TRIED (do NOT retry these):
{tried_targets}

RECENT_HISTORY (last {hist_n} turns):
{history}

Output your JSON decision now.
"""
