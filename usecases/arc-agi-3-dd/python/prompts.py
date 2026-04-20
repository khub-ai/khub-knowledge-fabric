"""INITIAL_ASSESSMENT prompt for ARC-AGI-3 DD v0.1.

The prompt is reply-only JSON, sections 1..4:
  1. ELEMENTS         — salient gameplay objects, closed function vocab
  2. SIMILAR_GROUPS   — same-glyph sets (rotation/reflection/scale/colour ok)
  3. INITIAL_STRATEGY — primary goal, first action, rationale, open questions
  4. PROBES           — DSL instructions the dumb harness can execute

Coordinate convention: [row_min, col_min, row_max, col_max], 0-indexed,
inclusive.  Frame is 64x64 for ls20.
"""
from __future__ import annotations


SYSTEM = """You are a visual reasoning subsystem for a symbolic game-playing engine.
You receive an initial frame of an ARC-AGI-3 game (small 2-D integer grid,
cells 0-15) and the available action labels.  You MUST reply with a single
JSON object matching the schema in the user message, and nothing else --
no prose outside the JSON, no code fences.

Your output is compared side-by-side with another model's output on the
same frame.  Honest low confidence is useful; fabricated high confidence
is harmful."""


USER_TEMPLATE = """You are shown the initial frame of ARC-AGI-3 game {game_id} level {level}.

FRAME (64x64 grid, palette 0-15):
{frame_text}

AVAILABLE_ACTIONS: {action_labels}
STATE: {state}
LEVELS_COMPLETED: {levels_completed} / {win_levels}

GAME_TAGS: {tags}
GAME_TITLE: {title}

QUESTION_TYPE: INITIAL_ASSESSMENT

Coordinate convention for every bbox below: [row_min, col_min, row_max, col_max],
rows top-to-bottom, columns left-to-right, both 0-indexed and inclusive.

Produce the four sections below as a single JSON object.  No prose outside JSON.

---------------------------------------------------------------------------
1. ELEMENTS -- objects likely to matter for gameplay.

   List every distinct visual element worth naming for a player: agent(s),
   targets, hazards, walls, collectibles, counters, portals, switches,
   doors, readouts, meaningful decor.

   EXCLUDE compositional fill: do NOT list each tile of a tiled background,
   each brick of a brick wall, each cell of a uniform floor.  Report the
   background/floor/wall ONCE as a single element (or not at all if it is
   pure fill with no gameplay role).

   For each element:
     id          integer, local to this response (1, 2, 3, ...)
     name        short descriptive name you invent (no palette numbers)
     bbox        [row_min, col_min, row_max, col_max]
     function    one of: agent, target, hazard, wall, collectible,
                 resource, counter, portal, switch, readout, decor, unknown
     rationale   one short sentence -- the visual cue for this function
     confidence  0.0..1.0

2. SIMILAR_GROUPS -- sets of elements that look like variants of each other.

   Group elements whose INTERNAL shape is the same pictogram, even if they
   differ by rotation, reflection, small scale change, or colour.  Each
   group must have >= 2 members.

   EXCLUDE compositional repetition: the identical tiles of a tiled
   background are NOT a group here.  Groups are meaningful same-kind
   elements (e.g. three keys in different colours; two enemies facing
   different directions).

   For each group:
     group_id          short local id ("G1", "G2", ...)
     member_ids        list of element ids from section 1
     similarity_axes   subset of: shape, rotation, reflection, scale, colour
                       ("shape" means the core glyph matches; the others
                        describe what VARIES across members)
     note              one short sentence on the shared visual concept

3. INITIAL_STRATEGY -- how to start playing.

     primary_goal      one sentence: what the player seems to be trying
                       to achieve on this level
     first_action      one of AVAILABLE_ACTIONS (by label, e.g. "ACTION2"),
                       or null if you cannot commit
     rationale         one or two sentences: why this action, what you
                       expect to observe
     open_questions    list of 1..3 short questions whose answers would
                       most change your strategy

4. PROBES -- specific experiments the dumb harness can run to reduce
   YOUR uncertainty about this frame.

   List 2..5 probes.  Each tests ONE hypothesis you hold with less than
   full confidence.  Use ONLY the closed DSL grammar below -- anything
   outside it will be rejected.

   INSTRUCTIONS (executed in order):
     DO <ACTION_LABEL>                              # execute one action
     DO_SEQ [<ACTION_LABEL>, <ACTION_LABEL>, ...]   # execute sequence
     REPEAT DO <ACTION_LABEL> <N>                   # 1 <= N <= 10
     RESET                                          # full reset

   OBSERVATIONS (run after the last instruction; result returned):
     REGION_DELTA [r0,c0,r1,c1]    # pixel-diff count in this bbox since
                                     probe start
     ELEMENT_MOVED <element_id>    # new bbox of this element, or null
                                     if vanished
     STATE                          # current GameState
     AVAILABLE_ACTIONS              # current action subset
     SCORE_DELTA                    # change in levels_completed

   <ACTION_LABEL> must be one of AVAILABLE_ACTIONS shown above.
   <element_id> must be an id from your ELEMENTS list.

   For each probe:
     probe_id       short local id ("P1", "P2", ...)
     hypothesis     one short sentence
     instructions   list of INSTRUCTION statements (strings)
     observe        list of OBSERVATION statements (strings)
     outcome_map    mapping: observation-outcome key -> conclusion.
                    2..4 entries.  Use short keys like
                    "element_3_bbox_changed", "state_became_GAME_OVER".

---------------------------------------------------------------------------
REPLY SCHEMA (valid JSON, nothing else):

{{
  "elements": [
    {{
      "id": 1,
      "name": "string",
      "bbox": [0, 0, 0, 0],
      "function": "agent|target|hazard|wall|collectible|resource|counter|portal|switch|readout|decor|unknown",
      "rationale": "string",
      "confidence": 0.0
    }}
  ],
  "similar_groups": [
    {{
      "group_id": "G1",
      "member_ids": [1, 2],
      "similarity_axes": ["shape"],
      "note": "string"
    }}
  ],
  "initial_strategy": {{
    "primary_goal": "string",
    "first_action": "ACTION2",
    "rationale": "string",
    "open_questions": ["string"]
  }},
  "probes": [
    {{
      "probe_id": "P1",
      "hypothesis": "string",
      "instructions": ["DO ACTION1"],
      "observe": ["ELEMENT_MOVED 3", "REGION_DELTA [0,0,63,63]"],
      "outcome_map": {{
        "element_3_bbox_changed": "element 3 is agent, ACTION1 moves it",
        "state_became_GAME_OVER": "ACTION1 is fatal from this state"
      }}
    }}
  ]
}}
"""


def build_user_message(
    *,
    frame_text:       str,
    action_labels:    list[str],
    state:            str,
    levels_completed: int,
    win_levels:       int,
    game_id:          str,
    title:            str,
    tags:             list[str],
    level:            int = 1,
) -> str:
    return USER_TEMPLATE.format(
        frame_text       = frame_text,
        action_labels    = action_labels,
        state            = state,
        levels_completed = levels_completed,
        win_levels       = win_levels,
        game_id          = game_id,
        title            = title,
        tags             = tags,
        level            = level,
    )


def format_frame_text(grid: list[list[int]]) -> str:
    """Format a 2-D grid as a python-literal list-of-lists on multiple
    lines -- one row per line so diffs / inspection are painless."""
    rows = [", ".join(f"{v:2d}" for v in row) for row in grid]
    return "[\n" + ",\n".join(f"  [{r}]" for r in rows) + "\n]"
