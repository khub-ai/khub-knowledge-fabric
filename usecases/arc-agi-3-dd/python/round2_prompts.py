"""Round-2 REVISED_ASSESSMENT prompt for ARC-AGI-3 DD.

Round 2 feeds both models: (a) the frame, (b) their own Round-1 assessment,
(c) a facts-only grounding pack derived from executing TUTOR's Round-1
probes, and optionally (d) prior_knowledge injected by the operator
(simulates knowledge carried from prior game plays).

Models return a revised assessment in the same schema as Round 1, plus a
revision_notes field explaining what changed and why.

No success gates here — the Round-2 output is diagnostic material.
"""
from __future__ import annotations

import json


SYSTEM_ROUND2 = """You are a visual reasoning subsystem for a symbolic game-playing engine,
now on Round 2 of a dialogic-distillation loop.

You will be given:
  * the frame (same 64x64 int grid as Round 1, re-shown for grounding)
  * YOUR Round-1 assessment
  * a facts-only grounding pack: empirical observations from executing
    another model's Round-1 probes against the live environment
  * (optional) prior knowledge carried from previous game plays

Your job is to REVISE your Round-1 assessment using the evidence.  Return
the same JSON schema as Round 1, plus a `revision_notes` section at the
top explaining what you changed and why -- ideally citing the specific
observation that prompted each change.

Honest retraction is useful.  If an element you named in Round 1 turns
out to be wrong (bad coordinates, wrong function, etc.), say so in
revision_notes and correct or remove it.

Reply with a single JSON object and nothing else -- no prose, no fences."""


USER_TEMPLATE = """ROUND 2 -- REVISED_ASSESSMENT

You are continuing your analysis of ARC-AGI-3 game {game_id} level {level}.

FRAME (64x64 grid, palette 0-15, same as Round 1):
{frame_text}

AVAILABLE_ACTIONS: {action_labels}
STATE: {state}
LEVELS_COMPLETED: {levels_completed} / {win_levels}
GAME_TAGS: {tags}
GAME_TITLE: {title}

Coordinate convention for every bbox: [row_min, col_min, row_max, col_max],
rows top-to-bottom, columns left-to-right, both 0-indexed and inclusive.
The grid is 64x64.  All bboxes MUST fit in 0..63.

-----------------------------------------------------------------------
YOUR_ROUND1_ASSESSMENT (what YOU said last round):
{round1_json}

-----------------------------------------------------------------------
GROUNDING_PACK (empirical observations from executing Round-1 probes,
no prose / no hypotheses):
{grounding_json}
{prior_knowledge_block}
-----------------------------------------------------------------------
TASK:
Produce a revised assessment in the SAME JSON schema as Round 1
(elements, similar_groups, initial_strategy, probes), plus a leading
`revision_notes` section.  Match Round-1 element ids where possible so
the two rounds can be diffed; if you retract an element, drop it and
note the retraction.

revision_notes must be a list of objects of the form:
  {{ "change": "<what you changed>",
     "reason": "<the observation or evidence that prompted it>",
     "round1_ref": "<element id / probe id / strategy field you revised>" }}

If you have no changes, return an empty list and explain why in a single
"no_changes_reason" top-level string field.

You may also propose up to 5 NEW probes to run in a potential Round 3
(same PROBES DSL as Round 1: DO, DO_SEQ, REPEAT DO, RESET; REGION_DELTA,
ELEMENT_MOVED, STATE, AVAILABLE_ACTIONS, SCORE_DELTA).  These must target
hypotheses still open after Round 2.

-----------------------------------------------------------------------
REPLY SCHEMA (valid JSON, nothing else):

{{
  "revision_notes": [
    {{ "change": "string", "reason": "string", "round1_ref": "string" }}
  ],
  "no_changes_reason": "string",   // only if revision_notes is empty
  "elements": [
    {{ "id": 1, "name": "string", "bbox": [0,0,0,0],
       "function": "agent|target|hazard|wall|collectible|resource|counter|portal|switch|readout|decor|unknown",
       "rationale": "string", "confidence": 0.0 }}
  ],
  "similar_groups": [
    {{ "group_id": "G1", "member_ids": [1,2], "similarity_axes": ["shape"], "note": "string" }}
  ],
  "initial_strategy": {{
    "primary_goal": "string",
    "first_action": "ACTION2",
    "rationale": "string",
    "open_questions": ["string"]
  }},
  "probes": [
    {{ "probe_id": "P1", "hypothesis": "string",
       "instructions": ["DO ACTION1"],
       "observe": ["ELEMENT_MOVED 3"],
       "outcome_map": {{ "key": "conclusion" }} }}
  ]
}}
"""


PRIOR_KNOWLEDGE_TEMPLATE = """
-----------------------------------------------------------------------
PRIOR_KNOWLEDGE (carried from previous game plays -- treat as given truth
unless the evidence in the grounding pack directly contradicts it):

{prior_knowledge}
"""


def build_round2_user_message(
    *,
    frame_text:       str,
    action_labels:    list[str],
    state:            str,
    levels_completed: int,
    win_levels:       int,
    game_id:          str,
    title:            str,
    tags:             list[str],
    level:            int,
    round1_assessment: dict,
    grounding_pack:    dict,
    prior_knowledge:   str | None = None,
) -> str:
    prior_block = ""
    if prior_knowledge:
        prior_block = PRIOR_KNOWLEDGE_TEMPLATE.format(prior_knowledge=prior_knowledge)
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
        round1_json      = json.dumps(round1_assessment, indent=2),
        grounding_json   = json.dumps(grounding_pack,    indent=2),
        prior_knowledge_block = prior_block,
    )
