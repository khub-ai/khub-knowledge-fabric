# ARC-AGI-3 Solver — Design Decisions

> **Status**: working document, not for publication yet.  
> **Private companion**: `.private/DESIGN_PRIVATE.md` (gitignored) — LLM architecture, air-gap strategy.

## Overview

The ARC-AGI-3 ensemble is a specialization of the Knowledge Fabric (KF) pipeline for
interactive game environments. Unlike ARC-AGI-2 (static input→output puzzles), ARC-AGI-3
tasks are sequential decision problems: the agent must explore an unknown game, learn its
mechanics, and navigate to a win condition purely through action and observation.

**Entry point:** `usecases/arc-agi-3/python/harness.py`
**Python runtime:** Windows Store Python 3.12 (arc_agi SDK installed there, not conda)
**Default run:** `python harness.py --env ls20 --episodes 1 --max-steps 10`

---

## File layout

```
usecases/arc-agi-3/
  python/
    harness.py          — CLI: --env, --episodes, --max-steps, --max-cycles, --playlog, ...
    ensemble.py         — Episode orchestrator, EpisodeLogger, wall detection, urgency goals
    agents.py           — OBSERVER/MEDIATOR runners, concept binding parsing, prediction inject
    object_tracker.py   — Zero-cost visual analysis (all object-level functions)
    rules.py            — Shim: sets DEFAULT_PATH to local rules.json, re-exports RuleEngine
    tools.py            — Shim: sets DEFAULT_PATH to local tools.json, re-exports ToolRegistry
    rules.json          — Accumulated rules (gitignored — local only)
    tools.json          — Accumulated tools (gitignored — local only)
    playlogs/           — Per-run output: step JSONs + episode_NN.log (gitignored)
  prompts/
    observer.md         — OBSERVER system prompt
    mediator.md         — MEDIATOR system prompt
  DESIGN.md             — This file (gitignored — not for publication yet)
```

---

## Episode structure

```
env.reset()
inject_initial_goals()        # "win game", "complete level 1", "understand mechanics"

while steps < max_steps and cycles < max_cycles and state != WIN/GAME_OVER:

    Round 0: Rule matching
        RuleEngine retrieves candidate+active rules matching current state

    Round 1: OBSERVER
        Input:  frame (64×64 grid), current objects, concept_bindings, trend predictions,
                action effects table, action history, matched rules
        Output: JSON with level_description, visual_observations, action_characterizations,
                identified_objects, concept_bindings, hypothesized_goal, reasoning
                All inferences labeled [GUESS] or [CONFIRMED]

    Round 2: MEDIATOR
        Input:  OBSERVER analysis, rules, goals+state, action history
        Output: JSON with action_plan, rule_updates, goal_updates, state_updates, reasoning

    Parse updates:
        - Merge concept_bindings from OBSERVER into state
        - Apply goal_updates and state_updates from MEDIATOR
        - Add rule_updates (all forced to "candidate" status)

    Round 3: ACTOR — for each action in plan:
        env.step(action)
        accumulate_action_effect()   pixel-level + object-level diffs with raw before/after
        detect_wall_contacts()       if piece didn't move but normally does → wall colors
        compute_trend_predictions()  → push urgency goals if [URGENT]
        write_step_log()             JSON playlog file
        log to episode_NN.log

    if level advance:
        update_level_goals()
        promote_fired_candidate_rules()
        break to re-observe

Post-episode:
    Record rule success/failure based on won/not
    auto_deprecate(min_candidate_fired=5)
    Write episode summary to log
```

---

## Core principle: observation-only learning

**The agent must learn entirely from what it can observe through the standard API.**

Legitimate inputs:
- `obs.frame` — the 64×64 pixel grid
- `obs.levels_completed` — level advance signal
- `obs.state` — NOT_FINISHED / GAME_OVER / WIN
- The action space (names, simple vs complex)

Prohibited shortcuts:
- Reading game source files (`.py`)
- Introspecting `env._game`, `env._game.current_level`, sprite tags, or internal positions
- Any knowledge of win conditions, object roles, or mechanics not derived from observation

**Why this matters:** ARC-AGI-3 has 25 games. Source-inspection BFS solvers are
single-game hacks — they cannot generalize and they actively prevent building the
observation-based learning mechanisms that are the actual challenge. The ensemble
(OBSERVER → rules → MEDIATOR → action) is the right architecture; improving it is the
right unit of progress.

**Implication for LS20:** The `_KNOWN_SUBPLANS` and BFS scripts were useful for
infrastructure testing, but represent a shortcut that would not exist in competition.
Future games must be approached purely through the ensemble.

---

## Design decisions

### D1 — Rule lifecycle differs from ARC-AGI-2

All ARC-AGI-3 rules start as `candidate`, never `active`, regardless of `add_rule()` defaults.
After `parse_mediator_rule_updates()`, any rule with `lineage.type == "new"` is immediately
forced to `"candidate"` before saving.

**Why:** ARC-AGI-2 rules are verified by an executor against demos — passing all demos
justifies `active`. ARC-AGI-3 rules are exploration hypotheses — they need independent
confirmation across multiple episodes before being trusted.

**Promotion:** Candidate → active only when a level advances during a cycle where the rule fired.

**Auto-deprecation threshold:** `min_candidate_fired=5` (vs default 1 for ARC-AGI-2).
The `min_candidate_fired` parameter was added to `core/knowledge/rules.py` for this.

---

### D2 — Extended colors are not pre-labeled

Colors 0–9 are standard ARC-AGI palette. Colors 10+ are game-specific extensions.
They are rendered as distinct lowercase letters (`j`=10, `k`=11, `x`=12, ...) so the LLM
can see them clearly, but no semantic meaning is hardcoded.

**Why:** Hardcoding "color12 = cursor" breaks the moment a different game uses color12
for something else, or the same game changes. Roles must be discovered through exploration
and stored in concept_bindings.

**What is allowed:** Rendering distinction (different char per color value). Not allowed:
pre-assigning semantic names to specific color integers in any prompt or constant.

---

### D3 — Concept bindings schema (two-level, game-local)

`state_manager._data["concept_bindings"]` uses two schemas:

1. `{color_int: role_name}` — color-keyed, proposed by OBSERVER per episode
   Example: `{12: "player_piece", 11: "step_counter"}`

2. `{"wall_colors": [c, ...]}` — concept-keyed list, populated by wall detection
   Example: `{"wall_colors": [4, 5]}`

**Critical invariant:** ALL concept_bindings are game-local and episode-local.
StateManager is created fresh each episode — nothing persists across episodes by design.
The same color may mean different things in different games.

**Why two schemas:** The wall concept travels across games (the concept "wall" is stable)
but the color instantiation does not. Storing it as `{"wall_colors": [...]}` separates the
concept name from the color, whereas `{4: "wall"}` would conflate them.

**Role name vocabulary:** OBSERVER must use generic names: `player_piece`, `step_counter`,
`goal_region`, `reference_pattern`, `wall`. Not game-specific names.

---

### D4 — Attribute change records store raw numeric before/after

Each `attribute_changes` entry in `object_observations` includes:
```python
{
  "color": int,
  "changed": ["size", "width"],      # which attributes changed
  "summary": "azure: size 12->18",   # human-readable
  "before": {"size": 12, "width": 4},  # raw numbers ← key addition
  "after":  {"size": 18, "width": 6},
}
```

**Why:** `compute_trend_predictions` must work from pure data, not string parsing.
Storing raw numbers means any future attribute added to `ObjectRecord` is automatically
eligible for trend analysis with no changes to the prediction code.

---

### D5 — Trend prediction is fully data-driven

`compute_trend_predictions(action_effects, steps_remaining)` iterates over every
`(color, attribute)` pair in the accumulated data. No hardcoded colors, no hardcoded
attribute names. Works for size, width, height, and any future attribute.

Two trend classes detected:
1. **Attribute trend:** consistent increase or decrease → predicts depletion or unbounded growth
2. **Position drift:** consistent directional movement → predicts boundary collision

Direction reversals (oscillation) are excluded — only monotonic drift triggers.
`[URGENT]` is added when predicted depletion is within `steps_remaining`.

**Urgency goal push:** When any prediction is `[URGENT]`, an urgency goal is automatically
pushed to GoalManager (priority 1, deduplicated by description prefix).

---

### D6 — Wall detection is zero-cost and color-agnostic

When a step produces no movement on an object that has moved before under the same action:
1. `infer_typical_direction()` retrieves the most common move direction for this action
2. `detect_wall_contacts()` scans cells immediately beyond the object's bounding box in
   that direction and counts colors found there
3. Those colors are added to `concept_bindings["wall_colors"]` (game-local)

**Why color-agnostic:** Wall color varies between games. The concept "wall" is stable;
the color is not. We discover it per-game rather than assuming any color is always a wall.

---

### D7 — Episode logger captures all reasoning explicitly

`EpisodeLogger` writes `playlogs/episode_NN.log` incrementally. Every significant event
is captured: cycle starts, matched rules, OBSERVER output (guesses labeled), MEDIATOR
reasoning and plan, rule proposals (condition + then), goal events, state key-value changes,
per-step action outcomes with object diffs, wall contacts, concept binding updates, urgency
goals, auto-deprecation events, episode summary.

**Key convention:** OBSERVER labels all inferences as `[GUESS]` or `[CONFIRMED]`.
`[CONFIRMED]` = directly supported by action effects table or prior rules.
`[GUESS]` = inferred from visual structure alone.
This makes the trace auditable — a developer can quickly identify what the system measured
vs. what it assumed.

---

### D8 — Object tracker design principles

**`detect_objects`:** BFS connected-component per color. Objects ≥25% of total cells
are `is_background=True`. `ObjectRecord` has: color, size, centroid, bbox, width, height,
orientation (horizontal/vertical/square), aspect_ratio.

**`diff_objects`:** Same-color greedy nearest-neighbour, match_radius=10. Produces ObjectDiff
with moved, appeared, disappeared, stationary, attribute_changes. Attribute changes tracked
for: size, width, height, orientation.

**Future extension point:** `match_cross_color=False` parameter for color-changing objects —
deferred until evidence of this in any game.

---

## Dynamic Discovery — Design Principles

### P1 — No hardcoded game knowledge

Everything in `game_knowledge.json` — `player_colors`, `walkable_colors`, `step_size`,
`action_map` — is an **initial hypothesis**, not a fact. These values may be wrong for a
different game (or even a slightly altered version of the same game). They serve only as
starting guesses until the system has gathered enough observations to replace them.

**Implementation:** `GameHypothesis` dataclass holds `prior_*` (from game_knowledge.json)
and `obs_*` (filled by inference functions at runtime). `effective_*` properties return
the observed value when available, falling back to the prior. All downstream code
uses `effective_*`, never the raw priors.

**Corollary:** Any hardcoded filter, threshold, or heuristic that assumes specific
properties of a particular game is a design violation. If the system relies on
"max_sprite_pixels=12" or "ignore the most common color" or "HUD is at y≥58", it
will break on a different game that doesn't match those assumptions.

---

### P2 — The OBSERVER identifies objects, not the Python code

The OBSERVER (LLM) looks at the game frame with human-like vision and common sense.
It identifies what objects exist, what they look like, and what role they may play (toggle,
changer, wall, player, status bar, etc.). It writes these identifications to
`concept_bindings`. These are to be treated as initial hypotheses, since they could be wrong.

The Python frame analysis code (`find_sprite_cells`, `build_level_model`) only does
**position mapping**: given a color that the OBSERVER says is interesting, where does
that color appear on the game grid? The Python code may override or
second-guess the OBSERVER's object identification with sufficient evidence.

**Division of labor:**

| Responsibility                        | Who does it       |
|---------------------------------------|-------------------|
| "The yellow ring is a step-counter"     | OBSERVER (LLM)    |
| "The yellow rings appears near cells (19,30) and (34,15)" | Python frame scan |
| "Those are ring_positions in the model"| `build_level_model` (joins the above) |
| "The yellow bar at the bottom is the HUD, not a ring" | OBSERVER (LLM) |

Python code may not
filter, exclude, or re-classify candidates
based on heuristics (dominant color frequency, pixel count thresholds, screen
region). These are the MEDIATOR and OBSERVER's job.

---

### P3 — Object discovery is behavioral, not static

Objects are discovered by **what happens when you interact with them**, not by
how they look in a single frame.

| Observation                                        | Classification    |
|----------------------------------------------------|-------------------|
| Large pixel diff at position, no level advance     | State changer (RC, color-changer, etc.) |
| Level advance while standing at position           | Win gate           |
| Object disappears from frame after being visited   | Consumable (ring)  |
| Step counter bar grows after visiting position     | Step-counter reset |
| Player didn't move despite taking an action        | Wall / blocked     |

A single frame can identify *candidates* (non-floor, non-player pixels), but
classification requires observing behavior over time. The system should never
assume an object's role from its color or appearance alone — it must confirm
through interaction.

---

### P4 — Continuous frame comparison, not one-shot initial scan

The current frame should always be compared to the previous frame — not just to
the level's initial frame. Things can appear, change, and disappear at any time
during gameplay (animations, moving platforms, consumables, state changes).

The system should detect:
- **Appeared:** pixels/objects present now but absent in the previous frame
- **Disappeared:** pixels/objects absent now but present in the previous frame
- **Changed:** same position, different color or shape

This continuous diff model generalizes beyond the current "compare initial vs
current to find consumed rings" approach, which misses objects that appear
mid-level or change state dynamically.

---

### P5 — Player state tracking must be adaptive

The player's visual appearance (color, shape, rotation) changes during gameplay
as the player visits state-changers. The system must track these changes and
update its internal model accordingly — it cannot assume the player always looks
the same as it did at level start.

**Current implementation:** `_tracked_player_colors` starts from the
game_knowledge prior and updates whenever `find_player_position` fails but a
frame diff reveals a new moving cluster. Resets to prior on level advance.

**General principle:** Any game property that can change during play (player
color, player shape, walkable set, action semantics) must be tracked as a
mutable variable, not stored as a constant.

---

### P6 — Inference functions are independent and fallible

Each inference function (`infer_step_size`, `infer_action_directions`,
`infer_walkable_from_visits`, `infer_player_colors_from_diff`) operates
independently. Each can fail (return `None`) without affecting the others.
When an inference fails, the system falls back to the prior hypothesis
from game_knowledge.json.

This means the system degrades gracefully: on the first step with no history,
all inferences fail and the system runs entirely on priors. As history
accumulates, inferences override priors one by one. There is no single
point of failure.

Hard-coded inference functions should be considered as temporary measures, since later we would want to allow such functions to be learned as well. 

---

### P7 — discovered_knowledge.json persists cross-episode facts

Facts confirmed through successful gameplay (e.g., "level 1 requires 2 RC
visits", "color 3 = rotation_changer") persisted to
`discovered_knowledge.json` so future episodes don't have to re-discover them.

The matching with previous facts must be fuzzy, so that the persisted facts can be generalized appropriately to cover similar (but not exactly matched) situations. Generalized and validated "facts" are also registered in the `discovered_knowledge.json`. There should be suitable triggering mechanism not only to allow fuzzy matching, but also support efficient triggering of very large knowledge store.

This is complementary to concept_bindings (which are episode-local and proposed
by the OBSERVER). Discovered knowledge is **confirmed** — it came from a
successful level completion, not from unconfirmed hypotheses.

---

## LS20 game mechanics (discovered empirically, not assumed)

| Element | Color | Value | Behavior |
|---------|-------|-------|----------|
| Cursor | `x` | 12 | Moves with ACTION1/2/3/4. Starts at centroid ~(46,36). 5w×2h. |
| White component | `W` | 9 | Moves with cursor as a unit — same logical object. 5w×3h. |
| Step counter | `k` | 11 | Shrinks 2 cells per action (any direction). Starts at 84. Turns green. |
| Play field | `G` | 3 | Large green background. Static unless piece moves through it. |
| Outer area | `Y` | 4 | Yellow. Blocks upward cursor movement. Wall candidate. |
| Borders | `b` | 5 | Grey. Also blocks upward movement. Wall candidate. |

**Action map:**
- ACTION1: cursor up 5 rows
- ACTION2: cursor down 5 rows
- ACTION3: cursor left 5 cols (produces diff=2 when at left wall — piece doesn't move)
- ACTION4: cursor right 5 cols

**Win condition:** Unknown. No level advance observed yet. Hypotheses: navigate cursor
to specific position, align with white W reference pattern in bordered box.

**Step budget:** Starting at 84 cells, losing 2/step → 42 actions before step counter
depletes. What happens at depletion is unknown (likely GAME_OVER).

---

## StateStore — Unified State and Relation Representation

### Motivation

The current system hard-codes game-specific concepts (player, RC, win_gate, ring) in
Python data structures that only work for ls20-style maze navigation. ARC-AGI-3 has
25 games spanning radically different mechanics:

| Game | Mechanic |
|------|----------|
| ls20 | Grid navigation, maze rotation via state-changers, shape/color/rotation matching |
| tr87 | Pattern-matching: rotate shapes until rule-pairs are satisfied |
| ft09 | Constraint satisfaction: click cells to cycle colors, satisfy adjacency constraints |
| r11l | Drag-and-drop: select pieces, move to goals, avoid obstacles, collect paint |
| wa30 | Push-coupling: grab objects, push them to goals, 4-way BFS pathfinding |
| dc22 | Hybrid keyboard+click: bridge attachment, pressure plates, pixel collision |
| lp85 | Cyclic permutation: click buttons to rotate tile sequences, dual-goal matching |
| sp80 | Two-phase: arrange ramps/buckets, then simulate particle flow with splitting |

A general schema must handle all of these without mentioning any game by name.

---

### Design principle: everything is a fact in a typed store

The `StateStore` holds three kinds of facts:

1. **Attribute facts** — a property of one entity (an object, the world, progress)
2. **Relation facts** — a relationship between two or more entities
3. **Delta events** — a change to any fact, emitted on every write

Rules read facts, match deltas, and write new facts. The planner reads facts to
decide actions. No Python code should hard-code game knowledge that could be
expressed as a fact + rule.

---

### Object identity

Objects are assigned stable integer IDs when first detected (via connected-component
analysis or OBSERVER identification). An ID persists within a level; on level advance,
IDs may be reassigned. The only built-in property of an object is its ID.

```python
@dataclass
class StateFact:
    value:      Any           # the fact's value — any Python type
    confidence: float         # 0.0–1.0
    source:     str           # "prior" | "inferred" | "observed" | "rule"
    scope:      str           # "step" | "level" | "episode" | "game"
    step_index: int           # when last written
    evidence:   int           # how many observations support this
```

---

### Attribute fact key schema

Keys are tuples. The first element is the namespace.

**World namespace — game physics, inferred from interaction:**
```
("world", "step_size")              → int         # e.g. 5
("world", "action_map")            → dict        # {"UP": "ACTION1", ...}
("world", "walkable_colors")       → set[int]    # {3, 5}
("world", "wall_colors")           → set[int]    # {4}
("world", "grid_offset")           → (int, int)  # (col%step, row%step)
("world", "playfield_bbox")        → (int,int,int,int) # (c0,r0,c1,r1)
("world", "input_model")          → str         # "keyboard" | "click" | "hybrid"
("world", "action_count")         → int         # how many distinct action IDs
("world", "phase")                → str         # "change" | "spill" | "play" (sp80-style)
("world", "orientation")          → int         # 0/90/180/270 (sp80 board rotation)
("world", "color_cycle")          → list[int]   # [9, 8, 12] (ft09 palette)
```

**Object namespace — keyed by object ID, not color or name:**
```
("obj", <id>, "colors")            → set[int]    # current pixel colors
("obj", <id>, "dominant_color")    → int         # most-frequent color
("obj", <id>, "size")              → int         # pixel count
("obj", <id>, "bbox")              → (c0,r0,c1,r1)
("obj", <id>, "centroid")          → (col, row)
("obj", <id>, "shape_hash")        → int         # rotation-normalized pixel hash
("obj", <id>, "shape_variant")     → int         # index within variant family
("obj", <id>, "rotation")          → int         # 0/90/180/270
("obj", <id>, "role")              → str         # OBSERVER-assigned, free-form
("obj", <id>, "selected")          → bool        # cursor/selection is on this obj
("obj", <id>, "visible")           → bool        # currently rendered
("obj", <id>, "layer")             → int         # rendering layer
("obj", <id>, "state")             → str         # "attached" | "detached" | "filled" etc.
("obj", <id>, "paint_colors")      → set[int]    # collected paint (r11l)
("obj", <id>, "visit_count")       → int         # how many times player visited
("obj", <id>, "blocked")           → bool        # player failed to enter this cell
```

**Progress namespace — what has happened this level:**
```
("progress", "steps_taken")         → int
("progress", "steps_remaining")     → int
("progress", "lives_remaining")     → int
("progress", "visited_positions")   → set[tuple]
("progress", "blocked_positions")   → set[tuple]
("progress", "goals_satisfied")     → int
("progress", "goals_total")         → int
("progress", "cursor_on")          → int         # object ID under cursor
("progress", "phase")              → str         # current game phase
("progress", "collision_count")    → int         # penalty hits (r11l)
```

The schema is open-ended: any `("world", <key>)`, `("obj", <id>, <key>)`, or
`("progress", <key>)` can be written at any time. No enumeration is exhaustive —
future games may introduce new keys.

---

### Relation fact schema

Relations are stored as facts keyed by a synthetic relation ID.

```python
@dataclass
class RelFact:
    rel_type:   str           # see vocabulary below
    subjects:   tuple[int,...]# object IDs (ordered where meaningful)
    properties: dict          # relation-specific data
    confidence: float
    scope:      str
    step_index: int
    evidence:   int
```

Stored as: `("rel", <rel_id>) → RelFact(...)`

**Relation type vocabulary** — organized by what observation creates them:

#### Spatial relations (from frame geometry)

```
"same_row"       (A, B)  {"row": 14, "tolerance": 3}
"same_col"       (A, B)  {"col": 23}
"adjacent"       (A, B)  {"distance": 5, "direction": "RIGHT"}
"contains"       (A, B)  {}          # A's bbox contains B's centroid
"left_of"        (A, B)  {"gap": 2}
"above"          (A, B)  {"gap": 9}
"overlaps"       (A, B)  {"area": 12}  # pixel overlap (dc22 collision)
```

#### Structural / grouping (from visual clustering + OBSERVER)

```
"member_of"        (A, G)     {"position": 2, "group_size": 3}
"ordered_sequence"  (G,)       {"members": [A,B,C], "axis": "horizontal"}
"paired_with"       (G1, G2)   {"via": C}     # two sequences linked through connector
"cycle_group"       (G,)       {"members": [A,B,C,D]}  # cyclic permutation group (lp85)
"stacked_on"        (A, B)     {}   # A sits on top of B (layer / z-order)
```

#### Similarity / equivalence (from shape analysis)

```
"same_shape"        (A, B)  {"hash": 0xA3F2}
"same_color"        (A, B)  {"color": 5}
"rotation_of"       (A, B)  {"angle": 90}
"variant_of"        (A, B)  {"delta": 1}  # next variant in family
"mirror_of"         (A, B)  {"axis": "horizontal"}
"color_match"       (A, B)  {}  # A's color set == B's color set (r11l paint)
```

#### Causal / dependency (from behavioral observation)

```
"blocks"            (A, B)  {"direction": "LEFT"}
"enables"           (A, B)  {"condition": "visited"}
"requires"          (A, B)  {"count": 1}  # B requires N visits to A
"triggers"          (A, B)  {"effect": "removes"}  # pressing A removes B (pressure plate)
"cycles_color_of"   (A, B)  {"pattern": [[0,0,0],[0,1,0],[0,0,0]]}  # clicking A cycles B (ft09)
"permutes"          (A, G)  {"direction": "left"}  # clicking A shifts cycle-group G (lp85)
"spawns"            (A, B)  {"direction": (0,1)}  # A produces B moving in direction (sp80 fountain)
"redirects"         (A, B)  {"from": (0,1), "to": (1,0)}  # A deflects B's direction (sp80 ramp)
"fills"             (A, B)  {}  # A is filled by B (sp80 bucket←particle)
```

#### Attachment / coupling (from behavioral observation)

```
"attached_to"       (A, B)  {"offset": (3, 0)}  # A moves with B, fixed offset
"grabbed_by"        (A, B)  {}  # A is being carried by B (wa30)
"co_moves_with"     (A, B)  {}  # A and B always displace identically (ls20 player body)
```

#### Selection / reachability (from player state + pathfinding)

```
"selected"          (cursor, A)  {}
"reachable"         (player, A)  {"path_length": 5}
"unreachable"       (player, A)  {"reason": "blocked"}
```

#### Constraint satisfaction (from ft09-style games)

```
"constrains"        (A, B)  {"must_equal": True, "target_color": 9}
"satisfied"         (A,)    {}  # constraint A is currently satisfied
"violated"          (A,)    {}  # constraint A is currently violated
```

The vocabulary is open-ended. New relation types can be introduced by rules or
the OBSERVER at any time. The relation type string is free-form; the vocabulary
above is a starting set, not a closed enumeration.

---

### Delta events

Every `StateStore.set()` emits a delta:

```python
@dataclass
class Delta:
    key:        tuple       # which fact changed
    old_value:  Any         # None if new fact
    new_value:  Any
    step_index: int
```

For relation facts, creating or removing a relation also emits a delta.

Rules pattern-match on deltas, not just static state. This enables event-driven
rules like:

```
ON  delta("obj", X, "centroid") changed
AND delta("world", "last_frame_diff") > 80 [same step]
THEN  set("obj", color_at(old_centroid), "role") → "state_changer"
```

---

### Scope lifecycle

| Scope | Cleared when | Examples |
|-------|-------------|----------|
| `"step"` | After each env.step() | last_frame_diff, last_action |
| `"level"` | Level advances | blocked_positions, visit_count, goals_satisfied |
| `"episode"` | Episode ends | action_map (may differ game to game) |
| `"game"` | Never (within game run) | wall_colors, walkable_colors, step_size |

`StateStore.clear_scope(scope)` removes all facts with that scope.

---

### Rules over StateStore

A rule has three parts:

```
CONDITION   — pattern over current facts and/or pending deltas
EFFECT      — writes new facts and/or emits actions
CONFIDENCE  — how certain this rule is (propagated to written facts)
```

**Rule tiers** (from raw signal to strategy):

| Tier | What it captures | Example |
|------|-----------------|---------|
| T1 | Action → player displacement | ACTION1 → player.pos += (0, -5) |
| T2 | Action → single object attribute change | any action → counter.size -= 2 |
| T3 | Action → relationship change | ACTION2 → distance(player, goal) decreases |
| T4 | Object attribute (static) | color 3 is dominant → role = "floor" |
| T5 | Object relationship (static) | A and B always move together → co_moves_with |
| T6 | Conditional effect (action + state → special) | at RC pos + action → large diff (maze rotates) |
| T7 | Temporal / sequential | visit RC once, then win_gate → level advance |
| T8 | World structure | step_size = 5, walkable = {3, 5} |
| T9 | Goal / strategy | win_gate unreachable → visit unvisited RC first |

Rules at every tier read and write `StateStore` facts. No tier has a privileged
code path — a T1 rule and a T9 rule are both `(condition, effect, confidence)` triples.

---

### Completeness check against known games

The schema must handle every mechanic discovered across the 25-game corpus.
Below is a verification against the 8 deeply-analyzed games:

| Mechanic | Game(s) | How represented |
|----------|---------|-----------------|
| Grid movement with step_size | ls20, wa30, g50t | `("world", "step_size")`, `("world", "action_map")` |
| Click-based interaction | ft09, r11l, lp85, dc22 | `("world", "input_model")` = "click" / "hybrid" |
| Object selection / cursor | tr87, r11l, lp85, sp80 | `("obj", id, "selected")`, `("progress", "cursor_on")` |
| Shape rotation (cyclic variants) | tr87, ls20 | `("obj", id, "shape_variant")`, `("obj", id, "rotation")`, rel: `variant_of` |
| Color cycling | ft09, ls20 | `("world", "color_cycle")`, `("obj", id, "dominant_color")`, rel: `cycles_color_of` |
| Constraint satisfaction | ft09 | rel: `constrains(A, B, must_equal, target_color)`, `satisfied(A)` |
| Pattern-matching rules | tr87 | rel: `paired_with(G1, G2, via=connector)`, `same_shape(A, B)` |
| Drag / move to goal | r11l | rel: `reachable`, fact: `("obj", id, "centroid")` delta |
| Paint collection | r11l | `("obj", id, "paint_colors")`, rel: `color_match(A, B)` |
| Push-coupling (grab+push) | wa30 | rel: `grabbed_by(obj, player)`, `attached_to(obj, player, offset)` |
| Bridge attachment | dc22 | `("obj", id, "state")` = "attached"/"detached", rel: `attached_to` |
| Pressure plates | dc22 | rel: `triggers(plate, blocker, effect="removes")` |
| Cyclic permutation buttons | lp85 | rel: `cycle_group(G, members=[...])`, `permutes(button, G, direction)` |
| Two-phase state machine | sp80 | `("world", "phase")`, `("progress", "phase")` |
| Particle flow / spawning | sp80 | rel: `spawns(fountain, particle, direction)`, `redirects(ramp, particle)` |
| Bucket fill detection | sp80 | rel: `fills(bucket, particle)`, spatial adjacency check |
| Board rotation | sp80 | `("world", "orientation")` affecting input + physics |
| Maze rotation (state-changer) | ls20 | rel: `enables(RC, win_gate)`, `requires(RC, win_gate, count=1)` |
| Step counter / budget | all games | `("progress", "steps_remaining")`, `("progress", "lives_remaining")` |
| Multi-goal levels | ls20 (L6), r11l, lp85 | `("progress", "goals_satisfied")`, `("progress", "goals_total")` |
| Obstacle penalty | r11l, wa30 | `("progress", "collision_count")`, rel: `blocks(obstacle, player)` |
| Co-moving objects | ls20 (player body) | rel: `co_moves_with(A, B)` |
| Spatial zones (free/blocked/hazard) | wa30 | multiple `("world", "zone_*")` facts or tagged relations |
| Pixel-level collision | dc22 | rel: `overlaps(A, B, area)` |
| Animation state blocking | dc22, sp80 | `("world", "animating")` → bool, rules should not fire during animation |
| Fog of war | ls20 (L7-8) | `("world", "fog")` → bool |
| Lives / retry | ls20 | `("progress", "lives_remaining")` |
| Dual coordinate systems | dc22 | `("world", "grid_to_pixel_offset")`, conversion is a world fact |
| Object invisibility | tr87 (hidden click targets) | `("obj", id, "visible")` |
| Collectible removal on visit | ls20 (rings), r11l (paint), sp80 (particles) | delta on `("obj", id, "visible")` False + source="consumed" |

**Gap analysis:** No mechanic from the 8 analyzed games falls outside the schema.
The schema's open-ended key design (any string key in any namespace) means that
unforeseen mechanics in the remaining 17 games can be added without schema changes —
only new keys and relation types, which is purely additive.

---

## Cross-Domain Generalization: Home Robot Assessment

The StateStore's core mechanisms — `StateFact`, `RelFact`, `Delta`, rules as
`(condition, effect, confidence)` triples — are domain-agnostic by design.
However, several concrete design choices bake in ARC-AGI-3 assumptions that
would break when applied to a physical-world home robot. This section
identifies each gap and proposes the minimal schema changes needed so that
a single StateStore implementation serves both ARC games and physical robots
(and any future domain).

### Gap 1: Scope lifecycle is game-specific

**Current:** `scope ∈ {"step", "level", "episode", "game"}` with hard-coded
clear semantics tied to game progression.

**Robot problem:** A home robot has no "levels." Its world is continuous.
Facts persist for wildly different durations: "the cup is on the table" lasts
minutes; "Alice prefers oat milk" lasts months; "the front door is locked"
lasts until someone unlocks it. The four game scopes cannot express this.

**Fix:** Replace the fixed scope enum with a two-field model:

```python
@dataclass
class StateFact:
    value:      Any
    confidence: float
    source:     str           # "prior" | "inferred" | "observed" | "rule" | "told"
    scope:      str           # domain-defined, open-ended
    ttl:        Optional[float]  # seconds until auto-expiry, None = permanent
    timestamp:  float         # Unix epoch (replaces step_index)
    evidence:   int
```

- `scope` becomes an open string: `"step"`, `"level"`, `"game"` for ARC;
  `"moment"`, `"task"`, `"session"`, `"day"`, `"permanent"` for robot.
- `ttl` (time-to-live) lets facts auto-expire: sensor readings expire in
  seconds, task-local facts expire when the task ends, preferences never expire.
- `timestamp` replaces `step_index` as the universal ordering key. In ARC,
  `timestamp` can simply be set to `step_index` (integer time). In the robot,
  it's `time.time()`.

`clear_scope(scope)` still works — it deletes all facts with that scope label.
Domain code defines which scopes exist and when to clear them.

---

### Gap 2: No real temporal model

**Current:** `step_index` provides discrete ordering within an episode.
No duration, no real-time clock, no temporal relations.

**Robot problem:** "Alice left for work 30 minutes ago." "The oven has been
preheating for 8 minutes." "Water the plants every Tuesday." These all require:
- Real-time timestamps (already addressed by Gap 1's `timestamp` field)
- Duration-aware facts
- Temporal relations between events
- Recurrence / schedule facts

**Fix — temporal relation types** (added to RelFact vocabulary):

```
"before"          (E1, E2)  {"gap_sec": 120.0}
"after"           (E1, E2)  {}
"during"          (E1, E2)  {}       # E1 occurred while E2 was active
"simultaneous"    (E1, E2)  {"tolerance_sec": 1.0}
"caused_by"       (E1, E2)  {}       # E2 triggered E1
"periodic"        (E,)      {"cron": "0 8 * * *", "next_at": 1712800000.0}
```

**Fix — duration attribute:**

```
("obj", <id>, "state_since")      → float   # timestamp when current state began
("task", <id>, "started_at")      → float
("task", <id>, "deadline")        → float
("task", <id>, "duration_est")    → float   # seconds
```

---

### Gap 3: 2D grid coordinates → 3D continuous space — RESOLVED BY OBSERVER

**Current:** All positions are `(col, row)` integer tuples on a pixel grid,
with `step_size` defining the discrete cell spacing.

**Robot problem:** Physical objects exist in continuous 3D space. A cup is
at `(1.23, 0.87, 0.75)` meters in the kitchen reference frame. The robot's
gripper has 6-DOF pose. Rooms are volumes, not grid cells.

**Resolution:** This is an OBSERVER responsibility, not a StateStore schema
issue. The OBSERVER is the perception boundary between the external world
(pixel frames, LIDAR point clouds, depth cameras, etc.) and the internal
representation. Its job is to convert raw sensory data into `StateFact` and
`RelFact` entries using whatever coordinate system the domain requires.

- In ARC: OBSERVER reads a pixel frame → produces `("obj", id, "centroid")`
  as `(col, row)` integers on a grid.
- In a home robot: OBSERVER fuses camera + LIDAR + IMU → produces
  `("obj", id, "position")` as `(x, y, z)` floats in meters.

The StateStore already stores positions as `Any`-typed values, so `(col, row)`
and `(x, y, z, roll, pitch, yaw)` both fit without schema changes. The
position tuple's dimensionality is implicit in the domain's OBSERVER.

Similarly, spatial relations like `on_top_of`, `inside`, `near`, `in_room`
are computed by the OBSERVER (or by a perception tool it delegates to) and
written as `RelFact` entries. The relation vocabulary is already open-ended —
adding 3D relation types is purely additive and requires no schema change.

**OBSERVER perception tool:** For expensive perception pipelines (e.g.,
3D scene reconstruction, object pose estimation, semantic segmentation),
the OBSERVER can delegate to a specialized **perception tool** that runs
asynchronously and writes results back to the StateStore. This keeps the
OBSERVER itself lightweight (orchestration + fact writing) while the heavy
computation lives in pluggable tools:

```
OBSERVER  →  calls perception_tool("identify_objects", sensor_data)
          →  perception_tool returns [{id, position, class, confidence}, ...]
          →  OBSERVER writes StateFacts and RelFacts into StateStore
```

This is the same pattern as ARC's OBSERVER calling an LLM to interpret
a game frame — the LLM *is* the perception tool.

**No schema changes required.** The StateStore's `Any`-typed values and
open-ended relation vocabulary handle 2D and 3D transparently. The
complexity lives in the OBSERVER and its perception tools, not in the
store.

---

### Gap 4: No entity type system

**Current:** Everything is an "object" identified by integer ID, discovered
via connected-component analysis or OBSERVER labeling.

**Robot problem:** A home has fundamentally different entity categories —
people, rooms, appliances, furniture, tools, consumables, pets — each with
category-specific attributes. A person has a name, face embedding, and
preferences. A room has a floor plan. An appliance has an operational state
and a manual. These are not just "objects with extra keys."

**Fix — namespace per entity category:**

```
# People (persistent across sessions)
("person", <id>, "name")           → str         # "Alice"
("person", <id>, "face_embedding") → ndarray     # for re-identification
("person", <id>, "role")           → str         # "resident" | "guest" | "child"
("person", <id>, "last_seen")      → float       # timestamp
("person", <id>, "location")       → str         # room ID or "away"
("person", <id>, "activity")       → str         # "sleeping" | "cooking" | "watching_tv"
("person", <id>, "mood")           → str         # "calm" | "stressed" (if detectable)

# Rooms / zones
("room", <id>, "name")            → str         # "kitchen"
("room", <id>, "floor_plan")      → Polygon     # walkable area boundary
("room", <id>, "temperature")     → float       # Celsius
("room", <id>, "lighting")        → str         # "bright" | "dim" | "off"
("room", <id>, "occupancy")       → int         # number of people detected

# Appliances / devices
("device", <id>, "name")          → str         # "oven"
("device", <id>, "type")          → str         # "cooking" | "cleaning" | "climate"
("device", <id>, "power_state")   → str         # "on" | "off" | "standby"
("device", <id>, "operational_state") → str     # "preheating" | "ready" | "error"
("device", <id>, "target_temp")   → float
("device", <id>, "current_temp")  → float

# Task / goal (replaces game's "progress" namespace for robot)
("task", <id>, "description")     → str
("task", <id>, "status")          → str         # "pending" | "active" | "done" | "failed"
("task", <id>, "parent_task")     → Optional[int]  # for hierarchical decomposition
("task", <id>, "assigned_to")     → str         # "robot" | person ID
("task", <id>, "priority")        → int         # 0 = urgent, 9 = low
("task", <id>, "requester")       → int         # person ID who asked for it
```

The `("obj", ...)` namespace still exists for generic physical objects
(cups, books, packages). Entity categories are just namespace conventions —
the StateStore doesn't enforce type constraints. A domain can define any
namespace it needs.

---

### Gap 5: No event / history log

**Current:** Deltas are emitted and consumed within a single step. There is
no persistent record of past events. ARC doesn't need one — each step is
essentially Markovian with respect to the current frame.

**Robot problem:** "When did Alice last take her medication?" "Has the front
door been opened today?" "What happened while I was away?" All require a
queryable event history that survives scope clears.

**Fix — EventFact as a new fact type:**

```python
@dataclass
class EventFact:
    event_type: str           # "person_entered", "device_state_changed", etc.
    subjects:   tuple[int,...]# entity IDs involved
    properties: dict          # event-specific data
    timestamp:  float         # when it happened
    source:     str           # "observed" | "inferred" | "reported"
    confidence: float
```

Stored as: `("event", <event_id>) → EventFact(...)`

Events are **immutable** — once recorded, they are never modified (unlike
state facts which are overwritten). They accumulate in an append-only log.
Retention policy is domain-defined (e.g., keep 30 days of events, then
archive or summarize).

**Event type vocabulary (starter):**

```
"person_entered"       {"person": id, "room": id}
"person_left"          {"person": id, "room": id}
"object_moved"         {"object": id, "from": pos, "to": pos, "by": person_id}
"device_state_changed" {"device": id, "old": "off", "new": "on"}
"task_completed"       {"task": id, "duration_sec": 300}
"utterance"            {"person": id, "text": "...", "intent": "request"}
"anomaly"              {"description": "front door open past midnight"}
"routine_triggered"    {"routine": "morning_lights", "trigger": "schedule"}
```

In ARC, the event log is optional (step-by-step replay can be reconstructed
from deltas). In the robot, it's essential infrastructure.

---

### Gap 6: No social / functional / normative relations

**Current:** Relation vocabulary covers spatial, structural, similarity,
causal, attachment, selection, and constraint. These are all physical or
logical relationships between objects in a game.

**Robot problem:** Human environments are rich with non-physical relationships:
- Social: "Alice is Bob's mother", "Charlie is a guest"
- Functional: "the mug is for drinking", "the broom is stored in the closet"
- Ownership: "this laptop belongs to Alice"
- Normative: "don't vacuum while someone is sleeping", "knock before entering"
- Preference: "Bob likes the thermostat at 72°F"

**Fix — extended relation vocabulary:**

```
# Social
"family_of"         (P1, P2)  {"relation": "parent"}
"lives_with"        (P1, P2)  {}
"caretaker_of"      (P1, P2)  {}   # P1 cares for P2 (child, elderly)
"guest_of"          (P1, P2)  {"until": timestamp}

# Functional / purpose
"used_for"          (O, purpose)  {"purpose": "drinking"}
"stored_in"         (O, L)     {}  # O's home location is L
"part_of"           (O1, O2)   {}  # O1 is a component of O2
"substitute_for"    (O1, O2)   {}  # O1 can replace O2 (oat milk for dairy)

# Ownership
"belongs_to"        (O, P)     {}
"shared_by"         (O, [P1,P2]) {}

# Normative (soft constraints on robot behavior)
"prohibited_when"   (action, condition)  {"reason": "noise during sleep"}
"required_before"   (A1, A2)   {}  # A1 must happen before A2
"preferred_by"      (setting, P) {"value": 72, "unit": "°F"}
"routine"           (sequence,) {"schedule": "0 7 * * 1-5", "steps": [...]}
```

---

### Gap 7: Multi-modal source tracking

**Current:** `source` is a simple string: `"prior" | "inferred" | "observed" | "rule"`.

**Robot problem:** A robot derives facts from multiple sensor modalities —
camera, LIDAR, microphone, touch, temperature, gas sensors — each with
different noise characteristics and update rates. "I see a cup on the table"
(camera, high confidence) vs "I heard something fall" (microphone, low
position confidence). Sensor fusion requires knowing which modalities
contributed to each fact.

**Fix — structured source:**

```python
@dataclass
class FactSource:
    origin:     str           # "observed" | "inferred" | "told" | "rule" | "prior"
    modality:   Optional[str] # "camera" | "lidar" | "microphone" | "touch" | "api" | None
    sensor_id:  Optional[str] # specific sensor instance
    model:      Optional[str] # ML model that produced inference, if any
    told_by:    Optional[int] # person ID, if source is "told"
```

In ARC, `source` stays a simple string (modality is always "camera" —
the game frame). The structured source is backward-compatible: ARC code
only reads `source.origin`.

---

### Gap 8: Hierarchical goals and task decomposition

**Current:** Goals are flat: `("progress", "goals_satisfied")` and
`("progress", "goals_total")`. A rule at tier T9 says "go to the win gate."

**Robot problem:** "Prepare dinner" decomposes into subtasks: check
pantry → plan recipe → retrieve ingredients → cook → plate → set table.
Each subtask may itself decompose. The robot must track which subtask is
active, what's blocked, and what's done. This is hierarchical task
network (HTN) planning.

**Fix — task tree as facts:**

```
("task", <id>, "description")     → str
("task", <id>, "status")          → str     # "pending" | "active" | "blocked" | "done" | "failed"
("task", <id>, "parent_task")     → int     # parent task ID, None for root
("task", <id>, "children")        → list[int]
("task", <id>, "preconditions")   → list[tuple]  # StateStore keys that must be true
("task", <id>, "postconditions")  → list[tuple]  # facts that should be true when done
("task", <id>, "priority")        → int
("task", <id>, "deadline")        → Optional[float]
```

**Task relations:**

```
"subtask_of"        (child, parent)  {"order": 2}
"depends_on"        (T1, T2)         {}   # T1 cannot start until T2 is done
"conflicts_with"    (T1, T2)         {}   # T1 and T2 cannot run simultaneously
"interrupted_by"    (T1, T2)         {}   # T2 preempted T1
```

In ARC, the task hierarchy is shallow (one root goal: "advance the level,"
maybe with sub-goals like "visit RC" and "reach win_gate"). The same
`("task", ...)` namespace works for both.

---

### Gap 9: Persistent identity across sessions

**Current:** Object IDs are assigned per level and may be reassigned on
level advance. The entire StateStore can be cleared between episodes.

**Robot problem:** People, rooms, and important objects persist indefinitely.
Alice is always Alice. The kitchen is always the kitchen. The robot needs
a persistent entity registry that survives restarts.

**Fix — persistence tier:**

Add a `persistence` field to StateFact:

```python
persistence: str   # "volatile" | "session" | "persistent"
```

- `"volatile"`: Cleared at scope boundary (same as current behavior).
- `"session"`: Survives scope clears within a session, cleared on restart.
- `"persistent"`: Written to disk, survives restarts. Used for people,
  rooms, long-term preferences, learned routines.

The StateStore serialization layer writes persistent facts to a backing
store (JSON file, SQLite, etc.). On startup, persistent facts are loaded
before any new observations arrive.

In ARC, all facts are volatile or session-scoped. No change to game code.

---

### Summary of schema changes

| Change | Affects | ARC impact |
|--------|---------|------------|
| `scope` → open string | StateFact, RelFact | ARC uses same values, no code change |
| Add `ttl` field | StateFact | ARC sets `ttl=None` (no expiry), no code change |
| `step_index` → `timestamp` (float) | StateFact, RelFact, Delta | ARC sets `timestamp=step_index`, no code change |
| Add `persistence` field | StateFact | ARC uses "volatile", no code change |
| Add `EventFact` type | StateStore | ARC doesn't use it, no code change |
| Add entity namespaces (person, room, device, task) | Key conventions | ARC ignores them, no code change |
| ~~Add 3D spatial relations~~ | ~~RelFact vocabulary~~ | **Not needed** — OBSERVER handles coordinate conversion |
| Add social/functional/normative relations | RelFact vocabulary | ARC ignores them, no code change |
| Add temporal relation types | RelFact vocabulary | ARC ignores them, no code change |
| Structured `FactSource` | StateFact.source | ARC uses string shorthand, no code change |
| Task hierarchy facts | Key conventions | ARC optionally uses shallow tasks |

**Design principle preserved:** Every change is additive. No existing ARC
field is removed or renamed. The schema remains open-ended — any new
namespace, key, or relation type can be introduced without modifying the
StateStore implementation itself.
