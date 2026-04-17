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
        Input:  frame (64×64 grid), current StateStore facts (objects, roles, world),
                action effects table, action history, matched rules
        Output: JSON with level_description, visual_observations, action_characterizations,
                identified_objects, role assignments, hypothesized_goal, reasoning
                → writes StateFact/RelFact entries to StateStore
                All inferences labeled [GUESS] or [CONFIRMED]

    Round 2: MEDIATOR
        Input:  OBSERVER analysis, rules, goals+state, action history
        Output: JSON with action_plan, rule_updates, goal_updates, state_updates, reasoning

    Parse updates:
        - OBSERVER writes object/role facts to StateStore (replaces concept_bindings merge)
        - MEDIATOR writes goal/strategy facts to StateStore (replaces state_updates)
        - Rule updates added to rule engine (all forced to "candidate" status)

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
and stored as `StateFact` entries in the StateStore (see D3).

**What is allowed:** Rendering distinction (different char per color value). Not allowed:
pre-assigning semantic names to specific color integers in any prompt or constant.

---

### D3 — Object roles live in the StateStore, not in a separate bindings dict

~~Previously, concept bindings were a two-level dict (`{color: role}` + `{"wall_colors": [...]}`)
stored in `state_manager._data`. This is superseded by the StateStore.~~

Object roles are now `StateFact` entries:

```
@(obj, <id>, "role")  →  "player" | "wall" | "goal" | "step_counter" | ...
```

**Source tracking replaces the two schemas.** The old `{color: role}` (OBSERVER guess)
vs `{"wall_colors": [...]}` (behavioral confirmation) distinction is captured by the
`source` and `confidence` fields of `StateFact`:
- OBSERVER's initial guess: `source="observed", confidence=0.5`
- Behavioral confirmation (wall blocked movement): `source="rule", confidence=0.9`

**Scope replaces the "episode-local" invariant.** Role facts have `scope="level"` or
`scope="episode"` — cleared at the appropriate boundary. Cross-game concept knowledge
lives in concept facts (`scope="game"` or persistent) separate from concrete role
assignments.

**Role name vocabulary:** OBSERVER assigns generic role names from its own common-sense
vocabulary. No fixed enumeration — the LLM-as-ontology-engine principle means the
system's concept vocabulary grows with what it encounters (see "LLM as the Ontology
Engine" section).

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
3. Adjacent cells are assigned `@(obj, X, "role") := "wall"` in the StateStore via a
   concept-binding rule (T4). The concept fact `@(concept, "wall", "discriminator") =
   "blocks movement, immovable"` is written once and persists.

**Why color-agnostic:** Wall color varies between games. The concept "wall" is stable;
the color is not. The concept-binding rule maps the concrete color to the abstract role,
so a wall-color change in a new level only requires re-learning the binding rule — the
abstract movement rules that check `role ≠ "wall"` survive unchanged (see Rule
Abstraction Ladder).

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

**Implementation:** Initial hypotheses from `game_knowledge.json` are loaded as
`StateFact` entries with `source="prior"` and low confidence (e.g., 0.3). As the
system observes evidence, inference rules write higher-confidence facts with
`source="inferred"` or `source="observed"` that supersede the priors. All downstream
code reads from the StateStore, which returns the highest-confidence fact for each key.

~~The former `GameHypothesis` dataclass with `prior_*`/`obs_*`/`effective_*` properties
is superseded by this mechanism — the StateStore's confidence-ranked facts provide the
same prior→observed fallback behavior without a separate data structure.~~

**Corollary:** Any hardcoded filter, threshold, or heuristic that assumes specific
properties of a particular game is a design violation. If the system relies on
"max_sprite_pixels=12" or "ignore the most common color" or "HUD is at y≥58", it
will break on a different game that doesn't match those assumptions.

---

### P2 — The OBSERVER identifies objects, not the Python code

The OBSERVER (LLM) looks at the game frame with human-like vision and common sense.
It identifies what objects exist, what they look like, and what role they may play (toggle,
changer, wall, player, status bar, etc.). It writes these identifications as `StateFact`
and `RelFact` entries in the StateStore. These are initial hypotheses at low confidence —
they may be wrong, and behavioral evidence (P3) can overwrite them.

The OBSERVER is also the source of **initial concept labels** — from the very first
frame, before any interaction, it guesses roles for all visible objects. This gives
the rule system a starting vocabulary immediately (see "LLM as the Ontology Engine").

The Python frame analysis code (perception tools) only does **position mapping**:
given an object ID that the OBSERVER has registered, where does it appear on the
game grid? The Python tool may also flag evidence that contradicts the OBSERVER's
labels (e.g., a "wall" that moved), which triggers concept refinement.

**Division of labor:**

| Responsibility                        | Who does it       |
|---------------------------------------|-------------------|
| "That yellow ring is a step-counter"   | OBSERVER (LLM) → `@(obj, 5, "role") := "step_counter"` |
| "Yellow pixels at (19,30) and (34,15)" | Perception tool (Python) |
| "The yellow bar at the bottom is HUD"  | OBSERVER (LLM) → `@(obj, 8, "role") := "hud"` |
| "Object 3 blocks movement"            | Rule engine (T4 concept-binding rule) |
| "Object 3's role should split"         | OBSERVER/MEDIATOR (concept split, LLM call) |

Python code may not filter, exclude, or re-classify candidates based on
heuristics (dominant color frequency, pixel count thresholds, screen region).
Classification is the OBSERVER's job; refinement is the rule engine's job;
ambiguity resolution is the MEDIATOR's job.

---

### P3 — Object discovery is behavioral, not static

Objects are discovered by **what happens when you interact with them**, not by
how they look in a single frame. The OBSERVER provides initial role guesses
from the first frame (P2), but these are hypotheses at low confidence.
Behavioral evidence writes higher-confidence role facts that supersede them.

| Observation (delta pattern)                        | Role fact written  |
|----------------------------------------------------|-------------------|
| Large pixel diff at position, no level advance     | `@(obj, X, "role") := "state_changer"` |
| Level advance while standing at position           | `@(obj, X, "role") := "goal"` |
| Object disappears from frame after being visited   | `@(obj, X, "role") := "consumable"` |
| Counter object grows after visiting position       | `@(obj, X, "role") := "resource_reset"` |
| Agent didn't move despite taking an action         | `@(obj, X, "role") := "wall"` |
| Object moves when pushed                           | Concept split: "wall" → "wall" + "pushable" |

Each row corresponds to a T4 concept-binding rule in the rule engine. The
rules fire on delta events and write role facts to the StateStore. The role
vocabulary is open-ended — the OBSERVER names new roles as needed (see
"LLM as the Ontology Engine").

A single frame can identify *candidates* (non-background, non-agent pixels), but
classification requires observing behavior over time. The OBSERVER's initial
labels are starting hypotheses; behavioral confirmation raises their confidence.

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

### P5 — Agent state tracking must be adaptive

The agent's visual appearance (color, shape, rotation) changes during gameplay
as it interacts with state-changers. The system must track these changes and
update its internal model accordingly — it cannot assume the agent always looks
the same as it did at level start.

**Implementation via StateStore:** The agent's appearance is stored as mutable
facts: `@(obj, agent_id, "colors")`, `@(obj, agent_id, "shape_hash")`,
`@(obj, agent_id, "rotation")`. When the perception tool detects that the
agent has changed appearance (frame diff reveals the old agent-colored cluster
disappeared and a new one appeared), it updates these facts. A delta event
fires, which downstream rules can react to.

**General principle:** Any environment property that can change during play
(agent appearance, traversable regions, action semantics) must be a `StateFact`
with appropriate scope — not a constant or a one-time assignment. The scope
lifecycle handles resetting on level advance.

---

### P6 — Inference is independent, fallible, and expressed as rules

Each type of inference (step size, action directions, traversable regions,
agent appearance) operates independently. Each can fail without affecting
the others. When an inference has not yet produced a result, the system
falls back to the prior-confidence facts loaded from initial hypotheses.

**Implementation via rules:** Each inference is a rule (or small rule chain)
in the rule engine. For example:

- T8 rule: after observing two agent displacements of the same magnitude,
  write `@(world, "step_size") := magnitude, confidence=0.8`
- T1 rule: after observing `ACTION1 → displacement(0, -N)`, write
  `@(world, "action_map", "ACTION1") := "UP", confidence=0.9`

This means the system degrades gracefully: on the first step with no history,
no inference rules have fired and the system runs entirely on prior-confidence
facts. As history accumulates, higher-confidence facts from fired rules
supersede the priors one by one. There is no single point of failure.

**Temporary hard-coded inference functions** (e.g., `infer_step_size`) may
exist during the transition to the rule-based architecture. These are
intermediate implementations that will be replaced by T8 rules as the rule
engine matures. They are not part of the target design.

---

### P7 — Cross-episode knowledge persists via scope and persistence tiers

Facts confirmed through successful gameplay (e.g., "level 1 requires visiting
a state-changer before the goal", "color 3 objects have role=state_changer")
persist in the StateStore with `scope="game"` or `persistence="persistent"`.
Future episodes begin with these facts already loaded, avoiding re-discovery.

**Scope-based persistence replaces `discovered_knowledge.json`.** The StateStore's
scope lifecycle (`step < level < episode < game`) and persistence tier
(`volatile < session < persistent`) provide the same layered persistence
without a separate file. Facts written with `scope="game"` survive across
episodes within a run; facts with `persistence="persistent"` survive across
runs (serialized to disk).

**Generalization of persisted facts.** A fact confirmed in one level
(e.g., `color=5 → role="wall"`) is a concrete binding. Through the rule
abstraction ladder, it may generalize to a concept-level fact
(`@(concept, "wall", "discriminator") = "blocks movement"`) that transfers
across levels and games. The concrete binding is level-scoped; the concept
fact is game-scoped or persistent.

**Rule-based matching replaces fuzzy matching.** Rather than fuzzy string
matching against a knowledge file, the rule engine matches incoming
observations against existing concept facts and rules. A T4 concept-binding
rule fires when it sees evidence matching a known concept's discriminator —
this is structural matching, not string similarity, and scales to large
knowledge stores through the rule engine's indexing.

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

Keys are tuples. The first element is the namespace. The schema defines
**only domain-agnostic core keys**. Domain-specific keys (e.g., ARC game
colors, robot sensor readings) are written by the OBSERVER and are not
part of the core schema — they are just facts like any other.

**World namespace — environment physics and affordances:**
```
("world", "input_model")          → str         # how the agent acts: "keyboard" | "click" | "continuous" | ...
("world", "action_count")         → int         # how many distinct actions available
("world", "action_map")           → dict        # maps semantic names to action IDs
("world", "phase")                → str         # current phase if env has phases
("world", "bounds")               → tuple       # environment boundary (any dimensionality)
```

**Object namespace — keyed by object ID, not name or type:**
```
("obj", <id>, "position")          → tuple       # location (dimensionality set by OBSERVER)
("obj", <id>, "orientation")       → Any         # rotation/heading (domain-defined)
("obj", <id>, "size")              → Any         # extent (pixels, meters, volume — domain-defined)
("obj", <id>, "role")              → str         # OBSERVER-assigned, free-form
("obj", <id>, "selected")          → bool        # agent is currently targeting this obj
("obj", <id>, "visible")           → bool        # currently perceivable
("obj", <id>, "state")             → str         # free-form state label
("obj", <id>, "visit_count")       → int         # how many times agent has interacted
```

**Progress namespace — what has happened in the current task/episode:**
```
("progress", "steps_taken")         → int
("progress", "steps_remaining")     → int         # if budget-limited
("progress", "goals_satisfied")     → int
("progress", "goals_total")         → int
("progress", "phase")              → str         # current task phase
```

The schema is **open-ended**: any `("<namespace>", <key>)` or
`("<namespace>", <id>, <key>)` can be written at any time by the OBSERVER,
rules, or the agent itself. No enumeration is exhaustive.

**Domain-specific keys are NOT part of the core schema.** They are written
by each domain's OBSERVER as ordinary facts. Examples:

*ARC-AGI-3 OBSERVER writes:*
```
("world", "step_size")            → 5           # grid cell size in pixels
("world", "walkable_colors")      → {3, 5}      # pixel colors the player can traverse
("world", "wall_colors")          → {4}
("world", "grid_offset")          → (2, 3)      # sub-cell alignment
("world", "color_cycle")          → [9, 8, 12]  # ft09 palette cycle
("obj", 7, "dominant_color")      → 1           # pixel color
("obj", 7, "shape_hash")          → 0xA3F2      # rotation-normalized shape
("obj", 7, "bbox")                → (10, 20, 25, 35)
("obj", 7, "paint_colors")        → {3, 5}      # r11l collected paint
("progress", "blocked_positions") → {(5,10)}     # BFS dead ends
("progress", "lives_remaining")   → 3
```

*Home robot OBSERVER writes:*
```
("world", "coordinate_frame")     → "house_floor1"
("world", "units")                → "meters"
("obj", 12, "class")              → "mug"
("obj", 12, "material")           → "ceramic"
("obj", 12, "weight_kg")          → 0.35
("obj", 12, "temperature_c")      → 62.0
("person", 1, "name")             → "Alice"
("person", 1, "activity")         → "reading"
("room", 3, "name")               → "kitchen"
("room", 3, "temperature_c")      → 22.5
```

The StateStore implementation treats all of these identically — it does
not know or care which keys exist. Domain knowledge lives in the OBSERVER
and in rules, never in the store itself.

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

### Rule Abstraction Ladder

Rules are not static. A rule evolves through levels of abstraction as the
system accumulates evidence, encounters failures, and extracts concepts.
This progression is the mechanism by which the system transfers knowledge
across levels, games, and domains.

#### The four levels

**Level 0 — Raw observation (T1).** Direct correlation from a single episode:

```
ACTION1  →  Δ(obj, agent, position.x) = +step_size
```

Confidence: 0.80. Sometimes fails; the system doesn't yet know why.

**Level 1 — Specialized with concrete condition (T6).** After observing that
ACTION1 fails when a specific concrete condition holds:

```
ACTION1
  ∧ adj(agent, RIGHT).color ≠ 5
  →  Δ(obj, agent, position.x) = +step_size
```

Confidence: 0.95. Fewer failures, but the condition `color ≠ 5` is brittle —
it breaks if wall color changes in level 2.

**Level 2 — Concept extraction (T4 + T6).** The concrete condition is replaced
by an abstract concept. Two rules replace one:

Rule A — *concept binding* (maps concrete attribute to abstract role):
```
obj.color = 5
  ∧ agent failed to enter obj.position
  →  @(obj, X, role) := "wall"
```

Rule B — *abstract movement* (references role, not color):
```
ACTION1
  ∧ adj(agent, RIGHT).role ≠ "wall"
  →  Δ(obj, agent, position.x) = +step_size
```

If wall color changes from 5 to 7 in a new level, only Rule A needs
re-learning. Rule B survives intact.

**Level 3 — Full generalization (T8 + T6).** The direction mapping itself
is extracted as a parameterized concept:

Rule C — *action-direction binding*:
```
ACTION1 → direction := RIGHT
ACTION2 → direction := LEFT
ACTION3 → direction := UP
ACTION4 → direction := DOWN
```

Rule D — *universal movement*:
```
ACTION(X)
  ∧ X.direction = D
  ∧ adj(agent, D).role ≠ "wall"
  →  Δ(obj, agent, position) += D.unit_vector × step_size
```

A single movement rule now handles all four directions. The only
game-specific knowledge is in Rules A (what is a wall) and C (which
button maps to which direction) — both are cheap leaf rules that re-learn
quickly on game or level change.

#### What transfers across contexts

| Scenario | Level 1 (concrete) | Level 2+ (abstract) |
|----------|-------------------|---------------------|
| New level changes wall color 5 → 7 | Rule breaks, must re-learn from scratch | Rule B survives; only Rule A re-learns |
| Different game uses color 5 as floor | Rule wrongly avoids color 5 | Rule A never fires (no blocked evidence), color 5 gets `role="floor"` |
| Wall becomes invisible (no color cue) | Cannot represent | Rule A learns from blocked movement alone, no color needed |
| Action buttons remapped | Level 3 Rule D breaks | Only Rule C re-learns; Rule D survives |

Each level separates **what changes between contexts** (color=5, ACTION1=RIGHT)
from **what is invariant** (walls block movement, actions cause directional
displacement). Invariant parts survive; variant parts are cheap leaf rules.

---

### Triggers for Climbing and Descending the Ladder

The system does not generalize or specialize on a fixed schedule. Specific
evidence patterns trigger transitions between abstraction levels.

#### Climbing (generalization)

**Repeated observation.** When the system observes the same pattern across
multiple contexts (different levels, different episodes), the recurring
structure is a candidate for extraction. For example, if `color=5 → blocked`
holds across three levels, the concept "wall" is worth extracting even if
the OBSERVER hasn't named it yet.

**Concept ladder match.** When a concrete condition in a Level 1 rule
(e.g., `color=5`) also appears in a concept fact written by the OBSERVER
(e.g., `@(concept, "wall", "indicators") contains color=5`), the system
can immediately generalize to Level 2 by referencing the concept instead
of the concrete value. This is the fast path — the OBSERVER's common-sense
knowledge short-circuits what would otherwise require many episodes of
bottom-up pattern extraction.

**OBSERVER top-down labeling.** The OBSERVER is prompted to label objects
and roles from the very first frame, before any interaction. These initial
guesses (at low confidence) provide a starting concept vocabulary that
rules can reference immediately. As behavioral evidence accumulates
(the labeled "wall" actually blocks movement), confidence rises. If a
label is wrong, low evidence makes it cheap to overwrite.

#### Descending (specialization)

**Confidence drop.** When a rule's confidence falls below a threshold
(e.g., < 0.7 after N failures), the system looks for a discriminating
condition between its success and failure cases, then creates a more
specialized version.

**Concept splitting.** When an abstract concept starts producing
contradictory evidence (e.g., some "walls" move when pushed, others
don't), the concept splits into sub-concepts. Split types include:

- *Behavioral split:* Same appearance, different response to action
  (wall vs pushable vs breakable).
- *Conditional split:* Same object, different behavior depending on state
  (locked door vs unlocked door).
- *Contextual split:* Same concept, different meaning in different phases
  (color 5 is floor in phase 1, hazard in phase 2).
- *Compositional split:* Concept conflates independent properties
  (color=5 means "wall AND blue" but wall-ness and blue-ness are
  independent).

Splits are triggered mechanically (the rule tool detects the confidence
drop and identifies the discriminating sub-condition) but *named* by
the LLM (the OBSERVER decides that the sub-concepts should be called
"wall" and "pushable" rather than "blocking_type_1" and "blocking_type_2").

#### MEDIATOR as tiebreaker

When the OBSERVER is uncertain between two plausible interpretations,
or when rules produce contradictory evidence about an object's role,
the MEDIATOR can be consulted. The OBSERVER stays focused on perception;
the MEDIATOR provides judgment. This separation prevents the OBSERVER
from stalling on ambiguity.

---

### Confidence Propagation Through Rule Chains

When Rule B reads a fact that was written by Rule A, Rule B's effective
confidence is bounded by Rule A's confidence:

```
effective_confidence(Rule B) =
    Rule_B.own_confidence × min(confidence of each input fact)
```

This prevents the system from acting confidently on shaky foundations.
A freshly-extracted concept with confidence 0.6 limits all downstream
rules that depend on it, regardless of how often those rules have
succeeded in the past. As the concept's evidence grows and its confidence
rises, all dependent rules benefit automatically.

---

### Unknown Facts and Exploration

When a rule condition references a fact that does not exist in the
StateStore (e.g., `adj(agent, RIGHT).role` when the adjacent cell has
never been classified), the condition evaluates to `UNKNOWN` rather
than `TRUE` or `FALSE`.

Unknown conditions do not silently pass. Instead, they trigger
**exploration subgoals**:

```
adj(agent, RIGHT).role = UNKNOWN
  → create subgoal: "determine role of cell (x+1, y)"
  → plan: traverse to cell → observe result
  → outcome writes fact: role="wall" or role="floor" or role="pushable" ...
```

Exploration subgoals are prioritized by relevance to the current goal.
Unknowns on the critical path to the active goal are explored first.
Unknowns far from any current objective are deferred. This prevents
exploration paralysis in environments with many unclassified cells.

---

### Rule Degradation and Subsumption

When a higher-level rule (Level 2) is created from a lower-level rule
(Level 1), both are retained. They coexist under a priority ordering:

```
priority:  Level 3 > Level 2 > Level 1 > Level 0
```

The highest-level applicable rule fires first. Lower-level rules serve
as **fallbacks**: if the abstract rule's confidence drops (e.g., the
concept binding broke in a new level), the concrete rule reactivates
while the concept re-learns.

Rules that have not fired successfully in N steps are **degraded**: their
priority is lowered, reducing trigger frequency. Degraded rules are not
deleted — they remain available for reactivation. A rule is **pruned**
(permanently removed) only when a higher-level rule strictly dominates it
in both evidence count and confidence over a sustained period.

This ensures that the rule base grows in abstraction over time but never
loses hard-won concrete knowledge prematurely.

---

## LLM as the Ontology Engine

Traditional AI systems rely on **fixed ontologies** — hand-crafted
hierarchies of concepts (obstacle → wall, door, fence; container → box,
cup, bucket) that define the vocabulary of the system. Fixed ontologies
are brittle in three well-known ways:

1. **Incompleteness.** No designer can anticipate every concept a system
   will encounter. A robot trained with an ontology that includes "wall"
   but not "curtain" cannot represent a soft barrier.

2. **Rigidity.** Splitting or merging concepts requires modifying the
   ontology definition — typically a code change, a schema migration,
   or a knowledge-engineering session. The system cannot adapt to novel
   distinctions at runtime.

3. **Domain lock-in.** An ontology built for household robotics is useless
   for chemistry, game-playing, or medical diagnosis. Cross-domain
   transfer requires building a new ontology from scratch.

The StateStore avoids all three by **using the LLM itself as the ontology
engine.** Concepts are not defined in code or configuration. They are
facts in the StateStore, created and managed by LLM calls at runtime:

```
@(concept, "wall", "parent")           = "obstacle"
@(concept, "wall", "discriminator")    = "blocks movement, immovable"
@(concept, "pushable", "parent")       = "obstacle"
@(concept, "pushable", "discriminator")= "blocks movement, moves when pushed"
```

These facts are written by the OBSERVER (when it labels objects in a frame)
or by the MEDIATOR (when it resolves ambiguity or names a concept split).
The rule engine's tool traverses concept facts mechanically — find children,
check parent, match discriminator. But the *creation* and *naming* of
concepts is always an LLM operation.

#### Why this matters

**No hardcoded vocabulary.** The system starts with zero concepts.
The OBSERVER examines the first frame and writes initial concept facts
based on what it sees. In an ARC game, it might write "player," "wall,"
"goal." For a home robot, it might write "person," "table," "mug."
For a chemistry simulation, it might write "molecule," "catalyst,"
"solvent." The StateStore and rule engine don't change — only the
LLM's perception prompt changes.

**Runtime adaptation.** When the concept-split mechanism detects that
"wall" conflates two distinct behaviors, it asks the LLM: "Object X
blocks movement but moved when pushed. Current label is 'wall.' What
should the sub-concepts be called?" The LLM responds with "wall" and
"pushable" (or domain-appropriate equivalents). The concept hierarchy
updates in-place, as ordinary StateStore writes.

**Domain transfer by LLM swap.** The entire domain knowledge of the
system lives in three places: the OBSERVER's system prompt (how to
interpret sensory input), the MEDIATOR's system prompt (how to resolve
ambiguity), and the concept facts these LLMs have written into the
StateStore. To move from game-playing to household robotics, swap the
OBSERVER and MEDIATOR prompts. To move from English-centric concepts
to domain-specialist concepts (e.g., materials science), swap the
underlying LLM. The StateStore, the rule engine, the abstraction
ladder, the tool — all remain identical.

**Graceful degradation.** If the LLM produces a poor concept name,
nothing breaks. The name is a string label for human readability; the
rule engine matches on structure (parent, discriminator), not on the
name itself. A concept called "blocking_thing_1" works identically to
one called "wall." Over time, better LLM calls or MEDIATOR intervention
can rename concepts without invalidating any rules that reference them.

#### Architecture: tool handles volume, LLM handles judgment

The concept hierarchy can grow large. Evaluating hundreds of rules
against thousands of facts every step is a mechanical operation that
should never touch the LLM context window. The division of labor:

| Operation | Executor | Rationale |
|-----------|----------|-----------|
| Rule condition matching | **Tool** (Python) | Iterate facts, check predicates, pure logic |
| Rule effect application | **Tool** (Python) | Write facts to StateStore |
| Confidence arithmetic | **Tool** (Python) | Update evidence counts, degrade unused rules |
| Concept split detection | **Tool** (Python) | Detect confidence drop + discriminating sub-condition |
| Generalization pattern matching | **Tool** (Python) | Structural comparison of rule conditions across contexts |
| Exploration subgoal creation | **Tool** (Python) | Generate subgoal from UNKNOWN fact on critical path |
| **Concept naming / labeling** | **LLM** (OBSERVER) | Requires common sense: "blocks movement" → "wall" |
| **Novel concept recognition** | **LLM** (OBSERVER) | Requires world knowledge: "this looks like a pressure plate" |
| **Ambiguous split naming** | **LLM** (OBSERVER/MEDIATOR) | "Is this a wall or a door?" requires judgment |
| **Strategy / goal prioritization** | **LLM** (Planner) | "Which unknown to explore first?" requires judgment |
| **Initial frame labeling** | **LLM** (OBSERVER) | Label all objects and guess roles from first frame |

The LLM never sees the full rule set. It receives focused, narrow
queries from the tool: "Cell at (15, 20) blocks movement and has color 5.
What concept is this?" or "Object X was labeled 'wall' but it moved when
pushed. What are the two sub-concepts?" This keeps LLM calls small,
fast, and cheap — typically a single sentence in, a single label out.

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
