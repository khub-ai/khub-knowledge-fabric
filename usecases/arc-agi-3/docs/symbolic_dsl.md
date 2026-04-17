# Symbolic Causal-Inference DSL for ARC-AGI-3

**Status:** Draft v0.1 — strawman to be reviewed, contested, and refined.
**Purpose:** The Kaggle ARC-AGI-3 competition disallows the use of commercial LLMs/VLMs. This design is meant to bypass the problem by defining the vocabulary of perceptual primitives, causal primitives,
object roles, and goal predicates over which the offline KF agent reasons,
plans, and self-improves. The DSL is the bet: if it covers the cognitive
primitives a human uses to play unfamiliar grid puzzles, the symbolic layers
can win without any commercial LLM at runtime.
**Last updated:** April 7, 2026

### Two deployment configurations

KF will be shipped in two configurations against the *same* DSL and the
*same* symbolic core. Only the LLM/VLM component swaps.

| Mode | Target | LLM/VLM | Constraints |
|---|---|---|---|
| **Constrained** | Kaggle ARC-AGI-3 competition | Small open VLM (e.g. Qwen3-VL-8B, GGUF Q4) | Air-gapped, ~6 GB VRAM, no network |
| **Frontier** | arcprize.com leaderboard | Claude (Sonnet/Opus) or strongest available frontier model | Internet allowed, no model size limit |

The DSL exists *because* of Constrained mode, but it must not handicap
Frontier mode. Both modes share the symbolic perception, causal tracker,
planner, and rule library. The only difference is which model fills the
"hypothesis proposer" slot defined in Layer 8 — and how much that proposer
is trusted relative to the symbolic tracker.

**Design implication:** every interface between the symbolic core and the
LLM/VLM must be expressible at *both* extremes — a small grammar-constrained
8B model and a frontier reasoner. That means structured I/O, no clever
prompt tricks that only work at scale, and a clean abstraction for "ask the
proposer for a candidate." Layer 8 specifies that contract.

## Design principles

1. **Cognitive primitives, not game mechanics.** "Translate," "match shape,"
   "fill region," "count," "before vs. after" — these are how humans reason
   about novel grid puzzles. Specific game mechanics (push pads, color
   changers, step counters) are *compositions* of primitives, not primitives
   themselves. The DSL stays small by refusing to encode mechanics.
2. **Closed under observation.** Every primitive must be either (a) directly
   observable from `object_tracker.py` outputs or (b) inferable from a finite
   sequence of observed `(state, action, next_state)` transitions. If a
   primitive can't be falsified by an observation, it doesn't belong here.
3. **Compositional.** Hypotheses are *expressions* over primitives, not flat
   labels. "ACTION1 translates objects of color C by (dr,dc) when not blocked
   by walls" is a single hypothesis built from `Action`, `Translate`, `Color`,
   `Predicate`. The Bayesian updater scores expressions, not strings.
4. **Bounded.** Every category below has an enumerated, finite set of values.
   Open-ended natural-language reasoning lives in the *optional* VLM layer,
   not here. Bounded means enumerable means searchable.
5. **Graceful degradation.** When the symbolic layer fails to form a confident
   hypothesis, the agent should *know it has failed* (high posterior entropy)
   rather than confidently emit garbage. Information-gain exploration and
   the LLM tiebreaker both depend on this.

## Layer 1 — Perceptual primitives

Things that can be extracted from a single frame, deterministically, by the
existing `object_tracker.py` (possibly with extensions). No interaction or
history required.

### 1.1 Object

The atomic unit. Already produced by `detect_objects`.

```
Object {
    id:         int                 # stable across frames via tracking
    color:      int                 # palette index, 0-19+
    cells:      set[(r,c)]          # connected component
    bbox:       (r0, c0, r1, c1)
    centroid:   (rf, cf)            # float
    area:       int
    is_background: bool
}
```

### 1.2 Spatial relations

Binary predicates over pairs of objects, computed once per frame.

| Relation | Definition |
|---|---|
| `adjacent(a, b)` | any cell in `a` 4-touches any cell in `b` |
| `contains(a, b)` | `b.bbox` strictly inside `a.bbox` and `a` forms a closed boundary |
| `aligned_h(a, b)` | centroids share a row band (within ±1 cell) |
| `aligned_v(a, b)` | centroids share a column band |
| `inside_region(a, R)` | all cells of `a` lie in named region `R` |
| `distance(a, b)` | Manhattan distance between centroids (integer-rounded) |
| `same_shape(a, b)` | shapes equal under one of D4 symmetries (use `compare_shapes`) |
| `same_color(a, b)` | `a.color == b.color` |
| `same_size(a, b)` | `a.area == b.area` |

### 1.3 Region primitives

Named subdivisions of the frame, derived from structural cues.

| Region | How detected |
|---|---|
| `border` | outermost ring of cells |
| `interior` | complement of border |
| `bounded_box(id)` | maximal rectangular region enclosed by a single-color boundary |
| `slot_strip(id)` | linear arrangement of equally-sized bounded boxes (puzzle slots) |
| `reference_pair(left, right)` | two bounded boxes side by side, possibly with different border colors |
| `quadrant(NW|NE|SW|SE)` | the four geometric quadrants |
| `connected_open_space` | maximal connected region of background color |

Regions are *findable*, not *labelable*. The DSL only knows that a
`bounded_box` exists; it does not know the box "means goal." That's a
hypothesis, not a primitive.

### 1.4 Visual gestalt features

Single-frame patterns that are not strictly objects or regions but matter for
goal inference.

| Feature | Definition |
|---|---|
| `symmetry_axis(axis, score)` | frame is approximately symmetric across `axis` ∈ {h, v, d, ad} |
| `repetition(period_r, period_c)` | a pattern tiles the frame with given period |
| `unique_cell` | a cell whose `(color, neighborhood)` tuple is unique in the frame |
| `progress_bar` | a contiguous strip of one color whose length varies across steps |
| `counter_digits` | small numeric-looking shapes in a fixed location |
| `miniature` | a small region whose downsampled content matches a region elsewhere |

## Layer 2 — Causal primitives (transition vocabulary)

The vocabulary used to *explain* what an action did. Every observed
transition `(s_t, a_t, s_{t+1})` is fitted to one or more of these primitives.
Hypotheses are conjunctions of primitives with optional preconditions.

### 2.1 Object-level effects

| Effect | Schema |
|---|---|
| `Translate(obj, dr, dc)` | object's cells shift uniformly by `(dr,dc)` |
| `Rotate(obj, k)` | object's mask rotates 90°·k around its centroid |
| `Reflect(obj, axis)` | mirrored across `axis` ∈ {h, v} |
| `Recolor(obj, c_old, c_new)` | every cell of `obj` changes color |
| `Resize(obj, ds)` | uniform scale by `ds` (rare; track separately) |
| `Spawn(c, region)` | a new object of color `c` appears inside `region` |
| `Destroy(obj)` | object disappears |
| `Toggle(obj, attr)` | a discrete property (color, shape) flips between two values |
| `Swap(obj_a, obj_b, attr)` | two objects exchange `attr` |
| `Merge(obj_a, obj_b)` | two objects combine into one |
| `Split(obj)` | one object becomes two |
| `NoOp` | nothing changed |

### 2.2 Frame-level effects

| Effect | Schema |
|---|---|
| `RegionFill(region, color)` | a region is uniformly recolored |
| `CounterTick(obj, delta)` | a numeric/length object changes by `delta` |
| `LevelAdvance` | pseudo-effect: env reports level completion |
| `StateReset` | frame returns to a previously seen state |

### 2.3 Preconditions

Effects can be conditional. The hypothesis tracker maintains preconditions
because that's how you distinguish "ACTION1 always moves X right" from
"ACTION1 moves X right *unless* a wall blocks it."

| Precondition | Schema |
|---|---|
| `IfRole(obj, role)` | only fires for a particular inferred role |
| `IfAdjacent(obj, other)` | only fires when `obj` is adjacent to `other` |
| `IfInside(obj, region)` | only fires when `obj` is in a region |
| `IfFocused(obj)` | only fires for the currently focused/cursor-selected object |
| `IfCounter(obj, op, value)` | only fires while a counter satisfies a comparison |
| `IfBlocked(obj, dir)` | otherwise the effect is suppressed |
| `Otherwise(NoOp)` | default precondition for unobserved cases |

### 2.4 Action models

The full hypothesis about what an action does:

```
ActionModel {
    action:        ActionId
    effects:       list[(Precondition, Effect)]   # ordered, first match wins
    posterior:     float                          # 0.0–1.0
    support:       int                            # observations consistent with it
    contradictions:int                            # observations inconsistent
}
```

Multiple `ActionModel`s can coexist for the same action with different
posteriors — the tracker maintains a distribution, not a point estimate.

## Layer 3 — Object roles

Roles are *inferred labels* on objects, derived from their causal behavior
plus visual cues. Roles are intentionally generic (not game-specific).

| Role | Inference cue |
|---|---|
| `agent` | object whose state most reliably changes in response to actions |
| `obstacle` | object that blocks `agent` movement (causes `IfBlocked`) |
| `target` | object whose state must change for `LevelAdvance`; often visually distinct |
| `trigger` | object whose contact with `agent` causes a non-translate effect |
| `container` | object satisfying `contains(self, other)` for some inner object |
| `key` | object whose `Destroy` precedes a `Recolor`/`Destroy` of another object |
| `counter` | object exhibiting `CounterTick` over time |
| `progress_indicator` | counter that monotonically approaches a fixed value |
| `cursor` | object that translates without changing other objects, used for focus |
| `slot` | object inside a `slot_strip` whose content can be modified |
| `reference` | object inside a `reference_pair` whose content does not change |
| `goal_state` | a `reference` interpreted as the desired state of a `slot` |
| `decoration` | object with no observed causal relationships (default fallback) |

Each role assignment carries a confidence score updated by the Bayesian
tracker. An object can hold multiple roles with different confidences.

## Layer 4 — Goal predicates

The agent doesn't know the goal, so it maintains a *ranked list* of goal
hypotheses. Each is a predicate over frames; planning targets the top one but
the agent re-ranks as evidence accumulates.

| Predicate | Schema | Example |
|---|---|---|
| `Reach(agent, target)` | agent's position equals target's position | move player to door |
| `Match(slot, goal_state)` | slot's content equals goal under some transform | rotate slot to match reference |
| `MatchAll(slots, goal_states)` | every slot in a strip matches its reference | finish all slots |
| `Cover(region, color)` | every cell in region has color | paint the floor |
| `Eliminate(role)` | no objects of a given role remain | clear all enemies |
| `EqualCount(role_a, role_b)` | counts of two roles become equal | balance the scales |
| `Sequence(events)` | a temporal sequence of events occurs in order | press buttons in order |
| `BeforeEqualsAfterUnder(transform)` | the goal is to apply `transform` to the frame | recolor the whole grid |
| `CounterReaches(counter, value)` | a counter reaches a target value | timer to zero |
| `PatternComplete(pattern)` | a partially-shown pattern is filled in | complete the symmetry |

Goals are scored by:
- **Visual plausibility prior** — does the frame *look like* this kind of puzzle? (boosted by gestalt features)
- **Behavioral consistency** — does observed reward / `LevelAdvance` correlate with the predicate becoming true?
- **Achievability** — does the planner believe this is reachable from the current state under the current `ActionModel`s?

## Layer 5 — Hypothesis tracker (Bayesian updater)

Maintains posteriors over:
- one or more `ActionModel` per `ActionId`
- a role assignment per `Object` (multinomial over roles)
- a ranked list of `GoalPredicate` candidates

Update rule (informal):

```
On observing transition (s, a, s'):
  1. For each ActionModel of a:
       if effects predict s' from s:  support += 1, posterior ∝ support / (support + contradictions)
       else:                          contradictions += 1
  2. If no model predicts s', spawn new candidate models by enumerating
     plausible (Precondition, Effect) compositions consistent with the diff.
     Cap the number of candidates per action; prune lowest-posterior tail.
  3. Update role posteriors using behavior signatures (e.g. "object that
     translated under ACTION1" gets +evidence for `agent`).
  4. Re-score goal predicates against s' (have any become true? false?).
```

Two computational tricks make this tractable:

- **Hypothesis space is enumerated lazily**, not eagerly. Only generate
  candidate `ActionModel`s when the current ones fail to explain a transition.
- **Sampling, not enumeration, when the hypothesis space is large.** When
  precondition combinatorics explode, sample from the prior over preconditions
  rather than enumerating.

## Layer 6 — Planning under the inferred model

Given the current best `ActionModel`s and the top-ranked `GoalPredicate`, the
planner runs in one of two modes:

### 6.1 Explore mode (high posterior entropy)

Pick the action whose expected outcome distribution most reduces hypothesis
entropy. Concretely, for each candidate action `a`:

```
EIG(a) = H(hypotheses) - E_{s'~P(s'|s,a,hypotheses)} [ H(hypotheses | s') ]
```

Approximate by sampling a small number of hypotheses, simulating each forward
under `a`, and measuring how much the resulting `s'` would discriminate them.
Pick the `a` with maximum `EIG`. This is the principled version of "try things
to learn what they do."

### 6.2 Exploit mode (low posterior entropy, confident goal)

Run BFS or MCTS over the inferred world model:
- **State:** abstract state derived from objects + roles + counter values.
- **Actions:** the discrete `ActionId`s, plus (for pixel-control games) a
  small set of *meaningful click targets* derived from `object_tracker`
  affordances. **Never plan over raw pixels.**
- **Transition:** apply the highest-posterior `ActionModel` for the chosen
  action. If multiple models have similar posterior, plan robustly (choose
  actions whose outcome is the same across all top-k models).
- **Goal test:** the top `GoalPredicate` evaluated on the predicted state.
- **Heuristic:** distance under the predicate (e.g. number of unmatched slots).

### 6.3 Mode switching

A simple controller:

```
if entropy(action_models) > θ_explore:        explore
elif goal_confidence < θ_goal:                explore (information about goal)
elif planner finds plan within budget:        execute plan
else:                                         explore (planning failed)
```

## Layer 7 — Persistent rule library (self-improvement)

After every episode, promote confirmed hypotheses into a persistent rule
library scoped by *signature*, not by game name (the agent doesn't know game
names in competition).

A signature is a hash of:
- Number and roles of objects detected
- Set of region types present
- Set of gestalt features present
- Number of available actions

Rules are not "if game == LS20 then …". They are "if a frame contains a
slot_strip and a reference_pair and 4 actions, then ACTION1 historically
behaves as a Translate(cursor) primitive — try this first." This generalizes
across unseen games that share structural signatures.

Rule library operations:
- **Promote:** when a hypothesis reaches posterior > 0.9 and survives an
  episode without contradiction, save it with its signature.
- **Match:** at episode start, retrieve all rules whose signature is similar
  to the current frame; use them as priors for the hypothesis tracker.
- **Demote:** if a rule is contradicted in a new episode, decay its weight.
- **Prune:** drop rules whose weight falls below a floor.

The rule library *is* the self-improvement artifact. It is a few hundred KB
of JSON, fully interpretable, fully diffable.

## Layer 8 — LLM/VLM interface contract

This layer specifies how the symbolic core talks to *whichever* LLM/VLM is
mounted — small open VLM in Constrained mode, frontier model in Frontier
mode. The contract is identical in both modes; only trust weights and
invocation frequency differ.

### 8.1 The "proposer" abstraction

The LLM/VLM is treated as a single role: **Proposer**. The Proposer never
selects actions, never updates state, and never owns any persistent memory
of its own. It exposes one method:

```
propose(request: ProposalRequest) -> ProposalResponse
```

A `ProposalRequest` always contains:
- The current frame as a rendered PNG (image input).
- The symbolic perception summary: objects, regions, gestalt features, all
  expressed in DSL vocabulary. This is what `object_tracker` already produces,
  serialized to a compact text block.
- The current top-k hypotheses from the tracker (action models, role
  assignments, goal predicates) with their posteriors.
- The recent action effect history (last N transitions, in DSL vocabulary).
- A `request_kind` field selecting one of a small set of question types:
  `BIND_ROLES`, `RANK_GOALS`, `PROPOSE_ACTION_MODEL`, `PROPOSE_PRECONDITION`,
  `EXPLAIN_STALL`. Each kind has its own typed response schema.

A `ProposalResponse` is *always* a DSL instance — never free-form prose.
Roles are drawn from the enumerated role set; goal predicates are drawn from
the enumerated predicate list; effects are drawn from the enumerated effect
types. The schema is enforced at decode time, not at parse time.

This means: in Constrained mode the response is produced via grammar-constrained
decoding (`llama.cpp` GBNF, `outlines`, or `lm-format-enforcer`). In Frontier
mode the response is produced via tool/structured-output APIs (Anthropic's
tool use, OpenAI's structured outputs). The Proposer code path branches on
backend, but the request and response *types* are identical.

### 8.2 Request kinds

| Kind | Trigger | Response schema |
|---|---|---|
| `BIND_ROLES` | New level entered, hypothesis tracker has no priors | Map of `object_id → (role, confidence)` over the enum |
| `RANK_GOALS` | Goal posterior is uniform across top-3 candidates | Reordered list of goal predicates with rationale slot per item |
| `PROPOSE_ACTION_MODEL` | An action's current models all contradict the latest transition | A new `ActionModel` expression to add to the candidate pool |
| `PROPOSE_PRECONDITION` | An action's effect is intermittent and tracker can't find a precondition | A `Precondition` expression (constrained to enum) |
| `EXPLAIN_STALL` | Planner has been unable to find a plan for K cycles | A short structured diagnosis: `{suspected_missing_primitive, suggested_exploration_target}` |

Each kind has a strict schema, a small max-token budget, and a fallback path
if the response is invalid (Constrained mode: re-decode with stricter
grammar; Frontier mode: retry once, then fall back to symbolic default).

### 8.3 Trust weights and invocation policy

The Proposer's output is *never* committed to state directly. It enters the
hypothesis tracker as one or more new candidates with a **prior weight**
that depends on the mode:

| Mode | Prior weight | Invocation cadence | Stall threshold |
|---|---|---|---|
| Constrained | 0.3 (treated as one candidate among many) | Only when symbolic stalls | High — let symbolic try first |
| Frontier | 0.6 (treated as a strong prior) | Proactively at level start, plus on stalls | Low — invoke early and often |

The asymmetry is deliberate. In Constrained mode the small VLM is
hallucination-prone and should not steer the tracker; the symbolic layer
does the heavy lifting and the VLM is a tiebreaker. In Frontier mode the
commercial model is good enough to be trusted as a source of strong priors,
which speeds up convergence dramatically — but its proposals are *still*
validated against observed transitions and can still be overridden by
contradictory evidence. The architecture never treats any model output as
ground truth.

### 8.4 What the Proposer is NOT allowed to do

- Select an action. Action selection is the planner's job. The Proposer can
  *suggest* a `promising_actions` list, which the planner uses as a tiebreaker
  among equally-good plans, but never as the plan itself.
- Mutate the rule library. Only confirmed hypotheses (posterior > 0.9 after
  validation) get promoted to rules. The Proposer cannot write rules.
- Output tokens outside the schema. Even in Frontier mode, structured-output
  APIs are mandatory. No free-form completions.
- See the rule library directly. Rules are applied as priors *before* the
  Proposer is called and as evidence *after*; the Proposer only sees the
  current hypothesis state, not the historical knowledge base. This keeps
  the Proposer stateless and the rule library auditable.
- Carry conversation history across cycles. Each `propose()` call is
  independent. Persistent learning happens in the rule library, not in
  context.

### 8.5 Fallback behavior when the Proposer is unavailable

The symbolic layers must be able to run with the Proposer disabled entirely.
This is the *true* Constrained-mode floor: even if the small VLM fails to
load, fails grammar validation repeatedly, or runs out of compute budget,
the agent still plays — slower and dumber, but it plays. Concretely:

- `BIND_ROLES` falls back to a behavioral classifier: assign roles based on
  observed action responses (the object that moves under most actions =
  agent, the object that disappears on contact = key, etc.).
- `RANK_GOALS` falls back to a visual-prior heuristic: prefer goal
  predicates whose visual fingerprint matches gestalt features detected in
  the current frame.
- `PROPOSE_ACTION_MODEL` falls back to enumeration over the DSL effect set,
  ranked by Occam's razor (fewer preconditions first).
- `PROPOSE_PRECONDITION` falls back to enumeration over the precondition set.
- `EXPLAIN_STALL` falls back to "increase exploration temperature."

This guarantees the Constrained-mode shipping artifact has a hard floor on
behavior even in the worst case, and that the Frontier-mode artifact can be
A/B tested with the Proposer ablated to measure its actual contribution.

### 8.6 Why this contract works in both modes

A strong frontier model and a small open VLM are *very* different in raw
capability, but the DSL contract reduces the gap by changing the question
each is asked. Both are asked to *fill in a structured form* over a fixed
vocabulary, not to *reason in prose*. Frontier models lose almost nothing
from this constraint (their reasoning still happens internally; they just
emit it as a typed object). Small models gain enormously, because grammar
decoding turns "produce coherent reasoning" into "select among enumerated
options," which is a task they're competent at.

The result is one architecture, two capability tiers, and a clean
attribution story: any difference in benchmark performance between
Constrained and Frontier modes is *exactly* the contribution of the
proposer. The symbolic core is held constant.

## Coverage check — does this cover ARC-AGI-3?

A self-test for the DSL: can we express, *as compositions of primitives*, the
mechanics we already know about?

- **LS20 navigation puzzles:** `agent` (player), `obstacle` (walls),
  `target` (goal cells), `Translate` effects with `IfBlocked`, goal =
  `Reach(agent, target)` or `Match(state, goal_state)`. ✅
- **TR87 transformation puzzles:** `slot_strip`, `reference_pair`, `cursor`,
  `Rotate`/`Recolor` effects with `IfFocused`, goal = `MatchAll(slots, refs)`. ✅
- **Hypothetical: counter-driven puzzles.** `counter` role, `CounterTick`
  effect, goal = `CounterReaches(counter, value)`. ✅
- **Hypothetical: pixel-mouse games.** Affordances become "click on
  detected object/region centroid." Action space stays bounded. ✅
- **Hypothetical: rule-rewriting games (level n changes what action k does).**
  Handled by `ActionModel` posteriors decaying as contradictions accumulate;
  the tracker spawns new models. Works in principle but is the slowest case
  to converge — this is where the VLM tiebreaker earns its keep.

Known weak spots — primitives we may need to add:

- **Modular arithmetic on positions.** "Object teleports from edge to opposite
  edge" (toroidal worlds) is not in the current `Translate` primitive.
- **Non-grid logic.** If a game introduces continuous angles or non-cell
  positions, the entire DSL is at risk. Probably a bridge too far for v0.1.
- **Multi-agent / NPC behavior.** No primitive currently models objects that
  move on their own between actions. Could add `AutonomousMotion` if observed.
- **Numeric/arithmetic goals.** "Make the sum of red cells equal 7" — would
  need a `Sum`/`Equation` predicate family.

## Open questions for review

1. **Is the role taxonomy too narrow?** The 13 roles above are a guess at
   what's sufficient. If competition games introduce roles outside this set,
   the tracker will misclassify silently. Do we want a `meta_rule` role for
   "this object changes the meaning of other objects"?
2. **How aggressive should hypothesis pruning be?** Keeping too many candidates
   makes information-gain computation expensive; keeping too few risks
   pruning the truth before it has support. Suggest a hard cap of 8 models
   per action initially.
3. **Should signatures be exact or fuzzy?** Exact signatures rarely match
   across games; fuzzy signatures risk applying wrong rules. Suggest:
   structural features as exact, count features as bucketed.
4. **What's the right default `θ_explore`?** Too low and the agent never
   explores; too high and it never exploits. Probably needs per-game
   self-tuning based on observed convergence rate.
5. **Does the VLM tiebreaker get to see history?** If yes, it's stronger but
   prompts blow up. If no, it's a cleaner abstraction. Suggest: history-free
   v0, add history if v0 underperforms.

## Implementation order

1. **Type definitions** for all primitives in a single `dsl.py`. Pure
   dataclasses, no logic. ~1 day.
2. **Hypothesis tracker** with naive enumeration over a small subset of
   primitives. Validate on synthetic transitions. ~2–3 days.
3. **Synthetic game generator** that samples random `ActionModel`s and
   `GoalPredicate`s and produces playable mini-games. This is the test
   harness for everything below. ~2 days.
4. **Information-gain explorer** as MEDIATOR replacement. Validate it
   converges faster than random on synthetic games. ~2 days.
5. **Goal-directed planner** (BFS over the inferred model). ~2 days.
6. **Persistent rule library** with signature matching. ~1 day.
7. **VLM tiebreaker integration** — last, only after layers 1–6 work on
   synthetics. ~1–2 days.

The order matters: every layer is testable on synthetic games drawn from the
DSL itself before any real ARC-AGI-3 frame is touched. If a layer can't beat
random on synthetics, no amount of real-game tuning will save it.

## Non-goals

To keep the bet honest, the DSL deliberately does **not** try to:

- Encode any specific game's mechanics by name.
- Learn primitives from data — the primitive set is fixed and human-authored.
- Compete with frontier LLMs on open-ended visual reasoning. The VLM
  tiebreaker handles cases the DSL can't.
- Be elegant. Coverage matters more than parsimony. Adding a primitive
  because a competition game needed it is fine.

---

**Next actions for the human reviewer:**

- Read the role list and goal predicate list and tell me what's missing from
  *human* gameplay strategies that you've observed in your runs.
- Confirm or override the "no game-specific primitives" rule. If competition
  rules let us see the game family in advance, we can be more aggressive.
- Decide whether the VLM tiebreaker is in scope for v1 or deferred to v2.
- Approve the implementation order, or re-order based on which layer you're
  least confident about (it should be built first to retire risk early).
