# Cognitive Engine — Design Specification

**Status:** Phase 3 complete (goal forest + planner + explorer).  Episode runner / miners / adapters in progress.
**Scope:** Domain-agnostic symbolic reasoning substrate shared by
sequential-reasoning benchmarks (ARC-style interactive environments)
and embodied robotics.

---

## 1. Why this design

An earlier attempt at a Cognitive OS accumulated game-specific
control flow over many debugging sessions: flags for required-visit
counters, ring refuels, win-gate unblocking, lethal-cell persistence,
candidate-goal registries.  The code drifted into implicit encoding
of one specific level's mechanics.

That falsified the Cognitive OS thesis — *"write cognitive logic once,
deploy across domains by swapping adapters"* — at the code level.
Every mechanic discovered during debugging was implemented as an
imperative flag in the controller loop rather than as a discovered
fact in a hypothesis store, so the logic was implicitly tied to the
game it was written against.  That earlier code has now been retired.

The new engine is a clean-slate rewrite with two invariants:

1. **No domain-specific code in the engine.** ARC, robotics, or any
   other domain's mechanics are expressed either as *discovered
   hypotheses* in a generic store, or as *adapter-side capabilities*
   exposed through generic protocols.
2. **The engine is a learning system that accumulates knowledge
   across episodes.** Hypotheses, learned macro-actions (Options),
   and strategy heuristics persist at scope `GAME` or broader and
   carry forward.  The system must demonstrably become a better
   player over time.

These two invariants are standing directives enforced at code review.

---

## 2. Architecture at a glance

```
┌──────────────────────────────────────────────────────────────┐
│  COGNITIVE ENGINE   (core/cognitive_os/engine/, domain-agnostic)          │
│                                                              │
│  ┌───────────────┐   ┌──────────────┐   ┌────────────────┐   │
│  │ HypothesisStore│  │   Planner    │   │   Explorer     │   │
│  │  + Refinement  │◀─▶  AO* search  │◀─▶ info-gain +    │   │
│  │                │  │  AND-OR-CHAN │   │   curiosity    │   │
│  └────────▲───────┘  └──────▲───────┘   └────────▲───────┘   │
│           │                 │                    │           │
│  ┌────────┴─────────────────┴────────────────────┴───────┐   │
│  │                    WorldState                         │   │
│  │ (entities, hypotheses, rules, goals, tools, options)  │   │
│  └────────▲──────────────────▲──────────────────▲───────┘   │
│           │                  │                  │           │
│  ┌────────┴───────┐  ┌──────┴──────┐   ┌────────┴─────┐    │
│  │    Miners      │  │EpisodeRunner │   │ PostMortem   │    │
│  │  (pattern gen) │  │  main loop   │   │   analyser   │    │
│  └────────▲───────┘  └──────▲───────┘   └──────────────┘    │
└───────────┼─────────────────┼──────────────────────────────┘
            │                 │
     symbolic events    typed actions            ┌──────────────┐
            │                 │                  │   OBSERVER   │
            │                 │       ◀─────────▶│  (visual LLM)│
┌───────────┴─────────────────┴─────────────┐    └──────────────┘
│      DOMAIN ADAPTER  (usecases/…/)         │    ┌──────────────┐
│  - perception: raw frame → symbolic events │◀──▶│   MEDIATOR   │
│  - action:     Action → domain-native API  │    │ (text LLM)   │
│  - tools:      domain primitives registry  │    └──────────────┘
│  - goals:      seeds primary Goal          │
└────────────────────────────────────────────┘
```

Three seams cross the engine/outside boundary:

1. **Adapter** — translation of raw observations to symbolic events,
   and of generic actions back to domain-native commands.
2. **Observer** — visual / perceptual oracle for appearance-based
   questions the engine cannot answer symbolically.
3. **Mediator** — common-sense / world-knowledge oracle for impasses,
   cold starts, unexplained surprises, and tool proposals.

The engine itself never inspects pixels, never calls an LLM directly,
and never contains any branching over game-specific identifiers.

---

## 3. Core data types

All types live in `core/cognitive_os/engine/` and are re-exported from
`core/cognitive_os/engine/__init__.py`.

### 3.1 Observation — the adapter's per-step output

```python
Observation(step, agent_state, events, entity_snapshots,
            raw_frame, metadata)
```

* `events` is a typed list (`AgentMoved`, `AgentDied`, `ResourceChanged`,
  `EntityStateChanged`, `GoalConditionMet`, `EntityAppeared`,
  `EntityDisappeared`, `SurpriseEvent`).  The closed vocabulary forces
  the adapter to discretise continuous domain events into symbolic ones.
* `raw_frame` is retained for the Observer to inspect on demand, but
  the engine itself never reads it.

### 3.2 WorldState — the engine's running model

```python
WorldState(
    step, agent, entities, hypotheses, rules, goal_forest,
    observation_history,
    tool_registry, options, pending_tool_calls,
    config
)
```

Updated by the `EpisodeRunner` each step from the incoming Observation.
Readable by every subsystem; writeable only through a narrow update API
that enforces scope rules and credence update disciplines.

### 3.3 Hypothesis and Claim — the learning unit

```python
Hypothesis(id, claim, credence, scope, source,
           supporting_steps, contradicting_steps,
           expires_at, parent_id, child_ids,
           created_at, rationale)
```

The `Claim` inside a hypothesis is one of seven typed forms:

| Claim type | Meaning |
|---|---|
| `PropertyClaim` | `entity.property == value` |
| `CausalClaim` | trigger condition → effect condition (with `min_occurrences`, `delay`) |
| `TransitionClaim` | action × pre-condition → post-condition |
| `RelationalClaim` | binary relation between two entities (appearance, co-occurrence, structural, spatial) |
| `ConstraintClaim` | condition → informal implication |
| `StructureMappingClaim` | partial mapping between two groups of entities with preserved relations and asymmetries (Gentner) |
| `StrategyClaim` | meta-claim: in contexts matching a pattern, a given OR-branch strategy succeeds at rate R |

**Canonical key vs full key.** Every claim exposes two keys:

* `canonical_key()` — structural identity.  Two claims with equal
  canonical keys refer to the *same phenomenon*.
* `full_key()` — exact identity.  Two claims with equal full keys are
  literally identical.

The gap between them is where parameter learning happens.  Three
`CausalClaim`s with the same trigger/effect but `min_occurrences ∈
{2,3,4}` share a canonical key and compete for evidence; one
converges, the others are pruned.

### 3.4 Credence — heuristic evidence accumulation

```python
Credence(point, evidence_weight, last_confirmed,
         last_contradicted, competing)
```

Update rules are deliberately simple (not Bayesian):

* Support: `point += (1 - point) * lr * source_strength`
* Contradiction: `point -= point * lr * strength`
* Decay: after `staleness_window` idle steps, `point` decreases linearly

Thresholds live in `CredenceConfig`; promotion (commit) at
`point >= 0.85` default, pruning (abandon) at `point <= 0.15` default.
Every source (miner, Observer, Mediator, user correction) has a prior
credence in `SourcePriors` — user corrections start at 0.95, LLM
proposals at 0.30.

### 3.5 Rule — externally imposed, principal-weighted

```python
Rule(id, condition, constraint, principal, violability,
     priority, scope, source, expires_at, rationale, created_at)
```

Rules differ from hypotheses in three ways:

1. **Source is authoritative, not evidential.**  They originate from
   operators, safety specs, or the system itself — not from observation.
2. **Compliance is not confidence.**  Rules carry a `Violability`
   enum (`INVIOLABLE`, `DEFEASIBLE`, `ADVISORY`) orthogonal to belief
   strength.
3. **Each rule has a principal** — the entity whose will it represents.
   Conflict resolution uses principal authority and context.

For ARC-AGI-3 Rules are rarely used; for robotics they are central
(operator corrections, safety specs, owner preferences).

### 3.6 Goal — AND-OR-CHANCE tree

```python
Goal(id, root: GoalNode, priority, deadline, source, principal)

GoalNode(id, node_type ∈ {ATOM, AND, OR, CHANCE, OPTION,
                          MAINTAIN, LOOP, ADVERSARIAL, INFO_SET},
         condition, children, ordering, status, priority,
         deadline, active_branch, outcome_priors,
         supporting_hypothesis_ids, source, created_at)
```

The four core node types (`ATOM`, `AND`, `OR`, `CHANCE`) are sufficient
for single-agent MDPs.  The extensions (`OPTION`, `MAINTAIN`, `LOOP`,
`ADVERSARIAL`, `INFO_SET`) are reserved for robotics and multi-agent
scenarios; the planner will add support for them as specific domains
demand.

**Multi-goal support.**  A `GoalForest` holds multiple concurrent goals
with detected `GoalConflict`s (MUTEX / RESOURCE / ADVERSARIAL /
TEMPORAL) and resolution policies.  For ARC-AGI-3 there is typically
one primary goal; robotics commonly has several simultaneous goals
possibly belonging to different principals.

### 3.7 Plan

```python
Plan(goal_id, steps: List[PlannedAction], computed_at,
     assumptions: List[HypothesisId],
     branch_selections: Dict[OrNodeId, ChildId],
     status, current_step_index)

PlannedAction(action, expected_effects, depends_on_hypotheses,
              pre_condition)
```

A Plan is a *selected path* through the AND-OR tree.  The
`assumptions` field lists hypothesis IDs whose truth the plan relies
on; any demotion of those hypotheses invalidates the plan.

### 3.8 Option — learned macro-action

```python
Option(id, name, parameters, internal_plan, applicability,
       expected_effects, success_rate, n_uses, source, scope,
       created_at, rationale)
```

When a plan fragment of ≥3 steps recurs with success across multiple
contexts, the `OptionSynthesiser` miner abstracts it into a parameterised
Option.  The Option joins the action space; future plans can invoke it as
a single step.  Default scope is `GAME`; promotion to `GLOBAL` happens
after cross-game reuse.

### 3.9 CachedSolution — recorded procedural knowledge

```python
CachedSolution(id, task_id, task_parameters, plan,
               recorded_at, n_uses, n_successes,
               deterministic, monitor_level, scope, source, rationale)
```

Represents a recorded, executable sequence that achieves a specific
task.  Structurally one type, two use cases:

1. **Game-level replay (ARC-AGI-3)** — an action sequence that solved
   a specific level.  `task_id` is typically `"arc:<game>:L<level>"`,
   scope is `LEVEL`, `deterministic=True` when the level allows exact
   replay, `monitor_level="low"` for rapid skip-ahead in training.

2. **Procedural / muscle-memory skills (robotics)** — a rehearsed
   motor sequence for a task family such as walking or grasping.
   `task_id` is typically `"robotics:walk_to"` or
   `"robotics:grasp_from_table"`; scope is `GAME` or `GLOBAL` for
   transferable skills; `deterministic=False` (the real world is
   stochastic); `monitor_level="low"` to bypass cognitive overhead
   during well-rehearsed execution.

CachedSolution is distinct from both:

* **Hypothesis** — a hypothesis is a probabilistic belief subject to
  falsification.  A CachedSolution is a recording / rehearsed procedure;
  failures update its success-rate statistic but do not falsify it.
  A single failure of a stochastic solution is expected and tolerated.

* **Option** — an Option is a parameterised macro-action that joins
  the action space and is composed with other actions during
  planning.  A CachedSolution is a complete procedure for a task and
  is invoked as a whole unit with reduced monitoring (see
  `monitor_level`).

**Stochasticity.**  `deterministic=False` marks a solution whose
outcome is not guaranteed even from an identical starting state.  The
Planner then treats the solution as a strong prior that informs
search, not as a guaranteed shortcut.  Applies to any stochastic game
(including some ARC-AGI-3 levels with timing-dependent mechanics) and
to essentially all real-world robotics tasks.

**Monitor levels.**  During replay the executor watches for deviation
with one of three intensities:

* `"low"` — muscle-memory mode; only hard failures interrupt
* `"moderate"` — precondition checks at declared waypoints
* `"full"` — standard planner monitoring per step

**Loading is mode-gated.**  LEVEL-scoped CachedSolutions are loaded
only in `OperatingMode.TRAINING` or `DEBUG`; they are purged in
`COMPETITION` / `EVALUATION`.  GAME- and GLOBAL-scoped CachedSolutions
are loaded in all modes.  See §11.

### 3.10 Tool system

```python
ToolSignature(name, description, input_schema, output_schema,
              cost, typical_latency_ms, determinism,
              side_effects, is_async)

ToolRegistry.register(sig)

ToolInvocation(invocation_id, tool_name, arguments,
               requester, requested_at, callback, timeout_ms, urgency)

ToolResult(invocation_id, success, result, error,
           cost_consumed, latency_ms, completed_at)

ToolProposal(signature, implementation_hint,
             expected_use_case, safety_notes, rationale)
```

Adapter-provided domain primitives (grid BFS, symmetry detection,
motion planning, etc.) are the engine's workaround for the
weakness of LLMs on large-grid and geometric tasks.  Synchronous
tools return results immediately; asynchronous tools return a pending
handle and deliver results via callback or pull-queue.  The Planner
uses `typical_latency_ms` to budget real-time expenditure; the
Runtime memoises deterministic-tool results.

### 3.11 Observer and Mediator protocols

Two typed oracle seams, both budgeted via `LLMBudget`:

**Observer** — visual questions about specific frames:

```python
QuestionType ∈ {STILL_SIMILAR, CLASSIFY, COMPARE, DESCRIBE, STRUCTURE_MAP}
ObserverQuery(query_id, question, targets, frames, claim_id, urgency, context)
ObserverAnswer(query_id, result, confidence, explanation)
```

**Mediator** — common-sense guidance given a symbolic summary:

```python
MediatorQuestion ∈ {
    IDENTIFY_ROLES, SUGGEST_MECHANICS, SUGGEST_STRATEGY,
    WARN_HAZARDS, PROPOSE_GOALS, EXPLAIN_SURPRISE, PROPOSE_TOOL
}
MediatorQuery(query_id, question, world_summary, focus_entities,
              focus_goals, surprise, urgency, context)
MediatorAnswer(query_id, proposed_claims, proposed_goals,
               proposed_rules, entity_roles,
               tool_invocations, tool_proposals,
               confidence, explanation)
```

Both oracles produce **typed** outputs; `explanation` fields are free
text for audit only and never enter the decision path.  Mediator
outputs are treated as low-prior evidence: claims enter the
HypothesisStore with LLM-source credence (~0.30) and must still
accumulate support through observation.  Mediator hallucinations
therefore fail naturally — they accumulate no evidence and are
pruned after a few idle steps.

### 3.12 PostMortem — cross-episode learning

```python
PostMortem(episode_id, final_status, final_step,
           goal_outcomes, failed_plans, contradicted_hypotheses,
           surprises, lessons, options_synthesised,
           mediator_usage, observer_usage, total_steps, wall_time_seconds)
```

Runs once at episode end.  Its output is the mechanism by which the
system accumulates knowledge across episodes:

* `lessons` (new `StrategyClaim`s, `ConstraintClaim`s) are written back
  to the hypothesis store at `Scope.GAME`.
* `options_synthesised` are added to the action registry for the next
  episode.
* `contradicted_hypotheses` signature informs the Mediator at the next
  impasse so the same dead ends are avoided.
* `mediator_usage` / `observer_usage` counters drive budget
  recalibration.

---

## 4. Episode lifecycle

```
1. Episode start
   ├── Load persisted knowledge (hypotheses + options at scope GAME or GLOBAL)
   ├── Adapter initialises: populates ToolRegistry, seeds primary Goal
   ├── Observer.full_scan(initial_frame) → preliminary RelationalClaims
   └── WorldState constructed

2. Main loop  (until terminal)
   │
   │  a. Adapter.step_observation() → Observation
   │     Engine.ingest(obs): update agent, entities, fire events to miners
   │
   │  b. Miners scan (event stream + history):
   │     FutilePattern / Surprise / Causal / CoOccurrence / DistalEffect /
   │     GoalPrecondition / Symmetry / StructureMapping / OptionSynth
   │     → propose Hypotheses via HypothesisStore.propose()
   │
   │  c. HypothesisStore reconciles:
   │     deduplicate by full_key / link competitors by canonical_key /
   │     update credences from supporting & contradicting events /
   │     apply staleness decay / run generalise+specialise operators /
   │     promote across commit threshold / prune below abandon threshold
   │
   │  d. Plan validity check:
   │     any plan assumption demoted below commit?  → invalidate plan.
   │
   │  e. Replan decision:
   │     no plan | invalidated | goal preempted | (opt) surprise | (opt) periodic
   │     → Planner.compute(active_goal, committed_hypotheses, rules, tools)
   │     → Plan or None
   │
   │  f. If Plan is None:
   │     Explorer.choose(world) → action from info_gain + curiosity goals
   │     OR Mediator.SUGGEST_STRATEGY / PROPOSE_GOALS (budget permitting)
   │
   │  g. Oracle queries (if triggered this step):
   │     - Re-validate cached visual RelationalClaims → ObserverQuery
   │     - At impasse → MediatorQuery
   │     - On unexplained SurpriseEvent → MediatorQuery.EXPLAIN_SURPRISE
   │
   │  h. Tool invocations resolve (sync return + async callbacks drain)
   │
   │  i. Execute next PlannedAction → Adapter dispatches → domain
   │
   │  j. Loop
   │
3. Episode end (success | failure | timeout | abandoned)
   ├── PostMortemAnalyzer.run(world) → PostMortem
   │     - extract StrategyClaim updates from branch outcomes
   │     - extract ConstraintClaim from repeated failure signatures
   │     - synthesise Options from recurring successful fragments
   ├── Persist hypotheses+options at scope GAME or broader to disk
   ├── Persist Options scoped GAME to the action registry cache
   └── Log mediator/observer usage for budget review
```

---

## 5. Hypothesis lifecycle

### 5.1 Proposal sources

Every Hypothesis enters through a single `HypothesisStore.propose(claim,
source, scope, initial_credence)` pipeline, regardless of origin:

1. **Miners** (continuous, automatic): structural pattern detectors
   over the event stream and observation history.
2. **Observer** (triggered): initial full scan + on-demand visual queries.
3. **Abductive reasoning** (surprise-driven): when a `SurpriseEvent`
   fires, templates generate candidate causal explanations.
4. **Analogy / transfer** (lazy): structurally similar past hypotheses
   (via ConceptRegistry) proposed with reduced prior credence.
5. **Mediator** (on request): common-sense claims proposed at
   impasses, cold starts, or in response to surprises.
6. **User correction** (bounded): NL corrections parsed into Rules or
   high-credence Hypotheses by the Mediator.

Sources differ only in their *priors* — the validation pipeline is
identical.  A Mediator-proposed claim and a miner-proposed claim
compete on the same terms.

### 5.2 Dedup, competition, and commitment

Proposal → `full_key` check:
* If an exact match exists → merge evidence, bump credence.

Proposal → `canonical_key` check:
* If canonically-equivalent claims exist → link via `Credence.competing`.
  Competitors compete for evidence; the store drives one to commitment
  and others below abandon threshold.

Otherwise → add new Hypothesis with source prior credence.

Each step, the store:
1. Scans all hypotheses against new events to find supporting or
   contradicting evidence.
2. Applies credence updates.
3. Applies staleness decay.
4. Promotes claims crossing commit threshold.
5. Prunes claims below abandon threshold.
6. Runs generalisation and specialisation operators (see §5.3).

### 5.3 Generalisation and specialisation

The hypothesis store is a **lattice**, not a flat list.  Each Hypothesis
has `parent_id` (more general) and `child_ids` (more specific).

**Generalisation trigger:** when multiple specific hypotheses with
high credence share a structural pattern, the `GeneralisationMiner`
proposes a more abstract parent claim via anti-unification:

```
PropertyClaim(cell_A, lethal)       ┐
PropertyClaim(cell_B, lethal)       ├─ pattern: cells adjacent to moving entity
PropertyClaim(cell_C, lethal)       ┘

→ proposed: PropertyClaim(entity_pattern=adjacent_to_moving, lethal)
```

If the generalisation confirms on new instances, the parent commits
and the specific children become derived (no longer need independent
credence tracking).

**Specialisation trigger:** when a committed hypothesis is contradicted
by a near-miss observation, the `SpecialisationMiner` diffs the
supporting and contradicting contexts and proposes a guarded
specialisation:

```
Committed:     CausalClaim(step_on(ring) → budget_resets)
Near-miss:     stepped on ring-shape at (14,30) but budget did not reset
Diff features: colour / animation_phase / prior_visits

→ specialisation child: CausalClaim(step_on(ring) ∧ F → budget_resets)
```

Children compete with parent; evidence settles which scope is correct.

Together, these two operators are the **concrete learning mechanism**:
they compress regularities and refine under contradiction.

---

## 6. Planning

### 6.1 AO* over AND-OR-CHANCE

The Planner performs AO* search over the goal tree rooted at the
active `Goal`:

* **ATOM**: leaf; directly achievable via BFS / action sequence using
  committed `TransitionClaim`s.
* **AND**: recurse into each child; cost = sum.
* **OR**: recurse into best child based on:
  * Cost estimate per child
  * `StrategyClaim` heuristics (learned branch preferences)
  * Mediator suggestions if present and recent
* **CHANCE**: recurse into children weighted by `outcome_priors`;
  plan must handle any outcome.

`Rule`s with violability `INVIOLABLE` or `DEFEASIBLE` filter the
action space before search; `ADVISORY` rules contribute cost penalties.

### 6.2 Replan triggers (hybrid policy)

Always-replan (cheap, mandatory):
* No current plan
* Current plan `status == INVALIDATED`
* Current plan `status == COMPLETE`
* Any hypothesis in plan `assumptions` demoted below commit
* Active goal changed

Optional (off for ARC, on for robotics):
* On `SurpriseEvent` (`replan_on_surprise=True`)
* Every N steps (`replan_periodic=True`)

Configured via `PlannerConfig` on `EngineConfig`.

---

## 7. Exploration

The Explorer is invoked when the Planner returns no plan, or when the
active goal is achieved and the episode has budget remaining.

Two drivers, combined:

1. **Information-gain:** prefer actions that would discriminate between
   currently competing hypotheses.
2. **Curiosity:** prefer actions involving entities or action types
   with low claim coverage (few hypotheses about them).

Both drivers are tuned by a **single knob**:
`ExplorerConfig.curiosity_level ∈ [0,1]`.  This knob derives five
downstream parameters coherently via
`ExplorerConfig.from_curiosity_level()`:

| Derived parameter | As a function of `level` |
|---|---|
| `curiosity_threshold` | 0.1 + 0.3 · level |
| `novelty_base` | 0.05 + 0.25 · level |
| `info_gain_weight` | 0.5 + 1.0 · level |
| `idle_boost` | 1.0 + 4.0 · (1 - level) |
| `generate_curiosity_goals` | `level > 0.0` |

At `level = 0`, no curiosity goals are generated at all; the agent
pure-exploits.  At `level = 1`, unknown entities are actively sought
even when progress is possible on the primary goal.  Default for ARC
is 0.3 (mostly exploit, curiosity on impasse); for robotics 0.5.

---

## 8. Oracle discipline

### 8.1 Observer — lazy visual queries

The Observer runs once on the initial frame (`full_scan`) to produce
preliminary `RelationalClaim`s (groupings, similarities).  After that,
calls are triggered by specific conditions:

| Trigger | Query |
|---|---|
| Entity in a group had an `EntityStateChanged` | `STILL_SIMILAR` — still in the group? |
| New entity appeared | `CLASSIFY` — category? |
| Relational hypothesis falsified | `STILL_SIMILAR` on other members of same group |
| Surprise miner flags a specific entity | `COMPARE` — what changed visually? |

Queries are batched per step; budget capped via
`LLMBudget.observer_per_episode`.

### 8.2 Mediator — impasse-driven common sense

Consulted only when code-level reasoning is insufficient:

| Trigger | Question |
|---|---|
| Cold start (unfamiliar game / scene) | `IDENTIFY_ROLES`, `SUGGEST_MECHANICS` |
| Planner exhausts without solution | `SUGGEST_STRATEGY`, `PROPOSE_GOALS` |
| `SurpriseEvent` without local explanation | `EXPLAIN_SURPRISE` |
| New entity types in robotics | `WARN_HAZARDS` |
| Persistent impasse after strategy exhaustion | `PROPOSE_TOOL` |

Budget capped via `LLMBudget.mediator_per_episode`.

### 8.3 Mediator outputs are evidence, not commands

A `MediatorAnswer` never causes an action directly.  Its `proposed_claims`
enter the HypothesisStore with LLM-source prior (~0.30); its
`proposed_goals` are inserted under the relevant parent but compete
with other branches on cost; its `tool_invocations` go through the
normal dispatch pipeline (including cost accounting); its
`tool_proposals` are subject to the adapter's adoption gate.  A
Mediator hallucination accumulates no supporting evidence and is
pruned within a few steps.

---

## 9. Tool system

### 9.1 Registry

Each adapter exposes its primitives through a `ToolRegistry`:

```python
# ARC adapter might register:
grid.bfs_distance(from, to, blocked) → int
grid.find_components(colour_or_predicate) → List[Component]
grid.detect_symmetry(region) → SymmetryReport
grid.diff(frame_a, frame_b) → DiffReport
grid.find_pattern(template) → List[Match]
grid.regions_rotationally_equivalent(a, b) → bool

# Robotics adapter might register:
scene.find_objects_matching(description) → List[Object]
scene.reachable(pose) → bool
scene.path_plan(from, to) → Trajectory     # async, latency ~800ms
scene.affordances(object_id) → List[Affordance]
```

The engine never imports these directly; it invokes by name via
`ToolInvocation`.

### 9.2 Sync vs. async

* **Synchronous** (`is_async = False`): result returned immediately.
  Used for fast deterministic primitives (grid ops).
* **Asynchronous** (`is_async = True`): invocation returns a pending
  handle; result delivered via callback or pull from
  `WorldState.pending_tool_calls`.  Used for slow primitives (motion
  planning).  Planner uses `typical_latency_ms` to budget wall-clock.

### 9.3 Learned tools (Options)

The `OptionSynthesiser` miner scans Plan history for recurring
successful fragments of ≥3 steps.  When K structurally similar
instances are found, it abstracts the varying parts into parameters
and proposes a new `Option`.  Options join the action space for the
next planning call; over many episodes the toolkit grows.

Default Option scope is `GAME`; after N successful cross-game uses
(tracked in `n_uses`), an Option is promoted to `GLOBAL` scope.

### 9.4 Mediator-proposed tools

At persistent impasse, the Mediator may return a `ToolProposal` in
response to `MediatorQuestion.PROPOSE_TOOL`.  The adapter decides
whether to implement (hand-write, sandboxed code-synthesis, human
operator).  Adoption gate:

1. Signature type-checks
2. Implementation passes regression on recorded past observations
3. Safety notes respected (e.g. read-only, size caps)
4. Determinism verified if claimed

Only then is the tool registered.  This is the "Claude-Code-as-tool-
builder" capability, operationalised with discipline.

---

## 10. Cross-episode accumulation

The system is claimed to be a learning system.  The concrete mechanisms:

1. **Hypothesis persistence.**  At episode end, hypotheses scoped
   `GAME` or broader are serialised to disk under a game-keyed
   directory.  The next episode loads them on startup.
2. **Option persistence.**  Synthesised Options at scope `GAME` or
   broader are serialised to the action registry cache.
3. **CachedSolution persistence.**  Level-specific recordings and
   cross-task procedural skills (robotics muscle memory) are
   serialised to a separate file so that competition-mode purging
   can be enforced structurally — see §11.
4. **StrategyClaim persistence.**  Branch-success statistics are
   preserved and guide OR-node selection in future episodes.
5. **Mediator usage analytics.**  Per-question-type call counts drive
   periodic recalibration of source priors and budgets.
6. **Failure signature memory.**  `contradicted_hypotheses` from
   PostMortem feed the Mediator at next impasse to avoid repeating
   dead ends.

Persistence layout (JSON with schema versioning):

```
.tmp/engine_state/<domain>/<game>/
    knowledge.json        ← hypotheses at scope GAME or GLOBAL,
                             Options, StrategyClaims, analytics
    solutions.json        ← CachedSolutions at every scope
```

At episode start, `knowledge.json` is always loaded; `solutions.json`
is loaded with a scope filter keyed on operating mode (see §11).
Transient intra-episode state (step-scoped hypotheses, observation
history, pending tool calls) is never written.

---

## 11. Operating modes — training vs competition

The engine distinguishes two kinds of persisted content and two kinds
of runs.  The cross product is resolved by :class:`OperatingMode`.

### 11.1 Two kinds of persisted content

| Kind | Examples | Legitimate in competition? |
|---|---|---|
| **Knowledge** | Committed hypotheses at scope `GAME`/`GLOBAL`, learned Options, StrategyClaims, procedural skills at scope `GAME`/`GLOBAL` (robotics muscle memory) | ✅ Yes — this is accumulated competence |
| **Level solutions** | `CachedSolution` entries at scope `LEVEL` — concrete action sequences that solved a specific (game, level) | ❌ No — these are answers to the specific problem the competition tests |

The separation is structural, not a policy check: the two kinds live
in separate persistence files and in separate `WorldState` fields
(`hypotheses` / `options` / ... vs. `cached_solutions`), loaded
through a single gate at episode start that reads
`cfg.operating_mode.loads_level_solutions()`.

### 11.2 Four modes

| Mode | Loads knowledge | Loads LEVEL-scoped solutions | Logging |
|---|---|---|---|
| `TRAINING` | ✅ | ✅ | normal |
| `COMPETITION` | ✅ | ❌ | normal |
| `EVALUATION` | ✅ | ❌ | verbose |
| `DEBUG` | ✅ | ✅ | verbose |

`TRAINING` is the default for development and multi-level progress
runs — the agent can skip past previously-solved levels to focus on
the target level.  `COMPETITION` is the default for benchmark
submissions — the agent brings all learned competence but must solve
each level from first principles.  `EVALUATION` matches competition
but enables telemetry.  `DEBUG` matches training but enables
diagnostics.

### 11.3 Robotics extension — muscle memory

The same mechanism supports rehearsed motor skills in robotics.  A
`CachedSolution` with `task_id="robotics:walk_to"`, scope `GAME` or
`GLOBAL`, `deterministic=False`, and `monitor_level="low"` represents
a walking skill that:

* Executes with minimal cognitive oversight (fast, efficient)
* Is treated as a strong prior, not a guaranteed outcome
  (the real world is stochastic)
* Survives all operating modes (competence, not a memorised answer)

A robot operating in a benchmark-evaluation scenario still has access
to its walking, grasping, and manipulation skills — they are
*knowledge*, not *level solutions*.  What gets purged in competition
mode is anything keyed to the specific benchmark task instance.

### 11.4 Audit contract

Because the separation is structural, it is audit-checkable:

1. `cached_solutions.json` is opened at exactly one point in the runtime.
2. That load point is gated by
   `cfg.operating_mode.loads_level_solutions()`.
3. In `COMPETITION`/`EVALUATION`, LEVEL-scoped entries are further
   filtered out in memory before any planner or replay logic sees them.
4. The planner never references a CachedSolution by scope
   classification — scope is enforced at load time, not at use time.

This lets a reviewer confirm competition-mode legality by inspecting
a single loader function rather than scanning the entire codebase.

---

## 12. Claude-Code-style capabilities imparted to the engine

| Claude Code capability | COS mechanism |
|---|---|
| Iterative debugging | Hypothesis credence updates + contradiction tracking |
| Composing solutions | AO* search + Plan synthesis + Option reuse |
| Creating tools | Adapter ToolRegistry + OptionSynthesiser + Mediator `PROPOSE_TOOL` |
| Getting better over time | PostMortem + cross-episode persistence + StrategyClaim learning |
| Experimenting + verifying | Explorer (info-gain) + observation-based falsification |

---

## 13. Phase roadmap

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Data types + protocols (`types.py`, `claims.py`, `conditions.py`, `credence.py`, `config.py`, `tools.py`) | **Complete** |
| 2 | `hypothesis_store.py` (propose, dedup, link, update, prune) + `refinement.py` (specialise, generalise-candidate detection) | **Complete** |
| 3 | `planner.py` (AO* over goal forest) + `explorer.py` (info-gain + curiosity) + `goal_forest.py` | **Complete** |
| 4 | `episode_runner.py` (main loop) + core miners + `adapters.py` protocol + `postmortem.py` | Pending |
| 5 | ARC adapter (new `usecases/<arc-target>/`) + Observer + Mediator implementations | Pending |
| 6 | First integration benchmark (target level TBD) | Pending |
| 7 | OptionSynthesiser + persistence layer | Pending |
| 8 | Robotics adapter (phase-5 of the robotics roadmap) | Future |

---

## 14. Standing invariants (code-review checklist)

These constraints apply to every change under `core/cognitive_os/engine/`:

1. No module under `core/cognitive_os/engine/` may import from `usecases/`, hard-code
   a game/level/task identifier, or branch on a domain-specific tag.
2. No claim, hypothesis, or rule may encode a specific game mechanic
   as a hardcoded type.  Mechanics are *discovered* and stored in
   `HypothesisStore`.
3. No LLM call may occur outside the `Observer` or `Mediator` seams.
4. No free-form LLM text may be parsed for decision-making.  Only
   typed `Claim` / `Goal` / `Rule` / `Tool*` outputs enter the
   decision path.
5. Every new feature must have a generalisation argument: *which
   future domain also benefits from this?*  If the answer is "none",
   the feature belongs in an adapter, not the engine.
6. Cross-episode accumulation must be considered from day one.  A new
   miner that produces hypotheses only useful within one episode is a
   red flag — it likely generalises to something broader that should
   persist.
7. **Capability audit on every phase.**  The engine exists to do
   three things well: *debug* (iteratively refine beliefs from
   contradiction), *problem-solve* (decompose and compose across
   primitives), and *create tools* (compress recurring patterns into
   reusable macros).  Every phase's opening design note must state
   explicitly which of the three it advances and which it defers,
   and no phase may ship without advancing at least one.  The final
   system must embody all three.  This invariant exists because the
   three capabilities are the COS value proposition — deferring them
   silently would let the implementation drift from the thesis.

---

## 15. File layout (current)

The engine lives as a **sub-package under `core/cognitive_os/`** so
that all COS work is grouped together and cleanly separated from
non-COS framework code (`core/knowledge/`, `core/pipeline/`,
`core/benchmark/`, `core/dialogic_distillation/`).

```
core/cognitive_os/                     ← COS namespace
    __init__.py                        ← package stub (no exports)

    engine/                            ← the cognitive engine
        DESIGN.md                      ← this document
        __init__.py                    ← public API
        config.py                      ← EngineConfig + tunable sub-configs
        conditions.py                  ← Condition ABC + 9 subtypes
        claims.py                      ← Claim ABC + 7 subtypes (incl. StructureMappingClaim)
        credence.py                    ← Credence + update rules
        tools.py                       ← ToolSignature / Registry / Invocation / Result / Proposal
        types.py                       ← Events / Observation / Hypothesis / Rule / Goal /
                                         Plan / Option / CachedSolution / PostMortem /
                                         Observer / Mediator / WorldState
        hypothesis_store.py            ← propose / dedup / link competitors / update on events /
                                         staleness decay / prune abandoned / query helpers
        refinement.py                  ← specialise on contradiction /
                                         detect generalisation candidates /
                                         parent-child lattice maintenance
        goal_forest.py                 ← add_goal / select_active / derive subgoals from
                                         CausalClaims / achievement walk / conflict detection
                                         (MUTEX / RESOURCE / TEMPORAL / ADVERSARIAL)
        planner.py                     ← AO* over AND-OR-CHANCE-OPTION trees; Rules filter
                                         (INVIOLABLE/DEFEASIBLE hard-prune, ADVISORY cost
                                         penalty); StrategyClaim branch heuristic;
                                         TransitionClaim-based state simulation
        explorer.py                    ← claim coverage / info-gain / curiosity goals /
                                         fallback action selection when planner exhausted
```

Domain-specific adapters will live under `usecases/<domain>/` when the
first one is implemented (Phase 5).  One name collision exists outside
the `cognitive_os` namespace: `Goal` is also defined in
`core/knowledge/goals.py` (a non-COS framework module); the engine's
`Goal` is in `core.cognitive_os.engine.types` and the two never mix
because they live in distinct packages.

---

## 16. Open design questions

Flagged for resolution in later phases rather than blocking Phase 2:

1. **Hypothesis store indexing.**  For stores of >10,000 hypotheses, a
   secondary index keyed on `canonical_key` avoids O(n) scans during
   dedup.  Defer until empirically needed.
2. **Option parameterisation inference.**  Anti-unification across
   recurring plan fragments requires a distance metric over
   `Condition`s.  Start with feature-overlap; may need refinement for
   robotics continuous spaces.
3. **Principal conflict resolution.**  The code supports multi-principal
   rules; the arbitration algorithm (priority × context × violability)
   needs precise specification before robotics Phase 1.
4. **Persistence format versioning.**  JSON is simple but fragile.
   Consider msgpack or sqlite if store sizes grow.  Add schema
   version field from the start.
5. **Bayesian credence alternative.**  The heuristic update rule is
   sufficient for Phase 1–3.  A Bayesian alternative may be worth
   exploring once enough training data exists to calibrate per-claim-
   type likelihood functions.
