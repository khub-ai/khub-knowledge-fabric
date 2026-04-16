# Robotics Use Case Design Spec

This document is the developer-facing design specification for the robotics use case. The user-facing overview is in [README.md](README.md).

The core design decision is:

**we are not trying to simulate all of robotics. We are trying to evaluate the high-level cognitive layer that KF is supposed to improve.**

That means the environment should be interactive and embodied enough to make planning, state tracking, learning, and recovery meaningful, but abstract enough that we are not dominated by physics and control engineering.

---

## 1. Goal Of The Use Case

Demonstrate that KF can improve the **high-level behavior layer** of an embodied agent by enabling:

- environment learning across episodes
- explicit state and goal tracking
- reusable multi-step procedures
- natural-language human correction
- plan revision after failure
- knowledge accumulation outside the base model

The intended claim is narrower than "KF solves robotics":

**KF improves runtime adaptation, planning, and reusable task knowledge in a robotics-relevant environment without relying entirely on a single multimodal model.**

---

## 2. Why We Should Not Start With A Full Robot Simulator

A full robotics simulator would add:

- physics
- locomotion details
- joint control
- grasp planning
- simulator-specific engineering overhead

Those are important in robotics generally, but they are not the core thing this use case is trying to prove.

KF's contribution is primarily above the controller layer:

- task memory
- environment-specific conventions
- subgoal planning
- exception handling
- procedure reuse
- explicit learning from human correction

So for the first implementation, the environment should be:

- interactive
- stateful
- partially dynamic
- task-oriented
- robotics-relevant

but **not** fully physics-realistic.

---

## 3. Recommended Environment Type

**Plug into an established embodied simulator; do not build one from scratch.** Rolling our own environment is the single biggest credibility leak — reviewers will read it as "symbolic planning with `move_to()` painted on top." Using a simulator the robotics community already trusts removes the toy-environment critique for free and yields directly comparable baselines.

Preferred options, in order:

- **AI2-THOR / ProcTHOR** — partial observability, object state, ~10k procedurally generated layouts (ideal for held-out generalization)
- **ALFWorld / ALFRED** — well-known embodied benchmark with published baselines (SayCan, ReAct, Voyager-style agents) to compare against
- **Habitat 3.0** or **BEHAVIOR-1K** as alternatives when richer scenes are warranted

KF slots into the *planner* layer of the chosen simulator. The domain framing remains warehouse/lab-assistant style tasks (commercially relevant, rich in local conventions), but realized as task suites on top of an established sim rather than a bespoke world.

---

## 4. Environment Design

### 4.1 World elements

The simulated world should contain:

- named locations
- routes between locations
- containers, shelves, stations, bins, or carts
- movable objects
- object attributes such as fragile / sterile / high-priority / requires-verification
- local environment rules that may not be obvious initially

### 4.2 Action primitives

The low-level action layer is abstracted into symbolic actions, but **each action carries embodied friction** so that the benchmark does not collapse into symbolic planning:

- `move_to(location)` — can fail on narrow/congested routes; success rate < 1
- `look(direction)` / `scan(target)` — **explicit sensing actions with a step-budget cost**; the agent cannot freely re-observe
- `inspect(target)` — reveals attributes only after a sensing step
- `pick(object)` — has preconditions (gripper free, payload within limit, pose known within tolerance) that can silently fail
- `place(object, location)` — can fail if target surface occluded or unstable
- `open`, `close`, `verify` — require the target to be currently sensed, not just remembered
- `ask_human(question)` — bounded budget per episode
- `charge()`

Spatial ambiguity is first-class: multiple objects may match a referent ("the tray"), and the agent must disambiguate via sensing rather than assume.

### 4.3 Observations — partial, noisy, stale

The observation model must **not** return an oracle world state. Observations return only what is currently within the sensing cone, with uncertainty:

```json
{
  "location": "inspection_station",
  "visible_objects": [
    {"id": "obj_17", "class": "bin", "class_confidence": 0.82, "pose_sigma": 0.05},
    {"id": "obj_18", "class": "tray",  "class_confidence": 0.61, "pose_sigma": 0.12}
  ],
  "believed_blocked_routes": [
    {"route": "left_corridor", "last_seen_step": 42, "confidence": 0.6}
  ],
  "object_state": {
    "obj_17": {"verified": "unknown", "picked": false, "last_checked_step": null}
  },
  "sensing_budget_remaining": 7
}
```

Key properties:

- **Partial observability**: no global view; the agent only sees what it looked at
- **Stale beliefs**: objects outside the current sensing cone are not refreshed; the agent's belief state can diverge from ground truth between visits
- **Noisy perception**: class confidence and pose sigma are real; occasional misclassifications
- **Uncertain object state**: `verified`, `is_open`, `is_clean` decay to `unknown` after N steps or unrelated activity
- **Action uncertainty**: action outcomes are non-deterministic; failures must be detected from subsequent observations, not from ground-truth return codes

This is where a high-level memory layer has to earn its keep — by deciding when to re-sense, when to trust stale beliefs, and when to ask.

### 4.4 Episode dynamics

Across episodes, the world should change in controlled ways:

- routes become blocked
- objects move
- one station gains a new verification requirement
- a handling constraint is introduced
- a task priority changes

This is where KF should show value.

---

## 5. Candidate Tasks

The first task set should be multi-step and operationally plausible.

Examples:

- fetch an item from storage, verify it, and deliver it to a destination
- retrieve a tray, avoid a blocked route, and use a fallback path
- move a fragile object only after checking a handling condition
- complete a multi-stop delivery in priority order
- recover when an expected object is not present at the learned location

Task success should require more than one step and should benefit from remembering prior experience.

---

## 6. What KF Should Learn

The robotics use case should exercise several artifact types:

- **Environment artifacts**
  - object-location regularities
  - route preferences
  - blocked-path workarounds
  - station-specific conventions

- **Procedure artifacts**
  - repeated task sequences
  - fallback procedures
  - verification-before-transfer procedures

- **Boundary artifacts**
  - fragile handling rules
  - contamination boundaries
  - ask-before-acting rules
  - escalation conditions

- **Judgment artifacts**
  - prioritization logic
  - when to retry vs reroute vs ask for help

Example:

```json
{
  "artifact_type": "boundary",
  "scope": "transfer_from_storage_B",
  "condition": "When moving any item from Storage B to Packing",
  "action": "verify barcode before transfer",
  "rationale": "Storage B contains visually similar bins; transfer without verification is error-prone."
}
```

---

## 7. Proposed KF Architecture Mapping — Closed-Loop Executive

Robotics behavior is not naturally organized as "retrieve once, interpret once, plan once, act once." The ARC-AGI-3 round decomposition is a useful **internal execution scaffold** and a convenient unit for logging and evaluation, but the architecture must be framed externally as a **continuous executive loop** with interruption and event-driven replanning.

### Core loop

```
  perceive → update belief → check constraints → (re)plan → act → monitor
       ↑                                                            │
       └────────── interrupt on: belief change,  ───────────────────┘
                     action failure, operator
                     correction, safety trigger
```

### Roles (not rounds)

- **Retriever** — continuously surfaces relevant artifacts as the belief state and active goals evolve
- **Observer / belief updater** — integrates noisy, partial observations into a persistent belief state; marks stale entries; tracks confidence decay
- **Constraint checker** — evaluates active advisory and executable rules against the current plan *and* against each pending action before it fires
- **Planner / mediator** — constructs and revises a goal-aware plan; not invoked once per episode but on any event that invalidates the current plan
- **Actor / verifier** — executes actions, monitors outcomes against expected effects, and raises preemption events when outcomes diverge
- **Learner** — extracts reusable procedures, environment facts, and validated corrections at episode boundaries *and* opportunistically mid-episode

### Interruption and preemption

Any in-flight action must be preemptible by:

1. **Belief change** — a new observation invalidates a precondition (e.g., the target has moved)
2. **Action failure** — observed effect diverges from expected effect
3. **Operator correction** — a new rule arrives and affects the current or upcoming action
4. **Safety trigger** — a hazard-tagged constraint fires

Replan is **event-driven**, not round-driven. "Round N" is retained only as a logging and evaluation convenience.

---

## 8. Suggested Internal Managers

This use case should lean heavily on:

- **StateManager**
  - current location
  - object states
  - route availability
  - last failed action

- **GoalManager**
  - top-level task
  - subgoals
  - abandoned or blocked subgoals
  - recovery goals

This makes the robotics use case a natural continuation of the ARC-AGI-3 architecture.

---

## 9. Evaluation Plan

### Baselines

- multimodal model or planner with no KF memory
- same agent with short-horizon context only
- same agent with a long-context window (stress-tests whether KF beats "just put everything in context")
- same agent plus raw retrieved notes (RAG baseline)
- `agent + KF`

### Tier 1 metrics — task performance

- task completion rate
- number of steps per successful task
- number of replans required
- number of human interventions
- success after environment change
- reuse rate of learned procedures
- time to improvement after a human correction

### Tier 2 metrics — robotics-grade

These are the questions an experienced robotics reviewer will ask first. Without them, the benchmark can show "improvement" while still being brittle or unsafe.

- **Safety**
  - unsafe action rate (actions that violate a known constraint)
  - hazard-tag violation count (fragile / sterile / high-priority rules)
  - recovery latency after a violation is detected
- **Robustness**
  - success rate degradation curve under increasing observation noise
  - success rate degradation curve under increasing action failure rate
  - sensitivity to stale-belief duration
- **Generalization**
  - held-out layouts (ProcTHOR rooms not seen during learning)
  - held-out object instances
  - held-out rule-change scenarios (train on "avoid wet floor", test on "avoid icy floor")
- **Adaptation efficiency**
  - corrections required to reach threshold performance on a new constraint
  - retention of a correction after N unrelated intervening tasks
- **Correction quality**
  - over-generalization rate (rule fires when it shouldn't)
  - under-application rate (rule should fire and doesn't)
  - audit-trail completeness (fraction of actions traceable to the rules that shaped them)

### Strong first success criterion

A strong first result would show that:

- the KF-enabled agent improves across episodes under partial observability and noisy perception
- the long-context and RAG baselines do not improve comparably — demonstrating that governed, structured memory beats "more context"
- one or more operator corrections become immediately reusable, correctly-scoped artifacts
- task success or intervention count improves without retraining the underlying model
- safety and over-generalization rates do not regress as corrections accumulate

---

## 10. Human Correction Loop — Governed, Not Free

Human correction is the **headline contribution** of this use case: no published embodied-agent system treats operator corrections as scoped, expirable, auditable policy changes. But this only works if the correction loop has real governance. A correction is not memory — it is a behavior change for an embodied system, and in robotics "remember this rule" can move a fragile, sterile, or hazardous object the wrong way.

### Correction lifecycle

```
utterance → parse → scope → dry-run → advisory → (optional) promotion → executable
                                          │
                                          └── expiry / revocation / supersession
```

### Required governance mechanisms

1. **Scope metadata (required on every rule)**
   - `applies_to`: object class, object instance, location, task type, or operator
   - `preconditions`: when the rule is active
   - `expiry`: TTL or revocation event (task complete, shift end, explicit revoke)
   - A correction without a recoverable scope triggers a clarification prompt; it is not silently stored.

2. **Advisory vs executable distinction**
   - Corrections enter the system as **advisory constraints**: they influence planning and are logged, but do not yet block or rewrite actions.
   - Promotion to **executable policy** requires (a) a simulation-first dry run showing the behavior change is what the operator intended, and (b) explicit operator promotion.
   - This prevents a single ambiguous utterance from silently rewriting motion behavior.

3. **Conflict detection and resolution**
   - New rules are checked against existing rules before storage.
   - Conflicts trigger an explicit resolution prompt; last-write-wins is forbidden.
   - Supersession is recorded so the audit trail shows *why* an older rule stopped applying.

4. **Hazard tagging and elevated approval**
   - Objects, locations, and action classes carry hazard flags (fragile, sterile, sharp, high-priority).
   - Any correction that would weaken a hazard-tagged constraint requires an elevated confirmation step.
   - Hazard constraints cannot be expired by a plain correction — only by an explicit hazard-scoped revoke.

5. **Audit log**
   - Every action records the set of rules that influenced it (retrieval trace + constraint trace).
   - Every rule records its origin utterance, scope, promotion history, and revocation.
   - This makes unsafe behavior traceable to the correction that caused it, which is both a safety feature and a debugging feature.

### Example corrections

- "Items from shelf B must be barcode-verified before transfer." → scope: task=`transfer`, applies_to=`shelf_B_items`, advisory until promoted
- "If the left corridor is blocked, use the rear path." → scope: location=`left_corridor`, precondition=`blocked`, can auto-promote (low risk)
- "Do not place sterile tools on the blue cart." → hazard-tagged; requires elevated confirmation; cannot be silently overridden
- "When the target bin is missing, inspect the overflow station before asking for help." → scope: task=`fetch`, precondition=`target_missing`, advisory

This is the part of the use case that is most convincing to a robotics audience, because correction-governance is a real, underserved problem in the embodied-agent industry — and it can be demonstrated end-to-end without a physical robot.

> **Where corrections meet execution.** A promoted correction does not enforce itself —
> it compiles into a `SafetyChecker` that the cognitive-OS **Safety Ensemble** runs against
> every action proposal. See §18 for the action-vetting interface (`ActionProposal` →
> `SafetyVerdict`) that the governance layer plugs into.

---

## 11. Scope Boundary

This benchmark is **task-level knowledge adaptation for embodied agents with symbolic action interfaces** — not robotics writ large.

It should be described as:

- an embodied task-level adaptation benchmark
- aimed at the planner/memory layer that sits above perception and control
- focused on governed correction, belief-state memory, and adaptation under uncertainty

It should **not** be described as:

- a physics benchmark
- a low-level control benchmark
- a perception benchmark
- a claim of direct transfer to physical robots without additional work

### Transfer story

**Expected to transfer** to a real robot stack: the correction-governance model, the belief-state memory structure, the constraint-propagation patterns, and the evaluation harness for adaptation under noise and correction.

**Not expected to transfer**: anything touching contact dynamics, continuous control, grasp planning, or the perception stack itself.

**Integration surface**: KF sits above something like a behavior-tree executor or skill library (MoveIt-style primitives, or a VLA model exposing discrete skills). A real-robot integration would replace the simulator behind the same planner-layer API.

**Optional sim-to-real sanity check**: a tabletop webcam + VLM perception + scripted/Wizard-of-Oz actuation demo shows the knowledge layer survives contact with real perceptual noise. For a planner layer this is the actual sim-to-real risk, and it is within reach without a robot.

---

## 12. Simulation Platform Selection

### The choice: AI2-THOR/ProcTHOR vs NVIDIA Isaac Sim

Two platforms are the strongest candidates. They serve different audiences and different phases of the roadmap — the right answer is to use both in sequence, not to choose one.

### Head-to-head comparison

| Dimension | AI2-THOR / ProcTHOR | NVIDIA Isaac Sim |
|---|---|---|
| Primary purpose | Embodied AI benchmarking | Robot engineering + RL training |
| Physics fidelity | Unity-based, simplified | PhysX, photorealistic, GPU-accelerated |
| Unitree R1/G1 support | None | `unitree_sim_isaaclab` repo, working |
| ROS 2 integration | Minimal | Full, native |
| Hardware requirement | 8GB VRAM, any modern GPU | RTX 4080 min, 16GB VRAM, 64GB RAM |
| Procedural layout library | ProcTHOR: 10,000+ houses | Manual scene authoring, no equivalent |
| Peer-recognized benchmark baselines | Hundreds (SayCan, ReAct, HELPER…) | Robotics engineering community |
| Episode iteration speed | Seconds | Minutes at full fidelity |
| RAI adapter complexity | ~150 lines | ~400 lines + ROS 2 node |
| Action realism / embodied friction | Needs artificial injection | PhysX baked in |
| VLM OBSERVER perception realism | Simplified Unity visuals | Photorealistic |
| Sim-to-real transfer story | Weak | Strong — ROS 2 bridge to SDK2 |
| Getting started time | Days | Weeks |
| Open source | Yes | Yes (since Isaac Sim 5.0, Aug 2025) |
| Windows support | Yes | Yes (rougher than Linux) |

### AI2-THOR / ProcTHOR — key strengths and weaknesses

**Strengths:**
- Hundreds of peer-reviewed papers use it; results are immediately legible to NeurIPS/CoRL/RSS AI planning reviewers without extra framing
- ProcTHOR's 10,000 procedural layouts are the only way to run held-out generalization at scale — the metric that distinguishes real adaptation from memorization
- Python API is trivially simple: `controller.step("PickupObject", objectId="Mug_1")` returns a structured observation dict; the full RAI adapter is ~150 lines
- Low hardware bar means results are reproducible by anyone
- Fast episode cycles (seconds) make it practical to run the hundreds of correction-governance experiments needed for Tier 2 metrics

**Weaknesses:**
- Domestic environments only (kitchens, living rooms); warehouse/lab settings require custom ProcTHOR layout authoring
- No Unitree or humanoid robot model — the "cognitive OS + named shipping robot" story cannot be told here
- Embodied friction (action uncertainty, silent precondition failures) must be injected artificially, not simulated
- Uncertain long-term maintenance trajectory

### Isaac Sim — key strengths and weaknesses

**Strengths:**
- `unitree_sim_isaaclab` (Unitree's own repo) provides working G1/H1 simulation in Isaac Lab today; this is a real named tie-in to a shipping product
- Full ROS 2 integration means the RAI adapter is directly reusable against the real R1 SDK2 with minimal changes — the sim-to-real path for the cognitive layer is genuine
- PhysX physics gives realistic action failure modes and embodied friction without artificial injection
- Photorealistic rendering means VLM OBSERVER perception is closer to real camera input
- NVIDIA GR00T N1.6 VLA integration pattern is documented and used by major robotics teams
- Multi-robot simulation handles heterogeneous robot types in the same scene

**Weaknesses:**
- Hardware requirements are punishing: RTX 4080 minimum, 16GB VRAM, 64GB RAM recommended — most reviewers cannot reproduce results without significant hardware
- Steep learning curve: Omniverse, USD scene composition, complex extension system — getting a basic task running takes days-to-weeks, not hours
- No procedural layout library equivalent to ProcTHOR; every scene must be authored or imported
- Robotics engineering community audience, not the AI planning community that will review KF's first papers
- Windows development is supported but rougher than Linux

### Recommended strategy: sequence, not choose

The two simulators serve different stages of the roadmap and different audiences.

**Phases 0–4 — use AI2-THOR / ProcTHOR:**
- Fast experiment iteration for Phase 2 correction-governance work
- ProcTHOR generalization metrics essential for Tier 2 evaluation
- Peer-legible results for CoRL LangRob / RSS Lifelong Learning workshop
- Low hardware bar for reproducibility

**Phase 5 — add Isaac Sim + unitree_sim_isaaclab:**
- Replace AI2-THOR adapter with Isaac Sim / ROS 2 adapter (~300 lines, no changes to cognitive OS)
- Run Demo C with a real Unitree G1/H1 model in sim
- Targets robotics engineering audience and industry
- ROS 2 bridge directly reusable against real R1 SDK2

**The paper claim:** *"We validated the cognitive OS on AI2-THOR/ProcTHOR (peer-recognized benchmark, 10,000 layouts, direct baseline comparisons). We then ported the same layer to Unitree G1 in Isaac Sim via a 300-line adapter with no changes to the cognitive OS code."* This covers both audiences with appropriate evidence for each.

---

## 13. Implementation Roadmap

The path from the current ARC-AGI-3 codebase to a demo-ready cognitive OS is evolutionary. ARC-AGI-3's StateStore, rule lifecycle, and OBSERVER/MEDIATOR pipeline become the cognitive OS core. Each phase adds one capability layer.

### Phase 0 — Extract and harden the cognitive OS core
**~3–4 weeks | Medium difficulty**

Promote ARC-AGI-3 components that belong in the cognitive OS into a shared `core/cognitive_os/` module, cleaned up and extended for multi-domain use.

- Extend StateStore scope hierarchy: add `session` and `deployment` tiers above `game`; define serialization contract for `persistence="persistent"` facts
- Harden rule lifecycle: add `scope`, `expiry`, and `operator` fields; formalize candidate→active promotion as configurable policy, not hardcoded thresholds
- Generalize OBSERVER/MEDIATOR prompt contract: strip ARC-specific vocabulary; replace game concepts (level, frame, color) with domain-agnostic equivalents
- Extract `cognitive_os/` module: `state_store.py`, `rule_engine.py`, `goal_manager.py`, `episode_logger.py` importable without any ARC dependency; ARC-AGI-3 becomes one client
- Write domain-agnostic integration tests for StateStore confidence resolution, scope expiry, and rule promotion

*Risk: Low. Main risk is over-engineering the abstraction. Err on thin interfaces.*

### Phase 1 — Robot Adapter Interface + AI2-THOR sim adapter
**~5–7 weeks | Medium-High difficulty**

Define the RAI schema and implement the first adapter.

- Define the RAI schema: `skill_call(name, params) → Outcome`, `observe() → BeliefUpdate`, `alert(type, payload)`; define `BeliefUpdate` schema with confidence and staleness fields
- Implement AI2-THOR adapter: map AI2-THOR action space to RAI `skill_call`; map observation dict to `BeliefUpdate` with confidence and partial visibility; inject controlled noise (~150 lines)
- Add partial observability model: sensing cone, stale cached beliefs with confidence decay, sensing as budget-costed explicit action
- Add action uncertainty model: configurable failure injection; failures detected from next `observe()`, not from error return codes
- Port OBSERVER/MEDIATOR prompts to robotics domain vocabulary
- Build warehouse task suite on AI2-THOR: 10–15 tasks using ProcTHOR layouts for held-out generalization

*Risk: Medium. AI2-THOR's action semantics are richer than ARC-AGI-3's discrete space; the adapter must handle failures without leaking sim concepts upward.*

### Phase 2 — Correction governance layer ← critical path
**~6–8 weeks | High difficulty**

The headline contribution. No published embodied-agent system has this.

- Correction parser: LLM call extracts intent, `applies_to`, `preconditions`, proposed expiry from natural-language utterance; unrecoverable scope triggers clarification, not silent storage
- Advisory/executable distinction: corrections enter as advisories (influence planning, logged); promotion to executable requires dry-run + operator confirmation
- Dry-run mechanism: replay last N steps with proposed rule active; show operator what behavior changes
- Conflict detection: check new rule against existing rules of overlapping scope before storage; conflicts surface as explicit resolution prompt; supersession written to audit log
- Hazard tagging: `hazard_flags` on objects/locations/action classes; corrections weakening hazard constraints require elevated confirmation; hazard constraints cannot be expired by plain correction
- Audit log: every action records `rule_trace` and `constraint_trace`; every rule records origin, scope, promotion history, revocations; queryable by rule, action, time window
- Operator CLI: text prompt at episode boundaries and mid-episode on `ask_human()`

*Risk: High. Conflict detection over compound scopes is non-trivial. LLM correction parser will occasionally hallucinate scope. Build a test suite of 20–30 correction examples early and run them through the parser before implementing the governance machinery.*

### Phase 3 — Closed-loop executive
**~3–4 weeks | Medium difficulty**

Replace the round-based pipeline with an event-driven executive.

- Define four preemption event types: `BeliefChange`, `ActionFailure`, `OperatorCorrection`, `SafetyTrigger`
- Implement lightweight event bus: skill calls publish outcomes; constraint checker and belief updater subscribe and emit preemption events
- Interruptible skill execution: skill calls yield at defined checkpoints; executive checks for pending events at each checkpoint
- Event-driven replan: MEDIATOR called with replan reason as context; triggered by events, not round completion
- Retain round logging as evaluation annotation only

*Risk: Low-medium. Main risk is regression on ARC-AGI-3 performance. Run ARC-AGI-3 regression suite before starting this phase.*

### Phase 4 — Wow demonstrations (AI2-THOR / ProcTHOR)
**~3–4 weeks | Medium difficulty**

Three integrated demo scenarios, each telling a different part of the cognitive OS story. All run in simulation; all are designed to be filmed.

**Demo A — "The Correction That Sticks"**
Agent runs 20 warehouse tasks. Baseline (long-context LLM) follows a correction for 3 tasks, forgets it by task 8. KF agent encodes it as a scoped artifact, follows it at task 18, and the audit log traces the rule at every relevant action. Side-by-side comparison.
*Wow factor: "The baseline forgot. Ours didn't — and here's the proof."*

**Demo B — "Shift Handoff"**
Agent runs tasks across two sessions (restart between them). Session 1 operator teaches conventions. Session 2 starts with those conventions loaded (deployment-scope persistence). Session 2 operator tries to override a hazard-tagged constraint; the system prompts for elevated confirmation. No retraining.
*Wow factor: "The robot remembered everything from yesterday, and it refused an unsafe instruction."*

**Demo C — "Same Brain, Different Robot"**
The same KF cognitive layer, configuration unchanged, runs against two different RAI adapters: AI2-THOR warehouse and a second minimal adapter (Spot-style GraphNav API mocked in Python). Identical correction governance, identical audit trail. Each adapter is ~250 lines.
*Wow factor: "We changed the robot. We didn't change the brain."*

*Risk: Medium. Demo reliability depends on Phase 2 robustness. Script corrections exhaustively and test before filming.*

**Target venue:** CoRL 2026 LangRob workshop or RSS 2026 Lifelong Learning workshop (submissions ~July 2026). Phases 0–4 are achievable by June 2026 with focused effort.

### Phase 5 — Unitree R1 adapter (Isaac Sim)
**~2–3 weeks | Low-Medium difficulty**

Complete the "cognitive OS" claim with a named shipping product.

- Implement Unitree R1 RAI adapter against Isaac Sim + `unitree_sim_isaaclab`: map `LocoClient.Move()`, `ArmActionClient` to RAI `skill_call`; map DDS observations to `BeliefUpdate`; handle Unitree's fire-and-observe failure semantics (~300 lines, ROS 2 node)
- Rerun Demo C replacing the mock Spot adapter with the Unitree G1/H1 Isaac Sim adapter; same cognitive layer, no changes
- Optional: tabletop webcam + VLM OBSERVER + Wizard-of-Oz actuation for a real-perception sanity-check video

*Risk: Low. The RAI was designed with Unitree in mind. The adapter is additive, not architectural.*

### Roadmap summary

```
Phase 0   Cognitive OS core extraction        3–4 wks    Medium
Phase 1   RAI + AI2-THOR adapter              5–7 wks    Medium-High
Phase 2   Correction governance           ★   6–8 wks    High
Phase 3   Closed-loop executive               3–4 wks    Medium
Phase 4   Wow demos (AI2-THOR)                3–4 wks    Medium
Phase 5   Unitree R1 / Isaac Sim adapter      2–3 wks    Low-Medium
──────────────────────────────────────────────────────────────────
Total                                        22–30 wks
```

★ Phase 2 is the critical path. Everything else is well-understood engineering or scoped integration. Starting Phase 2 design — especially the correction parser and scope representation — in parallel with Phase 1 implementation is strongly recommended.

With one focused developer plus AI assistance, Phases 0–4 are achievable in 6–8 months; Phase 5 adds ~2 months.

---

## 14. Suggested Repo Layout

```
usecases/
  robotics/
    README.md
    DESIGN.md
    python/
      harness.py
      ensemble.py
      env.py
      tasks.py
      rules.py
      tools.py
      prompts/
        observer.txt
        mediator.txt
        actor.txt
```

Suggested `dataset_tag`:

```text
robotics
```

---

## 15. Correction Governance — State of the Art

This section documents the research landscape for Phase 2 (Correction Governance Layer), explains where each prior-art stream reaches, and identifies exactly what KF must build from scratch vs. what can be borrowed.

### Four research streams — and their gaps

The field splits into four loosely connected streams that rarely cite each other. None fully addresses the combination KF needs.

---

#### Stream 1 — Formal safety enforcement (robotics conferences)

The strongest work on hard constraint guarantees for robot planners.

**Brown H2R Lab — "Plug in the Safety Chip" (Tellex group, ICRA 2024)**
Natural-language safety rules are compiled offline into **Linear Temporal Logic (LTL)** formulas. At runtime, every proposed action is checked against the LTL monitor; violations are caught and the LLM is reprompted with an explanation. Provides verifiable (not probabilistic) guarantees — the LTL monitor cannot be talked out of a constraint by a clever prompt.
*Gap vs KF:* Rules are compiled offline by engineers. No mechanism for an operator to add or modify a rule mid-deployment in natural language. No scope, expiry, conflict detection, or audit trail linking actions to rules. Static guard, not a governable knowledge store.

**RoboGuard (2025, follow-on to Safety Chip)**
Two-stage architecture: a root-of-trust LLM uses chain-of-thought reasoning to contextualize pre-defined safety rules against the current environment, generating tightened LTL constraints. A monitor checks execution against those constraints. Reduces unsafe plan execution from 92% to below 2.5%.
*Gap vs KF:* Same structural limitation — rules are pre-defined, not operator-correctable at runtime.

**SELP — Purdue, ICRA 2025 Best Paper Finalist (Lin Tan, Suresh Jagannathan)**
Three innovations: (1) *equivalence voting* — multiple LLM samples taken, only majority-agreed plans proceed; (2) *constrained decoding* — token generation guided to avoid LTL-forbidden sequences; (3) domain-specific fine-tuning. Achieves 95.2% safety rate on drone navigation, 93.6% on manipulation.
*Gap vs KF:* Closed set of user-specified constraints at planning time. No runtime correction loop, no persistence across sessions, no operator governance. Constraints are task-scoped inputs, not a managed knowledge store.

**Key institutions:** Brown H2R (Tellex), Purdue (Lin Tan / Jagannathan), Stanford ILIAD (Sadigh — adjacent, focused on reward learning).

---

#### Stream 2 — Runtime governance frameworks (AI safety / agents community)

Exploded in late 2024–2025, mostly outside the robotics community.

**"Harnessing Embodied Agents: Runtime Governance for Policy-Constrained Execution" (arxiv 2604.07833, April 2026)**
Most directly relevant paper. Separates agent cognition from execution oversight via a dedicated runtime layer performing: policy checking, capability admission, execution monitoring, rollback, human override. 1000 simulation trials: 96.2% interception of unauthorized actions, 91.4% recovery success.
*Gap vs KF:* Governance policies are pre-specified at deployment. No operator mid-deployment NL corrections, no scoping, no conflict resolution, no advisory/executable distinction, no audit trail linking actions to rules.

**Policy Cards (arxiv 2510.24383, Oct 2025)**
Machine-readable JSON schema for agent operational constraints. Includes `valid_from`/`valid_to` timestamps, allow/deny rules, scope (domain, jurisdiction, data sensitivity), compliance crosswalks to NIST AI RMF and ISO/IEC 42001. Agents can ingest and reason about their own policy cards at runtime. The `valid_from`/`valid_to` field is the closest published equivalent to KF's expiry mechanism.
*Gap vs KF:* Authored by compliance teams, not typed in by operators. No NL correction ingestion, no conflict detection between cards, no hazard-based elevation mechanism, no advisory promotion workflow.

**Agent Behavioral Contracts (arxiv 2602.22302, Feb 2026)**
Brings Design-by-Contract to LLM agents. Specifies Preconditions, Invariants, Governance policies, and Recovery mechanisms as runtime-enforceable components. Defines probabilistic compliance accounting for LLM non-determinism. Proves a "Drift Bounds Theorem": contracts with recovery rate > failure rate bound behavioral drift.
*Gap vs KF:* Contracts are authored formally by engineers. No operator correction loop, no scope lifecycle, no advisory period.

**AgentSpec (arxiv 2503.18666)**
A rule-based DSL for LLM agent safety properties. Both preventive enforcement (block before action) and corrective enforcement (intervene after). Supports conditional rules and runtime monitoring. Closest to KF's rule engine in structure, but requires technical authoring — no NL ingestion.

---

#### Stream 3 — Corrigibility and interactive correction learning (HRI / RL)

**Stanford ILIAD (Dorsa Sadigh) and Berkeley (Anca Dragan)**
Learn reward functions from comparative human language feedback: "the robot should have done X more than Y." The 2025 paper (Hirota, Yang, Dragan, Bıyık et al.) iteratively updates trajectory policies from language corrections. Most rigorous academic treatment of incorporating human preferences into robot behavior.
*Gap vs KF:* Corrections change *reward functions*, not *explicit rules*. Behavior change is opaque — cannot inspect what changed or why. Corrections cannot be revoked without retraining. No scoping, expiry, conflict detection, or audit trail. Fundamentally *implicit* correction (reward learning) vs. KF's *explicit* correction (governed artifacts).

**Anthropic — Corrigibility research (2025)**
A model trained for pure corrigibility can develop instrumental convergence toward empowering its principal — both an attractor toward genuine corrigibility and toward manipulation. The alignment-faking result (Claude 3 Opus reasoning to mislead testers 10% of the time) confirms that LLM-native correction acceptance is unreliable without an external enforcement layer. This is a strong argument *for* KF's approach of treating corrections as governed external artifacts rather than prompts the LLM reasons about.

---

#### Stream 4 — Instruction hierarchy and conflict detection

**OpenAI — Instruction Hierarchy (ICLR 2025)**
Trains models to respect a hierarchy (system prompt > user > injected content). Reduces prompt injection success significantly. Hierarchy is baked into training, not configurable at runtime. No scoping, no conflict resolution between rules at the same priority level. Works for simple priority ordering, not compound-scope conflicts.

**"Who Is In Charge? Role Conflicts in LLM Instruction Following" (OpenReview 2025)**
LLMs reliably *encode* instruction conflicts early in processing but resolve them based on social cues (authority framing, expertise) rather than structural priority. The model can be steered to follow the correct priority, but the steering is not role-aware. Confirms that instruction-level governance *inside* the LLM is unreliable; external enforcement is necessary.

---

### Gap map

```
                    NL         SCOPED/    CONFLICT   EXPIRY/    AUDIT   OPERATOR-
                    INGESTION  EXPIRABLE  DETECT     REVOKE     TRAIL   CORRECTABLE
                                                                        MID-DEPLOY
──────────────────────────────────────────────────────────────────────────────────
Safety Chip/SELP    ✓ (→LTL)   ✗          ✗          ✗          ✗       ✗
RoboGuard           ✗          ✗          ✗          ✗          ✗       ✗
Runtime Gov         ✗          ✗          ✗          ✗       partial    ✗
Policy Cards        ✗ (JSON)   ✓ (TTL)    ✗       partial       ✓       ✗
Agent Contracts     ✗ (formal) partial    ✗          ✗          ✗       ✗
AgentSpec           ✗ (DSL)    ✗          ✗          ✗          ✗       ✗
ILIAD/Dragan        ✓ (→reward) ✗         ✗          ✗          ✗    implicit
KF Phase 2 target   ✓          ✓          ✓          ✓          ✓       ✓
```

The governed corrections + full lifecycle column is genuinely unoccupied.

---

### What to borrow directly (reduces blank-page risk)

| KF component | Best prior art | What to take |
|---|---|---|
| Rule schema (scope, expiry, TTL) | Policy Cards (2510.24383) | JSON schema pattern, `valid_from/valid_to`, version-control model |
| NL → structured rule (correction parser) | SELP equivalence voting | Run parser N times, take consensus — reduces hallucinated scope |
| Pre-action constraint checking | Safety Chip / RoboGuard monitor | Monitor-before-execute pattern; LTL not required, structural check sufficient |
| Audit log schema | Runtime Governance (2604.07833) | Closest published audit design for embodied agents |
| Probabilistic compliance tracking | Agent Behavioral Contracts | (p, δ, k)-satisfaction model for tracking rule adherence over time |
| Hazard constraint invariants | Agent Behavioral Contracts | Invariant concept maps directly to KF's hazard-tagged constraints |

### What remains genuinely novel (no prior art to borrow)

1. **Conflict detection over compound scopes** — Rule A: "always verify items from shelf B"; Rule B: "skip verification for priority orders from shelf B." Detecting this overlap requires scope intersection reasoning. No published algorithm. Needs custom design.
2. **Advisory → executable promotion with dry-run** — The "show the operator what would change before committing" workflow has no robotics precedent. Closest analogy is A/B policy rollout in web systems.
3. **NL → scoped rule for operator-level corrections** — NL→LTL (Safety Chip, SELP) is proven but rigid. NL→KF's flexible scope schema is LLM-assisted and prone to hallucination. Validated test suite of 20–30 correction examples must be built before implementing the governance machinery.

### Revised difficulty rationale

Phase 2 remains **High** difficulty, but is now less "wilderness exploration" and more "careful integration of five partially-overlapping prior-art components." The Policy Cards schema and Safety Chip monitor pattern reduce the blank-page problem. Conflict detection and advisory promotion remain genuinely novel and are the true hard sub-problems.

---

## 16. Competitive Landscape — Who Is Doing What

Understanding what adjacent systems have already demonstrated is important for positioning KF's contribution clearly and avoiding redundant work.

### Category 1 — Memory architecture for embodied agents

**RoboMemory** (arxiv 2025) is the closest published system on the memory side. It unifies spatial, temporal, episodic, and semantic memory under a brain-inspired architecture (thalamus-like preprocessor, hippocampus-like memory, prefrontal-lobe-like planner) built on a dynamic knowledge graph. It outperforms Claude 3.5 Sonnet by 5% on EmbodiedBench. **Gap vs KF**: purely a memory architecture — no operator-facing correction mechanism, no governance, no scoping or audit. Knowledge accumulates but cannot be safely corrected or revoked.

**MemoryOS / MemOS** (MemTensor, 2025) is a memory OS for LLM agents with persistent skill memory and cross-task reuse, similar in spirit to KF's procedure store. **Gap vs KF**: agent-focused rather than embodied-robot-focused; no hazard-aware safety layer; no correction governance.

**ArmarX memory system** (KIT, ongoing academic) — long-running project on robot cognitive memory with episodic, semantic, and procedural stores. **Gap vs KF**: research prototype; not designed around operator-facing corrections or safety governance.

### Category 2 — Cross-platform / multi-robot OS

**EMOS** (NUS/HKU, ICLR 2025) is the most directly comparable system on the cross-platform angle. It is a heterogeneous multi-robot OS where LLM agents read robot URDF files to generate "robot resumes" (self-descriptions of physical capability), then use those for embodiment-aware task assignment across robots of different types. It benchmarks on Habitat-MAS (manipulation, perception, navigation, multi-floor rearrangement). **Gap vs KF**: EMOS is a planner and task dispatcher — it has no persistent knowledge across sessions, no operator correction mechanism, no governed memory layer. It answers "which robot should do this?" not "what has this robot learned across deployments?"

### Category 3 — LLM planner + feedback loop (foundational work)

These are the systems KF builds on top of:

- **SayCan** (Google, 2022) — LLM skill selection grounded by affordance value functions. Proved LLM+skills on real robots. Stateless; no persistent memory; no correction.
- **Inner Monologue** (Google, 2022) — closed-loop environment feedback to LLM planners. Better recovery and replanning. Still stateless per-episode; no persistence; no operator correction.
- **Voyager** (NVIDIA, 2023) — skill library accumulation in Minecraft. Closest in spirit to KF's procedure store. Game-environment only; no safety/governance; no operator correction; no cross-session belief state.

### Category 4 — Industry players

No major embodied-AI company has published a systematic answer to cross-session memory and governed correction at the task level:

| Company | Focus | Gap |
|---|---|---|
| Figure / OpenAI | End-to-end VLA, stateless per-episode | No persistent memory, no correction governance |
| Physical Intelligence (π0) | Generalist VLA policy | Same — no cross-session layer |
| 1X (NEO) | Cautious household deployment + teleoperation backup | Governance is human-in-the-loop, not systematic |
| Agility Robotics (Digit) | Warehouse deployment | Workspace rules are hard-coded, not learned or correctable |
| Boston Dynamics (Spot) | Cloud-assisted mission scripting | Mission scripting only; no learned/correctable knowledge |

The pattern is consistent: everyone is investing in perception and control (VLA), and no one has a published, systematic solution to cross-session memory + governed correction at the task level.

### Where KF's unoccupied territory is

```
                        PERSISTENT    GOVERNED     CROSS-         SAFETY-
                        MEMORY        CORRECTIONS  PLATFORM       GOVERNED
                                                   (cognitive OS) RULES
────────────────────────────────────────────────────────────────────────────
RoboMemory               ✓             ✗            ✗              ✗
MemoryOS                 ✓             ✗            ✗              ✗
EMOS                     ✗             ✗            ✓              ✗
SayCan / InnerMonologue  ✗             ✗            ✗              ✗
Voyager                  ✓ (skills)    ✗            ✗              ✗
Industry VLAs            ✗             ✗            ✗              ✗
KF (proposed)            ✓             ✓            ✓              ✓
```

The governed corrections + cross-platform + safety column is genuinely unoccupied by any published system. The positioning statement:

> "RoboMemory and MemoryOS show that persistent memory improves embodied-agent performance. EMOS shows that a cross-platform cognitive layer is tractable. KF combines both and adds the missing dimension neither addresses: governed, auditable, safely-scoped operator corrections that persist and compose across sessions and platforms."

The editorial "The Robot in Your Living Room Has No Rulebook" (AI Frontiers, 2025) directly names this gap and effectively writes the motivation section.

---

## 16. KF As A Cross-Robot Cognitive OS

The longer-term strategic opportunity is not a memory layer for one robot — it is infrastructure that the embodied-AI industry is missing above the skill layer, comparable to what an OS kernel provides above hardware.

### Why the abstraction is tractable

Every major humanoid (and wheeled robot) exposes a similar semantic API boundary, just with different names and transports:

| Robot | High-level API | Transport |
|---|---|---|
| Unitree R1 / G1 | `LocoClient`, `ArmActionClient` | DDS / JSON-RPC |
| Boston Dynamics Spot | `RobotCommandClient`, `GraphNavClient` | gRPC |
| Agility Digit | ROS 2 action servers | ROS 2 DDS |
| Fourier GR-1 / GR-2 | `FourierClient` skill API | ROS 2 / WebSocket |
| Hello Robot Stretch | `robot.move_to()`, `robot.arm` | ROS 2 |
| Figure 02 / 1X NEO | Proprietary skill API | LAN WebSocket |

The pattern is universal: discrete named skills + structured observation return + DDS or ROS 2 transport. If KF's planner speaks to a **Robot Adapter Interface (RAI)** that normalizes these, the cognitive layer above never needs to know which robot it's running on.

### Proposed architecture

```
┌──────────────────────────────────────────────────────┐
│              KF COGNITIVE OS                          │
│  Belief state │ Goal manager │ Correction governance  │
│  Procedure store │ Audit log │ Planner (LLM)          │
└──────────────────┬───────────────────────────────────┘
                   │  Robot Adapter Interface (RAI)
                   │  Standard schema:
                   │   skill_call(name, params) → outcome
                   │   observe() → BeliefUpdate
                   │   alert(type, payload)
        ┌──────────┴───────────────┬────────────────┐
        ▼                          ▼                ▼
  [Unitree Adapter]       [Spot Adapter]    [Sim Adapter]
  LocoClient / ArmClient  gRPC GraphNav     AI2-THOR / ALFWorld
  DDS                     ROS 2             Python API
```

The RAI is approximately 200–400 lines per robot adapter — largely a skill-name mapping and an observation schema normalizer. The cognitive layer is written once.

### What makes this tractable

- The hard part (locomotion, manipulation, perception) is already solved per-robot. KF does not touch it.
- Skill granularity is similar across platforms: `move_to`, `pick`, `place`, `inspect`, `ask` are universal task-level primitives.
- Observation schemas are structurally similar across platforms: location, visible objects, task status, blocked routes — field names differ but semantics don't.

### What makes this non-trivial

- Timing and failure semantics differ. Spot's action server provides preemption hooks; Unitree's DDS is more fire-and-observe. The RAI must absorb this.
- Safety estop integration is robot-specific and must stay in the adapter, not in KF.
- Object models differ (Spot's GraphNav uses waypoints; AI2-THOR uses instance IDs; real cameras need object detection). The VLM/perception layer needs its own normalization alongside the RAI.

### The compelling simulator proof

The cross-robot claim is uniquely well-suited to simulation — in fact, sim is better than hardware here because you can run multiple robot adapters in the same test harness. A convincing demonstration runs **the same KF cognitive layer, unmodified, against two or three different sim adapters** — e.g., AI2-THOR, a warehouse-layout ProcTHOR environment, and a minimal adapter mimicking Spot's GraphNav API — showing identical correction-governance behavior, identical belief-state management, and identical audit trails across all three. The cognitive layer does not change. The adapter for each is under 300 lines. That is a clean, falsifiable, peer-legible claim and it requires no hardware.

---

## 17. Why This Use Case Matters

The broader thesis this use case supports:

**Embodied systems need more than perception and control. They need an explicit runtime knowledge layer for long-horizon adaptive behavior — one that is persistent, correctable, governed, and portable across platforms.**

That is the gap KF targets. RoboMemory and MemoryOS are beginning to address persistence. EMOS is beginning to address cross-platform task allocation. No published system addresses the full combination — and correction governance in particular is completely open territory.

If this works, KF has a credible path from a simulator benchmark to a cognitive OS substrate that any robot team deploying above the skill layer would want to adopt.

---

## 18. Safety Governance Layer — Cognitive-OS Safety Ensemble

The cognitive OS includes a dedicated **safety governance layer** (`core/cognitive_os/safety.py`)
that vets every action the rest of the system wants to execute. It is a small, deliberately-shaped
component today, but the interface is the seed of a future **Safety Agent / Safety Ensemble** —
a supervisor that can grow from rule-based checkers into LLM-backed, simulator-backed, or
learned-classifier checkers without breaking callers.

The motivating event was an ARC-AGI-3 episode in which a BFS planner correctly avoided a
learned lethal cell, but its **LLM-mediator fallback** (used when BFS reports "no path")
walked into the same cell and lost a life. The lesson is general: any layer that can override
the planner — fallbacks, recovery routines, operator macros — needs a uniform veto path that
sees the same risk knowledge the planner does.

### 18.1 Interface contract

Four small types do all the work:

| Type | Role |
|---|---|
| `ActionProposal(agent_id, action, predicted_effect, context)` | What the caller wants to do, plus what it expects to happen if it executes. |
| `SafetyChecker` (Protocol) | A single rule. Has a `name: str` and `check(proposal) -> SafetyVerdict`. |
| `SafetyVerdict(allow, severity, checker, reason, alternatives)` | The outcome. `allow` is the only boolean callers must branch on; the rest is for logs, audit, and optional planner consultation. |
| `SafetyEnsemble(checkers, aggregator)` | Composable container. `register(c)` to add. `evaluate(proposal)` runs every checker and aggregates. Itself satisfies the `SafetyChecker` Protocol so ensembles are nestable. |

The shapes are intentionally small: a caller that has never seen the safety layer can wire it
in with three lines (build a proposal, call `evaluate`, branch on `verdict.allow`).

### 18.2 Key design decisions

These choices were made now, before the second checker exists, because changing them later
would force every caller and every checker to be rewritten.

1. **`predicted_effect` is a soft-key dict, not a typed struct.**
   Keys (`target_pos`, `ee_pose`, `token_output`, `resource_cost`, …) are a *convention*
   shared between callers and checkers. A checker that needs a new key can be added without
   modifying the dataclass; callers that don't emit it simply get `allow=True` (the checker
   abstains). This keeps the shape stable across ARC, robotics, and LLM-action domains
   that have very different "effect" vocabularies.

2. **Safety decides; planning replaces.**
   The ensemble never picks a substitute action. A checker MAY populate
   `SafetyVerdict.alternatives` as advice, but the planner — which knows about cost,
   goals, and counterfactuals — owns the choice. Conflating the two would mean the
   safety layer has to understand task semantics, which is the whole reason the cognitive
   OS exists separately.

3. **Caller policy stays outside the safety layer.**
   What to do on a block — break the plan chunk, substitute a wait, abort the skill,
   prompt the operator — is the caller's policy. The safety layer only answers
   "is this OK?". Otherwise every new policy would force a safety-layer change.

4. **Graceful degradation.** A checker that raises is logged as `severity="warn",
   allow=True`, not propagated. A bug in a safety rule must not brick the agent. Strict
   wrappers can be layered on top by callers that prefer fail-closed behavior.

5. **Ensemble is nestable.** `SafetyEnsemble` itself implements `SafetyChecker`, so a
   high-priority "kernel" ensemble can be registered inside an outer ensemble that holds
   advisory checkers — priority tiers via composition, not via a new abstraction.

6. **Aggregator is pluggable.** The default `strict_block_aggregator` vetoes on any
   `allow=False` and concatenates reasons by severity. The signature
   `Callable[[List[SafetyVerdict]], SafetyVerdict]` is wide enough to support a
   future learned arbiter that weighs checkers by track record, or a quorum-based
   policy, without changing the ensemble class.

7. **Domain-agnostic action handle.** `ActionProposal.action` is `Any`. ARC passes an
   action-name string; robotics passes a skill name or trajectory handle; an LLM-action
   layer passes the candidate token sequence. Checkers care about `predicted_effect`,
   not the raw action, so the layer transfers across domains untouched.

### 18.3 Reference checker — `LethalPosChecker`

The first concrete checker, used in the ARC validation episode, blocks any proposal whose
`predicted_effect["target_pos"]` falls in a learned-lethal cell set. The set is updated by
calling `update_lethal(new_set)` — typically once per planning cycle, fed from the
HypothesisRegistry's confirmed `lethal_at_pos` claims. The checker is stateless about
*how* the set was assembled; it just enforces it.

The robotics analogues already on the roadmap:

| Checker (planned) | Reads | Vetoes when |
|---|---|---|
| `CollisionRiskChecker` | `target_pos` (pose cell) | cell appears in persisted high-risk collision map |
| `JointLimitChecker` | `ee_pose` | predicted joint solution exceeds calibrated limits |
| `ResourceGuard` | `resource_cost` | predicted spend would push battery / time-budget below floor |
| `WorkspaceBoundChecker` | `target_pos` | predicted move would leave the operator-defined work envelope |

All plug into the **same** `SafetyEnsemble.register()` call. The ARC validation episode
exercises the same machinery a real robot run would.

### 18.4 Relationship to the correction governance layer (§10, §15)

The two layers are complementary, not redundant:

- **Correction governance** (§10) takes natural-language operator corrections, scopes them,
  expires them, and resolves their conflicts. Its outputs are *rules* — high-level policies
  the agent must respect.
- **Safety governance** (this section) takes a candidate action, checks it against
  rules + learned hazards, and returns a verdict. Its output is *per-action allow/block*.

A correction like "don't enter the kitchen between 9pm and 7am" (a §10 rule) compiles into a
checker that the §18 ensemble runs on every motion proposal whose `target_pos` falls in the
kitchen polygon during the relevant window. The governance layer is *where the rule comes
from*; the safety ensemble is *where the rule fires*.

### 18.5 Scale-up path

Each scale-up step lands as a new checker or aggregator — no caller changes:

1. **Today.** Hand-coded checkers (`LethalPosChecker`), strict-block aggregator.
2. **Near-term.** Simulator-backed checker: roll out the proposed action in a fast
   physics sim, block on collision/instability. Same `check()` signature.
3. **Mid-term.** LLM-backed checker for high-level plausibility ("does this action
   match the operator's stated intent?"). Same `check()` signature; latency budget
   handled by parallel evaluation inside the ensemble.
4. **Long-term.** Learned arbiter as the aggregator, trained on which historical
   vetoes were correct vs. overzealous. Checkers stay individually auditable; the
   weighting is what adapts.
5. **Eventually.** The ensemble itself becomes the public interface of a **Safety
   Agent** that owns memory, talks to the correction-governance layer directly, and
   surfaces its veto rationale to operators. The wire-format (`ActionProposal` →
   `SafetyVerdict`) does not change.

The point of locking in the interface now is that the migration above is additive at
every step.


## 19. Learning-Gap Closures — Causal, Resource, Distal, Precondition, Similarity

Five substrate additions land in `core/cognitive_os/` during the same iteration as
the safety ensemble (§18). They exist because an ARC-AGI-3 level-2 post-mortem on
an 8-run failure chain surfaced five generic reasoning gaps that any agent operating
under a depleting resource and device-shaped mechanics will hit — and which no
domain-specific patching would fix without re-breaking robotics.

The gaps and the modules that close them:

### 19.1 Causal attribution — `causal.py`

**Gap.** A single-cause post-mortem ("I just died at cell X therefore X is lethal")
is wrong whenever a slow-burning background cause was the real culprit. In the ARC
run, deaths caused by step-counter exhaustion were blamed on whatever cell the
agent happened to be standing on. Each such misattribution added a cell to the
lethal set, which forced longer detours, which depleted the counter faster — a
feedback loop that sank runs 8a–8f.

**Module.** `CausalAttributor` takes a `Surprise` (the event to explain) and an
`observation_snapshot`, and runs registered `AlternativeCauseHook`s to propose
competing `CauseCandidate`s. The built-in `resource_exhaustion_hook` queries
`resource_decay` hypotheses in the registry, reads the snapshot value at the
resource's `observation_key`, and returns a high-confidence candidate when the
value is at or below the critical threshold. The top candidate wins; the planner
branches on `cause == "resource_exhaustion"` vs `"surface"`.

**Robotics analogue.** A peer robot's tool collides with the end-effector mid-grasp.
Surface attribution: "that object position is lethal". Alternative causes: a
battery at 5% induces motor slip (`low_battery_hook`); a hand-over schedule
says the peer was supposed to be elsewhere (`schedule_conflict_hook`). Each
lands as its own hook; the aggregator stays the same.

**Invariant.** The same `seed_resource_decay_hypothesis` declaration is read by
both §19.1 (the attributor hook) and §19.2 (the tracker). Changing a resource's
definition once updates attribution and budgeting in lockstep.

### 19.2 Resource awareness — `resources.py`

**Gap.** A substrate that doesn't model depleting resources cheerfully plans a
50-step detour when 25 steps remain, then "mysteriously" dies and typically
misattributes the failure to whatever it was doing at the moment (§19.1).

**Module.** `ResourceTracker` owns a `Resource` per depleting quantity
(`step_counter`, `battery`, `gripper_cycles`, `tokens_of_llm_budget`,
`time_on_task`). Each `Resource` carries `current`, `critical_threshold`,
`decay_per_action`, `observation_key`, and a `refill_predicate` used to query
refill-location hypotheses. The planner consults:

- `can_afford(name, path_cost) -> BudgetStatus` → `AFFORDABLE / TIGHT / OVERRUN / UNKNOWN`
- `remaining_actions(name)` → floor on actions until critical
- `nearest_refill(name, from_pos, distance_fn)` → optional refill subgoal

**Robotics analogue.** Battery on a mobile base is the canonical case; dock
positions come from `dock_at_pose` hypotheses. Gripper-actuation cycles, tool-wear
counters, conversational LLM budget, mission-clock seconds — every one of them
drops into the same `Resource` slot with zero planner-code changes. The tracker's
rule that an observed value overrides a modelled one means the dock's "battery
full" reading wins over accumulated decay estimates — identical to how the
step-counter's HUD reading overrides per-action drain.

**Abstention.** If decay-per-action is unknown and no recent observation exists,
the tracker returns `UNKNOWN` rather than guessing. A robotics safety reviewer
reads that directly: the planner knows to ask for more data, not to proceed blind.

### 19.3 Distal effects — `miners.DistalEffectMiner`

**Gap.** Devices — buttons, switches, RC-changers, wall-mounted panels — by
definition cause non-local state change. A miner that only correlates an action
with its *local* diff will never discover that hitting cell A consistently
rotates the indicator at cell B on the other side of the map.

**Module.** `DistalEffectMiner` reads `tick.outcome["distal_changes"]` — a list
of cells/regions that changed this tick and sit *outside* the agent's local
radius. After `min_count` corroborations of the same `(source_cell, action,
affected_cell)` triple, it registers a `distal_effect` hypothesis. A planner that
reads this hypothesis can use A as a tool: *to change state at B, act at A*.

**Robotics analogue.** A button on cabinet #3 whose press unlatches a drawer on
shelf #7; a wall switch that toggles a light in another room; a gesture that
starts a peer robot's hand-over. Each is one `distal_effect` hypothesis; each
becomes a planning affordance.

**Domain bridge.** The ARC adapter (`arc_stream_adapter._compute_distal_changes`)
produces the tick's `distal_changes` list by diffing pre- and post-frames and
quantising to a coarse grid; robotics adapters emit the same list from their
perception stack (object-pose diffs, detected drawer state, etc.). The miner is
identical in both cases.

### 19.4 Goal preconditions — `miners.GoalPreconditionMiner`

**Gap.** Two attempts at the same goal — say, stepping on the WIN tile —
sometimes succeed and sometimes fail. A naive planner treats the goal as
non-deterministic and keeps retrying. A better substrate *learns which event
must precede the goal in the successful runs*: the ring indicator must be in a
matching orientation; the drawer must be open; the robot must be gripping the
object. That learned precondition becomes a subgoal of the parent goal.

**Module.** `GoalPreconditionMiner` listens for `goal_attempt:<id>`,
`goal_success:<id>`, `goal_failure:<id>` notable events. For each goal, it
computes events common to all successes and absent from failures, then
registers `precondition_of` hypotheses. The planner injects the event as a
subgoal before the next attempt.

**Robotics analogue.** "Handover succeeds only when the peer is facing the
operator" (orientation precondition); "drawer-close succeeds only when the
contents are clear" (state precondition). Exact same miner, different event
vocabulary sourced from the robot perception stack.

### 19.5 Similarity connections — `similarity.py` (VLM-assisted)

**Gap.** Two objects that *look alike* often matter together — keys and locks,
identical pill bottles on two shelves, a miniature indicator vs. the full-size
target it tracks. Pixel-level similarity detects the easy cases (same-colour
same-scale blobs) but misses anything that requires rotation-, scale-, or
colour-invariance. Those are exactly the cases where a human would say "those
two things are connected — investigate".

**Modules.** `SimilarityOracle` is a narrow Protocol (`compare(a, b) -> SimilarityResult`).
Three implementations ship:

- **`PixelSimilarityOracle`** — cheap baseline; catches same-shape same-scale matches.
- **`VLMSimilarityOracle`** — wraps an adapter-supplied `VLMCallable`. The adapter
  routes into whichever VLM stack the use case owns (ARC's OBSERVER/MEDIATOR; a
  robotics perception VLM). Oracle handles prompt, caching, per-cycle budget.
- **`CascadingOracle`** — pixel first; escalate to VLM only when the score is
  ambiguous. Saves expensive calls on obvious matches and obvious non-matches.

`SimilarityMiner` queries the oracle on region pairs (either via
`tick.outcome["similarity_pairs"]` or via a `pair_supplier` callback) and registers
`possible_connection_between` hypotheses for pairs the oracle scores above
`min_score`.

**Why VLM.** The ARC level-2 case requires recognising that a small L-shaped
sprite near the RC-changer and a large L-shaped indicator on the HUD are
*instances of the same icon at different scales*. Pixel overlap is near zero.
An OBSERVER-routed VLM call sees the match directly and returns
`connection_kind="miniature_indicator"`. That hypothesis lets the planner
formulate the experiment: "does hitting the RC-changer rotate the small L so
that it matches the large L?"

**Robotics analogue.** Two identical tools at different shelf positions →
`instance_of_category`, interchangeable. A pair of arrow-shaped floor markers
with the same colour and size → `paired`, waypoints on the same route. The
operator's mug on the desk and the one in the cupboard → `identity`, same
object category. Each becomes a planning affordance without changing the miner.

### 19.6 Cross-domain integrity check

Every closure above follows the same rule: **the module lives in `core/cognitive_os/`,
the domain adapter supplies a narrow input, and the robotics analogue is the
peer use case — not an afterthought**. Shipping the ARC fix lands the robotics
capability by construction; shipping the robotics fix lands the ARC capability
by construction. That is the two-usecase test for every substrate commit.
