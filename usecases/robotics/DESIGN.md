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

## 12. Suggested Repo Layout

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

## 13. Recommended First Build

The first concrete implementation should be:

**KF as the planner layer on top of AI2-THOR/ProcTHOR (or ALFWorld), running a warehouse/lab-assistant task suite with partial observability, noisy perception, action uncertainty, episodic environment changes, and a governed correction loop.**

Target headline experiment: **correction governance under partial observability.** Run the agent over a batch of tasks, inject operator corrections of varying scope, and measure Tier 2 metrics (safety, over-generalization, audit completeness, adaptation efficiency) against long-context and RAG baselines. This is the single most defensible story for a robotics audience and needs no hardware.

That is enough to test:

- environment learning under uncertainty
- procedure reuse across held-out layouts
- event-driven replanning
- failure recovery
- governed natural-language correction

without getting trapped in simulator complexity — and without the "toy environment" critique that would apply to a bespoke world.

---

## 14. Why This Use Case Matters

If this works, it would support a strong broader thesis:

**embodied systems need more than perception and control; they need an explicit runtime knowledge layer for long-horizon adaptive behavior.**

That is where KF could become genuinely important in robotics.
