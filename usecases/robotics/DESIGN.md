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

Use a **semi-structured embodied task simulator with abstracted low-level actions**.

This should look like a robotics task environment, not a generic symbolic planner.

Recommended setting:

- **warehouse assistant**
or
- **lab assistant**

These are preferable to household robotics because:

- easier to define clearly
- commercially meaningful
- rich in local conventions
- strong opportunities for high-level knowledge reuse

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

The low-level action layer should be abstracted into actions like:

- `move_to(location)`
- `inspect(target)`
- `pick(object)`
- `place(object, location)`
- `open(container)`
- `close(container)`
- `verify(object)`
- `ask_human(question)`
- `charge()`

These actions should have deterministic or semi-deterministic outcomes and return structured observations.

### 4.3 Observations

The environment should return observations such as:

```json
{
  "location": "inspection_station",
  "visible_objects": ["bin_A12", "tray_red", "scanner_1"],
  "task_status": "in_progress",
  "blocked_routes": ["left_corridor"],
  "object_state": {
    "bin_A12": {"verified": false, "picked": false}
  }
}
```

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

## 7. Proposed KF Architecture Mapping

This use case should map naturally onto the emerging ARC-AGI-3 style ensemble pattern.

### Round 0 — Rule retrieval

Retrieve relevant artifacts based on:

- current task
- current location
- visible objects
- known environment state

### Round 1 — Observer / state interpreter

Interpret the current environment state into a structured representation:

- current location
- available routes
- candidate task-relevant objects
- blocked paths
- current task progress

### Round 2 — Mediator / planner

Construct a goal-aware plan using:

- retrieved artifacts
- environment interpretation
- active goals and subgoals

### Round 3 — Actor / verifier

Execute actions in the environment, observe results, and revise when necessary.

### Post-task learning

Extract reusable procedures, environment facts, and failure corrections from the episode.

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
- same agent plus raw retrieved notes
- `agent + KF`

### Metrics

- task completion rate
- number of steps per successful task
- number of replans required
- number of human interventions
- success after environment change
- reuse rate of learned procedures
- time to improvement after a human correction

### Strong first success criterion

A strong first result would show that:

- the KF-enabled agent improves across episodes
- the baseline agent does not improve comparably
- one or more human corrections become immediately reusable artifacts
- task success or intervention count improves without retraining the underlying model

---

## 10. Human Correction Loop

Human correction should be central to the demonstration.

Examples:

- "Items from shelf B must be barcode-verified before transfer."
- "If the left corridor is blocked, use the rear path."
- "Do not place sterile tools on the blue cart."
- "When the target bin is missing, inspect the overflow station before asking for help."

KF should turn these into persistent artifacts and apply them in later episodes.

This is one of the most convincing parts of the use case because it makes the value of explicit runtime learning visible immediately.

---

## 11. Scope Boundary

This benchmark should be described as:

- a robotics-relevant embodied task benchmark
- aimed at the high-level behavior layer

It should **not** be described as:

- a physics benchmark
- a low-level control benchmark
- a claim of direct transfer to physical robots without additional work

That keeps the abstraction honest and the claim defensible.

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

**a warehouse or lab assistant simulator with 5–10 locations, 10–20 movable objects, route changes across episodes, and a small set of human-teachable workspace rules**

That is enough to test:

- environment learning
- procedure reuse
- high-level planning
- failure recovery
- natural-language correction

without getting trapped in simulator complexity.

---

## 14. Why This Use Case Matters

If this works, it would support a strong broader thesis:

**embodied systems need more than perception and control; they need an explicit runtime knowledge layer for long-horizon adaptive behavior.**

That is where KF could become genuinely important in robotics.
