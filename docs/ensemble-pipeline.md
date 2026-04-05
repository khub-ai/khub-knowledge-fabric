# KF Ensemble Pipeline — Developer Guide

This document describes the Knowledge Fabric ensemble pipeline: its domain-agnostic base architecture and the three active domain specializations. It is the primary reference for developers building or extending any KF use case.

**Related documents:**
- [Architecture](architecture.md) — knowledge artifact schema, storage, and tiered retrieval
- [Design decisions](design-decisions.md) — how KF differs from other agent memory / middleware
- [Glossary](glossary.md) — canonical definitions for KF terms used throughout this doc
- [Roadmap](roadmap.md) — planned benchmarks and future use cases

**Use case READMEs:**
- [ARC-AGI-2 use case](../usecases/arc-agi-2/python/../../../usecases/arc-agi-2/README.md) — if present
- [Image classification UC200](../usecases/image-classification/README.md) — bird and dermatology sub-cases

---

## Contents

1. [Overview](#1-overview)
2. [Repository layout](#2-repository-layout)
3. [Base ensemble architecture](#3-base-ensemble-architecture)
4. [Core abstractions](#4-core-abstractions)
5. [Specialization 1 — ARC-AGI-2](#5-specialization-1--arc-agi-2-static-puzzles)
6. [Specialization 2 — ARC-AGI-3](#6-specialization-2--arc-agi-3-interactive-environments)
7. [Specialization 3 — Image classification UC200](#7-specialization-3--image-classification-uc200-birds)
8. [Harness CLI conventions](#8-harness-cli-conventions)
9. [Extension guide](#9-extension-guide-adding-a-new-domain)
10. [Operational notes](#10-operational-notes)

---

## 1. Overview

Knowledge Fabric (KF) is a general-purpose runtime learning framework. The core ensemble pipeline is **domain-agnostic**. Each domain use case is a specialization of the same base architecture — different task adapters, agent prompts, and tool schemas, but the same round structure, rule engine, tool registry, state/goal managers, and harness.

See [design-decisions.md](design-decisions.md#how-this-differs-from-existing-agent-memory) for how KF's persistent, human-readable [knowledge artifacts](glossary.md#knowledge-artifact) differ from other agent memory systems.

**Three active use cases:**

| Use Case | Status | Dataset tag | Entry point |
|---|---|---|---|
| ARC-AGI-2 (static puzzles) | Active — 43/48 correct (89.6%) | `arc-agi-legacy` | `usecases/arc-agi-2/python/harness.py` |
| ARC-AGI-3 (interactive environments) | Design phase | `arc-agi-3` | TBD — new harness needed |
| Image classification UC200 (birds) | Design phase | `bird-uc200` | TBD — new harness needed |

---

## 2. Repository layout

```
khub-knowledge-fabric/
  core/
    knowledge/
      rules.py          ← RuleEngine (namespace-aware, full lifecycle)
      tools.py          ← ToolRegistry (namespace-aware; code and schema tools)
      state.py          ← StateManager (ephemeral task state + history/rollback)
      goals.py          ← GoalManager + Goal (ephemeral goal tree, 5-status lifecycle)
    pipeline/
      agents.py         ← Generic Anthropic API infra (call_agent, CostTracker)
                          call_agent() accepts str or list[content_block] for vision
  usecases/
    arc-agi/
      python/           ← ARC-AGI-2 specialization
        harness.py      ← CLI test runner
        ensemble.py     ← run_ensemble() orchestrator
        agents.py       ← ARC-specific agent runners (shim over core)
        executor.py     ← Deterministic pseudo-code runner
        rules.py        ← Thin shim: re-exports core RuleEngine, sets DEFAULT_PATH
        tools.py        ← Thin shim: re-exports core ToolRegistry, sets DEFAULT_PATH
        rules.json      ← Persisted rule knowledge base (gitignored)
        tools.json      ← Persisted verified tool code (gitignored)
    image-classification/
                        ← UC200 specialization (under active development)
      README.md         ← Full experiment design, results, and UC200 architecture
      python/           ← To be built — see Section 7
  tests/
    arc-agi-3/          ← Early ARC-AGI-3 exploration
      ls20_explorer.py  ← Novelty-driven LS20 environment explorer
      playlog_viewer.py ← Tkinter-based session replay viewer
  docs/
    ensemble-pipeline.md  ← this file
    architecture.md       ← knowledge artifact schema and storage
    design-decisions.md   ← positioning vs. other memory/middleware systems
    glossary.md           ← canonical KF term definitions
    roadmap.md            ← planned benchmarks and future use cases
```

**Python environment:**
```
C:/Users/kaihu/AppData/Local/pypoetry/Cache/virtualenvs/01os-TLh4bqwo-py3.10/Scripts/python.exe
```
The base conda env is Python 3.8 — avoid it; type annotations require 3.10+. ARC-AGI-3 additionally requires conda env `arc` with the `arc_agi` SDK installed.

---

## 3. Base ensemble architecture

The pipeline runs once per **task**. A task is any problem where:
- input(s) can be presented to a VLM or LLM
- correct output can be verified against ground truth or labeled examples
- discriminative rules can be expressed in natural language and reused across tasks

```
Round 0  RULE RETRIEVAL
           RuleEngine retrieves namespace-filtered active rules relevant to this task.
           If >30 candidate rules: two-stage filter
             Stage 1: cheap LLM call (256 tok) → pick relevant categories
             Stage 2: full match on ≤25 filtered rules
           Else: single full-match call
           → list[RuleMatch]

Round 1  PERCEIVER / HYPOTHESIS GENERATOR
           One or more agents observe the task input and produce a structured interpretation.
           Domain-specific role: what transformation? what features? what environment state?
           Prior knowledge (matched rule actions) injected into prompts.
           → structured hypothesis or observation record

Round 2  MEDIATOR / SYNTHESIZER
           Synthesizes a solution plan from the Round 1 output + matched rules.
           Produces a tool call sequence (pseudo-code) or decision chain.
           May emit goal_updates / state_updates JSON blocks for StateManager / GoalManager.
           → plan (list of tool calls with args, or decision chain)

Round 3  EXECUTOR / VERIFIER  (iterative, up to max_revisions=5)
           Executes or evaluates the plan against ground-truth demos or labeled examples.
           On failure: MEDIATOR revises the plan using the failure trace.
           On missing tool: tool_creator generates and verifies a new tool.
           Terminates when all demos pass or revision limit is reached.
           → final answer + pass/fail + accuracy metric

Post-task  LEARNING
           Rule generalizer extracts new candidate rules from success/failure patterns.
           Tools that pass verification are added to ToolRegistry.
           StateManager / GoalManager state logged if enabled.
```

---

## 4. Core abstractions

### 4.1 RuleEngine (`core/knowledge/rules.py`)

Stores, retrieves, and versions natural-language rules by namespace. See source for full API.

| Method | Purpose |
|---|---|
| `active_task_rules()` | Rules to match in Round 0 |
| `add_rule(..., observability_filter=False)` | Create a rule; pass `observability_filter=True` for image-classification use cases to reject non-visual cues (vocalizations, habitat, range, season) before storage |
| `is_visually_observable(text)` | Standalone helper — returns False if text contains non-visual keywords |
| `record_success(rule_id, task_id)` | Update stats after a successful task |
| `record_failure(rule_id, task_id)` | Update stats after a failed task |
| `auto_deprecate()` | Prune low-performing rules |
| `parse_mediator_rule_updates(text, task_id)` | Parse rule create/update JSON from MEDIATOR output |

**Rule status lifecycle:** `candidate → active → flagged → deprecated → archived`

**Pruning triggers:** fires ≥ 10 with 0 successes → deprecate; tasks_seen ≥ 50 → flag

**Rule schema (v3):**
```json
{
  "id": "r_001",
  "condition": "If the grid has a periodic tiling pattern...",
  "action": "Use repair_tiled_grid tool",
  "status": "active",
  "tags": ["arc-agi-legacy"],
  "stats_by_ns": {"arc-agi-legacy": {"fires": 8, "successes": 7, "failures": 1}},
  "tasks_seen": 12,
  "source": "mediator",
  "source_task": "0a2355a6",
  "created": "2026-03-15T..."
}
```

### 4.2 ToolRegistry (`core/knowledge/tools.py`)

Stores two types of tools, both persisted in `tools.json` and namespaced by `dataset_tag`:

| `tool_type` | Contents of `code` field | Used by |
|---|---|---|
| `"code"` (default) | Python function source string | EXECUTOR — loaded into executor at harness startup |
| `"schema"` | JSON string of feature observation form | OBSERVER — retrieved via `get_schema(name)` for prompt injection |

Key methods:
- `register(name, code, verified, tool_type="code")` — add or update a tool
- `get(name)` — retrieve a verified code tool
- `get_schema(name)` — retrieve a verified schema tool as a parsed dict
- `load_into_executor()` — re-register all verified code tools; skips schema tools
- `build_tool_section_for_prompt()` — lists code tools for MEDIATOR
- `build_schema_section_for_prompt()` — lists schema tools for OBSERVER

**Tool schema (v2):**
```json
{
  "name": "repair_tiled_grid",
  "code": "def repair_tiled_grid(grid, **kwargs): ...",
  "tool_type": "code",
  "verified": true,
  "tags": ["arc-agi-legacy"],
  "source_task": "0a2355a6",
  "created": "2026-03-10T..."
}
```

### 4.3 StateManager (`core/knowledge/state.py`)

Ephemeral, task-scoped free-form state with append-only history and rollback. Initialized per task in `run_ensemble()`. Formatted for injection into MEDIATOR prompts via `format_for_prompt()`.

### 4.4 GoalManager (`core/knowledge/goals.py`)

Ephemeral goal tree. Five statuses: `pending / active / succeeded / failed / abandoned`. Abandoning a goal cascades to all descendants. Formatted for injection via `format_for_prompt()`.

### 4.5 Agent update JSON (emitted by MEDIATOR in ` ```json ``` ` blocks)

```json
{
  "goal_updates": [
    {"action": "push", "description": "Identify the transformation type", "priority": 1},
    {"action": "resolve", "id": "goal-abc123", "result": "periodic tiling repair"}
  ],
  "state_updates": {
    "description": "Identified tiling pattern",
    "set": {"pattern_type": "periodic", "period": 3},
    "delete": []
  }
}
```

### 4.6 call_agent() (`core/pipeline/agents.py`)

```python
async def call_agent(
    agent_id: str,
    user_message: Union[str, list],   # str for text; list of content blocks for vision
    system_prompt: str = "",
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = 5,
) -> tuple[str, int]:                 # (response_text, duration_ms)
```

For vision calls (UC200 OBSERVER), pass a list of Anthropic content blocks:
```python
[
  {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
  {"type": "text", "text": "Fill in the feature observation form for this image."},
]
```

### 4.7 Namespace system

All rules and tools carry a `dataset_tag`. This prevents knowledge from different domains from cross-contaminating. Both `RuleEngine` and `ToolRegistry` filter by tag at retrieval time.

Example tags: `arc-agi-legacy`, `arc-agi-3`, `bird-uc200`, `derm-uc201`

### 4.8 Shim pattern

Each use case has thin shim files `rules.py` and `tools.py` that:
1. Set `_KF_ROOT` and insert it into `sys.path`
2. Override `DEFAULT_PATH` to point at the use-case-local JSON files
3. Re-export all public symbols from `core.knowledge.rules` / `core.knowledge.tools`

---

## 5. Specialization 1 — ARC-AGI-2 (static puzzles)

**Task definition:** A set of `(input_grid, output_grid)` demonstration pairs plus one test input. Goal: produce the correct output grid for the test input.

**Data source:** `C:/_backup/arctest2025/data/training/arc-agi_training_challenges.json` + `_solutions.json`

**Round mapping:**

| Round | Agent | Input | Output |
|---|---|---|---|
| 0 | RuleEngine | task_id, grid dimensions, colors | matched rules |
| 1 | SOLVER (×1–3) | demo grids + rule actions | transformation hypothesis (text) |
| 2 | MEDIATOR | hypothesis + rules | pseudo-code: ordered list of tool calls |
| 3 | EXECUTOR | pseudo-code + demo grids | pass/fail + cell accuracy |
| 3 (revision) | MEDIATOR | failure trace + insight | revised pseudo-code |
| 3 (tool gen) | tool_creator | tool name + failing demos | new Python function |

**Key files:**

| File | Purpose |
|---|---|
| `usecases/arc-agi-2/python/harness.py` | CLI — task selection, batching, results output |
| `usecases/arc-agi-2/python/ensemble.py` | `run_ensemble()` — main per-task orchestrator |
| `usecases/arc-agi-2/python/executor.py` | `run_executor()`, `test_tool_code()` |
| `usecases/arc-agi-2/python/agents.py` | `run_solvers_round1()`, `run_mediator()`, `run_tool_generator()`, `run_tool_generator_fix()` |
| `usecases/arc-agi-2/python/v2_progress.json` | Single source of truth for all processed v2 tasks |
| `usecases/arc-agi-2/python/failed.json` | Failure subset — 5 remaining tasks |

**v2 task sequencing:**
- v2 tasks = all tasks NOT in `v1_ids.json`
- Last processed: v2 index 47 (`1990f7a8`); next batch starts at index 48
- Use `--task-list <file>` — do NOT use `--all --offset N` (offset is broken with `--all`)

**Known issues resolved:**
- `agents.py`: bare `_cost_tracker.add()` → `get_cost_tracker().add()` (two sites)
- `executor.py`: `run_executor()` wrapped in try/except in `test_tool_code()` to handle tool crashes gracefully
- `tools.json`: `assemble_quadrant_shapes` uses adaptive midpoint (shifts split by 0.5 when cells land exactly on `mid_col` but not `mid_col+1`)

**Note:** `tools.json` and `rules.json` are gitignored — they are local workspace files. Document significant tool fixes in commit messages.

---

## 6. Specialization 2 — ARC-AGI-3 (interactive environments)

**Task definition:** An interactive game environment where the agent must take sequential actions to advance `levels_completed`. Unlike ARC-AGI-2, there are no fixed input/output demo pairs — the agent explores and learns through live interaction.

**Current status:** Early exploration only. No KF integration yet. See `tests/arc-agi-3/` for the existing explorer and playlog viewer.

**Environment facts (LS20):**
- Actions: ACTION1, ACTION2, ACTION3, ACTION4 (appear to be two reversal pairs)
- State: grid with color values; diffs are localized (~52 cells per directional action, ~2 for status strip)
- Goal: advance `levels_completed` from 0/7; no completion achieved in early runs
- SDK entry: `import arc_agi` — requires conda env `arc` (NOT the default Python env)

**Proposed round mapping:**

| Round | Agent | Input | Output |
|---|---|---|---|
| 0 | RuleEngine | env_id, grid hash, current level | matched rules (known action sequences, level patterns) |
| 1 | OBSERVER | current grid state + action history | structured state interpretation (what changed, what is goal state?) |
| 2 | MEDIATOR | interpretation + rules + GoalManager state | action plan (sequence of steps) |
| 3 | ACTOR | action plan + live environment | execute `env.step()` calls, observe results, update StateManager |
| 3 (revision) | MEDIATOR | failed plan + new observations | revised plan |

**Key differences from ARC-AGI-2:**

| Aspect | ARC-AGI-2 | ARC-AGI-3 |
|---|---|---|
| Verification signal | Cell accuracy vs expected grid | `levels_completed` increasing |
| State persistence | Reset per task | Sequential across steps — StateManager essential |
| Tool type | Python grid transform function | Reusable action sequence: `def advance_level_1(env): ...` |
| Data source | JSON challenge files | `arc_agi` Python SDK |
| GoalManager role | Optional | Central — tracks `explore → complete level N → try sequence X` |

**Observation format for OBSERVER:**
```json
{
  "action_taken": "ACTION2",
  "diff_count": 52,
  "changed_cells": [{"row": 10, "col": 5, "from": 3, "to": 12}, "..."],
  "bbox": {"x_min": 13, "x_max": 38, "y_min": 40, "y_max": 62},
  "levels_completed": 0,
  "game_state": "NOT_FINISHED"
}
```

**What needs to be built:**
1. `usecases/arc-agi-3/python/harness.py` — wraps `arc_agi` SDK, runs episodes, tracks `levels_completed`
2. `usecases/arc-agi-3/python/ensemble.py` — same 4-round structure; EXECUTOR → ACTOR calling `env.step()`
3. `usecases/arc-agi-3/python/rules.py` + `tools.py` — shims with `dataset_tag = "arc-agi-3"`
4. OBSERVER prompt — produces the structured diff record above
5. ACTOR — executes the plan and feeds results back to MEDIATOR

---

## 7. Specialization 3 — Image classification UC200 (birds)

For the full experiment background, dataset details, baseline results, and failure analysis, see the [UC200 README](../usecases/image-classification/README.md).

**Task definition:** Classify a test image into one of two confusable species, using expert-authored rules and optionally a few labeled examples. The challenge is fine-grained visual discrimination where zero-shot VLMs fail on targeted pairs.

**Current experiment results (GPT-4o, CUB-200-2011, 15 confusable pairs, 600 images):**

| Condition | Accuracy |
|---|---|
| Zero-shot | 78.0% |
| Few-shot (3 images/class) | 82.8% |
| KF-patched (best run) | 75.5% |

KF helped on 3 pairs (+3 to +10pp), was neutral on 5, and hurt on 7. Root causes and fixes:

| Failure mode | Example | Fix |
|---|---|---|
| Wrong patch source | Tree Sparrow described wrong species | Expert verification loop; never use `auto_accept=True` |
| Non-visual rules injected | Crow pair used vocalizations, habitat | `add_rule(observability_filter=True)` — already in core |
| Holistic classification despite rules | Tern pairs | Two-phase architecture (see below) |

**Why the current single-pass approach fails — and the fix:**

```
Current:  [rules + image] → model → label          (model still classifies holistically)
Proposed: image → model → feature_record → rules → label
```

**Proposed round mapping:**

| Round | Agent | Input | Output |
|---|---|---|---|
| 0 | RuleEngine | pair name (e.g. `crow_vs_fish_crow`) | matched visual rules |
| 1 | OBSERVER (VLM) | image + feature schema from ToolRegistry | structured feature observation record |
| 2 | MEDIATOR | feature record + rules | classification decision + reasoning |
| 3 | VERIFIER | decision + few-shot labeled images | consistency check; revision signal if inconsistent |

**Feature observation record (Round 1 output):**
```json
{
  "task_id": "crow_vs_fish_crow",
  "image_id": "test_001",
  "features": {
    "bill_length_vs_head_depth": {"value": "equal", "confidence": 0.7},
    "bare_facial_skin_extent":   {"value": "forehead_and_eye", "confidence": 0.9},
    "outer_tail_feathers_spotted": {"value": "no", "confidence": 0.8}
  },
  "notes": "Facial skin clearly extends past eye; bill pale yellowish."
}
```

The feature schema (which questions to ask) is stored as a `tool_type="schema"` entry in ToolRegistry, namespaced to `bird-uc200`. Retrieved via `tool_registry.get_schema("crow_vs_fish_crow_schema")`. Generated fresh if not present.

**Key design principles:**

1. **Observability filter** — `add_rule(observability_filter=True)` at rule-extraction time. Already implemented in `core/knowledge/rules.py`. Rejects rules mentioning vocalizations, habitat, range, season, or size-without-reference.

2. **Confidence gating** — MEDIATOR skips feature claims with confidence < 0.5 and returns "uncertain" rather than a wrong label.

3. **KF + few-shot combined** — strongest expected configuration. Few-shot grounds the VERIFIER; KF rules guide the OBSERVER's attention. They address orthogonal failure modes.

4. **Structured evidence as auditable artifact** — the feature record is the unit of expert review. An expert can correct `"outer_tail_feathers_spotted": "no"` and the decision updates automatically — without rewriting prompts or retraining the model.

**Vision API:** The OBSERVER calls `call_agent()` with a list of content blocks. Already supported in `core/pipeline/agents.py`:
```python
user_message = [
    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
    {"type": "text", "text": f"Fill in this feature schema:\n{json.dumps(schema, indent=2)}"},
]
```

**What needs to be built:**

| File | Purpose |
|---|---|
| `usecases/.../python/harness.py` | Loads CUB-200-2011 pairs, runs classification, reports per-pair and aggregate accuracy |
| `usecases/.../python/ensemble.py` | 4-round orchestrator; SOLVER→OBSERVER (VLM with image), EXECUTOR→VERIFIER |
| `usecases/.../python/rules.py` | Shim with `dataset_tag = "bird-uc200"` |
| `usecases/.../python/tools.py` | Shim with `dataset_tag = "bird-uc200"` |
| Task adapter | Produces `task` dict: `{image_path, class_a, class_b, few_shot_a: [...], few_shot_b: [...]}` |
| OBSERVER prompt | Instructs VLM to fill in the feature schema for the image |
| VERIFIER | Checks decision consistency against few-shot images; returns revision signal |

**Existing assets to reuse:**
- 118 expert rules already extracted for all 15 pairs (as of 2026-03-29) — need migration into RuleEngine with `dataset_tag = "bird-uc200"` and observability filtering
- Baseline results: `usecases/image-classification/results/results_20260329T104850.json`
- CUB-200-2011 dataset: confirm local path with user (default harness data dir is `C:/_backup/arctest2025/data/training` but CUB images are elsewhere)
- 15 confusable pairs and per-pair results: [README Section 7.3](../usecases/image-classification/README.md#73-bird-experiment-results)

---

## 8. Harness CLI conventions

All harness implementations should follow this shared convention:

```
python harness.py
  --task-id TASK_ID          # run a single task by ID
  --task-list FILE           # JSON list of task IDs to run
  --all                      # run all tasks in the dataset
  --offset N                 # skip first N tasks  ⚠ broken with --all; use --task-list instead
  --limit N                  # cap at N tasks
  --resume                   # append to existing output file instead of overwriting
  --output FILE              # results output path
  --dataset DATASET          # data source label
  --dataset-tag TAG          # namespace tag for rules and tools
  --prune                    # run pruning pass after the batch
  --quiet                    # suppress non-essential terminal output
```

---

## 9. Extension guide: adding a new domain

1. Create `usecases/<domain>/python/` with `harness.py`, `ensemble.py`, `rules.py`, `tools.py`
2. In shims `rules.py` and `tools.py`:
   - Set `_KF_ROOT = Path(__file__).resolve().parents[N]` and insert into `sys.path`
   - Override `DEFAULT_PATH` to the use-case-local JSON file
   - Re-export all public symbols from `core.knowledge.rules` / `core.knowledge.tools`
3. Write a task adapter that produces a standard `task` dict for your domain
4. Map domain-specific agents to the 4-round structure:
   - Round 1: Perceiver / Hypothesis Generator
   - Round 2: Mediator / Synthesizer
   - Round 3: Executor / Verifier (with revision loop)
5. Define what a **tool** means in your domain:
   - Grid transform function (ARC-AGI-2)
   - Feature observation schema / JSON form (UC200)
   - Reusable action sequence (ARC-AGI-3)
6. Write domain-specific prompts under `usecases/<domain>/prompts/`
7. Set `--dataset-tag <your-domain>` to namespace rules and tools away from other domains
8. For vision domains: use `call_agent(user_message=[...content blocks...])` — already supported in core

---

## 10. Operational notes

- `tools.json` and `rules.json` are gitignored in each use case directory — they are local workspace files that accumulate across runs. Document significant tool fixes or rule schema changes in commit messages.
- Always run harness commands in the foreground so token consumption can be monitored and interrupted with Ctrl+C.
- `usecases/image-classification/` is under active development by a separate developer as of 2026-04-01 — coordinate before editing files in that directory.
- ARC-AGI-3 requires the `arc` conda environment — the default Python venv does not have the `arc_agi` SDK.
- For ARC-AGI-2 v2 task sequencing: last processed is v2 index 47 (`1990f7a8`); 5 failures remain; next batch starts at index 48. Use `--task-list` not `--all --offset`.
