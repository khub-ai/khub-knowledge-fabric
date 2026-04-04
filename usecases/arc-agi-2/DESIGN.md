# ARC-AGI Ensemble — Design Document

## Overview

This system solves ARC-AGI puzzles using a multi-agent ensemble with a rule base. The core idea: separate *reasoning about the pattern* (text-only) from *executing the pattern* (deterministic code), and accumulate reusable knowledge across puzzle runs.

---

## Architecture

```
Round 0:  Rule Matching     — check rule base for known pattern
Round 1:  Solver(s)         — propose hypothesis in natural language (parallel if multiple)
Round 2:  MEDIATOR          — verify hypothesis against demos, write pseudo-code + optionally request new tools
          Tool Generator    — if new tools requested: generate Python, verify against demos, self-correct, register
Round 3:  EXECUTOR          — run pseudo-code deterministically against all demo pairs
          if all pass  →  apply to test input → done
          if fail      →  MEDIATOR revises (up to MAX_REVISIONS)
Final:    Rule Updates       — MEDIATOR writes/merges rules; auto-deprecate failing rules
```

---

## Key Design Principles

### 1. Reasoning and execution are strictly separated

Solvers and MEDIATOR never produce output grids. They produce *descriptions* and *pseudo-code*. EXECUTOR runs the actual transformations using deterministic Python tools. This separation means:
- Reasoning errors and implementation errors can be diagnosed independently
- The same pseudo-code can be re-run without repeating the reasoning step
- Tool correctness can be verified independently of the hypothesis

### 2. Verification before commitment

MEDIATOR is required to mentally trace every proposed pseudo-code sequence against **all demo pairs** before writing the output JSON. This is enforced in the prompt. If the trace reveals a mismatch, MEDIATOR must revise the hypothesis before committing. The goal: catch bad pseudo-code at reasoning time, not execution time.

### 3. Dynamic tool generation with demo-verified self-correction

When MEDIATOR needs a transformation that no existing tool can express, it requests a new tool via a `new_tools` JSON block. The tool generation loop:

1. MEDIATOR specifies `name`, `description`, `args`, `behavior` (precise natural language)
2. `run_tool_generator()` calls Claude with the spec **and the demo pairs** — giving the generator concrete input/output examples, not just prose
3. Generated Python is run against all demos deterministically
4. If any demo fails, the failing diff is sent back to Claude for self-correction (up to 3 attempts)
5. Only verified code is registered in `_TOOL_REGISTRY` and becomes available to EXECUTOR

**Why demo pairs matter**: English descriptions of geometric operations are ambiguous (inclusive vs exclusive endpoints, which cell is the corner, etc.). Concrete examples resolve the ambiguity the same way they do for MEDIATOR.

This is the mechanism that makes the system genuinely autonomous — it can acquire new capabilities without hand-coded tools.

### 4. Configurable solver ensemble

By default one solver (`SOLVER`) runs per puzzle. The solver prompt covers spatial, procedural, and analogical reasoning in a single pass. This is efficient for most puzzles.

For hard puzzles, switch to multiple specialist solvers:
```python
# In agents.py
DEFAULT_SOLVERS = ["SOLVER-SPATIAL", "SOLVER-PROCEDURAL", "SOLVER-ANALOGICAL"]
```
Or pass `solver_ids` to `run_ensemble()` at runtime. The three specialist prompts (`solver-spatial.md`, `solver-procedural.md`, `solver-analogical.md`) are preserved for this purpose.

**Why multiple solvers help (when they do)**: Different specialists notice different features. SPATIAL catches geometric structure; PROCEDURAL catches cell-by-cell formulas; ANALOGICAL matches to known transformation categories. If one is wrong, the others can outvote it. Most value on hard puzzles where the pattern is genuinely ambiguous.

**Why one solver is sufficient for easier puzzles**: The specialists tend to converge on the same answer when the pattern is clear. Three API calls for identical conclusions is waste. The combined `solver.md` prompt gets the same coverage in one call.

### 5. Rules as transferable knowledge

The rule base (`rules.json`) accumulates solved patterns across puzzle runs. After each puzzle:
- MEDIATOR writes rules describing the pattern it just solved
- Rules are structured as `condition → action` pairs with category tags
- On the next puzzle, Round 0 runs a matching pass — if a rule matches, MEDIATOR gets it as prior knowledge

**Rule quality principles** (enforced via prompt):
- Conditions begin with a category tag: `[gravity]`, `[path-drawing]`, `[proximity]`, etc.
- Conditions describe a *category* of puzzles, not a specific puzzle (no hardcoded colors, sizes, coordinates)
- Before writing `action: new`, MEDIATOR checks all existing rules — if a similar one exists, prefer `merge` or `generalize`
- `auto_deprecate()` runs after every puzzle: rules fired ≥3 times with 0 successes are deprecated automatically

**Rule lineage**: every rule records how it was created (`new`, `generalized`, `specialized`, `merged`, `consolidated`) and which parent rules it derived from. This allows tracing how knowledge evolves across puzzles.

### 6. Failure handling protocol (operator process)

When the system fails to solve a task autonomously, the correct operator response is **not** to provide the solution directly (e.g., via `--insight` or `--revision-hint`). That would mask the underlying capability gap and prevent the system from becoming self-sufficient.

The correct process is:

```
1. Understand the correct solution
   — Analyze the demo pairs manually (or ask the operator).
   — Identify the exact transformation rule, including any role asymmetries
     between different object groups (e.g., longest sequence vs. shorter ones).

2. Identify the system gap
   — Why did the solver not hypothesize the correct rule?
     (Missing reasoning step? Wrong default assumption? Bias toward simpler properties?)
   — Why did MEDIATOR fail to synthesize correct pseudo-code?
     (0 steps? Wrong tool requested? Tool behavior spec too vague?)
   — Why did the tool generator fail?
     (Too complex in one tool? Missing primitives? Namespace limitations?)
   — Was it a revision strategy failure?
     (MEDIATOR kept reusing the same broken tool across all revisions?)

3. Repair the gap
   — Fix prompts, tool generation context, revision strategy, or add builtin primitives
     as appropriate. The fix should address the *class of failure*, not the specific task.
   — Do NOT hardcode task-specific knowledge (e.g., exact color maps or coordinate offsets).
     The repair should generalize to structurally similar puzzles.

4. Validate autonomously
   — Re-run the failing task with NO hints. If it solves, the gap is repaired.
   — If it still fails, repeat from step 2 — the gap may be deeper or multiple gaps exist.
```

**Why this matters**: directly providing the solution via `--insight` produces a one-time fix that doesn't transfer. The system learns nothing about the *class of reasoning* it was missing. Gap-repair-first ensures that every hard puzzle makes the system permanently more capable — both for similar future puzzles (via task rules and improved prompts) and for different puzzles that share the same reasoning gap.

**When to use `--insight` legitimately**: only to trigger preference rule extraction after a correction event — i.e., after the gap has already been repaired and the task succeeds autonomously. The insight then documents *what the system was doing wrong before the repair*, not what it should do now.

### 7. Generalization policy

When a task rule is created, the post-success generalization pass asks MEDIATOR to propose broader candidate variants. Not every dimension of a rule should be generalized — the goal is to produce variants that will plausibly fire on structurally similar puzzles.

**What to generalize (free parameters):**
- Numeric thresholds that were inferred from a specific task (e.g., `max_size=2`, a particular color value)
- Conditions that reference specific colors by value rather than by role (e.g., "color 1 and 5" → "any non-background color")
- Positional descriptions that may not hold in all instances of the pattern class

**What NOT to generalize:**
- The core algorithm — if `recolor_small_components` is the right tool, that stays fixed
- Conditions that are essential to discriminate this pattern from others — removing them produces a rule that fires incorrectly on unrelated tasks
- Parameters whose variation would require a *different* tool, not just different arguments

**Generalization dimensions for common pattern classes:**

| Pattern class | Free parameters to generalize |
|---|---|
| Component-size recolor | `max_size` (threshold), `new_color`, background value |
| Diagonal radiation | Tip direction (topmost vs bottommost), number of sequences |
| Topology (hole count) | `object_color`, `color_map` values |
| Gravity / sorting | Direction (up/down/left/right), object selection criterion |
| Tiling / scaling | Scale factor, flip/rotation axis |
| Path drawing | Marker color, line color, connection rule |

**Candidate promotion policy:**
- New generalizations enter at `status: candidate`
- A candidate is promoted to `status: active` only after it fires and succeeds on a task different from its source task
- A candidate is auto-deprecated after 1 failure (stricter than active rules which allow 3)
- This prevents speculative generalizations from accumulating and polluting Round 0 matching

**Avoiding over-generalization:**
A rule that is too broad fires on tasks it cannot solve, accumulating failures and being deprecated. The right level of generality is: broad enough to cover the pattern class, specific enough to exclude unrelated patterns. When in doubt, prefer a narrower condition — it is easier to broaden later than to recover from a rule that fires incorrectly everywhere.

### 8. Human-in-the-loop is optional and non-blocking (for hints)

Prefilled hints (`--hypothesis`, `--insight`, `--revision-hint`) inject at checkpoints without waiting. If not provided, the system runs fully autonomously. The design intent: hints should *accelerate* convergence, not be required for correctness. A hint that is wrong can actively mislead MEDIATOR — the system does not yet have a mechanism to reject bad hints.

**Current guidance**: provide no hints by default. Only intervene if MEDIATOR's hypothesis is clearly wrong after Round 1 or fails all demos across multiple revisions.

### 7. Human-natural reasoning preference

ARC-AGI-V2 evaluation has an important property: when multiple transformation rules fit all demo pairs equally well, the one that is **preferred by humans** is considered correct. This systematically excludes solutions that are computationally convenient but unlikely to be considered by a human solver — e.g. "recolor objects where bounding box area > 12" vs "recolor objects with a hole in them."

The solver currently has no built-in preference for human-natural over computationally-natural hypotheses. This creates a failure mode: the solver picks the hypothesis that is easiest to express as code, which often differs from the one a human would reach first.

**Human-natural properties** (humans perceive these first):
- Topological structure (number of enclosed holes, connectedness, shape identity)
- Perceptual grouping (proximity, color, orientation)
- Relative position (left/right of divider, inside/outside boundary)
- Symmetry and reflection

**Non-human-natural properties** (easy to compute, hard to eyeball):
- Exact pixel/cell count
- Bounding box area or aspect ratio
- Lexicographic ordering of color values

This principle cannot be fully solved by hardcoding a preference list — such a list would be incomplete, potentially wrong, and wouldn't transfer to new puzzle categories. Instead, preferences are **learned from corrections** (see §8).

### 8. Failed-task gap-repair protocol

When the system fails on a task, the correct response is **never to hand it the answer**. Instead, follow this protocol:

#### Step 1 — Find the correct hypothesis independently
Analyze the demo pairs manually (or with assistance). Derive the complete, precise rule that transforms every demo input into its output. Do not proceed until the hypothesis is verified against all demos.

#### Step 2 — Identify the system gap
Ask: *why didn't the system reach this hypothesis on its own?* Every failure has a class-level cause:

| Gap type | Symptom | Example |
|---|---|---|
| **Missing tool** | MEDIATOR requested a tool; tool generator failed or produced wrong code | `radiate_sequences`, `fill_blocks_from_key` — no reliable implementation existed |
| **Solver reasoning gap** | Solver's hypothesis was partially correct but missed a critical property | Solver saw "groups radiate" but missed the asymmetric role (longest goes first, tip-first) |
| **Missing reasoning step** | Solver never considered a relevant property class | Solver didn't test sequential vs simultaneous application; didn't look for key-to-block rotation |
| **Rule base noise** | Conflicting or wrong rules fired and confused MEDIATOR | Multiple near-duplicate rules with wrong color_map blocked the correct one |
| **MEDIATOR prompt gap** | MEDIATOR has no guidance about a specific failure mode | No instruction to consider parallel vs sequential processing during revision |

#### Step 3 — Repair the gap at class level
Fix the **root cause**, not the symptom. The repair must generalize beyond this one task:

| Gap type | Class-level repair | What NOT to do |
|---|---|---|
| **Missing tool** | Add a verified builtin to `grid_tools.py` + `executor.py` + mention in `mediator.md` | Add a task-specific dynamic tool directly to `tools.json` |
| **Solver reasoning gap** | Add an analysis step to `prompts/solver.md` that prompts for the missing property | Add the correct answer as a hint in the run command |
| **Missing reasoning step** | Add the reasoning step to `solver.md` and/or MEDIATOR revision prompt in `agents.py` | Add a rule that encodes the complete solution for this task |
| **Rule base noise** | Deprecate incorrect rules; fix the condition text of ambiguous rules | Re-run until the system happens to get it right |
| **MEDIATOR prompt gap** | Add guidance to `prompts/mediator.md` or the revision prompt in `agents.py` | Bypass MEDIATOR with a pre-coded pseudocode rule |

#### Step 4 — Validate: let the system solve autonomously
After repairing the gap, run the task with **no hints and no task-specific rules**. The system must:
- Propose the correct hypothesis via the Solver (guided by the repaired reasoning prompts)
- Reach for the correct tool (which now exists as a verified builtin)
- Pass all demos and apply to the test input

If it still fails, identify the *next* gap and repeat. Do not proceed to the next failed task until this one passes autonomously.

#### Step 5 — Document in LEARNINGS.md and SOLVE_LOG.md
For each fixed failure:
- Record the failure type, root cause, and fix in `LEARNINGS.md`
- Add a solve entry to `SOLVE_LOG.md` under the correct session

#### The critical boundary: gap repair vs. answer injection

The bright line is this:

> **Filling a gap** = making the system *capable* of finding a class of solutions (tool, reasoning step, prompt guidance). The system still has to find this particular solution on its own.
>
> **Answer injection** = encoding the specific solution to this specific task (task-specific rule with explicit pseudocode, pre-filled hypothesis hint, direct `--insight` with the answer). This bypasses the solver entirely.

Gap repairs are permanent improvements — they help on all future similar tasks. Answer injections are one-time shortcuts that tell us nothing about whether the system actually understands the problem class. They must be avoided.

**Concrete examples of the distinction:**

| Action | Type | Why |
|---|---|---|
| Add `fill_blocks_from_key` builtin to `grid_tools.py` | Gap repair | Any puzzle in this class can now be solved; solver still decides when and how to use it |
| Add r_107 with `pseudocode: [{tool: fill_blocks_from_key, args: ...}]` pre-seeded for task 103eff5b | Answer injection | Round 0 fires this rule and skips the solver entirely |
| Add "check for key-to-block-layout rotation" step to `solver.md` | Gap repair | Solver now looks for this property class; must still work out the specific rotation |
| Run with `--insight "rotate key 90 degrees CW before mapping"` | Answer injection | Gives MEDIATOR the specific transformation to apply |

### 9. Preference rules: learning from corrections (after gap repair)

When the system gets a puzzle wrong and a human correction succeeds, that is a training event:

```
wrong_hypothesis → human insight → correct_hypothesis → success
```

The system extracts a **preference rule** from this triple: not a solution for the specific puzzle, but a general reasoning bias about *which hypothesis property to prefer* when evidence is ambiguous.

**Preference rules vs task rules**:

| Property | Task rule | Preference rule |
|---|---|---|
| Encodes | How to solve puzzle type X | Which hypothesis property to prefer |
| Applied | Per-puzzle (matched in Round 0) | Every puzzle (universal soft prior) |
| Created by | MEDIATOR after solving | MEDIATOR after correction event |
| Triggered by | Normal task completion | `--insight` used + task succeeded |
| Overridable | By stronger matching rules | By demo evidence or future corrections |

**Key design constraints**:
- Preferences are **soft priors**, not mandates. The solver is explicitly told: "demo evidence overrides a prior." A future puzzle can provide counter-evidence that causes the prior to be revised.
- Preferences are **not hardcoded by the developers**. They emerge from observed correction events. A developer-imposed preference list would embed developer assumptions (potentially wrong) and wouldn't generalize.
- Preferences **accumulate and can be revised**. If a preference rule leads the solver astray on a future puzzle, that failure generates a correction that specializes or contradicts the prior — just as a human refines intuitions from experience.

**Long-term alignment significance**: This mechanism models how human preferences are transferred to an AI system — not by explicit rule specification, but by observing corrections and learning what the corrector cares about. The same architecture that learns "prefer topology over pixel count for ARC-AGI" can in principle learn "prefer fairness over efficiency in resource allocation" from social corrections. The goal is a system that models the *cognitive process* a human uses, not just a lookup table of known preferences.

---

## File Map

| File | Role |
|------|------|
| `python/harness.py` | CLI entry point; parses args, loads tasks, calls `run_ensemble()` |
| `python/ensemble.py` | Main orchestrator; runs all rounds, manages rule updates |
| `python/agents.py` | All LLM calls (solvers, MEDIATOR, tool generator); retry on 529 |
| `python/executor.py` | Deterministic tool runner; parses pseudo-code JSON; registers dynamic tools |
| `python/grid_tools.py` | Grid transformation tools (`gravity`, `flood_fill`, `gravity_by_type`, etc.) |
| `python/rules.py` | `RuleEngine`: CRUD, matching, prompt builders, auto-deprecation |
| `python/display.py` | Rich terminal UI; prefilled checkpoint injection |
| `python/metadata.py` | Data classes (`SolverEntry`, `MediatorDecision`, `TaskMetadata`) |
| `prompts/solver.md` | Combined solver prompt (spatial + procedural + analogical) |
| `prompts/solver-spatial.md` | Specialist: geometric/visual reasoning |
| `prompts/solver-procedural.md` | Specialist: cell-by-cell algorithmic reasoning |
| `prompts/solver-analogical.md` | Specialist: classification by known transformation category |
| `prompts/mediator.md` | MEDIATOR: verification, pseudo-code synthesis, rule management, tool requests |
| `python/rules.json` | Runtime rule base (gitignored — per-environment state) |

---

## Namespace System

Rules and tools are tagged with dataset namespaces so knowledge from one dataset generation cannot pollute another.

### Schema

Every rule and tool carries two new fields:

```json
{
  "tags":  ["arc-agi-legacy"],
  "scope": "dataset"
}
```

- **`tags`**: free-string list of dataset names this entry applies to (e.g. `"arc-agi-legacy"`, `"arc-agi-3"`, `"my-custom-dataset"`). No enum — any string is valid; a simple validation list in `harness.py` guards against typos at runtime.
- **`scope`**: `"dataset"` (default) fires only when the current dataset tag is in `tags`; `"global"` fires for every dataset regardless of `tags`. Global scope must be **explicitly assigned** — the system never auto-promotes to global. Only pure primitives with no domain specificity (rotate, flood_fill, replace_color, etc.) should be global.

Matching rule:
```python
def is_active_for(entry, current_dataset):
    return entry["scope"] == "global" or current_dataset in entry["tags"]
```

### Per-namespace stats

Flat `stats.fired` / `stats.succeeded` are replaced by a per-namespace dict:

```json
"stats_by_ns": {
    "arc-agi-legacy": { "fires": 12, "successes": 10 },
    "arc-agi-3":      { "fires": 3,  "successes": 0  }
}
```

This enables per-namespace auto-deprecation (remove a namespace from `tags` when it underperforms) without killing the rule for other namespaces.

### Auto-tagging

When the MEDIATOR writes a new rule or a tool is registered, the harness stamps the current `dataset_tag` automatically. No manual tagging needed during normal operation.

### Cross-namespace promotion

If a `"dataset"`-scoped rule succeeds on a namespace it wasn't originally tagged for, a human can add that namespace to `tags`. The system flags candidates (high success rate on a new namespace) but never promotes automatically — cross-namespace transfer is a human decision.

### Migration

- All existing rules → `"tags": ["arc-agi-legacy"], "scope": "dataset"`
- All existing tools in `tools.json` → same
- All builtin functions in `grid_tools.py` / `executor.py` → `"scope": "global"`

---

## Registry Pruning

Rules and tools follow a four-state lifecycle to prevent the registry from accumulating low-quality entries indefinitely.

### Lifecycle

```
active  →  flagged  →  deprecated  →  archived
```

- **active**: in rotation, included in matching and prompt context
- **flagged**: underperforming or redundant, still active but marked for human review; shown with a warning label in prompts
- **deprecated**: excluded from matching; kept in `rules.json` / `tools.json` for audit history
- **archived**: moved to cold-storage files (`rules_archive.json`, `tools_archive.json`); never loaded at runtime. Archive is irreversible and a human-only action.

### Performance pruning (auto, per-namespace)

```
fires < 10                              → too early to judge, no action
fires ≥ 10,  success_rate == 0.0        → deprecate (remove namespace from tags)
fires ≥ 5,   success_rate < 0.20        → flag for review
fires ≥ 10,  success_rate < 0.40        → flag for review
```

Deprecation removes the **namespace** from `tags`, not the whole rule. If `tags` becomes empty, the rule is fully deprecated. Threshold is 10 fires (raised from the original 3 — new namespaces need time to accumulate signal before entries are killed).

### Staleness pruning (auto, per-namespace)

A rule that never fires provides no signal and wastes prompt space:

```
tasks_processed_in_ns ≥ 50,  fires_in_ns == 0  → flag
tasks_processed_in_ns ≥ 100, fires_in_ns == 0  → deprecate
```

`tasks_processed_in_ns` is tracked as a counter in the namespace stats block.

### Redundancy pruning (semi-auto)

Detected via two signals:
- **Condition overlap**: two rules fire on the same set of tasks repeatedly (high Jaccard on `fired_on` task ID lists). Auto-flag both; human confirms which to keep.
- **Action conflict**: two rules with overlapping conditions point to different tools. Flag both; higher success-rate rule survives.

The system surfaces redundancy candidates in the `--stats --section rules` report. Human confirms deprecation.

### `--prune` maintenance command

```bash
# Deprecate all underperforming rules for a namespace
python harness.py --prune --namespace arc-agi-legacy --min-success-rate 0.4 --min-fires 10

# Show flagged entries without changing anything
python harness.py --prune --dry-run --namespace arc-agi-legacy
```

---

## Known Limitations and Future Work

### Human hint trust
MEDIATOR currently trusts human hints unconditionally. A wrong hint is worse than no hint because MEDIATOR spends revision budget trying to reconcile the hint with the demos. **Future**: give MEDIATOR explicit permission to reject or downweight a human hint if it contradicts the demo evidence.

### Preference rule quality
Preference rules are extracted by MEDIATOR from correction events, which means their quality depends on MEDIATOR's ability to generalize correctly from a single example. A rule extracted too specifically will fail to transfer; a rule extracted too broadly may suppress correct hypotheses. **Future**: validate preference rules by re-running past puzzles with and without the new prior; only retain rules that improve accuracy on held-out tasks.

### Solver asymmetric role blindness
The solver may treat all non-zero groups as having the same role, missing cases where different groups transform differently based on a property like length rank, position, or color. The solver prompt now explicitly asks "do all groups transform identically or do different groups have different roles?" — but the solver may still converge on a symmetric description when an asymmetric one is correct. **Future**: add a solver reasoning step that explicitly enumerates all observable group properties (length, orientation, color, position) and tests whether those properties predict which groups have which behavior.

### Rule matching reliability
Rule matching uses an LLM call (MEDIATOR reads all rules as text and decides which apply). This can miss matches or produce false positives. **Future**: structured condition predicates (object count, color set, grid shape, transformation category) that can be matched programmatically before the LLM pass.

### Rule-triggering efficiency (two-stage retrieval) — Future
At ~100 rules the single LLM matching call is manageable (~100 tokens/rule × 100 rules). At 500+ rules per namespace it degrades: prompt noise reduces match quality and cost grows linearly.

**Planned two-stage approach:**
1. **Stage 1 — Python pre-filter (no LLM):** Analyze task grid properties (has_solid_border_row, color_count, grid_size, has_isolated_objects, etc.) and match against per-rule feature tags written at rule-creation time. Narrows 500 candidates to ~30 without any API call.
2. **Stage 2 — LLM matching (current approach):** Send only the 30 pre-filtered candidates to the LLM. Same quality, much smaller prompt.

**Embedding-based retrieval** (alternative for 5000+ rules): Embed each rule's condition text; embed a description of the current task; retrieve top-K by cosine similarity. No LLM call for stage 1 but adds embedding infrastructure.

**Implementation trigger**: build stage 1 when any namespace exceeds ~300 active rules. Design the feature-tag schema into the rule schema now so rules written today are taggable without migration.

### Generated tool persistence
Dynamically generated tools are registered at runtime but not saved to disk. If a tool was useful for puzzle A and puzzle B has a similar pattern, the tool must be regenerated. **Future**: persist verified tool code alongside rules.json so tools accumulate across runs.

### ARC-AGI v3 readiness
v3 tasks are expected to require qualitatively harder reasoning. Planned upgrades before tackling v3:
- **Model**: default SOLVER/MEDIATOR to Opus 4.6 for deeper reasoning
- **Multiple solvers**: run 2–3 solvers with intentionally diverse framings (spatial, procedural, analogical) when single-solver fails; architecture already supports `solver_ids`
- **Revision budget**: raise MAX_REVISIONS from 5 to 8–10 for harder tasks
- **Demo anomaly handling**: already improved (95% accuracy threshold for test output computation)

### Solver diversity
Currently, using multiple solvers with the same underlying model (Sonnet 4.6) provides limited diversity — they tend to notice the same features. **Future**: plug in different model families (e.g., Opus for deep spatial reasoning, a fine-tuned ARC specialist) as distinct solvers when the single-solver fails.

### Leaderboard tracking
The harness writes per-run stats to `results.json`. Full leaderboard submission requires: dataset split (training vs evaluation vs test), no human hints used, reproducible results. Track `human_hints_used` per task in the results.

---

## Running the System

```bash
# Single puzzle
bash usecases/arc-agi-ensemble/run-python.sh --task-id 0e671a1a

# Batch run (V2-only tasks, offset 30, 10 tasks)
bash usecases/arc-agi-ensemble/run-python.sh --skip-ids python/v1_ids.json --offset 30 --limit 10

# With human insight
bash usecases/arc-agi-ensemble/run-python.sh --task-id 0a2355a6 --insight "topological hole count"

# Override revision count (useful during debugging to cap API spend)
bash usecases/arc-agi-ensemble/run-python.sh --task-id 0e671a1a --max-revisions 2

# Show all agent prompts and MEDIATOR output
bash usecases/arc-agi-ensemble/run-python.sh --task-id 0e671a1a --prompts
```

## Performance Stats

`stats.py` aggregates results across all saved result files and prints a structured report. Run it any time to review system health:

```bash
# Full report (all sections)
bash usecases/arc-agi-ensemble/run-python.sh --stats

# Single section
bash usecases/arc-agi-ensemble/run-python.sh --stats --section rules
bash usecases/arc-agi-ensemble/run-python.sh --stats --section failed
bash usecases/arc-agi-ensemble/run-python.sh --stats --section generalization
bash usecases/arc-agi-ensemble/run-python.sh --stats --section methods
bash usecases/arc-agi-ensemble/run-python.sh --stats --section cost

# Rules + generalization only (no task results needed)
bash usecases/arc-agi-ensemble/run-python.sh --stats --rules-only

# Specific result files
bash usecases/arc-agi-ensemble/run-python.sh --stats python/results_v2_21_30.json
```

### Sections reported

| Section | Contents |
|---|---|
| **Overview** | Total tasks, correct/failed counts, accuracy |
| **Methods** | How tasks were solved: rule-matched vs dynamic-tool vs solver-only, with task IDs |
| **Rules** | Rule base size, lineage breakdown, fire→success rate, per-rule stats sorted by successes |
| **Generalization** | Generalized/merged/candidate rule counts, which generalizations have fired and on which tasks |
| **Tools** | All dynamically generated tools, whether each contributed to a solution, task associations |
| **Cost** | Total + per-task cost, avg duration, token breakdown, API call counts |
| **Failed** | Failed task list with cell accuracy, rounds, cost, tools tried, rules matched |

### Note on result file coverage

Each harness run writes results to `results.json` (or `--output <file>`). The stats reporter auto-discovers `results.json`, `results_v2_21_30.json`, `results_v2_31_35.json` in the `python/` directory and deduplicates by task ID. To include a new batch, either use the default output path or pass the file path explicitly. Tasks run before result-file persistence was added are not recoverable in structured form — consult `SOLVE_LOG.md` for narrative history of those runs.
