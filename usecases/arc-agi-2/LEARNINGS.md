# ARC-AGI Ensemble — Learnings Log

A living record of system failures, root cause analyses, fixes applied, and design insights gained. Each entry documents not just what broke and how it was repaired, but *why* the gap existed and how the repair generalizes to future tasks. This document is the primary reference for understanding how the system's capability evolved over time.

---

## How to read this document

Each failure event has five fields:
- **Failure**: what the system did wrong
- **Root cause**: the specific gap — in a prompt, a tool, the rule base, or the architecture
- **Fix**: what was changed
- **Generalization**: how future tasks benefit from this fix (beyond the one task that exposed it)
- **Design insight**: the broader principle learned

Entries are grouped by failure *type*, not by task, so patterns across tasks are visible. Cross-references link to individual case study articles in `.private/`.

---

## Failure type 1: Solver reasoning gaps (wrong or incomplete hypothesis)

### 1.1 Solver assumes symmetric roles for all non-zero groups

**Exposed by**: `1190bc91` (sequence radiation)
**Case study**: `.private/1190bc91.md`

**Failure**: SOLVER described "each sequence radiates diagonally" — treating all sequences identically. Missed that the longest sequence has a different role (spine → diagonal radiation) from shorter sequences (peripheral → BFS expansion).

**Root cause**: The solver prompt had no step asking whether different groups play different roles. Default assumption is symmetric: all non-zero groups transform the same way.

**Fix**: Added Step 3 to solver.md analysis protocol:
> *"Ask explicitly: do all non-zero groups transform the same way, or do different groups play different roles? Look for properties that could assign roles: length/size, position, color, orientation, rank."*

**Generalization**: Any puzzle where objects have different roles based on a measurable property (size rank, position, distance from center, color uniqueness) will now be caught. Relevant for object-manipulation, sorting, gravity-by-type, and classification puzzles.

**Design insight**: The solver's default is symmetric reasoning — it describes what the majority of cells do and assumes the rest follow the same rule. Asymmetric roles require an explicit prompt to look for outliers and hypothesize a discriminating property. This is a general bias to counteract.

---

### 1.2 Solver misses ordering / sequential processing constraints

**Exposed by**: `1190bc91` (sequence radiation)
**Case study**: `.private/1190bc91.md`

**Failure**: SOLVER described diagonal radiation from the spine sequence without noticing that elements are processed tip-to-end sequentially, with already-filled cells acting as barriers for subsequent elements and for peripheral sequence expansion.

**Root cause**: The solver reasons statically about the final output state. It doesn't naturally consider *when* each cell was set relative to other cells. Ordering only reveals itself through *contested cells* — cells reachable from multiple sources where one source's color wins. The solver prompt never asked for this analysis.

**Fix**: Added Step 4 to solver.md analysis protocol:
> *"Identify contested cells — output cells reachable from multiple sources. Which source's color appears? Is there a consistent priority/ordering rule? Does simultaneous vs sequential interpretation produce different results — commit to the correct one."*

Also added to solver.md "In later rounds" revision section and to the MEDIATOR revision prompt in agents.py.

**Generalization**: Any puzzle with multiple sources that could claim the same cell needs this analysis: radiation puzzles, BFS expansion, beam projection, gravity with multiple objects, path drawing with intersections. The contested-cell test is the universal diagnostic.

**Design insight**: Static output description ≠ algorithmic description. The solver needs to reason about execution order when transformations interact. The signal is always in the contested cells — they carry information about priority that non-contested cells do not.

---

### 1.3 Solver uses wrong discriminating property (size vs topology)

**Exposed by**: `0a2355a6` (hole count recoloring)

**Failure**: SOLVER hypothesized that objects are recolored by size rank (number of cells), when the correct property is topological hole count (number of enclosed holes).

**Root cause**: Size rank is visually salient and immediately computable. Topological properties (hole count) require a more abstract analysis step. The solver defaulted to the more obvious property.

**Fix**: Preference rule `r_085` extracted via `--insight "topological hole count"` after the gap was diagnosed. Rule encodes: "when objects have different hole counts, prefer topology over size as the discriminating property."

Also added `recolor_by_hole_count` as a verified builtin with `color_map` parameter.

**Generalization**: Preference rules generalize the lesson: whenever objects have both size-based and topology-based differences, the system now has a prior favoring topology. Future tasks with hole-count discrimination will match this preference before the solver even runs.

**Design insight**: When the solver picks a plausible-but-wrong property, the right fix is a *preference rule*, not a task rule. A task rule would only fire on identical patterns; a preference rule shapes the solver's reasoning across all structurally similar tasks.

---

## Failure type 2: MEDIATOR produces 0 pseudocode steps

### 2.1 Conflicting rules in rule base paralyze MEDIATOR

**Exposed by**: `0a2355a6` and `1190bc91` (multiple retry runs)

**Failure**: MEDIATOR produced 0 pseudocode steps — `parse_pseudocode()` returned an empty list. Task automatically failed.

**Root cause**: Multiple retry runs had accumulated conflicting rules for the same task (e.g., `r_082`, `r_083`, `r_084` all matched the same pattern but specified contradicting `color_map` values). MEDIATOR saw high-confidence rules with contradicting actions and produced long reasoning text without committing to pseudocode.

**Fix**:
1. Deprecated conflicting rules manually.
2. Added mandatory instruction to MEDIATOR prompt: *"Zero steps is never acceptable — you must always produce pseudocode or explicitly request a new tool."*
3. Added `failed_tools` tracking to revision loop so MEDIATOR knows which tools have already failed.

**Generalization**: The 0-steps failure mode is now explicitly prevented. MEDIATOR must always produce either a pseudocode block or a tool-request block. Rule conflicts that could cause paralysis are now caught earlier via the conflict detection in `rules.py`.

**Design insight**: When MEDIATOR is given two high-confidence contradicting instructions, it reasons but doesn't decide. The fix is twofold: prevent contradicting rules from accumulating (better deprecation), and make 0-step output a hard error (prompt constraint). Both are needed — the prompt constraint alone doesn't fix the underlying rule quality issue.

---

### 2.2 MEDIATOR can't synthesize pseudocode when no tool matches the hypothesis

**Exposed by**: `1190bc91` (sequence radiation)

**Failure**: Even when MEDIATOR didn't produce 0 steps, it defaulted to requesting `radiate_sequences` from the tool generator — which then failed all fix attempts because the algorithm was too complex to implement from a monolithic natural-language spec.

**Root cause**: No existing tool matched the two-phase algorithm. MEDIATOR's options were: (1) request a new tool (which failed), or (2) produce 0 steps (now prevented). Neither worked. The real gap was that the algorithm decomposes into two conceptually distinct sub-algorithms that can't be reliably fused into one tool spec.

**Fix**:
1. Implemented `radiate_sequences` as a verified builtin (Phase 1: diagonal radiation, Phase 2: BFS expansion). This bypasses the tool generator entirely for this pattern class.
2. Added rule `r_102` so MEDIATOR can match the pattern in Round 0 without needing the solver to discover it.

**Generalization**: When a puzzle's algorithm decomposes into two distinct sub-algorithms, the tool generator will likely fail regardless of how many attempts it gets — because it has no mechanism to discover the decomposition from a failing cell diff. The correct fix is to implement a builtin that encodes the decomposition correctly, then expose it via the rule base.

**Design insight**: The tool generator is effective for single-algorithm transformations with clear input/output semantics. It fails systematically for multi-phase algorithms where each phase has different logic. The signal: tool generator fails all 3–5 fix attempts with cell accuracy that fluctuates rather than converges. When this happens, the fix is a builtin, not more fix attempts.

---

## Failure type 3: Tool generator failures

### 3.1 Dynamically generated tool hardcodes task-specific parameters

**Exposed by**: `12eac192` (small component recoloring)
**Case study**: `.private/12eac192.md`

**Failure** (latent, caught before running): The dynamically generated `recolor_small_components` tool in the registry hardcoded `target_val in [1, 5, 8, 7]`. This would have worked for task 12eac192 but silently failed on any task with different colors.

**Root cause**: The tool generator produces code that passes the specific demo pairs it was verified against. It has no incentive to generalize beyond those demos — and no mechanism to test generalization. Hardcoded values that happen to match the demos look correct to the verifier.

**Fix**: Implemented `recolor_small_components` as a proper builtin that processes ALL non-background colors (determined by `background` parameter, not hardcoded), with `max_size` and `new_color` as explicit parameters inferred from demos.

**Generalization**: Any future task in the "small components → recolor" class will use the builtin regardless of which colors are involved. The mediator.md entry instructs MEDIATOR to infer `max_size` and `new_color` from demos, not use defaults blindly.

**Design insight**: Dynamically generated tools that pass demos by hardcoding demo-specific constants are a hidden reliability risk. The tool generator should be prompted to parameterize any value that appears in the demos rather than hardcoding it. **Future**: add a tool verification step that runs the generated tool against a color-shifted version of the demos — if it fails, the tool is hardcoding colors.

---

### 3.2 Revision loop reuses the same failing tool under a new name

**Exposed by**: `1190bc91` (multiple `diagonal_*` tool variants)

**Failure**: Across 5 revision attempts, MEDIATOR kept requesting variations of the same `diagonal_sequence_radiation` tool (`_v2`, `_fixed`, `_nearest`, `_v2`). Each new tool had different bugs. No progress toward the correct algorithm.

**Root cause**: The revision prompt told MEDIATOR to "fix the pseudocode" but didn't prevent it from requesting the same conceptually broken approach repeatedly. MEDIATOR had no memory of which approaches had already failed.

**Fix**: Added `failed_tools` list tracking across all revisions. Revision prompt now includes: *"Do NOT reuse any of these tools. Try a different decomposition."* Also added to MEDIATOR revision prompt: *"If the same tool has failed twice, try a fundamentally different decomposition."*

**Generalization**: Any future task where the first tool approach is wrong will now switch approaches after 2 failures rather than cycling through variations of the same broken idea. This reduces wasted API calls and prevents tool registry pollution.

**Design insight**: The revision loop without memory is a local search with no escape from local minima. The failed_tools list is a minimal memory mechanism. A stronger version would track not just tool names but the *approach* (e.g., "single-tool diagonal radiation") so that MEDIATOR avoids entire families of approaches, not just specific tool names.

---

## Failure type 4: Rule base quality issues

### 4.1 Wrong lesson extracted from correction event

**Exposed by**: `0a2355a6` (hole count recoloring), first correction attempt

**Failure**: After running with `--insight "topological hole count"`, MEDIATOR extracted a preference rule about tie-breaking by spatial position instead of the intended lesson (prefer topology over size as discriminating property).

**Root cause**: When `--insight` is provided before Round 1, the solver may already hypothesize correctly (because the tool name `recolor_by_hole_count` is visible in tool signatures). So the "wrong hypothesis" that the preference rule should correct was actually the correct hypothesis — and the extractor extracted a vacuous or wrong lesson.

**Fix**: Added `failed_hypotheses.json` sidecar to store the solver's hypotheses from prior *failed* runs. When `--insight` fires, the wrong hypotheses are loaded from the sidecar rather than from the current (potentially already-correct) Round 1 entries.

**Generalization**: Preference rule extraction is now correctly anchored to what was actually wrong in previous runs, not what the solver happens to say in the current run (which may already be correct due to the insight hint).

**Design insight**: The correction event has a temporal structure: the wrong hypothesis came from an earlier run; the insight comes now. Mixing them up produces a rule that tries to correct the right hypothesis instead of the wrong one. The sidecar pattern (persist wrong state from failed run, consume it in correction run) solves this cleanly.

---

### 4.2 Rule base accumulates conflicting rules from retry runs

**Exposed by**: `0a2355a6`, `1190bc91` (multiple retry runs each)

**Failure**: Each failed retry run created a new rule with a slightly different (wrong) `color_map` or tool spec. After 3–4 retries, the rule base had 3–4 conflicting rules all matching the same task pattern with contradicting actions.

**Root cause**: MEDIATOR creates a new rule after every run (including failed runs where it got a partial result). There was no deduplication or conflict detection between rules from the same task.

**Fix**:
1. `auto_deprecate()` now runs after every puzzle — rules fired ≥3 times with 0 successes are deprecated.
2. Rule conflict detection added: when two active rules share the same source_task and same action tool, flag them for review.
3. Operator manual review of rules.json after persistent failures.

**Generalization**: The rule base stays clean across all tasks. Retry-run artifacts are caught and deprecated before they accumulate to the point of paralysis.

**Design insight**: The rule base is only as useful as its signal-to-noise ratio. Every conflicting rule degrades the quality of Round 0 matching for all future tasks. Strict deprecation and conflict detection are essential maintenance operations, not optional cleanup.

---

## Design directions suggested by this log

### Direction 1: Contested-cell analysis as a first-class reasoning primitive
The ordering gap (1.2) reveals that static output description is insufficient for interaction-based transformations. A dedicated reasoning module that identifies contested cells, tests parallel vs sequential interpretations, and infers priority rules would address this systematically. This is likely one of the most broadly applicable missing capabilities.

### Direction 2: Tool generator quality gates
The hardcoded-parameter failure (3.1) suggests adding a post-generation test: run the generated tool on a color-shifted or size-scaled variant of the demos. Tools that fail this test are flagged as potentially brittle. This prevents silent reliability degradation in the tool registry.

### Direction 3: Approach-level memory in revision loop
The reuse-of-failing-approach failure (3.2) is only half-fixed by the `failed_tools` list. A stronger fix tracks the *approach category* (single-tool radiation, Chebyshev expansion, etc.) so MEDIATOR avoids entire families. This requires classifying failed approaches semantically, not just by tool name.

### Direction 4: Rule quality validation via held-out tasks
Preference rules and generalized rules are currently validated only on the task that triggered them. A background validation pass that runs each new rule against 2–3 randomly selected prior tasks would catch rules that are either too narrow (never fire) or too broad (fire and fail). This is the "held-out task" validation mentioned in Known Limitations.

### Direction 5: Structured condition predicates
Rule matching uses a full LLM call because conditions are free-text. Structured condition fields (object count range, color set size, grid aspect ratio, transformation category tag) would allow fast programmatic pre-filtering before the LLM pass. This reduces matching cost and false positives on large rule bases.

---

### 1.4 Solver uses grid-relative reference frame instead of shape-relative

**Exposed by**: `13f06aa5` (marker echo rays)

**Failure**: SOLVER said "the marker shoots toward the nearest grid edge" — measuring direction by which grid boundary is closest to the marker's absolute row/col. The correct rule is "the marker shoots toward the edge it faces within its own shape" — measuring direction by the marker's position relative to the shape's cells (top/bottom/left/right extremum).

**Root cause**: When reasoning about direction, the solver defaults to absolute (grid-relative) measurement because it's simpler to compute. The shape-relative interpretation requires first identifying which cells belong to the enclosing shape, then finding the marker's extremum position within that set.

**Fix**: Added Step 5 to solver.md analysis protocol:
> *"When an element seems to 'point toward' or 'shoot toward' something, ask: is this measured relative to the grid, or relative to a containing shape/object? Test both interpretations against all demo pairs."*

Also added the reference-frame check to the MEDIATOR revision prompt in agents.py.

**Outcome**: Task solved via tool generator (MEDIATOR revised once when the initial tool failed, then produced `shoot_markers` which passed all demos). The solver's wrong hypothesis was corrected by demo-verified tool generation, not by the prompt fix directly. The prompt fix will prevent this class of error in future similar tasks.

**Generalization**: Any puzzle where direction, alignment, or pointing is determined by a local/containment relationship rather than a global grid property requires this check. Relevant for: marker-relative direction puzzles, inside/outside determination, orientation of embedded shapes.

**Design insight**: Two reference frames — grid-global and shape-local — can produce identical predictions on well-aligned demos and diverge on others. The solver needs to make both framings explicit and test both against all demos, rather than defaulting to whichever is simpler to compute.

---

## Quick reference: failure types and their signals

| Signal observed | Likely failure type | First thing to check |
|---|---|---|
| MEDIATOR produces 0 steps | Rule conflict or no matching tool | Check rules.json for conflicting active rules on same task |
| Solver hypothesis is partially right but misses a key property | Solver reasoning gap | Which reasoning step is missing from solver.md? |
| Tool generator fails all fix attempts | Algorithm too complex for one tool | Does the algorithm decompose into 2+ sub-algorithms? If so, build a builtin |
| Correct in demos, wrong on test | Overfitting to demo-specific constants | Check if tool/rule hardcodes colors, sizes, or coordinates from demos |
| Preference rule extracts wrong lesson | Correction event timing issue | Were `failed_hypotheses.json` entries available? Did solver already have the right hypothesis in Round 1? |
| Retry runs accumulate conflicting rules | Rule base maintenance gap | Run `auto_deprecate()`, review rules for same-task conflicts |
| Rule fires but never succeeds | Condition too broad or action wrong | Review condition vs actual task pattern; check if action tool is correct |

---

## Failure type 1 (continued): Solver + MEDIATOR compounding gap — cross/plus shape completion

### 1.5 Solver misidentifies changed cell type; MEDIATOR oversimplifies cross-center algorithm

**Exposed by**: `14754a24` (cross/plus completion)
**Timestamp**: 2026-03-28 ~01:00 EDT
**Retry count**: Retry 9 (8 prior failed runs in this session)

**Failure**: System failed across 8 attempts, each with a different error:
- Early runs: SOLVER said "replace 0-valued cells with 2" — wrong cell type entirely (only 5→2 transitions occur)
- Later runs (post transition-census fix): SOLVER identified "5-cells between two 4s in same row/col" — right cell type, wrong spatial criterion
- Later runs (post shape-completion hint): SOLVER found "hub 4-cell with most 4-neighbors, change orth-5-neighbors" — correct for clusters where center is a 4-cell, wrong for clusters where center is a 5-cell
- Tool generator: even with a correct solver hypothesis, generated wrong tools because MEDIATOR simplified the algorithm to "between 4s" or "adj to 4s"

**Root cause (compound)**:
1. **Solver**: No instruction to do a transition census before forming a hypothesis. Solver defaulted to describing 0-cells without checking what actually changes.
2. **Solver**: No vocabulary for "cross/plus shape completion" pattern. Solver used "between pairs of 4s in same row/col" as a simpler approximation, missing cells that require the cross-center algorithm.
3. **Solver**: Cross-center algorithm has TWO cases (center inside cluster vs. outside cluster). The hint described both cases but solver kept collapsing to a single "hub = 4-cell with most 4-orth-neighbors" rule which fails for Case A (diagonal clusters where center is outside the cluster).
4. **MEDIATOR**: "Prefer simpler pseudo-code" principle caused MEDIATOR to discard the solver's correct (complex) hypothesis in favor of simpler (wrong) algorithms that covered 97% of cells but not 100%.
5. **MEDIATOR**: No template for "cross/plus completion" tool — the behavior description it wrote was always a variation of "between 4s" or "adj to 4s."

**Fix (multi-part)**:
1. Added **transition census** instruction to solver.md Step 1: solver must list all (input_value → output_value) pairs observed before forming any hypothesis.
2. Added **shape-completion check** to solver.md Step 2: ask whether changed cells + marker cells together form a complete geometric shape.
3. Added **cross/plus detection** with TWO explicit cases to solver.md Step 2: Case A (center is a non-cluster cell orth-adj to all cluster cells), Case B (a cluster cell is orth-adj to all other cluster cells). With concrete examples.
4. Added **"Shape completion"** category to solver.md's analogy lens.
5. Added **transition-census enforcement** to agents.py MEDIATOR revision prompt: "only cells with a specific input value change; use the correct target value in tool parameters."
6. Changed mediator.md decision principle from "prefer simpler" to **"prefer simpler CORRECT"**: a 97%-right algorithm is wrong; verify every cell.
7. Added **cross/plus completion tool behavior template** to mediator.md: exact two-case algorithm description that the MEDIATOR should copy into its `new_tools` behavior field when it encounters this pattern.

**Outcome**: Task solved on run 9. MEDIATOR generated `fill_cross_center_arms` → failed; then `fill_cross_arms_v2` → failed; then `fill_cross_center` → failed; then `fill_cross_from_markers` → CORRECT 100% | 744.5s | 6 rounds.

**Generalization**: All cross/plus completion puzzles benefit from the cross-center detection algorithm. The two-case rule (center inside vs. outside cluster) is now in solver.md and mediator.md. Transition census prevents wrong-cell-type hypotheses across all task types.

**Design insight**: The MEDIATOR's "prefer simpler" heuristic can be harmful when the task requires a complex but precise algorithm. A 97%-right approximation is not "simpler" — it's wrong. The mediator must verify 100% of demo cells before accepting any algorithm. Also: compound failures (solver wrong → MEDIATOR simplifies further) are harder to fix than single gaps; address each gap layer separately.

---

### 1.6 Tool creator uses wrong proxy check; template instruction wrong about interior content
**Exposed by**: `14b8e18c` (closed square ring outside-corner marking)
**Timestamp**: 2026-03-28 07:25 EDT
**Retry count**: 4th run (3 prior failed in this session)

**Failure**: The tool creator implemented "must have interior (dimensions ≥ 3)" as a proxy for "is a closed ring." This correctly excludes 2×4 non-square blocks (since they have h=2, excluded by the check) but ALSO incorrectly excludes valid 2×2 closed squares (also h=2). The correct distinguishing criterion is "bounding box must be square (h==w)", not "must have interior cells."

**Root cause** (3 layers):
1. **Solver didn't identify "square" as the constraint**: Said "closed rectangles" — correct pattern but too broad. Non-square rectangles (2×4, 1×3 lines) are excluded; only squares qualify.
2. **Template instruction was wrong**: Added "require no interior cell is object_color" — but a large ring (e.g., 8×8) may contain other same-colored components in its interior region (they're SEPARATE components). The interior check must be component-based: "component cells == perimeter cells exactly," not "grid interior values == background."
3. **executor.py lacked None-return handling**: A generated tool returning None instead of a grid caused an unhandled TypeError in diff_cells → crash.

**Fix**:
- `solver.md` Step 2: Added "Square vs. rectangle constraint" hint — explicitly check h==w; 2×2 is the minimum closed square.
- `solver.md` Step 2: Added "Outside-corner marking" hint — 8 orthogonal marks per shape (2 per corner), NOT 4 diagonal marks.
- `mediator.md`: Added closed-square-ring tool template with correct algorithm: (1) square check h==w, h≥2; (2) component cells == perimeter cells of bbox exactly; (3) do NOT check grid interior values; (4) 8 outside-corner marks.
- `executor.py` execute_steps: If tool returns None, treat as tool error with `error="Tool returned None"` rather than crashing.

**Generalization**: The "component membership check" vs. "grid value check" distinction is general: when verifying that a shape is a hollow ring, always check whether the component's own cells form the ring, not whether the interior region of the grid is background-colored. Nested objects are common in ARC.

**Design insight**: Proxy checks (e.g., "has interior" as proxy for "is a valid ring") work for the majority case but fail at boundary sizes. Always derive the check from the first principles of what actually differs between qualifying and non-qualifying shapes.

---

## 1.7 — 15113be4 (grid key-mask block coloring)
**Date:** 2026-03-28T18:XX EST: 2026-03-28 14:XX
**Retry count:** 5th run (4 prior failed)

### Root cause
Tool_creator kept generating code with wrong block-boundary formula (`br*3` instead of `br*4`). Without explicit formula in the behavior description, the tool_creator assumed uniform block-size spacing (3 cells per block) rather than band-size spacing (3 cells + 1 separator = 4 per band). This caused separator columns to be treated as data, producing 4s inside data positions. Secondary issue: tool hardcoded color values observed in demo 0 (e.g., `key_color=6`) instead of computing dynamically.

### Fixes applied

**`prompts/solver.md` — added hint for grid-with-key-mask pattern:**
- Described the 6×6 corner key area with 2×2 sub-blocks forming a 3×3 binary mask
- Emphasized all-or-nothing block matching (if any mask-1 position is 0, skip block entirely)
- Added explicit formula: "block (br,bc) starts at row=br*4, col=bc*4; band index br=r//4"

**`prompts/mediator.md` — added tool template with step-by-step pseudocode:**
- 5 explicit steps: find key_color, find key corner via `r//4` band indices, compute inner 6×6 area origin, extract 3×3 mask, apply all-or-nothing to non-key blocks
- Key formula: `kr = 1 if min_br==0 else min_br*4` (top corners start at offset 1 due to outer border row; bottom corners start at `min_br*4` directly)
- Explicit block formula: `rs=br*4, cs=bc*4`

**`python/agents.py` — added two CRITICAL warnings to tool_generator system prompt:**
- "NEVER hardcode color values — always compute dynamically from grid"
- "When behavior mentions blocks separated by separator value: find separator positions by scanning; don't assume `br*block_size`. For 1-separator-wide rows: each band is block_size+1 wide, so `br*4` for 3-wide blocks with 1-col separators"

### Key lesson
When a behavior description uses natural language like "blocks separated by 4-lines," the tool_creator defaults to uniform spacing (`br*block_size`) rather than correctly accounting for separator widths. **Always include the explicit formula** (`rs=br*4`) in the behavior description. The tool_generator cannot infer separator-adjusted offsets from the grid examples alone without guidance.

### Secondary lesson
Tools from failed runs are retained in the registry with their last-attempt code. A later run may find that one of those previously-failed tools is correct and reuses it. Registry reuse can unexpectedly solve tasks: `stamp_key_pattern_v3` and `stamp_key_pattern_v4` were generated in failed earlier runs but happened to have the correct logic, and were loaded from cache on the successful 5th run.

---

## 1.8 — 15663ba9 (closed-loop corner convexity marking)
**Date:** 2026-03-28
**Retry count:** 1st run (first attempt after prompt fix)

### Root cause
No explicit hint for the closed-loop corner convexity pattern. The pattern requires: (1) detecting 90° turn cells in 1-pixel-wide closed loops, (2) classifying each turn as convex or concave by whether its inside diagonal is enclosed within the loop (interior) or not (exterior), and (3) marking convex→4, concave→2. Without a hint, the solver might only notice "corners become 4" without recognizing the concave/convex distinction.

### Fix
Added to `solver.md`: "Closed-loop corner convexity marking" hint describing the flood-fill exterior detection and inside-diagonal classification rule.
Added to `mediator.md`: Exact 2-step behavior template — flood-fill exterior from boundary, then for each turn cell compute inside_diag and check exterior status.

### Key lesson
The inside diagonal of a 90° turn cell is always background (value 0) in both convex and concave cases — it cannot be distinguished by cell value alone. The distinguishing criterion is topological: is the inside diagonal reachable from the grid boundary through background cells? Flood-fill from boundary is the correct approach.

### Design insight
This is a 1st-run solve because the hint gave the exact algorithm. The solver reproduced it verbatim (high confidence). The mediator reused `mark_loop_corners` from the registry — a tool that had been generated in a prior failed run of a related task. This confirms that building up a registry of reusable tools accelerates solving.
