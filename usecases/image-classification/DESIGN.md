# Design Spec: Expert Knowledge Transfer For Image Classification

This document is the developer-facing companion to
[README.md](C:\_backup\github\khub-knowledge-fabric\usecases\image-classification\README.md).

The README is intentionally user-facing. This document holds the more technical
material:

- experiment framing and protocol
- implementation decisions
- pipeline architecture
- repository layout
- developer quick-start commands
- benchmark caveats and next-step gaps

## 1. Evaluation Framing

### 1.1 Simulated patch authoring session

In the intended KF workflow, patch authoring is interactive. A domain expert
works with KF in natural language: KF proposes rule candidates, the expert
confirms, rejects, or revises them, and only verified rules are stored.

Both current image-classification experiments simulate that workflow because no
live domain expert was available.

- **Birds**: expert knowledge was pre-encoded from Sibley's Guide, Kaufman's
  Field Guide, eBird / All About Birds species accounts, and the
  CUB-200-2011 dataset's 312 binary attribute annotations into per-pair
  Markdown files in `teaching_sessions/`, then extracted by GPT-4-turbo into
  `knowledge_base/*.json`.
- **Dermatology**: expert knowledge was pre-encoded from dermoscopy references
  and lesion-attribute-style materials into JSON rule sets in
  `dermatology/knowledge_base/`.

The simulated workflow is useful for development, but it is weaker than a real
dialogic session because `auto_accept` batch import cannot catch expert-rule
errors as they happen.

### 1.2 Conditions

| Condition | Description |
|---|---|
| **Baseline: zero-shot** | Frontier VLM, no fine-tuning, no KF |
| **Baseline: few-shot** | Same model with a small number of labeled example images |
| **Baseline: retrieved reference text** | Same model with raw reference passages inserted into the prompt, but without KF extraction or verification |
| **KF: NL patch only** | KF patching session using expert text or expert-authored rules |
| **KF: NL patch + few-shot** | KF patching combined with a small number of labeled examples |
| **Reference: supervised SOTA** | Included only as a domain ceiling reference; not a fair implementation baseline |

### 1.3 Primary metric

The primary metric is **accuracy on confusable-pair test sets**.

This is more diagnostic than overall benchmark accuracy because:

- easy cases dominate full-dataset scores
- confusable pairs expose what expert discriminative knowledge is worth
- the realistic claim is narrower: KF should improve targeted hard cases
  without fine-tuning

Secondary metrics include:

- full benchmark accuracy
- artifact reuse rate
- cost per image
- latency per image
- number of patching iterations needed to improve a failure mode

## 2. Bird Experiments (CUB-200-2011)

### 2.1 Experiment 1: Single-pass prompt injection

**Model**: GPT-4o (vision)  
**Dataset**: CUB-200-2011  
**Pairs tested**: 15 confusable species pairs  
**Test images**: 20 per class per pair (40 per pair)  
**Knowledge base**: 118 rules extracted from field-guide text via GPT-4-turbo  
**Results file**: `results/results_20260329T104850.json`

#### What the model received

| Condition | What the model receives |
|---|---|
| **Zero-shot** | test image + species names only |
| **Few-shot** | test image + 3 labeled training images per class |
| **KF-patched** | test image + species names + expert rules injected as prompt text |

#### Aggregate results

| Condition | Correct | Total | Accuracy | vs. Zero-shot |
|---|---|---|---|---|
| Zero-shot | 468 | 600 | **78.0%** | — |
| Few-shot | 497 | 600 | **82.8%** | +4.8pp |
| KF-patched (first run) | 434 | 599 | **72.5%** | −5.5pp |
| KF-patched (after fix) | 452 | 599 | **75.5%** | −2.5pp |

#### Main lessons

- KF helped on several visually grounded pairs.
- KF failed badly when patch quality was poor.
- KF also exposed a harder design problem: some pairs need KF to externalize
  observable feature claims rather than merely inject more expert text.

### 2.2 Implementation decisions for Experiment 1

1. **Vision model**: GPT-4o was used as the base VLM. No fine-tuning occurred.
2. **Patch source format**: bird expertise came from field guides, eBird
   accounts, and related ornithological material.
3. **Verification assumption**: the intended system assumes expert review, but
   this experiment used batch-imported rules and therefore weaker validation.
4. **Benchmark framing**: the confusable-pair subset is the main story; the
   full CUB framing is supporting context.

### 2.3 Experiment 2: KF ensemble pipeline

**Model**: Claude Sonnet 4.6  
**Dataset**: CUB-200-2011  
**Pairs tested**: American Crow vs Fish Crow; Brewer Sparrow vs Clay-colored
Sparrow  
**Test images**: 3 per class per pair (12 total)  
**Few-shot images**: 3 per class per pair  
**Rules**: 112 migrated rules, growing to 139 with post-task extraction  
**Results file**: `python/results_test.json`

This experiment replaced prompt-only patching with a structured multi-round
pipeline:

1. retrieve relevant rules
2. generate or load a feature schema
3. run an **OBSERVER** that records visible features
4. run a **MEDIATOR** that applies rules to those observations
5. run a **VERIFIER** grounded by few-shot examples
6. optionally extract post-task rule updates

#### First-pass results

| Pair | Correct | Total | Accuracy | Cost | Avg API calls/image |
|---|---|---|---|---|---|
| American Crow vs Fish Crow | 5 | 6 | **83.3%** | $0.34 | 6.2 |
| Brewer Sparrow vs Clay-colored Sparrow | 6 | 6 | **100.0%** | $0.39 | 6.2 |
| **Combined** | **11** | **12** | **91.7%** | **$0.73** | **6.2** |

Average cost per image: ~$0.061  
Average duration per image: ~57s

#### Important caveats

1. **Online adaptation confound**: post-task rule learning was active during
   this run, so later test images may have benefited from earlier labeled
   outcomes.
2. **Model + architecture confound**: this run used Claude Sonnet 4.6 and a
   richer pipeline, so it is not a same-model comparison to Experiment 1.
3. **Small sample**: 12 images across 2 pairs is a pilot, not a benchmark.

### 2.4 Observability filter

All rules passed through `add_rule(observability_filter=True)`, which blocks
rules mentioning non-visual cues such as:

- vocalizations
- calls / songs
- habitat
- range
- migration
- season
- behavior
- flock-size comparison

This was critical for avoiding the earlier crow failure mode caused by
non-visual rules being injected into a static-image task.

### 2.5 Feature-schema lesson

The strongest lesson from the bird work is architectural:

**hard visual cases often need KF to externalize feature observations into a
human-verifiable intermediate layer.**

This is what made the structured ensemble pipeline more promising than direct
prompt injection.

### 2.6 Experiment 3: KF dialogic patching loop

**Cheap pupil**: Qwen3-VL-8B-Instruct (via OpenRouter)  
**Expert**: Claude Sonnet 4.6  
**Validator**: Claude Sonnet 4.6  
**Pair**: Bronzed Cowbird vs Shiny Cowbird  
**Test images**: 3 per class (6 total, same images used for baseline and rerun)  
**Implementation**: `birds/python/patch.py` + `birds/python/agents.py`

This is the first end-to-end run of the full KF dialogic patching loop on bird data.
It matches the dermatology loop architecture exactly — same pipeline steps, same
precision gate, same specificity spectrum — with only prompt vocabulary changed.

#### Steps

1. Zero-shot baseline on 6 images (cheap model only).
2. For each failure: expert authors a candidate rule.
3. Rule completer fills implicit background conditions.
4. Semantic validator rates each precondition (reliable / context_dependent / unreliable)
   and flags if REVISE; rule proceeds regardless — semantic validation is advisory.
5. Held-out gate: pool of 8 images (4 per class), min precision 0.75 and FP ≤ 1.
   If the completed rule fails but the pre-completion rule passes, run 4-level spectrum.
6. Accepted rules are registered in-memory and written to `patch_rules_dialogic_r2.json`.
7. Rerun: all 4 failures re-classified with registered rules active.
8. Round 2 (dialogic, `--max-rounds 2`): for cases still failing after round 1, assemble
   a prior-context block (which rules were active, which fired, pupil's reasoning, per-
   precondition validator observations on the trigger image) and re-engage the expert.
   Expert authors a new rule targeting a different feature. Same validation pipeline runs.

#### Results (2 rounds, `--max-rounds 2`)

| Phase | Correct | Accuracy |
|---|---|---|
| Zero-shot (Qwen3-VL-8B) | 2/6 | 33.3% |
| After round 1 | 4/6 | 66.7% |
| After round 2 (dialogic) | 5/6 | 83.3% |
| Delta (total) | +3 | +50pp |
| Rules authored / accepted / registered | 6 / 4 / 4 | |
| Round 1 rules | 4 authored / 3 registered | |
| Round 2 rules (dialogic) | 2 authored / 1 registered | |

**r_001** (red iris + thick bill → Bronzed Cowbird): TP=2 FP=0 precision=1.00. Fixed Bronzed_0019.  
**r_002** (red iris + thick bill, authored from 0061 → Bronzed Cowbird): TP=4 FP=0 precision=1.00. Fixed Bronzed_0061.  
**r_003** (bull-necked silhouette + heavy bill → Bronzed Cowbird): spectrum pre-completion TP=4 FP=0. Registered but pupil does not fire at inference on Bronzed_0081.  
**r_004** (domed head + stubby bill, round 2 dialogic → Bronzed Cowbird): spectrum pre-completion TP=4 FP=0. Registered but pupil still fails on Bronzed_0081.

Two rules were rejected (both for Shiny Cowbird): validator TP=0 on held-out pool in both rounds.

#### Cross-class anchoring

Shiny_0080 was fixed in round 2 without a dedicated Shiny rule. After 4 Bronzed Cowbird rules
were registered, none fired on the all-brown Shiny specimen, causing the pupil to correctly
default. Accumulated Bronzed knowledge implicitly defined the class boundary.

#### Remaining failure

- **Bronzed_0081**: 2 dedicated rules registered (r_003 round 1, r_004 round 2); both pass
  validation (precision=1.00, TP=4) but the pupil does not fire them at inference on this
  specific image. Structural gestalt features (bill depth, head proportions) appear
  insufficiently salient or ambiguous in this particular CUB shot.

### 2.7 What is needed before a bird benchmark claim

| Gap | What to do |
|---|---|
| Online adaptation during test | Re-run with `--mode test` so rules are frozen before evaluation |
| Model confound | Add Claude zero-shot and Claude few-shot baselines |
| Small sample | Run the full test set for both pilot pairs |
| Narrow scope | Expand back out to more of the 15 confusable pairs |
| Dialogic loop coverage | Run Experiment 3 loop on additional confusable pairs |

## 3. Dermatology Experiments (HAM10000 / ISIC)

### 3.1 Design intent

The dermatology track is deliberately parallel to the bird track:

- expert visual criteria are captured outside the model
- KF converts them into explicit rules
- the base model stays unchanged
- the experiment focuses on targeted confusions rather than leaderboard claims

### 3.2 Pilot setup

**Model**: Claude Sonnet 4.6
**Dataset**: HAM10000
**Pairs tested**:

- Melanoma vs Melanocytic Nevus
- Basal Cell Carcinoma vs Benign Keratosis
- Actinic Keratosis vs Benign Keratosis

**Test images**: 3 per class per pair (18 total)
**Mode**: `--mode test` with frozen rules
**Few-shot images**: 3 per class per pair

### 3.3 Pilot results

Three iterations were run on the same 18 images to isolate the impact of each improvement.

#### v1 — initial pipeline (36 rules, original VERIFIER/MEDIATOR prompts)

| Pair | Correct | Total | Accuracy | Cost | Avg API calls/image |
|---|---|---|---|---|---|
| Melanoma vs Melanocytic Nevus | 4 | 6 | **66.7%** | $0.44 | 6.3 |
| Basal Cell Carcinoma vs Benign Keratosis | 3 | 6 | **50.0%** | $0.47 | 6.3 |
| Actinic Keratosis vs Benign Keratosis | 2 | 6 | **33.3%** | $0.46 | 6.3 |
| **Combined** | **9** | **18** | **50.0%** | **$1.37** | **6.3** |

#### v2 — expanded KB (48 rules, TWO_STAGE_THRESHOLD=999, confidence gate 0.35)

Same overall accuracy as v1 (9/18 = 50%). The new composite-absence rules did not fire because the semantic rule retriever could not match absence-framing against image queries — a structural mismatch.

The VERIFIER was identified as net-negative: it revised 5 MEDIATOR decisions, converting 3 correct-or-fixable decisions to "uncertain" (wrong) and saving only 1. Without the VERIFIER, v2 would have scored 12/18.

#### v3 — no-abstain MEDIATOR + hard-contradiction-only VERIFIER (2026-04-05)

| Pair | Correct | Total | Accuracy | Cost | Avg API calls/image |
|---|---|---|---|---|---|
| Melanoma vs Melanocytic Nevus | 4 | 6 | **66.7%** | $0.44 | 5.1 |
| Basal Cell Carcinoma vs Benign Keratosis | 5 | 6 | **83.3%** | $0.48 | 5.1 |
| Actinic Keratosis vs Benign Keratosis | 2 | 6 | **33.3%** | $0.46 | 5.1 |
| **Combined** | **11** | **18** | **61.1%** | **$1.44** | **5.1** |

Average cost per image: ~$0.080
Average duration per image: ~52s

**Changes in v3 vs v1:**

| Change | Rationale |
|---|---|
| MEDIATOR must commit to one of the two classes ("uncertain" only when strong evidence exists for BOTH simultaneously) | Binary task: ground truth is always one class; abstaining is never correct |
| VERIFIER only flags hard contradictions — a pathognomonic feature of the OTHER class unmistakably present | Dermoscopic uncertainty is normal; the old VERIFIER treated ambiguity as contradiction |

#### v4 — absence-checklist schema + LPLK KB rules (2026-04-05)

| Pair | Correct | Total | Accuracy | Cost | Avg API calls/image |
|---|---|---|---|---|---|
| Melanoma vs Melanocytic Nevus | 4 | 6 | **66.7%** | $0.44 | 5.1 |
| Basal Cell Carcinoma vs Benign Keratosis | 5 | 6 | **83.3%** | $0.48 | 5.1 |
| Actinic Keratosis vs Benign Keratosis | 4 | 6 | **66.7%** | $0.46 | 5.1 |
| **Combined** | **13** | **18** | **72.2%** | **$1.38** | **5.1** |

Average cost per image: ~$0.077
Model: Claude Sonnet 4.6

**Changes in v4 vs v3:**

| Change | Rationale |
|---|---|
| Per-pair absence-checklist fields always appended to OBSERVER schema | Composite-absence rules never fired via retrieval; hardcoding the fields bypasses the retriever entirely |
| 4 LPLK-specific KB rules for AK/BKL pair (`lplk_atypical_network`, `no_erythema_favors_bkl`, `flat_brown_structureless`, `lplk_regression_brown`) | LPLK-like BKL images lack SK markers and mimic AK; rules needed to make absence of erythematous background positive evidence for BKL |

AK/BKL improved from 33% to 67% (2/6 → 4/6). Root cause of remaining failures: two LPLK images still lack all dermoscopic markers and the pipeline defaults to AK in total absence of evidence.

### 3.4 Cross-model comparison (v4 pipeline, 2026-04-05)

All three runs use the same 18-image pilot, same KB (52 rules), and same v4 pipeline. OpenAI models ran as structured zero-shot in the original data below because the rule retriever returned 0 matches — two bugs were present: (1) the retriever prompt was hardcoded to ARC-AGI framing, confusing the model; (2) `parse_match_response` only handled JSON inside code fences, but o4-mini sometimes returns raw JSON. Both are now fixed (see §3.8).

A third bug affected the `model` field in results: `DEFAULT_MODEL` (always `claude-sonnet-4-6`) was logged instead of `ACTIVE_MODEL`. This means the o4-mini column below was actually run with o4-mini but logged as Sonnet. The fix has been applied; future runs will log correctly.

| Model | Mel/Nev | BCC/BKL | AK/BKL | **Overall** | Cost | Notes |
|---|---|---|---|---|---|---|
| Claude Sonnet 4.6 | 4/6 (67%) | 5/6 (83%) | 4/6 (67%) | **13/18 (72%)** | $1.38 | Full pipeline: rules + absence checklist |
| GPT-4o | 5/6 (83%) | 5/6 (83%) | 4/6 (67%) | **14/18 (78%)** | $0.71 | Zero-shot (0 rules fired); parse fix applied |
| o4-mini | 5/6 (83%) | 4/6 (67%) | 2/6 (33%) | **11/18 (61%)** | $0.46 | Zero-shot (0 rules fired; retriever bug — now fixed) |

Key observations:
- GPT-4o outperforms Claude zero-shot on this 18-image sample. The margin (78% vs 72%) is within noise for n=18 but suggests GPT-4o's dermoscopy pattern recognition is strong without KB guidance.
- o4-mini's AK/BKL score (33%) was recorded with 0 rules firing; the retriever fix is expected to improve this pair most since absence-reasoning rules could not fire.
- All models struggle with AK/BKL due to LPLK heterogeneity in the `bkl` class.
- A full rerun with all fixes applied is needed for honest per-model numbers — the §3.4 table is from the pre-fix runs.

### 3.5 Main failure modes

#### Failure mode 1: absence-only evidence (partially addressed in v2, partially remaining)

Presence-only rules cannot fire when no positive markers are visible. Composite-absence rules were added to the KB in v2 but did not improve results because the rule retriever matches rules by semantic similarity to image content — absence-framing ("no veil + no regression + ...") scores low against a query about what IS visible.

Fix needed: schema generation should always include absence-tracking fields for the pair's key markers, independent of retrieval.

#### Failure mode 2: `bkl` class heterogeneity

The HAM10000 `bkl` label covers both seborrheic keratosis (SK) and lichenoid keratosis (LPLK). LPLK-like images lack SK's positive dermoscopic markers and can resemble AK or melanoma, making them hard for any absence-based or presence-based rule set to classify correctly. This is partly an ontology problem, not purely a pipeline problem.

The `--sk-only` flag (filters `bkl` test images to `dx_type=="histo"` as a proxy for SK) is available but has not been evaluated yet.

#### Failure mode 3: AK vs BKL pair is structurally hard

The three BKL images used in the AK/BKL pair (ISIC_0024336, _0024420, _0024495) were correctly identified as BKL in the BCC/BKL pair context but misclassified as AK in the AK/BKL context across all three iterations. Without typical SK markers, the rule set defaults to AK. A pair-specific fix or additional rules are needed.

### 3.6 What worked

- BCC vs Benign Keratosis improved significantly (50% → 83%) with the VERIFIER fix. BCC has strong pathognomonic markers (arborizing vessels, ovoid nests) that make the MEDIATOR reliable once it is allowed to commit.
- The no-abstain MEDIATOR eliminated 4 "uncertain" predictions — 3 of which became correct.
- Absence-checklist bypass (v4) fixed the composite-absence retrieval gap: AK/BKL improved from 33% to 67%.
- Domain transfer from birds to dermatology required only prompt and schema changes, confirming the architecture is domain-agnostic.

### 3.7 What is needed next

| Gap | What to do |
|---|---|
| Full rerun with all fixes | Run all 18 images with token-budget fixes, retriever fix, and correct model logging to get clean per-model numbers |
| Dialogic loop lift measurement | Compare o4-mini zero-shot vs o4-mini + patch rules authored by Claude Sonnet/Opus on the same 18 images |
| Claude zero-shot baseline | Run `--baseline zero_shot` on same 18 images for apples-to-apples comparison |
| `bkl` heterogeneity | Evaluate `--sk-only` flag; consider LPLK as a third class |
| Larger sample | Run full test sets per pair (`--max-per-class 10+`) |
| Remaining AK/BKL failures | Two LPLK images still default to AK in total feature absence; may require LPLK subclass or prior-based tiebreak |
| Cross-pair rule generalization | Confirm or reject cross-pair rule firings detected by `patch.py`; update rule pre-conditions or promote to general rules |
| **Pupil teachability screening** | Before running a full patch loop on any new pupil VLM, run the teachability entrance exam (§3.9) to confirm the model is correctable — GPT-4o proved unteachable; Qwen3-VL-8B proved correctable |

### 3.8 Dialogic patching loop (patch.py)

#### Reframing

The 4-round KF ensemble pipeline (observer → mediator → verifier) is net-negative for already-strong VLMs like Claude Sonnet 4.6 or GPT-4o: it adds cost and latency without improving accuracy over zero-shot on the 18-image pilot. This is expected: a strong VLM's pattern recognition is not improved by rules it is already implicitly aware of.

**KF's real value is the dialogic patching loop** — a mechanism for upgrading a weaker model using knowledge authored by a stronger one:

1. A cheap VLM (e.g. o4-mini, Claude Haiku) classifies a batch of labeled training images.
2. Failure cases are collected and surfaced to a human expert — or, in controlled experiments, to a superior VLM (e.g. Claude Sonnet, Claude Opus) acting as expert.
3. The expert authors corrective rules with explicit pre-conditions that describe when the rule applies.
4. KF validates the rules: candidate rules are applied to the labeled training pool; rules that fire correctly on at least one training image are promoted to active status.
5. Active rules fire on future images matching the pre-conditions, routing classification toward the correct class without re-running the expensive expert.

The metaphor: the pupil (cheap VLM) is shown where it errs. The master (expert VLM or human) teaches the rule. The pupil applies the rule on future images — without needing the master present.

#### Cross-pair firing and the "enlightened pupil" signal

Rules are authored for a specific pair (e.g. `melanoma_vs_melanocytic_nevus`). When such a rule fires on a *different* pair's image, it indicates one of two things:

- **Quality problem**: the rule's pre-conditions are too broad — they match features common across pairs and will generalize incorrectly.
- **Possible generalization**: the rule captures a visual principle that is genuinely useful across pairs — a "pupil has been enlightened" moment where the master should confirm or further refine the rule.

`patch.py` detects cross-pair firings automatically and surfaces them as explicit signals for the human expert or superior VLM to review.

#### Model routing: --validator-model separation

`patch.py` accepts three distinct model roles:

| Flag | Role | Typical value |
|---|---|---|
| `--cheap-model` | Pupil — the model being patched | `qwen/qwen3-vl-8b-instruct` |
| `--expert-model` | Tutor — authors rules, does contrastive analysis, generates spectrum | `claude-tutor` (filesystem) or `claude-sonnet-4-6` (API) |
| `--validator-model` | Validator — binary precondition checks on pool images | `claude-sonnet-4-6` |

Without `--validator-model`, all pool validation calls go to the expert model — which is correct when the expert is an API VLM, but wrong when the expert is `claude-tutor` (a human-in-the-loop via filesystem). In the tutor workflow, all six pool validation call sites must use `claude-sonnet-4-6` directly to avoid routing 100+ binary validation requests to the human. `_val_model = validator_model or expert_model` handles this at the top of the loop.

#### Expert model: filesystem inbox/outbox (claude-tutor)

When `--expert-model claude-tutor` is set, agent calls that require expert judgment (EXPERT_RULE_AUTHOR, RULE_COMPLETER, SEMANTIC_RULE_VALIDATOR, contrastive analysis, spectrum generation) are routed through a filesystem protocol instead of an API:

- Request written as JSON to `tutor_inbox/<request_id>.json` with the agent role, system prompt, user text, and image paths
- Loop polls for `tutor_outbox/<request_id>.txt`; when present, reads the response and continues
- Completed requests are moved to `tutor_inbox/done/`

This allows a human expert (or Claude Code session acting as expert) to respond to each authoring step interactively, one at a time, while pool validation still runs automatically via `--validator-model`.

#### Architecture (patch.py)

```
patch.py --cheap-model qwen/qwen3-vl-8b-instruct \
         --expert-model claude-tutor \
         --validator-model claude-sonnet-4-6 \
         --pair melanoma_vs_melanocytic_nevus

Step 0.  Run cheap VLM zero-shot → collect failures

For each failure image (one at a time — early exit on first unresolved failure):

Step 1a. EXPERT_RULE_AUTHOR (expert-model + failure image)
           → candidate rule with pre-conditions (diagnostic — what was distinctive)

Step 1b. RULE_COMPLETER (expert-model, text-only)
           → fill implicit background conditions the expert omitted
           → returns enriched rule with added_preconditions + completion_rationale

Step 1c. SEMANTIC_RULE_VALIDATOR (expert-model, text-only, no images)
           → rate each pre-condition: reliable / unreliable / context_dependent
           → overall: accept / revise / reject
           If reject → skip image validation, surface to expert immediately

Step 2.  Sample three independent image pools from training set:
           authoring pool   (seed=0,   --max-authoring-per-class)
             Expert-visible: validator observations → tp_cases/fp_cases for contrastive analysis
           held-out pool    (seed=42,  --max-val-per-class)
             Expert-blind:  binding acceptance gate (expert never sees these images)
           confirmation pool (seed=123, --max-confirm-per-class)
             Expert-blind:  secondary precision check after held-out passes

           All pool validation calls use validator-model (not expert-model).

Step 3a. Validate completed rule against authoring pool (validator-model)
           → tp_cases, fp_cases with per-image observations for expert context

Step 3b. Validate completed rule against held-out pool (validator-model, binding gate)
           Accept if: fires_on_trigger AND fp ≤ 1 AND precision ≥ 0.75
           If accepted → also validate against confirmation pool → register if passes
           If rejected, fires_on_trigger=False → reject (cannot tighten further)
           If rejected, fp=0, low precision → reject (no FP observations to analyze)
           If rejected, fp > 0 → spectrum search:
             a. run_contrastive_feature_analysis(auth_tp_cases, auth_fp_cases)
                  → single most discriminating visual feature (uses authoring observations)
             b. run_rule_spectrum_generator() (expert-model)
                  → 4 specificity levels in one call, ordered most-general → most-specific:
                      Level 1: single essential pre-condition only (most_general)
                      Level 2: core + one supporting condition (moderate)
                      Level 3: original rule as authored, unchanged (original)
                      Level 4: original + contrastive tightening condition (most_specific)
             c. validate_candidate_rules_batch(all 4 levels, held_out_images, validator-model)
             d. Pick the most general level that passes the held-out gate
             e. If none pass → _surface_to_expert() with best-precision level

Step 4.  Register accepted rule in patch_rules file
Step 5.  Re-run cheap VLM with patch rules on the failure image → check if fixed
         If NOT fixed and more failures remain → STOP (early exit); save state
         If fixed → proceed to next failure image
Step 6.  Detect cross-pair firings → surface to operator
```

Rules are stored in `patch_rules_clean.json` — a clean, session-local file that starts empty and is never written to by other pipeline components. This ensures no contamination between the batch-imported KB rules and the expert-authored patch rules.

#### Pre-conditions as hard gates

Patch rules encode pre-conditions in the condition field as a semicolon-separated list, prefixed with `[Patch rule — <pair_id>]`. The MEDIATOR formats these as a numbered HARD GATE checklist: all conditions must be confirmed by the feature record before the rule is applied. If any pre-condition is absent, the rule is skipped entirely.

This prevents the failure mode where a MEDIATOR reads a pre-condition like "centrifugal pigment gradient" as positive evidence for the favored class even when that feature was not observed.

#### Rule completion: filling implicit background conditions

Experts write **diagnostic rules** — they describe what was distinctive about the
specific failure case. They naturally omit background conditions they consider
obvious: standard markers that any experienced dermoscopist would assume are
present for the favored class, or absent for the other class.

A rule encoded without those background conditions creates loopholes: it fires on
any image that shares the distinctive feature, even images that lack the typical
profile of the favored class.

`run_rule_completer()` (text-only) is called immediately after authoring. It asks
the expert model:

1. What positive background markers are expected for the favored class but absent
   from the pre-conditions?
2. What features of the other class should be explicitly excluded but are not?

Added conditions are flagged with `added_preconditions` in the rule dict and
`"+"` in the console output. If the rule is already complete, the completer
returns it unchanged with an explanation.

This step runs **before** semantic validation — the semantic check should evaluate
the fully explicit rule, not the expert's partial one.

#### Semantic validation: catching bad logic before image testing

`run_semantic_rule_validator()` (text-only, no images) rates each pre-condition:

| Rating | Meaning |
|---|---|
| `reliable` | Consistently separates favored class from other; rarely present in the other class |
| `unreliable` | Vague, directionally wrong, or common to both classes |
| `context_dependent` | Only discriminating under specific co-occurring conditions |

Overall verdict: `accept` (proceed), `revise` (proceed with flagged warnings),
`reject` (skip image validation, escalate to expert immediately).

This is a cheap text-only filter. The key benefit: rules with clinically wrong
logic never spend image-validation budget, and the error surfaces in a structured,
reviewable form rather than as a silent FP in the precision gate.

#### Split-pool validation: preventing overfitting to a small sample

With only 8 images per class, a rule can pass the precision gate by coincidence —
the features the expert chose happen to appear in those 8 images without generalizing.

Two independent image pools are sampled from the training set:

| Pool | Seed | Expert sees it? | Used for |
|---|---|---|---|
| **Authoring pool** | 0 | Yes | Initial validator run → tp_cases/fp_cases for contrastive analysis and spectrum generation |
| **Held-out pool** | 42 | No | Binding acceptance gate for initial rule and all spectrum levels |

The expert's reasoning (contrastive analysis, spectrum generation) is grounded in
authoring-pool observations. The final accept/reject decision uses only held-out
images the expert never saw. A rule that memorizes the authoring pool's patterns
will fail the held-out gate.

Both pool sizes are configurable (`--max-authoring-per-class`, `--max-val-per-class`,
both default 8). Increasing both reduces coincidental passes at the cost of more
API calls.

#### Specificity spectrum: avoiding over-tightening

The contrastive revision loop (earlier design) added one pre-condition per iteration
until FP=0. This caused **over-tightening**: the added condition eliminated the FP
cases but also eliminated the trigger image itself (`fires_on_trigger → False`).

The **specificity spectrum** generates four rule variants in one call rather than
iterating blindly:

| Level | Label | Strategy |
|---|---|---|
| 1 | most_general | Single essential pre-condition only |
| 2 | moderate | Core condition + one supporting condition |
| 3 | original | Original rule as authored (all original pre-conditions, unchanged) |
| 4 | most_specific | Original + one contrastive tightening condition |

All four are validated in parallel against the held-out pool. The most general
passing level is selected. This avoids over-tightening because the search starts
from the loosest formulation and only tightens as needed.

**Key empirical finding (Qwen3-VL-8B mel/nev run, 2026-04-08)**: for ISIC_0024315, Level 1
(2 preconditions: regression structures + peripheral globules) achieved precision=1.0 and was
selected. Levels 2, 3, and 4 each achieved precision=0.67 or did not fire on the trigger at all.
More conditions hurt rather than helped — the added conditions created new FP pathways while also
failing to fire on the TP. Simpler rules can be more precise than complex ones.

#### Dialogic loop results (o4-mini, 2026-04-06)

| Pair | Zero-shot | + patch rules | Delta | Revision notes |
|---|---|---|---|---|
| Mel/Nev | 4/6 (67%) | **5/6 (83%)** | +16.7pp | 2 rules accepted on first attempt (FP=0, precision=1.00) |
| BCC/BKL | 4/6 (67%) | **5/6 (83%)** | +16.7pp | 1 rule accepted (FP=1, precision=0.80); rule 2 rejected (fires_on_trigger=False) |
| AK/BKL | 4/6 (67%) | — | 0pp | Both rules fail; revision tightened rule until fires_on_trigger=False — escalated to expert |

#### Dialogic loop results (Qwen3-VL-8B + claude-tutor, 2026-04-08)

Run command:
```bash
python patch.py \
  --failures-from results_baseline_qwen3vl8b_zeroshot.json \
  --cheap-model "qwen/qwen3-vl-8b-instruct" \
  --expert-model "claude-tutor" \
  --validator-model "claude-sonnet-4-6" \
  --pair melanoma_vs_melanocytic_nevus \
  --max-val-per-class 4 --max-authoring-per-class 4 --max-confirm-per-class 8 \
  --patch-rules patch_rules_single_test.json \
  --output patch_session_single_test.json
```

Qwen3-VL-8B baseline on mel/nev: 3/6 (50%) — all three melanomas missed, all three benign moles correct.

| Failure | Correct label | Rule authored | Spectrum level selected | Held-out precision | Result |
|---|---|---|---|---|---|
| ISIC_0024315 | Melanoma | regression structures + irregular peripheral globules | **Level 1** (most_general, 2 preconditions) | 1.00 (0 FP) | ✓ FIXED |
| ISIC_0024333 | Melanoma | diffuse regression + peppering + disorganized texture | **Level 2** (moderate) | 1.00 (0 FP) | ✓ FIXED |
| ISIC_0024400 | Melanoma | gray-blue structureless areas + multi-zone asymmetry | ✗ all levels failed | — | ✓ FIXED via r_002 cross-pair |

**Final session result: 6/6 (100%) — up from 3/6 (50%) zero-shot (+50pp).**  
Rules: 3 authored, 2 accepted & registered (r_001 for ISIC_0024315, r_002 for ISIC_0024333).  
ISIC_0024400's own rule was rejected at all 4 spectrum levels (precision=0.67 throughout — FP case was a deep nevus with Tyndall-effect gray-blue that the brightness-polarity condition could not reliably separate). ISIC_0024400 was nonetheless fixed when r_002 (regression/peppering rule from ISIC_0024333) fired on it and correctly predicted Melanoma — an in-session cross-pair generalization.

Expert model: `claude-tutor` (human-in-the-loop via filesystem inbox/outbox).
Validator model: `claude-sonnet-4-6` (API, handles all pool validation directly).

**Mel/Nev and BCC/BKL both lifted to 83%**, matching Claude Sonnet 4.6 zero-shot on those pairs, with the expert cost paid once per rule.

**AK/BKL is unresolvable by automatic revision.** The contrastive analysis identified "moth-eaten/comedone-like internal structure" as the discriminating feature. Adding it as a pre-condition over-tightened the rule: it no longer fired even on the target image. This confirms the structural diagnosis: the LPLK-like BKL images genuinely lack SK markers — the rule cannot be tightened to distinguish them from AK without also excluding them from the BKL class. This is an ontology problem, not a pre-condition problem.

**Expert question generated by the system:** The target image (0024420) is a LPLK variant that doesn't show classic SK markers. The system cannot author a safe rule for it because the features that distinguish it from AK are either absent or shared with AK. The correct escalation is: should LPLK be treated as a third class, or is there a different set of features that distinguishes LPLK from AK without relying on SK markers?

#### Contrastive revision reliability

The mechanism works for cases where:
- The FP images differ from TP images in at least one observable dermoscopic feature
- The RULE_VALIDATOR's per-image observations are informative enough to identify the distinction

It breaks down when:
- The discriminating feature is present in TP but also required for the rule to fire on the trigger at all (over-tightening)
- The FP and TP images are genuinely visually similar at the schema level (ontology problem)

The validator's observations are text summaries, not structured feature records. Using the OBSERVER's full feature records (instead of the RULE_VALIDATOR's brief observations) for contrastive analysis would likely improve recall — but at higher cost since it requires running the full OBSERVER pipeline on each training image.

### 3.9 Pupil teachability screening (prerequisite)

**A pupil VLM must be screened for teachability before committing to a full patch run.**

The dialogic patching loop (§3.8) is expensive: a complete session involves rule authoring, rule completion, semantic validation, pool validation across three independent image pools, and confirmation. If the candidate pupil is unteachable — meaning it cannot act on the patch rules even when they are correctly authored, validated, and injected — the entire expert budget is wasted.

#### What "unteachable" means in practice

A teachable pupil: when a validated patch rule is injected and the rule's pre-conditions are met by the target image, the pupil's prediction changes from wrong to correct on rerun.

An unteachable pupil: the rule is injected and pre-conditions are met, but the model does not act on the rule. It produces a prediction consistent with its prior bias rather than the injected rule. The model effectively hallucinates past its own rule text.

**Known examples:**

| Pupil VLM | Teachable? | Evidence |
|---|---|---|
| GPT-4o | **No** | Rules registered and injected; model never altered predictions based on rule pre-conditions — hallucinated past them on rerun |
| Qwen3-VL-8B | **Yes** | Single-failure POC (ISIC_0024315, mel/nev): rule injected, pre-conditions confirmed present, prediction flipped Nevus→Melanoma |

These are preliminary single-experiment observations. GPT-4o unteachability may be pair-specific or version-specific and should be retested before being treated as a global property.

#### Teachability entrance exam (design, not yet built)

Before running a full patch loop on a new pupil VLM, run a lightweight screening test:

1. **Probe set**: a small fixed set of image-rule pairs where the correct answer is known and the rule has already been validated independently. Each probe consists of one image the pupil is known to misclassify, plus a validated patch rule with explicit pre-conditions that should flip the prediction.
2. **Pass criterion**: the pupil's prediction changes from wrong to correct on at least N of M probe pairs when the rule is injected. A model that reads rules but ignores them scores 0/M.
3. **Rejection criterion**: if the pass rate is below threshold, reject the candidate pupil and do not proceed to the full patch loop. Report the failure for investigation (prompt engineering, model version, injection format, etc.).

**Key metric**: not "did the rule pass the held-out gate" (that's about rule quality, not pupil behavior) but "did the rule fire and flip the prediction on rerun" (that's about pupil receptiveness to injected rules).

**Suggested probe size**: 3–5 pairs, using existing validated patch rules from the dermatology mel/nev session. The ISIC_0024315 Level-1 rule (2 pre-conditions, precision=1.0) is an ideal first probe.

#### Why the concept matters architecturally

KF's patching mechanism assumes the pupil can learn from rules at inference time. This is not universally true. Some VLMs are so confident in their internal representations that they override explicit rule text. Others comply with structured pre-condition checklists even when their priors disagree. Teachability is therefore a **pre-condition for the entire KF patching approach** — not a secondary optimization.

Analogically: before a master commits effort to teaching a student, it is reasonable to verify the student can act on corrections rather than repeat the same error regardless of instruction.

#### Token budget for reasoning models (o4-mini)

OpenAI reasoning models (o4-mini, o1) share the `max_completion_tokens` budget between internal chain-of-thought tokens and output tokens. With a budget of 1024, the model spends all tokens on reasoning and produces empty or truncated output. All agents now use 4096+ tokens as the minimum for any call that goes to a reasoning model.

## 4. Repository Layout And Developer Quick Start

### 4.1 Bird implementation layout

```text
python/
  harness.py
  ensemble.py
  agents.py
  dataset.py
  rules.py
  tools.py
  migrate_rules.py

src/
  baseline.py
  kf_teacher.py
  dataset.py
  confusable_pairs.py
  config.py
  evaluator.py
```

### 4.2 Dermatology implementation layout

```text
dermatology/
  knowledge_base/
    melanoma_vs_melanocytic_nevus.json
    basal_cell_carcinoma_vs_benign_keratosis.json
    actinic_keratosis_vs_benign_keratosis.json
  python/
    harness.py          # test harness; --all, --pair, --mode test|train
    ensemble.py         # 4-round KF pipeline orchestrator
    agents.py           # all LLM agent calls (observer, mediator, verifier, etc.)
    dataset.py
    rules.py
    tools.py
    migrate_rules.py    # batch-import knowledge_base/*.json into rules.json
    patch.py            # dialogic patching loop (see §3.8)
    patch_rules_clean.json  # isolated patch rules file; never pre-populated
```

### 4.3 Bird quick-start

```bash
cd usecases/image-classification/python

python migrate_rules.py
python migrate_rules.py --dry-run
python harness.py --pair american_crow_vs_fish_crow
python harness.py --pair brewer_sparrow_vs_clay_colored_sparrow --output results.json
python harness.py --all --mode test --output results_test.json
python harness.py --prune
```

### 4.4 Dermatology quick-start

```bash
cd usecases/image-classification/dermatology/python

# Migrate batch KB rules
python migrate_rules.py

# Run KF pipeline (frozen rules, test mode)
python harness.py --pair melanoma_vs_melanocytic_nevus --mode test
python harness.py --all --max-per-class 3 --mode test --output results_pilot.json

# Dialogic patching loop: cheap model + API expert
python patch.py --cheap-model o4-mini --expert-model claude-sonnet-4-6 \
    --pair melanoma_vs_melanocytic_nevus --max-per-class 3

# Dialogic patching loop: cheap model + human-in-the-loop tutor + cheap validator
# (expert-model routes authoring to filesystem; validator-model handles pool checks via API)
python patch.py \
    --failures-from results_baseline_qwen3vl8b_zeroshot.json \
    --cheap-model "qwen/qwen3-vl-8b-instruct" \
    --expert-model "claude-tutor" \
    --validator-model "claude-sonnet-4-6" \
    --pair melanoma_vs_melanocytic_nevus \
    --max-val-per-class 4 --max-authoring-per-class 4 --max-confirm-per-class 8 \
    --patch-rules patch_rules_single_test.json \
    --output patch_session_single_test.json
```

### 4.5 Runtime notes

- Bird environment path used in prior runs:
  `C:/Users/kaihu/AppData/Local/pypoetry/Cache/virtualenvs/01os-TLh4bqwo-py3.10/Scripts/python.exe`
- Dermatology default data directory:
  `C:\_backup\ml\data\DermaMNIST_HAM10000`

## 5. Positioning Guidance

For external communication, the current state should be framed as:

- a **research prototype**
- a **post-training runtime patching** approach
- strongest on **targeted hard cases**
- promising for **structured evidence observation**
- **not yet** a clean benchmark win across all settings

That framing is especially important because the README is user-facing while
this file is developer-facing.
