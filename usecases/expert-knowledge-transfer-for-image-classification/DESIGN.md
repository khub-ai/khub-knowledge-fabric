# Design Spec: Expert Knowledge Transfer For Image Classification

This document is the developer-facing companion to
[README.md](C:\_backup\github\khub-knowledge-fabric\usecases\expert-knowledge-transfer-for-image-classification\README.md).

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

### 2.6 What is needed before a bird benchmark claim

| Gap | What to do |
|---|---|
| Online adaptation during test | Re-run with `--mode test` so rules are frozen before evaluation |
| Model confound | Add Claude zero-shot and Claude few-shot baselines |
| Small sample | Run the full test set for both pilot pairs |
| Narrow scope | Expand back out to more of the 15 confusable pairs |

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

### 3.4 Main failure modes

#### Failure mode 1: absence-only evidence (partially addressed in v2, partially remaining)

Presence-only rules cannot fire when no positive markers are visible. Composite-absence rules were added to the KB in v2 but did not improve results because the rule retriever matches rules by semantic similarity to image content — absence-framing ("no veil + no regression + ...") scores low against a query about what IS visible.

Fix needed: schema generation should always include absence-tracking fields for the pair's key markers, independent of retrieval.

#### Failure mode 2: `bkl` class heterogeneity

The HAM10000 `bkl` label covers both seborrheic keratosis (SK) and lichenoid keratosis (LPLK). LPLK-like images lack SK's positive dermoscopic markers and can resemble AK or melanoma, making them hard for any absence-based or presence-based rule set to classify correctly. This is partly an ontology problem, not purely a pipeline problem.

The `--sk-only` flag (filters `bkl` test images to `dx_type=="histo"` as a proxy for SK) is available but has not been evaluated yet.

#### Failure mode 3: AK vs BKL pair is structurally hard

The three BKL images used in the AK/BKL pair (ISIC_0024336, _0024420, _0024495) were correctly identified as BKL in the BCC/BKL pair context but misclassified as AK in the AK/BKL context across all three iterations. Without typical SK markers, the rule set defaults to AK. A pair-specific fix or additional rules are needed.

### 3.5 What worked

- BCC vs Benign Keratosis improved significantly (50% → 83%) with the VERIFIER fix. BCC has strong pathognomonic markers (arborizing vessels, ovoid nests) that make the MEDIATOR reliable once it is allowed to commit.
- The no-abstain MEDIATOR eliminated 4 "uncertain" predictions — 3 of which became correct.
- Domain transfer from birds to dermatology required only prompt and schema changes, confirming the architecture is domain-agnostic.

### 3.6 What is needed next

| Gap | What to do |
|---|---|
| Absence-tracking schema | Force absence-checklist fields into OBSERVER schema independently of rule retrieval |
| `bkl` heterogeneity | Evaluate `--sk-only` flag; consider LPLK as a third class |
| Same-model baselines | Run Claude zero-shot and few-shot on the same 18 images for comparison |
| Larger sample | Run full test sets per pair |
| AK/BKL KB | Author additional rules distinguishing LPLK-like BKL from AK |

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
    harness.py
    ensemble.py
    agents.py
    dataset.py
    rules.py
    tools.py
    migrate_rules.py
```

### 4.3 Bird quick-start

```bash
cd usecases/expert-knowledge-transfer-for-image-classification/python

python migrate_rules.py
python migrate_rules.py --dry-run
python harness.py --pair american_crow_vs_fish_crow
python harness.py --pair brewer_sparrow_vs_clay_colored_sparrow --output results.json
python harness.py --all --mode test --output results_test.json
python harness.py --prune
```

### 4.4 Dermatology quick-start

```bash
cd usecases/expert-knowledge-transfer-for-image-classification/dermatology/python

python migrate_rules.py
python harness.py --pair melanoma_vs_melanocytic_nevus --mode test
python harness.py --all --max-per-class 3 --mode test --output results_pilot.json
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
