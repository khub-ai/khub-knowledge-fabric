# Dialogic Distillation — Developer Design Spec

This document is the developer-facing reference for the Dialogic Distillation use case.
For the user-facing overview, see [README.md](README.md).

---

## Contents

1. [Motivation and Core Problem](#1-motivation-and-core-problem)
2. [The Three-Party Protocol](#2-the-three-party-protocol)
3. [The Grounding Problem — Why Single-Shot Fails](#3-the-grounding-problem--why-single-shot-fails)
4. [KF Steering Moves](#4-kf-steering-moves)
5. [Pool Gate Mechanics](#5-pool-gate-mechanics)
6. [Knowledge Types Across Domains](#6-knowledge-types-across-domains)
7. [Domain: Image Classification (Dermatology and Ornithology)](#7-domain-image-classification-dermatology-and-ornithology)
8. [Domain: ARC-AGI-3 Competition Mode](#8-domain-arc-agi-3-competition-mode)
9. [Adapting the Protocol for ARC-AGI-3](#9-adapting-the-protocol-for-arc-agi-3)
10. [Rule Lifecycle](#10-rule-lifecycle)
11. [Relation to Published Work](#11-relation-to-published-work)
12. [Design Issues and Open Questions](#12-design-issues-and-open-questions)

---

## 1. Motivation and Core Problem

Dialogic Distillation addresses the capability gap between a high-quality expert model (TUTOR) and a constrained deployment model (PUPIL). The goal is to close that gap in a specific domain without modifying the PUPIL's weights.

The approach is motivated by two observations:

**Observation 1: Capability gaps are often narrow.** A cheap model is not uniformly worse than an expensive one. It fails on specific patterns — visual features it underweights, reasoning steps it skips, abstractions it cannot hold simultaneously. These failure patterns are diagnosable, and the corrective knowledge is often compact enough to fit in a few text rules.

**Observation 2: Fine-tuning is expensive and opaque.** Compressing TUTOR knowledge into PUPIL weights via supervised fine-tuning requires a large labeled dataset, GPU infrastructure, and produces a checkpoint whose learned knowledge cannot be inspected, revised, or selectively updated. When the domain shifts, the process starts over.

**The alternative**: extract the TUTOR's corrective reasoning as explicit text artifacts and inject them into the PUPIL's context at inference time. The PUPIL applies them as mandatory instructions on matching cases.

The critical challenge is that the TUTOR's natural reasoning does not always translate into rules that the PUPIL's validator can evaluate reliably. This requires multi-round dialogue — hence "dialogic" distillation.

---

## 2. The Three-Party Protocol

### Roles

**PUPIL** — the model being improved. Runs the target task and fails. Its failures become the training signal for rule authoring. Its weights are never modified. In image classification: a cheap VLM (e.g., Qwen3-VL-8B). In ARC-AGI-3: the open-source model used in competition mode.

**TUTOR** — the expert model (or human expert acting as one). Receives the PUPIL's failure in context and proposes corrective rules. In practice: Claude Sonnet or Opus acting as domain expert, or a human routing responses via file inbox/outbox (`claude-tutor` pattern). Does not need to know the PUPIL's architecture or weights.

**KF (Knowledge Fabric orchestrator)** — steers the entire loop. Surfaces PUPIL failures to TUTOR, runs grounding checks and pool validation, feeds validator observations back to TUTOR as steering input, and registers accepted rules into the knowledge base.

### Protocol skeleton

```
for each PUPIL failure f:

    Round 1:
        KF → TUTOR: present f (prediction, reasoning, ground truth, input)
        TUTOR → KF: proposed rule R (preconditions + corrective action)

    Grounding check:
        KF → VALIDATOR: does the trigger input meet R's preconditions?
        if yes: proceed to pool gate
        if no:
            KF → TUTOR: validator's exact observations + steering guidance
            TUTOR → KF: refined rule R' (Round 2, 3, …)
            repeat grounding check until grounded or max_rounds exceeded

    Pool gate:
        KF: sample held-out pool (balanced, train split only)
        KF → VALIDATOR (×N): does each pool image meet R's preconditions?
        compute precision, FP count
        if precision ≥ threshold AND FP ≤ 1 AND fires_on_trigger:
            register R into knowledge base
        else:
            optionally: contrastive tightening round → new R → back to pool gate

    Rule injection:
        on future PUPIL calls: prepend matching registered rules to system prompt
        as MANDATORY RULES with IF/THEN format
```

### VALIDATOR vs TUTOR

The VALIDATOR is a separate agent from the TUTOR. It answers binary questions ("do this image's visible features meet all preconditions in rule R?") rather than authoring rules. In practice it can be a cheaper model than the TUTOR. Keeping them separate prevents the TUTOR from self-grading, which would undermine the grounding check.

---

## 3. The Grounding Problem — Why Single-Shot Fails

**Empirical result**: In the dermatology three-party experiment (`distill_dialogic.py`), single-shot elicitation produced 0/4 grounded rules. The dialogic loop produced 4/4 grounded rules.

The failure mode is vocabulary mismatch. The TUTOR writes rules using domain-expert vocabulary ("irregular pigment network with abrupt cutoff at the periphery"). The VALIDATOR evaluates images using visual description vocabulary ("relatively symmetric oval shape with a corona pattern at the border"). Both descriptions may be correct for the same image at different levels of abstraction, but the rule's preconditions must match what the VALIDATOR actually observes in order to fire.

Single-shot elicitation has no feedback loop. The TUTOR proposes a rule based on their own vocabulary; if it does not fire during grounding check, the rule is discarded. There is no mechanism to tell the TUTOR *why* it failed or *what vocabulary to use instead*.

The dialogic loop fixes this by feeding VALIDATOR observations back to the TUTOR as explicit steering: "the validator described this image as X — try writing the preconditions in terms of X rather than Y." After one or two rounds, the TUTOR's rule language converges to the VALIDATOR's vocabulary and the grounding check passes.

---

## 4. KF Steering Moves

KF applies different steering strategies depending on what failed and how many rounds have elapsed:

**Vocabulary alignment** (primary strategy, round 1–2 failures):
> "The validator's description of this image was: [exact text]. Your precondition uses the phrase [X]. Try rewriting it using the terms the validator used."

**Specificity coaching** (when too many preconditions fail individually):
> "Your rule has N preconditions. The validator found that M of them did not hold. Consolidate to the 2–3 most visually concrete and reliable ones."

**Feature pivot** (round 3+, when vocabulary alignment is not converging):
> "The feature you are describing may not be reliably visible at this image resolution. Consider anchoring the rule on [alternative feature the VALIDATOR did confirm]."

**Contrastive tightening** (after pool gate — too many false positives):
> "The rule fires on [FP count] images from the negative class. Here are the observer's descriptions of those false positive cases vs the true positive cases: [side-by-side]. What distinguishes the true positives? Add that as an additional precondition."

KF chooses steering moves based on:
- Grounding check result (which preconditions fired, which did not, validator's exact text)
- Round number
- Pool gate result (FP count, precision, which pool images produced FPs)

---

## 5. Pool Gate Mechanics

The pool gate is a mandatory validation step that prevents overfitted or over-broad rules from reaching the knowledge base.

**Pool construction**:
- Sampled from the training split only (test images are never used)
- Balanced: equal images from both classes
- Deterministic seed per session (default 42 for held-out pool, 123 for confirmation pool)

**Pass conditions** (all three must hold):
1. `fires_on_trigger == True` — the rule fires on the original failure image
2. `precision ≥ min_precision` (default 0.75)
3. `FP_count ≤ 1`

Precision alone is not sufficient because a rule that never fires has precision 1.0. Requiring `fires_on_trigger` prevents degenerate rules that technically achieve high precision by not firing at all.

**Contrastive tightening**: if the pool gate fails due to excess FPs (but the rule fires on the trigger), KF runs a contrastive analysis round: the TUTOR sees side-by-side descriptions of the true positives and false positives in the pool and proposes a tightening condition. The tightened rule is re-validated from the grounding check.

**Spectrum generation** (optional, dermatology variant): the TUTOR generates four versions of the rule at different specificity levels (1 core condition → 2 conditions → full rule → full rule + tightening). KF validates all four and selects the tightest version that still passes the pool gate. This avoids both over-tightening (rule stops firing entirely) and under-tightening (still too many FPs).

---

## 6. Knowledge Types Across Domains

The distillation loop structure is domain-agnostic, but the nature of the rules varies significantly by domain.

### Visual classification (image-classification use case)

Rules are **declarative visual discrimination cues**:
```
preconditions: [
    "gray-blue or pink-gray structureless areas visible within the lesion body",
    "irregular dark dots or globules clustered peripherally on one side"
]
action: "Classify as Melanoma"
```

Properties:
- Static (same image every time — no state)
- Evaluatable by a vision-capable VALIDATOR without running the environment
- Checkable against a pool of labeled images cheaply
- Human-inspectable by a domain expert
- Narrow enough to have high precision on a balanced pool

### ARC-AGI-3 (planned)

Rules are **procedural and strategic heuristics**:
```
preconditions: [
    "game is in early exploration phase (fewer than 3 full action cycles completed)",
    "one or more non-background colored objects have not yet been tested for movement"
]
action: "Test each untested object by issuing each of the 4 base actions once and
         observing which objects change position. Record results before planning."
```

Properties:
- State-dependent (rule fires in a specific phase of an episode)
- Evaluatable only by running the environment (or replaying a recorded trajectory)
- Pool validation requires N replay episodes rather than N image lookups
- Knowledge is strategic, not perceptual — the PUPIL knows what it sees; it fails on what to do with that information

This distinction drives the adaptations required for ARC-AGI-3 (see §9).

---

## 7. Domain: Image Classification (Dermatology and Ornithology)

### Reference implementation

The full dialogic distillation protocol is implemented in:

```
usecases/image-classification/dermatology/python/distill_dialogic.py
usecases/image-classification/dermatology/python/patch.py
usecases/image-classification/birds/python/patch.py
```

`distill_dialogic.py` is the research protocol (three-party distillation, full transcript logging).
`patch.py` in each domain is the production-ready dialogic patching loop integrated with the dataset and rule store.

### Agent roles in the patching loop

| Agent | Model | Function |
|---|---|---|
| `EXPERT_RULE_AUTHOR` | TUTOR (Sonnet/Opus or human) | Authors initial corrective rule from failure image |
| `RULE_COMPLETER` | TUTOR | Adds implicit background conditions |
| `SEMANTIC_RULE_VALIDATOR` | VALIDATOR (Sonnet) | Reviews each precondition for reliability (ACCEPT/REVISE/REJECT) |
| `RULE_VALIDATOR` | VALIDATOR (Sonnet, ×N) | Binary: does this image meet all preconditions? |
| `CONTRASTIVE_ANALYSIS` | TUTOR | Identifies feature separating TP from FP pool images |
| `RULE_SPECTRUM_GENERATOR` | TUTOR | Generates 4 specificity levels |

Semantic validation is advisory (REVISE does not block the pool gate). It is useful for surfacing preconditions that are likely to produce FPs before running the pool, saving validator calls.

### Rule injection format

Rules are prepended to the PUPIL's system prompt as structured mandatory instructions:

```
MANDATORY RULES — if a rule's pre-conditions are met, you MUST apply it:

RULE 1:
  IF: <full precondition list as comma-separated conditions>
  THEN: <classification + confidence + rule text>

RULE 2:
  …
```

The PUPIL outputs `"rule_fired": "<rule_id or null>"` in its response JSON, enabling
post-hoc analysis of which rules are active vs. firing.

### Empirical results

| Experiment | Zero-shot | With rules | Gain |
|---|---|---|---|
| Mel/Nev 30/class (patch loop) | 55% | 93.3% | +38.3 pp |
| BCC/BKL 30/class (patch loop) | 56.7% | 75% | +18.3 pp |
| Mel/Nev 30/class (dialogic distill, 3 rules) | 55% | 91.7% | +36.7 pp |
| Mel/Nev single-shot elicitation (4 attempts) | — | 0/4 grounded | — |
| Mel/Nev dialogic (4 attempts) | — | 4/4 grounded, 3/4 accepted | — |
| Birds (Cowbird, pilot 6-image) | 33% | 83% | +50 pp |
| Birds (expanded 30/class) | 46.7% | 96.7% | +50 pp |

---

## 8. Domain: ARC-AGI-3 Competition Mode

### The capability gap

In non-competition mode, the ARC-AGI-3 ensemble uses Claude Sonnet 4.6 for the OBSERVER and MEDIATOR roles. Sonnet applies a systematic exploration discipline naturally: it catalogs objects, tests actions in isolation, withholds goal hypotheses until sufficient evidence is gathered, and revises when predictions fail.

In competition mode, a smaller open-source model replaces Sonnet. The failure mode is not perceptual — the model can describe the grid accurately. The failure is strategic: it forms premature goal hypotheses, skips systematic action testing, and does not revise plans when predictions fail.

This is exactly the structure that Dialogic Distillation addresses: the PUPIL sees the same inputs but applies a weaker reasoning process. The TUTOR's advantage is not access to privileged information — it is the discipline of structured exploration.

### What distillable knowledge looks like

**Category 1: Concept identification heuristics**
> "If a small cluster of non-background cells moves 5 cells uniformly when any single action is called, that cluster is the player-controlled piece. Assign it role `player_piece` in concept_bindings."

These are game-agnostic behavioral signatures. The PUPIL can see the grid and the action effects; it fails to systematically derive object roles from them. The rule makes the inference procedure explicit.

**Category 2: Goal recognition patterns**
> "If any numeric attribute decreases monotonically across consecutive actions and its current value is within 20% of its starting value, add an urgency goal: minimize remaining actions before that attribute reaches zero."

Depletion counters, level timers, and resource pools all share this behavioral signature across different games. The rule transfers the TUTOR's pattern recognition into an explicit, injectable heuristic.

**Category 3: Strategic meta-rules (highest value)**
> "In the first 3 action cycles of any unknown level: (1) catalog all distinct non-background colors; (2) test each action once in sequence; (3) record which objects moved, by how much, and whether any were blocked; (4) do not commit to a goal hypothesis until all 4 actions have been tested at least once."

This is procedural knowledge about *how to explore* rather than what to do in a specific situation. It is the type of knowledge that transfers most broadly across games and that the TUTOR applies implicitly but the PUPIL skips.

---

## 9. Adapting the Protocol for ARC-AGI-3

### What changes relative to the image-classification protocol

**1. Failure attribution requires trajectory diffing**

In image classification, a failure is atomic: the PUPIL saw image X and predicted the wrong class. In ARC-AGI-3, a failure is an episode: the PUPIL took up to N actions and either ran out of budget or never advanced the level. The actual root cause may be at step 3 of a 20-step episode, but the failure is only observable at the end.

KF must run both TUTOR and PUPIL on the same episode and compare their trajectories to find the **earliest meaningful divergence** — the first step where their action plans or reasoning differ in a consequential way. That divergence step becomes the "failure image" equivalent: it is presented to the TUTOR as the failure case to diagnose.

**2. Grounding check requires environment replay**

In image classification, the grounding check runs the VALIDATOR against a static image. In ARC-AGI-3, the only way to test whether a proposed rule leads to better behavior at the divergence step is to replay the episode with the rule injected and observe whether the PUPIL's MEDIATOR output at that step improves.

This is more expensive than an image lookup (one environment interaction per grounding check round vs. one VALIDATOR API call). Budget the grounding loop accordingly.

**3. Pool gate uses episode-level success metrics**

The pool gate in image classification samples N images and measures precision on the balanced pool. In ARC-AGI-3, the pool is N fresh episodes of the same game (or structurally similar games). The metric is level-advance rate or meaningful-progress rate (fraction of episodes where the episode ended in advance or improvement, not just budget exhaustion), compared against a PUPIL baseline without the rule.

**4. Game-agnostic rules are the priority**

Some rules are game-specific facts: "In LS20, the cursor step size is 5 cells per action." These should be stored in the game-specific knowledge base (`game_knowledge.json`), not in the dialogic distillation rule store. Dialogic Distillation should focus on **game-agnostic** strategic and procedural heuristics — those are what the PUPIL carries into the competition, where the game identity is unknown.

During pool gate validation, game-agnostic rules should be tested on multiple different games, not just the game where the failure was observed. A rule that only improves behavior on LS20 is a game-specific fact, not a distilled strategic insight.

### Modified protocol for ARC-AGI-3

```
for each PUPIL episode failure E:

    Step 0: Trajectory diff
        Run TUTOR on the same episode (same game, same starting seed)
        Compare OBSERVER/MEDIATOR outputs step by step
        Find earliest divergence step D where plans meaningfully differ

    Round 1:
        KF → TUTOR: present divergence state at step D
            (game frame, action history up to D, PUPIL's observation,
             PUPIL's action plan, TUTOR's action plan, episode outcome)
        TUTOR → KF: proposed strategic rule R
            (preconditions: what state/phase conditions trigger this rule;
             action: what the MEDIATOR should do differently)

    Grounding check:
        KF: replay episode from start, inject rule R into MEDIATOR context
        KF: observe MEDIATOR output at step D with rule present
        if MEDIATOR's plan at D matches TUTOR's intended direction:
            proceed to pool gate
        if not:
            KF → TUTOR: MEDIATOR's actual output at step D + steering guidance
            TUTOR → KF: refined rule R' (Round 2, 3, …)
            repeat

    Pool gate:
        KF: run PUPIL + rule R on N fresh episodes of:
            (a) same game
            (b) 1–2 different games (game-agnostic test)
        measure level-advance rate vs. PUPIL baseline (no rule)
        if improvement ≥ threshold on both (a) and (b):
            register R as a game-agnostic rule
        if improvement only on (a):
            register R as a game-specific rule (lower priority)
        if no improvement: discard or flag for further refinement

    Rule injection:
        on future PUPIL episodes: prepend matching registered rules
        to MEDIATOR system prompt as mandatory strategic instructions
```

### Trajectory diff implementation note

The trajectory diff requires running both TUTOR and PUPIL on the same episode with the same environment seed. The `harness.py` episode loop already records full OBSERVER/MEDIATOR outputs per step via `EpisodeLogger`. The diff logic needs to compare these logs across the two runs and identify the first step where the MEDIATOR's action plan deviates from a TUTOR reference run in a consequential way (not just a wording difference — a different action sequence or a different goal update).

A heuristic: compare the set of actions proposed in each plan (not the prose), and the goal additions/removals. A meaningful divergence is a step where the PUPIL proposes an action that the TUTOR did not and that led to no progress (wall collision, redundant move, wasted budget).

---

## 10. Rule Lifecycle

Rules go through the same lifecycle in both domains:

```
authored → grounded → pool-validated → registered → injected → monitored → (revised | retired)
```

**authored**: TUTOR has proposed a rule; it has not yet been grounded.

**grounded**: the rule fires on the trigger case (the specific failure that prompted it).

**pool-validated**: the rule passes the held-out pool gate (precision ≥ threshold, FP ≤ 1).

**registered**: the rule is written to the knowledge base with full provenance:
- which failure triggered it
- which model authored it
- grounding check transcript (all rounds)
- pool gate result (TP, FP, precision)

**injected**: the rule is prepended to matching PUPIL calls going forward.

**monitored**: in subsequent PUPIL runs, the rule's fire/correct/incorrect outcome is recorded.

**revised**: if a registered rule produces false positives in production (fires on cases it should not), the loop initiates a contrastive refinement pass — same dialogue structure as the original authoring, but now grounded on the FP cases.

**retired**: if a rule is consistently wrong or superseded by a more precise rule covering the same cases, it is removed from the knowledge base. The original rule record is preserved for provenance.

---

## 11. Relation to Published Work

| Paper | Closest feature | Key difference |
|---|---|---|
| **Inter-Cascade** (Wu et al., arXiv:2509.22984, 2025) | Failure-trigger → strategy artifact → inference-time injection | Single-shot (no multi-round grounding dialogue) |
| **GER** (Wang & Sudhir, arXiv:2408.07238, 2024) | Reusable textual guidance injected at inference time | Not failure-triggered; no PUPIL/TUTOR split |
| **ExpeL** (Zhao et al., AAAI 2024, arXiv:2308.10144) | Rules extracted from failure contrastive analysis | No pre-registration validation; no PUPIL/TUTOR split |
| **ProTeGi** (Pryzant et al., EMNLP 2023) | Held-out pool validation before committing a rule | Optimizes prompt, not rule artifact; single model |
| **Reflexion** (Shinn et al., NeurIPS 2023) | Verbal reflection injected into future context | Same model; per-task, not cross-task reusable; no pool gate |
| **Meta-Policy Reflexion** (Wu et al., arXiv:2509.03990, 2025) | Cross-task reusable structured rules, inference-time enforcement | No PUPIL/TUTOR split; no grounding dialogue |
| **DeepSeek-R1 distillation** (2025) | Strong model trains weak model on reasoning traces | Fine-tuning required; knowledge opaque; large data requirement |

The distinctive combination in Dialogic Distillation:
- failure-triggered + multi-round grounding dialogue + orchestrator-gated pool validation + inference-time injection without fine-tuning

No published paper as of April 2026 combines all four properties.

---

## 12. Design Issues and Open Questions

### 12.1 Trajectory diff is not yet implemented (ARC-AGI-3)

The episode diff logic needed to identify the earliest meaningful divergence between TUTOR and PUPIL trajectories does not yet exist. It is the primary new engineering requirement for the ARC-AGI-3 extension. See §9 for the design.

### 12.2 Pool validation cost in ARC-AGI-3

Each pool gate validation in ARC-AGI-3 requires N full episode runs, not N image-validator API calls. At ~8 LLM cycles per episode and $0.10–$0.20 per cycle, a pool of 5 episodes costs $4–$8 per rule candidate. The grounding loop and pool gate should be designed with tight round limits and early-exit conditions.

### 12.3 Game-agnostic vs. game-specific rule classification

The protocol distinguishes game-agnostic rules (test on multiple games, register globally) from game-specific rules (register in game_knowledge.json). The boundary is not always clear. A heuristic: if a rule refers to specific colors, step sizes, or object counts from a known game, it is game-specific. If it refers only to behavioral signatures (monotonic decrease, uniform movement, blocked movement), it is game-agnostic.

### 12.4 Rule store unification

In the image-classification use case, the patch loop's rule store (`patch_rules_*.json`) and the ensemble harness rule store (`rules.json`) are separate files with different formats. This creates friction when transferring validated patch rules into the harness. A unified rule store format that both loops can consume directly would simplify the architecture.

### 12.5 Semantic validation is advisory

The semantic validation step (SEMANTIC_RULE_VALIDATOR) reviews each precondition for reliability before pool testing. Its REVISE verdict does not currently block the pool gate. Enforcing REVISE as a blocking condition would reduce pool costs (fewer FPs make it to pool testing) but would also require the TUTOR to revise rules that are often salvageable with minor wording changes.

### 12.6 No cross-domain transfer yet

Rules authored in dermatology are specific to the dermatology pair. Rules authored in ARC-AGI-3 are game-agnostic but still domain-specific (reasoning about visual grids). Dialogic Distillation has not yet been applied to transfer knowledge across fundamentally different task types. This is a longer-term research question.
