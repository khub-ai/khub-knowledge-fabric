# ARC-AGI-3 Dialogic Distillation — Experiment Spec

> **For**: the ARC-AGI-3 competition submission that must run under an open-source VLM, and needs the PUPIL's typed-query answers to approach TUTOR-level quality without fine-tuning.
>
> **Status**: Early spec — directions and constraints below; concrete recipe (PUPIL model, question-type coverage, hierarchy structure, human involvement) still to be worked out. A scoped investigation brief exists in session memory (`project_arc_agi3_observer_dd.md`); this file is its runnable successor once the details below are decided.
>
> **Also see**: [Dialogic Distillation](../dialogic-distillation/README.md) · [Road-surface hierarchical DD](../image-classification/road-surface/README.md) · [ARC-AGI-2 usecase](../arc-agi-2/README.md)

---

## The One-Line Summary

A small open-source VLM (PUPIL) answers typed Observer and Mediator queries weakly on ARC-AGI-3 frames. A stronger TUTOR (e.g., Sonnet-class model, or a human expert) answers the same queries correctly. We harvest the TUTOR's answers into a Knowledge Fabric keyed on an abstracted *scene bucket*, retrieve relevant entries at inference time, and inject them into the PUPIL's prompt. The PUPIL's weights never change. The retrieval layer is inspectable, editable, and game-agnostic.

---

## 1. Why This Is a Plausible DD Target

Three properties make ARC-AGI-3 typed queries a promising DD surface:

1. **The question surface is narrow and typed.** The engine does not ask the VLM to act as a general reasoner — it asks a small set of JSON-schema-constrained queries (`ENUMERATE_OBJECTS`, `CLASSIFY`, `IDENTIFY_ROLES`, etc.). DD tends to work well on typed shards, and this surface is already sharded.
2. **The environment provides an outcome signal for free.** Unlike still-image classification where labels are fixed, a rule that helps PUPIL answer a typed query correctly ultimately shows up (or fails to) as completed episodes. Score / level-win is a second validator beyond TUTOR-agreement, usable alongside or instead of it.
3. **Related infrastructure exists.** Three-party DD runners, precision-gated pool evaluation, and grounding checks have been built for image-classification usecases; parts of them may be reusable here. How much actually transfers — and how much needs to be rebuilt for the temporal, multi-step nature of game frames — is an open question this experiment is meant to answer.

---

## 2. Scope — To Be Worked Out

This experiment's scope is deliberately under-specified at this stage. Each of the following decisions is open:

- **Question-type coverage.** The first run will cover more than a single typed query. The exact set (which Observer queries, whether any Mediator queries, and in what order they are tackled) is TBD and should follow from which queries actually get triggered with non-trivial frequency in live runs.
- **PUPIL model.** Not yet chosen. Candidate families include open-source VLMs such as Qwen-VL and Llama-VL classes. Selection criteria to be worked out (parse-success baseline on typed queries, hosting cost, swayability profile).
- **TUTOR model(s).** Plural. The protocol does not commit to a single TUTOR; a stronger API-hosted model and/or a human expert are both in play, and whether to mix them is TBD.
- **Human involvement.** Whether and how humans participate — as TUTOR, as KF curator reviewing accepted rules, as judge on outcome-ambiguous episodes — is TBD.
- **Hierarchy structure.** Whether the KF is organized flat (by question type only), two-level (scene-bucket × question type, as in road-surface / dermatology), or deeper, is TBD. The right shape depends on how redundantly frames cluster on cheap abstractions; that has not been measured.
- **Games and levels.** The first run targets ARC-AGI-3 games but the specific game(s) and level(s) are TBD.
- **Acceptance gates.** Whether to use a pool-precision gate, an outcome-lift gate, both in sequence, or something new tuned to the multi-step nature of games, is TBD.
- **Harvest source.** Whether the corpus for TUTOR harvest comes from live engine runs, a dedicated offline pass, human-curated hand traces, or a mix, is TBD.

The sections below sketch directions, not commitments. Every concrete number, model name, and command is a placeholder until an initial scoping pass lands.

---

## 3. The Pipeline — Sketch

```
  frame ─┬─► (optional) scene-abstraction router ───► scene_bucket
         │
         │                         ┌──────── KF (per question_type [× scene_bucket]) ────────┐
         │                         │  TUTOR-harvested entries, gated (gates TBD)             │
         │                         └───────────┬─────────────────────────────────────────────┘
         │                                     │  retrieval (strategy TBD)
         ▼                                     ▼
  PUPIL(typed-query prompt  +  [retrieved KF entries])
         │
         ▼
  parsed typed answer ──► engine continues ──► episode outcome (score, level-win)
                                                       │
                                                       ▼
                                          (optional) outcome-based feedback into KF
```

Every block above has open design questions — router structure, retrieval strategy, gate criteria, outcome feedback mechanism. The pipeline is presented as a shape to fill in, not a commitment.

---

## 4. Cast — Placeholder

| Role | Model | Notes |
|---|---|---|
| PUPIL | TBD (open-source VLM) | The model the competition submission will actually deploy. |
| TUTOR | TBD (one or more stronger models, and/or human) | Only during harvest + validation. Never in the deployed loop. |
| VALIDATOR | TBD | Grounding / consistency check on harvested entries. Possibly the same as TUTOR with a separate prompt; possibly distinct. |
| ENV-VALIDATOR | the ARC-AGI-3 harness itself | Provides the outcome signal (score, level-win) if we choose to use it as a gate. |

---

## 5. Scene-Bucket Router — TBD

Direction: a cheap, pixel-derived abstraction of each frame that produces a discrete bucket, used (if a hierarchy is adopted) as a retrieval key. Features under consideration include palette signature, grid size, object-count class, and sparsity. Whether to route at all, and what the right abstraction set is, has not been decided and should be driven by empirical bucket cardinality + consistency measurements on real frames.

**Open:** whether routing is needed at all; feature set; bucket cardinality targets; how to measure that the chosen bucketing is informative (i.e. that frames in the same bucket tend to admit the same typed answer).

---

## 6. KF Entry Schema — TBD

Direction: JSON entries keyed at minimum on question type, and on scene bucket if a hierarchy is adopted. Each entry carries the TUTOR-provided corrective content, preconditions for when it should fire, a rationale, and whatever validation statistics the chosen gate produces. Exact schema TBD.

---

## 7. Harvest and Acceptance Gates — TBD

Direction: produce candidate entries from (frame, prompt, TUTOR answer) triples where PUPIL and TUTOR disagree; apply one or more acceptance gates before an entry enters the deployed KF. Which gates to use, in what order, with what thresholds, and how to handle the multi-step nature of games (where a "correct" typed answer does not trivially decompose to a single-image label) is open.

---

## 8. Outcome Gate — TBD

Direction: use episode outcome (score, level-win) as a second or alternative validator for KF entries. Open: how to handle credit-assignment across a multi-step episode, whether to gate per-rule or per-bucket, and what statistical sample size is achievable within the experiment's compute budget.

---

## 9. Instrumentation — TBD

Non-negotiable direction: whatever we build must measure, per run, how often KF entries are actually retrieved, how often their preconditions match, and how often they enter the PUPIL's prompt. Prior image-classification work shipped hierarchical DD claims with effectively zero rule-fires end-to-end; we will not repeat that. Exact metric names and logging format TBD.

---

## 10. Step-by-Step First Run — TBD

To be written once §§2, 5, 6, 7 are decided.

---

## 11. Success Criteria and Failure Modes — TBD

To be written once §7 and §8 gate choices are decided. Will reference baseline metrics collected during initial PUPIL scoping.

---

## 12. Lessons to Carry Over From Image-Classification — TBD

Candidate lessons to consider (none adopted yet):
- Feature-denial bypass via pixel-derived (not VLM-answered) abstractions.
- L1 / routing accuracy as a make-or-break measurement before claiming hierarchical gain.
- Starting narrow before scaling coverage.
- Honest reporting of what was run vs. what was promised.

Which of these apply verbatim to game frames (vs. requiring adaptation for temporal / multi-step context) is itself something to work out.

---

## 13. Key Files and Their Intended Roles — TBD

Directory shape will emerge as §§5–10 are decided.

```
usecases/arc-agi-3-dd/
  README.md                                    ← this file
  python/                                      ← TBD
  benchmarks/                                  ← TBD
  knowledge_base/                              ← TBD
```

### Cross-repo pointers (reference only)
- Typed-query prompts (PUPIL/TUTOR surface): `C:\_backup\github\arc-agi-3\arc_agi_3\observer.py`
- Oracle triggers (engine side): `C:\_backup\github\cognitive-os-engine\cognitive_os\oracle.py`
- Backend swap point: `C:\_backup\github\arc-agi-3\arc_agi_3\backends\base.py`

---

## 14. Follow-ups — TBD

To be listed once the first run's scope is concrete.
