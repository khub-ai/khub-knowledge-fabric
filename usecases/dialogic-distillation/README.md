# Dialogic Distillation

**Bringing a weaker AI model up to expert level in a specific domain — without retraining it.**

---

> **Status**: Active research — initial results are encouraging (image classification: dermatology +38pp, ornithology +50pp at 30/class); expanding test coverage and continuing to refine the distillation mechanism; ARC-AGI-3 competition mode: planned  
> **Theme**: [Knowledge Fabric (KF)](../../docs/what-is-kf.md) as a distillation layer between a capable expert model and a constrained deployment model  

---

## Key Findings

Results across four domains, with links to full data and analysis:

### DD improves weaker models significantly — without retraining

| Domain | PUPIL model | Accuracy before DD | Accuracy after DD | Gain |
|---|---|---|---|---|
| Dermatology (melanoma vs nevus) | `Qwen3-VL-8B` | ~55% | ~83% | +28 pp |
| Ornithology (cowbird species) | `Qwen3-VL-8B` | ~33% | ~83% | +50 pp |
| Wildfire detection (smoke vs terrain) | `claude-sonnet-4-6` | 66.7% | **95.8%** | +29 pp ✓ GO |
| Maritime SAR (person vs whitecap) | `qwen/qwen3-vl-8b-instruct` | 66.7% | **95.8%** | +29 pp ✓ GO |

Full data: [image classification](../image-classification/README.md) · [wildfire detection](../ai-fleets/wildfire-detection/README.md) · [maritime SAR](../ai-fleets/seapatch/README.md)

### Human expert rules work as well as model-generated rules

When a domain has published expert guidelines, you can skip the TUTOR model entirely and inject those guidelines directly. Wildfire: NWCG fire-spotting rules matched Opus-generated rules exactly (both 95.8%). Maritime SAR: IAMSAR search criteria matched model rules (both 83.3%). This means DD is usable even in organisations with no access to a high-end TUTOR model.

### Models have different levels of "swayability" — and this matters for deployment

We tested whether human policy directives could change model behaviour by injecting two opposite instructions: one saying "bias toward detecting persons" and one saying "default to the negative class, be conservative." The results reveal a fundamental difference between model families:

**Heavily-conditioned models (Claude):** The rescue prior is immovable. Claude-Sonnet and Claude-Opus kept calling the same orange-object frames "person in water" regardless of which directive was active — rescue-biased or conservative. You cannot override a deep safety prior with a natural-language instruction. More capable models (Opus) were *harder* to sway, not easier.

**Less-conditioned models (Qwen3-VL-8B):** Fully swayable in both directions. The rescue directive produced GO at 87.5%. The conservative directive caused the model to call *every single victim* "whitecap" — 12/12 missed, 0 false alarms. The model did exactly what it was told, in both directions.

This is not "better vs worse" — it is a choice about where the safety responsibility sits. Full analysis and data: [SeaPatch §11](../ai-fleets/seapatch/README.md#11-patchbench-probe-model-selection-for-seapatch-dd).

### DD reveals model properties invisible to standard benchmarking

The swayability finding only becomes visible when you test opposing directives and compare results. A single-direction accuracy test would miss it entirely. The dialogic method — actively trying to change model behaviour, not just scoring outputs — surfaces properties that matter for real deployment decisions.

---

## The Problem in Plain Terms

Imagine you have access to a highly capable AI model — say, Claude Sonnet — that performs very well on a specialized task: diagnosing skin lesions, identifying rare bird species, or solving complex reasoning puzzles. Now imagine you need to deploy a different, cheaper model in production: one that runs locally, fits within a competition budget, or costs a fraction of the expert model per call. That cheaper model fails on exactly the cases the expert handles well.

The standard solution is fine-tuning: collect a large set of the expert model's outputs, use them as training data, and update the cheaper model's weights. This works, but it has significant costs:

- You need a large labeled dataset
- You need GPU infrastructure and ML expertise
- The resulting knowledge is opaque — baked into weights you cannot inspect or update
- When the domain shifts, you start over

**Dialogic Distillation offers a different path.** Instead of transferring knowledge through weight updates, it extracts the expert's reasoning into explicit, human-readable rules — and injects those rules into the weaker model's context at inference time. No retraining required. The weaker model's weights never change. The knowledge lives outside the model, as auditable text artifacts you can inspect, edit, version, and reuse.

---

## How It Works

Three parties collaborate in a structured dialogue:

**PUPIL** — the weaker model being improved. It runs on the task, fails on specific cases, and its reasoning is recorded. The PUPIL's weights are never modified.

**TUTOR** — the expert model (or human expert) that knows how to handle the cases the PUPIL gets wrong. The TUTOR diagnoses failures and proposes corrective rules in natural language.

**KF (Knowledge Fabric)** — the orchestrator. KF surfaces the PUPIL's failures to the TUTOR, steers the dialogue, validates proposed rules before accepting them, and injects accepted rules into the PUPIL's context for future tasks.

The loop looks like this:

```
PUPIL fails on a case
        │
        ▼
KF shows the failure to TUTOR
(what the PUPIL predicted, what it said, why it was wrong)
        │
        ▼
TUTOR proposes a corrective rule (Round 1)
        │
        ▼
KF immediately tests the rule on the failure case
(grounding check: does the rule fire on the very image/state it was authored for?)
        │
   ┌────┴────┐
   │ fires   │ does not fire
   ▼         ▼
pool gate   KF feeds TUTOR the validator's exact observations
            ("here is what the validator saw — your preconditions
             did not match. Try using these terms instead.")
                    │
                    ▼
             TUTOR refines (Round 2, 3…)
                    │
                    └──► grounding check repeats
                              │
                         once grounded → pool gate
```

**The grounding check** is the key insight that separates dialogic distillation from single-shot elicitation. A rule proposed in one round, even by an expert, often fails to fire because the TUTOR and the validator use different vocabulary for the same thing. The TUTOR may write "irregular pigment network with abrupt cutoff"; the validator may describe the same feature as "asymmetric peripheral network." Multi-round dialogue bridges that gap — the TUTOR refines the rule's language until it matches what the validator actually observes.

**The pool gate** is a validation step before any rule is accepted. Once a rule fires correctly on the failure case, KF tests it on a held-out pool of labeled examples. The rule must achieve sufficient precision — it should not fire too broadly on cases belonging to the opposite class. Only rules that pass both the grounding check and the pool gate are registered into the knowledge base.

---

## How This Differs from DeepSeek-Style Distillation

DeepSeek-R1 (January 2025) demonstrated impressive results by training smaller models on reasoning traces generated by a larger model. That is the classical distillation approach: the expert model's behavior is compressed into the student model's weights via supervised fine-tuning.

| | DeepSeek-style distillation | Dialogic Distillation |
|---|---|---|
| **Transfer mechanism** | Fine-tune student on expert outputs | Inject explicit rules at inference time |
| **Student weights** | Updated | Never modified |
| **Knowledge form** | Implicit, in model weights | Explicit, human-readable text |
| **Inspectable?** | No | Yes — every rule is plain text |
| **Updatable?** | Requires re-fine-tuning | Add, revise, or retire individual rules |
| **Triggering** | Proactive, batch, offline | Failure-triggered, reactive |
| **Validation before use** | None at rule level | Grounding check + held-out pool gate |
| **Data requirement** | Large set of expert outputs | A handful of failure cases |
| **Infrastructure** | GPU compute, ML pipeline | LLM API calls only |
| **Portability** | Specific fine-tuned checkpoint | Any capable model can consume the rules |
| **Domain shift** | Retrain | Revise or add rules |

The key tradeoff: DeepSeek-style distillation can transfer broader, more implicit knowledge and works well when you have the data and the infrastructure. Dialogic Distillation is better when you need targeted, auditable corrections, when fine-tuning is impractical (competition constraints, no GPU, unknown model in advance), or when you need to be able to inspect and govern what the model has learned.

---

## Reference Domains

### Image Classification: Dermatology and Ornithology

The image classification use case is the original home of Dialogic Distillation. A cheap vision-language model (Qwen3-VL-8B) fails on specific confusable class pairs — melanoma vs. benign mole, basal cell carcinoma vs. benign keratosis, Brown-headed Cowbird vs. similar species. The expert model (Claude) analyzes each failure, proposes a discriminative visual rule, and KF validates it through the grounding check and pool gate before registering it.

**Results** (research ongoing — test coverage is being expanded and the distillation mechanism continues to be refined):

| Domain | Pair | Scale | Zero-shot | With distilled rules | Gain |
|---|---|---|---|---|---|
| Dermatology | Melanoma vs. Nevus | 30/class | 55% | 93.3% | +38.3 pp |
| Dermatology | Melanoma vs. Nevus | 100/class | 52.5% | 81.0% | +28.5 pp |
| Dermatology | BCC vs. Benign Keratosis | 30/class | 56.7% | 75% | +18.3 pp |
| Ornithology | Cowbird (pilot) | 6 images | 33% | 83% | +50 pp |
| Ornithology | Cowbird (expanded) | 30/class | 46.7% | 96.7% | +50 pp |

The 100/class result comes from a controlled experiment (200 images, 10 trigger failures, held-out pool of 10/class, seed-controlled splits). At this larger scale the gain is smaller than at 30/class — expected, since the additional images include harder cases — but still substantial.

#### Why multi-round dialogue matters — and when single-shot catches up

The initial three-party experiment demonstrated why multi-round dialogue is necessary: **naive single-shot elicitation produced 0 out of 7 grounded rules** across two separate attempts (`elicit_from_failures.py`: 0/4; `elicit_nodular_rules.py`: 0/3). The dialogic loop produced 4/4 grounded rules, 3 of which passed the pool gate. The failure mode is vocabulary mismatch — the TUTOR writes rules using clinical dermoscopy vocabulary; the VALIDATOR describes the same features using colors, shapes, and spatial arrangements (see [DESIGN.md](DESIGN.md) §3).

However, in the controlled 200-image experiment, **both methods achieved identical conversion rates**: 10/10 grounded, 7/10 accepted. The reason: by this point, KF had already incorporated the meta-learnings from the earlier dialogic rounds into the single-shot prompt (see [Meta-Learning](#meta-learning-kf-learns-to-steer-better) below). The TUTOR prompt already framed the task for the validator's vocabulary, included GOOD/BAD vocabulary examples, and limited precondition count — all insights that were originally discovered through multi-round dialogue.

This reveals that **dialogic distillation's value has two dimensions**:

1. **Per-rule refinement** — multi-round dialogue bridges the vocabulary gap on individual rules that fail grounding on the first attempt
2. **Meta-learning** — KF learns behavioral patterns of the TUTOR and VALIDATOR through the dialogue, and encodes those patterns into improved prompts for future rounds

Once meta-learning saturates (KF's prompts are well-optimized for the TUTOR/VALIDATOR pair), single-shot can catch up — but only because dialogue happened first. Without the dialogue, the meta-learnings would never have been discovered.

#### Meta-learning: KF learns to steer better

KF's role is not just to relay messages — it adapts its prompts based on what works and what fails. Three concrete meta-learnings emerged from the dermatology dialogic experiments:

**1. Frame the task for the validator, not the clinician.** The naive TUTOR prompt (used in `elicit_from_failures.py`) asked for "dermoscopic features." The TUTOR naturally wrote rules in clinical vocabulary. After observing repeated grounding failures, KF's optimized prompt (`distill_dialogic.py: TUTOR_SYSTEM`) explicitly states: *"The rules you write will be tested by a separate validator model that describes images in terms of visible colors, shapes, edges, and spatial patterns — not clinical terminology."*

**2. Provide GOOD/BAD vocabulary examples.** The optimized Round 1 prompt includes concrete examples of vocabulary that works vs. vocabulary that fails:
> *GOOD: "dark brown or black irregularly shaped patches"*
> *BAD: "atypical pigment network" (clinical term the validator won't use)*

These examples were distilled from actual grounding-check failures in earlier dialogic rounds.

**3. Limit precondition count to 2–3.** Early TUTOR proposals often included 5–7 preconditions, making it unlikely that the validator would confirm all of them on any single image. KF learned to instruct the TUTOR: *"Write 2–3 preconditions maximum, each visually concrete."*

**Effect of meta-learning**: naive prompts achieved 0% grounding rate (0/7 rules grounded). KF-optimized prompts achieved 100% grounding rate (10/10 rules grounded) — even on the first attempt, without any dialogue rounds.

The knowledge in this domain is **visual discrimination rules**: when you see feature X combined with feature Y, classify as class A rather than class B. Rules are concrete, checkable against images, and narrow enough to be validated on a held-out pool.

See [usecases/image-classification/README.md](../image-classification/README.md) for the full walkthrough.

### ARC-AGI-3: Competition Mode

ARC-AGI-3 is an interactive reasoning benchmark where an agent must explore an unknown game environment, discover its hidden mechanics, and navigate to a win condition through sequential action and observation. In non-competition mode, the system uses Claude Sonnet 4.6 — a highly capable model that can reason about novel environments on the fly. In competition mode, the system must switch to a smaller open-source model that fits within budget constraints — and that model struggles with exactly the reasoning patterns the larger model handles naturally.

This is a direct parallel to the image classification case: a capable TUTOR (Sonnet 4.6) and a constrained PUPIL (open-source model), with a domain-specific failure mode that needs corrective rules.

The knowledge in this domain is different in character. ARC-AGI-3 rules are not visual discrimination cues — they are **strategic and procedural heuristics**: how to observe a new game systematically, how to form hypotheses about object roles from behavioral evidence, how to prioritize goals when resources are running low. A weaker model skips steps in this reasoning chain that a stronger model applies naturally.

Three categories of distillable knowledge have been identified:

- **Concept identification heuristics**: how to recognize the player piece, wall colors, and interactive objects from behavioral evidence (which things move, which things block)
- **Goal recognition patterns**: how to detect depletion counters, urgency signals, and win-condition proximity from observed state changes
- **Strategic meta-rules**: the discipline of systematic exploration before exploitation — cataloguing all distinct objects, testing each action in isolation, forming a goal hypothesis only after enough evidence is gathered

These heuristics transfer across games (game-agnostic), which is exactly what competition mode requires: the game identity is unknown in advance.

Sequential-reasoning work has moved to the new cognitive engine, now in
its own repository:
[github.com/khub-ai/cognitive-os-engine](https://github.com/khub-ai/cognitive-os-engine).

---

## What Makes This Novel

No published paper at the time of writing combines all of these properties in a single system:

1. **Failure-triggered** — rules are authored in response to specific failures, not proactively
2. **Multi-round dialogue for grounding** — the orchestrator steers TUTOR refinement until the rule's vocabulary matches what the validator actually observes
3. **Orchestrator-gated pool validation** — rules must pass a held-out precision check before registration, not just a single-example test
4. **Inference-time injection** — accepted rules are injected into the PUPIL's context; the PUPIL's weights are never modified
5. **Explicit, auditable artifacts** — every rule is plain text with provenance: which failure triggered it, who authored it, what the pool validation showed
6. **Meta-learning at the orchestrator level** — KF improves its own steering prompts based on patterns observed across dialogue rounds, compounding gains across sessions

The closest published work:
- **Inter-Cascade** (Wu et al., 2025): failure-trigger → strategy artifact → inference-time injection, but single-shot (no grounding dialogue)
- **GER** (Wang & Sudhir, 2024): reusable textual guidance library at inference time, but not failure-triggered
- **ExpeL** (Zhao et al., AAAI 2024): rules extracted from failure/success contrastive analysis, but no pre-registration validation
- **ProTeGi** (Pryzant et al., EMNLP 2023): held-out pool validation before committing a rule, but prompt-level (not a separate artifact for a weaker model)

---

## Scope and Limits of DD

DD works well for two specific problems:

**1. Correcting perceptual errors.** The model sees an image but misidentifies
it because it doesn't know what to look for. Once told — "early wildfire smoke
looks blue-white, not grey; it has a point source, not a diffuse edge" — the
model applies that knowledge immediately and correctly. The PatchBench results
for wildfire detection show a +29 percentage-point improvement in a single
session (66.7% → 95.8%, GO verdict).

**2. Encoding domain knowledge the model lacks.** The model is not wrong about
what it sees — it simply has no framework for what the features mean. Injecting
published expert guidelines (NWCG fire-spotting rules, IAMSAR SAR search
criteria) achieves the same lift as model-generated rules. This means DD can
work even without a TUTOR model, as long as the domain has written expert
guidance.

**What DD cannot do: override a model's trained values.**

During the SeaPatch maritime SAR experiments, we discovered a harder limit. Some
of the model's errors were not perceptual mistakes — they were the model
correctly following its training. Specifically, both `claude-sonnet-4-6` and
`claude-opus-4-6` persistently called orange floating objects "person in water,"
even when the ground truth said otherwise. We then tested whether this could be
changed by instruction.

We injected two opposite directives:
- *"In SAR operations, orange on water means a person — classify as person-in-water."*
- *"Default to whitecap. Orange equipment alone is not sufficient. When in doubt, say whitecap."*

Neither changed the model's behaviour on those frames. The same false alarms
appeared in every run, regardless of which instruction was active.

**Why:** Claude models (and most commercially deployed AI models) are shaped
during training by human raters who reward cautious, safety-oriented behaviour.
"Orange thing on water = possible person in danger" becomes a deep belief, not
just a rule. Natural-language instructions reach the part of the model that
follows directions. They do not reach the part that holds trained values. When
the two conflict, the values win.

More capable models hold these values more firmly, not less. Opus had more
immovable false alarms than Sonnet, not fewer.

**The practical implication:** If your deployment needs the model to behave in a
way that conflicts with its safety training — for example, being less sensitive
to potential victims in a harbour full of floating equipment — instructions alone
will not get you there with a heavily-conditioned model. We tested this with
`qwen/qwen3-vl-8b-instruct`, an open-source model with less human-feedback
conditioning. The result was striking: Qwen followed the conservative directive
completely, calling every person-in-water frame "whitecap" (50% accuracy, 12
missed victims). It is fully controllable in either direction — which is
powerful but puts the entire responsibility on the instruction author. With a
well-crafted model-rules directive, Qwen achieved 95.8% accuracy (GO verdict),
the best result of any model tested. The options for deployment are therefore:
(a) use a less-conditioned model and invest in instruction quality; (b) add a
pre-classification filter that handles the ambiguous object type before the VLM
sees the frame; or (c) treat the safety prior as a feature and design around it.

**This finding is only visible through the dialogic method.** Standard
benchmarking tests accuracy in one direction. The immovability of the prior only
becomes apparent when you test opposing instructions and compare results — which
is what the dialogic approach naturally does.

Full data and analysis: [SeaPatch README §11](../ai-fleets/seapatch/README.md#11-patchbench-probe-model-selection-for-seapatch-dd).

---

## Status

| Sub-domain | Status |
|---|---|
| Image classification — dermatology | 🔬 Results at 30/class (+38pp) and 100/class (+28.5pp); meta-learning findings documented; expanding coverage |
| Image classification — ornithology | 🔬 Results at 30/class (+50pp); expanding coverage and refining protocol |
| ARC-AGI-3 competition mode | Planned |

---

## Try It on Your PUPIL Model

The dermatology and ornithology benchmarks are designed to be reproducible with any vision-capable model as the PUPIL. If you run the benchmark against a model we haven't tested, we'd like to know the results.

**What to do:**
1. Pick a PUPIL model — any vision-capable model smaller or cheaper than Claude Sonnet.
2. Run the baseline evaluation (zero-shot, no injected rules) on the held-out image pool.
3. Run the dialogic distillation loop (TUTOR = Claude Sonnet 4.6, PUPIL = your model).
4. Re-evaluate the PUPIL on the same pool with the accepted rules injected.
5. Report the before/after accuracy and which PUPIL model you used.

See [usecases/image-classification/README.md](../image-classification/README.md) for the evaluation harness and step-by-step instructions.

**Report results:** Open an issue at [khub-knowledge-fabric](https://github.com/kaihuchen/khub-knowledge-fabric/issues) with the tag `benchmark-result`. Include your PUPIL model name, domain (dermatology / ornithology / other), dataset size, baseline accuracy, and post-distillation accuracy.

Results from the community will be aggregated into the Status table above.

---

## Navigation

- [DESIGN.md](DESIGN.md) — developer-facing spec: three-party protocol, grounding mechanics, domain adaptations, differences across knowledge types
- [usecases/image-classification/](../image-classification/) — reference implementation for visual classification
- [cognitive-os-engine](https://github.com/khub-ai/cognitive-os-engine) — domain-agnostic cognitive engine (successor to the retired arc-agi-3 use case), extracted to its own MIT-0 repo
