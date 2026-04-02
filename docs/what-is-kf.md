# What Knowledge Fabric Is

Knowledge Fabric (KF) is a **runtime knowledge layer** for AI systems.

In practical terms, KF is a **knowledge middleware layer** that wraps one or
more LLMs, VLMs, or multimodal models and gives the overall system capabilities
that the underlying models do not natively provide on their own.

KF is not itself a foundation model. It sits **around** one or more models and
helps them learn, store, retrieve, apply, revise, and govern reusable
knowledge over time.

## KF In One Sentence

**KF turns useful knowledge into persistent, human-readable artifacts that can
improve an AI system immediately, without retraining the underlying model.**

## What KF Adds

Compared with using a base model alone, KF can provide:

- **Runtime learning**: the system can improve during use rather than only
  through retraining.
- **Knowledge persistence outside the model**: what was learned survives across
  sessions, deployments, and even model changes.
- **Human-readable knowledge artifacts**: learned knowledge stays explicit,
  inspectable, editable, and auditable.
- **Dialogic learning**: the system can work interactively with a user or
  domain expert in natural language to extract and verify useful knowledge.
- **Controlled application of knowledge**: KF decides when knowledge should be
  applied, withheld, revised, or retired.
- **Portability across models and vendors**: the knowledge layer is not locked
  into one model provider.
- **Incremental patching**: specific failure modes can be corrected one patch
  at a time without waiting for a full model-release cycle.
- **Tool generation for precise repetitive work**: once a procedure becomes
  clear enough, KF can help compile it into a more reliable executable tool.
- **Multi-agent internal structure**: KF can use specialized internal roles for
  observation, reasoning, verification, planning, and revision instead of
  relying on a single opaque model pass.

## How KF Works

At a high level, KF follows this loop:

1. **Capture** useful knowledge from interaction, examples, or expert input.
2. **Structure** that knowledge into reusable artifacts such as rules,
   procedures, judgments, strategies, or boundaries.
3. **Persist** those artifacts outside the model.
4. **Retrieve and apply** the right artifacts when they are relevant.
5. **Revise** them when experience, feedback, or new evidence shows they should
   change.

The important point is that the knowledge does not disappear into model
weights. It remains available as an explicit layer the system can reason about
and the user can inspect.

## How KF Differs From Adjacent Approaches

KF overlaps with several familiar AI patterns, but it is not the same as any
one of them:

- **Prompt orchestration** helps control a model call; KF adds a persistent
  knowledge layer that survives beyond one prompt.
- **RAG** retrieves documents or notes; KF stores and applies distilled,
  task-shaped knowledge artifacts rather than only raw source text.
- **Platform memory** may remember facts or preferences; KF keeps knowledge
  explicit, portable, and user-governable.
- **Fine-tuning** changes model weights; KF improves behavior by changing the
  external knowledge layer instead.
- **Workflow middleware** routes tools and tasks; KF focuses on learning,
  reusing, and governing knowledge itself.

## KF, PIL, And Knowledge Artifacts

These terms are related but not identical:

- **Knowledge Fabric (KF)** is the broader system or runtime layer.
- **PIL (Persistable Interactive Learning)** is the core learning pattern used
  inside KF.
- A **knowledge artifact** is the stored unit of reusable knowledge produced
  and managed by KF.

So a good mental model is:

**KF is the overall runtime knowledge layer, PIL is its core learning
mechanism, and knowledge artifacts are the units of knowledge it stores and
applies.**

## What KF Is Good For

KF is most valuable when:

- important knowledge lives in experts' heads or repeated user corrections
- the system must improve after deployment
- the knowledge should remain reviewable and governable
- retraining is too slow, expensive, opaque, or narrowly scoped
- accumulated knowledge should survive model upgrades or vendor changes

That is why the repository explores domains such as assistant personalization,
image classification, cybersecurity, robotics, and long-horizon reasoning.

## Where To Go Next

- [README.md](../README.md): repo overview and current implementation status
- [glossary.md](glossary.md): canonical definitions for repository terms
- [design-decisions.md](design-decisions.md): why KF differs from existing
  agent memory and adjacent approaches
- [ensemble-pipeline.md](ensemble-pipeline.md): the current multi-round KF
  pipeline used in active use cases
