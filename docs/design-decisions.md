# Design Decisions

This document explains the key design choices behind the project, how it differs from existing approaches, the forward-compatibility strategy, and the known limitations of the current implementation.

## How this differs from existing agent memory

Most AI assistants in 2026 offer some form of persistent memory. Here is how this project differs:

| Feature | Context-window memory | Platform memory (e.g., ChatGPT Memory, Claude Projects) | Agent framework memory (e.g., Letta) | This project (PIL) |
|---|---|---|---|---|
| **Storage location** | Server-side | Server-side | Server-side (Letta server / database) | Local, user-owned |
| **User can inspect** | Limited | Partial (some platforms show stored memories) | Partial — via Letta API or UI | Full — artifacts are text files |
| **User can edit** | No | Limited | Limited — via Letta API only | Yes — any text editor |
| **User can delete** | Session reset only | Per-item deletion in some platforms | Yes — via API | Full control (file system) |
| **Portable** | No | No | No — Letta-specific database format | Yes — model-agnostic text |
| **Structured** | No (raw conversation) | Minimal (key-value or short statements) | Yes — typed memory blocks | Yes — typed, versioned, with provenance |
| **Generalized** | No | Rarely | No — stores observations; no systematic distillation into rules | Yes — the pipeline explicitly generalizes observations into rules |
| **Governable** | No | Platform-dependent | Partial — operator-controlled; no per-artifact governance | Yes — user owns the files, controls what persists |

The key difference is not just *where* the knowledge is stored, but *what form* it takes. Platform memory features typically store brief statements extracted from conversation ("user prefers dark mode"). Letta provides a richer memory architecture for developers building agents, but memory remains server-side and in a framework-specific format. PIL produces structured, typed, confidence-scored, versioned [knowledge artifacts](glossary.md#knowledge-artifact) that the user can reason about, edit, and share — and that outlive any particular platform or framework.

## PIL as a complement to fine-tuning

When an AI agent needs to work effectively in a specific domain, the standard tool is fine-tuning: training the base model on domain-specific data to adjust its weights. Fine-tuning is well-suited to *capability expansion* — teaching a new technical vocabulary, a new code style, a specialized subject domain. But for a different and increasingly important class of adaptation — adjusting an agent's *behavior to fit a specific user, team, or operational context* — fine-tuning is an ill-matched tool.

**The limitations of fine-tuning for behavioral adaptation:**

- **High cost per iteration.** Each behavioral correction requires data curation, a training run, regression evaluation, and a new deployment. This cycle takes days to weeks and is economically impractical for individual users or small teams.
- **Catastrophic forgetting.** Reinforcing one behavior can silently degrade others. Correcting a specific pattern risks breaking adjacent behaviors in unpredictable ways.
- **Opacity.** Fine-tuned weights carry no provenance. There is no record of which training examples caused which behavioral changes; debugging requires behavioral probing rather than direct inspection.
- **Static knowledge.** Behavior baked into model weights is frozen at training time. When a user's conventions change or a team adopts a new procedure, retraining is the only recourse.
- **Population-level, not individual-level.** Fine-tuning optimizes for a dataset — a domain archetype, a use-case category, a population of examples. It cannot practically adapt to the preferences, conventions, and judgment criteria of a specific individual.
- **Irreversibility.** Removing a specific learned behavior ("unlearning") without collateral damage is an unsolved problem. A wrong convention, once trained in, is difficult to excise cleanly.
- **No runtime feedback loop.** The model cannot update from deployment-time corrections; every real-world signal must wait for a new training cycle.

**What PIL covers instead:**

PIL does not change model weights and cannot expand model capabilities — it is not a replacement for fine-tuning. What it provides is a complementary mechanism for behavioral adaptation at the individual level, which fine-tuning cannot address practically:

- **Incremental.** Every interaction is a potential learning event; no batch training required.
- **Inspectable.** Knowledge lives as human-readable artifacts. A user can read, edit, or delete exactly what the agent has learned — impossible with weight-based learning.
- **Correctable in real time.** A wrong artifact can be revised or retired immediately, with no risk to any other learned behavior.
- **Individual-specific.** Learning accumulates for one person's specific preferences, conventions, and judgment — not a population average.
- **Immediately effective.** Corrections take effect in the next interaction, not after a training cycle.

**An analogy:**

If pretraining is the equivalent of a broad university education — producing a knowledgeable, capable generalist — then fine-tuning is a formal professional certification: structured, costly, and designed to bring a cohort to domain proficiency.

PIL is something qualitatively different: it is the equivalent of on-the-job learning from a specific person over the course of a working relationship. A new employee arrives with their education and certifications intact. What they don't yet know is how *this team*, *this organization*, *this particular person* does things — the unwritten conventions, the judgment calls that experienced colleagues make intuitively, the preferences that appear in no formal document. This knowledge is not taught in classrooms; it accumulates through daily interaction, observation, and correction. It is personal, contextual, and always current. And unlike a formal credential, it can be directly discussed, revised, and handed off when circumstances change.

**Fine-tuning adapts a model for a domain. PIL adapts an agent for a person.**

## Language support and interoperability

The current pipeline implementation is TypeScript-only. This section describes what non-TypeScript callers can do today and the intended path to Python support.

**PIL is a library, not a service.** The local-first architecture means the agent IS the process — it calls PIL functions the way it calls any library. There is no separate PIL daemon. A Python agent should use PIL the same way the TypeScript agent does: as a native library running inside the same process. This shapes which Python interop paths make sense.

**What non-TypeScript callers can do today:**

- **Read and write the artifact store directly.** Artifacts are stored as JSONL at `~/.openclaw/knowledge/artifacts.jsonl` — one JSON object per line, with a fully documented schema in [`docs/architecture.md`](architecture.md#knowledge-artifact-schema). Python, Ruby, Go, or any language that handles JSON can read, filter, and write artifacts without going through the TypeScript pipeline. This enables use cases like artifact inspection tooling, migration scripts, custom reporting, and artifact injection from external systems.

**What requires TypeScript today:**

Calling `processMessage`, `retrieve`, `apply`, and `revise` as library functions requires a TypeScript or JavaScript runtime. There is no Python package.

**Planned: Python library**

The correct path to Python support is a native Python port of the PIL pipeline — not a REST API. A REST API would impose a service model (a daemon that must be running before any agent can operate) that contradicts the local-first, zero-infrastructure-overhead design. The TypeScript reference implementation is the specification; the Python port would implement the same pipeline against the same JSON store.

A Python port is architecturally straightforward: the LLM adapter is a plain callable `(prompt: str) -> str`, the store is plain JSON, and the pipeline has no framework dependencies. It is planned for after the artifact format stabilizes through Phase 5 (Portability), when the schema is stable enough that a Python implementation can reasonably commit to full compatibility.

**Note on subprocess invocation:** Calling the TypeScript CLI from Python via subprocess is technically possible but is an integration workaround, not an architecture. It requires a Node.js installation, adds process-management complexity, and disappears when the Python library exists.

→ *[FAQ: Is a Python API available?](faq.md#is-a-python-api-available-for-khub-pil)*

## Artifact format: free-form text, not rigid schemas

[Knowledge artifacts](glossary.md#knowledge-artifact) are primarily **free-form text with lightweight conventions** — not rigid database schemas. This is a deliberate choice with several motivations:

- **Human-readable**: A user can open an artifact file and understand it immediately without specialized tooling.
- **Model-agnostic**: Any sufficiently capable LLM can consume the artifacts, regardless of vendor. There are no embeddings, token IDs, or model-specific representations.
- **Forward-compatible**: As the system evolves and new knowledge types emerge, existing artifacts don't require migration. The format accommodates what we haven't anticipated.
- **Editable**: Users can modify artifacts with any text editor, version-control them with git, diff them, and merge them.

The "schema" is a set of conventions — kind, confidence, provenance, timestamps — rather than a strict type system. This keeps artifacts open-ended enough to support diverse knowledge types while providing enough structure for the pipeline to operate on them programmatically.

## Forward-compatibility principles

The architecture is designed to evolve without invalidating existing artifacts. Three rules ensure this:

### 1. Core + optional enrichment

The core of an artifact is always `kind` + `content` + `confidence`. Everything else — tags, trigger conditions, graph relationships, lifecycle tracking — is optional enrichment metadata. An artifact without enrichments still works: it can be stored, retrieved (by content search), applied, and revised.

### 2. Additive-only field evolution

New fields are always optional, never required. The pipeline never refuses to process an artifact because it's missing metadata. This means:
- Old artifacts created before new fields were introduced remain valid
- A human can create an artifact by writing minimal JSON (kind + content + confidence) and it works
- The system gracefully degrades: an artifact without tags gets slower retrieval (content search instead of index lookup) but still functions

### 3. Text as the durable layer

Even if the storage backend changes (JSONL → SQLite → graph database), even if the triggering system is redesigned (keyword index → vector search → neural retrieval), even if the graph structure is reorganized — the artifacts themselves are text. They can always be read, understood, and consumed by any system that processes text. This is the deepest form of forward-compatibility: the knowledge outlives any particular implementation.

## Model-agnostic: what it means precisely

"Model-agnostic" applies to two different layers:

1. **The artifacts are fully model-agnostic.** They are free-form text. Any LLM that can read and reason over text can consume them. There are no vendor-specific embeddings, fine-tuning artifacts, or proprietary formats. A user can export artifacts from an OpenClaw instance running Claude and import them into one running GPT, Gemini, or an open-source model.

2. **The PIL pipeline requires a capable host.** The code that elicits, induces, validates, and applies knowledge relies on the host agent having sufficient interactive learning and reasoning ability. In practice, this means leading LLMs from major vendors work well, but smaller or older models may not have the reasoning capacity to perform effective induction or generalization.

This distinction is important: the knowledge is portable even if the pipeline that produced it is not universally deployable.

## Language-agnostic pattern extraction

The pipeline contains no English-specific heuristics. This is a deliberate design constraint, not an implementation detail.

Earlier approaches to knowledge extraction relied on signal words (`always`, `never`, `prefer`), hedge words (`maybe`, `might`, `possibly`), and regex patterns to classify input. These work only for English and fail entirely for Chinese, Farsi, Spanish, or any other language. They also fail for code-heavy input where the meaningful signal is in a comment or identifier, not in surrounding prose.

The solution is to delegate all natural language understanding to the LLM, which handles semantics across languages natively. The pipeline's job is to invoke the LLM with a well-structured prompt and act on its structured output — not to do linguistics itself.

This means:
- A user interacting in any language gets identical behavior
- The pipeline has no hardcoded vocabulary to maintain
- Classification quality scales with the LLM's capability, not with the quality of hand-crafted rules

The cost is one LLM call per message (or per conversation batch). This is already unavoidable at Milestone 1b and beyond; the language-agnostic design does not add LLM calls — it just moves the understanding work into the call that was always going to happen.

## Confidence gating: approach and evolution

Confidence is seeded at extraction time from the `certainty` field assigned by the LLM:

| `certainty` | Starting `confidence` | When to expect it |
|---|---|---|
| `"definitive"` | 0.65 | Strong, unhedged statement: "always use strict mode" |
| `"tentative"` | 0.35 | Qualified statement: "I usually prefer bullet points" |
| `"uncertain"` | 0.15 | Speculative: "I think I prefer shorter summaries?" |

This replaces the former English-specific hedge/assertion word heuristics entirely. The LLM infers the degree of certainty from the phrasing and intent, regardless of language.

Confidence then grows through evidence accumulation (each supporting observation nudges it upward) and is adjusted at application time by feedback signals (acceptance rate, decay, reinforcement). See [architecture.md](architecture.md#effective-confidence-calculation) for the full formula.

The confidence-gating mechanism itself — suggest below threshold, auto-apply above — is designed to be pluggable. The threshold, the scoring method, and the feedback loop can all evolve without changing the artifact format or storage model.

## Cognitive design cross-check

The architecture is informed by a systematic cross-check against human cognitive mechanisms. Key findings:

| Human mechanism | System analogue | Status |
|---|---|---|
| Memory consolidation | Evidence accumulates in `candidate`/`accumulating` artifacts → LLM consolidation call → `consolidated` generalization | ✅ Implemented (Phase 1b — via `stage`, `evidenceCount`, `evidence[]`, consolidation LLM call) |
| Forgetting / decay | Effective confidence decreases for unretrieved, unreinforced artifacts; consolidated artifacts decay to a floor, not zero | ✅ Implemented (Phase 2a — `effectiveConfidence()` with validation-modulated half-life and decay floor; stored `confidence` unchanged) |
| Emotional salience | `salience` field adjusts auto-apply threshold *and* decay floor independent of confidence | ✅ Implemented (Phase 1 threshold adjustment + Phase 2a floor multiplier: `high` × 1.25, `low` × 0.75) |
| Spreading activation | Graph traversal during retrieval surfaces connected artifacts | Designed (via `relations`); not yet implemented |
| Meta-cognition | Query layer over store for knowledge coverage awareness | Future work; no artifact changes needed |
| Habituation | Feedback tracking reduces influence of repeatedly rejected artifacts | ✅ Partially implemented (Phase 2a — `rejectedCount` lowers `validationStrength`, shortening half-life and lowering decay floor; fully suppressing persistent rejections is future work) |
| Context-dependent retrieval | `trigger` field expresses natural language conditions for when artifacts apply | Designed; not yet implemented |

For full details, see [Memory Taxonomy — Lessons from human cognition](memory-taxonomy.md#lessons-from-human-cognition).

## Security and governance considerations

Security deserves more than a brief note. PIL adds new attack surfaces on top of OpenClaw, which itself connects to 24+ external platforms and has known security concerns. The key risks are:

- **External sender injection**: PIL's `message_received` hook fires on all inbound messages, including those from external parties on connected platforms. Without sender verification, a malicious external message could trigger extraction and persist adversarial content.
- **Prompt injection via artifact injection**: Artifact `content` is injected directly into LLM prompts. Adversarial content in an artifact can manipulate LLM behavior for that session.
- **Sensitive data capture**: The extraction LLM may inadvertently capture and persist PII, credentials, or confidential information from conversation.
- **Cross-channel leakage**: Knowledge learned in a private channel may be injected in a public or professional one.
- **Knowledge poisoning via import**: Phase 4+ imports are a high-risk surface; a malicious package can introduce many adversarial artifacts at once.
- **Local file exposure**: The JSONL store is plaintext; anyone with filesystem access can read or modify it without going through the pipeline.

→ *[Full security threat model with mitigations by phase](security.md)*

Governance considerations for enterprise deployment (access controls, audit trails, org-level knowledge tiers) are covered in [enterprise-vision.md](enterprise-vision.md).

## Why an OpenClaw extension

This project is built as an extension for [OpenClaw](https://github.com/openclaw/openclaw) for several reasons:

- **Plugin architecture**: OpenClaw has a mature plugin SDK that allows extensions to register tools, CLI commands, and hooks without forking upstream.
- **Hook-based integration**: OpenClaw's `message_received` and `before_prompt_build` hooks enable passive observation and knowledge injection — the two capabilities PIL needs to operate within conversation flow.
- **Multi-channel**: OpenClaw connects to 24+ messaging platforms, which means knowledge learned in one channel is available across all channels.
- **Local-first**: OpenClaw runs on the user's own machine, which aligns with the project's principle that knowledge should be user-owned and local.
- **Additive**: Users can install and upgrade OpenClaw normally. This extension adds capabilities without modifying the core.
- **Open-source**: OpenClaw's open-source codebase gives this project full freedom to experiment — modifying hook behavior, inspecting internals, testing non-obvious integration patterns — without the constraints that proprietary platforms impose on what third-party extensions can observe or do. The surrounding developer ecosystem also means tooling, libraries, and prior art are openly available and rapidly iterated on by the community.

The knowledge artifact format itself is not OpenClaw-specific. The PIL pipeline is implemented as an OpenClaw plugin for convenience, but the artifacts it produces are portable to any system that can read text files.
