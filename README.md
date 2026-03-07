# OpenClaw Knowledge Management

**OpenClaw Knowledge Management** is an extension for [OpenClaw](https://github.com/openclaw/openclaw) that gives AI agents the ability to **learn from interaction, persist what they learn, and reuse it reliably** — all as **user-owned, local, portable knowledge artifacts** rather than opaque server-side state.

## Who this is for

**Individual users and knowledge workers** — if you use an AI agent daily and find yourself re-explaining your preferences, correcting the same mistakes, or re-teaching your workflow every session, this is for you. PIL makes the agent accumulate what it learns about you across sessions — your communication style, your conventions, your judgment criteria — stored as files on your own machine that you can inspect, edit, and take with you if you switch platforms.

**Enterprise AI adopters** — if you are evaluating AI agents for professional or organizational use, the hardest question is governance: what does the agent know, who approved it, and what happens when the expert who built up that knowledge leaves? PIL addresses this with a structured, auditable artifact model where knowledge is captured explicitly, versioned, and inheritable — not locked inside a model or a vendor's server.

**AI investors and ecosystem builders** — if you are tracking where durable value accumulates in the AI stack, PIL proposes a new asset class: portable, typed, user-owned knowledge artifacts. The artifact format, if it achieves adoption, defines a coordination layer — analogous to what npm did for packages or OpenAPI did for APIs — around which expert knowledge marketplaces, org custody services, and certification businesses can form.

**AI platforms and potential partners** — PIL is built as an OpenClaw extension, but the artifact format is platform-agnostic. A platform that supports PIL artifact import/export gains interoperability with an emerging knowledge ecosystem and a credible story for user data portability and governance.

→ *[Detailed enterprise and investment case](docs/enterprise-vision.md)* · *[Security threat model](docs/security.md)*

## Why this exists

Modern AI agents appear to learn from interaction, but the knowledge they accumulate — whether through context windows, compacted session memory, or fine-tuning — remains **server-side, opaque, and platform-locked**. The user cannot easily inspect, edit, govern, or move it. If the user switches platforms, the knowledge disappears.

This project takes a different approach: knowledge is extracted from interaction, stored locally as human-readable text, and owned entirely by the user. The artifacts are **model-agnostic** — any sufficiently capable LLM can consume them. They can be edited with a text editor, version-controlled with git, shared with colleagues, or imported into a different AI assistant.

→ *[How this differs from existing agent memory](docs/design-decisions.md#how-this-differs-from-existing-agent-memory)*

## What the agent learns

The project is built around **four types of knowledge**, with emphasis on their generalized forms:

| Memory type | What it captures | Generalized into |
|---|---|---|
| **Episodic** | What happened in conversation | Raw material for generalization of other types |
| **Semantic** | Facts, preferences, conventions | General rules applicable across contexts |
| **Procedural** | How to do things | Structured recipes, optionally executable programs |
| **Evaluative** | What counts as "good" | Judgment heuristics and value frameworks |

**Generalization is the key.** We don't just record that the user corrected a summary format five times — we distill it into a rule ("always use bullet points") that the agent applies proactively in new situations. For procedures, a structured recipe can optionally be compiled into a deterministic program for tasks that require perfect repeatability. For evaluative knowledge, patterns of preference become judgment frameworks the agent uses to make good choices in novel situations — the closest analogue to what we colloquially call "wisdom."

→ *[Full memory taxonomy and evaluative knowledge](docs/memory-taxonomy.md)*

## See it in action

A condensed example of how the agent learns progressively over multiple sessions:

> **Session 2** — Agent notices the user always names financial statements as `YYYY-MM-institution-account.pdf`. It proposes a convention. User confirms. → *Semantic artifact created.*
>
> **Session 3** — Agent learns the user downloads from Chase, Fidelity, and Amex monthly. It proposes a checklist procedure. → *Procedural artifact created.*
>
> **Session 4** — After three identical runs, the agent offers to compile the checklist into a script. → *Recipe retained; executable program linked as optional optimised form.*
>
> **Session 5** — User closes an account and opens a new one. Agent revises the institution list, the procedure, and the script together. → *Coherent revision across artifact types.*

→ *[Full example with dialogue](docs/example-learning-in-action.md)*

## Core idea: Persistable Interactive Learning (PIL)

The system treats dialogue as a learning substrate and produces durable knowledge artifacts through an 8-stage pipeline:

1. **Elicit** — surface candidate knowledge from interaction
2. **Induce** — classify into a typed artifact (semantic, procedural, evaluative, etc.)
3. **Validate** — estimate confidence and scope
4. **Compact** — compress into minimal form; deduplicate
5. **Persist** — store locally with provenance and versioning
6. **Retrieve** — recall by relevance and context
7. **Apply** — confidence-gated: suggest, auto-apply, or hold back
8. **Revise** — update or retire when contradicted or outdated

## Design goals

- **Extension for OpenClaw** — additive layer; users install/upgrade OpenClaw normally
- **User-owned and local** — knowledge lives on your machine, not a vendor's server
- **Model-agnostic portability** — artifacts are text; any capable LLM can consume them
- **Confidence-gated reuse** — learned knowledge is *suggested*, *auto-applied*, or *held back* based on certainty
- **Free-form artifacts** — lightweight conventions, not rigid schemas; human-readable and editable
- **Versioned and auditable** — changes are tracked; revisions are first-class

→ *[Design decisions and rationale](docs/design-decisions.md)* · *[Architecture (tiered triggering, knowledge graph, artifact schema)](docs/architecture.md)* · *[Security threat model](docs/security.md)*

## Roadmap

### Near-term (Phase 1 milestones)

| Milestone | What it delivers | Status |
|---|---|---|
| **1a — Scaffolding** | Pipeline architecture with placeholder heuristics, playground | ✅ Done |
| **1b — Explicit "remember"** | User says "remember this" → LLM-backed artifact creation → retrieval in future sessions | **Next** |
| **1c — Passive elicitation** | Agent observes conversation via hooks and proposes knowledge without explicit instruction | Planned |
| **1d — Tier 1 triggering** | Keyword index enables zero-cost retrieval on every message | Planned |

### Long-term (Phases 2–5)

| Phase | Focus | Status |
|---|---|---|
| **2 — Generalization Engine** | Episodic → semantic/evaluative generalization, Tier 2 triggering, decay, feedback | Planned |
| **3 — Procedural Memory & Code Synthesis** | Structured recipes, optional program compilation, tool library | Planned |
| **4 — Portability** | Standard artifact format, import/export, cross-agent compatibility | Planned |
| **5 — Governance & Ecosystem** | Team/org knowledge tiers, access controls, compliance audit trails, knowledge ecosystem and new business models | Long-term |

→ *[Detailed roadmap with milestones](docs/roadmap.md)* · *[Enterprise vision and investment thesis](docs/enterprise-vision.md)*

## Repository structure

```
openclaw-knowledge-management/
├── packages/
│   ├── openclaw-plus/          # Core PIL extension (OpenClaw plugin)
│   │   ├── index.ts            # Plugin entry point
│   │   ├── openclaw.plugin.json
│   │   └── src/
│   │       ├── pipeline.ts     # Stages 1–4: elicit, induce, validate, compact
│   │       ├── store.ts        # Stages 5–8: persist, retrieve, apply, revise
│   │       └── tools.ts        # knowledge_search tool via plugin SDK
│   └── skills-foo/             # Example PIL-aware skill
│       └── SKILL.md
├── apps/
│   └── playground/             # Dev harness — runs the pipeline without OpenClaw
│       └── index.ts
└── docs/                       # Design documents
    ├── memory-taxonomy.md      # Four memory types, generalization, cognitive mechanisms
    ├── architecture.md         # Tiered triggering, knowledge graph, artifact schema
    ├── example-learning-in-action.md
    ├── roadmap.md              # Phased roadmap with near-term milestones
    ├── design-decisions.md     # Rationale, differentiators, forward-compatibility
    ├── security.md             # Threat model, risks, and mitigations by phase
    └── enterprise-vision.md    # Enterprise deployment model and investment thesis
```

## Getting started

Requires Node.js ≥ 18 and pnpm.

```bash
git clone https://github.com/khub-ai/openclaw-knowledge-management
cd openclaw-knowledge-management
pnpm install
cd apps/playground
pnpm start        # runs all 8 PIL stages against sample input
pnpm dev          # re-runs on file changes
```

Artifacts are stored at `~/.openclaw/knowledge/artifacts.jsonl`.
Override with `KNOWLEDGE_STORE_PATH=/your/path pnpm start`.

## Implementation status

**The code is not yet functional.** The current implementation consists of placeholder heuristics that demonstrate the pipeline architecture but are not suitable for real use. Each stage will need to be replaced with LLM-backed logic to produce meaningful results.

### Placeholder implementations (scaffolding only)
| Stage | Module | Current state |
|---|---|---|
| 1 Elicit | `pipeline.ts` | Placeholder: matches sentences containing hardcoded signal words |
| 2 Induce | `pipeline.ts` | Placeholder: regex keyword matching to guess artifact kind |
| 3 Validate | `pipeline.ts` | Placeholder: adjusts confidence by counting hedging/assertive words |
| 4 Compact | `pipeline.ts` | Placeholder: whitespace normalisation only |
| 5 Persist | `store.ts` | Placeholder: JSONL file append with basic Jaccard dedup |
| 6 Retrieve | `store.ts` | Placeholder: keyword overlap scoring, no semantic understanding |
| 7 Apply | `store.ts` | Placeholder: simple threshold check (≥ 0.8 → auto-apply) |
| 8 Revise | `store.ts` | Placeholder: in-place update or retire based on content diff |
| Plugin | `index.ts` / `tools.ts` | Wiring only: `knowledge_search` tool registered with OpenClaw |

### What's needed to reach a functional state
- LLM-backed induction (replace regex heuristics with structured LLM calls)
- LLM-backed elicitation (contextual knowledge extraction, not keyword matching)
- Confidence calibration from user feedback signals
- Vector/semantic retrieval (replacing keyword Jaccard)
- Evaluative knowledge capture and generalization
- Procedural recipes with optional code synthesis
- Import/export primitives
- CLI for inspecting, editing, and deleting artifacts

## Non-goals

- Not fine-tuning base LLM weights
- Not replacing OpenClaw's core — this is an extension
- Not treating "memory" as a single bucket — knowledge types differ and require different controls
- Not prescribing a rigid schema — artifacts are free-form text with conventions

## Status

Early-stage / experimental. The pipeline architecture and storage layer are scaffolded with placeholder heuristics. **The code is not yet functional for real use** — each stage needs to be replaced with LLM-backed logic to produce meaningful knowledge artifacts. The current placeholders exist to validate the pipeline structure and provide a development harness.

## Contributing

Contributions are welcome — especially around:
- Knowledge artifact conventions and evaluation
- Retrieval and ranking methods for learned artifacts
- Evaluative knowledge representation
- Procedural knowledge and code synthesis
- Safe application policies and conflict resolution
- Tooling for inspection, export, and deletion

---

**Working thesis:** Agents become genuinely useful when they can **learn interactively**, **persist what they learn as user-owned artifacts**, and **reliably reuse that knowledge** — all without repeatedly re-prompting the user, retraining the model, or locking knowledge into a vendor's platform.
