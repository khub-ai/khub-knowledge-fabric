# Roadmap

This project is structured in five phases, progressing from a working personal knowledge store toward a portable, cross-agent knowledge ecosystem. Each phase produces something independently useful.

**The near-term priority is practical utility.** Phase 1 is broken into incremental milestones so that something simple but functional is available soon, while the longer-term vision (phases 2–5) demonstrates that the architecture can support ambitious goals.

## Phase 1 — Personal Knowledge Store *(current)*

**Goal:** A single user accumulates knowledge across sessions and can inspect and edit it.

Phase 1 is broken into four milestones. Each produces a working system — the next milestone extends it.

### Milestone 1a — Pipeline scaffolding ✅ *(done)*
- PIL pipeline (8 stages) with placeholder heuristics
- JSONL local storage with basic deduplication
- Playground for testing the pipeline end-to-end
- OpenClaw plugin wiring (`knowledge_search` tool)
- **What you get:** Architectural skeleton. Not functional for real use — placeholders only.

### Milestone 1b — Explicit "remember" command *(next)*
- The user says "remember this: [knowledge]" and the agent stores it as an artifact
- LLM-backed induction: classify the knowledge into a kind (semantic, procedural, evaluative) with appropriate confidence
- LLM-backed enrichment: generate tags, topic, summary, and trigger condition for the artifact
- On session start (`before_prompt_build`), retrieve all artifacts relevant to the current conversation context and inject them
- Basic import/export: load artifacts from a file, export to a file
- **What you get:** A working personal knowledge store. The user explicitly teaches the agent; the agent remembers and applies in future sessions. Artifacts are local files the user can inspect and edit.

### Milestone 1c — Passive elicitation via hooks
- Register `message_received` hook to passively observe conversation
- LLM-backed elicitation: on each message (or batched per conversation), identify candidate knowledge without the user explicitly saying "remember"
- Deduplicate against existing store; stage new candidates for user confirmation
- **What you get:** The agent learns from conversation without explicit instruction. It proposes new artifacts when it notices patterns.

### Milestone 1d — Tier 1 reflexive triggering
- Build inverted index on artifact tags and topics
- On each message, tokenize and look up matching artifacts (no LLM cost)
- Stage matched artifacts for injection in `before_prompt_build`
- **What you get:** Knowledge retrieval happens on every message at zero cost. High-confidence artifacts are applied automatically; lower-confidence ones are presented as suggestions.

### Phase 1 deliverable
A developer can install the extension, accumulate knowledge across sessions (both explicitly and passively), and see the agent apply learned knowledge in new conversations. Artifacts are local text files that can be inspected, edited, and version-controlled.

---

## Phase 2 — Generalization Engine

**Goal:** Turn specific observations into general rules and patterns.

This is the transition from "memory" to "knowledge management." The agent stops merely recording what the user said and starts producing generalized rules, conventions, and judgment heuristics that apply across contexts.

### Key capabilities
- Episodic → semantic generalization (e.g., 5 corrections of summary format → 1 preference rule: "always use bullet points")
- Episodic → evaluative generalization (e.g., patterns of user choices → judgment heuristics about what constitutes "good" output)
- Tier 2 triggering: cheap LLM reasoning to disambiguate partial matches from Tier 1
- Confidence calibration from user feedback signals (accept/reject/edit)
- Semantic retrieval (vector or hybrid search, augmenting keyword matching)
- Conflict detection (new knowledge contradicts existing rules)
- Background consolidation: periodic review of raw artifacts to produce consolidated generalizations
- Decay: artifacts that aren't retrieved or reinforced gradually lose effective confidence

### Deliverable
Agent behavior becomes noticeably more predictive — it generalizes from examples rather than requiring explicit instruction for every new situation.

---

## Phase 3 — Procedural Memory and Code Synthesis

**Goal:** Agent learns to produce and maintain structured procedures, and optionally compile them into executable programs.

### Key capabilities
- Procedural artifact type with structured recipe as the primary form
- Optional compilation of recipes into executable scripts after repeated identical execution
- The structured recipe is always retained — as documentation, as a fallback for flexible execution, and as the human-readable specification of what the program does
- Sandboxed runtime for executing stored programs
- Testing and verification step before a procedure graduates from recipe to code
- Library management for the user's growing collection of agent-generated tools

### Enterprise considerations
- Perfect repeatability: for critical workflows, a program guarantees identical results regardless of how many times the procedure runs
- Cost efficiency: running a program is vastly cheaper than having an LLM re-derive a complex workflow each time
- Auditability: generated code can be reviewed, tested, and version-controlled

### Deliverable
For repeatable tasks, the agent writes a program once then runs it reliably. For tasks requiring flexibility, the agent follows the structured recipe with judgment. The user accumulates a library of agent-generated tools.

---

## Phase 4 — Portability and Cross-Agent Compatibility

**Goal:** Knowledge artifacts work across agents and platforms.

### Key capabilities
- Standard artifact format specification (text-based, open, documented)
- Import/export adapters for other agent frameworks
- Validation that artifacts remain model-agnostic (no LLM-specific embeddings, tokens, or weights)

### What "model-agnostic" means precisely
The knowledge artifacts themselves are free-form text with lightweight conventions — any sufficiently capable LLM can read and reason over them. The PIL pipeline (the code that elicits, induces, validates, and applies knowledge) requires a capable host agent, but the artifacts it produces are not tied to any specific model or vendor. A user can export artifacts from an OpenClaw instance running Claude and import them into an instance running GPT or Gemini.

### Deliverable
A user can export their knowledge from OpenClaw and import it into another assistant. Knowledge survives platform transitions.

---

## Phase 5 — Governance and Ecosystem *(long-term)*

**Goal:** Knowledge as a shareable, governed, and potentially tradeable unit — beyond the individual user.

This phase converts the artifact format and ownership model built in phases 1–4 into a foundation for team and organizational knowledge management. It is also where a new kind of economic value becomes possible.

→ *[Enterprise vision: detailed considerations and business models](enterprise-vision.md)*

### Enterprise: from personal to organizational knowledge

The hardest knowledge management problem in organizations is not storage — it is capture, curation, and continuity. Subject-matter expertise is locked in individuals' heads, in email threads, and in chat histories that are impossible to search. When a key person leaves, that knowledge largely leaves with them.

PIL's artifact model addresses this differently.

**Tiered ownership**

Knowledge flows through a natural hierarchy:

| Tier | Scope | Who can see it |
|---|---|---|
| **Personal** | Private to the individual | Owner only |
| **Team** | Shared within a defined group | Team members |
| **Org** | Curated, reviewed, and approved | All employees |
| **Public** | Open registry | Anyone who imports the package |

This mirrors how knowledge actually moves in organizations: individuals learn → teams develop shared conventions → organizations codify policy → industries develop standards. Each tier promotion is gated by an explicit human decision, not automatic.

**Access controls and role model**

A governance layer separates four roles:

| Role | What they can do |
|---|---|
| **Contributor** | Create and edit personal artifacts; submit to team review |
| **Reviewer** | Approve or reject team submissions; annotate with rationale |
| **Publisher** | Promote reviewed artifacts to org-level; set scope and expiry |
| **Auditor** | Read-only access to all artifacts and their full provenance chain |

Because every artifact already carries a provenance record — who created it, from what source, when it was revised, with what confidence — the governance layer can be enforced without modifying the artifact format. Access control is layered on top of what phases 1–4 already produce.

**Compliance-friendly audit trail**

Each artifact's lifecycle — creation, retrieval, application, revision, retirement — is recorded in its provenance fields. This means:

- You can answer "what knowledge was injected into this conversation?" for any past session.
- You can trace a specific agent recommendation back to the artifact that informed it, the conversation that generated it, and the person who confirmed it.
- Artifact retirement records *when* a rule was superseded and by what — nothing is silently overwritten.

For regulated industries — legal, financial services, healthcare — this is not a nice-to-have. It is a prerequisite for deploying AI agents in consequential workflows.

**Organizational continuity**

The knowledge a skilled employee accumulates over years of agent use — preferred formats, client-specific conventions, learned shortcuts, evaluated quality standards — does not have to leave when they do. With an org-level knowledge tier and appropriate consent controls, curated portions of that accumulated knowledge can be retained, reviewed, and inherited by successors or shared with the team.

This is qualitatively different from document retention. The knowledge is *active*: it informs the successor's agent from day one, surfaces relevant conventions at the moment they are needed, and can be revised as the organization's practices evolve.

---

### Ecosystem: a new economic layer for knowledge artifacts

When knowledge artifacts are user-owned, portable, verifiable, and typed, they become a tradeable unit. This is the economic opportunity Phase 5 opens.

**Knowledge packages as products**

A domain expert — a senior tax attorney, a clinical pharmacist, a structural engineer — accumulates judgment that takes years to develop. Today they can publish a textbook. With PIL, they can publish a *knowledge package*: a curated set of evaluative, procedural, and semantic artifacts representing their expertise, importable directly into an AI agent.

The recipient's agent does not merely read a document. It applies the expert's judgment framework in real time, surfaces the relevant heuristics when the situation calls for them, and treats imported knowledge with appropriately calibrated confidence.

**The coordination layer opportunity**

The PIL artifact format — if it achieves adoption — defines a coordination layer analogous to what OpenAPI did for REST APIs or what npm did for JavaScript packages: a standard format that allows producers and consumers to exchange value without direct coordination.

Whoever establishes this standard captures network effects: more expert producers attract more consumers, more consumers attract more producers, and switching costs increase as org-level knowledge accumulates in the format.

**Potential business models**

| Model | Description |
|---|---|
| **Expert packages** | Domain professionals publish curated knowledge packages (e.g., "EU GDPR compliance for SaaS") on a subscription or per-download basis |
| **Org knowledge custody** | A managed service that hosts, versions, and distributes org-level knowledge packages with audit trails and access controls — like a private registry for knowledge |
| **Certification** | Third parties verify and certify knowledge packages (e.g., "reviewed by licensed practitioners"), commanding a premium for high-stakes domains |
| **Knowledge migration** | Services that convert existing documentation, process manuals, and policy documents into structured PIL artifacts |
| **Continuity-as-a-service** | Managed retention and curation of individual knowledge accumulated during employment, with org licensing and handover options |

None of these require fundamental changes to the artifact format. They are services built on top of a portable, open format — the same pattern that has worked repeatedly in software infrastructure.

---

### Technical foundation for Phase 5

The architecture built in phases 1–4 is designed to support these use cases without rearchitecting:

- **Artifact format** is text-based, versioned, and typed — it can carry governance metadata as optional enrichment fields without breaking existing consumers.
- **Provenance fields** already record creator, source, confidence, and revision history — a compliance audit trail is primarily a query and presentation layer on top of what already exists.
- **Model-agnostic** design means org knowledge is not locked to whichever LLM a team uses today; it survives vendor transitions.
- **Additive-only field evolution** means Phase 5 governance fields are backwards-compatible with Phase 1 artifacts.

We defer the specifics deliberately: the right design will be clearer once phases 1–4 have generated real-world experience with how artifacts are used, shared, and valued. What we avoid is making decisions in phases 1–4 that would foreclose Phase 5 options.

---

## Summary timeline

| Milestone | What it delivers | LLM cost |
|---|---|---|
| **1a** ✅ | Pipeline scaffolding, placeholder heuristics | None |
| **1b** | Explicit "remember" + retrieval + injection | Per explicit command |
| **1c** | Passive elicitation from conversation | Per message (batch possible) |
| **1d** | Tier 1 reflexive triggering (index-based) | None (index lookup) |
| **Phase 2** | Generalization, Tier 2 triggering, decay, feedback | Occasional cheap LLM calls |
| **Phase 3** | Procedural recipes, optional code synthesis | Per procedure compilation |
| **Phase 4** | Standard format, import/export, cross-agent | None (format work) |
| **Phase 5** | Governance, sharing, ecosystem | TBD |
