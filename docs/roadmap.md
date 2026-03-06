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
Agent behaviour becomes noticeably more predictive — it generalizes from examples rather than requiring explicit instruction for every new situation.

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

**Goal:** Knowledge as a shareable, portable unit beyond the individual user.

This phase is deliberately under-specified. The artifact format and ownership model built in phases 1–4 are designed to preserve design freedom for future possibilities:

- Org-level knowledge management and distribution
- Publishing and sharing of curated knowledge packages
- Access control and provenance tracking
- A potential ecosystem where domain experts publish knowledge artifacts that other users can import

The technical foundation — user-owned, text-based, model-agnostic artifacts with versioning and provenance — is intended to support these use cases without requiring fundamental changes to the artifact format or storage model.

We mention this phase for completeness but intentionally avoid over-specifying it. The right design for knowledge governance and sharing will become clearer as the foundation matures through real-world use.

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
