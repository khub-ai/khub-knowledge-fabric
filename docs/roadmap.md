# Roadmap

This project is structured in five phases, progressing from a working personal knowledge store toward a portable, cross-agent knowledge ecosystem. Each phase produces something independently useful.

## Phase 1 — Personal Knowledge Store *(current)*

**Goal:** A single user accumulates knowledge across sessions and can inspect and edit it.

### Done
- PIL pipeline (8 stages) with text-based artifacts
- JSONL local storage with deduplication and retrieval
- Human-readable, editable artifact files
- Playground for development and testing
- OpenClaw plugin wiring (`knowledge_search` tool)

### Next
- LLM-backed induction (replace heuristic extraction with structured LLM calls)
- Basic import/export (load artifacts from a file, export to a file)
- CLI commands for inspecting, editing, and deleting stored artifacts

### Deliverable
Something a developer can install today and start accumulating knowledge from interaction. Artifacts are local files that can be inspected and edited with any text editor.

---

## Phase 2 — Generalization Engine

**Goal:** Turn specific observations into general rules and patterns.

This is the transition from "memory" to "knowledge management." The agent stops merely recording what the user said and starts producing generalized rules, conventions, and judgment heuristics that apply across contexts.

### Key capabilities
- Episodic → semantic generalization (e.g., 5 corrections of summary format → 1 preference rule: "always use bullet points")
- Episodic → evaluative generalization (e.g., patterns of user choices → judgment heuristics about what constitutes "good" output)
- Confidence calibration from user feedback signals (accept/reject/edit)
- Semantic retrieval (vector or hybrid search, augmenting keyword matching)
- Conflict detection (new knowledge contradicts existing rules)

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
