# Architecture

This document describes the system architecture: how knowledge artifacts are structured, how they are stored and indexed, how the tiered trigger cascade retrieves them efficiently, and how the system integrates with OpenClaw's plugin SDK.

## Design principles

1. **The core of an artifact is always `kind` + `content` + `confidence`.** All other fields are optional enrichment metadata. An artifact without tags, trigger conditions, or graph relationships still works — it can be stored, retrieved (by content search), applied, and revised. Enrichments make retrieval faster and triggering smarter, but they are not structural requirements.

2. **Old artifacts remain valid.** New fields are always optional. The pipeline never refuses to process an artifact because it's missing metadata. A human can create an artifact with two lines of text (kind + content) and it works.

3. **The architecture is designed to evolve.** Even if the triggering system, graph structure, or storage backend change fundamentally, the underlying artifacts — which are text — remain usable. Forward-compatibility is a first-class concern.

## Knowledge artifact schema

Each artifact is a JSON object with required core fields and optional enrichment fields:

### Core fields (required)

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique identifier (UUID) |
| `kind` | string | `"semantic"`, `"procedural"`, `"evaluative"`, or `"fact"` |
| `content` | string | The knowledge itself, in free-form text |
| `confidence` | number | 0–1; certainty that this artifact is correct and applicable |
| `provenance` | string | Where this knowledge came from (session ID, user statement, etc.) |
| `createdAt` | string | ISO 8601 timestamp |

### Enrichment fields (optional, for triggering and retrieval)

| Field | Type | Description | Cognitive analogue |
|---|---|---|---|
| `trigger` | string | Natural language condition for when this artifact applies | Context-dependent retrieval |
| `tags` | string[] | Keywords for Tier 1 index lookup | Associative priming |
| `topic` | string | High-level domain cluster (e.g., `"code-style"`, `"financial-ops"`) | Chunking |
| `summary` | string | One-line distillation for cheap LLM matching in Tier 2 | Gist memory |
| `relations` | object[] | Graph edges: `{ type, targetId }` where type is `"references"`, `"constrains"`, `"supersedes"`, or `"supports"` | Spreading activation |

### Lifecycle fields (optional, for decay, habituation, and revision)

| Field | Type | Description | Cognitive analogue |
|---|---|---|---|
| `salience` | string | `"low"`, `"medium"`, or `"high"` — how much this matters (separate from confidence) | Emotional valence |
| `stage` | string | `"raw"` (freshly captured) or `"consolidated"` (reviewed during background consolidation) | Memory consolidation |
| `lastRetrievedAt` | string | ISO 8601; last time this artifact was retrieved | Decay tracking |
| `reinforcementCount` | number | How many times this artifact has been confirmed or reused | Rehearsal strengthening |
| `appliedCount` | number | How many times this artifact was applied or suggested | Habituation tracking |
| `acceptedCount` | number | How many times the user accepted the application | Feedback calibration |
| `rejectedCount` | number | How many times the user rejected or overrode | Feedback calibration |
| `revisedAt` | string | ISO 8601; last revision timestamp | Revision trail |
| `retired` | boolean | True when superseded or invalidated | Graceful forgetting |

**All optional fields default to absent.** An artifact with only the core fields is fully functional. Enrichment and lifecycle fields are populated progressively — by LLM-backed enrichment, by the triggering system, and by user feedback.

## Knowledge graph

Artifacts form a graph through their `relations` field. The graph is implicit — stored as edges on each artifact, not as a separate data structure — and traversed during retrieval.

### Edge types

| Edge type | Meaning | Example |
|---|---|---|
| `references` | This artifact uses or depends on another | Procedure "download statements" → references → fact "institution list" |
| `constrains` | This artifact limits how another is applied | Convention "naming pattern" → constrains → procedure "download statements" |
| `supersedes` | This artifact replaced another (revision trail) | Updated rule → supersedes → old rule (now retired) |
| `supports` | This artifact provides evaluative guidance for another | Heuristic "respect reader's time" → supports → preference "concise summaries" |

### Graph traversal during retrieval

When a Tier 1 or Tier 2 match identifies a relevant artifact, the retrieval system performs a 1–2 hop traversal to also surface connected artifacts. This implements **spreading activation**: retrieving the "monthly statement download" procedure also brings the "institution list" and "naming convention" into scope, even if those didn't directly match the query.

Activation diminishes with hop distance — directly connected artifacts receive a strong boost; second-hop artifacts receive a weaker one.

## Tiered trigger cascade

The system retrieves and applies knowledge through a cost-ordered cascade. Cheap checks run first; expensive LLM reasoning is reserved for cases that need it.

```
Message arrives (message_received hook)
  │
  ├─ Tier 1: Reflexive (no LLM)
  │   Tokenize message → lookup in inverted index (tags, topics)
  │   → high-confidence matches → stage for injection
  │   → partial matches → collect candidates for Tier 2
  │
  ├─ Tier 2: Light reasoning (cheap/fast LLM)
  │   Send message + candidate artifact summaries to a fast model
  │   → model returns which artifacts are relevant
  │   → stage matches for injection
  │
  └─ (Tier 3 is implicit: the primary LLM reasons over injected context)

Agent prepares response (before_prompt_build hook)
  │
  └─ Inject staged artifacts into prompt context
     → high-confidence: injected as established knowledge
     → lower-confidence: injected as suggestions for the agent to consider
```

### Tier 1 — Reflexive (no LLM cost)

- Runs on **every message**
- Tokenizes the message into keywords
- Looks up the inverted index: `tag → [artifact IDs]`, `topic → [artifact IDs]`
- Artifacts matching with `confidence ≥ auto-apply threshold` and `salience`-adjusted thresholds are staged immediately
- Partial matches (some keyword overlap) are collected as candidates for Tier 2
- **Cost: zero** (in-memory index lookup)

### Tier 2 — Light reasoning (cheap LLM)

- Runs only when Tier 1 produces candidates that need disambiguation, or when the message seems potentially knowledge-relevant but Tier 1 found no matches
- Sends the message plus a batch of **candidate artifact summaries** (one line each, not full content) to a fast model
- The model returns which artifact IDs are relevant, with brief reasoning
- **Cost: low** (one call to a fast model, small payload of summaries)

### Tier 3 — Deep reasoning (primary LLM)

- Not a separate triggering step — it happens naturally when the primary LLM processes the prompt with injected artifacts
- The primary model reasons over the full content of injected artifacts and the conversation context
- This is where conflict resolution, nuanced application, and generalization can occur
- **Cost: already included** in the normal agent response

### Index structure

Built in memory when the store is loaded:

```
tagIndex:     Map<string, Set<artifactId>>      // "summary" → {a1, a5, a12}
topicIndex:   Map<string, Set<artifactId>>      // "code-style" → {a3, a7, a8}
kindIndex:    Map<string, Set<artifactId>>      // "procedural" → {a2, a6}
```

Updated incrementally when artifacts are persisted or revised. Rebuilt from scratch on store load (fast — the store is local).

## OpenClaw integration

The system integrates with OpenClaw through three hook points provided by the plugin SDK:

### `message_received` — passive observation

Fires on every inbound user message. The PIL extension:
1. Tokenizes the message and runs Tier 1 trigger search
2. If Tier 2 is needed, calls the LLM via OpenClaw's configured model provider (`api.runtime`)
3. Stages matched artifacts in session state for injection
4. Optionally runs background elicitation: sends the message to the LLM for knowledge extraction

### `before_prompt_build` — knowledge injection

Fires after session load, before the model generates a response. The PIL extension:
1. Reads staged artifacts from session state
2. Injects high-confidence artifacts as established context
3. Injects lower-confidence artifacts as suggestions
4. The agent then responds with learned knowledge already available

### `api.registerTool()` — explicit search

The `knowledge_search` tool allows the agent (or user) to explicitly query the knowledge store at any time, independent of the passive trigger cascade.

### Hook maturity note

As of February 2026, OpenClaw's hook system has the following working hooks: `before_agent_start`, `agent_end`, `message_received`, and `tool_result_persist`. Hooks for `before_tool_call`, `after_tool_call`, `message_sending`, and `message_sent` were reported as not yet fully wired. The `llm_input`/`llm_output` hooks are observational only, not mutable.

This means PIL can **observe inbound messages** and **inject into prompts**, but cannot yet observe or intercept the agent's outgoing responses. This is a minor limitation — most learnable knowledge comes from what the user says, not from the agent's output. When outgoing message hooks become available, they can be added to capture knowledge from the agent's reasoning.

## Effective confidence calculation

When an artifact is being considered for application, its raw `confidence` score is adjusted by several factors:

```
effectiveConfidence = confidence
  × decayFactor(age, lastRetrievedAt, reinforcementCount)
  × acceptanceRate(acceptedCount, rejectedCount)
```

The `salience` field modifies the auto-apply threshold rather than the confidence itself:
- `salience: "low"` → auto-apply at `effectiveConfidence ≥ 0.75`
- `salience: "medium"` → auto-apply at `effectiveConfidence ≥ 0.85`
- `salience: "high"` → auto-apply at `effectiveConfidence ≥ 0.95` (almost always suggest rather than auto-apply)

These thresholds are configurable and expected to be tuned through experience.

## Storage

### Current implementation

JSONL file at `~/.openclaw/knowledge/artifacts.jsonl` (overridable via `KNOWLEDGE_STORE_PATH`). One artifact per line. Simple, human-inspectable, git-friendly.

### Future considerations

As the artifact count grows and graph traversal becomes important, the storage layer may evolve to:
- SQLite for indexed queries
- A lightweight graph database or adjacency list structure
- Vector embeddings for semantic retrieval (Phase 2+)

The storage backend is abstracted behind `persist()` and `retrieve()` functions in `store.ts`. Changing the storage implementation does not affect the artifact format or the rest of the pipeline.
