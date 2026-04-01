# Architecture

This document describes the system architecture: how [knowledge artifacts](glossary.md#knowledge-artifact) are structured, how they are stored and indexed, how the tiered trigger cascade retrieves them efficiently, and how the system integrates with OpenClaw's plugin SDK.

## Design principles

1. **The core of an artifact is always `kind` + `content` + `confidence`.** All other fields are optional enrichment metadata. An artifact without tags, trigger conditions, or graph relationships still works — it can be stored, retrieved (by content search), applied, and revised. Enrichments make retrieval faster and triggering smarter, but they are not structural requirements.

2. **Old artifacts remain valid.** New fields are always optional. The pipeline never refuses to process an artifact because it's missing metadata. A human can create an artifact with two lines of text (kind + content) and it works.

3. **The architecture is designed to evolve.** Even if the triggering system, graph structure, or storage backend change fundamentally, the underlying artifacts — which are text — remain usable. Forward-compatibility is a first-class concern.

4. **Extraction is language-agnostic.** The pipeline delegates all natural language understanding to the LLM. No English-specific heuristics (signal words, hedge words, regex patterns) appear anywhere in the pipeline logic. A user interacting in Chinese, Farsi, Spanish, or any other language gets identical behavior.

## Knowledge artifact schema

Each artifact is a JSON object with required core fields and optional enrichment fields:

### Core fields (required)

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique identifier (UUID) |
| `kind` | string | `"preference"`, `"convention"`, `"fact"`, `"procedure"`, `"judgment"`, or `"strategy"` — see [Kind taxonomy](#kind-taxonomy) |
| `content` | string | The knowledge itself, in free-form text. For `candidate` and `accumulating` artifacts: verbatim from the user's input (any language). For `consolidated` artifacts: LLM-distilled generalization. |
| `confidence` | number | 0–1; certainty that this artifact is correct and applicable. Seeded from `certainty` at extraction; grows with evidence. |
| `provenance` | string | Where this knowledge came from (session ID, user statement, etc.) |
| `createdAt` | string | ISO 8601 timestamp |

### Enrichment fields (optional, for triggering and retrieval)

| Field | Type | Description | Cognitive analogue |
|---|---|---|---|
| `scope` | string | `"specific"` (applies only to this situation) or `"general"` (applies broadly across contexts) — LLM-assigned at extraction | Context-dependent retrieval |
| `certainty` | string | `"definitive"`, `"tentative"`, or `"uncertain"` — how strongly the knowledge was expressed, regardless of language — LLM-assigned at extraction | Confidence seeding |
| `trigger` | string | Natural language condition for when this artifact applies | Context-dependent retrieval |
| `tags` | string[] | Normalized topic tags: lowercase, hyphenated, English noun phrases (e.g., `"code-style"`, `"file-naming"`). LLM-assigned; anchored to the store's existing tag vocabulary for consistency. | Associative priming |
| `topic` | string | High-level domain cluster (e.g., `"code-style"`, `"financial-ops"`) | Chunking |
| `summary` | string | One-line distillation for cheap LLM matching in Tier 2 | Gist memory |
| `relations` | object[] | Graph edges: `{ type, targetId }` where type is `"references"`, `"constrains"`, `"supersedes"`, or `"supports"` | Spreading activation |

### Lifecycle fields (optional, for decay, habituation, and revision)

| Field | Type | Description | Cognitive analogue |
|---|---|---|---|
| `salience` | string | `"low"`, `"medium"`, or `"high"` — how much this matters (separate from confidence) | Emotional valence |
| `stage` | string | `"candidate"` (single observation, not yet confirmed), `"accumulating"` (evidence building toward threshold), or `"consolidated"` (LLM-distilled generalization, ready for injection) | Memory consolidation |
| `evidenceCount` | number | How many observations support this pattern. When this reaches the consolidation threshold (default: **3**), a consolidation LLM call is triggered. | Pattern accumulation |
| `evidence` | string[] | Raw observation snippets — verbatim from user input, in any language — that will be passed to the consolidation LLM call | Episodic substrate |
| `lastRetrievedAt` | string | ISO 8601; last time this artifact was retrieved | Decay tracking |
| `reinforcementCount` | number | How many times this artifact has been confirmed or reused | Rehearsal strengthening |
| `appliedCount` | number | How many times this artifact was applied or suggested | Habituation tracking |
| `acceptedCount` | number | How many times the user accepted the application | Feedback calibration |
| `rejectedCount` | number | How many times the user rejected or overrode | Feedback calibration |
| `revisedAt` | string | ISO 8601; last revision timestamp | Revision trail |
| `retired` | boolean | True when superseded or invalidated | Graceful forgetting |

**All optional fields default to absent.** An artifact with only the core fields is fully functional. Enrichment and lifecycle fields are populated progressively — by LLM-backed enrichment, by the triggering system, and by user feedback.

## Kind taxonomy

The `kind` field uses nine values. The first six are assigned by the LLM during passive Phase 1 extraction from any input language. The final three are produced only by Phase 4 dialogic learning sessions and are never emitted by the passive pipeline.

| Kind | What it captures | Example | Phase |
|---|---|---|---|
| `preference` | Subjective style or taste | "Always use bullet points for summaries" | 1 |
| `convention` | Agreed naming, terminology, or standards | "We call it the staging environment" | 1 |
| `fact` | Objective, verifiable information | "The API endpoint is https://api.example.com/v2" | 1 |
| `procedure` | A specific step-by-step process or recipe | "To deploy: run build, then push, then notify team" | 1 + 4 |
| `judgment` | An evaluative heuristic or quality criterion | "Favor brevity over completeness in executive summaries" | 1 + 4 |
| `strategy` | A general approach to a class of problems | "When debugging, isolate variables before forming hypotheses" | 1 + 4 |
| `boundary` | When a rule does not apply | "This survivability check is weaker in commodity-driven businesses" | 4 only |
| `revision-trigger` | Evidence that should cause revision of a conclusion | "Downgrade the thesis if confidence depends on management promises" | 4 only |
| `failure-case` | A past mistake that refined later judgment | "Applied the fragility rule too mechanically; rejected a sound business" | 4 only |

**Relationship to the cognitive taxonomy** described in [memory-taxonomy.md](memory-taxonomy.md): `preference + convention + fact` map to *semantic* memory; `procedure` maps to *procedural* memory; `judgment + strategy` map to *evaluative* memory; `boundary + revision-trigger + failure-case` are produced by Phase 4 dialogic learning and carry structured expert know-how. The cognitive taxonomy is conceptual framing; the `kind` field is what appears in artifacts.

---

## Extraction pipeline

The pipeline translates raw user input into typed, confidence-scored artifacts. All natural language understanding is delegated to the LLM; no English-specific heuristics appear in the pipeline logic, making the system language-agnostic.

### Stage 1 — Extract (LLM call)

A single LLM call replaces the former English-specific Elicit + Induce + Validate stages.

**Input:** the user's message (any language) + the store's current tag vocabulary (for tag normalization)

**The LLM is asked to:**
- Decide whether the message contains any persistable, reusable knowledge
- If yes, extract each piece as a separate candidate with: `content` (verbatim from input), `kind`, `scope`, `certainty`, `tags`, `rationale`
- If no persistable knowledge is present, return an empty list

**Output per candidate:**

| Field | Description |
|---|---|
| `content` | Exact quote or minimal paraphrase from the input, in the user's original language |
| `kind` | One of the six values above |
| `scope` | `"specific"` or `"general"` |
| `certainty` | `"definitive"`, `"tentative"`, or `"uncertain"` |
| `tags` | 2–5 normalized topic tags (lowercase, hyphenated, English noun phrases; prefer existing store vocabulary) |
| `rationale` | One sentence explaining why this is worth remembering (stored in `provenance`) |

**Initial confidence from `certainty`** (replaces hedge/assertion word heuristics):

| `certainty` | Starting `confidence` | Rationale |
|---|---|---|
| `"definitive"` | 0.65 | One strong observation; grows with further evidence |
| `"tentative"` | 0.35 | Weak signal; needs reinforcement before use |
| `"uncertain"` | 0.15 | Barely a signal; consolidation required before injection |

### Stage 2 — Match (store lookup)

For each extracted candidate, compare against existing active artifacts of the same `kind`:

- Match criteria: same `kind` + overlapping `tags` + compatible `scope`
- Three outcomes: **novel**, **accumulating**, or **confident**

### Stage 3 — Resolve (evidence accumulation)

| Outcome | What happens |
|---|---|
| **Novel** | Create a new artifact at `stage: "candidate"`, `evidenceCount: 1`, `evidence: [content]` |
| **Accumulating** | Increment `evidenceCount` on the matched artifact; append `content` to `evidence[]`. If `evidenceCount` reaches the consolidation threshold (**3**, configurable) → trigger consolidation LLM call → promote to `stage: "consolidated"` |
| **Confident** | Retrieve the consolidated artifact for injection; update retrieval and reinforcement counts |

### Consolidation LLM call

Triggered when `evidenceCount` reaches the threshold (default: 3). The LLM receives all entries in `evidence[]` and is asked to:

> *"Given these N observations from a user, what is the general, reusable rule or pattern they reflect? Produce a concise artifact that will guide the agent in future sessions."*

The output becomes the new `content` of the artifact (now a generalized rule rather than a verbatim quote). The original observations remain in `evidence[]` as an audit trail.

*Future: the threshold may become dynamic — adjusted by conversation progress, `certainty` mix, or user feedback signals.*

### Injection rules

Not all artifacts are injected immediately:

| `stage` | `certainty` | Injectable? | Label in context |
|---|---|---|---|
| `consolidated` | any | ✅ Yes | confidence-gated per normal rules |
| `candidate` | `"definitive"` | ✅ Yes (provisional) | `[provisional]` — single strong observation |
| `candidate` | `"tentative"` or `"uncertain"` | ❌ No | awaiting consolidation |
| `accumulating` | any | ❌ No | awaiting consolidation |

The `[provisional]` label signals to the agent that this knowledge comes from a single unconfirmed observation and should be applied with extra caution. This policy is experimental and may be revised as we learn from real use.

---

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

Confidence exists in two layers: the stored `confidence` field (the durable audit record, never modified by decay) and `effectiveConfidence` (computed at every read, never persisted to disk).

### Stored confidence

`confidence` is seeded at extraction time from the `certainty` field assigned by the LLM:

```
certainty "definitive" → confidence = 0.65
certainty "tentative"  → confidence = 0.35
certainty "uncertain"  → confidence = 0.15
```

It grows through evidence accumulation: supporting observations increment `evidenceCount`, and when the consolidation threshold is reached the artifact is distilled by an LLM call into a generalization at `stage: "consolidated"`. The stored `confidence` is ground truth — it reflects the strength of the evidence record, not recency of use.

### Effective confidence (validation-modulated decay)

`effectiveConfidence` is computed on every read from the stored `confidence`, validation history, and time since last retrieval. Implementation lives in `effectiveConfidence()` in `store.ts`; constants are in `DECAY_CONSTANTS` in `types.ts`.

```
validationStrength = reinforcementCount + 2 × acceptedCount − rejectedCount
halfLifeDays       = BASE_HALF_LIFE_DAYS × (1 + VALIDATION_ALPHA × validationStrength)
decayFactor        = 0.5 ^ (daysSinceLastRetrieved / halfLifeDays)
                     (1.0 if lastRetrievedAt is absent — no decay until first use)
```

Constants:

| Constant | Value | Meaning |
|---|---|---|
| `BASE_HALF_LIFE_DAYS` | 30 | Half-life (days) for an artifact with zero validation |
| `VALIDATION_ALPHA` | 0.20 | Each validation unit extends half-life by 20% |
| `DECAY_FLOOR_BASE` | 0.20 | Minimum floor for a consolidated artifact with no validation |
| `DECAY_FLOOR_MAX` | 0.60 | Ceiling — the most validated artifact's floor cannot exceed this |
| `DECAY_FLOOR_PER_VALIDATION` | 0.04 | Each validation unit raises the floor by 4% |

For `consolidated` artifacts a floor prevents complete forgetting:

```
floor = min(DECAY_FLOOR_MAX,
            DECAY_FLOOR_BASE + DECAY_FLOOR_PER_VALIDATION × validationStrength)
      × salience_multiplier   // high: × 1.25  |  low: × 0.75  |  medium: × 1.0
```

`candidate` and `accumulating` artifacts have no floor (floor = 0): unconfirmed observations expire naturally if the user never reinforces them.

Final formula:

```
effectiveConfidence = max(floor, floor + (confidence − floor) × decayFactor)
```

**Side effect of `retrieve()`:** every call to `retrieve()` that returns at least one result updates `lastRetrievedAt` on all returned artifacts. Actively-used knowledge therefore keeps its effective confidence high automatically. Empty-query list-all calls are exempted from this update to avoid artificially resetting decay for bulk inspection.

### Injection label gating

`getInjectLabel()` uses `effectiveConfidence` (not raw `confidence`) when choosing the label for `consolidated` artifacts:

```
effectiveConfidence ≥ auto-apply threshold  →  [established]  (injected as settled knowledge)
effectiveConfidence <  auto-apply threshold  →  [suggestion]   (presented as a suggestion)
```

A consolidated artifact that was once `[established]` automatically falls back to `[suggestion]` as its effective confidence decays, without any explicit user action.

### Auto-apply thresholds (adjusted by salience)

The `salience` field adjusts the per-artifact auto-apply threshold:

| `salience` | Auto-apply threshold | Notes |
|---|---|---|
| `"low"` | ≥ 0.75 | Minor preferences; auto-apply liberally |
| *(unset)* | ≥ 0.80 | Default |
| `"medium"` | ≥ 0.85 | Standard knowledge; normal caution |
| `"high"` | ≥ 0.95 | Critical rules; almost always suggest rather than auto-apply |

Note that `salience: "high"` simultaneously raises the auto-apply threshold *and* the decay floor: high-salience artifacts are harder to auto-apply silently (more likely to surface as explicit suggestions) but also more resistant to being forgotten entirely.

## Storage

All storage files are plain JSON or JSONL — no binary formats, no TypeScript-specific serialisation. Any language that can read JSON can read and write them directly.

### Artifact store

JSONL file at `~/.openclaw/knowledge/artifacts.jsonl` (overridable via `KNOWLEDGE_STORE_PATH`). One JSON object per line. Simple, human-inspectable, git-friendly.

### Phase 4 session files

Dialogic learning sessions are persisted separately as JSON objects at:

```
~/.openclaw/knowledge/sessions/<session-id>.json
```

Each file contains the full state of one expert session: the learning objective, domain, all dialogue turns, candidate rules with gap status, question history, and the IDs of artifacts promoted at session end. Session files are retained permanently as the audit record after the session ends.

### Communication Profile

User-level dialogue style preferences are stored at:

```
~/.openclaw/knowledge/communication-profile.json
```

This file is separate from session files and from `artifacts.jsonl` because it is user-level meta-knowledge — about how to conduct dialogue with this expert — rather than domain-specific knowledge.

### Future considerations

As the artifact count grows and graph traversal becomes important, the storage layer may evolve to:
- SQLite for indexed queries
- A lightweight graph database or adjacency list structure
- Vector embeddings for semantic retrieval (Phase 2+)

The storage backend is abstracted behind `persist()` and `retrieve()` operations (TypeScript reference: `store.ts`). Changing the storage implementation does not affect the artifact format or the rest of the pipeline.
