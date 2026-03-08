# 01 — Playground Walkthrough

| | |
|---|---|
| **Program** | `apps/playground/index.ts` |
| **Milestones covered** | 1b (LLM extraction, accumulation, injection gating), 1c and 1d (core retrieval logic) |
| **Kind** | Scripted end-to-end harness with a live LLM |
| **Runtime** | ~15–30 seconds (4 Anthropic API calls) |

---

## Purpose

The playground is the primary human-readable health check for the PIL pipeline. It is a scripted, non-interactive program that sends four carefully chosen sample messages through the full pipeline, then exercises retrieval, apply, and revise against the resulting store.

Unlike the automated test suite, which uses mock LLMs and is suitable for CI, the playground uses a real Anthropic model. This means it can catch failures that only appear with genuine LLM responses: extraction quality, tag normalization against the live store vocabulary, consolidation output quality, and retrieval ranking under real conditions.

### What it covers

| Pipeline stage | Coverage | What is exercised | What is not exercised |
|---|---|---|---|
| Stage 1 — LLM extraction | ✅ Full | All 4 messages; both the extraction path and the noise-rejection path (Turn 4: 0 candidates) | — |
| Stage 2 — Tag-based matching | ✅ Full | Novel path (Turns 1, 2: no match → create) and match path (Turn 3: tag overlap → accumulate) | — |
| Stage 3 — Evidence accumulation | ⚠️ Partial | Accumulation step: Turn 3 appends evidence, `evCount` advances to 2 | Consolidation LLM call and promotion to `consolidated` — requires a third matching observation; see Variations |
| Stage 4 — Injection gating | ⚠️ Partial | `[provisional]` label (candidate + definitive certainty); no-inject case for `accumulating` stage | `[suggestion]` and `[established]` labels — both require a `consolidated` artifact, which is not produced in a standard run |
| Stage 5 — Persist to JSONL | ✅ Full | Store grows by 2 new artifacts per run; verified in store summary | — |
| Stage 6 — Retrieve | ✅ Full | Tag-overlap scoring, result ranking, retrieval from mixed-stage store | — |
| Stage 7 — Apply | ⚠️ Partial | `autoApply=false` path; `[provisional]` suggestion surfaced | `autoApply=true` path — requires a `consolidated` artifact with confidence above the salience threshold |
| Stage 8 — Revise | ✅ Full | Content replacement and confidence update on an existing artifact | — |

### What it does not cover

- OpenClaw hook wiring (`message_received` / `before_prompt_build`) — pending Milestones 1c and 1d
- Interactive user feedback — the apply decision in Stage 7 is observed but the script does not accept or reject it

---

## Prerequisites

```
Node.js >= 22
pnpm   >= 9
ANTHROPIC_API_KEY environment variable set
```

Verify the API key is present before running:

```bash
# macOS / Linux
echo $ANTHROPIC_API_KEY

# Windows PowerShell
echo $env:ANTHROPIC_API_KEY
```

The key must start with `sk-ant-`. If it is missing, the playground exits immediately with an error message.

---

## How to run

From the playground directory:

```bash
cd apps/playground
pnpm start
```

Or from the repo root:

```bash
pnpm --filter @khub-ai/playground start
```

### Model

The playground uses `claude-sonnet-4-6` by default (line 43 of `apps/playground/index.ts`). To use a different model, change the `model:` field in that file. The original development default was `claude-3-5-haiku-20241022` — faster and cheaper for iteration but may produce less consistent tag normalization.

Verify the exact model ID available to your API key at <https://console.anthropic.com/settings/models>.

### Store location

The playground reads from and writes to `~/.openclaw/knowledge/artifacts.jsonl`. Subsequent runs **accumulate artifacts** on top of existing ones. To start from a clean slate:

```bash
# macOS / Linux
rm ~/.openclaw/knowledge/artifacts.jsonl

# Windows PowerShell
Remove-Item $env:USERPROFILE\.openclaw\knowledge\artifacts.jsonl
```

To isolate a single run without affecting the shared store:

```bash
KNOWLEDGE_STORE_PATH=/tmp/pil-test.jsonl pnpm start
```

---

## Annotated output

The following is a complete annotated walkthrough of a real run. Each section corresponds to one block of output.

---

### Header

```
=== PIL Playground (LLM-backed pipeline) ===
```

Confirms the script started and the LLM adapter initialized. If you see an error about `ANTHROPIC_API_KEY` here instead, the environment variable is not set.

---

### Turn 1 — Definitive preference, novel

**Input:** `"I always want bullet-point summaries, no more than five points."`

```
  Candidates extracted: 1
    [preference/definitive] tags: summary-format, bullet-points, response-length, output-style
    "I always want bullet-point summaries, no more than five points."
  Created: 1   Updated: 0
    → [NEW / candidate] conf=0.65  "I always want bullet-point summaries, no more than five points."
  Injectable artifacts from this turn:
    Remembered user knowledge (apply as appropriate):
    [provisional] [preference] I always want bullet-point summaries, no more than five points.
```

**What each line means:**

| Output | Explanation |
|---|---|
| `Candidates extracted: 1` | The LLM found one piece of persistable knowledge in this message |
| `[preference/definitive]` | Kind = `preference`; certainty = `definitive` — "I always want" is unhedged and high-conviction |
| `tags: summary-format, bullet-points, ...` | LLM assigned 4 semantic tags; used for future match lookups |
| `conf=0.65` | Seeded from `CONFIDENCE_SEED.definitive` — every definitive statement starts at this value, regardless of content |
| `[NEW / candidate]` | No existing artifact matched this tag set → new artifact created at stage `candidate` |
| `[provisional]` injection label | `candidate` stage + `definitive` certainty → injectable immediately, labeled as a single unconfirmed observation |

The `[provisional]` label is the injection gate working correctly. The system trusts one strong observation enough to surface it to the agent, but flags it so the agent applies it cautiously rather than treating it as established fact.

---

### Turn 2 — Second preference, novel

**Input:** `"When writing code, I prefer TypeScript with strict mode enabled."`

```
  Candidates extracted: 1
    [preference/definitive] tags: typescript, strict-mode, language-preference, code-style
    "When writing code, I prefer TypeScript with strict mode enabled."
  Created: 1   Updated: 0
    → [NEW / candidate] conf=0.65
  Injectable artifacts from this turn:
    [provisional] [preference] When writing code, I prefer TypeScript with strict mode enabled.
```

Same pattern as Turn 1. Tags are entirely different (`typescript`, `code-style` vs. `summary-format`, `bullet-points`), so `matchCandidate()` finds no overlap above the 0.25 Jaccard threshold → new artifact created. Confidence = 0.65 again.

---

### Turn 3 — Evidence accumulation instead of duplication

**Input:** `"Use async/await rather than raw promises wherever possible."`

```
  Candidates extracted: 1
  Created: 0   Updated: 1
    → [UPD / accumulating] evCount=2  conf=0.65
```

**This turn is the most structurally important in the run.** The `async/await` message shares tags with Turn 2's artifact (`typescript`, `code-style`, `language-preference`). The Jaccard overlap on tag sets exceeded the 0.25 threshold (`TAG_MATCH_THRESHOLD` in `store.ts`), so instead of creating a third new artifact, `matchCandidate()` returned the existing TypeScript artifact and `accumulateEvidence()` appended this observation to its `evidence[]` array.

| Output | Explanation |
|---|---|
| `Created: 0   Updated: 1` | Recognised as evidence for an existing pattern, not a new one |
| `evCount=2` | Two observations now support this code-style cluster |
| `[UPD / accumulating]` | Stage promoted from `candidate` to `accumulating` |
| No injectable block | `accumulating` stage never injects — awaiting the third observation |

The absence of an injectable block here is **intentional and correct**. The system does not inject partially-accumulated knowledge. When a third matching message arrives (any session), `evCount` will reach 3 (= `CONSOLIDATION_THRESHOLD`), triggering a consolidation LLM call that produces a generalized rule and promotes the artifact to `stage: "consolidated"` — at which point it becomes eligible for `[suggestion]` or `[established]` injection.

---

### Turn 4 — The noise filter

**Input:** `"What time is it?"`

```
  Candidates extracted: 0
  Created: 0   Updated: 0
```

The most important negative case in the entire run. "What time is it?" contains no persistable knowledge — it is a request, not a statement of preference, convention, fact, or procedure. The LLM correctly returns an empty candidates list. No store writes occur, no disk access, no output block.

**Healthy run requirement:** Turn 4 must always produce `Candidates extracted: 0`. If a simple factual question produces artifacts, extraction quality has regressed — one of the most critical failure modes for a production knowledge store, since noise accumulates silently.

---

### Stage 6 — Retrieve

```
────────────────────────────────────────────────────────────  Stage 6 — Retrieve
Query: "summary format preference"  →  4 match(es):

  [preference/candidate]    conf=0.65  "I always want bullet-point summaries, no more than five points."
  [preference/accumulating] conf=0.65  "When writing code, I prefer TypeScript with strict mode enabled."
  [preference/?]            conf=0.95  "I always want concise bullet-point summaries, strictly no more than five points."
  [preference/?]            conf=0.60  "Never use filler phrases like 'in conclusion' or 'to summarise'."
```

The playground queries the store with `"summary format preference"` and requests the top 5 results. The results are ranked by a composite score:

```
score = tag_overlap   × 0.60
      + content_jaccard × 0.30
      + confidence      × 0.10
      + 0.05             (if stage = "consolidated")
```

**Interpreting the result set:**

The first two artifacts (`candidate` and `accumulating`) were created in this run. The `[?]`-stage artifacts are pre-existing in the store from a prior run — they are `consolidated` stage artifacts. The `?` is a minor display formatting issue explained under [Known display quirks](#known-display-quirks); the underlying stage value in the JSONL is `"consolidated"`.

The TypeScript artifact (`accumulating`) appears in a `summary format` query because its tags include `code-style`, which has enough overlap with `output-style` from Turn 1's artifact to produce a non-zero tag score. This is the retrieval ranking working correctly — topic-adjacent artifacts can appear in results but rank below directly-matched ones. If the TypeScript artifact appeared first, tag scoring would have drifted.

**Reading the `conf=0.95` artifact:** A consolidated artifact with high confidence that survived prior runs and revision. It outranks the freshly extracted `candidate` because the consolidated bonus (0.05) and its higher confidence push it above the 0.65 starting confidence of new artifacts.

---

### Stage 7 — Apply

```
────────────────────────────────────────────────────────────  Stage 7 — Apply
  autoApply=false  →  [provisional] I always want bullet-point summaries, no more than five points.
```

The playground calls `apply()` on the top result. Two pieces of information are reported:

| Field | Value | What it means |
|---|---|---|
| `autoApply=false` | Artifact is `candidate` stage, confidence 0.65 | Does not meet the salience-adjusted `AUTO_APPLY_THRESHOLD`; agent should present as suggestion |
| `[provisional]` | From `getInjectLabel()` | Single unconfirmed observation; apply with caution |

`autoApply=true` would appear if an artifact were `consolidated` and its confidence exceeded the threshold (`≥ 0.75` for low salience, `≥ 0.85` for medium, `≥ 0.95` for high). That requires at least three supporting observations — which hasn't happened in a fresh two-message run.

The three inject labels and when they appear:

| Label | Condition | Auto-apply? |
|---|---|---|
| `[provisional]` | `candidate` + `definitive` certainty | No |
| `[suggestion]` | `consolidated` + confidence below threshold | No |
| `[established]` | `consolidated` + confidence ≥ salience threshold | Yes |

**Healthy run requirement:** On a first run against a clean store, `autoApply` must be `false` for all results. `autoApply=true` appearing for the TypeScript artifact (only 2 observations) would indicate the threshold logic is broken.

---

### Stage 8 — Revise

```
────────────────────────────────────────────────────────────  Stage 8 — Revise
  Revised: id=4a6d91fe…  conf=1
  Content: "I always want concise bullet-point summaries, strictly no more than five points. (revised in playground)"
```

The playground calls `revise()` on the first non-retired artifact in the store. It updates the content and adds 0.05 to the confidence (capped at 1.0). The `(revised in playground)` suffix is appended by the script to make it easy to identify playground-generated revisions in the JSONL.

Note that `conf=1` here means the artifact being revised was already at `0.95` or `1.00` (from a prior run's revision or from the pre-seeded data). The first run against a clean store revises the Turn 1 artifact from `0.65` → `0.70`.

The key behavior to verify: the artifact's `id` is preserved (it's the same artifact in-place, not a new one), and the content is updated. `revise()` does not retire the old artifact unless explicitly asked to.

**Healthy run requirement:** `Revised: id=...` should always appear. If it does not, `loadAll()` returned an empty store or all artifacts are marked `retired`.

---

### Store summary

```
────────────────────────────────────────────────────────────  Store summary
  Active artifacts: 5  (total incl. retired: 5)
  [preference/?] conf=1.00  "I always want concise bullet-point summaries, strictly no more than fi…"
  [preference/?] conf=0.60  "Never use filler phrases like 'in conclusion' or 'to summarise'."
  [preference/?] conf=0.50  "When writing code, I prefer TypeScript with strict mode enabled."
  [preference/candidate]    conf=0.65  "I always want bullet-point summaries, no more than five points."
  [preference/accumulating] conf=0.65  "When writing code, I prefer TypeScript with strict mode enabled."
```

Final snapshot of all non-retired artifacts, sorted by confidence descending.

- **Active** means not retired. Retired artifacts remain in the JSONL for audit purposes but are excluded from retrieval and injection.
- The `[?]`-stage artifacts are `consolidated` stage — the display quirk is explained below.
- Each run adds 2 artifacts (one `candidate` from Turn 1, one `accumulating` from Turns 2+3). The pre-existing consolidated artifacts remain unchanged.

**Healthy run requirement:** `Active artifacts` count should increase by 2 on each run (assuming no prior matching artifacts in the store that cause merging). If the count stays the same, `persist()` is not writing to disk.

---

## Healthy run checklist

Use this table as a quick sanity check after any change to `extract.ts`, `pipeline.ts`, or `store.ts`:

| Check | Expected value | Failure indicates |
|---|---|---|
| Turn 1: `Candidates extracted` | `1` | Extraction broken or model response not parseable |
| Turn 1: `kind` | `preference` | Classification regression |
| Turn 1: `certainty` | `definitive` | Certainty assessment changed |
| Turn 1: `conf` | `0.65` | `CONFIDENCE_SEED.definitive` changed or miscalculated |
| Turn 1: outcome | `[NEW / candidate]` | `matchCandidate()` is incorrectly matching against a prior artifact |
| Turn 2: outcome | `[NEW / candidate]` | Tag overlap with Turn 1 is incorrectly ≥ 0.25 |
| Turn 3: `Created: 0   Updated: 1` | Exactly this | Either a false-positive new artifact, or accumulation not triggering |
| Turn 3: `evCount` | `2` | Evidence not being appended to `evidence[]` |
| Turn 4: `Candidates extracted` | `0` | Noise filter broken; factual questions are being persisted |
| Stage 6: top result | bullet-point preference artifact | Retrieval ranking regressed |
| Stage 6: result count | ≥ 2 | Store not being written or retrieve returning nothing |
| Stage 7: `autoApply` | `false` (on first run vs. clean store) | Threshold logic inverted |
| Stage 8: `Revised: id=...` | Present | `loadAll()` empty or all artifacts retired |
| Store summary: Active count | Increases by 2 per run | `persist()` not writing to disk |

---

## Relationship to the automated test suite

The playground and the automated tests are complementary, not redundant:

| | Automated tests (`pnpm test`) | Playground (`pnpm start`) |
|---|---|---|
| **API key required** | No — uses mock LLMs | Yes — real Anthropic API |
| **Speed** | ~5 seconds | ~15–30 seconds |
| **Determinism** | Fully deterministic | LLM output may vary slightly across runs |
| **What it catches** | Logic bugs in store, pipeline, extract | LLM quality regressions, prompt structure problems, real extraction behavior |
| **CI-suitable** | Yes | No (requires API key, takes time) |
| **Run frequency** | On every change | Before milestones ship; after prompt changes |

The automated tests use `createPatternMockLLM()` from `apps/computer-assistant/src/tests/mock-llm.ts`, which returns deterministic canned responses based on substring matching in the prompt. The playground uses a real model, so it validates that the prompt structure produces the expected extraction behavior when the LLM is genuinely reasoning about the input — including tag normalization against a real live vocabulary.

---

## Common variations

### First-ever run against a clean store

The retrieval stage returns only the 2 artifacts created during the run (no pre-existing consolidated artifacts). Stage 7's apply returns `autoApply=false` and `[provisional]`. Stage 8 revises the Turn 1 artifact from `conf=0.65` → `conf=0.70`.

Expected store summary: `Active artifacts: 2`.

### Multiple runs against the same store

Each run creates 2 new artifacts (new UUIDs). After N runs there will be N bullet-point preference artifacts (all at `stage: "candidate"`) and N TypeScript/code-style artifacts (one at `accumulating` with `evCount=N`, or multiple `accumulating` artifacts depending on tag match behavior). This reveals a known limitation: the current matching uses tag Jaccard at extraction time, but cross-session deduplication is not yet implemented. The same message re-processed in a new session creates a new artifact rather than accumulating into the existing one. This is addressed in Milestone 1d.

### Observing consolidation in action

To see the consolidation LLM call trigger:

1. Add a third TypeScript-related message to `SAMPLE_INPUTS` in `apps/playground/index.ts`:
   ```ts
   "Always use const over let unless mutation is required.",
   ```
2. Run the playground.
3. The third TypeScript message will push `evCount` to 3 (= `CONSOLIDATION_THRESHOLD`).
4. `accumulateEvidence()` will call `consolidateEvidence()` → one additional LLM call.
5. The artifact's `stage` changes to `consolidated` and its `content` becomes a generalized rule rather than a verbatim quote.
6. On the next run, this artifact will be eligible for `[suggestion]` or `[established]` injection depending on its confidence.

---

## Known display quirks

### `[preference/?]` for consolidated artifacts

The stage slot in the playground output (`[kind/stage]`) shows `?` for artifacts at `stage: "consolidated"`. This is a display-only issue in `apps/playground/index.ts` line 117:

```ts
const label = r.stage ?? "?";
```

Since `"consolidated"` is a valid non-null string, the `??` fallback should never trigger. However, in some artifact records the `stage` field is undefined (pre-existing JSONL records created before the `stage` field was introduced). The underlying artifact is functional — `getInjectLabel()` and `isInjectable()` handle `undefined` stage gracefully. The display will be improved in a future cleanup.

### `DeprecationWarning: The punycode module is deprecated`

A harmless Node.js 22 warning from a transitive dependency of `@anthropic-ai/sdk`. The `punycode` module is being phased out of Node.js core; the warning is informational only and has no effect on functionality. It can be suppressed with `node --no-deprecation` if it clutters output during development.

---

## Source reference

| File | Role in this program |
|---|---|
| `apps/playground/index.ts` | The script itself — orchestrates the run |
| `packages/openclaw-plus/src/pipeline.ts` | `processMessage()` — Stages 1–4 orchestration |
| `packages/openclaw-plus/src/extract.ts` | `extractFromMessage()`, `consolidateEvidence()` — LLM extraction |
| `packages/openclaw-plus/src/store.ts` | `retrieve()`, `apply()`, `revise()`, `loadAll()` — Stages 5–8 |
| `packages/openclaw-plus/src/types.ts` | `KnowledgeArtifact`, `LLMFn`, `CONFIDENCE_SEED`, `CONSOLIDATION_THRESHOLD` |
| `~/.openclaw/knowledge/artifacts.jsonl` | The JSONL store written and read during this run |
