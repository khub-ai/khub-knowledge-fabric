# 08 — End-to-End Scenarios Benchmark

**Pipeline stages:** All stages (1–8)
**Modules:** `pipeline.ts`, `extract.ts`, `store.ts`
**Implementation status:** ✅ Core pipeline; hook wiring pending (Milestones 1c/1d)
**Automated coverage:** ⚠️ Partial (mock scenarios); live LLM scenarios not yet automated

---

## Purpose

This benchmark verifies the system's behavior across realistic multi-turn learning scenarios — the kind of use cases that motivated the project. Each scenario exercises the full pipeline from raw user input through extraction, accumulation, consolidation, persistence, retrieval, and injection.

Individual stage benchmarks verify that each component works correctly in isolation. This benchmark verifies that the components work correctly *together*, and that the integrated system produces sensible behavior across the realistic patterns an actual user would generate.

## Design rationale this benchmark validates

- **Progressive learning**: The system should learn incrementally — a single message produces a `candidate`, repeated similar messages produce an `accumulating` artifact, and eventually a `consolidated` rule. No single session should be required for full learning.
- **Knowledge specificity**: The system should learn what the user actually said, not a generic paraphrase. The content preserved in `evidence[]` must match the user's original phrasing (possibly in their original language).
- **Noise tolerance**: Real conversations contain a mix of knowledge-bearing and non-knowledge-bearing messages. The system must accumulate only from the former without polluting the store with the latter.
- **Knowledge reuse**: Once consolidated, knowledge should be retrieved and injected correctly in subsequent sessions — even if phrased differently in the query.
- **Graceful degradation**: If extraction or consolidation produces unexpected output, the pipeline should fail gracefully (empty result) rather than persist garbage.

---

## Scenarios

### Group EE-A: Single-session learning arc

#### EE-A-1: Preference learned in one session — three-message arc
| Field | Value |
|---|---|
| **ID** | EE-A-1 |
| **Session 1, Turn 1** | `"I always want bullet-point summaries."` → candidate artifact created |
| **Session 1, Turn 2** | `"Please use bullet points for all output."` → accumulating (evCount=2) |
| **Session 1, Turn 3** | `"Always use bullet points, max 5 items."` → consolidated; generalized rule stored |
| **Session 2, Query** | `retrieve("summary format")` → consolidated rule retrieved |
| **Pass criterion** | After turn 3: stage=`consolidated`, content is a generalized English rule; retrieved in session 2 |
| **Automated** | ✅ `pipeline.test.ts: "consolidates after 3 observations"` (mock); live LLM run via playground |

#### EE-A-2: Mixed knowledge and noise in one session
| Field | Value |
|---|---|
| **ID** | EE-A-2 |
| **Messages** | `"What time is it?"`, `"I always use TypeScript."`, `"Open the file."`, `"Use strict mode always."` |
| **Pass criterion** | Exactly 1 artifact created (TypeScript preference); messages 1, 3 produce 0 candidates; message 4 accumulates |
| **Automated** | ✅ Partially (individual turns tested in `pipeline.test.ts`; multi-turn sequence not tested as unit) |

#### EE-A-3: Two unrelated preferences, separate artifacts
| Field | Value |
|---|---|
| **ID** | EE-A-3 |
| **Messages** | `"Always use bullet points."`, `"Our API is at https://api.example.com"` |
| **Pass criterion** | 2 separate artifacts created; no merging (tag overlap = 0) |
| **Automated** | ✅ `scenarios.test.ts: "two separate preferences create two artifacts"` |

---

### Group EE-B: Cross-session persistence

#### EE-B-1: Knowledge persists across simulated sessions
| Field | Value |
|---|---|
| **ID** | EE-B-1 |
| **Session 1** | `processMessage("I always want bullet-point summaries", llm)` |
| **Session 2 (new process, same store path)** | `retrieve("summary format")` |
| **Pass criterion** | Artifact from session 1 is retrieved in session 2; all fields intact |
| **Automated** | ✅ `scenarios.test.ts: "PIL learning — alias definition persists for retrieval"` |

#### EE-B-2: Store grows monotonically across sessions
| Field | Value |
|---|---|
| **ID** | EE-B-2 |
| **Action** | Run playground twice against the same store |
| **Pass criterion** | Active artifact count increases by 2 on each run |
| **Automated** | ⚠️ Observed in playground output; not formalized as a test |

---

### Group EE-C: Noise filtering

These cases verify that the pipeline's noise filter (Stage 1 returning 0 candidates) prevents non-knowledge from reaching the store.

#### EE-C-1: Factual question produces no artifact
| Field | Value |
|---|---|
| **ID** | EE-C-1 |
| **Input** | `"What time is it?"` |
| **Pass criterion** | `processMessage()` returns `created: 0, updated: 0`; store unchanged |
| **Automated** | ✅ `pipeline.test.ts: "produces no artifact for non-knowledge input"` |

#### EE-C-2: One-off command produces no artifact
| Field | Value |
|---|---|
| **ID** | EE-C-2 |
| **Input** | `"Open README.md for me"` |
| **Pass criterion** | 0 candidates; store unchanged |
| **Automated** | ✅ `scenarios.test.ts: "does not learn from simple task requests"` |

#### EE-C-3: Greeting produces no artifact
| Field | Value |
|---|---|
| **ID** | EE-C-3 |
| **Input** | `"Hello! How are you?"` |
| **Pass criterion** | 0 candidates |
| **Automated** | ⚠️ Covered by live scenario (`none-3`); not in unit tests |

---

### Group EE-D: Language agnosticism end-to-end

#### EE-D-1: Chinese input → English tags → retrieval in English query
| Field | Value |
|---|---|
| **ID** | EE-D-1 |
| **Input** | `"我总是希望摘要使用项目符号"` (Chinese: "I always want summaries to use bullet points") |
| **Expected** | Artifact with English tags `[bullet-points, summary-format]`; content in Chinese |
| **Retrieval** | `retrieve("bullet point summary")` retrieves the artifact via tag match |
| **Pass criterion** | Tags in English; content in Chinese; retrieval succeeds |
| **Automated** | ✅ `extraction.test.ts: "handles non-English input (Chinese)"` + `scenarios.test.ts` |

#### EE-D-2: Same preference stated in two languages → accumulates
| Field | Value |
|---|---|
| **ID** | EE-D-2 |
| **Turn 1** | `"I always want bullet-point summaries."` (English) → candidate, tags `[bullet-points, summary-format]` |
| **Turn 2** | `"我总是希望摘要使用项目符号"` (Chinese, same meaning) → should accumulate into turn 1's artifact |
| **Pass criterion** | `created: 0, updated: 1`; evidence contains both observations |
| **Automated** | 🔲 Not yet specified — depends on LLM assigning the same tags to both |
| **Note** | This is the strongest test of language agnosticism. Tag normalization must be consistent across languages for this to pass. |

---

### Group EE-E: Contradiction and revision

#### EE-E-1: User corrects a preference
| Field | Value |
|---|---|
| **ID** | EE-E-1 |
| **Setup** | Artifact: `"Always use bullet points for summaries."` (confidence=0.65) |
| **User says** | `"Actually, I prefer numbered lists, not bullet points."` |
| **Expected** | New extraction: same tags, contradicting content; system should either accumulate (if matching) or create new artifact |
| **Pass criterion** | New observation captured; the contradiction surfaced (via conflicting artifacts or via revision) |
| **Automated** | 🔲 Not yet specified — conflict detection is a Phase 2 feature |
| **Note** | Currently the system creates a second artifact. Conflict detection and merge/retire logic is deferred to Phase 2. |

#### EE-E-2: Explicit revision updates content
| Field | Value |
|---|---|
| **ID** | EE-E-2 |
| **Setup** | Artifact with content A |
| **Action** | `revise(artifact, { content: B, confidence: 0.90 })` |
| **Pass criterion** | `loadAll()` returns artifact with content B, confidence=0.90, id unchanged |
| **Automated** | ✅ `store.test.ts: "revise — updates content in-place"` |

---

## Open questions

1. **EE-D-2 (cross-language accumulation)**: This is the key test of whether language-agnostic extraction produces stable tags. If the LLM assigns `[bullet-points, summary-format]` to the Chinese input but `[bullet-format, summaries]` to the English one, Jaccard overlap may fall below threshold. We need to run live LLM experiments to characterize tag stability across languages.
2. **EE-E-1 (contradiction)**: Currently the system does not detect contradictions. A user can state two conflicting preferences and both will be stored and potentially injected. This is acceptable for Milestone 1b but must be addressed in Phase 2.
3. **EE-B-2 (store growth)**: Without cross-session matching (Milestone 1d), the same message run twice creates two separate artifacts. The playground demonstrates this. The benchmark should capture this as a known limitation, not a failure.
4. **Scale scenarios**: All scenarios use small stores (< 10 artifacts). The behavior at 1,000 or 10,000 artifacts — retrieval quality, consolidation behavior, apply decision time — has not been characterized.

---

## Automated evaluation notes

The scenario tests in `apps/computer-assistant/src/tests/scenarios.test.ts` use mock LLMs and cover EE-A-3, EE-B-1, EE-C-1, EE-C-2, and EE-D-1.

**To add (live LLM)**:
- EE-A-1 end-to-end three-message arc with real Anthropic model
- EE-D-2 cross-language accumulation stability
- EE-A-2 mixed session with correct artifact count

**Evaluation signal for future self-improvement**: The scenarios in this document can be expressed as (input sequence → expected store state) pairs. An automated evaluator could run the input sequence through the pipeline, inspect the resulting store, and score the outcome against the expected state. This score is a direct signal for tuning extraction prompts, consolidation prompts, and matching thresholds.
