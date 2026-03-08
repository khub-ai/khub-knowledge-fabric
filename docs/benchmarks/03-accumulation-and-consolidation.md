# 03 — Accumulation and Consolidation Benchmark (Stages 2–3)

| | |
|---|---|
| **Pipeline stages** | Stage 2 — Match; Stage 3 — Resolve (accumulate or create) |
| **Modules** | `packages/openclaw-plus/src/store.ts` → `matchCandidate()`, `accumulateEvidence()`; `packages/openclaw-plus/src/extract.ts` → `consolidateEvidence()` |
| **Implementation status** | ✅ Implemented |
| **Automated coverage** | ✅ Unit tests (mock LLM) |

---

## Purpose

This benchmark verifies that the system correctly decides whether a newly extracted candidate is novel (warranting a new artifact) or related to an existing one (warranting evidence accumulation), and that when enough evidence accumulates, the consolidation LLM call produces a well-formed generalized rule.

This is the system's core learning mechanism. Without correct matching, the store fills with redundant artifacts that fragment the same underlying preference across multiple records. Without correct accumulation, the system never reaches the consolidation threshold that promotes `candidate` knowledge to `consolidated` (trusted, injectable) form.

## Design rationale this benchmark validates

- **Tag-based matching over content matching**: Matching is done on `tags[]` using Jaccard overlap (threshold 0.25), not on raw content. This means semantically related messages (even phrased differently) are recognized as belonging to the same cluster. The threshold is a critical hyperparameter — too low causes false merges; too high causes false splits.
- **Three-stage lifecycle**: The artifact moves through `candidate` → `accumulating` → `consolidated`. This progression is the system's confidence mechanism — it never auto-applies knowledge seen only once.
- **Consolidation as generalization**: The consolidation LLM call receives all raw evidence strings (verbatim, possibly multilingual) and produces a single generalized English rule. This is qualitatively different from averaging — the output should capture what the observations have in common, not merely summarize them.
- **Stage preservation**: Once consolidated, an artifact must not be demoted back to `accumulating` by further evidence. This is a critical correctness property.

---

## Metrics

| Metric | Definition | Target |
|---|---|---|
| **Match precision** | Fraction of accumulation decisions where the matched artifact is genuinely the right one | ≥ 0.90 |
| **Match recall** | Fraction of truly-related message pairs that are accumulated (not duplicated) | ≥ 0.85 |
| **False merge rate** | Fraction of unrelated message pairs that are incorrectly accumulated | ≤ 0.05 |
| **Consolidation trigger rate** | Fraction of artifacts that correctly promote at threshold | 1.0 (deterministic) |
| **Consolidation quality** | Does the consolidated output generalize correctly? (LLM-evaluated) | 🔲 Metric TBD |
| **Stage preservation** | Consolidated artifacts never re-demoted | 1.0 (deterministic, unit-tested) |

---

## Test cases

### Group AC-A: Matching — accumulate vs. create

#### AC-A-1: Related messages accumulate
| Field | Value |
|---|---|
| **ID** | AC-A-1 |
| **Setup** | Artifact in store: `preference`, tags `[typescript, code-style, language-preference]`, stage `candidate` |
| **Input** | `"Use async/await rather than raw promises wherever possible."` |
| **Expected extraction** | kind=preference, tags include `[async-await, code-style, typescript]` |
| **Expected match** | Jaccard overlap with stored artifact ≥ 0.25 → accumulate |
| **Pass criterion** | `processMessage()` returns `created: 0, updated: 1`; stored artifact `evCount` = 2, stage = `accumulating` |
| **Automated** | ✅ `pipeline.test.ts: "accumulates into existing artifact on second related message"` |

#### AC-A-2: Unrelated messages do not accumulate
| Field | Value |
|---|---|
| **ID** | AC-A-2 |
| **Setup** | Artifact in store: `preference`, tags `[typescript, code-style]` |
| **Input** | `"I always want bullet-point summaries."` |
| **Expected extraction** | tags include `[bullet-points, summary-format]` — no overlap with `[typescript, code-style]` |
| **Pass criterion** | `processMessage()` returns `created: 1, updated: 0`; new artifact created with separate id |
| **Automated** | ✅ `pipeline.test.ts: "creates new artifact for novel knowledge"` (indirectly) |

#### AC-A-3: Jaccard threshold boundary — above threshold
| Field | Value |
|---|---|
| **ID** | AC-A-3 |
| **Setup** | Artifact tags: `[code-style, output-format, brevity]`; candidate tags: `[code-style, conciseness]` |
| **Jaccard** | 1/4 = 0.25 → meets threshold |
| **Pass criterion** | Accumulates (not creates) |
| **Automated** | ⚠️ Threshold logic tested in `store.test.ts: matchCandidate`; boundary value not explicitly tested |

#### AC-A-4: Jaccard threshold boundary — below threshold
| Field | Value |
|---|---|
| **ID** | AC-A-4 |
| **Setup** | Artifact tags: `[typescript, strict-mode, code-style]`; candidate tags: `[summary-format, bullet-points]` |
| **Jaccard** | 0/5 = 0 → below threshold |
| **Pass criterion** | Creates new artifact |
| **Automated** | ✅ `store.test.ts: matchCandidate — returns null when no artifacts match` |

#### AC-A-5: Same-kind constraint — different kinds do not match
| Field | Value |
|---|---|
| **ID** | AC-A-5 |
| **Setup** | Artifact in store: kind=`convention`, tags `[alias, github]` |
| **Input candidate** | kind=`fact`, tags `[alias, github]` (same tags, different kind) |
| **Pass criterion** | No match returned — `matchCandidate()` only matches within the same kind |
| **Automated** | 🔲 Not explicitly tested; implicit in `matchCandidate()` implementation |

---

### Group AC-B: Evidence accumulation

#### AC-B-1: Evidence appended correctly
| Field | Value |
|---|---|
| **ID** | AC-B-1 |
| **Setup** | Artifact with `evidence: ["I always want bullet-point summaries"]`, `evCount: 1` |
| **Action** | Call `accumulateEvidence(artifact, "Please use bullet points for all output", llm)` |
| **Pass criterion** | Returned artifact: `evCount: 2`; `evidence` array has 2 entries; second entry matches new observation |
| **Automated** | ✅ `store.test.ts: "increments evidenceCount and appends evidence"` |

#### AC-B-2: Stage advances from candidate to accumulating
| Field | Value |
|---|---|
| **ID** | AC-B-2 |
| **Setup** | Artifact at stage `candidate`, `evCount: 1` |
| **Action** | Accumulate one more observation |
| **Pass criterion** | Stage = `accumulating`; `evCount: 2` |
| **Automated** | ✅ `store.test.ts: "promotes to accumulating stage on second observation"` |

#### AC-B-3: Consolidation triggers at threshold
| Field | Value |
|---|---|
| **ID** | AC-B-3 |
| **Setup** | Artifact at stage `accumulating`, `evCount: 2`; `CONSOLIDATION_THRESHOLD = 3` |
| **Action** | Accumulate a third observation |
| **Pass criterion** | Consolidation LLM call fires; stage promoted to `consolidated`; content replaced with generalized rule |
| **Automated** | ✅ `store.test.ts: "triggers consolidation at threshold"`, `pipeline.test.ts: "consolidates after 3 observations"` |

#### AC-B-4: Already-consolidated artifact is not demoted
| Field | Value |
|---|---|
| **ID** | AC-B-4 |
| **Setup** | Artifact at stage `consolidated`, `evCount: 3` |
| **Action** | Accumulate a fourth observation |
| **Pass criterion** | Stage remains `consolidated` (not demoted to `accumulating`) |
| **Automated** | ✅ `store.test.ts: "does not re-consolidate an already-consolidated artifact"` |
| **Note** | This was a real bug (fixed): `accumulateEvidence()` previously overwrote stage unconditionally |

#### AC-B-5: Configurable threshold
| Field | Value |
|---|---|
| **ID** | AC-B-5 |
| **Setup** | `CONSOLIDATION_THRESHOLD` env var = `2` |
| **Action** | Second observation on a `candidate` artifact |
| **Pass criterion** | Consolidation triggers at count 2 (not 3) |
| **Automated** | 🔲 Not yet tested with env var override |

---

### Group AC-C: Consolidation quality

#### AC-C-1: Output generalizes correctly from English observations
| Field | Value |
|---|---|
| **ID** | AC-C-1 |
| **Input observations** | `["I always want bullet-point summaries", "Please use bullet points for all output", "Always use bullet points, max 5 items"]` |
| **Expected output** | A generalized rule capturing: preference for bullet points, maximum 5 items |
| **Pass criterion** | Output is non-empty, length > 10 chars, contains semantic reference to bullet points |
| **Automated** | ✅ `extraction.test.ts: "returns a consolidated rule from multiple observations"` (mock) |
| **LLM evaluation** | Full quality assessment requires live LLM and human judgment |

#### AC-C-2: Output is in English regardless of input language
| Field | Value |
|---|---|
| **ID** | AC-C-2 |
| **Input observations** | Mixed: `["我总是希望摘要使用项目符号", "Please use bullet points", "Siempre quiero puntos"]` |
| **Pass criterion** | Consolidated output is in English |
| **Automated** | 🔲 Not yet tested; live LLM run required |

#### AC-C-3: Empty observations return empty string
| Field | Value |
|---|---|
| **ID** | AC-C-3 |
| **Input observations** | `[]` |
| **Pass criterion** | `consolidateEvidence()` returns `""` without calling LLM |
| **Automated** | ✅ `extraction.test.ts: "returns empty string for empty observations"` |

---

## Gaps and open questions

1. **AC-A-5 (same-kind constraint)**: The implementation filters by kind before computing Jaccard overlap, but this is not explicitly tested. A test should verify that two messages with identical tags but different kinds create separate artifacts.
2. **AC-A-3/AC-A-4 (boundary conditions)**: The 0.25 Jaccard threshold is a hyperparameter that has not been empirically validated. We should run a calibration study: sample pairs of messages, compute human judgments of "related / not related," and measure threshold accuracy.
3. **AC-B-5 (configurable threshold)**: The `CONSOLIDATION_THRESHOLD` env var is used in code but not tested with a non-default value.
4. **Consolidation quality (AC-C-1, AC-C-2)**: "Quality" of the consolidated rule is not automatable with deterministic tests. Options: (a) LLM-as-judge evaluation, (b) checklist of properties (non-empty, English, semantically related to inputs), (c) human review in the benchmark run.
5. **Cross-session matching**: Currently, `matchCandidate()` only runs at `processMessage()` time. Identical knowledge from two separate sessions creates two artifacts rather than accumulating. This is a known gap addressed in Milestone 1d.

---

## Automated evaluation notes

No live LLM benchmark scenario currently exercises matching and accumulation end-to-end with a real model. The `EXTRACTION_SCENARIOS` in `benchmarks/scenarios.ts` test extraction only.

**To add**: A `ACCUMULATION_SCENARIOS` array in `benchmarks/scenarios.ts` specifying pairs of messages that should accumulate, with the expected resulting `evCount` and stage. The benchmark runner would: (1) process message 1, (2) process message 2, (3) verify the store state.

Consolidation quality could be evaluated by an LLM-as-judge call: given the input observations and the consolidated output, ask a model to score coherence and generalization on a 1–5 scale.
