# 02 — Extraction Benchmark (Stage 1)

**Pipeline stage:** Stage 1 — Extract
**Module:** `packages/openclaw-plus/src/extract.ts` → `extractFromMessage()`
**Implementation status:** ✅ Implemented
**Automated coverage:** ✅ Unit tests (mock LLM) + live LLM benchmark

---

## Purpose

This benchmark verifies that the LLM extraction call correctly distinguishes persistable knowledge from noise, assigns the right kind and certainty, and produces well-formed tags — across a range of input styles, knowledge types, and natural languages.

Stage 1 is the system's front door. Everything downstream (matching, accumulation, injection) operates on what extraction produces. A false positive here (extracting from noise) pollutes the store permanently. A false negative (missing genuine knowledge) means the system fails to learn. A miscategorized kind or certainty cascades into incorrect confidence seeding and injection gating.

## Design rationale this benchmark validates

- **Language-agnostic extraction**: The system delegates all natural language understanding to the LLM. No English-specific heuristics appear in the code. This benchmark validates that the extraction prompt produces correct results for Chinese, Spanish, and other languages with the same structure as English.
- **Six-kind taxonomy**: The LLM assigns one of six kind values. This benchmark validates that the taxonomy is expressive enough to cover real inputs without ambiguity, and that the LLM's classification is consistent.
- **Certainty seeding**: Certainty (`definitive` / `tentative` / `uncertain`) seeds the artifact's initial confidence. This benchmark validates that the LLM correctly reads hedging and strength from phrasing.
- **Noise rejection**: The most important failure mode is over-extraction — persisting things that aren't reusable knowledge. This benchmark validates that the rejection rate is high for non-knowledge inputs.

---

## Metrics

| Metric | Definition | Target |
|---|---|---|
| **Extraction precision** | Fraction of extracted candidates that are genuinely persistable knowledge | ≥ 0.90 |
| **Extraction recall** | Fraction of knowledge-bearing inputs that produce ≥ 1 candidate | ≥ 0.90 |
| **Noise rejection rate** | Fraction of non-knowledge inputs that produce 0 candidates | ≥ 0.95 |
| **Kind accuracy** | Fraction of extracted candidates with correct kind | ≥ 0.85 |
| **Certainty accuracy** | Fraction with correct certainty (definitive/tentative/uncertain) | ≥ 0.85 |
| **Tag coverage** | Fraction of cases where ≥ 1 expected tag appears | ≥ 0.90 |

Targets are aspirational starting points. They will be revised as we accumulate empirical results from live LLM runs.

---

## Test cases

### Group EX-A: Positive extraction (knowledge should be extracted)

#### EX-A-1: Definitive preference — output format
| Field | Value |
|---|---|
| **ID** | EX-A-1 |
| **Input** | `"I always want bullet-point summaries, no more than five points."` |
| **Expected kind** | `preference` |
| **Expected certainty** | `definitive` ("always" is unhedged) |
| **Expected tags (any)** | `bullet-points`, `summary-format`, `output-style`, `response-length` |
| **Pass criterion** | 1 candidate; kind = preference; certainty = definitive; ≥1 expected tag present |
| **Automated** | ✅ `extraction.test.ts: "extracts a strong preference from English input"` |
| **Live scenario** | `benchmarks/scenarios.ts: pref-1` |

#### EX-A-2: Definitive preference — code style
| Field | Value |
|---|---|
| **ID** | EX-A-2 |
| **Input** | `"When writing code, always use TypeScript with strict mode enabled."` |
| **Expected kind** | `preference` |
| **Expected certainty** | `definitive` |
| **Expected tags (any)** | `typescript`, `strict-mode`, `code-style` |
| **Pass criterion** | 1 candidate; kind = preference or convention; ≥1 expected tag present |
| **Automated** | ✅ `extraction.test.ts: "normalizes tags to lowercase-hyphenated format"` |
| **Live scenario** | `benchmarks/scenarios.ts: pref-2` |

#### EX-A-3: Convention — alias definition
| Field | Value |
|---|---|
| **ID** | EX-A-3 |
| **Input** | `"When I say 'gh', I mean https://github.com"` |
| **Expected kind** | `convention` |
| **Expected certainty** | `definitive` |
| **Expected tags (any)** | `alias`, `github`, `url-mapping` |
| **Pass criterion** | 1 candidate; kind = convention or fact; ≥1 expected tag present |
| **Automated** | ✅ `extraction.test.ts: "extracts a convention/fact (alias definition)"` |
| **Live scenario** | `benchmarks/scenarios.ts: fact-1` |

#### EX-A-4: Procedure — step-by-step
| Field | Value |
|---|---|
| **ID** | EX-A-4 |
| **Input** | `"To deploy: run pnpm build, then git push origin main, then notify the team on Slack."` |
| **Expected kind** | `procedure` |
| **Expected certainty** | `definitive` |
| **Expected tags (any)** | `deployment`, `workflow`, `build-process` |
| **Pass criterion** | 1 candidate; kind = procedure; ≥1 expected tag present |
| **Automated** | ✅ `extraction.test.ts: "extracts a procedure correctly"` |
| **Live scenario** | `benchmarks/scenarios.ts: proc-1` |

#### EX-A-5: Judgment — quality criterion
| Field | Value |
|---|---|
| **ID** | EX-A-5 |
| **Input** | `"Good code is readable first, then efficient. Never sacrifice clarity for performance."` |
| **Expected kind** | `judgment` |
| **Expected tags (any)** | `code-quality`, `readability`, `code-style` |
| **Pass criterion** | 1 candidate; kind = judgment or strategy; ≥1 expected tag present |
| **Automated** | ⚠️ Covered in live scenario (`judge-1`) but not in unit tests |
| **Live scenario** | `benchmarks/scenarios.ts: judge-1` |

#### EX-A-6: Tentative certainty — hedged preference
| Field | Value |
|---|---|
| **ID** | EX-A-6 |
| **Input** | `"I usually prefer shorter responses, maybe 3-5 sentences."` |
| **Expected kind** | `preference` |
| **Expected certainty** | `tentative` ("usually", "maybe") |
| **Expected tags (any)** | `output-style`, `conciseness` |
| **Pass criterion** | 1 candidate; certainty = tentative; confidence seeded at 0.35 |
| **Automated** | ✅ `extraction.test.ts: "assigns tentative certainty for hedged statements"` |
| **Live scenario** | `benchmarks/scenarios.ts: tent-1` |

#### EX-A-7: Uncertain certainty — speculative
| Field | Value |
|---|---|
| **ID** | EX-A-7 |
| **Input** | `"I'm not sure, but I think I might prefer dark mode for the UI?"` |
| **Expected certainty** | `uncertain` |
| **Pass criterion** | certainty = uncertain; confidence seeded at 0.15 |
| **Automated** | ⚠️ Certainty seed tested in `candidateToArtifact` test; full extraction not covered |
| **Live scenario** | `benchmarks/scenarios.ts: tent-2` |

---

### Group EX-B: Noise rejection (nothing should be extracted)

#### EX-B-1: Simple question
| Field | Value |
|---|---|
| **ID** | EX-B-1 |
| **Input** | `"What time is it?"` |
| **Pass criterion** | 0 candidates |
| **Automated** | ✅ `extraction.test.ts: "returns empty array for non-persistable messages"` |
| **Live scenario** | `benchmarks/scenarios.ts: none-1` |

#### EX-B-2: One-off command
| Field | Value |
|---|---|
| **ID** | EX-B-2 |
| **Input** | `"Open README.md for me"` |
| **Pass criterion** | 0 candidates — this is a request, not a reusable rule |
| **Automated** | ⚠️ Not explicitly unit-tested; covered in live scenario |
| **Live scenario** | `benchmarks/scenarios.ts: none-2` |

#### EX-B-3: Greeting
| Field | Value |
|---|---|
| **ID** | EX-B-3 |
| **Input** | `"Hello! How are you?"` |
| **Pass criterion** | 0 candidates |
| **Automated** | ⚠️ Not explicitly unit-tested; covered in live scenario |
| **Live scenario** | `benchmarks/scenarios.ts: none-3` |

#### EX-B-4: Calculation request
| Field | Value |
|---|---|
| **ID** | EX-B-4 |
| **Input** | `"What is 15% of 240?"` |
| **Pass criterion** | 0 candidates |
| **Automated** | ⚠️ Not explicitly unit-tested; covered in live scenario |
| **Live scenario** | `benchmarks/scenarios.ts: none-4` |

#### EX-B-5: Acknowledgement
| Field | Value |
|---|---|
| **ID** | EX-B-5 |
| **Input** | `"OK thanks"` |
| **Pass criterion** | 0 candidates |
| **Automated** | ⚠️ Not unit-tested; covered in live scenario |
| **Live scenario** | `benchmarks/scenarios.ts: none-5` |

---

### Group EX-C: Language agnosticism

#### EX-C-1: Chinese preference
| Field | Value |
|---|---|
| **ID** | EX-C-1 |
| **Input** | `"我总是希望摘要使用项目符号，不超过五点。"` |
| **Expected kind** | `preference` |
| **Expected certainty** | `definitive` |
| **Expected tags (any)** | `bullet-points`, `summary-format` (in English) |
| **Content language** | Chinese (preserved verbatim) |
| **Pass criterion** | 1 candidate; kind, certainty, tags in English; content contains `摘要` or `项目符号` |
| **Automated** | ✅ `extraction.test.ts: "handles non-English input (Chinese)"` |
| **Live scenario** | `benchmarks/scenarios.ts: lang-1` |

#### EX-C-2: Spanish convention
| Field | Value |
|---|---|
| **ID** | EX-C-2 |
| **Input** | `"Cuando digo 'gh' me refiero a https://github.com"` |
| **Expected tags (any)** | `alias`, `github` (in English) |
| **Pass criterion** | 1 candidate; tags in English; content in Spanish |
| **Automated** | ❌ Not yet unit-tested |
| **Live scenario** | `benchmarks/scenarios.ts: lang-2` |

#### EX-C-3: Mixed-language input
| Field | Value |
|---|---|
| **ID** | EX-C-3 |
| **Input** | `"Please always use TypeScript. 我喜欢严格模式。"` |
| **Pass criterion** | 1 candidate capturing the combined TypeScript + strict mode preference; tags in English |
| **Automated** | 🔲 Not yet specified |
| **Live scenario** | 🔲 Not yet in scenarios.ts |

---

### Group EX-D: Robustness

#### EX-D-1: Malformed LLM response
| Field | Value |
|---|---|
| **ID** | EX-D-1 |
| **Input** | Any message; LLM returns invalid JSON |
| **Pass criterion** | 0 candidates; no exception thrown |
| **Automated** | ✅ `extraction.test.ts: "gracefully handles malformed LLM response"` |

#### EX-D-2: Markdown-fenced JSON response
| Field | Value |
|---|---|
| **ID** | EX-D-2 |
| **Input** | Any message; LLM wraps response in ` ```json ` fence |
| **Pass criterion** | Candidates extracted correctly (fence stripped by `parseJSON()`) |
| **Automated** | ✅ `extraction.test.ts: "gracefully handles LLM response wrapped in markdown code fence"` |

#### EX-D-3: Empty input
| Field | Value |
|---|---|
| **ID** | EX-D-3 |
| **Input** | `""` |
| **Pass criterion** | 0 candidates; no exception thrown |
| **Automated** | ✅ `extraction.test.ts: "returns empty array for empty input"` |

#### EX-D-4: Multiple candidates in one message
| Field | Value |
|---|---|
| **ID** | EX-D-4 |
| **Input** | `"Always use bullet points in summaries. Our API endpoint is https://api.example.com/v2."` |
| **Pass criterion** | 2 candidates: one preference (output format), one fact (API endpoint) |
| **Automated** | 🔲 Not yet specified |
| **Live scenario** | 🔲 Not yet in scenarios.ts |

---

## Gaps and open questions

1. **Strategy kind coverage**: No test case currently exercises `kind: "strategy"`. Need a scenario like *"When debugging, isolate variables before forming hypotheses."*
2. **Multiple candidates from one message** (EX-D-4): The extraction prompt supports multiple candidates but no test verifies this.
3. **Spanish and other languages** (EX-C-2, EX-C-3): Live scenarios exist but no unit tests.
4. **Long messages**: Extraction behavior on long messages (> 500 tokens) has not been characterized. The LLM may extract too many or too few candidates.
5. **Certainty precision**: The boundary between `tentative` and `uncertain` is fuzzy. We lack a clear specification of what phrases map to each.

---

## Automated evaluation notes

The live benchmark in `apps/computer-assistant/benchmarks/index.ts` runs all `EXTRACTION_SCENARIOS` with a real Anthropic model and reports:
- Per-scenario pass/fail on kind, certainty, and tag presence
- Aggregate precision, recall, and F1 across the scenario set

To add new scenarios, append to `EXTRACTION_SCENARIOS` in `apps/computer-assistant/benchmarks/scenarios.ts`. Each scenario maps directly to a test case in this document.

**Future**: Per-case certainty evaluation could be automated by checking `confidence` values at artifact creation time — `definitive` always seeds at 0.65, `tentative` at 0.35, `uncertain` at 0.15. This is already verified by unit tests on `candidateToArtifact()`.
