# Benchmarks and Walkthroughs

This directory contains the specification and instructional documentation for every runnable test program and benchmark in the PIL project.

---

## Purposes

These documents serve three distinct purposes, and every benchmark should be designed with all three in mind.

### 1. Inspection by interested parties

A reader — whether a potential adopter, a researcher, or a technical evaluator — should be able to open any benchmark document and understand:

- What specific question about the system this benchmark answers
- What the system must do correctly for the benchmark to pass
- What failure would mean for the system's practical usefulness
- Whether the design choices reflected in the benchmark represent a sound direction

Benchmarks are not just test harnesses. They are also a form of argument: "here is what we believe the system should do, here is how we verify it, and here is why it matters."

### 2. Developer communication

Benchmarks are the primary medium through which developers (human or AI) agree on what the system is supposed to do. Each document captures:

- The precise specification of a pipeline stage's correct behavior
- The test cases that define the boundary between passing and failing
- What is currently implemented, what is pending, and what is deliberately deferred
- Which module owns the behavior being tested
- Open questions about the design that are not yet resolved

When two developers disagree about whether a behavior is correct, the benchmark document should be the reference. When a developer proposes changing the system's behavior, the benchmark document is what must be updated first.

### 3. Foundation for automated self-evaluation

Every test case specified here is a candidate for inclusion in an automated evaluation pipeline. The goal is that over time:

- All benchmark cases have corresponding automated tests (unit, integration, or LLM-evaluated)
- Metrics are defined precisely enough that they can be computed without human judgment
- The evaluation pipeline can be run against any candidate system change to detect regressions
- Eventually, the evaluation results can be fed back into the system itself as training signal for self-improvement

Documents mark which cases are already automated, which are partially automated, and which require LLM-evaluation or human judgment.

---

## How to run automated benchmarks

The automated unit tests (mock LLMs, no API key required):

```bash
pnpm --filter @khub-ai/computer-assistant test
```

The live LLM benchmarks (Anthropic API key required, ~2–5 minutes):

```bash
cd apps/computer-assistant
pnpm benchmark
```

The playground end-to-end walkthrough (API key required, ~15–30 seconds):

```bash
cd apps/playground
pnpm start
```

To run against a clean store without affecting the shared artifact file:

```bash
KNOWLEDGE_STORE_PATH=/tmp/pil-bench.jsonl pnpm benchmark
```

---

## Document index

| # | Document | Pipeline stages | Implementation status | Automated? |
|---|---|---|---|---|
| 01 | [playground.md](./01-playground.md) | All stages (end-to-end) | ✅ Implemented | Partial (live LLM) |
| 02 | [extraction.md](./02-extraction.md) | Stage 1 — Extract | ✅ Implemented | ✅ Yes (mock + live) |
| 03 | [accumulation-and-consolidation.md](./03-accumulation-and-consolidation.md) | Stages 2–3 — Match, Resolve | ✅ Implemented | ✅ Yes (mock + live) |
| 04 | [injection-gating.md](./04-injection-gating.md) | Stage 4 — Decide | ✅ Implemented | ✅ Yes (mock) |
| 05 | [persistence.md](./05-persistence.md) | Stage 5 — Persist | ✅ Implemented | ✅ Yes (mock) |
| 06 | [retrieval.md](./06-retrieval.md) | Stage 6 — Retrieve | ✅ Implemented | ✅ Yes (mock + live) |
| 07 | [apply-and-revise.md](./07-apply-and-revise.md) | Stages 7–8 — Apply, Revise | ✅ Implemented | ✅ Yes (mock) |
| 08 | [end-to-end.md](./08-end-to-end.md) | All stages (scenarios) | ✅ Core; hooks pending | Partial (mock scenarios) |
| 09 | [openClaw-integration.md](./09-openClaw-integration.md) | Hook wiring | 🔄 Pending (1c/1d) | ❌ Not yet |
| 10 | [lmp-acronym.md](./10-lmp-acronym.md) | Semantic matching · Consolidation | ✅ Implemented | ✅ Yes (mock) |
| 11 | [financial-statements.md](./11-financial-statements.md) | Multi-artifact extraction · Retrieval · Revision | ✅ Implemented | ✅ Yes (mock) |
| 12 | [compile-procedure.md](./12-compile-procedure.md) | Recipe → program escalation · EXECUTABLE hint | ✅ Implemented | ✅ Yes (mock) |

---

## Conventions used in benchmark documents

### Test case format

Each test case has:
- **ID** — unique identifier, e.g. `EX-3` (extraction, case 3)
- **Input** — the exact message or state the system receives
- **Expected output** — what the system must produce
- **Pass criterion** — the precise condition that determines pass or fail
- **Automated coverage** — whether a corresponding unit test exists

### Coverage notation

| Symbol | Meaning |
|---|---|
| ✅ Full | All meaningful sub-cases are covered |
| ⚠️ Partial | Some sub-cases are covered; gaps are listed explicitly |
| 🔲 Placeholder | Specification written; no implementation or test yet |
| ❌ Not covered | Known gap; reason documented |

### Automated vs. human-evaluated

Some test cases can be verified programmatically (exact output match, type check, threshold comparison). Others require LLM evaluation (is this consolidation output a good generalization?) or human judgment. The distinction is marked per case.

---

## Relationship between benchmark documents and the automated test suite

| Benchmark doc | Corresponding test file(s) | Corresponding benchmark file |
|---|---|---|
| 02-extraction.md | `src/tests/extraction.test.ts` | `benchmarks/scenarios.ts` (EXTRACTION_SCENARIOS) |
| 03-accumulation-and-consolidation.md | `src/tests/store.test.ts`, `src/tests/pipeline.test.ts` | — |
| 04-injection-gating.md | `src/tests/store.test.ts` | — |
| 05-persistence.md | `src/tests/store.test.ts` | — |
| 06-retrieval.md | `src/tests/store.test.ts` | `benchmarks/scenarios.ts` (RETRIEVAL_SCENARIOS) |
| 07-apply-and-revise.md | `src/tests/store.test.ts` | — |
| 08-end-to-end.md | `src/tests/scenarios.test.ts`, `src/tests/pipeline.test.ts` | — |
| 09-openClaw-integration.md | — (not yet) | — (not yet) |
| 10-lmp-acronym.md | `src/tests/bench-lmp-acronym.test.ts` | — |
| 11-financial-statements.md | `src/tests/bench-financial-statements.test.ts` | — |
| 12-compile-procedure.md | `src/tests/bench-compile-procedure.test.ts` | — |

All test files are in `apps/computer-assistant/src/tests/`. All benchmark scenarios are in `apps/computer-assistant/benchmarks/`.
