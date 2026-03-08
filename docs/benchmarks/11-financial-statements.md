# Benchmark 11: Financial Statements — Multi-Artifact, Retrieval, and Revision

| # | Scenario | What it tests | Automated test | Run with |
|---|---|---|---|---|
| 11 | Financial statements | Multi-artifact extraction from a single exchange; cross-kind retrieval; in-place artifact revision | `bench-financial-statements.test.ts` | `pnpm test` |

## Scenario

This benchmark validates the scenario described in full in
[`docs/example-learning-in-action.md`](../example-learning-in-action.md).

A user manages financial statements across multiple institutions. Over five sessions an
agent progressively learns: a file-naming convention, the user's institution list, and a
monthly download procedure — then retrieves and applies that knowledge on demand, and
correctly revises it when circumstances change.

### Goals

1. **No over-extraction**: a single one-off task (Session 1) must not trigger artifact creation.
2. **Convention learning**: an explicit user confirmation ("Yes, always use that pattern") is the trigger, not the first occurrence.
3. **Multi-artifact from one exchange**: a single Session 3 exchange produces two independent artifacts (a `fact` and a `procedure`) in one extraction call.
4. **Natural language retrieval**: "time for statements" must surface the procedure, not the naming convention or institution fact.
5. **Revision**: `revise()` updates an artifact in place for minor changes (Jaccard ≥ 0.5), preserving the existing id and adding `revisedAt`.

---

## Why this is interesting

### Problem 1 — No spurious learning from one-off tasks

Session 1 asks the agent to rename a file. The agent completes the task, and PIL sees the
exchange. But a single isolated task should not create a convention — the naming pattern
has only been observed once and has not been confirmed. The mock LLM returns an empty
candidates list for this exchange, matching the expected behavior of a real extraction LLM
that understands the difference between episodic execution and reusable knowledge.

### Problem 2 — Multi-candidate extraction from a single turn

Session 3 is where the user both names their institutions *and* requests a checklist in a
single reply ("Good idea. Add Amex too — I always forget that one."). The extraction LLM
returns two candidates in one JSON response. The pipeline iterates over both independently:

- Candidate 1 (`fact` — institution list) → no existing `fact` artifacts → created
- Candidate 2 (`procedure` — monthly checklist) → no existing `procedure` artifacts → created

Because `matchCandidate` filters by `kind` first, the convention from Session 2 is never
compared against either new candidate. Both are stored cleanly.

### Problem 3 — Cross-kind retrieval scoring

Three artifacts exist after sessions 1–3:

| Kind | Tags |
|---|---|
| `convention` | `file-naming`, `financial-statements`, `naming-convention` |
| `fact` | `financial-institutions`, `accounts`, `chase`, `fidelity`, `amex` |
| `procedure` | `monthly-statements`, `download-procedure`, `financial-statements` |

Query `"time for statements"` tokenises to `["time", "for", "statements"]`.

- `procedure`: "statements" matches `monthly-statements` *and* `financial-statements` → 2/3 tag hits → score ≈ 0.44
- `convention`: "statements" matches `financial-statements` → 1/3 tag hits → score ≈ 0.30
- `fact`: no tag overlap → score below threshold

The procedure correctly ranks highest — the agent follows the right checklist.

### Problem 4 — In-place revision for minor content changes

When Amex closes and Schwab opens, the institution list changes from:

> Chase (checking), Fidelity (brokerage), **Amex** (credit card)

to:

> Chase (checking), Fidelity (brokerage), **Schwab** (brokerage)

Word-set Jaccard between the two contents ≈ **0.64** (7 shared words: *user, financial,
institutions, chase, checking, fidelity, brokerage* — out of a union of 11). Since 0.64 > 0.5,
`revise()` updates the artifact in place (same id, no retirement), setting `revisedAt`.
The store stays at 3 active artifacts; retrieval immediately reflects the updated content.

---

## Conversation trace

| Session | Turn | Input | PIL events |
|---|---|---|---|
| 1 | User | Download January Chase statement, rename `2026-01-chase-checking.pdf`, put in Finance/Statements | — |
|   | Agent | Done. Saved to `~/Finance/Statements/2026-01-chase-checking.pdf`. | (none — one-off task) |
| 2 | User | Download February Chase statement. Same as last time. | — |
|   | Agent | Renamed to `2026-02-chase-checking.pdf` and saved. I notice a naming pattern: YYYY-MM-institution-account.pdf. Want me to remember this? | — |
|   | User | Yes, always use that pattern. | **convention created** — `YYYY-MM-institution-account.pdf` in `~/Finance/Statements/` |
| 3 | User | Get my February Fidelity statement too. | — |
|   | Agent | Saved. You now have two institutions. Want a monthly checklist? | — |
|   | User | Good idea. Add Amex too — I always forget that one. | **fact created** — Chase, Fidelity, Amex; **procedure created** — monthly download checklist |
| 4 | User | It's March 2nd — time for statements. | retrieve("time for statements") → procedure |
|   | Agent | Following your monthly checklist: ✓ Chase ✓ Fidelity ✓ Amex. All saved. | (apply — no new artifacts) |
| 5 | User | I closed my Amex account. I opened a Schwab brokerage account. | — |
|   | Agent | Updated institution list and procedure. | **fact revised** — Amex → Schwab (in-place update) |

---

## Test phases

| Phase | Description | Key assertions |
|---|---|---|
| 1 | Session 1 → no artifacts | `created=0`, `updated=0`, store empty |
| 2 | Session 2 → naming convention | `kind=convention`, content matches `/YYYY-MM/`, injectable as `[provisional]` |
| 3 | Session 3 → fact + procedure | `created=2`, one `fact` and one `procedure`, both `[provisional]`; store has 3 active |
| 4 | Session 4 → retrieval | `retrieve("time for statements")` returns procedure; `retrieve("file naming format")` returns convention; execution exchange creates 0 new artifacts |
| 5 | Session 5 → revision | `revised.content` has Schwab, not Amex; `revisedAt` set; store still has 3 active |

---

## Running the benchmark

```bash
# All tests in the project
pnpm test

# Only this benchmark
pnpm test bench-financial-statements

# Verbose output (shows each test name)
pnpm test bench-financial-statements -- --reporter=verbose
```

No special CLI flags are needed. The benchmark uses the default matching behaviour
(`matchLlm = llm`) — semantic matching is not exercised here because the three artifact
kinds (`convention`, `fact`, `procedure`) are distinct and `matchCandidate` never compares
across kinds.
