# 05 — Persistence Benchmark (Stage 5)

| | |
|---|---|
| **Pipeline stage** | Stage 5 — Persist |
| **Module** | `packages/openclaw-plus/src/store.ts` → `persist()`, `loadAll()` |
| **Implementation status** | ✅ Implemented |
| **Automated coverage** | ✅ Unit tests (mock) |

---

## Purpose

This benchmark verifies that artifacts are correctly written to and read from the JSONL store, that updates are applied in-place (upsert by id), and that no data is silently lost or corrupted across the round-trip.

Persistence is the durability layer of the system. All upstream logic — extraction, accumulation, consolidation — produces results that have no lasting effect unless they are correctly stored. All downstream logic — retrieval, apply, revise — depends entirely on what was persisted.

## Design rationale this benchmark validates

- **JSONL format**: One artifact per line, human-readable, git-friendly. The format must be stable across field additions (additive-only schema evolution) and resistant to partial-write corruption (an interrupted write should not corrupt earlier entries).
- **Upsert by id**: `persist()` updates an existing artifact if its `id` is already in the store, rather than appending a duplicate. This is how accumulation and revision work — they mutate an artifact in-place without creating a new record.
- **Store isolation via env var**: `KNOWLEDGE_STORE_PATH` overrides the default `~/.openclaw/knowledge/artifacts.jsonl`. This is required for test isolation and for user-controlled store locations.
- **Retirement via flag**: Retired artifacts are kept in the JSONL with `retired: true` (not deleted). They are excluded from retrieval but preserved for audit purposes.

---

## Metrics

| Metric | Definition | Target |
|---|---|---|
| **Round-trip fidelity** | Fraction of fields correctly preserved across write/read | 1.0 |
| **Upsert correctness** | Updates modify the correct record; no duplicate created | 1.0 |
| **Isolation correctness** | `KNOWLEDGE_STORE_PATH` env var correctly redirects all reads and writes | 1.0 |
| **Retirement preservation** | Retired artifacts remain in file; excluded from retrieval | 1.0 |

---

## Test cases

### Group PE-A: Basic read/write

#### PE-A-1: Persist and load back
| Field | Value |
|---|---|
| **ID** | PE-A-1 |
| **Action** | `persist(artifact)` then `loadAll()` |
| **Pass criterion** | `loadAll()` returns an array containing the artifact; all fields match |
| **Automated** | ✅ `store.test.ts: "persists a new artifact and loads it back"` |

#### PE-A-2: Multiple artifacts persist independently
| Field | Value |
|---|---|
| **ID** | PE-A-2 |
| **Action** | `persist(a1)`, `persist(a2)`, then `loadAll()` |
| **Pass criterion** | `loadAll()` returns both artifacts; order is preserved |
| **Automated** | ✅ `store.test.ts: "persists multiple artifacts"` |

#### PE-A-3: Store file is created on first write
| Field | Value |
|---|---|
| **ID** | PE-A-3 |
| **Setup** | `KNOWLEDGE_STORE_PATH` points to a non-existent file |
| **Action** | `persist(artifact)` |
| **Pass criterion** | File is created; no error thrown; subsequent `loadAll()` returns the artifact |
| **Automated** | ✅ (implicit in test isolation setup — each test uses a fresh temp path) |

---

### Group PE-B: Upsert behavior

#### PE-B-1: Updating an existing artifact replaces it in-place
| Field | Value |
|---|---|
| **ID** | PE-B-1 |
| **Action** | `persist(artifact)`, then `persist({...artifact, confidence: 0.95})` (same id) |
| **Pass criterion** | `loadAll()` returns exactly 1 artifact (not 2); `confidence` = 0.95 |
| **Automated** | ✅ `store.test.ts: "upserts existing artifact by id"` |

#### PE-B-2: New artifact with different id is appended, not merged
| Field | Value |
|---|---|
| **ID** | PE-B-2 |
| **Action** | `persist(a1)`, `persist(a2)` where `a2.id ≠ a1.id` |
| **Pass criterion** | `loadAll()` returns 2 artifacts |
| **Automated** | ✅ `store.test.ts: "persists multiple artifacts"` |

---

### Group PE-C: Retirement

#### PE-C-1: Retired artifact is preserved in file
| Field | Value |
|---|---|
| **ID** | PE-C-1 |
| **Action** | `persist({...artifact, retired: true})` |
| **Pass criterion** | `loadAll()` includes the artifact with `retired: true` |
| **Automated** | ✅ `store.test.ts: "retired artifacts appear in loadAll"` |

#### PE-C-2: Retired artifact excluded from retrieval
| Field | Value |
|---|---|
| **ID** | PE-C-2 |
| **Action** | Persist a retired artifact with matching tags; call `retrieve(query)` |
| **Pass criterion** | Retired artifact does not appear in results |
| **Automated** | ✅ `store.test.ts: "retrieve — excludes retired artifacts"` |

---

### Group PE-D: Store isolation

#### PE-D-1: `KNOWLEDGE_STORE_PATH` env var redirects store
| Field | Value |
|---|---|
| **ID** | PE-D-1 |
| **Action** | Set `KNOWLEDGE_STORE_PATH=/tmp/custom.jsonl`; `persist(artifact)`; verify file created at that path |
| **Pass criterion** | Default path (`~/.openclaw/...`) untouched; custom path contains the artifact |
| **Automated** | ✅ (every test in `store.test.ts` uses isolated temp path via `beforeEach`) |

---

## Gaps and open questions

1. **Partial-write corruption**: What happens if the process is interrupted mid-write? JSONL format is somewhat resilient (incomplete line can be detected), but this is not tested.
2. **Large store performance**: The current implementation reads the entire file on every `loadAll()` call. Performance characteristics for stores with thousands of artifacts have not been benchmarked. A separate performance benchmark is needed before any production deployment.
3. **Concurrent writes**: The store has no locking mechanism. Concurrent writes from two processes (e.g., the playground and the computer-assistant running simultaneously) could corrupt the file. This is a known limitation.
4. **Schema evolution**: Adding a new optional field to `KnowledgeArtifact` must not break reading of older artifacts that lack the field. This is tested implicitly (older artifacts load fine with `undefined` for new fields) but should be explicitly tested with a fixture file from a prior schema version.

---

## Automated evaluation notes

Persistence is fully deterministic and requires no LLM. All test cases can be and mostly are covered by unit tests. The main gap is the partial-write and concurrent-write scenarios, which require process-level testing rather than unit tests.
