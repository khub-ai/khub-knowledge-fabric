# 07 — Apply and Revise Benchmark (Stages 7–8)

**Pipeline stages:** Stage 7 — Apply; Stage 8 — Revise
**Module:** `packages/openclaw-plus/src/store.ts` → `apply()`, `revise()`, `recordAccepted()`, `recordRejected()`
**Implementation status:** ✅ Implemented
**Automated coverage:** ✅ Unit tests (mock)

---

## Purpose

This benchmark verifies that the apply stage correctly surfaces knowledge to the agent (or withholds it), and that the revise stage correctly updates or retires artifacts in response to user feedback.

Apply and revise are the system's output and feedback mechanisms. Without correct apply behavior, the agent either ignores accumulated knowledge or applies it indiscriminately. Without correct revise behavior, the system cannot update its model of the user when circumstances change, and errors persist indefinitely.

## Design rationale this benchmark validates

- **Apply as suggestion, not command**: `apply()` returns a suggestion string and an `autoApply` flag. The agent decides what to do with it. For non-auto-apply cases, the user's next response implicitly confirms or overrides the suggestion. This keeps the human in the loop without requiring explicit confirmation dialogs for every injection.
- **Revise as in-place update**: By default, `revise()` updates the artifact in-place (same id). For major changes, `revise()` can optionally retire the old artifact and create a new one. The in-place case is the common path; the retire-and-replace case is for when the underlying knowledge has fundamentally changed, not just been refined.
- **Feedback loop**: `recordAccepted()` and `recordRejected()` update confidence counters that will eventually feed into confidence decay and reinforcement (Phase 2). Currently they increment `acceptedCount` / `rejectedCount` fields without changing `confidence` directly. Future versions will adjust effective confidence based on these signals.
- **appliedCount tracking**: `apply()` increments `appliedCount` each time it surfaces a suggestion. This is the habituation signal — frequently-applied artifacts that are rarely rejected are candidates for confidence promotion.

---

## Metrics

| Metric | Definition | Target |
|---|---|---|
| **autoApply accuracy** | Fraction of apply() calls where autoApply flag matches the expected value | 1.0 (deterministic) |
| **Suggestion string correctness** | Suggestion content reflects artifact content with correct label | 1.0 |
| **Revise round-trip fidelity** | All specified fields updated; id preserved (in-place case) | 1.0 |
| **Retire correctness** | Old artifact retired; new artifact created; relation established | 1.0 |
| **Counter accuracy** | `acceptedCount` / `rejectedCount` / `appliedCount` increment correctly | 1.0 |

---

## Test cases

### Group AR-A: Apply

#### AR-A-1: Consolidated + high confidence → autoApply=true
| Field | Value |
|---|---|
| **ID** | AR-A-1 |
| **Artifact** | stage=`consolidated`, confidence=0.90, salience=`low` (threshold 0.75) |
| **Pass criterion** | `apply()` returns `{ suggestion: "[established] ...", autoApply: true }` |
| **Automated** | ✅ `store.test.ts: "apply — consolidated high confidence → autoApply true"` |

#### AR-A-2: Candidate + definitive → autoApply=false, suggestion present
| Field | Value |
|---|---|
| **ID** | AR-A-2 |
| **Artifact** | stage=`candidate`, certainty=`definitive`, confidence=0.65 |
| **Pass criterion** | `apply()` returns `{ suggestion: "[provisional] ...", autoApply: false }` |
| **Automated** | ✅ `store.test.ts: "apply — candidate + definitive → autoApply false"` |

#### AR-A-3: Non-injectable artifact → no suggestion
| Field | Value |
|---|---|
| **ID** | AR-A-3 |
| **Artifact** | stage=`accumulating` |
| **Pass criterion** | `apply()` returns `{ suggestion: null, autoApply: false }` |
| **Automated** | ✅ `store.test.ts: "apply — accumulating → null suggestion"` |

#### AR-A-4: appliedCount increments on each apply call
| Field | Value |
|---|---|
| **ID** | AR-A-4 |
| **Action** | Call `apply()` twice on the same artifact |
| **Pass criterion** | After second call, `appliedCount` = 2 in the persisted artifact |
| **Automated** | ✅ `store.test.ts: "apply — increments appliedCount"` |

---

### Group AR-B: Revise (in-place)

#### AR-B-1: Content update preserves id
| Field | Value |
|---|---|
| **ID** | AR-B-1 |
| **Action** | `revise(artifact, { content: "Updated rule." })` |
| **Pass criterion** | Returned artifact has same `id`; `content` = "Updated rule."; `revisedAt` is set |
| **Automated** | ✅ `store.test.ts: "revise — updates content in-place"` |

#### AR-B-2: Confidence update within bounds
| Field | Value |
|---|---|
| **ID** | AR-B-2 |
| **Action** | `revise(artifact, { confidence: 0.95 })` on artifact with confidence=0.65 |
| **Pass criterion** | `confidence` = 0.95; capped at 1.0 |
| **Automated** | ✅ `store.test.ts: "revise — updates confidence"` |

#### AR-B-3: Revised artifact is persisted
| Field | Value |
|---|---|
| **ID** | AR-B-3 |
| **Action** | `revise(artifact, updates)` then `loadAll()` |
| **Pass criterion** | `loadAll()` returns the updated artifact; original version not present |
| **Automated** | ✅ `store.test.ts: "revise — persists updated artifact"` |

---

### Group AR-C: Revise (retire and replace)

#### AR-C-1: Old artifact retired, new one created
| Field | Value |
|---|---|
| **ID** | AR-C-1 |
| **Action** | `revise(artifact, { content: "New rule." }, { retire: true })` |
| **Pass criterion** | Old artifact has `retired: true`; new artifact created with new id and content "New rule." |
| **Automated** | ✅ `store.test.ts: "revise — retire and replace"` |

#### AR-C-2: Supersession relation created
| Field | Value |
|---|---|
| **ID** | AR-C-2 |
| **Action** | Retire-and-replace revision |
| **Pass criterion** | New artifact has `relations` entry `{ type: "supersedes", targetId: oldId }` |
| **Automated** | ⚠️ Relation creation tested but `supersedes` type not explicitly asserted |

---

### Group AR-D: Feedback tracking

#### AR-D-1: recordAccepted increments acceptedCount
| Field | Value |
|---|---|
| **ID** | AR-D-1 |
| **Action** | `recordAccepted(artifactId)` |
| **Pass criterion** | Artifact in store has `acceptedCount` incremented by 1 |
| **Automated** | ✅ `store.test.ts: "recordAccepted — increments acceptedCount"` |

#### AR-D-2: recordRejected increments rejectedCount
| Field | Value |
|---|---|
| **ID** | AR-D-2 |
| **Action** | `recordRejected(artifactId)` |
| **Pass criterion** | Artifact in store has `rejectedCount` incremented by 1 |
| **Automated** | ✅ `store.test.ts: "recordRejected — increments rejectedCount"` |

#### AR-D-3: Feedback counters inform future confidence (Phase 2)
| Field | Value |
|---|---|
| **ID** | AR-D-3 |
| **Status** | 🔲 Placeholder — not yet implemented |
| **Description** | After N rejections, effective confidence should decrease. After M acceptances, effective confidence should increase. The exact formula is deferred to Phase 2. |

---

## Gaps and open questions

1. **AR-C-2 (supersedes relation)**: The `relations` field is written during retire-and-replace, but no test asserts the `type: "supersedes"` value specifically.
2. **AR-D-3 (confidence from feedback)**: `acceptedCount` and `rejectedCount` are tracked but do not yet modify `confidence`. The formula for feedback-driven confidence adjustment is a Phase 2 concern. The benchmark stub is here to ensure we don't skip it.
3. **Suggestion string format**: The format of the suggestion string (e.g., `"[established] preference — Always use bullet points..."`) is not formally specified. If the format changes, the agent's prompt interpretation may break. A format spec and corresponding test would make this explicit.
4. **Multi-artifact apply sequence**: The agent may apply several artifacts in sequence for a single prompt. The interaction between multiple apply calls (especially `appliedCount` for each) has not been tested.

---

## Automated evaluation notes

Apply and revise are fully deterministic. All test cases can be unit-tested without LLM involvement. The main gaps (AR-C-2, AR-D-3, suggestion format) are straightforward to add to `store.test.ts`.

**For Phase 2**: When feedback-driven confidence adjustment is implemented, a property-based test would be valuable: for any sequence of `recordAccepted()` / `recordRejected()` calls, `confidence` should stay within [0, 1] and the direction of change should match the feedback direction.
