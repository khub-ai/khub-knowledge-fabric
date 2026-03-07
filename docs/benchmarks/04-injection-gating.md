# 04 — Injection Gating Benchmark (Stage 4)

**Pipeline stage:** Stage 4 — Decide (injection gating)
**Module:** `packages/openclaw-plus/src/store.ts` → `getInjectLabel()`, `isInjectable()`
**Implementation status:** ✅ Implemented
**Automated coverage:** ✅ Unit tests (mock)

---

## Purpose

This benchmark verifies that the system surfaces knowledge to the agent at the right confidence level, with the right label, and withholds it when confidence is insufficient.

Injection gating is the system's credibility mechanism. An agent that injects everything it has ever heard — including unconfirmed, contradicted, or weakly-held knowledge — is not trustworthy. An agent that never injects until knowledge is perfectly certain is not useful. The gating logic defines the operating point between these extremes.

## Design rationale this benchmark validates

- **Three-label hierarchy**: `[established]`, `[suggestion]`, and `[provisional]` communicate the system's confidence to the agent. The agent is expected to treat these differently: established knowledge is applied without question; suggestions are applied unless context contradicts; provisional knowledge is flagged for the user to confirm.
- **Stage gating**: `accumulating` stage artifacts are never injected — the system requires at least consolidation before surfacing to the agent without a `[provisional]` caveat. This prevents half-formed patterns from becoming agent behavior.
- **Confidence thresholding for auto-apply**: The `[established]` label (which triggers `autoApply=true`) requires both `consolidated` stage and confidence above the salience-adjusted threshold. This ensures the agent auto-applies only knowledge that has been both generalized and reinforced.
- **Salience-adjusted thresholds**: Higher-salience artifacts require higher confidence to auto-apply, reflecting the higher cost of applying them incorrectly.

---

## Label assignment rules (specification)

| Stage | Certainty | Confidence | Label | autoApply |
|---|---|---|---|---|
| `consolidated` | any | ≥ salience threshold | `[established]` | true |
| `consolidated` | any | < salience threshold | `[suggestion]` | false |
| `candidate` | `definitive` | any | `[provisional]` | false |
| `candidate` | `tentative` or `uncertain` | any | (none — not injected) | false |
| `accumulating` | any | any | (none — not injected) | false |
| `retired` | any | any | (none — not injected) | false |

Salience thresholds (from `AUTO_APPLY_THRESHOLDS`):
- `salience: "low"` → auto-apply at confidence ≥ 0.75
- `salience: "medium"` → auto-apply at confidence ≥ 0.85
- `salience: "high"` → auto-apply at confidence ≥ 0.95

---

## Metrics

| Metric | Definition | Target |
|---|---|---|
| **Label accuracy** | Fraction of artifacts assigned the correct label given their stage and confidence | 1.0 (deterministic) |
| **Injection precision** | Fraction of injected artifacts that are genuinely appropriate for the context | 🔲 Requires evaluation |
| **False injection rate** | Fraction of retired or accumulating artifacts incorrectly returned as injectable | 0.0 (deterministic) |

---

## Test cases

### Group IG-A: Label assignment

#### IG-A-1: Consolidated + high confidence → [established]
| Field | Value |
|---|---|
| **ID** | IG-A-1 |
| **Artifact** | stage=`consolidated`, confidence=0.90, salience=`low` (threshold 0.75) |
| **Pass criterion** | `getInjectLabel()` = `[established]`; `apply()` returns `autoApply: true` |
| **Automated** | ✅ `store.test.ts: "getInjectLabel — consolidated + high confidence → [established]"` |

#### IG-A-2: Consolidated + medium confidence → [suggestion]
| Field | Value |
|---|---|
| **ID** | IG-A-2 |
| **Artifact** | stage=`consolidated`, confidence=0.65, salience=`low` (threshold 0.75) |
| **Pass criterion** | `getInjectLabel()` = `[suggestion]`; `apply()` returns `autoApply: false` |
| **Automated** | ✅ `store.test.ts: "getInjectLabel — consolidated + below threshold → [suggestion]"` |

#### IG-A-3: Candidate + definitive → [provisional]
| Field | Value |
|---|---|
| **ID** | IG-A-3 |
| **Artifact** | stage=`candidate`, certainty=`definitive`, confidence=0.65 |
| **Pass criterion** | `getInjectLabel()` = `[provisional]`; `apply()` returns `autoApply: false` |
| **Automated** | ✅ `store.test.ts: "getInjectLabel — candidate + definitive → [provisional]"` |

#### IG-A-4: Candidate + tentative → not injected
| Field | Value |
|---|---|
| **ID** | IG-A-4 |
| **Artifact** | stage=`candidate`, certainty=`tentative` |
| **Pass criterion** | `getInjectLabel()` = `null`; `isInjectable()` = `false` |
| **Automated** | ✅ `store.test.ts: "getInjectLabel — candidate + tentative → null"` |

#### IG-A-5: Candidate + uncertain → not injected
| Field | Value |
|---|---|
| **ID** | IG-A-5 |
| **Artifact** | stage=`candidate`, certainty=`uncertain` |
| **Pass criterion** | `getInjectLabel()` = `null`; `isInjectable()` = `false` |
| **Automated** | ✅ `store.test.ts: "getInjectLabel — candidate + uncertain → null"` |

#### IG-A-6: Accumulating → not injected
| Field | Value |
|---|---|
| **ID** | IG-A-6 |
| **Artifact** | stage=`accumulating`, any certainty |
| **Pass criterion** | `getInjectLabel()` = `null`; `isInjectable()` = `false` |
| **Automated** | ✅ `store.test.ts: "getInjectLabel — accumulating → null"` |

#### IG-A-7: Retired → not injected
| Field | Value |
|---|---|
| **ID** | IG-A-7 |
| **Artifact** | `retired: true`, any stage |
| **Pass criterion** | `isInjectable()` = `false`; not returned by `retrieve()` |
| **Automated** | ✅ `store.test.ts: "retrieve — excludes retired artifacts"` |

---

### Group IG-B: Salience-adjusted thresholds

#### IG-B-1: Medium salience raises threshold
| Field | Value |
|---|---|
| **ID** | IG-B-1 |
| **Artifact** | stage=`consolidated`, confidence=0.80, salience=`medium` (threshold 0.85) |
| **Pass criterion** | `getInjectLabel()` = `[suggestion]` (not `[established]`) — confidence < 0.85 |
| **Automated** | 🔲 Not explicitly tested; threshold constants tested but not per-salience label |

#### IG-B-2: High salience raises threshold further
| Field | Value |
|---|---|
| **ID** | IG-B-2 |
| **Artifact** | stage=`consolidated`, confidence=0.92, salience=`high` (threshold 0.95) |
| **Pass criterion** | `getInjectLabel()` = `[suggestion]` (not `[established]`) — confidence < 0.95 |
| **Automated** | 🔲 Not explicitly tested |

---

### Group IG-C: Integration with processMessage output

#### IG-C-1: Injectable artifacts correctly identified in pipeline output
| Field | Value |
|---|---|
| **ID** | IG-C-1 |
| **Action** | Run `processMessage()` on a definitive preference message (novel) |
| **Pass criterion** | `result.injectable` contains 1 entry with label `[provisional]` |
| **Automated** | ✅ `pipeline.test.ts: "returns [provisional] injectable for definitive candidate"` |

#### IG-C-2: Accumulating artifacts produce no injectable output
| Field | Value |
|---|---|
| **ID** | IG-C-2 |
| **Action** | Run `processMessage()` on a message that accumulates into an existing artifact |
| **Pass criterion** | `result.injectable` is empty |
| **Automated** | ✅ `pipeline.test.ts: "returns no injectable for accumulating update"` |

#### IG-C-3: Consolidated artifact becomes [established] after threshold
| Field | Value |
|---|---|
| **ID** | IG-C-3 |
| **Action** | Run `processMessage()` three times with related messages; third call triggers consolidation |
| **Pass criterion** | After third call, the consolidated artifact is returned as `[established]` in subsequent `apply()` calls if confidence ≥ threshold |
| **Automated** | ⚠️ Consolidation is tested; `[established]` after consolidation not separately tested |

---

## Gaps and open questions

1. **Salience threshold tests (IG-B-1, IG-B-2)**: Per-salience-level label assignment is not explicitly tested. The constants are defined but their effect on `getInjectLabel()` is not verified for each salience value.
2. **`[established]` after consolidation (IG-C-3)**: The path from third-message consolidation to `[established]` injection requires a consolidated artifact with confidence above threshold. Currently the default confidence after consolidation may not exceed the threshold. This interaction needs a test.
3. **Label semantics**: The labels are injected into the prompt as text strings. The agent's interpretation of `[provisional]` vs. `[suggestion]` vs. `[established]` depends on the agent's system prompt — which is outside the scope of the store. A future benchmark should test agent behavior under each label.

---

## Automated evaluation notes

Label assignment is fully deterministic given stage, certainty, confidence, and salience. It requires no LLM and can be exhaustively unit-tested. The current test coverage is good but missing the salience-threshold variations (IG-B-1, IG-B-2).

**To add**: Parametric tests over the full matrix of (stage × certainty × salience × confidence) to verify the label table above is correctly implemented in all cells.
