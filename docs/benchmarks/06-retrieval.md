# 06 — Retrieval Benchmark (Stage 6)

| | |
|---|---|
| **Pipeline stage** | Stage 6 — Retrieve |
| **Module** | `packages/openclaw-plus/src/store.ts` → `retrieve()` |
| **Implementation status** | ✅ Implemented |
| **Automated coverage** | ✅ Unit tests (mock) + live LLM benchmark scenarios |

---

## Purpose

This benchmark verifies that retrieval surfaces the most relevant artifacts for a given query, ranks them correctly, excludes ineligible artifacts, and respects the `limit` parameter.

Retrieval is the system's memory recall mechanism. Poor retrieval means the agent proceeds with stale or irrelevant knowledge injected, or misses applicable knowledge entirely. The retrieval quality determines how much practical value the accumulated knowledge provides.

## Design rationale this benchmark validates

- **Composite scoring**: `retrieve()` combines tag overlap (weight 0.60), content Jaccard (weight 0.30), confidence (weight 0.10), and a consolidated-stage bonus (0.05). The weights encode a design judgment: semantic relevance (tags) matters most, followed by content similarity, then confidence. This weighting has not yet been empirically validated.
- **Tag vocabulary anchoring**: Tags are normalized at extraction time to a shared vocabulary. This means two messages that express the same concept in different words will share tags and thus match each other. The retrieval quality is therefore upstream-dependent on extraction quality.
- **Retired artifact exclusion**: Retired artifacts are in the JSONL but must never surface in retrieval results.
- **Limit parameter**: The `limit` parameter constrains result size. The default is 5. Retrieval must respect this regardless of store size.

---

## Scoring formula

```
score = tag_overlap(query_tags, artifact_tags) × 0.60
      + content_jaccard(query, artifact.content) × 0.30
      + artifact.confidence × 0.10
      + 0.05 (if artifact.stage === "consolidated")
```

Where `tag_overlap` is the Jaccard similarity between the set of tags derived from the query and the artifact's `tags[]` field.

---

## Metrics

| Metric | Definition | Target |
|---|---|---|
| **Precision@k** | Fraction of top-k results that are genuinely relevant | ≥ 0.80 |
| **Recall@k** | Fraction of relevant artifacts that appear in top-k | ≥ 0.80 |
| **Hit rate** | Fraction of scenarios where the expected artifact appears in top-k | ≥ 0.90 |
| **Ranking correctness** | Expected top artifact ranks first | ≥ 0.85 |
| **Retired exclusion** | Retired artifacts never appear in results | 1.0 |
| **Limit compliance** | Result count ≤ limit | 1.0 |

---

## Test cases

### Group RE-A: Basic retrieval

#### RE-A-1: Exact tag match retrieves correct artifact
| Field | Value |
|---|---|
| **ID** | RE-A-1 |
| **Store** | Artifact: preference, `content="Always use bullet points for summaries"`, tags=`[summary-format, bullet-points]`, confidence=0.80 |
| **Query** | `"summary bullet points"` |
| **Pass criterion** | Artifact appears in top-k results |
| **Automated** | ✅ `store.test.ts: "retrieve — returns relevant artifacts by tag overlap"` |
| **Live scenario** | `benchmarks/scenarios.ts: retr-1` |

#### RE-A-2: Alias lookup via tag match
| Field | Value |
|---|---|
| **ID** | RE-A-2 |
| **Store** | Convention: `"When user says 'gh', they mean https://github.com"`, tags=`[alias, github, url-mapping]` |
| **Query** | `"open gh"` |
| **Pass criterion** | Convention artifact appears in top-k |
| **Automated** | ✅ `store.test.ts: "retrieve — alias lookup"` |
| **Live scenario** | `benchmarks/scenarios.ts: retr-2` |

#### RE-A-3: Unrelated query returns low-score results
| Field | Value |
|---|---|
| **ID** | RE-A-3 |
| **Store** | Artifact about TypeScript code style |
| **Query** | `"financial statements naming convention"` |
| **Pass criterion** | TypeScript artifact either does not appear or ranks very low (score < 0.15) |
| **Automated** | 🔲 Not yet specified |

---

### Group RE-B: Ranking

#### RE-B-1: Consolidated artifact outranks accumulating, same tags
| Field | Value |
|---|---|
| **ID** | RE-B-1 |
| **Store** | A1: stage=`consolidated`, tags=`[bullet-points, summary-format]`, confidence=0.70; A2: stage=`accumulating`, same tags, confidence=0.70 |
| **Pass criterion** | A1 ranks above A2 (consolidated bonus = +0.05) |
| **Automated** | ⚠️ Sorting tested in `store.test.ts`; this specific comparison not isolated |

#### RE-B-2: Higher confidence breaks tie among same-stage artifacts
| Field | Value |
|---|---|
| **ID** | RE-B-2 |
| **Store** | A1: stage=`candidate`, tags=`[bullet-points]`, confidence=0.65; A2: stage=`candidate`, same tags, confidence=0.50 |
| **Pass criterion** | A1 ranks above A2 |
| **Automated** | ⚠️ Confidence component is part of scoring formula; not independently tested |

#### RE-B-3: Tag overlap dominates content similarity
| Field | Value |
|---|---|
| **ID** | RE-B-3 |
| **Store** | A1: low content overlap with query, high tag overlap; A2: high content overlap, no tag overlap |
| **Pass criterion** | A1 ranks above A2 (tag weight 0.60 > content weight 0.30) |
| **Automated** | 🔲 Not yet specified |

---

### Group RE-C: Exclusion

#### RE-C-1: Retired artifact excluded
| Field | Value |
|---|---|
| **ID** | RE-C-1 |
| **Store** | Artifact with `retired: true`, tags matching the query |
| **Pass criterion** | Retired artifact does not appear in results |
| **Automated** | ✅ `store.test.ts: "retrieve — excludes retired artifacts"` |

#### RE-C-2: Limit parameter respected
| Field | Value |
|---|---|
| **ID** | RE-C-2 |
| **Store** | 10 artifacts, all with high tag overlap |
| **Query** | Any matching query; `limit = 3` |
| **Pass criterion** | `retrieve(query, 3)` returns ≤ 3 artifacts |
| **Automated** | ✅ `store.test.ts: "retrieve — respects limit parameter"` |

---

### Group RE-D: Edge cases

#### RE-D-1: Empty store returns empty array
| Field | Value |
|---|---|
| **ID** | RE-D-1 |
| **Store** | Empty |
| **Pass criterion** | `retrieve(query)` returns `[]` without error |
| **Automated** | ✅ `store.test.ts: "retrieve — empty store"` |

#### RE-D-2: Query with no matching tags falls back to content Jaccard
| Field | Value |
|---|---|
| **ID** | RE-D-2 |
| **Store** | Artifact with no tags but content matching the query |
| **Pass criterion** | Artifact retrieved via content Jaccard fallback |
| **Automated** | ⚠️ Content Jaccard path exists in code but not explicitly tested in isolation |

---

## Gaps and open questions

1. **Scoring weight calibration**: The weights (0.60/0.30/0.10/0.05) are initial design choices, not empirically validated. A calibration study should: (a) collect human relevance judgments for query-artifact pairs, (b) measure nDCG and MRR under different weight combinations, (c) update the defaults.
2. **RE-B-3 (tag vs. content dominance)**: No test isolates the relative contribution of tag overlap and content Jaccard. This is needed to validate the weighting design.
3. **Multi-word query handling**: The current tag overlap computation tokenizes the query. The behavior on multi-word queries with complex phrasing has not been characterized.
4. **Semantic drift**: If the tag vocabulary drifts over time (e.g., early artifacts tagged `summary` and later ones tagged `summary-format`), retrieval quality degrades. A benchmark monitoring tag vocabulary consistency would catch this.
5. **Performance at scale**: `retrieve()` currently scans all artifacts. At 10,000 artifacts, this will be slow. The in-memory inverted index planned for Milestone 1d will address this; its retrieval quality should be verified against these same scenarios.

---

## Automated evaluation notes

The live benchmark in `apps/computer-assistant/benchmarks/index.ts` runs `RETRIEVAL_SCENARIOS` with a real Anthropic model and reports hit rate (fraction of scenarios where the expected artifact appears in top-k).

Currently only 2 retrieval scenarios are defined. The benchmark should be expanded with at least:
- A ranking scenario (verifying order, not just presence)
- A negative scenario (verifying irrelevant artifacts don't appear)
- A multi-artifact store with realistic noise

**Metrics to add to the live benchmark runner:**
- MRR (mean reciprocal rank): `1 / rank_of_expected_artifact`
- nDCG: normalized discounted cumulative gain over the result list
