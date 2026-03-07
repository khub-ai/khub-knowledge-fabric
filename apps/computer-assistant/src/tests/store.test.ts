/**
 * Unit tests for the knowledge store (store.ts).
 *
 * Uses a temporary file path per test to ensure isolation.
 * All LLM calls use mock implementations — no API calls required.
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { unlink } from "node:fs/promises";
import { existsSync } from "node:fs";
import {
  persist,
  retrieve,
  apply,
  revise,
  getActiveTags,
  matchCandidate,
  accumulateEvidence,
  getInjectLabel,
  isInjectable,
  loadAll,
} from "@khub-ai/openclaw-plus/store";
import { candidateToArtifact } from "@khub-ai/openclaw-plus/extract";
import { CONFIDENCE_SEED, CONSOLIDATION_THRESHOLD } from "@khub-ai/openclaw-plus/types";
import type { KnowledgeArtifact } from "@khub-ai/openclaw-plus/types";
import {
  createPatternMockLLM,
  CONSOLIDATED_BULLET_RULE,
} from "./mock-llm.js";

// ---------------------------------------------------------------------------
// Test isolation — unique store path per test
// ---------------------------------------------------------------------------

let testStorePath: string;

beforeEach(() => {
  testStorePath = join(tmpdir(), `pil-test-${Date.now()}-${Math.random().toString(36).slice(2)}.jsonl`);
  process.env["KNOWLEDGE_STORE_PATH"] = testStorePath;
});

afterEach(async () => {
  delete process.env["KNOWLEDGE_STORE_PATH"];
  if (existsSync(testStorePath)) {
    await unlink(testStorePath);
  }
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeArtifact(overrides: Partial<KnowledgeArtifact> = {}): KnowledgeArtifact {
  return {
    id: `test-${Math.random().toString(36).slice(2)}`,
    kind: "preference",
    content: "I always want bullet-point summaries",
    confidence: CONFIDENCE_SEED.definitive,
    provenance: "test",
    createdAt: new Date().toISOString(),
    stage: "candidate",
    certainty: "definitive",
    tags: ["summary-format", "bullet-points", "output-style"],
    evidenceCount: 1,
    evidence: ["I always want bullet-point summaries"],
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// persist / loadAll
// ---------------------------------------------------------------------------

describe("persist / loadAll", () => {
  it("persists a new artifact and loads it back", async () => {
    const artifact = makeArtifact();
    await persist(artifact);

    const all = await loadAll();
    expect(all).toHaveLength(1);
    expect(all[0]!.id).toBe(artifact.id);
    expect(all[0]!.content).toBe(artifact.content);
  });

  it("upserts an artifact by id", async () => {
    const artifact = makeArtifact();
    await persist(artifact);

    const updated = { ...artifact, confidence: 0.9, content: "updated content" };
    await persist(updated);

    const all = await loadAll();
    expect(all).toHaveLength(1);
    expect(all[0]!.confidence).toBe(0.9);
    expect(all[0]!.content).toBe("updated content");
  });

  it("persists multiple artifacts", async () => {
    await persist(makeArtifact({ id: "a1" }));
    await persist(makeArtifact({ id: "a2" }));
    await persist(makeArtifact({ id: "a3" }));

    const all = await loadAll();
    expect(all).toHaveLength(3);
  });
});

// ---------------------------------------------------------------------------
// getActiveTags
// ---------------------------------------------------------------------------

describe("getActiveTags", () => {
  it("returns unique tags from active artifacts", async () => {
    await persist(makeArtifact({ tags: ["summary-format", "bullet-points"] }));
    await persist(makeArtifact({ id: "a2", tags: ["code-style", "bullet-points"] }));

    const tags = await getActiveTags();
    expect(tags).toContain("summary-format");
    expect(tags).toContain("bullet-points");
    expect(tags).toContain("code-style");
    // No duplicates
    expect(tags.filter((t) => t === "bullet-points")).toHaveLength(1);
  });

  it("excludes tags from retired artifacts", async () => {
    await persist(makeArtifact({ tags: ["retired-tag"], retired: true }));

    const tags = await getActiveTags();
    expect(tags).not.toContain("retired-tag");
  });

  it("returns empty array when store is empty", async () => {
    const tags = await getActiveTags();
    expect(tags).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// matchCandidate
// ---------------------------------------------------------------------------

describe("matchCandidate", () => {
  it("returns null when store is empty", async () => {
    const match = await matchCandidate({
      kind: "preference",
      tags: ["summary-format", "bullet-points"],
    });
    expect(match).toBeNull();
  });

  it("returns null when no artifact matches kind", async () => {
    await persist(makeArtifact({ kind: "convention" }));
    const match = await matchCandidate({
      kind: "preference",
      tags: ["summary-format"],
    });
    expect(match).toBeNull();
  });

  it("returns matching artifact with sufficient tag overlap", async () => {
    const existing = makeArtifact({
      id: "existing-1",
      kind: "preference",
      tags: ["summary-format", "bullet-points", "output-style"],
    });
    await persist(existing);

    const match = await matchCandidate({
      kind: "preference",
      tags: ["summary-format", "output-style"],
    });

    expect(match).not.toBeNull();
    expect(match!.id).toBe("existing-1");
  });

  it("returns null when tag overlap is below threshold", async () => {
    await persist(makeArtifact({
      kind: "preference",
      tags: ["file-naming", "directory-structure"],
    }));

    const match = await matchCandidate({
      kind: "preference",
      tags: ["summary-format", "bullet-points"],
    });

    expect(match).toBeNull();
  });

  it("returns null for retired artifacts", async () => {
    await persist(makeArtifact({
      tags: ["summary-format", "bullet-points"],
      retired: true,
    }));

    const match = await matchCandidate({
      kind: "preference",
      tags: ["summary-format", "bullet-points"],
    });

    expect(match).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// accumulateEvidence
// ---------------------------------------------------------------------------

describe("accumulateEvidence", () => {
  it("increments evidenceCount and appends evidence", async () => {
    const artifact = makeArtifact({ evidenceCount: 1, evidence: ["first obs"] });
    await persist(artifact);

    const updated = await accumulateEvidence(
      artifact,
      "second observation",
      createPatternMockLLM([]),
    );

    expect(updated.evidenceCount).toBe(2);
    expect(updated.evidence).toContain("second observation");
    expect(updated.stage).toBe("accumulating");
  });

  it(`consolidates when evidenceCount reaches CONSOLIDATION_THRESHOLD (${CONSOLIDATION_THRESHOLD})`, async () => {
    // Start at threshold - 1 so the next accumulation triggers consolidation
    const artifact = makeArtifact({
      evidenceCount: CONSOLIDATION_THRESHOLD - 1,
      evidence: Array.from({ length: CONSOLIDATION_THRESHOLD - 1 }, (_, i) => `obs ${i + 1}`),
      stage: "accumulating",
    });
    await persist(artifact);

    const consolLlm = createPatternMockLLM([
      { match: "consolidation assistant", response: CONSOLIDATED_BULLET_RULE },
    ]);

    const updated = await accumulateEvidence(
      artifact,
      `obs ${CONSOLIDATION_THRESHOLD}`,
      consolLlm,
    );

    expect(updated.stage).toBe("consolidated");
    expect(updated.content).toBe(CONSOLIDATED_BULLET_RULE);
    expect(updated.evidenceCount).toBe(CONSOLIDATION_THRESHOLD);
    // Confidence should have increased
    expect(updated.confidence).toBeGreaterThan(artifact.confidence);
  });

  it("does not re-consolidate an already-consolidated artifact", async () => {
    const artifact = makeArtifact({
      stage: "consolidated",
      content: "Already consolidated rule",
      evidenceCount: CONSOLIDATION_THRESHOLD,
      confidence: 0.85,
    });
    await persist(artifact);

    const updated = await accumulateEvidence(
      artifact,
      "new observation",
      createPatternMockLLM([{ match: "consolidation assistant", response: "New rule" }]),
    );

    // Stage stays consolidated, content unchanged
    expect(updated.stage).toBe("consolidated");
    // evidenceCount still increments (for tracking)
    expect(updated.evidenceCount).toBe(CONSOLIDATION_THRESHOLD + 1);
  });
});

// ---------------------------------------------------------------------------
// retrieve
// ---------------------------------------------------------------------------

describe("retrieve", () => {
  it("returns empty array from empty store", async () => {
    const results = await retrieve("summary format preference");
    expect(results).toHaveLength(0);
  });

  it("retrieves artifact with matching tags", async () => {
    await persist(makeArtifact({
      tags: ["summary-format", "bullet-points"],
      stage: "consolidated",
    }));

    const results = await retrieve("summary bullet points");
    expect(results.length).toBeGreaterThan(0);
  });

  it("excludes retired artifacts", async () => {
    await persist(makeArtifact({ retired: true }));
    const results = await retrieve("bullet-point summary");
    expect(results).toHaveLength(0);
  });

  it("returns all active artifacts when query is empty", async () => {
    await persist(makeArtifact({ id: "a1" }));
    await persist(makeArtifact({ id: "a2" }));

    const results = await retrieve("");
    expect(results).toHaveLength(2);
  });

  it("respects the limit parameter", async () => {
    for (let i = 0; i < 5; i++) {
      await persist(makeArtifact({
        id: `a${i}`,
        tags: ["summary-format", "bullet-points"],
      }));
    }

    const results = await retrieve("summary", 3);
    expect(results).toHaveLength(3);
  });
});

// ---------------------------------------------------------------------------
// getInjectLabel / isInjectable
// ---------------------------------------------------------------------------

describe("getInjectLabel", () => {
  it("returns [established] for consolidated high-confidence artifact", () => {
    const artifact = makeArtifact({
      stage: "consolidated",
      confidence: 0.85,
    });
    expect(getInjectLabel(artifact)).toBe("[established]");
  });

  it("returns [suggestion] for consolidated lower-confidence artifact", () => {
    const artifact = makeArtifact({
      stage: "consolidated",
      confidence: 0.50,
    });
    expect(getInjectLabel(artifact)).toBe("[suggestion]");
  });

  it("returns [provisional] for candidate with definitive certainty", () => {
    const artifact = makeArtifact({
      stage: "candidate",
      certainty: "definitive",
    });
    expect(getInjectLabel(artifact)).toBe("[provisional]");
  });

  it("returns null for accumulating artifact (not injectable)", () => {
    const artifact = makeArtifact({ stage: "accumulating" });
    expect(getInjectLabel(artifact)).toBeNull();
  });

  it("returns null for candidate with tentative certainty (not injectable)", () => {
    const artifact = makeArtifact({
      stage: "candidate",
      certainty: "tentative",
    });
    expect(getInjectLabel(artifact)).toBeNull();
  });

  it("returns null for retired artifact", () => {
    const artifact = makeArtifact({
      stage: "consolidated",
      confidence: 0.9,
      retired: true,
    });
    expect(getInjectLabel(artifact)).toBeNull();
  });

  it("uses higher threshold for high-salience artifacts", () => {
    // High salience requires confidence >= 0.95 for [established]
    const artifact = makeArtifact({
      stage: "consolidated",
      confidence: 0.85,
      salience: "high",
    });
    // 0.85 < 0.95 → [suggestion], not [established]
    expect(getInjectLabel(artifact)).toBe("[suggestion]");
  });
});

// ---------------------------------------------------------------------------
// apply
// ---------------------------------------------------------------------------

describe("apply", () => {
  it("auto-applies consolidated high-confidence artifact", async () => {
    const artifact = makeArtifact({ stage: "consolidated", confidence: 0.85 });
    await persist(artifact);

    const { suggestion, autoApply } = await apply(artifact, "context");
    expect(autoApply).toBe(true);
    expect(suggestion).toContain("[established]");
  });

  it("suggests consolidated lower-confidence artifact", async () => {
    const artifact = makeArtifact({ stage: "consolidated", confidence: 0.50 });
    await persist(artifact);

    const { suggestion, autoApply } = await apply(artifact, "context");
    expect(autoApply).toBe(false);
    expect(suggestion).toContain("[suggestion]");
  });

  it("returns empty for non-injectable artifact", async () => {
    const artifact = makeArtifact({ stage: "accumulating" });
    await persist(artifact);

    const { suggestion, autoApply } = await apply(artifact, "context");
    expect(autoApply).toBe(false);
    expect(suggestion).toBe("");
  });
});

// ---------------------------------------------------------------------------
// revise
// ---------------------------------------------------------------------------

describe("revise", () => {
  it("updates an artifact in place for minor content change", async () => {
    const artifact = makeArtifact({ content: "I always want bullet summaries" });
    await persist(artifact);

    // Small change — Jaccard should be high enough to update in place
    const revised = await revise(artifact, {
      content: "I always want bullet summaries with max 5 items",
    });

    // Same id (in-place update)
    expect(revised.id).toBe(artifact.id);
    expect(revised.revisedAt).toBeTruthy();
  });

  it("retires old artifact and creates new one for major content change", async () => {
    const artifact = makeArtifact({ content: "Use markdown headers for all output" });
    await persist(artifact);

    const revised = await revise(artifact, {
      content: "Always use bullet points, never use headers, keep it concise and scannable",
    });

    // New id
    expect(revised.id).not.toBe(artifact.id);

    // Original is retired
    const all = await loadAll();
    const original = all.find((a) => a.id === artifact.id);
    expect(original?.retired).toBe(true);
  });
});
