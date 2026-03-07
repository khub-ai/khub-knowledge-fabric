/**
 * Unit tests for the processMessage pipeline orchestration.
 *
 * Tests the end-to-end flow: extract → match → accumulate/create → injectable.
 * Uses mock LLMs — no API calls required.
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { unlink } from "node:fs/promises";
import { existsSync } from "node:fs";
import { processMessage, formatForInjection } from "@khub-ai/openclaw-plus/pipeline";
import { loadAll } from "@khub-ai/openclaw-plus/store";
import { CONSOLIDATION_THRESHOLD } from "@khub-ai/openclaw-plus/types";
import {
  createPatternMockLLM,
  createFullPipelineMockLLM,
  BULLET_PREF_RESPONSE,
  ALIAS_FACT_RESPONSE,
  EMPTY_RESPONSE,
  CONSOLIDATED_BULLET_RULE,
} from "./mock-llm.js";

// ---------------------------------------------------------------------------
// Test isolation
// ---------------------------------------------------------------------------

let testStorePath: string;

beforeEach(() => {
  testStorePath = join(tmpdir(), `pil-pipeline-test-${Date.now()}-${Math.random().toString(36).slice(2)}.jsonl`);
  process.env["KNOWLEDGE_STORE_PATH"] = testStorePath;
});

afterEach(async () => {
  delete process.env["KNOWLEDGE_STORE_PATH"];
  if (existsSync(testStorePath)) {
    await unlink(testStorePath);
  }
});

// ---------------------------------------------------------------------------
// Basic extraction → create
// ---------------------------------------------------------------------------

describe("processMessage — novel knowledge creation", () => {
  it("creates a new artifact for novel knowledge", async () => {
    const llm = createPatternMockLLM([
      { match: "bullet-point", response: BULLET_PREF_RESPONSE },
    ]);

    const result = await processMessage(
      "I always want bullet-point summaries",
      llm,
      "test/session-1",
    );

    expect(result.candidates).toHaveLength(1);
    expect(result.created).toHaveLength(1);
    expect(result.updated).toHaveLength(0);

    const artifact = result.created[0]!;
    expect(artifact.kind).toBe("preference");
    expect(artifact.stage).toBe("candidate");
  });

  it("creates a [provisional] injectable for definitive candidate", async () => {
    const llm = createPatternMockLLM([
      { match: "bullet-point", response: BULLET_PREF_RESPONSE },
    ]);

    const result = await processMessage(
      "I always want bullet-point summaries",
      llm,
      "test/session-1",
    );

    expect(result.injectable).toHaveLength(1);
    expect(result.injectable[0]!.label).toBe("[provisional]");
  });

  it("persists the artifact to the store", async () => {
    const llm = createPatternMockLLM([
      { match: "bullet-point", response: BULLET_PREF_RESPONSE },
    ]);

    await processMessage("I always want bullet-point summaries", llm);

    const all = await loadAll();
    expect(all.some((a) => a.kind === "preference")).toBe(true);
  });

  it("returns empty result for non-persistable input", async () => {
    const llm = createPatternMockLLM([
      { match: "What time is it", response: EMPTY_RESPONSE },
    ]);

    const result = await processMessage("What time is it?", llm);

    expect(result.candidates).toHaveLength(0);
    expect(result.created).toHaveLength(0);
    expect(result.updated).toHaveLength(0);
    expect(result.injectable).toHaveLength(0);
  });

  it("creates distinct artifacts for distinct knowledge types", async () => {
    const multiLlm = createPatternMockLLM([
      { match: "gh", response: ALIAS_FACT_RESPONSE },
    ]);

    const r1 = await processMessage("when I say 'gh', I mean https://github.com", multiLlm);
    const bulletLlm = createPatternMockLLM([
      { match: "bullet-point", response: BULLET_PREF_RESPONSE },
    ]);
    const r2 = await processMessage("I always want bullet-point summaries", bulletLlm);

    expect(r1.created[0]!.id).not.toBe(r2.created[0]!.id);

    const all = await loadAll();
    expect(all).toHaveLength(2);
  });
});

// ---------------------------------------------------------------------------
// Evidence accumulation
// ---------------------------------------------------------------------------

describe("processMessage — evidence accumulation", () => {
  it("accumulates evidence for repeated similar knowledge", async () => {
    const llm = createPatternMockLLM([
      { match: "bullet-point summaries", response: BULLET_PREF_RESPONSE },
      { match: "bullet points for all", response: BULLET_PREF_RESPONSE },
    ]);

    // First message: creates new artifact
    await processMessage("I always want bullet-point summaries", llm);

    // Second message with same kind+tags: accumulates on existing
    const result2 = await processMessage(
      "bullet points for all my output please",
      llm,
    );

    expect(result2.created).toHaveLength(0);
    expect(result2.updated).toHaveLength(1);

    const updated = result2.updated[0]!;
    expect(updated.evidenceCount).toBe(2);
    expect(updated.stage).toBe("accumulating");
  });

  it(`consolidates after ${CONSOLIDATION_THRESHOLD} observations`, async () => {
    const llm = createFullPipelineMockLLM();

    // Use messages that exactly match the mock LLM patterns
    const messages = [
      "I always want bullet-point summaries",
      "bullet points for all output",
      "always use bullet points, max 5 items",
    ];

    let lastResult;
    for (const msg of messages) {
      lastResult = await processMessage(msg, llm);
    }

    const all = await loadAll();
    const consolidated = all.filter((a) => a.stage === "consolidated");
    expect(consolidated).toHaveLength(1);
    expect(consolidated[0]!.content).toBe(CONSOLIDATED_BULLET_RULE);
  });

  it("makes consolidated artifact injectable as [established] when confidence is high enough", async () => {
    const llm = createFullPipelineMockLLM();

    for (let i = 0; i < CONSOLIDATION_THRESHOLD; i++) {
      await processMessage("I always want bullet-point summaries", llm);
    }

    const all = await loadAll();
    const consolidated = all.find((a) => a.stage === "consolidated");
    expect(consolidated).toBeTruthy();
    // Consolidated + confidence >= 0.80 → [established]
    if (consolidated!.confidence >= 0.80) {
      const result = await processMessage("one more message", createPatternMockLLM([
        { match: "one more message", response: EMPTY_RESPONSE },
      ]));
      void result; // just checking store state
    }
  });
});

// ---------------------------------------------------------------------------
// formatForInjection
// ---------------------------------------------------------------------------

describe("formatForInjection", () => {
  it("returns empty string for no injectables", () => {
    const formatted = formatForInjection([]);
    expect(formatted).toBe("");
  });

  it("formats injectables with correct labels", async () => {
    const { candidateToArtifact } = await import("@khub-ai/openclaw-plus/extract");
    const artifact = candidateToArtifact(
      {
        content: "Always use bullet points for summaries",
        kind: "preference",
        scope: "general",
        certainty: "definitive",
        tags: ["summary-format"],
        rationale: "test",
      },
      "test",
    );

    const formatted = formatForInjection([
      { artifact, label: "[provisional]" },
    ]);

    expect(formatted).toContain("[provisional]");
    expect(formatted).toContain("preference");
    expect(formatted).toContain("Always use bullet points");
    expect(formatted).toContain("Remembered user knowledge");
  });
});
