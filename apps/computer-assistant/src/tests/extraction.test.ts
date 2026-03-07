/**
 * Unit tests for PIL extraction (extract.ts).
 *
 * All tests use mock LLMs — no API calls required.
 */

import { describe, it, expect, beforeEach } from "vitest";
import {
  extractFromMessage,
  consolidateEvidence,
  candidateToArtifact,
} from "@khub-ai/openclaw-plus/extract";
import { CONFIDENCE_SEED } from "@khub-ai/openclaw-plus/types";
import {
  createPatternMockLLM,
  BULLET_PREF_RESPONSE,
  TYPESCRIPT_CONVENTION_RESPONSE,
  ALIAS_FACT_RESPONSE,
  CHINESE_PREF_RESPONSE,
  PROCEDURE_RESPONSE,
  TENTATIVE_PREF_RESPONSE,
  EMPTY_RESPONSE,
  CONSOLIDATED_BULLET_RULE,
} from "./mock-llm.js";

// ---------------------------------------------------------------------------
// extractFromMessage
// ---------------------------------------------------------------------------

describe("extractFromMessage", () => {
  it("extracts a strong preference from English input", async () => {
    const llm = createPatternMockLLM([
      { match: "bullet-point", response: BULLET_PREF_RESPONSE },
    ]);

    const candidates = await extractFromMessage(
      "I always want bullet-point summaries",
      [],
      llm,
    );

    expect(candidates).toHaveLength(1);
    const c = candidates[0]!;
    expect(c.kind).toBe("preference");
    expect(c.certainty).toBe("definitive");
    expect(c.scope).toBe("general");
    expect(c.tags).toContain("summary-format");
    expect(c.tags).toContain("bullet-points");
    expect(c.content).toBeTruthy();
  });

  it("extracts a convention/fact (alias definition)", async () => {
    const llm = createPatternMockLLM([
      { match: "gh", response: ALIAS_FACT_RESPONSE },
    ]);

    const candidates = await extractFromMessage(
      "when I say 'gh', I mean https://github.com",
      [],
      llm,
    );

    expect(candidates).toHaveLength(1);
    const c = candidates[0]!;
    expect(c.kind).toBe("convention");
    expect(c.tags).toContain("alias");
  });

  it("extracts a procedure correctly", async () => {
    const llm = createPatternMockLLM([
      { match: "To deploy:", response: PROCEDURE_RESPONSE },
    ]);

    const candidates = await extractFromMessage(
      "To deploy: run build, then push to main, then notify the team in Slack",
      [],
      llm,
    );

    expect(candidates).toHaveLength(1);
    expect(candidates[0]!.kind).toBe("procedure");
    expect(candidates[0]!.tags).toContain("deployment");
  });

  it("assigns tentative certainty for hedged statements", async () => {
    const llm = createPatternMockLLM([
      { match: "usually prefer", response: TENTATIVE_PREF_RESPONSE },
    ]);

    const candidates = await extractFromMessage(
      "I usually prefer concise summaries, maybe 3-5 lines",
      [],
      llm,
    );

    expect(candidates[0]!.certainty).toBe("tentative");
  });

  it("returns empty array for non-persistable messages", async () => {
    const llm = createPatternMockLLM([
      { match: "What time is it", response: EMPTY_RESPONSE },
    ]);

    const candidates = await extractFromMessage("What time is it?", [], llm);
    expect(candidates).toHaveLength(0);
  });

  it("returns empty array for empty input", async () => {
    const llm = createPatternMockLLM([]);
    const candidates = await extractFromMessage("", [], llm);
    expect(candidates).toHaveLength(0);
  });

  it("handles non-English input (Chinese) with same output structure", async () => {
    const llm = createPatternMockLLM([
      { match: "我总是希望摘要", response: CHINESE_PREF_RESPONSE },
    ]);

    const candidates = await extractFromMessage(
      "我总是希望摘要使用项目符号",
      [],
      llm,
    );

    expect(candidates).toHaveLength(1);
    const c = candidates[0]!;
    // Kind, certainty, and tags are in English regardless of input language
    expect(c.kind).toBe("preference");
    expect(c.certainty).toBe("definitive");
    expect(c.tags).toContain("bullet-points");
    // Content is preserved in the original language
    expect(c.content).toContain("摘要");
  });

  it("normalizes tags to lowercase-hyphenated format", async () => {
    const llm = createPatternMockLLM([
      { match: "TypeScript", response: TYPESCRIPT_CONVENTION_RESPONSE },
    ]);

    const candidates = await extractFromMessage(
      "Use TypeScript with strict mode enabled",
      [],
      llm,
    );

    const tags = candidates[0]!.tags;
    // All tags should be lowercase and hyphenated
    expect(tags.every((t) => t === t.toLowerCase())).toBe(true);
    expect(tags.every((t) => !/\s/.test(t))).toBe(true);
  });

  it("gracefully handles malformed LLM response", async () => {
    const llm = createPatternMockLLM([
      { match: "anything", response: "this is not valid JSON" },
    ]);

    const candidates = await extractFromMessage("anything at all", [], llm);
    expect(candidates).toHaveLength(0);
  });

  it("gracefully handles LLM response wrapped in markdown code fence", async () => {
    const fenced = "```json\n" + BULLET_PREF_RESPONSE + "\n```";
    const llm = createPatternMockLLM([
      { match: "bullet-point", response: fenced },
    ]);

    const candidates = await extractFromMessage(
      "I always want bullet-point summaries",
      [],
      llm,
    );
    expect(candidates).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// consolidateEvidence
// ---------------------------------------------------------------------------

describe("consolidateEvidence", () => {
  it("returns a consolidated rule from multiple observations", async () => {
    const llm = createPatternMockLLM([
      { match: "consolidation assistant", response: CONSOLIDATED_BULLET_RULE },
    ]);

    const result = await consolidateEvidence(
      "preference",
      [
        "I always want bullet-point summaries",
        "Please use bullet points for all output",
        "Always use bullet points, max 5 items",
      ],
      llm,
    );

    expect(result).toBeTruthy();
    expect(result.length).toBeGreaterThan(10);
  });

  it("returns empty string for empty observations", async () => {
    const llm = createPatternMockLLM([]);
    const result = await consolidateEvidence("preference", [], llm);
    expect(result).toBe("");
  });
});

// ---------------------------------------------------------------------------
// candidateToArtifact
// ---------------------------------------------------------------------------

describe("candidateToArtifact", () => {
  it("creates a candidate-stage artifact with correct confidence", () => {
    const candidate = {
      content: "I always want bullet-point summaries",
      kind: "preference" as const,
      scope: "general" as const,
      certainty: "definitive" as const,
      tags: ["summary-format", "bullet-points"],
      rationale: "User preference",
    };

    const artifact = candidateToArtifact(candidate, "test/session");

    expect(artifact.kind).toBe("preference");
    expect(artifact.confidence).toBe(CONFIDENCE_SEED.definitive);
    expect(artifact.stage).toBe("candidate");
    expect(artifact.evidenceCount).toBe(1);
    expect(artifact.evidence).toHaveLength(1);
    expect(artifact.evidence![0]).toBe(candidate.content);
    expect(artifact.tags).toEqual(["summary-format", "bullet-points"]);
    expect(artifact.id).toBeTruthy();
    expect(artifact.createdAt).toBeTruthy();
  });

  it("sets tentative confidence for tentative certainty", () => {
    const candidate = {
      content: "I usually prefer concise summaries",
      kind: "preference" as const,
      scope: "general" as const,
      certainty: "tentative" as const,
      tags: ["summary-format"],
      rationale: "Qualified preference",
    };

    const artifact = candidateToArtifact(candidate, "test/session");
    expect(artifact.confidence).toBe(CONFIDENCE_SEED.tentative);
  });

  it("sets uncertain confidence for uncertain certainty", () => {
    const candidate = {
      content: "I think I might prefer shorter responses?",
      kind: "preference" as const,
      scope: "general" as const,
      certainty: "uncertain" as const,
      tags: ["output-style"],
      rationale: "Speculative preference",
    };

    const artifact = candidateToArtifact(candidate, "test/session");
    expect(artifact.confidence).toBe(CONFIDENCE_SEED.uncertain);
  });

  it("includes rationale in provenance", () => {
    const candidate = {
      content: "Use TypeScript with strict mode",
      kind: "convention" as const,
      scope: "general" as const,
      certainty: "definitive" as const,
      tags: ["typescript"],
      rationale: "User requires strict TypeScript",
    };

    const artifact = candidateToArtifact(candidate, "session-abc");
    expect(artifact.provenance).toContain("session-abc");
    expect(artifact.provenance).toContain("User requires strict TypeScript");
  });

  it("generates a unique id for each artifact", () => {
    const candidate = {
      content: "content",
      kind: "fact" as const,
      scope: "general" as const,
      certainty: "definitive" as const,
      tags: ["test"],
      rationale: "test",
    };

    const a1 = candidateToArtifact(candidate, "s");
    const a2 = candidateToArtifact(candidate, "s");
    expect(a1.id).not.toBe(a2.id);
  });
});
