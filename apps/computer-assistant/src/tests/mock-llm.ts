/**
 * Mock LLM implementations for unit tests.
 *
 * No API calls — responses are crafted to simulate realistic LLM output
 * for specific test scenarios.
 */

import type { LLMFn } from "@khub-ai/openclaw-plus/types";

// ---------------------------------------------------------------------------
// Generic mock: map prompt substrings to responses
// ---------------------------------------------------------------------------

/**
 * Create a mock LLM that matches prompts by substring and returns
 * a predetermined response.
 *
 * Falls back to returning `{ "candidates": [] }` for unmatched prompts.
 */
export function createPatternMockLLM(
  patterns: Array<{ match: string; response: string }>,
): LLMFn {
  return async (prompt: string): Promise<string> => {
    for (const { match, response } of patterns) {
      if (prompt.includes(match)) return response;
    }
    return JSON.stringify({ candidates: [] });
  };
}

// ---------------------------------------------------------------------------
// Scenario-specific mock responses (extraction)
// ---------------------------------------------------------------------------

/** Extraction response for "I always want bullet-point summaries" */
export const BULLET_PREF_RESPONSE = JSON.stringify({
  candidates: [
    {
      content: "I always want bullet-point summaries",
      kind: "preference",
      scope: "general",
      certainty: "definitive",
      tags: ["summary-format", "bullet-points", "output-style"],
      rationale: "User has a strong preference for bullet-point format in summaries",
    },
  ],
});

/** Extraction response for a TypeScript coding convention */
export const TYPESCRIPT_CONVENTION_RESPONSE = JSON.stringify({
  candidates: [
    {
      content: "Use TypeScript with strict mode enabled",
      kind: "convention",
      scope: "general",
      certainty: "definitive",
      tags: ["typescript", "code-style", "strict-mode"],
      rationale: "User consistently requires TypeScript with strict mode",
    },
  ],
});

/** Extraction response for an alias/fact */
export const ALIAS_FACT_RESPONSE = JSON.stringify({
  candidates: [
    {
      content: "When user says 'gh', they mean https://github.com",
      kind: "convention",
      scope: "general",
      certainty: "definitive",
      tags: ["alias", "url-mapping", "github"],
      rationale: "User has defined a shortcut alias for GitHub",
    },
  ],
});

/** Extraction response for Chinese input: "我总是希望摘要使用项目符号" (I always want bullet summaries) */
export const CHINESE_PREF_RESPONSE = JSON.stringify({
  candidates: [
    {
      content: "我总是希望摘要使用项目符号",
      kind: "preference",
      scope: "general",
      certainty: "definitive",
      tags: ["summary-format", "bullet-points", "output-style"],
      rationale: "User wants bullet-point format for summaries (input in Chinese)",
    },
  ],
});

/** Extraction response for a multi-step procedure */
export const PROCEDURE_RESPONSE = JSON.stringify({
  candidates: [
    {
      content: "To deploy: run build, then push to main, then notify the team in Slack",
      kind: "procedure",
      scope: "general",
      certainty: "definitive",
      tags: ["deployment", "workflow", "build-process"],
      rationale: "User described a repeatable deployment procedure",
    },
  ],
});

/** Extraction response for tentative preference */
export const TENTATIVE_PREF_RESPONSE = JSON.stringify({
  candidates: [
    {
      content: "I usually prefer concise summaries, maybe 3-5 lines",
      kind: "preference",
      scope: "general",
      certainty: "tentative",
      tags: ["summary-format", "conciseness", "output-style"],
      rationale: "User expressed a qualified preference for concise output",
    },
  ],
});

/** No persistable knowledge (simple question) */
export const EMPTY_RESPONSE = JSON.stringify({ candidates: [] });

// ---------------------------------------------------------------------------
// Consolidation mock
// ---------------------------------------------------------------------------

/** Consolidated rule for multiple bullet-point preference observations */
export const CONSOLIDATED_BULLET_RULE =
  "Always use bullet points for summaries. Maximum 5 items. No filler phrases.";

/** Consolidated rule for multiple TypeScript observations */
export const CONSOLIDATED_TS_RULE =
  "All code should use TypeScript with strict mode enabled and async/await over raw promises.";

// ---------------------------------------------------------------------------
// Pre-built mock LLMs for common test scenarios
// ---------------------------------------------------------------------------

/**
 * Mock LLM that handles extraction for common test inputs.
 */
export function createExtractionMockLLM(): LLMFn {
  return createPatternMockLLM([
    { match: "bullet-point summaries",        response: BULLET_PREF_RESPONSE },
    { match: "TypeScript with strict mode",   response: TYPESCRIPT_CONVENTION_RESPONSE },
    { match: "when I say 'gh'",               response: ALIAS_FACT_RESPONSE },
    { match: "gh means https://github.com",   response: ALIAS_FACT_RESPONSE },
    { match: "我总是希望摘要",                  response: CHINESE_PREF_RESPONSE },
    { match: "To deploy:",                    response: PROCEDURE_RESPONSE },
    { match: "usually prefer concise",        response: TENTATIVE_PREF_RESPONSE },
  ]);
}

/**
 * Mock LLM that handles both extraction and consolidation.
 */
export function createFullPipelineMockLLM(): LLMFn {
  return createPatternMockLLM([
    // Consolidation patterns come FIRST — the consolidation prompt contains
    // evidence text that would also match extraction patterns if checked later.
    { match: "consolidation assistant",       response: CONSOLIDATED_BULLET_RULE },
    // Extraction patterns
    { match: "bullet-point summaries",        response: BULLET_PREF_RESPONSE },
    { match: "bullet points for all output",  response: BULLET_PREF_RESPONSE },
    { match: "always use bullet points",      response: BULLET_PREF_RESPONSE },
    { match: "TypeScript with strict mode",   response: TYPESCRIPT_CONVENTION_RESPONSE },
    { match: "when I say 'gh'",               response: ALIAS_FACT_RESPONSE },
    { match: "gh means https://github.com",   response: ALIAS_FACT_RESPONSE },
    { match: "我总是希望摘要",                  response: CHINESE_PREF_RESPONSE },
    { match: "To deploy:",                    response: PROCEDURE_RESPONSE },
    { match: "usually prefer concise",        response: TENTATIVE_PREF_RESPONSE },
  ]);
}
