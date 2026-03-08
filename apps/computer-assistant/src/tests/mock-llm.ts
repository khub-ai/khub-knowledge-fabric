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

/**
 * Create a mock LLM that supports multi-string matching.
 *
 * When `match` is an array, ALL strings must appear in the prompt.
 * When `match` is a string, behaves like createPatternMockLLM.
 *
 * Falls back to returning `{ "candidates": [] }` for unmatched prompts.
 *
 * Use this when you need to distinguish prompts that share a common
 * substring (e.g., a semantic match prompt that also mentions "atp" vs
 * an extraction prompt that also mentions "atp").
 */
export function createMultiPatternMockLLM(
  patterns: Array<{ match: string | string[]; response: string }>,
): LLMFn {
  return async (prompt: string): Promise<string> => {
    for (const { match, response } of patterns) {
      const matched = Array.isArray(match)
        ? match.every((m) => prompt.includes(m))
        : prompt.includes(match);
      if (matched) return response;
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

// ---------------------------------------------------------------------------
// lmp acronym benchmark mocks
// ---------------------------------------------------------------------------
//
// These mocks simulate the LLM's behavior in the "lmp" acronym scenario.
// Key design choice: the ATP and DS extraction responses use DIFFERENT tags
// from the LMP response, so Jaccard tag matching fails — forcing the test to
// exercise the semantic matching path.
//
// Semantic match prompts are identified by the phrase
// "SAME underlying behavioral habit" (from buildSemanticMatchPrompt in store.ts).

/** Exchange-pass extraction for "lmp = list my preferences" */
export const LMP_EXCHANGE_RESPONSE = JSON.stringify({
  candidates: [
    {
      content: "'lmp' is the user's shorthand for 'list my preferences'",
      kind: "convention",
      scope: "general",
      certainty: "definitive",
      tags: ["lmp", "shorthand-command", "acronym"],
      rationale: "User defined a personal acronym for a frequently-used command",
    },
  ],
});

/**
 * Exchange-pass extraction for "atp = add to preferences".
 *
 * Intentionally uses DIFFERENT tags from LMP_EXCHANGE_RESPONSE so that
 * Jaccard tag similarity = 0, forcing the semantic matching path.
 */
export const ATP_EXCHANGE_RESPONSE = JSON.stringify({
  candidates: [
    {
      content: "'atp' is the user's shorthand for 'add to preferences'",
      kind: "convention",
      scope: "general",
      certainty: "definitive",
      tags: ["atp", "command-alias"],  // no overlap with lmp tags → Jaccard = 0
      rationale: "User defined a personal acronym for a frequently-used command",
    },
  ],
});

/**
 * Exchange-pass extraction for "ds = debug session".
 *
 * Also uses different tags to force the semantic matching path.
 */
export const DS_EXCHANGE_RESPONSE = JSON.stringify({
  candidates: [
    {
      content: "'ds' is the user's shorthand for 'debug session'",
      kind: "convention",
      scope: "general",
      certainty: "definitive",
      tags: ["ds", "abbreviation"],  // no overlap with prior tags → Jaccard = 0
      rationale: "User defined a personal acronym for a debugging workflow command",
    },
  ],
});

/** Consolidated rule produced after 3 acronym observations */
export const ACRONYM_CONSOLIDATION_RULE =
  "User regularly defines personal shorthand acronyms for frequently-used commands (e.g., lmp = list preferences, atp = add to preferences, ds = debug session).";

// ---------------------------------------------------------------------------
// Financial statements benchmark mocks
// ---------------------------------------------------------------------------
//
// These mocks simulate the LLM's behaviour for the scenario described in
// docs/example-learning-in-action.md.
//
// Key design choices:
//   - Session 1 ("January" one-off) → falls through to default empty response
//   - Session 2 ("always use that pattern") → convention artifact
//   - Session 3 ("Add Amex too") → BOTH a fact AND a procedure in one call
//   - Sessions 4+ (retrieval / revision) → no extraction, no mock needed

/** Extraction response for Session 2 — user confirms naming convention */
export const FINANCE_NAMING_CONVENTION_RESPONSE = JSON.stringify({
  candidates: [
    {
      content:
        "Financial statements are named YYYY-MM-institution-account.pdf and stored in ~/Finance/Statements/",
      kind: "convention",
      scope: "general",
      certainty: "definitive",
      tags: ["file-naming", "financial-statements", "naming-convention"],
      rationale:
        "User confirmed a consistent naming pattern for all financial statement files",
    },
  ],
});

/**
 * Extraction response for Session 3.
 *
 * Returns TWO candidates from a single extraction call — this exercises the
 * pipeline's ability to create multiple artifacts from one exchange:
 *   1. fact      — the user's institution list (Chase, Fidelity, Amex)
 *   2. procedure — the monthly download checklist
 *
 * The tags are deliberately distinct so Jaccard matching between them is zero
 * and each is stored as an independent artifact.
 */
export const FINANCE_SESSION3_RESPONSE = JSON.stringify({
  candidates: [
    {
      content:
        "User's financial institutions: Chase (checking), Fidelity (brokerage), Amex (credit card)",
      kind: "fact",
      scope: "general",
      certainty: "definitive",
      tags: ["financial-institutions", "accounts", "chase", "fidelity", "amex"],
      rationale:
        "User listed the financial institutions they track for monthly statement downloads",
    },
    {
      content:
        "Monthly statement download: for each institution [Chase, Fidelity, Amex]: log in, download PDF, rename to YYYY-MM-institution-account.pdf, save to ~/Finance/Statements/. Confirm all files are present. Frequency: once per month after the 1st.",
      kind: "procedure",
      scope: "general",
      certainty: "definitive",
      tags: ["monthly-statements", "download-procedure", "financial-statements"],
      rationale:
        "User requested a monthly checklist for downloading statements from all their institutions",
    },
  ],
});

/**
 * Mock LLM for the financial statements benchmark.
 *
 * Handles, in priority order:
 *   1. Session 2 — user confirms naming pattern ("always use that pattern")
 *   2. Session 3 — user adds institution list and checklist ("Add Amex too")
 *   3. Everything else → empty candidates (one-off task or execution phase)
 */
export function createFinanceBenchmarkLLM(): LLMFn {
  return createMultiPatternMockLLM([
    {
      match: "always use that pattern",
      response: FINANCE_NAMING_CONVENTION_RESPONSE,
    },
    {
      match: "Add Amex too",
      response: FINANCE_SESSION3_RESPONSE,
    },
  ]);
}

// ---------------------------------------------------------------------------
// Compile-procedure benchmark mocks
// ---------------------------------------------------------------------------
//
// These simulate the code-generation LLM call triggered by compileToProgram().
// The mock returns a realistic Python script so tests can verify the full
// artifact → code → persist → retrieve lifecycle without hitting an API.
//
// The code-gen prompt is identified by the phrase "PROCEDURE TO AUTOMATE"
// (from buildCodeGenPrompt in pipeline.ts).

/**
 * Realistic Python automation script returned by the mock code-gen LLM.
 *
 * Structured to match what a real LLM would generate from the financial
 * statements procedure and its supporting conventions / facts.
 */
export const COMPILE_CODE_RESPONSE = `#!/usr/bin/env python3
"""
Download financial statements from all institutions, rename them to
YYYY-MM-institution-account.pdf, and save to ~/Finance/Statements/.

Credentials are read from environment variables:
  CHASE_USER, CHASE_PASS, FIDELITY_USER, FIDELITY_PASS, AMEX_USER, AMEX_PASS
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# -- Configuration -----------------------------------------------------------
STATEMENTS_DIR = Path.home() / "Finance" / "Statements"

INSTITUTIONS = [
    {"name": "chase",    "account": "checking",  "user_env": "CHASE_USER",    "pass_env": "CHASE_PASS"},
    {"name": "fidelity", "account": "brokerage", "user_env": "FIDELITY_USER", "pass_env": "FIDELITY_PASS"},
    {"name": "amex",     "account": "credit",    "user_env": "AMEX_USER",     "pass_env": "AMEX_PASS"},
]


def statement_filename(institution: dict) -> str:
    """Return the canonical filename for this month's statement."""
    now = datetime.now()
    return f"{now.year:04d}-{now.month:02d}-{institution['name']}-{institution['account']}.pdf"


def download_statement(institution: dict) -> None:
    """Download and save one institution's statement."""
    filename = statement_filename(institution)
    dest = STATEMENTS_DIR / filename
    print(f"  Downloading {filename} ...")
    # TODO: implement institution-specific download using requests/playwright
    # dest.write_bytes(pdf_bytes)
    print(f"  Saved to {dest}")


def main() -> None:
    STATEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving statements to: {STATEMENTS_DIR}")
    for inst in INSTITUTIONS:
        try:
            download_statement(inst)
        except Exception as exc:
            print(f"ERROR ({inst['name']}): {exc}", file=sys.stderr)
            sys.exit(1)
    print("All statements downloaded successfully.")


if __name__ == "__main__":
    main()
`;

/**
 * Mock LLM for the compile-procedure benchmark.
 *
 * Handles, in priority order:
 *   1. Code-generation prompt (identified by "PROCEDURE TO AUTOMATE") — raw Python
 *   2. Financial-statements setup exchanges (finance benchmark patterns)
 */
export function createCompileBenchmarkLLM(): LLMFn {
  return createMultiPatternMockLLM([
    // Code-gen call: identified by the unique phrase from buildCodeGenPrompt in pipeline.ts.
    // Returns raw Python source (not JSON) — compileToProgram() uses the response verbatim.
    { match: "PROCEDURE TO AUTOMATE", response: COMPILE_CODE_RESPONSE },
    // Finance benchmark patterns — used to set up sessions 1–3 before compiling
    { match: "always use that pattern", response: FINANCE_NAMING_CONVENTION_RESPONSE },
    { match: "Add Amex too",           response: FINANCE_SESSION3_RESPONSE },
  ]);
}

/**
 * Mock LLM for the lmp acronym benchmark.
 *
 * Handles, in order:
 *   1. Consolidation prompts (identified by "consolidation assistant")
 *   2. Semantic match prompts (identified by "SAME underlying behavioral habit")
 *      — returns "1" when the candidate is an acronym-definition convention
 *   3. Exchange-pass extraction prompts
 */
export function createLmpBenchmarkLLM(): LLMFn {
  return createMultiPatternMockLLM([
    // ── Consolidation (must come first: the prompt contains evidence text
    //    that would also match the extraction patterns below) ─────────────────
    { match: "consolidation assistant",
      response: ACRONYM_CONSOLIDATION_RULE },

    // ── Semantic matching (identified by unique phrase from store.ts prompt)
    //    Returns "1" when the new candidate matches the stored acronym pattern.
    //    All three acronyms (atp, ds) should group with the first (lmp). ─────
    { match: ["SAME underlying behavioral habit", "add to preferences"],
      response: "1" },
    { match: ["SAME underlying behavioral habit", "debug session"],
      response: "1" },
    // Fallback: no semantic match for anything else
    { match: "SAME underlying behavioral habit",
      response: "NONE" },

    // ── Exchange-pass extraction ─────────────────────────────────────────────
    //    Matched by unique content phrases from the simulated exchange texts.
    { match: "list my preferences",  response: LMP_EXCHANGE_RESPONSE },
    { match: "add to preferences",   response: ATP_EXCHANGE_RESPONSE },
    { match: "debug session",        response: DS_EXCHANGE_RESPONSE },
  ]);
}
