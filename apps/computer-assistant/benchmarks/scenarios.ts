/**
 * Curated benchmark scenarios for evaluating PIL pipeline effectiveness.
 *
 * Each scenario specifies:
 *   - input:         the user message
 *   - expectedKind:  the expected knowledge kind (or null if no knowledge expected)
 *   - expectedCertainty: the expected certainty level
 *   - expectedTags:  tags that MUST be present in the extracted artifact
 *   - shouldExtract: whether ANY knowledge should be extracted
 */

export type ExtractionScenario = {
  id: string;
  description: string;
  input: string;
  shouldExtract: boolean;
  expectedKind?: string;
  expectedCertainty?: string;
  /** At least one of these tags must appear in the extracted artifact */
  expectedTagsAny?: string[];
  /** All of these tags must appear */
  expectedTagsAll?: string[];
};

export const EXTRACTION_SCENARIOS: ExtractionScenario[] = [
  // ── Definitive preferences ──────────────────────────────────────────────
  {
    id: "pref-1",
    description: "Strong preference for bullet points",
    input: "I always want bullet-point summaries, no more than five points.",
    shouldExtract: true,
    expectedKind: "preference",
    expectedCertainty: "definitive",
    expectedTagsAny: ["bullet-points", "summary-format", "output-style"],
  },
  {
    id: "pref-2",
    description: "Code style convention",
    input: "When writing code, always use TypeScript with strict mode enabled.",
    shouldExtract: true,
    expectedKind: "preference",
    expectedCertainty: "definitive",
    expectedTagsAny: ["typescript", "code-style", "strict-mode"],
  },
  {
    id: "pref-3",
    description: "Async/await preference",
    input: "Use async/await rather than raw promises wherever possible.",
    shouldExtract: true,
    expectedCertainty: "definitive",
    expectedTagsAny: ["async-await", "code-style", "javascript"],
  },

  // ── Facts and conventions ────────────────────────────────────────────────
  {
    id: "fact-1",
    description: "Alias definition",
    input: "When I say 'gh', I mean https://github.com",
    shouldExtract: true,
    expectedKind: "convention",
    expectedCertainty: "definitive",
    expectedTagsAny: ["alias", "github", "url-mapping"],
  },
  {
    id: "fact-2",
    description: "API endpoint fact",
    input: "Our staging API is at https://staging.api.example.com/v2",
    shouldExtract: true,
    expectedKind: "fact",
    expectedTagsAny: ["api", "endpoint", "staging"],
  },
  {
    id: "fact-3",
    description: "File naming convention",
    input: "I always name financial statements as YYYY-MM-institution-account.pdf",
    shouldExtract: true,
    expectedTagsAny: ["file-naming", "naming-convention"],
  },

  // ── Procedures ───────────────────────────────────────────────────────────
  {
    id: "proc-1",
    description: "Deployment procedure",
    input: "To deploy: run pnpm build, then git push origin main, then notify the team on Slack.",
    shouldExtract: true,
    expectedKind: "procedure",
    expectedCertainty: "definitive",
    expectedTagsAny: ["deployment", "workflow", "build-process"],
  },
  {
    id: "proc-2",
    description: "Morning routine procedure",
    input: "Every morning I check emails first, then Slack, then review GitHub PRs.",
    shouldExtract: true,
    expectedKind: "procedure",
    expectedTagsAny: ["workflow", "morning-routine"],
  },

  // ── Judgment / evaluative ────────────────────────────────────────────────
  {
    id: "judge-1",
    description: "Quality criterion",
    input: "Good code is readable first, then efficient. Never sacrifice clarity for performance.",
    shouldExtract: true,
    expectedKind: "judgment",
    expectedTagsAny: ["code-quality", "readability"],
  },
  {
    id: "judge-2",
    description: "Summary quality criterion",
    input: "A good summary leads with the takeaway, not the background. Always action-first.",
    shouldExtract: true,
    expectedTagsAny: ["summary-format", "communication-style"],
  },

  // ── Tentative / uncertain ────────────────────────────────────────────────
  {
    id: "tent-1",
    description: "Hedged preference",
    input: "I usually prefer shorter responses, maybe 3-5 sentences.",
    shouldExtract: true,
    expectedCertainty: "tentative",
    expectedTagsAny: ["output-style", "conciseness"],
  },
  {
    id: "tent-2",
    description: "Uncertain/speculative preference",
    input: "I'm not sure, but I think I might prefer dark mode for the UI?",
    shouldExtract: true,
    expectedCertainty: "uncertain",
  },

  // ── Non-persistable inputs (should extract nothing) ─────────────────────
  {
    id: "none-1",
    description: "Simple question — no persistable knowledge",
    input: "What time is it?",
    shouldExtract: false,
  },
  {
    id: "none-2",
    description: "One-off request — no generalizable knowledge",
    input: "Open README.md for me",
    shouldExtract: false,
  },
  {
    id: "none-3",
    description: "Greeting — no knowledge",
    input: "Hello! How are you?",
    shouldExtract: false,
  },
  {
    id: "none-4",
    description: "Simple calculation request",
    input: "What is 15% of 240?",
    shouldExtract: false,
  },
  {
    id: "none-5",
    description: "Acknowledgement",
    input: "OK thanks",
    shouldExtract: false,
  },

  // ── Multilingual inputs ──────────────────────────────────────────────────
  {
    id: "lang-1",
    description: "Chinese preference (bullet points)",
    input: "我总是希望摘要使用项目符号，不超过五点。",
    shouldExtract: true,
    expectedKind: "preference",
    expectedTagsAny: ["bullet-points", "summary-format"],
  },
  {
    id: "lang-2",
    description: "Spanish convention (alias)",
    input: "Cuando digo 'gh' me refiero a https://github.com",
    shouldExtract: true,
    expectedTagsAny: ["alias", "github"],
  },
];

// ---------------------------------------------------------------------------
// Retrieval benchmark scenarios
// ---------------------------------------------------------------------------

export type RetrievalScenario = {
  id: string;
  description: string;
  /** Artifacts to seed the store with before retrieval */
  seedArtifacts: Array<{
    kind: string;
    content: string;
    tags: string[];
    confidence: number;
  }>;
  query: string;
  /** IDs of seed artifacts that MUST appear in top-k results */
  expectedTopK: number[];
  k: number;
};

export const RETRIEVAL_SCENARIOS: RetrievalScenario[] = [
  {
    id: "retr-1",
    description: "Exact tag match retrieval",
    seedArtifacts: [
      {
        kind: "preference",
        content: "Always use bullet points for summaries",
        tags: ["summary-format", "bullet-points"],
        confidence: 0.80,
      },
      {
        kind: "preference",
        content: "Use async/await over raw promises",
        tags: ["code-style", "async-await"],
        confidence: 0.75,
      },
    ],
    query: "summary bullet points",
    expectedTopK: [0],
    k: 3,
  },
  {
    id: "retr-2",
    description: "Alias lookup",
    seedArtifacts: [
      {
        kind: "convention",
        content: "When user says 'gh', they mean https://github.com",
        tags: ["alias", "github", "url-mapping"],
        confidence: 0.65,
      },
    ],
    query: "open gh",
    expectedTopK: [0],
    k: 3,
  },
];
