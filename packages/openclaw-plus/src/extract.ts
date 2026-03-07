/**
 * LLM-backed knowledge extraction — PIL stages 1–3 (unified).
 *
 * All natural language understanding is delegated to the LLM. No
 * English-specific heuristics (signal words, hedge words, regex) appear here.
 * A user interacting in any language gets identical behavior.
 */

import { randomUUID } from "node:crypto";
import {
  type KnowledgeArtifact,
  type KnowledgeKind,
  type KnowledgeCertainty,
  type KnowledgeScope,
  type LLMFn,
  CONFIDENCE_SEED,
} from "./types.js";

// ---------------------------------------------------------------------------
// Intermediate representation: extraction candidate
// ---------------------------------------------------------------------------

/**
 * Structured output from the extraction LLM call (Stage 1).
 * One candidate per piece of persistable knowledge found in the message.
 */
export type ExtractionCandidate = {
  /** Verbatim or minimally paraphrased from the input, in the user's language. */
  content: string;
  kind: KnowledgeKind;
  scope: KnowledgeScope;
  certainty: KnowledgeCertainty;
  /**
   * 2–5 normalized topic tags.
   * Lowercase, hyphenated, English noun phrases.
   * Prefer vocabulary already in the store for consistency.
   */
  tags: string[];
  /** One sentence explaining why this is worth remembering. */
  rationale: string;
};

// ---------------------------------------------------------------------------
// Prompt builders
// ---------------------------------------------------------------------------

function buildExtractionPrompt(message: string, existingTags: string[]): string {
  const tagHint =
    existingTags.length > 0
      ? `Existing tag vocabulary (prefer these for consistency, add new ones if needed):\n${existingTags.slice(0, 60).join(", ")}`
      : "No existing tags yet — create new ones as needed.";

  return `You are a knowledge extraction assistant. Analyze the user message below and identify any reusable, persistable knowledge worth remembering for future interactions.

${tagHint}

Kind definitions (pick the best fit):
- preference:  subjective style, taste, or personal choice ("I prefer dark mode")
- convention:  agreed naming, terminology, or standards ("we call it the staging environment")
- fact:        objective, verifiable information ("the API endpoint is https://api.example.com/v2")
- procedure:   step-by-step process or recipe ("to deploy: run build, then push, then notify team")
- judgment:    evaluative heuristic or quality criterion ("favor brevity over completeness")
- strategy:    general approach to a class of problems ("when debugging, isolate variables first")

Certainty (how strongly was it expressed?):
- definitive: strong, unhedged — "always use strict mode", "I always want..."
- tentative:  qualified or hedged — "I usually prefer...", "maybe try...", "sometimes..."
- uncertain:  speculative or exploratory — "I think I might...", "not sure but..."

Scope:
- general:  applies broadly across many contexts
- specific: applies only to a particular situation or task

Return a JSON object with this exact structure:
{
  "candidates": [
    {
      "content": "exact quote or minimal paraphrase from the input, in the user's original language",
      "kind": "preference|convention|fact|procedure|judgment|strategy",
      "scope": "general|specific",
      "certainty": "definitive|tentative|uncertain",
      "tags": ["2-5 lowercase hyphenated English noun phrases"],
      "rationale": "one sentence explaining why this is worth remembering"
    }
  ]
}

Return { "candidates": [] } if the message contains no persistable, reusable knowledge (e.g., a simple question, greeting, one-off request with no generalizable content).

User message:
---
${message}
---

Return only valid JSON, no other text.`;
}

function buildConsolidationPrompt(kind: KnowledgeKind, observations: string[]): string {
  const numbered = observations.map((o, i) => `${i + 1}. ${o}`).join("\n");

  return `You are a knowledge consolidation assistant. Given the following ${observations.length} observations from a user (which may be in any language), distill them into a single, concise, reusable rule or pattern.

The distilled artifact should:
- Capture the general principle rather than the specific instances
- Be concise (1–3 sentences maximum)
- Be immediately actionable as a guideline for an AI assistant
- Be written in English (regardless of the input language)

Kind: ${kind}

Observations:
${numbered}

Return only the distilled rule or pattern as plain text. No JSON, no preamble, no explanation.`;
}

// ---------------------------------------------------------------------------
// JSON parse helper with markdown fence stripping
// ---------------------------------------------------------------------------

function parseJSON<T>(text: string): T | null {
  const trimmed = text.trim();
  // LLMs sometimes wrap output in markdown code fences
  const stripped = trimmed.startsWith("```")
    ? trimmed.replace(/^```(?:json)?\s*\n?/, "").replace(/\n?```\s*$/, "")
    : trimmed;
  try {
    return JSON.parse(stripped) as T;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

const VALID_KINDS = new Set<string>(["preference", "convention", "fact", "procedure", "judgment", "strategy"]);
const VALID_CERTAINTIES = new Set<string>(["definitive", "tentative", "uncertain"]);
const VALID_SCOPES = new Set<string>(["general", "specific"]);

function isValidCandidate(c: unknown): c is ExtractionCandidate {
  if (typeof c !== "object" || c === null) return false;
  const obj = c as Record<string, unknown>;
  return (
    typeof obj["content"] === "string" && obj["content"].trim().length > 0 &&
    typeof obj["kind"] === "string" && VALID_KINDS.has(obj["kind"]) &&
    typeof obj["certainty"] === "string" && VALID_CERTAINTIES.has(obj["certainty"]) &&
    typeof obj["scope"] === "string" && VALID_SCOPES.has(obj["scope"]) &&
    Array.isArray(obj["tags"]) && (obj["tags"] as unknown[]).every(t => typeof t === "string")
  );
}

// ---------------------------------------------------------------------------
// Stage 1 — Extract (LLM call)
// ---------------------------------------------------------------------------

/**
 * Call the LLM to extract persistable knowledge candidates from a message.
 *
 * Returns an empty array if the message contains no persistable knowledge.
 * Language-agnostic: the LLM handles all natural language understanding.
 *
 * @param message      - User's message in any language
 * @param existingTags - Tag vocabulary from the current store (for normalization)
 * @param llm          - LLM adapter function (dependency-injected)
 */
export async function extractFromMessage(
  message: string,
  existingTags: string[],
  llm: LLMFn,
): Promise<ExtractionCandidate[]> {
  if (!message.trim()) return [];

  const prompt = buildExtractionPrompt(message, existingTags);
  const response = await llm(prompt);

  const parsed = parseJSON<{ candidates: unknown[] }>(response);
  if (!parsed || !Array.isArray(parsed.candidates)) return [];

  // Validate and normalize each candidate
  return parsed.candidates
    .filter(isValidCandidate)
    .map((c) => ({
      ...c,
      content: c.content.trim(),
      tags: c.tags.map((t) => t.toLowerCase().replace(/\s+/g, "-")).filter(Boolean),
      rationale: typeof c.rationale === "string" ? c.rationale : "",
    }));
}

// ---------------------------------------------------------------------------
// Consolidation LLM call
// ---------------------------------------------------------------------------

/**
 * Consolidate multiple raw observations into a single distilled generalization.
 *
 * Called automatically by the store when `evidenceCount` reaches
 * `CONSOLIDATION_THRESHOLD`. The result replaces `content` in the artifact
 * (now a generalized rule); original observations remain in `evidence[]`.
 *
 * @param kind         - Artifact kind (included for prompt context)
 * @param observations - Raw verbatim observation strings in any language
 * @param llm          - LLM adapter function
 */
export async function consolidateEvidence(
  kind: KnowledgeKind,
  observations: string[],
  llm: LLMFn,
): Promise<string> {
  if (observations.length === 0) return "";
  const prompt = buildConsolidationPrompt(kind, observations);
  const result = await llm(prompt);
  return result.trim();
}

// ---------------------------------------------------------------------------
// Build artifact from extraction candidate
// ---------------------------------------------------------------------------

/**
 * Convert an `ExtractionCandidate` into a `KnowledgeArtifact`.
 *
 * Sets `stage: "candidate"`, `evidenceCount: 1`, `evidence: [content]`.
 * High-certainty ("definitive") candidates are injectable immediately as
 * `[provisional]` — see `getInjectLabel()` in store.ts.
 *
 * @param candidate  - Validated extraction candidate
 * @param provenance - Source session/message reference
 */
export function candidateToArtifact(
  candidate: ExtractionCandidate,
  provenance: string,
): KnowledgeArtifact {
  const confidence = CONFIDENCE_SEED[candidate.certainty] ?? CONFIDENCE_SEED.tentative;

  // Rationale is stored as part of the provenance string
  const fullProvenance = candidate.rationale
    ? `${provenance} — ${candidate.rationale}`
    : provenance;

  return {
    id: randomUUID(),
    kind: candidate.kind,
    content: candidate.content,
    confidence,
    provenance: fullProvenance,
    createdAt: new Date().toISOString(),
    // Enrichment
    scope: candidate.scope,
    certainty: candidate.certainty,
    tags: candidate.tags,
    // Lifecycle
    stage: "candidate",
    evidenceCount: 1,
    evidence: [candidate.content],
  };
}
