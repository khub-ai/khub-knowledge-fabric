/**
 * PIL pipeline — orchestration layer.
 *
 * Combines extraction (extract.ts) and storage (store.ts) to process a user
 * message end-to-end: extract knowledge → match against store → accumulate
 * or create → determine what is injectable.
 *
 * Backward-compatibility note: the old synchronous elicit / induce / validate /
 * compact functions are retained below (marked @deprecated) so that existing
 * code (e.g., the playground) continues to compile without changes.
 */

import { randomUUID } from "node:crypto";
import {
  type KnowledgeArtifact,
  type KnowledgeKind,
  type LLMFn,
} from "./types.js";
import {
  type ExtractionCandidate,
  extractFromMessage,
  candidateToArtifact,
} from "./extract.js";
import {
  getActiveTags,
  matchCandidate,
  accumulateEvidence,
  persist,
  getInjectLabel,
  type InjectLabel,
} from "./store.js";

// Re-export core types for backward compatibility
export type { KnowledgeKind, KnowledgeArtifact } from "./types.js";
export type { ExtractionCandidate } from "./extract.js";

// ---------------------------------------------------------------------------
// processMessage result types
// ---------------------------------------------------------------------------

export type InjectableArtifact = {
  artifact: KnowledgeArtifact;
  label: InjectLabel;
};

export type ProcessResult = {
  /** Candidates the LLM extracted from the message */
  candidates: ExtractionCandidate[];
  /** Newly created artifacts (novel knowledge not seen before) */
  created: KnowledgeArtifact[];
  /** Updated artifacts (evidence accumulated or newly consolidated) */
  updated: KnowledgeArtifact[];
  /**
   * Artifacts from this message that are injectable into the current session.
   *
   * Note: for retrieving artifacts from *previous* sessions relevant to a
   * query, use retrieve() from store.ts directly.
   */
  injectable: InjectableArtifact[];
};

// ---------------------------------------------------------------------------
// Main pipeline entrypoint: processMessage
// ---------------------------------------------------------------------------

/**
 * Process a user message through the full PIL pipeline.
 *
 * Stages:
 *   1. Extract — LLM identifies persistable knowledge candidates
 *   2. Match   — compare each candidate against existing store artifacts
 *   3. Resolve — novel → create; accumulating → add evidence; confident → update stats
 *   4. Decide  — determine which newly-created/updated artifacts are injectable
 *
 * @param message    - User's message in any language
 * @param llm        - LLM adapter function (used for extraction + consolidation)
 * @param provenance - Session or message reference stored with each artifact
 */
export async function processMessage(
  message: string,
  llm: LLMFn,
  provenance = "unknown",
): Promise<ProcessResult> {
  // ── Stage 1: Extract ──────────────────────────────────────────────────────
  const existingTags = await getActiveTags();
  const candidates = await extractFromMessage(message, existingTags, llm);

  const created: KnowledgeArtifact[] = [];
  const updated: KnowledgeArtifact[] = [];

  // ── Stages 2 & 3: Match → Resolve ────────────────────────────────────────
  for (const candidate of candidates) {
    const match = await matchCandidate(candidate);

    if (match === null) {
      // Novel: create new candidate artifact
      const artifact = candidateToArtifact(candidate, provenance);
      await persist(artifact);
      created.push(artifact);
    } else {
      // Accumulating or confident: add evidence, trigger consolidation if ready
      const updated_ = await accumulateEvidence(match, candidate.content, llm);
      updated.push(updated_);
    }
  }

  // ── Stage 4: Decide injectable ────────────────────────────────────────────
  const injectable: InjectableArtifact[] = [];
  for (const a of [...created, ...updated]) {
    const label = getInjectLabel(a);
    if (label) injectable.push({ artifact: a, label });
  }

  return { candidates, created, updated, injectable };
}

// ---------------------------------------------------------------------------
// Format injectable artifacts for LLM prompt injection
// ---------------------------------------------------------------------------

/**
 * Render a list of injectable artifacts as a formatted string suitable for
 * insertion into an LLM system prompt or context block.
 *
 * Labels:
 *   [established]  — consolidated, high-confidence knowledge; treat as fact
 *   [suggestion]   — consolidated but lower-confidence; apply with judgment
 *   [provisional]  — single strong observation; use with caution
 */
export function formatForInjection(injectables: InjectableArtifact[]): string {
  if (injectables.length === 0) return "";

  const lines = injectables.map(
    ({ artifact, label }) => `${label} [${artifact.kind}] ${artifact.content}`,
  );

  return (
    "Remembered user knowledge (apply as appropriate):\n" + lines.join("\n")
  );
}

// ---------------------------------------------------------------------------
// Backward-compatible synchronous stubs (deprecated)
// ---------------------------------------------------------------------------

/** @deprecated Use processMessage() with an LLM adapter instead. */
export function elicit(input: string): string[] {
  // Naive sentence splitter — retained for backward compat only
  return input
    .split(/(?<=[.!?])\s+|(?<=[.!?])$/)
    .map((s) => s.trim())
    .filter(Boolean);
}

/** @deprecated Use processMessage() with an LLM adapter instead. */
export function induce(
  candidate: string,
  provenance = "unknown",
): KnowledgeArtifact | null {
  const trimmed = candidate.trim();
  if (!trimmed) return null;
  return {
    id: randomUUID(),
    kind: "fact" as KnowledgeKind,
    content: trimmed,
    confidence: 0.5,
    provenance,
    createdAt: new Date().toISOString(),
  };
}

/** @deprecated Use processMessage() with an LLM adapter instead. */
export function validate(artifact: KnowledgeArtifact): KnowledgeArtifact {
  return artifact; // No-op; validation now performed by LLM at extraction
}

/** @deprecated Use processMessage() with an LLM adapter instead. */
export function compact(artifact: KnowledgeArtifact): KnowledgeArtifact {
  const content = artifact.content
    .trim()
    .replace(/\s+/g, " ")
    .replace(/\s([.,!?;:])/g, "$1");
  return { ...artifact, content };
}
