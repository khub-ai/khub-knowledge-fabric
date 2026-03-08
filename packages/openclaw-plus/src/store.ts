/**
 * Knowledge store — PIL stages 5–8 (persist, retrieve, apply, revise).
 *
 * Persistence: JSONL file, one artifact per line.
 * Default location: ~/.openclaw/knowledge/artifacts.jsonl
 * Override via env: KNOWLEDGE_STORE_PATH
 */

import { readFile, writeFile, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join, dirname } from "node:path";
import {
  type KnowledgeArtifact,
  type KnowledgeKind,
  type LLMFn,
  CONSOLIDATION_THRESHOLD,
  DEFAULT_AUTO_APPLY_THRESHOLD,
  AUTO_APPLY_THRESHOLDS,
} from "./types.js";
import { consolidateEvidence } from "./extract.js";

// ---------------------------------------------------------------------------
// Storage helpers
// ---------------------------------------------------------------------------

export function storePath(): string {
  return (
    process.env["KNOWLEDGE_STORE_PATH"] ??
    join(homedir(), ".openclaw", "knowledge", "artifacts.jsonl")
  );
}

async function ensureDir(filePath: string): Promise<void> {
  const dir = dirname(filePath);
  if (!existsSync(dir)) await mkdir(dir, { recursive: true });
}

export async function loadAll(): Promise<KnowledgeArtifact[]> {
  const path = storePath();
  await ensureDir(path);
  if (!existsSync(path)) return [];
  const raw = await readFile(path, "utf-8");
  return raw
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as KnowledgeArtifact);
}

async function saveAll(artifacts: KnowledgeArtifact[]): Promise<void> {
  const path = storePath();
  await ensureDir(path);
  const content = artifacts.map((a) => JSON.stringify(a)).join("\n");
  await writeFile(path, content ? content + "\n" : "", "utf-8");
}

// ---------------------------------------------------------------------------
// Tag overlap scoring (Tier 1 retrieval)
// ---------------------------------------------------------------------------

function tagOverlap(a: string[], b: string[]): number {
  const sa = new Set(a);
  const sb = new Set(b);
  const intersection = [...sa].filter((t) => sb.has(t)).length;
  const union = new Set([...sa, ...sb]).size;
  return union === 0 ? 0 : intersection / union;
}

/**
 * Minimum Jaccard overlap on tags for two artifacts to be considered
 * the same underlying pattern.
 */
const TAG_MATCH_THRESHOLD = 0.25;

// ---------------------------------------------------------------------------
// Jaccard similarity on word sets (content-based fallback)
// ---------------------------------------------------------------------------

function wordSet(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .split(/\W+/)
      .filter((w) => w.length > 2),
  );
}

function jaccard(a: string, b: string): number {
  const sa = wordSet(a);
  const sb = wordSet(b);
  const intersection = [...sa].filter((w) => sb.has(w)).length;
  const union = new Set([...sa, ...sb]).size;
  return union === 0 ? 0 : intersection / union;
}

// ---------------------------------------------------------------------------
// Inject label helpers
// ---------------------------------------------------------------------------

export type InjectLabel = "[provisional]" | "[established]" | "[suggestion]";

/**
 * Determine whether an artifact is injectable and what label it gets.
 *
 * Injection rules:
 *   consolidated + confidence ≥ threshold  → [established]  (auto-apply)
 *   consolidated + confidence < threshold  → [suggestion]
 *   candidate    + certainty "definitive"  → [provisional]  (single strong obs)
 *   accumulating / candidate non-definitive→ null            (not injectable)
 */
export function getInjectLabel(artifact: KnowledgeArtifact): InjectLabel | null {
  if (artifact.retired) return null;

  if (artifact.stage === "consolidated") {
    const threshold =
      artifact.salience !== undefined
        ? AUTO_APPLY_THRESHOLDS[artifact.salience]
        : DEFAULT_AUTO_APPLY_THRESHOLD;
    return artifact.confidence >= threshold ? "[established]" : "[suggestion]";
  }

  if (artifact.stage === "candidate" && artifact.certainty === "definitive") {
    return "[provisional]";
  }

  return null; // accumulating, or tentative/uncertain candidate
}

export function isInjectable(artifact: KnowledgeArtifact): boolean {
  return getInjectLabel(artifact) !== null;
}

// ---------------------------------------------------------------------------
// Tag vocabulary
// ---------------------------------------------------------------------------

/**
 * Get the set of all tags present in active (non-retired) artifacts.
 * Used by extractFromMessage() to normalize new tags against existing vocabulary.
 */
export async function getActiveTags(): Promise<string[]> {
  const all = await loadAll();
  const tags = new Set<string>();
  for (const a of all) {
    if (!a.retired && a.tags) {
      for (const t of a.tags) tags.add(t);
    }
  }
  return [...tags];
}

// ---------------------------------------------------------------------------
// Match candidate against existing store (Stage 2)
// ---------------------------------------------------------------------------

/**
 * Find an existing active artifact that matches a candidate's kind and tags.
 *
 * Match criteria: same `kind` + Jaccard(tags) ≥ TAG_MATCH_THRESHOLD.
 * Returns the best-matching artifact, or null if no match.
 */
export async function matchCandidate(candidate: {
  kind: KnowledgeKind;
  tags: string[];
}): Promise<KnowledgeArtifact | null> {
  const all = await loadAll();
  const active = all.filter((a) => !a.retired && a.kind === candidate.kind);

  if (!candidate.tags || candidate.tags.length === 0) return null;

  let best: { artifact: KnowledgeArtifact; score: number } | null = null;

  for (const artifact of active) {
    if (!artifact.tags || artifact.tags.length === 0) continue;
    const score = tagOverlap(artifact.tags, candidate.tags);
    if (score >= TAG_MATCH_THRESHOLD) {
      if (!best || score > best.score) {
        best = { artifact, score };
      }
    }
  }

  return best?.artifact ?? null;
}

// ---------------------------------------------------------------------------
// Evidence accumulation (Stage 3 — Resolve)
// ---------------------------------------------------------------------------

/**
 * Add a new observation to an existing artifact's evidence log.
 *
 * If `evidenceCount` reaches `CONSOLIDATION_THRESHOLD` and the artifact is
 * not yet consolidated, triggers a consolidation LLM call and promotes the
 * artifact to `stage: "consolidated"`.
 *
 * @param artifact       - The existing artifact to update
 * @param newObservation - Raw verbatim observation to append
 * @param llm            - LLM adapter (needed for consolidation)
 */
export async function accumulateEvidence(
  artifact: KnowledgeArtifact,
  newObservation: string,
  llm: LLMFn,
): Promise<KnowledgeArtifact> {
  const all = await loadAll();
  const idx = all.findIndex((a) => a.id === artifact.id);
  if (idx < 0) return artifact;

  const existing = all[idx]!;
  const evidenceCount = (existing.evidenceCount ?? 1) + 1;
  const evidence = [...(existing.evidence ?? [existing.content]), newObservation];

  let updated: KnowledgeArtifact = {
    ...existing,
    evidenceCount,
    evidence,
    // Preserve the consolidated stage — never demote back to accumulating
    stage: existing.stage === "consolidated" ? "consolidated" : "accumulating",
    revisedAt: new Date().toISOString(),
  };

  // Trigger consolidation when threshold is reached (skip if already consolidated)
  if (evidenceCount >= CONSOLIDATION_THRESHOLD && existing.stage !== "consolidated") {
    const consolidatedContent = await consolidateEvidence(existing.kind, evidence, llm);
    // Confidence grows on consolidation: definitive seeded at 0.65 → grows toward ~0.80
    const newConfidence = Math.min(0.92, existing.confidence + 0.20);
    updated = {
      ...updated,
      content: consolidatedContent,
      stage: "consolidated",
      confidence: Math.round(newConfidence * 100) / 100,
    };
  }

  all[idx] = updated;
  await saveAll(all);
  return updated;
}

// ---------------------------------------------------------------------------
// Stage 5 — Persist
// ---------------------------------------------------------------------------

/**
 * Stage 5 — Persist: write an artifact to the store.
 *
 * - If an artifact with the same id exists, it is replaced (upsert).
 * - Otherwise the artifact is appended.
 *
 * Note: near-duplicate detection is now handled by matchCandidate() +
 * accumulateEvidence() in the pipeline before persist() is called.
 */
export async function persist(artifact: KnowledgeArtifact): Promise<void> {
  const all = await loadAll();

  const existingIdx = all.findIndex((a) => a.id === artifact.id);
  if (existingIdx >= 0) {
    all[existingIdx] = artifact;
  } else {
    all.push(artifact);
  }

  await saveAll(all);
}

// ---------------------------------------------------------------------------
// Stage 6 — Retrieve
// ---------------------------------------------------------------------------

/**
 * Stage 6 — Retrieve: recall relevant active artifacts for a query.
 *
 * Scoring (Tier 1 + content fallback):
 *   - Tag overlap with query tokens (60% weight)
 *   - Jaccard similarity against content (30% weight)
 *   - Confidence boost (10% weight)
 *
 * Consolidated artifacts are weighted higher than candidates.
 * Retired artifacts are excluded.
 *
 * @param query - Natural language query in any language
 * @param limit - Maximum number of results to return (default: 10)
 */
export async function retrieve(
  query: string,
  limit = 10,
): Promise<KnowledgeArtifact[]> {
  const all = await loadAll();
  const active = all.filter((a) => !a.retired);

  if (!query || !query.trim()) return active.slice(0, limit);

  // Tokenize query into potential tag fragments
  const queryTokens = query
    .toLowerCase()
    .split(/\W+/)
    .filter((w) => w.length > 2);

  const scored = active
    .map((artifact) => {
      let score = 0;

      // Tag overlap (Tier 1 — primary signal)
      if (artifact.tags && artifact.tags.length > 0 && queryTokens.length > 0) {
        const matchingTags = artifact.tags.filter((t) =>
          queryTokens.some((qt) => t.includes(qt) || qt.includes(t)),
        );
        const tagScore = matchingTags.length / Math.max(artifact.tags.length, queryTokens.length);
        score += tagScore * 0.60;
      }

      // Content similarity (fallback for tag-free artifacts)
      const contentScore = jaccard(query, artifact.content);
      score += contentScore * 0.30;

      // Confidence boost
      score += artifact.confidence * 0.10;

      // Consolidated artifacts get a slight priority boost
      if (artifact.stage === "consolidated") score += 0.05;

      return { artifact, score };
    })
    .filter(({ score }) => score > 0.05)
    .sort((a, b) => b.score - a.score);

  return scored.slice(0, limit).map(({ artifact }) => artifact);
}

// ---------------------------------------------------------------------------
// Stage 7 — Apply
// ---------------------------------------------------------------------------

/**
 * Stage 7 — Apply: confidence-gated decision.
 *
 * Returns the inject label and whether the artifact should be auto-applied
 * (silently injected) or surfaced as a suggestion.
 */
export async function apply(
  artifact: KnowledgeArtifact,
  _context: string,
): Promise<{ suggestion: string; autoApply: boolean }> {
  const label = getInjectLabel(artifact);

  if (!label) {
    return { suggestion: "", autoApply: false };
  }

  const autoApply = label === "[established]";
  // Update application count
  const all = await loadAll();
  const idx = all.findIndex((a) => a.id === artifact.id);
  if (idx >= 0) {
    const existing = all[idx]!;
    all[idx] = {
      ...existing,
      appliedCount: (existing.appliedCount ?? 0) + 1,
      lastRetrievedAt: new Date().toISOString(),
    };
    await saveAll(all);
  }

  return {
    suggestion: `${label} ${artifact.content}`,
    autoApply,
  };
}

// ---------------------------------------------------------------------------
// Stage 8 — Revise
// ---------------------------------------------------------------------------

/**
 * Stage 8 — Revise: update an artifact in place.
 *
 * If content changes significantly (Jaccard < 0.5 vs. original), the old
 * artifact is retired and the updated one is stored as a fresh entry with a
 * new id, preserving the audit trail.
 */
export async function revise(
  artifact: KnowledgeArtifact,
  update: Partial<KnowledgeArtifact>,
): Promise<KnowledgeArtifact> {
  const { randomUUID } = await import("node:crypto");
  const now = new Date().toISOString();
  const revised = { ...artifact, ...update, revisedAt: now };

  const contentChanged =
    update.content !== undefined &&
    jaccard(artifact.content, update.content) < 0.5;

  if (contentChanged) {
    // Retire original, persist new entry with fresh id
    await persist({ ...artifact, retired: true, revisedAt: now });
    revised.id = randomUUID();
    revised.createdAt = now;
    delete revised.revisedAt;
  }

  await persist(revised);
  return revised;
}

// ---------------------------------------------------------------------------
// Feedback helpers
// ---------------------------------------------------------------------------

/**
 * Record that a suggestion was accepted by the user.
 * Increments `acceptedCount` and nudges confidence upward.
 */
export async function recordAccepted(artifactId: string): Promise<void> {
  const all = await loadAll();
  const idx = all.findIndex((a) => a.id === artifactId);
  if (idx < 0) return;
  const a = all[idx]!;
  all[idx] = {
    ...a,
    acceptedCount: (a.acceptedCount ?? 0) + 1,
    reinforcementCount: (a.reinforcementCount ?? 0) + 1,
    confidence: Math.min(1, Math.round((a.confidence + 0.02) * 100) / 100),
  };
  await saveAll(all);
}

/**
 * Record that a suggestion was rejected by the user.
 * Increments `rejectedCount` and nudges confidence downward.
 */
export async function recordRejected(artifactId: string): Promise<void> {
  const all = await loadAll();
  const idx = all.findIndex((a) => a.id === artifactId);
  if (idx < 0) return;
  const a = all[idx]!;
  all[idx] = {
    ...a,
    rejectedCount: (a.rejectedCount ?? 0) + 1,
    confidence: Math.max(0, Math.round((a.confidence - 0.05) * 100) / 100),
  };
  await saveAll(all);
}
