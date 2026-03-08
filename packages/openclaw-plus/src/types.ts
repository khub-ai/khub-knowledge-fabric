/**
 * Core types for the PIL (Persistable Interactive Learning) pipeline.
 *
 * All other modules import from here; this file has no internal imports.
 */

// ---------------------------------------------------------------------------
// Artifact classification
// ---------------------------------------------------------------------------

/**
 * The kind taxonomy: six values mapping to the four cognitive memory types.
 *
 * Cognitive mapping:
 *   preference + convention + fact → semantic memory
 *   procedure                      → procedural memory
 *   judgment + strategy            → evaluative memory
 */
export type KnowledgeKind =
  | "preference"   // subjective style, taste, or personal choice
  | "convention"   // agreed naming, terminology, or standards
  | "fact"         // objective, verifiable information
  | "procedure"    // step-by-step process or recipe
  | "judgment"     // evaluative heuristic or quality criterion
  | "strategy";    // general approach to a class of problems

/**
 * How strongly the knowledge was expressed (LLM-assigned at extraction).
 * Seeds the initial confidence score independent of language.
 */
export type KnowledgeCertainty =
  | "definitive"  // strong, unhedged statement: "always use strict mode"
  | "tentative"   // qualified: "I usually prefer bullet points"
  | "uncertain";  // speculative: "I think I prefer shorter summaries?"

/**
 * Whether the artifact applies broadly or only to a specific situation.
 */
export type KnowledgeScope =
  | "specific"    // applies only to this situation or task
  | "general";    // applies broadly across many contexts

/**
 * Lifecycle stage of an artifact through the evidence-accumulation pipeline.
 *
 *   candidate    — single observation; not yet confirmed; may be [provisional]-injectable
 *   accumulating — evidence building toward consolidation threshold; NOT injectable
 *   consolidated — LLM-distilled generalization; fully injectable
 */
export type ArtifactStage = "candidate" | "accumulating" | "consolidated";

// ---------------------------------------------------------------------------
// Knowledge graph
// ---------------------------------------------------------------------------

export type RelationType =
  | "references"   // this artifact uses or depends on another
  | "constrains"   // this artifact limits how another is applied
  | "supersedes"   // this artifact replaced another (revision trail)
  | "supports";    // this artifact provides evaluative guidance for another

export type ArtifactRelation = {
  type: RelationType;
  targetId: string;
};

// ---------------------------------------------------------------------------
// Artifact schema
// ---------------------------------------------------------------------------

/**
 * A knowledge artifact: the primary unit of persisted, reusable knowledge.
 *
 * Core fields (id, kind, content, confidence, provenance, createdAt) are
 * required. All other fields are optional enrichments populated progressively
 * by the pipeline, the triggering system, and user feedback.
 *
 * An artifact with only the core fields is fully functional: it can be stored,
 * retrieved (by content search), applied, and revised.
 */
export type KnowledgeArtifact = {
  // ── Core fields (required) ──────────────────────────────────────────────
  id: string;
  kind: KnowledgeKind;
  /**
   * The knowledge itself.
   *
   * - candidate / accumulating: verbatim from the user's input, any language.
   * - consolidated: LLM-distilled generalization (English).
   */
  content: string;
  /**
   * Certainty score 0–1. Seeded from `certainty` at extraction time;
   * grows with evidence accumulation and positive user feedback.
   */
  confidence: number;
  /** Where this knowledge came from (session ID, user statement, etc.). */
  provenance: string;
  createdAt: string;            // ISO 8601

  // ── Enrichment fields (optional — for triggering and retrieval) ─────────
  scope?: KnowledgeScope;
  certainty?: KnowledgeCertainty;
  /** Natural language condition for when this artifact applies. */
  trigger?: string;
  /**
   * Normalized topic tags. Lowercase, hyphenated, English noun phrases.
   * LLM-assigned; anchored to the store's existing vocabulary.
   * Examples: "code-style", "file-naming", "summary-format"
   */
  tags?: string[];
  /** High-level domain cluster, e.g. "code-style", "financial-ops". */
  topic?: string;
  /** One-line distillation for cheap Tier-2 LLM matching. */
  summary?: string;
  /** Graph edges: relationships to other artifacts. */
  relations?: ArtifactRelation[];

  // ── Lifecycle fields (optional — for decay, habituation, and revision) ──
  salience?: "low" | "medium" | "high";
  /**
   * Evidence accumulation stage.
   *
   * candidate → accumulating → consolidated
   */
  stage?: ArtifactStage;
  /** Number of observations that support this pattern. */
  evidenceCount?: number;
  /**
   * Raw observation snippets in the user's original language.
   * Passed to the consolidation LLM call when evidenceCount reaches threshold.
   */
  evidence?: string[];
  lastRetrievedAt?: string;
  reinforcementCount?: number;
  appliedCount?: number;
  acceptedCount?: number;
  rejectedCount?: number;
  revisedAt?: string;           // ISO 8601, set on each revision
  /**
   * Optional compiled form of a procedure artifact.
   *
   * Generated by compileToProgram(); the procedure recipe (content) is always
   * retained as primary documentation and manual fallback. The program is an
   * optimised executable form that the agent can try first on subsequent runs.
   *
   * Only meaningful on kind="procedure" artifacts.
   */
  program?: {
    language: string;    // "python" | "bash" | etc.
    code: string;        // full source — retained even when path is set
    path?: string;       // absolute path to the saved file on disk
    generatedAt: string; // ISO 8601
  };
  retired?: boolean;            // true when superseded or invalidated
};

// ---------------------------------------------------------------------------
// LLM adapter
// ---------------------------------------------------------------------------

/**
 * Dependency-injected LLM function.
 *
 * The core `openclaw-plus` package has no hard dependency on any specific LLM
 * SDK. Callers provide a concrete implementation (e.g. wrapping @anthropic-ai/sdk).
 *
 * @param prompt - The full prompt to send to the LLM.
 * @returns The LLM's text response.
 */
export type LLMFn = (prompt: string) => Promise<string>;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/**
 * Starting confidence values, seeded from the LLM-assigned `certainty` field.
 * Replaces English-specific hedge / assertion word heuristics.
 */
export const CONFIDENCE_SEED: Record<KnowledgeCertainty, number> = {
  definitive: 0.65,  // strong, unhedged statement — grows with further evidence
  tentative:  0.35,  // weak signal — needs reinforcement before use
  uncertain:  0.15,  // barely a signal — consolidation required before injection
} as const;

/**
 * Number of observations required to trigger the consolidation LLM call.
 * When evidenceCount reaches this value, the accumulating artifact is
 * distilled into a consolidated generalization.
 *
 * Default: 3. Configurable via CONSOLIDATION_THRESHOLD env variable.
 */
export const CONSOLIDATION_THRESHOLD =
  parseInt(process.env["CONSOLIDATION_THRESHOLD"] ?? "3", 10);

// ---------------------------------------------------------------------------
// Auto-apply thresholds (adjusted by salience)
// ---------------------------------------------------------------------------

export const AUTO_APPLY_THRESHOLDS: Record<NonNullable<KnowledgeArtifact["salience"]>, number> = {
  low:    0.75,
  medium: 0.85,
  high:   0.95,
} as const;

/** Default auto-apply threshold when salience is not set. */
export const DEFAULT_AUTO_APPLY_THRESHOLD = 0.80;
