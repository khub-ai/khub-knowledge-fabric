/**
 * Core types for the PIL (Persistable Interactive Learning) pipeline.
 *
 * All other modules import from here; this file has no internal imports.
 */

// ---------------------------------------------------------------------------
// Artifact classification
// ---------------------------------------------------------------------------

/**
 * The kind taxonomy: nine values mapping to the four cognitive memory types.
 *
 * Cognitive mapping:
 *   preference + convention + fact               → semantic memory
 *   procedure                                    → procedural memory
 *   judgment + strategy                          → evaluative memory
 *   boundary + revision-trigger + failure-case   → dialogic knowledge (Phase 4)
 *
 * The three Phase 4 kinds are produced only by dialogic learning sessions.
 * They are not created by the Phase 1 passive extraction pipeline.
 */
export type KnowledgeKind =
  | "preference"        // subjective style, taste, or personal choice
  | "convention"        // agreed naming, terminology, or standards
  | "fact"              // objective, verifiable information
  | "procedure"         // step-by-step process or recipe
  | "judgment"          // evaluative heuristic or quality criterion
  | "strategy"          // general approach to a class of problems
  // Phase 4 — produced only by dialogic learning sessions
  | "boundary"          // when a rule does not apply
  | "revision-trigger"  // evidence that should cause revision of a conclusion
  | "failure-case";     // past mistake that refined later judgment

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
 * The core `knowledge-fabric` package has no hard dependency on any specific LLM
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

// ---------------------------------------------------------------------------
// Phase 4 — Expert-to-Agent Dialogic Learning
//
// Types for the structured expert elicitation system. These extend the Phase 1
// artifact store; DialogueSession is persisted separately at:
//   ~/.openclaw/knowledge/sessions/<session-id>.json
// ---------------------------------------------------------------------------

/**
 * The nine question types from the dialogic learning taxonomy.
 *
 * The taxonomy is open-ended: CustomQuestionType records additional types
 * introduced by the expert during a session.
 */
export type QuestionType =
  | "case-elicitation"   // obtain a real example
  | "process-extraction" // recover sequence and decision logic
  | "priority"           // determine which signals matter most
  | "abstraction"        // separate general method from specific case details
  | "boundary"           // determine where a rule stops applying
  | "counterexample"     // test whether the rule survives difficult cases
  | "revision"           // learn how the expert changes their mind
  | "transfer"           // test whether knowledge applies beyond the original case
  | "confidence";        // calibrate certainty and scope

/**
 * The six lifecycle stages of a dialogic session.
 */
export type SessionStage =
  | "eliciting-case"     // asking for a concrete example; no rule yet proposed
  | "extracting-process" // unpacking the expert's reasoning sequence
  | "abstracting"        // generalizing from the case to a tentative rule
  | "testing-boundaries" // asking for exceptions, failures, and limits
  | "synthesizing"       // agent proposes a rule for the expert to correct
  | "complete";          // all candidate rules consolidated or archived

/**
 * How an expert turn qualifies as a correction of a prior agent synthesis.
 */
export type CorrectionType =
  | "rule-revision"        // expert explicitly revises the rule statement
  | "scope-adjustment"     // expert narrows or widens the rule's scope
  | "counterexample-added"; // expert introduces a case that invalidates the rule

/**
 * Which of the five minimum consolidation criteria have been met
 * for a candidate rule.
 *
 * All five must be true before the candidate rule is promoted to the
 * main artifact store.
 */
export type ConsolidationGapStatus = {
  /** At least one concrete real-world example has been provided. */
  hasConcreteCase: boolean;
  /**
   * The agent has proposed a generalized version and the expert has
   * accepted or corrected it.
   */
  hasGeneralizedRestatement: boolean;
  /** A scope statement or boundary condition has been captured. */
  hasScopeOrBoundary: boolean;
  /** At least one exception, counterexample, or failure mode has been recorded. */
  hasExceptionOrFailureMode: boolean;
  /** At least one revision trigger has been stated by the expert. */
  hasRevisionTrigger: boolean;
};

/**
 * A single turn in the session transcript.
 *
 * Agent turns carry questionType; expert turns may carry correctionType.
 */
export type DialogueTurn = {
  turnId: string;
  role: "agent" | "expert";
  content: string;
  timestamp: string;              // ISO 8601
  /** Which question type was used. Present on agent turns only. */
  questionType?: QuestionType;
  /** Which candidate rule this turn was primarily advancing. */
  candidateRuleId?: string;
  /**
   * Candidate knowledge fragments parsed from this expert turn.
   * Populated after extraction; absent on agent turns.
   */
  extractedUnits?: string[];
  /**
   * Set when an expert turn contains a correction of a prior synthesis.
   * Triggers correction processing in the pipeline.
   */
  correctionType?: CorrectionType;
};

/**
 * Records which question types were asked for which candidate rule,
 * used to prevent repetition within a session.
 */
export type QuestionHistoryEntry = {
  questionType: QuestionType;
  candidateRuleId: string;
  turnId: string;
};

/**
 * A question type introduced by the expert during a session.
 *
 * Extends the base nine-type taxonomy for domain-specific needs.
 * Propagated to future sessions in the same domain.
 */
export type CustomQuestionType = {
  id: string;
  /** Short label, e.g. "competitive-moat". */
  name: string;
  /** What learning gap this question type addresses. */
  purpose: string;
  /** Representative example of the question. */
  exampleQuestion: string;
  addedAt: string;               // ISO 8601
  addedBySessionId: string;
};

/**
 * A candidate rule being developed within a session.
 *
 * Promoted to the artifact store when all five consolidation criteria
 * in `gaps` are satisfied.
 */
export type CandidateRule = {
  id: string;
  /**
   * Current best statement of the rule.
   * Updated in place when a correction is applied; prior versions
   * are archived in the related DialogueTurn's extractedUnits.
   */
  content: string;
  /** Expected KnowledgeKind of the promoted artifact. */
  kind: KnowledgeKind;
  /** Progress toward the five minimum consolidation criteria. */
  gaps: ConsolidationGapStatus;
  /** IDs of the turns that contributed evidence to this rule. */
  relatedTurnIds: string[];
};

/**
 * The full state of one expert-to-agent dialogic learning session.
 *
 * Persisted at: ~/.openclaw/knowledge/sessions/<session-id>.json
 * Retained permanently after completion as the audit record.
 *
 * Artifacts promoted from the session are written to artifacts.jsonl
 * with provenance `session:<id>`.
 */
export type DialogueSession = {
  id: string;
  /**
   * The declared learning goal in natural language.
   * Example: "learn how this expert screens investment opportunities."
   */
  objective: string;
  /**
   * The topic area. Used to match this session to prior sessions in
   * the same domain and to gate gap inheritance.
   * Example: "long-term-fundamental-investing"
   */
  domain: string;
  stage: SessionStage;
  createdAt: string;             // ISO 8601
  lastActiveAt: string;          // ISO 8601; updated on every turn
  /** Full session transcript in chronological order. */
  turns: DialogueTurn[];
  /** Rules being developed in this session, each with a gap tracker. */
  candidateRules: CandidateRule[];
  /**
   * History of which question types were asked for which candidate
   * rule. Used to prevent repetition and to select rephrased variants.
   */
  questionHistory: QuestionHistoryEntry[];
  /**
   * IDs of artifacts promoted to the main store when this session ended.
   * Empty while the session is in progress.
   */
  artifactIds: string[];
  /** IDs of prior sessions in the same domain loaded at session start. */
  priorSessionIds: string[];
  /**
   * IDs of artifacts inherited from prior sessions.
   * These informed the starting state but were not created in this session.
   */
  inheritedArtifactIds: string[];
  /**
   * Question types introduced by the expert during this session.
   * Propagated to future sessions in the same domain.
   */
  customQuestionTypes: CustomQuestionType[];
};

/**
 * User-level communication preferences for dialogic sessions.
 *
 * Meta-knowledge about HOW to conduct dialogue with this expert —
 * not about any specific domain. Applies to all sessions regardless
 * of topic.
 *
 * Stored at: ~/.openclaw/knowledge/communication-profile.json
 * Populated via pre-session calibration, adaptive signals, and
 * Phase 1 preference capture.
 */
export type CommunicationProfile = {
  /**
   * Whether the expert prefers one question per turn or can handle
   * grouped related questions.
   * Default: "single"
   */
  questionGranularity: "single" | "grouped";
  /**
   * Whether the expert thinks better starting from a concrete example
   * or from the principle.
   * Default: "example-first"
   */
  framingPreference: "example-first" | "principle-first";
  /**
   * Whether the agent's questions should be brief and direct, or
   * include contextual setup before the question itself.
   */
  verbosity: "brief" | "contextual";
  /**
   * Whether the expert wants the agent to reflect back what it heard
   * before asking the next question, or move forward immediately.
   */
  acknowledgmentStyle: "reflect-back" | "forward";
  /**
   * How often the agent should synthesize and propose tentative rules.
   *
   * - "often": check understanding every few turns.
   * - "at-milestones": only when all five consolidation criteria are met.
   * - "session-end": only at the close of the session.
   */
  synthesisFrequency: "often" | "at-milestones" | "session-end";
  /**
   * Whether the agent can use domain-specific terms freely or should
   * let the expert introduce terminology organically.
   */
  terminologyTolerance: "free" | "expert-led";

  // ── Metadata ─────────────────────────────────────────────────────────
  createdAt: string;             // ISO 8601; set after first calibration
  updatedAt: string;             // ISO 8601; set after each adaptive update
  /**
   * Whether the initial pre-session calibration has been completed.
   * When false, the session starts with the 3–4 calibration questions.
   * When true, calibration is skipped and the stored profile is used.
   */
  calibrationComplete: boolean;
};
