/**
 * session.ts — Phase 4 dialogic learning session management.
 *
 * Owns CRUD for DialogueSession objects and the idempotent promotion
 * pathway from candidate rules to the main artifact store.
 *
 * Sessions are persisted as plain JSON files:
 *   ~/.openclaw/knowledge/sessions/<session-id>.json
 *
 * Mirrors store.ts patterns: fs/promises, existsSync guard, console.warn
 * on failure, no throws from async operations.
 */

import { readFile, writeFile, mkdir, readdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import type {
  DialogueSession,
  CandidateRule,
  ConsolidationGapStatus,
  DialogueTurn,
  QuestionType,
  CorrectionType,
  LLMFn,
} from "./types.js";
import { candidateToArtifact } from "./extract.js";
import { loadAll, persist } from "./store.js";

// ── Storage paths ────────────────────────────────────────────────────────────

export function sessionsDir(): string {
  return (
    process.env["SESSIONS_DIR"] ??
    join(homedir(), ".openclaw", "knowledge", "sessions")
  );
}

function sessionPath(id: string): string {
  return join(sessionsDir(), `${id}.json`);
}

async function ensureSessionsDir(): Promise<void> {
  const dir = sessionsDir();
  if (!existsSync(dir)) {
    await mkdir(dir, { recursive: true });
  }
}

// ── CRUD ─────────────────────────────────────────────────────────────────────

/**
 * Create a new, unsaved DialogueSession.
 * Caller must call saveSession() to persist it.
 */
export function createSession(
  domain: string,
  objective: string,
): DialogueSession {
  const now = new Date().toISOString();
  return {
    id: randomUUID(),
    objective,
    domain,
    stage: "eliciting-case",
    createdAt: now,
    lastActiveAt: now,
    turns: [],
    candidateRules: [],
    questionHistory: [],
    artifactIds: [],
    priorSessionIds: [],
    inheritedArtifactIds: [],
    customQuestionTypes: [],
    committed: false,
  };
}

/**
 * Load a session by ID. Returns null if the file does not exist or cannot
 * be parsed.
 */
export async function loadSession(
  id: string,
): Promise<DialogueSession | null> {
  const path = sessionPath(id);
  if (!existsSync(path)) return null;
  try {
    const raw = await readFile(path, "utf-8");
    return JSON.parse(raw) as DialogueSession;
  } catch {
    console.warn(`[session] Failed to load session ${id}`);
    return null;
  }
}

/**
 * Persist a session to disk. Creates the sessions directory if needed.
 */
export async function saveSession(session: DialogueSession): Promise<void> {
  await ensureSessionsDir();
  const path = sessionPath(session.id);
  await writeFile(path, JSON.stringify(session, null, 2), "utf-8");
}

/**
 * Return all sessions whose domain exactly matches the given string,
 * sorted by createdAt ascending (oldest first).
 *
 * Domain matching is exact and case-sensitive per the Phase 4 spec.
 */
export async function listSessionsByDomain(
  domain: string,
): Promise<DialogueSession[]> {
  await ensureSessionsDir();
  let files: string[];
  try {
    files = await readdir(sessionsDir());
  } catch {
    return [];
  }

  const sessions: DialogueSession[] = [];
  for (const file of files) {
    if (!file.endsWith(".json")) continue;
    const id = file.slice(0, -5);
    const session = await loadSession(id);
    if (session && session.domain === domain) {
      sessions.push(session);
    }
  }

  return sessions.sort((a, b) => a.createdAt.localeCompare(b.createdAt));
}

// ── Turn helpers ─────────────────────────────────────────────────────────────

/**
 * Append an agent turn to the session transcript. Returns a new session
 * object (immutable update pattern).
 */
export function addAgentTurn(
  session: DialogueSession,
  content: string,
  questionType?: QuestionType,
  candidateRuleId?: string,
): DialogueSession {
  const now = new Date().toISOString();
  const turn: DialogueTurn = {
    turnId: randomUUID(),
    role: "agent",
    content,
    timestamp: now,
    questionType,
    candidateRuleId,
  };
  return {
    ...session,
    lastActiveAt: now,
    turns: [...session.turns, turn],
  };
}

/**
 * Append an expert turn to the session transcript. Returns a new session
 * object (immutable update pattern).
 */
export function addExpertTurn(
  session: DialogueSession,
  content: string,
  extractedUnits?: string[],
  correctionType?: CorrectionType,
): DialogueSession {
  const now = new Date().toISOString();
  const turn: DialogueTurn = {
    turnId: randomUUID(),
    role: "expert",
    content,
    timestamp: now,
    extractedUnits,
    correctionType,
  };
  return {
    ...session,
    lastActiveAt: now,
    turns: [...session.turns, turn],
  };
}

// ── Candidate rule helpers ────────────────────────────────────────────────────

/**
 * Return the existing active candidate rule, or create one if none exists.
 * Returns [updatedSession, rule].
 */
export function getOrCreateRule(
  session: DialogueSession,
): [DialogueSession, CandidateRule] {
  const existing = session.candidateRules.find((r) => r.status === "active");
  if (existing) return [session, existing];

  const rule: CandidateRule = {
    id: randomUUID(),
    content: session.objective,
    kind: "procedure",
    status: "active",
    gaps: {
      hasConcreteCase: false,
      hasGeneralizedRestatement: false,
      hasScopeOrBoundary: false,
      hasExceptionOrFailureMode: false,
      hasRevisionTrigger: false,
    },
    relatedTurnIds: [],
  };

  const updated: DialogueSession = {
    ...session,
    candidateRules: [...session.candidateRules, rule],
  };

  return [updated, rule];
}

/**
 * Apply partial updates to a candidate rule identified by ruleId.
 * Returns a new session object.
 */
export function updateRule(
  session: DialogueSession,
  ruleId: string,
  updates: Partial<CandidateRule>,
): DialogueSession {
  return {
    ...session,
    candidateRules: session.candidateRules.map((r) =>
      r.id === ruleId ? { ...r, ...updates } : r,
    ),
  };
}

/**
 * Merge gap truths into a candidate rule's gap status.
 *
 * Uses || to ensure a gap that was already true is never reverted to false.
 * Only partial gap updates are required — omit fields that haven't changed.
 */
export function updateGaps(
  session: DialogueSession,
  ruleId: string,
  gaps: Partial<ConsolidationGapStatus>,
): DialogueSession {
  return {
    ...session,
    candidateRules: session.candidateRules.map((r) => {
      if (r.id !== ruleId) return r;
      return {
        ...r,
        gaps: {
          hasConcreteCase:
            r.gaps.hasConcreteCase || (gaps.hasConcreteCase ?? false),
          hasGeneralizedRestatement:
            r.gaps.hasGeneralizedRestatement ||
            (gaps.hasGeneralizedRestatement ?? false),
          hasScopeOrBoundary:
            r.gaps.hasScopeOrBoundary || (gaps.hasScopeOrBoundary ?? false),
          hasExceptionOrFailureMode:
            r.gaps.hasExceptionOrFailureMode ||
            (gaps.hasExceptionOrFailureMode ?? false),
          hasRevisionTrigger:
            r.gaps.hasRevisionTrigger || (gaps.hasRevisionTrigger ?? false),
        },
      };
    }),
  };
}

// ── Promotion ─────────────────────────────────────────────────────────────────

/**
 * Promote all synthesized candidate rules to the main artifact store.
 *
 * Idempotent: if session.committed is already true, returns immediately.
 * If a partial promotion was interrupted, re-running is safe — existing
 * artifacts with provenance "session:<id>" are skipped.
 *
 * After all promotions succeed, saves the session with committed: true.
 */
export async function promoteSession(
  session: DialogueSession,
  _llm: LLMFn,
): Promise<DialogueSession> {
  if (session.committed) return session;

  // Build a set of artifact IDs already promoted from this session
  const existing = await loadAll();
  const alreadyPromoted = new Set(
    existing
      .filter((a) => a.provenance.startsWith(`session:${session.id}`))
      .map((a) => a.id),
  );

  const newArtifactIds: string[] = [];

  for (const rule of session.candidateRules) {
    if (rule.status !== "synthesized") continue;

    // Build an ExtractionCandidate-compatible shape
    const candidate = {
      content: rule.content,
      kind: rule.kind,
      scope: "general" as const,
      certainty: "definitive" as const,
      tags: [session.domain],
      rationale: `Elicited via dialogic learning. Objective: ${session.objective}`,
    };

    const artifact = candidateToArtifact(candidate, `session:${session.id}`);

    if (alreadyPromoted.has(artifact.id)) continue;

    // Dialogic artifacts are fully expert-confirmed — upgrade lifecycle
    const promoted = {
      ...artifact,
      stage: "consolidated" as const,
      confidence: 0.80,
    };

    await persist(promoted);
    newArtifactIds.push(promoted.id);
  }

  const committed: DialogueSession = {
    ...session,
    artifactIds: [...session.artifactIds, ...newArtifactIds],
    committed: true,
    stage: "complete",
  };

  await saveSession(committed);
  return committed;
}
