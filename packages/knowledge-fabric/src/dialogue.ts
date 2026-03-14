/**
 * dialogue.ts — Phase 4 dialogic learning engine.
 *
 * Owns:
 *   - Gap-driven question type selection (rule-based, no LLM)
 *   - LLM prompt builders for gap assessment, question generation, synthesis,
 *     and correction parsing
 *   - processTurn() — the main entrypoint called by pil-chat on each expert
 *     message while a teach session is active
 *
 * Default communication profile is hardcoded for the demo; a full
 * CommunicationProfile calibration phase is deferred to a future iteration.
 */

import { randomUUID } from "node:crypto";
import type {
  DialogueSession,
  CandidateRule,
  ConsolidationGapStatus,
  QuestionType,
  SessionStage,
  CorrectionType,
  LLMFn,
  CommunicationProfile,
  QuestionHistoryEntry,
  DialogueTurn,
} from "./types.js";
import {
  addAgentTurn,
  addExpertTurn,
  getOrCreateRule,
  updateRule,
  updateGaps,
  saveSession,
  promoteSession,
} from "./session.js";

// ── Default communication profile ────────────────────────────────────────────

export const DEFAULT_PROFILE: CommunicationProfile = {
  questionGranularity: "single",
  framingPreference: "example-first",
  verbosity: "contextual",
  acknowledgmentStyle: "reflect-back",
  synthesisFrequency: "at-milestones",
  terminologyTolerance: "expert-led",
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
  calibrationComplete: false,
};

// ── JSON parse helper ─────────────────────────────────────────────────────────

function parseJSON<T>(text: string): T | null {
  const trimmed = text.trim();
  // Strip markdown code fences if present
  const stripped = trimmed.startsWith("```")
    ? trimmed.replace(/^```(?:json)?\s*\n?/, "").replace(/\n?```\s*$/, "")
    : trimmed;
  try {
    return JSON.parse(stripped) as T;
  } catch {
    return null;
  }
}

// ── Question type selection ───────────────────────────────────────────────────

/**
 * Rule-based priority ladder: returns the next question type to ask, or
 * "synthesize" when all five gaps are closed.
 *
 * Dedup: if a question type was already used for this ruleId, skip it and
 * advance to the next eligible type. This prevents verbatim repetition
 * while still converging on gap closure.
 */
export function selectNextQuestionType(
  gaps: ConsolidationGapStatus,
  questionHistory: QuestionHistoryEntry[],
  ruleId: string,
): QuestionType | "synthesize" {
  if (
    gaps.hasConcreteCase &&
    gaps.hasGeneralizedRestatement &&
    gaps.hasScopeOrBoundary &&
    gaps.hasExceptionOrFailureMode &&
    gaps.hasRevisionTrigger
  ) {
    return "synthesize";
  }

  // Priority ladder: gap key → question type to ask when gap is false
  const ladder: Array<[keyof ConsolidationGapStatus, QuestionType]> = [
    ["hasConcreteCase", "case-elicitation"],
    ["hasGeneralizedRestatement", "process-extraction"],
    ["hasScopeOrBoundary", "boundary"],
    ["hasExceptionOrFailureMode", "counterexample"],
    ["hasRevisionTrigger", "revision"],
  ];

  const askedForRule = new Set(
    questionHistory
      .filter((h) => h.candidateRuleId === ruleId)
      .map((h) => h.questionType),
  );

  // First pass: prefer a question type that hasn't been asked yet
  for (const [gapKey, qType] of ladder) {
    if (!gaps[gapKey] && !askedForRule.has(qType)) return qType;
  }

  // Second pass: all remaining false-gap types have been asked once —
  // repeat the first false-gap type to avoid deadlock
  for (const [gapKey, qType] of ladder) {
    if (!gaps[gapKey]) return qType;
  }

  // All gaps true (should have been caught above)
  return "synthesize";
}

// ── Prompt builders ───────────────────────────────────────────────────────────

/**
 * Build the gap assessment prompt.
 *
 * LLM returns a JSON object with only the gaps that are *newly* true.
 * It is explicitly told not to return false for already-true criteria.
 */
export function buildGapAssessmentPrompt(
  expertResponse: string,
  rule: CandidateRule,
  turns: DialogueTurn[],
): string {
  const transcript = turns
    .slice(-6)
    .map((t) => `${t.role === "agent" ? "Agent" : "Expert"}: ${t.content}`)
    .join("\n");

  return `You are evaluating whether an expert's response advances the consolidation of a knowledge rule.

CURRENT RULE BEING DEVELOPED:
"${rule.content}"

RECENT DIALOGUE:
${transcript}

LATEST EXPERT RESPONSE:
"${expertResponse}"

CONSOLIDATION CRITERIA — for each criterion, return true ONLY if the expert's latest response provides clear evidence that criterion is now met. Never change a criterion from true to false.

Current gap status (already true = do not change):
- hasConcreteCase: ${rule.gaps.hasConcreteCase}
- hasGeneralizedRestatement: ${rule.gaps.hasGeneralizedRestatement}
- hasScopeOrBoundary: ${rule.gaps.hasScopeOrBoundary}
- hasExceptionOrFailureMode: ${rule.gaps.hasExceptionOrFailureMode}
- hasRevisionTrigger: ${rule.gaps.hasRevisionTrigger}

Criterion definitions:
- hasConcreteCase: Expert described at least one real, specific example
- hasGeneralizedRestatement: Expert confirmed or corrected a general principle
- hasScopeOrBoundary: Expert stated when the rule applies or does not apply
- hasExceptionOrFailureMode: Expert described at least one case where the rule fails or an exception exists
- hasRevisionTrigger: Expert stated what evidence would cause them to change their conclusion

Return a JSON object containing ONLY the criteria that are NOW newly true (do not include already-true criteria, never include false values):
{ "hasConcreteCase": true }

Return {} if the response advances no criteria. Return only valid JSON, no other text.`.trim();
}

/**
 * Build the question generation prompt.
 *
 * LLM returns a single natural-sounding conversational question.
 */
export function buildQuestionPrompt(
  questionType: QuestionType,
  rule: CandidateRule,
  turns: DialogueTurn[],
): string {
  const transcript = turns
    .slice(-4)
    .map((t) => `${t.role === "agent" ? "Agent" : "Expert"}: ${t.content}`)
    .join("\n");

  const instructions: Record<QuestionType, string> = {
    "case-elicitation":
      "Ask the expert to share a specific, real example where they applied this rule or judgment. Be concrete — ask for a specific situation, company, decision, or moment, not a general description.",
    "process-extraction":
      "Ask the expert to walk through their reasoning step by step in that case. What did they look at first? What decision points did they encounter? What told them to proceed or stop?",
    priority:
      "Ask the expert which signals or criteria matter most when multiple factors are present. What would they prioritize if they could only consider one thing?",
    abstraction:
      "Ask the expert to generalize from the specific case. What is the underlying principle? Would they apply the same logic to a different company or situation? How would they state the rule to a junior colleague?",
    boundary:
      "Ask the expert when this rule does NOT apply. What conditions must be present for it to work? What changes would make them set it aside?",
    counterexample:
      "Ask the expert to describe a situation where following this rule would be wrong or misleading, or a past case where they got it wrong.",
    revision:
      "Ask the expert what new evidence or changed circumstances would cause them to revise this conclusion. What would make them change their mind?",
    transfer:
      "Ask whether this logic applies in a different domain, industry, or situation. Would they use the same approach in a different context?",
    confidence:
      "Ask how confident the expert is in this rule and why. Under what conditions are they most or least certain?",
  };

  return `You are conducting a structured expert knowledge elicitation session. Your task is to generate ONE natural, conversational question that advances the dialogue.

RULE BEING DEVELOPED:
"${rule.content}"

RECENT DIALOGUE:
${transcript}

QUESTION TYPE TO USE: ${questionType}
INSTRUCTION: ${instructions[questionType]}

Requirements:
- Ask only ONE question
- Make it feel like genuine curiosity, not an interrogation
- Reference specifics from the dialogue where relevant
- Keep it concise — 1 to 3 sentences maximum
- Do not include preamble, explanation, or a label like "Question:"

Return only the question text.`.trim();
}

/**
 * Build the synthesis prompt.
 *
 * LLM distils all expert turns into a single 3–6 sentence procedure-form rule.
 */
export function buildSynthesisPrompt(
  rule: CandidateRule,
  turns: DialogueTurn[],
): string {
  const expertTurns = turns
    .filter((t) => t.role === "expert")
    .map((t) => `- ${t.content}`)
    .join("\n");

  return `You are synthesizing expert knowledge into a single reusable rule.

LEARNING OBJECTIVE:
"${rule.content}"

WHAT THE EXPERT HAS SAID (all expert turns):
${expertTurns}

Your task: write a single consolidated rule statement that captures the expert's knowledge in a form that is:
- Stated as a procedure (what to do, step by step, or as a conditional judgment)
- Generalized beyond the specific examples — applicable to future cases
- Bounded: mentions when the rule applies and when it does not
- Includes the key exception or failure mode the expert described
- Mentions what evidence would trigger a revision of the rule
- Written in 3 to 6 sentences

Format: plain text, no bullet points, no section headers. Write the rule as a single coherent paragraph.

Return only the rule text.`.trim();
}

/**
 * Build the correction parsing prompt.
 *
 * LLM determines whether the expert accepted, corrected, or redirected,
 * and returns the final rule text with the correction type.
 */
export function buildCorrectionPrompt(
  synthesis: string,
  expertResponse: string,
): string {
  return `You are analyzing an expert's response to a proposed knowledge rule.

PROPOSED RULE:
"${synthesis}"

EXPERT RESPONSE:
"${expertResponse}"

Determine:
1. The revised rule text (incorporate the expert's correction; if they accepted without changes, return the original text unchanged)
2. The correction type

Correction types:
- "rule-revision": expert explicitly changes the core logic, sequence, or conclusion
- "scope-adjustment": expert narrows or widens the conditions under which the rule applies
- "counterexample-added": expert introduces a specific case the rule must now account for
- "redirect": expert rejects the framing or refocuses the discussion on a different aspect

Return a JSON object:
{
  "revised": "<the corrected or confirmed rule text as a single paragraph>",
  "correctionType": "rule-revision"
}

Return only valid JSON, no other text.`.trim();
}

// ── LLM-backed operations ─────────────────────────────────────────────────────

export async function assessGaps(
  response: string,
  rule: CandidateRule,
  turns: DialogueTurn[],
  llm: LLMFn,
): Promise<Partial<ConsolidationGapStatus>> {
  const prompt = buildGapAssessmentPrompt(response, rule, turns);
  const raw = await llm(prompt);
  const parsed = parseJSON<Partial<ConsolidationGapStatus>>(raw);
  if (!parsed) {
    console.warn("[dialogue] Gap assessment parse failed; returning no updates");
    return {};
  }
  return parsed;
}

export async function generateQuestion(
  qType: QuestionType,
  rule: CandidateRule,
  turns: DialogueTurn[],
  llm: LLMFn,
): Promise<string> {
  const prompt = buildQuestionPrompt(qType, rule, turns);
  const raw = await llm(prompt);
  return raw.trim();
}

export async function synthesize(
  rule: CandidateRule,
  turns: DialogueTurn[],
  llm: LLMFn,
): Promise<string> {
  const prompt = buildSynthesisPrompt(rule, turns);
  const raw = await llm(prompt);
  return raw.trim();
}

export async function parseCorrection(
  synthesis: string,
  expertResponse: string,
  llm: LLMFn,
): Promise<{ revised: string; correctionType: CorrectionType }> {
  const prompt = buildCorrectionPrompt(synthesis, expertResponse);
  const raw = await llm(prompt);
  const parsed = parseJSON<{ revised: string; correctionType: CorrectionType }>(
    raw,
  );
  if (!parsed?.revised || !parsed?.correctionType) {
    console.warn("[dialogue] Correction parse failed; keeping original synthesis");
    return { revised: synthesis, correctionType: "rule-revision" };
  }
  return parsed;
}

// ── processTurn — main entrypoint ─────────────────────────────────────────────

export type ProcessTurnResult = {
  session: DialogueSession;
  agentResponse: string;
  stage: SessionStage;
};

const stageMap: Record<QuestionType, SessionStage> = {
  "case-elicitation": "eliciting-case",
  "process-extraction": "extracting-process",
  priority: "extracting-process",
  abstraction: "abstracting",
  confidence: "abstracting",
  boundary: "testing-boundaries",
  counterexample: "testing-boundaries",
  revision: "testing-boundaries",
  transfer: "testing-boundaries",
};

/**
 * Process one expert input turn within an active teach session.
 *
 * Steps:
 *   1. Record the expert's turn in the transcript
 *   2. Get or create the active candidate rule
 *   3. Assess which gaps the expert's response closes (LLM)
 *   4. Select the next action (rule-based question type or "synthesize")
 *   5a. If synthesize + session is already in "synthesizing" stage:
 *       parse the expert's correction, mark rule synthesized, promote
 *   5b. If synthesize + session is not yet in "synthesizing" stage:
 *       generate a synthesis proposal, transition stage
 *   5c. Otherwise: generate the next targeted question (LLM)
 *   6. Save session and return
 *
 * Cold-start: the /teach command passes a synthetic expert opening message
 * ("I want to learn about: <objective>") so processTurn generates the first
 * case-elicitation question immediately without a special codepath.
 */
export async function processTurn(
  session: DialogueSession,
  expertInput: string,
  llm: LLMFn,
): Promise<ProcessTurnResult> {
  // 1. Record expert's turn
  let s = addExpertTurn(session, expertInput);

  // 2. Get or create the active candidate rule
  let rule: CandidateRule;
  [s, rule] = getOrCreateRule(s);

  // 3. Assess gaps from the expert's response
  const gapUpdates = await assessGaps(expertInput, rule, s.turns, llm);
  s = updateGaps(s, rule.id, gapUpdates);

  // Refresh rule reference after gap update
  rule = s.candidateRules.find((r) => r.id === rule.id)!;

  // 4. Determine next action
  const nextActionFixed = selectNextQuestionType(rule.gaps, s.questionHistory, rule.id);

  let agentResponse: string;
  let newStage: SessionStage = s.stage;

  if (nextActionFixed === "synthesize") {
    if (s.stage === "synthesizing") {
      // ── Expert is responding to our synthesis proposal ──────────────────
      // Find the synthesis text we proposed (last agent turn without a questionType)
      const lastSynthesisTurn = [...s.turns]
        .reverse()
        .find((t) => t.role === "agent" && !t.questionType);
      const priorSynthesis = lastSynthesisTurn?.content ?? rule.content;

      const { revised, correctionType } = await parseCorrection(
        priorSynthesis,
        expertInput,
        llm,
      );

      s = updateRule(s, rule.id, { content: revised, status: "synthesized" });
      s = await promoteSession(s, llm);

      agentResponse =
        `I've recorded that as the rule for "${s.domain}":\n\n` +
        `"${revised}"\n\n` +
        `It's been added to your knowledge store (provenance: session:${s.id}). ` +
        `Type /endteach to exit teach mode or continue teaching.`;
      newStage = "complete";
    } else {
      // ── All gaps just closed — propose synthesis ──────────────────────────
      const synthesisText = await synthesize(rule, s.turns, llm);
      s = updateRule(s, rule.id, { content: synthesisText });

      // Record the synthesis as an agent turn (no questionType — used as
      // marker by the correction parser on the next turn)
      s = addAgentTurn(s, synthesisText, undefined, rule.id);

      agentResponse =
        `Based on everything you've shared, here's what I've learned:\n\n` +
        `"${synthesisText}"\n\n` +
        `Does this capture it correctly? If anything is wrong or missing, just tell me and I'll revise it.`;
      newStage = "synthesizing";
    }
  } else {
    // ── Ask the next targeted question ────────────────────────────────────
    const question = await generateQuestion(
      nextActionFixed,
      rule,
      s.turns,
      llm,
    );

    // Append to question history (dedup guard for next turn)
    const historyEntry: QuestionHistoryEntry = {
      questionType: nextActionFixed,
      candidateRuleId: rule.id,
      turnId: randomUUID(),
    };
    s = {
      ...s,
      questionHistory: [...s.questionHistory, historyEntry],
    };

    // Record the agent's question in the transcript
    s = addAgentTurn(s, question, nextActionFixed, rule.id);

    agentResponse = question;
    newStage = stageMap[nextActionFixed] ?? s.stage;
  }

  s = { ...s, stage: newStage };
  await saveSession(s);

  return { session: s, agentResponse, stage: newStage };
}
