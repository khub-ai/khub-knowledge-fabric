/**
 * PIL-aware agent for the computer assistant.
 *
 * Each user turn:
 *   1. Retrieve relevant PIL artifacts from previous sessions (Tier 1/2)
 *   2. Build agent prompt with injected PIL context
 *   3. Call agent LLM to determine action
 *   4. Execute the action
 *   5. Process user message through PIL pipeline (learn for future sessions)
 */

import { homedir } from "node:os";
import { join } from "node:path";
import {
  retrieve,
  isInjectable,
  getInjectLabel,
} from "@khub-ai/openclaw-plus/store";
import {
  processMessage,
  compileToProgram,
  formatForInjection,
  type InjectableArtifact,
} from "@khub-ai/openclaw-plus/pipeline";
import type { LLMFn } from "@khub-ai/openclaw-plus/types";
import type { KnowledgeArtifact } from "@khub-ai/openclaw-plus/types";
import { parseAgentResponse, executeAction, type Action, type ActionResult } from "./actions.js";

// ---------------------------------------------------------------------------
// Agent turn result
// ---------------------------------------------------------------------------

export type AgentTurnResult = {
  /** Human-readable message from the agent */
  message: string;
  /** The action determined by the agent LLM */
  action: Action;
  /** Result of executing the action */
  result: ActionResult;
  /** Artifacts retrieved from the store and injected into this turn's prompt */
  injectedArtifacts: KnowledgeArtifact[];
  /** Artifacts newly learned from this turn */
  learned: {
    created: KnowledgeArtifact[];
    updated: KnowledgeArtifact[];
  };
};

// ---------------------------------------------------------------------------
// Context formatting
// ---------------------------------------------------------------------------

/**
 * Format stored artifacts for injection into the agent prompt.
 *
 * Procedure artifacts with a compiled program attached get an EXECUTABLE
 * annotation so the agent LLM knows to prefer run-command over manual steps.
 */
export function formatArtifactsForPrompt(artifacts: KnowledgeArtifact[]): string {
  const injectable = artifacts
    .filter(isInjectable)
    .map((a) => {
      const label = getInjectLabel(a) ?? "[suggestion]";
      let line = `${label} [${a.kind}] ${a.content}`;
      // Append EXECUTABLE hint so the agent LLM can prefer the compiled script
      if (a.kind === "procedure" && a.program?.path) {
        line += ` [EXECUTABLE: ${a.program.path}]`;
      }
      return line;
    });

  if (injectable.length === 0) return "";

  return (
    "\n\nPIL KNOWLEDGE CONTEXT (remembered from previous sessions):\n" +
    injectable.join("\n") +
    "\n\nApply the above knowledge when interpreting the user's request."
  );
}

// ---------------------------------------------------------------------------
// Main agent turn
// ---------------------------------------------------------------------------

/**
 * Run one agent turn: retrieve → inject → decide → execute → learn.
 *
 * @param userInput  - The user's raw input text
 * @param agentLlm   - LLM adapter for agent decision-making (with system prompt)
 * @param pilLlm     - LLM adapter for PIL pipeline (extraction + consolidation)
 * @param sessionId  - Session identifier for provenance tracking
 * @param dryRun     - If true, determine action but do not execute it
 */
export async function runAgentTurn(
  userInput: string,
  agentLlm: LLMFn,
  pilLlm: LLMFn,
  sessionId: string,
  dryRun = false,
): Promise<AgentTurnResult> {
  // ── Step 1: Retrieve relevant artifacts (Tier 1 + content fallback) ──────
  const retrieved = await retrieve(userInput, 8);
  const injectedArtifacts = retrieved.filter(isInjectable);

  // ── Step 2: Build agent prompt with PIL context ───────────────────────────
  const pilContext = formatArtifactsForPrompt(retrieved);
  const agentPrompt = `${userInput}${pilContext}`;

  // ── Step 3: Call agent LLM to determine action ────────────────────────────
  const rawResponse = await agentLlm(agentPrompt);
  const action = parseAgentResponse(rawResponse);

  // ── Step 3.5: Handle compile-procedure (needs PIL store access) ──────────
  //
  // compile-procedure is intercepted here rather than in executeAction because
  // it requires the retrieved artifacts and the pilLlm — neither of which are
  // accessible from the stateless executeAction function.
  if (action.kind === "compile-procedure") {
    // Find the highest-confidence injectable procedure in the retrieved set
    const procedure = retrieved
      .filter((a) => a.kind === "procedure" && isInjectable(a))
      .sort((a, b) => b.confidence - a.confidence)[0];

    if (!procedure) {
      return {
        message: action.message,
        action,
        result: { success: false, error: "No stored procedure found to compile. Teach me a procedure first." },
        injectedArtifacts,
        learned: { created: [], updated: [] },
      };
    }

    const relatedArtifacts = retrieved.filter((a) => a.id !== procedure.id);
    const language = action.target || "python";
    const saveDir = join(homedir(), ".openclaw", "programs");
    const updated = await compileToProgram(procedure, relatedArtifacts, language, pilLlm, saveDir);

    return {
      message: action.message,
      action,
      result: {
        success: true,
        output: `Script saved to ${updated.program!.path ?? "(in-memory only)"}`,
      },
      injectedArtifacts,
      learned: { created: [], updated: [updated] },
    };
  }

  // ── Step 4: Execute the action (unless dry-run) ───────────────────────────
  const result = dryRun
    ? { success: true, output: `[dry-run] Would execute: ${action.kind} ${action.target}` }
    : await executeAction(action);

  // ── Step 5: Learn from the user's input (PIL pipeline) ────────────────────
  const provenance = `computer-assistant/${sessionId}`;
  const processResult = await processMessage(userInput, pilLlm, provenance);

  // Also format newly-injectable artifacts from this turn for display
  const newInjectable: InjectableArtifact[] = processResult.injectable;
  void newInjectable; // available to callers via learned field

  return {
    message: action.message,
    action,
    result,
    injectedArtifacts,
    learned: {
      created: processResult.created,
      updated: processResult.updated,
    },
  };
}

// ---------------------------------------------------------------------------
// Format a turn result for display in the REPL
// ---------------------------------------------------------------------------

export function formatTurnResult(turn: AgentTurnResult): string {
  const lines: string[] = [];

  // Action taken
  if (turn.action.kind !== "say") {
    lines.push(`[Action: ${turn.action.kind} → ${turn.action.target}]`);
  }

  // Agent's message
  if (turn.message) {
    lines.push(turn.message);
  }

  // Execution result
  if (turn.result.output) {
    lines.push(turn.result.output);
  }
  if (!turn.result.success && turn.result.error) {
    lines.push(`Error: ${turn.result.error}`);
  }

  // PIL learning summary
  const learnedCount = turn.learned.created.length + turn.learned.updated.length;
  if (learnedCount > 0) {
    const parts: string[] = [];
    if (turn.learned.created.length > 0) {
      const kinds = turn.learned.created.map((a) => a.kind).join(", ");
      parts.push(`${turn.learned.created.length} new (${kinds})`);
    }
    if (turn.learned.updated.length > 0) {
      parts.push(`${turn.learned.updated.length} updated`);
    }
    lines.push(`[PIL learned: ${parts.join("; ")}]`);
  }

  // PIL context used
  if (turn.injectedArtifacts.length > 0) {
    lines.push(
      `[PIL context: ${turn.injectedArtifacts.length} artifact(s) injected]`,
    );
  }

  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Format all stored artifacts (for /list command)
// ---------------------------------------------------------------------------

import { loadAll } from "@khub-ai/openclaw-plus/store";

export async function listArtifacts(): Promise<string> {
  const all = await loadAll();
  const active = all.filter((a) => !a.retired);

  if (active.length === 0) return "No artifacts stored yet.";

  const lines = active.map((a) => {
    const stage = a.stage ?? "?";
    const conf = (a.confidence ?? 0).toFixed(2);
    const tags = a.tags?.join(", ") ?? "";
    return `[${a.kind}/${stage}] conf=${conf}  ${a.content.slice(0, 80)}${a.content.length > 80 ? "…" : ""}${tags ? `\n   tags: ${tags}` : ""}`;
  });

  return `${active.length} active artifact(s):\n\n${lines.join("\n\n")}`;
}
