/**
 * OpenClaw lifecycle hooks for Milestones 1c and 1d.
 *
 * 1c — message_received: passive extraction from every inbound message.
 *      Gated by sender check. Requires an LLM adapter for extraction.
 *
 * 1d — before_prompt_build: automatic retrieval and artifact injection
 *      at near-zero cost (in-memory tag-overlap lookup, no LLM call needed).
 *
 * Both hooks are registered together; either can be skipped gracefully if
 * its prerequisites are not met (e.g. no LLM adapter = 1c disabled but 1d
 * still works).
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { processMessage } from "./pipeline.js";
import { retrieve, getInjectLabel } from "./store.js";
import type { LLMFn } from "./types.js";

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

export type KnowledgeHookOptions = {
  /**
   * Optional LLM adapter for PIL extraction (Milestone 1c).
   * If null or undefined, `message_received` fires but extraction is skipped;
   * artifact retrieval and injection (1d) still work.
   */
  llm?: LLMFn | null;

  /**
   * Senders whose messages may be used as knowledge sources (Milestone 1c).
   *
   * Security guard: PIL must only extract knowledge from messages originating
   * from the authenticated agent owner. External senders in a multi-channel
   * OpenClaw deployment could otherwise inject false knowledge.
   *
   * Values are compared case-insensitively. Use ["*"] to trust all senders
   * (only appropriate for local dev / single-user deployments).
   *
   * Default: [] — extraction is disabled until trustedSenders is configured.
   */
  trustedSenders?: string[];
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function isTrustedSender(from: string, trustedSenders: string[]): boolean {
  if (trustedSenders.length === 0) return false;
  if (trustedSenders.includes("*")) return true;
  const fromLower = from.toLowerCase();
  return trustedSenders.some((s) => s.toLowerCase() === fromLower);
}

/**
 * Format retrieved artifacts as a context block for prompt injection.
 * Returns null if there is nothing worth injecting.
 */
function formatRetrievedContext(artifacts: Awaited<ReturnType<typeof retrieve>>): string | null {
  const injectable = artifacts.filter((a) => getInjectLabel(a) !== null);
  if (injectable.length === 0) return null;

  const lines = injectable
    .map((a) => {
      const label = getInjectLabel(a);
      return `${label} ${a.content}`;
    })
    .join("\n");

  return `## Knowledge context\n${lines}`;
}

// ---------------------------------------------------------------------------
// registerKnowledgeHooks
// ---------------------------------------------------------------------------

/**
 * Register OpenClaw lifecycle hooks for passive knowledge elicitation (1c)
 * and automatic artifact retrieval and injection (1d).
 *
 * Call this from the plugin's `register()` function after `registerKnowledgeTools`.
 */
export function registerKnowledgeHooks(
  api: OpenClawPluginApi,
  opts: KnowledgeHookOptions = {},
): void {
  const { llm = null, trustedSenders = [] } = opts;

  if (!llm) {
    api.logger.warn(
      "[knowledge-fabric] No LLM adapter provided — passive extraction (Milestone 1c) " +
        "is disabled. Set ANTHROPIC_API_KEY and ensure @anthropic-ai/sdk is installed to enable it.",
    );
  }

  if (trustedSenders.length === 0) {
    api.logger.warn(
      "[knowledge-fabric] trustedSenders is empty — extraction is disabled for all senders. " +
        "Configure the 'trustedSenders' plugin option to enable Milestone 1c.",
    );
  } else if (trustedSenders.includes("*")) {
    api.logger.warn(
      "[knowledge-fabric] trustedSenders includes '*' — all senders are trusted. " +
        "This is only appropriate for local dev / single-user deployments.",
    );
  }

  // ── Milestone 1c: passive extraction on every inbound message ─────────────
  // ── Milestone 1d: stage retrieved artifacts for prompt injection ──────────
  api.on("message_received", async (event, _ctx) => {
    // ── 1c: extract knowledge if sender is trusted and LLM is available ──
    if (llm && isTrustedSender(event.from, trustedSenders)) {
      try {
        await processMessage(event.content, llm, `message_received:${event.from}`);
      } catch (err) {
        api.logger.warn(`[knowledge-fabric] extraction failed: ${String(err)}`);
      }
    }
  });

  // ── Milestone 1d: inject relevant artifacts into every prompt ─────────────
  //
  // Uses event.prompt as the retrieval query — no LLM cost, in-memory index.
  // Returns prependContext so the artifacts appear before the agent's system
  // prompt and the user's message.
  api.on("before_prompt_build", async (event, _ctx) => {
    try {
      const artifacts = await retrieve(event.prompt);
      const context = formatRetrievedContext(artifacts);
      if (!context) return;
      return { prependContext: context };
    } catch (err) {
      api.logger.warn(`[knowledge-fabric] retrieval failed: ${String(err)}`);
    }
  });
}
