/**
 * OpenClaw lifecycle hooks for Milestones 1c and 1d.
 *
 * 1c — Two-pass passive extraction:
 *      • message_received (pre-pass)   — extracts from the raw inbound message
 *        before the agent responds. Gated by per-sender trust check.
 *      • agent_end (exchange-pass)     — extracts from the full user→agent turn
 *        after the agent finishes. Carries richer signal (agent phrasing may
 *        confirm or clarify facts the user stated). Gated on trustedSenders
 *        being non-empty and on the turn having succeeded.
 *      Both passes require an LLM adapter for extraction.
 *
 * 1d — before_prompt_build: automatic retrieval and artifact injection
 *      at near-zero cost (in-memory tag-overlap lookup, no LLM call needed).
 *
 * All hooks are registered together; each degrades gracefully when its
 * prerequisites are not met (e.g. no LLM adapter = 1c disabled, 1d still
 * works; trustedSenders empty = exchange-pass skipped).
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
 * Format an `agent_end` messages array as a readable exchange string for
 * extraction. The messages array uses Anthropic message format internally.
 *
 * Takes the last `maxMessages` entries to bound token cost on long sessions.
 * Returns null when no usable text can be extracted.
 */
function formatExchangeForExtraction(
  messages: unknown[],
  maxMessages = 8,
): string | null {
  const tail = messages.slice(-maxMessages);
  const lines: string[] = [];

  for (const msg of tail) {
    if (typeof msg !== "object" || msg === null) continue;
    const { role, content } = msg as { role?: unknown; content?: unknown };
    if (role !== "user" && role !== "assistant") continue;

    let text: string | null = null;
    if (typeof content === "string") {
      text = content;
    } else if (Array.isArray(content)) {
      const parts = content
        .filter(
          (b): b is { type: "text"; text: string } =>
            typeof b === "object" &&
            b !== null &&
            (b as { type?: unknown }).type === "text" &&
            typeof (b as { text?: unknown }).text === "string",
        )
        .map((b) => b.text);
      if (parts.length > 0) text = parts.join(" ");
    }

    if (text?.trim()) {
      lines.push(`${role === "user" ? "User" : "Agent"}: ${text.trim()}`);
    }
  }

  return lines.length > 0 ? lines.join("\n") : null;
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

  // ── Milestone 1c (exchange-pass): extract from the completed turn ─────────
  //
  // Fires after the agent finishes responding. At this point we have the full
  // user→agent exchange, which carries richer extraction signal than the raw
  // inbound message alone (the agent's phrasing may confirm or clarify facts).
  //
  // Sender-check rationale: agent_end has no per-message `from` field, so we
  // gate on trustedSenders being non-empty (user has opted into learning) and
  // on the turn having succeeded (avoid extracting from error sessions).
  api.on("agent_end", async (event, _ctx) => {
    if (!llm || trustedSenders.length === 0 || !event.success) return;

    const exchange = formatExchangeForExtraction(event.messages);
    if (!exchange) return;

    try {
      await processMessage(exchange, llm, "agent_end");
    } catch (err) {
      api.logger.warn(`[knowledge-fabric] exchange-pass extraction failed: ${String(err)}`);
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
