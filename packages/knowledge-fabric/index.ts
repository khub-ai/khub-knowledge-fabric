import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { registerKnowledgeTools } from "./src/tools.js";
import { registerKnowledgeHooks } from "./src/hooks.js";
import type { LLMFn } from "./src/types.js";

// ---------------------------------------------------------------------------
// Plugin config schema
//
// Validated at plugin load time by OpenClaw. Fields:
//   trustedSenders  — array of sender IDs allowed as knowledge sources (1c).
//   llmModel        — Anthropic model used for extraction (optional, has default).
// ---------------------------------------------------------------------------

const configSchema = {
  jsonSchema: {
    type: "object",
    properties: {
      trustedSenders: {
        type: "array",
        items: { type: "string" },
        description:
          "Sender IDs whose messages may be extracted as knowledge. " +
          'Use ["*"] to trust all senders (local dev only).',
        default: [],
      },
      llmModel: {
        type: "string",
        description: "Anthropic model ID for PIL extraction. Defaults to claude-3-5-haiku-20241022.",
        default: "claude-3-5-haiku-20241022",
      },
    },
    additionalProperties: false,
  },
};

// ---------------------------------------------------------------------------
// Optional Anthropic LLM adapter
//
// knowledge-fabric is LLM-agnostic by design; the adapter is created here in
// the plugin entry point (not in the core library) so the core stays portable.
// Requires: ANTHROPIC_API_KEY env var + @anthropic-ai/sdk installed.
// ---------------------------------------------------------------------------

async function tryCreateAnthropicLlm(model: string): Promise<LLMFn | null> {
  const apiKey = process.env["ANTHROPIC_API_KEY"];
  if (!apiKey) return null;

  try {
    // Dynamic import keeps @anthropic-ai/sdk optional at the package level.
    const { default: Anthropic } = await import("@anthropic-ai/sdk");
    const client = new Anthropic({ apiKey });

    return async (prompt: string): Promise<string> => {
      const message = await client.messages.create({
        model,
        max_tokens: 1024,
        messages: [{ role: "user", content: prompt }],
      });
      const block = message.content[0];
      return block?.type === "text" ? block.text : "";
    };
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Plugin definition
// ---------------------------------------------------------------------------

const plugin = {
  id: "knowledge-fabric",
  name: "KHUB Knowledge Fabric",
  description:
    "A knowledge store that learns from your conversations, persists across sessions and agents, and stays on your machine — inspectable and portable by design.",
  kind: "knowledge",
  configSchema,

  async register(api: OpenClawPluginApi) {
    // ── Static tools (knowledge_search) ────────────────────────────────────
    registerKnowledgeTools(api);

    // ── Resolve plugin config ───────────────────────────────────────────────
    const cfg = (api.pluginConfig ?? {}) as Record<string, unknown>;
    const trustedSenders = Array.isArray(cfg["trustedSenders"])
      ? (cfg["trustedSenders"] as string[]).map(String)
      : [];
    const llmModel =
      typeof cfg["llmModel"] === "string"
        ? cfg["llmModel"]
        : "claude-3-5-haiku-20241022";

    // ── Optional Anthropic LLM adapter for extraction (Milestone 1c) ───────
    const llm = await tryCreateAnthropicLlm(llmModel);

    // ── Lifecycle hooks: passive extraction (1c) + auto-inject (1d) ────────
    registerKnowledgeHooks(api, { llm, trustedSenders });
  },
};

export default plugin;
