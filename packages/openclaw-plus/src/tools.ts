/**
 * OpenClaw tool registrations for the knowledge-management plugin.
 *
 * Tools are exposed to the agent at runtime via api.registerTool().
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { retrieve } from "./store.js";

export function registerKnowledgeTools(api: OpenClawPluginApi): void {
  api.registerTool(
    (_ctx) => ({
      name: "knowledge_search",
      description:
        "Search persisted knowledge artifacts (procedures, preferences, conventions, etc.) relevant to the current task",
      parameters: {
        type: "object" as const,
        properties: {
          query: {
            type: "string",
            description: "Natural language query describing what to look up",
          },
        },
        required: ["query"],
      },
      async execute(_id: string, params: Record<string, unknown>) {
        const query =
          typeof params["query"] === "string" ? params["query"] : "";
        const artifacts = await retrieve(query);
        return {
          content: [{ type: "text", text: JSON.stringify(artifacts, null, 2) }],
          details: { artifacts },
        };
      },
    }),
    { names: ["knowledge_search"] },
  );
}
