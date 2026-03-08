/**
 * Anthropic LLM adapter for the computer-assistant demo.
 *
 * Provides two factory functions:
 *   createPilLLM()   — for PIL pipeline calls (extraction, consolidation)
 *   createAgentLLM() — for agent decision-making (with system prompt)
 *
 * Both return a `LLMFn` compatible with the PIL pipeline's dependency-injection
 * pattern. The core @khub-ai/openclaw-plus package has no direct dependency on
 * @anthropic-ai/sdk; the adapter lives here in the app layer.
 */

import Anthropic from "@anthropic-ai/sdk";
import type { LLMFn } from "@khub-ai/openclaw-plus/types";

// Default model used for all calls — fast and cost-efficient
const DEFAULT_MODEL = "claude-3-5-haiku-20241022";

// ---------------------------------------------------------------------------
// Shared Anthropic client (singleton)
// ---------------------------------------------------------------------------

let _client: Anthropic | null = null;

function getClient(): Anthropic {
  if (!_client) {
    const apiKey = process.env["ANTHROPIC_API_KEY"];
    if (!apiKey) {
      throw new Error(
        "ANTHROPIC_API_KEY environment variable is required.\n" +
          "Set it before running: export ANTHROPIC_API_KEY=sk-ant-...",
      );
    }
    _client = new Anthropic({ apiKey });
  }
  return _client;
}

// ---------------------------------------------------------------------------
// PIL pipeline LLM — extraction and consolidation
// ---------------------------------------------------------------------------

/**
 * Create an LLMFn suitable for PIL pipeline calls (extraction, consolidation).
 *
 * No system prompt — the PIL pipeline embeds all instructions in the user prompt.
 *
 * @param model - Anthropic model ID (default: claude-3-5-haiku-20241022)
 */
export function createPilLLM(model = DEFAULT_MODEL): LLMFn {
  return async (prompt: string): Promise<string> => {
    const client = getClient();
    const message = await client.messages.create({
      model,
      max_tokens: 1024,
      messages: [{ role: "user", content: prompt }],
    });

    const block = message.content[0];
    return block?.type === "text" ? block.text : "";
  };
}

// ---------------------------------------------------------------------------
// Agent decision LLM — computer assistant
// ---------------------------------------------------------------------------

const AGENT_SYSTEM_PROMPT = `You are a computer assistant that helps users work with files, folders, URLs, and system commands.

When the user gives you a command, determine exactly what they want to do and respond with a JSON object.

Available actions:
- "open-file":         Open a specific file with its default application
- "open-folder":       Open a folder in the file explorer
- "open-url":          Open a URL or web address in the default browser
- "run-command":       Execute a shell command
- "compile-procedure": Generate and save an executable script from the stored PIL procedure knowledge.
                       Set "target" to the language (e.g. "python"). The script will be saved to
                       ~/.openclaw/programs/ automatically.
- "say":               Respond with text only (no system action needed)

Decision rules:
- If target is a URL (starts with http/https) or refers to a known web service, use "open-url"
- If target looks like a folder path (no file extension, ends with / or \\, or is clearly a directory), use "open-folder"
- If target looks like a file (has a file extension, or the user says "file"), use "open-file"
- If the user asks to run a command (ls, dir, git, npm, etc.), use "run-command"
- If the user asks to automate a task, create a script, or generate executable code from a
  learned workflow, use "compile-procedure" with the appropriate language as "target"
- When PIL context includes a procedure line ending with [EXECUTABLE: /path/to/script],
  prefer { "action": "run-command", "target": "python /path/to/script" } for task-execution
  requests. Always name the script in your "message" so the user knows what is running.
- If unclear, use "say" and ask for clarification

Respond with ONLY a JSON object in this exact format:
{
  "action": "open-file|open-folder|open-url|run-command|say",
  "target": "the specific target (file path, folder path, URL, or command)",
  "message": "brief human-readable message about what you are doing"
}`;

/**
 * Create an LLMFn for the agent decision layer.
 *
 * Uses a computer-assistant system prompt. Wraps the user's message as the
 * human turn; any PIL knowledge context should be prepended to the prompt
 * string before calling this function.
 *
 * @param model - Anthropic model ID (default: claude-3-5-haiku-20241022)
 */
export function createAgentLLM(model = DEFAULT_MODEL): LLMFn {
  return async (prompt: string): Promise<string> => {
    const client = getClient();
    const message = await client.messages.create({
      model,
      max_tokens: 512,
      system: AGENT_SYSTEM_PROMPT,
      messages: [{ role: "user", content: prompt }],
    });

    const block = message.content[0];
    return block?.type === "text" ? block.text : "";
  };
}
