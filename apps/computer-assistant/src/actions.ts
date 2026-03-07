/**
 * OS-level actions for the computer assistant.
 *
 * Cross-platform support:
 *   Windows: cmd /c start "" "target"
 *   macOS:   open "target"
 *   Linux:   xdg-open "target"
 */

import { exec } from "node:child_process";
import { promisify } from "node:util";
import { platform } from "node:process";
import { existsSync, statSync } from "node:fs";
import { resolve } from "node:path";

const execAsync = promisify(exec);

// ---------------------------------------------------------------------------
// Action types
// ---------------------------------------------------------------------------

export type ActionKind =
  | "open-file"
  | "open-folder"
  | "open-url"
  | "run-command"
  | "say"
  | "unknown";

export type Action = {
  kind: ActionKind;
  /** Target: file path, folder path, URL, or shell command */
  target: string;
  /** Human-readable message from the agent */
  message: string;
};

export type ActionResult = {
  success: boolean;
  output?: string;
  error?: string;
};

// ---------------------------------------------------------------------------
// Platform detection
// ---------------------------------------------------------------------------

function openCommand(target: string): string {
  // Escape double quotes in target
  const escaped = target.replace(/"/g, '\\"');
  switch (platform) {
    case "win32":
      // `start ""` — first arg is window title (empty), second is the path/URL
      return `cmd /c start "" "${escaped}"`;
    case "darwin":
      return `open "${escaped}"`;
    default:
      return `xdg-open "${escaped}"`;
  }
}

// ---------------------------------------------------------------------------
// Action helpers
// ---------------------------------------------------------------------------

/**
 * Classify a raw target string as file, folder, or URL.
 * Used by the agent when the LLM is uncertain.
 */
export function classifyTarget(
  target: string,
): "file" | "folder" | "url" | "unknown" {
  if (target.startsWith("http://") || target.startsWith("https://")) {
    return "url";
  }

  // Resolve relative paths
  const resolved = resolve(target);
  if (existsSync(resolved)) {
    try {
      const stat = statSync(resolved);
      return stat.isDirectory() ? "folder" : "file";
    } catch {
      return "unknown";
    }
  }

  // Heuristic: has file extension → likely a file
  if (/\.\w{1,8}$/.test(target)) return "file";

  return "unknown";
}

// ---------------------------------------------------------------------------
// Executors
// ---------------------------------------------------------------------------

async function openFile(target: string): Promise<ActionResult> {
  const resolved = resolve(target);
  if (!existsSync(resolved)) {
    return { success: false, error: `File not found: ${resolved}` };
  }
  try {
    await execAsync(openCommand(resolved));
    return { success: true, output: `Opened file: ${resolved}` };
  } catch (err) {
    return { success: false, error: String(err) };
  }
}

async function openFolder(target: string): Promise<ActionResult> {
  const resolved = resolve(target);
  if (!existsSync(resolved)) {
    return { success: false, error: `Folder not found: ${resolved}` };
  }
  try {
    await execAsync(openCommand(resolved));
    return { success: true, output: `Opened folder: ${resolved}` };
  } catch (err) {
    return { success: false, error: String(err) };
  }
}

async function openUrl(target: string): Promise<ActionResult> {
  // Ensure the target has a protocol
  const url = target.startsWith("http") ? target : `https://${target}`;
  try {
    await execAsync(openCommand(url));
    return { success: true, output: `Opened URL: ${url}` };
  } catch (err) {
    return { success: false, error: String(err) };
  }
}

async function runCommand(command: string): Promise<ActionResult> {
  try {
    const { stdout, stderr } = await execAsync(command);
    const output = [stdout, stderr].filter(Boolean).join("\n");
    return { success: true, output: output.trim() || "(no output)" };
  } catch (err) {
    const e = err as { message: string; stdout?: string; stderr?: string };
    const output = [e.stdout, e.stderr].filter(Boolean).join("\n");
    return {
      success: false,
      output: output.trim() || undefined,
      error: e.message,
    };
  }
}

// ---------------------------------------------------------------------------
// Main dispatcher
// ---------------------------------------------------------------------------

/**
 * Execute an action determined by the agent.
 *
 * @param action - Parsed action from the agent LLM response
 * @returns ActionResult indicating success or failure
 */
export async function executeAction(action: Action): Promise<ActionResult> {
  switch (action.kind) {
    case "open-file":
      return openFile(action.target);
    case "open-folder":
      return openFolder(action.target);
    case "open-url":
      return openUrl(action.target);
    case "run-command":
      return runCommand(action.target);
    case "say":
      return { success: true, output: action.message };
    case "unknown":
    default:
      return { success: false, error: `Unknown action kind: ${action.kind}` };
  }
}

// ---------------------------------------------------------------------------
// Parse agent LLM response → Action
// ---------------------------------------------------------------------------

/**
 * Parse the agent LLM's JSON response into a typed Action.
 *
 * Falls back to a "say" action if parsing fails.
 */
export function parseAgentResponse(raw: string): Action {
  // Strip markdown code fences if present
  const stripped = raw
    .trim()
    .replace(/^```(?:json)?\s*\n?/, "")
    .replace(/\n?```\s*$/, "");

  try {
    const obj = JSON.parse(stripped) as {
      action?: string;
      target?: string;
      message?: string;
    };

    const kind = (obj.action ?? "say") as ActionKind;
    const validKinds: ActionKind[] = [
      "open-file",
      "open-folder",
      "open-url",
      "run-command",
      "say",
    ];

    return {
      kind: validKinds.includes(kind) ? kind : "say",
      target: obj.target ?? "",
      message: obj.message ?? raw,
    };
  } catch {
    return { kind: "say", target: "", message: raw };
  }
}
