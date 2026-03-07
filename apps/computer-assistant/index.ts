/**
 * Computer Assistant — PIL demo REPL
 *
 * Demonstrates Milestones 1b, 1c, and 1d:
 *   1b — User teaches the assistant: "when I say 'gh', open https://github.com"
 *        → PIL extracts and persists a 'convention' artifact
 *   1c — On subsequent messages, PIL passively observes patterns
 *        → after 3 similar observations, consolidates into a rule
 *   1d — Tag-based Tier-1 retrieval injects relevant artifacts into every turn
 *
 * Usage:
 *   export ANTHROPIC_API_KEY=sk-ant-...
 *   pnpm start
 *
 * REPL commands:
 *   open <path>           — open a file, folder, or URL
 *   run <command>         — run a shell command
 *   remember <fact>       — explicitly teach the assistant something
 *   /list                 — show all stored PIL artifacts
 *   /clear                — clear the PIL artifact store (irreversible)
 *   /help                 — show this help
 *   exit | quit           — exit
 */

import * as readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { writeFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import { createPilLLM, createAgentLLM } from "./src/llm.js";
import { runAgentTurn, formatTurnResult, listArtifacts } from "./src/agent.js";
import { storePath } from "@khub-ai/openclaw-plus/store";

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------

const VERSION = "0.1.0 (Milestone 1b/1c/1d)";
const SESSION_ID = randomUUID().slice(0, 8);

function printWelcome(): void {
  console.log(`
╔══════════════════════════════════════════════════╗
║         PIL Computer Assistant  ${VERSION.padEnd(16)}║
║  Type /help for commands, exit to quit.          ║
╚══════════════════════════════════════════════════╝
Session: ${SESSION_ID}
Store:   ${storePath()}
`);
}

function printHelp(): void {
  console.log(`
Commands:
  open <path|url>      Open a file, folder, or URL
  run <command>        Execute a shell command
  remember <fact>      Teach the assistant something explicitly
  /list                Show all stored PIL artifacts
  /clear               Clear the entire artifact store
  /help                Show this help
  exit | quit          Exit

Examples:
  open README.md
  open ~/Downloads
  open https://github.com
  open gh                     (after teaching: 'gh means https://github.com')
  remember when I say gh I mean https://github.com
  run ls -la
`);
}

// ---------------------------------------------------------------------------
// /clear command — wipe the store
// ---------------------------------------------------------------------------

async function clearStore(rl: readline.Interface): Promise<void> {
  const confirm = await rl.question(
    "Clear all PIL artifacts? This cannot be undone. Type YES to confirm: ",
  );
  if (confirm.trim() === "YES") {
    await writeFile(storePath(), "", "utf-8");
    console.log("Store cleared.\n");
  } else {
    console.log("Cancelled.\n");
  }
}

// ---------------------------------------------------------------------------
// REPL
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  printWelcome();

  const pilLlm = createPilLLM();
  const agentLlm = createAgentLLM();

  const rl = readline.createInterface({ input, output });

  // Graceful Ctrl-C
  rl.on("SIGINT", () => {
    console.log("\nUse 'exit' to quit.\n");
  });

  while (true) {
    let userInput: string;
    try {
      userInput = await rl.question("\n> ");
    } catch {
      // EOF / Ctrl-D
      break;
    }

    const trimmed = userInput.trim();
    if (!trimmed) continue;

    // ── Built-in commands ─────────────────────────────────────────────────
    if (trimmed === "exit" || trimmed === "quit") break;
    if (trimmed === "/help") { printHelp(); continue; }
    if (trimmed === "/list") { console.log("\n" + await listArtifacts()); continue; }
    if (trimmed === "/clear") { await clearStore(rl); continue; }

    // ── Agent turn ────────────────────────────────────────────────────────
    try {
      const turn = await runAgentTurn(
        trimmed,
        agentLlm,
        pilLlm,
        SESSION_ID,
      );
      console.log("\n" + formatTurnResult(turn));
    } catch (err) {
      if (err instanceof Error && err.message.includes("ANTHROPIC_API_KEY")) {
        console.error(`\n${err.message}\n`);
        break;
      }
      console.error(`\nError: ${err instanceof Error ? err.message : String(err)}\n`);
    }
  }

  rl.close();
  console.log("\nGoodbye.");
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
