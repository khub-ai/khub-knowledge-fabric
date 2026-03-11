/**
 * pil-chat — Interactive CLI chatbot for testing PIL.
 *
 * Works independently of OpenClaw. Shares the same artifact store and
 * pipeline as the OpenClaw plugin, so knowledge accumulated here is
 * immediately visible to the agent in OpenClaw (and vice-versa).
 *
 * Usage (from repo root):
 *   pnpm chat                              # default store
 *   pnpm chat -- --store /tmp/test.jsonl   # isolated test store
 *   pnpm chat -- --no-persist              # ephemeral (discarded on exit)
 *   pnpm chat -- --verbose                 # show PIL pipeline details
 *   pnpm chat -- --log /tmp/session.log    # mirror all I/O to a log file
 *   pnpm chat -- --model claude-sonnet-4-6
 *   pnpm chat -- --match-model claude-3-5-haiku-20241022  # cheap model for semantic matching
 *   pnpm chat -- --dashboard               # browser-based chat + PIL monitor
 *   pnpm chat -- --dashboard --port 8080   # custom port (default: 7331)
 *   pnpm chat -- --fresh                   # clear store + log at startup
 *   pnpm chat -- --help
 *
 * Requires: ANTHROPIC_API_KEY environment variable
 */

import * as readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { randomUUID } from "node:crypto";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createWriteStream, rmSync, truncateSync } from "node:fs";
import type { WriteStream } from "node:fs";
import { writeFile } from "node:fs/promises";
import Anthropic from "@anthropic-ai/sdk";
import {
  processMessage,
  formatForInjection,
  type InjectableArtifact,
} from "@khub-ai/knowledge-fabric/pipeline";
import {
  retrieve,
  loadAll,
  storePath,
  getInjectLabel,
  effectiveConfidence,
} from "@khub-ai/knowledge-fabric/store";
import {
  CONSOLIDATION_THRESHOLD,
  DEFAULT_AUTO_APPLY_THRESHOLD,
  DECAY_CONSTANTS,
} from "@khub-ai/knowledge-fabric/types";
import type { KnowledgeArtifact, LLMFn } from "@khub-ai/knowledge-fabric/types";
import type {
  TurnResult,
  PilActivity,
  StoreEntry,
  ProcessTurnFn,
  SystemParams,
  DeleteArtifactsFn,
  SetArtifactsConfidenceFn,
} from "./dashboard.js";
import { startDashboard } from "./dashboard.js";

// ─── CLI options ─────────────────────────────────────────────────────────────

interface Options {
  model: string;
  /** Separate model for Stage 2 semantic matching. null = use same as model. */
  matchModel: string | null;
  verbose: boolean;
  noPersist: boolean;
  customStore: string | null;
  logPath: string | null;
  /** Launch browser-based dashboard instead of CLI readline loop. */
  dashboard: boolean;
  /** Port for the dashboard HTTP server (default: 7331). */
  port: number;
  /**
   * Fresh start: truncate the store and log file at startup.
   * The store remains persistent for this session (unlike --no-persist).
   */
  fresh: boolean;
}

function parseArgs(): Options {
  const args = process.argv.slice(2);
  const opts: Options = {
    model: "claude-sonnet-4-6",
    matchModel: null,
    verbose: false,
    noPersist: false,
    customStore: null,
    logPath: null,
    dashboard: false,
    port: 7331,
    fresh: false,
  };

  for (let i = 0; i < args.length; i++) {
    const a = args[i]!;
    if ((a === "--store" || a === "-s") && args[i + 1]) {
      opts.customStore = args[++i]!;
    } else if (a === "--no-persist") {
      opts.noPersist = true;
    } else if (a === "--verbose" || a === "-v") {
      opts.verbose = true;
    } else if ((a === "--model" || a === "-m") && args[i + 1]) {
      opts.model = args[++i]!;
    } else if ((a === "--match-model" || a === "-M") && args[i + 1]) {
      opts.matchModel = args[++i]!;
    } else if ((a === "--log" || a === "-l") && args[i + 1]) {
      opts.logPath = args[++i]!;
    } else if (a === "--dashboard" || a === "-d") {
      opts.dashboard = true;
    } else if (a === "--port" && args[i + 1]) {
      const p = parseInt(args[++i]!, 10);
      if (!isNaN(p)) opts.port = p;
    } else if (a === "--fresh" || a === "-F") {
      opts.fresh = true;
    } else if (a === "--help" || a === "-h") {
      printHelp();
      process.exit(0);
    }
  }

  return opts;
}

function printHelp(): void {
  console.log(`
pil-chat — Interactive CLI chatbot for testing PIL

Usage:
  pnpm chat -- --help          ← note the "--" separator required by pnpm
  pnpm chat -- <options>
  pnpm chat:help               ← shortcut for --help

Store options (mutually exclusive):
  (default)              Read/write from the default store
                         (~/.openclaw/knowledge/artifacts.jsonl or
                         KNOWLEDGE_STORE_PATH env var)
  --store <path>         Read/write from a specific file — useful for
                         an isolated test store that won't pollute your
                         real knowledge base
  --no-persist           Run with a temporary store that is deleted when
                         you exit — PIL runs normally but nothing is saved
  --fresh, -F            Truncate the store and log at startup, then run
                         normally with full persistence. Combine with
                         --store to wipe a specific test store. Combine
                         with --log to start the log from a clean slate.
                         (Ignored when --no-persist is set)

Other options:
  --dashboard, -d        Launch browser-based dashboard (chat + PIL monitor)
                         When active the browser is the primary interface;
                         the terminal only shows the startup line and URL.
  --port <n>             Dashboard HTTP port (default: 7331)
  --log <path>           Mirror all I/O to a log file (appended each run)
  --model <model>        Anthropic model (default: claude-sonnet-4-6)
  --match-model <model>  Separate model for semantic pattern matching
                         (default: same as --model; use haiku for cheaper
                         matching — the task is a simple YES/NO classification)
  --verbose, -v          Show PIL pipeline activity for every message
  --help, -h             Show this help

REPL commands (CLI mode and dashboard /cmd buttons):
  /store                 Show current store path and artifact counts
  /list                  List all active artifacts in the store
  /clear                 Clear conversation history (store is unchanged)
  /clearlog              Truncate the session log file (no-op if --log not set)
  /reset                 Delete all artifacts in the current store
  /help                  Show this command list
  exit / quit / Ctrl-D   Exit
`.trim());
}

// ─── Session log ─────────────────────────────────────────────────────────────

interface SessionLog {
  logInput: (line: string) => void;
  close: () => void;
  /** Truncate the log file and write a fresh header. Returns a status string. */
  clearLog: () => string;
}

function setupLog(logPath: string | null, fresh = false): SessionLog {
  if (!logPath) {
    return {
      logInput: () => {},
      close: () => {},
      clearLog: () => "No log configured — use --log <path> to enable logging.",
    };
  }

  // fresh=true → "w" truncates the file on open; false → "a" appends to it.
  const stream: WriteStream = createWriteStream(logPath, { flags: fresh ? "w" : "a" });
  const sep = "─".repeat(72);
  stream.write(`\n${sep}\nSession started: ${new Date().toISOString()}\n${sep}\n`);

  // Tee console.log and console.error to the log file.
  // All program output already goes through console.log, so patching it once
  // here is sufficient — no changes needed elsewhere in the code.
  const origLog   = console.log.bind(console);
  const origError = console.error.bind(console);

  console.log = (...args: unknown[]): void => {
    const line = args.map((a) => (typeof a === "string" ? a : String(a))).join(" ");
    origLog(line);
    stream.write(line + "\n");
  };

  console.error = (...args: unknown[]): void => {
    const line = args.map((a) => (typeof a === "string" ? a : String(a))).join(" ");
    origError(line);
    stream.write("[err] " + line + "\n");
  };

  // readline.question() echoes input to the terminal but bypasses console.log,
  // so we log each user input line explicitly via logInput().
  const logInput = (line: string): void => {
    stream.write(`You: ${line}\n`);
  };

  const close = (): void => {
    stream.write(`\nSession ended: ${new Date().toISOString()}\n`);
    stream.end();
    console.log   = origLog;
    console.error = origError;
  };

  const clearLog = (): string => {
    // Truncate the underlying file. Because the stream is opened with O_APPEND,
    // all subsequent writes go to the current end-of-file — which after truncation
    // is position 0, so the log restarts cleanly without reopening the stream.
    truncateSync(logPath, 0);
    stream.write(`${sep}\nLog cleared: ${new Date().toISOString()}\n${sep}\n`);
    return `Log cleared: ${logPath}`;
  };

  return { logInput, close, clearLog };
}

// ─── Store setup ─────────────────────────────────────────────────────────────

function setupStore(opts: Options): { displayPath: string; cleanup: () => void } {
  let tempPath: string | null = null;

  if (opts.noPersist) {
    tempPath = join(tmpdir(), `pil-chat-${randomUUID()}.jsonl`);
    process.env["KNOWLEDGE_STORE_PATH"] = tempPath;
  } else if (opts.customStore) {
    process.env["KNOWLEDGE_STORE_PATH"] = opts.customStore;
  }
  // else: leave KNOWLEDGE_STORE_PATH as-is (env var or default)

  const displayPath = opts.noPersist
    ? "(ephemeral — discarded on exit)"
    : storePath();

  const cleanup = (): void => {
    if (tempPath) {
      try { rmSync(tempPath); } catch { /* ignore if never created */ }
    }
  };

  return { displayPath, cleanup };
}

// ─── LLM setup ───────────────────────────────────────────────────────────────

interface LLMHandle {
  pilLlm: LLMFn;
  /** LLM used for semantic pattern matching in Stage 2. May be cheaper than pilLlm. */
  matchLlm: LLMFn;
  chat: (userMessage: string, systemPrompt: string) => Promise<string>;
  clearHistory: () => void;
}

function setupLLM(opts: Options): LLMHandle {
  const apiKey = process.env["ANTHROPIC_API_KEY"];
  if (!apiKey) {
    console.error(
      "Error: ANTHROPIC_API_KEY is not set.\n" +
      "  export ANTHROPIC_API_KEY=sk-ant-...\n",
    );
    process.exit(1);
  }

  const client = new Anthropic({ apiKey });

  // Single-turn adapter for PIL extraction/consolidation prompts
  const pilLlm: LLMFn = async (prompt: string): Promise<string> => {
    const msg = await client.messages.create({
      model: opts.model,
      max_tokens: 1024,
      messages: [{ role: "user", content: prompt }],
    });
    const block = msg.content[0];
    return block?.type === "text" ? block.text : "";
  };

  // Separate adapter for semantic matching: may use a cheaper/faster model.
  // The task is only a simple pattern-equivalence classification — the response
  // is always a single number ("1", "2", …) or "NONE", so max_tokens can be low.
  const matchLlm: LLMFn = opts.matchModel
    ? async (prompt: string): Promise<string> => {
        const msg = await client.messages.create({
          model: opts.matchModel!,
          max_tokens: 32,
          messages: [{ role: "user", content: prompt }],
        });
        const block = msg.content[0];
        return block?.type === "text" ? block.text : "";
      }
    : pilLlm; // default: reuse same model

  // Multi-turn conversation history for the actual chat
  const history: Array<{ role: "user" | "assistant"; content: string }> = [];

  const chat = async (userMessage: string, systemPrompt: string): Promise<string> => {
    history.push({ role: "user", content: userMessage });
    const msg = await client.messages.create({
      model: opts.model,
      max_tokens: 2048,
      system: systemPrompt,
      messages: history,
    });
    const text = msg.content[0]?.type === "text" ? msg.content[0].text : "";
    history.push({ role: "assistant", content: text });
    return text;
  };

  const clearHistory = (): void => { history.length = 0; };

  return { pilLlm, matchLlm, chat, clearHistory };
}

// ─── PIL activity helpers ─────────────────────────────────────────────────────

/** Convert raw processMessage() output to the serialisable PilActivity shape. */
function toPilActivity(
  result: Awaited<ReturnType<typeof processMessage>>,
): PilActivity {
  return {
    created: result.created.map((a) => ({
      id: a.id,
      kind: a.kind,
      stage: a.stage ?? "candidate",
      confidence: a.confidence,
      content: a.content,
      tags: a.tags ?? [],
    })),
    updated: result.updated.map((a) => ({
      id: a.id,
      evidenceCount: a.evidenceCount ?? 0,
      confidence: a.confidence,
      content: a.content,
    })),
    injectable: result.injectable.map(({ label, artifact }) => ({
      label,
      kind: artifact.kind,
      content: artifact.content,
      confidence: artifact.confidence,
    })),
    candidates: result.candidates.map((c) => ({
      kind: c.kind,
      certainty: c.certainty,
      tags: c.tags ?? [],
      content: c.content,
    })),
  };
}

/** Convert a KnowledgeArtifact to the dashboard-safe StoreEntry shape. */
function toStoreEntry(a: KnowledgeArtifact): StoreEntry {
  const eff = effectiveConfidence(a);

  // Compute decay components for the dashboard decay panel.
  const validationStrength = Math.max(
    0,
    (a.reinforcementCount ?? 0) +
    2 * (a.acceptedCount    ?? 0) -
      (a.rejectedCount      ?? 0),
  );
  const halfLifeDays =
    DECAY_CONSTANTS.BASE_HALF_LIFE_DAYS *
    (1 + DECAY_CONSTANTS.VALIDATION_ALPHA * validationStrength);
  let decayFactor = 1.0;
  if (a.lastRetrievedAt) {
    const daysSince =
      (Date.now() - new Date(a.lastRetrievedAt).getTime()) / 86_400_000;
    decayFactor = Math.pow(0.5, daysSince / halfLifeDays);
  }

  return {
    id: a.id,
    kind: a.kind,
    stage: a.stage ?? "candidate",
    confidence: a.confidence,
    effectiveConfidence: Math.round(eff * 1000) / 1000,
    content: a.content,
    tags: a.tags ?? [],
    evidenceCount: a.evidenceCount ?? 0,
    label: getInjectLabel(a) ?? "",
    lastRetrievedAt: a.lastRetrievedAt ?? null,
    reinforcementCount: a.reinforcementCount ?? 0,
    acceptedCount:      a.acceptedCount      ?? 0,
    rejectedCount:      a.rejectedCount       ?? 0,
    validationStrength,
    halfLifeDays:  Math.round(halfLifeDays),
    decayFactor:   Math.round(decayFactor * 1000) / 1000,
  };
}

/** Load and map all active (non-retired) artifacts to StoreEntry[]. */
async function buildStoreSnapshot(): Promise<StoreEntry[]> {
  const all = await loadAll();
  return all.filter((a) => !a.retired).map(toStoreEntry);
}

/**
 * Print PIL activity to the CLI terminal.
 * Used only in CLI mode; the dashboard renders from the PilActivity JSON
 * that is broadcast via SSE.
 */
function showPilActivityFromData(activity: PilActivity, verbose: boolean): void {
  if (verbose) {
    if (activity.candidates.length > 0) {
      console.log(`  [PIL] Extracted ${activity.candidates.length} candidate(s):`);
      for (const c of activity.candidates) {
        console.log(`    • [${c.kind}/${c.certainty}] tags: ${c.tags.join(", ")}`);
        console.log(`      "${c.content}"`);
      }
    }
    for (const a of activity.created) {
      console.log(
        `  [PIL] ✚ Created  [${a.kind}/${a.stage}] conf=${a.confidence.toFixed(2)}` +
        `  "${a.content.slice(0, 70)}${a.content.length > 70 ? "…" : ""}"`,
      );
    }
    for (const a of activity.updated) {
      console.log(
        `  [PIL] ↺ Updated  evidence=${a.evidenceCount}` +
        ` conf=${a.confidence.toFixed(2)}` +
        `  "${a.content.slice(0, 60)}${a.content.length > 60 ? "…" : ""}"`,
      );
    }
    for (const i of activity.injectable) {
      console.log(
        `  [PIL] → ${i.label} [${i.kind}]` +
        `  "${i.content.slice(0, 60)}${i.content.length > 60 ? "…" : ""}"`,
      );
    }
  } else {
    // Brief one-liner
    const parts: string[] = [];
    if (activity.created.length > 0)    parts.push(`+${activity.created.length} stored`);
    if (activity.updated.length > 0)    parts.push(`~${activity.updated.length} updated`);
    if (activity.injectable.length > 0) parts.push(`${activity.injectable.length} injectable`);
    if (parts.length > 0) console.log(`  [PIL: ${parts.join("  ")}]`);
  }
}

// ─── Core turn processor ──────────────────────────────────────────────────────

/**
 * Build a ProcessTurnFn that captures the LLM handles and runs the full PIL
 * pipeline for each user message or REPL command.
 *
 * The returned function is called identically by both the CLI readline loop
 * and the dashboard's POST /chat handler — the only difference is that CLI
 * mode intercepts /reset for a confirmation prompt before calling it, and
 * handles "exit"/"quit" at the loop level.
 */
function buildProcessTurn(
  pilLlm: LLMFn,
  matchLlm: LLMFn,
  chat: (userMessage: string, systemPrompt: string) => Promise<string>,
  clearHistory: () => void,
  displayPath: string,
  sessionLog: SessionLog,
): ProcessTurnFn {
  return async function processTurn(userInput: string): Promise<TurnResult> {

    // ── REPL commands ──────────────────────────────────────────────────────────

    if (userInput === "exit" || userInput === "quit") {
      return {
        isCommand: true,
        shouldExit: true,
        storeSnapshot: await buildStoreSnapshot(),
      };
    }

    if (userInput === "/help") {
      return {
        isCommand: true,
        commandOutput: "Commands: /store  /list  /clear  /reset  /help  exit",
        storeSnapshot: await buildStoreSnapshot(),
      };
    }

    if (userInput === "/store") {
      const all    = await loadAll();
      const active = all.filter((a) => !a.retired);
      return {
        isCommand: true,
        commandOutput:
          `Path    : ${displayPath}\n` +
          `Total   : ${all.length}  (${active.length} active, ${all.length - active.length} retired)`,
        storeSnapshot: active.map(toStoreEntry),
      };
    }

    if (userInput === "/list") {
      const active = (await loadAll()).filter((a) => !a.retired);
      const output = active.length === 0
        ? "(no artifacts)"
        : active
            .map((a) => {
              const label = (getInjectLabel(a) ?? "—").padEnd(15);
              const conf  = a.confidence.toFixed(2);
              const snip  = a.content.slice(0, 64) + (a.content.length > 64 ? "…" : "");
              return `${label} [${a.kind}/${a.stage ?? "?"}] conf=${conf}  "${snip}"`;
            })
            .join("\n");
      return { isCommand: true, commandOutput: output, storeSnapshot: active.map(toStoreEntry) };
    }

    if (userInput === "/clear") {
      clearHistory();
      return {
        isCommand: true,
        commandOutput: "Conversation history cleared.",
        storeSnapshot: await buildStoreSnapshot(),
      };
    }

    if (userInput === "/clearlog") {
      return {
        isCommand: true,
        commandOutput: sessionLog.clearLog(),
        storeSnapshot: await buildStoreSnapshot(),
      };
    }

    if (userInput === "/reset") {
      await writeFile(storePath(), "");
      return { isCommand: true, commandOutput: "Store cleared.", storeSnapshot: [] };
    }

    // ── PIL pre-pass ───────────────────────────────────────────────────────────

    let pilPreResult: Awaited<ReturnType<typeof processMessage>>;
    try {
      pilPreResult = await processMessage(userInput, pilLlm, "pil-chat", matchLlm);
    } catch (err) {
      return {
        isCommand: false,
        error: `PIL error: ${err instanceof Error ? err.message : String(err)}`,
        storeSnapshot: await buildStoreSnapshot(),
      };
    }

    const pilPre = toPilActivity(pilPreResult);

    // ── Context retrieval + system prompt ──────────────────────────────────────

    const contextArtifacts = await retrieve(userInput);
    const alreadyInjected  = new Set(pilPreResult.injectable.map((i) => i.artifact.id));
    const contextInjectables: InjectableArtifact[] = contextArtifacts
      .filter((a) => !alreadyInjected.has(a.id))
      .flatMap((a) => {
        const label = getInjectLabel(a);
        return label ? [{ artifact: a, label }] : [];
      });

    const injectionText = formatForInjection([
      ...pilPreResult.injectable,
      ...contextInjectables,
    ]);

    const systemPrompt = [
      "You are a helpful AI assistant.",
      injectionText ? `\n${injectionText}` : "",
    ].join("\n").trim();

    // ── LLM response ───────────────────────────────────────────────────────────

    let response: string;
    try {
      response = await chat(userInput, systemPrompt);
    } catch (err) {
      return {
        isCommand: false,
        error: `LLM error: ${err instanceof Error ? err.message : String(err)}`,
        pilPre,
        storeSnapshot: await buildStoreSnapshot(),
      };
    }

    // ── Exchange PIL pass ──────────────────────────────────────────────────────
    //
    // Knowledge often emerges only from seeing both sides of a conversation.
    // Example: user says "lmp" (opaque), assistant says it doesn't know, user
    // clarifies "it means list my preferences" — the fact that 'lmp' = 'list
    // my preferences' is only extractable when both turns are read together.
    //
    // Solution: after each turn, run a second PIL pass on the full exchange so
    // the extractor can resolve pronouns and co-references.  This pass is for
    // persistence only — it does not affect the current turn's system prompt.

    let pilExchange: PilActivity | undefined;
    try {
      const exchangeText  = `User: ${userInput}\nAssistant: ${response}`;
      const exchangeResult = await processMessage(
        exchangeText, pilLlm, "pil-chat:exchange", matchLlm,
      );
      pilExchange = toPilActivity(exchangeResult);
    } catch {
      // Non-fatal: log the exchange-pass result only in verbose CLI mode
    }

    const storeSnapshot = await buildStoreSnapshot();
    return { isCommand: false, response, pilPre, pilExchange, storeSnapshot };
  };
}

// ─── Artifact mutation helpers ────────────────────────────────────────────────
//
// Both functions load the full JSONL, apply the mutation, rewrite the file, then
// return a fresh store snapshot. The JSONL format is one JSON object per line.

async function rewriteStore(artifacts: KnowledgeArtifact[]): Promise<void> {
  const content = artifacts.map((a) => JSON.stringify(a)).join("\n");
  await writeFile(storePath(), content ? content + "\n" : "");
}

const deleteArtifactsFn: DeleteArtifactsFn = async (ids: string[]): Promise<StoreEntry[]> => {
  const all       = await loadAll();
  const remaining = all.filter((a) => !ids.includes(a.id));
  await rewriteStore(remaining);
  return remaining.filter((a) => !a.retired).map(toStoreEntry);
};

const setArtifactsConfidenceFn: SetArtifactsConfidenceFn = async (
  ids: string[],
  confidence: number,
): Promise<StoreEntry[]> => {
  const all     = await loadAll();
  const updated = all.map((a) => ids.includes(a.id) ? { ...a, confidence } : a);
  await rewriteStore(updated);
  return updated.filter((a) => !a.retired).map(toStoreEntry);
};

// ─── Main ────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const opts = parseArgs();

  // ── Fresh start: wipe store + log before opening anything ─────────────────
  // setupStore() sets KNOWLEDGE_STORE_PATH, so we run it first to resolve the
  // effective path, then immediately overwrite with an empty file.
  const { displayPath, cleanup } = setupStore(opts);
  if (opts.fresh && !opts.noPersist) {
    await writeFile(storePath(), "");
  }

  // Pass fresh flag to setupLog so it opens the file in "w" (truncate) mode
  // rather than "a" (append) — the session header becomes the first line.
  const sessionLog = setupLog(opts.logPath, opts.fresh);
  const { pilLlm, matchLlm, chat, clearHistory } = setupLLM(opts);

  const exit = (): never => {
    sessionLog.close();
    cleanup();
    process.exit(0);
  };

  process.on("exit", cleanup);
  process.on("SIGINT",  () => exit());
  process.on("SIGTERM", () => exit());

  const sessionStart = new Date().toISOString();
  const processTurn  = buildProcessTurn(pilLlm, matchLlm, chat, clearHistory, displayPath, sessionLog);

  // ── Dashboard mode ──────────────────────────────────────────────────────────

  if (opts.dashboard) {
    console.log("\npil-chat — PIL test harness");
    console.log(`  Store : ${displayPath}`);
    console.log(`  Model : ${opts.model}`);
    if (opts.matchModel) console.log(`  Match : ${opts.matchModel} (semantic matching)`);
    console.log(`  Mode  : dashboard`);
    if (opts.logPath) console.log(`  Log   : ${opts.logPath}`);

    const systemParams: SystemParams = {
      model:                  opts.model,
      matchModel:             opts.matchModel,
      storePath:              displayPath,
      consolidationThreshold: CONSOLIDATION_THRESHOLD,
      defaultInjectThreshold: DEFAULT_AUTO_APPLY_THRESHOLD,
    };

    startDashboard(
      opts.port,
      processTurn,
      displayPath,
      sessionStart,
      () => buildStoreSnapshot(),
      deleteArtifactsFn,
      setArtifactsConfidenceFn,
      systemParams,
    );

    // Keep process alive — the HTTP server handles all interaction from here.
    // SIGINT/SIGTERM handlers above will call exit() when the user presses Ctrl+C.
    await new Promise<never>(() => { /* intentionally never resolves */ });
    return;
  }

  // ── CLI mode (readline loop) ────────────────────────────────────────────────

  const iface = readline.createInterface({ input, output });

  console.log("\npil-chat — PIL test harness");
  console.log(`  Store : ${displayPath}`);
  console.log(`  Model : ${opts.model}`);
  if (opts.matchModel) console.log(`  Match : ${opts.matchModel} (semantic matching)`);
  console.log(`  Mode  : ${opts.verbose ? "verbose" : "normal"}`);
  if (opts.logPath) console.log(`  Log   : ${opts.logPath}`);
  console.log("\nType /help for commands, exit to quit.\n");

  while (true) {
    let userInput: string;
    try {
      userInput = (await iface.question("You: ")).trim();
    } catch {
      break; // Ctrl-D
    }

    sessionLog.logInput(userInput);
    if (!userInput) continue;

    // ── Exit ─────────────────────────────────────────────────────────────────
    if (userInput === "exit" || userInput === "quit") break;

    // ── /reset: ask for confirmation before wiping the store ─────────────────
    if (userInput === "/reset") {
      const answer = (
        await iface.question("  Delete all artifacts in the current store? (yes/no): ")
      ).trim();
      sessionLog.logInput(answer);
      if (answer === "yes" || answer === "y") {
        await writeFile(storePath(), "");
        console.log("  Store cleared.");
      } else {
        console.log("  Cancelled.");
      }
      continue;
    }

    // ── All other commands and regular chat ───────────────────────────────────
    let result: TurnResult;
    try {
      result = await processTurn(userInput);
    } catch (err) {
      console.error(`  [Error] ${err instanceof Error ? err.message : String(err)}`);
      continue;
    }

    if (result.isCommand) {
      if (result.commandOutput) console.log(result.commandOutput);
      continue;
    }

    if (result.error) {
      console.error(result.error);
      continue;
    }

    if (result.pilPre)      showPilActivityFromData(result.pilPre,      opts.verbose);
    if (result.response)    console.log(`\nAssistant: ${result.response}\n`);
    if (result.pilExchange) showPilActivityFromData(result.pilExchange, opts.verbose);
  }

  iface.close();
  console.log("\nGoodbye.");
  sessionLog.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
