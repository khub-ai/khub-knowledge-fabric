/**
 * Benchmark: procedure → executable program escalation
 *
 * Validates the full recipe-to-program lifecycle described in
 * docs/example-learning-in-action.md Session 4:
 *
 *   "I've now run this procedure three months in a row and the steps are
 *    identical each time. Want me to turn it into a script?"
 *   → Agent generates a Python script, attaches it to the procedure artifact,
 *     and on the next relevant request uses run-command instead of manual steps.
 *
 * Four phases:
 *   Phase 1 — compileToProgram() attaches code + path to the procedure artifact
 *   Phase 2 — the program survives a persist/loadAll round-trip
 *   Phase 3 — formatArtifactsForPrompt() includes [EXECUTABLE: …] for the agent LLM
 *   Phase 4 — a second compile updates in-place (no orphan artifacts)
 *
 * This file serves two purposes:
 *   - Automated regression test (run with: pnpm test)
 *   - Instructional documentation of the recipe → program escalation
 *
 * See also: docs/example-learning-in-action.md (Session 4)
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { unlink, readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { processMessage, compileToProgram } from "@khub-ai/openclaw-plus/pipeline";
import { loadAll } from "@khub-ai/openclaw-plus/store";
import {
  createCompileBenchmarkLLM,
  COMPILE_CODE_RESPONSE,
} from "./mock-llm.js";
import { formatArtifactsForPrompt } from "../agent.js";

// ---------------------------------------------------------------------------
// Test isolation — each test gets its own ephemeral store + program dir
// ---------------------------------------------------------------------------

let testStorePath: string;
let testProgramDir: string;

beforeEach(() => {
  const tag = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  testStorePath = join(tmpdir(), `pil-compile-bench-${tag}.jsonl`);
  testProgramDir = join(tmpdir(), `pil-compile-programs-${tag}`);
  process.env["KNOWLEDGE_STORE_PATH"] = testStorePath;
});

afterEach(async () => {
  delete process.env["KNOWLEDGE_STORE_PATH"];
  if (existsSync(testStorePath)) await unlink(testStorePath);
  // Program files are cleaned up by the OS tmpdir GC; no explicit cleanup needed
});

// ---------------------------------------------------------------------------
// Shared setup: run sessions 1–3 from the financial statements scenario
// ---------------------------------------------------------------------------

const SESSION1_EXCHANGE =
  'User: Download my January statement from Chase, rename it "2026-01-chase-checking.pdf",' +
  " and put it in my Finance/Statements folder.\n" +
  "Assistant: Done. Statement saved to ~/Finance/Statements/2026-01-chase-checking.pdf.";

const SESSION2_EXCHANGE =
  "User: Download my February statement from Chase. Same as last time.\n" +
  'Assistant: Got it — renamed to "2026-02-chase-checking.pdf" and saved to ~/Finance/Statements/. ' +
  "I notice you follow a consistent naming pattern: YYYY-MM-institution-account.pdf. " +
  "Want me to remember this for future downloads?\n" +
  "User: Yes, always use that pattern.";

const SESSION3_EXCHANGE =
  "User: Get my February statement from Fidelity too.\n" +
  "Assistant: Saved to ~/Finance/Statements/2026-02-fidelity-brokerage.pdf. " +
  "By the way — you now download statements from two institutions. " +
  "Want me to create a monthly checklist so we don't miss any?\n" +
  "User: Good idea. Add Amex too — I always forget that one.";

async function setupSessions123(): Promise<{
  llm: ReturnType<typeof createCompileBenchmarkLLM>;
  procedure: Awaited<ReturnType<typeof loadAll>>[number];
  relatedArtifacts: Awaited<ReturnType<typeof loadAll>>;
}> {
  const llm = createCompileBenchmarkLLM();
  await processMessage(SESSION1_EXCHANGE, llm, "pil-chat:exchange");
  await processMessage(SESSION2_EXCHANGE, llm, "pil-chat:exchange");
  await processMessage(SESSION3_EXCHANGE, llm, "pil-chat:exchange");

  const all = await loadAll();
  const active = all.filter((a) => !a.retired);
  const procedure = active.find((a) => a.kind === "procedure")!;
  const relatedArtifacts = active.filter((a) => a.kind !== "procedure");

  return { llm, procedure, relatedArtifacts };
}

// ---------------------------------------------------------------------------
// Phase 1 — compileToProgram() generates and attaches code
// ---------------------------------------------------------------------------

describe("Benchmark compile/Phase 1 — code generation and attachment", () => {
  it("compileToProgram attaches a program block to the procedure artifact", async () => {
    const { llm, procedure, relatedArtifacts } = await setupSessions123();

    const updated = await compileToProgram(
      procedure, relatedArtifacts, "python", llm, testProgramDir,
    );

    expect(updated.program).toBeDefined();
    expect(updated.program!.language).toBe("python");
    expect(updated.program!.code).toBeTruthy();
    expect(updated.program!.generatedAt).toBeTruthy();
  });

  it("generates code matching the mock LLM response", async () => {
    const { llm, procedure, relatedArtifacts } = await setupSessions123();
    const updated = await compileToProgram(
      procedure, relatedArtifacts, "python", llm, testProgramDir,
    );
    // Code is the raw LLM output, trimmed
    expect(updated.program!.code).toBe(COMPILE_CODE_RESPONSE.trim());
  });

  it("saves the script to disk and sets program.path", async () => {
    const { llm, procedure, relatedArtifacts } = await setupSessions123();
    const updated = await compileToProgram(
      procedure, relatedArtifacts, "python", llm, testProgramDir,
    );

    expect(updated.program!.path).toBeDefined();
    expect(updated.program!.path).toMatch(/\.py$/);
    expect(existsSync(updated.program!.path!)).toBe(true);
  });

  it("file on disk contains the generated code", async () => {
    const { llm, procedure, relatedArtifacts } = await setupSessions123();
    const updated = await compileToProgram(
      procedure, relatedArtifacts, "python", llm, testProgramDir,
    );

    const onDisk = await readFile(updated.program!.path!, "utf-8");
    expect(onDisk).toBe(COMPILE_CODE_RESPONSE.trim());
  });

  it("procedure content (recipe) is preserved unchanged", async () => {
    const { llm, procedure, relatedArtifacts } = await setupSessions123();
    const originalContent = procedure.content;
    const updated = await compileToProgram(
      procedure, relatedArtifacts, "python", llm, testProgramDir,
    );
    // The recipe is never replaced — only the program field is added
    expect(updated.content).toBe(originalContent);
  });
});

// ---------------------------------------------------------------------------
// Phase 2 — program field survives persist/loadAll round-trip
// ---------------------------------------------------------------------------

describe("Benchmark compile/Phase 2 — persistence", () => {
  it("program field is present after loadAll()", async () => {
    const { llm, procedure, relatedArtifacts } = await setupSessions123();
    const updated = await compileToProgram(
      procedure, relatedArtifacts, "python", llm, testProgramDir,
    );

    // Reload from disk to verify the field survived serialisation
    const all = await loadAll();
    const reloaded = all.find((a) => a.id === updated.id);

    expect(reloaded).toBeDefined();
    expect(reloaded!.program).toBeDefined();
    expect(reloaded!.program!.path).toBe(updated.program!.path);
    expect(reloaded!.program!.code).toBe(updated.program!.code);
  });

  it("store still has exactly 3 active artifacts after compiling (in-place update)", async () => {
    // compileToProgram calls revise() with only the program field changed.
    // Jaccard of content vs itself = 1.0 >> 0.5 threshold → in-place update,
    // no retirement + fresh-id cycle. Active count stays at 3.
    const { llm, procedure, relatedArtifacts } = await setupSessions123();
    await compileToProgram(procedure, relatedArtifacts, "python", llm, testProgramDir);

    const all = await loadAll();
    const active = all.filter((a) => !a.retired);
    expect(active).toHaveLength(3); // convention + fact + procedure (with program)
  });
});

// ---------------------------------------------------------------------------
// Phase 3 — formatArtifactsForPrompt includes [EXECUTABLE: …] hint
// ---------------------------------------------------------------------------

describe("Benchmark compile/Phase 3 — EXECUTABLE hint in agent context", () => {
  it("includes [EXECUTABLE: path] for a procedure with a saved program", async () => {
    const { llm, procedure, relatedArtifacts } = await setupSessions123();
    const updated = await compileToProgram(
      procedure, relatedArtifacts, "python", llm, testProgramDir,
    );

    // Reload so we get the persisted version with program.path populated
    const all = await loadAll();
    const context = formatArtifactsForPrompt(all.filter((a) => !a.retired));

    expect(context).toContain("[EXECUTABLE:");
    expect(context).toContain(updated.program!.path!);
  });

  it("does NOT include [EXECUTABLE:] for a procedure without a compiled program", async () => {
    await setupSessions123(); // procedure exists but no compile yet
    const all = await loadAll();
    const context = formatArtifactsForPrompt(all.filter((a) => !a.retired));

    expect(context).not.toContain("[EXECUTABLE:");
  });
});

// ---------------------------------------------------------------------------
// Phase 4 — re-compiling updates in-place (no orphan artifacts)
// ---------------------------------------------------------------------------

describe("Benchmark compile/Phase 4 — re-compile updates in place", () => {
  it("second compile overwrites program.code, not creates a new artifact", async () => {
    const { llm, procedure, relatedArtifacts } = await setupSessions123();

    // First compile
    await compileToProgram(procedure, relatedArtifacts, "python", llm, testProgramDir);

    // Reload the updated procedure for the second compile
    const afterFirst = await loadAll();
    const compiledProc = afterFirst.find((a) => a.kind === "procedure" && !a.retired)!;
    expect(compiledProc.program).toBeDefined();

    // Second compile (same mock → same code, but revisedAt will update)
    const recompiled = await compileToProgram(
      compiledProc, afterFirst.filter((a) => a.kind !== "procedure"), "python", llm, testProgramDir,
    );

    // Still the same artifact id (in-place update)
    expect(recompiled.id).toBe(compiledProc.id);
    expect(recompiled.program!.code).toBe(COMPILE_CODE_RESPONSE.trim());
  });

  it("store still has exactly 3 active artifacts after re-compiling", async () => {
    const { llm, procedure, relatedArtifacts } = await setupSessions123();
    await compileToProgram(procedure, relatedArtifacts, "python", llm, testProgramDir);

    const afterFirst = await loadAll();
    const compiledProc = afterFirst.find((a) => a.kind === "procedure" && !a.retired)!;
    await compileToProgram(
      compiledProc, afterFirst.filter((a) => a.kind !== "procedure"), "python", llm, testProgramDir,
    );

    const final = await loadAll();
    const active = final.filter((a) => !a.retired);
    expect(active).toHaveLength(3);
  });
});
