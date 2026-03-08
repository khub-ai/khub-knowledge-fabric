/**
 * Benchmark: "financial statements" scenario
 *
 * Validates the PIL pipeline against the narrative in
 * docs/example-learning-in-action.md — an agent that progressively learns from
 * real interactions rather than explicit "remember this" commands.
 *
 * Five sessions, each testing a distinct PIL capability:
 *
 *   Session 1 — One-off task: no artifacts created (episodic only)
 *   Session 2 — User confirms a naming pattern → convention extracted
 *   Session 3 — Two artifacts extracted in one exchange (fact + procedure)
 *   Session 4 — Stored procedure retrieved by natural language query
 *   Session 5 — Artifacts revised when an account closes and a new one opens
 *
 * This file serves two purposes:
 *   - Automated regression test (run with: pnpm test)
 *   - Verification that docs/example-learning-in-action.md works as advertised
 *
 * See also: docs/benchmarks/11-financial-statements.md
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { unlink } from "node:fs/promises";
import { existsSync } from "node:fs";
import { processMessage } from "@khub-ai/openclaw-plus/pipeline";
import { retrieve, loadAll, revise } from "@khub-ai/openclaw-plus/store";
import {
  createFinanceBenchmarkLLM,
  FINANCE_NAMING_CONVENTION_RESPONSE,
} from "./mock-llm.js";

// ---------------------------------------------------------------------------
// Test isolation — each test gets its own ephemeral store
// ---------------------------------------------------------------------------

let testStorePath: string;

beforeEach(() => {
  testStorePath = join(
    tmpdir(),
    `pil-finance-bench-${Date.now()}-${Math.random().toString(36).slice(2)}.jsonl`,
  );
  process.env["KNOWLEDGE_STORE_PATH"] = testStorePath;
});

afterEach(async () => {
  delete process.env["KNOWLEDGE_STORE_PATH"];
  if (existsSync(testStorePath)) await unlink(testStorePath);
});

// ---------------------------------------------------------------------------
// Simulated conversation exchanges (full User+Assistant turns)
//
// Each string is passed to processMessage() as the "exchange" text, giving PIL
// the full conversational context needed to resolve references and learn
// patterns that span multiple turns.
// ---------------------------------------------------------------------------

// Session 1: agent executes a one-off download task — no pattern confirmed
const SESSION1_EXCHANGE =
  'User: Download my January statement from Chase, rename it "2026-01-chase-checking.pdf",' +
  " and put it in my Finance/Statements folder.\n" +
  "Assistant: Done. Statement saved to ~/Finance/Statements/2026-01-chase-checking.pdf.";

// Session 2: agent spots the naming pattern and asks to remember it; user confirms
const SESSION2_EXCHANGE =
  "User: Download my February statement from Chase. Same as last time.\n" +
  'Assistant: Got it — renamed to "2026-02-chase-checking.pdf" and saved to ~/Finance/Statements/. ' +
  "I notice you follow a consistent naming pattern: YYYY-MM-institution-account.pdf. " +
  "Want me to remember this for future downloads?\n" +
  "User: Yes, always use that pattern.";

// Session 3: second institution, agent proposes a monthly checklist, user adds Amex
const SESSION3_EXCHANGE =
  "User: Get my February statement from Fidelity too.\n" +
  "Assistant: Saved to ~/Finance/Statements/2026-02-fidelity-brokerage.pdf. " +
  "By the way — you now download statements from two institutions. " +
  "Want me to create a monthly checklist so we don't miss any?\n" +
  "User: Good idea. Add Amex too — I always forget that one.";

// Session 4: user triggers the stored procedure by date — no new learning,
//            just retrieval and execution. Modelled as an exchange-level text
//            to show PIL correctly extracts nothing from a pure execution turn.
const SESSION4_EXCHANGE =
  "User: It's March 2nd — time for statements.\n" +
  "Assistant: On it. Following your monthly checklist:\n" +
  "✓ 2026-02-chase-checking.pdf\n" +
  "✓ 2026-02-fidelity-brokerage.pdf\n" +
  "✓ 2026-02-amex-credit.pdf\n" +
  "All saved to ~/Finance/Statements/.\n" +
  "I've now run this procedure three months in a row and the steps are identical each time. " +
  "Want me to turn it into a script you can run without needing me?\n" +
  "User: Yes, that would be great.";

// ---------------------------------------------------------------------------
// Shared setup: run sessions 1–3 and return the llm instance
// ---------------------------------------------------------------------------

async function setupSessions123() {
  const llm = createFinanceBenchmarkLLM();
  await processMessage(SESSION1_EXCHANGE, llm, "pil-chat:exchange");
  await processMessage(SESSION2_EXCHANGE, llm, "pil-chat:exchange");
  await processMessage(SESSION3_EXCHANGE, llm, "pil-chat:exchange");
  return llm;
}

// ---------------------------------------------------------------------------
// Phase 1 — Session 1: one-off task produces no artifacts
// ---------------------------------------------------------------------------

describe("Benchmark finance/Phase 1 — episodic-only (Session 1)", () => {
  it("creates no artifacts for a one-off download task with no confirmed pattern", async () => {
    // Session 1 is a single execution — the agent completes the task but has
    // not yet seen a repeated pattern worth generalising. PIL should extract
    // nothing and leave the store empty.
    const llm = createFinanceBenchmarkLLM();
    const result = await processMessage(SESSION1_EXCHANGE, llm, "pil-chat:exchange");

    expect(result.candidates).toHaveLength(0);
    expect(result.created).toHaveLength(0);
    expect(result.updated).toHaveLength(0);
    expect(result.injectable).toHaveLength(0);

    const all = await loadAll();
    expect(all).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// Phase 2 — Session 2: naming convention created after explicit user confirmation
// ---------------------------------------------------------------------------

describe("Benchmark finance/Phase 2 — convention creation (Session 2)", () => {
  it("extracts a naming convention when the user explicitly confirms the pattern", async () => {
    // "Yes, always use that pattern." is an explicit, definitive confirmation.
    // PIL sees this in context and extracts a convention about file naming.
    const llm = createFinanceBenchmarkLLM();
    await processMessage(SESSION1_EXCHANGE, llm, "pil-chat:exchange");
    const result = await processMessage(SESSION2_EXCHANGE, llm, "pil-chat:exchange");

    expect(result.candidates).toHaveLength(1);
    expect(result.created).toHaveLength(1);
    expect(result.updated).toHaveLength(0);

    const convention = result.created[0]!;
    expect(convention.kind).toBe("convention");
    expect(convention.content).toMatch(/YYYY-MM/);
    expect(convention.content).toMatch(/Finance\/Statements/);
    expect(convention.certainty).toBe("definitive");
  });

  it("convention is injectable immediately as [provisional] (definitive certainty)", async () => {
    const llm = createFinanceBenchmarkLLM();
    await processMessage(SESSION1_EXCHANGE, llm, "pil-chat:exchange");
    const result = await processMessage(SESSION2_EXCHANGE, llm, "pil-chat:exchange");

    expect(result.injectable).toHaveLength(1);
    expect(result.injectable[0]!.label).toBe("[provisional]");
  });

  it("uses the configured response verbatim (mock sanity check)", async () => {
    // Regression guard: verify the mock is actually firing.
    // If FINANCE_NAMING_CONVENTION_RESPONSE changes structure, the test above
    // would fail first — but this makes the failure message unambiguous.
    const parsed = JSON.parse(FINANCE_NAMING_CONVENTION_RESPONSE) as {
      candidates: { content: string }[];
    };
    const llm = createFinanceBenchmarkLLM();
    await processMessage(SESSION1_EXCHANGE, llm, "pil-chat:exchange");
    const result = await processMessage(SESSION2_EXCHANGE, llm, "pil-chat:exchange");

    expect(result.created[0]!.content).toBe(parsed.candidates[0]!.content);
  });
});

// ---------------------------------------------------------------------------
// Phase 3 — Session 3: fact + procedure extracted in a single exchange
// ---------------------------------------------------------------------------

describe("Benchmark finance/Phase 3 — multi-artifact extraction (Session 3)", () => {
  it("extracts two distinct artifacts (fact + procedure) from one exchange", async () => {
    // A single exchange produces two candidates in one LLM call:
    //   1. fact      — the user's institution list (Chase, Fidelity, Amex)
    //   2. procedure — the monthly download checklist
    // The pipeline iterates over both candidates independently and creates
    // two separate artifacts, each stored with its own id and kind.
    const llm = createFinanceBenchmarkLLM();
    await processMessage(SESSION1_EXCHANGE, llm, "pil-chat:exchange");
    await processMessage(SESSION2_EXCHANGE, llm, "pil-chat:exchange");
    const result = await processMessage(SESSION3_EXCHANGE, llm, "pil-chat:exchange");

    expect(result.candidates).toHaveLength(2);
    expect(result.created).toHaveLength(2);
    expect(result.updated).toHaveLength(0);

    const fact = result.created.find((a) => a.kind === "fact");
    const procedure = result.created.find((a) => a.kind === "procedure");

    expect(fact).toBeDefined();
    expect(procedure).toBeDefined();
  });

  it("fact artifact lists all three institutions", async () => {
    const llm = createFinanceBenchmarkLLM();
    await processMessage(SESSION1_EXCHANGE, llm, "pil-chat:exchange");
    await processMessage(SESSION2_EXCHANGE, llm, "pil-chat:exchange");
    const result = await processMessage(SESSION3_EXCHANGE, llm, "pil-chat:exchange");

    const fact = result.created.find((a) => a.kind === "fact")!;
    expect(fact.content).toContain("Chase");
    expect(fact.content).toContain("Fidelity");
    expect(fact.content).toContain("Amex");
  });

  it("procedure artifact describes a monthly download checklist", async () => {
    const llm = createFinanceBenchmarkLLM();
    await processMessage(SESSION1_EXCHANGE, llm, "pil-chat:exchange");
    await processMessage(SESSION2_EXCHANGE, llm, "pil-chat:exchange");
    const result = await processMessage(SESSION3_EXCHANGE, llm, "pil-chat:exchange");

    const procedure = result.created.find((a) => a.kind === "procedure")!;
    expect(procedure.content).toMatch(/month/i);
    expect(procedure.content).toMatch(/Chase/);
    // Procedure references the naming convention (learned in Session 2)
    expect(procedure.content).toMatch(/YYYY-MM/);
  });

  it("both new artifacts are injectable as [provisional]", async () => {
    const llm = createFinanceBenchmarkLLM();
    await processMessage(SESSION1_EXCHANGE, llm, "pil-chat:exchange");
    await processMessage(SESSION2_EXCHANGE, llm, "pil-chat:exchange");
    const result = await processMessage(SESSION3_EXCHANGE, llm, "pil-chat:exchange");

    // Two new definitive-certainty artifacts → two [provisional] injectables
    expect(result.injectable).toHaveLength(2);
    expect(result.injectable.every((i) => i.label === "[provisional]")).toBe(true);
  });

  it("store has exactly three active artifacts after sessions 1–3: convention + fact + procedure", async () => {
    await setupSessions123();

    const all = await loadAll();
    const active = all.filter((a) => !a.retired);
    expect(active).toHaveLength(3);

    expect(active.find((a) => a.kind === "convention")).toBeDefined();
    expect(active.find((a) => a.kind === "fact")).toBeDefined();
    expect(active.find((a) => a.kind === "procedure")).toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// Phase 4 — Session 4: retrieval surfaces the right artifact for the query
// ---------------------------------------------------------------------------

describe("Benchmark finance/Phase 4 — retrieval (Session 4)", () => {
  it("retrieve('time for statements') surfaces the procedure", async () => {
    // When the user says "time for statements" the agent should retrieve the
    // monthly download procedure. The tags "monthly-statements" and
    // "financial-statements" on the procedure artifact give it the highest
    // relevance score for this query.
    await setupSessions123();

    const results = await retrieve("time for statements");

    expect(results.length).toBeGreaterThan(0);
    const procedure = results.find((a) => a.kind === "procedure");
    expect(procedure).toBeDefined();
  });

  it("retrieve('file naming format') surfaces the naming convention", async () => {
    await setupSessions123();

    const results = await retrieve("file naming format financial statements");

    expect(results.length).toBeGreaterThan(0);
    // Convention has tags ["file-naming", "financial-statements", "naming-convention"]
    // which give it the highest score for this query.
    const convention = results.find((a) => a.kind === "convention");
    expect(convention).toBeDefined();
    expect(convention!.content).toMatch(/YYYY-MM/);
  });

  it("retrieve('accounts institutions') surfaces the institution fact", async () => {
    await setupSessions123();

    const results = await retrieve("financial accounts institutions");

    expect(results.length).toBeGreaterThan(0);
    const fact = results.find((a) => a.kind === "fact");
    expect(fact).toBeDefined();
    expect(fact!.content).toContain("Chase");
  });

  it("session 4 execution exchange produces no new artifacts", async () => {
    // Session 4 is a pure application turn — the agent runs the stored
    // procedure but learns nothing new. PIL should extract nothing.
    const llm = await setupSessions123();
    const before = (await loadAll()).filter((a) => !a.retired).length;

    const result = await processMessage(SESSION4_EXCHANGE, llm, "pil-chat:exchange");

    expect(result.created).toHaveLength(0);
    expect(result.updated).toHaveLength(0);

    const after = (await loadAll()).filter((a) => !a.retired).length;
    expect(after).toBe(before);
  });
});

// ---------------------------------------------------------------------------
// Phase 5 — Session 5: artifact revision when circumstances change
// ---------------------------------------------------------------------------

describe("Benchmark finance/Phase 5 — revision (Session 5)", () => {
  it("revises the institution fact when Amex closes and Schwab opens", async () => {
    // Session 5: "I closed my Amex account. I opened a Schwab (brokerage) account."
    // The agent must update the fact artifact. revise() handles minor content
    // changes (Jaccard ≥ 0.5) as in-place updates — the id is preserved,
    // `content` is replaced, and `revisedAt` is set.
    await setupSessions123();

    const all = await loadAll();
    const factArtifact = all.find((a) => a.kind === "fact" && !a.retired)!;
    expect(factArtifact).toBeDefined();
    expect(factArtifact.content).toContain("Amex");

    const updatedContent =
      "User's financial institutions: Chase (checking), Fidelity (brokerage), Schwab (brokerage)";
    const revised = await revise(factArtifact, { content: updatedContent });

    // Revised artifact reflects the change
    expect(revised.content).toContain("Schwab");
    expect(revised.content).not.toContain("Amex");
    expect(revised.revisedAt).toBeDefined();
    expect(revised.retired).toBeFalsy();
  });

  it("store still has exactly three active artifacts after revision (in-place update)", async () => {
    // The Jaccard similarity between old and new institution list content is
    // ≈ 0.64 (7 shared words out of 11 in the union: user, financial,
    // institutions, chase, checking, fidelity, brokerage — Amex/credit/card
    // dropped, schwab added). Since 0.64 > 0.5, revise() updates in place
    // rather than retiring the original and creating a new entry.
    await setupSessions123();

    const all = await loadAll();
    const factArtifact = all.find((a) => a.kind === "fact" && !a.retired)!;
    await revise(factArtifact, {
      content:
        "User's financial institutions: Chase (checking), Fidelity (brokerage), Schwab (brokerage)",
    });

    const afterRevision = await loadAll();
    const active = afterRevision.filter((a) => !a.retired);
    expect(active).toHaveLength(3); // convention + fact (updated) + procedure

    const activeFact = active.find((a) => a.kind === "fact")!;
    expect(activeFact.content).toContain("Schwab");
    expect(activeFact.content).not.toContain("Amex");
  });

  it("retrieve after revision surfaces the updated institution list", async () => {
    await setupSessions123();

    const all = await loadAll();
    const factArtifact = all.find((a) => a.kind === "fact" && !a.retired)!;
    await revise(factArtifact, {
      content:
        "User's financial institutions: Chase (checking), Fidelity (brokerage), Schwab (brokerage)",
    });

    const results = await retrieve("financial accounts institutions");
    const fact = results.find((a) => a.kind === "fact");
    expect(fact).toBeDefined();
    expect(fact!.content).toContain("Schwab");
    expect(fact!.content).not.toContain("Amex");
  });

  it("procedure artifact can also be revised to replace Amex with Schwab", async () => {
    // In the full scenario the agent also updates the procedure (and any
    // generated script). We test revise() on the procedure independently.
    await setupSessions123();

    const all = await loadAll();
    const proc = all.find((a) => a.kind === "procedure" && !a.retired)!;
    expect(proc.content).toContain("Amex");

    // Replace Amex with Schwab throughout the procedure description
    const updatedProcedure = proc.content.replace(/Amex/g, "Schwab");
    const revised = await revise(proc, { content: updatedProcedure });

    expect(revised.content).toContain("Schwab");
    expect(revised.content).not.toContain("Amex");
    expect(revised.revisedAt).toBeDefined();
  });
});
