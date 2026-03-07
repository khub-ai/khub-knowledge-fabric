/**
 * Playground — manual harness for exercising the full PIL pipeline
 * without needing a live OpenClaw runtime.
 *
 * Run:   pnpm start          (single run)
 *        pnpm dev            (re-run on file changes)
 *
 * Requires: ANTHROPIC_API_KEY environment variable
 * Storage:  ~/.openclaw/knowledge/artifacts.jsonl
 *           (override via KNOWLEDGE_STORE_PATH env var)
 *
 * This playground exercises the LLM-backed processMessage() pipeline
 * introduced in Milestones 1b/1c/1d. For a full interactive demo with
 * OS actions, see apps/computer-assistant/.
 */

import Anthropic from "@anthropic-ai/sdk";
import {
  processMessage,
  formatForInjection,
} from "@khub-ai/openclaw-plus/pipeline";
import { retrieve, apply, revise, loadAll } from "@khub-ai/openclaw-plus/store";
import type { LLMFn } from "@khub-ai/openclaw-plus/types";

// ---------------------------------------------------------------------------
// LLM adapter (Anthropic)
// ---------------------------------------------------------------------------

const apiKey = process.env["ANTHROPIC_API_KEY"];
if (!apiKey) {
  console.error(
    "Error: ANTHROPIC_API_KEY is not set.\n" +
      "Set it with: export ANTHROPIC_API_KEY=sk-ant-...\n",
  );
  process.exit(1);
}

const client = new Anthropic({ apiKey });

const llm: LLMFn = async (prompt: string): Promise<string> => {
  const message = await client.messages.create({
    model: "claude-3-5-haiku-20241022",
    max_tokens: 1024,
    messages: [{ role: "user", content: prompt }],
  });
  const block = message.content[0];
  return block?.type === "text" ? block.text : "";
};

// ---------------------------------------------------------------------------
// Sample inputs that exercise different knowledge types
// ---------------------------------------------------------------------------

const SAMPLE_INPUTS = [
  // Definitive preference — will be injectable immediately as [provisional]
  "I always want bullet-point summaries, no more than five points.",
  // Convention / alias
  "When writing code, I prefer TypeScript with strict mode enabled.",
  // Second TypeScript message — same kind + tags → evidence accumulates
  "Use async/await rather than raw promises wherever possible.",
  // Non-knowledge — should produce no artifacts
  "What time is it?",
];

const hr = (label = "") =>
  console.log("\n" + "─".repeat(60) + (label ? `  ${label}` : ""));

console.log("=== PIL Playground (LLM-backed pipeline) ===\n");

// ---------------------------------------------------------------------------
// Process each sample input through the full pipeline
// ---------------------------------------------------------------------------

for (const input of SAMPLE_INPUTS) {
  hr();
  console.log(`Input: "${input}"\n`);

  const result = await processMessage(input, llm, "playground/sample");

  console.log(`  Candidates extracted: ${result.candidates.length}`);
  result.candidates.forEach((c) => {
    console.log(
      `    [${c.kind}/${c.certainty}] tags: ${c.tags.join(", ")}\n    "${c.content}"`,
    );
  });

  console.log(`  Created: ${result.created.length}   Updated: ${result.updated.length}`);
  result.created.forEach((a) => {
    console.log(`    → [NEW / ${a.stage}] conf=${a.confidence.toFixed(2)}  "${a.content}"`);
  });
  result.updated.forEach((a) => {
    console.log(`    → [UPD / ${a.stage}] evCount=${a.evidenceCount}  conf=${a.confidence.toFixed(2)}`);
  });

  if (result.injectable.length > 0) {
    console.log("\n  Injectable artifacts from this turn:");
    console.log(
      formatForInjection(result.injectable)
        .split("\n")
        .map((l) => "    " + l)
        .join("\n"),
    );
  }
}

// ---------------------------------------------------------------------------
// Stage 6: Retrieve
// ---------------------------------------------------------------------------

hr("Stage 6 — Retrieve");
const query = "summary format preference";
const results = await retrieve(query, 5);
console.log(`Query: "${query}"  →  ${results.length} match(es):\n`);
results.forEach((r) => {
  const label = r.stage ?? "?";
  console.log(`  [${r.kind}/${label}] conf=${r.confidence.toFixed(2)}  "${r.content}"`);
});

// ---------------------------------------------------------------------------
// Stage 7: Apply
// ---------------------------------------------------------------------------

hr("Stage 7 — Apply");
for (const result of results.slice(0, 2)) {
  const { suggestion, autoApply } = await apply(result, "summarise this document for me");
  if (suggestion) {
    console.log(`  autoApply=${autoApply}  →  ${suggestion}`);
  }
}

// ---------------------------------------------------------------------------
// Stage 8: Revise
// ---------------------------------------------------------------------------

hr("Stage 8 — Revise");
const all = await loadAll();
const first = all.find((a) => !a.retired);
if (first) {
  const revised = await revise(first, {
    content: first.content + " (revised in playground)",
    confidence: Math.min(1, first.confidence + 0.05),
  });
  console.log(`  Revised: id=${revised.id.slice(0, 8)}…  conf=${revised.confidence}`);
  console.log(`  Content: "${revised.content}"`);
}

// ---------------------------------------------------------------------------
// Store summary
// ---------------------------------------------------------------------------

hr("Store summary");
const finalAll = await loadAll();
const active = finalAll.filter((a) => !a.retired);
console.log(`  Active artifacts: ${active.length}  (total incl. retired: ${finalAll.length})`);
active.forEach((a) => {
  console.log(
    `  [${a.kind}/${a.stage ?? "?"}] conf=${a.confidence.toFixed(2)}  "${a.content.slice(0, 70)}${a.content.length > 70 ? "…" : ""}"`,
  );
});

hr();
console.log("Done.");
