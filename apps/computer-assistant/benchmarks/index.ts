/**
 * PIL benchmark suite — measures extraction and retrieval effectiveness.
 *
 * Requires ANTHROPIC_API_KEY (uses real LLM for meaningful results).
 *
 * Usage:
 *   export ANTHROPIC_API_KEY=sk-ant-...
 *   pnpm benchmark
 *
 * Output:
 *   - Extraction precision/recall/F1 by scenario
 *   - Retrieval top-k hit rate
 *   - Overall summary
 */

import { tmpdir } from "node:os";
import { join } from "node:path";
import { unlink } from "node:fs/promises";
import { existsSync } from "node:fs";
import { randomUUID } from "node:crypto";
import { extractFromMessage } from "@khub-ai/openclaw-plus/extract";
import { persist, retrieve } from "@khub-ai/openclaw-plus/store";
import type { KnowledgeArtifact } from "@khub-ai/openclaw-plus/types";
import { EXTRACTION_SCENARIOS, RETRIEVAL_SCENARIOS } from "./scenarios.js";
import { createPilLLM } from "../src/llm.js";

// ---------------------------------------------------------------------------
// Benchmark state
// ---------------------------------------------------------------------------

const BOLD = (s: string) => `\x1b[1m${s}\x1b[0m`;
const GREEN = (s: string) => `\x1b[32m${s}\x1b[0m`;
const RED = (s: string) => `\x1b[31m${s}\x1b[0m`;
const YELLOW = (s: string) => `\x1b[33m${s}\x1b[0m`;
const DIM = (s: string) => `\x1b[2m${s}\x1b[0m`;

function fmt(n: number): string {
  return (n * 100).toFixed(1) + "%";
}

// ---------------------------------------------------------------------------
// Extraction benchmark
// ---------------------------------------------------------------------------

type ExtractionResult = {
  id: string;
  description: string;
  input: string;
  extracted: boolean;
  extractedKind?: string;
  extractedCertainty?: string;
  extractedTags?: string[];
  // Evaluation
  shouldExtract: boolean;
  correctExtraction: boolean;   // true/false positive matches shouldExtract
  correctKind: boolean;
  correctCertainty: boolean;
  correctTagsAny: boolean;
  pass: boolean;
};

async function runExtractionBenchmark(llm: ReturnType<typeof createPilLLM>): Promise<void> {
  console.log(BOLD("\n═══ Extraction Benchmark ═══\n"));

  const results: ExtractionResult[] = [];

  for (const scenario of EXTRACTION_SCENARIOS) {
    process.stdout.write(`  [${scenario.id}] ${scenario.description.padEnd(50)} `);

    const candidates = await extractFromMessage(scenario.input, [], llm);
    const extracted = candidates.length > 0;
    const candidate = candidates[0];

    const correctExtraction = extracted === scenario.shouldExtract;

    // Kind check
    const correctKind = !scenario.expectedKind ||
      (candidate !== undefined && candidate.kind === scenario.expectedKind);

    // Certainty check
    const correctCertainty = !scenario.expectedCertainty ||
      (candidate !== undefined && candidate.certainty === scenario.expectedCertainty);

    // Tag check (at least one of expectedTagsAny must appear)
    let correctTagsAny = true;
    if (scenario.expectedTagsAny && scenario.expectedTagsAny.length > 0) {
      const tags = candidate?.tags ?? [];
      correctTagsAny =
        candidate !== undefined &&
        scenario.expectedTagsAny.some((expected) =>
          tags.some((t) => t.includes(expected) || expected.includes(t)),
        );
    }

    // All expectedTagsAll must appear
    let correctTagsAll = true;
    if (scenario.expectedTagsAll && scenario.expectedTagsAll.length > 0) {
      const tags = candidate?.tags ?? [];
      correctTagsAll = scenario.expectedTagsAll.every((expected) =>
        tags.some((t) => t.includes(expected) || expected.includes(t)),
      );
    }

    const pass =
      correctExtraction &&
      correctKind &&
      correctCertainty &&
      correctTagsAny &&
      correctTagsAll;

    results.push({
      id: scenario.id,
      description: scenario.description,
      input: scenario.input,
      extracted,
      extractedKind: candidate?.kind,
      extractedCertainty: candidate?.certainty,
      extractedTags: candidate?.tags,
      shouldExtract: scenario.shouldExtract,
      correctExtraction,
      correctKind,
      correctCertainty,
      correctTagsAny,
      pass,
    });

    if (pass) {
      console.log(GREEN("✓ PASS"));
    } else {
      console.log(RED("✗ FAIL"));
      const failures: string[] = [];
      if (!correctExtraction)
        failures.push(`extraction=${extracted} (expected ${scenario.shouldExtract})`);
      if (!correctKind)
        failures.push(`kind=${candidate?.kind} (expected ${scenario.expectedKind})`);
      if (!correctCertainty)
        failures.push(`certainty=${candidate?.certainty} (expected ${scenario.expectedCertainty})`);
      if (!correctTagsAny)
        failures.push(`tags=${JSON.stringify(candidate?.tags)} (none of ${JSON.stringify(scenario.expectedTagsAny)})`);
      console.log(DIM(`         └─ ${failures.join(" | ")}`));
    }
  }

  // Summary metrics
  const total = results.length;
  const passed = results.filter((r) => r.pass).length;
  const failed = total - passed;

  // Precision / recall on extraction detection
  const tp = results.filter((r) => r.shouldExtract && r.extracted).length;
  const fp = results.filter((r) => !r.shouldExtract && r.extracted).length;
  const fn = results.filter((r) => r.shouldExtract && !r.extracted).length;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

  console.log(BOLD("\n  Extraction Benchmark Results:"));
  console.log(`  Total scenarios: ${total}`);
  console.log(`  Passed: ${GREEN(String(passed))}   Failed: ${RED(String(failed))}`);
  console.log(`  Pass rate: ${fmt(passed / total)}`);
  console.log(`\n  Extraction detection (should/should-not extract):`);
  console.log(`    Precision: ${fmt(precision)}`);
  console.log(`    Recall:    ${fmt(recall)}`);
  console.log(`    F1:        ${fmt(f1)}`);
}

// ---------------------------------------------------------------------------
// Retrieval benchmark
// ---------------------------------------------------------------------------

async function runRetrievalBenchmark(): Promise<void> {
  console.log(BOLD("\n═══ Retrieval Benchmark ═══\n"));

  let totalScenarios = 0;
  let passedScenarios = 0;

  for (const scenario of RETRIEVAL_SCENARIOS) {
    // Isolated store per scenario
    const storeFile = join(tmpdir(), `pil-bench-retr-${randomUUID()}.jsonl`);
    process.env["KNOWLEDGE_STORE_PATH"] = storeFile;

    try {
      // Seed the store
      for (let i = 0; i < scenario.seedArtifacts.length; i++) {
        const seed = scenario.seedArtifacts[i]!;
        const artifact: KnowledgeArtifact = {
          id: `bench-seed-${i}`,
          kind: seed.kind as KnowledgeArtifact["kind"],
          content: seed.content,
          confidence: seed.confidence,
          provenance: "benchmark",
          createdAt: new Date().toISOString(),
          tags: seed.tags,
          stage: "consolidated",
        };
        await persist(artifact);
      }

      // Run retrieval
      const results = await retrieve(scenario.query, scenario.k);
      const resultIds = new Set(results.map((a) => {
        // Map back to seed index by matching content
        const idx = scenario.seedArtifacts.findIndex(
          (s) => a.content.includes(s.content.slice(0, 30)),
        );
        return idx;
      }));

      const hitCount = scenario.expectedTopK.filter((i) => resultIds.has(i)).length;
      const hitRate = hitCount / scenario.expectedTopK.length;
      const pass = hitRate === 1.0;

      totalScenarios++;
      if (pass) passedScenarios++;

      process.stdout.write(`  [${scenario.id}] ${scenario.description.padEnd(50)} `);
      console.log(pass ? GREEN("✓ PASS") : YELLOW(`~ ${fmt(hitRate)} hit rate`));
    } finally {
      delete process.env["KNOWLEDGE_STORE_PATH"];
      if (existsSync(storeFile)) await unlink(storeFile);
    }
  }

  console.log(BOLD("\n  Retrieval Benchmark Results:"));
  console.log(`  Scenarios: ${totalScenarios}   Passed: ${GREEN(String(passedScenarios))}`);
  console.log(`  Hit rate: ${fmt(passedScenarios / Math.max(1, totalScenarios))}`);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  console.log(BOLD("PIL Benchmark Suite"));
  console.log(`Using model: claude-3-5-haiku-20241022`);
  console.log(`Scenarios: ${EXTRACTION_SCENARIOS.length} extraction, ${RETRIEVAL_SCENARIOS.length} retrieval`);

  if (!process.env["ANTHROPIC_API_KEY"]) {
    console.error(RED("\nError: ANTHROPIC_API_KEY is not set."));
    console.error("Set it with: export ANTHROPIC_API_KEY=sk-ant-...\n");
    process.exit(1);
  }

  const llm = createPilLLM();

  const start = Date.now();

  await runExtractionBenchmark(llm);
  await runRetrievalBenchmark();

  const elapsed = ((Date.now() - start) / 1000).toFixed(1);
  console.log(BOLD(`\n═══ Benchmark complete in ${elapsed}s ═══\n`));
}

main().catch((err) => {
  console.error("Benchmark error:", err);
  process.exit(1);
});
