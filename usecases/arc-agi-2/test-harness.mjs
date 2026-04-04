#!/usr/bin/env node
// ---------------------------------------------------------------------------
// ARC-AGI Ensemble Test Harness
//
// Loads ARC-AGI-2 tasks, sends them to the /api/ensemble endpoint,
// compares results to known solutions, and reports accuracy.
//
// Usage:
//   node test-harness.mjs [--limit 1] [--offset 0] [--server http://localhost:4800]
//                         [--data-dir /path/to/training] [--output results.json]
//                         [--task-id <specific-task-id>]
// ---------------------------------------------------------------------------
import { readFileSync, writeFileSync } from 'fs';
import { resolve } from 'path';

function getArg(name, fallback) {
  const idx = process.argv.indexOf(`--${name}`);
  return idx >= 0 && process.argv[idx + 1] ? process.argv[idx + 1] : fallback;
}

const SERVER   = getArg('server', 'http://localhost:4800');
const DATA_DIR = getArg('data-dir', 'C:/_backup/arctest2025/data/training');
const LIMIT    = parseInt(getArg('limit', '1'), 10);
const OFFSET   = parseInt(getArg('offset', '0'), 10);
const OUTPUT   = getArg('output', 'results.json');
const TASK_ID  = getArg('task-id', '');

// Load data
const challenges = JSON.parse(readFileSync(resolve(DATA_DIR, 'arc-agi_training_challenges.json'), 'utf8'));
const solutions  = JSON.parse(readFileSync(resolve(DATA_DIR, 'arc-agi_training_solutions.json'), 'utf8'));

// Select tasks
let taskIds = TASK_ID ? [TASK_ID] : Object.keys(challenges).slice(OFFSET, OFFSET + LIMIT);
console.log(`\n=== ARC-AGI Ensemble Test Harness ===`);
console.log(`Server:  ${SERVER}`);
console.log(`Tasks:   ${taskIds.length} (offset=${OFFSET}, limit=${LIMIT})`);
console.log(`Output:  ${OUTPUT}\n`);

function gridsEqual(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (!b[i] || a[i].length !== b[i].length) return false;
    for (let j = 0; j < a[i].length; j++) {
      if (a[i][j] !== b[i][j]) return false;
    }
  }
  return true;
}

function gridToString(grid) {
  if (!grid) return '(null)';
  return grid.map(row => row.map(c => c.toString()).join(' ')).join('\n');
}

async function runTask(taskId) {
  const task = challenges[taskId];
  const solution = solutions[taskId]?.[0]; // first test case solution

  console.log(`─── Task: ${taskId} ───`);
  console.log(`  Train pairs: ${task.train.length}`);
  console.log(`  Test input:  ${task.test[0].input.length}×${task.test[0].input[0].length}`);
  console.log(`  Expected:    ${solution.length}×${solution[0].length}`);

  const startMs = Date.now();
  let result;
  try {
    const resp = await fetch(`${SERVER}/api/ensemble`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task })
    });
    result = await resp.json();
  } catch (err) {
    console.log(`  ERROR: ${err.message}\n`);
    return { taskId, correct: false, error: err.message, durationMs: Date.now() - startMs };
  }

  const durationMs = Date.now() - startMs;
  const correct = gridsEqual(result.answer, solution);

  console.log(`  Result:      ${correct ? '✓ CORRECT' : '✗ WRONG'}`);
  console.log(`  Converged:   ${result.metadata?.converged ?? false}`);
  console.log(`  Rounds:      ${result.metadata?.rounds ?? '?'}`);
  console.log(`  Duration:    ${(durationMs / 1000).toFixed(1)}s`);

  if (!correct) {
    console.log(`  Expected output:`);
    console.log(gridToString(solution).split('\n').map(l => '    ' + l).join('\n'));
    console.log(`  Got:`);
    console.log(gridToString(result.answer).split('\n').map(l => '    ' + l).join('\n'));
  }

  // Show debate summary
  if (result.debate) {
    console.log(`  Debate (${result.debate.length} entries):`);
    for (const entry of result.debate) {
      const gridInfo = extractGridSummary(entry.content);
      console.log(`    R${entry.round} ${entry.agent}: ${gridInfo}`);
    }
  }
  console.log('');

  return {
    taskId,
    correct,
    converged: result.metadata?.converged ?? false,
    rounds: result.metadata?.rounds ?? 0,
    durationMs,
    debate: result.debate?.map(e => ({ round: e.round, agent: e.agent, length: e.content.length }))
  };
}

function extractGridSummary(content) {
  if (!content) return '(empty)';
  const first100 = content.slice(0, 120).replace(/\n/g, ' ');
  return first100 + (content.length > 120 ? '...' : '');
}

async function main() {
  const results = [];

  for (const taskId of taskIds) {
    const r = await runTask(taskId);
    results.push(r);
  }

  // Summary
  const correct = results.filter(r => r.correct).length;
  const total   = results.length;
  const avgMs   = results.reduce((s, r) => s + (r.durationMs || 0), 0) / total;
  const convRate = results.filter(r => r.converged).length / total;

  console.log('═══════════════════════════════════');
  console.log(`Results: ${correct}/${total} correct (${(100 * correct / total).toFixed(1)}%)`);
  console.log(`Avg duration: ${(avgMs / 1000).toFixed(1)}s`);
  console.log(`Convergence rate: ${(100 * convRate).toFixed(1)}%`);
  console.log('═══════════════════════════════════\n');

  writeFileSync(OUTPUT, JSON.stringify({ summary: { correct, total, accuracy: correct / total, avgMs, convRate }, tasks: results }, null, 2));
  console.log(`Full results written to ${OUTPUT}`);
}

main().catch(err => { console.error(err); process.exit(1); });
