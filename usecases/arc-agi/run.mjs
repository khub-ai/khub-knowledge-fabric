#!/usr/bin/env node
/**
 * run.mjs — Launcher for the ARC-AGI Python ensemble harness.
 *
 * Usage (from repo root):
 *   npm run arc                                   # first task, no interaction
 *   npm run arc -- --human                        # human-in-the-loop mode
 *   npm run arc -- --human --task-id 1e0a9b12     # specific task
 *   npm run arc -- --limit 5 --charts             # batch run with charts
 *
 * Python resolution order:
 *   1. PYTHON env var  (set this to your venv python if needed)
 *   2. 'python' on PATH
 */

import { spawnSync } from "child_process";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __dir  = dirname(fileURLToPath(import.meta.url));
const harness = join(__dir, "python", "harness.py");
const python  = process.env.PYTHON || "python";
const args    = process.argv.slice(2);   // pass everything after 'node run.mjs'

const result = spawnSync(python, [harness, ...args], {
  stdio: "inherit",
  env: { ...process.env, PYTHONUTF8: "1", PYTHONIOENCODING: "utf-8" },
});

process.exit(result.status ?? 1);
