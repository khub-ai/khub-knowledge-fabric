#!/usr/bin/env node
/**
 * Cross-platform launcher for the Python ARC-AGI ensemble harness.
 * Works from the repo root via npm scripts:
 *   npm run arc:py:human
 *   npm run arc:py -- --task-id 1e0a9b12
 */

import { execFileSync, execSync } from "child_process";
import { existsSync, readFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const harness = resolve(__dirname, "python", "harness.py");

const PYTHON = "C:/Users/kaihu/AppData/Local/pypoetry/Cache/virtualenvs/01os-TLh4bqwo-py3.10/Scripts/python.exe";
const KEY_FILE = "P:/_access/Security/api_keys.env";

// Load API key if not already set
if (!process.env.ANTHROPIC_API_KEY && existsSync(KEY_FILE)) {
  const lines = readFileSync(KEY_FILE, "utf-8").split("\n");
  for (const line of lines) {
    if (line.startsWith("ANTHROPIC_API_KEY=")) {
      process.env.ANTHROPIC_API_KEY = line.slice("ANTHROPIC_API_KEY=".length).trim();
      break;
    }
  }
}

// Ensure Python can output Unicode on Windows
process.env.PYTHONUTF8 = "1";

// Forward all CLI args after the script name
const args = process.argv.slice(2);

try {
  execFileSync(PYTHON, [harness, ...args], {
    stdio: "inherit",
    env: process.env,
  });
} catch (e) {
  process.exit(e.status || 1);
}
