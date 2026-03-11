/**
 * FileReporter — a vitest custom reporter that writes a detailed plain-text
 * log file alongside the normal verbose terminal output.
 *
 * Log files are written to apps/computer-assistant/logs/test-YYYYMMDD_HHMMSS.log
 * and are ignored by git (*.log in .gitignore).
 *
 * Usage: listed in vitest.config.ts reporters array alongside "verbose".
 */

import type { Reporter, File, Task } from "vitest";
import { writeFileSync, mkdirSync, existsSync } from "node:fs";
import { join, relative } from "node:path";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeTimestamp(): string {
  // Format: YYYYMMDD_HHMMSS  e.g. 20260311_161204
  const d = new Date();
  const pad = (n: number, w = 2) => String(n).padStart(w, "0");
  return (
    `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}` +
    `_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`
  );
}

/** Remove ANSI colour/style escape codes for clean plain-text output. */
function stripAnsi(s: string): string {
  // eslint-disable-next-line no-control-regex
  return s.replace(/\x1b\[[0-9;]*m/g, "");
}

function fmtDuration(ms: number | undefined): string {
  if (ms === undefined) return "";
  return ms < 1000 ? ` (${Math.round(ms)}ms)` : ` (${(ms / 1000).toFixed(2)}s)`;
}

function stateIcon(state: string | undefined): string {
  switch (state) {
    case "pass": return "✓";
    case "fail": return "✗";
    case "skip":
    case "todo": return "○";
    default:     return "·";
  }
}

// ---------------------------------------------------------------------------
// Reporter
// ---------------------------------------------------------------------------

export default class FileReporter implements Reporter {
  private readonly logFile: string;
  private readonly startedAt: number = Date.now();
  private lines: string[] = [];

  constructor() {
    const dir = join(process.cwd(), "logs");
    if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
    this.logFile = join(dir, `test-${makeTimestamp()}.log`);
  }

  // ── Lifecycle hooks ────────────────────────────────────────────────────────

  onInit(): void {
    const line = "═".repeat(72);
    this.lines.push(line);
    this.lines.push("  PIL Knowledge Fabric — Test Run");
    this.lines.push(line);
    this.lines.push(`  Started:   ${new Date().toISOString()}`);
    this.lines.push(`  Node:      ${process.version}`);
    this.lines.push(`  Platform:  ${process.platform}/${process.arch}`);
    this.lines.push(`  Log file:  ${this.logFile}`);
    this.lines.push(line);
  }

  onCollected(files: File[] = []): void {
    const total = files.reduce((n, f) => n + this.countTests(f.tasks ?? []), 0);
    this.lines.push(`\nDiscovered ${files.length} file(s), ${total} test(s):`);
    for (const f of files) {
      const rel = relative(process.cwd(), f.name);
      const count = this.countTests(f.tasks ?? []);
      this.lines.push(`  ${rel}  [${count} tests]`);
    }
  }

  onFinished(files: File[] = [], errors: unknown[] = []): void {
    let passed = 0;
    let failed = 0;
    let skipped = 0;
    const failures: string[] = [];

    for (const file of files) {
      const rel = relative(process.cwd(), file.name);
      const divider = "─".repeat(72);
      this.lines.push(`\n${divider}`);
      this.lines.push(`File: ${rel}`);
      this.lines.push(divider);

      const counts = this.writeTasks(file.tasks ?? [], 1, failures);
      passed  += counts.passed;
      failed  += counts.failed;
      skipped += counts.skipped;
    }

    const elapsed = ((Date.now() - this.startedAt) / 1000).toFixed(2);
    const border = "═".repeat(72);

    this.lines.push(`\n${border}`);
    this.lines.push(`  Finished:  ${new Date().toISOString()}`);
    this.lines.push(`  Duration:  ${elapsed}s`);
    this.lines.push(
      `  Results:   ${passed} passed | ${failed} failed | ${skipped} skipped`,
    );
    this.lines.push(`  Files:     ${files.length}`);

    if (failures.length > 0) {
      this.lines.push(`\n  Failed tests (${failures.length}):`);
      for (const name of failures) {
        this.lines.push(`    ✗ ${name}`);
      }
    }

    if (errors.length > 0) {
      this.lines.push(`\n  Unhandled errors (${errors.length}):`);
      for (const e of errors) {
        this.lines.push(`    ${stripAnsi(String(e))}`);
      }
    }

    this.lines.push(border);

    // ── Write the file ──────────────────────────────────────────────────────
    try {
      writeFileSync(this.logFile, this.lines.join("\n") + "\n", "utf-8");
    } catch (e) {
      console.error(`[FileReporter] Could not write log: ${e}`);
      return;
    }

    // ── Announce in terminal (appears after the verbose summary) ───────────
    const bar = "━".repeat(60);
    console.log(`\n${bar}`);
    console.log(`Test log: ${this.logFile}`);
    console.log(bar);
  }

  // ── Internal helpers ───────────────────────────────────────────────────────

  /**
   * Recursively write tasks to this.lines and tally results.
   * Suites (describe blocks) are identified by the presence of a `tasks` array.
   */
  private writeTasks(
    tasks: Task[],
    depth: number,
    failures: string[],
  ): { passed: number; failed: number; skipped: number } {
    let passed  = 0;
    let failed  = 0;
    let skipped = 0;
    const indent = "  ".repeat(depth);

    for (const task of tasks) {
      const sub: Task[] | undefined = (task as any).tasks;

      if (Array.isArray(sub)) {
        // Suite / describe block — recurse; don't count the suite itself
        this.lines.push(`${indent}▸ ${task.name}`);
        const c = this.writeTasks(sub, depth + 1, failures);
        passed  += c.passed;
        failed  += c.failed;
        skipped += c.skipped;
      } else {
        // Leaf test
        const state = task.result?.state;
        const d     = fmtDuration(task.result?.duration);
        this.lines.push(`${indent}${stateIcon(state)} ${task.name}${d}`);

        if (state === "pass") {
          passed++;
        } else if (state === "fail") {
          failed++;
          failures.push(task.name);
          for (const err of (task.result?.errors ?? [])) {
            const msg = stripAnsi(err.message ?? "error");
            this.lines.push(`${indent}  ↳ ${msg}`);
            if (err.stack) {
              stripAnsi(err.stack)
                .split("\n")
                .slice(1, 8)                        // first 7 stack frames
                .forEach((l) => this.lines.push(`${indent}    ${l.trim()}`));
            }
          }
        } else if (state === "skip" || state === "todo") {
          skipped++;
        }
      }
    }

    return { passed, failed, skipped };
  }

  /** Count leaf tests recursively (used in onCollected summary). */
  private countTests(tasks: Task[]): number {
    let n = 0;
    for (const task of tasks) {
      const sub: Task[] | undefined = (task as any).tasks;
      if (Array.isArray(sub)) {
        n += this.countTests(sub);
      } else {
        n++;
      }
    }
    return n;
  }
}
