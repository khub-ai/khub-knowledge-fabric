/**
 * End-to-end scenario tests for the computer assistant.
 *
 * Tests the full agent flow without executing real OS actions (dry-run mode).
 * Uses mock LLMs — no API calls required.
 *
 * These tests verify that PIL learning across sessions works correctly
 * for the core "open X" learning scenario.
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { unlink } from "node:fs/promises";
import { existsSync } from "node:fs";
import { parseAgentResponse, classifyTarget } from "../actions.js";
import { processMessage } from "@khub-ai/openclaw-plus/pipeline";
import { retrieve } from "@khub-ai/openclaw-plus/store";
import {
  createPatternMockLLM,
  ALIAS_FACT_RESPONSE,
  EMPTY_RESPONSE,
} from "./mock-llm.js";
import type { LLMFn } from "@khub-ai/openclaw-plus/types";

// ---------------------------------------------------------------------------
// Test isolation
// ---------------------------------------------------------------------------

let testStorePath: string;

beforeEach(() => {
  testStorePath = join(tmpdir(), `pil-scenario-test-${Date.now()}-${Math.random().toString(36).slice(2)}.jsonl`);
  process.env["KNOWLEDGE_STORE_PATH"] = testStorePath;
});

afterEach(async () => {
  delete process.env["KNOWLEDGE_STORE_PATH"];
  if (existsSync(testStorePath)) {
    await unlink(testStorePath);
  }
});

// ---------------------------------------------------------------------------
// parseAgentResponse
// ---------------------------------------------------------------------------

describe("parseAgentResponse", () => {
  it("parses open-file action correctly", () => {
    const raw = JSON.stringify({
      action: "open-file",
      target: "README.md",
      message: "Opening README.md with default editor",
    });

    const action = parseAgentResponse(raw);
    expect(action.kind).toBe("open-file");
    expect(action.target).toBe("README.md");
    expect(action.message).toContain("README.md");
  });

  it("parses open-url action correctly", () => {
    const raw = JSON.stringify({
      action: "open-url",
      target: "https://github.com",
      message: "Opening GitHub in browser",
    });

    const action = parseAgentResponse(raw);
    expect(action.kind).toBe("open-url");
    expect(action.target).toBe("https://github.com");
  });

  it("parses open-folder action correctly", () => {
    const raw = JSON.stringify({
      action: "open-folder",
      target: "~/Downloads",
      message: "Opening Downloads folder",
    });

    const action = parseAgentResponse(raw);
    expect(action.kind).toBe("open-folder");
  });

  it("parses run-command action correctly", () => {
    const raw = JSON.stringify({
      action: "run-command",
      target: "ls -la",
      message: "Listing directory contents",
    });

    const action = parseAgentResponse(raw);
    expect(action.kind).toBe("run-command");
    expect(action.target).toBe("ls -la");
  });

  it("falls back to say action on invalid JSON", () => {
    const action = parseAgentResponse("this is not JSON");
    expect(action.kind).toBe("say");
  });

  it("falls back to say action for unknown action kind", () => {
    const raw = JSON.stringify({
      action: "fly-to-moon",
      target: "moon",
      message: "Flying to moon",
    });

    const action = parseAgentResponse(raw);
    expect(action.kind).toBe("say");
  });

  it("handles markdown code-fenced JSON", () => {
    const raw = "```json\n" + JSON.stringify({
      action: "open-url",
      target: "https://example.com",
      message: "Opening URL",
    }) + "\n```";

    const action = parseAgentResponse(raw);
    expect(action.kind).toBe("open-url");
  });
});

// ---------------------------------------------------------------------------
// classifyTarget
// ---------------------------------------------------------------------------

describe("classifyTarget", () => {
  it("classifies https URLs as url", () => {
    expect(classifyTarget("https://github.com")).toBe("url");
  });

  it("classifies http URLs as url", () => {
    expect(classifyTarget("http://localhost:3000")).toBe("url");
  });

  it("classifies paths with file extensions as file (when not found)", () => {
    // Non-existent file with extension — heuristic fallback
    expect(classifyTarget("nonexistent-file-xyz.md")).toBe("file");
  });

  it("classifies paths without extensions as unknown (when not found)", () => {
    expect(classifyTarget("nonexistent-folder-xyz")).toBe("unknown");
  });
});

// ---------------------------------------------------------------------------
// PIL learning scenario: alias teaching
// ---------------------------------------------------------------------------

describe("PIL learning: alias definition", () => {
  it("stores an alias fact when user teaches it", async () => {
    const llm = createPatternMockLLM([
      { match: "gh", response: ALIAS_FACT_RESPONSE },
    ]);

    const result = await processMessage(
      "when I say 'gh', I mean https://github.com",
      llm,
      "test/session",
    );

    expect(result.created).toHaveLength(1);
    const artifact = result.created[0]!;
    expect(artifact.kind).toBe("convention");
    expect(artifact.tags).toContain("alias");
  });

  it("retrieves the alias fact on a relevant query", async () => {
    // Teach
    const learnLlm = createPatternMockLLM([
      { match: "gh", response: ALIAS_FACT_RESPONSE },
    ]);
    await processMessage("when I say 'gh', I mean https://github.com", learnLlm);

    // Retrieve
    const results = await retrieve("open gh");
    expect(results.length).toBeGreaterThan(0);

    const aliasArtifact = results.find(
      (a) => a.tags?.includes("alias") || a.content.includes("github"),
    );
    expect(aliasArtifact).toBeTruthy();
  });
});

// ---------------------------------------------------------------------------
// PIL learning scenario: progressive open-file pattern
// ---------------------------------------------------------------------------

describe("PIL learning: open-file pattern accumulation", () => {
  it("accumulates evidence across multiple 'open file' messages", async () => {
    // Simulate the agent learning "when user says open X.md, it means open-file X.md"
    const openFileFact = JSON.stringify({
      candidates: [
        {
          content: "When user says 'open X.md', open X.md as a file with default editor",
          kind: "convention",
          scope: "general",
          certainty: "definitive",
          tags: ["file-handling", "open-command", "markdown-files"],
          rationale: "User consistently opens markdown files via text editor",
        },
      ],
    });

    const llm = createPatternMockLLM([
      { match: "open README", response: openFileFact },
      { match: "open CHANGELOG", response: openFileFact },
      { match: "open NOTES", response: openFileFact },
    ]);

    await processMessage("open README.md", llm);
    await processMessage("open CHANGELOG.md", llm);
    const result3 = await processMessage("open NOTES.md", llm);

    // Third message may trigger consolidation
    const allArtifacts = result3.created.length > 0
      ? result3.created
      : result3.updated;

    expect(allArtifacts.length).toBeGreaterThan(0);

    // The store should have artifacts about file opening
    const retrieved = await retrieve("open file");
    expect(retrieved.some((a) => a.tags?.includes("file-handling"))).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Cross-session persistence simulation
// ---------------------------------------------------------------------------

describe("cross-session persistence", () => {
  it("persists knowledge from one session and retrieves it in another", async () => {
    // Session 1: teach the alias
    const session1Llm = createPatternMockLLM([
      { match: "gh", response: ALIAS_FACT_RESPONSE },
    ]);
    await processMessage("gh means https://github.com", session1Llm, "session-1");

    // Session 2 (same store): retrieve the alias
    // (In a real scenario, a new process would start with the same store path)
    const session2Results = await retrieve("open gh");
    expect(session2Results.some((a) => a.tags?.includes("alias"))).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// PIL pipeline: no learning from non-knowledge messages
// ---------------------------------------------------------------------------

describe("PIL pipeline: no spurious learning", () => {
  it("does not create artifacts for simple questions", async () => {
    const llm = createPatternMockLLM([
      { match: "What time is it", response: EMPTY_RESPONSE },
      { match: "open the file", response: EMPTY_RESPONSE },
      { match: "ls", response: EMPTY_RESPONSE },
    ]);

    await processMessage("What time is it?", llm);
    await processMessage("open the file README.md", llm);
    await processMessage("ls", llm);

    const { loadAll } = await import("@khub-ai/openclaw-plus/store");
    const all = await loadAll();
    expect(all).toHaveLength(0);
  });
});
