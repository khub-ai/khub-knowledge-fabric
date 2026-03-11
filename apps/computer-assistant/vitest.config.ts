import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // Run tests sequentially to avoid store file conflicts
    pool: "forks",
    poolOptions: {
      forks: {
        singleFork: true,
      },
    },
    // Timeout per test (LLM calls in integration tests can be slow)
    testTimeout: 30_000,
    // Reporters:
    //   "verbose"              — coloured pass/fail tree in the terminal
    //   "./src/test-file-reporter" — detailed plain-text log in logs/test-<timestamp>.log
    reporters: ["verbose", "./src/test-file-reporter"],
  },
});
