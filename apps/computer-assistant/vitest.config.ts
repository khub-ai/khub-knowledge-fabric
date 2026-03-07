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
    // Report
    reporter: "verbose",
  },
});
