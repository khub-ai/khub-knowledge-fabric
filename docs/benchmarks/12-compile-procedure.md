# Benchmark 12: Compile Procedure — Recipe → Executable Program Escalation

| | |
|---|---|
| **Scenario** | Procedure artifact compiled into a runnable script |
| **What it tests** | `compileToProgram()` code generation · disk persistence · `[EXECUTABLE:]` context hint · in-place re-compile |
| **Automated test** | `apps/computer-assistant/src/tests/bench-compile-procedure.test.ts` |
| **Run with** | `pnpm test` |

---

## Scenario

After a procedure artifact has been learned over several sessions, the agent can escalate it
from a natural-language recipe into a runnable program. This benchmark validates the full
lifecycle described in `docs/example-learning-in-action.md` Session 4:

> *"I've now run this procedure three months in a row and the steps are identical each time.
> Want me to turn it into a script?"*
> → Agent generates a Python script, attaches it to the procedure artifact,
> and on the next relevant request uses `run-command` instead of manual steps.

---

## Why this is interesting

### Problem 1 — Keeping the recipe alongside the program

When a procedure is compiled, the natural-language recipe (`content`) is always retained as
primary documentation and manual fallback. Only the `program` field is added via `revise()`.
The agent should **never replace** the readable recipe with the script — both coexist.

This matters because:
- The script may not run on a new machine without installing dependencies
- The user may want to inspect or modify the steps before automating them
- A future `revise()` of the procedure must still start from the recipe text, not code

### Problem 2 — In-place update, not retire-and-create

`compileToProgram()` calls `revise(procedure, { program: {...} })`. Because only the `program`
field changes (the recipe `content` is identical), Jaccard similarity = 1.0 >> 0.5 threshold.
`revise()` updates the artifact in place: same `id`, `revisedAt` set, store count unchanged.

If this were naïvely treated as a content change, each compile would retire the old artifact
and create a fresh one — accumulating orphan artifacts and breaking retrieval continuity.

### Problem 3 — Context hint for the agent LLM

After compilation, the agent must know a script exists so it can choose `run-command` over
manual steps on subsequent requests. This is done via `formatArtifactsForPrompt()`, which
appends `[EXECUTABLE: /path/to/script]` to any procedure artifact whose `program.path` is set.

The agent system prompt instructs:
> *"When PIL context includes a procedure line ending with `[EXECUTABLE: /path/to/script]`,
> prefer `run-command python /path/to/script` for execution requests."*

### Problem 4 — Re-compiling doesn't create a second program

When the user asks to regenerate the script (e.g., after revising the procedure), calling
`compileToProgram()` again must overwrite the previous `program` block in-place, not produce
a second artifact. The store must remain at the same artifact count after both compiles.

---

## Conversation trace

| Session | Turn | Input | PIL / agent events |
|---|---|---|---|
| 1 | User | Download January Chase statement, rename `2026-01-chase-checking.pdf`, put in Finance/Statements | — |
|   | Agent | Done. Saved to `~/Finance/Statements/2026-01-chase-checking.pdf`. | *(none — one-off task)* |
| 2 | User | Download February Chase statement. Same as last time. | — |
|   | Agent | Renamed to `2026-02-chase-checking.pdf` and saved. Naming pattern: YYYY-MM-institution-account.pdf. Want me to remember? | — |
|   | User | Yes, always use that pattern. | **convention created** |
| 3 | User | Get my February Fidelity statement too. | — |
|   | Agent | Saved. You now have two institutions. Want a monthly checklist? | — |
|   | User | Good idea. Add Amex too — I always forget that one. | **fact created** (Chase, Fidelity, Amex); **procedure created** (monthly download checklist) |
| 4 | User | Create a Python script to automate monthly statement downloads. | — |
|   | Agent | *(action: compile-procedure, target: python)* | **procedure revised in-place**: `program.code`, `program.path`, `program.generatedAt` set |
|   | Agent | Script saved to `~/.openclaw/programs/monthly-statements-download-procedure.py`. | — |
| 5 | User | It's March — time for statements. | PIL context includes `[EXECUTABLE: ~/.openclaw/programs/monthly-statements-download-procedure.py]` |
|   | Agent | Found a script for this — running it now. *(action: run-command python ~/.openclaw/programs/monthly-statements-download-procedure.py)* | *(announce-then-run)* |

---

## What the code does, step by step

### `compileToProgram(procedure, relatedArtifacts, language, llm, saveDir)`

Located in `packages/openclaw-plus/src/pipeline.ts`.

1. **Build prompt** — `buildCodeGenPrompt()` formats the procedure recipe and all supporting
   artifacts (conventions, facts) into a code-generation prompt. The phrase
   `"PROCEDURE TO AUTOMATE"` uniquely identifies this prompt (used by the mock LLM to
   distinguish it from extraction/consolidation prompts).

2. **Call LLM** — the response is raw source code (no JSON wrapper, no markdown fences).

3. **Write to disk** — if `saveDir` is provided, `mkdir -p` the directory and write the code
   to `<saveDir>/<derived-filename>.<ext>`. The filename is derived from the procedure's tags
   (e.g. tags `["monthly-statements", "download-procedure"]` → `monthly-statements-download-procedure.py`).

4. **Revise in place** — `revise(procedure, { program: { language, code, path, generatedAt } })`
   updates only the `program` field; recipe `content` is untouched; artifact `id` is preserved.

### Agent interception (`runAgentTurn` in `agent.ts`)

`compile-procedure` is intercepted in `runAgentTurn` before `executeAction()` because it needs
two things unavailable in the stateless `executeAction` function: the PIL artifact store
(to find the highest-confidence injectable procedure) and the `pilLlm` adapter (for code
generation). `executeAction` contains a safety-net error case that should never be reached.

---

## Test phases

| Phase | Test cases | Key assertions |
|---|---|---|
| **Phase 1** — Code generation and attachment | 5 | `program.language === "python"`; `program.code` matches mock; `program.path` ends `.py`; file exists on disk; recipe `content` unchanged |
| **Phase 2** — Persistence round-trip | 2 | `program.path` and `program.code` survive `loadAll()`; store stays at **3** active artifacts (in-place update) |
| **Phase 3** — EXECUTABLE hint | 2 | `formatArtifactsForPrompt()` includes `[EXECUTABLE: /path]` for compiled procedure; absent for uncompiled procedure |
| **Phase 4** — Re-compile in place | 2 | Second compile: same artifact `id`; store still **3** active artifacts |

**Total: 11 tests.** All run in < 1 s (no API calls; mock LLM).

---

## Running the automated test

No API key is needed. The mock LLM (`createCompileBenchmarkLLM()` in `mock-llm.ts`) handles
all three prompt types: extraction for sessions 1–3, and code-generation for the compile call.

```bash
# From the monorepo root
pnpm test

# Only this benchmark (faster feedback loop)
pnpm --filter @khub-ai/computer-assistant test -- --reporter=verbose bench-compile-procedure

# Watch mode (re-runs on file save)
pnpm --filter @khub-ai/computer-assistant test -- --watch bench-compile-procedure
```

Expected output:
```
✓ Benchmark compile/Phase 1 — code generation and attachment > compileToProgram attaches a program block to the procedure artifact
✓ Benchmark compile/Phase 1 — code generation and attachment > generates code matching the mock LLM response
✓ Benchmark compile/Phase 1 — code generation and attachment > saves the script to disk and sets program.path
✓ Benchmark compile/Phase 1 — code generation and attachment > file on disk contains the generated code
✓ Benchmark compile/Phase 1 — code generation and attachment > procedure content (recipe) is preserved unchanged
✓ Benchmark compile/Phase 2 — persistence > program field is present after loadAll()
✓ Benchmark compile/Phase 2 — persistence > store still has exactly 3 active artifacts after compiling (in-place update)
✓ Benchmark compile/Phase 3 — EXECUTABLE hint in agent context > includes [EXECUTABLE: path] for a procedure with a saved program
✓ Benchmark compile/Phase 3 — EXECUTABLE hint in agent context > does NOT include [EXECUTABLE:] for a procedure without a compiled program
✓ Benchmark compile/Phase 4 — re-compile updates in place > second compile overwrites program.code, not creates a new artifact
✓ Benchmark compile/Phase 4 — re-compile updates in place > store still has exactly 3 active artifacts after re-compiling

Tests  11 passed (11)
```

---

## Manual walkthrough (live LLM)

This walkthrough uses the `computer-assistant` REPL with a real Anthropic API key.
It produces an actual Python file in `~/.openclaw/programs/`.

### Prerequisites

```bash
node --version     # must be ≥ 18
echo $ANTHROPIC_API_KEY  # must be set (sk-ant-...)
```

### Step 1 — Start the agent

```bash
cd C:/_backup/openclaw/openclaw-knowledge-management   # Windows path; adjust for your OS
pnpm --filter @khub-ai/computer-assistant start
```

You will see a `>` prompt.

### Step 2 — Session 1: one-off download (no learning)

Type this at the `>` prompt:

```
Download my January statement from Chase, rename it "2026-01-chase-checking.pdf", and put it in my Finance/Statements folder.
```

Expected: agent responds with a `say` or `run-command` action. **No `[PIL learned:]` line**
should appear — a single one-off task must not create a convention.

### Step 3 — Session 2: confirm the naming pattern

```
Download my February statement from Chase. Same as last time.
```

Agent may notice the pattern. Then:

```
Yes, always use that pattern for all financial statement filenames.
```

Expected: `[PIL learned: 1 new (convention)]` appears.

### Step 4 — Session 3: add institution list and procedure

```
Get my February statement from Fidelity too.
```

```
Good idea. Add Amex too — I always forget that one. Create a monthly checklist.
```

Expected: `[PIL learned: 2 new (fact, procedure)]` — both artifacts created in one turn.

### Step 5 — Verify the store

```
/list
```

Expected: three active artifacts — a `convention`, a `fact`, and a `procedure` (no `[EXECUTABLE:]`
in the context line yet because nothing has been compiled).

### Step 6 — Compile the procedure

```
Create a Python script to automate my monthly statement downloads.
```

Expected output includes:
```
[Action: compile-procedure → python]
Script saved to /home/<user>/.openclaw/programs/monthly-statements-download-procedure.py
```

Verify the file was created:

```bash
# In a separate terminal (not the agent REPL)
cat ~/.openclaw/programs/monthly-statements-download-procedure.py
```

You should see a complete, runnable Python script with `INSTITUTIONS` list, `download_statement()`,
and `main()`.

### Step 7 — Check the EXECUTABLE hint

Back in the agent REPL:

```
/list
```

The procedure line should now end with:
```
[EXECUTABLE: /home/<user>/.openclaw/programs/monthly-statements-download-procedure.py]
```

### Step 8 — Trigger announce-then-run

```
It's March 2nd — time for my monthly statements.
```

Expected:
```
[Action: run-command → python /home/<user>/.openclaw/programs/monthly-statements-download-procedure.py]
Found a script for this — running it now.
```

The script will run (and fail gracefully with `TODO` stubs for the actual site-specific
download logic — credentials and selectors are left for the user to fill in).

### Step 9 — Re-compile after a procedure revision (optional)

Change the institution list (e.g., Amex closed, Schwab opened):

```
I closed my Amex account and opened a Schwab brokerage account.
```

Then ask to update the script:

```
Please update the Python script to reflect the new institution list.
```

Expected: `compileToProgram()` runs again; the store still has **3** active artifacts
(same IDs, `revisedAt` updated on the procedure; new script file on disk).

---

## Does this affect the official OpenClaw plugin?

**No breaking changes.** The OpenClaw plugin surface is limited to `index.ts` and `tools.ts`,
which expose a single `knowledge_search` tool. That tool calls `retrieve()` and returns
`JSON.stringify(artifacts)` — it is entirely unaware of the `program?` field.

The only observable difference in OpenClaw: if a procedure artifact has been compiled,
`knowledge_search` will now return a `program` block in the JSON alongside `content`.
The LLM agent receiving the tool result will see the script path and code, which is
informative rather than harmful. A future improvement could strip `program.code` from
the tool result (keeping `program.path` for reference) to avoid verbosity.

| Component | Changed? | Breaking? |
|---|---|---|
| `packages/openclaw-plus/src/types.ts` | ✅ Yes — `program?` field added | ❌ No — optional field; old artifacts remain valid |
| `packages/openclaw-plus/src/pipeline.ts` | ✅ Yes — `compileToProgram()` added | ❌ No — additive export; existing functions unchanged |
| `packages/openclaw-plus/src/store.ts` | ❌ No changes | — |
| `packages/openclaw-plus/src/tools.ts` | ❌ No changes | — |
| `packages/openclaw-plus/index.ts` | ❌ No changes | — |
| `apps/computer-assistant/src/agent.ts` | ✅ Yes — `compile-procedure` handler, EXECUTABLE hint | ❌ No — app-local changes only; not part of plugin |
| `apps/computer-assistant/src/actions.ts` | ✅ Yes — new ActionKind | ❌ No — app-local |
| `apps/computer-assistant/src/llm.ts` | ✅ Yes — updated system prompt | ❌ No — app-local |

**Store format compatibility:** The JSONL store is append-only and additive. Existing artifact
lines do not have a `program` field; new compiled artifacts do. Both are valid JSON and
loadable by the current `loadAll()` parser (unknown fields are ignored by the type system
at runtime; TypeScript's optional `?` marks the field absent, not invalid).
