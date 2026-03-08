# 09 — OpenClaw Integration Benchmark

| | |
|---|---|
| **Pipeline stages** | Hook wiring for all stages; `knowledge_search` tool |
| **Modules** | `packages/openclaw-plus/index.ts`, `packages/openclaw-plus/src/tools.ts` |
| **Implementation status** | 🔄 Partial — `knowledge_search` tool wired; hook wiring pending (Milestones 1c/1d) |
| **Automated coverage** | ❌ Not yet |

---

## Purpose

This benchmark verifies that the PIL extension integrates correctly with a running OpenClaw instance: that hooks fire at the right moments, that the extension correctly observes inbound messages for passive learning, that retrieved artifacts are injected into prompts before the model responds, and that the `knowledge_search` tool surfaces relevant knowledge on demand.

The individual stage benchmarks (02–08) verify the pipeline in isolation. This benchmark verifies the pipeline as it operates inside the OpenClaw runtime — including the hook dispatch mechanism, sender verification security guard, and the timing of injection relative to prompt construction.

## Design rationale this benchmark validates

- **Passive learning via hooks**: The `message_received` hook is the mechanism by which PIL learns from conversation without requiring the user to issue explicit "remember this" commands. Once wired, every inbound message passes through `processMessage()` automatically.
- **Injection via `before_prompt_build`**: Retrieved artifacts are injected into the model's context before it generates a response. The timing and format of this injection determine how effectively the model uses the accumulated knowledge.
- **Sender verification**: The `message_received` hook fires on all inbound messages, including those from external parties. Without sender verification, a malicious external message could trigger extraction and persist adversarial content. The security guard must check sender identity before passing messages to `processMessage()`.
- **Explicit retrieval via `knowledge_search` tool**: The agent can query the knowledge store explicitly, independent of the passive injection pipeline. This is the currently-wired capability and must continue to work correctly after hook wiring is added.

---

## Current state

| Capability | Status | Notes |
|---|---|---|
| `knowledge_search` tool registration | ✅ Wired | Registered in `index.ts` via `registerKnowledgeTools(api)` |
| `message_received` hook → `processMessage()` | 🔲 Placeholder | Core logic exists; hook not yet registered |
| Sender verification guard | 🔲 Placeholder | No implementation yet |
| `before_prompt_build` hook → inject staged artifacts | 🔲 Placeholder | Core logic exists; hook not yet registered |
| Session state for staged artifacts | 🔲 Placeholder | No session state mechanism yet |

---

## Test cases

### Group OC-A: `knowledge_search` tool (currently wired)

#### OC-A-1: Tool registered and callable
| Field | Value |
|---|---|
| **ID** | OC-A-1 |
| **Action** | Call `knowledge_search` tool with `{ query: "summary format preference" }` via OpenClaw tool invocation |
| **Pass criterion** | Tool executes without error; returns JSON array of matching artifacts |
| **Automated** | ❌ Requires live OpenClaw instance; no automated test yet |

#### OC-A-2: Tool returns relevant artifacts
| Field | Value |
|---|---|
| **ID** | OC-A-2 |
| **Setup** | Store contains a bullet-point preference artifact |
| **Action** | Call `knowledge_search` with `{ query: "bullet points summaries" }` |
| **Pass criterion** | Returned JSON includes the bullet-point preference artifact |
| **Automated** | ❌ Requires live OpenClaw instance |

#### OC-A-3: Tool returns empty array for irrelevant query
| Field | Value |
|---|---|
| **ID** | OC-A-3 |
| **Setup** | Store contains only a TypeScript preference artifact |
| **Action** | Call `knowledge_search` with `{ query: "financial reporting" }` |
| **Pass criterion** | Returned JSON is `[]` or an array with score=0 results |
| **Automated** | ❌ Requires live OpenClaw instance |

---

### Group OC-B: Passive learning via `message_received` hook
*🔲 Pending Milestone 1c*

#### OC-B-1: Hook fires on inbound user message
| Field | Value |
|---|---|
| **ID** | OC-B-1 |
| **Status** | 🔲 Placeholder |
| **Action** | Send a preference-bearing message in an OpenClaw conversation |
| **Pass criterion** | `message_received` hook fires; `processMessage()` is called; artifact persisted to store |

#### OC-B-2: Hook does not fire on agent's own messages
| Field | Value |
|---|---|
| **ID** | OC-B-2 |
| **Status** | 🔲 Placeholder |
| **Action** | Agent responds; `message_received` hook observed |
| **Pass criterion** | Agent response does not trigger extraction (only inbound user messages do) |

#### OC-B-3: Sender verification blocks external messages
| Field | Value |
|---|---|
| **ID** | OC-B-3 |
| **Status** | 🔲 Placeholder |
| **Action** | An external (non-user) message arrives via OpenClaw's connected platforms |
| **Pass criterion** | Sender verification guard rejects the message before extraction; no artifact persisted |
| **Note** | This is a security-critical test; must pass before Milestone 1c is considered complete |

---

### Group OC-C: Knowledge injection via `before_prompt_build` hook
*🔲 Pending Milestone 1d*

#### OC-C-1: Staged artifacts injected before model call
| Field | Value |
|---|---|
| **ID** | OC-C-1 |
| **Status** | 🔲 Placeholder |
| **Setup** | Store contains a consolidated bullet-point preference; user sends a message requesting a summary |
| **Pass criterion** | Before the model call, the prompt contains the injected preference with `[established]` label |

#### OC-C-2: Auto-apply threshold respected in injection
| Field | Value |
|---|---|
| **ID** | OC-C-2 |
| **Status** | 🔲 Placeholder |
| **Setup** | Artifact at `candidate` stage (confidence 0.65) |
| **Pass criterion** | Artifact injected as `[provisional]` (not `[established]`); `autoApply: false` |

#### OC-C-3: No injection when store is empty
| Field | Value |
|---|---|
| **ID** | OC-C-3 |
| **Status** | 🔲 Placeholder |
| **Pass criterion** | Empty store → no injection block added to prompt; model proceeds normally |

---

## Open questions

1. **Session state mechanism**: Artifacts retrieved by the `message_received` hook need to be staged somewhere between hook execution and the `before_prompt_build` call. OpenClaw's session state API for this has not yet been explored. The implementation of OC-C-1 depends on understanding this mechanism.
2. **Sender identity**: OpenClaw provides identity information with each message (user, agent, external platform). The sender verification guard needs to use this to distinguish user messages from others. The exact identity field names in the plugin SDK must be confirmed before implementation.
3. **Injection format**: The format of the injected block in the prompt (header text, label style, ordering of multiple artifacts) is not yet finalized. The format must be consistent with what the model's system prompt expects.
4. **Token budget**: Injecting many artifacts could consume a significant fraction of the context window. A token budget mechanism (inject only the top N by score, truncate if total tokens exceed threshold) is needed but not yet designed.
5. **Hook maturity**: As of February 2026, not all OpenClaw hooks are fully wired (see `docs/architecture.md#hook-maturity-note`). The integration tests should be written against stable hooks only.

---

## How to run integration tests (once implemented)

```bash
# Start OpenClaw in dev mode with the plugin loaded
cd /path/to/openclaw
ANTHROPIC_API_KEY=sk-ant-... pnpm gateway:watch

# In a separate terminal, run integration tests
cd apps/computer-assistant
pnpm test:integration   # (not yet implemented)
```

Integration tests will require a live OpenClaw instance and an Anthropic API key. They will be significantly slower than unit tests and are not suitable for CI.

---

## Automated evaluation notes

Integration testing against a live runtime is significantly more complex than unit testing. Options:

1. **In-process testing**: Mock the OpenClaw plugin API (`OpenClawPluginApi`) and test hook registration and dispatch without a live runtime. This is the most feasible short-term approach.
2. **Contract testing**: Define the interface the plugin expects from `api` (hook registration, tool registration, session state) and verify the plugin's usage against that interface.
3. **End-to-end testing**: Use OpenClaw's headless mode (if available) to run full integration tests against a real instance. Most reliable but most expensive to maintain.

The `tools.ts` `knowledge_search` tool can be unit-tested by mocking `retrieve()` — this does not require a live OpenClaw instance and should be added as a near-term gap closure.
