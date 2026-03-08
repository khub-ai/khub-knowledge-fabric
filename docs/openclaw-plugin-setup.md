# Running PIL inside OpenClaw

This guide documents how to install `packages/openclaw-plus` as a live
plugin inside OpenClaw, giving the agent access to persisted knowledge
artifacts through the `knowledge_search` tool.

---

## Target environment

This guide is written for **globally installed OpenClaw** — the common
end-user setup where OpenClaw is installed with:

```bash
npm install -g openclaw@latest
# or
npm install -g openclaw@2026.3.2
```

and the `openclaw` command is available in `$PATH`.

> **Cloned GitHub repo users:** if you run OpenClaw from a local clone of
> the `openclaw/openclaw` repository rather than from a global npm
> install, replace every `openclaw <command>` below with
> `node openclaw.mjs <command>` (run from the root of the cloned repo).
> Everything else — config file location, plugin discovery, `allow` list
> behaviour — is identical.

---

## What works right now

| Capability | Status |
|---|---|
| `knowledge_search` tool — agent queries the artifact store on demand | ✅ Wired |
| Pre-populating the store via the standalone playground | ✅ Works |
| Passive elicitation — agent learns automatically from every message | 🚧 Planned (Milestone 1c) |
| Automatic injection into every prompt | 🚧 Planned (Milestone 1d) |

The agent can **retrieve and apply** what it already knows. It does not
yet **learn passively** from OpenClaw conversations — you teach it
explicitly via the standalone playground or computer-assistant demo, and
the agent draws on that accumulated knowledge inside OpenClaw.

---

## Prerequisites

| Requirement | Check |
|---|---|
| OpenClaw ≥ 2026.1.26 (globally installed) | `openclaw --version` |
| Node.js ≥ 18 | `node --version` |
| OpenClaw gateway running | `openclaw status` |
| Repo cloned and dependencies installed | `cd ~/src/openclaw-knowledge-management && pnpm install` |

---

## Step 1 — Fix WSL/NTFS permissions (WSL2 + Windows users only)

> Skip this step if the plugin directory is on a native Linux filesystem
> (e.g. `/home/...`). If it is under `/mnt/c/` or any other NTFS mount,
> this step is required.

NTFS volumes mounted in WSL2 appear with `mode=777` (world-writable).
OpenClaw's security policy refuses to load plugins from world-writable
paths. Attempting to start the gateway with such a path produces:

```
blocked plugin candidate: world-writable path (/mnt/c/..., mode=777)
```

**Fix:** add metadata-aware mount options to `/etc/wsl.conf`:

```ini
[automount]
options = "metadata,umask=22,fmask=11"
```

Then restart WSL from a Windows PowerShell or CMD window:

```powershell
wsl --shutdown
```

Re-open your WSL terminal. NTFS paths will now appear as `755` and
OpenClaw will accept the plugin directory.

---

## Step 2 — Add the plugin to `load.paths`

OpenClaw's config file lives at `~/.openclaw/openclaw.json` (JSON5
format). Edit it to add the plugin directory under `plugins.load.paths`.

> **Use the absolute path.** JSON does not expand `~`, so write the full
> path, e.g. `/home/kaihu/src/...`.

```json5
{
  // ... rest of your existing config ...
  plugins: {
    load: {
      paths: ["/home/kaihu/src/openclaw-knowledge-management/packages/openclaw-plus"]
    }
    // leave your existing allow / entries / etc. untouched for now
  }
}
```

Restart the gateway:

```bash
openclaw gateway restart
```

The plugin is now **discovered** but will appear as **disabled** in
`openclaw plugins list`. This is expected — continue to Step 3.

---

## Step 3 — Add the plugin to the `allow` list

> **Skip this step if your config has no `allow` field.** The `allow`
> list is an optional strict allowlist. If it is absent, all discovered
> plugins are permitted to be enabled.

If your config already contains an `allow` array (e.g.
`"allow": ["discord", "line"]`), you must add `openclaw-plus` to it
**after** the gateway has discovered the plugin (i.e. after Step 2).

Adding it before discovery causes a validation error —
`plugins.allow: plugin not found: openclaw-plus` — because OpenClaw
verifies every ID in the `allow` list against the set of currently
known plugins.

```json5
allow: ["discord", "line", "openclaw-plus"]
```

---

## Step 4 — Enable the plugin

```bash
openclaw plugins enable openclaw-plus
```

Alternatively, add it to `entries` in the config:

```json5
entries: {
  "openclaw-plus": {
    enabled: true
  }
}
```

Then restart the gateway:

```bash
openclaw gateway restart
```

---

## Step 5 — Verify it loaded

```bash
openclaw plugins list
# look for: openclaw-plus | enabled

openclaw plugins info openclaw-plus
# should list the knowledge_search tool
```

---

## Step 6 — Populate the knowledge store

The artifact store is a JSONL file at:

```
~/.openclaw/knowledge/artifacts.jsonl
```

The playground, computer-assistant demo, and OpenClaw plugin all read
from and write to this same file by default.

### Option A — Run the playground (quickest seed)

The playground processes a hardcoded sample dialogue end-to-end and
writes whatever the LLM extracts to the artifact store:

```bash
cd ~/src/openclaw-knowledge-management
export ANTHROPIC_API_KEY=sk-ant-...     # if not already in environment
pnpm start
```

Confirm artifacts were written:

```bash
cat ~/.openclaw/knowledge/artifacts.jsonl
```

### Option B — Interactive learning via computer-assistant

The computer-assistant is a REPL that learns from your actual
instructions, giving you a more realistic view of what the agent will
accumulate over time:

```bash
cd ~/src/openclaw-knowledge-management
pnpm --filter @khub-ai/computer-assistant start
```

Teach it something specific (e.g. "always format dates as YYYY-MM-DD")
and the artifact will be retrievable in OpenClaw immediately.

### Option C — Write an artifact directly

For a quick smoke-test without running any pipeline:

```bash
cat >> ~/.openclaw/knowledge/artifacts.jsonl << 'EOF'
{"id":"test-001","kind":"preference","content":"User always wants dates formatted as YYYY-MM-DD.","certainty":"definitive","confidence":0.95,"stage":"consolidated","scope":"global","tags":["dates","format","preference"],"evidence":["manually added for testing"],"evidenceCount":1,"salience":"high","provenance":{"createdAt":"2026-03-07T00:00:00Z","sourceConversation":"manual","createdBy":"user"},"lifecycle":{"status":"active"}}
EOF
```

---

## Step 7 — Test in a chat session

Open any chat with your OpenClaw agent and ask:

> "Do you know anything about how I like dates formatted?"

or more explicitly:

> "Search your knowledge base for anything about my preferences."

The agent should call `knowledge_search`, retrieve the artifact, and
apply it in its response. Until passive injection (Milestone 1d) is
wired, you can prompt the agent to search at the start of sessions:

> "Before we begin, search your knowledge base for anything relevant to
> this task."

---

## Troubleshooting

### `blocked plugin candidate: world-writable path (mode=777)`

The plugin path is on an NTFS mount. Follow Step 1 (wsl.conf fix and
WSL restart).

### `plugins.allow: plugin not found: openclaw-plus`

The ID `openclaw-plus` was added to `allow` before the gateway had a
chance to discover it via `load.paths`. Remove it from `allow`, restart
the gateway once (so the plugin is discovered), then add it back and
restart again. See Step 3.

### `plugin id mismatch (manifest uses "...", entry hints "openclaw-plus")`

Older versions of the plugin used `id: "knowledge-management"` in the
manifest, while OpenClaw infers `openclaw-plus` from the npm package
name. This mismatch produced a warning on every restart. It is fixed in
the current source — pull the latest and restart the gateway.

### `knowledge_search` returns a trim error / tool call fails

Older versions of `tools.ts` had the wrong `execute` function signature.
OpenClaw calls `execute(_id: string, params: Record<string, unknown>)`;
the old code accepted one argument and destructured it as `{ query }`,
receiving the tool-call ID string instead of the params object, causing
`query` to be `undefined` and `.trim()` to throw. Fixed in current
source — pull the latest and restart.

### Plugin shows `disabled` despite `entries.openclaw-plus.enabled = true`

If an `allow` list is present and `openclaw-plus` is not in it, the
plugin is blocked regardless of `entries`. Add it to `allow` (Step 3).

### Store path mismatch between playground and OpenClaw

Both default to `~/.openclaw/knowledge/artifacts.jsonl`. If
`OPENCLAW_STATE_DIR` is set in your environment, set
`KNOWLEDGE_STORE_PATH` to match in `~/.openclaw/.env`:

```bash
KNOWLEDGE_STORE_PATH=$OPENCLAW_STATE_DIR/knowledge/artifacts.jsonl
```

---

## Upgrading the plugin

Because the plugin is loaded from source via `load.paths` (not copied),
upgrading is just a pull and a gateway restart:

```bash
cd ~/src/openclaw-knowledge-management
git pull
pnpm install          # only needed if package.json dependencies changed
openclaw gateway restart
```

---

## What comes next

When Milestones 1c and 1d are complete:

- **1c (Passive elicitation):** the plugin will hook into OpenClaw's
  `message_received` event to extract knowledge from every conversation
  automatically — no explicit teaching step needed.
- **1d (Tier 1 triggering):** relevant artifacts will be injected into
  every prompt via `before_prompt_build`, so the agent applies learned
  knowledge transparently without being asked to search.

At that point the explicit "search your knowledge base" prompt and the
standalone playground teaching step both become unnecessary.
