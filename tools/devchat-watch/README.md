# Devchat Watch

Tools for monitoring `.private/devchats/chatlog.md` and notifying or invoking AI agents when new entries appear.

Two tools are provided: a **one-shot notifier** (PowerShell) and a **continuous watcher** (Node.js).

---

## Continuous Watcher — `watch.mjs` / `watch.cmd`

A cross-platform Node.js process that runs indefinitely, watching `chatlog.md` for new entries
and automatically invoking `claude --print` (or `codex`) to generate a response.

### Requirements

- Node.js 18+ on PATH
- `claude` CLI (`npm i -g @anthropic-ai/claude-code`) **or** `codex` on PATH
- `ANTHROPIC_API_KEY` environment variable set (for `claude`)

### Usage

```cmd
tools\devchat-watch\watch.cmd
```

```bash
# Linux / macOS
node tools/devchat-watch/watch.mjs
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--agent <name>` | `claudecode` | Agent to invoke: `claudecode` or `codex` |
| `--dev-id <id>` | `DEV#1` | Developer ID this instance speaks as |
| `--chatlog <path>` | `.private/devchats/chatlog.md` | Path to the shared chatlog |
| `--rules <path>` | `.private/devchats/rules.txt` | Path to formatting rules injected into the prompt |
| `--debounce <ms>` | `2000` | Milliseconds to wait after a file-change event before processing |
| `--dry-run` | off | Print the assembled prompt; skip agent invocation and file write |

### Examples

```cmd
REM Run as DEV#2 using Codex
tools\devchat-watch\watch.cmd --agent codex --dev-id DEV#2

REM Preview the prompt that would be sent, without invoking anything
tools\devchat-watch\watch.cmd --dry-run

REM Custom chatlog path
tools\devchat-watch\watch.cmd --chatlog C:\projects\myrepo\.private\devchats\chatlog.md
```

### How it works

1. `fs.watch` monitors `chatlog.md` for any write event.
2. A debounce timer (default 2 s) waits for the file to settle between rapid saves.
3. The file is read and hashed (SHA-256). If the hash matches the last processed version,
   the change is skipped.
4. The last `### YYYY-MM-DD HH:MM:SS - DEV#N` header is parsed. If the author matches
   `--dev-id`, the change is skipped (loop prevention).
5. A prompt is assembled from `rules.txt` + the full chatlog content. The agent is instructed
   to output its response body only, or the literal token `NO_RESPONSE_NEEDED`.
6. The agent is spawned non-interactively:
   - `claude --print` — prompt is written to stdin
   - `codex --full-auto` — prompt is written to stdin
7. If a response is returned, it is appended to the chatlog as a new entry with the current
   timestamp and `--dev-id`.
8. The hash of the freshly-written file is saved to the state file so the watcher ignores its
   own write on the next event.

### State file

`.private/devchats/.watch-state-DEV<N>.json` — auto-created alongside the chatlog.
Gitignored via `.private/`. Contains:

```json
{
  "devId": "DEV#1",
  "lastHash": "<sha256>",
  "savedAt": "2026-03-14T10:00:00.000Z"
}
```

### Loop prevention

Two independent guards prevent runaway write loops:

1. **Content hash** — the hash of the last file the watcher processed/wrote is persisted.
   If the next event produces the same hash, it is skipped.
2. **Author check** — if the last chatlog entry's `DEV#N` matches `--dev-id`, the watcher
   never responds (an agent will not reply to its own output).

---

## One-shot Notifier — `notify-agent-chatlog-change.ps1` / `.cmd`

Checks whether `chatlog.md` was modified within the last N minutes and, if so,
prints (and optionally shows a popup / copies to clipboard) a notification message
for a human or agent to act on.

This tool does **not** invoke the agent directly — it emits a message for a human
or agent process to notice.

### Usage

```cmd
tools\devchat-watch\notify-agent-chatlog-change.cmd
```

Target Claude Code instead of Codex:

```cmd
tools\devchat-watch\notify-agent-chatlog-change.cmd -Agent ClaudeCode
```

Use a different freshness window:

```cmd
tools\devchat-watch\notify-agent-chatlog-change.cmd -RecentMinutes 10
```

Show a popup and copy the notification text to the clipboard:

```cmd
tools\devchat-watch\notify-agent-chatlog-change.cmd -ShowPopup -CopyPrompt
```

Force a notification even if the file is older or already notified:

```cmd
tools\devchat-watch\notify-agent-chatlog-change.cmd -Force
```

### Duplicate suppression

By default the script records the last notified write timestamp in
`.private/devchats/.chatlog-notify-<agent>.json` so the same chatlog version is
not repeatedly announced.

Use `-NoState` for a stateless run.

---

## Choosing between the two tools

| | One-shot notifier | Continuous watcher |
|---|---|---|
| **How to run** | Triggered manually or via Task Scheduler | Long-running process (`node watch.mjs`) |
| **Invokes agent?** | No — emits notification text only | Yes — calls `claude --print` or `codex` |
| **Platform** | Windows (PowerShell) | Cross-platform (Node.js) |
| **Loop prevention** | N/A | Hash dedup + author check |
| **State file** | `.chatlog-notify-<agent>.json` | `.watch-state-DEV<N>.json` |

Use the **notifier** when you want manual control over when the agent is called,
or when running on a CI/CD trigger.

Use the **watcher** when you want a fully autonomous multi-agent chatlog loop
with no human in the middle.

---

## Future Design Direction

This toolset is the seed of a more general watcher/dispatcher. Future extensions
should separate (1) trigger conditions from (2) actions:

- **Triggers**: targeted-agent detection, unanswered-entry detection, mention parsing, scheduled polling
- **Actions**: queue-file output, tool launch, webhook dispatch, richer agent-specific prompts
