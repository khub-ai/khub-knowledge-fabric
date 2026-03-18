#!/usr/bin/env node
/**
 * watch.mjs — Cross-platform devchat watcher
 *
 * Watches chatlog.md for changes. When a new entry appears from a developer
 * other than this agent, invokes `claude --print` (or `codex`) with the full
 * chatlog as context and appends the response as a new formatted entry.
 *
 * Usage:
 *   node tools/devchat-watch/watch.mjs [options]
 *
 * Options:
 *   --agent <name>      claudecode | codex          (default: claudecode)
 *   --dev-id <id>       DEV#N                       (default: DEV#1)
 *   --chatlog <path>    path to chatlog.md          (default: .private/devchats/chatlog.md)
 *   --rules <path>      path to rules.txt           (default: .private/devchats/rules.txt)
 *   --debounce <ms>     write-settle wait time      (default: 2000)
 *   --dry-run           print prompt; skip invoke and write
 *
 * Loop prevention:
 *   After writing a response the new content hash is saved to a state file in
 *   .private/devchats/. If the next change event reads the same hash, it is
 *   skipped. In addition, if the last chatlog entry's author matches --dev-id
 *   the agent never responds to its own output.
 *
 * State file:
 *   .private/devchats/.watch-state-DEV<N>.json  (gitignored via .private/)
 */

import { watch }                from 'node:fs';
import { readFile, writeFile }  from 'node:fs/promises';
import { existsSync }           from 'node:fs';
import { createHash }           from 'node:crypto';
import { spawn }                from 'node:child_process';
import { resolve, dirname }     from 'node:path';
import { fileURLToPath }        from 'node:url';

// ── CLI args ──────────────────────────────────────────────────────────────────

const argv = process.argv.slice(2);

function argVal(flag, def) {
  const i = argv.indexOf(flag);
  return i !== -1 && argv[i + 1] ? argv[i + 1] : def;
}
function argFlag(flag) { return argv.includes(flag); }

const __dir    = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dir, '..', '..');

const AGENT    = argVal('--agent',    'claudecode');
const DEV_ID   = argVal('--dev-id',   'DEV#1');
const CHATLOG  = resolve(repoRoot, argVal('--chatlog', '.private/devchats/chatlog.md'));
const RULES    = resolve(repoRoot, argVal('--rules',   '.private/devchats/rules.txt'));
const DEBOUNCE = parseInt(argVal('--debounce', '2000'), 10);
const DRY_RUN  = argFlag('--dry-run');

// State file lives alongside the chatlog in the gitignored .private dir.
const STATE = CHATLOG.replace(/\.md$/, `.watch-state-${DEV_ID.replace('#', '')}.json`);

// ── Chatlog parsing ───────────────────────────────────────────────────────────

/**
 * Find the last ### YYYY-MM-DD HH:MM:SS - DEV#N header in the chatlog.
 * Returns { timestamp, author } or null.
 */
function lastEntry(content) {
  const re = /^###\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+-\s+(DEV#\d+)\s*$/gm;
  let match, last = null;
  while ((match = re.exec(content)) !== null) last = match;
  return last ? { timestamp: last[1], author: last[2] } : null;
}

// ── Hashing ───────────────────────────────────────────────────────────────────

function sha256(s) {
  return createHash('sha256').update(s).digest('hex');
}

// ── State persistence ─────────────────────────────────────────────────────────

async function loadState() {
  if (!existsSync(STATE)) return { lastHash: null };
  try { return JSON.parse(await readFile(STATE, 'utf8')); }
  catch { return { lastHash: null }; }
}

async function saveState(hash) {
  const obj = { devId: DEV_ID, lastHash: hash, savedAt: new Date().toISOString() };
  await writeFile(STATE, JSON.stringify(obj, null, 2), 'utf8');
}

// ── Prompt builder ────────────────────────────────────────────────────────────

async function buildPrompt(chatlogContent) {
  let rules = '(rules file not found — follow standard developer chat conventions)';
  try { rules = await readFile(RULES, 'utf8'); } catch { /* use default */ }

  return `\
You are ${DEV_ID}, a developer collaborating on this project through a shared chatlog file.

RULES (from rules.txt — follow these exactly for formatting):
${rules.trim()}

CURRENT CHATLOG:
${chatlogContent.trim()}

---

Task: Read the chatlog above. Decide whether ${DEV_ID} should write a response to the most recent entry.

Respond if ALL of these are true:
  - The most recent entry was written by a different developer (not ${DEV_ID}).
  - The entry raises a question, issue, or work item that ${DEV_ID} is responsible for or has relevant context on.
  - No response from ${DEV_ID} already follows that entry.

Do NOT respond if:
  - The last entry was written by ${DEV_ID}.
  - The entry is directed at a different developer and does not concern ${DEV_ID}.
  - The entry is informational and does not require acknowledgment.

If a response IS needed:
  - Write ONLY the response body. Do NOT include the ### timestamp header — it is added automatically.
  - Follow the formatting rules in rules.txt exactly (Markdown, one blank line after paragraphs, etc.).

If no response is needed:
  - Output exactly this token and nothing else: NO_RESPONSE_NEEDED
`.trim();
}

// ── Agent invocation ──────────────────────────────────────────────────────────

/**
 * Invoke claude or codex non-interactively.
 * Writes the prompt to the process's stdin and captures stdout.
 *
 * claude: `claude --print`  (prompt via stdin)
 * codex:  `codex --full-auto` (prompt via stdin; adjust if your version differs)
 */
function invokeAgent(prompt) {
  return new Promise((res, rej) => {
    const [cmd, ...flags] =
      AGENT === 'codex'
        ? ['codex', '--full-auto']
        : ['claude', '--print'];

    const child = spawn(cmd, flags, {
      stdio: ['pipe', 'pipe', 'pipe'],
      // On Windows, shell: true is needed if cmd is a .cmd/.bat shim.
      shell: process.platform === 'win32',
    });

    let stdout = '', stderr = '';
    child.stdout.on('data', d => (stdout += d));
    child.stderr.on('data', d => (stderr += d));

    child.on('error', e =>
      rej(new Error(`Failed to spawn "${cmd}": ${e.message}\nIs it installed and on PATH?`)),
    );
    child.on('close', code => {
      if (code !== 0) {
        rej(new Error(`"${cmd}" exited with code ${code}.\nstderr: ${stderr.trim()}`));
      } else {
        res(stdout.trim());
      }
    });

    child.stdin.write(prompt, 'utf8');
    child.stdin.end();
  });
}

// ── Response formatting ───────────────────────────────────────────────────────

function formatEntry(devId, body) {
  const now = new Date();
  // ISO local-like timestamp: YYYY-MM-DD HH:MM:SS
  const ts = [
    now.getFullYear(),
    String(now.getMonth() + 1).padStart(2, '0'),
    String(now.getDate()).padStart(2, '0'),
  ].join('-') + ' ' + [
    String(now.getHours()).padStart(2, '0'),
    String(now.getMinutes()).padStart(2, '0'),
    String(now.getSeconds()).padStart(2, '0'),
  ].join(':');

  // Rules require LF line endings. Normalize and wrap with one blank line on each side.
  const normalizedBody = body.replace(/\r\n/g, '\n').trim();
  return `\n### ${ts} - ${devId}\n\n${normalizedBody}\n`;
}

// ── Main processing ───────────────────────────────────────────────────────────

let processing  = false;
let debounceTimer = null;

async function processChange() {
  if (processing) return;
  processing = true;

  try {
    // Read chatlog — normalize CRLF to LF throughout.
    const raw     = await readFile(CHATLOG, 'utf8');
    const content = raw.replace(/\r\n/g, '\n');
    const hash    = sha256(content);

    // Dedup: same content as last run.
    const state = await loadState();
    if (hash === state.lastHash) {
      log('No change since last processed version — skipping.');
      return;
    }

    const entry = lastEntry(content);
    if (!entry) {
      log('No parseable entry found in chatlog.');
      await saveState(hash);
      return;
    }

    log(`Last entry: ${entry.author} at ${entry.timestamp}`);

    // Loop prevention: last entry is our own output.
    if (entry.author === DEV_ID) {
      log(`Last entry is from ${DEV_ID} — nothing to respond to.`);
      await saveState(hash);
      return;
    }

    log(`Entry from ${entry.author}. Composing response as ${DEV_ID}...`);

    const prompt = await buildPrompt(content);

    if (DRY_RUN) {
      log('DRY RUN — prompt follows, no agent invoked.\n');
      console.log('─'.repeat(60));
      console.log(prompt);
      console.log('─'.repeat(60));
      await saveState(hash);
      return;
    }

    // Invoke agent.
    let response;
    try {
      response = await invokeAgent(prompt);
    } catch (e) {
      console.error(`[watch] Agent error: ${e.message}`);
      // Don't update state — will retry on next file change.
      return;
    }

    if (!response || response.includes('NO_RESPONSE_NEEDED')) {
      log('Agent determined no response needed.');
      await saveState(hash);
      return;
    }

    // Append entry to chatlog.
    const newEntry  = formatEntry(DEV_ID, response);
    const updated   = content.endsWith('\n') ? content + newEntry : content + '\n' + newEntry;

    await writeFile(CHATLOG, updated, 'utf8');

    // Save hash of what WE just wrote — our own next change event will be skipped.
    await saveState(sha256(updated));

    log(`Response written as ${DEV_ID}.`);
    log(`Preview: ${response.slice(0, 160)}${response.length > 160 ? '…' : ''}`);

  } catch (e) {
    console.error(`[watch] Unexpected error: ${e.message}`);
  } finally {
    processing = false;
  }
}

function log(msg) {
  const ts = new Date().toISOString().slice(11, 19);
  console.log(`[${ts}] ${msg}`);
}

function scheduleProcess() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(processChange, DEBOUNCE);
}

// ── Start ─────────────────────────────────────────────────────────────────────

if (!existsSync(CHATLOG)) {
  console.error(`[watch] Chatlog not found: ${CHATLOG}`);
  process.exit(1);
}

console.log('[watch] devchat-watch starting');
console.log(`[watch]   agent   : ${AGENT}`);
console.log(`[watch]   dev-id  : ${DEV_ID}`);
console.log(`[watch]   chatlog : ${CHATLOG}`);
console.log(`[watch]   debounce: ${DEBOUNCE}ms`);
if (DRY_RUN) console.log('[watch]   mode    : DRY RUN');
console.log('');

watch(CHATLOG, { persistent: true }, scheduleProcess);

// Process once on startup to handle any change that occurred while the
// watcher was not running.
scheduleProcess();

console.log('[watch] Watching. Press Ctrl+C to stop.\n');
