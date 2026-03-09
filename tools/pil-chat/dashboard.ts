/**
 * pil-chat — Browser-based dashboard server.
 *
 * Loaded only when pil-chat is started with --dashboard. Serves a two-panel
 * HTML interface: chat on the left, PIL activity + artifact store on the right.
 *
 * No additional npm dependencies — uses only Node.js built-ins (http,
 * child_process).
 */

import * as http from "node:http";
import { exec } from "node:child_process";

// ─── Interfaces shared with index.ts ─────────────────────────────────────────

export interface PilActivity {
  created: Array<{
    id: string;
    kind: string;
    stage: string;
    confidence: number;
    content: string;
    tags: string[];
  }>;
  updated: Array<{
    id: string;
    evidenceCount: number;
    confidence: number;
    content: string;
  }>;
  injectable: Array<{
    label: string;
    kind: string;
    content: string;
  }>;
  candidates: Array<{
    kind: string;
    certainty: string;
    tags: string[];
    content: string;
  }>;
}

export interface StoreEntry {
  id: string;
  kind: string;
  stage: string;
  confidence: number;
  content: string;
  tags: string[];
  evidenceCount: number;
  /** Inject label — mirrors getInjectLabel() from store.ts; "" when not injectable. */
  label: string;
}

export interface TurnResult {
  isCommand: boolean;
  /** true when the user typed "exit" or "quit" */
  shouldExit?: boolean;
  /** Output text for REPL commands (/list, /store, /help, /clear) */
  commandOutput?: string;
  /** LLM response text */
  response?: string;
  /** PIL activity from the pre-pass (user message) */
  pilPre?: PilActivity;
  /** PIL activity from the exchange pass (full turn) */
  pilExchange?: PilActivity;
  /** Snapshot of all active (non-retired) artifacts after the turn */
  storeSnapshot: StoreEntry[];
  /** Set on error — response and pil fields will be absent */
  error?: string;
}

export type ProcessTurnFn = (userInput: string) => Promise<TurnResult>;
export type GetSnapshotFn = () => Promise<StoreEntry[]>;

// ─── SSE broadcaster ─────────────────────────────────────────────────────────

const sseClients = new Set<http.ServerResponse>();

function emitSse(event: string, data: unknown): void {
  const payload = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  for (const client of sseClients) {
    client.write(payload);
  }
}

// ─── Embedded HTML dashboard ─────────────────────────────────────────────────
// All CSS and JS is inlined — no build step, no external assets, no CDN.

function escHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function buildHtml(displayPath: string, sessionStart: string): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PIL Chat</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; overflow: hidden; }
body {
  display: flex; flex-direction: column;
  font-family: Consolas, "Courier New", monospace; font-size: 13px;
  background: #1e1e1e; color: #d4d4d4;
}
#topbar {
  background: #3c3c3c; padding: 5px 12px; font-size: 11px;
  color: #858585; flex-shrink: 0; display: flex;
  justify-content: space-between; align-items: center;
}
#topbar .title { color: #d4d4d4; font-weight: bold; }
#main { display: flex; flex: 1; overflow: hidden; }

/* ── Left: chat ─────────────────────────────────────────────────────── */
#chat-panel {
  display: flex; flex-direction: column;
  width: 45%; min-width: 300px; border-right: 1px solid #3e3e42;
}
#messages {
  flex: 1; overflow-y: auto; padding: 10px 12px;
  display: flex; flex-direction: column; gap: 8px;
}
.msg { line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
.msg-label {
  color: #608b9f; user-select: none; display: inline-block;
  width: 42px; vertical-align: top;
}
.msg.user     .msg-label { color: #6a9fd8; }
.msg.assistant .msg-label { color: #4ec9b0; }
.msg.error    .msg-label { color: #f44747; }
.msg.error    .msg-text  { color: #f44747; }
.msg.command  .msg-text  { color: #858585; font-style: italic; }
.msg.status   .msg-text  { color: #555; font-style: italic; }
#input-area {
  border-top: 1px solid #3e3e42; padding: 8px;
  display: flex; gap: 6px; flex-shrink: 0;
}
#msg-input {
  flex: 1; background: #2d2d2d; border: 1px solid #3e3e42; color: #d4d4d4;
  font-family: inherit; font-size: 13px; padding: 6px 8px;
  resize: none; height: 52px; outline: none; border-radius: 2px;
}
#msg-input:focus { border-color: #569cd6; }
#send-btn {
  background: #0e639c; color: white; border: none;
  padding: 0 14px; cursor: pointer; border-radius: 2px;
  font-family: inherit; font-size: 13px;
}
#send-btn:hover { background: #1177bb; }
#send-btn:disabled { background: #37373d; color: #555; cursor: default; }
#cmd-bar {
  border-top: 1px solid #3e3e42; padding: 5px 8px;
  display: flex; gap: 5px; flex-shrink: 0; background: #252526;
}
.cmd-btn {
  background: #37373d; color: #d4d4d4; border: 1px solid #3e3e42;
  padding: 2px 9px; cursor: pointer; border-radius: 2px;
  font-family: inherit; font-size: 12px;
}
.cmd-btn:hover { background: #4e4e55; }
.cmd-btn.danger { color: #f44747; border-color: #5a2020; }
.cmd-btn.danger:hover { background: #5a2020; }

/* ── Right: monitor ─────────────────────────────────────────────────── */
#monitor-panel { display: flex; flex-direction: column; flex: 1; overflow: hidden; }
.panel-section { display: flex; flex-direction: column; overflow: hidden; }
#events-section { flex: 2; border-bottom: 1px solid #3e3e42; }
#store-section  { flex: 3; }
.section-hdr {
  background: #252526; border-bottom: 1px solid #3e3e42;
  padding: 4px 10px; font-size: 11px; color: #858585;
  text-transform: uppercase; letter-spacing: 0.08em; flex-shrink: 0;
}
#events-feed {
  overflow-y: auto; flex: 1; padding: 6px 10px;
  display: flex; flex-direction: column; gap: 2px;
}
.ev { font-size: 12px; line-height: 1.4; padding: 2px 0; border-bottom: 1px solid #242424; }
.ev-ts    { color: #4a4a4a; }
.ev-phase {
  color: #858585; background: #2d2d2d; padding: 0 4px;
  border-radius: 2px; margin: 0 4px; font-size: 10px;
}
.ev-created    { color: #4ec9b0; }
.ev-updated    { color: #dcdcaa; }
.ev-injectable { color: #569cd6; }
.ev-candidate  { color: #666; }
.ev-system     { color: #4a4a4a; font-style: italic; }
.ev-error      { color: #f44747; }
.ev-command    { color: #666; }
#store-scroll { overflow-y: auto; flex: 1; }
#store-table { width: 100%; border-collapse: collapse; font-size: 12px; }
#store-table th {
  background: #252526; color: #858585; font-weight: normal; text-align: left;
  padding: 4px 8px; border-bottom: 1px solid #3e3e42;
  position: sticky; top: 0; font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.06em;
}
#store-table td {
  padding: 3px 8px; border-bottom: 1px solid #242424; vertical-align: middle;
}
#store-table tr:hover td { background: #2a2a2a; }
.badge {
  display: inline-block; padding: 1px 5px; border-radius: 2px;
  font-size: 10px; white-space: nowrap;
}
.kd-preference  { background: #1a2a4a; color: #9cdcfe; }
.kd-convention  { background: #1a3a4a; color: #9cdcfe; }
.kd-fact        { background: #1a1a3a; color: #c586c0; }
.kd-procedure   { background: #1a3a1a; color: #b5cea8; }
.kd-judgment    { background: #3a1a1a; color: #f44747; }
.kd-strategy    { background: #3a2a1a; color: #ce9178; }
.st-candidate    { background: #2d2d2d; color: #858585; }
.st-accumulating { background: #1a2a3a; color: #569cd6; }
.st-consolidated { background: #1a3a1a; color: #4ec9b0; }
.conf-wrap {
  width: 52px; background: #2d2d2d; height: 5px; border-radius: 3px;
  display: inline-block; vertical-align: middle; margin-right: 4px; overflow: hidden;
}
.conf-fill { height: 100%; border-radius: 3px; }
.lbl-established { color: #4ec9b0; font-size: 11px; }
.lbl-suggestion  { color: #dcdcaa; font-size: 11px; }
.lbl-provisional { color: #569cd6; font-size: 11px; }
.lbl-none        { color: #555;    font-size: 11px; }
.content-cell {
  color: #d4d4d4; max-width: 240px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
</style>
</head>
<body>
<div id="topbar">
  <span class="title">PIL Chat</span>
  <span>store: ${escHtml(displayPath)} &nbsp;·&nbsp; session: ${escHtml(sessionStart)}</span>
</div>

<div id="main">
  <!-- Left: chat panel -->
  <div id="chat-panel">
    <div id="messages"></div>
    <div id="input-area">
      <textarea id="msg-input" placeholder="Type a message… (Enter to send, Shift+Enter for newline)"></textarea>
      <button id="send-btn" onclick="sendMsg()">Send</button>
    </div>
    <div id="cmd-bar">
      <button class="cmd-btn" onclick="sendCmd('/list')">/list</button>
      <button class="cmd-btn" onclick="sendCmd('/store')">/store</button>
      <button class="cmd-btn" onclick="sendCmd('/clear')">/clear history</button>
      <button class="cmd-btn danger" onclick="confirmReset()">/reset store</button>
    </div>
  </div>

  <!-- Right: monitor panel -->
  <div id="monitor-panel">
    <div id="events-section" class="panel-section">
      <div class="section-hdr">PIL Events</div>
      <div id="events-feed"></div>
    </div>
    <div id="store-section" class="panel-section">
      <div class="section-hdr">Artifact Store</div>
      <div id="store-scroll">
        <table id="store-table">
          <thead>
            <tr>
              <th>Kind</th>
              <th>Stage</th>
              <th>Conf</th>
              <th>Label</th>
              <th>Content</th>
            </tr>
          </thead>
          <tbody id="store-body"></tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<script>
const messagesEl = document.getElementById('messages');
const inputEl    = document.getElementById('msg-input');
const sendBtnEl  = document.getElementById('send-btn');
const feedEl     = document.getElementById('events-feed');
const storeEl    = document.getElementById('store-body');
let busy = false;

// ── SSE ──────────────────────────────────────────────────────────────────────
const es = new EventSource('/events');
es.addEventListener('connected',          () => addFeed('system', null, 'Connected to pil-chat.'));
es.addEventListener('thinking',           () => { setBusy(true); addMsg('status', '⋯  thinking…'); });
es.addEventListener('assistant-response', e  => { setBusy(false); removeStatus(); addMsg('assistant', JSON.parse(e.data).text); });
es.addEventListener('command-output',     e  => { setBusy(false); removeStatus(); const o = JSON.parse(e.data).output; if (o) addMsg('command', o); });
es.addEventListener('pil-activity',       e  => renderPilActivity(JSON.parse(e.data)));
es.addEventListener('store-snapshot',     e  => renderStore(JSON.parse(e.data)));
es.addEventListener('error-msg',          e  => { setBusy(false); removeStatus(); addMsg('error', JSON.parse(e.data).message); });

// ── Input ─────────────────────────────────────────────────────────────────────
inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMsg(); }
});

function sendMsg() {
  if (busy) return;
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = '';
  addMsg('user', text);
  post(text);
}

function sendCmd(cmd) {
  if (busy) return;
  addMsg('user', cmd);
  post(cmd);
}

function confirmReset() {
  if (busy) return;
  if (!confirm('Delete ALL artifacts in the current store? This cannot be undone.')) return;
  addMsg('user', '/reset');
  post('/reset');
}

function post(message) {
  setBusy(true);
  fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  }).catch(err => { setBusy(false); removeStatus(); addMsg('error', 'Network error: ' + err.message); });
}

function setBusy(b) {
  busy = b;
  sendBtnEl.disabled = b;
}

// ── Messages ─────────────────────────────────────────────────────────────────
function addMsg(role, text) {
  if (role === 'status') removeStatus();
  const el = document.createElement('div');
  el.className = 'msg ' + role;
  const labels = { user: 'You', assistant: 'AI', command: 'CMD', error: 'ERR', status: '···' };
  el.innerHTML =
    '<span class="msg-label">' + (labels[role] || '') + '</span>' +
    '<span class="msg-text">' + esc(text) + '</span>';
  if (role === 'status') el.dataset.status = '1';
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function removeStatus() {
  const s = messagesEl.querySelector('[data-status]');
  if (s) s.remove();
}

// ── PIL event feed (newest first) ─────────────────────────────────────────────
function nowStr() { return new Date().toTimeString().slice(0, 8); }

function addFeed(type, phase, text) {
  const el = document.createElement('div');
  el.className = 'ev';
  const phSpan = phase ? \`<span class="ev-phase">\${phase}</span>\` : '';
  el.innerHTML = \`<span class="ev-ts">\${nowStr()}</span>\${phSpan} <span class="ev-\${type}">\${esc(text)}</span>\`;
  feedEl.insertBefore(el, feedEl.firstChild);
}

function renderPilActivity(data) {
  const p = data.phase;
  for (const a of (data.created    || []))
    addFeed('created',    p, \`✚ [\${a.kind}/\${a.stage}] conf=\${a.confidence.toFixed(2)}  "\${snip(a.content, 55)}"\`);
  for (const a of (data.updated    || []))
    addFeed('updated',    p, \`↺ evidence=\${a.evidenceCount} conf=\${a.confidence.toFixed(2)}  "\${snip(a.content, 50)}"\`);
  for (const i of (data.injectable || []))
    addFeed('injectable', p, \`→ \${i.label} [\${i.kind}]  "\${snip(i.content, 50)}"\`);
  for (const c of (data.candidates || []))
    addFeed('candidate',  p, \`? [\${c.kind}/\${c.certainty}]  "\${snip(c.content, 45)}"\`);
}

// ── Store table ───────────────────────────────────────────────────────────────
function renderStore(entries) {
  if (!entries || !entries.length) {
    storeEl.innerHTML = '<tr><td colspan="5" style="color:#555;padding:8px;">(no artifacts)</td></tr>';
    return;
  }
  const sorted = [...entries].sort((a, b) => b.confidence - a.confidence);
  storeEl.innerHTML = sorted.map(a => {
    const pct   = Math.round(a.confidence * 100);
    const color = a.confidence >= 0.95 ? '#4ec9b0'
                : a.confidence >= 0.85 ? '#dcdcaa'
                : a.confidence >= 0.75 ? '#569cd6'
                : '#858585';
    const kdCls = ['preference','convention','fact','procedure','judgment','strategy'].includes(a.kind)
                  ? 'kd-' + a.kind : 'kd-fact';
    const stCls = ['candidate','accumulating','consolidated'].includes(a.stage)
                  ? 'st-' + a.stage : 'st-candidate';
    const lbl   = a.label
                  ? \`<span class="\${labelCls(a.label)}">\${esc(a.label)}</span>\`
                  : '<span class="lbl-none">—</span>';
    return \`<tr>
      <td><span class="badge \${kdCls}">\${esc(a.kind)}</span></td>
      <td><span class="badge \${stCls}">\${esc(a.stage)}</span></td>
      <td style="white-space:nowrap">
        <span class="conf-wrap"><span class="conf-fill" style="width:\${pct}%;background:\${color}"></span></span>\${a.confidence.toFixed(2)}
      </td>
      <td>\${lbl}</td>
      <td class="content-cell" title="\${esc(a.content)}">\${esc(snip(a.content, 65))}</td>
    </tr>\`;
  }).join('');
}

function labelCls(label) {
  if (label.includes('established')) return 'lbl-established';
  if (label.includes('suggestion'))  return 'lbl-suggestion';
  if (label.includes('provisional')) return 'lbl-provisional';
  return 'lbl-none';
}

// ── Utils ─────────────────────────────────────────────────────────────────────
function snip(s, n) { return s.length > n ? s.slice(0, n) + '…' : s; }
function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
</script>
</body>
</html>`;
}

// ─── HTTP server ──────────────────────────────────────────────────────────────

/**
 * Start the dashboard HTTP server and open a browser window.
 *
 * @param port              Local port to listen on.
 * @param processTurn       Function that executes one chat/command turn and
 *                          returns a TurnResult with PIL activity + store snapshot.
 * @param displayPath       Store path string shown in the top bar.
 * @param sessionStart      ISO timestamp string shown in the top bar.
 * @param getInitialSnapshot Called once on first SSE connection to populate the
 *                          store panel before any chat has occurred.
 */
export function startDashboard(
  port: number,
  processTurn: ProcessTurnFn,
  displayPath: string,
  sessionStart: string,
  getInitialSnapshot: GetSnapshotFn,
): void {
  const html = buildHtml(displayPath, sessionStart);

  const server = http.createServer((req, res) => {
    const url    = req.url    ?? "/";
    const method = req.method ?? "GET";

    // ── Dashboard HTML ───────────────────────────────────────────────────────
    if (method === "GET" && url === "/") {
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      res.end(html);
      return;
    }

    // ── SSE stream ───────────────────────────────────────────────────────────
    if (method === "GET" && url === "/events") {
      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
      });
      sseClients.add(res);
      res.write(`event: connected\ndata: "connected"\n\n`);

      // Send initial store snapshot so the panel is populated before any chat
      getInitialSnapshot()
        .then((entries) => {
          res.write(`event: store-snapshot\ndata: ${JSON.stringify(entries)}\n\n`);
        })
        .catch(() => { /* non-fatal */ });

      req.on("close", () => sseClients.delete(res));
      return;
    }

    // ── Chat endpoint ────────────────────────────────────────────────────────
    if (method === "POST" && url === "/chat") {
      let body = "";
      req.on("data", (chunk) => { body += chunk; });
      req.on("end", () => {
        // Respond immediately so the browser doesn't block on the HTTP reply
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end('{"ok":true}');

        let userInput: string;
        try {
          const parsed = JSON.parse(body) as { message?: unknown };
          userInput = (typeof parsed.message === "string" ? parsed.message : "").trim();
        } catch {
          emitSse("error-msg", { message: "Invalid request body." });
          return;
        }
        if (!userInput) return;

        emitSse("thinking", {});

        processTurn(userInput)
          .then((result) => {
            if (result.shouldExit) {
              emitSse("command-output", {
                output: "Use Ctrl+C in the terminal to exit pil-chat.",
              });
              return;
            }
            if (result.error) {
              emitSse("error-msg", { message: result.error });
              emitSse("store-snapshot", result.storeSnapshot);
              return;
            }
            if (result.isCommand) {
              emitSse("command-output", { output: result.commandOutput ?? "" });
            } else {
              if (result.pilPre) {
                emitSse("pil-activity", { phase: "pre", ...result.pilPre });
              }
              if (result.response != null) {
                emitSse("assistant-response", { text: result.response });
              }
              if (result.pilExchange) {
                emitSse("pil-activity", { phase: "exchange", ...result.pilExchange });
              }
            }
            emitSse("store-snapshot", result.storeSnapshot);
          })
          .catch((err: unknown) => {
            emitSse("error-msg", {
              message: err instanceof Error ? err.message : String(err),
            });
          });
      });
      return;
    }

    res.writeHead(404);
    res.end();
  });

  server.listen(port, "127.0.0.1", () => {
    const url = `http://localhost:${port}`;
    console.log(`\npil-chat dashboard  →  ${url}`);
    console.log("Browser interface active. Use Ctrl+C to exit.\n");

    // Best-effort: auto-open the browser (ignore errors — the URL is printed above)
    const cmd =
      process.platform === "win32"  ? `start "" "${url}"` :
      process.platform === "darwin" ? `open "${url}"` :
                                      `xdg-open "${url}"`;
    exec(cmd, () => { /* ignore */ });
  });
}
