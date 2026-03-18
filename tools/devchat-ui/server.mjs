#!/usr/bin/env node
/**
 * server.mjs — devchat-ui backend
 *
 * Serves the browser UI on http://localhost:PORT and provides:
 *   - REST API  (/api/entries, /api/agents, /api/config, /api/rules)
 *   - WebSocket real-time push (new entries, agent status/logs)
 *   - AgentManager  — spawns/kills watch.mjs instances per agent
 *   - ChatlogWatcher — fs.watch → parse → diff → broadcast
 *   - AutomationEngine — fires agent triggers on new human entries
 */

import { createServer }              from 'node:http';
import { readFileSync, existsSync,
         mkdirSync, watch as fsWatch } from 'node:fs';
import { readFile, writeFile,
         appendFile }                from 'node:fs/promises';
import { createHash, randomUUID }    from 'node:crypto';
import { spawn, exec }               from 'node:child_process';
import { resolve, dirname, extname } from 'node:path';
import { fileURLToPath }             from 'node:url';
import { WebSocketServer }           from 'ws';

const __dir    = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dir, '..', '..');

// ── CLI args ──────────────────────────────────────────────────────────────────

const argv = process.argv.slice(2);
const argVal  = (f, d) => { const i = argv.indexOf(f); return i !== -1 && argv[i+1] ? argv[i+1] : d; };
const argFlag = f => argv.includes(f);

const CLI_PORT    = parseInt(argVal('--port', '0'), 10) || null;
const CLI_CHATLOG = argVal('--chatlog', '');
const CONFIG_PATH = resolve(repoRoot, argVal('--config', '.private/devchats/ui-config.json'));
const OPEN_BROWSER = argFlag('--open');

// ── Config ────────────────────────────────────────────────────────────────────

const DEFAULT_CONFIG = {
  chatlog: '.private/devchats/chatlog.md',
  rules:   '.private/devchats/rules.txt',
  port:    3737,
  agents: [
    { id: 'DEV#1', agent: 'claudecode', respondTo: ['DEV#0'] },
    { id: 'DEV#2', agent: 'claudecode', respondTo: ['DEV#0'] },
  ],
  automationRules: [],
};

function loadConfigSync() {
  if (existsSync(CONFIG_PATH)) {
    try { return { ...DEFAULT_CONFIG, ...JSON.parse(readFileSync(CONFIG_PATH, 'utf8')) }; }
    catch { /* fall through */ }
  }
  return { ...DEFAULT_CONFIG };
}

async function saveConfig(cfg) {
  mkdirSync(dirname(CONFIG_PATH), { recursive: true });
  await writeFile(CONFIG_PATH, JSON.stringify(cfg, null, 2), 'utf8');
}

let _config = loadConfigSync();
const getConfig = () => _config;

async function updateConfig(patch) {
  _config = { ..._config, ...patch };
  await saveConfig(_config);
  broadcast({ type: 'config_updated', config: _config });
  return _config;
}

// ── Paths ─────────────────────────────────────────────────────────────────────

const chatlogPath = () => resolve(repoRoot, CLI_CHATLOG || getConfig().chatlog);
const rulesPath   = () => resolve(repoRoot, getConfig().rules);

// ── Chatlog parsing ───────────────────────────────────────────────────────────

const sha256 = s => createHash('sha256').update(s).digest('hex');

function parseEntries(content) {
  const re   = /^###\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+-\s+(DEV#\d+)\s*$/gm;
  const hits = [];
  let m;
  while ((m = re.exec(content)) !== null) {
    hits.push({ start: m.index, end: m.index + m[0].length, timestamp: m[1], author: m[2] });
  }
  return hits.map((h, i) => {
    // Body runs from end-of-this-header to start-of-next-header (or EOF)
    const nextIdx = i + 1 < hits.length ? hits[i + 1].start : content.length;
    const body = content.slice(h.end, nextIdx).trim();
    return { id: sha256(`${h.timestamp}:${h.author}`), timestamp: h.timestamp, author: h.author, body };
  });
}

// ── Chatlog watcher ───────────────────────────────────────────────────────────

let _lastHash    = '';
let _lastEntries = [];
let _debTimer    = null;

async function processChange() {
  const CHATLOG = chatlogPath();
  let raw;
  try { raw = await readFile(CHATLOG, 'utf8'); } catch { return; }

  const content = raw.replace(/\r\n/g, '\n');
  const hash    = sha256(content);
  if (hash === _lastHash) return;
  _lastHash = hash;

  const entries    = parseEntries(content);
  const knownIds   = new Set(_lastEntries.map(e => e.id));
  const newEntries = entries.filter(e => !knownIds.has(e.id));
  _lastEntries     = entries;

  if (newEntries.length > 0) {
    broadcast({ type: 'entries_added', entries: newEntries });
    newEntries.forEach(evaluateRules);
  }
}

function scheduleChange() {
  clearTimeout(_debTimer);
  _debTimer = setTimeout(processChange, 600);
}

function startChatlogWatcher() {
  const CHATLOG = chatlogPath();
  if (!existsSync(CHATLOG)) {
    console.warn(`[server] Chatlog not found: ${CHATLOG}`);
    return;
  }
  processChange(); // initial read
  fsWatch(CHATLOG, { persistent: true }, scheduleChange);
  console.log(`[server] Watching: ${CHATLOG}`);
}

// ── Agent manager ─────────────────────────────────────────────────────────────

const _agents = new Map(); // id → { process, logs, status, lastResponseAt, pid }

function agentStatus(id) {
  const a = _agents.get(id);
  return {
    id,
    status:         a ? a.status : 'stopped',
    pid:            a ? a.pid    : null,
    lastResponseAt: a ? a.lastResponseAt : null,
    logs:           a ? a.logs.slice(-50) : [],
  };
}

const allAgentStatuses = () => getConfig().agents.map(a => agentStatus(a.id));

function startAgent(id) {
  if (_agents.has(id)) return agentStatus(id);
  const cfg = getConfig().agents.find(a => a.id === id);
  if (!cfg) return { error: `Unknown agent: ${id}` };

  const watchScript = resolve(__dir, '..', 'devchat-watch', 'watch.mjs');
  const args = [
    watchScript,
    '--agent',      cfg.agent,
    '--dev-id',     cfg.id,
    '--respond-to', (cfg.respondTo || ['DEV#0']).join(','),
    '--chatlog',    chatlogPath(),
    '--rules',      rulesPath(),
  ];

  const proc  = spawn('node', args, { stdio: ['ignore', 'pipe', 'pipe'] });
  const entry = { process: proc, logs: [], status: 'running', lastResponseAt: null, pid: proc.pid };

  const handleLine = line => {
    entry.logs.push(line);
    if (entry.logs.length > 500) entry.logs.shift();
    if (line.includes('Response written')) entry.lastResponseAt = new Date().toISOString();
    broadcast({ type: 'agent_log', id, line });
    broadcast({ type: 'agent_status', ...agentStatus(id) });
  };

  proc.stdout.on('data', d => d.toString().split('\n').filter(Boolean).forEach(handleLine));
  proc.stderr.on('data', d => d.toString().split('\n').filter(Boolean).forEach(l => handleLine(`[err] ${l}`)));
  proc.on('close', () => {
    const last = entry.lastResponseAt;
    _agents.delete(id);
    broadcast({ type: 'agent_status', id, status: 'stopped', pid: null, lastResponseAt: last, logs: [] });
  });

  _agents.set(id, entry);
  broadcast({ type: 'agent_status', ...agentStatus(id) });
  return agentStatus(id);
}

function stopAgent(id) {
  const a = _agents.get(id);
  if (!a) return agentStatus(id);
  a.process.kill('SIGTERM');
  return agentStatus(id);
}

async function triggerAgent(id) {
  const cfg = getConfig().agents.find(a => a.id === id);
  if (!cfg) return { error: `Unknown agent: ${id}` };

  const CHATLOG = chatlogPath();
  let raw, rules;
  try { raw   = await readFile(CHATLOG, 'utf8'); } catch { return { error: 'Chatlog unreadable' }; }
  try { rules = await readFile(rulesPath(), 'utf8'); } catch { rules = ''; }

  const content = raw.replace(/\r\n/g, '\n').trim();
  const prompt  = buildPrompt(id, content, rules);

  const [cmd, ...flags] = cfg.agent === 'codex'
    ? ['codex', '--full-auto']
    : ['claude', '--print'];

  const log = line => broadcast({ type: 'agent_log', id, line: `[trigger] ${line}` });
  log(`Invoking one-shot response as ${id}…`);

  const child = spawn(cmd, flags, { stdio: ['pipe', 'pipe', 'pipe'], shell: process.platform === 'win32' });
  let stdout = '', stderr = '';
  child.stdout.on('data', d => (stdout += d));
  child.stderr.on('data', d => (stderr += d));
  child.stdin.write(prompt, 'utf8');
  child.stdin.end();

  child.on('close', async code => {
    const response = stdout.trim();
    if (code !== 0 || !response || response === 'NO_RESPONSE_NEEDED') {
      log(code !== 0 ? `Agent error (code ${code}): ${stderr.trim()}` : 'Agent decided: NO_RESPONSE_NEEDED');
      return;
    }
    const suffix = content.endsWith('\n') ? formatEntry(id, response) : '\n' + formatEntry(id, response);
    await appendFile(CHATLOG, suffix, 'utf8');
    log('Response written.');
  });

  return { queued: true };
}

function buildPrompt(devId, chatlogContent, rules) {
  return `You are ${devId}, a developer collaborating through a shared chatlog file.

RULES:
${rules.trim()}

CURRENT CHATLOG:
${chatlogContent}

---
Task: Read the chatlog. Decide whether ${devId} should write a response to the most recent entry.
Respond if the most recent entry raises a question or issue ${devId} should address, and ${devId} has not already replied.
Write ONLY the response body — the header is added automatically.
If no response is needed, output exactly: NO_RESPONSE_NEEDED`.trim();
}

function formatEntry(devId, body) {
  const d  = new Date();
  const p2 = n => String(n).padStart(2, '0');
  const ts = `${d.getFullYear()}-${p2(d.getMonth()+1)}-${p2(d.getDate())} ${p2(d.getHours())}:${p2(d.getMinutes())}:${p2(d.getSeconds())}`;
  return `\n### ${ts} - ${devId}\n\n${body.replace(/\r\n/g, '\n').trim()}\n`;
}

// ── Automation engine ─────────────────────────────────────────────────────────

const _pending = new Map();

function evaluateRules(entry) {
  for (const rule of getConfig().automationRules) {
    if (!rule.enabled) continue;
    if (rule.trigger.type === 'entry_from' && entry.author === rule.trigger.author) {
      const key = `${rule.id}:${entry.id}`;
      if (_pending.has(key)) continue;
      const t = setTimeout(() => {
        _pending.delete(key);
        triggerAgent(rule.action.agentId).catch(console.error);
      }, rule.action.delayMs ?? 5000);
      _pending.set(key, t);
    }
  }
}

// ── WebSocket hub ─────────────────────────────────────────────────────────────

const _clients = new Set();

function broadcast(obj) {
  const msg = JSON.stringify(obj);
  for (const ws of _clients) if (ws.readyState === 1) ws.send(msg);
}

// ── HTTP helpers ──────────────────────────────────────────────────────────────

const MIME = {
  '.html': 'text/html;charset=utf-8',
  '.js':   'application/javascript',
  '.css':  'text/css',
  '.json': 'application/json',
  '.ico':  'image/x-icon',
};

const sendJSON = (res, status, body) => {
  const j = JSON.stringify(body);
  res.writeHead(status, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(j) });
  res.end(j);
};

const readBody = req => new Promise((ok, fail) => {
  let d = '';
  req.on('data', c => (d += c));
  req.on('end', () => { try { ok(JSON.parse(d)); } catch { ok(d); } });
  req.on('error', fail);
});

async function serveStatic(res, rel) {
  const safe = rel.replace(/\.\./g, '');
  const full  = resolve(__dir, 'client', safe || 'index.html');
  try {
    const data = await readFile(full);
    res.writeHead(200, { 'Content-Type': MIME[extname(full)] || 'text/plain', 'Content-Length': data.length });
    res.end(data);
  } catch {
    res.writeHead(404); res.end('Not found');
  }
}

// ── REST router ───────────────────────────────────────────────────────────────

async function handleAPI(req, res) {
  const url = req.url.split('?')[0];
  const M   = req.method;

  if (M === 'GET'  && url.startsWith('/api/entries')) {
    const qs     = new URLSearchParams(req.url.split('?')[1] || '');
    const before = qs.get('before') || '';
    const limit  = parseInt(qs.get('limit') || '0', 10);
    let entries  = before ? _lastEntries.filter(e => e.timestamp < before) : _lastEntries;
    if (limit > 0) entries = entries.slice(-limit);
    return sendJSON(res, 200, { entries, total: _lastEntries.length });
  }
  if (M === 'GET'  && url === '/api/agents')   return sendJSON(res, 200, allAgentStatuses());
  if (M === 'GET'  && url === '/api/config')   return sendJSON(res, 200, getConfig());
  if (M === 'GET'  && url === '/api/rules')    return sendJSON(res, 200, getConfig().automationRules);

  if (M === 'POST' && url === '/api/entries') {
    const { author, body } = await readBody(req);
    if (!author || !body) return sendJSON(res, 400, { error: 'author and body required' });
    const raw    = await readFile(chatlogPath(), 'utf8').catch(() => '');
    const suffix = raw.replace(/\r\n/g, '\n').endsWith('\n')
      ? formatEntry(author, body)
      : '\n' + formatEntry(author, body);
    await appendFile(chatlogPath(), suffix, 'utf8');
    return sendJSON(res, 201, { ok: true });
  }

  if (M === 'PUT' && url === '/api/config') {
    return sendJSON(res, 200, await updateConfig(await readBody(req)));
  }

  // /api/agents/:id/:action
  const agentM = url.match(/^\/api\/agents\/([^/]+)\/(start|stop|trigger)$/);
  if (agentM) {
    const id = decodeURIComponent(agentM[1]);
    if (agentM[2] === 'start')   return sendJSON(res, 200, startAgent(id));
    if (agentM[2] === 'stop')    return sendJSON(res, 200, stopAgent(id));
    if (agentM[2] === 'trigger') return sendJSON(res, 200, await triggerAgent(id));
  }

  // /api/rules  POST | /api/rules/:id  PUT | DELETE
  if (M === 'POST' && url === '/api/rules') {
    const rule = { id: randomUUID(), enabled: true, ...await readBody(req) };
    const cfg  = getConfig();
    cfg.automationRules.push(rule);
    await updateConfig({ automationRules: cfg.automationRules });
    return sendJSON(res, 201, rule);
  }

  const ruleM = url.match(/^\/api\/rules\/([^/]+)$/);
  if (ruleM) {
    const id  = ruleM[1];
    const cfg = getConfig();
    const idx = cfg.automationRules.findIndex(r => r.id === id);
    if (idx === -1) return sendJSON(res, 404, { error: 'Rule not found' });
    if (M === 'PUT') {
      cfg.automationRules[idx] = { ...cfg.automationRules[idx], ...await readBody(req) };
      await updateConfig({ automationRules: cfg.automationRules });
      return sendJSON(res, 200, cfg.automationRules[idx]);
    }
    if (M === 'DELETE') {
      cfg.automationRules.splice(idx, 1);
      await updateConfig({ automationRules: cfg.automationRules });
      return sendJSON(res, 200, { ok: true });
    }
  }

  sendJSON(res, 404, { error: 'Not found' });
}

// ── Main ──────────────────────────────────────────────────────────────────────

const PORT = CLI_PORT || getConfig().port;

const httpServer = createServer(async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  if (req.method === 'OPTIONS') { res.writeHead(204); return res.end(); }
  if (req.url.startsWith('/api/')) return handleAPI(req, res).catch(e => sendJSON(res, 500, { error: e.message }));
  const rel = req.url === '/' ? 'index.html' : req.url.slice(1);
  await serveStatic(res, rel);
});

const wss = new WebSocketServer({ server: httpServer });

const INIT_LIMIT = 500; // max entries sent on WebSocket connect; client can page for older ones

wss.on('connection', ws => {
  _clients.add(ws);
  const initEntries = _lastEntries.slice(-INIT_LIMIT);
  ws.send(JSON.stringify({ type: 'init', entries: initEntries, total: _lastEntries.length, agents: allAgentStatuses(), config: getConfig() }));
  ws.on('close',   () => _clients.delete(ws));
  ws.on('error',   () => _clients.delete(ws));
});

setInterval(() => broadcast({ type: 'ping' }), 30_000);

httpServer.listen(PORT, '127.0.0.1', () => {
  const url = `http://localhost:${PORT}`;
  console.log(`\n  devchat-ui  →  ${url}\n`);
  startChatlogWatcher();
  if (OPEN_BROWSER) {
    const cmd = process.platform === 'win32' ? `start "" "${url}"`
              : process.platform === 'darwin' ? `open "${url}"`
              : `xdg-open "${url}"`;
    exec(cmd, err => { if (err) console.warn(`[server] Could not open browser: ${err.message}`); });
  }
});

httpServer.on('error', err => {
  if (err.code === 'EADDRINUSE')
    console.error(`[server] Port ${PORT} already in use — try --port <n>`);
  else
    console.error(`[server] ${err.message}`);
  process.exit(1);
});
