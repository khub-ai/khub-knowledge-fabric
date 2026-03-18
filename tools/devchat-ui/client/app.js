/**
 * app.js — devchat-ui browser client
 *
 * Connects to the server via WebSocket, renders the chatlog in real time,
 * and provides controls for agent management, search, filtering, and composing.
 */

// ── Author colours (matches style.css custom properties) ──────────────────────

const AUTHOR_COLORS = ['#2563eb','#16a34a','#9333ea','#ea580c','#0891b2','#db2777'];
const authorColor = a => AUTHOR_COLORS[parseInt(a.replace('DEV#',''),10) % AUTHOR_COLORS.length];

// ── Constants ─────────────────────────────────────────────────────────────────

const INITIAL_VISIBLE = 20;   // newest N entries expanded on first load; older ones collapsed

// ── State ─────────────────────────────────────────────────────────────────────

let _entries       = [];      // Entry[]
let _agents        = [];      // AgentStatus[]
let _config        = {};
let _collapseState = {};      // id → boolean (true = collapsed)
let _knownIds      = new Set();
let _newSinceView  = 0;
let _ws            = null;
let _scrollLocked  = false;   // user scrolled up
let _logTarget     = null;    // agent id currently in log drawer
let _logLines      = {};      // id → string[]
let _searchQuery   = '';
let _filterAuthors = new Set();   // empty = all
let _filterFrom    = '';
let _filterTo      = '';
let _filterNewOnly = false;
let _filterCode    = false;

// Entries received since page load (for "new only" filter)
const _sessionNewIds = new Set();

// ── DOM refs ──────────────────────────────────────────────────────────────────

const $list         = document.getElementById('chatlog-list');
const $newBadge     = document.getElementById('new-badge');
const $newCount     = document.getElementById('new-count');
const $entryCount   = document.getElementById('entry-count');
const $agentPills   = document.getElementById('agent-pills');
const $agentCards   = document.getElementById('agent-cards');
const $ruleList     = document.getElementById('rule-list');
const $authorFilters= document.getElementById('author-filters');
const $searchInput  = document.getElementById('search-input');
const $searchClear  = document.getElementById('search-clear');
const $logDrawer    = document.getElementById('log-drawer');
const $logContent   = document.getElementById('log-drawer-content');
const $logTitle     = document.getElementById('log-drawer-title');
const $composeArea  = document.getElementById('compose-area');
const $composeBody  = document.getElementById('compose-body');
const $composeAuthor= document.getElementById('compose-author');
const $composePreview= document.getElementById('compose-preview');
const $configModal  = document.getElementById('config-modal');
const $backdrop     = document.getElementById('modal-backdrop');

// ── WebSocket ─────────────────────────────────────────────────────────────────

function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  _ws = new WebSocket(`${proto}://${location.host}`);

  _ws.onmessage = e => {
    const msg = JSON.parse(e.data);
    switch (msg.type) {
      case 'init':
        _config = msg.config;
        handleEntries(msg.entries, false);
        collapseInitialOldEntries();
        handleAgents(msg.agents);
        renderConfig();
        renderAuthorFilters();
        renderRules();
        break;
      case 'entries_added':
        handleEntries(msg.entries, true);
        break;
      case 'agent_status':
        updateAgentStatus(msg);
        break;
      case 'agent_log':
        appendLog(msg.id, msg.line);
        break;
      case 'config_updated':
        _config = msg.config;
        renderConfig();
        break;
      case 'ping':
        _ws.send(JSON.stringify({ type: 'pong' }));
        break;
    }
  };

  _ws.onclose = () => setTimeout(connectWS, 3000);
}

// ── Entry handling ────────────────────────────────────────────────────────────

function handleEntries(entries, isNew) {
  entries.forEach(e => {
    if (_knownIds.has(e.id)) return;
    _knownIds.add(e.id);
    _entries.push(e);
    if (isNew) _sessionNewIds.add(e.id);

    const card = buildCard(e, isNew);
    $list.appendChild(card);

    if (isNew) {
      if (_scrollLocked) {
        _newSinceView++;
        $newCount.textContent = _newSinceView;
        $newBadge.hidden = false;
      } else {
        card.scrollIntoView({ behavior: 'smooth', block: 'end' });
      }
    }

    // Register author filter chip
    if (![..._filterAuthors.keys()].includes(e.author)) ensureAuthorFilter(e.author);
  });

  updateEntryCount();
}

// Collapse all entries except the newest INITIAL_VISIBLE on first load,
// then scroll the list to the bottom so those recent entries are in view.
function collapseInitialOldEntries() {
  const cutoff = _entries.length - INITIAL_VISIBLE;
  _entries.forEach((e, i) => {
    if (i < cutoff) {
      _collapseState[e.id] = true;
      rebuildCard(e.id);
    }
  });
  // Scroll to bottom after collapse so newest entries are visible
  requestAnimationFrame(() => {
    $list.scrollTop = $list.scrollHeight;
  });
}

function handleAgents(agents) {
  _agents = agents;
  renderAgentPills();
  renderAgentCards();
  renderComposeAuthorOptions();
}

function updateAgentStatus(status) {
  const idx = _agents.findIndex(a => a.id === status.id);
  if (idx !== -1) _agents[idx] = { ..._agents[idx], ...status };
  else _agents.push(status);
  renderAgentPills();
  renderAgentCards();
}

// ── Timeago ───────────────────────────────────────────────────────────────────

function parseTs(ts) {
  // ts: "YYYY-MM-DD HH:MM:SS"
  return new Date(ts.replace(' ', 'T'));
}

function timeAgo(ts) {
  const sec = Math.floor((Date.now() - parseTs(ts)) / 1000);
  if (sec < 10)   return 'just now';
  if (sec < 60)   return `${sec}s ago`;
  const min = Math.floor(sec / 60);
  if (min < 60)   return `${min}m ago`;
  const hr  = Math.floor(min / 60);
  if (hr  < 24)   return `${hr}h ago`;
  const day = Math.floor(hr / 24);
  if (day < 7)    return `${day}d ago`;
  return ts.slice(0, 10); // date only
}

function ageClass(ts) {
  const sec = Math.floor((Date.now() - parseTs(ts)) / 1000);
  if (sec < 45)   return 'age-fresh';
  if (sec < 300)  return 'age-new';
  if (sec < 3600) return 'age-recent';
  return '';
}

// Refresh all timeago labels every 20 seconds
setInterval(() => {
  document.querySelectorAll('.entry-ts[data-ts]').forEach(el => {
    el.textContent = timeAgo(el.dataset.ts);
  });
  document.querySelectorAll('.entry-card[data-ts]').forEach(card => {
    const ts = card.dataset.ts;
    ['age-fresh','age-new','age-recent'].forEach(c => card.classList.remove(c));
    const cls = ageClass(ts);
    if (cls) card.classList.add(cls);
    // Update badge
    const badge = card.querySelector('.badge-live, .badge-new');
    if (badge) {
      const sec = Math.floor((Date.now() - parseTs(ts)) / 1000);
      if (sec >= 45) badge.remove();
    }
  });
}, 20_000);

// ── Card builder ──────────────────────────────────────────────────────────────

function buildCard(entry, isNew = false) {
  const { id, timestamp, author, body } = entry;
  const color    = authorColor(author);
  const age      = ageClass(timestamp);
  const snippet  = body.replace(/[#`*_\[\]]/g, '').slice(0, 110).trim();
  const collapsed = _collapseState[id] ?? false;

  const card = document.createElement('div');
  card.className  = `entry-card ${age} ${collapsed ? '' : 'expanded'}`;
  card.id         = `card-${id}`;
  card.dataset.id = id;
  card.dataset.ts = timestamp;
  card.style.borderLeftColor = color;

  // Badge
  let badgeHtml = '';
  if (age === 'age-fresh') badgeHtml = `<span class="badge badge-live">● Live</span>`;
  else if (isNew && age === 'age-new') badgeHtml = `<span class="badge badge-new">New</span>`;

  // Lines count
  const lineCount = body.split('\n').filter(l => l.trim()).length;
  const lineBadge = collapsed ? `<span class="muted" style="font-size:11px;margin-left:auto">▸ ${lineCount} lines</span>` : '';

  card.innerHTML = `
    <div class="entry-header">
      <span class="entry-toggle">${collapsed ? '▶' : '▼'}</span>
      <span class="entry-author" style="color:${color}">${author}</span>
      <span class="entry-ts" data-ts="${timestamp}" title="${timestamp}">${timeAgo(timestamp)}</span>
      ${badgeHtml}
      ${lineBadge}
    </div>
    ${collapsed
      ? `<div class="entry-snippet">${escHtml(snippet)}…</div>`
      : `<div class="entry-body">${renderMarkdown(body)}</div>`
    }
  `;

  card.querySelector('.entry-header').addEventListener('click', () => toggleCard(id));
  return card;
}

function renderMarkdown(md) {
  if (typeof marked !== 'undefined') return marked.parse(md, { breaks: true, gfm: true });
  return `<pre>${escHtml(md)}</pre>`;
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Collapse / expand ─────────────────────────────────────────────────────────

function toggleCard(id) {
  _collapseState[id] = !(_collapseState[id] ?? false);
  rebuildCard(id);
}

function rebuildCard(id) {
  const entry = _entries.find(e => e.id === id);
  if (!entry) return;
  const old   = document.getElementById(`card-${id}`);
  if (!old) return;
  const card  = buildCard(entry, _sessionNewIds.has(id));
  old.replaceWith(card);
  applyFilters();
}

function collapseAll()  { _entries.forEach(e => { _collapseState[e.id] = true;  rebuildCard(e.id); }); }
function expandAll()    { _entries.forEach(e => { _collapseState[e.id] = false; rebuildCard(e.id); }); }
function collapseBeforeDate(dateStr) {
  _entries.filter(e => e.timestamp.slice(0,10) < dateStr)
    .forEach(e => { _collapseState[e.id] = true; rebuildCard(e.id); });
}

document.getElementById('btn-collapse-all').addEventListener('click', collapseAll);
document.getElementById('btn-expand-all').addEventListener('click', expandAll);
document.getElementById('btn-collapse-before').addEventListener('click', () => {
  const d = prompt('Collapse entries before date (YYYY-MM-DD):');
  if (d) collapseBeforeDate(d);
});

// ── Search ────────────────────────────────────────────────────────────────────

function setSearch(val) {
  $searchInput.value = val;
  _searchQuery = val.toLowerCase().trim();
  const hasQ = _searchQuery.length > 0;
  $searchInput.classList.toggle('has-query', hasQ);
  $searchClear.hidden = !hasQ;
  applyFilters();
}

$searchInput.addEventListener('input', e => setSearch(e.target.value));
$searchInput.addEventListener('keydown', e => { if (e.key === 'Escape') setSearch(''); });
$searchClear.addEventListener('click', () => { setSearch(''); $searchInput.focus(); });

// ── Filter sidebar ────────────────────────────────────────────────────────────

const knownAuthors = new Set();

function ensureAuthorFilter(author) {
  if (knownAuthors.has(author)) return;
  knownAuthors.add(author);
  renderAuthorFilters();
  renderComposeAuthorOptions();
}

function renderAuthorFilters() {
  $authorFilters.innerHTML = '';
  [...knownAuthors].forEach(author => {
    const row = document.createElement('div');
    row.className = 'author-filter-row';
    const id = `af-${author}`;
    row.innerHTML = `
      <div class="author-dot" style="background:${authorColor(author)}"></div>
      <label for="${id}">${author}</label>
      <input type="checkbox" id="${id}" checked style="margin-left:auto">
    `;
    row.querySelector('input').addEventListener('change', e => {
      if (e.target.checked) _filterAuthors.delete(author);
      else _filterAuthors.add(author);
      applyFilters();
    });
    $authorFilters.appendChild(row);
  });
}

document.getElementById('filter-from').addEventListener('change', e => { _filterFrom = e.target.value; applyFilters(); });
document.getElementById('filter-to').addEventListener('change',   e => { _filterTo   = e.target.value; applyFilters(); });
document.getElementById('filter-new').addEventListener('change',  e => { _filterNewOnly = e.target.checked; applyFilters(); });
document.getElementById('filter-code').addEventListener('change', e => { _filterCode    = e.target.checked; applyFilters(); });
document.getElementById('btn-clear-filters').addEventListener('click', () => {
  _filterAuthors.clear(); _filterFrom = ''; _filterTo = '';
  _filterNewOnly = false; _filterCode = false;
  setSearch('');
  document.getElementById('filter-from').value = '';
  document.getElementById('filter-to').value   = '';
  document.getElementById('filter-new').checked  = false;
  document.getElementById('filter-code').checked = false;
  document.querySelectorAll('#author-filters input[type=checkbox]').forEach(cb => cb.checked = true);
  applyFilters();
});

function entryMatchesFilters(entry) {
  if (_filterAuthors.size > 0 && _filterAuthors.has(entry.author)) return false;
  const d = entry.timestamp.slice(0,10);
  if (_filterFrom && d < _filterFrom) return false;
  if (_filterTo   && d > _filterTo)   return false;
  if (_filterNewOnly && !_sessionNewIds.has(entry.id)) return false;
  if (_filterCode && !entry.body.includes('```')) return false;
  if (_searchQuery) {
    const hay = (entry.author + ' ' + entry.timestamp + ' ' + entry.body).toLowerCase();
    if (!hay.includes(_searchQuery)) return false;
  }
  return true;
}

function applyFilters() {
  let visible = 0;
  _entries.forEach(entry => {
    const card = document.getElementById(`card-${entry.id}`);
    if (!card) return;
    const show = entryMatchesFilters(entry);
    card.classList.toggle('search-hidden', !show);
    if (show) visible++;

    // Highlight search matches in visible cards
    if (show && _searchQuery) {
      card.querySelectorAll('.entry-body, .entry-snippet').forEach(el => {
        el.innerHTML = highlightQuery(el.innerHTML, _searchQuery);
      });
    }
  });
  updateEntryCount(visible);
}

function highlightQuery(html, q) {
  if (!q) return html;
  // Simple highlight: only operate on text nodes via a replace on the raw HTML
  // This is approximate but safe enough for single-word queries
  const escaped = q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return html.replace(new RegExp(`(${escaped})`, 'gi'), '<mark>$1</mark>');
}

function updateEntryCount(visible) {
  const total = _entries.length;
  const isFiltered = visible !== undefined && visible !== total;
  $entryCount.textContent = isFiltered ? `${visible} / ${total} entries` : `${total} entries`;
  $entryCount.classList.toggle('filtered', isFiltered);
}

// ── Scroll detection ──────────────────────────────────────────────────────────

$list.addEventListener('scroll', () => {
  const { scrollTop, scrollHeight, clientHeight } = $list;
  _scrollLocked = scrollHeight - scrollTop - clientHeight > 180;
  if (!_scrollLocked) {
    _newSinceView = 0;
    $newBadge.hidden = true;
  }
});

$newBadge.addEventListener('click', () => {
  $list.lastElementChild?.scrollIntoView({ behavior: 'smooth' });
  _newSinceView = 0;
  $newBadge.hidden = true;
});

// ── Agent pills + cards ───────────────────────────────────────────────────────

function renderAgentPills() {
  $agentPills.innerHTML = '';
  _agents.forEach(a => {
    const pill = document.createElement('div');
    pill.className = `agent-pill ${a.status === 'running' ? 'running' : ''}`;
    pill.innerHTML = `<span class="dot"></span>${a.id}`;
    $agentPills.appendChild(pill);
  });
}

function renderAgentCards() {
  $agentCards.innerHTML = '';
  const cfgAgents = (_config.agents || []);
  // Merge config + status
  const allIds = new Set([...cfgAgents.map(a => a.id), ..._agents.map(a => a.id)]);

  allIds.forEach(id => {
    const status = _agents.find(a => a.id === id) || { id, status: 'stopped' };
    const running = status.status === 'running';
    const color   = authorColor(id);

    const card = document.createElement('div');
    card.className = `agent-card ${running ? 'running' : ''}`;
    card.id = `agent-card-${id.replace('#','_')}`;

    const lastResp = status.lastResponseAt
      ? `Last response: ${timeAgo(status.lastResponseAt.replace('T',' ').slice(0,19))}`
      : 'No response yet';

    card.innerHTML = `
      <div class="agent-card-header">
        <div class="dot"></div>
        <span class="agent-name" style="color:${color}">${id}</span>
        <span class="muted" style="margin-left:auto;font-size:11px">${running ? 'Running' : 'Stopped'}</span>
      </div>
      <div class="agent-last">${lastResp}</div>
      <div class="agent-actions">
        ${running
          ? `<button class="btn-sm btn-stop" data-id="${id}">■ Stop</button>`
          : `<button class="btn-sm btn-start" data-id="${id}">▶ Start</button>`}
        <button class="btn-sm btn-trigger" data-id="${id}">⚡ Trigger</button>
        <button class="btn-sm btn-log"     data-id="${id}">📄 Log</button>
      </div>
    `;

    card.querySelector('.btn-start,  .btn-stop')?.addEventListener('click', e => {
      const action = e.target.classList.contains('btn-start') ? 'start' : 'stop';
      api('POST', `/api/agents/${encodeURIComponent(id)}/${action}`);
    });
    card.querySelector('.btn-trigger')?.addEventListener('click', () => {
      api('POST', `/api/agents/${encodeURIComponent(id)}/trigger`);
    });
    card.querySelector('.btn-log')?.addEventListener('click', () => openLog(id));

    $agentCards.appendChild(card);
  });
}

// ── Log drawer ────────────────────────────────────────────────────────────────

function openLog(id) {
  _logTarget = id;
  $logTitle.textContent = `Log — ${id}`;
  const lines = (_logLines[id] || []).join('\n');
  $logContent.textContent = lines;
  $logContent.scrollTop   = $logContent.scrollHeight;
  $logDrawer.hidden       = false;
}

function appendLog(id, line) {
  if (!_logLines[id]) _logLines[id] = [];
  _logLines[id].push(line);
  if (_logLines[id].length > 500) _logLines[id].shift();
  if (_logTarget === id) {
    $logContent.textContent += '\n' + line;
    $logContent.scrollTop    = $logContent.scrollHeight;
  }
}

document.getElementById('btn-log-close').addEventListener('click', () => {
  $logDrawer.hidden = true;
  _logTarget = null;
});

// ── Compose ───────────────────────────────────────────────────────────────────

function renderComposeAuthorOptions() {
  $composeAuthor.innerHTML = '';
  const authors = ['DEV#0', ...(_config.agents || []).map(a => a.id), ...[...knownAuthors].filter(a => a !== 'DEV#0')];
  [...new Set(authors)].forEach(a => {
    const opt = document.createElement('option');
    opt.value = a; opt.textContent = a;
    $composeAuthor.appendChild(opt);
  });
}

document.getElementById('btn-compose-toggle').addEventListener('click', () => {
  $composeArea.hidden = !$composeArea.hidden;
  if (!$composeArea.hidden) $composeBody.focus();
});

document.getElementById('btn-preview-toggle').addEventListener('click', () => {
  const show = $composePreview.hidden;
  $composePreview.hidden = !show;
  if (show) $composePreview.innerHTML = renderMarkdown($composeBody.value);
});

$composeBody.addEventListener('input', () => {
  if (!$composePreview.hidden) $composePreview.innerHTML = renderMarkdown($composeBody.value);
});

$composeBody.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') postEntry();
});

document.getElementById('btn-post').addEventListener('click', postEntry);

async function postEntry() {
  const body   = $composeBody.value.trim();
  const author = $composeAuthor.value;
  if (!body) return;
  const res = await api('POST', '/api/entries', { author, body });
  if (res.ok !== false) {
    $composeBody.value      = '';
    $composePreview.innerHTML = '';
    $composeArea.hidden       = true;
  }
}

// ── Automation rules ──────────────────────────────────────────────────────────

function renderRules() {
  $ruleList.innerHTML = '';
  ((_config.automationRules) || []).forEach(rule => {
    const row = document.createElement('div');
    row.className = 'rule-row';
    row.innerHTML = `
      <label>
        <input type="checkbox" ${rule.enabled ? 'checked' : ''}>
        <span>${rule.label || `${rule.trigger?.author} → ${rule.action?.agentId}`}</span>
      </label>
      <button class="btn-rule-del" title="Delete rule" data-id="${rule.id}">✕</button>
    `;
    row.querySelector('input').addEventListener('change', e => {
      api('PUT', `/api/rules/${rule.id}`, { enabled: e.target.checked });
    });
    row.querySelector('.btn-rule-del').addEventListener('click', async () => {
      if (confirm(`Delete rule "${rule.label}"?`)) {
        await api('DELETE', `/api/rules/${rule.id}`);
        const cfg = await api('GET', '/api/config');
        _config = cfg;
        renderRules();
      }
    });
    $ruleList.appendChild(row);
  });
}

document.getElementById('btn-add-rule').addEventListener('click', async () => {
  const from    = prompt('Trigger: entry from which author? (e.g. DEV#0)');
  const target  = prompt('Action: which agent should respond? (e.g. DEV#1)');
  const delayMs = parseInt(prompt('Delay in seconds before triggering (default 5):') || '5', 10) * 1000;
  if (!from || !target) return;
  const label = `${from} → ${target}`;
  await api('POST', '/api/rules', {
    label,
    enabled: true,
    trigger: { type: 'entry_from', author: from },
    action:  { type: 'trigger_agent', agentId: target, delayMs },
  });
  const cfg = await api('GET', '/api/config');
  _config = cfg;
  renderRules();
});

// ── Config modal ──────────────────────────────────────────────────────────────

function renderConfig() {
  document.getElementById('cfg-chatlog').value = _config.chatlog || '';
  document.getElementById('cfg-rules').value   = _config.rules   || '';
  document.getElementById('cfg-port').value    = _config.port    || 3737;
}

document.getElementById('btn-config').addEventListener('click', () => {
  renderConfig();
  $configModal.hidden = false;
  $backdrop.hidden    = false;
});

function closeConfigModal() {
  $configModal.hidden = true;
  $backdrop.hidden    = true;
}

document.getElementById('btn-config-cancel').addEventListener('click', closeConfigModal);
$backdrop.addEventListener('click', closeConfigModal);

document.getElementById('btn-config-save').addEventListener('click', async () => {
  const patch = {
    chatlog: document.getElementById('cfg-chatlog').value,
    rules:   document.getElementById('cfg-rules').value,
    port:    parseInt(document.getElementById('cfg-port').value, 10),
  };
  _config = await api('PUT', '/api/config', patch);
  closeConfigModal();
});

// ── API helper ────────────────────────────────────────────────────────────────

async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body) { opts.body = JSON.stringify(body); opts.headers['Content-Type'] = 'application/json'; }
  const res  = await fetch(path, opts);
  return res.json().catch(() => ({}));
}

// ── Init ──────────────────────────────────────────────────────────────────────

connectWS();
