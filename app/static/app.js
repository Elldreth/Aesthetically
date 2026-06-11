/* Aesthetically — shared helpers: api(), toast(), initNav(), helpOverlay() */
'use strict';

/* ---------- icons (Lucide-style, 16px strokes, currentColor) ---------- */

const ICONS = {
  thumbsUp: '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7 10v12"/><path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z"/></svg>',
  thumbsDown: '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M17 14V2"/><path d="M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22a3.13 3.13 0 0 1-3-3.88Z"/></svg>',
  helpCircle: '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/></svg>',
  x: '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>',
  undo: '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>',
  chevronLeft: '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m15 18-6-6 6-6"/></svg>',
  chevronRight: '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m9 18 6-6-6-6"/></svg>',
};

/* ---------- api ---------- */

function getCookie(name) {
  const m = document.cookie.match('(?:^|; )' + name + '=([^;]*)');
  return m ? decodeURIComponent(m[1]) : null;
}

async function api(path, body) {
  let opts = {};
  if (body !== undefined) {
    const headers = { 'Content-Type': 'application/json' };
    const token = getCookie('aesth_token');
    if (token) headers['X-Aesth-Token'] = token;
    opts = { method: 'POST', headers, body: JSON.stringify(body) };
  }
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error((await res.text()).slice(0, 300));
  return res.json();
}

/* ---------- toast ---------- */

function toast(message, { action } = {}) {
  let host = document.getElementById('toasts');
  if (!host) {
    host = document.createElement('div');
    host.id = 'toasts';
    document.body.appendChild(host);
  }
  const t = document.createElement('div');
  t.className = 'toast';
  t.setAttribute('role', 'status');
  const msg = document.createElement('span');
  msg.textContent = message;
  t.appendChild(msg);

  const dismiss = () => { clearTimeout(timer); t.remove(); };
  if (action) {
    const b = document.createElement('button');
    b.className = 'toast-action';
    b.textContent = action.label;
    b.addEventListener('click', () => { dismiss(); action.onClick(); });
    t.appendChild(b);
  }
  host.appendChild(t);
  const timer = setTimeout(dismiss, 4000);
  return { dismiss };
}

/* ---------- nav ---------- */

function initNav(activeTab) {
  const tabs = [
    ['rate', 'Rate', '/'],
    ['grid', 'Grid', '/static/grid.html'],
    ['tournament', 'Tournament', '/static/tournament.html'],
    ['studio', 'Studio', '/static/studio.html'],
  ];
  const header = document.createElement('header');
  header.className = 'app-header';

  const brand = document.createElement('span');
  brand.className = 'brand';
  brand.textContent = 'Aesthetically';
  header.appendChild(brand);

  const nav = document.createElement('nav');
  nav.setAttribute('aria-label', 'Main navigation');
  for (const [key, label, href] of tabs) {
    const a = document.createElement('a');
    a.className = 'nav-tab';
    a.href = href;
    a.textContent = label;
    if (key === activeTab) {
      a.classList.add('active');
      a.setAttribute('aria-current', 'page');
    }
    nav.appendChild(a);
  }
  header.appendChild(nav);

  const right = document.createElement('div');
  right.className = 'nav-spacer';
  const add = document.createElement('button');
  add.className = 'btn btn-quiet';
  add.textContent = 'Add folder';
  add.title = 'Register a local image folder (files stay in place)';
  add.addEventListener('click', _openAddFolder);
  right.appendChild(add);
  const help = document.createElement('button');
  help.className = 'btn btn-quiet icon-btn';
  help.setAttribute('aria-label', 'Keyboard shortcuts');
  help.title = 'Keyboard shortcuts (?)';
  help.innerHTML = ICONS.helpCircle;
  help.addEventListener('click', () => _toggleHelp());
  right.appendChild(help);
  header.appendChild(right);

  document.body.prepend(header);
  return header;
}

/* ---------- help overlay ---------- */

let _helpShortcuts = [];
let _helpEl = null;

function _isTyping(e) {
  const t = e.target;
  return t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable);
}

function _closeHelp() {
  if (_helpEl) { _helpEl.remove(); _helpEl = null; }
}

function _openHelp() {
  if (_helpEl) return;
  const backdrop = document.createElement('div');
  backdrop.className = 'overlay-backdrop';
  backdrop.addEventListener('click', e => { if (e.target === backdrop) _closeHelp(); });

  const panel = document.createElement('div');
  panel.className = 'overlay-panel';
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-modal', 'true');
  panel.setAttribute('aria-label', 'Keyboard shortcuts');

  const h = document.createElement('h2');
  h.textContent = 'Keyboard shortcuts';
  panel.appendChild(h);

  for (const [key, desc] of _helpShortcuts) {
    const row = document.createElement('div');
    row.className = 'shortcut-row';
    const k = document.createElement('kbd');
    k.textContent = key;
    const d = document.createElement('span');
    d.className = 'desc';
    d.textContent = desc;
    row.appendChild(d);
    row.appendChild(k);
    panel.appendChild(row);
  }
  const hint = document.createElement('div');
  hint.className = 'close-hint';
  hint.textContent = 'Press Escape to close';
  panel.appendChild(hint);

  backdrop.appendChild(panel);
  document.body.appendChild(backdrop);
  _helpEl = backdrop;
}

function _toggleHelp() { _helpEl ? _closeHelp() : _openHelp(); }

function helpOverlay(shortcuts) {
  _helpShortcuts = shortcuts;
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') { _closeHelp(); return; }
    if (e.key === '?' && !_isTyping(e)) { e.preventDefault(); _toggleHelp(); }
  });
}

/* helpOverlayOpen() — pages can check it to suppress their own key handlers
   while the dialog is up. */
function helpOverlayOpen() { return !!_helpEl; }

/* ---------- add-folder dialog ---------- */

function _openAddFolder() {
  const backdrop = document.createElement('div');
  backdrop.className = 'overlay-backdrop';
  backdrop.addEventListener('click', e => { if (e.target === backdrop) backdrop.remove(); });

  const panel = document.createElement('div');
  panel.className = 'overlay-panel';
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-modal', 'true');
  panel.setAttribute('aria-label', 'Add an image folder');

  const h = document.createElement('h2');
  h.textContent = 'Add an image folder';
  const p = document.createElement('p');
  p.className = 'dim';
  p.style.marginBottom = '12px';
  p.textContent = 'Registers every image in the folder (read-only — nothing is '
    + 'moved or copied), then hashes, embeds and scores them so they join the '
    + 'rating queue. Use a path this machine can see, e.g. a drive letter or UNC share.';
  const input = document.createElement('input');
  input.placeholder = 'D:\\images  or  \\\\server\\share\\folder';
  input.style.width = '100%';
  input.setAttribute('aria-label', 'Folder path');
  const row = document.createElement('div');
  row.style.cssText = 'display:flex;gap:8px;margin-top:12px;justify-content:flex-end';
  const cancel = document.createElement('button');
  cancel.className = 'btn btn-quiet';
  cancel.textContent = 'Cancel';
  cancel.addEventListener('click', () => backdrop.remove());
  const go = document.createElement('button');
  go.className = 'btn btn-primary';
  go.textContent = 'Add';
  go.addEventListener('click', async () => {
    const path = input.value.trim();
    if (!path) return;
    go.disabled = true;
    try {
      await api('/api/ingest_folder', { path });
      backdrop.remove();
      _pollFolderJob();
    } catch (e) {
      go.disabled = false;
      toast('Could not start: ' + e.message);
    }
  });
  input.addEventListener('keydown', e => { if (e.key === 'Enter') go.click(); });
  row.append(cancel, go);
  panel.append(h, p, input, row);
  backdrop.appendChild(panel);
  document.body.appendChild(backdrop);
  input.focus();
}

async function _pollFolderJob() {
  let status = null;
  const timer = setInterval(async () => {
    try {
      const s = await api('/api/ingest_folder/status');
      status?.dismiss();
      if (s.state === 'done') {
        clearInterval(timer);
        toast(`Folder added: ${s.added} new image${s.added === 1 ? '' : 's'}`
          + (s.scored ? ', scored and queued' : '')
          + ' — refresh to see them',
          { action: { label: 'Refresh', onClick: () => location.reload() } });
      } else if (s.state === 'failed') {
        clearInterval(timer);
        toast('Folder ingest failed — check the server log');
      } else {
        status = toast(`${s.state}… ${s.done || 0}/${s.total || '?'}`);
      }
    } catch { /* server briefly busy — keep polling */ }
  }, 2000);
}
