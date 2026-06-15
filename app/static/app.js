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
  expand: '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M15 3h6v6"/><path d="M9 21H3v-6"/><path d="M21 3l-7 7"/><path d="M3 21l7-7"/></svg>',
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
    ['home', 'Home', '/'],
    ['rate', 'Rate', '/static/index.html'],
    ['grid', 'Grid', '/static/grid.html'],
    ['hands', 'Hands', '/static/rate_hands.html'],
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
  const jobsBtn = document.createElement('button');
  jobsBtn.className = 'btn btn-quiet';
  jobsBtn.id = 'jobs-indicator';
  jobsBtn.title = 'Background jobs';
  jobsBtn.innerHTML = '<span class="jobs-dot" hidden></span><span id="jobs-label">Jobs</span>';
  jobsBtn.addEventListener('click', _toggleJobsPanel);
  right.appendChild(jobsBtn);
  const add = document.createElement('button');
  add.className = 'btn btn-quiet';
  add.textContent = 'Add folder';
  add.title = 'Register a local image folder (files stay in place)';
  add.addEventListener('click', _openAddFolder);
  right.appendChild(add);
  const exp = document.createElement('button');
  exp.className = 'btn btn-quiet';
  exp.textContent = 'Export best';
  exp.title = 'Copy the highest-predicted images to a folder';
  exp.addEventListener('click', _openExportBest);
  right.appendChild(exp);
  const scan = document.createElement('button');
  scan.className = 'btn btn-quiet';
  scan.textContent = 'Score folder';
  scan.title = 'Predict your taste for any folder WITHOUT adding it to the collection';
  scan.addEventListener('click', _openScanFolder);
  right.appendChild(scan);
  const help = document.createElement('button');
  help.className = 'btn btn-quiet icon-btn';
  help.setAttribute('aria-label', 'Keyboard shortcuts');
  help.title = 'Keyboard shortcuts (?)';
  help.innerHTML = ICONS.helpCircle;
  help.addEventListener('click', () => _toggleHelp());
  right.appendChild(help);
  header.appendChild(right);

  document.body.prepend(header);
  _jobsPoll();
  return header;
}

/* ---------- background jobs indicator + panel ---------- */

let _jobsTimer = null;
const KIND_LABEL = { ingest: 'Add folder', scan: 'Score folder', export: 'Export',
                     classify: 'Style tagging' };

function _shortPath(p) {
  if (!p) return '';
  return p.replace(/[\\/]+$/, '').split(/[\\/]/).pop() || p;
}

function _updateIndicator(data) {
  const dot = document.querySelector('#jobs-indicator .jobs-dot');
  const label = document.getElementById('jobs-label');
  if (!label) return;
  if (data.active > 0) {
    dot.hidden = false;
    label.textContent = `Jobs · ${data.active}`;
  } else {
    dot.hidden = true;
    label.textContent = 'Jobs';
  }
}

function _renderJobsPanel(items) {
  const panel = document.getElementById('jobs-panel');
  if (!panel) return;
  const body = panel.querySelector('.jobs-body');
  body.textContent = '';
  if (!items.length) {
    const e = document.createElement('p');
    e.className = 'dim'; e.style.padding = '12px';
    e.textContent = 'No jobs yet.';
    body.appendChild(e);
    return;
  }
  for (const j of items.slice().reverse()) {
    const row = document.createElement('div');
    row.className = 'job-row';
    const title = document.createElement('div');
    title.className = 'job-title';
    const k = document.createElement('span');
    k.textContent = `${KIND_LABEL[j.kind] || j.kind}: ${_shortPath(j.label)}`;
    const st = document.createElement('span');
    st.className = 'badge ' + (j.state === 'done' ? 'ok' : j.state === 'failed' ? 'fail'
      : (j.state === 'running' || j.state === 'queued') ? 'busy' : '');
    st.textContent = j.state === 'running' ? (j.phase || 'running') : j.state;
    title.append(k, st);
    row.appendChild(title);
    if (j.state === 'running' || j.state === 'queued') {
      const bar = document.createElement('div');
      bar.className = 'job-bar';
      const fill = document.createElement('div');
      fill.style.width = j.total ? `${Math.round(100 * j.done / j.total)}%` : '0%';
      bar.appendChild(fill);
      row.appendChild(bar);
      const meta = document.createElement('div');
      meta.className = 'job-meta';
      meta.textContent = j.total ? `${j.phase || ''} ${j.done}/${j.total}` : (j.phase || 'starting…');
      const cancel = document.createElement('button');
      cancel.className = 'toast-action'; cancel.textContent = 'Cancel';
      cancel.addEventListener('click', async () => {
        try { await api(`/api/jobs/${j.id}/cancel`, {}); _jobsPoll(true); } catch {}
      });
      meta.appendChild(cancel);
      row.appendChild(meta);
    } else if (j.error) {
      const m = document.createElement('div');
      m.className = 'job-meta dim'; m.textContent = j.error;
      row.appendChild(m);
    } else if (j.result && j.result.added != null) {
      const m = document.createElement('div');
      m.className = 'job-meta dim'; m.textContent = `added ${j.result.added}`;
      row.appendChild(m);
    } else if (j.result && j.result.count != null) {
      const m = document.createElement('div');
      m.className = 'job-meta dim'; m.textContent = `exported ${j.result.count}`;
      row.appendChild(m);
    }
    body.appendChild(row);
  }
}

async function _jobsPoll(force) {
  if (_jobsTimer) { clearTimeout(_jobsTimer); _jobsTimer = null; }
  let data;
  try { data = await api('/api/jobs'); } catch { return; }
  _updateIndicator(data);
  const panelOpen = !!document.querySelector('#jobs-panel:not([hidden])');
  if (panelOpen) _renderJobsPanel(data.items);
  if (data.active > 0 || panelOpen || force) {
    _jobsTimer = setTimeout(() => _jobsPoll(), 1500);
  }
}

function _toggleJobsPanel() {
  let panel = document.getElementById('jobs-panel');
  if (!panel) {
    panel = document.createElement('div');
    panel.id = 'jobs-panel';
    panel.innerHTML = '<div class="jobs-head">Background jobs'
      + '<button class="btn btn-quiet" id="jobs-close" aria-label="Close">✕</button></div>'
      + '<div class="jobs-body"></div>';
    panel.hidden = true;          // created closed; the toggle below opens it
    document.body.appendChild(panel);
    panel.querySelector('#jobs-close').addEventListener('click', () => { panel.hidden = true; });
  }
  panel.hidden = !panel.hidden;
  if (!panel.hidden) _jobsPoll(true);
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

/* ---------- lightbox ---------- */

/* lightbox(entries, start) — full-size image viewer. entries: [{url, label}].
   Arrow keys / on-screen arrows navigate; Escape or backdrop click closes. */
let _lightboxOpen = false;
function lightbox(entries, start = 0) {
  if (!entries.length || _lightboxOpen) return;
  _lightboxOpen = true;
  let i = Math.max(0, Math.min(entries.length - 1, start));

  const bk = document.createElement('div');
  bk.className = 'overlay-backdrop lightbox';
  const fig = document.createElement('figure');
  fig.className = 'lightbox-fig';
  const img = document.createElement('img');
  const cap = document.createElement('figcaption');
  fig.append(img, cap);
  const prev = document.createElement('button');
  prev.className = 'lb-nav lb-prev'; prev.innerHTML = '‹'; prev.setAttribute('aria-label', 'Previous');
  const next = document.createElement('button');
  next.className = 'lb-nav lb-next'; next.innerHTML = '›'; next.setAttribute('aria-label', 'Next');
  const close = document.createElement('button');
  close.className = 'lb-close'; close.innerHTML = '✕'; close.setAttribute('aria-label', 'Close');
  bk.append(prev, fig, next, close);

  function show() {
    const e = entries[i];
    img.src = e.url; img.alt = e.label || '';
    cap.textContent = (entries.length > 1 ? `${i + 1} / ${entries.length}` : '') +
      (e.label ? `   ${e.label}` : '');
    prev.style.visibility = i > 0 ? 'visible' : 'hidden';
    next.style.visibility = i < entries.length - 1 ? 'visible' : 'hidden';
  }
  function go(d) { const j = i + d; if (j >= 0 && j < entries.length) { i = j; show(); } }
  function done() { bk.remove(); document.removeEventListener('keydown', key, true); _lightboxOpen = false; }
  function key(e) {
    if (e.key === 'Escape') { e.stopPropagation(); done(); }
    else if (e.key === 'ArrowLeft') { e.stopPropagation(); go(-1); }
    else if (e.key === 'ArrowRight') { e.stopPropagation(); go(1); }
  }
  prev.onclick = e => { e.stopPropagation(); go(-1); };
  next.onclick = e => { e.stopPropagation(); go(1); };
  close.onclick = done;
  bk.addEventListener('click', e => { if (e.target === bk) done(); });
  document.addEventListener('keydown', key, true);   // capture: beats page hotkeys
  document.body.appendChild(bk);
  show();
}

function lightboxOpen() { return _lightboxOpen; }

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
      toast('Folder queued — open Jobs to watch progress');
      _jobsPoll(true);
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

/* ---------- score-folder dialog ---------- */

function _openScanFolder() {
  const backdrop = document.createElement('div');
  backdrop.className = 'overlay-backdrop';
  backdrop.addEventListener('click', e => { if (e.target === backdrop) backdrop.remove(); });

  const panel = document.createElement('div');
  panel.className = 'overlay-panel';
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-modal', 'true');
  panel.setAttribute('aria-label', 'Score a folder');

  const h = document.createElement('h2');
  h.textContent = 'Score a folder';
  const p = document.createElement('p');
  p.className = 'dim';
  p.style.marginBottom = '12px';
  p.textContent = 'Predicts how much you would like every image in a folder — '
    + 'nothing is added to your collection or rating queue. Results open in a '
    + 'ranked view where you can export the top picks.';
  const input = document.createElement('input');
  input.placeholder = 'D:\\new-images  or  \\\\server\\share\\folder';
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
  go.textContent = 'Score';
  go.addEventListener('click', async () => {
    const path = input.value.trim();
    if (!path) return;
    go.disabled = true;
    try {
      await api('/api/scan', { path });
      location.href = '/static/scan.html';
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

/* ---------- export-best dialog ---------- */

function _openExportBest() {
  const backdrop = document.createElement('div');
  backdrop.className = 'overlay-backdrop';
  backdrop.addEventListener('click', e => { if (e.target === backdrop) backdrop.remove(); });

  const panel = document.createElement('div');
  panel.className = 'overlay-panel';
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-modal', 'true');
  panel.setAttribute('aria-label', 'Export best images');

  const h = document.createElement('h2');
  h.textContent = 'Export best images';
  const p = document.createElement('p');
  p.className = 'dim';
  p.style.marginBottom = '12px';
  p.textContent = 'Copies the highest-predicted images (named score_id.ext) '
    + 'into a folder on this machine. Originals are never touched.';

  const mkRow = (labelText, control) => {
    const row = document.createElement('label');
    row.style.cssText = 'display:flex;align-items:center;gap:8px;margin:8px 0;font-size:13px';
    const span = document.createElement('span');
    span.style.cssText = 'min-width:110px;color:var(--text-dim)';
    span.textContent = labelText;
    row.append(span, control);
    return row;
  };

  const out = document.createElement('input');
  out.placeholder = 'D:\\keepers';
  out.style.flex = '1';

  const pick = document.createElement('select');
  for (const [v, t] of [['top', 'top N images'], ['min_score', 'score at least…'],
                        ['buckets', 'group everything into score buckets']]) {
    const o = document.createElement('option');
    o.value = v; o.textContent = t;
    pick.appendChild(o);
  }
  const amount = document.createElement('input');
  amount.type = 'number'; amount.value = '200'; amount.style.width = '90px';
  pick.addEventListener('change', () => {
    amount.hidden = pick.value === 'buckets';
    amount.value = pick.value === 'min_score' ? '0.8' : '200';
    amount.step = pick.value === 'min_score' ? '0.05' : '1';
  });

  const unl = document.createElement('input');
  unl.type = 'checkbox'; unl.checked = true;
  const unlRow = mkRow('unrated only', unl);
  unlRow.title = 'only pure predictions — images you have not rated yourself';

  const modeSel = document.createElement('select');
  for (const [v, t] of [['copy', 'copy'], ['link', 'hardlink (no extra disk)'],
                        ['move', 'move']]) {
    const o = document.createElement('option');
    o.value = v; o.textContent = t;
    modeSel.appendChild(o);
  }

  const row = document.createElement('div');
  row.style.cssText = 'display:flex;gap:8px;margin-top:14px;justify-content:flex-end';
  const cancel = document.createElement('button');
  cancel.className = 'btn btn-quiet';
  cancel.textContent = 'Cancel';
  cancel.addEventListener('click', () => backdrop.remove());
  const go = document.createElement('button');
  go.className = 'btn btn-primary';
  go.textContent = 'Export';
  go.addEventListener('click', async () => {
    const outPath = out.value.trim();
    if (!outPath) return;
    const body = { out: outPath, unlabeled_only: unl.checked, mode: modeSel.value };
    if (pick.value === 'top') body.top = +amount.value;
    else if (pick.value === 'min_score') body.min_score = +amount.value;
    else body.buckets = true;
    go.disabled = true;
    try {
      await api('/api/select', body);
      backdrop.remove();
      toast('Export queued — open Jobs to watch progress');
      _jobsPoll(true);
    } catch (e) {
      go.disabled = false;
      toast('Could not start: ' + e.message);
    }
  });
  row.append(cancel, go);

  panel.append(h, p,
    mkRow('output folder', out),
    mkRow('what to take', pick),
    mkRow('amount', amount),
    unlRow,
    mkRow('transfer', modeSel),
    row);
  backdrop.appendChild(panel);
  document.body.appendChild(backdrop);
  out.focus();
}

