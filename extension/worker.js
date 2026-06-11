// Service worker: receives rate requests from the content script and POSTs
// image bytes to the local Aesthetically API.
//
// Security posture:
// - The content script extracts pixels via canvas when it can (no network).
//   We only fetch here when the canvas is tainted (cross-origin, no CORS).
// - Credentials are attached ONLY when the image host matches the page host —
//   cookies apply where the user is actually browsing, and an embedded
//   <img src="https://intranet/..."> on a hostile page gets no credentials.
// - http(s) only; the API token comes from chrome.storage (options page).

const DEFAULTS = { apiUrl: 'http://127.0.0.1:8787', token: '' };

function settings() {
  return new Promise((resolve) => {
    chrome.storage.sync.get(DEFAULTS, resolve);
  });
}

async function toBase64(buf) {
  let s = '';
  const bytes = new Uint8Array(buf);
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }
  return btoa(s);
}

function sameSite(a, b) {
  try {
    return new URL(a).host === new URL(b).host;
  } catch {
    return false;
  }
}

async function getImageB64(msg) {
  if (msg.dataB64) return msg.dataB64; // canvas-extracted in the page
  const url = new URL(msg.imageUrl);
  if (!['http:', 'https:', 'file:'].includes(url.protocol)) {
    throw new Error('unsupported URL scheme');
  }
  // file: needs "Allow access to file URLs" enabled for the extension
  const res = await fetch(msg.imageUrl, {
    credentials: url.protocol !== 'file:' && sameSite(msg.imageUrl, msg.pageUrl)
      ? 'include' : 'omit',
  });
  if (!res.ok) throw new Error('image fetch ' + res.status);
  return toBase64(await res.arrayBuffer());
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type !== 'rate') return;
  (async () => {
    try {
      const cfg = await settings();
      const data_b64 = await getImageB64(msg);
      const headers = { 'Content-Type': 'application/json' };
      if (cfg.token) headers['X-Aesth-Token'] = cfg.token;
      const api = await fetch(cfg.apiUrl.replace(/\/$/, '') + '/api/ingest', {
        method: 'POST',
        headers,
        body: JSON.stringify({
          data_b64,
          image_url: msg.imageUrl,
          page_url: msg.pageUrl,
          value: msg.value,
        }),
      });
      if (api.status === 403) throw new Error('token missing/invalid — set it in extension options');
      if (!api.ok) throw new Error('api ' + api.status);
      sendResponse({ ok: true, ...(await api.json()) });
    } catch (e) {
      sendResponse({ ok: false, error: String(e.message || e) });
    }
  })();
  return true; // async sendResponse
});
