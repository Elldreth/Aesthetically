// Service worker: receives {imageUrl, pageUrl, value} from the content script,
// fetches the image bytes (page credentials apply via host_permissions, which
// defeats hotlink protection), and POSTs base64 to the local Aesthetically API.
const API = 'http://127.0.0.1:8787/api/ingest';

async function toBase64(buf) {
  let s = '';
  const bytes = new Uint8Array(buf);
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }
  return btoa(s);
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type !== 'rate') return;
  (async () => {
    try {
      const res = await fetch(msg.imageUrl, { credentials: 'include' });
      if (!res.ok) throw new Error('image fetch ' + res.status);
      const data_b64 = await toBase64(await res.arrayBuffer());
      const api = await fetch(API, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data_b64,
          image_url: msg.imageUrl,
          page_url: msg.pageUrl,
          value: msg.value,
        }),
      });
      if (!api.ok) throw new Error('api ' + api.status + ': ' + await api.text());
      sendResponse({ ok: true, ...(await api.json()) });
    } catch (e) {
      sendResponse({ ok: false, error: String(e) });
    }
  })();
  return true; // async sendResponse
});
