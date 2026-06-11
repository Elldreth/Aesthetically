// Content script: tracks the hovered <img>, shows a subtle outline + badge,
// and sends a/w/d ratings to the service worker. No page DOM is altered
// beyond the floating badge element.
(() => {
  const MIN_SIZE = 120; // ignore icons/avatars
  let hovered = null;

  const badge = document.createElement('div');
  badge.id = 'aesthetically-badge';
  badge.textContent = 'a 👎 · w 🤔 · d 👍';
  document.documentElement.appendChild(badge);

  const BG_URL_RE = /url\(["']?(.*?)["']?\)/;

  // SPAs love transparent overlays on top of images, so the <img> never
  // receives mouse events — look through the hit-test stack instead. Also
  // accepts elements drawn with CSS background-image.
  function findImageAt(x, y) {
    for (const el of document.elementsFromPoint(x, y)) {
      if (el instanceof HTMLImageElement &&
          el.naturalWidth >= MIN_SIZE && el.naturalHeight >= MIN_SIZE) {
        return { el, url: el.currentSrc || el.src, isImg: true };
      }
      if (el instanceof HTMLElement &&
          el.clientWidth >= MIN_SIZE && el.clientHeight >= MIN_SIZE) {
        const m = BG_URL_RE.exec(getComputedStyle(el).backgroundImage || '');
        if (m) return { el, url: new URL(m[1], location.href).href, isImg: false };
      }
    }
    return null;
  }

  function setHover(hit) {
    if (hovered) hovered.el.classList.remove('aesthetically-hover');
    hovered = hit;
    if (hit) {
      hit.el.classList.add('aesthetically-hover');
      const r = hit.el.getBoundingClientRect();
      badge.style.left = Math.max(4, r.left + 6) + 'px';
      badge.style.top = Math.max(4, r.top + 6) + 'px';
      badge.style.display = 'block';
    } else {
      badge.style.display = 'none';
    }
  }

  let raf = 0;
  document.addEventListener('mousemove', (e) => {
    if (raf) return;                      // throttle to one hit-test per frame
    raf = requestAnimationFrame(() => {
      raf = 0;
      const hit = findImageAt(e.clientX, e.clientY);
      if (hit?.el !== hovered?.el) setHover(hit);
    });
  }, { passive: true });

  function feedback(text, ok) {
    badge.textContent = text;
    badge.classList.toggle('aesthetically-err', !ok);
    setTimeout(() => {
      badge.textContent = 'a 👎 · w 🤔 · d 👍';
      badge.classList.remove('aesthetically-err');
    }, 1200);
  }

  function canvasExtract(img) {
    // Pulls exactly the displayed pixels with NO extra network request.
    // Throws on tainted (cross-origin, non-CORS) images — caller falls back.
    const c = document.createElement('canvas');
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    c.getContext('2d').drawImage(img, 0, 0);
    return c.toDataURL('image/png').split(',')[1]; // throws if tainted
  }

  document.addEventListener('keydown', (e) => {
    if (!hovered || e.repeat || e.ctrlKey || e.altKey || e.metaKey) return;
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || e.target.isContentEditable) return;
    const value = { a: 0, w: 0.5, d: 1 }[e.key];
    if (value === undefined) return;

    const imageUrl = hovered.url;
    if (!imageUrl || !/^(https?|file):/.test(imageUrl)) {
      feedback('no fetchable URL', false);
      return;
    }
    let dataB64 = null;
    // canvas only works for real <img> elements; file: origins always taint
    if (hovered.isImg && !imageUrl.startsWith('file:')) {
      try {
        dataB64 = canvasExtract(hovered.el);
      } catch {
        /* tainted canvas — worker will fetch instead */
      }
    }
    badge.textContent = '…';
    chrome.runtime.sendMessage(
      { type: 'rate', imageUrl, pageUrl: location.href, value, dataB64 },
      (res) => {
        if (res && res.ok) {
          feedback(value === 1 ? 'saved 👍' : value === 0.5 ? 'saved 🤔' : 'saved 👎', true);
        } else {
          feedback((res && res.error) || 'failed — app running?', false);
        }
      }
    );
  });
})();
