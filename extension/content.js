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

  function setHover(img) {
    if (hovered) hovered.classList.remove('aesthetically-hover');
    hovered = img;
    if (img) {
      img.classList.add('aesthetically-hover');
      const r = img.getBoundingClientRect();
      badge.style.left = Math.max(4, r.left + 6) + 'px';
      badge.style.top = Math.max(4, r.top + 6) + 'px';
      badge.style.display = 'block';
    } else {
      badge.style.display = 'none';
    }
  }

  document.addEventListener('mouseover', (e) => {
    const img = e.target instanceof HTMLImageElement ? e.target : null;
    if (img && img.naturalWidth >= MIN_SIZE && img.naturalHeight >= MIN_SIZE) {
      setHover(img);
    }
  }, { passive: true });

  document.addEventListener('mouseout', (e) => {
    if (e.target === hovered) setHover(null);
  }, { passive: true });

  function feedback(text, ok) {
    badge.textContent = text;
    badge.classList.toggle('aesthetically-err', !ok);
    setTimeout(() => {
      badge.textContent = 'a 👎 · w 🤔 · d 👍';
      badge.classList.remove('aesthetically-err');
    }, 1200);
  }

  document.addEventListener('keydown', (e) => {
    if (!hovered || e.repeat || e.ctrlKey || e.altKey || e.metaKey) return;
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || e.target.isContentEditable) return;
    const value = { a: 0, w: 0.5, d: 1 }[e.key];
    if (value === undefined) return;

    const imageUrl = hovered.currentSrc || hovered.src;
    if (!imageUrl || imageUrl.startsWith('data:')) {
      feedback('no fetchable URL', false);
      return;
    }
    badge.textContent = '…';
    chrome.runtime.sendMessage(
      { type: 'rate', imageUrl, pageUrl: location.href, value },
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
