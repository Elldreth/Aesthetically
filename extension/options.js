const DEFAULTS = { apiUrl: 'http://127.0.0.1:8787', token: '' };

chrome.storage.sync.get(DEFAULTS, (cfg) => {
  document.getElementById('apiUrl').value = cfg.apiUrl;
  document.getElementById('token').value = cfg.token;
});

document.getElementById('save').addEventListener('click', () => {
  const apiUrl = document.getElementById('apiUrl').value.trim() || DEFAULTS.apiUrl;
  const token = document.getElementById('token').value.trim();
  chrome.storage.sync.set({ apiUrl, token }, () => {
    const s = document.getElementById('status');
    s.textContent = 'saved';
    setTimeout(() => (s.textContent = ''), 1500);
  });
});
