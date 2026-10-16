const DEFAULTS = {
  apiBaseUrl: "http://127.0.0.1:8000/api/v1",
  webBaseUrl: "http://127.0.0.1:3000"
};

const CACHE_TTL_MS = 60 * 1000;
const cache = new Map();

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "open-newsproof-report",
    title: "Open NewsProof",
    contexts: ["action"]
  });
});

chrome.contextMenus.onClicked.addListener(async (info) => {
  if (info.menuItemId !== "open-newsproof-report") {
    return;
  }
  const { webBaseUrl } = await getSettings();
  await chrome.tabs.create({ url: webBaseUrl });
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "VERIFY_ACTIVE_TAB") {
    verifyActiveTab()
      .then((payload) => sendResponse({ ok: true, payload }))
      .catch((error) => sendResponse({ ok: false, error: error.message || "Verification failed" }));
    return true;
  }

  if (message?.type === "GET_SETTINGS") {
    getSettings()
      .then((settings) => sendResponse({ ok: true, payload: settings }))
      .catch((error) => sendResponse({ ok: false, error: error.message || "Could not load settings" }));
    return true;
  }

  if (message?.type === "SAVE_SETTINGS") {
    saveSettings(message.payload || {})
      .then(() => sendResponse({ ok: true }))
      .catch((error) => sendResponse({ ok: false, error: error.message || "Could not save settings" }));
    return true;
  }

  return false;
});

async function verifyActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const url = tab?.url;

  if (!url || !/^https?:\/\//.test(url)) {
    throw new Error("Open a standard http or https page before running NewsProof.");
  }

  const cached = cache.get(url);
  if (cached && Date.now() - cached.createdAt < CACHE_TTL_MS) {
    return cached.payload;
  }

  const { apiBaseUrl, webBaseUrl } = await getSettings();
  const response = await fetch(`${apiBaseUrl}/verify/url?fast=true`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url })
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `NewsProof API returned ${response.status}`);
  }

  const result = await response.json();
  const payload = {
    pageUrl: url,
    domain: new URL(url).hostname,
    status: result.status,
    shortReason: result.short_reason,
    headlineMessage: result.headline_message,
    jobId: result.job_id,
    resultUrl: `${webBaseUrl}/results/${result.job_id}`
  };

  cache.set(url, { createdAt: Date.now(), payload });
  return payload;
}

async function getSettings() {
  const stored = await chrome.storage.sync.get(Object.keys(DEFAULTS));
  return {
    apiBaseUrl: normalizeBase(stored.apiBaseUrl || DEFAULTS.apiBaseUrl),
    webBaseUrl: normalizeBase(stored.webBaseUrl || DEFAULTS.webBaseUrl)
  };
}

async function saveSettings(payload) {
  const nextSettings = {
    apiBaseUrl: normalizeBase(payload.apiBaseUrl || DEFAULTS.apiBaseUrl),
    webBaseUrl: normalizeBase(payload.webBaseUrl || DEFAULTS.webBaseUrl)
  };
  await chrome.storage.sync.set(nextSettings);
}

function normalizeBase(value) {
  return String(value || "").trim().replace(/\/+$/, "");
}
