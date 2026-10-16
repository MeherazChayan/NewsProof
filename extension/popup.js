const statusPanel = document.getElementById("status-panel");
const settingsPanel = document.getElementById("settings-panel");
const settingsToggle = document.getElementById("settings-toggle");
const saveSettingsButton = document.getElementById("save-settings");
const refreshButton = document.getElementById("refresh-button");
const openReportLink = document.getElementById("open-report");
const apiBaseUrlInput = document.getElementById("api-base-url");
const webBaseUrlInput = document.getElementById("web-base-url");

settingsToggle.addEventListener("click", async () => {
  settingsPanel.classList.toggle("hidden");
  if (!settingsPanel.classList.contains("hidden")) {
    await loadSettings();
  }
});

saveSettingsButton.addEventListener("click", async () => {
  const response = await sendMessage("SAVE_SETTINGS", {
    apiBaseUrl: apiBaseUrlInput.value,
    webBaseUrl: webBaseUrlInput.value
  });

  if (!response.ok) {
    renderError(response.error);
    return;
  }

  settingsPanel.classList.add("hidden");
  await runVerification();
});

refreshButton.addEventListener("click", async () => {
  await runVerification();
});

document.addEventListener("DOMContentLoaded", async () => {
  await loadSettings();
  await runVerification();
});

async function loadSettings() {
  const response = await sendMessage("GET_SETTINGS");
  if (!response.ok) {
    renderError(response.error);
    return;
  }

  apiBaseUrlInput.value = response.payload.apiBaseUrl;
  webBaseUrlInput.value = response.payload.webBaseUrl;
}

async function runVerification() {
  openReportLink.classList.add("hidden");
  statusPanel.innerHTML = `<p class="muted">Checking the current tab against NewsProof…</p>`;

  const response = await sendMessage("VERIFY_ACTIVE_TAB");
  if (!response.ok) {
    renderError(response.error);
    return;
  }

  renderResult(response.payload);
}

function renderResult(payload) {
  statusPanel.innerHTML = `
    <span class="status-badge ${escapeStatusClass(payload.status)}">${escapeHtml(payload.status)}</span>
    <p class="domain">${escapeHtml(payload.domain)}</p>
    <p class="headline">${escapeHtml(payload.headlineMessage)}</p>
    <p class="reason">${escapeHtml(payload.shortReason)}</p>
  `;

  openReportLink.href = payload.resultUrl;
  openReportLink.classList.remove("hidden");
}

function renderError(message) {
  statusPanel.innerHTML = `
    <span class="status-badge Timeout">Unavailable</span>
    <p class="headline">NewsProof could not verify this page.</p>
    <p class="reason">${escapeHtml(message)}</p>
  `;
}

function sendMessage(type, payload = {}) {
  return chrome.runtime.sendMessage({ type, payload });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function escapeStatusClass(value) {
  return String(value).replaceAll(/[^a-zA-Z ]/g, "").trim();
}
