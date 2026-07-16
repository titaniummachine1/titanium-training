/**
 * Content script on quoridors.com — injects ACE legal oracle + page hook,
 * and relays events to the extension background.
 */
const ACE_URL = chrome.runtime.getURL("inject/ace_legal.js");
const HOOK_URL = chrome.runtime.getURL("inject/page_hook.js");

function injectPageScripts() {
  if (document.documentElement.dataset.quoridorsBridgeInjected) return;
  document.documentElement.dataset.quoridorsBridgeInjected = "1";
  const parent = document.documentElement || document.head || document.body;
  const ace = document.createElement("script");
  ace.src = ACE_URL;
  ace.async = false;
  parent.prepend(ace);
  const hook = document.createElement("script");
  hook.src = HOOK_URL;
  hook.async = false;
  // Insert after ACE so page_hook sees AceLegal.
  ace.after(hook);
}

injectPageScripts();

function relay(type, detail) {
  chrome.runtime.sendMessage({ channel: "quoridors-bridge", type, detail, url: location.href }).catch(() => {});
}

window.addEventListener("quoridors-bridge-state", (ev) => relay("state:update", ev.detail));
window.addEventListener("quoridors-bridge-event", (ev) => relay("socket:event", ev.detail));
window.addEventListener("quoridors-bridge-status", (ev) => relay("socket:status", ev.detail));
window.addEventListener("quoridors-bridge-reject", (ev) => relay("move:rejected", ev.detail));
window.addEventListener("quoridors-bridge-log", (ev) => relay("socket:log", ev.detail));
window.addEventListener("quoridors-bridge-local-heartbeat", (ev) => relay("local:heartbeat", ev.detail));

window.addEventListener("quoridors-bridge-cmd-result", (ev) => {
  relay("cmd:result", ev.detail);
});

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.channel !== "quoridors-bridge-cmd") return;
  const requestId = msg.requestId || `${Date.now()}-${Math.random().toString(16).slice(2)}`;

  let settled = false;
  function finish(payload) {
    if (settled) return;
    settled = true;
    window.removeEventListener("quoridors-bridge-cmd-result", onResult);
    sendResponse(payload);
  }

  function onResult(ev) {
    const d = ev.detail || {};
    if (d.requestId !== requestId) return;
    finish(d);
  }

  window.addEventListener("quoridors-bridge-cmd-result", onResult);
  window.dispatchEvent(
    new CustomEvent("quoridors-bridge-cmd", {
      detail: { ...msg, requestId },
    }),
  );

  setTimeout(() => finish({ ok: false, error: "timeout waiting for page hook" }), 5000);

  return true;
});
