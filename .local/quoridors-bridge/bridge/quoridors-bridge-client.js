/**
 * Drop into any allowed page to talk to the Quoridors Bridge extension.
 *
 * Requires the extension loaded unpacked from .local/quoridors-bridge/extension
 */
(function quoridorsBridgeClient(global) {
  const EXT_ID_KEY = "quoridorsBridgeExtensionId";

  function getExtensionId() {
    return global.localStorage.getItem(EXT_ID_KEY) || global.__QUORIDORS_BRIDGE_EXTENSION_ID__ || "";
  }

  function setExtensionId(id) {
    global.localStorage.setItem(EXT_ID_KEY, id);
  }

  function hasChromeRuntime() {
    return typeof global.chrome !== "undefined" && global.chrome?.runtime?.sendMessage;
  }

  function send(action, extra = {}) {
    const extensionId = getExtensionId();
    if (!extensionId) {
      return Promise.reject(
        new Error("Set localStorage.quoridorsBridgeExtensionId or window.__QUORIDORS_BRIDGE_EXTENSION_ID__"),
      );
    }
    if (!hasChromeRuntime()) {
      return Promise.reject(
        new Error("chrome.runtime not available — page must be allowed in externally_connectable"),
      );
    }
    return new Promise((resolve, reject) => {
      global.chrome.runtime.sendMessage(
        extensionId,
        { channel: "quoridors-bridge-api", action, ...extra },
        (response) => {
          const err = global.chrome.runtime.lastError;
          if (err) reject(new Error(err.message));
          else resolve(response);
        },
      );
    });
  }

  function connectPort(onEvent) {
    const extensionId = getExtensionId();
    if (!extensionId) throw new Error("extension id not configured");
    const port = global.chrome.runtime.connect(extensionId, { name: "quoridors-bridge" });
    port.onMessage.addListener((msg) => {
      if (typeof onEvent === "function") onEvent(msg);
    });
    return port;
  }

  global.QuoridorsBridgeClient = {
    setExtensionId,
    getExtensionId,
    getState: () => send("getState"),
    playMove: (move) => send("playMove", { move }),
    playAlgebraic: (text) => send("playAlgebraic", { text }),
    connectPort,
  };
})(window);
