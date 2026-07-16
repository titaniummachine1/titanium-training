/** @typedef {{event:string,args:unknown[],ackId:number|null,direction:string}} SocketEventDetail */

const ports = new Set();
/** @type {any} */
let latestState = null;
/** @type {SocketEventDetail[]} */
let recentEvents = [];
/** @type {any} */
let lastReject = null;
let siteTabId = null;
let socketConnected = false;
const LOG_KEY = "quoridorsBridgeLogs";
const MAX_LOGS = 500;
const MAX_LOG_DETAIL_CHARS = 8_000;
const LOG_PERSIST_DEBOUNCE_MS = 1_000;
const memoryLogs = [];
let logsReadyResolve;
const logsReady = new Promise((resolve) => {
  logsReadyResolve = resolve;
});
let persistTimer = null;
let persistInFlight = false;
let logsDirty = false;

chrome.storage.local.get(LOG_KEY, (data) => {
  const storedLogs = Array.isArray(data?.[LOG_KEY]) ? data[LOG_KEY] : [];
  memoryLogs.push(...storedLogs.slice(-MAX_LOGS));
  while (memoryLogs.length > MAX_LOGS) memoryLogs.shift();
  logsReadyResolve();
});

function schedulePersistLogs() {
  logsDirty = true;
  if (persistTimer || persistInFlight) return;
  persistTimer = setTimeout(() => {
    persistTimer = null;
    persistLogs();
  }, LOG_PERSIST_DEBOUNCE_MS);
}

function persistLogs() {
  if (persistInFlight || !logsDirty) return;
  logsDirty = false;
  persistInFlight = true;
  const snapshot = memoryLogs.slice();
  try {
    chrome.storage.local.set({ [LOG_KEY]: snapshot }, () => {
      persistInFlight = false;
      if (logsDirty) schedulePersistLogs();
    });
  } catch {
    persistInFlight = false;
    /* storage unavailable during extension shutdown */
  }
}

function slimStateDetail(detail) {
  if (!detail || typeof detail !== "object") return detail;
  const {
    legal_wall_placements,
    legal_pawn_moves,
    legalWallPlacements,
    legalPawnMoves,
    aceLegalMoves,
    ...rest
  } = detail;
  return {
    ...rest,
    legal_pawn_moves_count: Array.isArray(legal_pawn_moves) ? legal_pawn_moves.length : undefined,
    legal_wall_placements_count: Array.isArray(legal_wall_placements) ? legal_wall_placements.length : undefined,
    aceLegalMoves_count: Array.isArray(aceLegalMoves) ? aceLegalMoves.length : undefined,
  };
}

function addLog(type, detail) {
  let safeDetail = detail;
  try {
    const serialized = JSON.stringify(detail);
    if (serialized && serialized.length > MAX_LOG_DETAIL_CHARS) {
      safeDetail = {
        truncated: true,
        preview: serialized.slice(0, MAX_LOG_DETAIL_CHARS),
      };
    }
  } catch {
    safeDetail = { truncated: true, preview: String(detail) };
  }
  const entry = { ts: new Date().toISOString(), type, detail: safeDetail };
  memoryLogs.push(entry);
  while (memoryLogs.length > MAX_LOGS) memoryLogs.shift();
  schedulePersistLogs();
}

async function getStoredLogs() {
  await logsReady;
  return memoryLogs.slice();
}

async function exportLogs() {
  const logs = await getStoredLogs();
  const payload = {
    exportedAt: new Date().toISOString(),
    extensionId: chrome.runtime.id,
    siteTabId,
    socketConnected,
    latestState,
    lastReject,
    logs,
  };
  const url = `data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(payload, null, 2))}`;
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const downloadId = await chrome.downloads.download({
    url,
    filename: `quoridors-bridge-logs/quoridors-bridge-${stamp}.json`,
    saveAs: false,
    conflictAction: "uniquify",
  });
  return { ok: true, downloadId, count: logs.length };
}

function pushEvent(type, detail) {
  const safeDetail = type === "state:update" ? slimStateDetail(detail) : detail;
  const entry = { ts: Date.now(), type, detail: safeDetail };
  const isSuccessfulCmdResult =
    type === "cmd:result" && !detail?.error && detail?.ok !== false;
  if (type !== "local:heartbeat" && !isSuccessfulCmdResult) addLog(type, safeDetail);
  if (type === "state:update") latestState = safeDetail;
  if (type === "move:rejected") lastReject = detail;
  if (type === "socket:event" || (type === "cmd:result" && !isSuccessfulCmdResult)) {
    recentEvents.push(entry);
    if (recentEvents.length > 100) recentEvents.shift();
  }
  if (type === "socket:status") socketConnected = Boolean(detail?.connected);
  for (const port of ports) {
    try {
      port.postMessage({ channel: "quoridors-bridge", ...entry });
    } catch {
      ports.delete(port);
    }
  }
}

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== "quoridors-bridge") return;
  ports.add(port);
  port.postMessage({
    channel: "quoridors-bridge",
    type: "hello",
    detail: {
      extensionId: chrome.runtime.id,
      latestState,
      recentEvents: recentEvents.slice(-30),
      lastReject,
      socketConnected,
      siteTabId,
    },
  });
  port.onDisconnect.addListener(() => ports.delete(port));
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.channel === "quoridors-bridge") {
    if (sender.tab?.id) siteTabId = sender.tab.id;
    pushEvent(msg.type, msg.detail);
    sendResponse({ ok: true });
    return false;
  }

  if (msg?.channel === "quoridors-bridge-api") {
    handleApi(msg, sendResponse);
    return true;
  }

  // Offscreen streams engine info/progress here; fan out to quoridors tabs.
  if (msg?.channel === "quoridors-bridge-engine-progress") {
    void forwardEngineProgress(msg);
    return false;
  }

  if (msg?.channel === "quoridors-bridge-engine" && sender.tab) {
    handleEngineRequest(msg, sendResponse);
    return true;
  }

  // Allow popup / extension pages to warm without a tab sender.
  if (msg?.channel === "quoridors-bridge-engine") {
    handleEngineRequest(msg, sendResponse);
    return true;
  }

  return false;
});

// ---------------------------------------------------------------------
// Titanium engine: relay content-script requests to the offscreen
// document, which is the only place a chrome-extension:// Worker can be
// constructed from a quoridors.com tab (Workers must share origin with the
// page that spawns them; the content script's page origin is quoridors.com).
// ---------------------------------------------------------------------

const OFFSCREEN_URL = "offscreen/offscreen.html";
let creatingOffscreen = null;

async function ensureOffscreenDocument() {
  if (creatingOffscreen) {
    await creatingOffscreen;
    return;
  }
  creatingOffscreen = chrome.offscreen
    .createDocument({
      url: OFFSCREEN_URL,
      reasons: ["WORKERS"],
      justification: "Host the Titanium WASM engine worker at the extension origin.",
    })
    .catch((err) => {
      const message = String(err?.message || err);
      if (!message.includes("single offscreen document")) throw err;
    });
  try {
    await creatingOffscreen;
  } finally {
    creatingOffscreen = null;
  }
}

async function closeOffscreenDocument() {
  try {
    await chrome.offscreen.closeDocument();
  } catch {
    /* no offscreen document, or Chrome already closed it */
  } finally {
    creatingOffscreen = null;
  }
}

function relayTimeoutForOp(msg) {
  const op = msg.op;
  // analyze responds immediately ({ analyzing: true }) — short relay budget.
  if (op === "cancel" || op === "sync" || op === "analyze") return 5_000;
  if (op === "warm" || op === "init") return 65_000;
  // Play search: relay budget ≈ engine timeMs + short grace (not 45s hang).
  return Math.max(5_000, Number(msg.timeMs) || 3_000) + 10_000;
}

function buildOffscreenPayload(msg) {
  return {
    channel: "quoridors-bridge-engine-offscreen",
    op: msg.op,
    algebraicMoves: msg.algebraicMoves,
    timeMs: msg.timeMs,
    maxDepth: msg.maxDepth,
    engineMode: msg.engineMode,
    catLmrCeiling: msg.catLmrCeiling,
    threads: msg.threads,
    isFreshGame: msg.isFreshGame,
    forceFullSync: msg.forceFullSync,
    forceReset: msg.forceReset,
    multipv: msg.multipv,
    rootScores: msg.rootScores,
    reason: msg.reason,
  };
}

function sendToOffscreen(payload) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(payload, (response) => {
      const err = chrome.runtime.lastError;
      if (err) {
        resolve({ ok: false, error: err.message || String(err) });
        return;
      }
      resolve(response);
    });
  });
}

async function softCancelOffscreen() {
  try {
    await ensureOffscreenDocument();
    return await sendToOffscreen({
      channel: "quoridors-bridge-engine-offscreen",
      op: "cancel",
    });
  } catch (err) {
    addLog("engine:cancel-failed", { message: String(err?.message || err) });
    return { ok: false, error: String(err?.message || err) };
  }
}

async function handleEngineRequest(msg, sendResponse) {
  let settled = false;
  const finish = (payload) => {
    if (settled) return;
    settled = true;
    sendResponse(payload);
  };
  try {
    addLog("engine:request", {
      op: msg.op,
      moves: Array.isArray(msg.algebraicMoves) ? msg.algebraicMoves.length : undefined,
      timeMs: msg.timeMs,
      maxDepth: msg.maxDepth,
      engineMode: msg.engineMode,
      catLmrCeiling: msg.catLmrCeiling,
      threads: msg.threads,
      multipv: msg.multipv,
      rootScores: msg.rootScores,
      reason: msg.reason,
    });
    if (msg.op === "reset") {
      await closeOffscreenDocument();
      addLog("engine:reset", { reason: msg.reason });
      finish({ ok: true, reset: true });
      return;
    }
    await ensureOffscreenDocument();
    const relayTimeoutMs = relayTimeoutForOp(msg);
    const softOps =
      msg.op === "cancel" ||
      msg.op === "sync" ||
      msg.op === "warm" ||
      msg.op === "init" ||
      msg.op === "analyze";
    const result = await new Promise((resolve) => {
      const timeoutId = setTimeout(async () => {
        addLog("engine:relay-timeout", {
          op: msg.op,
          timeMs: msg.timeMs,
          soft: softOps,
        });
        if (softOps) {
          // Keep the warm worker; only log the soft-op timeout.
          resolve({ ok: false, error: `engine ${msg.op} timed out` });
          return;
        }
        // Search timeout: soft-cancel first so TT / thread pool stay warm.
        // Only close the offscreen document if cancel itself cannot be delivered.
        const cancelResult = await softCancelOffscreen();
        if (cancelResult?.ok === false) {
          addLog("engine:relay-timeout-close", {
            op: msg.op,
            cancelError: cancelResult.error,
          });
          await closeOffscreenDocument();
        }
        resolve({ ok: false, error: "engine relay timed out" });
      }, relayTimeoutMs);
      chrome.runtime.sendMessage(buildOffscreenPayload(msg), (response) => {
        clearTimeout(timeoutId);
        const err = chrome.runtime.lastError;
        if (err) {
          resolve({ ok: false, error: err.message || String(err) });
          return;
        }
        resolve(response);
      });
    });
    addLog(result?.ok === false ? "engine:response:error" : "engine:response", result);
    finish(result);
  } catch (err) {
    addLog("engine:exception", { message: String(err?.message || err), stack: err?.stack });
    finish({ ok: false, error: String(err?.message || err) });
  }
}

chrome.runtime.onMessageExternal.addListener((msg, _sender, sendResponse) => {
  if (msg?.channel !== "quoridors-bridge-api") return;
  handleApi(msg, sendResponse);
  return true;
});

async function findQuoridorsTab() {
  if (siteTabId != null) {
    try {
      const tab = await chrome.tabs.get(siteTabId);
      if (tab?.url?.includes("quoridors.com")) return siteTabId;
    } catch {
      siteTabId = null;
    }
  }
  const tabs = await chrome.tabs.query({ url: ["https://www.quoridors.com/*", "https://quoridors.com/*"] });
  siteTabId = tabs[0]?.id ?? null;
  return siteTabId;
}

/** Fan out streamed engine progress from offscreen to every quoridors.com tab. */
let lastProgressLogAt = 0;
let lastProgressLogKey = "";

async function forwardEngineProgress(msg) {
  // Throttled progress samples so exports can prove ghosts/rootMoves streamed.
  const now = Date.now();
  const rootN = Array.isArray(msg.rootMoves) ? msg.rootMoves.length : 0;
  const multiN = Array.isArray(msg.multiPv) ? msg.multiPv.length : 0;
  const key = `${msg.kind}|${msg.depth}|${msg.algebraic}|${rootN}|${multiN}`;
  if (key !== lastProgressLogKey || now - lastProgressLogAt > 2000) {
    lastProgressLogKey = key;
    lastProgressLogAt = now;
    addLog("engine:progress", {
      kind: msg.kind,
      algebraic: msg.algebraic,
      depth: msg.depth,
      nodes: msg.nodes,
      rootScore: msg.rootScore,
      rootMoves: rootN,
      multiPv: multiN,
      top: Array.isArray(msg.multiPv)
        ? msg.multiPv.slice(0, 3).map((e) => e?.move)
        : Array.isArray(msg.rootMoves)
          ? msg.rootMoves.slice(0, 3).map((e) => e?.move)
          : [],
    });
  }

  let tabs;
  try {
    tabs = await chrome.tabs.query({
      url: ["https://www.quoridors.com/*", "https://quoridors.com/*"],
    });
  } catch {
    return;
  }
  for (const tab of tabs) {
    if (tab?.id == null) continue;
    try {
      await chrome.tabs.sendMessage(tab.id, {
        channel: "quoridors-bridge-engine-progress",
        kind: msg.kind,
        algebraic: msg.algebraic,
        rootScore: msg.rootScore,
        depth: msg.depth,
        whiteDist: msg.whiteDist,
        blackDist: msg.blackDist,
        nodes: msg.nodes,
        pv: msg.pv,
        depthLog: msg.depthLog,
        rootMoves: msg.rootMoves,
        multiPv: msg.multiPv,
      });
    } catch {
      /* tab closed or content script not ready — ignore */
    }
  }
}

async function handleApi(msg, sendResponse) {
  const action = msg.action;
  try {
    if (action === "getState") {
      sendResponse({
        ok: true,
        latestState,
        recentEvents: recentEvents.slice(-30),
        lastReject,
        socketConnected,
        siteTabId,
      });
      return;
    }

    if (action === "exportLogs") {
      sendResponse(await exportLogs());
      return;
    }

    if (action === "playMove" || action === "playAlgebraic") {
      const tabId = await findQuoridorsTab();
      if (!tabId) throw new Error("No quoridors.com tab open");
      const result = await chrome.tabs.sendMessage(tabId, {
        channel: "quoridors-bridge-cmd",
        action,
        move: msg.move,
        text: msg.text,
        requestId: msg.requestId,
      });
      sendResponse(result);
      return;
    }

    sendResponse({ ok: false, error: `unknown action: ${action}` });
  } catch (err) {
    sendResponse({ ok: false, error: String(err?.message || err) });
  }
}
