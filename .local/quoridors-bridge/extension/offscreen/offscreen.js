/**
 * Runs at the extension's own origin (chrome-extension://<id>/offscreen/…).
 * Hosts the one Titanium WASM worker for the whole extension and answers
 * `quoridors-bridge-engine-offscreen` messages relayed by the background
 * service worker on behalf of content scripts.
 *
 * Keep-hot: the worker stays alive between searches. Soft cancel prefers
 * `op: "cancel"` over terminate; full algebraic history travels with each
 * search, while `op: "sync"` is reserved for recovery.
 *
 * Multiplex: a single worker serves both play (`op: "search"`) and always-on
 * analysis (`op: "analyze"`). Analysis responds immediately and streams
 * progress; play cancels analysis first and awaits bestmove.
 */

const ENGINE_MODE = "titanium-v17";
const CAT_LMR_CEILING = 1000;
const THREADS_SAFETY_CAP = 32;
const INIT_TIMEOUT_MS = 60_000;
const CANCEL_CLEAR_MS = 1_000;
/** Play must preempt analyze quickly — long clear windows caused hung searches. */
const PLAY_PREEMPT_CLEAR_MS = 400;
const PLAY_PREEMPT_SETTLE_MS = 140;
const PLAY_TIMEOUT_GRACE_MS = 2_500;
const PLAY_TIMEOUT_FLOOR_MS = 3_000;
const PLAY_TIMEOUT_PARTIAL_WAIT_MS = 400;
const ANALYZE_TIME_MS = 86_400_000;
const ANALYZE_RESTART_MS = 250;
const MAX_DEPTH_LOG_ENTRIES = 64;

let engineWorker = null;
let engineReady = false;
let readyWaiter = null;
let pendingSearch = null;
/** @type {"play" | "analyze" | null} */
let searchKind = null;
/** Serialize sync/cancel/search/analyze so they never overlap on the single worker. */
let opQueue = Promise.resolve();
/** Last streamed progress move during the in-flight search (best-so-far). */
let lastSearchPartial = null;
let lastEngineMode = ENGINE_MODE;
let lastCatLmrCeiling = CAT_LMR_CEILING;
let lastThreads = 1;
let warnedCrossOriginIsolation = false;
/**
 * Monotonic search generation. Worker echoes `seq` on info/bestmove; offscreen
 * ignores messages whose seq does not match pendingSearch.seq so a stale
 * analyze bestmove cannot resolve a new play search.
 */
let activeSearchSeq = 0;

function boundedDepthLog(value) {
  return Array.isArray(value) ? value.slice(-MAX_DEPTH_LOG_ENTRIES) : [];
}

/** Analysis session state (site analysisEngineSession parity). */
let analyzeGen = 0;
let analyzeFingerprint = null;
let analyzeRestartTimer = null;
let analyzeLastOpts = null;
let analyzeLastMoves = null;
let analyzeSyncSkipLogged = false;

function resolveThreads(requested) {
  if (typeof crossOriginIsolated !== "undefined" && !crossOriginIsolated) {
    if (!warnedCrossOriginIsolation) {
      warnedCrossOriginIsolation = true;
      console.warn("[quoridors-bridge] SharedArrayBuffer unavailable; using 1 engine thread");
    }
    return 1;
  }
  const hardware = hardwareThreadCount();
  const value = Number(requested);
  if (requested == null || !Number.isFinite(value) || value <= 0) {
    return hardware;
  }
  return Math.max(1, Math.min(hardware, value));
}

function hardwareThreadCount() {
  return Math.max(
    1,
    Math.min(THREADS_SAFETY_CAP, Number(navigator.hardwareConcurrency) || 8),
  );
}

function fingerprintMoves(algebraicMoves) {
  return (Array.isArray(algebraicMoves) ? algebraicMoves : []).join(" ");
}

function firstMoveFromPv(pv) {
  if (Array.isArray(pv)) {
    return pv.length ? String(pv[0] || "").trim() : "";
  }
  return String(pv || "")
    .trim()
    .split(/\s+/)[0] || "";
}

function extractPartialMove(data) {
  if (!data || typeof data !== "object") return null;
  const direct = data.algebraicMove || data.rootMove || data.move || data.bestmove;
  if (typeof direct === "string" && direct.trim() && direct !== "(none)") {
    return direct.trim().split(/\s+/)[0];
  }
  const fromPv = firstMoveFromPv(data.pv);
  if (fromPv && fromPv !== "(none)") return fromPv;
  const log = boundedDepthLog(data.depthLog);
  if (log.length) {
    const last = log[log.length - 1];
    const fromLog = firstMoveFromPv(last?.pv);
    if (fromLog && fromLog !== "(none)") return fromLog;
  }
  return null;
}

function rememberSearchPartial(data) {
  const algebraicMove = extractPartialMove(data);
  if (!algebraicMove && lastSearchPartial == null) {
    // Still remember score/depth even before a PV appears.
    lastSearchPartial = {
      algebraicMove: null,
      rootScore: data.rootScore,
      depth: data.searchDepth ?? data.depth,
      whiteDist: data.whiteDist,
      blackDist: data.blackDist,
      nodes: data.totalNodesAcrossWorkers ?? data.totalNodes ?? data.nodes,
      depthLog: boundedDepthLog(data.depthLog),
      rootMoves: data.rootMoves,
      multiPv: data.multiPv,
      pv: data.pv,
      searchWallMs: data.elapsedMs ?? data.searchWallMs,
    };
    return;
  }
  lastSearchPartial = {
    algebraicMove: algebraicMove || lastSearchPartial?.algebraicMove,
    rootScore: data.rootScore ?? lastSearchPartial?.rootScore,
    depth: data.searchDepth ?? data.depth ?? lastSearchPartial?.depth,
    whiteDist: data.whiteDist ?? lastSearchPartial?.whiteDist,
    blackDist: data.blackDist ?? lastSearchPartial?.blackDist,
    nodes:
      data.totalNodesAcrossWorkers ??
      data.totalNodes ??
      data.nodes ??
      lastSearchPartial?.nodes,
    depthLog: Array.isArray(data.depthLog)
      ? boundedDepthLog(data.depthLog)
      : lastSearchPartial?.depthLog,
    rootMoves: data.rootMoves ?? lastSearchPartial?.rootMoves,
    multiPv: data.multiPv ?? lastSearchPartial?.multiPv,
    pv: data.pv ?? lastSearchPartial?.pv,
    searchWallMs: data.elapsedMs ?? data.searchWallMs ?? lastSearchPartial?.searchWallMs,
  };
}

function broadcastEngineProgress(data) {
  if (!searchKind) return;
  const partial = lastSearchPartial;
  try {
    chrome.runtime
      .sendMessage({
        channel: "quoridors-bridge-engine-progress",
        kind: searchKind,
        algebraic: partial?.algebraicMove || extractPartialMove(data) || null,
        rootScore: data.rootScore ?? partial?.rootScore,
        depth: data.searchDepth ?? data.depth ?? partial?.depth,
        whiteDist: data.whiteDist ?? partial?.whiteDist,
        blackDist: data.blackDist ?? partial?.blackDist,
        nodes:
          data.totalNodesAcrossWorkers ??
          data.totalNodes ??
          data.nodes ??
          partial?.nodes,
        depthLog: partial?.depthLog || boundedDepthLog(data.depthLog) || null,
        rootMoves: data.rootMoves || partial?.rootMoves || null,
        multiPv: data.multiPv || partial?.multiPv || null,
        pv: data.pv ?? partial?.pv,
      })
      .catch(() => {});
  } catch {
    /* service worker may be restarting */
  }
}

function clearAnalyzeRestart() {
  if (analyzeRestartTimer != null) {
    clearTimeout(analyzeRestartTimer);
    analyzeRestartTimer = null;
  }
}

function clearReadyWaiter() {
  if (readyWaiter?.timeoutId) clearTimeout(readyWaiter.timeoutId);
  readyWaiter = null;
}

function resetEngineWorker(reason) {
  const err = new Error(reason || "Titanium engine worker reset");
  clearAnalyzeRestart();
  analyzeGen += 1;
  searchKind = null;
  analyzeFingerprint = null;
  analyzeSyncSkipLogged = false;
  if (readyWaiter) {
    readyWaiter.reject(err);
    clearReadyWaiter();
  }
  if (pendingSearch) {
    if (pendingSearch.timeoutId) clearTimeout(pendingSearch.timeoutId);
    pendingSearch.reject(err);
    pendingSearch = null;
  }
  lastSearchPartial = null;
  try {
    engineWorker?.terminate();
  } catch {
    /* already gone */
  }
  engineWorker = null;
  engineReady = false;
}

function ensureEngineWorker() {
  if (engineWorker) return;
  const url = chrome.runtime.getURL("engine/titaniumWasmWorker.js");
  engineWorker = new Worker(url, { type: "module" });
  engineWorker.onmessage = onEngineMessage;
  engineWorker.onerror = onEngineError;
}

function scheduleAnalyzeRestart(delayMs = ANALYZE_RESTART_MS) {
  clearAnalyzeRestart();
  if (searchKind !== "analyze" || !analyzeLastMoves || !analyzeLastOpts) return;
  const gen = analyzeGen;
  const fp = analyzeFingerprint;
  analyzeRestartTimer = setTimeout(() => {
    analyzeRestartTimer = null;
    if (searchKind !== "analyze" || gen !== analyzeGen) return;
    if (fingerprintMoves(analyzeLastMoves) !== fp) return;
    void runAnalyzeSearch(analyzeLastMoves, analyzeLastOpts, gen, fp);
  }, delayMs);
}

/** True when worker message belongs to the in-flight pendingSearch. */
function messageMatchesPending(data) {
  if (!pendingSearch) return false;
  if (data?.seq != null) return data.seq === pendingSearch.seq;
  // Weak fallback for workers that omit seq — prefer fingerprint match.
  if (pendingSearch.fingerprint != null && data?.fingerprint != null) {
    return data.fingerprint === pendingSearch.fingerprint;
  }
  // Prefer requiring seq: without seq and without fingerprint, reject.
  return false;
}

function onEngineMessage(ev) {
  const data = ev.data || {};
  const type = data.type;

  // Progress / lifecycle noise — never fail a pending request on these.
  if (
    type === "search-started" ||
    type === "info" ||
    type === "progress" ||
    type === "depth" ||
    type === "synced"
  ) {
    if (type === "info" || type === "progress" || type === "depth") {
      if (data.seq != null && pendingSearch && data.seq !== pendingSearch.seq) {
        return;
      }
      if (data.seq != null && !pendingSearch) {
        return;
      }
      rememberSearchPartial(data);
      broadcastEngineProgress(data);
    }
    return;
  }

  if (type === "ready") {
    engineReady = true;
    if (readyWaiter) {
      readyWaiter.resolve();
      clearReadyWaiter();
    }
    return;
  }

  if (type === "error") {
    const err = new Error(data.message || "Titanium engine error");
    if (readyWaiter) {
      readyWaiter.reject(err);
      clearReadyWaiter();
      return;
    }
    if (pendingSearch && messageMatchesPending(data)) {
      if (pendingSearch.timeoutId) clearTimeout(pendingSearch.timeoutId);
      const kind = pendingSearch.kind;
      pendingSearch.reject(err);
      pendingSearch = null;
      lastSearchPartial = null;
      if (kind === "analyze" && searchKind === "analyze") {
        scheduleAnalyzeRestart(600);
      } else if (kind === "play") {
        searchKind = null;
      }
    }
    return;
  }

  if (type === "bestmove" && pendingSearch) {
    if (!messageMatchesPending(data)) {
      // Stale analyze bestmove must not resolve a newer play pendingSearch.
      return;
    }
    if (pendingSearch.timeoutId) clearTimeout(pendingSearch.timeoutId);
    const pending = pendingSearch;
    const kind = pending.kind;
    const gen = pending.gen;
    const fp = pending.fingerprint;
    pendingSearch = null;
    pending.resolve(data);
    lastSearchPartial = null;

    if (kind === "analyze" && searchKind === "analyze" && gen === analyzeGen) {
      if (fp === analyzeFingerprint) {
        scheduleAnalyzeRestart(ANALYZE_RESTART_MS);
      }
    } else if (kind === "play") {
      searchKind = null;
    }
  }
}

function onEngineError(ev) {
  resetEngineWorker(ev?.message || "Titanium engine worker crashed");
}

function softCancel() {
  try {
    engineWorker?.postMessage({ op: "cancel" });
  } catch {
    /* worker may already be gone */
  }
}

function resolvePendingWithPartial(reason) {
  if (!pendingSearch) return false;
  const partial = lastSearchPartial;
  if (!partial?.algebraicMove) return false;
  if (pendingSearch.timeoutId) clearTimeout(pendingSearch.timeoutId);
  const pending = pendingSearch;
  pendingSearch = null;
  const stopReason =
    reason === "space" || reason === "space_interrupt"
      ? "space_interrupt"
      : reason === "timeout_partial"
        ? "timeout_partial"
        : "cancel_partial";
  pending.resolve({
    ...partial,
    algebraicMove: partial.algebraicMove,
    stopReason,
  });
  lastSearchPartial = null;
  return true;
}

function cancelPendingSearch(reason, clearMs = CANCEL_CLEAR_MS) {
  return new Promise((resolve) => {
    clearAnalyzeRestart();
    // Bump generation so late bestmoves/info from the cancelled search cannot
    // attach to a subsequent play/analyze pendingSearch.
    activeSearchSeq += 1;
    if (!pendingSearch) {
      softCancel();
      resolve(true);
      return;
    }
    const pending = pendingSearch;
    softCancel();
    if (resolvePendingWithPartial(reason)) {
      resolve(true);
      return;
    }
    const clearTimer = setTimeout(() => {
      if (pendingSearch === pending) {
        // Late info may have arrived during the wait — prefer partial over reject.
        if (resolvePendingWithPartial(reason)) {
          resolve(true);
          return;
        }
        if (pending.timeoutId) clearTimeout(pending.timeoutId);
        pendingSearch = null;
        pending.reject(new Error(reason || "engine search cancelled"));
        resolve(false);
      } else {
        resolve(true);
      }
    }, clearMs);
    const origResolve = pending.resolve;
    const origReject = pending.reject;
    pending.resolve = (data) => {
      clearTimeout(clearTimer);
      origResolve(data);
      resolve(true);
    };
    pending.reject = (err) => {
      clearTimeout(clearTimer);
      origReject(err);
      resolve(true);
    };
  });
}

/** Force-drop a stuck pendingSearch so play can begin (after soft-cancel wait). */
function forceAbortPendingSearch(reason) {
  if (!pendingSearch) return;
  if (pendingSearch.timeoutId) clearTimeout(pendingSearch.timeoutId);
  const pending = pendingSearch;
  pendingSearch = null;
  lastSearchPartial = null;
  try {
    pending.reject(new Error(reason || "play preempt force"));
  } catch {
    /* already settled */
  }
}

function enqueueOp(fn) {
  const run = opQueue.then(() => fn());
  // Keep the chain alive after failures so later ops still serialize.
  opQueue = run.then(
    () => undefined,
    () => undefined,
  );
  return run;
}

function initEngine({ engineMode, catLmrCeiling, threads } = {}) {
  const mode = engineMode || ENGINE_MODE;
  const cat = Number.isFinite(Number(catLmrCeiling))
    ? Number(catLmrCeiling)
    : CAT_LMR_CEILING;
  const threadCount = resolveThreads(threads);

  if (
    engineReady &&
    (lastThreads !== threadCount ||
      lastEngineMode !== mode ||
      lastCatLmrCeiling !== cat)
  ) {
    resetEngineWorker("engine profile changed");
  }
  lastEngineMode = mode;
  lastCatLmrCeiling = cat;
  lastThreads = threadCount;

  ensureEngineWorker();
  if (engineReady) return Promise.resolve();
  if (readyWaiter) return readyWaiter.promise;

  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  readyWaiter = { resolve, reject, promise };
  readyWaiter.timeoutId = setTimeout(() => {
    resetEngineWorker("Titanium engine init timed out");
  }, INIT_TIMEOUT_MS);
  try {
    engineWorker.postMessage({
      op: "init",
      engineMode: mode,
      catLmrCeiling: cat,
      threads: threadCount,
    });
  } catch (err) {
    resetEngineWorker(String(err?.message || err));
  }
  return promise;
}

function syncPosition(algebraicMoves, { engineMode, catLmrCeiling, threads, forceReset } = {}) {
  ensureEngineWorker();
  const mode = engineMode || lastEngineMode || ENGINE_MODE;
  const cat = Number.isFinite(Number(catLmrCeiling))
    ? Number(catLmrCeiling)
    : lastCatLmrCeiling;
  const threadCount = resolveThreads(
    threads != null ? threads : lastThreads,
  );
  lastEngineMode = mode;
  lastCatLmrCeiling = cat;
  lastThreads = threadCount;
  if (forceReset) {
    engineWorker.postMessage({
      op: "sync",
      algebraicMoves: [],
      engineMode: mode,
      catLmrCeiling: cat,
      threads: threadCount,
    });
  }
  engineWorker.postMessage({
    op: "sync",
    algebraicMoves: Array.isArray(algebraicMoves) ? algebraicMoves : [],
    engineMode: mode,
    catLmrCeiling: cat,
    threads: threadCount,
  });
}

/**
 * Start a worker search. For play: caller awaits the promise.
 * For analyze: fire-and-forget; bestmove triggers auto-restart.
 */
function beginSearch(algebraicMoves, timeMs, maxDepth, opts = {}, meta = {}) {
  const mode = opts.engineMode || ENGINE_MODE;
  const cat = Number.isFinite(Number(opts.catLmrCeiling))
    ? Number(opts.catLmrCeiling)
    : CAT_LMR_CEILING;
  const threadCount = resolveThreads(opts.threads);
  const moves = Array.isArray(algebraicMoves) ? algebraicMoves : [];
  const kind = meta.kind || "play";
  const forceFull = Boolean(opts.forceFullSync);
  const isFreshGame = forceFull
    ? true
    : opts.isFreshGame != null
      ? Boolean(opts.isFreshGame)
      : moves.length === 0;
  const gen = meta.gen ?? 0;
  const fingerprint = meta.fingerprint ?? fingerprintMoves(moves);
  const seq = ++activeSearchSeq;

  lastSearchPartial = null;

  return new Promise((resolve, reject) => {
    pendingSearch = {
      resolve,
      reject,
      kind,
      gen,
      fingerprint,
      seq,
    };
    // Analyze uses a huge budget; do not arm a near-term timeout that would
    // kill continuous eval. Play keeps timeMs + grace.
    const timeoutMs = Math.max(
      PLAY_TIMEOUT_FLOOR_MS,
      Number(timeMs) + PLAY_TIMEOUT_GRACE_MS,
    );
    const timeoutId =
      kind === "analyze"
        ? null
        : setTimeout(async () => {
            if (pendingSearch?.resolve !== resolve) return;
            if (pendingSearch?.seq !== seq) return;
            // Prefer best-so-far over hard failure.
            if (resolvePendingWithPartial("timeout_partial")) {
              return;
            }
            softCancel();
            await new Promise((r) => setTimeout(r, PLAY_TIMEOUT_PARTIAL_WAIT_MS));
            if (pendingSearch?.resolve !== resolve) return;
            if (pendingSearch?.seq !== seq) return;
            if (resolvePendingWithPartial("timeout_partial")) return;
            // Last resort: a true hang with no usable PV — never leave relay hanging.
            if (pendingSearch?.timeoutId) clearTimeout(pendingSearch.timeoutId);
            pendingSearch = null;
            lastSearchPartial = null;
            searchKind = null;
            resetEngineWorker("play search hung after preempt");
            reject(new Error("engine search hung — worker reset"));
          }, timeoutMs);
    pendingSearch.timeoutId = timeoutId;
    try {
      engineWorker.postMessage({
        op: "search",
        seq,
        algebraicMoves: moves,
        timeMs,
        maxNodes: 0,
        maxDepth: maxDepth || 0,
        isFreshGame,
        engineMode: mode,
        catLmrCeiling: cat,
        threads: threadCount,
        streamProgress: true,
        rootScores: opts.rootScores !== false,
        multipv: Math.max(1, Math.min(64, Number(opts.multipv) || 3)),
      });
    } catch (err) {
      if (timeoutId) clearTimeout(timeoutId);
      pendingSearch = null;
      lastSearchPartial = null;
      reject(err);
    }
  });
}

async function search(algebraicMoves, timeMs, maxDepth, opts = {}) {
  const mode = opts.engineMode || ENGINE_MODE;
  const cat = Number.isFinite(Number(opts.catLmrCeiling))
    ? Number(opts.catLmrCeiling)
    : CAT_LMR_CEILING;
  const threadCount = resolveThreads(opts.threads);

  await initEngine({ engineMode: mode, catLmrCeiling: cat, threads: threadCount });

  // Cancel analyze / prior work, wait briefly, then hard-abort if still stuck.
  clearAnalyzeRestart();
  analyzeGen += 1;
  analyzeFingerprint = null;
  const preemptNeeded = searchKind === "analyze" || pendingSearch;
  if (preemptNeeded) {
    await cancelPendingSearch("play preempt", PLAY_PREEMPT_CLEAR_MS);
    softCancel();
    await new Promise((resolve) => setTimeout(resolve, PLAY_PREEMPT_SETTLE_MS));
    if (pendingSearch) {
      forceAbortPendingSearch("play preempt force");
    }
  } else {
    softCancel();
  }
  searchKind = null;
  lastSearchPartial = null;

  searchKind = "play";
  const meta = {
    kind: "play",
    fingerprint: fingerprintMoves(algebraicMoves),
  };
  try {
    return await beginSearch(algebraicMoves, timeMs, maxDepth, opts, meta);
  } catch (err) {
    if (!/timed out|hung/i.test(String(err?.message || err))) throw err;
    resetEngineWorker("play search recovery");
    await initEngine({ engineMode: mode, catLmrCeiling: cat, threads: threadCount });
    searchKind = "play";
    return beginSearch(algebraicMoves, timeMs, maxDepth, opts, meta);
  }
}

async function runAnalyzeSearch(algebraicMoves, opts, gen, fingerprint) {
  if (searchKind !== "analyze" || gen !== analyzeGen) return;
  if (fingerprintMoves(algebraicMoves) !== fingerprint) return;
  try {
    await initEngine({
      engineMode: opts.engineMode,
      catLmrCeiling: opts.catLmrCeiling,
      threads: opts.threads,
    });
    if (searchKind !== "analyze" || gen !== analyzeGen) return;
    if (pendingSearch) {
      await cancelPendingSearch("analyze restart");
    }
    if (searchKind !== "analyze" || gen !== analyzeGen) return;

    // Continuous analysis: use the full budget (default ~24h). Do not slice into
    // short time-management windows — play preempt soft-cancels when needed.
    const timeMs = Number(opts.timeMs) > 0 ? Number(opts.timeMs) : ANALYZE_TIME_MS;
    const promise = beginSearch(algebraicMoves, timeMs, 0, opts, {
      kind: "analyze",
      gen,
      fingerprint,
    });
    // Analyze does not await bestmove for the message response; swallow reject
    // from soft-cancel so it does not surface as an unhandled rejection.
    promise.catch(() => {});
  } catch {
    if (searchKind === "analyze" && gen === analyzeGen) {
      scheduleAnalyzeRestart(600);
    }
  }
}

async function startAnalyze(msg) {
  if (searchKind === "play" && pendingSearch) {
    return { ok: false, error: "play search active", analyzing: false };
  }

  const moves = Array.isArray(msg.algebraicMoves) ? msg.algebraicMoves : [];
  const fp = fingerprintMoves(moves);
  const opts = {
    engineMode: msg.engineMode,
    catLmrCeiling: msg.catLmrCeiling,
    threads: msg.threads,
    timeMs: msg.timeMs ?? ANALYZE_TIME_MS,
    isFreshGame: moves.length === 0,
    rootScores: msg.rootScores !== false,
    multipv: Math.max(1, Math.min(64, Number(msg.multipv) || 3)),
  };

  // Same position already analyzing — keep the in-flight search.
  // Do not require pendingSearch: beginSearch sets it asynchronously, and a
  // second analyze for the same moves was canceling the first before progress.
  if (
    searchKind === "analyze" &&
    analyzeFingerprint === fp
  ) {
    return { ok: true, analyzing: true, already: true };
  }

  // A new position replaces the old analysis only after its soft cancel has
  // settled. This keeps the worker's single pendingSearch slot unambiguous.
  if (searchKind === "analyze" || pendingSearch) {
    clearAnalyzeRestart();
    analyzeGen += 1;
    searchKind = null;
    analyzeFingerprint = null;
    analyzeLastMoves = null;
    if (pendingSearch) {
      await cancelPendingSearch("analyze position changed", 150);
    } else {
      softCancel();
    }
    await new Promise((resolve) => setTimeout(resolve, 150));
  }

  const gen = ++analyzeGen;
  searchKind = "analyze";
  analyzeFingerprint = fp;
  analyzeSyncSkipLogged = false;
  analyzeLastMoves = moves;
  analyzeLastOpts = opts;

  // Respond immediately — do not block until bestmove.
  void runAnalyzeSearch(moves, opts, gen, fp);
  return { ok: true, analyzing: true };
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.channel !== "quoridors-bridge-engine-offscreen") return false;

  enqueueOp(async () => {
    try {
      if (msg.op === "warm" || msg.op === "init") {
        await initEngine({
          engineMode: msg.engineMode,
          catLmrCeiling: msg.catLmrCeiling,
          threads: msg.threads,
          forceReset: msg.forceReset,
        });
        sendResponse({
          ok: true,
          ready: engineReady,
          engineMode: lastEngineMode,
          threads: lastThreads,
          catLmrCeiling: lastCatLmrCeiling,
          crossOriginIsolated: Boolean(
            typeof crossOriginIsolated !== "undefined" && crossOriginIsolated,
          ),
        });
        return;
      }

      if (msg.op === "cancel") {
        clearAnalyzeRestart();
        analyzeGen += 1;
        const wasKind = searchKind;
        searchKind = null;
        analyzeFingerprint = null;
        // Wait for soft-cancel bestmove / first info (up to CANCEL_CLEAR_MS).
        // Do not immediately reject when no partial exists yet — Space early in
        // a search needs that grace window.
        // Never auto-restart analyze here: play/undo owns the next op.
        // Overlay ensureAnalysis restarts eval when appropriate.
        const cleared = await cancelPendingSearch(
          msg.reason || "cancel",
          msg.reason === "play" || msg.reason === "play preempt"
            ? PLAY_PREEMPT_CLEAR_MS
            : CANCEL_CLEAR_MS,
        );
        sendResponse({
          ok: true,
          cancelled: true,
          cleared: Boolean(cleared),
          wasKind,
        });
        return;
      }

      if (msg.op === "sync") {
        if (!engineReady) {
          await initEngine({
            engineMode: msg.engineMode,
            catLmrCeiling: msg.catLmrCeiling,
            threads: msg.threads,
          });
        }
        const activeAnalysis = searchKind === "analyze";
        if (activeAnalysis && !msg.forceReset) {
          if (!analyzeSyncSkipLogged) {
            analyzeSyncSkipLogged = true;
            console.info("[quoridors-bridge] sync skipped during active analyze");
          }
          sendResponse({ ok: true, synced: true, skipped: true });
          return;
        }
        if (activeAnalysis && msg.forceReset) {
          clearAnalyzeRestart();
          analyzeGen += 1;
          searchKind = null;
          analyzeFingerprint = null;
          analyzeLastMoves = null;
          analyzeSyncSkipLogged = false;
          await cancelPendingSearch("sync position changed");
        }
        syncPosition(msg.algebraicMoves ?? [], {
          engineMode: msg.engineMode,
          catLmrCeiling: msg.catLmrCeiling,
          threads: msg.threads,
          forceReset: msg.forceReset,
        });
        sendResponse({ ok: true, synced: true });
        return;
      }

      if (msg.op === "analyze") {
        sendResponse(await startAnalyze(msg));
        return;
      }

      if (msg.op === "search") {
        // search() cancels analyze + hard-aborts stuck pending within ~400ms.
        const data = await search(
          msg.algebraicMoves ?? [],
          msg.timeMs ?? 3000,
          msg.maxDepth,
          {
            engineMode: msg.engineMode,
            catLmrCeiling: msg.catLmrCeiling,
            threads: msg.threads,
            isFreshGame: msg.forceFullSync ? true : msg.isFreshGame,
            forceFullSync: msg.forceFullSync,
            rootScores: msg.rootScores !== false,
            multipv: Math.max(1, Math.min(64, Number(msg.multipv) || 3)),
          },
        );
        const algebraic =
          extractPartialMove(data) ||
          (typeof data.algebraicMove === "string" ? firstMoveFromPv(data.algebraicMove) : null);
        if (!algebraic || algebraic === "(none)") {
          throw new Error("engine returned no move");
        }
        sendResponse({
          ok: true,
          algebraic,
          whiteDist: data.whiteDist,
          blackDist: data.blackDist,
          rootScore: data.rootScore,
          depth: data.searchDepth ?? data.depth,
          nodes: data.totalNodesAcrossWorkers ?? data.totalNodes ?? data.nodes,
          stopReason: data.stopReason,
          searchWallMs: data.searchWallMs,
          depthLog: boundedDepthLog(data.depthLog),
          pv: data.pv,
          rootMoves: data.rootMoves || null,
          multiPv: data.multiPv || null,
        });
        return;
      }

      sendResponse({ ok: false, error: `unknown op: ${msg.op}` });
    } catch (err) {
      if (searchKind === "play") searchKind = null;
      sendResponse({ ok: false, error: String(err?.message || err) });
    }
  });

  return true;
});
