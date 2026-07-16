/**
 * On-page HUD for quoridors.com — eval bar, ghost hint, play-best button.
 *
 * Titanium runs entirely inside this extension as a WASM Web Worker (same
 * engine build as the site's browser client) — no network calls, no local
 * server. The worker itself lives in an offscreen document
 * (offscreen/offscreen.js), because a page at https://www.quoridors.com cannot
 * construct a chrome-extension:// Worker directly (browser-enforced
 * same-origin rule for Worker scripts). This content script just sends
 * `quoridors-bridge-engine` requests through the background service worker,
 * which relays them to the offscreen document and back.
 *
 * This script is an ISOLATED-world content script; it talks to
 * inject/page_hook.js (MAIN world) via window CustomEvents, same pattern as
 * content/quoridors.js.
 */
(function quoridorsBridgeOverlay() {
  if (window.__quoridorsBridgeOverlayInstalled) return;
  window.__quoridorsBridgeOverlayInstalled = true;

  const WHOLE_GAME_PLAN_MOVES = 30;
  const DEFAULT_SETTINGS = {
    timeSec: 3,
    wallClockSeconds: 60,
    wholeGameTime: true,
    engineMode: "titanium-v17",
    catLmrCeiling: 1000,
    threads: 0,
    maxDepth: null, // null/0 = no explicit cap
    pawnDelayAvgSec: 0.2,
    wallDelayAvgSec: 1.2,
    endgamePawnDelayAvgSec: 0.2,
    endgameWallDelayAvgSec: 0.8,
    delayJitter: 0.4,
    complexityScaleSec: 1,
    humanizerEnabled: true,
    autoplayEnabled: false,
    enableVisuals: true,
    showGhost: true,
    maxGhosts: 3,
    showEvalBar: true,
    settingsVersion: 7,
  };
  const OPENING_DELAY_SCALE_IGNORE_PLIES = 12;
  // Opening guard only on ply 0; Play best / Space skip humanizer entirely.
  const OPENING_AUTOPLAY_GUARD_SEC = 1.5;
  const ANALYZE_TIME_MS = 86_400_000;
  const ANALYZE_DEBOUNCE_MS = 100;
  /** Soft cap when using extension popup time (whole-game bank or per-move). */
  const MENU_PLAY_SEARCH_CAP_MS = 60_000;
  /** Soft cap when budgeting from a live in-game human clock. */
  const LIVE_PLAY_SEARCH_CAP_MS = 30_000;
  const PLAY_HARD_TIMEOUT_GRACE_MS = 8_000;
  const RECOVER_RETRY_TIME_MS = 800;
  const MAX_RECOVER_RETRIES = 128;
  const MAX_DEPTH_LOG_ENTRIES = 64;
  const EARLY_PLAY_MIN_DEPTH = 6;
  const EARLY_PLAY_MIN_ELAPSED_MS = 350;
  const ALG_MOVE_RE = /^[a-i][1-9][hv]?$/i;

  let settings = { ...DEFAULT_SETTINGS };
  let hud = null;
  let evalBarEl, evalFillEl, evalLabelEl, evalDepthEl, evalNodesEl, statusEl, playBestBtn;
  let busy = false;
  let playBestClicked = false;
  let forcePlayNow = false;
  /** Set when tick() drops a board update because busy — forces a retick in finally. */
  let pendingRetick = false;
  let spaceHeld = false;
  let playGeneration = 0;
  let lastFingerprint = null;
  let lastSeenPlyCount = 0;
  let sawLocalGame = false;
  let lastBoardDetailAt = 0;
  let prewarmStarted = false;
  let lastPlayPartial = null;
  /** Last algebraic move drawn as ghost — dedupe live progress re-renders. */
  let lastGhostMove = null;
  let lastGhostKey = null;
  let lastGhostBoardFp = null;
  let lastPaintedGhosts = [];
  let ghostMovesByFingerprint = new Map();
  /** Throttle thinking-mode ghost refreshes (progress floods ~every info tick). */
  let lastGhostModeFireAt = 0;
  let lastGhostHealAt = 0;
  const GHOST_MODE_THROTTLE_MS = 200;
  const GHOST_HEAL_MS = 1500;
  /** Cached think result for the current fingerprint (Space / Play best reuse). */
  let lastCompletedResult = null;
  /** True when the active search budget came from the live in-game clock. */
  let usingLiveClock = false;
  /** True while a play genmove search owns the WASM worker. */
  let playSearchActive = false;
  /** Hint-mode: kick another play search after a completed think (not analyze). */
  let continuePlayThink = false;
  let continuePlayTimer = null;
  /** Fingerprint of the position currently under always-on analysis. */
  let analysisFingerprint = null;
  let analysisPendingFp = null;
  let analysisDebounceTimer = null;
  let lastSyncedMovesFp = null;
  let recoverRetryByFingerprint = Object.create(null);
  let forceResetNextSync = false;
  /** performance.now() when the in-flight play search started (early-play gate). */
  let playSearchStartedAt = 0;
  /** performance.now() when our turn first became actionable — humanizer min clock. */
  let ourTurnBeganAt = 0;
  let ourTurnBeganFingerprint = null;
  let earlyPlayCancelArmed = false;

  function boundedDepthLog(value) {
    return Array.isArray(value) ? value.slice(-MAX_DEPTH_LOG_ENTRIES) : [];
  }

  function rememberRecoverRetry(fingerprint) {
    if (!fingerprint || recoverRetryByFingerprint[fingerprint]) return false;
    const keys = Object.keys(recoverRetryByFingerprint);
    if (keys.length >= MAX_RECOVER_RETRIES) {
      // Retry guards are only useful for the current session; discard the
      // oldest batch before retaining another position fingerprint.
      for (const key of keys.slice(0, Math.floor(MAX_RECOVER_RETRIES / 2))) {
        delete recoverRetryByFingerprint[key];
      }
    }
    recoverRetryByFingerprint[fingerprint] = true;
    return true;
  }

  // Whole-game clock bank (site-parity allocateWholeGameTime).
  let clockUsedMs = 0;
  let ownMovesPlayed = 0;
  let lastFingerprintForClock = null;
  let lastAlloc = null;
  let lastKnownOwnDist = null;

  function resolveExpectedMovesLeft({ ownMovesPlayed: played = 0, distanceToWin = null } = {}) {
    const n = Math.max(0, Number(played) || 0);
    const planTail = Math.max(1, WHOLE_GAME_PLAN_MOVES - n);
    const dist = Number(distanceToWin);
    const distFloor = Number.isFinite(dist) && dist > 0 ? Math.ceil(dist) : 0;
    return Math.max(planTail, distFloor, 1);
  }

  /** Inline port of site/web/src/lib/timeControl.js allocateWholeGameTime. */
  function allocateWholeGameTime({
    totalMs,
    usedMs,
    ownMovesPlayed: played,
    distanceToWin = null,
  }) {
    const total = Math.max(250, Number(totalMs) || 0);
    const used = Math.max(0, Number(usedMs) || 0);
    const remainingMs = Math.max(0, total - used);
    const expectedMovesLeft = resolveExpectedMovesLeft({
      ownMovesPlayed: played,
      distanceToWin,
    });
    const remainingFraction = remainingMs / total;
    const spendFactor =
      remainingFraction <= 0.1 ? 0.75 : remainingFraction <= 0.25 ? 1 : 1.35;
    const shareCap = remainingFraction <= 0.1 ? 0.1 : 0.2;
    const grossBudgetMs = Math.min(
      remainingMs * shareCap,
      (remainingMs / expectedMovesLeft) * spendFactor,
    );
    const handoffReserveMs = Math.min(300, Math.max(50, grossBudgetMs * 0.05));
    const moveBudgetMs =
      remainingMs > 0 ? Math.max(1, grossBudgetMs - handoffReserveMs) : 0;
    return {
      totalMs: total,
      remainingMs,
      moveBudgetMs,
      expectedMovesLeft,
      handoffReserveMs,
    };
  }

  /** Simplified chargeThinkMsForSeat — whole-game bank always charges wall time. */
  function chargeThinkMs(wallThinkMs, moveBudgetMs, handoffMs, usesWholeGameClock) {
    if (wallThinkMs == null || !Number.isFinite(Number(wallThinkMs))) return 0;
    const wall = Math.max(0, Math.round(Number(wallThinkMs)));
    if (usesWholeGameClock) return wall;
    const budget = Math.max(0, Math.round(Number(moveBudgetMs) || 0));
    const handoff = Math.max(0, Math.round(Number(handoffMs) || 0));
    if (budget > 0) return Math.min(wall, budget + handoff);
    return wall;
  }

  function resetClockBank() {
    clockUsedMs = 0;
    ownMovesPlayed = 0;
    lastFingerprintForClock = null;
    lastAlloc = null;
    lastKnownOwnDist = null;
    usingLiveClock = false;
    clearOurTurnBegan();
  }

  function updateManualHudClass() {
    if (!hud) return;
    hud.classList.toggle("wzb-manual", settings.autoplayEnabled !== true);
  }

  function fmtClock(ms) {
    const s = Math.ceil(Math.max(0, Number(ms) || 0) / 1000);
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }

  /** Positive live in-game remaining time only; null if missing/zero/non-finite. */
  function liveClockMyMs(detail) {
    const ms = Number(detail?.clock?.myMs);
    return Number.isFinite(ms) && ms > 0 ? ms : null;
  }

  /** Prefer freshest lastLocal clock when present (heartbeat keeps it ticking). */
  function detailForClockRead(detail) {
    const lastLocal = window.__quoridorsBridgeLastLocal;
    if (lastLocal?.clock?.myMs != null) {
      if (!detail?.fingerprint || lastLocal.fingerprint === detail.fingerprint) {
        return lastLocal;
      }
    }
    return detail;
  }

  function isGameOver(detail) {
    return Boolean(
      detail?.terminal ||
        detail?.game?.gameOver ||
        detail?.game?.winner ||
        detail?.game?.isDraw,
    );
  }

  function gameOverStatus(detail) {
    if (detail.game?.isDraw) return "draw";
    if (detail.game?.winner === detail.bottomPlayer) return "you win";
    if (detail.game?.winner) return "you lose";
    return "game over";
  }

  function isTypingTarget(el) {
    if (!el || !(el instanceof Element)) return false;
    const tag = el.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
    if (el.isContentEditable) return true;
    return Boolean(el.closest?.("input, textarea, select, [contenteditable='true']"));
  }

  function shouldHandleSpace() {
    return Boolean(shouldShowHud() || isActiveGameDetail(window.__quoridorsBridgeLastLocal));
  }

  function migrateSettings(raw) {
    const next = { ...DEFAULT_SETTINGS, ...(raw || {}) };
    const ver = Number(raw?.settingsVersion) || 0;
    if (!raw || ver < 2) {
      next.showEvalBar = true;
    }
    if (!raw || ver < 3) {
      next.wholeGameTime = true;
      if (raw?.wallClockSeconds == null) next.wallClockSeconds = 60;
      next.engineMode = "titanium-v17";
      next.catLmrCeiling = 1000;
      next.settingsVersion = 3;
    }
    if (!raw || ver < 4) {
      next.humanizerEnabled = true;
      next.settingsVersion = 4;
    }
    if (!raw || ver < 5) {
      next.maxGhosts = 3;
      next.settingsVersion = 5;
    }
    if (!raw || ver < 6) {
      next.threads = 0;
      next.settingsVersion = 6;
    }
    if (!raw || ver < 7) {
      const legacyDelay = Number(raw?.delayAvgSec);
      const legacyEndgame = Number(raw?.endgameDelayAvgSec);
      next.wallDelayAvgSec = Number.isFinite(legacyDelay)
        ? legacyDelay
        : DEFAULT_SETTINGS.wallDelayAvgSec;
      next.pawnDelayAvgSec = DEFAULT_SETTINGS.pawnDelayAvgSec;
      next.endgameWallDelayAvgSec = Number.isFinite(legacyEndgame)
        ? legacyEndgame
        : DEFAULT_SETTINGS.endgameWallDelayAvgSec;
      next.endgamePawnDelayAvgSec = DEFAULT_SETTINGS.endgamePawnDelayAvgSec;
      next.settingsVersion = 7;
    }
    next.maxGhosts = Math.max(1, Math.min(8, Number(next.maxGhosts) || 3));
    next.threads = Math.max(0, Math.min(32, Number(next.threads) || 0));
    next.pawnDelayAvgSec = Math.max(0, Number(next.pawnDelayAvgSec) || 0);
    next.wallDelayAvgSec = Math.max(0, Number(next.wallDelayAvgSec) || 0);
    next.endgamePawnDelayAvgSec = Math.max(0, Number(next.endgamePawnDelayAvgSec) || 0);
    next.endgameWallDelayAvgSec = Math.max(0, Number(next.endgameWallDelayAvgSec) || 0);
    return next;
  }

  function applySettings(nextSettings) {
    const previousThreads = Number(settings.threads) || 0;
    settings = migrateSettings(nextSettings);
    if (previousThreads !== (Number(settings.threads) || 0)) {
      prewarmStarted = false;
      prewarmEngine();
    }
    updateManualHudClass();
    updateVisualVisibility();
    if (settings.showEvalBar === false) {
      clearAnalysisLocalState();
    } else {
      ensureAnalysis(window.__quoridorsBridgeLastLocal);
    }
  }

  function loadSettings() {
    try {
      chrome.storage.local.get("quoridorsBridgeSettings", (data) => {
        const raw = data && data.quoridorsBridgeSettings;
        const migrated = migrateSettings(raw);
        applySettings(migrated);
        if (!raw || Number(raw.settingsVersion) < 6) {
          chrome.storage.local.set({ quoridorsBridgeSettings: migrated });
        }
      });
    } catch {
      /* extension context invalidated (reload race) — keep defaults */
    }
  }
  loadSettings();

  try {
    chrome.storage.onChanged.addListener((changes, area) => {
      if (area === "local" && changes.quoridorsBridgeSettings) {
        const wasAutoplayEnabled = settings.autoplayEnabled === true;
        applySettings(changes.quoridorsBridgeSettings.newValue || {});
        const detail = window.__quoridorsBridgeLastLocal;
        if (
          !wasAutoplayEnabled &&
          settings.autoplayEnabled === true &&
          detail &&
          canActNow(detail)
        ) {
          forcePlayNow = false;
          void tick(detail);
        }
      }
    });
  } catch {
    /* ignore */
  }

  // ---------------------------------------------------------------------
  // Titanium engine — hosted in the offscreen document, reached via the
  // background service worker. Prewarm keeps WASM hot; analyze/search requests
  // carry the complete algebraic history, while sync is reserved for recovery.
  // ---------------------------------------------------------------------

  function engineProfile() {
    const maxGhosts = Math.max(1, Math.min(8, Number(settings.maxGhosts) || 3));
    return {
      engineMode: settings.engineMode || "titanium-v17",
      catLmrCeiling: Number(settings.catLmrCeiling) || 1000,
      threads: Number(settings.threads) || 0,
      // Ask Titanium for ranked root dump + enough MultiPV slots for ghosts.
      rootScores: true,
      multipv: maxGhosts,
    };
  }

  function prewarmEngine() {
    if (prewarmStarted) return;
    prewarmStarted = true;
    const profile = engineProfile();
    chrome.runtime
      .sendMessage({
        channel: "quoridors-bridge-engine",
        op: "warm",
        ...profile,
      })
      .catch((err) => {
        console.warn("[quoridors-bridge] prewarm failed (non-blocking)", err);
      });
  }

  function syncEnginePosition(algebraicMoves, opts = {}) {
    const forceReset = Boolean(opts.forceReset || forceResetNextSync);
    if (forceReset) forceResetNextSync = false;
    const moves = Array.isArray(algebraicMoves) ? algebraicMoves : [];
    const movesFp = moves.join(" ");
    if (!forceReset && movesFp === lastSyncedMovesFp) return Promise.resolve();
    lastSyncedMovesFp = movesFp;
    const profile = engineProfile();
    return chrome.runtime
      .sendMessage({
        channel: "quoridors-bridge-engine",
        op: "sync",
        algebraicMoves: moves,
        forceReset,
        ...profile,
      })
      .catch(() => {});
  }

  function cancelEngineSearch(reason) {
    chrome.runtime
      .sendMessage({
        channel: "quoridors-bridge-engine",
        op: "cancel",
        reason,
      })
      .catch(() => {});
  }

  function playerToMoveFromPly(plyCount) {
    // Even ply → p1/white to move (site analysisEngineSession parity).
    return Number(plyCount) % 2 === 0 ? "p1" : "p2";
  }

  /** Play search owns the worker whenever we can move. */
  function canPlayNow(detail) {
    if (!assertOurTurn(detail)) return false;
    // When the site says it's our turn, don't block on ACE history bookkeeping.
    if (detail?.siteSaysOurTurn === true || detail?.isMyTurn === true) return true;
    return !detail?.historyDesynced;
  }

  function hasLegalMovesToPlay(detail) {
    const ace = detail?.aceLegalMoves;
    if (Array.isArray(ace) && ace.length > 0) return true;
    const pawns = detail?.legalPawnMoves || detail?.game?.legal_pawn_moves;
    const walls = detail?.legalWallPlacements || detail?.game?.legal_wall_placements;
    if (
      (Array.isArray(pawns) && pawns.length > 0) ||
      (Array.isArray(walls) && walls.length > 0)
    ) {
      return true;
    }
    // Site shows move dots only when we can move — enough to start a search.
    return detail?.siteSaysOurTurn === true || detail?.isMyTurn === true;
  }

  /** True only when a move may actually be applied. */
  function canActNow(detail) {
    return (
      isActiveGameDetail(detail) &&
      canPlayNow(detail) &&
      hasLegalMovesToPlay(detail) &&
      !isGameOver(detail)
    );
  }

  function manualAssistWanted(detail) {
    return Boolean(
      isActiveGameDetail(detail) &&
        canActNow(detail) &&
        settings.autoplayEnabled !== true &&
        !playSearchActive &&
        !busy,
    );
  }

  function analysisAllowed(detail) {
    return Boolean(
      analysisVisualsWanted() &&
        detail &&
        (isGameOver(detail) || isActiveGameDetail(detail)) &&
        !playSearchActive &&
        !busy,
    );
  }

  function clearAnalysisLocalState() {
    analysisFingerprint = null;
    analysisPendingFp = null;
    if (analysisDebounceTimer != null) {
      clearTimeout(analysisDebounceTimer);
      analysisDebounceTimer = null;
    }
  }

  function clearAnalysisVisualState() {
    clearAnalysisLocalState();
    lastCompletedResult = null;
    lastPlayPartial = null;
    if (evalFillEl) evalFillEl.style.setProperty("--scale", "0.5");
    if (evalLabelEl) evalLabelEl.textContent = "–";
    if (evalDepthEl) evalDepthEl.textContent = "";
    if (evalNodesEl) evalNodesEl.textContent = "";
  }

  function analysisVisualsWanted() {
    return visualsEnabled() && (settings.showEvalBar !== false || settings.showGhost !== false);
  }

  function analysisPositionKey(detail) {
    if (!detail) return null;
    const moves = Array.isArray(detail.algebraicMoves) ? detail.algebraicMoves : [];
    const p1 = detail.game?.pawns?.p1;
    const p2 = detail.game?.pawns?.p2;
    return [
      detail.gameId || detail.game?.game_id || "-",
      moves.length,
      p1 ? `${p1.x},${p1.y}` : "-",
      p2 ? `${p2.x},${p2.y}` : "-",
      detail.terminal || detail.game?.gameOver ? 1 : 0,
    ].join(":");
  }

  function ensureAnalysis(detail) {
    if (!analysisAllowed(detail) || busy) return;

    // Position-only key — ignore controllable/seat noise that was restarting
    // analyze every few hundred ms and killing live rootMoves streams.
    const fp = analysisPositionKey(detail);
    if (!fp || fp === analysisFingerprint || fp === analysisPendingFp) return;

    if (analysisDebounceTimer != null) clearTimeout(analysisDebounceTimer);
    analysisDebounceTimer = setTimeout(() => {
      analysisDebounceTimer = null;
      if (playSearchActive || busy) return;
      const current = window.__quoridorsBridgeLastLocal;
      if (!current || !analysisAllowed(current) || busy) return;
      if (analysisPositionKey(current) !== fp) return;
      if (fp === analysisFingerprint || fp === analysisPendingFp) return;
      paintImmediateGhosts(current);
      const profile = engineProfile();
      analysisPendingFp = fp;
      chrome.runtime
        .sendMessage({
          channel: "quoridors-bridge-engine",
          op: "analyze",
          algebraicMoves: Array.isArray(current.algebraicMoves) ? current.algebraicMoves : [],
          timeMs: ANALYZE_TIME_MS,
          isFreshGame: !(current.algebraicMoves && current.algebraicMoves.length),
          ...profile,
        })
        .then((response) => {
          // Only an accepted request owns this fingerprint. This prevents a
          // transient relay failure from suppressing the next fresh request.
          if (
            response?.ok === true &&
            analysisPositionKey(window.__quoridorsBridgeLastLocal) === fp
          ) {
            analysisFingerprint = fp;
          }
          if (analysisPendingFp === fp) analysisPendingFp = null;
        })
        .catch(() => {
          if (analysisPendingFp === fp) analysisPendingFp = null;
        });
    }, ANALYZE_DEBOUNCE_MS);
  }

  function sendRuntimeMessageWithTimeout(message, timeoutMs) {
    return new Promise((resolve, reject) => {
      let settled = false;
      const timer = setTimeout(() => {
        if (settled) return;
        settled = true;
        const partial = lastPlayPartial;
        if (partial?.algebraic) {
          cancelEngineSearch("timeout_partial");
          resolve({
            ok: true,
            algebraic: partial.algebraic,
            whiteDist: partial.whiteDist,
            blackDist: partial.blackDist,
            rootScore: partial.rootScore,
            depth: partial.depth,
            nodes: partial.nodes,
            stopReason: "timeout_partial",
            depthLog: boundedDepthLog(partial.depthLog),
            pv: partial.pv,
            rootMoves: partial.rootMoves,
            multiPv: partial.multiPv,
          });
          return;
        }
        cancelEngineSearch("content request timed out");
        reject(new Error("engine request timed out"));
      }, timeoutMs);
      chrome.runtime.sendMessage(message, (response) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        const err = chrome.runtime.lastError;
        if (err) {
          reject(new Error(err.message || String(err)));
          return;
        }
        resolve(response);
      });
    });
  }

  function resolveSearchTimeMs(detail) {
    const clockDetail = detailForClockRead(detail);
    const liveMyMs = liveClockMyMs(clockDetail);
    // Human game WITH a live timer: budget from side-to-move remaining (our clock).
    if (liveMyMs != null) {
      usingLiveClock = true;
      const remaining = Math.max(250, liveMyMs);
      const alloc = allocateWholeGameTime({
        totalMs: remaining,
        usedMs: 0,
        ownMovesPlayed,
        distanceToWin: lastKnownOwnDist,
      });
      lastAlloc = alloc;
      const safetyReserveMs = Math.min(1500, Math.max(400, remaining * 0.02));
      const budget = Math.min(
        alloc.moveBudgetMs,
        remaining - safetyReserveMs,
        LIVE_PLAY_SEARCH_CAP_MS,
      );
      return Math.max(200, Math.round(budget));
    }
    // No site timer (AI / casual online / hotseat): use popup settings as-is.
    usingLiveClock = false;
    if (settings.wholeGameTime !== false) {
      const totalMs = Math.max(1000, Number(settings.wallClockSeconds) * 1000 || 60_000);
      const alloc = allocateWholeGameTime({
        totalMs,
        usedMs: clockUsedMs,
        ownMovesPlayed,
        distanceToWin: lastKnownOwnDist,
      });
      lastAlloc = alloc;
      const bankLeft = Math.max(0, totalMs - clockUsedMs);
      const budget = Math.min(
        alloc.moveBudgetMs,
        bankLeft,
        MENU_PLAY_SEARCH_CAP_MS,
      );
      return Math.max(200, Math.round(budget));
    }
    lastAlloc = null;
    const sec = Math.max(0.2, Number(settings.timeSec) || 1.2);
    return Math.max(200, Math.min(MENU_PLAY_SEARCH_CAP_MS, Math.round(sec * 1000)));
  }

  function rememberOwnDistance(result, bottomPlayer) {
    const whiteDist = Number(result?.whiteDist);
    const blackDist = Number(result?.blackDist);
    if (!Number.isFinite(whiteDist) || !Number.isFinite(blackDist)) return;
    // Site seats: white = p1 (bottom often), black = p2.
    lastKnownOwnDist = bottomPlayer === "p2" ? blackDist : whiteDist;
  }

  function chargeClockAfterThink(result, didPlayMove) {
    // Live in-game clocks are re-read each search; do not deplete the popup bank.
    if (usingLiveClock) {
      if (didPlayMove) ownMovesPlayed += 1;
      return;
    }
    const wall =
      Number(result?.searchWallMs) || Number(result?.requestWallMs) || 0;
    const charged = chargeThinkMs(
      wall,
      lastAlloc?.moveBudgetMs,
      lastAlloc?.handoffReserveMs,
      settings.wholeGameTime !== false,
    );
    clockUsedMs += charged;
    if (didPlayMove) ownMovesPlayed += 1;
  }

  async function genmove(detail, opts = {}) {
    playSearchActive = true;
    lastPlayPartial = null;
    // Local only — search handler cancels any in-flight analyze; avoid triple-hit race.
    clearAnalysisLocalState();
    const clockDetail = detailForClockRead(detail);
    const algebraicMoves = Array.isArray(detail?.algebraicMoves) ? detail.algebraicMoves : [];
    const timeMs =
      Number(opts.timeMs) > 0 ? Math.round(Number(opts.timeMs)) : resolveSearchTimeMs(detail);
    const liveMs = liveClockMyMs(clockDetail);
    if (usingLiveClock && liveMs != null) {
      setStatus(
        `your turn · thinking… · live ${fmtClock(liveMs)} · think ${(timeMs / 1000).toFixed(1)}s`,
      );
    } else if (settings.wholeGameTime !== false) {
      const bankLeft = Math.max(
        0,
        (Number(settings.wallClockSeconds) || 60) * 1000 - clockUsedMs,
      );
      setStatus(
        `your turn · thinking… · bank ${fmtClock(bankLeft)} · think ${(timeMs / 1000).toFixed(1)}s`,
      );
    } else {
      setStatus(`your turn · thinking… · ${(timeMs / 1000).toFixed(1)}s/move`);
    }
    paintImmediateGhosts(detail);

    if (algebraicMoves.length === 0) {
      if (!pawnsAtStart(detail)) {
        throw new Error("history desynced: empty moves but pawns not at start");
      }
      return {
        algebraic: "e2",
        whiteDist: 8,
        blackDist: 8,
        rootScore: 0,
        depth: 0,
        nodes: 0,
        stopReason: "extension_opening",
        searchWallMs: 0,
        requestWallMs: 0,
        depthLog: [{ depth: 1, pv: "e2" }],
        pv: "e2",
      };
    }

    const maxDepth = Number(settings.maxDepth) || 0;
    const profile = engineProfile();
    earlyPlayCancelArmed = false;
    playSearchStartedAt = performance.now();
    console.info("[quoridors-bridge] genmove request", {
      algebraicMoves,
      timeMs,
      maxDepth,
      forceFullSync: false,
      isFreshGame: algebraicMoves.length === 0,
      wholeGameTime: settings.wholeGameTime !== false,
      usingLiveClock,
      liveMyMs: liveMs,
      clockUsedMs,
      ownMovesPlayed,
      ...profile,
    });
    // Incremental make_move keeps TT warm; full reset is only for recovery.
    const requestStartedAt = performance.now();
    try {
      const res = await sendRuntimeMessageWithTimeout(
        {
          channel: "quoridors-bridge-engine",
          op: "search",
          algebraicMoves,
          timeMs,
          maxDepth,
          isFreshGame: algebraicMoves.length === 0,
          forceFullSync: false,
          ...profile,
        },
        timeMs + PLAY_HARD_TIMEOUT_GRACE_MS,
      );
      console.info("[quoridors-bridge] genmove result", res);

      if (!res || res.ok === false) {
        const errMsg = res?.error || "engine request failed";
        if (/timed out|cancel/i.test(errMsg) && lastPlayPartial?.algebraic) {
          const requestWallMs = Math.max(0, performance.now() - requestStartedAt);
          return {
            algebraic: lastPlayPartial.algebraic,
            whiteDist: lastPlayPartial.whiteDist,
            blackDist: lastPlayPartial.blackDist,
            rootScore: lastPlayPartial.rootScore,
            depth: lastPlayPartial.depth,
            nodes: lastPlayPartial.nodes,
            stopReason: lastPlayPartial.stopReason || "timeout_partial",
            searchWallMs: requestWallMs,
            requestWallMs,
            depthLog: boundedDepthLog(lastPlayPartial.depthLog),
            pv: lastPlayPartial.pv,
            rootMoves: lastPlayPartial.rootMoves,
            multiPv: lastPlayPartial.multiPv,
          };
        }
        throw new Error(errMsg);
      }
      const requestWallMs = Math.max(0, performance.now() - requestStartedAt);
      const algebraic = firstMoveFromPv(res.algebraic || res.pv);
      const engineWall = Number(res.searchWallMs);
      return {
        algebraic: algebraic && algebraic !== "(none)" ? algebraic : null,
        whiteDist: res.whiteDist,
        blackDist: res.blackDist,
        rootScore: res.rootScore,
        depth: res.depth,
        nodes: res.nodes,
        stopReason: res.stopReason,
        // Prefer our wall clock; engine elapsed can under-report vs humanizer hold.
        searchWallMs: Math.max(
          requestWallMs,
          Number.isFinite(engineWall) ? engineWall : 0,
        ),
        requestWallMs,
        depthLog: boundedDepthLog(res.depthLog),
        pv: res.pv,
        rootMoves: res.rootMoves || res.data?.rootMoves,
        multiPv: res.multiPv || res.data?.multiPv,
      };
    } finally {
      // tick owns the lifecycle flag for a play search.
    }
  }

  // ---------------------------------------------------------------------
  // HUD
  // ---------------------------------------------------------------------

  function sleepUnlessForced(ms, onProgress) {
    return new Promise((resolve) => {
      const deadline = Date.now() + ms;
      const total = Math.max(1, ms);
      function step() {
        if (forcePlayNow) {
          onProgress?.(0);
          resolve("forced");
          return;
        }
        const remaining = deadline - Date.now();
        onProgress?.(Math.max(0, Math.min(1, remaining / total)));
        if (remaining <= 0) {
          resolve("elapsed");
          return;
        }
        setTimeout(step, Math.min(100, remaining));
      }
      step();
    });
  }

  function withHardTimeout(promise, timeoutMs, onTimeout) {
    let timeoutId = null;
    const timeout = new Promise((_, reject) => {
      timeoutId = setTimeout(() => {
        try {
          onTimeout?.();
        } catch {
          /* ignore timeout cleanup errors */
        }
        reject(new Error("engine search timed out"));
      }, timeoutMs);
    });
    return Promise.race([
      Promise.resolve(promise).finally(() => {
        if (timeoutId) clearTimeout(timeoutId);
      }),
      timeout,
    ]);
  }

  function setStatus(text) {
    if (statusEl) statusEl.textContent = text;
  }

  function updatePlayBestEnabled() {
    if (!playBestBtn) return;
    const detail = window.__quoridorsBridgeLastLocal;
    const show =
      isActiveGameDetail(detail) &&
      canActNow(detail);
    playBestBtn.hidden = !show;
    playBestBtn.disabled = !show;
    playBestBtn.style.display = show ? "" : "none";
  }

  function setButtonsDisabled(_disabled) {
    // Play best follows board state, not the engine's busy/thinking state.
    updatePlayBestEnabled();
  }

  function findBoardMount() {
    const boardEl =
      document.getElementById("board") ||
      document.querySelector(".board-wrap #board") ||
      document.querySelector("#board");
    if (!boardEl) return null;
    const aspect = boardEl.closest(".board-wrap") || boardEl.parentElement;
    const parent = aspect && aspect.parentElement;
    if (!aspect || !parent) return null;
    return { aspect, parent };
  }

  function isElementVisible(el) {
    if (!el || !el.isConnected) return false;
    const style = window.getComputedStyle(el);
    if (style.display === "none" || style.visibility === "hidden" || Number(style.opacity) === 0) {
      return false;
    }
    const rect = el.getBoundingClientRect();
    if (rect.width < 80 || rect.height < 80) return false;
    return rect.bottom > 0 && rect.right > 0 && rect.top < window.innerHeight && rect.left < window.innerWidth;
  }

  function isActiveGameDetail(detail) {
    if (Date.now() - lastBoardDetailAt > 2000) return false;
    const game = detail?.game;
    return Boolean(
      game &&
        game.pawns?.p1 &&
        game.pawns?.p2 &&
        (game.turn === "p1" || game.turn === "p2"),
    );
  }

  function boardLooksInProgress() {
    const boardEl = document.getElementById("board");
    if (!boardEl) return false;
    return Boolean(
      boardEl.querySelector(".pawn") ||
        boardEl.querySelector(".move-dot") ||
        boardEl.querySelector("[data-r][data-c]"),
    );
  }

  function shouldShowHud() {
    const mount = findBoardMount();
    if (!visualsEnabled() || !mount || !isElementVisible(mount.aspect)) return false;
    return Boolean(
      isActiveGameDetail(window.__quoridorsBridgeLastLocal) ||
        sawLocalGame ||
        boardLooksInProgress(),
    );
  }

  function cleanupSiteLayoutClass() {
    for (const el of document.querySelectorAll(".wzb-board-row")) {
      el.classList.remove("wzb-board-row");
    }
  }

  function updateHudPlacement() {
    if (!hud || !hud.isConnected) return;
    cleanupSiteLayoutClass();
    const mount = findBoardMount();
    if (!mount) return;

    const rect = mount.aspect.getBoundingClientRect();
    const gap = 10;
    const width = 110;
    const left = Math.max(8, rect.left - width - gap);
    const top = Math.max(8, rect.top);
    const bottom = Math.min(window.innerHeight - 8, rect.bottom);
    const height = Math.max(96, bottom - top);

    hud.style.left = `${left}px`;
    hud.style.top = `${top}px`;
    hud.style.height = `${height}px`;
  }

  function buildHud() {
    if (hud && hud.isConnected) {
      updateHudPlacement();
      updateVisualVisibility();
      return hud;
    }
    hud = null;

    const mount = findBoardMount();
    if (!mount) return null;
    cleanupSiteLayoutClass();

    const existing = document.getElementById("quoridors-bridge-hud");
    if (existing) {
      hud = existing;
      if (hud.parentElement !== document.body) document.body.appendChild(hud);
    } else {
      hud = document.createElement("div");
      hud.id = "quoridors-bridge-hud";
      hud.innerHTML = `
        <div class="wzb-eval-bar" id="wzb-eval-bar" title="Titanium eval: black dist − white dist">
          <div class="wzb-eval-fill" id="wzb-eval-fill"></div>
          <div class="wzb-eval-readout">
            <div class="wzb-eval-label" id="wzb-eval-label">&ndash;</div>
            <div class="wzb-eval-depth" id="wzb-eval-depth"></div>
            <div class="wzb-eval-nodes" id="wzb-eval-nodes"></div>
          </div>
        </div>
        <button id="wzb-play-best" class="wzb-btn" type="button">Play best</button>
        <div class="wzb-status" id="wzb-status">idle</div>
      `;
      document.body.appendChild(hud);
    }

    hud.querySelector("#wzb-autoplay-toggle")?.closest(".wzb-autoplay")?.remove();
    hud.querySelector("#wzb-autoplay-toggle")?.remove();
    if (!hud.querySelector("#wzb-eval-bar")) {
      hud.insertAdjacentHTML(
        "afterbegin",
        `<div class="wzb-eval-bar" id="wzb-eval-bar" title="Titanium eval: black dist − white dist">
          <div class="wzb-eval-fill" id="wzb-eval-fill"></div>
          <div class="wzb-eval-readout">
            <div class="wzb-eval-label" id="wzb-eval-label">&ndash;</div>
            <div class="wzb-eval-depth" id="wzb-eval-depth"></div>
            <div class="wzb-eval-nodes" id="wzb-eval-nodes"></div>
          </div>
        </div>`,
      );
    } else if (!hud.querySelector("#wzb-eval-depth")) {
      const label = hud.querySelector("#wzb-eval-label");
      if (label && !label.parentElement?.classList?.contains("wzb-eval-readout")) {
        const readout = document.createElement("div");
        readout.className = "wzb-eval-readout";
        label.replaceWith(readout);
        readout.appendChild(label);
        const depthEl = document.createElement("div");
        depthEl.className = "wzb-eval-depth";
        depthEl.id = "wzb-eval-depth";
        readout.appendChild(depthEl);
        const nodesEl = document.createElement("div");
        nodesEl.className = "wzb-eval-nodes";
        nodesEl.id = "wzb-eval-nodes";
        readout.appendChild(nodesEl);
      } else if (label?.parentElement?.classList?.contains("wzb-eval-readout")) {
        const depthEl = document.createElement("div");
        depthEl.className = "wzb-eval-depth";
        depthEl.id = "wzb-eval-depth";
        label.parentElement.appendChild(depthEl);
      }
    }
    if (!hud.querySelector("#wzb-eval-nodes")) {
      const readout = hud.querySelector(".wzb-eval-readout");
      if (readout) {
        const nodesEl = document.createElement("div");
        nodesEl.className = "wzb-eval-nodes";
        nodesEl.id = "wzb-eval-nodes";
        readout.appendChild(nodesEl);
      }
    }

    evalBarEl = hud.querySelector("#wzb-eval-bar");
    evalFillEl = hud.querySelector("#wzb-eval-fill");
    evalLabelEl = hud.querySelector("#wzb-eval-label");
    evalDepthEl = hud.querySelector("#wzb-eval-depth");
    evalNodesEl = hud.querySelector("#wzb-eval-nodes");
    statusEl = hud.querySelector("#wzb-status");
    playBestBtn = hud.querySelector("#wzb-play-best");
    prewarmEngine();
    updatePlayBestEnabled();

    if (!playBestBtn.dataset.wired) {
      playBestBtn.dataset.wired = "1";
      playBestBtn.addEventListener("click", () => {
        const detail = window.__quoridorsBridgeLastLocal;
        if (isGameOver(detail)) return;
        void playReadyBestImmediate("play-best");
      });
    }

    updateManualHudClass();
    updateHudPlacement();
    updateVisualVisibility();
    updatePlayBestEnabled();
    return hud;
  }

  function visualsEnabled() {
    return settings.enableVisuals !== false;
  }

  function updateVisualVisibility() {
    const enabled = shouldShowHud();
    if (hud) {
      hud.style.display = enabled ? "" : "none";
    }
    if (evalBarEl) {
      evalBarEl.style.display = enabled && settings.showEvalBar !== false ? "" : "none";
    }
    if (statusEl) {
      statusEl.style.display = enabled ? "" : "none";
    }
    // Do not clear a live ghost when HUD visibility flickers false mid-think.
    // Only clear when the user disabled visuals or the ghost itself.
    if (!visualsEnabled() || settings.showGhost === false) clearGhost();
  }

  function formatEngineEval(score) {
    const value = Number(score);
    if (!Number.isFinite(value)) return null;
    const abs = Math.abs(value);
    if (abs >= 99000) {
      const mateIn = Math.max(1, 100000 - abs);
      return `${value >= 0 ? "" : "-"}M${mateIn}`;
    }
    if (abs >= 31000 && abs <= 32500) {
      const mateIn = Math.max(1, 32000 - abs);
      return `${value >= 0 ? "" : "-"}M${mateIn}`;
    }
    const pawns = value / 100;
    return `${pawns > 0 ? "+" : ""}${pawns.toFixed(2)}`;
  }

  function scoreToScale(score) {
    const value = Number(score);
    if (!Number.isFinite(value)) return null;
    return Math.max(0.05, Math.min(0.95, 1 / (1 + Math.exp(-value / 350))));
  }

  function formatCompactNodes(nodes) {
    const n = Number(nodes);
    if (!Number.isFinite(n) || n < 0) return "";
    if (n < 1000) return String(Math.round(n));
    if (n < 1e6) return `${(n / 1e3).toFixed(2)}k`;
    if (n < 1e9) return `${(n / 1e6).toFixed(2)}M`;
    return `${(n / 1e9).toFixed(2)}B`;
  }

  function updateEvalBar(result, bottomPlayer, playerToMove) {
    updateVisualVisibility();
    if (!visualsEnabled()) return;
    if (settings.showEvalBar === false) return;
    if (!evalFillEl || !evalLabelEl) return;

    const depthNum = Number(result?.depth ?? result?.searchDepth);
    const depthText = Number.isFinite(depthNum) && depthNum > 0 ? `d${depthNum}` : "";
    const nodesText = formatCompactNodes(
      result?.nodes ?? result?.totalNodes ?? result?.totalNodesAcrossWorkers,
    );
    const setDepth = () => {
      if (evalDepthEl) evalDepthEl.textContent = depthText;
      if (evalNodesEl) evalNodesEl.textContent = nodesText;
    };
    const nodesNote = nodesText
      ? `, nodes ${Math.round(Number(result?.nodes ?? result?.totalNodes ?? result?.totalNodesAcrossWorkers))}`
      : "";

    // rootScore is negamax (side-to-move). Flip to white, then map fill to
    // bottomPlayer — same as site analysisResultToEvalState.
    const sideScore = Number(result?.rootScore);
    const hasScore = Number.isFinite(sideScore);
    if (hasScore) {
      const turn =
        playerToMove ||
        result?.playerToMove ||
        (Array.isArray(window.__quoridorsBridgeLastLocal?.algebraicMoves)
          ? playerToMoveFromPly(window.__quoridorsBridgeLastLocal.algebraicMoves.length)
          : "p1");
      const blackToMove = turn === "p2" || turn === 2;
      const whiteScore = blackToMove ? -sideScore : sideScore;
      const whiteScale = scoreToScale(whiteScore);
      const scoreLabel = formatEngineEval(whiteScore);
      if (whiteScale != null && scoreLabel != null) {
        const mine = bottomPlayer === "p2" ? 1 - whiteScale : whiteScale;
        evalFillEl.style.setProperty("--scale", String(mine));
        evalLabelEl.textContent = scoreLabel;
        setDepth();
        const depthNote = depthText ? `, ${depthText}` : "";
        evalBarEl?.setAttribute("title", `Titanium eval (white perspective)${depthNote}${nodesNote}`);
        return;
      }
    }

    const whiteDist = Number(result.whiteDist);
    const blackDist = Number(result.blackDist);
    if (!Number.isFinite(whiteDist) || !Number.isFinite(blackDist)) {
      evalLabelEl.textContent = "–";
      if (evalDepthEl) evalDepthEl.textContent = "";
      if (evalNodesEl) evalNodesEl.textContent = nodesText;
      return;
    }
    const margin = blackDist - whiteDist;
    const p1 = Math.max(0.05, Math.min(0.95, 0.5 + margin * 0.07));
    const mine = bottomPlayer === "p1" ? p1 : 1 - p1;
    evalFillEl.style.setProperty("--scale", String(mine));
    evalLabelEl.textContent = margin > 0 ? `d+${margin}` : `d${margin}`;
    setDepth();
    if (depthText || nodesNote) {
      const depthNote = depthText ? `, ${depthText}` : "";
      evalBarEl?.setAttribute("title", `Titanium eval: black dist − white dist${depthNote}${nodesNote}`);
    }
  }

  function isForcedEndgameScore(score) {
    const value = Number(score);
    if (!Number.isFinite(value)) return false;
    const abs = Math.abs(value);
    return abs >= 99000 || (abs >= 31000 && abs <= 32500);
  }

  function dispatchCmd(action, extra) {
    return new Promise((resolve) => {
      const requestId = `wzb-${Date.now()}-${Math.random().toString(16).slice(2)}`;
      let settled = false;
      let timeoutId = null;
      const onResult = (ev) => {
        const d = ev.detail || {};
        if (d.requestId !== requestId) return;
        if (settled) return;
        settled = true;
        if (timeoutId != null) clearTimeout(timeoutId);
        window.removeEventListener("quoridors-bridge-cmd-result", onResult);
        resolve(d);
      };
      window.addEventListener("quoridors-bridge-cmd-result", onResult);
      window.dispatchEvent(
        new CustomEvent("quoridors-bridge-cmd", { detail: { action, requestId, ...extra } }),
      );
      timeoutId = setTimeout(() => {
        if (settled) return;
        settled = true;
        window.removeEventListener("quoridors-bridge-cmd-result", onResult);
        resolve({ ok: false, error: "timeout waiting for page hook" });
      }, 5000);
    });
  }

  function dispatchCmdFire(action, extra) {
    const requestId = `wzb-fire-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    window.dispatchEvent(
      new CustomEvent("quoridors-bridge-cmd", {
        detail: { action, requestId, ...extra },
      }),
    );
  }

  function renderGhostsCmd(ghosts, bottomPlayer, player) {
    if (!visualsEnabled() || settings.showGhost === false) {
      clearGhostFire();
      return Promise.resolve();
    }
    return dispatchCmd("renderGhosts", { ghosts, bottomPlayer, player });
  }

  function renderGhostsFire(ghosts, bottomPlayer, player) {
    if (!visualsEnabled() || settings.showGhost === false) {
      clearGhostFire();
      return;
    }
    if (Array.isArray(ghosts) && ghosts.length) {
      lastPaintedGhosts = ghosts.map((ghost) => ({
        move: ghost.move,
        rank: ghost.rank,
        blunder: Boolean(ghost.blunder),
      }));
    }
    dispatchCmdFire("renderGhosts", { ghosts, bottomPlayer, player });
  }

  function prunePaintedGhosts(detail) {
    const ghosts = legalGhosts(lastPaintedGhosts, detail);
    if (!ghosts.length) {
      clearGhostFire();
      return;
    }
    lastGhostMove = ghosts[0].move;
    lastGhostKey = null;
    renderGhostsFire(
      ghosts.map((ghost) => ({
        move: ghost.move,
        rank: ghost.rank,
        blunder: Boolean(ghost.blunder),
        mode: ghost.rank === 1 ? "thinking" : "static",
        progress: 1,
      })),
      detail.bottomPlayer,
      detail.game?.turn,
    );
  }

  function clearGhostCmd() {
    clearGhostFire();
    return Promise.resolve();
  }

  function clearGhostFire() {
    lastGhostMove = null;
    lastGhostKey = null;
    lastPaintedGhosts = [];
    ghostMovesByFingerprint = new Map();
    dispatchCmdFire("clearGhost");
  }

  function setGhostProgressCmd(progress) {
    return dispatchCmd("setGhostProgress", { progress });
  }

  function setGhostModeCmd(mode) {
    return dispatchCmd("setGhostMode", { mode });
  }

  const TEMPO_CP = 100;
  function isMateWinScore(s) {
    const n = Number(s);
    return Number.isFinite(n) && (n >= 99000 || (n >= 31000 && n <= 32500));
  }
  function isMateLossScore(s) {
    const n = Number(s);
    return Number.isFinite(n) && (n <= -99000 || (n <= -31000 && n >= -32500));
  }
  function isBlunderVsBest(bestScore, moveScore) {
    return (
      (isMateLossScore(moveScore) && !isMateLossScore(bestScore)) ||
      (isMateWinScore(bestScore) && !isMateWinScore(moveScore)) ||
      Number(bestScore) - Number(moveScore) > TEMPO_CP
    );
  }
  function filterAntiBlunderGhosts(entries, limit) {
    const list = Array.isArray(entries) ? entries : [];
    if (!list.length) return [];
    const max = Math.max(1, Number(limit) || 1);
    const bestScore = Number(list[0].score);
    const result = [list[0]];
    for (const entry of list.slice(1)) {
      if (result.length >= max || isBlunderVsBest(bestScore, entry.score)) break;
      result.push(entry);
    }
    return result;
  }

  function normalizeRootMoveEntry(entry) {
    if (!entry || typeof entry !== "object") return null;
    const move = firstMoveFromPv(entry.move ?? entry.algebraic).toLowerCase();
    if (!ALG_MOVE_RE.test(move) || move === "(none)") return null;
    const score = Number(entry.score);
    const rank = Number(entry.rank);
    return {
      move,
      score: Number.isFinite(score) ? score : Number.NEGATIVE_INFINITY,
      rank: Number.isFinite(rank) && rank > 0 ? rank : null,
    };
  }

  /**
   * Ghosts prefer Titanium rootMoves / multiPv (rank + score).
   * Never invent multi-move alternatives from depth-log history.
   * Until a root dump arrives, show the current best only (algebraic / last PV).
   */
  function coalesceTopGhostMoves({
    algebraic,
    pv,
    depthLog,
    rootMoves,
    multiPv,
    rootScore,
    limit = settings.maxGhosts,
  } = {}) {
    const max = Math.max(1, Math.min(8, Number(limit) || 3));
    let best = firstMoveFromPv(algebraic || pv).toLowerCase();
    if (!best || best === "(none)") {
      const log = Array.isArray(depthLog) ? depthLog : [];
      if (log.length) {
        best = firstMoveFromPv(log[log.length - 1]?.pv).toLowerCase();
      }
    }

    const raw =
      (Array.isArray(rootMoves) && rootMoves.length > 0 && rootMoves) ||
      (Array.isArray(multiPv) && multiPv.length > 0 && multiPv) ||
      null;

    if (raw) {
      const byMove = new Map();
      for (const entry of raw) {
        const normalized = normalizeRootMoveEntry(entry);
        if (!normalized) continue;
        const prior = byMove.get(normalized.move);
        if (
          !prior ||
          (normalized.rank != null &&
            (prior.rank == null || normalized.rank < prior.rank)) ||
          (normalized.rank == null &&
            prior.rank == null &&
            normalized.score > prior.score)
        ) {
          byMove.set(normalized.move, normalized);
        }
      }
      let ranked = [...byMove.values()].sort((a, b) => {
        if (a.rank != null && b.rank != null && a.rank !== b.rank) {
          return a.rank - b.rank;
        }
        if (a.rank != null && b.rank == null) return -1;
        if (a.rank == null && b.rank != null) return 1;
        return b.score - a.score;
      });
      if (!best && ranked[0]?.move) best = ranked[0].move;
      // Keep engine best first even if score sort drifted.
      if (best && ALG_MOVE_RE.test(best)) {
        const idx = ranked.findIndex((e) => e.move === best);
        if (idx > 0) {
          const [top] = ranked.splice(idx, 1);
          ranked.unshift(top);
        } else if (idx < 0) {
          ranked.unshift({
            move: best,
            score: Number.isFinite(Number(rootScore))
              ? Number(rootScore)
              : Number.POSITIVE_INFINITY,
            rank: 1,
          });
        }
      }
      if (!ranked.length) {
        /* fall through to best-only */
      } else {
        const bestScore = Number(ranked[0]?.score);
        ranked = ranked.map((entry, index) => ({
          move: entry.move,
          score: entry.score,
          rank: index + 1,
          blunder: isBlunderVsBest(bestScore, entry.score),
          label: String(index + 1),
        }));
        // Keep anti-blunder for #2..N, but #1 always stays visible.
        return filterAntiBlunderGhosts(ranked, max).map((entry, index) => ({
          ...entry,
          rank: index + 1,
        }));
      }
    }

    // No root dump yet — show only the current best, never fake alternatives.
    if (best && ALG_MOVE_RE.test(best) && best !== "(none)") {
      return [
        {
          move: best,
          score: Number.isFinite(Number(rootScore)) ? Number(rootScore) : 0,
          rank: 1,
          blunder: false,
          label: "1",
        },
      ];
    }
    return [];
  }

  function legalGhosts(ghosts, detail) {
    const list = Array.isArray(ghosts) ? ghosts : [];
    if (!list.length) return [];
    const legal = list
      .filter((ghost) => isLegalGhostMove(ghost.move, detail))
      .map((ghost, index) => ({ ...ghost, rank: index + 1 }));
    // Display must not go blank if ACE legal list is briefly empty/desynced —
    // keep the engine's top move as a visual hint.
    if (!legal.length && list[0]?.move) {
      return [{ ...list[0], rank: 1 }];
    }
    return legal;
  }

  function ghostsForResult(result, detail, modes = {}) {
    const ghosts = legalGhosts(
      coalesceTopGhostMoves({
        algebraic: result?.algebraic,
        pv: result?.pv,
        depthLog: result?.depthLog,
        rootMoves: result?.rootMoves,
        multiPv: result?.multiPv,
        rootScore: result?.rootScore,
      }),
      detail,
    );
    return ghosts.map((ghost) => ({
      move: ghost.move,
      rank: ghost.rank,
      blunder: Boolean(ghost.blunder),
      mode: ghost.rank === 1 ? (modes.rank1 || "static") : "static",
      progress: ghost.rank === 1 ? modes.progress : 1,
    }));
  }

  function paintImmediateGhosts(detail, mode = "thinking") {
    if (!detail || !visualsEnabled() || settings.showGhost === false) return;
    const posKey = analysisPositionKey(detail);
    const cached =
      lastCompletedResult?.result &&
      (lastCompletedResult.posKey === posKey ||
        lastCompletedResult.fingerprint === detail.fingerprint)
        ? lastCompletedResult.result
        : null;
    let ghosts =
      cached && (cached.algebraic || cached.rootMoves || cached.multiPv)
        ? ghostsForResult(cached, detail, { rank1: mode, progress: 1 })
        : [];
    if (!ghosts.length && lastGhostMove && isLegalGhostMove(lastGhostMove, detail)) {
      ghosts = [{ move: lastGhostMove, rank: 1, mode, progress: 1 }];
    }
    if (!ghosts.length) return;
    lastGhostBoardFp = posKey || detail.fingerprint;
    lastGhostKey = null;
    renderGhostsFire(ghosts, detail.bottomPlayer, detail.game?.turn);
  }

  function playMoveCmd(text) {
    const detail = window.__quoridorsBridgeLastLocal;
    if (!canActNow(detail)) {
      return Promise.resolve({ ok: false, error: "cannot act now" });
    }
    const move = firstMoveFromPv(text);
    if (!ALG_MOVE_RE.test(move) || move === "(none)") {
      return Promise.resolve({ ok: false, error: "bad algebraic" });
    }
    if (pawnMoveIsCurrentSquare(move, detail)) {
      return Promise.resolve({ ok: false, error: "illegal or stale move" });
    }
    if (!isLegalGhostMove(move, detail)) {
      return Promise.resolve({ ok: false, error: `illegal per AceLegal: ${move}` });
    }
    return dispatchCmd("playAlgebraic", { text: move });
  }

  function pawnsAtStart(detail) {
    const p1 = detail?.game?.pawns?.p1;
    const p2 = detail?.game?.pawns?.p2;
    return p1?.x === 4 && p1?.y === 0 && p2?.x === 4 && p2?.y === 8;
  }

  async function recoverEngineFromBoard(detail, reason) {
    setStatus(`desync — resyncing engine (${String(reason)})`);
    cancelEngineSearch(reason);
    forceResetNextSync = true;
    await syncEnginePosition(detail?.algebraicMoves || [], { forceReset: true });
    lastCompletedResult = null;
    lastFingerprint = null;
    lastPlayPartial = null;
    analysisFingerprint = null;
    clearGhostFire();
  }

  /**
   * Wallz pawns are {x:file, y:rank-1}. Quor legal lists use [r,c] with
   * r = 9-rank (pawn) / 8-rank (wall). Match page_hook parseAlgebraic.
   */
  function pawnMoveIsCurrentSquare(algebraic, detail) {
    const move = firstMoveFromPv(algebraic).toLowerCase();
    if (!/^[a-i][1-9]$/.test(move)) return false;
    const player = detail?.game?.turn === "p2" ? detail.game.pawns?.p2 : detail?.game?.pawns?.p1;
    if (!player) return false;
    const file = move.charCodeAt(0) - 97;
    const rank = Number(move[1]);
    // wallz: x = file, y = rank - 1  (from quorToWallzCoord: y = 8 - r, r = 9 - rank)
    return player.x === file && player.y === rank - 1;
  }

  function isLegalGhostMove(algebraic, detail) {
    const move = firstMoveFromPv(algebraic).toLowerCase();
    if (!ALG_MOVE_RE.test(move)) return false;
    if (pawnMoveIsCurrentSquare(move, detail)) return false;
    // Prefer ACE legal list from page_hook; never gate on site legal_pawn_moves.
    const aceList = detail?.aceLegalMoves;
    if (Array.isArray(aceList) && aceList.length > 0) {
      return aceList.some((m) => String(m).toLowerCase() === move);
    }
    const Ace = typeof AceLegal !== "undefined" ? AceLegal : globalThis.AceLegal;
    if (Ace && Array.isArray(detail?.algebraicMoves)) {
      const game = Ace.create();
      Ace.loadAlgebraic(game, detail.algebraicMoves);
      return Ace.isLegal(game, move);
    }
    // No ACE available — refuse rather than fall back to site legal lists.
    return false;
  }

  async function validateMoveCmd(move) {
    const normalized = firstMoveFromPv(move);
    try {
      const res = await dispatchCmd("validateMove", { move: normalized });
      if (res && typeof res.legal === "boolean") {
        return { ok: res.ok !== false, legal: res.legal, reason: res.reason || null };
      }
    } catch {
      /* fall through to local */
    }
    const detail = window.__quoridorsBridgeLastLocal;
    return {
      ok: true,
      legal: isLegalGhostMove(normalized, detail),
      reason: "ace-local-fallback",
    };
  }

  function looksLikeEngineAtStartDesync(result, detail) {
    const plyCount = Array.isArray(detail?.algebraicMoves) ? detail.algebraicMoves.length : 0;
    if (plyCount <= 0) return false;
    const whiteDist = Number(result?.whiteDist);
    const blackDist = Number(result?.blackDist);
    if (whiteDist === 8 && blackDist === 8) return true;
    const move = firstMoveFromPv(result?.algebraic || result?.pv);
    return pawnMoveIsCurrentSquare(move, detail);
  }

  function assertOurTurn(detail) {
    if (!detail || isGameOver(detail)) return false;
    // Site chrome is authoritative (#status-line "Your move" / Interaction / dots).
    if (detail.siteSaysOurTurn === true) return true;
    if (detail.siteSaysOurTurn === false) return false;
    if (detail.isMyTurn === true) return true;
    if (detail.isMyTurn === false) return false;
    return Boolean(
      detail?.controllable &&
        detail.game?.turn === detail.bottomPlayer &&
        (detail.mySeat == null ||
          detail.turnSeat === detail.mySeat ||
          detail.game?.turnSeat === detail.mySeat),
    );
  }

  async function recoverAndRetry(detail, reason) {
    const fp = detail?.fingerprint;
    if (!rememberRecoverRetry(fp)) return false;
    await recoverEngineFromBoard(detail, reason);
    // Brief settle so forceReset sync lands before the short re-search.
    await new Promise((r) => setTimeout(r, 50));
    const fresh = window.__quoridorsBridgeLastLocal;
    if (!fresh || !canActNow(fresh)) return false;
    const retry = await genmove(fresh, { timeMs: RECOVER_RETRY_TIME_MS });
    const move = firstMoveFromPv(retry?.algebraic || retry?.pv);
    if (!move || move === "(none)" || !ALG_MOVE_RE.test(move)) {
      clearGhostFire();
      setStatus(`desync — engine retry returned ${move || "(none)"}`);
      return false;
    }
    if (pawnMoveIsCurrentSquare(move, fresh) || !isLegalGhostMove(move, fresh)) {
      clearGhostFire();
      const aceSample = (fresh.aceLegalMoves || []).slice(0, 8).join(",") || "?";
      setStatus(`titanium PV illegal per ACE: ${move} (legal: ${aceSample})`);
      return false;
    }
    const playRes = await playMoveCmd(move);
    if (!playRes || playRes.ok === false || playRes.result?.applied === false) {
      clearGhostFire();
      setStatus(`desync — retry failed: ${playRes?.error || "move not applied"}`);
      return false;
    }
    clearGhostFire();
    setStatus(`played ${move} after resync`);
    syncEnginePosition([...(fresh.algebraicMoves || []), move], { forceReset: true });
    return true;
  }

  function looksLikeIllegalPlayError(message) {
    return /illegal|stale|bad move|not legal|rejected|not applied|current square|AceLegal/i.test(
      String(message || ""),
    );
  }

  function firstMoveFromPv(pv) {
    if (Array.isArray(pv)) {
      return pv.length ? String(pv[0] || "").trim() : "";
    }
    return String(pv || "").trim().split(/\s+/)[0] || "";
  }

  function firstBestDepth(result) {
    const bestMove = String(result?.algebraic || "").trim();
    const depthLog = Array.isArray(result?.depthLog) ? result.depthLog : [];
    if (!bestMove || !depthLog.length) return null;

    let firstDepth = null;
    for (const entry of depthLog) {
      const entryDepth = Number(entry?.depth);
      if (!Number.isFinite(entryDepth) || entryDepth <= 0) continue;
      if (firstMoveFromPv(entry?.pv) !== bestMove) continue;
      if (firstDepth == null || entryDepth < firstDepth) firstDepth = entryDepth;
    }
    return firstDepth;
  }

  function complexityExtraSecForDepth(depth) {
    const value = Number(depth);
    if (!Number.isFinite(value) || value <= 2) return 0;
    const scale = Math.max(0, Number(settings.complexityScaleSec) || 0);
    return Math.min(8, value - 2) * scale;
  }

  function shouldScaleDelay(plyCount) {
    return Number(plyCount) >= OPENING_DELAY_SCALE_IGNORE_PLIES;
  }

  function isPawnAlgebraic(move) {
    return /^[a-i][1-9]$/i.test(String(move || "").trim());
  }

  function resolveHumanizerAvgSec(result) {
    const forcedEndgame = isForcedEndgameScore(result?.rootScore);
    const pawn = isPawnAlgebraic(result?.algebraic);
    if (forcedEndgame) {
      return pawn
        ? settings.endgamePawnDelayAvgSec
        : settings.endgameWallDelayAvgSec;
    }
    return pawn ? settings.pawnDelayAvgSec : settings.wallDelayAvgSec;
  }

  function markOurTurnBegan(detail) {
    const fp = detail?.fingerprint || null;
    if (!fp) {
      ourTurnBeganAt = performance.now();
      ourTurnBeganFingerprint = null;
      return;
    }
    if (ourTurnBeganFingerprint === fp && ourTurnBeganAt > 0) return;
    ourTurnBeganAt = performance.now();
    ourTurnBeganFingerprint = fp;
  }

  function clearOurTurnBegan() {
    ourTurnBeganAt = 0;
    ourTurnBeganFingerprint = null;
  }

  /**
   * Roll humanizer once when the engine stops: avg (±jitter) + complexity +
   * opening guard. That spit-out is the minimum wall time since turn began;
   * engine think already counts — wait only the shortfall.
   * Under 20s on the live clock: cap min at 1.5× engine think so we don't
   * look like we stalled after deciding (flag avoidance).
   */
  function computeHumanizerMinMs(result, plyCount) {
    if (settings.humanizerEnabled === false) return 0;
    const liveRemaining = liveClockMyMs(detailForClockRead(window.__quoridorsBridgeLastLocal));

    const forcedEndgame = isForcedEndgameScore(result?.rootScore);
    const avg = Math.max(0, Number(resolveHumanizerAvgSec(result)) || 0);
    const jitter = Math.max(0, Math.min(1, Number(settings.delayJitter) || 0));
    const spread = avg * jitter;
    let minSec = avg + (Math.random() * 2 - 1) * spread;
    if (Number(plyCount) === 0) {
      minSec += OPENING_AUTOPLAY_GUARD_SEC + Math.random() * OPENING_AUTOPLAY_GUARD_SEC * jitter;
    }
    if (!forcedEndgame && shouldScaleDelay(plyCount)) {
      minSec += complexityExtraSecForDepth(firstBestDepth(result));
    }
    if (liveRemaining != null && liveRemaining < 45_000 && liveRemaining >= 20_000) {
      minSec = Math.min(minSec, 0.3);
    }
    let minMs = Math.max(0, Math.round(minSec * 1000));

    if (liveRemaining != null && liveRemaining < 20_000) {
      const thoughtMs = Math.max(
        0,
        Number(result?.requestWallMs) || 0,
        Number(result?.searchWallMs) || 0,
        ourTurnBeganAt > 0 ? performance.now() - ourTurnBeganAt : 0,
      );
      // Total time-to-move ≤ 150% of what the engine already thought.
      minMs = Math.min(minMs, Math.round(thoughtMs * 1.5));
    }
    return Math.max(0, minMs);
  }

  /** Remaining wait after crediting time already spent since our turn began. */
  function remainingDelayMs(result, plyCount, minMs = null) {
    const targetMs =
      minMs != null && Number.isFinite(minMs)
        ? Math.max(0, Math.round(minMs))
        : computeHumanizerMinMs(result, plyCount);
    if (targetMs <= 0) return 0;
    const sinceTurnMs =
      ourTurnBeganAt > 0 ? Math.max(0, performance.now() - ourTurnBeganAt) : 0;
    const searchMs = Math.max(
      0,
      Number(result?.requestWallMs) || 0,
      Number(result?.searchWallMs) || 0,
    );
    const elapsedMs = Math.max(sinceTurnMs, searchMs);
    return Math.max(0, Math.round(targetMs - elapsedMs));
  }

  function isMyTurnLocal(detail) {
    return assertOurTurn(detail);
  }

  function formatBestStatus(algebraic, { hintSpace = false } = {}) {
    if (hintSpace && settings.autoplayEnabled !== true) {
      return `best: ${algebraic} · Space to play`;
    }
    if (usingLiveClock) {
      const myMs = liveClockMyMs(window.__quoridorsBridgeLastLocal);
      if (myMs != null) {
        return `best: ${algebraic} · clock ${fmtClock(myMs)}`;
      }
    }
    if (settings.wholeGameTime !== false) {
      const remainingSec = Math.max(
        0,
        ((Number(settings.wallClockSeconds) || 60) * 1000 - clockUsedMs) / 1000,
      );
      return `best: ${algebraic} · bank ${remainingSec.toFixed(0)}s`;
    }
    const sec = Math.max(0.2, Number(settings.timeSec) || 1.2);
    return `best: ${algebraic} · ${sec.toFixed(1)}s/move`;
  }

  async function tick(detail) {
    if (!buildHud()) return;
    if (!detail) {
      updatePlayBestEnabled();
      setStatus(sawLocalGame ? "no board detected" : "waiting for a game…");
      return;
    }

    updatePlayBestEnabled();
    prewarmEngine();

    const plyCountNow = Array.isArray(detail.algebraicMoves) ? detail.algebraicMoves.length : 0;
    const posKey = analysisPositionKey(detail);
    const positionChanged = posKey != null && posKey !== lastGhostBoardFp;
    if (positionChanged && lastGhostBoardFp != null) {
      ghostMovesByFingerprint = new Map();
      lastGhostBoardFp = posKey;
      // Drop stale eval/ghosts from the previous ply immediately. Do NOT sync
      // here — sync-before-analyze was killing the worker (log: progress only
      // for ply1, silence for ply2..N). ensureAnalysis alone applies moves.
      analysisFingerprint = null;
      clearGhostFire();
      clearAnalysisVisualState();
    } else if (posKey && !lastGhostBoardFp) {
      lastGhostBoardFp = posKey;
    }
    // Still track full fingerprint for play/search identity, but do not prune
    // ghosts when only controllable/seat noise flips inside it.
    if (busy) {
      pendingRetick = true;
      return;
    }

    if (plyCountNow < lastSeenPlyCount) {
      // Soft cancel + sync — keep WASM/TT warm across undo / new game.
      cancelEngineSearch(plyCountNow === 0 ? "new game detected" : "undo detected");
      syncEnginePosition(detail.algebraicMoves || [], { forceReset: true });
      resetClockBank();
      lastFingerprint = null;
      lastCompletedResult = null;
      analysisFingerprint = null;
      continuePlayThink = false;
      if (continuePlayTimer != null) {
        clearTimeout(continuePlayTimer);
        continuePlayTimer = null;
      }
      clearAnalysisVisualState();
      clearGhostCmd();
    }
    lastSeenPlyCount = plyCountNow;

    if (isGameOver(detail)) {
      clearGhost();
      playBestClicked = false;
      forcePlayNow = false;
      continuePlayThink = false;
      if (continuePlayTimer != null) {
        clearTimeout(continuePlayTimer);
        continuePlayTimer = null;
      }
      updatePlayBestEnabled();
      setStatus(gameOverStatus(detail));
      lastFingerprint = detail.fingerprint;
      ensureAnalysis(detail);
      return;
    }

    const myTurn = isMyTurnLocal(detail);
    if (spaceHeld && myTurn) {
      forcePlayNow = true;
      playBestClicked = true;
    }

    const samePosition = detail.fingerprint === lastFingerprint;
    if (
      canActNow(detail) &&
      settings.autoplayEnabled !== true &&
      !playBestClicked &&
      !forcePlayNow &&
      !spaceHeld &&
      !continuePlayThink
    ) {
      lastFingerprint = detail.fingerprint;
      ensureAnalysis(detail);
      setStatus("your turn · analyzing…");
      updatePlayBestEnabled();
      return;
    }
    if (samePosition && !playBestClicked && !forcePlayNow && !spaceHeld && !continuePlayThink) {
      ensureAnalysis(detail);
      if (!canActNow(detail)) setStatus("waiting for opponent…");
      return;
    }

    lastFingerprint = detail.fingerprint;

    if (!canActNow(detail)) {
      // Analysis is display-only while waiting or when the current position
      // has no legal move to apply.
      clearOurTurnBegan();
      lastPlayPartial = null;
      forcePlayNow = false;
      playBestClicked = false;
      setStatus("waiting for opponent…");
      continuePlayThink = false;
      if (continuePlayTimer != null) {
        clearTimeout(continuePlayTimer);
        continuePlayTimer = null;
      }
      ensureAnalysis(detail);
      return;
    }

    markOurTurnBegan(detail);

    if (detail.historyDesynced) {
      clearGhostCmd();
      if (rememberRecoverRetry(detail.fingerprint)) {
        await recoverEngineFromBoard(detail, "history desynced");
        await dispatchCmd("refreshState");
      }
      setStatus("desync — refreshed board; waiting for authoritative history");
      playBestClicked = false;
      return;
    }

    clearAnalysisLocalState();

    console.info("[quoridors-bridge] play turn", {
      ply: Array.isArray(detail.algebraicMoves) ? detail.algebraicMoves.length : 0,
      turn: detail.turnSeat ?? detail.game?.turn,
      mySeat: detail.mySeat,
      isMyTurn: detail.isMyTurn,
      moves: detail.algebraicMoves,
      aceLegalCount: Array.isArray(detail.aceLegalMoves) ? detail.aceLegalMoves.length : 0,
    });

    // Manual paths (Play best / Space / hold) always skip humanization delay.
    const wasManualClick = playBestClicked || forcePlayNow || spaceHeld;
    playBestClicked = false;
    continuePlayThink = false;

    busy = true;
    playSearchActive = true;
    const genAtStart = playGeneration;
    setButtonsDisabled(true);
    setStatus("your turn · thinking…");
    let didPlayMove = false;
    let result = null;
    try {
      const cached =
        wasManualClick &&
        lastCompletedResult?.fingerprint === detail.fingerprint &&
        lastCompletedResult.result?.algebraic
          ? lastCompletedResult.result
          : null;

      if (cached) {
        result = cached;
        usingLiveClock = liveClockMyMs(detail) != null;
      } else {
        const searchBudgetMs = resolveSearchTimeMs(detail);
        const hardTimeoutStartedAt = performance.now();
        playSearchStartedAt = hardTimeoutStartedAt;
        earlyPlayCancelArmed = false;
        try {
          result = await withHardTimeout(
            genmove(detail),
            searchBudgetMs + PLAY_HARD_TIMEOUT_GRACE_MS,
            () => {
              cancelEngineSearch("overlay hard search timeout");
              clearGhostCmd();
            },
          );
        } catch (hardErr) {
          if (/timed out|cancel/i.test(hardErr?.message || "") && lastPlayPartial?.algebraic) {
            const requestWallMs = Math.max(0, performance.now() - hardTimeoutStartedAt);
            result = {
              ...lastPlayPartial,
              stopReason: lastPlayPartial.stopReason || "timeout_partial",
              searchWallMs: requestWallMs,
              requestWallMs,
            };
          } else {
            throw hardErr;
          }
        }
        lastCompletedResult = { fingerprint: detail.fingerprint, posKey: analysisPositionKey(detail), result };
      }

      rememberOwnDistance(result, detail.bottomPlayer);
      const move = firstMoveFromPv(result?.algebraic || result?.pv);
      if (result) result.algebraic = move && move !== "(none)" ? move : null;

      const autoplayOn = settings.autoplayEnabled === true;
      const shouldPlay =
        canActNow(detail) &&
        (wasManualClick || forcePlayNow || spaceHeld || autoplayOn);
      const plyCount = Array.isArray(detail.algebraicMoves) ? detail.algebraicMoves.length : 0;

      const localLegal = Boolean(move && isLegalGhostMove(move, detail));
      const engineDesync = looksLikeEngineAtStartDesync(result, detail);
      const needsRecover = !move || !localLegal || engineDesync;

      if (needsRecover) {
        const aceSample = (detail.aceLegalMoves || []).slice(0, 12).join(",") || "?";
        setStatus(
          `titanium PV illegal per ACE: ${move || "(none)"} (legal: ${aceSample})`,
        );

        // Force-reset + one short ACE-gated retry. Never play ACE-illegal moves.
        const retried = await recoverAndRetry(
          detail,
          engineDesync
            ? `engine-at-start desync ${move || "(none)"}`
            : `ACE-illegal PV ${move || "(none)"}`,
        );
        if (retried) {
          didPlayMove = true;
          result = null;
          return;
        }
        clearGhostFire();
        return;
      }

      // Humanization only for pure autoplay — never for button/Space/hold.
      // Roll spit-out once when engine stops; credit think time against that min.
      const delayedAutoplay =
        shouldPlay &&
        autoplayOn &&
        !wasManualClick &&
        !forcePlayNow &&
        !spaceHeld &&
        settings.humanizerEnabled !== false;
      const humanizerMinMs = delayedAutoplay
        ? computeHumanizerMinMs(result, plyCount)
        : 0;
      const delayMs = delayedAutoplay
        ? remainingDelayMs(result, plyCount, humanizerMinMs)
        : 0;
      updateEvalBar(result, detail.bottomPlayer, playerToMoveFromPly(plyCount));

      if (delayedAutoplay && delayMs > 0) {
        await renderGhostsCmd(
          ghostsForResult(result, detail, { rank1: "countdown", progress: 0 }),
          detail.bottomPlayer,
          detail.game?.turn,
        );
        setStatus(
          `${formatBestStatus(result.algebraic)} · hold ${(delayMs / 1000).toFixed(1)}s more (min ${(humanizerMinMs / 1000).toFixed(1)}s)`,
        );
        await sleepUnlessForced(delayMs, (progress) => {
          void setGhostProgressCmd(1 - progress);
        });
      } else if (!delayedAutoplay) {
        // Hint or immediate play (Space/Play best): static full ghost.
        if (isLegalGhostMove(result.algebraic, detail)) {
          await renderGhostsCmd(
            ghostsForResult(result, detail, { rank1: "static", progress: 1 }),
            detail.bottomPlayer,
            detail.game?.turn,
          );
        }
        setStatus(formatBestStatus(result.algebraic, { hintSpace: !shouldPlay }));
      } else {
        // Autoplay with delay already exhausted: keep thinking/full ghost, play now.
        // Do NOT re-render at progress 0 (avoids countdown flash).
        await setGhostModeCmd("static");
        setStatus(formatBestStatus(result.algebraic));
      }

      if (shouldPlay) {
        // Delay+jitter already handled above for autoplay; Play best /
        // Space / hold are deliberate and play at once.
        if (!spaceHeld) forcePlayNow = false;
        const currentDetail = window.__quoridorsBridgeLastLocal;
        if (
          playGeneration !== genAtStart ||
          !currentDetail ||
          currentDetail.fingerprint !== detail.fingerprint ||
          !canActNow(currentDetail)
        ) {
          setStatus("position changed — skipped stale move");
          clearGhostFire();
          return;
        }
        if (
          !isLegalGhostMove(result.algebraic, currentDetail) ||
          pawnMoveIsCurrentSquare(result.algebraic, currentDetail) ||
          !ALG_MOVE_RE.test(result.algebraic)
        ) {
          const aceSample = (currentDetail.aceLegalMoves || []).slice(0, 12).join(",") || "?";
          setStatus(
            `titanium PV illegal per ACE: ${result.algebraic || "(none)"} (legal: ${aceSample})`,
          );
          const retried = await recoverAndRetry(detail, "ACE-illegal at play time");
          if (retried) {
            didPlayMove = true;
            result = null;
            return;
          }
          clearGhostFire();
          return;
        }
        const playRes = await playMoveCmd(result.algebraic);
        const playApplied =
          playRes &&
          playRes.ok !== false &&
          playRes.result?.applied !== false;
        if (!playApplied) {
          const error = playRes?.error || "unknown error";
          if (looksLikeIllegalPlayError(error) || /current square|not applied|desync/i.test(error)) {
            const retried = await recoverAndRetry(detail, error);
            if (retried) {
              didPlayMove = true;
              result = null;
              return;
            }
          }
          lastCompletedResult = null;
          lastFingerprint = null;
          clearGhostFire();
          setStatus(`play failed: ${error}`);
        } else {
          didPlayMove = true;
          lastCompletedResult = null;
          lastPlayPartial = null;
          lastGhostMove = null;
          setStatus(`played ${result.algebraic}`);
          clearGhostFire();
        }
      }
    } catch (err) {
      clearGhostCmd();
      const error = err && err.message ? err.message : String(err);
      if (looksLikeIllegalPlayError(error) || /current square|not applied|desync/i.test(error)) {
        const retried = await recoverAndRetry(detail, error).catch(() => false);
        if (retried) didPlayMove = true;
      }
      if (!didPlayMove) setStatus(`error: ${error}`);
    } finally {
      if (result) {
        // Charge think time for hint-only and played moves alike (site parity).
        chargeClockAfterThink(result, didPlayMove);
        lastFingerprintForClock = detail.fingerprint;
      }
      // Keep force flag while Space is held so the next turn stays instant.
      const retrySpaceHold = spaceHeld && !didPlayMove;
      forcePlayNow = spaceHeld && !didPlayMove;
      playSearchActive = false;
      busy = false;
      const lastLocal = window.__quoridorsBridgeLastLocal;
      updatePlayBestEnabled();
      const positionAdvanced =
        Boolean(lastLocal?.fingerprint) && lastLocal.fingerprint !== detail.fingerprint;
      const autoplayOn = settings.autoplayEnabled === true;
      // Only autoplay chains timed play searches. Manual/eval mode resumes
      // continuous analyze — never invent another time-managed think.
      const shouldContinuePlaySearch =
        autoplayOn &&
        canActNow(lastLocal) &&
        (pendingRetick || !didPlayMove || positionAdvanced);
      const pendingManualRetick = pendingRetick && !autoplayOn;
      pendingRetick = false;

      if (isGameOver(lastLocal)) {
        ensureAnalysis(lastLocal);
      } else if (!canActNow(lastLocal)) {
        lastPlayPartial = null;
        setStatus("waiting for opponent…");
        ensureAnalysis(lastLocal);
      } else if (!autoplayOn && canActNow(lastLocal)) {
        // Back to unlimited eval after Play best / Space / interrupted search.
        continuePlayThink = false;
        ensureAnalysis(lastLocal);
        if (!didPlayMove || pendingManualRetick) setStatus("your turn · analyzing…");
      } else if (shouldContinuePlaySearch) {
        if (continuePlayTimer != null) clearTimeout(continuePlayTimer);
        continuePlayTimer = setTimeout(() => {
          continuePlayTimer = null;
          const d = window.__quoridorsBridgeLastLocal;
          if (!busy && !playSearchActive && d && canActNow(d)) {
            continuePlayThink = true;
            void tick(d);
          }
        }, 50);
      }
      // If Space interrupted before any PV existed, kick another think while held.
      if (retrySpaceHold) {
        queueMicrotask(() => {
          const d = window.__quoridorsBridgeLastLocal;
          if (spaceHeld && !busy && d && canActNow(d)) void tick(d);
        });
      }
    }
  }

  /**
   * Trusted best move for the current board only.
   * Prefers rootMoves rank-1, then algebraic — never a stale ghost from another position.
   */
  function resolveTrustedBestMove(detail) {
    const fp = detail?.fingerprint;
    if (!fp) return null;
    const cached =
      lastCompletedResult?.fingerprint === fp ? lastCompletedResult.result : null;
    const partialOk =
      lastPlayPartial?.algebraic &&
      (lastCompletedResult?.fingerprint === fp || playSearchActive);
    const root =
      (Array.isArray(cached?.rootMoves) && cached.rootMoves[0]) ||
      (Array.isArray(cached?.multiPv) && cached.multiPv[0]) ||
      null;
    const fromRoot = firstMoveFromPv(root?.move || root?.algebraic).toLowerCase();
    const fromAlg = firstMoveFromPv(
      (partialOk && lastPlayPartial?.algebraic) || cached?.algebraic || "",
    ).toLowerCase();
    const candidate = fromRoot || fromAlg;
    if (!candidate || !ALG_MOVE_RE.test(candidate) || candidate === "(none)") return null;
    if (pawnMoveIsCurrentSquare(candidate, detail)) return null;
    if (!isLegalGhostMove(candidate, detail)) return null;
    return candidate;
  }

  async function playReadyBestImmediate(source) {
    const detail = window.__quoridorsBridgeLastLocal;
    if (!canActNow(detail)) return;

    const move = resolveTrustedBestMove(detail);
    if (!move) {
      // No trusted eval best yet — run a timed play search (play mode only).
      forcePlayNow = true;
      playBestClicked = true;
      if (!busy) void tick(detail);
      else cancelEngineSearch(source);
      return;
    }

    forcePlayNow = true;
    playGeneration += 1;
    cancelEngineSearch(source);
    clearGhostFire();
    const playRes = await playMoveCmd(move);
    const applied = playRes && playRes.ok !== false && playRes.result?.applied !== false;
    if (applied) {
      if (usingLiveClock) {
        ownMovesPlayed += 1;
      } else {
        chargeClockAfterThink({ searchWallMs: 0 }, true);
      }
      setStatus(`played ${move}`);
      lastPlayPartial = null;
      lastCompletedResult = null;
      lastGhostMove = null;
    } else {
      setStatus(`play failed: ${playRes?.error || "unknown error"}`);
      // Resume unlimited eval — do not start a timed think loop.
      ensureAnalysis(detail);
    }
  }

  function clearGhost() {
    void clearGhostCmd();
  }

  function onSpaceDown(ev) {
    if (ev.code !== "Space" && ev.key !== " ") return;
    if (isTypingTarget(ev.target)) return;
    if (!shouldHandleSpace()) return;

    // Prevent page scroll while HUD/game is active.
    ev.preventDefault();

    // Autorepeat: keep hold, do not stack plays.
    if (ev.repeat) {
      spaceHeld = true;
      return;
    }

    spaceHeld = true;
    const detail = window.__quoridorsBridgeLastLocal;
    if (isGameOver(detail)) {
      forcePlayNow = false;
      playBestClicked = false;
      return;
    }
    if (!canActNow(detail)) return;

    void playReadyBestImmediate("space");
  }

  function onSpaceUp(ev) {
    if (ev.code !== "Space" && ev.key !== " ") return;
    spaceHeld = false;
  }

  window.addEventListener("keydown", onSpaceDown, true);
  window.addEventListener("keyup", onSpaceUp, true);

  window.addEventListener("quoridors-bridge-local", (ev) => {
    sawLocalGame = true;
    lastBoardDetailAt = Date.now();
    window.__quoridorsBridgeLastLocal = ev.detail;
    prewarmEngine();
    void tick(ev.detail);
  });

  window.addEventListener("quoridors-bridge-local-heartbeat", (ev) => {
    sawLocalGame = true;
    lastBoardDetailAt = Date.now();
    const d = ev.detail || {};
    const prev = window.__quoridorsBridgeLastLocal;
    const next = { ...prev, ...d };
    const prevPos = analysisPositionKey(prev);
    const nextPos = analysisPositionKey(next);
    // Clear immediately in the heartbeat path; do not let updating
    // lastGhostBoardFp suppress tick's position-change cleanup.
    if (nextPos && nextPos !== (prevPos || lastGhostBoardFp)) {
      ghostMovesByFingerprint = new Map();
      clearGhostFire();
      clearAnalysisVisualState();
      lastGhostBoardFp = nextPos;
    }
    window.__quoridorsBridgeLastLocal = next;
    updatePlayBestEnabled();
    updateHudPlacement();
    updateVisualVisibility();
    const last = window.__quoridorsBridgeLastLocal;
    if (isGameOver(last)) {
      ensureAnalysis(last);
    }
  });

  // Streamed engine progress (play or analyze) → live eval bar + ghost.
  try {
    chrome.runtime.onMessage.addListener((msg) => {
      if (msg?.channel !== "quoridors-bridge-engine-progress") return;
      // Analyze must not clobber an in-flight play search.
      // Manual mode (autoplay off) is allowed to stream analyze on our turn.
      if (msg.kind === "analyze") {
        if (busy) return;
      }

      const algebraic =
        (typeof msg.algebraic === "string" && msg.algebraic.trim()) ||
        (typeof msg.algebraicMove === "string" && msg.algebraicMove.trim()) ||
        (typeof msg.rootMove === "string" && msg.rootMove.trim()) ||
        (typeof msg.move === "string" && msg.move.trim()) ||
        (Array.isArray(msg.pv)
          ? String(msg.pv[0] || "").trim()
          : String(msg.pv || "").trim().split(/\s+/)[0]) ||
        "";
      const move =
        algebraic && algebraic !== "(none)" ? firstMoveFromPv(algebraic) : "";
      const depth = msg.depth ?? msg.searchDepth;
      const detail = window.__quoridorsBridgeLastLocal;
      if (!detail) return;
      const posKey = analysisPositionKey(detail) || detail.fingerprint;
      const searchFingerprint = `${msg.kind || "search"}:${posKey}`;
      if (posKey !== lastGhostBoardFp) {
        ghostMovesByFingerprint = new Map();
        lastGhostBoardFp = posKey;
      }
      const streamedGhosts = legalGhosts(coalesceTopGhostMoves({
        algebraic: move,
        pv: msg.pv,
        depthLog: msg.depthLog,
        rootMoves: msg.rootMoves || msg.data?.rootMoves,
        multiPv: msg.multiPv || msg.data?.multiPv,
        rootScore: msg.rootScore,
      }), detail);
      const streamedGhostKey = `${searchFingerprint}|d${depth || 0}|${streamedGhosts
        .map((g) => `${g.move}:${g.rank}:${g.score ?? ""}`)
        .join(",")}`;

      const moveLegal = Boolean(move && isLegalGhostMove(move, detail));

      if (msg.kind === "analyze") {
        // Autoplay owns timed play searches on our turn — drop leftover analyze.
        // Manual mode (autoplay off): keep streaming unlimited analysis + top-N
        // root ghosts on our turn like an eval bar.
        if (playSearchActive || busy) return;

        if (moveLegal || streamedGhosts.length) {
          const bestMove = streamedGhosts[0]?.move || move;
          lastCompletedResult = {
            fingerprint: detail.fingerprint,
            posKey: analysisPositionKey(detail),
            result: {
              algebraic: bestMove,
              rootScore: msg.rootScore,
              depth,
              nodes: msg.nodes,
              whiteDist: msg.whiteDist,
              blackDist: msg.blackDist,
              pv: msg.pv,
              depthLog: msg.depthLog,
              rootMoves: msg.rootMoves || msg.data?.rootMoves,
              multiPv: msg.multiPv || msg.data?.multiPv,
            },
          };
          if (bestMove) lastGhostMove = bestMove;
        }
        if (settings.showEvalBar !== false) {
          const ply = Array.isArray(detail.algebraicMoves) ? detail.algebraicMoves.length : 0;
          updateEvalBar(
            {
              rootScore: msg.rootScore,
              whiteDist: msg.whiteDist,
              blackDist: msg.blackDist,
              depth,
              nodes: msg.nodes,
            },
            detail.bottomPlayer,
            playerToMoveFromPly(ply),
          );
        }
        if (streamedGhosts.length && visualsEnabled() && settings.showGhost !== false) {
          if (streamedGhostKey !== lastGhostKey) {
            lastGhostMove = streamedGhosts[0].move;
            lastGhostKey = streamedGhostKey;
            lastGhostModeFireAt = performance.now();
            renderGhostsFire(
              streamedGhosts.map((ghost) => ({
                move: ghost.move,
                rank: ghost.rank,
                blunder: Boolean(ghost.blunder),
                mode: ghost.rank === 1 ? "thinking" : "static",
                progress: 1,
              })),
              detail.bottomPlayer,
              detail.game?.turn,
            );
          } else {
            const now = performance.now();
            if (now - lastGhostHealAt >= GHOST_HEAL_MS) {
              // Heal: page_hook used to wipe ghosts on fingerprint noise while
              // overlay kept lastGhostKey — mode-only updates then painted nothing.
              lastGhostHealAt = now;
              lastGhostModeFireAt = now;
              lastGhostKey = null;
              renderGhostsFire(
                streamedGhosts.map((ghost) => ({
                  move: ghost.move,
                  rank: ghost.rank,
                  blunder: Boolean(ghost.blunder),
                  mode: ghost.rank === 1 ? "thinking" : "static",
                  progress: 1,
                })),
                detail.bottomPlayer,
                detail.game?.turn,
              );
              lastGhostKey = streamedGhostKey;
            } else if (now - lastGhostModeFireAt >= GHOST_MODE_THROTTLE_MS) {
              lastGhostModeFireAt = now;
              dispatchCmdFire("setGhostMode", { mode: "thinking" });
            }
          }
          const depthNote = Number.isFinite(Number(depth)) && Number(depth) > 0
            ? ` · d${depth}`
            : "";
          const topNote =
            streamedGhosts.length > 1
              ? ` · top ${streamedGhosts.length}`
              : "";
          setStatus(
            `analyzing · ${move}${depthNote}${topNote}`,
          );
        } else if (Number.isFinite(Number(depth)) && Number(depth) > 0) {
          setStatus(`analyzing… · d${depth}`);
        }
        return;
      }

      // Stale play progress after our search already ended (cancelled/finished) —
      // never cancel or ghost off it; stops ghost spam and early-cancel false hits.
      if (msg.kind === "play" && !playSearchActive) return;

      if (msg.kind === "play" && !canActNow(detail)) {
        if (settings.showEvalBar !== false && msg.rootScore != null) {
          const ply = Array.isArray(detail.algebraicMoves) ? detail.algebraicMoves.length : 0;
          updateEvalBar(
            {
              rootScore: msg.rootScore,
              whiteDist: msg.whiteDist,
              blackDist: msg.blackDist,
              depth,
              nodes: msg.nodes,
            },
            detail.bottomPlayer,
            playerToMoveFromPly(ply),
          );
        }
        if (lastGhostMove) {
          // Do not wipe analyze/manual ghosts when a late play tick arrives
          // off-turn; only drop play ghosts that are no longer legal.
          if (!isLegalGhostMove(lastGhostMove, detail)) clearGhostFire();
        }
        return;
      }

      if (msg.kind === "play" && moveLegal) {
        lastPlayPartial = {
          algebraic: move,
          rootScore: msg.rootScore,
          depth,
          whiteDist: msg.whiteDist,
          blackDist: msg.blackDist,
          nodes: msg.nodes,
          pv: msg.pv,
          depthLog: msg.depthLog,
          rootMoves: msg.rootMoves || msg.data?.rootMoves,
          multiPv: msg.multiPv || msg.data?.multiPv,
          stopReason: "timeout_partial",
        };

        // Autoplay early-out: legal PV, depth≥6, ≥350ms elapsed → cancel for partial.
        const autoplayOn = settings.autoplayEnabled === true;
        if (
          autoplayOn &&
          playSearchActive &&
          canActNow(detail) &&
          !earlyPlayCancelArmed &&
          Number(depth) >= EARLY_PLAY_MIN_DEPTH &&
          playSearchStartedAt > 0 &&
          performance.now() - playSearchStartedAt >= EARLY_PLAY_MIN_ELAPSED_MS
        ) {
          earlyPlayCancelArmed = true;
          cancelEngineSearch("timeout_partial");
        }
      }

      if (msg.kind === "play" && move && !moveLegal) {
        // Illegal/stale PV must not clear a previous still-legal ghost.
        if (lastGhostMove && !isLegalGhostMove(lastGhostMove, detail)) {
          clearGhostFire();
        }
        // Still update eval from illegal progress if we have scores.
        if (settings.showEvalBar !== false && msg.rootScore != null) {
          const ply = Array.isArray(detail.algebraicMoves) ? detail.algebraicMoves.length : 0;
          updateEvalBar(
            {
              rootScore: msg.rootScore,
              whiteDist: msg.whiteDist,
              blackDist: msg.blackDist,
              depth,
              nodes: msg.nodes,
            },
            detail.bottomPlayer,
            playerToMoveFromPly(ply),
          );
        }
        return;
      }

      if (settings.showEvalBar !== false) {
        const ply = Array.isArray(detail.algebraicMoves) ? detail.algebraicMoves.length : 0;
        updateEvalBar(
          {
            rootScore: msg.rootScore,
            whiteDist: msg.whiteDist,
            blackDist: msg.blackDist,
            depth,
            nodes: msg.nodes,
          },
          detail.bottomPlayer,
          playerToMoveFromPly(ply),
        );
      }

      // Live ghost from first legal PV onward; thinking march while search active.
      // Throttle mode refreshes — progress streams every info tick and used to flood wzb-fire.
      const ghostSearchActive =
        msg.kind === "play" && playSearchActive && canActNow(detail);
      if (ghostSearchActive && streamedGhosts.length && visualsEnabled() && settings.showGhost !== false) {
        if (streamedGhostKey !== lastGhostKey) {
          lastGhostMove = streamedGhosts[0].move;
          lastGhostKey = streamedGhostKey;
          lastGhostModeFireAt = performance.now();
          renderGhostsFire(
            streamedGhosts.map((ghost) => ({
              move: ghost.move,
              rank: ghost.rank,
              blunder: Boolean(ghost.blunder),
              mode: ghost.rank === 1 ? "thinking" : "static",
              progress: 1,
            })),
            detail.bottomPlayer,
            detail.game?.turn,
          );
        } else {
          const now = performance.now();
          if (now - lastGhostHealAt >= GHOST_HEAL_MS) {
            lastGhostHealAt = now;
            lastGhostModeFireAt = now;
            lastGhostKey = null;
            renderGhostsFire(
              streamedGhosts.map((ghost) => ({
                move: ghost.move,
                rank: ghost.rank,
                blunder: Boolean(ghost.blunder),
                mode: ghost.rank === 1 ? "thinking" : "static",
                progress: 1,
              })),
              detail.bottomPlayer,
              detail.game?.turn,
            );
            lastGhostKey = streamedGhostKey;
          } else if (now - lastGhostModeFireAt >= GHOST_MODE_THROTTLE_MS) {
            lastGhostModeFireAt = now;
            dispatchCmdFire("setGhostMode", { mode: "thinking" });
          }
        }
      }
      // Never clear live ghosts just because this play tick did not repaint.
      if (msg.kind === "play" && move && playSearchActive) {
        const depthNote = Number.isFinite(Number(depth)) && Number(depth) > 0 ? ` · d${depth}` : "";
        if (usingLiveClock) {
          const liveMs = liveClockMyMs(detailForClockRead(detail));
          const thinkSec = lastAlloc?.moveBudgetMs
            ? Math.min(lastAlloc.moveBudgetMs, LIVE_PLAY_SEARCH_CAP_MS) / 1000
            : null;
          const clockNote = liveMs != null ? ` · live ${fmtClock(liveMs)}` : "";
          const thinkNote = thinkSec != null ? ` · think ${thinkSec.toFixed(1)}s` : "";
          setStatus(`your turn · thinking… · ${move}${depthNote}${clockNote}${thinkNote}`);
        } else if (settings.wholeGameTime !== false) {
          const bankLeft = Math.max(
            0,
            (Number(settings.wallClockSeconds) || 60) * 1000 - clockUsedMs,
          );
          const thinkSec = lastAlloc?.moveBudgetMs
            ? Math.min(lastAlloc.moveBudgetMs, MENU_PLAY_SEARCH_CAP_MS) / 1000
            : null;
          const thinkNote = thinkSec != null ? ` · think ${thinkSec.toFixed(1)}s` : "";
          setStatus(
            `your turn · thinking… · ${move}${depthNote} · bank ${fmtClock(bankLeft)}${thinkNote}`,
          );
        } else {
          const sec = Math.max(0.2, Number(settings.timeSec) || 1.2);
          setStatus(`your turn · thinking… · ${move}${depthNote} · ${sec.toFixed(1)}s/move`);
        }
      }
    });
  } catch {
    /* extension context invalidated */
  }

  // Keep trying to mount the HUD until the board shows up.
  setInterval(() => {
    buildHud();
    updatePlayBestEnabled();
    updateHudPlacement();
    updateVisualVisibility();
    if (shouldShowHud() && !isActiveGameDetail(window.__quoridorsBridgeLastLocal)) {
      setStatus(sawLocalGame ? "reconnecting…" : "waiting for a game…");
    }
  }, 1000);
  window.addEventListener("resize", updateHudPlacement);
  window.addEventListener("scroll", updateHudPlacement, true);
})();
