/**
 * Runs in the quoridors.com page context (not the extension isolated world).
 * Hooks WebSocket traffic, API helpers, and exposes game state + move play.
 */
(function quoridorsBridgePageHook() {
  if (window.__QUORIDORS_BRIDGE__) return;

  const state = {
    ws: null,
    sock: null,
    connected: false,
    gameState: null,
    gameId: null,
    mode: null,
    aiPlayer: 1,
    mySeat: null,
    clock: null,
    lastFingerprint: null,
    lastPositionKey: null,
    lastGhostAlgebraic: null,
    lastGhostKey: null,
    lastGhostPlayer: null,
    lastReject: null,
    blockedBotHistory: 0,
    /** Latest Interaction ctx from Interaction.setState — has onMove → submitMove. */
    interactionCtx: null,
    log: [],
  };

  function pageGlobal(name) {
    try {
      // Indirect eval reads global lexical bindings (const/let) that are not on window.
      return (0, eval)(name);
    } catch {
      return undefined;
    }
  }

  function getBoard() {
    return pageGlobal("Board");
  }

  function isBotHistoryPayload(payload) {
    return payload && typeof payload === "object" && payload.kind === "ai";
  }

  function noteBlockedBotHistory(via, payload) {
    state.blockedBotHistory += 1;
    console.info("[quoridors-bridge] blocked bot game history upload", { via, kind: payload?.kind });
    window.dispatchEvent(
      new CustomEvent("quoridors-bridge-log", {
        detail: {
          ts: Date.now(),
          direction: "blocked",
          raw: JSON.stringify({ via, kind: payload?.kind }).slice(0, 4000),
        },
      }),
    );
  }

  function fakeOkJson(extra = {}) {
    return new Response(JSON.stringify({ ok: true, blocked: true, ...extra }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }

  function captureGameStateFromFetch(path, method, init, data) {
    if (!data) return;

    if (path === "/api/matchmaking/status" || path === "/api/matchmaking/queue") {
      // Site uses { status: "matched", game_id, seat, seat_token } — not `.matched`.
      if (data.status === "matched" || data.game_id) {
        const seat = normalizeSeat(data.seat);
        if (seat != null) state.mySeat = seat;
        if (data.game_id) state.gameId = data.game_id;
        state.mode = "online";
      }
    }

    if (!Array.isArray(data.pawns)) return;

    if (method === "POST" && path === "/api/game/new") {
      let body = null;
      try {
        body = init?.body ? JSON.parse(init.body) : null;
      } catch {
        /* ignore */
      }
      const mode = body?.mode ?? data.mode;
      const aiPlayer = body?.ai_player ?? data.ai_player ?? 1;
      const mySeat = mode === "ai" ? humanSeatForAi(aiPlayer) : null;
      state.sock = null;
      updateGameState(data, { mode, aiPlayer, mySeat });
      return;
    }

    if (method === "POST" && /\/api\/game\/online\/(new|join)$/.test(path)) {
      state.mode = "online";
      state.mySeat = normalizeSeat(data.seat) ?? normalizeSeat(state.mySeat);
      state.gameId = data.game_id ?? state.gameId;
      updateGameState(data, { mode: "online", mySeat: state.mySeat });
      return;
    }

    const match = path.match(/^\/api\/game\/([^/]+)(?:\/(move|ai-move|undo))?$/);
    if (match) {
      const [, gid] = match;
      if (gid !== "new" && gid !== "online") state.gameId = gid;
      updateGameState(data);
    }
  }

  async function tryUpdateStateFromFetchResponse(url, method, init, response) {
    if (!response?.ok) return;
    let path = url;
    try {
      path = new URL(url, location.origin).pathname;
    } catch {
      /* use raw url */
    }
    const isGameEndpoint =
      path === "/api/game/new" ||
      /\/api\/game\/online\/(new|join)$/.test(path) ||
      path === "/api/matchmaking/status" ||
      /^\/api\/game\/[^/]+(?:\/(?:move|ai-move|undo))?$/.test(path);
    if (!isGameEndpoint) return;
    try {
      const data = await response.clone().json();
      captureGameStateFromFetch(path, method, init, data);
    } catch {
      /* non-json or unreadable */
    }
  }

  function hookFetch() {
    if (window.__quoridorsBridgeFetchHooked) return;
    window.__quoridorsBridgeFetchHooked = true;
    const origFetch = window.fetch.bind(window);
    window.fetch = async function patchedFetch(input, init) {
      const url = typeof input === "string" ? input : input?.url || "";
      const method = String(init?.method || "GET").toUpperCase();
      if (method === "POST" && /\/api\/games\/casual(?:\?|$)/.test(url)) {
        try {
          const body = init?.body ? JSON.parse(init.body) : null;
          if (isBotHistoryPayload(body)) {
            noteBlockedBotHistory("fetch", body);
            return fakeOkJson();
          }
        } catch {
          /* fall through */
        }
      }
      const response = await origFetch(input, init);
      await tryUpdateStateFromFetchResponse(url, method, init, response);
      return response;
    };
  }

  function hookStats() {
    const Stats = pageGlobal("Stats");
    if (!Stats || Stats.__quoridorsBridgeHooked) return;
    Stats.__quoridorsBridgeHooked = true;
    const origAdd = Stats.add.bind(Stats);
    Stats.add = function patchedAdd(result, meta) {
      if (meta?.mode === "ai" || state.mode === "ai") {
        noteBlockedBotHistory("stats.add", { kind: "ai", result, meta });
        return;
      }
      return origAdd(result, meta);
    };
  }

  function pushLog(direction, raw) {
    const entry = { ts: Date.now(), direction, raw: String(raw).slice(0, 4000) };
    state.log.push(entry);
    if (state.log.length > 200) state.log.shift();
    window.dispatchEvent(new CustomEvent("quoridors-bridge-log", { detail: entry }));
  }

  function readHumanColor() {
    try {
      return parseInt(localStorage.getItem("quoridor.humanColor"), 10) === 1 ? 1 : 0;
    } catch {
      return 0;
    }
  }

  function readOnlineSeat(gid) {
    try {
      const rooms = JSON.parse(localStorage.getItem("quoridor.online") || "{}");
      for (const info of Object.values(rooms)) {
        if (info && String(info.gid) === String(gid)) return info.seat;
      }
    } catch {
      /* ignore */
    }
    return null;
  }

  function readRankedSeat(gid) {
    try {
      const info = JSON.parse(localStorage.getItem("quoridor.ranked") || "null");
      if (info && String(info.gid) === String(gid)) {
        const seat = Number(info.seat);
        return seat === 0 || seat === 1 ? seat : null;
      }
    } catch {
      /* ignore */
    }
    return null;
  }

  function normalizeSeat(value) {
    const seat = Number(value);
    return seat === 0 || seat === 1 ? seat : null;
  }

  /** Parse site clocks object/array into [ms0, ms1] or null. */
  function parseClocksMs(clocks) {
    if (!clocks) return null;
    if (Array.isArray(clocks)) {
      const c0 = Number(clocks[0]);
      const c1 = Number(clocks[1]);
      if (Number.isFinite(c0) && Number.isFinite(c1)) return [c0, c1];
      return null;
    }
    if (typeof clocks === "object") {
      const c0 = Number(clocks["0"] ?? clocks[0]);
      const c1 = Number(clocks["1"] ?? clocks[1]);
      if (Number.isFinite(c0) && Number.isFinite(c1)) return [c0, c1];
    }
    return null;
  }

  /** Read visible #time-1 / #time-2 (m:ss) for online/ranked DOM fallback. */
  function parseDomClockMs(seat) {
    const el = document.getElementById(`time-${seat + 1}`);
    if (!el || el.classList.contains("hidden")) return null;
    const text = String(el.textContent || "").trim();
    const match = text.match(/^(\d+):(\d{2})$/);
    if (!match) return null;
    const ms = (Number(match[1]) * 60 + Number(match[2])) * 1000;
    return Number.isFinite(ms) && ms > 0 ? ms : null;
  }

  function resolveStoredSeat(gid) {
    return normalizeSeat(readOnlineSeat(gid)) ?? normalizeSeat(readRankedSeat(gid));
  }

  function isOnlineLike() {
    return state.mode === "online" || Boolean(state.sock) || Boolean(state.ws);
  }

  /** Resolve online seat: live turn signals → storage → cached mySeat. */
  function inferOnlineSeat(gs) {
    const turn = normalizeSeat(gs?.turn);
    // Live chrome beats localStorage — ranked/casual seat blobs can be stale
    // and were causing us to read the opponent's clock as ours.
    if (state.interactionCtx?.isMyTurn === true && turn != null) {
      return { seat: turn, source: "interaction" };
    }
    if (document.querySelector(".move-dot") && turn != null) {
      return { seat: turn, source: "move-dot" };
    }
    if (state.interactionCtx?.isMyTurn === false && turn != null) {
      return { seat: 1 - turn, source: "interaction-opp" };
    }
    const status = readSiteTurnSignal();
    if (status === true && turn != null) {
      return { seat: turn, source: "status-line" };
    }
    if (status === false && turn != null) {
      return { seat: 1 - turn, source: "status-line-opp" };
    }
    const gid = state.gameId || gs?.game_id;
    if (gid) {
      const stored = resolveStoredSeat(gid);
      if (stored != null) return { seat: stored, source: "storage" };
    }
    const existing = normalizeSeat(state.mySeat);
    if (existing != null) return { seat: existing, source: null };
    return { seat: null, source: null };
  }

  function humanSeatForAi(aiPlayer) {
    const ai = normalizeSeat(aiPlayer);
    return ai == null ? readHumanColor() : 1 - ai;
  }

  function playerName(index) {
    return index === 0 ? "p1" : "p2";
  }

  function quorToWallzCoord(r, c) {
    return { x: c, y: 8 - r };
  }

  function wallzCoordToQuor(x, y) {
    return { r: 8 - y, c: x };
  }

  function actionToAlgebraic(action) {
    if (!action || typeof action !== "object") return "";
    if (action.type === "pawn" && Array.isArray(action.to)) {
      const [r, c] = action.to;
      return `${String.fromCharCode(97 + c)}${9 - r}`;
    }
    if (action.type === "wall" && Array.isArray(action.slot)) {
      const [r, c] = action.slot;
      const o = action.orientation === "V" ? "v" : "h";
      // Wall slots are an 8-row grid (0..7), not the 9-row pawn grid, so the
      // row-flip constant is one less than the pawn case (verified against
      // tools/quoridors_match/quoridors_match.mjs, which plays clean games
      // end-to-end against the live site with this exact mapping).
      return `${String.fromCharCode(97 + c)}${8 - r}${o}`;
    }
    return "";
  }

  function parseAlgebraic(text) {
    const move = String(text || "").trim().toLowerCase();
    const column = move[0];
    const rank = Number(move[1]);
    const c = column.charCodeAt(0) - 97;
    if (!/[a-i]/.test(column) || !Number.isInteger(rank) || rank < 1 || rank > 9) {
      throw new Error(`bad move: ${text}`);
    }
    if (move.length > 2) {
      const wallType = move[2];
      const r = 8 - rank; // wall grid is 0..7 — see actionToAlgebraic
      if ((wallType !== "h" && wallType !== "v") || c > 7 || r < 0 || r > 7) {
        throw new Error(`bad wall: ${text}`);
      }
      return { type: "wall", orientation: wallType === "h" ? "H" : "V", slot: [r, c] };
    }
    const r = 9 - rank;
    return { type: "pawn", to: [r, c] };
  }

  function bridgeMoveToAction(move) {
    if (move?.type === "pawn" && move.to) {
      if (Array.isArray(move.to)) return { type: "pawn", to: move.to.slice() };
      const { r, c } = wallzCoordToQuor(move.to.x, move.to.y);
      return { type: "pawn", to: [r, c] };
    }
    if (move?.type === "wall") {
      if (move.slot && move.orientation) {
        return {
          type: "wall",
          orientation: move.orientation,
          slot: move.slot.slice(),
        };
      }
      const w = move.wall;
      if (!w) throw new Error("bad wall move");
      // Wall grid is 0..7, one row shorter than the pawn grid — see
      // actionToAlgebraic for why this constant differs from wallzCoordToQuor.
      return {
        type: "wall",
        orientation: w.o === "v" ? "V" : "H",
        slot: [7 - w.y, w.x],
      };
    }
    throw new Error("unsupported move");
  }

  function historyToAlgebraic(history) {
    if (!Array.isArray(history)) return [];
    return history.map(actionToAlgebraic).filter(Boolean);
  }

  /** Site walls H/V are [r,c] lists; ACE uses hw/vw bit arrays. */
  function wallsMatchAce(game, gs) {
    const H = gs?.walls?.H;
    const V = gs?.walls?.V;
    const hasWalls =
      (Array.isArray(H) && H.length > 0) || (Array.isArray(V) && V.length > 0);
    if (!hasWalls) return true;
    let aceH = 0;
    let aceV = 0;
    for (let s = 0; s < 64; s++) {
      if (game.hw[s]) aceH += 1;
      if (game.vw[s]) aceV += 1;
    }
    const siteH = Array.isArray(H) ? H.length : 0;
    const siteV = Array.isArray(V) ? V.length : 0;
    if (aceH !== siteH || aceV !== siteV) return false;
    if (Array.isArray(H)) {
      for (const slotRC of H) {
        if (!Array.isArray(slotRC) || slotRC.length < 2) return false;
        const slot = slotRC[0] * 8 + slotRC[1];
        if (!game.hw[slot]) return false;
      }
    }
    if (Array.isArray(V)) {
      for (const slotRC of V) {
        if (!Array.isArray(slotRC) || slotRC.length < 2) return false;
        const slot = slotRC[0] * 8 + slotRC[1];
        if (!game.vw[slot]) return false;
      }
    }
    return true;
  }

  /**
   * Prefer history algebraics when ACE replay matches site turn+pawns (+walls).
   * Otherwise mark historyDesynced and kick an API refresh when possible.
   */
  function buildVerifiedMoves(gs) {
    const fromHist = historyToAlgebraic(gs?.history);
    const Ace = typeof AceLegal !== "undefined" ? AceLegal : globalThis.AceLegal;
    if (!Ace || !gs) {
      return { moves: fromHist, verified: false, historyDesynced: false };
    }
    const probe = Ace.create();
    Ace.loadAlgebraic(probe, fromHist);
    const turnOk = gs.turn == null || probe.turn === gs.turn;
    const pawnsOk =
      typeof Ace.pawnsMatchSite === "function"
        ? Ace.pawnsMatchSite(probe, gs)
        : true;
    const wallsOk = wallsMatchAce(probe, gs);
    if (turnOk && pawnsOk && wallsOk) {
      return { moves: fromHist, verified: true, historyDesynced: false };
    }
    // Overlay already refreshes on historyDesynced; avoid sync thrash here.
    return { moves: fromHist, verified: false, historyDesynced: true };
  }

  function refreshSeatFromStorage(gs) {
    if (!isOnlineLike() && !state.gameId) return;
    const inferred = inferOnlineSeat(gs || state.gameState);
    if (inferred.seat != null) state.mySeat = inferred.seat;
  }

  function updateGameState(next, meta = {}) {
    if (!next || !Array.isArray(next.pawns)) return;
    state.gameState = next;
    if (next.game_id && next.game_id !== state.gameId) {
      // New game: drop any leftover timer from the previous match.
      state.clock = null;
      state.gameId = next.game_id;
    } else if (next.game_id) {
      state.gameId = next.game_id;
    }
    if (meta.mode) state.mode = meta.mode;
    if (meta.aiPlayer != null) state.aiPlayer = meta.aiPlayer;
    if (meta.mySeat != null) state.mySeat = normalizeSeat(meta.mySeat);
    refreshSeatFromStorage(next);
    if (next.code) state.mode = state.mode || "online";
    // Server-authoritative ranked/online clocks (mirrors main.js): only update
    // when next.clocks is present. Intermediate messages often omit clocks —
    // do not wipe the previous snapshot. Clear when the game ends.
    const parsedClocks = parseClocksMs(next.clocks);
    if (parsedClocks) {
      state.clock = {
        ms: parsedClocks,
        turn: next.turn,
        stamp: performance.now(),
        live: next.winner == null,
      };
    }
    if (next.winner != null) {
      state.clock = null;
    }
    window.dispatchEvent(new CustomEvent("quoridors-bridge-state", { detail: next }));
    emitLocalDetail();
  }

  async function refreshStateFromApi() {
    if (!state.gameId) return null;
    const API = pageGlobal("API");
    if (!API?.getState) return null;
    try {
      const next = await API.getState(state.gameId);
      if (next && Array.isArray(next.pawns)) {
        updateGameState(next);
        return next;
      }
    } catch {
      /* ignore */
    }
    return null;
  }

  function localSeat() {
    if (state.mode === "online") return state.mySeat;
    if (state.mode === "ai") return humanSeatForAi(state.aiPlayer);
    if (state.mode === "hotseat") return null;
    return null;
  }

  /** Mirror scraped main.js isMyTurnNow (AI / online / hotseat). */
  /** Authoritative site chrome: #status-line + Interaction + move dots. */
  function readSiteTurnSignal() {
    const sl = document.getElementById("status-line");
    const text = String(sl?.textContent || "").trim();
    if (/^Your move/i.test(text)) return true;
    if (/Opponent'?s move/i.test(text)) return false;
    if (/Computer is thinking/i.test(text)) return false;
    if (/Opponent disconnected/i.test(text)) return false;
    if (state.interactionCtx?.isMyTurn === true) return true;
    if (state.interactionCtx?.isMyTurn === false) return false;
    if (document.querySelector(".move-dot")) return true;
    return null;
  }

  function isMyTurnNow(gs) {
    if (!gs || gs.winner != null || Boolean(gs.is_draw)) return false;
    if (Boolean(gs.resigned) || gs.end_reason === "resign") return false;
    if (Boolean(gs.lost_on_time) || gs.end_reason === "timeout" || gs.end_reason === "time") {
      return false;
    }
    // Site status line / Interaction / move-dots beat our seat bookkeeping.
    const siteSignal = readSiteTurnSignal();
    if (siteSignal === true) {
      const turn = normalizeSeat(gs.turn);
      if (turn != null && isOnlineLike()) state.mySeat = turn;
      return true;
    }
    if (siteSignal === false) return false;

    const mode = state.mode || (state.sock || state.ws ? "online" : null);
    if (mode === "ai") {
      const aiPlayer = normalizeSeat(state.aiPlayer);
      return aiPlayer != null && gs.turn !== aiPlayer;
    }
    if (mode === "online" || state.sock || state.ws) {
      refreshSeatFromStorage(gs);
      const seat = normalizeSeat(state.mySeat);
      if (seat != null && gs.turn === seat) return true;
      if (state.interactionCtx?.isMyTurn === true) {
        const turn = normalizeSeat(gs.turn);
        if (turn != null) state.mySeat = turn;
        return true;
      }
      if (document.querySelector(".move-dot")) {
        const turn = normalizeSeat(gs.turn);
        if (turn != null) state.mySeat = turn;
        return true;
      }
      return false;
    }
    if (mode === "hotseat") return true;
    return false;
  }

  function isControllable(gs) {
    if (!gs || gs.winner != null || Boolean(gs.is_draw)) return false;
    if (Boolean(gs.resigned) || gs.end_reason === "resign") return false;
    if (Boolean(gs.lost_on_time) || gs.end_reason === "timeout" || gs.end_reason === "time") {
      return false;
    }
    return isMyTurnNow(gs);
  }

  function resolveLiveGameState() {
    const live = pageGlobal("state");
    if (live && Array.isArray(live.pawns) && live.pawns.length >= 2) {
      // Prefer live page state; keep mirror in sync for playMove paths.
      state.gameState = live;
      return live;
    }
    return state.gameState;
  }

  function aceGameFromGs(gs, algebraicMoves) {
    const Ace = typeof AceLegal !== "undefined" ? AceLegal : globalThis.AceLegal;
    if (!Ace) return null;
    const game = Ace.create();
    Ace.loadFromSiteState(game, gs, algebraicMoves);
    return game;
  }

  function computeHistoryDesynced(gs, algebraicMoves) {
    const emptyHist = (gs.history?.length || 0) === 0 && algebraicMoves.length === 0;
    const atStart =
      gs.pawns[0]?.[0] === 8 &&
      gs.pawns[0]?.[1] === 4 &&
      gs.pawns[1]?.[0] === 0 &&
      gs.pawns[1]?.[1] === 4;
    if (emptyHist && !atStart) return true;

    const plyCount = algebraicMoves.length;
    const expectedTurnFromPly = plyCount % 2;
    if (gs.turn != null && expectedTurnFromPly !== gs.turn) return true;

    const Ace = typeof AceLegal !== "undefined" ? AceLegal : globalThis.AceLegal;
    if (Ace && algebraicMoves.length > 0) {
      const probe = Ace.create();
      Ace.loadAlgebraic(probe, algebraicMoves);
      if (probe.turn !== gs.turn) return true;
      if (typeof Ace.pawnsMatchSite === "function" && !Ace.pawnsMatchSite(probe, gs)) {
        return true;
      }
    }
    return false;
  }

  function buildBridgeDetail() {
    const gs = resolveLiveGameState();
    if (!gs || !Array.isArray(gs.pawns) || gs.pawns.length < 2) return null;

    const gameOver = gs.winner != null || Boolean(gs.is_draw);
    const resigned = Boolean(gs.resigned) || gs.end_reason === "resign";
    const onTime =
      Boolean(gs.lost_on_time) || gs.end_reason === "timeout" || gs.end_reason === "time";
    const terminal = gameOver || resigned || onTime || gs.winner != null;
    const mode = state.mode || (state.sock || state.ws ? "online" : "local");
    // AI: human seat = opposite of aiPlayer / humanColor.
    // Online: never invent seat 0 — leave null until ranked/casual storage or Interaction says so.
    let seatSource = null;
    if (mode === "online" || state.sock || state.ws) {
      const inferred = inferOnlineSeat(gs);
      if (inferred.seat != null) state.mySeat = inferred.seat;
      seatSource = inferred.source;
    }
    let mySeat =
      mode === "ai"
        ? humanSeatForAi(state.aiPlayer)
        : mode === "online" || state.sock || state.ws
          ? normalizeSeat(state.mySeat)
          : normalizeSeat(localSeat());
    if (mode === "ai") state.mySeat = mySeat;
    const verified = buildVerifiedMoves(gs);
    const algebraicMoves = verified.moves;
    const plyCount = algebraicMoves.length;
    const expectedTurnFromPly = plyCount % 2;
    const historyDesynced =
      verified.historyDesynced || computeHistoryDesynced(gs, algebraicMoves);
    const siteSaysOurTurn = readSiteTurnSignal();
    const myTurn = isMyTurnNow(gs);
    // Site chrome is authoritative for who we are. Storage can be stale/wrong;
    // isMyTurnNow may have already corrected state.mySeat — sync the local copy
    // before clock / bottomPlayer so we never budget from the enemy timer.
    if (mode === "online" || state.sock || state.ws) {
      const turn = normalizeSeat(gs.turn);
      if (siteSaysOurTurn === true && turn != null) {
        mySeat = turn;
        state.mySeat = turn;
        seatSource = seatSource || "site-turn";
      } else if (siteSaysOurTurn === false && turn != null) {
        mySeat = 1 - turn;
        state.mySeat = mySeat;
        seatSource = seatSource || "site-opp-turn";
      } else {
        mySeat = normalizeSeat(state.mySeat) ?? mySeat;
      }
    }
    const bottomPlayer = mySeat === 1 ? "p2" : mySeat === 0 ? "p1" : "p1";
    const controllable = myTurn && !terminal;
    const aceGame = aceGameFromGs(gs, algebraicMoves);
    const Ace = typeof AceLegal !== "undefined" ? AceLegal : globalThis.AceLegal;
    const aceLegalMoves = aceGame && Ace ? Ace.legalAlgebraic(aceGame) : [];
    const fingerprint = [
      state.gameId || "-",
      algebraicMoves.length,
      gs.turn,
      gs.pawns[0]?.join(","),
      gs.pawns[1]?.join(","),
      gs.winner ?? "-",
      gs.is_draw ? 1 : 0,
      resigned ? 1 : 0,
      onTime ? 1 : 0,
      controllable ? 1 : 0,
      mode || "-",
      verified.verified ? 1 : 0,
    ].join(":");

    return {
      mode,
      source: "quoridors-state",
      game: {
        turn: playerName(gs.turn),
        winner: gs.winner != null ? playerName(gs.winner) : null,
        pawns: {
          p1: quorToWallzCoord(gs.pawns[0][0], gs.pawns[0][1]),
          p2: quorToWallzCoord(gs.pawns[1][0], gs.pawns[1][1]),
        },
        isDraw: Boolean(gs.is_draw),
        resigned,
        onTime,
        endReason: gs.end_reason || null,
        gameOver: terminal,
      },
      terminal,
      moves: gs.history || [],
      algebraicMoves,
      movesVerified: verified.verified,
      plyCount,
      expectedTurnFromPly,
      legalPawnMoves: gs.legal_pawn_moves || [],
      legalWallPlacements: gs.legal_wall_placements || [],
      aceLegalMoves,
      historyDesynced,
      bottomPlayer,
      controllable,
      isMyTurn: myTurn,
      siteSaysOurTurn,
      fingerprint,
      mySeat,
      seatSource,
      aiPlayer: state.aiPlayer,
      turnSeat: gs.turn,
      sideToMove: playerName(gs.turn),
      mode,
      modeRaw: state.mode,
      clock: (() => {
        const seat = mySeat;
        if (seat == null) return null;
        const isOnlineLike =
          mode === "online" || state.sock || state.ws || state.mode === "online";
        // Do not invent clocks for AI/hotseat.
        if (!isOnlineLike) return null;

        const turnSeat = normalizeSeat(gs.turn);
        // On our turn the side-to-move clock IS ours — never index by a stale seat.
        const ourSeat =
          (myTurn || siteSaysOurTurn === true) && turnSeat != null ? turnSeat : seat;
        const theirSeat = 1 - ourSeat;

        let source = "game-state";
        let myMs = null;
        let oppMs = null;
        let turn = gs.turn;
        let live = false;

        if (state.clock && state.clock.ms[0] != null) {
          myMs = state.clock.ms[ourSeat];
          oppMs = state.clock.ms[theirSeat];
          turn = state.clock.turn;
          live = state.clock.live;
          // Smooth countdown like main.js renderRankedClocks
          if (state.clock.live && turnSeat === ourSeat) {
            myMs = Math.max(0, myMs - (performance.now() - state.clock.stamp));
          } else if (state.clock.live && turnSeat === theirSeat) {
            oppMs = Math.max(0, oppMs - (performance.now() - state.clock.stamp));
          }
        } else {
          // Fallback: parse visible ranked/online clock DOM when server snapshot missing.
          const dom0 = parseDomClockMs(0);
          const dom1 = parseDomClockMs(1);
          if (dom0 != null && dom1 != null) {
            myMs = ourSeat === 0 ? dom0 : dom1;
            oppMs = theirSeat === 0 ? dom0 : dom1;
            live = myTurn && !terminal;
            source = "dom";
          }
        }

        if (!(Number.isFinite(myMs) && myMs > 0)) return null;
        return {
          myMs,
          oppMs: Number.isFinite(oppMs) ? oppMs : null,
          turn,
          live,
          source,
          seat: ourSeat,
        };
      })(),
    };
  }

  function positionKeyFromDetail(detail) {
    if (!detail) return null;
    const p1 = detail.game?.pawns?.p1;
    const p2 = detail.game?.pawns?.p2;
    return [
      detail.gameId || "-",
      detail.plyCount ?? (detail.algebraicMoves?.length ?? 0),
      detail.turnSeat ?? detail.game?.turn ?? "-",
      p1 ? `${p1.x},${p1.y}` : "-",
      p2 ? `${p2.x},${p2.y}` : "-",
      detail.terminal || detail.game?.gameOver ? 1 : 0,
    ].join(":");
  }

  function emitLocalDetail() {
    const detail = buildBridgeDetail();
    if (!detail) return;
    if (detail.fingerprint === state.lastFingerprint) {
      window.dispatchEvent(
        new CustomEvent("quoridors-bridge-local-heartbeat", { detail }),
      );
      return;
    }
    // Fingerprint includes controllable/seat noise — do NOT wipe ghosts unless
    // the actual board position changed. That was why suggestions vanished
    // after the first paint.
    const posKey = positionKeyFromDetail(detail);
    if (posKey && posKey !== state.lastPositionKey) {
      clearGhost();
      state.lastPositionKey = posKey;
    }
    state.lastFingerprint = detail.fingerprint;
    window.dispatchEvent(new CustomEvent("quoridors-bridge-local", { detail }));
  }

  function hookWebSocket(ws, url) {
    if (!String(url).includes("/ws/game/")) return;
    state.ws = ws;
    state.connected = ws.readyState === WebSocket.OPEN;

    const origSend = ws.send.bind(ws);
    ws.send = function patchedSend(data) {
      pushLog("out", data);
      try {
        const msg = JSON.parse(data);
        window.dispatchEvent(
          new CustomEvent("quoridors-bridge-event", {
            detail: { direction: "out", ...msg },
          }),
        );
      } catch {
        /* non-json */
      }
      return origSend(data);
    };

    ws.addEventListener("open", () => {
      state.connected = true;
      window.dispatchEvent(
        new CustomEvent("quoridors-bridge-status", { detail: { connected: true } }),
      );
    });
    ws.addEventListener("close", () => {
      state.connected = false;
      state.ws = null;
      window.dispatchEvent(
        new CustomEvent("quoridors-bridge-status", { detail: { connected: false } }),
      );
    });
    ws.addEventListener("message", (ev) => {
      pushLog("in", ev.data);
      try {
        const msg = JSON.parse(ev.data);
        window.dispatchEvent(
          new CustomEvent("quoridors-bridge-event", {
            detail: { direction: "in", ...msg },
          }),
        );
        if (msg.type === "state") updateGameState(msg);
        if (msg.type === "error") {
          state.lastReject = msg;
          window.dispatchEvent(new CustomEvent("quoridors-bridge-reject", { detail: msg }));
        }
      } catch {
        /* ignore */
      }
    });
  }

  const OrigWebSocket = window.WebSocket;
  function PatchedWebSocket(url, protocols) {
    const ws =
      protocols !== undefined ? new OrigWebSocket(url, protocols) : new OrigWebSocket(url);
    try {
      hookWebSocket(ws, url);
    } catch (err) {
      console.warn("[quoridors-bridge] ws hook failed", err);
    }
    return ws;
  }
  PatchedWebSocket.prototype = OrigWebSocket.prototype;
  for (const key of Object.getOwnPropertyNames(OrigWebSocket)) {
    if (key !== "prototype" && key !== "length" && key !== "name") {
      try {
        PatchedWebSocket[key] = OrigWebSocket[key];
      } catch {
        /* read-only */
      }
    }
  }
  window.WebSocket = PatchedWebSocket;

  function hookAPI() {
    const API = pageGlobal("API");
    if (!API || API.__quoridorsBridgeHooked) return;
    API.__quoridorsBridgeHooked = true;

    const origConnect = API.connect.bind(API);
    API.connect = function connectWrapped(gid, token, handlers = {}) {
      const storedSeat = resolveStoredSeat(gid);
      if (storedSeat != null) state.mySeat = storedSeat;
      const wrapped = { ...handlers };
      wrapped.onState = (msg) => {
        if (msg?.type === "state") {
          state.mode = "online";
          state.gameId = gid;
          const seat = resolveStoredSeat(gid);
          if (seat != null) state.mySeat = seat;
          updateGameState(msg);
        }
        handlers.onState?.(msg);
      };
      wrapped.onError = (msg) => {
        state.lastReject = msg;
        window.dispatchEvent(new CustomEvent("quoridors-bridge-reject", { detail: msg }));
        handlers.onError?.(msg);
      };
      wrapped.onOpen = () => {
        state.connected = true;
        handlers.onOpen?.();
      };
      const sock = origConnect(gid, token, wrapped);
      state.sock = sock;
      state.gameId = gid;
      return sock;
    };

    const origNewGame = API.newGame.bind(API);
    API.newGame = async function newGameWrapped(mode, aiPlayer = 1, difficulty = null) {
      const result = await origNewGame(mode, aiPlayer, difficulty);
      state.mode = mode;
      state.aiPlayer = aiPlayer;
      state.mySeat = mode === "ai" ? humanSeatForAi(aiPlayer) : null;
      state.sock = null;
      updateGameState(result, { mode, aiPlayer, mySeat: state.mySeat });
      return result;
    };

    const wrapStateFn = (fn, after) => async (...args) => {
      const result = await fn(...args);
      if (result?.pawns) {
        if (after) after(...args, result);
        updateGameState(result);
      }
      return result;
    };

    API.move = wrapStateFn(API.move.bind(API));
    API.aiMove = wrapStateFn(API.aiMove.bind(API));
    API.getState = wrapStateFn(API.getState.bind(API));
    API.undo = wrapStateFn(API.undo.bind(API));

    const origJoin = API.joinOnline?.bind(API);
    if (origJoin) {
      API.joinOnline = async (...args) => {
        const result = await origJoin(...args);
        state.mode = "online";
        state.mySeat = normalizeSeat(result.seat);
        state.gameId = result.game_id;
        updateGameState(result, { mode: "online", mySeat: state.mySeat });
        return result;
      };
    }

    const origCreate = API.createOnline?.bind(API);
    if (origCreate) {
      API.createOnline = async (...args) => {
        const result = await origCreate(...args);
        state.mode = "online";
        state.mySeat = normalizeSeat(result.seat);
        state.gameId = result.game_id;
        updateGameState(result, { mode: "online", mySeat: state.mySeat });
        return result;
      };
    }

    const origRecordCasual = API.recordCasual?.bind(API);
    if (origRecordCasual) {
      API.recordCasual = async function recordCasualWrapped(payload) {
        if (isBotHistoryPayload(payload)) {
          noteBlockedBotHistory("api.recordCasual", payload);
          return { ok: true, blocked: true };
        }
        return origRecordCasual(payload);
      };
    }

    function applyMatchmakingHandoff(result) {
      if (!result) return;
      if (result.status !== "matched" && !result.game_id) return;
      const seat = normalizeSeat(result.seat);
      if (seat != null) state.mySeat = seat;
      if (result.game_id) state.gameId = result.game_id;
      state.mode = "online";
    }

    const origMatchmakingStatus = API.matchmakingStatus?.bind(API);
    if (origMatchmakingStatus) {
      API.matchmakingStatus = async (...args) => {
        const result = await origMatchmakingStatus(...args);
        applyMatchmakingHandoff(result);
        return result;
      };
    }

    const origQueueRanked = API.queueRanked?.bind(API);
    if (origQueueRanked) {
      API.queueRanked = async (...args) => {
        const result = await origQueueRanked(...args);
        applyMatchmakingHandoff(result);
        return result;
      };
    }
  }

  // ---------------------------------------------------------------------
  // Move EXECUTION — prefer Interaction.onMove (same path as a real click →
  // submitMove → renderAll / sock.sendMove). Bare API.move() skips the site's
  // render loop and leaves the board stale. DOM click is kept only as fallback
  // when Interaction ctx is not yet captured.
  // ---------------------------------------------------------------------

  function hookInteraction() {
    const Interaction = pageGlobal("Interaction");
    if (!Interaction || Interaction.__quoridorsBridgeHooked) return;
    if (typeof Interaction.setState !== "function") return;
    Interaction.__quoridorsBridgeHooked = true;
    const origSetState = Interaction.setState.bind(Interaction);
    Interaction.setState = function setStateWrapped(s, c) {
      state.interactionCtx = c;
      // Site already computed isMyTurn from its real seat — learn seat when we can move.
      if (
        c?.isMyTurn === true &&
        s &&
        (state.mode === "online" || state.sock || state.ws) &&
        normalizeSeat(s.turn) != null
      ) {
        state.mySeat = normalizeSeat(s.turn);
      }
      if (s && Array.isArray(s.pawns)) {
        updateGameState(s);
      }
      return origSetState(s, c);
    };
  }

  function boardPoint(pctX, pctY) {
    const Board = getBoard();
    if (!Board?.boardEl) throw new Error("Board not ready");
    const rect = Board.boardEl.getBoundingClientRect();
    return { x: rect.left + (pctX / 100) * rect.width, y: rect.top + (pctY / 100) * rect.height };
  }

  function fireAt(el, x, y, type) {
    const opts = { clientX: x, clientY: y, bubbles: true, cancelable: true, view: window };
    if (type === "pointermove") {
      el.dispatchEvent(new PointerEvent(type, { ...opts, pointerId: 1, pointerType: "mouse", isPrimary: true }));
    } else {
      el.dispatchEvent(new MouseEvent(type, opts));
    }
  }

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async function clickPawn(r, c) {
    const dot = document.querySelector(`.move-dot[data-r="${r}"][data-c="${c}"]`);
    if (!dot) throw new Error(`no move-dot at ${r},${c} (not our turn, or illegal target)`);
    dot.click();
  }

  // H and V walls at the same true (r,c) share one pixel centre (verified
  // live), so an exact-centre click is an ambiguous tie that interaction.js
  // resolves via a hidden default ('H'). To force V we nudge off-centre past
  // interaction.js's own INTERSECTION tie zone (CELL*0.28) but short of its
  // IN_GROOVE cutoff (CELL*0.42) — CELL*0.32 lands squarely in that window.
  async function clickWall(orientation, r, c) {
    const Board = getBoard();
    if (!Board?.boardEl) throw new Error("Board not ready");
    const g = Board.wallGeom(orientation, r, c);
    const nudge = orientation === "V" ? Board.geom.CELL * 0.32 : 0;
    const { x, y } = boardPoint(g.x, g.y + nudge);
    fireAt(Board.boardEl, x, y, "pointermove");
    await sleep(60);
    fireAt(Board.boardEl, x, y, "click");
  }

  function isLegalAlgebraic(text) {
    let action;
    try {
      action = parseAlgebraic(text);
    } catch (err) {
      return { ok: false, legal: false, reason: String(err?.message || err) };
    }
    const gs = resolveLiveGameState();
    if (!gs) return { ok: true, legal: false, reason: "no game state" };
    if (
      gs.winner != null ||
      gs.is_draw ||
      Boolean(gs.resigned) ||
      gs.end_reason === "resign" ||
      Boolean(gs.lost_on_time) ||
      gs.end_reason === "timeout" ||
      gs.end_reason === "time"
    ) {
      return { ok: true, legal: false, reason: "game over" };
    }
    if (
      action.type === "pawn" &&
      gs.pawns &&
      gs.turn != null &&
      gs.pawns[gs.turn]?.[0] === action.to?.[0] &&
      gs.pawns[gs.turn]?.[1] === action.to?.[1]
    ) {
      return { ok: true, legal: false, reason: "current square" };
    }
    const Ace = typeof AceLegal !== "undefined" ? AceLegal : globalThis.AceLegal;
    if (!Ace) {
      return { ok: true, legal: false, reason: "AceLegal unavailable" };
    }
    const algebraic = actionToAlgebraic(action);
    const game = Ace.create();
    Ace.loadFromSiteState(game, gs, historyToAlgebraic(gs.history));
    if (!Ace.isLegal(game, algebraic)) {
      return { ok: true, legal: false, reason: "not in AceLegal moves" };
    }
    return { ok: true, legal: true, reason: null };
  }

  async function playMove(action) {
    if (!action || typeof action !== "object") throw new Error("bad action");

    const gs = resolveLiveGameState();
    if (
      gs?.winner != null ||
      gs?.is_draw ||
      Boolean(gs?.resigned) ||
      gs?.end_reason === "resign" ||
      Boolean(gs?.lost_on_time) ||
      gs?.end_reason === "timeout" ||
      gs?.end_reason === "time"
    ) {
      throw new Error("game over — cannot play");
    }
    if (
      action.type === "pawn" &&
      gs?.pawns &&
      gs.turn != null &&
      gs.pawns[gs.turn]?.[0] === action.to?.[0] &&
      gs.pawns[gs.turn]?.[1] === action.to?.[1]
    ) {
      throw new Error("illegal pawn move: destination is current square");
    }
    const Ace = typeof AceLegal !== "undefined" ? AceLegal : globalThis.AceLegal;
    if (Ace && gs) {
      const algebraic = actionToAlgebraic(action);
      const game = Ace.create();
      Ace.loadFromSiteState(game, gs, historyToAlgebraic(gs.history));
      if (!Ace.isLegal(game, algebraic)) {
        throw new Error(`illegal move per AceLegal: ${algebraic}`);
      }
    }

    const beforeFp = state.lastFingerprint;
    const beforeHistLen = Array.isArray(gs?.history) ? gs.history.length : null;
    const beforePawnKey = Array.isArray(gs?.pawns)
      ? gs.pawns.map((p) => p.join(",")).join("|")
      : null;

    // Pawn dots follow the site's human click path. Interaction.onMove is the
    // fallback for hidden dots and remains preferred for wall confirmation.
    const ctx = state.interactionCtx;
    let mode;
    if (action.type === "pawn") {
      const [r, c] = action.to;
      const dot = document.querySelector(`.move-dot[data-r="${r}"][data-c="${c}"]`);
      if (dot) {
        dot.click();
        mode = "move-dot-click";
      } else if (ctx && typeof ctx.onMove === "function") {
        const result = ctx.onMove(action);
        if (result?.then) await result;
        mode = "interaction-onMove";
      } else {
        await clickPawn(r, c);
        mode = "dom-click";
      }
    } else if (ctx && typeof ctx.onMove === "function") {
      try {
        const result = ctx.onMove(action);
        if (result?.then) await result;
      } catch (e) {
        throw new Error(e?.detail || e?.message || "Illegal move");
      }
      mode = "interaction-onMove";
    } else {
      // Fallback: synthesize the same DOM gestures a mouse would use.
      if (action.type === "pawn") {
        const [r, c] = action.to;
        await clickPawn(r, c);
      } else if (action.type === "wall") {
        const [r, c] = action.slot;
        await clickWall(action.orientation, r, c);
      } else {
        throw new Error(`unsupported move type: ${action.type}`);
      }
      mode = "dom-click";
    }

    const deadline = performance.now() + 2000;
    while (true) {
      emitLocalDetail();
      const currentGs = state.gameState;
      const currentPawnKey = Array.isArray(currentGs?.pawns)
        ? currentGs.pawns.map((p) => p.join(",")).join("|")
        : null;
      const currentHistLen = Array.isArray(currentGs?.history)
        ? currentGs.history.length
        : null;
      if (
        state.lastFingerprint !== beforeFp ||
        (beforeHistLen != null && currentHistLen > beforeHistLen) ||
        (beforePawnKey != null && currentPawnKey !== beforePawnKey)
      ) {
        return { mode, action, applied: true };
      }
      if (performance.now() >= deadline) break;
      await sleep(40);
    }
    throw new Error("move not applied (board state unchanged)");
  }

  // ---------------------------------------------------------------------
  // Ghost overlay — independent of site #preview-layer.
  // Site interaction.js clears #preview-layer on every groove hover via
  // Board.clearPreview / Board.showPreview, which also wiped Board.showGhost.
  // ---------------------------------------------------------------------

  const GHOST_LAYER_ID = "quoridors-bridge-ghost-layer";
  const GHOST_ID = "quoridors-bridge-ghost";
  const GHOST_STYLE_ID = "quoridors-bridge-ghost-styles";
  const BLUNDER_GHOST_COLORS = { stroke: "#ff4d62", fill: "rgba(255,77,98,0.22)" };

  function ghostColorForPlayer(player) {
    return player === "p2"
      ? { stroke: "#ff6b86", fill: "rgba(255, 107, 134, 0.22)" }
      : { stroke: "#2dd4c8", fill: "rgba(45, 212, 200, 0.22)" };
  }

  // Fixed dash pattern: pathLength=100, units are % of perimeter.
  // Countdown reveal appends whole units (hide-rest pattern).
  // Thinking mode uses a repeating dash/gap that marches via dashoffset.
  const GHOST_DASH = 2.5;
  const GHOST_GAP = 1.6;
  const GHOST_UNIT = GHOST_DASH + GHOST_GAP;
  const GHOST_UNIT_COUNT = Math.max(1, Math.round(100 / GHOST_UNIT));
  const GHOST_ACTUAL_UNIT = 100 / GHOST_UNIT_COUNT;
  const GHOST_ACTUAL_DASH = GHOST_ACTUAL_UNIT * (GHOST_DASH / GHOST_UNIT);
  const GHOST_ACTUAL_GAP = GHOST_ACTUAL_UNIT - GHOST_ACTUAL_DASH;

  function clampProgress(progress) {
    const value = Number(progress);
    if (!Number.isFinite(value)) return 1;
    return Math.max(0, Math.min(1, value));
  }

  function normalizeGhostMode(mode) {
    if (mode === "thinking" || mode === "countdown" || mode === "static") return mode;
    return "static";
  }

  function buildGhostDashArray(visiblePercent) {
    const clamped = Math.max(0, Math.min(100, visiblePercent));
    // Whole units only, no partial/growing dash at the current progress tip
    // — a dash pops in fully-formed the instant progress reaches it. That
    // makes the reveal read as pre-existing, evenly-spaced dashes being
    // uncovered starting from position 0 toward wherever progress currently
    // is, not new geometry being generated at the leading edge.
    const revealedUnits = Math.floor((clamped / 100) * GHOST_UNIT_COUNT);
    const parts = [];
    for (let i = 0; i < revealedUnits; i++) parts.push(GHOST_ACTUAL_DASH, GHOST_ACTUAL_GAP);
    // stroke-dasharray alternates dash,gap,dash,gap,... from index 0 (dash).
    // The "hide the rest" value MUST land at an odd (gap) index, or a
    // single trailing value gets read as dash=gap=that value — which is
    // exactly why progress=0 used to render as a fully solid ring: parts
    // was empty, so a lone "1000" push was interpreted as dash=1000 (>> the
    // path's own length), covering the whole circle instead of hiding it.
    if (parts.length % 2 === 1) {
      parts.push(1000);
    } else {
      parts.push(0, 1000);
    }
    return parts.join(" ");
  }

  /** Full repeating dash/gap for thinking march (no hide-rest). */
  function buildGhostThinkingDashArray() {
    return `${GHOST_ACTUAL_DASH} ${GHOST_ACTUAL_GAP}`;
  }

  function ensureGhostStyles() {
    if (document.getElementById(GHOST_STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = GHOST_STYLE_ID;
    style.textContent =
      "@keyframes wzb-ghost-march{to{stroke-dashoffset:-100}}" +
      "[data-wzb-ghost-outline='1'][data-wzb-ghost-mode='thinking']{" +
      "animation:wzb-ghost-march 2.5s linear infinite}";
    (document.head || document.documentElement).appendChild(style);
  }

  function getGhostOutline() {
    return document.querySelector(
      `#${GHOST_LAYER_ID} [data-wzb-ghost-rank="1"] [data-wzb-ghost-outline='1']`,
    );
  }

  function applyGhostVisual(outline, mode, progress = 1) {
    if (!outline) return false;
    const m = normalizeGhostMode(mode);
    const shown = clampProgress(progress);
    const prev = outline.getAttribute("data-wzb-ghost-mode");
    // Same thinking mode: keep dashoffset animation running (do not restart).
    if (m === "thinking" && prev === "thinking") return true;
    // Same static mode: do not reset dashoffset on every progress tick.
    if (m === "static" && prev === "static") return true;
    outline.setAttribute("data-wzb-ghost-mode", m);
    outline.setAttribute("stroke-dashoffset", "0");
    if (m === "thinking") {
      outline.setAttribute("stroke-dasharray", buildGhostThinkingDashArray());
    } else if (m === "countdown") {
      outline.setAttribute("stroke-dasharray", buildGhostDashArray(shown * 100));
    } else {
      outline.setAttribute("stroke-dasharray", buildGhostDashArray(100));
    }
    return true;
  }

  function ensureGhostLayer() {
    const Board = getBoard();
    const boardEl = Board?.boardEl || document.getElementById("board");
    if (!boardEl) return null;
    ensureGhostStyles();
    let layer = document.getElementById(GHOST_LAYER_ID);
    if (!layer || layer.parentElement !== boardEl) {
      layer?.remove();
      layer = document.createElement("div");
      layer.id = GHOST_LAYER_ID;
      console.info("[quoridors-bridge] ghost layer mounted");
    }
    layer.style.cssText = "position:absolute;inset:0;pointer-events:none;z-index:80;";
    boardEl.appendChild(layer);
    return layer;
  }

  function clearGhost() {
    state.lastGhostAlgebraic = null;
    state.lastGhostKey = null;
    const outline = getGhostOutline();
    if (outline) {
      outline.removeAttribute("data-wzb-ghost-mode");
      outline.style.animation = "";
      outline.removeAttribute("stroke-dasharray");
      outline.removeAttribute("stroke-dashoffset");
    }
    const layer = document.getElementById(GHOST_LAYER_ID);
    if (layer) layer.innerHTML = "";
    const ghost = document.getElementById(GHOST_ID);
    if (ghost && (!layer || ghost.parentElement !== layer)) ghost.remove();
  }

  function createSvgEl(name) {
    return document.createElementNS("http://www.w3.org/2000/svg", name);
  }

  /** Implies countdown mode — stops thinking march animation. */
  function setGhostProgress(progress) {
    return applyGhostVisual(getGhostOutline(), "countdown", progress);
  }

  /** Switch mode without rebuilding shape when outline already exists. */
  function setGhostMode(mode) {
    const outline = getGhostOutline();
    if (!outline) return false;
    const m = normalizeGhostMode(mode);
    const progress = m === "countdown" ? 0 : 1;
    return applyGhostVisual(outline, m, progress);
  }

  function configureProgressOutline(el, length, progress, mode = "static") {
    el.setAttribute("data-wzb-ghost-outline", "1");
    el.setAttribute("data-wzb-length", String(length));
    el.setAttribute("pathLength", "100");
    el.setAttribute("fill", "none");
    el.setAttribute("stroke-opacity", "0.95");
    el.setAttribute("stroke-linecap", "butt");
    const m = normalizeGhostMode(mode);
    applyGhostVisual(el, m, progress);
    // Re-apply after attach so CSS animation / dasharray stick in the DOM.
    setTimeout(() => applyGhostVisual(el, m, progress), 0);
  }

  function renderGhosts({ ghosts, bottomPlayer, player } = {}) {
    const list = Array.isArray(ghosts) ? ghosts : [];
    const normalized = list
      .map((ghost) => ({
        move: String(ghost?.move || "").trim().toLowerCase(),
        rank: Number(ghost?.rank),
        blunder: Boolean(ghost?.blunder),
        mode: normalizeGhostMode(ghost?.mode),
        progress: clampProgress(ghost?.progress),
      }))
      .filter((ghost) => ghost.move && Number.isFinite(ghost.rank) && ghost.rank > 0)
      .sort((a, b) => b.rank - a.rank);
    const ghostKey = normalized.map((ghost) => `${ghost.move}:${ghost.rank}`).join("|");
    if (state.lastGhostKey === ghostKey && normalized.length) {
      const outline = getGhostOutline();
      const top = normalized.find((ghost) => ghost.rank === 1);
      if (outline && top) return applyGhostVisual(outline, top.mode, top.progress);
    }
    clearGhost();
    const Board = getBoard();
    const layer = ensureGhostLayer();
    if (!Board || !layer || typeof Board.cellCenter !== "function") return false;
    const colorsFor = (ghost) => ghost.blunder ? BLUNDER_GHOST_COLORS : ghostColorForPlayer(player);
    state.lastGhostPlayer = player || state.lastGhostPlayer;
    for (const ghost of normalized) {
      const { move, rank, mode, progress } = ghost;
      const colors = colorsFor(ghost);
      let action;
      try {
        action = parseAlgebraic(move);
      } catch {
        continue;
      }
      const wrapper = document.createElement("div");
      wrapper.id = rank === 1 ? GHOST_ID : "";
      wrapper.dataset.wzbGhostRank = String(rank);
      wrapper.style.cssText =
        `position:absolute;inset:0;pointer-events:none;opacity:${rank === 1 ? 1 : rank === 2 ? 0.72 : 0.55};`;
      let svg;
      let badgeX = 30;
      let badgeY = 30;
      if (action.type === "pawn") {
      const [r, c] = action.to;
      const { x, y } = Board.cellCenter(r, c);
      if (![x, y].every(Number.isFinite)) return false;
      svg = createSvgEl("svg");
      svg.setAttribute("viewBox", "0 0 60 60");
      svg.style.cssText =
        `position:absolute;left:${x}%;top:${y}%;width:60px;height:60px;` +
        "transform:translate(-50%,-50%);overflow:visible;pointer-events:none;";

      const fill = createSvgEl("circle");
      fill.setAttribute("cx", "30");
      fill.setAttribute("cy", "30");
      fill.setAttribute("r", "22");
      fill.setAttribute("fill", colors.fill);
      fill.setAttribute("stroke", "none");

      const ring = createSvgEl("circle");
      ring.setAttribute("cx", "30");
      ring.setAttribute("cy", "30");
      ring.setAttribute("r", "22");
      ring.setAttribute("stroke", colors.stroke);
      ring.setAttribute("stroke-width", "4");
      ring.setAttribute("transform", "rotate(-90 30 30)");
      configureProgressOutline(ring, 2 * Math.PI * 22, progress, mode);
      svg.appendChild(fill);
      svg.appendChild(ring);
      } else if (action.type === "wall") {
      const [r, c] = action.slot;
      const wg = Board.wallGeom(action.orientation, r, c);
      const boardEl = Board.boardEl || document.getElementById("board");
      const boardRect = boardEl?.getBoundingClientRect();
      const wPx = boardRect && wg ? (wg.w / 100) * boardRect.width : NaN;
      const hPx = boardRect && wg ? (wg.h / 100) * boardRect.height : NaN;
      if (
        !wg ||
        ![wg.x, wg.y, wg.w, wg.h, wPx, hPx].every(Number.isFinite) ||
        wPx <= 0 ||
        hPx <= 0
      ) {
        return false;
      }
      svg = createSvgEl("svg");
      svg.setAttribute("viewBox", `0 0 ${wPx} ${hPx}`);
      svg.style.cssText =
        `position:absolute;left:${wg.x}%;top:${wg.y}%;width:${wPx}px;height:${hPx}px;` +
        "transform:translate(-50%,-50%);overflow:visible;pointer-events:none;";

      const fill = createSvgEl("rect");
      fill.setAttribute("x", "0");
      fill.setAttribute("y", "0");
      fill.setAttribute("width", String(wPx));
      fill.setAttribute("height", String(hPx));
      fill.setAttribute("rx", "3");
      fill.setAttribute("fill", colors.fill);
      fill.setAttribute("stroke", "none");

      const rect = createSvgEl("rect");
      rect.setAttribute("x", "0");
      rect.setAttribute("y", "0");
      rect.setAttribute("width", String(wPx));
      rect.setAttribute("height", String(hPx));
      rect.setAttribute("rx", "3");
      rect.setAttribute("stroke", colors.stroke);
      rect.setAttribute("stroke-width", "4");
      configureProgressOutline(rect, 2 * (wPx + hPx), progress, mode);
      svg.appendChild(fill);
      svg.appendChild(rect);
        badgeX = wPx / 2;
        badgeY = hPx / 2;
      } else {
        continue;
      }
      if (normalized.length > 1) {
        const badge = createSvgEl("rect");
        badge.setAttribute("x", String(badgeX - 9));
        badge.setAttribute("y", String(badgeY - 9));
        badge.setAttribute("width", "18");
        badge.setAttribute("height", "18");
        badge.setAttribute("rx", "4");
        badge.setAttribute("fill", "rgba(8,14,22,0.88)");
        badge.setAttribute("stroke", "rgba(255,255,255,0.35)");
        badge.setAttribute("stroke-width", "1");
        const label = createSvgEl("text");
        label.setAttribute("x", String(badgeX));
        label.setAttribute("y", String(badgeY + 4));
        label.setAttribute("text-anchor", "middle");
        label.setAttribute("font-size", "12");
        label.setAttribute("font-weight", "700");
        label.setAttribute("fill", "#e8f6ff");
        const rankNum = Number(rank);
        label.textContent =
          rank != null && rank !== "" && Number.isFinite(rankNum) && rankNum > 0
            ? String(Math.round(rankNum))
            : "?";
        svg.appendChild(badge);
        svg.appendChild(label);
      }
      wrapper.appendChild(svg);
      layer.appendChild(wrapper);
    }
    if (!layer.childElementCount) return false;
    state.lastGhostKey = ghostKey;
    const top = normalized.find((ghost) => ghost.rank === 1);
    state.lastGhostAlgebraic = top?.move || null;
    const boardEl = Board.boardEl || document.getElementById("board");
    if (boardEl) boardEl.appendChild(layer);
    console.info("[quoridors-bridge] ghost rendered", {
      ghosts: normalized,
    });
    return true;
  }

  function renderGhost(moveText, bottomPlayer, player, progress = 1, mode = "static") {
    return renderGhosts({
      ghosts: [{ move: moveText, rank: 1, progress, mode }],
      bottomPlayer,
      player,
    });
  }

  function cmdResult(detail, payload) {
    window.dispatchEvent(
      new CustomEvent("quoridors-bridge-cmd-result", {
        detail: { ...payload, requestId: detail.requestId },
      }),
    );
  }

  window.addEventListener("quoridors-bridge-cmd", (ev) => {
    const detail = ev.detail || {};
    try {
      if (detail.action === "playMove") {
        playMove(bridgeMoveToAction(detail.move))
          .then((result) => cmdResult(detail, { ok: true, result }))
          .catch((err) => cmdResult(detail, { ok: false, error: String(err?.message || err) }));
      } else if (detail.action === "playAlgebraic") {
        playMove(parseAlgebraic(detail.text))
          .then((result) => cmdResult(detail, { ok: true, result }))
          .catch((err) => cmdResult(detail, { ok: false, error: String(err?.message || err) }));
      } else if (detail.action === "getState") {
        cmdResult(detail, { ok: true, state: window.__QUORIDORS_BRIDGE__.getState() });
      } else if (detail.action === "refreshState") {
        refreshStateFromApi()
          .then((result) => cmdResult(detail, { ok: true, state: result }))
          .catch((err) => cmdResult(detail, { ok: false, error: String(err?.message || err) }));
      } else if (detail.action === "renderGhost") {
        cmdResult(detail, {
          ok: true,
          rendered: renderGhost(
            detail.move,
            detail.bottomPlayer,
            detail.player,
            detail.progress,
            detail.mode,
          ),
        });
      } else if (detail.action === "renderGhosts") {
        cmdResult(detail, {
          ok: true,
          rendered: renderGhosts({
            ghosts: detail.ghosts,
            bottomPlayer: detail.bottomPlayer,
            player: detail.player,
          }),
        });
      } else if (detail.action === "setGhostProgress") {
        cmdResult(detail, { ok: true, updated: setGhostProgress(detail.progress) });
      } else if (detail.action === "setGhostMode") {
        cmdResult(detail, { ok: true, updated: setGhostMode(detail.mode) });
      } else if (detail.action === "clearGhost") {
        clearGhost();
        cmdResult(detail, { ok: true });
      } else if (detail.action === "validateMove") {
        cmdResult(detail, isLegalAlgebraic(detail.move ?? detail.text));
      }
    } catch (err) {
      cmdResult(detail, { ok: false, error: String(err?.message || err) });
    }
  });

  function tryHookAPI() {
    hookFetch();
    hookStats();
    hookAPI();
    hookInteraction();
  }

  tryHookAPI();
  const apiTimer = setInterval(() => {
    tryHookAPI();
    emitLocalDetail();
  }, 350);
  window.addEventListener("load", tryHookAPI);

  window.__QUORIDORS_BRIDGE__ = {
    getState: () => ({
      connected: state.connected,
      gameId: state.gameId,
      mode: state.mode,
      mySeat: state.mySeat,
      aiPlayer: state.aiPlayer,
      gameState: state.gameState,
      lastReject: state.lastReject,
      bridgeDetail: buildBridgeDetail(),
      blockedBotHistory: state.blockedBotHistory,
      log: state.log.slice(-20),
    }),
    playMove,
    playAlgebraic: (text) => playMove(parseAlgebraic(text)),
    isLegalAlgebraic,
    refreshState: refreshStateFromApi,
    actionToAlgebraic,
    parseAlgebraic,
    buildBridgeDetail,
    clearInterval: () => clearInterval(apiTimer),
  };

  console.info("[quoridors-bridge] page hook installed");
})();
