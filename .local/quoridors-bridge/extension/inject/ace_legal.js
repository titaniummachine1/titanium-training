/**
 * Slim ACE Quoridor rules oracle (legal moves only — no search/TT/DOM).
 * Source: ACE HTML enginecode reset→legalMoves. Algebraic codec = site V3/ACE.
 */
(function aceLegalOracle(global) {
  "use strict";

  var DELTA = [-9, 9, -1, 1],
    DIRBIT = [1, 2, 4, 8];

  var zSeed = 0x9e3779b9 | 0;
  function zrand() {
    zSeed ^= zSeed << 13;
    zSeed ^= zSeed >>> 17;
    zSeed ^= zSeed << 5;
    return zSeed >>> 0;
  }
  var Z_PAWN_LO = [],
    Z_PAWN_HI = [];
  for (var zi = 0; zi < 2; zi++) {
    Z_PAWN_LO.push(new Uint32Array(81));
    Z_PAWN_HI.push(new Uint32Array(81));
    for (var zj = 0; zj < 81; zj++) {
      Z_PAWN_LO[zi][zj] = zrand();
      Z_PAWN_HI[zi][zj] = zrand();
    }
  }
  var Z_HW_LO = new Uint32Array(64),
    Z_HW_HI = new Uint32Array(64);
  var Z_VW_LO = new Uint32Array(64),
    Z_VW_HI = new Uint32Array(64);
  for (var zs = 0; zs < 64; zs++) {
    Z_HW_LO[zs] = zrand();
    Z_HW_HI[zs] = zrand();
    Z_VW_LO[zs] = zrand();
    Z_VW_HI[zs] = zrand();
  }
  var Z_TURN_LO = zrand(),
    Z_TURN_HI = zrand();

  function Quoridor() {
    this.histM = new Int16Array(1024);
    this.histFrom = new Int16Array(1024);
    this.histLw = new Int16Array(1024);
    this.hashesU = new Uint32Array(2048);
    this.reset();
  }

  Quoridor.prototype.reset = function () {
    this.pawn = [76, 4];
    this.wl = [10, 10];
    this.turn = 0;
    this.hw = new Uint8Array(64);
    this.vw = new Uint8Array(64);
    this.blocked = new Uint8Array(81);
    this.hashLo = (Z_PAWN_LO[0][76] ^ Z_PAWN_LO[1][4]) >>> 0;
    this.hashHi = (Z_PAWN_HI[0][76] ^ Z_PAWN_HI[1][4]) >>> 0;
    this.histLen = 0;
    this.lastWallPly = 0;
    this.wallStamp = 0;
  };

  var BORDER = new Uint8Array(81);
  for (var bc = 0; bc < 81; bc++) {
    var br = (bc / 9) | 0,
      bcl = bc % 9;
    BORDER[bc] =
      (br === 0 ? 1 : 0) | (br === 8 ? 2 : 0) | (bcl === 0 ? 4 : 0) | (bcl === 8 ? 8 : 0);
  }

  Quoridor.prototype.canStep = function (cell, dir) {
    return ((this.blocked[cell] | BORDER[cell]) & DIRBIT[dir]) === 0;
  };

  Quoridor.prototype.setWallBits = function (type, slot, on) {
    var r = (slot / 8) | 0,
      c = slot % 8,
      a,
      b,
      cc,
      dd;
    if (type === 0) {
      a = r * 9 + c;
      b = a + 1;
      cc = a + 9;
      dd = b + 9;
      if (on) {
        this.blocked[a] |= 2;
        this.blocked[b] |= 2;
        this.blocked[cc] |= 1;
        this.blocked[dd] |= 1;
      } else {
        this.blocked[a] &= ~2;
        this.blocked[b] &= ~2;
        this.blocked[cc] &= ~1;
        this.blocked[dd] &= ~1;
      }
    } else {
      a = r * 9 + c;
      b = a + 9;
      cc = a + 1;
      dd = b + 1;
      if (on) {
        this.blocked[a] |= 8;
        this.blocked[b] |= 8;
        this.blocked[cc] |= 4;
        this.blocked[dd] |= 4;
      } else {
        this.blocked[a] &= ~8;
        this.blocked[b] &= ~8;
        this.blocked[cc] &= ~4;
        this.blocked[dd] &= ~4;
      }
    }
  };

  Quoridor.prototype.wallFits = function (type, slot) {
    var r = (slot / 8) | 0,
      c = slot % 8;
    if (this.hw[slot] || this.vw[slot]) return false;
    if (type === 0) {
      if (c > 0 && this.hw[slot - 1]) return false;
      if (c < 7 && this.hw[slot + 1]) return false;
    } else {
      if (r > 0 && this.vw[slot - 8]) return false;
      if (r < 7 && this.vw[slot + 8]) return false;
    }
    return true;
  };

  Quoridor.prototype.wallNeedsPathCheck = function (type, slot) {
    var r = (slot / 8) | 0,
      c = slot % 8,
      anchors = 0;
    if (type === 0) {
      if (c === 0) anchors++;
      if (c === 7) anchors++;
    } else {
      if (r === 0) anchors++;
      if (r === 7) anchors++;
    }
    for (var dr = -2; dr <= 2 && anchors < 2; dr++) {
      var rr = r + dr;
      if (rr < 0 || rr > 7) continue;
      for (var dc = -2; dc <= 2; dc++) {
        var ccc = c + dc;
        if (ccc < 0 || ccc > 7) continue;
        var ss = rr * 8 + ccc;
        if (this.hw[ss] || this.vw[ss]) {
          anchors++;
          if (anchors >= 2) break;
        }
      }
    }
    return anchors >= 2;
  };

  var BFS_Q = new Int16Array(81);
  Quoridor.prototype.hasPath = function (player) {
    var goal = player === 0 ? 0 : 8,
      start = this.pawn[player];
    if (((start / 9) | 0) === goal) return true;
    var seen = this._seen || (this._seen = new Uint8Array(81));
    seen.fill(0);
    var head = 0,
      tail = 0;
    BFS_Q[tail++] = start;
    seen[start] = 1;
    var blk2 = this.blocked;
    while (head < tail) {
      var u = BFS_Q[head++],
        bm2 = blk2[u] | BORDER[u];
      for (var d = 0; d < 4; d++) {
        if (bm2 & DIRBIT[d]) continue;
        var v = u + DELTA[d];
        if (seen[v]) continue;
        if (((v / 9) | 0) === goal) return true;
        seen[v] = 1;
        BFS_Q[tail++] = v;
      }
    }
    return false;
  };

  Quoridor.prototype.wallLegal = function (type, slot) {
    if (this.wl[this.turn] <= 0) return false;
    if (!this.wallFits(type, slot)) return false;
    if (!this.wallNeedsPathCheck(type, slot)) return true;
    this.setWallBits(type, slot, true);
    var ok = this.hasPath(0) && this.hasPath(1);
    this.setWallBits(type, slot, false);
    return ok;
  };

  Quoridor.prototype.genPawnMoves = function (out, n) {
    var me = this.turn,
      s = this.pawn[me],
      o = this.pawn[1 - me];
    for (var d = 0; d < 4; d++) {
      if (!this.canStep(s, d)) continue;
      var t = s + DELTA[d];
      if (t !== o) {
        out[n++] = t;
        continue;
      }
      if (this.canStep(o, d)) {
        out[n++] = o + DELTA[d];
        continue;
      }
      var p1 = d < 2 ? 2 : 0,
        p2 = d < 2 ? 3 : 1;
      if (this.canStep(o, p1)) {
        var w1 = o + DELTA[p1];
        if (w1 !== s) out[n++] = w1;
      }
      if (this.canStep(o, p2)) {
        var w2 = o + DELTA[p2];
        if (w2 !== s) out[n++] = w2;
      }
    }
    return n;
  };

  Quoridor.prototype.makeMove = function (m) {
    var hl = this.histLen;
    this.histM[hl] = m;
    this.histLw[hl] = this.lastWallPly;
    if (m < 100) {
      var p = this.turn;
      this.histFrom[hl] = this.pawn[p];
      this.hashLo = (this.hashLo ^ Z_PAWN_LO[p][this.pawn[p]] ^ Z_PAWN_LO[p][m]) >>> 0;
      this.hashHi = (this.hashHi ^ Z_PAWN_HI[p][this.pawn[p]] ^ Z_PAWN_HI[p][m]) >>> 0;
      this.pawn[p] = m;
    } else if (m < 200) {
      var s0 = m - 100;
      this.hw[s0] = 1;
      this.setWallBits(0, s0, true);
      this.wl[this.turn]--;
      this.wallStamp++;
      this.hashLo = (this.hashLo ^ Z_HW_LO[s0]) >>> 0;
      this.hashHi = (this.hashHi ^ Z_HW_HI[s0]) >>> 0;
      this.lastWallPly = hl + 1;
    } else {
      var s1 = m - 200;
      this.vw[s1] = 1;
      this.setWallBits(1, s1, true);
      this.wl[this.turn]--;
      this.wallStamp++;
      this.hashLo = (this.hashLo ^ Z_VW_LO[s1]) >>> 0;
      this.hashHi = (this.hashHi ^ Z_VW_HI[s1]) >>> 0;
      this.lastWallPly = hl + 1;
    }
    this.turn ^= 1;
    this.hashLo = (this.hashLo ^ Z_TURN_LO) >>> 0;
    this.hashHi = (this.hashHi ^ Z_TURN_HI) >>> 0;
    this.hashesU[hl * 2] = this.hashLo;
    this.hashesU[hl * 2 + 1] = this.hashHi;
    this.histLen = hl + 1;
  };

  Quoridor.prototype.unmakeMove = function () {
    var hl = --this.histLen;
    var m = this.histM[hl];
    this.lastWallPly = this.histLw[hl];
    this.turn ^= 1;
    this.hashLo = (this.hashLo ^ Z_TURN_LO) >>> 0;
    this.hashHi = (this.hashHi ^ Z_TURN_HI) >>> 0;
    if (m < 100) {
      var p = this.turn,
        from = this.histFrom[hl];
      this.hashLo = (this.hashLo ^ Z_PAWN_LO[p][from] ^ Z_PAWN_LO[p][m]) >>> 0;
      this.hashHi = (this.hashHi ^ Z_PAWN_HI[p][from] ^ Z_PAWN_HI[p][m]) >>> 0;
      this.pawn[p] = from;
    } else if (m < 200) {
      var s0 = m - 100;
      this.hw[s0] = 0;
      this.setWallBits(0, s0, false);
      this.wl[this.turn]++;
      this.wallStamp--;
      this.hashLo = (this.hashLo ^ Z_HW_LO[s0]) >>> 0;
      this.hashHi = (this.hashHi ^ Z_HW_HI[s0]) >>> 0;
    } else {
      var s1 = m - 200;
      this.vw[s1] = 0;
      this.setWallBits(1, s1, false);
      this.wl[this.turn]++;
      this.wallStamp--;
      this.hashLo = (this.hashLo ^ Z_VW_LO[s1]) >>> 0;
      this.hashHi = (this.hashHi ^ Z_VW_HI[s1]) >>> 0;
    }
  };

  Quoridor.prototype.legalMoves = function () {
    var out = new Int16Array(160),
      n = 0;
    n = this.genPawnMoves(out, n);
    if (this.wl[this.turn] > 0) {
      for (var slot = 0; slot < 64; slot++) {
        if (this.wallLegal(0, slot)) out[n++] = 100 + slot;
        if (this.wallLegal(1, slot)) out[n++] = 200 + slot;
      }
    }
    return Array.prototype.slice.call(out.subarray(0, n));
  };

  /** Site / ACE V3 algebraic codec. */
  function algebraicToAce(alg) {
    var s = String(alg || "").toLowerCase();
    var col = s.charCodeAt(0) - 97,
      row = +s[1];
    if (s.length === 2) return (9 - row) * 9 + col;
    var slot = (8 - row) * 8 + col;
    return s[2] === "h" ? 100 + slot : 200 + slot;
  }

  function aceToAlgebraic(m) {
    var move = Number(m) | 0;
    if (move < 100) {
      var r = (move / 9) | 0,
        c = move % 9;
      return String.fromCharCode(97 + c) + (9 - r);
    }
    var base = move < 200 ? 100 : 200;
    var slot = move - base;
    var wr = (slot / 8) | 0,
      wc = slot % 8;
    var suffix = move < 200 ? "h" : "v";
    return String.fromCharCode(97 + wc) + (8 - wr) + suffix;
  }

  function create() {
    return new Quoridor();
  }

  function loadAlgebraic(game, moves) {
    game.reset();
    var list = Array.isArray(moves) ? moves : [];
    for (var i = 0; i < list.length; i++) {
      var alg = list[i];
      if (!alg) continue;
      game.makeMove(algebraicToAce(alg));
    }
    return game;
  }

  function pawnsMatchSite(game, gs) {
    if (!gs || !Array.isArray(gs.pawns) || gs.pawns.length < 2) return false;
    for (var i = 0; i < 2; i++) {
      var p = gs.pawns[i];
      if (!Array.isArray(p) || p.length < 2) return false;
      if (game.pawn[i] !== p[0] * 9 + p[1]) return false;
    }
    return true;
  }

  function applyWallList(game, type, list) {
    if (!Array.isArray(list)) return;
    for (var i = 0; i < list.length; i++) {
      var slotRC = list[i];
      if (!Array.isArray(slotRC) || slotRC.length < 2) continue;
      var slot = slotRC[0] * 8 + slotRC[1];
      if (slot < 0 || slot > 63) continue;
      if (type === 0) game.hw[slot] = 1;
      else game.vw[slot] = 1;
      game.setWallBits(type, slot, true);
    }
  }

  function recomputeHash(game) {
    var lo = 0,
      hi = 0;
    lo = (lo ^ Z_PAWN_LO[0][game.pawn[0]] ^ Z_PAWN_LO[1][game.pawn[1]]) >>> 0;
    hi = (hi ^ Z_PAWN_HI[0][game.pawn[0]] ^ Z_PAWN_HI[1][game.pawn[1]]) >>> 0;
    for (var s = 0; s < 64; s++) {
      if (game.hw[s]) {
        lo = (lo ^ Z_HW_LO[s]) >>> 0;
        hi = (hi ^ Z_HW_HI[s]) >>> 0;
      }
      if (game.vw[s]) {
        lo = (lo ^ Z_VW_LO[s]) >>> 0;
        hi = (hi ^ Z_VW_HI[s]) >>> 0;
      }
    }
    if (game.turn) {
      lo = (lo ^ Z_TURN_LO) >>> 0;
      hi = (hi ^ Z_TURN_HI) >>> 0;
    }
    game.hashLo = lo;
    game.hashHi = hi;
  }

  /**
   * Preferred: load pawns/walls/turn/wl from scraped site state.
   * Falls back to replaying algebraicMoves / history algebraics when walls missing.
   */
  function loadFromSiteState(game, gs, algebraicMoves) {
    if (!game) game = create();
    if (!gs || !Array.isArray(gs.pawns) || gs.pawns.length < 2) {
      if (Array.isArray(algebraicMoves)) return loadAlgebraic(game, algebraicMoves);
      game.reset();
      return game;
    }

    // Empty H/V arrays are still Array.isArray-truthy — require real walls.
    var hasWalls =
      gs.walls &&
      ((Array.isArray(gs.walls.H) && gs.walls.H.length > 0) ||
        (Array.isArray(gs.walls.V) && gs.walls.V.length > 0));

    if (!hasWalls) {
      // Prefer history replay when walls are absent/empty (verified list path).
      var replay = Array.isArray(algebraicMoves) ? algebraicMoves : [];
      loadAlgebraic(game, replay);
      if (gs.turn != null && game.turn !== gs.turn) {
        /* history may be incomplete — still return replayed state */
      }
      return game;
    }

    game.reset();
    game.pawn[0] = gs.pawns[0][0] * 9 + gs.pawns[0][1];
    game.pawn[1] = gs.pawns[1][0] * 9 + gs.pawns[1][1];
    if (Array.isArray(gs.walls_left) && gs.walls_left.length >= 2) {
      game.wl[0] = Number(gs.walls_left[0]) | 0;
      game.wl[1] = Number(gs.walls_left[1]) | 0;
    }
    applyWallList(game, 0, gs.walls.H);
    applyWallList(game, 1, gs.walls.V);
    game.turn = gs.turn != null ? Number(gs.turn) | 0 : 0;
    game.histLen = 0;
    game.lastWallPly = 0;
    recomputeHash(game);
    return game;
  }

  function legalAlgebraic(game) {
    if (!game) return [];
    return game.legalMoves().map(aceToAlgebraic);
  }

  function isLegal(game, alg) {
    if (!game || !alg) return false;
    var want = String(alg).toLowerCase();
    var moves = legalAlgebraic(game);
    for (var i = 0; i < moves.length; i++) {
      if (moves[i] === want) return true;
    }
    return false;
  }

  var AceLegal = {
    create: create,
    loadAlgebraic: loadAlgebraic,
    loadFromSiteState: loadFromSiteState,
    legalAlgebraic: legalAlgebraic,
    isLegal: isLegal,
    algebraicToAce: algebraicToAce,
    aceToAlgebraic: aceToAlgebraic,
    pawnsMatchSite: pawnsMatchSite,
    Quoridor: Quoridor,
  };

  global.AceLegal = AceLegal;
  if (typeof globalThis !== "undefined") globalThis.AceLegal = AceLegal;
})(typeof window !== "undefined" ? window : globalThis);
