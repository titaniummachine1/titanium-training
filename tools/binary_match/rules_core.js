
/* PATHFIX CANDIDATE — gen11_ghi + RaceProof/ThreatPrice/WallSense (PLAN_RUNG200
   T2+T3). Built by make_pathfix_engine.js; flags raceProof/threatPrice/wallSense
   on Search (default true) each no-op cleanly when false. DO NOT EDIT BY HAND. */
/* Quoridor engine v3: blocked-edge board, full rules (jumps, wall legality),
   iterative-deepening alpha-beta, typed-array TT, killers/history/countermoves,
   null move, graduated LMR, frontier LMP, reverse futility, lazy wall legality,
   repetition detection, wall-stamp dist caching, aspiration windows,
   easy-move early stop, BFS path eval (Texel-tuned on self-play data), race terms.
   Pure JS, no DOM, no per-node allocation. Runs in browser, Worker, or Node. */
"use strict";

var DELTA = [-9, 9, -1, 1], DIRBIT = [1, 2, 4, 8];
var MATE = 100000, MAX_PLY = 64;

// ---------- zobrist ----------
var zSeed = 0x9e3779b9 | 0;
function zrand() { zSeed ^= zSeed << 13; zSeed ^= zSeed >>> 17; zSeed ^= zSeed << 5; return zSeed >>> 0; }
var Z_PAWN_LO = [], Z_PAWN_HI = [];
for (var zi = 0; zi < 2; zi++) {
  Z_PAWN_LO.push(new Uint32Array(81)); Z_PAWN_HI.push(new Uint32Array(81));
  for (var zj = 0; zj < 81; zj++) { Z_PAWN_LO[zi][zj] = zrand(); Z_PAWN_HI[zi][zj] = zrand(); }
}
var Z_HW_LO = new Uint32Array(64), Z_HW_HI = new Uint32Array(64);
var Z_VW_LO = new Uint32Array(64), Z_VW_HI = new Uint32Array(64);
for (var zs = 0; zs < 64; zs++) { Z_HW_LO[zs] = zrand(); Z_HW_HI[zs] = zrand(); Z_VW_LO[zs] = zrand(); Z_VW_HI[zs] = zrand(); }
var Z_TURN_LO = zrand(), Z_TURN_HI = zrand();

// ---------- game state ----------
function Quoridor() {
  this.histM = new Int16Array(1024);
  this.histFrom = new Int16Array(1024);
  this.histLw = new Int16Array(1024);
  this.hashesU = new Uint32Array(2048);
  this.reset();
}

Quoridor.prototype.reset = function () {
  this.pawn = [76, 4];           // player 0 bottom (8,4) goal row 0; player 1 top (0,4) goal row 8
  this.wl = [10, 10];
  this.turn = 0;
  this.hw = new Uint8Array(64);
  this.vw = new Uint8Array(64);
  this.blocked = new Uint8Array(81); // bits N=1 S=2 W=4 E=8 (walls only; bounds checked separately)
  this.hashLo = (Z_PAWN_LO[0][76] ^ Z_PAWN_LO[1][4]) >>> 0;
  this.hashHi = (Z_PAWN_HI[0][76] ^ Z_PAWN_HI[1][4]) >>> 0;
  this.histLen = 0;
  this.lastWallPly = 0;  // repetition can only reach back to the last wall placement
  this.wallStamp = 0;    // bumped on every wall make/unmake; dist fields depend only on walls
};

Quoridor.prototype.loadState = function (st) {
  this.reset();
  for (var i = 0; i < st.moves.length; i++) this.makeMove(st.moves[i]);
};

var BORDER = new Uint8Array(81);
for (var bc = 0; bc < 81; bc++) {
  var br = (bc / 9) | 0, bcl = bc % 9;
  BORDER[bc] = (br === 0 ? 1 : 0) | (br === 8 ? 2 : 0) | (bcl === 0 ? 4 : 0) | (bcl === 8 ? 8 : 0);
}
Quoridor.prototype.canStep = function (cell, dir) {
  return ((this.blocked[cell] | BORDER[cell]) & DIRBIT[dir]) === 0;
};

Quoridor.prototype.winner = function () {
  if (this.pawn[0] < 9) return 0;
  if (this.pawn[1] >= 72) return 1;
  return -1;
};

// ---------- wall mechanics ----------
Quoridor.prototype.setWallBits = function (type, slot, on) {
  var r = (slot / 8) | 0, c = slot % 8, a, b, cc, dd;
  if (type === 0) {
    a = r * 9 + c; b = a + 1; cc = a + 9; dd = b + 9;
    if (on) { this.blocked[a] |= 2; this.blocked[b] |= 2; this.blocked[cc] |= 1; this.blocked[dd] |= 1; }
    else { this.blocked[a] &= ~2; this.blocked[b] &= ~2; this.blocked[cc] &= ~1; this.blocked[dd] &= ~1; }
  } else {
    a = r * 9 + c; b = a + 9; cc = a + 1; dd = b + 1;
    if (on) { this.blocked[a] |= 8; this.blocked[b] |= 8; this.blocked[cc] |= 4; this.blocked[dd] |= 4; }
    else { this.blocked[a] &= ~8; this.blocked[b] &= ~8; this.blocked[cc] &= ~4; this.blocked[dd] &= ~4; }
  }
};

Quoridor.prototype.wallFits = function (type, slot) {
  var r = (slot / 8) | 0, c = slot % 8;
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

// Conservative "cannot possibly seal" precheck (over-counts anchors, so safe to skip BFS)
Quoridor.prototype.wallNeedsPathCheck = function (type, slot) {
  var r = (slot / 8) | 0, c = slot % 8, anchors = 0;
  if (type === 0) { if (c === 0) anchors++; if (c === 7) anchors++; }
  else { if (r === 0) anchors++; if (r === 7) anchors++; }
  for (var dr = -2; dr <= 2 && anchors < 2; dr++) {
    var rr = r + dr; if (rr < 0 || rr > 7) continue;
    for (var dc = -2; dc <= 2; dc++) {
      var ccc = c + dc; if (ccc < 0 || ccc > 7) continue;
      var ss = rr * 8 + ccc;
      if (this.hw[ss] || this.vw[ss]) { anchors++; if (anchors >= 2) break; }
    }
  }
  return anchors >= 2;
};

var BFS_Q = new Int16Array(81);
Quoridor.prototype.hasPath = function (player) {
  var goal = player === 0 ? 0 : 8, start = this.pawn[player];
  if (((start / 9) | 0) === goal) return true;
  var seen = this._seen || (this._seen = new Uint8Array(81));
  seen.fill(0);
  var head = 0, tail = 0;
  BFS_Q[tail++] = start; seen[start] = 1;
  var blk2 = this.blocked;
  while (head < tail) {
    var u = BFS_Q[head++], bm2 = blk2[u] | BORDER[u];
    for (var d = 0; d < 4; d++) {
      if (bm2 & DIRBIT[d]) continue;
      var v = u + DELTA[d];
      if (seen[v]) continue;
      if (((v / 9) | 0) === goal) return true;
      seen[v] = 1; BFS_Q[tail++] = v;
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

// ---------- pawn moves ----------
Quoridor.prototype.genPawnMoves = function (out, n) {
  var me = this.turn, s = this.pawn[me], o = this.pawn[1 - me];
  for (var d = 0; d < 4; d++) {
    if (!this.canStep(s, d)) continue;
    var t = s + DELTA[d];
    if (t !== o) { out[n++] = t; continue; }
    if (this.canStep(o, d)) { out[n++] = o + DELTA[d]; continue; }
    var p1 = d < 2 ? 2 : 0, p2 = d < 2 ? 3 : 1;
    if (this.canStep(o, p1)) { var w1 = o + DELTA[p1]; if (w1 !== s) out[n++] = w1; }
    if (this.canStep(o, p2)) { var w2 = o + DELTA[p2]; if (w2 !== s) out[n++] = w2; }
  }
  return n;
};

// ---------- make / unmake (allocation-free) ----------
Quoridor.prototype.makeMove = function (m) {
  var hl = this.histLen;
  this.histM[hl] = m; this.histLw[hl] = this.lastWallPly;
  if (m < 100) {
    var p = this.turn;
    this.histFrom[hl] = this.pawn[p];
    this.hashLo = (this.hashLo ^ Z_PAWN_LO[p][this.pawn[p]] ^ Z_PAWN_LO[p][m]) >>> 0;
    this.hashHi = (this.hashHi ^ Z_PAWN_HI[p][this.pawn[p]] ^ Z_PAWN_HI[p][m]) >>> 0;
    this.pawn[p] = m;
  } else if (m < 200) {
    var s0 = m - 100;
    this.hw[s0] = 1; this.setWallBits(0, s0, true); this.wl[this.turn]--; this.wallStamp++;
    this.hashLo = (this.hashLo ^ Z_HW_LO[s0]) >>> 0; this.hashHi = (this.hashHi ^ Z_HW_HI[s0]) >>> 0;
    this.lastWallPly = hl + 1;
  } else {
    var s1 = m - 200;
    this.vw[s1] = 1; this.setWallBits(1, s1, true); this.wl[this.turn]--; this.wallStamp++;
    this.hashLo = (this.hashLo ^ Z_VW_LO[s1]) >>> 0; this.hashHi = (this.hashHi ^ Z_VW_HI[s1]) >>> 0;
    this.lastWallPly = hl + 1;
  }
  this.turn ^= 1;
  this.hashLo = (this.hashLo ^ Z_TURN_LO) >>> 0; this.hashHi = (this.hashHi ^ Z_TURN_HI) >>> 0;
  this.hashesU[hl * 2] = this.hashLo; this.hashesU[hl * 2 + 1] = this.hashHi;
  this.histLen = hl + 1;
};

Quoridor.prototype.unmakeMove = function () {
  var hl = --this.histLen;
  var m = this.histM[hl];
  this.lastWallPly = this.histLw[hl];
  this.turn ^= 1;
  this.hashLo = (this.hashLo ^ Z_TURN_LO) >>> 0; this.hashHi = (this.hashHi ^ Z_TURN_HI) >>> 0;
  if (m < 100) {
    var p = this.turn, from = this.histFrom[hl];
    this.hashLo = (this.hashLo ^ Z_PAWN_LO[p][from] ^ Z_PAWN_LO[p][m]) >>> 0;
    this.hashHi = (this.hashHi ^ Z_PAWN_HI[p][from] ^ Z_PAWN_HI[p][m]) >>> 0;
    this.pawn[p] = from;
  } else if (m < 200) {
    var s0 = m - 100;
    this.hw[s0] = 0; this.setWallBits(0, s0, false); this.wl[this.turn]++; this.wallStamp--;
    this.hashLo = (this.hashLo ^ Z_HW_LO[s0]) >>> 0; this.hashHi = (this.hashHi ^ Z_HW_HI[s0]) >>> 0;
  } else {
    var s1 = m - 200;
    this.vw[s1] = 0; this.setWallBits(1, s1, false); this.wl[this.turn]++; this.wallStamp--;
    this.hashLo = (this.hashLo ^ Z_VW_LO[s1]) >>> 0; this.hashHi = (this.hashHi ^ Z_VW_HI[s1]) >>> 0;
  }
};

// ---------- distance fields ----------
Quoridor.prototype.computeDist = function (player, dist) {
  dist.fill(255);
  var goal = player === 0 ? 0 : 8, head = 0, tail = 0;
  for (var c = 0; c < 9; c++) { var cell = goal * 9 + c; dist[cell] = 0; BFS_Q[tail++] = cell; }
  var blk = this.blocked;
  while (head < tail) {
    var u = BFS_Q[head++], du = dist[u] + 1, bm = blk[u] | BORDER[u];
    for (var d = 0; d < 4; d++) {
      if (bm & DIRBIT[d]) continue;
      var v = u + DELTA[d];
      if (dist[v] > du) { dist[v] = du; BFS_Q[tail++] = v; }
    }
  }
};

Quoridor.prototype.markPath = function (player, dist, mark) {
  var cur = this.pawn[player], bit = 1 << player, guard = 0;
  mark[cur] |= bit;
  while (dist[cur] > 0 && guard++ < 100) {
    for (var d = 0; d < 4; d++) {
      if (!this.canStep(cur, d)) continue;
      var v = cur + DELTA[d];
      if (dist[v] === dist[cur] - 1) { cur = v; mark[cur] |= bit; break; }
    }
  }
};

Quoridor.prototype.legalMoves = function () {
  var out = new Int16Array(160), n = 0;
  n = this.genPawnMoves(out, n);
  if (this.wl[this.turn] > 0) {
    for (var slot = 0; slot < 64; slot++) {
      if (this.wallLegal(0, slot)) out[n++] = 100 + slot;
      if (this.wallLegal(1, slot)) out[n++] = 200 + slot;
    }
  }
  return Array.prototype.slice.call(out.subarray(0, n));
};



module.exports = Quoridor;
