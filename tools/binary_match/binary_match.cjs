#!/usr/bin/env node
"use strict";
// Cross-BINARY engine match driver: plays two different titanium builds
// against each other over the warm `session` stdio protocol, so search
// changes can be A/B-gated as two clean binaries (main vs branch) instead of
// polluting the engine with harness switch statements.
//
// Rules/legality/openings come from the in-repo JS rules core
// (rules_core.js — the same ACE Quoridor object engine/src/titanium/game.rs
// is a 1:1 port of), so the driver never trusts either engine on legality.
//
// Methodology mirrors the in-process `titanium match` harness: N games,
// mirrored opening pairs (same random opening, seats swapped), fixed
// time/move, warm per-game state (TT persists across a game; `position`
// rebuild between games), max-ply draw cap, Wilson 95% lower bound on the
// A-side score.
//
// Usage:
//   node binary_match.cjs --a bin/titanium_sfhist.exe --b bin/titanium_base.exe \
//        [--engine titanium-v16] [--engine-a titanium-v16-sfhist] \
//        [--engine-b titanium-v16] [--games 50] [--time 2] [--open 4] \
//        [--maxply 200] [--seed 1] [--threads 1]
//
// --engine sets the default protocol engine for both binaries. --engine-a and
// --engine-b override it independently, allowing a new engine flag to be
// tested against a baseline binary that does not recognize that flag.

const { spawn } = require("node:child_process");
const path = require("node:path");
const readline = require("node:readline");
const Quoridor = require("./rules_core.js");

// ---------- args ----------
function parseArgs() {
  const a = process.argv.slice(2);
  const opts = {
    a: null, b: null, engine: "titanium-v16", games: 50, time: 0,
    engineA: null, engineB: null, clock: 0, open: 4, maxply: 200,
    seed: 1, threads: 1,
  };
  for (let i = 0; i < a.length; i++) {
    const k = a[i];
    const v = a[i + 1];
    switch (k) {
      case "--a": opts.a = v; i++; break;
      case "--b": opts.b = v; i++; break;
      case "--engine": opts.engine = v; i++; break;
      case "--engine-a": opts.engineA = v; i++; break;
      case "--engine-b": opts.engineB = v; i++; break;
      case "--games": opts.games = +v; i++; break;
      case "--time": opts.time = +v; i++; break;      // fixed sec/move
      case "--clock": opts.clock = +v; i++; break;    // sudden-death sec/side/game
      case "--open": opts.open = +v; i++; break;
      case "--maxply": opts.maxply = +v; i++; break;
      case "--seed": opts.seed = +v >>> 0; i++; break;
      case "--threads": opts.threads = +v; i++; break;
      default:
        console.error(`unknown arg ${k}`);
        process.exit(2);
    }
  }
  if (!opts.time && !opts.clock) opts.clock = 60;
  if (opts.time && opts.clock) {
    console.error("--time and --clock are mutually exclusive");
    process.exit(2);
  }
  if (!opts.a || !opts.b) {
    console.error("required: --a <exe> --b <exe>");
    process.exit(2);
  }
  opts.engineA ??= opts.engine;
  opts.engineB ??= opts.engine;
  return opts;
}

// ---------- deterministic RNG (xorshift32) ----------
function makeRng(seed) {
  let s = seed >>> 0 || 1;
  return function () {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
}

// ---------- move id <-> algebraic (mirrors engine/src/titanium/mod.rs) ----------
function moveIdToAlgebraic(m) {
  if (m < 100) {
    const r = (m / 9) | 0, c = m % 9;
    return String.fromCharCode(97 + c) + String(9 - r);
  }
  const base = m < 200 ? 100 : 200, suffix = m < 200 ? "h" : "v";
  const slot = m - base, r = (slot / 8) | 0, c = slot % 8;
  return String.fromCharCode(97 + c) + String(8 - r) + suffix;
}
function algebraicToMoveId(s) {
  const c = s.charCodeAt(0) - 97;
  if (s.length <= 2) {
    const row = +s.slice(1) - 1;
    return (8 - row) * 9 + c;
  }
  const row = +s.slice(1, -1) - 1;
  const slot = (7 - row) * 8 + c;
  return (s.endsWith("h") ? 100 : 200) + slot;
}

// ---------- engine session wrapper ----------
class EngineSession {
  constructor(exe, engineFlag, threads, tag) {
    this.tag = tag;
    this.proc = spawn(exe, ["session", "--engine", engineFlag, "--threads", String(threads)], {
      // stderr ignored: the session streams verbose search progress there,
      // which would drown the per-game log over a 50-game match.
      stdio: ["pipe", "pipe", "ignore"],
    });
    this.rl = readline.createInterface({ input: this.proc.stdout });
    this.queue = [];
    this.waiters = [];
    this.rl.on("line", (line) => {
      const w = this.waiters.shift();
      if (w) w(line);
      else this.queue.push(line);
    });
    this.proc.on("exit", (code) => {
      if (!this.quitting) {
        console.error(`[${tag}] engine exited unexpectedly (code ${code})`);
        process.exit(1);
      }
    });
  }
  readLine() {
    if (this.queue.length) return Promise.resolve(this.queue.shift());
    return new Promise((res) => this.waiters.push(res));
  }
  send(cmd) {
    this.proc.stdin.write(cmd + "\n");
  }
  /// position + wait for "ready"
  async setPosition(algMoves) {
    this.send("position" + (algMoves.length ? " " + algMoves.join(" ") : ""));
    for (;;) {
      const line = await this.readLine();
      if (line.startsWith("ready")) return;
      if (line.startsWith("error")) throw new Error(`[${this.tag}] position: ${line}`);
    }
  }
  /// go + wait for "bestmove"
  async go(timeSec) {
    this.send(`go ${timeSec}`);
    for (;;) {
      const line = await this.readLine();
      if (line.startsWith("bestmove")) {
        const mv = line.split(/\s+/)[1];
        if (mv === "(none)") throw new Error(`[${this.tag}] bestmove (none)`);
        return mv;
      }
      if (line.startsWith("error")) throw new Error(`[${this.tag}] go: ${line}`);
      // "info json ..." lines fall through
    }
  }
  quit() {
    this.quitting = true;
    try { this.send("quit"); } catch (e) { /* already dead */ }
  }
}

// ---------- Wilson 95% lower bound on a proportion ----------
function wilsonLower(successes, n, z = 1.96) {
  if (n === 0) return 0;
  const p = successes / n;
  const z2 = z * z;
  const denom = 1 + z2 / n;
  const center = p + z2 / (2 * n);
  const margin = z * Math.sqrt((p * (1 - p) + z2 / (4 * n)) / n);
  return (center - margin) / denom;
}

// ---------- main ----------
(async function main() {
  const opts = parseArgs();
  const rng = makeRng(opts.seed);
  const engA = new EngineSession(path.resolve(opts.a), opts.engineA, opts.threads, "A");
  const engB = new EngineSession(path.resolve(opts.b), opts.engineB, opts.threads, "B");
  console.log(`A = ${opts.a}`);
  console.log(`B = ${opts.b}`);
  const tcDesc = opts.clock ? `clock=${opts.clock}s/side/game` : `time=${opts.time}s/move`;
  console.log(`engine-a=${opts.engineA} engine-b=${opts.engineB} games=${opts.games} ${tcDesc} open=${opts.open} maxply=${opts.maxply} seed=${opts.seed} threads=${opts.threads}`);

  let aWins = 0, bWins = 0, draws = 0;
  let opening = [];

  for (let game = 0; game < opts.games; game++) {
    // mirrored pairs: even game draws a fresh opening, odd game replays it with seats swapped
    if (game % 2 === 0) {
      opening = [];
      const g0 = new Quoridor();
      for (let i = 0; i < opts.open; i++) {
        const legal = g0.legalMoves();
        if (!legal.length) break;
        const m = legal[(rng() * legal.length) | 0];
        g0.makeMove(m);
        opening.push(m);
      }
    }
    const aIsP0 = game % 2 === 0;

    const g = new Quoridor();
    const algMoves = [];
    for (const m of opening) { g.makeMove(m); algMoves.push(moveIdToAlgebraic(m)); }

    // sudden-death clocks (driver-side allocation: remaining/20, floored so a
    // flagged side plays on at minimal depth instead of forfeiting — engines
    // slightly overshoot budgets, so hard flag-falls would punish the engine
    // for driver rounding, not for chess^H^Hquoridor strength)
    const clockMs = { A: opts.clock * 1000, B: opts.clock * 1000 };

    let result; // 'A' | 'B' | 'draw'
    for (;;) {
      if (g.winner() >= 0) {
        const p0Won = g.winner() === 0;
        result = p0Won === aIsP0 ? "A" : "B";
        break;
      }
      if (g.histLen >= opts.maxply) { result = "draw"; break; }
      const mover = (g.turn === 0) === aIsP0 ? engA : engB;
      let moveSec;
      if (opts.clock) {
        moveSec = Math.max(0.15, clockMs[mover.tag] / 1000 / 20);
      } else {
        moveSec = opts.time;
      }
      await mover.setPosition(algMoves);
      const t0 = Date.now();
      const alg = await mover.go(moveSec);
      if (opts.clock) {
        clockMs[mover.tag] = Math.max(0, clockMs[mover.tag] - (Date.now() - t0));
      }
      const mid = algebraicToMoveId(alg);
      if (g.legalMoves().indexOf(mid) < 0) {
        console.error(`game ${game}: ${mover.tag} played ILLEGAL move ${alg} at ply ${g.histLen} — forfeits`);
        result = mover.tag === "A" ? "B" : "A";
        break;
      }
      g.makeMove(mid);
      algMoves.push(alg);
    }

    if (result === "A") aWins++;
    else if (result === "B") bWins++;
    else draws++;
    const n = game + 1;
    const score = (aWins + 0.5 * draws) / n;
    const clockNote = opts.clock
      ? `  clocks A=${(clockMs.A / 1000).toFixed(1)}s B=${(clockMs.B / 1000).toFixed(1)}s`
      : "";
    console.log(
      `game ${String(game).padStart(3)}  ${result === "draw" ? "draw" : result + " wins"}  ` +
      `(A as ${aIsP0 ? "p0" : "p1"}, ${g.histLen} plies)  ` +
      `running A: ${aWins}W ${draws}D ${bWins}L  score=${score.toFixed(3)}${clockNote}`
    );
  }

  const n = opts.games;
  const score = (aWins + 0.5 * draws) / n;
  const wl = wilsonLower(aWins + 0.5 * draws, n);
  console.log("");
  console.log(`FINAL  A: ${aWins}W ${draws}D ${bWins}L / ${n}  score=${score.toFixed(3)}`);
  console.log(`Wilson 95% lower bound on A score: ${wl.toFixed(3)}  (${wl > 0.5 ? "PASSES" : "does NOT pass"} the >0.5 ship gate)`);
  engA.quit();
  engB.quit();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
