#!/usr/bin/env node
// Benchmarks Titanium against quoridors.com's built-in bot (vs-computer mode,
// difficulty selectable up to Expert) over the site's public REST API
// (POST /api/game/new, /move, /ai-move — no auth, no socket, no ranked play).
//
// Titanium always plays the "player 0" seat (the one quoridors.com assigns to
// the human by default); the site's own AI is "player 1". quoridors.com's
// player-0 pawn starts at row 8 (goal row 0) while Titanium's engine binary
// always starts its first mover (the side that moves first from history) at
// row 0 (goal row 8) — so board rows are flipped (row_t = 8 - row_q for pawn
// squares, row_t = 7 - row_q for wall slots) when translating to/from the
// engine's algebraic notation. Columns and wall orientation (h/v) are
// unaffected: a pure row-flip is a valid symmetry of Quoridor.

import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const monorepoRoot = path.resolve(here, '..', '..');
const binName = process.platform === 'win32' ? 'titanium.exe' : 'titanium';
const defaultBin = path.join(monorepoRoot, 'engine', 'target', 'release', binName);

const BASE_URL = 'https://quoridors.com';

function colLetter(c) {
  return String.fromCharCode(97 + c);
}

function pawnToAlgebraic([rowQ, colQ]) {
  const rowT = 8 - rowQ;
  return `${colLetter(colQ)}${rowT + 1}`;
}

function algebraicToPawn(alg) {
  const colQ = alg.charCodeAt(0) - 97;
  const rowT = Number(alg.slice(1)) - 1;
  return [8 - rowT, colQ];
}

function wallToAlgebraic(orientation, [rowQ, colQ]) {
  const rowT = 7 - rowQ;
  return `${colLetter(colQ)}${rowT + 1}${orientation === 'H' ? 'h' : 'v'}`;
}

function algebraicToWall(alg) {
  const colQ = alg.charCodeAt(0) - 97;
  const orientation = alg.slice(-1) === 'h' ? 'H' : 'V';
  const rowT = Number(alg.slice(1, -1)) - 1;
  return { orientation, slot: [7 - rowT, colQ] };
}

function moveToAlgebraic(move) {
  if (move.type === 'pawn') return pawnToAlgebraic(move.to);
  return wallToAlgebraic(move.orientation, move.slot);
}

function actionFromAlgebraic(alg) {
  if (alg.length === 2) return { type: 'pawn', to: algebraicToPawn(alg) };
  const { orientation, slot } = algebraicToWall(alg);
  return { type: 'wall', orientation, slot };
}

function parseArgs(argv) {
  const opts = {
    difficulty: 'expert',
    games: 1,
    timeSec: 2,
    engine: 'titanium-v16',
    bin: process.env.TITANIUM_BIN || defaultBin,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--difficulty') opts.difficulty = argv[++i];
    else if (a === '--games') opts.games = Number(argv[++i]);
    else if (a === '--time') opts.timeSec = Number(argv[++i]);
    else if (a === '--engine') opts.engine = argv[++i];
    else if (a === '--bin') opts.bin = argv[++i];
    else throw new Error(`unknown arg: ${a}`);
  }
  return opts;
}

function titaniumGenmove(bin, engine, timeSec, history) {
  const args = ['genmove', ...history, '--engine', engine, '--time', String(timeSec)];
  const result = spawnSync(bin, args, { encoding: 'utf8', cwd: monorepoRoot, maxBuffer: 4 * 1024 * 1024 });
  if (result.error) throw new Error(`Titanium binary not found at ${bin}`);
  if (result.status !== 0) throw new Error(result.stderr?.trim() || `titanium exited ${result.status}`);
  const line = (result.stdout || '').trim().split(/\r?\n/).pop() || '';
  const match = /^bestmove\s+(\S+)/.exec(line);
  if (!match || match[1] === '(none)') throw new Error(`no legal move: ${line}`);
  return match[1];
}

async function api(pathSuffix, body) {
  const res = await fetch(`${BASE_URL}${pathSuffix}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `${pathSuffix} failed (${res.status})`);
  return data;
}

async function playGame(opts) {
  const state0 = await api('/api/game/new', { mode: 'ai', ai_player: 1, ai_depth: 2, difficulty: opts.difficulty });
  const gameId = state0.game_id;
  const history = []; // Titanium-space algebraic moves
  let state = state0;

  while (state.winner === null) {
    if (state.turn === 0) {
      const alg = titaniumGenmove(opts.bin, opts.engine, opts.timeSec, history);
      const action = actionFromAlgebraic(alg);
      state = await api(`/api/game/${gameId}/move`, action);
      history.push(alg);
    } else {
      state = await api(`/api/game/${gameId}/ai-move`, undefined);
      const alg = moveToAlgebraic(state.last_move);
      history.push(alg);
    }
  }

  // winner: 0 = Titanium (player 0), 1 = site bot (player 1)
  return { gameId, winner: state.winner, plies: history.length };
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  console.log(
    `Titanium (${opts.engine}, ${opts.timeSec}s/move) vs quoridors.com bot (${opts.difficulty}) — ${opts.games} game(s)`,
  );

  let wins = 0;
  let losses = 0;
  for (let g = 1; g <= opts.games; g++) {
    const result = await playGame(opts);
    if (result.winner === 0) wins++;
    else losses++;
    console.log(
      `game ${g}/${opts.games}: ${result.winner === 0 ? 'Titanium won' : 'bot won'} in ${result.plies} plies (${result.gameId})`,
    );
  }
  console.log(`\nResult: Titanium ${wins} - ${losses} bot (${opts.games} games, difficulty=${opts.difficulty})`);
}

main().catch((err) => {
  console.error(err.message ?? err);
  process.exit(1);
});
