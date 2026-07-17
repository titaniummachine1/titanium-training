#!/usr/bin/env node
/**
 * Bounded, deterministic Ka alpha-beta teacher-label adapter.
 *
 * The supplied Ace bundle is read in place and its browser scripts are
 * evaluated in a Node VM. No browser, copied bundle, or reference/ace.html
 * is used.
 */
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { performance } from 'node:perf_hooks';

const DEFAULT_ACE = 'C:\\Users\\Terminatort8000\\Downloads\\ace.html';
const PROJECT_ROOT = path.resolve(import.meta.dirname, '..', '..', '..');
const RUNTIME_DIR = path.join(PROJECT_ROOT, 'reference', 'ka_weights_export');
const WORK_DIR = path.join(PROJECT_ROOT, 'work', 'ka_ab_teacher');
const SCHEMA = 'ace-ka-ab-teacher-v1';
const VERSION = 1;
const REQUIRED = [
  'engine-core', 'ka-encoder', 'ka-forward', 'ka-solver',
  'ka-engine', 'ka-ab', 'ka-weights',
];

function sha256(data) {
  return crypto.createHash('sha256').update(data).digest('hex');
}

function extractTags(html) {
  const found = new Map();
  const open = /<script\b[^>]*>/gi;
  let match;
  while ((match = open.exec(html)) !== null) {
    const close = /<\/script\s*>/gi;
    close.lastIndex = open.lastIndex;
    const end = close.exec(html);
    if (!end) throw new Error('Unterminated <script> in Ace bundle');
    const idMatch = /\bid\s*=\s*(["'])(.*?)\1/i.exec(match[0]);
    const id = idMatch ? idMatch[2] : null;
    if (id && REQUIRED.includes(id)) {
      if (found.has(id)) throw new Error(`Duplicate Ace script id: ${id}`);
      found.set(id, html.slice(open.lastIndex, end.index));
    }
    open.lastIndex = close.lastIndex;
  }
  const missing = REQUIRED.filter((id) => !found.has(id));
  if (missing.length) throw new Error(`Missing Ace script id(s): ${missing.join(', ')}`);
  return found;
}

function assertAceModes(html) {
  const certified = /<option\b[^>]*\bvalue\s*=\s*(["'])mcts\1[^>]*\bselected(?:\s*=\s*(["'])selected\2)?[^>]*>\s*Ace\s*\(MCTS\)/i.test(html);
  const beta = /<option\b[^>]*\bvalue\s*=\s*(["'])ab\1[^>]*>\s*Ace-AB\s*\(beta\)/i.test(html);
  if (!certified || !beta) {
    throw new Error('Ace bundle lacks certified MCTS default and beta Ace-AB option');
  }
}

function makeSandbox() {
  const quietConsole = {
    log: (...args) => process.stderr.write(`${args.join(' ')}\n`),
    info: (...args) => process.stderr.write(`${args.join(' ')}\n`),
    warn: (...args) => process.stderr.write(`${args.join(' ')}\n`),
    error: (...args) => process.stderr.write(`${args.join(' ')}\n`),
  };
  const sandbox = {
    console: quietConsole, Math, Date, Error, performance, Promise,
    Array, Object, JSON, Map, Set, WeakMap, RegExp,
    Number, String, Boolean, BigInt,
    Int8Array, Uint8Array, Uint8ClampedArray, Uint16Array, Int16Array,
    Int32Array, Uint32Array, Float32Array, Float64Array, BigInt64Array,
    BigUint64Array, DataView, ArrayBuffer, SharedArrayBuffer,
    setTimeout, clearTimeout, setInterval, clearInterval,
    atob: (value) => Buffer.from(value, 'base64').toString('binary'),
    btoa: (value) => Buffer.from(value, 'binary').toString('base64'),
    module: { exports: {} }, exports: {},
    Buffer, WebAssembly,
  };
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;
  sandbox.window = sandbox;
  return sandbox;
}

function evaluateModule(code, sandbox, name) {
  sandbox.module = { exports: {} };
  sandbox.exports = sandbox.module.exports;
  vm.runInNewContext(code, sandbox, { filename: name });
  return sandbox.module.exports;
}

function loadExtracted(filename) {
  for (const directory of [RUNTIME_DIR, WORK_DIR]) {
    const candidate = path.join(directory, filename);
    if (fs.existsSync(candidate)) return fs.readFileSync(candidate, 'utf8');
  }
  throw new Error(`missing extracted Ka asset ${filename}`);
}

function loadExtractedJson(filename) {
  return JSON.parse(loadExtracted(filename));
}

function buildRuntime(html, backend, batchChunk) {
  const scripts = extractTags(html);
  const sandbox = makeSandbox();
  vm.runInNewContext(
    `${scripts.get('engine-core')}\n;globalThis.Quoridor = Quoridor;`,
    sandbox,
    { filename: 'ace:engine-core' },
  );
  const encoder = evaluateModule(scripts.get('ka-encoder'), sandbox, 'ace:ka-encoder');
  const forward = evaluateModule(scripts.get('ka-forward'), sandbox, 'ace:ka-forward');
  const solver = evaluateModule(scripts.get('ka-solver'), sandbox, 'ace:ka-solver');
  const engineLib = evaluateModule(scripts.get('ka-engine'), sandbox, 'ace:ka-engine');
  const abLib = evaluateModule(scripts.get('ka-ab'), sandbox, 'ace:ka-ab');
  const wasmForward = evaluateModule(loadExtracted('ka-forward-wasm.js'), sandbox, 'ka-forward-wasm');
  const backendLib = evaluateModule(loadExtracted('ka-backend.js'), sandbox, 'ka-backend');
  const exportedWeights = path.join(RUNTIME_DIR, 'ka-weights.json');
  const weights = fs.existsSync(exportedWeights)
    ? JSON.parse(fs.readFileSync(exportedWeights, 'utf8'))
    : vm.runInNewContext(`(${scripts.get('ka-weights').trim()})`, sandbox, {
      filename: 'ace:ka-weights',
    });
  const KaNet = forward.KaNet || forward;
  if (typeof KaNet !== 'function' || typeof abLib?.makeEngine !== 'function'
      || typeof backendLib?.makeEvaluate !== 'function') {
    throw new Error('Ace Ka forward or Ka-AB module has an invalid export');
  }
  const wasmBin = loadExtractedJson('ka-wasm-bin.json');
  const wasmBytes = wasmBin.b64 || wasmBin.base64 || wasmBin.bytes;
  return backendLib.makeEvaluate({
    Quoridor: sandbox.Quoridor,
    KaEncoder: encoder,
    Solver: solver,
    KaEngineLib: engineLib,
    KaNet,
    KaWasm: wasmForward,
    weights,
    wasmBytes,
    backend,
    strict: backend === 'wasm',
    config: { timeMs: 0, maxEvals: 0, seed: 13 },
  }).then((built) => {
    const engine = abLib.makeEngine({
      Quoridor: sandbox.Quoridor,
      KaEncoder: encoder,
      Solver: solver,
      KaEngineLib: engineLib,
      evaluate: built.evaluate,
      config: { timeMs: 0, maxEvals: 0, batchChunk, seed: 13 },
    });
    return {
      encoder,
      engine,
      Quoridor: sandbox.Quoridor,
      backend: built.backend,
      ladder: built.ladder,
    };
  });
}

function parseArgs(argv) {
  const args = {
    ace: DEFAULT_ACE, moves: [], nodes: 8, timeMs: 0, backend: 'auto', bench: 0,
    batchChunk: 8,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--ace' || arg === '--nodes' || arg === '--time-ms'
        || arg === '--backend' || arg === '--bench' || arg === '--batch-chunk') {
      const value = argv[++i];
      if (value === undefined || value.startsWith('--')) throw new Error(`${arg} requires a value`);
      if (arg === '--ace') args.ace = path.resolve(value);
      else if (arg === '--nodes') args.nodes = Number(value);
      else if (arg === '--time-ms') args.timeMs = Number(value);
      else if (arg === '--backend') args.backend = value;
      else if (arg === '--bench') args.bench = Number(value);
      else args.batchChunk = Number(value);
    } else if (arg === '--moves') {
      let count = 0;
      while (argv[i + 1] && !argv[i + 1].startsWith('--')) {
        args.moves.push(argv[++i]);
        count += 1;
      }
      if (count === 0) throw new Error('--moves requires at least one official move');
    } else {
      throw new Error(`Unknown or misplaced argument: ${arg}`);
    }
  }
  if (!Number.isInteger(args.nodes) || args.nodes < 1 || args.nodes > 100000) {
    throw new Error('--nodes must be an integer in [1, 100000]');
  }
  if (!['auto', 'wasm', 'js'].includes(args.backend)) {
    throw new Error('--backend must be one of auto, wasm, js');
  }
  if (!Number.isInteger(args.bench) || args.bench < 0 || args.bench > 10000) {
    throw new Error('--bench must be an integer in [0, 10000]');
  }
  if (!Number.isInteger(args.batchChunk) || args.batchChunk < 1 || args.batchChunk > 32) {
    throw new Error('--batch-chunk must be an integer in [1, 32]');
  }
  if (args.timeMs !== 0) throw new Error('--time-ms is accepted only as exactly 0');
  return args;
}

function officialMovesToOur(encoder, Quoridor, moves) {
  const game = new Quoridor();
  return moves.map((move) => {
    const id = encoder.officialToOurId(move);
    if (!Number.isInteger(id) || id < 0 || id > 263) throw new Error(`Invalid official move: ${move}`);
    if (!game.legalMoves().includes(id)) throw new Error(`Illegal official move: ${move}`);
    game.makeMove(id);
    return id;
  });
}

function finite(value, label) {
  const number = Number(value);
  if (!Number.isFinite(number)) throw new Error(`AB result has non-finite ${label}`);
  return number;
}

async function main() {
  const args = parseArgs(process.argv);
  if (!fs.existsSync(args.ace)) throw new Error(`Ace bundle missing: ${args.ace}`);
  const html = fs.readFileSync(args.ace);
  const text = html.toString('utf8');
  assertAceModes(text);
  const runtime = await buildRuntime(text, args.backend, args.batchChunk);
  const ourMoves = officialMovesToOur(runtime.encoder, runtime.Quoridor, args.moves);
  runtime.engine.setPosition(ourMoves);
  const repeats = args.bench || 1;
  let raw;
  let elapsed = 0;
  let benchEvals = 0;
  for (let i = 0; i < repeats; i += 1) {
    runtime.engine.setPosition(ourMoves);
    const started = performance.now();
    raw = await runtime.engine.search({ maxEvals: args.nodes, timeMs: 0 });
    elapsed += performance.now() - started;
    benchEvals += finite(raw.forwards, 'forwards/evals');
  }
  if (!raw || raw.bestOfficial == null || raw.bestOfficial === '') {
    throw new Error('AB result has no best move');
  }
  const winProb = finite(raw.winProb, 'winProb');
  if (winProb < 0 || winProb > 1) throw new Error('AB result winProb is outside [0, 1]');
  const score = finite(raw.score, 'score');
  const actualEvals = finite(raw.forwards, 'forwards/evals');
  const depth = finite(raw.depth, 'depth');
  if (actualEvals < 0 || actualEvals > args.nodes || depth < 0) throw new Error('AB result exceeded bounded budget');
  const pv = Array.isArray(raw.pv) ? raw.pv : [];
  const output = {
    schema: SCHEMA,
    schema_version: VERSION,
    source: {
      ace_file: args.ace,
      ace_bundle_sha256: sha256(html),
      bytes: html.byteLength,
    },
    engine: {
      name: 'Ace',
      mode: 'ab',
      beta_ab_used: true,
      certified_default: true,
      backend: runtime.backend,
      backend_requested: args.backend,
      backend_ladder: runtime.ladder,
      seed: 13,
      config: { maxEvals: args.nodes, timeMs: 0, batchChunk: args.batchChunk },
    },
    position: { moves_official: args.moves, ply: args.moves.length },
    budget: {
      requested_evals: args.nodes,
      requested_nodes: args.nodes,
      requested_time_ms: 0,
      actual_evals: actualEvals,
      elapsed_ms: finite(elapsed, 'elapsed_ms'),
    },
    teacher: {
      best_move_official: raw.bestOfficial,
      move_official: raw.bestOfficial,
      value_win_prob_stm: winProb,
      value_stm: 2 * winProb - 1,
      score,
      proven: !!raw.proven,
      depth,
      pv_official: pv,
      pv: pv,
    },
    raw_result: raw,
  };
  if (args.bench) {
    // Backend is a WASM-SIMD or plain-JS CPU evaluator; never a GPU. Throughput
    // is actual evals (summed over all repeats) divided by elapsed seconds, i.e.
    // the average per-run eval rate; guarded against a zero-duration run so it
    // never divides by zero.
    const elapsedSeconds = elapsed / 1000;
    const throughputEvalsPerSec = elapsedSeconds > 0 ? benchEvals / elapsedSeconds : 0;
    output.benchmark = {
      backend: runtime.backend,
      batch_chunk: args.batchChunk,
      repeats,
      total_ms: elapsed,
      per_run_ms: elapsed / repeats,
      nodes_per_run: args.nodes,
      actual_evals: benchEvals,
      throughput_evals_per_sec: throughputEvalsPerSec,
    };
  }
  process.stdout.write(`${JSON.stringify(output)}\n`);
}

main().catch((error) => {
  process.stderr.write(`${error?.stack || error}\n`);
  process.exitCode = 1;
});
