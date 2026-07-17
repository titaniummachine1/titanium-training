#!/usr/bin/env node
/**
 * Headless Ace epoch15000 forward harness (JS reference backend).
 *
 * Loads extracted scripts + ka-weights.json, encodes a position prefix, and
 * emits policy/value JSON with latency stats.  WebGPU/WASM paths require a
 * browser worker pool; this harness validates the JS reference path used for
 * parity locking before any teacher labels are written.
 *
 * Usage:
 *   node training/tools/ka_teacher/extract_ace_runtime.js
 *   node training/tools/ka_teacher/ace_harness.mjs --moves e2 e8
 *   node training/tools/ka_teacher/ace_harness.mjs --bench --repeats 20
 */
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { pathToFileURL } from 'node:url';
import { performance } from 'node:perf_hooks';

const ROOT = path.resolve(import.meta.dirname, '..', '..', '..');
const RUNTIME_DIR = path.join(ROOT, 'reference', 'ka_weights_export');
const WORK_DIR = path.join(ROOT, 'work', 'ka_ab_teacher');

function loadScript(filename) {
  for (const dir of [RUNTIME_DIR, WORK_DIR]) {
    const full = path.join(dir, filename);
    if (fs.existsSync(full)) return fs.readFileSync(full, 'utf8');
  }
  throw new Error(`missing script ${filename}; run extract_ace_runtime.js first`);
}

function loadWeights() {
  const weightsPath = path.join(RUNTIME_DIR, 'ka-weights.json');
  if (!fs.existsSync(weightsPath)) {
    throw new Error(`missing ${weightsPath}; run extract_ace_runtime.js first`);
  }
  return JSON.parse(fs.readFileSync(weightsPath, 'utf8'));
}

function makeSandbox() {
  const sandbox = {
    console,
    Math,
    Int16Array,
    Int32Array,
    Float32Array,
    Uint8Array,
    Array,
    Object,
    JSON,
    Map,
    Set,
    Promise,
    setTimeout,
    clearTimeout,
    atob: (b64) => Buffer.from(b64, 'base64').toString('binary'),
    btoa: (bin) => Buffer.from(bin, 'binary').toString('base64'),
    module: { exports: {} },
    exports: {},
  };
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;
  sandbox.window = sandbox;
  return sandbox;
}

function evalFactory(code, sandbox, exportName) {
  vm.runInNewContext(code, sandbox, { filename: exportName });
  return sandbox.module.exports;
}

function buildRuntime() {
  const sandbox = makeSandbox();
  const coreCode = loadScript('engine-core.js');
  vm.runInNewContext(`${coreCode}\n;globalThis.Quoridor = Quoridor;`, sandbox, { filename: 'engine-core.js' });
  const KaEncoder = evalFactory(loadScript('ka-encoder.js'), sandbox, 'ka-encoder.js');
  const KaForward = evalFactory(loadScript('ka-forward.js'), sandbox, 'ka-forward.js');
  const weights = loadWeights();
  const NetCtor = KaForward.KaNet || KaForward;
  const net = new NetCtor(weights);
  return { Quoridor: sandbox.Quoridor, KaEncoder, net };
}

function replayMoves(Quoridor, KaEncoder, moves) {
  const g = new Quoridor();
  for (const mv of moves) {
    const id = KaEncoder.officialToOurId(mv);
    g.makeMove(id);
  }
  return g;
}

function topPolicy(p, k = 5) {
  const ranked = [];
  for (let i = 0; i < p.length; i += 1) ranked.push([i, p[i]]);
  ranked.sort((a, b) => b[1] - a[1]);
  return ranked.slice(0, k).map(([id, prob]) => ({ ka_id: id, prob }));
}

function parseArgs(argv) {
  const moves = [];
  let bench = false;
  let repeats = 10;
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--bench') bench = true;
    else if (arg === '--repeats') repeats = Number(argv[++i] || 10);
    else if (arg === '--moves') {
      while (argv[i + 1] && !argv[i + 1].startsWith('--')) moves.push(argv[++i]);
    } else if (!arg.startsWith('--')) moves.push(arg);
  }
  return { moves, bench, repeats };
}

function main() {
  const { moves, bench, repeats } = parseArgs(process.argv);
  const { Quoridor, KaEncoder, net } = buildRuntime();
  const g = replayMoves(Quoridor, KaEncoder, moves);
  const feat = new Float32Array(1215);
  KaEncoder.encode(g, feat);
  if (bench) {
    for (let i = 0; i < 3; i += 1) net.forward(feat);
    const t0 = performance.now();
    for (let i = 0; i < repeats; i += 1) net.forward(feat);
    const elapsed = performance.now() - t0;
    const out = {
      backend: 'js',
      repeats,
      total_ms: elapsed,
      per_call_ms: elapsed / repeats,
      moves,
    };
    console.log(JSON.stringify(out, null, 2));
    return;
  }
  const t0 = performance.now();
  const raw = net.forward(feat);
  const elapsed = performance.now() - t0;
  const out = {
    backend: 'js',
    latency_ms: elapsed,
    moves,
    policy_top: topPolicy(raw.p || raw.policy || raw[0]),
    value_black: raw.value ?? raw.v ?? raw[1],
    ply: g.histLen || 0,
  };
  console.log(JSON.stringify(out, null, 2));
}

try {
  main();
} catch (error) {
  console.error(String(error?.stack || error));
  process.exitCode = 1;
}
