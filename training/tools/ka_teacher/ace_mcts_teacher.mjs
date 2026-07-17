#!/usr/bin/env node
/**
 * Bounded, auditable Ace MCTS teacher-label adapter.
 *
 * The supplied ace.html is read in place and evaluated in a Node VM. Its
 * certified default PUCT MCTS (ka-engine) is used directly; the beta AB
 * implementation is never loaded or selected. No copy of the bundle is made.
 *
 * Example:
 *   node training/tools/ka_teacher/ace_mcts_teacher.mjs --nodes 32
 *   node training/tools/ka_teacher/ace_mcts_teacher.mjs --moves e2 e8 --nodes 64
 */
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_ACE = path.join(process.env.USERPROFILE || 'C:/Users/Terminatort8000', 'Downloads', 'ace.html');
const SCHEMA = 'ace-mcts-teacher-v1';
const VERSION = 1;
const REQUIRED = ['engine-core', 'ka-encoder', 'ka-forward', 'ka-solver', 'ka-engine'];

function sha256(data) {
  return crypto.createHash('sha256').update(data).digest('hex');
}

function extractTags(html, ids) {
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
    if (id && ids.includes(id)) {
      if (found.has(id)) throw new Error(`Duplicate Ace script id: ${id}`);
      found.set(id, html.slice(open.lastIndex, end.index));
    }
    open.lastIndex = close.lastIndex;
  }
  const missing = ids.filter((id) => !found.has(id));
  if (missing.length) throw new Error(`Missing Ace script id(s): ${missing.join(', ')}`);
  return found;
}

function assertCertifiedDefault(html) {
  const certified = /<option\s+value=["']mcts["']\s+selected(?:\s*=\s*["']selected["'])?>\s*Ace\s*\(MCTS\)/i.test(html);
  const beta = /<option\s+value=["']ab["'][^>]*>\s*Ace-AB\s*\(beta\)/i.test(html);
  if (!certified || !beta) {
    throw new Error('Ace bundle does not expose the certified MCTS default and beta AB option');
  }
}

function makeSandbox() {
  const sandbox = {
    console, Math, Date, Error, performance, Promise,
    Array, Object, JSON, Map, Set,
    Int8Array, Uint8Array, Uint16Array, Int16Array, Int32Array,
    Float32Array, Float64Array,
    setTimeout, clearTimeout, setInterval, clearInterval,
    atob: (b64) => Buffer.from(b64, 'base64').toString('binary'),
    btoa: (bin) => Buffer.from(bin, 'binary').toString('base64'),
    module: { exports: {} }, exports: {},
  };
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;
  sandbox.window = sandbox;
  return sandbox;
}

function evalFactory(code, sandbox, name) {
  vm.runInNewContext(code, sandbox, { filename: name });
  return sandbox.module.exports;
}

function buildRuntime(html) {
  const scripts = extractTags(html, REQUIRED);
  const sandbox = makeSandbox();
  vm.runInNewContext(`${scripts.get('engine-core')}\n;globalThis.Quoridor = Quoridor;`, sandbox, {
    filename: 'ace:engine-core',
  });
  const encoder = evalFactory(scripts.get('ka-encoder'), sandbox, 'ace:ka-encoder');
  const forward = evalFactory(scripts.get('ka-forward'), sandbox, 'ace:ka-forward');
  const solver = evalFactory(scripts.get('ka-solver'), sandbox, 'ace:ka-solver');
  const engineLib = evalFactory(scripts.get('ka-engine'), sandbox, 'ace:ka-engine');
  const weightsTag = extractTags(html, ['ka-weights']).get('ka-weights').trim();
  const weights = JSON.parse(weightsTag);
  const Net = forward.KaNet || forward;
  const net = new Net(weights);
  const evaluate = (rows) => Promise.resolve(rows.map((row) => {
    const raw = net.forward(row);
    return { p: raw.p || raw.policy || raw[0], value: raw.value ?? raw.v ?? raw[1] };
  }));
  const engine = engineLib.makeEngine({
    Quoridor: sandbox.Quoridor,
    KaEncoder: encoder,
    Solver: solver,
    evaluate,
    config: {
      // Explicitly match the certified default PUCT path.
      batchSize: 1,
      gumbel: false,
      jitterEps: 0,
      seed: 13,
    },
  });
  return { encoder, engine };
}

function parseArgs(argv) {
  const moves = [];
  let ace = DEFAULT_ACE;
  let nodes = 32;
  let timeMs = 0;
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--ace') ace = path.resolve(argv[++i]);
    else if (arg === '--nodes') nodes = Number(argv[++i]);
    else if (arg === '--time-ms') timeMs = Number(argv[++i]);
    else if (arg === '--moves') {
      while (argv[i + 1] && !argv[i + 1].startsWith('--')) moves.push(argv[++i]);
    } else if (!arg.startsWith('--')) moves.push(arg);
  }
  if (!Number.isInteger(nodes) || nodes < 1 || nodes > 100000) {
    throw new Error('--nodes must be an integer in [1, 100000]');
  }
  if (!Number.isFinite(timeMs) || timeMs < 0) throw new Error('--time-ms must be non-negative');
  return { ace, moves, nodes, timeMs };
}

function officialMovesToOur(encoder, moves) {
  return moves.map((move) => {
    const id = encoder.officialToOurId(move);
    if (!Number.isInteger(id) || id < 0) throw new Error(`Invalid official move: ${move}`);
    return id;
  });
}

function rootPolicy(rootStats, encoder) {
  if (!rootStats || !Array.isArray(rootStats.moves)) return [];
  return rootStats.moves.map((our, i) => {
    const visits = Number(rootStats.N?.[i] || 0);
    const winSum = Number(rootStats.W?.[i] || 0);
    return {
      official: encoder.ourIdToOfficial(our),
      ka_id: our,
      prior: Number(rootStats.P?.[i] || 0),
      visits,
      win_sum: winSum,
      q: visits > 0 ? winSum / visits : null,
    };
  });
}

async function main() {
  const args = parseArgs(process.argv);
  if (!fs.existsSync(args.ace)) throw new Error(`Ace bundle missing: ${args.ace}`);
  const html = fs.readFileSync(args.ace, 'utf8');
  assertCertifiedDefault(html);
  const runtime = buildRuntime(html);
  const ourMoves = officialMovesToOur(runtime.encoder, args.moves);
  runtime.engine.setPosition(ourMoves);
  const started = performance.now();
  const result = await runtime.engine.search({
    maxNodes: args.nodes,
    timeMs: args.timeMs,
    gumbel: false,
  });
  const elapsed = performance.now() - started;
  const legalPolicy = rootPolicy(result.rootStats, runtime.encoder);
  const visits = legalPolicy.reduce((sum, row) => sum + row.visits, 0);
  const teacherValue = Number(result.winProb);
  const output = {
    schema: SCHEMA,
    schema_version: VERSION,
    source: {
      ace_file: args.ace,
      ace_bundle_sha256: sha256(html),
      bytes: Buffer.byteLength(html),
    },
    engine: {
      name: 'Ace',
      mode: 'mcts',
      certified_default: true,
      beta_ab_used: false,
      backend: 'node-vm-js',
      seed: 13,
    },
    position: {
      moves_official: args.moves,
      ply: args.moves.length,
    },
    budget: {
      requested_nodes: args.nodes,
      requested_time_ms: args.timeMs,
      actual_nodes: Number(result.evals || 0),
      actual_visits: visits,
      elapsed_ms: elapsed,
    },
    teacher: {
      move_official: result.bestOfficial,
      value_win_prob_stm: teacherValue,
      value_stm: 2 * teacherValue - 1,
      proven: !!result.proven,
      pv_official: result.pv || [],
    },
    legal_policy: legalPolicy,
    metadata: {
      root_pin: result.rootStats?.pin ?? null,
      root_pin_ply: result.rootStats?.pinPly ?? 0,
      stats: runtime.engine.stats,
    },
  };
  process.stdout.write(`${JSON.stringify(output, null, 2)}\n`);
}

main().catch((error) => {
  process.stderr.write(`${error?.stack || error}\n`);
  process.exitCode = 1;
});
