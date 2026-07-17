#!/usr/bin/env node
/**
 * Persistent batched Ka epoch-15000 teacher.
 *
 * Protocol: one JSON request per stdin line, one JSON response per stdout line.
 * A request is {"id": ..., "positions": [{"id": ..., "moves": [...]}]}.
 * The Ka runtime and its 12 MB weight blob are initialized once; each request
 * is encoded together and evaluated through the existing WASM-SIMD batch path.
 */
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import vm from 'node:vm';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..', '..', '..');
const RUNTIME = path.join(ROOT, 'reference', 'ka_weights_export');
const WORK = path.join(ROOT, 'work', 'ka_ab_teacher');
const NATIVE_RUNTIME = path.join(HERE, 'native_runtime');
const SCHEMA = 'ka-nn-batch-v1';

function parseArgs(argv) {
  const out = {
    backend: 'directml', batchMax: 64, modelBatch: 64, deviceId: 1, model: null, threads: 1,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--backend') out.backend = String(argv[++i] || '');
    else if (arg === '--batch-max') out.batchMax = Number(argv[++i]);
    else if (arg === '--model-batch') out.modelBatch = Number(argv[++i]);
    else if (arg === '--device-id') out.deviceId = Number(argv[++i]);
    else if (arg === '--model') out.model = path.resolve(String(argv[++i] || ''));
    else if (arg === '--threads') out.threads = Number(argv[++i]);
    else throw new Error(`unknown argument: ${arg}`);
  }
  if (!['directml', 'cpu', 'wasm', 'js', 'auto'].includes(out.backend)) {
    throw new Error('--backend must be directml, cpu, wasm, js, or auto');
  }
  if (!Number.isInteger(out.batchMax) || out.batchMax < 1 || out.batchMax > 1024) {
    throw new Error('--batch-max must be an integer in [1, 1024]');
  }
  if (!Number.isInteger(out.modelBatch) || out.modelBatch < 1 || out.modelBatch > 1024) {
    throw new Error('--model-batch must be an integer in [1, 1024]');
  }
  if (!Number.isInteger(out.deviceId) || out.deviceId < 0) {
    throw new Error('--device-id must be a non-negative integer');
  }
  if (!Number.isInteger(out.threads) || out.threads < 1 || out.threads > 64) {
    throw new Error('--threads must be an integer in [1, 64]');
  }
  return out;
}

function loadText(name) {
  for (const dir of [RUNTIME, WORK]) {
    const candidate = path.join(dir, name);
    if (fs.existsSync(candidate)) return fs.readFileSync(candidate, 'utf8');
  }
  throw new Error(`missing extracted Ka asset: ${name}`);
}

function loadJson(name) {
  return JSON.parse(loadText(name));
}

function makeSandbox() {
  const quiet = (...args) => process.stderr.write(`${args.join(' ')}\n`);
  const sandbox = {
    console: { log: quiet, info: quiet, warn: quiet, error: quiet },
    Math, Date, Error, Promise, Array, Object, JSON, Map, Set, WeakMap, RegExp,
    Number, String, Boolean, BigInt,
    Int8Array, Uint8Array, Uint8ClampedArray, Uint16Array, Int16Array,
    Int32Array, Uint32Array, Float32Array, Float64Array, BigInt64Array,
    BigUint64Array, DataView, ArrayBuffer, SharedArrayBuffer,
    setTimeout, clearTimeout, setInterval, clearInterval,
    atob: (value) => Buffer.from(value, 'base64').toString('binary'),
    btoa: (value) => Buffer.from(value, 'binary').toString('base64'),
    module: { exports: {} }, exports: {}, Buffer, WebAssembly,
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

function decodeWasmBytes(doc) {
  const encoded = doc.b64 || doc.base64 || doc.bytes;
  if (typeof encoded !== 'string') throw new Error('ka-wasm-bin.json has no base64 payload');
  return encoded;
}

async function buildTeacher(args) {
  const sandbox = makeSandbox();
  vm.runInNewContext(
    `${loadText('engine-core.js')}\n;globalThis.Quoridor = Quoridor;`,
    sandbox,
    { filename: 'engine-core.js' },
  );
  const encoder = evaluateModule(loadText('ka-encoder.js'), sandbox, 'ka-encoder.js');
  const weightsText = loadText('ka-weights.json');
  const weights = JSON.parse(weightsText);
  let built;
  if (args.backend === 'directml' || args.backend === 'cpu') {
    const require = createRequire(path.join(NATIVE_RUNTIME, 'package.json'));
    const ort = require('onnxruntime-node');
    const modelPath = args.model || path.join(NATIVE_RUNTIME, `ka_epoch15000_b${args.modelBatch}.onnx`);
    if (!fs.existsSync(modelPath)) throw new Error(`missing native Ka model: ${modelPath}`);
    const executionProviders = args.backend === 'directml'
      ? [{ name: 'dml', deviceId: args.deviceId }]
      : ['cpu'];
    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders,
      executionMode: 'sequential',
      intraOpNumThreads: args.threads,
      interOpNumThreads: 1,
      enableMemPattern: args.backend !== 'directml',
      graphOptimizationLevel: 'all',
    });
    built = {
      backend: args.backend === 'directml' ? `directml:${args.deviceId}` : 'onnx-cpu',
      ladder: [args.backend],
      evaluate: async (featureRows, requested = {}) => {
        const results = [];
        const outputNames = ['policy', 'value_black'];
        if (requested.includeAttention) outputNames.push('attention_l15');
        if (requested.includeTrunk) outputNames.push('trunk_l5');
        for (let offset = 0; offset < featureRows.length; offset += args.modelBatch) {
          const count = Math.min(args.modelBatch, featureRows.length - offset);
          const packed = new Float32Array(args.modelBatch * 1215);
          for (let row = 0; row < count; row += 1) {
            packed.set(featureRows[offset + row], row * 1215);
          }
          const feeds = { features: new ort.Tensor('float32', packed, [args.modelBatch, 9, 9, 15]) };
          // Fetch large diagnostic tensors only for visualization requests. Normal
          // labeling needs policy + value and previously copied tens of MB per batch.
          const output = await session.run(feeds, outputNames);
          const probabilities = output.policy.data;
          const values = output.value_black.data;
          const attention = output.attention_l15?.data;
          const trunk = output.trunk_l5?.data;
          for (let row = 0; row < count; row += 1) {
            const result = {
              p: probabilities.slice(row * 137, (row + 1) * 137),
              value: values[row],
            };
            if (attention) {
              const stride = 4 * 81 * 81;
              result.attention_l15 = attention.slice(row * stride, (row + 1) * stride);
            }
            if (trunk) {
              const stride = 128 * 81;
              result.trunk_l5 = trunk.slice(row * stride, (row + 1) * stride);
            }
            results.push(result);
          }
        }
        return results;
      },
    };
  } else {
    const kaForward = evaluateModule(loadText('ka-forward.js'), sandbox, 'ka-forward.js');
    const kaWasm = evaluateModule(loadText('ka-forward-wasm.js'), sandbox, 'ka-forward-wasm.js');
    const backend = evaluateModule(loadText('ka-backend.js'), sandbox, 'ka-backend.js');
    const wasmBytes = decodeWasmBytes(loadJson('ka-wasm-bin.json'));
    built = await backend.makeEvaluate({
      backend: args.backend,
      strict: args.backend === 'wasm',
      batchMax: args.batchMax,
      weights,
      wasmBytes,
      KaNet: kaForward.KaNet || kaForward,
      KaWasm: kaWasm,
    });
  }
  return {
    Quoridor: sandbox.Quoridor,
    encoder,
    evaluate: built.evaluate,
    backend: built.backend,
    ladder: built.ladder,
    checkpoint: weights.meta?.checkpoint || 'unknown',
    weightsSha256: crypto.createHash('sha256').update(weightsText).digest('hex'),
    native: args.backend === 'directml' || args.backend === 'cpu',
  };
}

function replay(runtime, moves, verifyFastLegality = false) {
  if (!Array.isArray(moves)) throw new Error('moves must be an array');
  const game = new runtime.Quoridor();
  const pawnMoves = new Int16Array(8);
  for (const official of moves) {
    const move = runtime.encoder.officialToOurId(String(official));
    let legal = false;
    if (Number.isInteger(move) && move >= 0 && move < 100) {
      const count = game.genPawnMoves(pawnMoves, 0);
      for (let index = 0; index < count; index += 1) {
        if (pawnMoves[index] === move) {
          legal = true;
          break;
        }
      }
    } else if (Number.isInteger(move) && move >= 100 && move < 164) {
      legal = game.wallLegal(0, move - 100);
    } else if (Number.isInteger(move) && move >= 200 && move < 264) {
      legal = game.wallLegal(1, move - 200);
    }
    if (verifyFastLegality) {
      const legacyLegal = game.legalMoves().includes(move);
      if (legal !== legacyLegal) {
        throw new Error(`fast replay legality mismatch: ${official}`);
      }
    }
    if (!legal) {
      throw new Error(`illegal move in prefix: ${official}`);
    }
    game.makeMove(move);
  }
  return game;
}

function policySummary(runtime, game, probabilities) {
  const mask = new Uint8Array(137);
  const legalMoves = runtime.encoder.legalKaMask(game, mask);
  let legalMass = 0;
  for (let action = 0; action < 137; action += 1) {
    if (mask[action]) legalMass += probabilities[action];
  }
  const denom = legalMass > 0 ? legalMass : 1;
  const ranked = legalMoves.map((move) => {
    const action = runtime.encoder.ourMoveToKaId(game, move);
    return {
      move: runtime.encoder.ourIdToOfficial(move),
      action,
      probability: probabilities[action] / denom,
    };
  }).sort((a, b) => b.probability - a.probability || a.action - b.action);
  let entropy = 0;
  for (const row of ranked) {
    const p = row.probability;
    if (p > 0) entropy -= p * Math.log(p);
  }
  return {
    best_move: ranked[0]?.move || null,
    confidence: ranked[0]?.probability || 0,
    entropy,
    legal_mass: legalMass,
    top: ranked.slice(0, 8),
  };
}

function trunkSummary(values, limit = 16) {
  const channels = [];
  for (let channel = 0; channel < 128; channel += 1) {
    const offset = channel * 81;
    const map = Array.from(values.slice(offset, offset + 81), (value) => Number(value.toFixed(6)));
    const mean = map.reduce((sum, value) => sum + value, 0) / map.length;
    const variance = map.reduce((sum, value) => sum + (value - mean) ** 2, 0) / map.length;
    channels.push({
      channel,
      mean: Number(mean.toFixed(6)),
      std: Number(Math.sqrt(variance).toFixed(6)),
      min: Math.min(...map),
      max: Math.max(...map),
      map,
    });
  }
  channels.sort((a, b) => b.std - a.std || a.channel - b.channel);
  return channels.slice(0, Math.max(1, Math.min(128, Number(limit) || 16)));
}

async function evaluateRequest(runtime, request) {
  if (!request || !Array.isArray(request.positions) || request.positions.length === 0) {
    throw new Error('request.positions must be a non-empty array');
  }
  const started = performance.now();
  const prepared = [];
  const rejected = [];
  for (const [index, position] of request.positions.entries()) {
    try {
      const game = replay(
        runtime,
        position.moves || [],
        Boolean(request.verify_replay_legality),
      );
      const features = new Float32Array(1215);
      runtime.encoder.encode(game, features);
      prepared.push({ position, game, features });
    } catch (error) {
      rejected.push({
        id: position?.id ?? index,
        error: String(error?.message || error),
      });
    }
  }
  const preparedAt = performance.now();
  const raw = prepared.length
    ? await runtime.evaluate(
      prepared.map((row) => row.features),
      runtime.native
        ? {
          includeAttention: Boolean(request.include_attention),
          includeTrunk: Boolean(request.include_trunk),
        }
        : prepared.length,
    )
    : [];
  const evaluatedAt = performance.now();
  const rows = prepared.map((row, index) => {
    const valueBlack = Number(raw[index].value);
    if (!Number.isFinite(valueBlack) || valueBlack < -1 || valueBlack > 1) {
      throw new Error(`non-finite/out-of-range Ka value at row ${index}`);
    }
    const sideToMove = row.game.turn % 2;
    const result = {
      id: row.position.id ?? index,
      ply: row.position.moves?.length || 0,
      side_to_move: sideToMove,
      value_black: valueBlack,
      value_stm: sideToMove === 0 ? valueBlack : -valueBlack,
      policy: policySummary(runtime, row.game, raw[index].p),
    };
    if (request.include_attention && raw[index].attention_l15) {
      result.attention_l15 = Array.from(raw[index].attention_l15);
    }
    if (request.include_trunk && raw[index].trunk_l5) {
      result.trunk_l5 = trunkSummary(raw[index].trunk_l5, request.trunk_top_channels);
    }
    return result;
  });
  return {
    schema: SCHEMA,
    id: request.id ?? null,
    ok: true,
    backend: runtime.backend,
    checkpoint: runtime.checkpoint,
    weights_sha256: runtime.weightsSha256,
    rows,
    rejected,
    timings_ms: {
      prepare: preparedAt - started,
      inference: evaluatedAt - preparedAt,
      summarize: performance.now() - evaluatedAt,
    },
  };
}

async function main() {
  const args = parseArgs(process.argv);
  const runtime = await buildTeacher(args);
  process.stderr.write(
    `ka_nn_batch_worker ready backend=${runtime.backend} checkpoint=${runtime.checkpoint}\n`,
  );
  const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of input) {
    if (!line.trim()) continue;
    let request;
    try {
      request = JSON.parse(line);
      const response = await evaluateRequest(runtime, request);
      process.stdout.write(`${JSON.stringify(response)}\n`);
    } catch (error) {
      process.stdout.write(`${JSON.stringify({
        schema: SCHEMA,
        id: request?.id ?? null,
        ok: false,
        error: String(error?.message || error),
      })}\n`);
    }
  }
}

main().catch((error) => {
  process.stderr.write(`${String(error?.stack || error)}\n`);
  process.exitCode = 1;
});
