#!/usr/bin/env node
'use strict';

/**
 * Extract the full Ace epoch15000 runtime surface from reference/ace.html:
 * Ka scripts, embedded weights JSON, and WASM binary blob.
 *
 * Outputs to reference/ka_weights_export/ (gitignored bulk) plus a manifest.
 */

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const PROJECT_ROOT = path.resolve(__dirname, '..', '..', '..');
const SOURCE_PATH = path.join(PROJECT_ROOT, 'reference', 'ace.html');
const DEFAULT_OUTPUT = path.join(PROJECT_ROOT, 'reference', 'ka_weights_export');
const WORK_OUTPUT = path.join(PROJECT_ROOT, 'work', 'ka_ab_teacher');

const SCRIPT_IDS = [
  'engine-core',
  'ka-encoder',
  'ka-forward',
  'ka-forward-wasm',
  'ka-forward-webgpu',
  'ka-backend',
  'ka-solver',
  'ka-engine',
  'ka-ab',
  'ka-worker',
];

const PAYLOAD_IDS = ['ka-weights', 'ka-wasm-bin'];

function sha256(data) {
  return crypto.createHash('sha256').update(data).digest('hex');
}

function scriptId(openingTag) {
  const match = /\bid\s*=\s*(["'])(.*?)\1/i.exec(openingTag);
  return match ? match[2] : null;
}

function extractTags(html, ids) {
  const found = new Map();
  const openingTag = /<script\b[^>]*>/gi;
  let match;
  while ((match = openingTag.exec(html)) !== null) {
    const close = /<\/script\s*>/gi;
    close.lastIndex = openingTag.lastIndex;
    const closingMatch = close.exec(html);
    if (!closingMatch) throw new Error('Unterminated <script> in ace.html');
    const id = scriptId(match[0]);
    if (id && ids.includes(id)) {
      if (found.has(id)) throw new Error(`Duplicate script id: ${id}`);
      found.set(id, html.slice(openingTag.lastIndex, closingMatch.index));
    }
    openingTag.lastIndex = close.lastIndex;
  }
  const missing = ids.filter((id) => !found.has(id));
  if (missing.length) throw new Error(`Missing script id(s): ${missing.join(', ')}`);
  return found;
}

function main() {
  const outDir = process.argv.includes('--out')
    ? path.resolve(process.argv[process.argv.indexOf('--out') + 1])
    : DEFAULT_OUTPUT;
  const html = fs.readFileSync(SOURCE_PATH, 'utf8');
  const scripts = extractTags(html, SCRIPT_IDS);
  const payloads = extractTags(html, PAYLOAD_IDS);

  fs.mkdirSync(outDir, { recursive: true });
  fs.mkdirSync(WORK_OUTPUT, { recursive: true });

  const manifest = {
    source: { file: SOURCE_PATH, sha256: sha256(html) },
    scripts: {},
    payloads: {},
  };

  for (const id of SCRIPT_IDS) {
    const body = scripts.get(id);
    const filename = `${id}.js`;
    for (const dir of [outDir, WORK_OUTPUT]) {
      fs.writeFileSync(path.join(dir, filename), body, 'utf8');
    }
    manifest.scripts[id] = { file: filename, sha256: sha256(body) };
  }

  for (const id of PAYLOAD_IDS) {
    const text = payloads.get(id).trim();
    const ext = id === 'ka-weights' ? 'json' : 'json';
    const filename = `${id}.${ext}`;
    fs.writeFileSync(path.join(outDir, filename), text, 'utf8');
    manifest.payloads[id] = { file: filename, sha256: sha256(text), bytes: Buffer.byteLength(text) };
  }

  const manifestPath = path.join(outDir, 'ace_runtime_manifest.json');
  fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
  process.stdout.write(`Extracted Ace runtime to ${outDir}\n`);
  process.stdout.write(`Manifest: ${manifestPath}\n`);
}

try {
  main();
} catch (error) {
  process.stderr.write(`${error.message}\n`);
  process.exitCode = 1;
}
