#!/usr/bin/env node
'use strict';

/*
 * Copy Ka-AB browser scripts for offline comparison/teacher-data work only.
 * This tool deliberately does not load the scripts or connect them to Titanium.
 */

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const REQUIRED_IDS = [
  'engine-core',
  'ka-encoder',
  'ka-forward',
  'ka-solver',
  'ka-engine',
  'ka-ab',
];

const PROJECT_ROOT = path.resolve(__dirname, '..', '..', '..');
const SOURCE_PATH = path.join(PROJECT_ROOT, 'reference', 'ace.html');
const DEFAULT_OUTPUT = path.join(PROJECT_ROOT, 'work', 'ka_ab_teacher');

function usage() {
  return [
    'Usage: node training/tools/ka_teacher/extract_ka_ab.js [--out <directory>]',
    '',
    'Extracts the six Ka-AB script bodies from reference/ace.html for offline',
    'comparison and teacher-data work. It never runs the scripts or changes the engine.',
    `Default output: ${DEFAULT_OUTPUT}`,
  ].join('\n');
}

function sha256(data) {
  return crypto.createHash('sha256').update(data).digest('hex');
}

function parseOutputDirectory(argv) {
  if (argv.length === 0) return DEFAULT_OUTPUT;
  if (argv.length === 1 && argv[0] === '--help') {
    process.stdout.write(`${usage()}\n`);
    process.exit(0);
  }
  if (argv.length === 2 && argv[0] === '--out' && argv[1].length > 0) {
    return path.resolve(argv[1]);
  }
  throw new Error(usage());
}

function scriptId(openingTag) {
  const match = /\bid\s*=\s*(["'])(.*?)\1/i.exec(openingTag);
  return match ? match[2] : null;
}

function extractRequiredBodies(html) {
  const found = new Map();
  const openingTag = /<script\b[^>]*>/gi;
  let match;

  while ((match = openingTag.exec(html)) !== null) {
    const close = /<\/script\s*>/gi;
    close.lastIndex = openingTag.lastIndex;
    const closingMatch = close.exec(html);
    if (!closingMatch) throw new Error('Unterminated <script> tag in reference/ace.html.');

    const id = scriptId(match[0]);
    if (id && REQUIRED_IDS.includes(id)) {
      if (found.has(id)) throw new Error(`Duplicate required script id: ${id}`);
      found.set(id, html.slice(openingTag.lastIndex, closingMatch.index));
    }
    openingTag.lastIndex = close.lastIndex;
  }

  const missing = REQUIRED_IDS.filter((id) => !found.has(id));
  if (missing.length > 0) {
    throw new Error(`Missing required script id(s): ${missing.join(', ')}`);
  }
  return found;
}

function main() {
  const outputDirectory = parseOutputDirectory(process.argv.slice(2));
  const source = fs.readFileSync(SOURCE_PATH);
  const bodies = extractRequiredBodies(source.toString('utf8'));

  fs.mkdirSync(outputDirectory, { recursive: true });
  const scripts = {};
  for (const id of REQUIRED_IDS) {
    const body = bodies.get(id);
    const filename = `${id}.js`;
    fs.writeFileSync(path.join(outputDirectory, filename), body, 'utf8');
    scripts[id] = { file: filename, sha256: sha256(body) };
  }

  const manifest = {
    source: { file: SOURCE_PATH, sha256: sha256(source) },
    scripts,
  };
  fs.writeFileSync(
    path.join(outputDirectory, 'manifest.json'),
    `${JSON.stringify(manifest, null, 2)}\n`,
    'utf8',
  );
  process.stdout.write(`Extracted ${REQUIRED_IDS.length} Ka-AB scripts to ${outputDirectory}\n`);
}

try {
  main();
} catch (error) {
  process.stderr.write(`${error.message}\n`);
  process.exitCode = 1;
}
