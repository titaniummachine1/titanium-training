#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const bundleText = fs.readFileSync(path.join(here, 'wallz_bundle.js'), 'utf8');

function extractFactory(id) {
  const marker = `${id},e=>{`;
  const start = bundleText.indexOf(marker);
  if (start < 0) {
    throw new Error(`Wallz factory ${id} not found`);
  }
  const fnStart = start + String(id).length + 1;
  let i = bundleText.indexOf('{', fnStart);
  let depth = 0;
  let quote = '';
  let escaped = false;
  for (; i < bundleText.length; i++) {
    const ch = bundleText[i];
    if (quote) {
      if (escaped) {
        escaped = false;
      } else if (ch === '\\') {
        escaped = true;
      } else if (ch === quote) {
        quote = '';
      }
      continue;
    }
    if (ch === '"' || ch === "'" || ch === '`') {
      quote = ch;
      continue;
    }
    if (ch === '{') {
      depth++;
    } else if (ch === '}') {
      depth--;
      if (depth === 0) {
        return bundleText.slice(fnStart, i + 1);
      }
    }
  }
  throw new Error(`Wallz factory ${id} did not terminate`);
}

const modules = new Map();
let currentModuleId = null;
const runtime = {
  i(id) {
    const mod = modules.get(id);
    if (!mod) {
      throw new Error(`Wallz module ${id} not loaded`);
    }
    return mod;
  },
  s(spec, moduleId = currentModuleId) {
    const mod = modules.get(moduleId) ?? {};
    for (let i = 0; i + 2 < spec.length; i += 3) {
      mod[spec[i]] = spec[i + 2];
    }
    modules.set(moduleId, mod);
  },
  q() {},
  r() {
    throw new Error('Wallz worker import is not available in node helper');
  },
};

function evalFactory(src) {
  return Function(`"use strict"; return (${src});`)();
}

function runFactory(id, src) {
  currentModuleId = id;
  try {
    evalFactory(src)(runtime);
  } finally {
    currentModuleId = null;
  }
}

runFactory(10222, extractFactory(10222));
runFactory(
  31026,
  extractFactory(31026).replace(
    ',"humanProfileForElo",0,',
    ',"wallzStats",0,B,"humanProfileForElo",0,',
  ),
);

const gameApi = modules.get(38023);
const aiApi = modules.get(31026);

function parseMove(text) {
  const move = String(text || '').trim().toLowerCase();
  const x = move.charCodeAt(0) - 97;
  const y = Number(move[1]) - 1;
  if (!Number.isInteger(x) || x < 0 || x > 8 || !Number.isInteger(y) || y < 0 || y > 8) {
    throw new Error(`bad Wallz move: ${text}`);
  }
  if (move.length > 2) {
    const o = move[2];
    if ((o !== 'h' && o !== 'v') || x > 7 || y > 7) {
      throw new Error(`bad Wallz wall: ${text}`);
    }
    return { type: 'wall', wall: { x, y, o } };
  }
  return { type: 'pawn', to: { x, y } };
}

function formatMove(move) {
  if (move.type === 'pawn') {
    return `${String.fromCharCode(97 + move.to.x)}${move.to.y + 1}`;
  }
  return `${String.fromCharCode(97 + move.wall.x)}${move.wall.y + 1}${move.wall.o}`;
}

function stateFromMoves(moves) {
  let state = gameApi.initialState();
  for (const text of moves) {
    const result = gameApi.applyMove(state, parseMove(text));
    if (!result.ok) {
      throw new Error(`Wallz rejected history move ${text}: ${result.error}`);
    }
    state = result.state;
  }
  return state;
}

function profileOptions(profile, timeMs) {
  const elo = profile === 'expert_2200' ? 2200 : Number(profile) || 2200;
  return {
    ...aiApi.humanProfileForElo(elo, { deepThink: true }),
    timeMs: Math.max(50, Math.round(Number(timeMs) || 1600)),
  };
}

function genmove(req) {
  const state = stateFromMoves(Array.isArray(req.moves) ? req.moves : []);
  const opts = profileOptions(req.profile, req.timeMs);
  const move = aiApi.findBestMove(state, opts);
  const stats = { ...(aiApi.wallzStats ?? {}) };
  return {
    id: req.id,
    move: formatMove(move),
    stats,
    profileName: `Wallz ${req.profile ?? 'expert_2200'}`,
  };
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  terminal: false,
});

rl.on('line', (line) => {
  if (!line.trim()) {
    return;
  }
  try {
    const req = JSON.parse(line);
    console.log(JSON.stringify(genmove(req)));
  } catch (err) {
    let id = null;
    try {
      id = JSON.parse(line).id ?? null;
    } catch {}
    console.log(JSON.stringify({ id, error: err?.message ?? String(err) }));
  }
});
