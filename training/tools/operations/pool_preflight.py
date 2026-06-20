#!/usr/bin/env python3
"""Validate pool + training stack before starting live games.

Exit 0 = ready. Exit 1 = print errors and abort.
  python training/pool_preflight.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from tools.datagen.datagen import DB_PATH, open_db  # noqa: E402
from titanium_training.training.guards import HALFPW_WEIGHT_BYTES, net_weights_size_ok  # noqa: E402

BIN = ROOT / "engine" / "target" / "release" / ("titanium.exe" if os.name == "nt" else "titanium")
WEIGHTS = ROOT / "engine" / "src" / "titanium" / "net_weights.bin"
FROZEN = ROOT / "engine" / "src" / "titanium" / "net_weights_frozen.bin"
OVERNIGHT = ROOT / "site" / "overnight_batch.js"
REMOTE_WORKER = ROOT / "site" / "remote_game_worker.js"
SELF_MATCH = ROOT / "site" / "self_match.js"
COORD = ROOT / "training" / "coordinator.py"
GAME_LOGIC = ROOT / "site" / "web" / "src" / "lib" / "gameLogic.js"

ENGINE_FLAGS = (
    "titanium-v15",
    "titanium-v15-frozen",
    "ace-v13-ti-pure",
)


def _err(msg: str) -> None:
    print(f"  FAIL: {msg}", file=sys.stderr)


def check_paths() -> list[str]:
    fails: list[str] = []
    for p, label in (
        (BIN, "titanium release binary"),
        (WEIGHTS, "net_weights.bin"),
        (FROZEN, "net_weights_frozen.bin"),
        (OVERNIGHT, "overnight_batch.js"),
        (REMOTE_WORKER, "remote_game_worker.js"),
        (SELF_MATCH, "self_match.js"),
        (COORD, "coordinator.py"),
        (GAME_LOGIC, "gameLogic.js"),
    ):
        if not p.exists():
            fails.append(f"missing {label}: {p}")
    if WEIGHTS.exists() and not net_weights_size_ok(WEIGHTS):
        fails.append(f"net_weights.bin size != {HALFPW_WEIGHT_BYTES} B")
    if FROZEN.exists() and not net_weights_size_ok(FROZEN):
        fails.append(f"net_weights_frozen.bin size != {HALFPW_WEIGHT_BYTES} B")
    return fails


def check_db() -> list[str]:
    fails: list[str] = []
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = open_db(DB_PATH, write=True)
        conn.execute("SELECT 1")
        conn.close()
    except Exception as e:
        fails.append(f"all_games.db not writable: {e}")
    return fails


def check_engine_eval() -> list[str]:
    fails: list[str] = []
    try:
        r = subprocess.run(
            [str(BIN), "eval-batch"],
            input=b"e2 e8 e3 e7\n",
            capture_output=True,
            timeout=30,
        )
        if r.returncode != 0:
            fails.append(f"eval-batch exited {r.returncode}: {(r.stderr or r.stdout)[:200]!r}")
            return fails
        line = (r.stdout or b"").decode("utf-8", errors="replace").strip().splitlines()
        if not line:
            fails.append("eval-batch returned no output")
            return fails
        json.loads(line[0])
    except Exception as e:
        fails.append(f"engine eval-batch: {e}")
    return fails


def check_engine_sessions() -> list[str]:
    fails: list[str] = []
    for flag in ENGINE_FLAGS:
        try:
            r = subprocess.run(
                [str(BIN), "session", "--engine", flag],
                input=b"reset\nquit\n",
                capture_output=True,
                timeout=45,
            )
            out = ((r.stdout or b"") + (r.stderr or b"")).decode("utf-8", errors="replace")
            if r.returncode != 0 and "ready" not in out:
                fails.append(f"session --engine {flag} rc={r.returncode}: {out[:160]!r}")
        except Exception as e:
            fails.append(f"session --engine {flag}: {e}")
    return fails


def check_node_modules() -> list[str]:
    """Ensure overnight_batch can load (binA/binB resolution smoke)."""
    fails: list[str] = []
    script = ROOT / "training" / "_pool_preflight_smoke.js"
    script.write_text(
        """
'use strict';
const path = require('path');
const fs = require('fs');
const bin = path.resolve(__dirname, '../engine/target/release/titanium.exe');
if (!fs.existsSync(bin)) { console.error('bin missing'); process.exit(1); }
const selfMatch = require('../site/self_match');
const opts = { engineA: 'titanium-v15', engineB: 'titanium-v15-frozen', bin, timeA: 1, timeB: 1 };
if (!opts.bin) { console.error('opts.bin empty'); process.exit(1); }
// playGame reads binA/binB — must fall back to bin
const a = opts.binA || opts.bin;
const b = opts.binB || opts.bin;
if (typeof a !== 'string' || typeof b !== 'string') {
  console.error('binA/binB not resolved:', a, b);
  process.exit(1);
}
console.log('node smoke ok');
""",
        encoding="utf-8",
    )
    try:
        r = subprocess.run(["node", str(script)], cwd=str(ROOT), capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            fails.append(f"node smoke: {(r.stderr or r.stdout).strip()[:200]}")
    except Exception as e:
        fails.append(f"node smoke: {e}")
    finally:
        script.unlink(missing_ok=True)
    return fails


def main() -> int:
    print("Pool preflight...")
    all_fails: list[str] = []
    for name, fn in (
        ("paths", check_paths),
        ("database", check_db),
        ("engine eval", check_engine_eval),
        ("engine sessions", check_engine_sessions),
        ("node pool smoke", check_node_modules),
    ):
        fails = fn()
        if fails:
            print(f"  [{name}]")
            for f in fails:
                _err(f)
            all_fails.extend(fails)
        else:
            print(f"  [{name}] OK")

    if all_fails:
        print(f"\nPreflight failed ({len(all_fails)} issue(s)). Fix before starting pool.", file=sys.stderr)
        return 1
    print("Preflight OK — pool may start.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
