"""Fresh-process weight isolation probe for the continuation candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "training/runs/oracle_horizon_pilot_v1/continuation_e3"
ENGINE = ROOT / "engine/target-catv5-accepted-03856fe/release/titanium.exe"
DEFAULT_OUT = RUN / "diagnostics/SEARCH_ISOLATION_PROBE.json"
NETS = {
    "parent": ROOT / "training/runs/v16/accepted/epoch_0003.bin",
    "raw": RUN / "exports/continuation_raw.bin",
    "ema": RUN / "exports/continuation_ema.bin",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_stdout(proc: subprocess.CompletedProcess[str]) -> dict:
    for line in reversed(proc.stdout.splitlines()):
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return {"ok": False, "stdout": proc.stdout[-1000:], "stderr": proc.stderr[-1000:]}


def run_score(net: Path, packed: str, nodes: int) -> dict:
    env = os.environ.copy()
    env.update(TITANIUM_BOOK_MODE="off", TITANIUM_NET_WEIGHTS_PATH=str(net.resolve()))
    p = subprocess.run(
        [str(ENGINE), "score-out", "--nodes", str(nodes), "--packed", packed],
        cwd=ROOT, env=env, text=True, capture_output=True, check=False,
    )
    result = json_stdout(p)
    result["_returncode"] = p.returncode
    return result


def run_eval(net: Path) -> dict:
    env = os.environ.copy()
    env.update(TITANIUM_BOOK_MODE="off", TITANIUM_NET_WEIGHTS_PATH=str(net.resolve()))
    p = subprocess.run(
        [str(ENGINE), "eval", "e2"], cwd=ROOT, env=env,
        text=True, capture_output=True, check=False,
    )
    result = json_stdout(p)
    if p.stdout.strip().startswith("eval "):
        result = {"ok": True, "eval_output": p.stdout.strip()}
    result["_returncode"] = p.returncode
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--holdout", type=Path, default=RUN / "holdout_labels.jsonl")
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.holdout.read_text().splitlines() if line.strip()]
    chosen = []
    for band in (1, 2, 3, 0):
        chosen.extend(row for row in rows if row.get("band") == band)
    chosen = chosen[:5]
    positions = [
        {"position_id": row.get("position_id"), "band": row.get("band"),
         "packed_state_hex": row["packed_state_hex"]}
        for row in chosen
    ]
    data = {
        "schema": "search-isolation-probe-v1",
        "engine": str(ENGINE),
        "engine_exists": ENGINE.is_file(),
        "weights": {},
        "positions": positions,
        "runs": {},
        "tt_process_local_expected": True,
        "source_inspection": {
            "net_loader": "engine/src/titanium/net.rs::net uses OnceLock and reads TITANIUM_NET_WEIGHTS_PATH at cold start",
            "genmove_api": "engine/src/search/genmove.rs delegates to search::pipeline::select_move",
            "score_out_path": "engine/src/main.rs::run_score_out builds a packed Board and runs bounded AB",
            "tt_note": "Each score-out call is a fresh subprocess; TT is therefore process-local and cannot carry state between nets.",
        },
    }
    for name, path in NETS.items():
        data["weights"][name] = {
            "path": str(path), "exists": path.is_file(),
            "sha256": sha(path) if path.is_file() else None,
            "opening_eval_after_e2": run_eval(path) if path.is_file() else {"ok": False},
        }
        data["runs"][name] = {}
        if path.is_file():
            for pos in positions:
                key = str(pos["position_id"])
                data["runs"][name][key] = {
                    str(n): run_score(path, pos["packed_state_hex"], n)
                    for n in (50_000, 200_000)
                }
    diffs = []
    for pos in positions:
        key = str(pos["position_id"])
        for n in (50_000, 200_000):
            vals = {name: data["runs"].get(name, {}).get(key, {}).get(str(n), {})
                    for name in NETS}
            tuples = {name: (v.get("selected_move"), v.get("score"), v.get("proven"),
                             v.get("nodes")) for name, v in vals.items()}
            if len(set(tuples.values())) > 1:
                diffs.append({"position_id": key, "nodes": n, "values": tuples})
    data["pairwise_diffs"] = diffs
    loaded = all(data["weights"][name]["opening_eval_after_e2"].get("_returncode") == 0
                 for name in NETS if data["weights"][name]["exists"])
    data["conclusion"] = (
        "WEIGHTS_NOT_LOADED" if not loaded else
        "ISOLATION_CLEAN_BUT_SEARCH_IDENTICAL" if not diffs else "ISOLATION_SUSPECT"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "conclusion": data["conclusion"],
                      "pairwise_diffs": len(diffs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
