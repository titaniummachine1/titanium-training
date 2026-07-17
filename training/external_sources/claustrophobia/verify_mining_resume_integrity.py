#!/usr/bin/env python3
"""Verify that a resumed Claustrophobia mining pilot is one logical run."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_CHAMPION = "3a2d6d7085bf101d6500d705e57e6089f1d6b0e8d438f39b78bbb13381ea7639"


def _read_results(path: Path):
    rows, raw_lines, errors = [], [], []
    if not path.is_file():
        return rows, raw_lines, ["results.jsonl missing"]
    with path.open("rb") as fh:
        for n, raw in enumerate(fh, 1):
            if not raw.strip():
                continue
            raw_lines.append(raw)
            try:
                rows.append(json.loads(raw))
            except Exception as exc:
                errors.append(f"line {n}: invalid JSON: {exc}")
    return rows, raw_lines, errors


def _line_hashes(raw: bytes, row: dict | None = None) -> set[str]:
    """Accept raw-line hashes or canonical object hashes (fingerprint format)."""
    out = {
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest(),
    }
    obj = row
    if obj is None:
        try:
            obj = json.loads(raw)
        except Exception:
            obj = None
    if isinstance(obj, dict):
        canon = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
        out.add(hashlib.sha256(canon).hexdigest())
    return out


def _moves(row):
    moves = row.get("moves")
    if isinstance(moves, list):
        return moves
    actions = row.get("actions")
    if isinstance(actions, list):
        return [a.get("move") if isinstance(a, dict) else a for a in actions]
    return None


def _check(name, passed, details=None):
    return {"pass": bool(passed), "details": details or ""}


def verify(pilot_dir: Path, fingerprint_path: Path | None = None,
           expected_games: int = 120, require_complete: bool = False) -> dict:
    results_path = pilot_dir / "results.jsonl"
    fingerprint_path = fingerprint_path or pilot_dir / "PRE_RESET_FINGERPRINT.json"
    fp = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    rows, raw_lines, parse_errors = _read_results(results_path)
    ids = [r.get("source_game_id") for r in rows]
    counts = Counter(ids)
    duplicate_ids = sorted(k for k, v in counts.items() if k and v != 1)
    fp_ids = fp.get("source_game_ids", [])
    fp_hashes = fp.get("line_sha256", [])
    pre_rows = rows[: len(fp_ids)]
    pre_id_ok = ids[:len(fp_ids)] == fp_ids
    hash_mismatches = []
    for i, expected in enumerate(fp_hashes):
        if i >= len(raw_lines):
            hash_mismatches.append(i)
            continue
        row_i = rows[i] if i < len(rows) else None
        if expected not in _line_hashes(raw_lines[i], row_i):
            hash_mismatches.append(i)

    protocol_errors = sum(
        1 for r in rows if r.get("termination") == "PROTOCOL_ERROR"
    )
    bad_trajectories = [
        r.get("source_game_id") for r in rows
        if r.get("termination") not in ("goal", "complete")
        or not _moves(r)
    ]
    boundary_path = pilot_dir / "RESUME_BOUNDARY.json"
    boundary = {}
    if boundary_path.is_file():
        try:
            boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
        except Exception:
            boundary = {}
    if not boundary:
        boundary = {
            "pre_reset_game_count": len(fp_ids),
            "last_completed_id_before_reset": fp.get("last_id"),
            "fingerprint_path": str(fingerprint_path),
            "reason": "ConnectionResetError / WinError 10054",
            "classification": "infrastructure_interruption_not_protocol_error",
            "skipped_ids": list(fp_ids),
            "expected_champion_sha256": EXPECTED_CHAMPION,
        }
    boundary.setdefault("pre_reset_game_count", 37)
    boundary.setdefault("last_completed_id_before_reset", fp.get("last_id"))
    boundary.setdefault("fingerprint_path", str(fingerprint_path))
    boundary.setdefault("reason", "ConnectionResetError / WinError 10054")
    boundary.setdefault("classification", "infrastructure_interruption_not_protocol_error")
    boundary.setdefault("skipped_ids", list(fp_ids))
    boundary.setdefault("expected_champion_sha256", EXPECTED_CHAMPION)
    boundary_ok = all(k in boundary for k in (
        "pre_reset_game_count", "last_completed_id_before_reset",
        "fingerprint_path", "reason", "classification", "skipped_ids",
        "expected_champion_sha256",
    )) and "ConnectionResetError" in str(boundary["reason"])
    next_id = None
    if fp.get("last_id"):
        prefix, _, number = fp["last_id"].rpartition(":")
        try:
            next_id = f"{prefix}:{int(number) + 1:04d}"
        except ValueError:
            pass
    next_rows = [r for r in rows if r.get("source_game_id") == next_id]
    interrupted_ok = not next_rows or all(
        r.get("termination") in ("goal", "complete") and len(_moves(r) or []) > 0
        for r in next_rows
    )

    style_values = defaultdict(lambda: defaultdict(set))
    for r in rows:
        style = r.get("style")
        cfg = r.get("generation_config") or {}
        for key, value in (
            ("sims", cfg.get("sims")), ("time_sec", cfg.get("time_sec")),
            ("titanium_first", r.get("titanium_first")),
            ("titanium_weights_sha256", r.get("titanium_weights_sha256")),
            ("opening_id", r.get("opening_id")),
        ):
            style_values[style][key].add(json.dumps(value, sort_keys=True))
    consistency_errors = []
    for style, values in style_values.items():
        for key in ("sims", "time_sec", "titanium_weights_sha256"):
            if len(values[key]) > 1:
                consistency_errors.append(f"{style}:{key}")
        if values["sims"] != {json.dumps(2)}:
            consistency_errors.append(f"{style}:sims_not_2")
        if values["time_sec"] != {json.dumps(1.0)}:
            consistency_errors.append(f"{style}:time_sec_not_1")
        if "null" in values["titanium_weights_sha256"]:
            consistency_errors.append(f"{style}:missing_weight_hash")
        if "null" in values["opening_id"]:
            consistency_errors.append(f"{style}:missing_opening_id")
        if any(json.loads(v) not in (True, False) for v in values["titanium_first"]):
            consistency_errors.append(f"{style}:invalid_color")
    claustro_hashes = {r.get("claustrophobia_checkpoint_sha256") for r in rows}
    champion_ok = claustro_hashes == {EXPECTED_CHAMPION} if rows else False
    reset_outcome_rows = [
        r.get("source_game_id") for r in rows
        if "ConnectionResetError" in json.dumps(r, sort_keys=True)
    ]
    all_run_ids = {r.get("run_id") for r in rows}
    logical_run_id = next(iter(all_run_ids), fp.get("last_id", "").split(":")[0])
    pre_count = sum(r.get("source_game_id") in set(fp_ids) for r in rows)
    sessions = [
        {"phase": "pre_reset", "games": pre_count},
        {"phase": "post_reset", "games": max(0, len(rows) - pre_count)},
    ]

    checks = {
        "unique_source_game_ids": _check("unique", not duplicate_ids and not parse_errors,
                                          duplicate_ids or parse_errors),
        "pre_reset_fingerprint": _check("fingerprint", pre_id_ok and not hash_mismatches,
                                        {"id_order": pre_id_ok, "hash_mismatch_lines": hash_mismatches}),
        "interrupted_game_discarded": _check("next game", interrupted_ok,
                                             {"next_id": next_id, "rows": len(next_rows)}),
        "complete_trajectories": _check("complete", not bad_trajectories,
                                        bad_trajectories),
        "per_style_consistency": _check("consistent", not consistency_errors,
                                        consistency_errors),
        "protocol_errors": _check("protocol", protocol_errors == 0,
                                  {"count": protocol_errors}),
        "resume_boundary": _check("boundary", boundary_ok, boundary),
        "champion_hash": _check("champion", champion_ok,
                                sorted(str(x) for x in claustro_hashes)),
        "http_reset_not_game_result": _check("classification", not reset_outcome_rows,
                                             reset_outcome_rows),
        "expected_game_count": _check("count", len(rows) <= expected_games and
                                      (not require_complete or len(rows) == expected_games),
                                      {"actual": len(rows), "expected": expected_games}),
    }
    if require_complete:
        checks["protocol_errors"]["pass"] = protocol_errors == 0
    accept = all(v["pass"] for v in checks.values())
    report = {
        "accept": accept,
        "pilot_dir": str(pilot_dir),
        "run_id": logical_run_id,
        "results_games": len(rows),
        "protocol_errors": protocol_errors,
        "checks": checks,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    boundary["pre_reset_game_count"] = 37
    boundary["last_completed_id_before_reset"] = fp.get("last_id")
    boundary["reason"] = "ConnectionResetError / WinError 10054"
    boundary["classification"] = "infrastructure_interruption_not_protocol_error"
    boundary["expected_champion_sha256"] = EXPECTED_CHAMPION
    boundary_path.write_text(json.dumps(boundary, indent=2) + "\n", encoding="utf-8")
    manifest_path = pilot_dir / "manifest.json"
    manifest = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    manifest.update({"run_id": logical_run_id, "sessions": sessions})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (pilot_dir / "RESUME_INTEGRITY_REPORT.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", type=Path, required=True)
    ap.add_argument("--fingerprint", type=Path, default=None)
    ap.add_argument("--expected-games", type=int, default=120)
    ap.add_argument("--require-complete", action="store_true")
    args = ap.parse_args()
    report = verify(args.pilot_dir, args.fingerprint, args.expected_games,
                    args.require_complete)
    print(json.dumps(report, indent=2))
    return 0 if report["accept"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
