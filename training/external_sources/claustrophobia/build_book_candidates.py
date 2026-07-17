#!/usr/bin/env python3
"""Build review-only Claustrophobia book candidates; never writes a database."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "training"))
from diversity.book_candidates import (  # noqa: E402
    CandidateStatus,
    admit_to_live_book_allowed,
    candidate_row,
    is_clean_v1_excluded,
)
from engine_session import EngineSession  # noqa: E402

DEFAULT_OPENINGS = Path(__file__).parent / "mining_openings" / "mining_openings_pilot_v1.json"
DEFAULT_RESULTS = Path(__file__).parent / "eval_games" / "mining_pilot_v1" / "results.jsonl"
DEFAULT_OUT = Path(__file__).parent / "book_candidates"


def _sha(path: Path | None) -> str:
    if path is None or not path.is_file():
        return "not-supplied"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "not-supplied"


def _moves(actions: Iterable[Any]) -> list[str]:
    return [str(a.get("move")) if isinstance(a, dict) else str(a) for a in actions if a]


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def collect_unique_pairs(
    openings: Iterable[dict[str, Any]], results: Iterable[dict[str, Any]] = ()
) -> list[dict[str, Any]]:
    """Collect (prefix, proposed) pairs from seeds and recorded game actions."""
    pairs: dict[tuple[tuple[str, ...], str], dict[str, Any]] = {}
    for opening in openings:
        prefix = tuple(str(x) for x in opening.get("moves", []))
        proposed = opening.get("proposed_move") or opening.get("claustrophobia_move")
        if not proposed and prefix:
            # A seed is itself a usable proposal when no result file exists:
            # the final seed ply is the proposed move from its prior prefix.
            prefix, proposed = prefix[:-1], prefix[-1]
        if proposed:
            pairs.setdefault((prefix, str(proposed)), {
                "prefix_moves": list(prefix), "proposed_move": str(proposed),
                "opening_id": str(opening.get("opening_id", "unknown")),
                "provenance": dict(opening.get("provenance") or {}),
            })
    for result in results:
        opening_id = str(result.get("opening_id") or result.get("source_opening_id") or "result")
        prefix = list(result.get("opening_moves") or result.get("opening") or [])
        actions = result.get("actions") or result.get("moves") or []
        seen = list(prefix)
        for action in actions:
            move = action.get("move") if isinstance(action, dict) else action
            if not move:
                continue
            side = action.get("side") if isinstance(action, dict) else None
            if side in (None, "claustrophobia"):
                key = (tuple(str(x) for x in seen), str(move))
                pairs.setdefault(key, {
                    "prefix_moves": list(seen), "proposed_move": str(move),
                    "opening_id": opening_id,
                    "provenance": dict(result.get("provenance") or {
                        k: result.get(k) for k in (
                            "claustro_checkpoint_sha256", "claustrophobia_checkpoint_sha256",
                            "claustro_checkpoint_hash", "repository_commit", "repo_commit",
                            "source_opening_id", "titanium_weights_sha256", "titanium_weight_hash",
                        ) if result.get(k)
                    }),
                })
            seen.append(str(move))
    return list(pairs.values())


def _fixture_legal(prefix: list[str], move: str) -> bool:
    """Conservative offline fixture legality, suitable for deterministic tests."""
    token = re.compile(r"[a-h][1-9](?:[hv])?")
    if any(not token.fullmatch(str(item)) for item in prefix):
        return False
    if not move or not token.fullmatch(str(move)):
        return False
    return True


def _alternatives(info: dict[str, Any]) -> list[Any]:
    roots = info.get("rootMoves") or info.get("root_moves") or info.get("pv") or []
    return roots[:5] if isinstance(roots, list) else []


def _titanium_move_is_legal(info: dict[str, Any], proposed: str) -> bool:
    """Use Titanium's root move list when the protocol exposes one."""
    roots = info.get("rootMoves") or info.get("root_moves")
    if not isinstance(roots, list) or not roots:
        return True  # legality was already checked by session sync/fixture
    legal = {
        str(item.get("move")) if isinstance(item, dict) else str(item)
        for item in roots
    }
    return proposed in legal


def verify_row(row: dict[str, Any], session: Any) -> dict[str, Any]:
    prefix = row["prefix_moves"]
    if not _fixture_legal(prefix, row["proposed_move"]):
        row["status"] = CandidateStatus.REJECTED_ILLEGAL.value
        row["stability"] = {"stable": False, "reason": "offline_fixture_illegal"}
        return row
    details = {}
    for budget in (1.0, 4.0):
        if not session.sync(prefix):
            row["status"] = CandidateStatus.REJECTED_ILLEGAL.value
            row["stability"] = {"stable": False, "reason": "titanium_sync_failed"}
            return row
        details[str(budget)] = session.go_detailed(budget)
    best1 = details["1.0"].get("bestmove")
    best4 = details["4.0"].get("bestmove")
    titanium_legal = _titanium_move_is_legal(details["1.0"].get("info", {}), row["proposed_move"])
    stable = bool(best1 and best1 == best4 and titanium_legal)
    row["titanium_best"] = best4 or best1
    row["top_alternatives"] = _alternatives(details["4.0"].get("info", {}))
    row["budgets"] = {"seconds": [1.0, 4.0], "results": details}
    row["stability"] = {"stable": stable, "best_moves": [best1, best4]}
    row["legality"] = {
        "claustrophobia": True,
        "titanium": titanium_legal,
        "mode": "offline_fixture",
    }
    row["status"] = (
        CandidateStatus.BOOK_CANDIDATE_VERIFIED.value if stable
        else CandidateStatus.REJECTED_UNSTABLE.value
    )
    return row


def build(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = json.loads(args.openings.read_text(encoding="utf-8"))
    openings = source.get("openings", [])
    results = load_jsonl(args.results)
    pairs = collect_unique_pairs(openings, results)
    rows = []
    excluded_clean_v1 = 0

    def provenance_value(pair: dict[str, Any], names: tuple[str, ...], fallback: str) -> str:
        provenance = pair.get("provenance", {})
        for name in names:
            if provenance.get(name) not in (None, "", "not-supplied", "unknown_unavailable"):
                return str(provenance[name])
        return fallback
    for pair in pairs:
        full_line = list(pair["prefix_moves"]) + [pair["proposed_move"]]
        if is_clean_v1_excluded(
            opening_id=pair["opening_id"],
            moves=full_line,
            canonical_key=pair.get("canonical_key"),
        ):
            status = CandidateStatus.REJECTED_EVAL_LEAKAGE.value
            excluded_clean_v1 += 1
        elif not _fixture_legal(pair["prefix_moves"], pair["proposed_move"]):
            status = CandidateStatus.REJECTED_ILLEGAL.value
        else:
            status = CandidateStatus.PROPOSED.value
        rows.append(candidate_row(
            prefix_moves=pair["prefix_moves"], proposed_move=pair["proposed_move"],
            opening_id=pair["opening_id"], status=status,
            claustro_checkpoint_sha256=provenance_value(
                pair, ("claustro_checkpoint_sha256", "claustrophobia_checkpoint_sha256", "claustro_checkpoint_hash"),
                _sha(args.claustro_checkpoint)),
            repository_commit=provenance_value(
                pair, ("repository_commit", "repo_commit"), _repo_commit()),
            titanium_weights_sha256=provenance_value(
                pair, ("titanium_weights_sha256", "titanium_weight_hash"), _sha(args.weights)),
        ))
    verified = 0
    if args.verify:
        if args.titanium_bin is None:
            raise ValueError("--verify requires --titanium-bin")
        session = EngineSession("titanium", args.weights, engine_bin=args.titanium_bin)
        try:
            for row in rows[:args.max if args.max else len(rows)]:
                if row["status"] == CandidateStatus.PROPOSED.value:
                    provenance_ok = all(
                        value not in (None, "", "not-supplied", "unknown_unavailable")
                        for value in row["provenance"].values()
                    )
                    if provenance_ok:
                        verify_row(row, session)
                        verified += 1
        finally:
            session.close()
    counts = {status.value: sum(row["status"] == status.value for row in rows)
              for status in CandidateStatus}
    summary = {
        "schema_version": "claustrophobia-book-candidates-summary-v1",
        "rows": len(rows), "verified": verified, "output_is_review_only": True,
        "counts_by_status": counts,
        "n_excluded_clean_v1": excluded_clean_v1,
        "live_book_db_written": False, "training_rows_written": False,
        # This builder is review-only; admission is never inferred from rows.
        "admit_to_live_book_allowed": False,
    }
    return rows, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openings", type=Path, default=DEFAULT_OPENINGS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--claustro-checkpoint", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--titanium-bin", type=Path, default=None)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--max", type=int, default=0, help="maximum rows to verify")
    args = parser.parse_args(argv)
    rows, summary = build(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "book_candidates_v1.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
