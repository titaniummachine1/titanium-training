"""Build child-value action-contrast rows from oracle horizon labels.

The preserving action is selected from the proven ladder first, then from
``best_move`` only when it is legal in the packed parent.  Other legal moves
are capped and scored with the parent net as a shallow proxy.  The proxy is
not an oracle label; targets are deliberately attached to child states.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
from titanium_training.store.state import (  # noqa: E402
    PositionState, apply_move, decode_move, legal_wall_codes, valid_pawn_destinations,
)


def legal_moves(state: PositionState) -> list[str]:
    from titanium_training.store.state import cell_to_notation
    return [cell_to_notation(c) for c in valid_pawn_destinations(state)] + [
        decode_move(state, c) for c in legal_wall_codes(state)
    ]


def _score_out(engine: Path, net: Path, packed: bytes, nodes: int) -> float | None:
    env = os.environ.copy()
    env.update(TITANIUM_BOOK_MODE="off", TITANIUM_NET_WEIGHTS_PATH=str(net.resolve()))
    p = subprocess.run(
        [str(engine), "score-out", "--nodes", str(nodes), "--packed", packed.hex()],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=180,
    )
    if p.returncode:
        return None
    for line in reversed(p.stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("score"), (int, float)):
            return float(value["score"])
    return None


def _first(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def build_row(
    row: dict[str, Any], *, engine: Path | None, net: Path | None,
    nodes: int, alt_cap: int,
) -> dict[str, Any] | None:
    packed_hex = row.get("packed_state_hex") or row.get("parent_packed_hex")
    if not isinstance(packed_hex, str):
        return None
    state = PositionState.unpack_state(bytes.fromhex(packed_hex))
    legal = legal_moves(state)
    proven = _first(row, ("ladder_selected_move", "last_proven_ladder_selected_move",
                          "proven_ladder_selected_move", "selected_move"))
    best = _first(row, ("best_move",))
    preserving = proven if proven in legal else (best if best in legal else None)
    if preserving is None:
        return {
            "parent_packed_hex": packed_hex, "preserving_children": [],
            "losing_or_worsening_children": [], "band": row.get("band"),
            "lineage_id": row.get("lineage_id") or row.get("proof_lineage_id"),
            "oracle_wdl": row.get("oracle_wdl") or row.get("oracle_wdl_stm"),
            "label_class": row.get("label_class", "UNRESOLVED_MOVE"),
            "supervision": "child_value_backed", "status": "no_legal_preserving_move",
        }
    alternatives = [move for move in legal if move != preserving][: max(0, alt_cap)]
    scores: dict[str, float | None] = {}
    if engine and net and engine.is_file() and net.is_file():
        for move in alternatives:
            scores[move] = _score_out(engine, net, apply_move(state, move).packed_state(), nodes)
    # score-out returns the child STM score, therefore negate it to parent STM.
    root_scores = {m: (-v if v is not None else None) for m, v in scores.items()}
    preserving_score = root_scores.get(preserving)
    if preserving_score is None and engine and net and engine.is_file() and net.is_file():
        preserving_score = -(_score_out(engine, net, apply_move(state, preserving).packed_state(), nodes) or 0.0)
    worsening = [
        move for move in alternatives
        if root_scores.get(move) is not None and preserving_score is not None
        and root_scores[move] < preserving_score
    ]
    def child(move: str, source: str) -> dict[str, Any]:
        return {"move": move, "child_packed_hex": apply_move(state, move).packed_state().hex(),
                "source": source, "proxy_root_stm_score": root_scores.get(move)}
    return {
        "parent_packed_hex": packed_hex,
        "preserving_children": [child(preserving, "ladder_proven" if proven == preserving else "labeled")],
        "losing_or_worsening_children": [child(m, "shallow_proxy") for m in worsening],
        "band": row.get("band"),
        "lineage_id": row.get("lineage_id") or row.get("proof_lineage_id"),
        "oracle_wdl": row.get("oracle_wdl") or row.get("oracle_wdl_stm"),
        "label_class": row.get("label_class", "ORACLE_SUPPORTED_PARTIAL"),
        "supervision": "child_value_backed",
        "proxy_definition": "parent-net score-out on child, negated to parent STM; lower than preserving is worsening",
        "legal_move_count": len(legal), "best_move_legal": best in legal if best else None,
        "selected_move_legal": proven in legal if proven else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--engine", type=Path, default=None)
    ap.add_argument("--net", type=Path, default=None)
    ap.add_argument("--nodes", type=int, default=50000)
    ap.add_argument("--alt-cap", type=int, default=8)
    ap.add_argument("--max-lineages", type=int, default=16)
    args = ap.parse_args()
    rows = [json.loads(x) for x in args.labels.read_text(encoding="utf-8").splitlines() if x.strip()]
    out = []
    seen: set[str] = set()
    for row in rows:
        lineage = str(row.get("lineage_id") or row.get("proof_lineage_id") or row.get("packed_state_hex"))
        if lineage in seen or len(out) >= args.max_lineages:
            continue
        seen.add(lineage)
        built = build_row(row, engine=args.engine, net=args.net, nodes=args.nodes, alt_cap=args.alt_cap)
        if built:
            out.append(built)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(x, sort_keys=True) for x in out) + ("\n" if out else ""), encoding="utf-8")
    print(json.dumps({"rows": len(out), "output": str(args.out), "proxy": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
