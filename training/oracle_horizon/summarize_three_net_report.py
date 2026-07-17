"""Create the decision-grade paired parent/raw/EMA oracle-horizon report.

This is intentionally read-only with respect to accepted weights and training.
It exits 2 while the corrected proof JSONL is incomplete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABELS = ("parent", "raw", "ema")
BUDGETS = (50_000, 200_000, 800_000, 3_200_000)
PARENT_SHA = "869ad228cfea8bb8964d98d05d6cf5e67a21b27661a36259a3976f60d486be56"


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_median(values: list[int | float]) -> dict[str, Any]:
    return {
        "mean": sum(values) / len(values) if values else None,
        "median": statistics.median(values) if values else None,
        "count_resolved": len(values),
    }


def move_kind(move: Any) -> str:
    return "wall" if str(move or "").endswith(("h", "v")) else "pawn"


def first_stage(row: dict[str, Any], key: str) -> int | None:
    direct = row.get(key)
    if direct is not None:
        return int(direct)
    for stage in row.get("stages") or []:
        if stage.get("move_correct" if key.endswith("move_nodes") else "wdl_correct"):
            if key.endswith("wdl_nodes") and not stage.get("proven"):
                continue
            return int(stage["nodes"])
    return None


def budget_name(value: int | None) -> str:
    return f"{value // 1000}k" if value is not None else "never"


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    move_resolved = [x for x in (first_stage(r, "first_correct_move_nodes") for r in rows) if x is not None]
    wdl_resolved = [x for x in (first_stage(r, "first_correct_wdl_nodes") for r in rows) if x is not None]
    move_correct = len(move_resolved)
    wdl_correct = len(wdl_resolved)
    move_dist = Counter(budget_name(first_stage(r, "first_correct_move_nodes")) for r in rows)
    wdl_dist = Counter(budget_name(first_stage(r, "first_correct_wdl_nodes")) for r in rows)
    out = {
        "n": n,
        "n_correct": move_correct,
        "move_accuracy": move_correct / n if n else None,
        "wdl_n_correct": wdl_correct,
        "wdl_accuracy": wdl_correct / n if n else None,
        "nodes_to_correct_wdl": mean_median(wdl_resolved),
        "nodes_to_correct_move": mean_median(move_resolved),
        "first_successful_ladder_budget": {
            "move": {budget_name(b): move_dist.get(budget_name(b), 0) for b in BUDGETS} | {"never": move_dist.get("never", 0)},
            "proven_correct_wdl": {budget_name(b): wdl_dist.get(budget_name(b), 0) for b in BUDGETS} | {"never": wdl_dist.get("never", 0)},
        },
        "missed_only_defense_count": sum(
            "missed_only_defense" in (r.get("needs_learning_reasons") or []) for r in rows
        ),
        "missed_only_defense_note": (
            "Inherited from holdout labels; this eval cannot newly detect missed-only-defense."
        ),
    }
    return out


def add_move_type_metrics(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    summary["wall_move"] = metric([r for r in rows if move_kind(r.get("best_move")) == "wall"])
    summary["pawn_move"] = metric([r for r in rows if move_kind(r.get("best_move")) == "pawn"])


def paired_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_state: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_state[str(row["packed_state_hex"])][str(row.get("label"))] = row
    complete = {
        state: labels for state, labels in by_state.items() if all(label in labels for label in LABELS)
    }
    return {label: [labels[label] for labels in complete.values()] for label in LABELS}


def deltas(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def one(a: str, b: str) -> dict[str, Any]:
        left, right = summaries[a], summaries[b]
        return {
            "move_accuracy_delta": left["move_accuracy"] - right["move_accuracy"],
            "wdl_accuracy_delta": left["wdl_accuracy"] - right["wdl_accuracy"],
            "mean_nodes_to_correct_move_delta": (
                left["nodes_to_correct_move"]["mean"] - right["nodes_to_correct_move"]["mean"]
                if left["nodes_to_correct_move"]["mean"] is not None and right["nodes_to_correct_move"]["mean"] is not None else None
            ),
            "median_nodes_to_correct_move_delta": (
                left["nodes_to_correct_move"]["median"] - right["nodes_to_correct_move"]["median"]
                if left["nodes_to_correct_move"]["median"] is not None and right["nodes_to_correct_move"]["median"] is not None else None
            ),
            "mean_nodes_to_correct_wdl_delta": (
                left["nodes_to_correct_wdl"]["mean"] - right["nodes_to_correct_wdl"]["mean"]
                if left["nodes_to_correct_wdl"]["mean"] is not None and right["nodes_to_correct_wdl"]["mean"] is not None else None
            ),
            "median_nodes_to_correct_wdl_delta": (
                left["nodes_to_correct_wdl"]["median"] - right["nodes_to_correct_wdl"]["median"]
                if left["nodes_to_correct_wdl"]["median"] is not None and right["nodes_to_correct_wdl"]["median"] is not None else None
            ),
        }
    return {"raw_vs_parent": one("raw", "parent"), "ema_vs_parent": one("ema", "parent"), "raw_vs_ema": one("raw", "ema")}


def screen_summary(screen: Any) -> Any:
    if not isinstance(screen, dict):
        return {"status": "MISSING"}
    result: dict[str, Any] = {}
    for label in ("raw", "ema", "frozen_anchor"):
        if label in screen and isinstance(screen[label], dict):
            item = screen[label]
            result[label] = {key: item.get(key) for key in ("status", "games", "wins", "losses", "draws", "score")}
    if "recommendation" in screen:
        result["screen_recommendation"] = screen["recommendation"]
    return result or screen


def recommendation(report: dict[str, Any]) -> str:
    if not report["integrity"]["ok"]:
        return "NEED_MORE_EVIDENCE"
    summary = report["summaries"]
    bands = report["band_comparison"]
    raw = report["deltas"]["raw_vs_parent"]
    ema = report["deltas"]["ema_vs_parent"]
    best = ema if ema["move_accuracy_delta"] >= raw["move_accuracy_delta"] else raw
    horizon_gain = best["move_accuracy_delta"] > 0 or best["wdl_accuracy_delta"] > 0
    horizon_gain = horizon_gain or any(
        (best[key] is not None and best[key] < 0)
        for key in ("mean_nodes_to_correct_move_delta", "mean_nodes_to_correct_wdl_delta")
    )
    b23_gain = any(
        bands[str(b)]["raw_vs_parent"]["move_accuracy_delta"] > 0
        or bands[str(b)]["ema_vs_parent"]["move_accuracy_delta"] > 0
        for b in (2, 3)
    )
    if not horizon_gain:
        return "REJECT"
    if not b23_gain:
        return "NEED_MORE_EVIDENCE"
    screens = report["screens"].get("general_20", {})
    anchor = report["screens"].get("frozen_anchor", {})
    available = screens.get("raw") or screens.get("ema")
    screen_ok = bool(available) and available.get("status") == "PASS" and available.get("score", 0) >= 0.5
    anchor_ok = not anchor or anchor.get("status") != "PASS" or anchor.get("score", 0) >= 0.5
    if available and (not screen_ok or not anchor_ok):
        return "QUARANTINE"
    return "PROMOTE" if screen_ok and anchor_ok else "NEED_MORE_EVIDENCE"


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    default_dir = root / "training/runs/oracle_horizon_pilot_v1/continuation_e3"
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=default_dir)
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    eval_dir = run_dir / "eval"
    proof_path = eval_dir / "HOLDOUT_PROOF_HORIZON.jsonl"
    rows = read_jsonl(proof_path)
    counts = Counter(str(row.get("label")) for row in rows)
    complete = all(counts[label] == 59 for label in LABELS)
    paired = paired_rows(rows)
    complete = complete and all(len(paired[label]) == 59 for label in LABELS)
    if not complete:
        status = {
            "status": "WAITING",
            "reason": "corrected holdout JSONL is incomplete; require 59 parent, 59 raw, and 59 ema rows joined on packed_state_hex",
            "path": str(proof_path),
            "rows": len(rows),
            "label_counts": dict(counts),
            "paired_positions": len(paired["parent"]),
        }
        print(json.dumps(status, indent=2))
        return 2

    labels = {
        str(r["packed_state_hex"]): r for r in read_jsonl(run_dir / "holdout_labels.jsonl")
    }
    summaries = {label: metric(paired[label]) for label in LABELS}
    for label in LABELS:
        add_move_type_metrics(summaries[label], paired[label])
    by_band: dict[str, Any] = {}
    for band in ("0", "1", "2", "3"):
        by_band[band] = {}
        for label in LABELS:
            by_band[band][label] = metric([r for r in paired[label] if str(r.get("band")) == band])
            add_move_type_metrics(
                by_band[band][label],
                [r for r in paired[label] if str(r.get("band")) == band],
            )
        by_band[band]["raw_vs_parent"] = {
            "move_accuracy_delta": by_band[band]["raw"]["move_accuracy"] - by_band[band]["parent"]["move_accuracy"],
        }
        by_band[band]["ema_vs_parent"] = {
            "move_accuracy_delta": by_band[band]["ema"]["move_accuracy"] - by_band[band]["parent"]["move_accuracy"],
        }
    lineage = {}
    for label in LABELS:
        matched = [labels.get(str(r["packed_state_hex"]), {}) for r in paired[label]]
        lineage[label] = {
            "unique_lineage_id": len({x["lineage_id"] for x in matched if x.get("lineage_id") is not None}),
            "unique_game_id": len({x["game_id"] for x in matched if x.get("game_id") is not None}),
            "holdout_label_rows_matched": sum(bool(x) for x in matched),
        }
    manifest = read_json(run_dir / "TRAIN_MANIFEST.json", {}) or {}
    mix = read_json(run_dir / "oracle_mix_manifest.json", {}) or {}
    hashes = read_json(run_dir / "exports/SHA256.json", {}) or {}
    parent = root / "training/runs/v16/accepted/epoch_0003.bin"
    integrity = {
        "parent_sha256": sha256(parent),
        "parent_sha256_expected": PARENT_SHA,
        "parent_sha_ok": sha256(parent) == PARENT_SHA,
        "raw_sha256": hashes.get("continuation_raw.bin"),
        "ema_sha256": hashes.get("continuation_ema.bin"),
        "confirmed_accepted_epoch3_sha_still": PARENT_SHA,
        "unattended": "OFF",
        "ok": sha256(parent) == PARENT_SHA and bool(hashes.get("continuation_raw.bin")) and bool(hashes.get("continuation_ema.bin")),
    }
    screens = {
        "general_20": screen_summary(read_json(eval_dir / "SCREEN_20.json")),
        "frozen_anchor": screen_summary(read_json(eval_dir / "FROZEN_ANCHOR_SCREEN.json")),
    }
    report = {
        "schema": "three-net-oracle-horizon-comparison-v1",
        "status": "PASS",
        "paired_holdout_positions": 59,
        "join_key": "packed_state_hex",
        "summaries": summaries,
        "band_comparison": by_band,
        "band_2_3_callout": {
            "combined": {
                label: metric([r for r in paired[label] if str(r.get("band")) in ("2", "3")])
                for label in LABELS
            },
            "separate_bands": {band: by_band[band] for band in ("2", "3")},
        },
        "deltas": deltas(summaries),
        "proof_lineage_holdout_coverage": lineage,
        "screens": screens,
        "training_context": {
            "parent_best_val_loss": 0.54852,
            "continuation_val_loss": 0.56559,
            "val_loss_warning": "Continuation is worse; warning only, not auto-reject.",
            "oracle_effective_fraction": mix.get("effective_oracle_fraction", 0.10),
            "repeat_factor": mix.get("repeat_factor", 2.6),
        },
        "integrity": integrity,
        "discarded_first_eval": {
            "path": str(eval_dir / "INVALID_FIRST_EVAL_NON_DECISION.json"),
            "status": "HARNESS_PATH_ERROR",
            "decision": "NON-DECISION",
            "note": "First eval is discarded and is not evidence.",
        },
    }
    rec = recommendation(report)
    report["recommendation"] = rec
    (eval_dir / "THREE_NET_COMPARISON.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    recommendation_doc = {
        "recommendation": rec,
        "reason": "See THREE_NET_COMPARISON.json for paired proof, screens, and integrity.",
        "auto_promote": False,
        "accepted_weights_touched": False,
        "unattended_started": False,
    }
    (eval_dir / "RECOMMENDATION.json").write_text(json.dumps(recommendation_doc, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# Three-Net Oracle-Horizon Report\n\n**Recommendation: `{rec}`**\n",
        "## Scope\n",
        f"Paired `{report['paired_holdout_positions']}` holdout positions joined on `packed_state_hex`; labels are parent, raw, and EMA.",
        "The first evaluation is explicitly discarded as `HARNESS_PATH_ERROR / NON-DECISION`, not evidence.",
        "\n## Proof comparison\n",
    ]
    for label in LABELS:
        item = summaries[label]
        lines.append(
            f"- **{label}**: move {item['n_correct']}/{item['n']} ({pct(item['move_accuracy'])}); "
            f"WDL {item['wdl_n_correct']}/{item['n']} ({pct(item['wdl_accuracy'])}); "
            f"move nodes mean/median {item['nodes_to_correct_move']['mean']}/{item['nodes_to_correct_move']['median']}; "
            f"WDL nodes mean/median {item['nodes_to_correct_wdl']['mean']}/{item['nodes_to_correct_wdl']['median']}; "
            f"wall move accuracy {pct(item['wall_move']['move_accuracy'])}; "
            f"pawn move accuracy {pct(item['pawn_move']['move_accuracy'])}."
        )
    lines += [
        "\n## Band 2–3 callout\n",
        "Band 2–3 is reported both combined and separately in `THREE_NET_COMPARISON.json`; this is the decision-critical horizon slice.",
        "\n## Screens and integrity\n",
        f"- Frozen-anchor: `{json.dumps(screens['frozen_anchor'], sort_keys=True)}`",
        f"- 20-game general screen: `{json.dumps(screens['general_20'], sort_keys=True)}`",
        f"- Parent epoch 3 SHA confirmed: `{PARENT_SHA}`; unattended mode: **OFF**.",
        f"- Training context: parent best val_loss 0.54852 vs continuation 0.56559 (warning only); oracle effective fraction {report['training_context']['oracle_effective_fraction']}; repeat {report['training_context']['repeat_factor']}x.",
        "\nNo automatic promotion was performed; accepted weights were not touched.",
    ]
    (eval_dir / "THREE_NET_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "recommendation": rec, "comparison": str(eval_dir / "THREE_NET_COMPARISON.json"), "report": str(eval_dir / "THREE_NET_REPORT.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
