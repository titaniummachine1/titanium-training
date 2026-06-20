#!/usr/bin/env python3
"""Train a sidecar leaf search-pressure head from shallow-vs-deep labels.

The head reuses frozen HalfPW feature construction. The JSONL contains only
move prefixes and labels; this script regenerates features through eval-batch
at training time so the sidecar does not balloon into stored position tensors.
It is intentionally not exported into the Rust engine yet; use it to prove that
a leaf-local scalar can predict whether a child is already searched enough or
deserves more budget than the engine would normally spend.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent

from titanium_training.training.trainer import (  # noqa: E402
    HalfPW,
    NET_H,
    QuoridorDataset,
    ROUTE_CONTESTED,
    ROUTE_ME,
    ROUTE_NEAR_ME,
    ROUTE_NEAR_OPP,
    ROUTE_OPP,
    WEIGHTS,
)
from tools.datagen.datagen import eval_batch  # noqa: E402
from titanium_training.store.move_codec import unpack_moves  # noqa: E402


def row_moves(row: dict) -> list[str]:
    if row.get("moves_bin"):
        return unpack_moves(base64.b64decode(row["moves_bin"]))
    return list(row.get("moves", []))


def row_target(row: dict) -> float:
    target = row.get("search_pressure")
    if target is None:
        target = row.get("depth_scalar")
    if target is None:
        target = 2.0 * float(row["importance"]) - 1.0
    return float(target)


def row_source(row: dict) -> str:
    teacher = str(row.get("teacher") or "titanium-native")
    return "zero" if "zero" in teacher else "native"


def row_is_trainable(row: dict) -> bool:
    """Exclude native positions where mate/race/terminal overrides own search."""
    if row_source(row) != "native":
        return True
    shallow = row.get("shallow") or {}
    deep = row.get("deep") or {}
    if shallow.get("best") in (None, "(none)") or deep.get("best") in (None, "(none)"):
        return False
    return abs(int(shallow.get("score", 0))) < 31_000 and abs(int(deep.get("score", 0))) < 31_000


def grouped_split(rows: list[dict], seed: int, val_fraction: float = 0.10) -> tuple[list[dict], list[dict]]:
    """Split whole games with a stable hash, stratified by teacher family."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = str(row.get("source_game_key") or row.get("moves_bin") or "")
        grouped[key].append(row)
    by_source_mix: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for key, batch in grouped.items():
        by_source_mix[tuple(sorted({row_source(row) for row in batch}))].append(key)
    val_keys: set[str] = set()
    for keys in by_source_mix.values():
        scored = sorted(
            (
                int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big"),
                key,
            )
            for key in keys
        )
        cutoff = int(val_fraction * (1 << 64))
        selected = [key for score, key in scored if score < cutoff]
        if not selected and len(scored) > 1:
            selected = [scored[0][1]]
        val_keys.update(selected)
    train = [row for key, batch in grouped.items() if key not in val_keys for row in batch]
    val = [row for key, batch in grouped.items() if key in val_keys for row in batch]
    return train, val


class ImportanceDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows
        features = []
        records = eval_batch([row_moves(r) for r in rows])
        if len(records) != len(rows):
            raise RuntimeError(f"eval-batch returned {len(records)} rows for {len(rows)} positions")
        for rec, row in zip(records, rows):
            rec["outcome"] = row.get("outcome", 1)
            features.append(rec)
        self.base = QuoridorDataset(features)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        item = self.base[idx]
        item["search_pressure"] = torch.tensor(row_target(self.rows[idx]), dtype=torch.float32)
        return item


class ImportanceHead(nn.Module):
    ROUTE_KEYS = (ROUTE_ME, ROUTE_OPP, ROUTE_NEAR_ME, ROUTE_NEAR_OPP, ROUTE_CONTESTED)
    RICH_SCALARS = 8
    RICH_ROUTE_STATS = len(ROUTE_KEYS) * 3

    def __init__(self, weights_path: Path, feature_set: str):
        super().__init__()
        self.trunk = HalfPW(weights_path)
        self.feature_set = feature_set
        for p in self.trunk.parameters():
            p.requires_grad = False
        # One tiny leaf-local actuator head: this is not a policy over moves,
        # only a trust/budget signal for the child node already reached.
        if feature_set == "hidden32":
            in_features = NET_H
        elif feature_set == "rich":
            in_features = NET_H + self.RICH_SCALARS + self.RICH_ROUTE_STATS
        else:
            in_features = NET_H + self.RICH_SCALARS + len(self.ROUTE_KEYS) * 81
        self.head = nn.Linear(in_features, 1)

    def pressure_features(self, batch):
        hid = self.trunk.hidden_features(batch)
        if self.feature_set == "hidden32":
            return hid
        d_me = batch["d_me"].float()
        d_opp = batch["d_opp"].float()
        w_me = batch["w_me"].float()
        w_opp = batch["w_opp"].float()
        scalars = torch.stack((
            d_me / 16.0,
            d_opp / 16.0,
            w_me / 10.0,
            w_opp / 10.0,
            batch["legal_wall_norm"].float(),
            batch["width_opp"].float() / 9.0,
            (d_opp - d_me) / 16.0,
            (w_me - w_opp) / 10.0,
        ), dim=1)
        if self.feature_set == "routefull":
            return torch.cat((hid, scalars, *(batch[key].float() for key in self.ROUTE_KEYS)), dim=1)
        route_stats = []
        for key in self.ROUTE_KEYS:
            route = batch[key].float()
            route_stats.extend((
                route.mean(dim=1),
                route.amax(dim=1),
                (route > 0).float().mean(dim=1),
            ))
        return torch.cat((hid, scalars, torch.stack(route_stats, dim=1)), dim=1)

    def forward(self, batch):
        with torch.no_grad():
            features = self.pressure_features(batch)
        return torch.tanh(self.head(features)).squeeze(-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", action="append", default=None,
                    help="JSONL input; repeat to combine native and zero labels")
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--out", default="training/checkpoints/search_pressure_head.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--min-rows", type=int, default=200)
    ap.add_argument("--min-val-games", type=int, default=2)
    ap.add_argument("--features", choices=("hidden32", "rich", "routefull"), default="rich")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--allow-zero", action="store_true",
                    help="Explicitly allow zero labels after an external correlation study")
    args = ap.parse_args()

    data_paths = args.data or ["training/data/exports/search_pressure_export.jsonl"]
    for data_path in data_paths:
        p = Path(data_path)
        if str(p).replace("\\", "/") == "training/data/search_pressure.jsonl":
            raise SystemExit(
                "train_search_importance blocked: legacy search_pressure.jsonl.\n"
                "Export from canonical store or pass --data with an export path.\n"
                "See training/CANONICAL_DATASTORE.md"
            )
    rows = []
    for data_path in data_paths:
        rows.extend(
            json.loads(line)
            for line in Path(data_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    raw_rows = len(rows)
    rows = [row for row in rows if row_is_trainable(row)]
    if len(rows) != raw_rows:
        print(f"filtered {raw_rows - len(rows)} terminal/forced-result rows; {len(rows)} remain")
    if any(row_source(row) == "zero" for row in rows) and not args.allow_zero:
        print(
            "zero labels are disabled for this pressure head: run same-position correlation first, "
            "then pass --allow-zero only if the targets agree"
        )
        return 1
    if len(rows) < args.min_rows:
        print(f"need at least {args.min_rows} rows, got {len(rows)}")
        return 1
    train_rows, val_rows = grouped_split(rows, args.seed)
    if not val_rows:
        print("need labels from at least two source games for grouped validation")
        return 1
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    weights_path = Path(args.weights)
    weights_sha256 = hashlib.sha256(weights_path.read_bytes()).hexdigest()

    targets = [row_target(row) for row in rows]
    train_mean = sum(row_target(row) for row in train_rows) / max(1, len(train_rows))
    val_targets = [row_target(row) for row in val_rows]
    baseline_val = (
        sum((t - train_mean) ** 2 for t in val_targets) / len(val_targets)
        if val_targets
        else float("inf")
    )
    print(
        f"rows={len(rows)} train={len(train_rows)} val={len(val_rows)} "
        f"target_mean={sum(targets)/len(targets):+.3f} "
        f"target_min={min(targets):+.3f} target_max={max(targets):+.3f} "
        f"const_baseline_val={baseline_val:.5f}"
    )

    train_means = {}
    for source in {row_source(r) for r in train_rows}:
        source_targets = [row_target(r) for r in train_rows if row_source(r) == source]
        train_means[source] = sum(source_targets) / len(source_targets)
    source_baselines = {}
    for source in {row_source(r) for r in val_rows}:
        source_targets = [row_target(r) for r in val_rows if row_source(r) == source]
        mean = train_means.get(source, train_mean)
        source_baselines[source] = sum((v - mean) ** 2 for v in source_targets) / len(source_targets)

    model = ImportanceHead(weights_path, args.features).to(device)
    opt = torch.optim.Adam(model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_dl = DataLoader(ImportanceDataset(train_rows), batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(ImportanceDataset(val_rows), batch_size=args.batch)

    def to_device(batch):
        return {k: v.to(device) for k, v in batch.items()}

    def run_val(details: bool = False):
        model.eval()
        total, n = 0.0, 0
        predictions = []
        with torch.no_grad():
            for batch in val_dl:
                batch = to_device(batch)
                pred = model(batch)
                loss = F.mse_loss(pred, batch["search_pressure"])
                total += loss.item() * len(pred)
                n += len(pred)
                if details:
                    predictions.extend(pred.detach().cpu().tolist())
        model.train()
        return (total / max(1, n), predictions) if details else total / max(1, n)

    best = float("inf")
    best_state = None
    for ep in range(args.epochs):
        total, n = 0.0, 0
        model.train()
        for batch in train_dl:
            batch = to_device(batch)
            opt.zero_grad()
            pred = model(batch)
            loss = F.mse_loss(pred, batch["search_pressure"])
            loss.backward()
            opt.step()
            total += loss.item() * len(pred)
            n += len(pred)
        val = run_val()
        rel = "beats" if val < baseline_val else "worse_than"
        print(f"epoch {ep+1}/{args.epochs} train={total/max(1,n):.5f} val={val:.5f} {rel}_baseline={baseline_val:.5f}")
        if val < best:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "kind": "leaf_search_pressure_sidecar_v1",
                "head": best_state,
                "val": best,
                "validated": False,
                "base_weights_sha256": weights_sha256,
                "feature_set": args.features,
            }, out)
    model.head.load_state_dict(best_state)
    _, predictions = run_val(details=True)
    per_source = {}
    for source in source_baselines:
        pairs = [
            (pred, row_target(row))
            for pred, row in zip(predictions, val_rows)
            if row_source(row) == source
        ]
        mse = sum((pred - target) ** 2 for pred, target in pairs) / len(pairs)
        per_source[source] = mse
        print(f"holdout {source}: mse={mse:.5f} baseline={source_baselines[source]:.5f}")
    high_recall_by_source = {}
    for source in {row_source(row) for row in val_rows}:
        indices = [i for i, row in enumerate(val_rows) if row_source(row) == source]
        targets_sorted = sorted(row_target(val_rows[i]) for i in indices)
        threshold = targets_sorted[max(0, int(0.75 * (len(targets_sorted) - 1)))]
        k = max(1, len(indices) // 4)
        predicted_top = sorted(indices, key=lambda i: predictions[i], reverse=True)[:k]
        high_recall_by_source[source] = (
            sum(row_target(val_rows[i]) >= threshold for i in predicted_top) / k
        )
    high_recall = high_recall_by_source.get("native", 0.0)
    sources_pass = all(per_source[s] < source_baselines[s] for s in per_source)
    val_games_by_source = {
        source: len({
            str(row.get("source_game_key") or row.get("moves_bin") or "")
            for row in val_rows if row_source(row) == source
        })
        for source in {row_source(row) for row in val_rows}
    }
    required_sources = {"native", "zero"}
    holdouts_ready = required_sources.issubset(val_games_by_source) and all(
        val_games_by_source[source] >= args.min_val_games for source in required_sources
    )
    native_validated = (
        val_games_by_source.get("native", 0) >= args.min_val_games
        and per_source.get("native", float("inf")) < source_baselines.get("native", 0.0)
        and high_recall_by_source.get("native", 0.0) > 0.25
    )
    validated = (
        best < baseline_val
        and sources_pass
        and holdouts_ready
        and all(high_recall_by_source.get(source, 0.0) > 0.25 for source in required_sources)
    )
    torch.save({
        "kind": "leaf_search_pressure_sidecar_v1",
        "head": best_state,
        "val": best,
        "validated": validated,
        "native_validated": native_validated,
        "holdout_mse": per_source,
        "holdout_baseline": source_baselines,
        "high_pressure_recall_at_quartile": high_recall,
        "high_pressure_recall_by_source": high_recall_by_source,
        "base_weights_sha256": weights_sha256,
        "val_games_by_source": val_games_by_source,
        "required_holdouts_ready": holdouts_ready,
        "feature_set": args.features,
    }, Path(args.out))
    print(
        f"best val={best:.5f} high_pressure_recall@quartile={high_recall:.1%} "
        f"holdouts={val_games_by_source} native_validated={native_validated} "
        f"required_holdouts_ready={holdouts_ready} "
        f"validated={validated} -> {args.out}"
    )
    return 0 if validated else 2


if __name__ == "__main__":
    raise SystemExit(main())
