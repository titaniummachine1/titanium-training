"""Fine-tune HalfPW NNUE weights from self-play game outcomes.

Uses sigmoid cross-entropy (WDL loss): outcome +1/-1 is mapped to a target
win probability, and the net's centipawn eval is passed through sigmoid to
get a predicted probability.  Trains ALL weights (ws, b1, w2, w1c, po, px)
starting from the current net_weights.bin.

Checkpoints are saved every --checkpoint-steps steps and on every best-val-loss.
Resume is allowed only from checkpoints stamped with the current feature schema.

Usage:
    python training/nnue_cli.py train --data training/data/games.jsonl
    python training/nnue_cli.py train --data training/data/games.jsonl --resume  # auto-finds latest ckpt
    python training/nnue_cli.py train --data training/data/games.jsonl --resume --ckpt path/to/ckpt.pt

Options:
    --data PATH          JSONL file from datagen.py
    --weights PATH       Starting weights (default: engine/src/titanium/net_weights.bin)
    --out-dir DIR        Checkpoint + output directory (default: training/checkpoints)
    --epochs N           Number of passes over data (default: 20)
    --batch N            Batch size (default: 512)
    --lr LR              Learning rate (default: 1e-3)
    --scale S            Sigmoid temperature in cp (default: 400)
    --checkpoint-steps N Save every N steps (default: 1000)
    --val-split F        Fraction of data held out for validation (default: 0.05)
    --resume             Resume from latest checkpoint in --out-dir
    --ckpt PATH          Resume from specific checkpoint file
    --cpu                Force CPU even if CUDA is available
"""

import sys
from pathlib import Path

_TRAINING_ROOT = Path(__file__).resolve().parents[2]
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

import argparse
import json
import math
import random
import sqlite3
import struct
import time
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from titanium_training.models.field_planes import (
    CAT_HEAT,
    CHOKE_P0,
    CHOKE_P1,
    CONTESTED,
    CORRIDOR_DELTA_P0,
    CORRIDOR_DELTA_P1,
    encode_contested,
    FIELD_PLANE_COUNT,
    GOAL_INV_P0,
    GOAL_INV_P1,
    PATH_CROSS_P0,
    PATH_CROSS_P1,
    PAWN_FWD_P0,
    PAWN_FWD_P1,
    ROUTE_CONTESTED,
    ROUTE_ME,
    ROUTE_NEAR_ME,
    ROUTE_NEAR_OPP,
    ROUTE_OPP,
    compact_cat_heat_vector,
    compact_route_vectors,
    rec_field,
)
from titanium_training.validation.engine_identity import assert_engine_ready

# ── constants matching halfpw.py / net.rs ────────────────────────────────────

# NET_H is per-blob now, not a fixed constant: every net_weights.bin-format
# file starts with an 8-byte little-endian NET_H header (see
# engine/src/titanium/net.rs read_h_header / training/tools/net2net_widen.py),
# read once when a HalfPW is constructed. This is what lets a differently
# -widened net (e.g. the net2net_widen.py Net2Net experiment) load and train
# with zero source edits, matching the engine side.
H_HEADER_LEN = 8
WSKIP_LEN = 20
FIELD_SHAPE = (81,)


def _payload_f64s(h: int) -> int:
    return WSKIP_LEN + h + h + 9 * 128 * h + 81 * h + 81 * h + math.prod(FIELD_SHAPE) * FIELD_PLANE_COUNT


def read_net_h(weights_path) -> int:
    with open(weights_path, "rb") as f:
        header = f.read(H_HEADER_LEN)
    (h,) = struct.unpack("<Q", header)
    return h


# Backward-compat default (only used by the couple of standalone/manual tools
# that still assume a fixed 32-wide legacy blob, e.g. train_linear_only.py).
NET_H = 32
W1C_SHAPE = (9, 128, NET_H)
PO_SHAPE = (81, NET_H)
PX_SHAPE = (81, NET_H)
NET_WEIGHT_F64S = _payload_f64s(NET_H)

NET_MIRC = [(8 - i // 9) * 9 + i % 9 for i in range(81)]
NET_MIRS = [(7 - i // 8) * 8 + i % 8 for i in range(64)]
NET_BKT  = [(i // 9 // 3) * 3 + (i % 9) // 3 for i in range(81)]

ROOT    = Path(__file__).resolve().parents[3]
WEIGHTS = ROOT / "engine" / "src" / "titanium" / "net_weights.bin"
TRAINING_SCHEMA = "halfpw-sparse-route5-catheat-ws20-cat-v2"

from titanium_training.store.config import GAME_STORE_DB
from titanium_training.store.guards import LegacyTrainingSourceError, assert_canonical_training_db
from titanium_training.store.lib import load_games_for_training, load_games_for_training_ids

# ── model ─────────────────────────────────────────────────────────────────────

class HalfPW(nn.Module):
    """Differentiable HalfPW forward pass, initialised from net_weights.bin."""

    def __init__(self, weights_path):
        super().__init__()
        raw = Path(weights_path).read_bytes()
        (h,) = struct.unpack("<Q", raw[:H_HEADER_LEN])
        data = raw[H_HEADER_LEN:]
        w1c_shape = (9, 128, h)
        po_shape = (81, h)
        px_shape = (81, h)
        legacy_f64s = _payload_f64s(h) - math.prod(FIELD_SHAPE)
        full_f64s = _payload_f64s(h)
        assert len(data) in (legacy_f64s * 8, full_f64s * 8), (
            f"net_weights.bin size {len(data)} for declared NET_H={h}; expected legacy or CAT-heat shape"
        )
        vals  = list(struct.unpack(f"<{len(data) // 8}d", data))
        if len(vals) == legacy_f64s:
            vals.extend([0.0] * math.prod(FIELD_SHAPE))
        o = 0
        def take(n):
            nonlocal o; s = vals[o:o+n]; o += n; return s

        self.h = h
        self.ws  = nn.Parameter(torch.tensor(take(WSKIP_LEN), dtype=torch.float32))
        self.b1  = nn.Parameter(torch.tensor(take(h),     dtype=torch.float32))
        self.w2  = nn.Parameter(torch.tensor(take(h),     dtype=torch.float32))
        self.w1c = nn.Parameter(torch.tensor(take(math.prod(w1c_shape)), dtype=torch.float32).view(*w1c_shape))
        self.po  = nn.Parameter(torch.tensor(take(math.prod(po_shape)),  dtype=torch.float32).view(*po_shape))
        self.px  = nn.Parameter(torch.tensor(take(math.prod(px_shape)),  dtype=torch.float32).view(*px_shape))
        self.route_me = nn.Parameter(torch.tensor(take(math.prod(FIELD_SHAPE)), dtype=torch.float32).view(*FIELD_SHAPE))
        self.route_opp = nn.Parameter(torch.tensor(take(math.prod(FIELD_SHAPE)), dtype=torch.float32).view(*FIELD_SHAPE))
        self.route_near_me = nn.Parameter(torch.tensor(take(math.prod(FIELD_SHAPE)), dtype=torch.float32).view(*FIELD_SHAPE))
        self.route_near_opp = nn.Parameter(torch.tensor(take(math.prod(FIELD_SHAPE)), dtype=torch.float32).view(*FIELD_SHAPE))
        self.route_contested = nn.Parameter(torch.tensor(take(math.prod(FIELD_SHAPE)), dtype=torch.float32).view(*FIELD_SHAPE))
        self.cat_heat = nn.Parameter(torch.tensor(take(math.prod(FIELD_SHAPE)), dtype=torch.float32).view(*FIELD_SHAPE))

    def hidden_features(self, b):
        """Frozen leaf representation before value projection; useful for sidecar heads."""
        bucket     = b["bucket"]
        wall_mask  = b["wall_mask"].float()
        pawn_me    = b["pawn_me"]
        pawn_opp   = b["pawn_opp"]

        w1c_sel = self.w1c[bucket]
        acc     = (w1c_sel * wall_mask.unsqueeze(-1)).sum(dim=1)
        hid     = self.b1 + acc + self.po[pawn_me] + self.px[pawn_opp]
        return hid.clamp(0.0, 1.0)

    def forward_trace(self, b):
        """Batched forward with intermediates — raw scalar path matches ``search.rs``."""
        from titanium_training.models.eval_forward import EvalTrace

        ws = self.ws
        d_me_raw = b["d_me"].float()
        d_opp_raw = b["d_opp"].float()
        w_me_raw = b["w_me"].float()
        w_opp_raw = b["w_opp"].float()
        width_opp = b["width_opp"].float()
        pd = d_opp_raw - d_me_raw
        wd = w_me_raw - w_opp_raw

        out = (
            ws[0]
            + ws[1] * pd
            + ws[2] * wd
            + ws[3] * d_me_raw
            + ws[4] * d_opp_raw
            + ws[9] * pd * (w_me_raw + w_opp_raw) / 20.0
            + ws[10] * wd * (d_me_raw + d_opp_raw) / 16.0
        )
        w_opp_zero = w_opp_raw == 0.0
        w_me_zero = w_me_raw == 0.0
        # search.rs uses if/elif — opponent-zero branch wins when both are zero.
        out = out + ws[6] * w_opp_zero.float()
        out = out + ws[5] * (w_opp_zero & (d_me_raw <= d_opp_raw)).float()
        out = out + ws[8] * (w_me_zero & ~w_opp_zero).float()
        out = out + ws[7] * (w_me_zero & ~w_opp_zero & (d_opp_raw <= d_me_raw - 1.0)).float()
        three = torch.tensor(3.0, device=d_me_raw.device, dtype=d_me_raw.dtype)
        out = out + ws[11] * torch.minimum(w_me_raw, three) * (d_opp_raw <= 4.0).float()
        out = out + ws[12] * torch.minimum(w_opp_raw, three) * (d_me_raw <= 4.0).float()
        out = out + ws[13] * pd * w_opp_raw / 10.0
        scalar_out = out

        route_out = (
            (b[ROUTE_ME] * self.route_me).sum(dim=1)
            + (b[ROUTE_OPP] * self.route_opp).sum(dim=1)
            + (b[ROUTE_NEAR_ME] * self.route_near_me).sum(dim=1)
            + (b[ROUTE_NEAR_OPP] * self.route_near_opp).sum(dim=1)
            + (b[ROUTE_CONTESTED] * self.route_contested).sum(dim=1)
        )
        # See forward() -- no cat_active gate; would permanently block gradient.
        cat_out = (b[CAT_HEAT] * self.cat_heat).sum(dim=1)
        width_contrib = ws[15] * width_opp

        bucket = b["bucket"]
        wall_mask = b["wall_mask"].float()
        pawn_me = b["pawn_me"]
        pawn_opp = b["pawn_opp"]
        w1c_sel = self.w1c[bucket]
        acc = (w1c_sel * wall_mask.unsqueeze(-1)).sum(dim=1)
        hid = self.b1.unsqueeze(0) + acc + self.po[pawn_me] + self.px[pawn_opp]
        hidden_clip = hid.clamp(0.0, 1.0)
        neural_out = (self.w2.unsqueeze(0) * hidden_clip * 200.0).sum(dim=-1)
        final_cp = (scalar_out + route_out + cat_out + width_contrib + neural_out)
        traces: list[EvalTrace] = []
        batch = int(d_me_raw.shape[0])
        for i in range(batch):
            traces.append(
                EvalTrace(
                    scalar_inputs={
                        "d_me": float(d_me_raw[i]),
                        "d_opp": float(d_opp_raw[i]),
                        "w_me": float(w_me_raw[i]),
                        "w_opp": float(w_opp_raw[i]),
                        "pd": float(pd[i]),
                        "wd": float(wd[i]),
                        "width_opp": float(width_opp[i]),
                    },
                    scalar_out=float(scalar_out[i].detach()),
        route_out=float(route_out[i].detach()),
        cat_out=float(cat_out[i].detach()),
        width_contrib=float(width_contrib[i].detach()),
                    wall_acc=acc[i].detach().cpu().tolist(),
                    hidden_pre=hid[i].detach().cpu().tolist(),
                    hidden_clip=hidden_clip[i].detach().cpu().tolist(),
                    neural_out=float(neural_out[i].detach()),
                    final_cp=int(final_cp[i].item()),
                )
            )
        return traces

    def forward(self, b):
        """
        b: dict of batched tensors (see QuoridorDataset.__getitem__).
        Returns centipawn eval [N] from the side-to-move's perspective.

        Raw scalar inputs match ``search.rs::evaluate()`` and ``halfpw.forward(normed=False)``.
        """
        ws = self.ws
        d_me_raw = b["d_me"].float()
        d_opp_raw = b["d_opp"].float()
        w_me_raw = b["w_me"].float()
        w_opp_raw = b["w_opp"].float()
        pd = d_opp_raw - d_me_raw
        wd = w_me_raw - w_opp_raw

        out = (
            ws[0]
            + ws[1] * pd
            + ws[2] * wd
            + ws[3] * d_me_raw
            + ws[4] * d_opp_raw
            + ws[9] * pd * (w_me_raw + w_opp_raw) / 20.0
            + ws[10] * wd * (d_me_raw + d_opp_raw) / 16.0
        )
        w_opp_zero = w_opp_raw == 0.0
        w_me_zero = w_me_raw == 0.0
        out = out + ws[6] * w_opp_zero.float()
        out = out + ws[5] * (w_opp_zero & (d_me_raw <= d_opp_raw)).float()
        out = out + ws[8] * (w_me_zero & ~w_opp_zero).float()
        out = out + ws[7] * (w_me_zero & ~w_opp_zero & (d_opp_raw <= d_me_raw - 1.0)).float()
        three = torch.tensor(3.0, device=d_me_raw.device, dtype=d_me_raw.dtype)
        out = out + ws[11] * torch.minimum(w_me_raw, three) * (d_opp_raw <= 4.0).float()
        out = out + ws[12] * torch.minimum(w_opp_raw, three) * (d_me_raw <= 4.0).float()
        out = out + ws[13] * pd * w_opp_raw / 10.0
        out = out + ws[15] * b["width_opp"].float()
        out = out + (b[ROUTE_ME] * self.route_me).sum(dim=1)
        out = out + (b[ROUTE_OPP] * self.route_opp).sum(dim=1)
        out = out + (b[ROUTE_NEAR_ME] * self.route_near_me).sum(dim=1)
        out = out + (b[ROUTE_NEAR_OPP] * self.route_near_opp).sum(dim=1)
        out = out + (b[ROUTE_CONTESTED] * self.route_contested).sum(dim=1)
        # No cat_active gate here: gating on the *current* weight value means a
        # zero-initialized cat_heat can never receive gradient (it's excluded
        # from the graph before it ever gets a chance to move) -- a
        # self-perpetuating trap that left this term permanently untrained
        # across the whole checkpoint chain. Multiply-by-zero is identical to
        # the gated version when the weight really is zero, so this changes
        # nothing for an untrained net and unblocks training going forward.
        out = out + (b[CAT_HEAT] * self.cat_heat).sum(dim=1)

        bucket = b["bucket"]
        wall_mask = b["wall_mask"].float()
        pawn_me = b["pawn_me"]
        pawn_opp = b["pawn_opp"]
        w1c_sel = self.w1c[bucket]
        acc = (w1c_sel * wall_mask.unsqueeze(-1)).sum(dim=1)
        hid = self.b1 + acc + self.po[pawn_me] + self.px[pawn_opp]
        hid_act = hid.clamp(0.0, 1.0)
        out = out + (self.w2 * hid_act * 200.0).sum(dim=-1)
        return out

    def save_weights(self, path):
        """Serialize back to the engine's little-endian format: an 8-byte
        NET_H header (matching engine/src/titanium/net.rs) followed by the
        f64 payload."""
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", self.h))
            def w(t):
                vals = t.detach().cpu().double().flatten().tolist()
                f.write(struct.pack(f"<{len(vals)}d", *vals))
            w(self.ws);   w(self.b1);  w(self.w2)
            w(self.w1c);  w(self.po);  w(self.px)
            w(self.route_me); w(self.route_opp)
            w(self.route_near_me); w(self.route_near_opp); w(self.route_contested)
            w(self.cat_heat)
        print(f"  weights saved -> {path}")


# ── dataset ───────────────────────────────────────────────────────────────────

class QuoridorDataset(Dataset):
    def __init__(self, records):
        self.recs = records

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, idx):
        r = self.recs[idx]
        me  = r["turn"]
        opp = 1 - me

        # Side-to-move perspective scalars
        d_me  = r["d0"] if me == 0 else r["d1"]
        d_opp = r["d1"] if me == 0 else r["d0"]
        w_me  = r["wl0"] if me == 0 else r["wl1"]
        w_opp = r["wl1"] if me == 0 else r["wl0"]

        # Opponent corridor width for ws[15] — raw count from engine JSON.
        from titanium_training.models.eval_forward import opponent_corridor_width

        width_opp = opponent_corridor_width(r, me, int(d_opp))

        legal_wall_norm = 0.0
        legal_cross_me_norm = 0.0
        legal_cross_opp_norm = 0.0

        # ws[18]/ws[19] are retired zero inputs. Keep the batch keys for cache
        # compatibility, but do not expose scalar CAT shortcuts to the model.
        cat_best_me_norm = 0.0
        cat_best_opp_norm = 0.0

        route_me, route_opp, route_near_me, route_near_opp, route_contested = compact_route_vectors(r, NET_MIRC)
        cat_heat = compact_cat_heat_vector(r, NET_MIRC)

        # Wall accumulator inputs (mirror when me=1 to share weights)
        hw = r["hw"];  vw = r["vw"]
        if me == 0:
            pawn_me_idx  = r["pawn0"]
            pawn_opp_idx = r["pawn1"]
            bucket       = NET_BKT[r["pawn0"]]
            wall_mask    = [hw[s] for s in range(64)] + [vw[s] for s in range(64)]
        else:
            pawn_me_idx  = NET_MIRC[r["pawn1"]]
            pawn_opp_idx = NET_MIRC[r["pawn0"]]
            bucket       = NET_BKT[pawn_me_idx]
            wall_mask    = ([hw[NET_MIRS[s]] for s in range(64)]
                          + [vw[NET_MIRS[s]] for s in range(64)])

        # Outcome: +1 = P0 wins, -1 = P1 wins.
        # Convert to win-probability from side-to-move's perspective.
        outcome_p0 = float(r["outcome"])
        outcome_stm = outcome_p0 if me == 0 else -outcome_p0
        target = (outcome_stm + 1.0) / 2.0  # +1→1.0  -1→0.0

        return {
            "d_me":      torch.tensor(d_me,       dtype=torch.float32),
            "d_opp":     torch.tensor(d_opp,      dtype=torch.float32),
            "w_me":      torch.tensor(w_me,        dtype=torch.float32),
            "w_opp":     torch.tensor(w_opp,       dtype=torch.float32),
            "legal_wall_norm":     torch.tensor(legal_wall_norm,     dtype=torch.float32),
            "width_opp":           torch.tensor(width_opp,           dtype=torch.float32),
            "legal_cross_me_norm": torch.tensor(legal_cross_me_norm,  dtype=torch.float32),
            "legal_cross_opp_norm":torch.tensor(legal_cross_opp_norm, dtype=torch.float32),
            "cat_best_me_norm":    torch.tensor(cat_best_me_norm,     dtype=torch.float32),
            "cat_best_opp_norm":   torch.tensor(cat_best_opp_norm,    dtype=torch.float32),
            ROUTE_ME: torch.tensor(route_me, dtype=torch.float32),
            ROUTE_OPP: torch.tensor(route_opp, dtype=torch.float32),
            ROUTE_NEAR_ME: torch.tensor(route_near_me, dtype=torch.float32),
            ROUTE_NEAR_OPP: torch.tensor(route_near_opp, dtype=torch.float32),
            ROUTE_CONTESTED: torch.tensor(route_contested, dtype=torch.float32),
            CAT_HEAT: torch.tensor(cat_heat, dtype=torch.float32),
            "bucket":    torch.tensor(bucket,      dtype=torch.long),
            "wall_mask": torch.tensor(wall_mask,   dtype=torch.float32),
            "pawn_me":   torch.tensor(pawn_me_idx, dtype=torch.long),
            "pawn_opp":  torch.tensor(pawn_opp_idx,dtype=torch.long),
            "target":    torch.tensor(target,      dtype=torch.float32),
        }


# ── cached dataset (full-corpus memmap) ──────────────────────────────────────

class CachedDataset(Dataset):
    """Streams positions from a pre-built feature cache (build_feature_cache.py).

    Loads positions.bin as a numpy memmap (no RAM copy) and reads rows via
    a shuffled index array so every epoch visits all positions exactly once.
    Skips rows retired by position_usage (>=5 epoch touches).
    """
    FV_LEN = 628

    def __init__(self, cache_dir: Path, split: str):
        self.cache_dir = Path(cache_dir)
        meta = json.loads((self.cache_dir / "meta.json").read_text(encoding="utf-8"))
        n_all = meta["n_total"]
        self.data = np.memmap(
            self.cache_dir / "positions.bin",
            dtype="float32",
            mode="r",
            shape=(n_all, self.FV_LEN),
        )
        self.refresh_indices(split)
        self._epoch_indices = None

    def refresh_indices(self, split: str | None = None) -> None:
        if split is None:
            split = getattr(self, "_split", "train")
        self._split = split
        self._epoch_indices = None
        if split == "recent_val":
            p = self.cache_dir / "recent_val_indices.npy"
            if p.is_file():
                self.indices = np.load(p)
                return
        try:
            from position_usage import active_indices

            self.indices = active_indices(self.cache_dir, split)
        except Exception:
            self.indices = np.load(self.cache_dir / f"{split}_indices.npy")

    def set_epoch_indices(self, indices: np.ndarray) -> None:
        self._epoch_indices = np.asarray(indices, dtype=np.int32)

    def clear_epoch_indices(self) -> None:
        self._epoch_indices = None

    def active_row_indices(self) -> np.ndarray:
        if self._epoch_indices is not None:
            return self._epoch_indices
        return self.indices

    def __len__(self):
        return len(self.active_row_indices())

    def __getitem__(self, i):
        row = int(self.active_row_indices()[i])
        fv = np.array(self.data[row])  # copy from memmap
        return {
            "target":               torch.tensor(float(fv[0]),   dtype=torch.float32),
            "d_me":                 torch.tensor(float(fv[1]),   dtype=torch.float32),
            "d_opp":                torch.tensor(float(fv[2]),   dtype=torch.float32),
            "w_me":                 torch.tensor(float(fv[3]),   dtype=torch.float32),
            "w_opp":                torch.tensor(float(fv[4]),   dtype=torch.float32),
            "legal_wall_norm":      torch.tensor(float(fv[5]),   dtype=torch.float32),
            "width_opp":            torch.tensor(float(fv[6]),   dtype=torch.float32),
            "legal_cross_me_norm":  torch.tensor(float(fv[7]),   dtype=torch.float32),
            "legal_cross_opp_norm": torch.tensor(float(fv[8]),   dtype=torch.float32),
            "cat_best_me_norm":     torch.tensor(float(fv[9]),   dtype=torch.float32),
            "cat_best_opp_norm":    torch.tensor(float(fv[10]),  dtype=torch.float32),
            "wall_mask":            torch.from_numpy(fv[11:139].copy()),
            ROUTE_ME:               torch.from_numpy(fv[139:220].copy()),
            ROUTE_OPP:              torch.from_numpy(fv[220:301].copy()),
            ROUTE_NEAR_ME:          torch.from_numpy(fv[301:382].copy()),
            ROUTE_NEAR_OPP:         torch.from_numpy(fv[382:463].copy()),
            ROUTE_CONTESTED:        torch.from_numpy(fv[463:544].copy()),
            CAT_HEAT:               torch.from_numpy(fv[544:625].copy()),
            "bucket":               torch.tensor(int(fv[625]),   dtype=torch.long),
            "pawn_me":              torch.tensor(int(fv[626]),   dtype=torch.long),
            "pawn_opp":             torch.tensor(int(fv[627]),   dtype=torch.long),
        }


# ── training loop ─────────────────────────────────────────────────────────────

def wdl_loss(eval_cp, target, scale, sample_weight=None):
    """Binary cross-entropy between sigmoid(eval/scale) and target win prob."""
    pred = torch.sigmoid(eval_cp / scale)
    per_sample = F.binary_cross_entropy(pred, target, reduction="none")
    if sample_weight is None:
        return per_sample.mean()
    w = sample_weight.float()
    denom = w.sum().clamp_min(1e-8)
    return (per_sample * w).sum() / denom


class OptimizerGroup:
    """Combines multiple optimizers behind the single-optimizer interface the
    rest of the training loop (step/zero_grad/state_dict/save_checkpoint)
    already uses -- so --optimizer muon (Muon on 2D+ matrices + an Adam
    fallback for biases/1D params, which Muon isn't designed for) doesn't
    require touching every call site."""

    def __init__(self, optimizers: list):
        self.optimizers = optimizers

    def step(self):
        for o in self.optimizers:
            o.step()

    def zero_grad(self, set_to_none: bool = True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return [o.state_dict() for o in self.optimizers]

    def load_state_dict(self, states):
        for o, s in zip(self.optimizers, states):
            o.load_state_dict(s)


def build_optimizer(model, *, kind: str, lr: float, weight_decay: float, aux_lr: float | None = None):
    """kind: "adam" (plain Adam, L2 weight decay -- the literal production
    default, unchanged), "adamw" (decoupled weight decay), or "muon" (Muon on
    exactly-2D hidden matrices, AdamW on everything else -- biases, norms,
    and this model's 3D feature-plane embeddings aren't matrices Muon
    supports). aux_lr overrides the fallback AdamW's LR for the muon case
    (Muon's update-norm geometry differs from Adam's -- same numeric LR is
    not a comparable step size, so the two optimizers inside a hybrid
    usually want different LRs, not one shared value)."""
    if kind == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if kind == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if kind == "muon":
        # torch.optim.Muon requires EXACTLY 2D parameters (not "2D+") --
        # anything else (biases, norms, and this model's 3D feature-plane
        # embeddings) falls back to AdamW.
        matrix_params = [p for p in model.parameters() if p.requires_grad and p.dim() == 2]
        other_params = [p for p in model.parameters() if p.requires_grad and p.dim() != 2]
        fallback_lr = aux_lr if aux_lr is not None else lr
        optimizers = []
        if matrix_params:
            optimizers.append(torch.optim.Muon(matrix_params, lr=lr, weight_decay=weight_decay))
        if other_params:
            optimizers.append(torch.optim.AdamW(other_params, lr=fallback_lr, weight_decay=weight_decay))
        return OptimizerGroup(optimizers)
    raise ValueError(f"unknown optimizer kind: {kind}")


def save_checkpoint(path, model, optimizer, step, epoch, best_val):
    torch.save({
        "schema": TRAINING_SCHEMA,
        "step": step, "epoch": epoch, "best_val": best_val,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)


def load_checkpoint(path, model, optimizer, *, weights_path=WEIGHTS):
    ckpt = torch.load(path, weights_only=False)
    schema = ckpt.get("schema")
    if schema != TRAINING_SCHEMA:
        raise RuntimeError(
            f"checkpoint schema {schema!r} != {TRAINING_SCHEMA!r}; "
            "do not resume checkpoints trained before ws20 CAT-best features"
        )
    try:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        return ckpt["step"], ckpt["epoch"], ckpt["best_val"], optimizer
    except RuntimeError as e:
        print(f"WARN: checkpoint incompatible ({e}); re-init from net_weights.bin")
        device = next(model.parameters()).device
        fresh = HalfPW(weights_path).to(device)
        model.load_state_dict(fresh.state_dict())
        lr = optimizer.param_groups[0]["lr"]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        return 0, 0, float("inf"), optimizer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data",             default=str(GAME_STORE_DB))
    ap.add_argument("--weights",          default=str(WEIGHTS))
    ap.add_argument("--out-dir",          default="training/checkpoints")
    ap.add_argument("--epochs",           type=int,   default=20)
    ap.add_argument("--batch",            type=int,   default=512)
    ap.add_argument("--lr",               type=float, default=1e-3)
    ap.add_argument("--weight-decay",     type=float, default=0.0,
                    help="Adam L2 weight decay (1e-5 recommended for fresh NNUE training)")
    ap.add_argument("--optimizer",        choices=["adam", "adamw", "muon"], default="adam",
                    help="adam (default, unchanged prior behavior), adamw "
                         "(decoupled weight decay), or muon (torch.optim.Muon "
                         "on exactly-2D hidden matrices, AdamW fallback on "
                         "biases/norms/3D embeddings -- experimental)")
    ap.add_argument("--aux-lr",           type=float, default=None,
                    help="Muon hybrid only: LR for the AdamW fallback "
                         "optimizer on non-matrix params. Muon's update "
                         "geometry differs from Adam's, so reusing --lr "
                         "for both isn't a fair same-step-size comparison "
                         "by default; defaults to --lr if unset.")
    ap.add_argument("--grad-clip",        type=float, default=1.0,
                    help="Max gradient L2 norm (0=disabled; 1.0 is standard for NNUE stability)")
    ap.add_argument("--scale",            type=float, default=400.0)
    ap.add_argument("--checkpoint-steps", type=int,   default=1000)
    ap.add_argument("--val-split",        type=float, default=0.05)
    ap.add_argument("--resume",           action="store_true")
    ap.add_argument("--ckpt",             default=None)
    ap.add_argument("--cpu",              action="store_true")
    ap.add_argument("--min-ply",          type=int,   default=4)
    ap.add_argument("--max-ply",          type=int,   default=150)
    ap.add_argument("--sample-rate",      type=float, default=1.0)
    ap.add_argument("--game-ids",         default=None,
                    help="Comma-separated SQLite game ids (incremental / per-game train)")
    ap.add_argument("--micro",            action="store_true",
                    help="Fast single-game fine-tune: 1 epoch, no val split, low checkpoint churn")
    ap.add_argument("--max-samples",      type=int, default=0,
                    help="Cap teacher-dataset samples when --data is a dataset directory")
    ap.add_argument("--seed",             type=int, default=0,
                    help="Shuffle seed for teacher-dataset sampling")
    ap.add_argument("--coverage-min",     type=float, default=None,
                    help="Minimum featurization coverage ratio for teacher dataset")
    ap.add_argument("--min-val",          type=int, default=0,
                    help="Minimum validation samples (teacher dataset)")
    ap.add_argument("--min-train",        type=int, default=1,
                    help="Minimum training samples")
    ap.add_argument("--patience",         type=int, default=100,
                    help="Early-stop after N epochs with no val_loss improvement (0=disabled)")
    ap.add_argument("--no-parity",        action="store_true",
                    help="Skip parity_check before training (unsafe; use only when engine/Python eval diverge)")
    ap.add_argument("--log-every",        type=int, default=100,
                    help="Print training progress every N steps (0=disable)")
    ap.add_argument("--log-interval-sec", type=float, default=10.0,
                    help="Also print progress every N seconds (0=disable)")
    ap.add_argument("--recent-replay-fraction", type=float, default=0.0,
                    help="Fraction of epoch steps drawn from recent self-play (0=uniform)")
    ap.add_argument("--recent-window-games", type=int, default=128,
                    help="Recent self-play game window for replay sampler")
    ap.add_argument("--cache-dir",        default=None,
                    help="Pre-built feature cache dir (from build_feature_cache.py). "
                         "Skips all on-the-fly featurization and uses all 1.4M positions.")
    ap.add_argument("--labels-db",        default=None,
                    help="Stream training from labels.db (database-first path; cache optional).")
    ap.add_argument("--stream-epoch-size", type=int, default=0,
                    help="Fresh generated positions sampled per epoch from labels.db (0=auto).")
    ap.add_argument("--stream-featurize-chunk", type=int, default=4096,
                    help="Featurize chunk size for --labels-db streaming.")
    ap.add_argument("--stream-max-positions", type=int, default=0,
                    help="Cap total sampled positions (smoke tests).")
    ap.add_argument("--stream-retired-replay-fraction", type=float, default=0.05,
                    help="Extra retired replay rows added on top of the active epoch; excludes sanity-purged rows.")
    ap.add_argument("--stream-old-refresh-fraction", type=float, default=0.05,
                    help="Extra low-visit old/teacher rows added on top of fresh generated rows.")
    ap.add_argument("--stream-full-active-epoch", action="store_true",
                    help="Train one cold-start epoch over every active labels.db row before steady-state 2048+5% epochs.")
    ap.add_argument("--no-stream-phase-quota", action="store_true",
                    help="Disable opening/mid/endgame quota rebalance for streaming epochs")
    ap.add_argument("--defer-usage-commit", action="store_true",
                    help="Commit position training_visits only after successful epoch checkpoint save.")
    ap.add_argument("--no-usage-commit", action="store_true",
                    help="Never touch training_visits at all (position selection state stays "
                         "untouched). For running multiple candidates on IDENTICAL "
                         "sample_epoch_keys() output -- only the last candidate in such a "
                         "group should omit this flag so the real epoch position consumption "
                         "still happens exactly once.")
    args = ap.parse_args()

    if args.micro:
        args.epochs = 1
        args.val_split = 0.0
        args.checkpoint_steps = max(args.checkpoint_steps, 999_999)
        if args.lr == 1e-3:
            args.lr = 5e-4

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")
    # H=32 net: OMP thread overhead exceeds compute benefit even on 4 cores.
    # Measured: 1t=128.9ms/step, 4t=177ms/step, 8t=276ms/step.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print("Threads: intra=1 inter=1 (measured optimum regardless of net width)")

    try:
        stamp = assert_engine_ready(write_if_missing=True, parity=not args.no_parity)
        if args.no_parity:
            print("Engine stamp OK (parity check skipped)")
        else:
            print(f"Engine stamp OK: {stamp['sha256'][:12]}")
    except Exception as e:
        print(f"Training blocked by engine validation: {e}")
        sys.exit(1)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── data loading ──────────────────────────────────────────────────────────
    teacher_meta: dict | None = None
    recent_val_ds = None
    cache_dir_arg = getattr(args, "cache_dir", None)
    labels_db_arg = getattr(args, "labels_db", None)
    streaming_ds = None
    if labels_db_arg:
        labels_path = Path(labels_db_arg)
        if not labels_path.is_absolute():
            labels_path = (ROOT / labels_path).resolve()
        if not labels_path.is_file():
            print(f"ERROR: labels.db not found: {labels_path}")
            sys.exit(1)
        sys.path.insert(0, str(ROOT / "training"))
        from position_usage_db import open_labels_db
        from streaming_db_loader import DbTrainingIterableDataset, db_counts, sample_epoch_keys

        con = open_labels_db(labels_path)
        counts = db_counts(labels_path)
        cap = int(getattr(args, "stream_max_positions", 0) or 0)
        epoch_size = int(getattr(args, "stream_epoch_size", 0) or 0)
        if epoch_size <= 0:
            epoch_size = min(8192, max(1, counts.eligible_positions))
        if cap > 0:
            epoch_size = min(epoch_size, cap)
        seed = int(getattr(args, "seed", 0) or 0)
        pos_keys = sample_epoch_keys(
            con,
            epoch_size=epoch_size,
            seed=seed,
            retired_replay_fraction=float(getattr(args, "stream_retired_replay_fraction", 0.05) or 0.0),
            old_refresh_fraction=float(getattr(args, "stream_old_refresh_fraction", 0.05) or 0.0),
            full_active_epoch=bool(getattr(args, "stream_full_active_epoch", False)),
        )
        con.close()
        if not pos_keys:
            print("ERROR: no eligible positions in labels.db for streaming training")
            sys.exit(1)
        if not getattr(args, "no_stream_phase_quota", False):
            from canonical_sampling import apply_phase_sampling_quota

            pos_keys = apply_phase_sampling_quota(pos_keys, labels_path, seed=seed)
        chunk = int(getattr(args, "stream_featurize_chunk", 4096) or 4096)
        n_all = len(pos_keys)
        n_val = max(1, int(n_all * args.val_split)) if args.val_split > 0 and n_all > 8 else 0
        if n_val > 0:
            from streaming_val_split import split_streaming_epoch_keys

            train_keys, val_keys = split_streaming_epoch_keys(
                pos_keys,
                labels_db=labels_path,
                val_fraction=args.val_split,
                seed=seed,
            )
            train_ds = DbTrainingIterableDataset(
                labels_path,
                train_keys,
                trainer_batch_size=args.batch,
                chunk_size=chunk,
            )
            val_ds = DbTrainingIterableDataset(
                labels_path,
                val_keys,
                trainer_batch_size=args.batch,
                chunk_size=chunk,
            )
        else:
            train_ds = DbTrainingIterableDataset(
                labels_path,
                pos_keys,
                trainer_batch_size=args.batch,
                chunk_size=chunk,
            )
            val_ds = None
        print(f"Streaming DB training: {labels_path}")
        print(f"  eligible={counts.eligible_positions:,}  labeled={counts.labeled_positions:,}")
        print(
            f"  mode={'full-active' if bool(getattr(args, 'stream_full_active_epoch', False)) else 'fresh+refresh'}  "
            f"fresh_target={epoch_size:,}  old_refresh={float(getattr(args, 'stream_old_refresh_fraction', 0.05) or 0.0):.3f}  "
            f"epoch_sample={len(pos_keys):,}  train={len(train_ds):,}  val={len(val_ds) if val_ds else 0:,}"
        )
        print(f"  featurize_chunk={chunk}  fv_len={DbTrainingIterableDataset.FV_LEN}")
    elif cache_dir_arg:
        _cdir = Path(cache_dir_arg)
        if not _cdir.is_absolute():
            _cdir = (ROOT / _cdir).resolve()
        sys.path.insert(0, str(ROOT / "training"))
        from build_feature_cache import check_fingerprint as _cfp_check
        _ok, _reason = _cfp_check(_cdir)
        if not _ok:
            print(f"ERROR: feature cache invalid: {_reason}")
            print("  Rebuild: python training/build_feature_cache.py")
            sys.exit(1)
        train_ds = CachedDataset(_cdir, "train")
        val_ds   = CachedDataset(_cdir, "val")
        recent_val_ds = None
        if (_cdir / "recent_val_indices.npy").is_file():
            recent_val_ds = CachedDataset(_cdir, "recent_val")
        print(f"Feature cache: {_cdir}")
        print(f"  train={len(train_ds):,}  val={len(val_ds):,}  (full corpus)")
        if recent_val_ds is not None:
            print(f"  recent_val={len(recent_val_ds):,}")
    else:
        # On-the-fly featurization path (original logic).
        # The DB stores raw game sequences; expand to per-position records via eval-batch.
        print(f"Loading {args.data}...")
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = (ROOT / data_path).resolve()
        if data_path.is_dir() and (data_path / "manifest.json").is_file():
            from titanium_training.data.teacher_value import load_teacher_value_training_records

            records, teacher_meta = load_teacher_value_training_records(
                data_path,
                max_samples=int(getattr(args, "max_samples", 0) or 200_000),
                min_samples=4 if args.micro else 64,
                seed=int(getattr(args, "seed", 0) or 0),
                coverage_min=args.coverage_min,
            )
            meta_path = out_dir / "run_metadata.json"
            meta_path.write_text(json.dumps(teacher_meta, indent=2), encoding="utf-8")
            print(
                f"  {len(records)} teacher-value positions  "
                f"(manifest {teacher_meta['dataset_manifest_sha256'][:16]}..., "
                f"mode {teacher_meta['featurization_mode']}, "
                f"coverage {teacher_meta.get('coverage_percentage', 0):.2f}%)"
            )
        elif data_path.suffix == ".db":
            try:
                data_path = assert_canonical_training_db(data_path, context="train.py")
            except LegacyTrainingSourceError as e:
                print(f"Training blocked: {e}")
                sys.exit(1)
            from tools.datagen.datagen import load_games_by_ids, expand_games
            import sqlite3

            conn = sqlite3.connect(str(data_path))
            has_canonical = bool(
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='game_paths'"
                ).fetchone()
            )
            conn.close()
            if args.game_ids:
                ids = [int(x.strip()) for x in args.game_ids.split(",") if x.strip()]
                if has_canonical and load_games_for_training_ids is not None:
                    games = load_games_for_training_ids(data_path, ids)
                else:
                    games = load_games_by_ids(data_path, ids)
                print(f"  {len(games)} game(s) ids={ids}  ->  expanding via eval-batch...")
            elif has_canonical and load_games_for_training is not None:
                games = load_games_for_training(data_path)
                print(f"  {len(games)} games from canonical position store  ->  expanding via eval-batch...")
            else:
                from tools.datagen.datagen import load_games_from_db

                games = load_games_from_db(data_path)
                print(f"  {len(games)} games  ->  expanding positions via eval-batch...")
            records = expand_games(games, args.min_ply, args.max_ply, args.sample_rate)
        else:
            records = [json.loads(l) for l in data_path.read_text().splitlines() if l.strip()]
        if teacher_meta is None:
            print(f"  {len(records)} positions  (WDL/self-play outcome only)")

        if not records:
            print("  no training positions (empty game list or filters)")
            sys.exit(0)

        from titanium_training.data.split import ValidationSplitError, deterministic_train_val_split

        split_meta: dict | None = None
        if teacher_meta is not None and args.val_split > 0:
            min_val = args.min_val if args.min_val > 0 else 64
            try:
                train_recs, val_recs, split_meta = deterministic_train_val_split(
                    records,
                    val_fraction=args.val_split,
                    seed=int(args.seed),
                    min_val=min_val,
                    min_train=max(1, args.min_train),
                )
            except ValidationSplitError as e:
                print(f"Training blocked: {e}")
                sys.exit(1)
            teacher_meta.update(split_meta)
            (out_dir / "run_metadata.json").write_text(
                json.dumps(teacher_meta, indent=2), encoding="utf-8"
            )
        elif args.val_split <= 0 or len(records) < 4:
            val_recs = []
            train_recs = records
        else:
            random.shuffle(records)
            n_val = max(1, int(len(records) * args.val_split))
            val_recs = records[:n_val]
            train_recs = records[n_val:]
        print(f"  train={len(train_recs)}  val={len(val_recs)}")

        train_ds = QuoridorDataset(train_recs)
        val_ds   = QuoridorDataset(val_recs) if val_recs else None

    from torch.utils.data import IterableDataset

    micro_batch = min(args.batch, max(1, len(train_ds)))
    if isinstance(train_ds, IterableDataset):
        train_dl = DataLoader(train_ds, batch_size=None, shuffle=False, num_workers=0)
    else:
        train_dl = DataLoader(
            train_ds,
            batch_size=micro_batch if args.micro else args.batch,
            shuffle=True,
            num_workers=0,
        )
    val_dl = (
        DataLoader(val_ds, batch_size=None, shuffle=False, num_workers=0)
        if isinstance(val_ds, IterableDataset)
        else DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
        if val_ds
        else None
    )

    model     = HalfPW(args.weights).to(device)
    print(f"Model: NET_H={model.h} (from {args.weights} header)")
    optimizer = build_optimizer(
        model,
        kind=getattr(args, "optimizer", "adam"),
        lr=args.lr,
        weight_decay=args.weight_decay,
        aux_lr=getattr(args, "aux_lr", None),
    )

    step      = 0
    start_ep  = 0
    best_val  = float("inf")

    # Resume
    ckpt_path = args.ckpt
    if ckpt_path is None and args.resume:
        # Prefer latest step checkpoint; fall back to latest epoch checkpoint
        candidates = sorted(out_dir.glob("ckpt_step*.pt")) or sorted(out_dir.glob("ckpt_epoch*.pt"))
        if candidates:
            ckpt_path = str(candidates[-1])
    if ckpt_path and Path(ckpt_path).exists():
        try:
            step, start_ep, best_val, optimizer = load_checkpoint(ckpt_path, model, optimizer)
            print(f"Resumed from {ckpt_path}  (step={step}, epoch={start_ep}, best_val={best_val:.5f})")
        except RuntimeError as e:
            if "checkpoint schema" not in str(e):
                raise
            print(f"WARN: {e}")
            print("  Starting fresh from net_weights.bin (ws20 era: CAT-best features)")

    from titanium_training.training.guards import enforce_artifact_cap, post_train_check, pretrain_sanity_ok
    ok, msg = pretrain_sanity_ok(batch=False, parity=not args.no_parity)
    if not ok:
        print(f"Training blocked by guards: {msg}")
        sys.exit(1)
    cap_ok, cap_msg = enforce_artifact_cap(out_dir)
    print(f"Artifact guard: {cap_msg}")
    if not cap_ok:
        print(f"Training blocked: {cap_msg}")
        sys.exit(1)

    def to_device(batch):
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}

    usage_con = None
    pending_usage_keys: list[str] = []
    if labels_db_arg:
        from position_usage_db import bump_training_visits, open_labels_db

        labels_db_path = Path(labels_db_arg)
        if not labels_db_path.is_absolute():
            labels_db_path = ROOT / labels_db_path
        usage_con = open_labels_db(labels_db_path)

    def run_val(dl=None):
        loader = dl if dl is not None else val_dl
        if loader is None:
            return float("inf")
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                batch = to_device(batch)
                batch.pop("_pos_keys", None)
                sample_w = batch.pop("sample_weight", None)
                out   = model(batch)
                loss  = wdl_loss(out, batch["target"], args.scale, sample_w)
                w_n = float(sample_w.sum().item()) if sample_w is not None else float(len(batch["target"]))
                total += loss.item() * w_n
                n     += int(w_n)
        model.train()
        return total / n if n else 0.0

    from tools.audit.epoch_diagnostics import (
        assert_finite_tensor,
        grad_norm_stats,
        param_norm,
        prediction_label_stats,
        update_norm,
        write_epoch_diagnostics,
    )

    recent_val_ds_ref = recent_val_ds

    end_epoch = start_ep + args.epochs
    patience = args.patience if not args.micro else 0
    no_improve = 0
    print(f"\nTraining for {args.epochs} epochs, lr={args.lr}, scale={args.scale}, "
          f"batch={args.batch}, target=WDL, patience={patience or 'disabled'}")
    batch_sz = micro_batch if args.micro else args.batch
    steps_per_epoch = max(1, (len(train_ds) + batch_sz - 1) // batch_sz)
    print(f"  train_samples={len(train_ds):,}  ~{steps_per_epoch:,} steps/epoch  "
          f"log_every={args.log_every or 'off'}  log_interval={args.log_interval_sec or 0}s",
          flush=True)
    model.train()

    def _maybe_log_progress(
        *,
        epoch_idx: int,
        epoch_step: int,
        epoch_loss: float,
        epoch_n: int,
        epoch_t0: float,
        last_log_t: float,
    ) -> float:
        """Print step/loss/ETA; return updated last_log_t.

        Loss is position-weighted: sum(batch_loss * batch_size) / total_positions.
        """
        if epoch_n <= 0:
            return last_log_t
        now = time.perf_counter()
        by_step = args.log_every and epoch_step % args.log_every == 0
        by_time = args.log_interval_sec and (now - last_log_t) >= args.log_interval_sec
        if not (by_step or by_time):
            return last_log_t
        elapsed = now - epoch_t0
        step_rate = epoch_step / elapsed if elapsed > 0 else 0.0
        pos_rate = epoch_n / elapsed if elapsed > 0 else 0.0
        remaining = max(0, steps_per_epoch - epoch_step)
        eta = remaining / step_rate if step_rate > 0 else 0.0
        pct = 100.0 * epoch_step / steps_per_epoch
        print(
            f"  epoch {epoch_idx + 1}/{args.epochs}  "
            f"step {epoch_step:,}/{steps_per_epoch:,} ({pct:.1f}%)  "
            f"loss={epoch_loss / epoch_n:.5f}  "
            f"{step_rate:.0f} step/s  {pos_rate:,.0f} pos/s  ETA {eta:.0f}s",
            flush=True,
        )
        return now

    for epoch in range(start_ep, end_epoch):
        epoch_loss = 0.0
        epoch_n    = 0
        epoch_step = 0
        epoch_t0   = time.perf_counter()
        last_log_t = epoch_t0
        first_loss: float | None = None
        diag_preds: list[float] = []
        diag_labels: list[float] = []
        grad_means: list[float] = []
        grad_maxes: list[float] = []
        weight_diag = None
        if labels_db_arg:
            sys.path.insert(0, str(ROOT / "training"))
            from epoch_weight_diagnostics import EpochWeightDiagnostics

            weight_diag = EpochWeightDiagnostics()
        print(f"\n--- epoch {epoch + 1}/{end_epoch} start ---", flush=True)

        if isinstance(train_ds, CachedDataset):
            from position_usage import epoch_indices_low_visits_first

            ordered = epoch_indices_low_visits_first(
                train_ds.cache_dir,
                train_ds.indices,
                seed=int(getattr(args, "seed", 0) or 0) + epoch,
            )
            train_ds.set_epoch_indices(ordered)

        if isinstance(train_ds, CachedDataset) and args.recent_replay_fraction > 0:
            sys.path.insert(0, str(ROOT / "training"))
            from training_sampler import mix_train_indices

            row_keys_path = train_ds.cache_dir / "row_position_keys.npy"
            if row_keys_path.is_file():
                row_keys = list(np.load(row_keys_path, allow_pickle=True))
                from db_import import GAMES_DB_PATH

                mixed = mix_train_indices(
                    train_ds.indices,
                    row_keys,
                    GAMES_DB_PATH,
                    recent_fraction=args.recent_replay_fraction,
                    recent_window_games=args.recent_window_games,
                    seed=42 + epoch,
                )
                train_ds.set_epoch_indices(mixed)
                print(
                    f"  recent replay mix: fraction={args.recent_replay_fraction} "
                    f"window_games={args.recent_window_games}",
                    flush=True,
                )
        prev_params = {n: p.data.clone() for n, p in model.named_parameters()}
        for batch in train_dl:
            batch = to_device(batch)
            pos_keys = batch.pop("_pos_keys", None)
            sample_w = batch.pop("sample_weight", None)
            batch_tiers = batch.pop("_source_tier", None)
            batch_phases = batch.pop("_game_phase", None)
            if weight_diag is not None and sample_w is not None:
                weight_diag.record_batch(
                    tiers=batch_tiers,
                    phases=batch_phases,
                    weights=sample_w.detach().cpu().numpy(),
                )
            optimizer.zero_grad()
            out  = model(batch)
            loss = wdl_loss(out, batch["target"], args.scale, sample_w)
            loss.backward()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            gstats = grad_norm_stats(model)
            if gstats["max"] > 0:
                grad_means.append(gstats["mean"])
                grad_maxes.append(gstats["max"])
            optimizer.step()
            for n, p in model.named_parameters():
                assert_finite_tensor(f"param {n}", p.data)
            if usage_con is not None and pos_keys and not args.no_usage_commit:
                if args.defer_usage_commit:
                    pending_usage_keys.extend(list(pos_keys))
                else:
                    bump_training_visits(usage_con, list(pos_keys))
                    usage_con.commit()

            with torch.no_grad():
                pred = torch.sigmoid(out / args.scale)
                diag_preds.extend(pred.detach().cpu().tolist())
                diag_labels.extend(batch["target"].detach().cpu().tolist())

            w_n = float(sample_w.sum().item()) if sample_w is not None else float(len(batch["target"]))
            step       += 1
            epoch_step += 1
            epoch_loss += loss.item() * w_n
            epoch_n    += int(w_n)
            if first_loss is None and epoch_n > 0:
                first_loss = epoch_loss / epoch_n

            last_log_t = _maybe_log_progress(
                epoch_idx=epoch,
                epoch_step=epoch_step,
                epoch_loss=epoch_loss,
                epoch_n=epoch_n,
                epoch_t0=epoch_t0,
                last_log_t=last_log_t,
            )

            if step % args.checkpoint_steps == 0:
                val_loss = run_val()
                ckpt_file = out_dir / f"ckpt_step{step:07d}.pt"
                save_checkpoint(str(ckpt_file), model, optimizer, step, epoch, best_val)
                print(f"  step={step:7d}  train_loss={epoch_loss/epoch_n:.5f}  val_loss={val_loss:.5f}  -> {ckpt_file.name}")
                if val_loss < best_val:
                    best_val = val_loss
                    best_file = out_dir / "best.pt"
                    save_checkpoint(str(best_file), model, optimizer, step, epoch, best_val)
                    # Also export the weights in engine format for quick testing
                    model.save_weights(out_dir / "net_weights_best.bin")
                    print(f"  ** new best val_loss={best_val:.5f}")

        epoch_rows_used = None
        if isinstance(train_ds, CachedDataset):
            epoch_rows_used = train_ds.active_row_indices().copy()
            train_ds.clear_epoch_indices()
        pnorm = param_norm(model)
        unorm = update_norm(model, prev_params)
        upd_ratio = unorm / pnorm if pnorm > 0 else 0.0
        ep_loss = epoch_loss / max(epoch_n, 1)
        train_elapsed = time.perf_counter() - epoch_t0
        pos_rate = epoch_n / train_elapsed if train_elapsed > 0 else 0.0
        loss_span = (
            f"  loss {first_loss:.5f} -> {ep_loss:.5f}"
            if first_loss is not None and first_loss != ep_loss
            else f"  loss={ep_loss:.5f}"
        )
        print(
            f"Epoch {epoch - start_ep + 1}/{args.epochs}  train{loss_span}  "
            f"({epoch_step:,} steps, {epoch_n:,} positions in {train_elapsed:.1f}s, "
            f"{pos_rate:,.0f} pos/s)",
            flush=True,
        )
        if weight_diag is not None:
            weight_diag.log()
            weight_diag.write_json(out_dir / f"epoch_weight_diagnostics_{epoch + 1:04d}.json")
        if isinstance(train_ds, CachedDataset):
            from position_usage import bump_epoch

            rows_to_bump = epoch_rows_used if epoch_rows_used is not None else train_ds.indices
            u = bump_epoch(train_ds.cache_dir, rows_to_bump)
            train_ds.refresh_indices("train")
            print(
                f"  usage: touched={u['touched']:,} retired_total={u['retired_total']:,} "
                f"active_train={u['active_train']:,}",
                flush=True,
            )
        elif labels_db_arg:
            if usage_con is not None:
                from position_usage_db import status as usage_status

                u = usage_status(usage_con)
                print(
                    f"  db usage: retired_total={u['retired']:,} active={u['active']:,}",
                    flush=True,
                )
        # End-of-epoch checkpoint
        ep_file = out_dir / f"ckpt_epoch{epoch+1:04d}.pt"
        val_t0 = time.perf_counter()
        val_loss = run_val()
        recent_val_loss = (
            run_val(DataLoader(recent_val_ds_ref, batch_size=args.batch, shuffle=False, num_workers=0))
            if recent_val_ds_ref is not None and len(recent_val_ds_ref) > 0
            else None
        )
        val_elapsed = time.perf_counter() - val_t0
        ckpt_t0 = time.perf_counter()
        save_checkpoint(str(ep_file), model, optimizer, step, epoch + 1, best_val)
        ckpt_elapsed = time.perf_counter() - ckpt_t0
        if usage_con is not None and args.defer_usage_commit and pending_usage_keys:
            # Do NOT commit here -- write the sampled keys out and let the
            # caller (training_coordinator.py) commit them only if this
            # candidate actually gets ACCEPTED by the strength gate. A
            # quarantined candidate never shaped a promoted checkpoint, so
            # its sampled positions must stay exactly as fresh as before this
            # run (else repeated failed attempts would burn through the
            # visit budget on data that never influenced anything real).
            keys_path = out_dir / "pending_usage_keys.json"
            keys_path.write_text(json.dumps(pending_usage_keys), encoding="utf-8")
            print(
                f"  usage keys staged (commit deferred to caller): {len(pending_usage_keys):,}",
                flush=True,
            )
            pending_usage_keys.clear()
        diag = {
            "epoch": epoch + 1,
            "lr": args.lr,
            "train_loss_start": first_loss,
            "train_loss_end": ep_loss,
            "val_loss": val_loss,
            "recent_val_loss": recent_val_loss,
            "train_seconds": train_elapsed,
            "val_seconds": val_elapsed,
            "ckpt_seconds": ckpt_elapsed,
            "positions_per_second": pos_rate,
            "grad_norm_mean": sum(grad_means) / len(grad_means) if grad_means else 0.0,
            "grad_norm_max": max(grad_maxes) if grad_maxes else 0.0,
            "param_norm": pnorm,
            "update_norm": unorm,
            "update_over_param_norm": upd_ratio,
            **prediction_label_stats(diag_preds[-8192:], diag_labels[-8192:]),
        }
        write_epoch_diagnostics(out_dir / f"epoch_diagnostics_{epoch+1:04d}.json", diag)
        print(
            f"  val_loss={val_loss:.5f}"
            + (f"  recent_val_loss={recent_val_loss:.5f}" if recent_val_loss is not None else "")
            + f"  val_time={val_elapsed:.1f}s  ckpt_time={ckpt_elapsed:.1f}s  -> {ep_file.name}",
            flush=True,
        )
        print(
            f"  diagnostics: grad_norm={diag['grad_norm_mean']:.4f}/{diag['grad_norm_max']:.4f}  "
            f"update/param={upd_ratio:.2e}  pred_mean={diag.get('pred_mean', 0):.4f}",
            flush=True,
        )
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            save_checkpoint(str(out_dir / "best.pt"), model, optimizer, step, epoch + 1, best_val)
            model.save_weights(out_dir / "net_weights_best.bin")
            print(f"  ** new best val_loss={best_val:.5f}")
        elif args.micro:
            # Micro-train has no val split — persist latest weights for checkpoint resume.
            model.save_weights(out_dir / "net_weights_best.bin")
            print(f"  micro: saved latest -> net_weights_best.bin")
        else:
            no_improve += 1
            if patience and no_improve >= patience:
                print(f"\nEARLY STOP: val_loss has not improved for {patience} consecutive epochs "
                      f"(best={best_val:.5f}, current={val_loss:.5f}).  Stopping.")
                break

    print(f"\nTraining complete.  Best val_loss={best_val:.5f}")
    print(f"Best weights: {out_dir / 'net_weights_best.bin'}")
    post_train_check()
    if usage_con is not None:
        usage_con.close()
    print("To test: copy net_weights_best.bin -> engine/src/titanium/net_weights.bin, rebuild, run match vs baseline.")


if __name__ == "__main__":
    main()
