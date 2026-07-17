"""
Fine-tune only the LINEAR weights (ws, route_*) starting from frozen ACE v13.

Keeps the hidden layer (b1, w2, w1c, po, px) frozen so it can't inflate and
swamp the strategic signal from ws[1] (pawn distance contribution).

Usage:
    python training/train_linear_only.py [--epochs N] [--lr LR]
"""
import argparse, math, struct, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training" / "titanium_training"))

import torch
import torch.nn.functional as F
import numpy as np

NET_H       = 32
WSKIP_LEN   = 20
W1C_SHAPE   = (9, 128, NET_H)
PO_SHAPE    = (81, NET_H)
PX_SHAPE    = (81, NET_H)
FIELD_SHAPE = (81,)
FIELD_PLANES = 5
NET_WEIGHT_F64S = (WSKIP_LEN + NET_H + NET_H
                   + math.prod(W1C_SHAPE) + math.prod(PO_SHAPE) + math.prod(PX_SHAPE)
                   + math.prod(FIELD_SHAPE) * FIELD_PLANES)

ROUTE_ME        = "route_me"
ROUTE_OPP       = "route_opp"
ROUTE_NEAR_ME   = "route_near_me"
ROUTE_NEAR_OPP  = "route_near_opp"
ROUTE_CONTESTED = "route_contested"

CACHE_DIR      = ROOT / "training" / "data" / "feature_cache"
FROZEN_WEIGHTS = ROOT / "site" / "engine" / "src" / "titanium" / "net_weights_frozen.bin"
OUT_DIR        = ROOT / "training" / "runs" / "v16_linear_only"
SCALE          = 400.0


import torch.nn as nn

class HalfPW(nn.Module):
    def __init__(self, weights_path):
        super().__init__()
        data = Path(weights_path).read_bytes()
        assert len(data) == NET_WEIGHT_F64S * 8, f"wrong size: {len(data)} vs {NET_WEIGHT_F64S*8}"
        vals = list(struct.unpack(f"<{NET_WEIGHT_F64S}d", data))
        o = 0
        def take(n):
            nonlocal o; s = vals[o:o+n]; o += n; return s
        self.ws  = nn.Parameter(torch.tensor(take(WSKIP_LEN), dtype=torch.float32))
        self.b1  = nn.Parameter(torch.tensor(take(NET_H), dtype=torch.float32))
        self.w2  = nn.Parameter(torch.tensor(take(NET_H), dtype=torch.float32))
        self.w1c = nn.Parameter(torch.tensor(take(math.prod(W1C_SHAPE)), dtype=torch.float32).view(*W1C_SHAPE))
        self.po  = nn.Parameter(torch.tensor(take(math.prod(PO_SHAPE)),  dtype=torch.float32).view(*PO_SHAPE))
        self.px  = nn.Parameter(torch.tensor(take(math.prod(PX_SHAPE)),  dtype=torch.float32).view(*PX_SHAPE))
        self.route_me        = nn.Parameter(torch.tensor(take(81), dtype=torch.float32))
        self.route_opp       = nn.Parameter(torch.tensor(take(81), dtype=torch.float32))
        self.route_near_me   = nn.Parameter(torch.tensor(take(81), dtype=torch.float32))
        self.route_near_opp  = nn.Parameter(torch.tensor(take(81), dtype=torch.float32))
        self.route_contested = nn.Parameter(torch.tensor(take(81), dtype=torch.float32))

    def freeze_hidden(self):
        for p in [self.b1, self.w2, self.w1c, self.po, self.px]:
            p.requires_grad_(False)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Frozen hidden layer. Trainable: {trainable:,} / {total:,} params")

    def forward(self, b):
        ws = self.ws
        d_me_raw  = b["d_me"].float()
        d_opp_raw = b["d_opp"].float()
        w_me_raw  = b["w_me"].float()
        w_opp_raw = b["w_opp"].float()
        d_me = d_me_raw / 16.0; d_opp = d_opp_raw / 16.0
        w_me = w_me_raw / 10.0; w_opp = w_opp_raw / 10.0
        pd = d_opp - d_me; wd = w_me - w_opp
        out = (ws[0] + ws[1]*pd + ws[2]*wd + ws[3]*d_me + ws[4]*d_opp
               + ws[9]*pd*(w_me+w_opp) + ws[10]*wd*(d_me+d_opp))
        w_opp_zero = (w_opp_raw == 0.0); w_me_zero = (w_me_raw == 0.0)
        out = out + ws[6]*w_opp_zero.float()
        out = out + ws[5]*(w_opp_zero & (d_me_raw <= d_opp_raw)).float()
        out = out + ws[8]*w_me_zero.float()
        out = out + ws[7]*(w_me_zero & (d_opp_raw <= d_me_raw - 1.0)).float()
        out = out + ws[11]*w_me.clamp(max=0.3)*(d_opp_raw <= 4.0).float()
        out = out + ws[12]*w_opp.clamp(max=0.3)*(d_me_raw  <= 4.0).float()
        out = out + ws[13]*pd*w_opp
        out = out + ws[15]*b["width_opp"].float()/9.0
        out = out + (b[ROUTE_ME]        * self.route_me).sum(dim=1)
        out = out + (b[ROUTE_OPP]       * self.route_opp).sum(dim=1)
        out = out + (b[ROUTE_NEAR_ME]   * self.route_near_me).sum(dim=1)
        out = out + (b[ROUTE_NEAR_OPP]  * self.route_near_opp).sum(dim=1)
        out = out + (b[ROUTE_CONTESTED] * self.route_contested).sum(dim=1)
        # hidden (frozen)
        w1c_sel = self.w1c[b["bucket"]]
        acc = (w1c_sel * b["wall_mask"].float().unsqueeze(-1)).sum(dim=1)
        hid = self.b1 + acc + self.po[b["pawn_me"]] + self.px[b["pawn_opp"]]
        out = out + (self.w2 * hid.clamp(0.0, 1.0) * 200.0).sum(dim=-1)
        return out

    def save_weights(self, path):
        with open(path, "wb") as f:
            def w(t):
                vals = t.detach().cpu().double().flatten().tolist()
                f.write(struct.pack(f"<{len(vals)}d", *vals))
            w(self.ws); w(self.b1); w(self.w2)
            w(self.w1c); w(self.po); w(self.px)
            w(self.route_me); w(self.route_opp)
            w(self.route_near_me); w(self.route_near_opp); w(self.route_contested)
        print(f"  saved -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",  type=int,   default=5)
    ap.add_argument("--lr",      type=float, default=0.001)
    ap.add_argument("--batch",   type=int,   default=128)
    ap.add_argument("--weights", default=str(FROZEN_WEIGHTS))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting from: {args.weights}")

    model = HalfPW(args.weights)
    model.freeze_hidden()

    from training.trainer import CachedDataset
    train_ds = CachedDataset(CACHE_DIR, split="train")
    val_ds   = CachedDataset(CACHE_DIR, split="val")
    # If position_usage retired all samples, fall back to raw index files
    if len(train_ds) == 0:
        print("position_usage retired all train samples; using train_indices.npy directly")
        train_ds.indices = np.load(CACHE_DIR / "train_indices.npy")
    if len(val_ds) == 0:
        val_ds.indices = np.load(CACHE_DIR / "val_indices.npy")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=256, shuffle=False)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def wdl_loss(out, target):
        return F.binary_cross_entropy(torch.sigmoid(out / SCALE), target.float())

    def validate():
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                out = model(batch)
                total += wdl_loss(out, batch["target"]).item() * len(out)
                n += len(out)
        model.train()
        return total / n if n > 0 else float("inf")

    best_val = float("inf")
    print(f"\nTraining {args.epochs} epochs  lr={args.lr}  batch={args.batch}")

    for epoch in range(args.epochs):
        model.train()
        train_loss, steps = 0.0, 0
        for batch in train_dl:
            optimizer.zero_grad()
            out = model(batch)
            loss = wdl_loss(out, batch["target"])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            steps += 1

        val_loss = validate()
        print(f"Epoch {epoch+1}/{args.epochs}  train={train_loss/steps:.5f}  val={val_loss:.5f}", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            model.save_weights(out_dir / "net_weights_best.bin")
            print(f"  ** new best val={best_val:.5f}")

        model.save_weights(out_dir / f"net_weights_epoch{epoch+1}.bin")

    print(f"\nBest val_loss: {best_val:.5f}")

    import struct as _struct, numpy as _np
    data = (out_dir / "net_weights_best.bin").read_bytes()
    ws = list(_struct.unpack_from("<20d", data))
    print(f"Final ws[1]/16 = {ws[1]/16:.3f} cp/step  (frozen ref: {160.81/16:.3f})")

    w2 = list(_struct.unpack_from(f"<{NET_H}d", data, 20*8 + NET_H*8))
    print(f"Final w2 abs-sum = {sum(abs(x) for x in w2):.3f}  (frozen ref: 6.63)")


if __name__ == "__main__":
    main()
