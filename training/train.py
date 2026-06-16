"""Fine-tune HalfPW NNUE weights from self-play game outcomes.

Uses sigmoid cross-entropy (WDL loss): outcome +1/-1 is mapped to a target
win probability, and the net's centipawn eval is passed through sigmoid to
get a predicted probability.  Trains ALL weights (ws, b1, w2, w1c, po, px)
starting from the current net_weights.bin.

Checkpoints are saved every --checkpoint-steps steps and on every best-val-loss.
Training is always resumable from the latest checkpoint.

Usage:
    python training/train.py --data training/data/games.jsonl
    python training/train.py --data training/data/games.jsonl --resume  # auto-finds latest ckpt
    python training/train.py --data training/data/games.jsonl --resume --ckpt path/to/ckpt.pt

Options:
    --data PATH          JSONL file from datagen.py
    --weights PATH       Starting weights (default: engine/src/acev13/net_weights.bin)
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

import argparse
import json
import math
import random
import sqlite3
import struct
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── constants matching halfpw.py / net.rs ────────────────────────────────────

NET_H     = 32
WSKIP_LEN = 16
W1C_SHAPE = (9, 128, NET_H)   # pawn buckets × wall slots × hidden
PO_SHAPE  = (81, NET_H)
PX_SHAPE  = (81, NET_H)

NET_MIRC = [(8 - i // 9) * 9 + i % 9 for i in range(81)]
NET_MIRS = [(7 - i // 8) * 8 + i % 8 for i in range(64)]
NET_BKT  = [(i // 9 // 3) * 3 + (i % 9) // 3 for i in range(81)]

ROOT    = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "engine" / "src" / "acev13" / "net_weights.bin"

# ── model ─────────────────────────────────────────────────────────────────────

class HalfPW(nn.Module):
    """Differentiable HalfPW forward pass, initialised from net_weights.bin."""

    def __init__(self, weights_path):
        super().__init__()
        data = Path(weights_path).read_bytes()
        total = WSKIP_LEN + NET_H + NET_H + math.prod(W1C_SHAPE) + math.prod(PO_SHAPE) + math.prod(PX_SHAPE)
        vals  = list(struct.unpack(f"<{total}d", data))
        o = 0
        def take(n):
            nonlocal o; s = vals[o:o+n]; o += n; return s

        self.ws  = nn.Parameter(torch.tensor(take(WSKIP_LEN), dtype=torch.float32))
        self.b1  = nn.Parameter(torch.tensor(take(NET_H),     dtype=torch.float32))
        self.w2  = nn.Parameter(torch.tensor(take(NET_H),     dtype=torch.float32))
        self.w1c = nn.Parameter(torch.tensor(take(math.prod(W1C_SHAPE)), dtype=torch.float32).view(*W1C_SHAPE))
        self.po  = nn.Parameter(torch.tensor(take(math.prod(PO_SHAPE)),  dtype=torch.float32).view(*PO_SHAPE))
        self.px  = nn.Parameter(torch.tensor(take(math.prod(PX_SHAPE)),  dtype=torch.float32).view(*PX_SHAPE))

    def forward(self, b):
        """
        b: dict of batched tensors (see QuoridorDataset.__getitem__).
        Returns centipawn eval [N] from the side-to-move's perspective.
        """
        ws  = self.ws
        d_me  = b["d_me"].float()
        d_opp = b["d_opp"].float()
        w_me  = b["w_me"].float()
        w_opp = b["w_opp"].float()

        pd = d_opp - d_me
        wd = w_me  - w_opp

        out = (ws[0]
               + ws[1]  * pd
               + ws[2]  * wd
               + ws[3]  * d_me
               + ws[4]  * d_opp
               + ws[9]  * pd * (w_me + w_opp) / 20.0
               + ws[10] * wd * (d_me + d_opp) / 16.0)

        w_opp_zero = (w_opp == 0.0)
        w_me_zero  = (w_me  == 0.0)
        out = out + ws[6] * w_opp_zero.float()
        out = out + ws[5] * (w_opp_zero & (d_me <= d_opp)).float()
        out = out + ws[8] * w_me_zero.float()
        out = out + ws[7] * (w_me_zero & (d_opp <= d_me - 1.0)).float()

        w_me_capped  = w_me.clamp(max=3.0)
        w_opp_capped = w_opp.clamp(max=3.0)
        out = out + ws[11] * w_me_capped  * (d_opp <= 4.0).float()
        out = out + ws[12] * w_opp_capped * (d_me  <= 4.0).float()

        # ws[13]: fragile-lead; ws[14..15]: corridor-width
        out = out + ws[13] * pd * w_opp / 10.0
        out = out + ws[14] * b["width_me"].float()
        out = out + ws[15] * b["width_opp"].float()

        # Neural hidden layer
        bucket     = b["bucket"]        # [N]
        wall_mask  = b["wall_mask"].float()  # [N, 128]
        pawn_me    = b["pawn_me"]       # [N]
        pawn_opp   = b["pawn_opp"]      # [N]

        w1c_sel = self.w1c[bucket]                              # [N, 128, H]
        acc     = (w1c_sel * wall_mask.unsqueeze(-1)).sum(dim=1) # [N, H]
        hid     = self.b1 + acc + self.po[pawn_me] + self.px[pawn_opp]  # [N, H]
        hid_act = hid.clamp(0.0, 1.0)                          # clipped ReLU
        out     = out + (self.w2 * hid_act * 200.0).sum(dim=-1)

        return out  # centipawns, side-to-move positive

    def save_weights(self, path):
        """Serialize back to the engine's little-endian f64 binary format."""
        with open(path, "wb") as f:
            def w(t):
                vals = t.detach().cpu().double().flatten().tolist()
                f.write(struct.pack(f"<{len(vals)}d", *vals))
            w(self.ws);   w(self.b1);  w(self.w2)
            w(self.w1c);  w(self.po);  w(self.px)
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

        # Corridor-width features (computed by datagen)
        width_me  = r["corridor_width0"] if me == 0 else r["corridor_width1"]
        width_opp = r["corridor_width1"] if me == 0 else r["corridor_width0"]

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
            "width_me":  torch.tensor(width_me,    dtype=torch.float32),
            "width_opp": torch.tensor(width_opp,   dtype=torch.float32),
            "bucket":    torch.tensor(bucket,      dtype=torch.long),
            "wall_mask": torch.tensor(wall_mask,   dtype=torch.float32),
            "pawn_me":   torch.tensor(pawn_me_idx, dtype=torch.long),
            "pawn_opp":  torch.tensor(pawn_opp_idx,dtype=torch.long),
            "target":    torch.tensor(target,      dtype=torch.float32),
        }


# ── training loop ─────────────────────────────────────────────────────────────

def wdl_loss(eval_cp, target, scale):
    """Binary cross-entropy between sigmoid(eval/scale) and target win prob."""
    pred = torch.sigmoid(eval_cp / scale)
    return F.binary_cross_entropy(pred, target)


def save_checkpoint(path, model, optimizer, step, epoch, best_val):
    torch.save({
        "step": step, "epoch": epoch, "best_val": best_val,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)


def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"], ckpt["epoch"], ckpt["best_val"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data",             default="training/data/all_games.db")
    ap.add_argument("--weights",          default=str(WEIGHTS))
    ap.add_argument("--out-dir",          default="training/checkpoints")
    ap.add_argument("--epochs",           type=int,   default=20)
    ap.add_argument("--batch",            type=int,   default=512)
    ap.add_argument("--lr",               type=float, default=1e-3)
    ap.add_argument("--scale",            type=float, default=400.0)
    ap.add_argument("--checkpoint-steps", type=int,   default=1000)
    ap.add_argument("--val-split",        type=float, default=0.05)
    ap.add_argument("--resume",           action="store_true")
    ap.add_argument("--ckpt",             default=None)
    ap.add_argument("--cpu",              action="store_true")
    ap.add_argument("--min-ply",          type=int,   default=4)
    ap.add_argument("--max-ply",          type=int,   default=150)
    ap.add_argument("--sample-rate",      type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data.  The DB stores raw game sequences; expand to per-position
    # records via eval-batch here (single subprocess, all positions at once).
    print(f"Loading {args.data}...")
    data_path = Path(args.data)
    if data_path.suffix == ".db":
        from datagen import load_games_from_db, expand_games
        games = load_games_from_db(data_path)
        print(f"  {len(games)} games  ->  expanding positions via eval-batch...")
        records = expand_games(games, args.min_ply, args.max_ply, args.sample_rate)
    else:
        records = [json.loads(l) for l in data_path.read_text().splitlines() if l.strip()]
    print(f"  {len(records)} positions")

    random.shuffle(records)
    n_val  = max(1, int(len(records) * args.val_split))
    val_recs   = records[:n_val]
    train_recs = records[n_val:]
    print(f"  train={len(train_recs)}  val={n_val}")

    train_ds = QuoridorDataset(train_recs)
    val_ds   = QuoridorDataset(val_recs)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    model     = HalfPW(args.weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
        step, start_ep, best_val = load_checkpoint(ckpt_path, model, optimizer)
        print(f"Resumed from {ckpt_path}  (step={step}, epoch={start_ep}, best_val={best_val:.5f})")

    def to_device(batch):
        return {k: v.to(device) for k, v in batch.items()}

    def run_val():
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                batch = to_device(batch)
                out   = model(batch)
                loss  = wdl_loss(out, batch["target"], args.scale)
                total += loss.item() * len(batch["target"])
                n     += len(batch["target"])
        model.train()
        return total / n if n else 0.0

    print(f"\nTraining for {args.epochs} epochs, lr={args.lr}, scale={args.scale}, batch={args.batch}")
    model.train()

    for epoch in range(start_ep, args.epochs):
        epoch_loss = 0.0
        epoch_n    = 0
        for batch in train_dl:
            batch = to_device(batch)
            optimizer.zero_grad()
            out  = model(batch)
            loss = wdl_loss(out, batch["target"], args.scale)
            loss.backward()
            optimizer.step()

            step       += 1
            epoch_loss += loss.item() * len(batch["target"])
            epoch_n    += len(batch["target"])

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

        ep_loss = epoch_loss / max(epoch_n, 1)
        print(f"Epoch {epoch+1}/{args.epochs}  avg_train_loss={ep_loss:.5f}")
        # End-of-epoch checkpoint
        ep_file = out_dir / f"ckpt_epoch{epoch+1:04d}.pt"
        val_loss = run_val()
        save_checkpoint(str(ep_file), model, optimizer, step, epoch + 1, best_val)
        print(f"  epoch checkpoint -> {ep_file.name}  val_loss={val_loss:.5f}")
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(str(out_dir / "best.pt"), model, optimizer, step, epoch + 1, best_val)
            model.save_weights(out_dir / "net_weights_best.bin")
            print(f"  ** new best val_loss={best_val:.5f}")

    print(f"\nTraining complete.  Best val_loss={best_val:.5f}")
    print(f"Best weights: {out_dir / 'net_weights_best.bin'}")
    print("To test: copy net_weights_best.bin -> engine/src/acev13/net_weights.bin, rebuild, run match vs baseline.")


if __name__ == "__main__":
    main()
