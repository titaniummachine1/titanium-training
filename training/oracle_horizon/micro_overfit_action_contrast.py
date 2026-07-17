"""Disposable child-value micro-overfit for action-contrast rows.

This intentionally refuses to train when WDL perspective is ambiguous.  The
model output is child-STM perspective, so a root-STM oracle target is negated
after the move before constructing the child target.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))


def _target(wdl: object) -> float:
    text = str(wdl).strip().upper()
    if text in {"W", "WIN", "1"}:
        return 1.0
    if text in {"L", "LOSS", "-1"}:
        return 0.0
    if text in {"D", "DRAW", "0"}:
        return 0.5
    raise ValueError(f"unclear oracle_wdl={wdl!r}; refusing to guess sign")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sample", type=Path)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--scale", type=float, default=400.0)
    args = ap.parse_args()
    result = {
        "schema": "micro-overfit-action-contrast-v1",
        "sample": str(args.sample), "out_dir": str(args.out_dir),
        "status": "READY_TO_RUN", "steps_requested": args.steps,
        "sign_convention": "child output is child STM; child target = 1 - root win target",
    }
    try:
        rows = [json.loads(x) for x in args.sample.read_text(encoding="utf-8").splitlines() if x.strip()]
        examples = []
        for row in rows:
            root = _target(row.get("oracle_wdl"))
            preserving = row.get("preserving_children") or []
            losing = row.get("losing_or_worsening_children") or []
            if not preserving:
                continue
            # Root target is from the root STM. After one move, the child STM
            # is the opponent, hence complement the WDL target.
            child_target = 1.0 - root
            examples.extend((c, child_target, "preserving") for c in preserving)
            losing_target = 0.5 if root == 0.5 else root
            examples.extend((c, 1.0 - losing_target, "losing") for c in losing)
        if not examples:
            result.update(status="READY_TO_RUN", reason="no usable contrast rows")
        else:
            # Keep the real fitting path explicit and fail closed if optional
            # dependencies or packed featurization cannot be loaded.
            import torch
            from titanium_training.data.eval_packed import eval_packed_batch_allow_errors
            from titanium_training.training.trainer import HalfPW, wdl_loss

            packed = [(i, bytes.fromhex(e[0]["child_packed_hex"])) for i, e in enumerate(examples)]
            evals = eval_packed_batch_allow_errors(packed)
            records = []
            for e, ev in zip(examples, evals):
                if not ev.get("ok", True):
                    raise RuntimeError(f"child featurization failed: {ev}")
                record = dict(ev)
                record["outcome"] = 1 if e[1] >= 0.5 else -1
                records.append(record)
            from titanium_training.training.trainer import QuoridorDataset
            ds = QuoridorDataset(records)
            model = HalfPW(args.weights)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            initial = model({k: torch.stack([ds[i][k] for i in range(len(ds))]) for k in ds[0] if k != "target"})
            initial_root = (-initial.detach()).tolist()
            target = torch.tensor([e[1] for e in examples], dtype=torch.float32)
            for _ in range(max(1, args.steps)):
                batch = {k: torch.stack([ds[i][k] for i in range(len(ds))]) for k in ds[0] if k != "target"}
                # QuoridorDataset's target is outcome-derived; replace it with
                # the verified action-contrast target.
                batch["target"] = target
                opt.zero_grad()
                loss = wdl_loss(model(batch), target, args.scale)
                loss.backward()
                opt.step()
            with torch.no_grad():
                final_child = model(batch)
                final_root = -final_child
                pairs = []
                cursor = 0
                for row in rows:
                    p = row.get("preserving_children") or []
                    l = row.get("losing_or_worsening_children") or []
                    if not p:
                        continue
                    pscore = float(final_root[cursor].item())
                    cursor += len(p)
                    lscore = float(final_root[cursor].mean().item()) if l else None
                    cursor += len(l)
                    pairs.append({"preserving_root_score": pscore, "losing_root_score": lscore})
            passed = all(x["losing_root_score"] is None or x["preserving_root_score"] > x["losing_root_score"] for x in pairs)
            result.update(status="PASS" if passed else "FAIL", usable_examples=len(examples),
                          lineages=len(pairs), ranking_perfect=passed, pairs=pairs)
            args.out_dir.mkdir(parents=True, exist_ok=True)
            model.save_weights(args.out_dir / "micro_overfit_weights.bin")
    except Exception as exc:
        result.update(status="READY_TO_RUN", reason=f"not run: {type(exc).__name__}: {exc}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir.parent / "MICRO_OVERFIT_RESULT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in {"PASS", "READY_TO_RUN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
