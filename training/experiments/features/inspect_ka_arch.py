#!/usr/bin/env python3
"""Inspect Ka MCTS CNN from TensorFlow checkpoint (KaAiData).

Usage:
    python training/inspect_ka_arch.py
    python training/inspect_ka_arch.py --ckpt path/to/epoch15000.ckpt
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = (
    ROOT
    / "KaAiData/Quoridor-master KA ENgine/KA Engine Weights MCTS/application_data/parameter/epoch15000"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    args = ap.parse_args()

    meta_path = Path(str(args.ckpt) + ".json")
    if meta_path.exists():
        cfg = json.loads(meta_path.read_text(encoding="utf-8"))
        print("config:", cfg)
    else:
        cfg = {}

    import tensorflow as tf

    tf.compat.v1.disable_v2_behavior()
    reader = tf.compat.v1.train.NewCheckpointReader(str(args.ckpt))
    var = {
        k: v
        for k, v in reader.get_variable_to_shape_map().items()
        if not k.endswith("/Adagrad") and k not in ("current_loss_scale", "good_steps")
    }

    conv3 = [v for v in var.values() if len(v) == 4 and v[0] == 3 and v[1] == 3]
    conv1 = [v for v in var.values() if len(v) == 4 and v[0] == 1 and v[1] == 1]
    attn = [v for v in var.values() if v == [128, 128]]
    policy137 = [k for k, v in var.items() if v == [2592, 137]]
    value1 = sum(1 for v in var.values() if v == [2592, 1])
    policy81 = sum(1 for v in var.values() if v == [2592, 81])
    aux32 = sum(1 for v in var.values() if v == [3, 3, 128, 32])

    params = sum(reader.get_tensor(k).size for k in var)
    print(f"\nvariables: {len(var)}  trainable scalars: {params:,}")
    print(f"conv 3x3 kernels: {len(conv3)}  (stem 15->128 + {len(conv3)-1} internal)")
    print(f"conv 1x1 SE gates: {len(conv1)}")
    print(f"self-attn 128x128 (Q/K/V): {len(attn)}  (~{len(attn)//3} blocks)")
    print(f"aux policy conv 128->32: {aux32}")
    print(f"policy heads 2592x137: {len(policy137)}")
    print(f"policy heads 2592x81: {policy81}")
    print(f"value heads 2592x1: {value1}")
    print("\n2592 = 81*32  ->  9x9 spatial map with 32 channels before heads")

    print("\n=== vs titanium HalfPW (leaf NNUE) ===")
    print("Ka:  15 raw board planes -> 3x3 conv -> 128ch ResNet+SE+attention x18 -> MCTS policy/value")
    print("Ours: BFS 11 planes + sparse wall buckets + H=32 halfpw (no spatial conv until extend)")
    print("Ka weights CANNOT load: different inputs (raw vs BFS), H=128 vs 32, MCTS heads vs scalar cp")


if __name__ == "__main__":
    main()
