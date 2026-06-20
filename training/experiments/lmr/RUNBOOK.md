# Phase-3 LMRH Runbook

This runbook is for the detached Phase-3 learned LMR head pipeline.

Runtime activation stays OFF.

## Preconditions

PowerShell from repo root:

```powershell
$env:RUSTFLAGS = "-C target-cpu=native"
cargo -C engine build --release --bin titanium
python training/nnue_cli.py preflight
```

Expected trunk SHA-256 at the time of this runbook update:

```text
dc2e3e95b099409361ce5682ab6f7d85dfe32503107e32e4e6467340a65ffed6
```

## Exact CLI discovery

Collector help:

```powershell
python training/collect_reduction_counterfactuals_v3.py --help
```

Trainer help:

```powershell
python training/train_lmr_head_v3.py --help
```

The collector now reconfigures stdout/stderr to UTF-8 on Windows, so help no longer crashes on `cp1250`.

## Local smoke run

Natural smoke collection:

```powershell
python training/collect_reduction_counterfactuals_v3.py `
  --natural-target 120 `
  --out-dir training/data/lmr_phase3_smoke `
  --depth 8 `
  --min-event-depth 6 `
  --min-ply 11 `
  --event-scan-limit 64 `
  --samples-per-position 2 `
  --minimum-nodes-saved 8 `
  --minimum-savings-ratio 0.05 `
  --seed 777
```

Hard-negative smoke pass:

```powershell
python training/collect_reduction_counterfactuals_v3.py `
  --hard-negative-pass `
  --natural-file training/data/lmr_phase3_smoke/natural.jsonl `
  --out-dir training/data/lmr_phase3_smoke `
  --depth 8 `
  --min-event-depth 6 `
  --minimum-nodes-saved 8 `
  --minimum-savings-ratio 0.05 `
  --seed 777 `
  --hard-negative-target 24
```

Smoke training phases without opening holdout:

```powershell
python training/train_lmr_head_v3.py `
  --natural training/data/lmr_phase3_smoke/natural.jsonl `
  --hard-negatives training/data/lmr_phase3_smoke/hard_negatives.jsonl `
  --out-dir training/checkpoints/lmr_v3_smoke `
  --phase narrowing `
  --epochs 80 `
  --lr 0.002
```

```powershell
python training/train_lmr_head_v3.py `
  --natural training/data/lmr_phase3_smoke/natural.jsonl `
  --hard-negatives training/data/lmr_phase3_smoke/hard_negatives.jsonl `
  --out-dir training/checkpoints/lmr_v3_smoke `
  --phase stability `
  --epochs 80 `
  --lr 0.002
```

```powershell
python training/train_lmr_head_v3.py `
  --natural training/data/lmr_phase3_smoke/natural.jsonl `
  --hard-negatives training/data/lmr_phase3_smoke/hard_negatives.jsonl `
  --out-dir training/checkpoints/lmr_v3_smoke `
  --phase manifest `
  --epochs 80 `
  --lr 0.002
```

```powershell
python training/train_lmr_head_v3.py `
  --natural training/data/lmr_phase3_smoke/natural.jsonl `
  --hard-negatives training/data/lmr_phase3_smoke/hard_negatives.jsonl `
  --out-dir training/checkpoints/lmr_v3_smoke `
  --phase shadow `
  --epochs 80 `
  --lr 0.002
```

## Cluster-scale collection

Natural collection target:

```powershell
python training/collect_reduction_counterfactuals_v3.py `
  --natural-target 10000 `
  --out-dir training/data/lmr_phase3 `
  --depth 8 `
  --min-event-depth 6 `
  --min-ply 11 `
  --event-scan-limit 128 `
  --samples-per-position 2 `
  --minimum-nodes-saved 8 `
  --minimum-savings-ratio 0.05 `
  --seed 777
```

Hard-negative enrichment:

```powershell
python training/collect_reduction_counterfactuals_v3.py `
  --hard-negative-pass `
  --natural-file training/data/lmr_phase3/natural.jsonl `
  --out-dir training/data/lmr_phase3 `
  --depth 8 `
  --min-event-depth 6 `
  --minimum-nodes-saved 8 `
  --minimum-savings-ratio 0.05 `
  --seed 777 `
  --hard-negative-target 200
```

Phase-3 training sequence:

```powershell
python training/train_lmr_head_v3.py `
  --natural training/data/lmr_phase3/natural.jsonl `
  --hard-negatives training/data/lmr_phase3/hard_negatives.jsonl `
  --out-dir training/checkpoints/lmr_v3 `
  --phase narrowing
```

```powershell
python training/train_lmr_head_v3.py `
  --natural training/data/lmr_phase3/natural.jsonl `
  --hard-negatives training/data/lmr_phase3/hard_negatives.jsonl `
  --out-dir training/checkpoints/lmr_v3 `
  --phase stability
```

```powershell
python training/train_lmr_head_v3.py `
  --natural training/data/lmr_phase3/natural.jsonl `
  --hard-negatives training/data/lmr_phase3/hard_negatives.jsonl `
  --out-dir training/checkpoints/lmr_v3 `
  --phase manifest
```

```powershell
python training/train_lmr_head_v3.py `
  --natural training/data/lmr_phase3/natural.jsonl `
  --hard-negatives training/data/lmr_phase3/hard_negatives.jsonl `
  --out-dir training/checkpoints/lmr_v3 `
  --phase holdout
```

```powershell
python training/train_lmr_head_v3.py `
  --natural training/data/lmr_phase3/natural.jsonl `
  --hard-negatives training/data/lmr_phase3/hard_negatives.jsonl `
  --out-dir training/checkpoints/lmr_v3 `
  --phase shadow
```

## Smoke verdict

- The smoke run proved end-to-end mechanics: collection, grouped split validation, narrowing, stability, manifest freeze, artifact export, and fixed-depth shadow parity.
- The smoke data is too soft for any GO decision.
- Most rows came from cheap fail-low scouts, hard-negative mining found `0` rows, and there were `0` unsafe examples.
- This means the next real collection must increase pressure on more expensive and tactically unstable events rather than trusting this exact parameter mix.
