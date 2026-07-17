# Titanium vs Titanium — per-think flamegraphs

Regular match clock (`parallel_engine_match`):

- **60s per side** per game
- think allotment = `remaining_ms / 20`

Each Titanium think gets its own Inferno SVG so ply averages are meaningful.

## 1) Collect thinks (warm session play)

```powershell
$env:RUSTFLAGS = '-C target-cpu=native'
cargo build --release -p titanium --manifest-path engine\Cargo.toml

python tools\profile_tt_per_think\collect_thinks.py `
  --games 10 --clock-sec 60 --workers 4 --threads 1 `
  --out-dir training\data\profiles\tt_per_think_run
```

`--workers 4` runs 4 games at once (each side is 1 search thread → ~8 engine processes).

## 2) Flamegraph each think (Administrator / ETW)

```powershell
powershell -ExecutionPolicy Bypass -File tools\profile_tt_per_think\flamegraph_each_think.ps1 `
  -ThinksJsonl training\data\profiles\tt_per_think_run\thinks.jsonl `
  -OutDir training\data\profiles\tt_per_think_run `
  -Workers 8
```

`-Workers 8` runs 8 parallel flamegraph processes (sharded). Existing SVGs are skipped.

## 3) Aggregate by ply

```powershell
python tools\profile_tt_per_think\aggregate_by_ply.py `
  --out-dir training\data\profiles\tt_per_think_run
```

Outputs: `ply_summary.txt`, `ply_summary.json`, `flamegraphs\gXXX_plyYYY_sZ.svg`.

## 4) Absolute NPS: release (native) vs profiling (debug=2)

Flamegraphs use `--profile profiling` (release opts + `debug=2` + frame pointers). Absolute NPS must be measured on **both** builds on a quiet machine so hotspot % can be mapped to production speed.

```powershell
$rel  = "engine\target\release\search_bench.exe"      # RUSTFLAGS=-C target-cpu=native
$prof = "engine\target\profiling\search_bench.exe"  # inherits release + debug=2

# Alternating A/B, same args:
& $rel  think --ms 5000 --full --threads 1
& $prof think --ms 5000 --full --threads 1
```

Save medians to `nps_release_vs_profiling_pinned_rt.json` next to the run.

**Measurement rules (absolute NPS):**
1. Sample per-CPU load; pick the quietest logical CPUs (1 for 1-thread, N for N-thread).
2. Set `ProcessorAffinity` to those CPUs only + `PriorityClass = RealTime`.
3. Alternating A/B release vs profiling; report median of ≥5 runs.

Use release NPS for production speed; use profiling stacks for “where CPU goes”. Profiling inherits release opts (`debug=2` only), so the two should be within a few percent when pinned.
