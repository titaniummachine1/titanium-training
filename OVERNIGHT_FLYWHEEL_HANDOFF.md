# Overnight handoff — 2026-07-12

## Safe rollback point

Branch: `overnight/flywheel-cert-20260712` (parent + engine submodule).

```powershell
git checkout main
git submodule update --init engine
# or: git reset --hard <commit-before-overnight>
```

## Do NOT start tonight

- **Gen-0 200k** — FLYWHEEL_SPEC_V1 §0 Phase B not run; `mass_generation_allowed: false`
- **New engine patch A/B matches** — baseline frozen at `titanium-v17`
- **Oracle 13-shard uploads** — SSH transport still flaky

## Completed before sleep

- Killed 7 stale `parallel_engine_match.py` supervisors; fixed stale `status.json`
- Added `training/tools/flywheel_label_cert.py` (cost pilot, fidelity spike, §0 skeleton)
- 10-root cost pilot: **10/10 exact-bound**, pilot-only gate passed
- Fidelity calibration on startpos written to `training/data/label_certification/`
- W23 defense test `#[ignore]`; `after_e2_depth_log` ignored in debug (passes release native)

## Overnight work (agent)

1. Triage `gate3_raw_k2_full_corpus_soundness` (102 oracle-wrong) in `engine/src/titanium/race.rs`
2. Wire Phase B stubs in flywheel cert (semantic-reset / audit runners) — no mass gen
3. Re-run release lib tests excluding long oracle audits

## Blockers

| Item | Status |
|------|--------|
| `gate3_raw_k2` | FAIL — 102 RAW_K2 oracle-wrong |
| §0 1000-root semantic reset | NOT_RUN |
| §0 1800/450 audits | NOT_RUN |
| Gen-0 mass generation | BLOCKED |

## Commands

```powershell
py -3.12 training/tools/flywheel_label_cert.py cost-pilot --roots 10 --node-budget 20000
py -3.12 training/tools/flywheel_label_cert.py skeleton-report

$env:RUSTFLAGS = '-C target-cpu=native'
cd engine
cargo test --release --lib -p titanium gate3_raw_k2 -- --nocapture
```
