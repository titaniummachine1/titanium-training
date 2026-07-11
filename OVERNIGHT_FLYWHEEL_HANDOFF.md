# Overnight handoff — 2026-07-12 (updated 01:20)

## Safe rollback point

Branch: `overnight/flywheel-cert-20260712` (parent + engine submodule).

```powershell
git checkout master
git submodule update --init engine
```

## Live overnight processes

| Process | Log |
|---------|-----|
| `flywheel_label_cert.py phase-b-all` | `training/data/overnight_logs/phase_b_all.log` |
| Race gate lib tests | `training/data/overnight_logs/lib_test_race_gates.log` |
| 30m agent loop | sentinel `AGENT_LOOP_WAKE_FLYWHEEL` |

## Resolved since sleep

- **`gate3_raw_k2_full_corpus_soundness` — FIXED** (0 oracle-wrong). Root cause: vacuous AND win when all defender moves exceeded slack budget but legal moves existed. Fix in `engine/src/titanium/race.rs` (`nm == 0` guard).

## Phase B (running)

Sequence in `phase-b-all`:
1. 1000-root semantic-reset equivalence
2. 1800 partial audit @ 20k nodes
3. 450 exhaustive audit @ 200k nodes
4. Drift canaries 180 + 45
5. If all pass + cost pilot exact → **Gen-0 pilot 15k** (not 200k)

Reports: `training/data/label_certification/`

## Blockers cleared / remaining

| Item | Status |
|------|--------|
| `gate3_raw_k2` | **PASSED** (0 oracle-wrong) |
| §0 1000-root semantic reset | **RUNNING** |
| §0 1800/450 audits | **RUNNING** (queued in phase-b-all) |
| Gen-0 200k mass | **BLOCKED** by spec |
| Gen-0 15k pilot | Starts automatically when Phase B + cost pilot pass |

## Do NOT start

- Gen-0 **200k** until prefix pilot shows ≥2% regret improvement
- New engine patch A/B matches — baseline frozen at `titanium-v17`
- Oracle 13-shard uploads — SSH still flaky

## Commands

```powershell
# Check progress
Get-Content training\data\overnight_logs\phase_b_all.log -Tail 30
py -3.12 training\tools\flywheel_label_cert.py skeleton-report

# Re-run single gate
$env:RUSTFLAGS = '-C target-cpu=native'
cd engine
cargo test --release --lib -p titanium gate3_raw_k2 -- --nocapture

# Stop flywheel cert only (not loop)
Get-Process python | Where-Object { $_.CommandLine -like '*phase-b-all*' } | Stop-Process
```
