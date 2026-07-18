# QBR LMR reference (next strength experiment)

**Status:** READY — architecture/cache/SMP baseline is frozen; open as an
isolated LMR A/B only.  
**Priority:** First strength experiment after the 2026-07-18 cache lock.

## Claim

External Quoridor engine (QBR) LMR is reportedly much stronger than Titanium’s current LMR.

Cited merge evidence (from extract header):

- Merged E-011: SPRT(0,+5) H1, n=1028, **57.9%**, Elo **+55.2** [34.2, 76.7] @50ms
- Production: `lmr=true`, `lmr_probable_walls=false` → uniform **1-ply** reductions in practice
- Source: `rust/src/search.rs` @ master `70dab79`

## What makes it different (from extract)

1. **Eligibility:** depth≥3, ordered_index≥6, walls only (not pawns), not TT move, not “tight”
2. **Tight-wall classifier:** wall touches either pawn’s BFS **shortest-path edge set** → never reduced
3. **Reduction + verified re-search:** reduce then re-search on fail-high (classic verified LMR)
4. **Optional probable-walls tier** (dormant in production config): 1 vs 2 ply by classifier

Titanium already has CAT/path LMR and route-touch ideas; this is a **candidate redesign / A-B**, not a blind copy.

## Frozen baseline (do not change during the gate)

```text
cache architecture: locked
LazySMP: locked
TT: unchanged
NNUE: unchanged
eval: 21-bit f32
dist: inline-16
```

## Experiment protocol (avoid implementation drift)

Before games:

1. **Exact change only** — QBR LMR code path; no unrelated cleanup, move ordering, or pruning knobs.
2. **Declare SPRT first** — target Elo, accepted loss, draw handling, game count / confidence bounds.
3. **Mechanical checks** — perft if applicable; search correctness; instrument reduce/re-search; watch node counts.
4. **A/B** — same binary except LMR; same hardware; same TC; enough games for the pre-declared SPRT stop.

## Files in this folder

| File | Role |
|------|------|
| `qbr_lmr_extract.rs` | Exact LMR eligibility / tight edges / reduce+re-search snippets |
| `search.rs` | Full QBR search.rs reference (~0.4 MB) for context |

Original Downloads paths (owner machine):

- `C:\Users\Terminatort8000\Downloads\qbr_lmr_extract.rs`
- `C:\Users\Terminatort8000\Downloads\search.rs`

## Do not

- Mix into cache, TT, LazySMP, TM, or NNUE work
- Enable probable-walls tier without its own gate
- Assume CAT “route touch” ≡ QBR “tight edges” without measuring
- Bundle move-order or pruning-parameter changes with the LMR patch
