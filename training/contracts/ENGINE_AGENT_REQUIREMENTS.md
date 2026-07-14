# Engine agent requirements — frozen semantics contract

While `TRAINING_PREP_ONLY=1`, no corpus row, label, or training checkpoint may be
produced against unfrozen or unknown engine semantics.

## Required contract fields

Every finalized corpus manifest must include:

- `engine_semantic_version`
- `game_rules_version`
- `canonical_state_version`
- `move_encoding_version`
- `nnue_feature_schema_version`
- `evaluation_semantics_version`
- `score_band_version`
- `oracle_semantics_version`
- `search_label_semantics_version`
- `zobrist_version`
- `binary_sha256`
- `source_commit`
- `dirty_tree_hash` (null if clean tree)
- `generated_at`

Schema: `training/contracts/engine_semantics.schema.json`  
Implementation: `training/engine_semantic_contract.py`

## Compatibility classes

| Class | Meaning |
|-------|---------|
| `compatible` | Byte-identical semantics hash — rows may mix |
| `relabel_required` | Board encoding unchanged; eval/score/search labels must be recomputed |
| `regeneration_required` | Rules, state key, move encoding, or zobrist changed — discard corpus |
| `invalid` | Missing, unknown, or placeholder field — reject |

Unknown or missing semantics are **never** compatible.

## Agent obligations

1. Bump the appropriate version field on every semantic change.
2. Record `dirty_tree_hash` when the source tree is not clean at generation time.
3. Never train on rows whose manifest semantics hash differs from the frozen launch hash.
4. Never claim DIVERSITY_SPEC_V1 compliance until real diversity lanes pass the certificate.
5. Do not create `training/APPROVE_GENERATION.json` without explicit human approval.

## Canonical corpus key components

Finalized sampled rows deduplicate on:

- pawn positions
- horizontal wall topology
- vertical wall topology
- both wall stocks
- side to move
- rule-state fields
- `game_rules_version`
- `canonical_state_version`

## Temporary opening filter

`TEMPORARY_GARBAGE_FILTER_NOT_DIVERSITY_COMPLIANCE` in `game_opening_gate.py` rejects
wall-first garbage only. It does **not** satisfy or imply `N_eff(2) >= 16`.

The sequence `e2 e8 e3 e7` is permitted only for:

- deploy collapse detection (`opening_sanity.py`)
- strength/Elo evaluation openings (`strength_gate.py`)
- regression documentation and tests

It must never appear in training SQL filters, sampling weights, or acceptance logic.
