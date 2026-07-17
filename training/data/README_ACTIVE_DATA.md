# Active Training Data

Use `teacher_dataset_good` for value/NNUE training.

Good sources currently included:
- `friend_selfplay:*`
- `quoridor-zero.ink`
- KA / Ishtar cohorts

Do not train value NNUE on collapsed Titanium self-play:
- `titanium-overnight`
- `titanium-selfplay`
- `titanium-native`
- `pool-titanium*`
- `random-titanium*`
- `self-match`

Misleading or old artifacts are archived under:

`training/data/archive/excluded_from_training/`

Old 545-wide feature caches are archived there too. Rebuild caches from
`teacher_dataset_good` with the current engine/schema.
