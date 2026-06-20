# Repository cleanup plan

Generated: 2026-06-20T11:52:58.735413+00:00

## Summary

- Files inventoried: 51,027
- Tracked: 201
- Delete candidates: 37,386
- Merge candidates: 2

## Delete candidates (proven dead / generated)

- `dist/oracle_upload_code/training/__pycache__/collect_reduction_counterfactuals.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/collect_reduction_counterfactuals_v3.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/collect_search_importance.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/color_rotation.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/compare_pressure_sources.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/conftest.cpython-312-pytest-9.1.1.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/coordinator.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/datagen.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/datagen.cpython-314.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/engine_identity.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/field_planes.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/halfpw.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/housekeeping.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/import_clipboard_game.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/ka_api_teacher.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/ka_teacher_worker.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/manifest.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/manifest.cpython-314.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/migrate_sparse_routes.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/move_codec.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/nnue_guards.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/nnue_learning_metrics.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/opponent_curriculum.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/parse_flamegraph.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/plateau_probe.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/pool_labels.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/pool_preflight.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/position_store_compact.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/position_store_config.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/position_store_friend.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/position_store_guards.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/position_store_lib.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/position_store_migration.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/position_store_split.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/position_store_state.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/position_store_teacher.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/reduction_counterfactual_schema.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/run_nnue_cycle.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/run_search_pressure_experiment.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/run_swiss_overnight.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/supervise.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/swiss_tournament.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/swiss_tournament.cpython-314.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/test_color_rotation.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/test_evidence_canonical.cpython-312-pytest-9.1.1.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/test_lmr_head_v3.cpython-312-pytest-9.1.1.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/test_lmr_head_v3.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/test_opponent_curriculum.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/test_oracle_bundle.cpython-312-pytest-9.1.1.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/test_position_store.cpython-312-pytest-9.1.1.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/test_position_store.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/test_position_store_migration.cpython-312-pytest-9.1.1.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/test_reduction_counterfactuals.cpython-312-pytest-9.1.1.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/test_reduction_counterfactuals.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/test_run_nnue_cycle.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/test_search_importance.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/test_teacher_dataset.cpython-312-pytest-9.1.1.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/train.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/train_lmr_head_v3.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/train_reduction_sidecar.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/train_reduction_sidecar_v2.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/train_search_importance.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/verify_db_games.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/__pycache__/zero_teacher_client.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/__init__.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/audit_policies.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/build.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/canonical_identity.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/catalog.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/cli.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/config.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/dataset_semantic_parity.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/evidence_canonical.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/evidence_envelope.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/finalize.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/freeze_reference.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/friend_state.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/gate_audits.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/jsonl_miss_audit.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/jsonl_policy_index.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/loader_smoke.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/policy_binary.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/policy_lookup.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/policy_payload_audit.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/policy_recovery.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/position_parity.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/promote.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/promotion_gates.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/reconcile.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/recovery_collision_audit.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/schema.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/sidecar_paths.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/sidecar_policy_index.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/sidecar_reader.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/update_v10_provenance.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/verify_artifacts.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/write_test_evidence.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/teacher_dataset/__pycache__/write_v10_provenance.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/zero_teacher/__pycache__/__init__.cpython-312.pyc` — Build/cache artifact
- `dist/oracle_upload_code/training/zero_teacher/__pycache__/client.cpython-312.pyc` — Build/cache artifact
- … and 37286 more

## Merge / consolidate

- `training/AUDIT_REPORT.md` — Superseded by docs/ — merge or remove
- `training/data/handoff.txt` — Superseded by docs/ — merge or remove
