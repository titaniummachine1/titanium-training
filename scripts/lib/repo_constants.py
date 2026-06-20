"""Shared repository paths and teacher-dataset identity for doctor and Oracle tooling."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ACTIVE_TEACHER_DATASET = REPO_ROOT / "training" / "data" / "teacher_dataset"
ACTIVE_MANIFEST_PATH = ACTIVE_TEACHER_DATASET / "manifest.json"
ACTIVE_MANIFEST_SHA256 = "31a422f25a8c701ebfa72410f59fab9dff52c2717e30985a3f8e6929be007d02"

APPROVED_CANDIDATE_MANIFEST_SHA256 = "95fa0dd5c7f5d0376bfcfd89d0933341adc5c495ce5268c6559bb16a2e23c38c"

APPROVED_DATASET_COUNTS = {
    "positions": 1_405_888,
    "labels": 2_281_163,
    "observations": 1_454_824,
    "unique_policies": 1_927_597,
    "has_policy_labels": 2_275_885,
    "labels_without_policy": 5_278,
    "policy_quarantined": 0,
    "unresolved": 0,
}

AUDIT_TIMESTAMP = "20260620T101843Z"
AUDIT_PAYLOAD_SHA256 = "03244540a4db0e2a857491afde8c96a3c2240a8bfe29d75737a7615e87290d93"
FINAL_EVIDENCE_ENVELOPE_SHA256 = "7b2d9297c4fe7fd9e85ce81f300568d3ab92d955a82bb01403cf9b4500076b44"
TEST_EVIDENCE_SHA256 = "4f6fc09f35e95494d934a065451160688204d899818efc9b2eade30a4b1af785"
PROMOTION_RECEIPT_SHA256 = "97e69900c95232f516c337f1861025ad8af44f56893d16c19904b5ffaeacf92b"

PROVENANCE_V10 = REPO_ROOT / "training" / "teacher_dataset" / "candidate_provenance" / "teacher_dataset_v10.json"
PROMOTION_RECEIPT = REPO_ROOT / "training" / "teacher_dataset" / "candidate_provenance" / "teacher_dataset_v10_promotion_receipt.json"

ROLLBACK_GLOB = "training/data/teacher_dataset_rollback_*"

FORBIDDEN_BUNDLE_PREFIXES = (
    ".git/",
    ".cleanup_quarantine/",
    "dist/",
    ".pytest-temp/",
    "pytest-temp/",
    "training/data/teacher_dataset_rollback_",
    "training/data/teacher_dataset_candidate",
    "training/checkpoints/",
    "training/checkpoints_smoke/",
    "training/runs/",
    "training/experiments/",
    "training/.pytest-temp/",
    "KaAiData/",
    "web/node_modules/",
    "engine/target/",
    "tools/position_store_importer/target/",
    "__pycache__/",
    ".pytest_cache/",
)

ORACLE_CODE_PATHS = (
    "README.md",
    "docs/",
    "scripts/",
    "training/",
    "tools/position_store_importer/",
    ".env.example",
    "training/requirements.txt",
    "training/requirements-teacher-dataset.txt",
)

CANONICAL_DOCS = (
    "docs/README.md",
    "docs/REPOSITORY_MAP.md",
    "docs/ARCHITECTURE.md",
    "docs/DATASET.md",
    "docs/TRAINING.md",
    "docs/ORACLE_DEPLOYMENT.md",
    "docs/ENGINE_INTEGRATION.md",
    "docs/OPERATIONS.md",
    "docs/TROUBLESHOOTING.md",
    "docs/ROADMAP.md",
)
