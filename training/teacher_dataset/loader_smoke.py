"""Value-only and value+policy loader smoke tests for candidate datasets."""
from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from position_store_config import ROOT

try:
    import resource
except ImportError:  # Windows
    resource = None  # type: ignore[assignment]


def _peak_rss_bytes() -> int | None:
    if resource is None:
        return None
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if usage.ru_maxrss > 1_000_000:
            return int(usage.ru_maxrss)
        return int(usage.ru_maxrss) * 1024
    except (AttributeError, ValueError):
        return None


@dataclass
class LoaderSmokeResult:
    passed: bool
    row_count: int
    opened_policy_files: bool
    no_policy_rows: int
    policy_rows: int
    expected_no_policy: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "row_count": self.row_count,
            "opened_policy_files": self.opened_policy_files,
            "no_policy_rows": self.no_policy_rows,
            "policy_rows": self.policy_rows,
            "expected_no_policy": self.expected_no_policy,
            "error": self.error,
        }


@dataclass
class LoaderSmokeAudit:
    passed: bool = False
    candidate_dir: str = ""
    value_only: LoaderSmokeResult | None = None
    policy_bearing: LoaderSmokeResult | None = None
    mixed: LoaderSmokeResult | None = None
    random_lookups: int = 0
    sequential_lookups: int = 0
    repeated_lookups: int = 0
    concurrent_lookups: int = 0
    startup_seconds: float = 0.0
    first_batch_seconds: float = 0.0
    steady_state_seconds: float = 0.0
    peak_rss_bytes: int | None = None
    file_opens: int = 0
    bytes_read: int = 0
    record_reads: int = 0
    bin_size_bytes: int = 0
    error: str | None = None
    checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "candidate_dir": self.candidate_dir,
            "value_only": self.value_only.to_dict() if self.value_only else None,
            "policy_bearing": self.policy_bearing.to_dict() if self.policy_bearing else None,
            "mixed": self.mixed.to_dict() if self.mixed else None,
            "random_lookups": self.random_lookups,
            "sequential_lookups": self.sequential_lookups,
            "repeated_lookups": self.repeated_lookups,
            "concurrent_lookups": self.concurrent_lookups,
            "startup_seconds": self.startup_seconds,
            "first_batch_seconds": self.first_batch_seconds,
            "steady_state_seconds": self.steady_state_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
            "file_opens": self.file_opens,
            "bytes_read": self.bytes_read,
            "record_reads": self.record_reads,
            "bin_size_bytes": self.bin_size_bytes,
            "bytes_read_ratio_to_bin": (
                round(self.bytes_read / self.bin_size_bytes, 4) if self.bin_size_bytes else None
            ),
            "error": self.error,
            "checks": self.checks,
        }


def smoke_value_only_loader(candidate_dir: Path, *, root: Path = ROOT) -> LoaderSmokeResult:
    manifest = json.loads((candidate_dir / "manifest.json").read_text(encoding="utf-8"))
    counts = manifest.get("counts") or {}
    labels_path = root / manifest["parts"]["labels"][0]
    table = pq.read_table(labels_path, columns=["has_policy", "value_i16"])
    n = table.num_rows
    expected = int(counts.get("labels", n))
    no_pol = sum(1 for i in range(n) if not bool(table.column("has_policy")[i].as_py()))
    with_pol = n - no_pol
    expected_no = int(counts.get("labels", 0)) - int(counts.get("has_policy_labels", with_pol))
    passed = n == expected and n > 0
    return LoaderSmokeResult(
        passed=passed,
        row_count=n,
        opened_policy_files=False,
        no_policy_rows=no_pol,
        policy_rows=with_pol,
        expected_no_policy=expected_no,
    )


def smoke_value_policy_loader(candidate_dir: Path, *, root: Path = ROOT) -> LoaderSmokeResult:
    from .policy_binary import PolicyChunkReader

    manifest = json.loads((candidate_dir / "manifest.json").read_text(encoding="utf-8"))
    counts = manifest.get("counts") or {}
    labels_path = root / manifest["parts"]["labels"][0]
    bin_path = root / manifest["parts"]["policies"][0]
    idx_path = root / manifest["parts"]["policies"][1]
    table = pq.read_table(labels_path, columns=["has_policy", "policy_record_id"])
    n = table.num_rows
    expected = int(counts.get("labels", n))
    no_pol = 0
    with_pol = 0
    seen_rids: set[int] = set()
    with PolicyChunkReader(bin_path, idx_path) as reader:
        for i in range(n):
            if not bool(table.column("has_policy")[i].as_py()):
                no_pol += 1
                continue
            with_pol += 1
            rid = int(table.column("policy_record_id")[i].as_py())
            if rid not in seen_rids:
                reader.read(rid)
                seen_rids.add(rid)
    expected_no = int(counts.get("labels", 0)) - int(counts.get("has_policy_labels", with_pol))
    passed = (
        n == expected
        and with_pol + no_pol == n
        and no_pol == expected_no
        and with_pol == int(counts.get("has_policy_labels", with_pol))
    )
    return LoaderSmokeResult(
        passed=passed,
        row_count=n,
        opened_policy_files=True,
        no_policy_rows=no_pol,
        policy_rows=with_pol,
        expected_no_policy=expected_no,
    )


def run_loader_smoke_audit(candidate_dir: Path, *, root: Path = ROOT) -> LoaderSmokeAudit:
    """Focused loader smoke with timing, IO, and lookup-pattern coverage."""
    from .policy_binary import PolicyChunkReader

    audit = LoaderSmokeAudit(candidate_dir=str(candidate_dir))
    t0 = time.perf_counter()
    try:
        audit.value_only = smoke_value_only_loader(candidate_dir, root=root)
        manifest = json.loads((candidate_dir / "manifest.json").read_text(encoding="utf-8"))
        counts = manifest.get("counts") or {}
        labels_path = root / manifest["parts"]["labels"][0]
        bin_path = root / manifest["parts"]["policies"][0]
        idx_path = root / manifest["parts"]["policies"][1]
        audit.bin_size_bytes = bin_path.stat().st_size if bin_path.is_file() else 0

        table = pq.read_table(labels_path, columns=["has_policy", "policy_record_id"])
        policy_rids = [
            int(table.column("policy_record_id")[i].as_py())
            for i in range(table.num_rows)
            if bool(table.column("has_policy")[i].as_py())
        ]
        unique_rids = sorted(set(policy_rids))
        audit.startup_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        with PolicyChunkReader(bin_path, idx_path) as reader:
            batch = unique_rids[: min(256, len(unique_rids))]
            for rid in batch:
                reader.read(rid)
            audit.first_batch_seconds = time.perf_counter() - t1

            t2 = time.perf_counter()
            step = max(1, len(unique_rids) // 512)
            for rid in unique_rids[::step][:512]:
                reader.read(rid)
            audit.steady_state_seconds = time.perf_counter() - t2

            if unique_rids:
                rng = random.Random(0)
                sample = [rng.choice(unique_rids) for _ in range(min(128, len(unique_rids)))]
                audit.random_lookups = len(sample)
                for rid in sample:
                    reader.read(rid)

                seq = unique_rids[: min(128, len(unique_rids))]
                audit.sequential_lookups = len(seq)
                for rid in seq:
                    reader.read(rid)

                repeat_rid = unique_rids[0]
                audit.repeated_lookups = 32
                for _ in range(32):
                    reader.read(repeat_rid)

                def _read_batch(rids: list[int]) -> int:
                    n_read = 0
                    for rid in rids:
                        reader.read(rid)
                        n_read += 1
                    return n_read

                chunks = [unique_rids[i::4][:32] for i in range(4)]
                with ThreadPoolExecutor(max_workers=4) as pool:
                    audit.concurrent_lookups = sum(pool.map(_read_batch, chunks))

            audit.file_opens = reader.stats.file_opens
            audit.bytes_read = reader.stats.bytes_read
            audit.record_reads = reader.stats.record_reads

        audit.policy_bearing = smoke_value_policy_loader(candidate_dir, root=root)
        audit.mixed = LoaderSmokeResult(
            passed=audit.value_only.passed and audit.policy_bearing.passed,
            row_count=audit.value_only.row_count,
            opened_policy_files=True,
            no_policy_rows=audit.policy_bearing.no_policy_rows,
            policy_rows=audit.policy_bearing.policy_rows,
            expected_no_policy=audit.policy_bearing.expected_no_policy,
        )

        io_ok = audit.bin_size_bytes == 0 or audit.bytes_read <= audit.bin_size_bytes + idx_path.stat().st_size + 4096
        audit.checks = {
            "value_only_batch": audit.value_only.passed,
            "policy_bearing_batch": audit.policy_bearing.passed,
            "mixed_batch": audit.mixed.passed,
            "no_policy_rows_match_manifest": audit.policy_bearing.no_policy_rows
            == audit.policy_bearing.expected_no_policy,
            "single_bin_load": audit.file_opens == 2,
            "bytes_not_scaled_by_unique_policies": io_ok,
            "random_lookup": audit.random_lookups > 0,
            "sequential_lookup": audit.sequential_lookups > 0,
            "repeated_lookup": audit.repeated_lookups > 0,
            "concurrent_readers": audit.concurrent_lookups > 0,
        }
        audit.passed = all(audit.checks.values())
        audit.peak_rss_bytes = _peak_rss_bytes()
    except Exception as exc:
        audit.error = str(exc)
        audit.passed = False
    return audit
