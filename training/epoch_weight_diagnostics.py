"""Aggregate and log per-epoch training weight distribution by source tier."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class EpochWeightDiagnostics:
    tiers: list[str] = field(default_factory=list)
    phases: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)

    def record_batch(
        self,
        *,
        tiers: list[str] | None,
        phases: list[str] | None,
        weights,
    ) -> None:
        if tiers is None or phases is None or weights is None:
            return
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        n = min(len(tiers), len(phases), len(w))
        if n <= 0:
            return
        self.tiers.extend(str(t) for t in tiers[:n])
        self.phases.extend(str(p) for p in phases[:n])
        self.weights.extend(float(x) for x in w[:n])

    def _percentile(self, arr: np.ndarray, q: float) -> float:
        if arr.size == 0:
            return 0.0
        return float(np.percentile(arr, q))

    def _phase_summary(self, total_samples: int, total_mass: float) -> list[str]:
        phase_counts: dict[str, int] = defaultdict(int)
        phase_mass: dict[str, float] = defaultdict(float)
        for phase, weight in zip(self.phases, self.weights, strict=True):
            phase_counts[phase] += 1
            phase_mass[phase] += float(weight)
        lines = ["  weight diagnostics by game phase (global):"]
        for phase in ("opening", "midgame", "endgame"):
            count = phase_counts.get(phase, 0)
            mass = phase_mass.get(phase, 0.0)
            sample_share = 100.0 * count / total_samples if total_samples else 0.0
            mass_share = 100.0 * mass / total_mass if total_mass else 0.0
            lines.append(
                f"    [{phase}] samples={count:,} ({sample_share:.1f}% of rows)  "
                f"loss_mass={mass:.3f} ({mass_share:.1f}% of weighted loss)"
            )
        return lines

    def format_report(self) -> str:
        if not self.weights:
            return "  weight diagnostics: no weighted samples recorded"

        w_all = np.asarray(self.weights, dtype=np.float64)
        total_samples = len(self.weights)
        total_mass = float(w_all.sum())
        lines = [
            "  weight diagnostics by source tier:",
            f"    total_samples={total_samples:,}  total_loss_mass={total_mass:.3f}",
        ]
        lines.extend(self._phase_summary(total_samples, total_mass))
        lines.append("")

        by_tier: dict[str, list[float]] = defaultdict(list)
        phase_by_tier: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for tier, phase, weight in zip(self.tiers, self.phases, self.weights, strict=True):
            by_tier[tier].append(float(weight))
            phase_by_tier[tier][phase] += 1

        tier_order = sorted(by_tier.keys(), key=lambda t: (-sum(by_tier[t]), t))
        for tier in tier_order:
            arr = np.asarray(by_tier[tier], dtype=np.float64)
            count = int(arr.size)
            mass = float(arr.sum())
            sample_share = 100.0 * count / total_samples
            mass_share = 100.0 * mass / total_mass if total_mass > 0 else 0.0
            phases = phase_by_tier[tier]
            phase_bits = ", ".join(f"{k}={phases[k]}" for k in sorted(phases))
            lines.extend(
                [
                    f"    [{tier}]",
                    f"      samples={count:,} ({sample_share:.1f}% of rows)  "
                    f"loss_mass={mass:.3f} ({mass_share:.1f}% of weighted loss)",
                    f"      mean={arr.mean():.4f}  median={float(np.median(arr)):.4f}  "
                    f"p90={self._percentile(arr, 90):.4f}  "
                    f"p99={self._percentile(arr, 99):.4f}  max={arr.max():.4f}",
                    f"      phases: {phase_bits}",
                ]
            )
        return "\n".join(lines)

    def log(self) -> None:
        print(self.format_report(), flush=True)

    def write_json(self, path: Path) -> None:
        out: dict = {
            "total_samples": len(self.weights),
            "total_loss_mass": float(sum(self.weights)),
            "phases": {},
            "tiers": {},
        }
        phase_counts: dict[str, int] = defaultdict(int)
        phase_mass: dict[str, float] = defaultdict(float)
        for phase, weight in zip(self.phases, self.weights, strict=True):
            phase_counts[phase] += 1
            phase_mass[phase] += float(weight)
        total_samples = len(self.weights)
        total_mass = out["total_loss_mass"]
        for phase in ("opening", "midgame", "endgame"):
            count = phase_counts.get(phase, 0)
            mass = phase_mass.get(phase, 0.0)
            out["phases"][phase] = {
                "sample_count": count,
                "sample_share_pct": 100.0 * count / total_samples if total_samples else 0.0,
                "loss_mass": mass,
                "loss_mass_share_pct": 100.0 * mass / total_mass if total_mass else 0.0,
            }
        by_tier: dict[str, list[float]] = defaultdict(list)
        phase_by_tier: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for tier, phase, weight in zip(self.tiers, self.phases, self.weights, strict=True):
            by_tier[tier].append(float(weight))
            phase_by_tier[tier][phase] += 1
        total_samples = len(self.weights)
        total_mass = out["total_loss_mass"]
        for tier, weights in sorted(by_tier.items()):
            arr = np.asarray(weights, dtype=np.float64)
            out["tiers"][tier] = {
                "sample_count": int(arr.size),
                "sample_share_pct": 100.0 * arr.size / total_samples if total_samples else 0.0,
                "loss_mass": float(arr.sum()),
                "loss_mass_share_pct": 100.0 * float(arr.sum()) / total_mass if total_mass else 0.0,
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "p90": float(np.percentile(arr, 90)),
                "p99": float(np.percentile(arr, 99)),
                "max": float(arr.max()),
                "phases": dict(phase_by_tier[tier]),
            }
        Path(path).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
