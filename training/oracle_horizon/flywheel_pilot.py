"""Create a bounded, supervised oracle-horizon pilot plan.

The dry-run is intentionally declarative: it never launches a coordinator,
producer, endless loop, engine, or trainer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .bands import active_bands_for_pilot
    from .config import PilotConfig
    from .data_mix import pilot_mix
    from .safety import CRITICAL_FLAGS
    from .search_ladder import DEFAULT_DEPLOYMENT_NODES, ladder_stages
except ImportError:  # direct ``py training/.../flywheel_pilot.py``
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from training.oracle_horizon.bands import active_bands_for_pilot
    from training.oracle_horizon.config import PilotConfig
    from training.oracle_horizon.data_mix import pilot_mix
    from training.oracle_horizon.safety import CRITICAL_FLAGS
    from training.oracle_horizon.search_ladder import DEFAULT_DEPLOYMENT_NODES, ladder_stages


def design_manifest(config: PilotConfig | None = None) -> dict:
    config = config or PilotConfig()
    return {
        "format": "oracle-horizon-pilot-v1",
        "loop": {
            "A": "Generate",
            "B": "Mine",
            "C": "Relabel",
            "D": "Train",
            "E": "Cheap screen",
            "F": "Full gate",
            "G": "Accept or quarantine",
            "H": "Staleness audit",
        },
        "label_classes": [
            "EXACT_ORACLE", "ORACLE_BACKED_MINIMAX",
            "ORACLE_SUPPORTED_PARTIAL", "SEARCH_ONLY",
        ],
        "bands": {"definitions": {"0": "0", "1": "1-2", "2": "3-4", "3": "5-8", "4": "9-16", "5": ">16"},
                  "active": sorted(active_bands_for_pilot())},
        "search_ladder": {"multipliers": ladder_stages(), "deployment_nodes": DEFAULT_DEPLOYMENT_NODES},
        "mix": pilot_mix(),
        "safety": {"fail_closed": True, "pause_flags": sorted(CRITICAL_FLAGS)},
        "budgets": {
            "max_candidate_positions": config.max_candidate_positions,
            "max_generated_games": config.max_generated_games,
            "max_deep_search_cpu_hours": config.max_deep_search_cpu_hours,
            "max_retained_rows": config.max_retained_rows,
            "screen_games": config.screen_games,
            "full_gate_games": config.full_gate_games,
        },
        "training": {"retain_only_exact_or_backed": config.retain_only_exact_or_backed,
                     "curriculum_mix": config.curriculum_mix, "book_mode": config.book_mode_training},
    }


def write_dry_run(out_dir: Path) -> dict:
    config = PilotConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = design_manifest(config)
    (out_dir / "DESIGN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cycle = {
        "dry_run": True, "unattended": config.unattended_repeat, "book_off": config.book_mode_training == "off",
        "unresolved_cannot_be_exact": True, "started_coordinator": False,
        "started_producer": False, "started_training": False, "started_endless_loop": False,
    }
    (out_dir / "CYCLE0_DRY_RUN.json").write_text(json.dumps(cycle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Oracle-horizon pilot v1\n\n"
        "This is a supervised first cycle. Review DESIGN_MANIFEST.json and "
        "CYCLE0_DRY_RUN.json, then run screening and full-gate matches manually. "
        "Do not enable unattended repeat, book mode, coordinator, producer, or "
        "training until safety review accepts the bounded artifacts.\n",
        encoding="utf-8",
    )
    return {"out_dir": str(out_dir), **cycle, "manifest": str(out_dir / "DESIGN_MANIFEST.json")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error("only --dry-run is supported; no live loop is implemented")
    print(json.dumps(write_dry_run(args.out_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
