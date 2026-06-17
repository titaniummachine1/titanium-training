#!/usr/bin/env python3
"""Deprecated Ka single-position teacher worker.

The continuous training stack no longer launches this worker. It remains as a
fail-closed compatibility stub so stale scripts do not create teacher caches.
"""

from __future__ import annotations


def main() -> int:
    print("Ka teacher worker is deprecated and disabled.")
    print("Training now uses completed games only: moves_bin + final WDL outcome.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
