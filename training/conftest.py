"""pytest configuration for training tests.

Windows workaround: stale pytest-current junction under TEMP may raise
PermissionError during cleanup_dead_symlinks at session teardown.
"""
from __future__ import annotations

import sys

if sys.platform == "win32":
    import _pytest.pathlib as _pathlib_mod

    _ORIG_CLEANUP_DEAD = _pathlib_mod.cleanup_dead_symlinks

    def _safe_cleanup_dead_symlinks(root):  # type: ignore[no-untyped-def]
        try:
            _ORIG_CLEANUP_DEAD(root)
        except PermissionError as exc:
            if "pytest-current" not in str(exc):
                raise  # re-raise anything that is not the known stale junction

    _pathlib_mod.cleanup_dead_symlinks = _safe_cleanup_dead_symlinks
