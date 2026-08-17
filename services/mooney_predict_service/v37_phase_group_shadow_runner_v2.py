"""Compatibility entry point for the legacy V3.7 pickle import namespace."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _configure_legacy_model_import() -> None:
    explicit = os.environ.get("MOONEY_MODEL_PACKAGE_DIR", "").strip()
    if not explicit:
        return
    package_dir = Path(explicit).resolve()
    pipeline_package_root = str(package_dir.parents[1])
    if pipeline_package_root not in sys.path:
        sys.path.insert(0, pipeline_package_root)


_configure_legacy_model_import()

from .v37_phase_group_shadow_runner import main  # noqa: E402


if __name__ == "__main__":
    main()
