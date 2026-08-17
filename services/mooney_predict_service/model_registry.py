from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib

from .db import fetch_active_model


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_active_package() -> tuple[object, dict, dict]:
    meta = fetch_active_model()
    package_dir = Path(str(meta["PackagePath"]))
    model_path = package_dir / "hybrid_model.joblib"
    schema_path = package_dir / "feature_metadata.json"
    if not model_path.is_file() or not schema_path.is_file():
        raise FileNotFoundError(f"Incomplete model package: {package_dir}")
    registered_hash = str(meta["PackageHash"] or "").upper()
    actual_hash = sha256_file(model_path)
    if registered_hash and registered_hash != actual_hash:
        raise RuntimeError("Active model package hash does not match MooneyModel")
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    return joblib.load(model_path), schema, meta

