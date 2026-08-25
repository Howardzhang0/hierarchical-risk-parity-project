"""Create compact provenance manifests without redistributing licensed data."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    path: str | Path,
    *,
    row_count: int,
    asset_count: int,
    start_date: str,
    end_date: str,
    source: str,
    query_id: str | int | None = None,
    duplicate_rows: int = 0,
) -> dict[str, Any]:
    data_path = Path(path)
    return {
        "source": source,
        "wrds_query_id": None if query_id is None else str(query_id),
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "licensed_data_redistributed": False,
        "raw_filename": data_path.name,
        "raw_bytes": data_path.stat().st_size,
        "raw_sha256": sha256_file(data_path),
        "row_count": int(row_count),
        "asset_count": int(asset_count),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "duplicate_asset_date_rows": int(duplicate_rows),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def write_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output

