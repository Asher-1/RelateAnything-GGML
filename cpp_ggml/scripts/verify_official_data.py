#!/usr/bin/env python3
"""Verify a local official dataset snapshot against its Hub revision."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="maelic/RA-4M")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "test_data/official")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "benchmarks/ra4m_verification.json")
    args = parser.parse_args()
    try:
        from huggingface_hub import HfApi
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise SystemExit(f"huggingface_hub and pyarrow are required: {exc}") from exc

    info = HfApi().dataset_info(args.dataset, files_metadata=True)
    directory = args.root / args.dataset.replace("/", "_")
    missing = []
    size_mismatches = []
    split_rows = defaultdict(int)
    expected_bytes = 0
    for sibling in info.siblings or []:
        relative = sibling.rfilename
        expected_size = sibling.size
        local = directory / relative
        expected_bytes += expected_size or 0
        if not local.is_file():
            missing.append(relative)
            continue
        if expected_size is not None and local.stat().st_size != expected_size:
            size_mismatches.append({"file": relative, "expected": expected_size,
                                    "actual": local.stat().st_size})
            continue
        if relative.startswith("data/") and relative.endswith(".parquet"):
            source = parquet.ParquetFile(local)
            if "image" not in source.schema_arrow.names:
                raise RuntimeError(f"image column is missing: {local}")
            split_rows[Path(relative).name.split("-", 1)[0]] += source.metadata.num_rows

    result = {
        "dataset": args.dataset,
        "revision": info.sha,
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "local_path": str(directory),
        "expected_files": len(info.siblings or []),
        "expected_bytes": expected_bytes,
        "missing": missing,
        "size_mismatches": size_mismatches,
        "parquet_image_rows": dict(sorted(split_rows.items())),
        "status": "complete" if not missing and not size_mismatches else "incomplete",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
