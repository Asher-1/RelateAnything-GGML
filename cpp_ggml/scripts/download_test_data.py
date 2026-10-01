#!/usr/bin/env python3
"""Download declared public test datasets without hidden global state."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_DATASETS = ("maelic/RA-4M", "maelic/VG150-coco-format",
                    "maelic/PSG-coco-format", "maelic/IndoorVG-coco-format")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=Path(__file__).resolve().parents[1] / "test_data/official")
    parser.add_argument("--dataset", action="append", help="Hugging Face dataset repository (repeatable; defaults to available official dataset metadata)")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--allow-large", action="store_true", help="download the complete dataset snapshot")
    args = parser.parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required; install it before downloading datasets") from exc
    args.dest.mkdir(parents=True, exist_ok=True)
    allow_patterns = None if args.allow_large else [
        "README*", "categories.json", "predicates.json", "export_stats.json",
        "clean_train_excluded.json",
    ]
    failures = []
    for dataset in tuple(args.dataset or DEFAULT_DATASETS):
        try:
            path = snapshot_download(repo_id=dataset, repo_type="dataset", revision=args.revision,
                                     local_dir=str(args.dest / dataset.replace("/", "_")),
                                     allow_patterns=allow_patterns)
        except Exception as exc:  # surface gated/private/renamed repositories without hiding others
            failures.append(f"{dataset}: {exc}")
            continue
        print(f"downloaded {dataset}@{args.revision} to {path}")
    if failures:
        for failure in failures:
            print(f"download failed: {failure}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
