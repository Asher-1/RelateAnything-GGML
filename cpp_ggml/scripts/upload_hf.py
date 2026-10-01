#!/usr/bin/env python3
"""Upload GGUF artifacts after explicit authenticated user invocation."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Asher-1/relateanything-ggml-full")
    parser.add_argument("--models-dir", type=Path, default=Path(__file__).resolve().parents[1] / "models" / "gguf")
    parser.add_argument("--card", type=Path, default=Path(__file__).resolve().parents[1] / "models" / "HF_MODEL_CARD.md")
    parser.add_argument("--token", help="Hugging Face token; pass explicitly or use the CLI credential store")
    parser.add_argument("--create", action="store_true", help="create the model repository if absent")
    parser.add_argument("--commit-message", default="Upload RelateAnything GGUF artifacts")
    args = parser.parse_args()
    try:
        from gguf import GGUFReader
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required for upload") from exc
    if not args.models_dir.exists():
        raise SystemExit(f"models directory does not exist: {args.models_dir}")
    models = sorted(args.models_dir.glob("*.gguf"))
    if not models:
        raise SystemExit("no official converted GGUF models to upload; test fixtures stay under test_data/fixtures")
    if any(not path.name.startswith("relsgg-") for path in models):
        raise SystemExit("GGUF upload accepts only relsgg-* official checkpoint names")
    for path in models:
        reader = GGUFReader(str(path))
        graph = reader.fields.get("ra.graph_kind")
        graph_name = bytes(graph.parts[-1]).decode("utf-8") if graph is not None else ""
        if not graph_name or graph_name == "relation_pair_linear_v1":
            raise SystemExit(f"{path} is not a converted official image graph")
    api = HfApi(token=args.token)
    if args.create:
        api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(repo_id=args.repo_id, repo_type="model", folder_path=str(args.models_dir),
                      path_in_repo="gguf", allow_patterns=["relsgg-*.gguf", "README.md"], commit_message=args.commit_message)
    if args.card.exists():
        api.upload_file(repo_id=args.repo_id, repo_type="model", path_or_fileobj=str(args.card),
                        path_in_repo="README.md", commit_message="Document verified native full-graph models and measurements")
    print(f"uploaded {args.models_dir} to https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
