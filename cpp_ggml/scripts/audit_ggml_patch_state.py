#!/usr/bin/env python3
"""Prove that every dirty file in the pinned GGML checkout comes from a patch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


def run(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    ggml = root / "cpp_ggml" / "third_party" / "ggml"
    patch_dir = root / "cpp_ggml" / "third_party" / "patches"
    patches = sorted(patch_dir.glob("*.patch"))
    patch_paths: set[str] = set()
    patch_records = []
    for patch in patches:
        paths = []
        for line in patch.read_text(encoding="utf-8").splitlines():
            if line.startswith("diff --git a/"):
                path = line.split(" b/", 1)[1]
                paths.append(path)
                patch_paths.add(path)
        patch_records.append({"name": patch.name, "sha256": sha256(patch), "paths": paths})

    tracked = [x for x in run(ggml, "diff", "--name-only", "HEAD").splitlines() if x]
    untracked = [
        x for x in run(ggml, "ls-files", "--others", "--exclude-standard").splitlines() if x
    ]
    dirty = sorted(set(tracked + untracked))
    with tempfile.TemporaryDirectory(prefix="ra-ggml-patch-audit-") as directory:
        env = os.environ.copy()
        env["GIT_INDEX_FILE"] = str(Path(directory) / "index")
        subprocess.run(["git", "-C", str(ggml), "read-tree", "HEAD"], env=env, check=True)
        for patch in patches:
            subprocess.run(
                ["git", "-C", str(ggml), "apply", "--cached", str(patch)],
                env=env,
                check=True,
            )
        exact_match = subprocess.run(
            ["git", "-C", str(ggml), "diff", "--quiet"], env=env, check=False
        ).returncode == 0
        unexpected = sorted(
            x for x in subprocess.check_output(
                ["git", "-C", str(ggml), "ls-files", "--others"],
                text=True,
                env=env,
            ).splitlines() if x
        )
    payload = {
        "ggml_head": run(ggml, "rev-parse", "HEAD"),
        "ggml_version": "v0.21.0",
        "patches": patch_records,
        "dirty_paths": dirty,
        "unexpected_dirty_paths": unexpected,
        "all_dirty_paths_patch_owned": set(dirty) <= patch_paths,
        "exact_patched_tree_match": exact_match and not unexpected,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if unexpected or not exact_match:
        raise SystemExit("GGML differs from the exact pinned-commit-plus-patches tree")


if __name__ == "__main__":
    main()
