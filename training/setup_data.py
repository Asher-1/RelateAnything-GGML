#!/usr/bin/env python3
"""Download everything train.sh reads to retrain the released mixture.

    pip install -e ".[train,hub]"
    hf auth login                  # the DINOv3 backbones are gated: accept the license on
                                   # https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m
    python training/setup_data.py  # about 52 GB of images
    ./train.sh

1. ``maelic/OV-SGG-Bench`` (the hicodet and psg packs, the text space, the text student, the
   IndoorVG holdout) and the packs of ``maelic/RA-4M`` (megasg, megasg_clean, vg_raw,
   megasg_proxy50k), linked into ``runs/packed`` and ``runs/datamix``.
2. The images, into ``$RA_DATASETS`` (default ``datasets/``) under the folder names the packs
   record, keeping only the images the packs name:

   | pack                        | from                                  | size   |
   |-----------------------------|---------------------------------------|--------|
   | megasg_clean (train), megasg (val) | maelic/RA-4M                   | 43 GB  |
   | vg_raw (train)              | maelic/VG150-coco-format (train, val) | 5.3 GB |
   | hicodet (train)             | maelic/HICO-DET-coco-format (train)   | 3.7 GB |
   | psg (val, checkpoint selection) | maelic/PSG-coco-format (val)      | 0.2 GB |

3. Every pack is pointed at ``$RA_DATASETS`` and every image it names is checked.

``--proxy`` takes ``megasg_proxy50k`` instead of ``megasg_clean`` (6 GB of MegaSG images
instead of 43 GB) for a development run on one GPU; ``--mix zeroshot`` leaves out HICO-DET.
Resumable: rerun after an interruption.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from benchmark.setup_data import (  # noqa: E402
    fetch_bench,
    from_parquet,
    link,
    link_into,
    relocate_and_check,
)
from relsgg.paths import DATASETS, PACKED, RUNS  # noqa: E402

RA4M = "maelic/RA-4M"
BACKBONE = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
# (pack, split) -> (dataset, its splits holding the images, folder under $RA_DATASETS, re-encode)
SOURCES = {
    ("megasg_clean", "train"): (RA4M, ["train"], "MEGASG/train", False),
    ("megasg_proxy50k", "train"): (RA4M, ["train"], "MEGASG/train", False),
    ("megasg", "val"): (RA4M, ["val"], "MEGASG/val", False),
    ("vg_raw", "train"): (
        "maelic/VG150-coco-format",
        ["train", "val"],
        "VG150_coco_format/train",
        False,
    ),
    ("hicodet", "train"): (
        "maelic/HICO-DET-coco-format",
        ["train"],
        "HICO_DET/train_images",
        True,
    ),
    ("psg", "val"): ("maelic/PSG-coco-format", ["val"], "PSG_coco_format/val", False),
}


def fetch_ra4m_packs(revision: str | None) -> None:
    from huggingface_hub import snapshot_download

    local = RUNS / "ra4m"
    print(f"[ra4m] {RA4M} packs -> {local}")
    snapshot_download(
        RA4M,
        repo_type="dataset",
        local_dir=str(local),
        allow_patterns=["packs/*"],
        revision=revision,
    )
    link_into(PACKED, sorted((local / "packs").iterdir()))


def check_backbone(repo: str) -> None:
    """The DINOv3 repositories are gated: say so now rather than at the first step."""
    from huggingface_hub import auth_check
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        auth_check(repo)
        print(f"[backbone] {repo}: accessible")
    except (GatedRepoError, RepositoryNotFoundError):
        print(
            f"[backbone] {repo} is gated: accept its license on "
            f"https://huggingface.co/{repo}, then run `hf auth login`"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--mix",
        choices=["released", "zeroshot"],
        default="released",
        help="zeroshot: the evaluation control without HICO-DET",
    )
    ap.add_argument(
        "--proxy",
        action="store_true",
        help="megasg_proxy50k instead of megasg_clean (development runs)",
    )
    ap.add_argument("--backbone", default=BACKBONE, help="checked for access only")
    ap.add_argument("--revision", default=None, help="dataset revision (default: main)")
    ap.add_argument(
        "--skip-images", action="store_true", help="only the packs and the links"
    )
    a = ap.parse_args()

    link(fetch_bench(a.revision))
    fetch_ra4m_packs(a.revision)
    splits = [
        ("megasg_proxy50k" if a.proxy else "megasg_clean", "train"),
        ("vg_raw", "train"),
        ("megasg", "val"),
        ("psg", "val"),
    ]
    if a.mix == "released":
        splits.append(("hicodet", "train"))
    if not a.skip_images:
        for pack, split in splits:
            repo, src_splits, folder, reencode = SOURCES[pack, split]
            keep = set(json.load(open(PACKED / pack / split / "file_names.json")))
            print(
                f"[images] {pack}/{split}: {len(keep):,} images from {repo} -> "
                f"{DATASETS / folder}"
            )
            from_parquet(repo, src_splits, DATASETS / folder, keep, reencode=reencode)
    missing = relocate_and_check(splits)
    check_backbone(a.backbone)
    if missing:
        sys.exit(f"{missing:,} images missing; rerun to resume")
    data = (
        ["--data_roots", "runs/packed/megasg_proxy50k", "runs/packed/vg_raw"]
        + (["runs/packed/hicodet"] if a.mix == "released" else [])
        if a.proxy
        else []
    )
    print(
        "\nready: "
        + (f"MIX={a.mix} " if a.mix != "released" else "")
        + "./train.sh"
        + (" " + " ".join(data) if data else "")
    )


if __name__ == "__main__":
    main()
