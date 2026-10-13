#!/usr/bin/env python3
"""Write the RA-4M images from the maelic/RA-4M parquet shards to <out>/<split>/<file_name>.

The packs (runs/packed/megasg, megasg_clean, megasg_proxy50k) name their images by file_name
under MEGASG/<split>/. maelic/RA-4M carries those images inline (497,308 rows in 100 shards,
~44 GB) with the same file names, so this needs neither JosephZ/mega_1m nor the manifest. The
JPEG bytes are written unchanged. Shards are downloaded one at a time and deleted after use, so
peak disk is one shard (~0.5 GB) plus the images (~41 GB).

    pip install -e ".[train,hub]"
    python release/ra4m/export_images.py --out "$RA_DATASETS/MEGASG"             # train + val
    python release/ra4m/export_images.py --out "$RA_DATASETS/MEGASG" --split val

Resumable: files already present in <out> are skipped.
"""
import argparse
import os
import time

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO = "maelic/RA-4M"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="image root; images go to <out>/<split>/")
    ap.add_argument("--split", nargs="+", default=["train", "val"], choices=["train", "val"])
    ap.add_argument("--keep-shards", action="store_true", help="do not delete shards after use")
    ap.add_argument("--cache", default=None, help="HF cache dir for the shards")
    a = ap.parse_args()

    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
                   if f.startswith("data/") and f.endswith(".parquet"))
    for split in a.split:
        shards = [f for f in files if os.path.basename(f).startswith(split + "-")]
        dst = os.path.join(a.out, split)
        os.makedirs(dst, exist_ok=True)
        written = present = 0
        start = time.time()
        for i, shard in enumerate(shards):
            path = hf_hub_download(REPO, shard, repo_type="dataset", cache_dir=a.cache)
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(columns=["file_name", "image"], batch_size=256):
                for name, image in zip(batch.column("file_name").to_pylist(),
                                       batch.column("image").to_pylist()):
                    target = os.path.join(dst, name)
                    if os.path.exists(target):
                        present += 1
                        continue
                    with open(target + ".part", "wb") as fh:
                        fh.write(image["bytes"])
                    os.replace(target + ".part", target)
                    written += 1
            if not a.keep_shards:
                os.remove(os.path.realpath(path))
            print(f"[{split}] shard {i + 1}/{len(shards)}: {written:,} written, "
                  f"{present:,} already present, {time.time() - start:.0f}s", flush=True)
        print(f"[{split}] {written + present:,} images in {dst}")


if __name__ == "__main__":
    main()
