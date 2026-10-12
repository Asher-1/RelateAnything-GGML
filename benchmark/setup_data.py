#!/usr/bin/env python3
"""Download OV-SGG-Bench and its evaluation images, ready for run_benchmark.py.

    pip install -e ".[train,hub]"
    python benchmark/setup_data.py                     # every axis (about 15 GB of images)
    python benchmark/setup_data.py --axes a1 a4 a6     # only what these axes read

1. ``maelic/OV-SGG-Bench`` (packs, negatives, text space, shared detections) goes to
   ``runs/ovsgg_bench`` and is linked into ``runs/packed``, ``runs/datamix`` and ``runs/detect``.
2. The images go to ``$RA_DATASETS`` (default ``datasets/``) under the folder names the packs
   record. Only the images of the packs' test splits are kept.
3. Every pack is pointed at ``$RA_DATASETS``, and every image it names is checked.

Resumable: finished sources are skipped. Images come from their sources:

| source       | axes   | from                                        | size   |
|--------------|--------|---------------------------------------------|--------|
| VG150        | A1, A3 | maelic/VG150-coco-format (test)             | 4.2 GB |
| PSG          | A1, A3, A4, A5 | maelic/PSG-coco-format (test)       | 0.4 GB |
| IndoorVG     | A1, A3 | maelic/IndoorVG-coco-format (test)          | 0.5 GB |
| HICO-DET     | A1, A2 | maelic/HICO-DET-coco-format (test)          | 1.1 GB |
| Haystack     | A2     | the authors' SA-1B subset (SA-1B license)   | 7.7 GB |
| SpatialSense | A6     | Zenodo record 8104370                       | 1.8 GB |
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from relsgg.paths import DATASETS, DATAMIX, PACKED, RUNS  # noqa: E402

BENCH_REPO = "maelic/OV-SGG-Bench"
HAYSTACK_URL = "https://myweb.rz.uni-augsburg.de/~lorenjul/haystack/v1/sa1b-subset.zip"
SPATIALSENSE_URL = "https://zenodo.org/api/records/8104370/files/images.tar.gz/content"

# source -> (packs whose images it provides, folder under $RA_DATASETS)
SOURCES = {
    "vg150": (["vg150"], "VG150_coco_format/test"),
    "psg": (["psg"], "PSG_coco_format/test"),
    "indoorvg": (["indoorvg"], "IndoorVG_coco_format/test"),
    "hicodet": (["hicodet"], "HICO_DET/test_images"),
    "haystack": (["haystack"], "Haystack/images"),
    "spatialsense": (
        ["spatialsense_test", "spatialsense_valid"],
        "SpatialSense/images",
    ),
}
AXES = {
    "a1": ["vg150", "psg", "indoorvg", "hicodet"],
    "a2": ["haystack", "hicodet"],
    "a3": ["vg150", "psg", "indoorvg"],
    "a4": ["psg"],
    "a5": ["psg"],
    "a6": ["spatialsense"],
}
# packs evaluated on their test split (the others carry only a train/ for the vocabulary)
TEST_SPLIT = {
    "vg150": "test",
    "psg": "test",
    "indoorvg": "test",
    "hicodet": "test",
    "haystack": "test",
    "spatialsense_test": "test",
    "spatialsense_valid": "test",
}


# ---------------------------------------------------------------- the dataset
def fetch_bench(revision: str | None) -> Path:
    from huggingface_hub import snapshot_download

    local = RUNS / "ovsgg_bench"
    print(f"[bench] {BENCH_REPO} -> {local}")
    snapshot_download(
        BENCH_REPO, repo_type="dataset", local_dir=str(local), revision=revision
    )
    return local


def link_into(target_dir: Path, sources: list[Path]) -> None:
    """Relative symlinks target_dir/<name> -> source; an existing entry is kept."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for src in sources:
        dst = target_dir / src.name
        if dst.exists() or dst.is_symlink():
            if not (dst.is_symlink() and dst.resolve() == src.resolve()):
                print(f"[link] {dst} exists and is kept")
            continue
        dst.symlink_to(os.path.relpath(src, target_dir))


def link(local: Path) -> None:
    link_into(
        PACKED,
        sorted((local / "packs").iterdir())
        + [local / "text_student_v2_512", local / "datamix_v22"],
    )
    link_into(DATAMIX, sorted((local / "datamix").iterdir()))
    link_into(RUNS / "detect", sorted((local / "detect").iterdir()))
    print(f"[link] {PACKED}, {DATAMIX} and {RUNS / 'detect'} point into {local}")


# ---------------------------------------------------------------- the images
def wanted_names(packs: list[str]) -> set[str]:
    names = set()
    for p in packs:
        names |= set(json.load(open(PACKED / p / TEST_SPLIT[p] / "file_names.json")))
    return names


def from_parquet(
    repo: str, splits: list[str], out: Path, keep: set[str], reencode: bool = False
) -> None:
    """Write the images of `keep` from the parquet shards of a dataset's `splits`;
    shards are deleted after use. `reencode` writes RGB JPEG at quality 95, as the
    HICO-DET packs' copies were made; otherwise the bytes are written unchanged.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    out.mkdir(parents=True, exist_ok=True)
    todo = {n for n in keep if not (out / n).exists()}
    shards = sorted(
        f
        for f in HfApi().list_repo_files(repo, repo_type="dataset")
        if f.endswith(".parquet") and any(f.startswith(f"data/{s}-") for s in splits)
    )
    for i, shard in enumerate(shards):
        if not todo:
            break
        path = hf_hub_download(
            repo, shard, repo_type="dataset", cache_dir=str(DATASETS / ".cache")
        )
        batches = pq.ParquetFile(path).iter_batches(
            columns=["file_name", "image"], batch_size=128
        )
        for batch in batches:
            images = batch.column("image").to_pylist()
            names = batch.column("file_name").to_pylist()
            for name, im in zip(names, images):
                if name not in todo:
                    continue
                tmp = out / (name + ".part")
                if reencode:
                    from PIL import Image

                    Image.open(io.BytesIO(im["bytes"])).convert("RGB").save(
                        tmp, "JPEG", quality=95
                    )
                else:
                    tmp.write_bytes(im["bytes"])
                os.replace(tmp, out / name)
                todo.discard(name)
        os.remove(os.path.realpath(path))
        print(
            f"[images] {repo} {shard}: {len(keep) - len(todo):,}/{len(keep):,}",
            flush=True,
        )
    if todo:
        sys.exit(
            f"[images] {repo}: {len(todo):,} images not found, e.g. {sorted(todo)[:3]}"
        )


def download(url: str, dst: Path) -> None:
    """Resumable HTTP download; only a complete file gets the final name."""
    if dst.exists():
        return
    part = dst.with_suffix(dst.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(
        url, headers={"Range": f"bytes={done}-"} if done else {}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        if done and r.status != 206:  # the server ignored the range: start over
            done = 0
        total = done + int(r.headers.get("Content-Length", 0))
        with open(part, "ab" if done else "wb") as f:
            while chunk := r.read(1 << 22):
                f.write(chunk)
                done += len(chunk)
                print(
                    f"\r[download] {dst.name}: {done / 1e9:.2f}/{total / 1e9:.2f} GB",
                    end="",
                    flush=True,
                )
    print()
    os.replace(part, dst)


def from_archive(url: str, out: Path, marker_dir: str) -> None:
    """Download an archive and move the folder holding `marker_dir` (a glob) to `out`."""
    if (out / ".complete").exists():
        return
    tmp = DATASETS / ".download"
    tmp.mkdir(parents=True, exist_ok=True)
    arc = tmp / url.rstrip("/").split("/")[-1 if not url.endswith("/content") else -2]
    download(url, arc)
    unpack = tmp / (arc.name + ".unpacked")
    shutil.rmtree(unpack, ignore_errors=True)
    print(f"[images] unpacking {arc.name}")
    if zipfile.is_zipfile(arc):
        zipfile.ZipFile(arc).extractall(unpack)
    else:
        with tarfile.open(arc) as t:
            t.extractall(unpack, filter="data")
    hits = [p for p in unpack.rglob(marker_dir) if p.is_dir()]
    if not hits:
        sys.exit(f"[images] {arc.name}: no {marker_dir} folder inside")
    shutil.rmtree(out, ignore_errors=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(hits[0].parent), str(out))
    shutil.rmtree(unpack, ignore_errors=True)
    arc.unlink()
    (out / ".complete").touch()


def fetch_images(source: str) -> None:
    packs, folder = SOURCES[source]
    out = DATASETS / folder
    print(f"[images] {source} -> {out}")
    if source in ("vg150", "psg", "indoorvg"):
        repo = {
            "vg150": "maelic/VG150-coco-format",
            "psg": "maelic/PSG-coco-format",
            "indoorvg": "maelic/IndoorVG-coco-format",
        }[source]
        from_parquet(repo, ["test"], out, wanted_names(packs))
    elif source == "hicodet":
        from_parquet(
            "maelic/HICO-DET-coco-format",
            ["test"],
            out,
            wanted_names(packs),
            reencode=True,
        )
    elif source == "haystack":
        from_archive(HAYSTACK_URL, out, "sa_0*")  # SA-1B shard folders
    elif source == "spatialsense":
        from_archive(SPATIALSENSE_URL, out, "flickr")


# ---------------------------------------------------------------- relocation
def relocate_and_check(splits: list[tuple[str, str]]) -> int:
    """Point every split of each (pack, split)'s pack at $RA_DATASETS
    (training/relocate_packs.py), then count the images the named splits are missing."""
    from training.relocate_packs import relocated

    root = str(DATASETS.resolve())
    missing = 0
    for p in sorted({pack for pack, _ in splits}):
        for meta_path in sorted((PACKED / p).glob("*/meta.json")):
            meta = json.load(open(meta_path))
            new = relocated(meta.get("img_dir_built", meta["img_dir"]), root)
            if meta["img_dir"] != new:
                meta.setdefault("img_dir_built", meta["img_dir"])
                meta["img_dir"] = new
                tmp = meta_path.with_suffix(".json.tmp")
                json.dump(meta, open(tmp, "w"))
                os.replace(tmp, meta_path)
    for p, split in splits:
        d = PACKED / p / split
        names = json.load(open(d / "file_names.json"))
        img_dir = Path(json.load(open(d / "meta.json"))["img_dir"])
        lost = sum(not (img_dir / n).is_file() for n in names)
        missing += lost
        print(
            f"[check] {p}/{split}: {len(names) - lost:,}/{len(names):,} images in {img_dir}"
        )
    return missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--axes",
        nargs="+",
        default=sorted(AXES),
        choices=sorted(AXES),
        help="download only the images these axes read (default: all)",
    )
    ap.add_argument(
        "--revision", default=None, help="OV-SGG-Bench revision (default: main)"
    )
    ap.add_argument(
        "--skip-images", action="store_true", help="only the dataset and the links"
    )
    a = ap.parse_args()

    link(fetch_bench(a.revision))
    sources = sorted({s for axis in a.axes for s in AXES[axis]})
    if not a.skip_images:
        for source in sources:
            fetch_images(source)
    packs = sorted({p for s in sources for p in SOURCES[s][0]})
    missing = relocate_and_check([(p, TEST_SPLIT[p]) for p in packs])
    if missing:
        sys.exit(f"{missing:,} images missing; rerun to resume")
    print(
        f"\nready: python benchmark/run_benchmark.py --checkpoint maelic/relsgg-vits16plus "
        f"--axes {' '.join(a.axes)}"
    )


if __name__ == "__main__":
    main()
