#!/usr/bin/env python3
"""Build maelic/HICO-DET-coco-format: the HICO-DET images and RelateAnything's COCO-SGG
annotations, in the layout of maelic/PSG-coco-format.

    python release/build_hicodet_coco.py --ann <dir with hicodet_{train,test}_coco.json> \\
        --out /tmp/hicodet_coco
    hf upload maelic/HICO-DET-coco-format /tmp/hicodet_coco --repo-type dataset --private

Rows: ``image`` (the JPEG bytes of zhimeng/hico_det, unchanged), ``image_id``, ``width``,
``height``, ``file_name``, ``objects`` (COCO boxes, xywh) and ``relations`` (subject,
object and predicate ids). The two source files number categories and predicates per
split; here both splits share one list: the 80 COCO classes in COCO order (ids 1-80,
HICO-DET spellings) and the 116 interaction predicates in alphabetical order (ids 0-115).
The COCO json of each split is written beside the parquet shards with the same ids.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SOURCE = "zhimeng/hico_det"
TEMPLATE = (
    "maelic/PSG-coco-format"  # its parquet schema and `datasets` features are reused
)
COCO80 = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic_light",
    "fire_hydrant",
    "stop_sign",
    "parking_meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports_ball",
    "kite",
    "baseball_bat",
    "baseball_glove",
    "skateboard",
    "surfboard",
    "tennis_racket",
    "bottle",
    "wine_glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot_dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted_plant",
    "bed",
    "dining_table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell_phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy_bear",
    "hair_drier",
    "toothbrush",
]


def template_schema() -> pa.Schema:
    from huggingface_hub import HfApi, HfFileSystem

    shard = next(
        f
        for f in HfApi().list_repo_files(TEMPLATE, repo_type="dataset")
        if f.startswith("data/val-") and f.endswith(".parquet")
    )
    with HfFileSystem().open(f"datasets/{TEMPLATE}/{shard}", "rb") as fh:
        return pq.ParquetFile(fh).schema_arrow


def canonical(split_json: dict, predicates: list[str]) -> dict:
    """The split's COCO json with the shared category and predicate ids."""
    cat = {c["id"]: COCO80.index(c["name"]) + 1 for c in split_json["categories"]}
    pred = {r["id"]: predicates.index(r["name"]) for r in split_json["rel_categories"]}
    return {
        "images": split_json["images"],
        "annotations": [
            {**a, "category_id": cat[a["category_id"]]}
            for a in split_json["annotations"]
        ],
        "categories": [
            {"id": i + 1, "name": n, "supercategory": "none"}
            for i, n in enumerate(COCO80)
        ],
        "rel_categories": [{"id": i, "name": n} for i, n in enumerate(predicates)],
        "rel_annotations": [
            {"id": i, **r, "predicate_id": pred[r["predicate_id"]]}
            for i, r in enumerate(split_json["rel_annotations"])
        ],
    }


def rows_of(coco: dict) -> dict:
    """file_name -> the row without its image."""
    objects, relations = {}, {}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        objects.setdefault(a["image_id"], []).append(
            {
                "area": float(w * h),
                "bbox": [float(v) for v in a["bbox"]],
                "category_id": a["category_id"],
                "id": a["id"],
                "iscrowd": 0,
                "segmentation": [],
            }
        )
    for r in coco["rel_annotations"]:
        relations.setdefault(r["image_id"], []).append(
            {
                "id": r["id"],
                "object_id": r["object_id"],
                "predicate_id": r["predicate_id"],
                "subject_id": r["subject_id"],
            }
        )
    return {
        im["file_name"]: {
            "image_id": im["id"],
            "width": im["width"],
            "height": im["height"],
            "file_name": im["file_name"],
            "objects": objects.get(im["id"], []),
            "relations": relations.get(im["id"], []),
        }
        for im in coco["images"]
    }


def write_split(
    split: str, rows: dict, schema: pa.Schema, out: Path, per_shard: int
) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    n_shards = math.ceil(len(rows) / per_shard)
    sources = sorted(
        f
        for f in HfApi().list_repo_files(SOURCE, repo_type="dataset")
        if f.startswith(f"data/{split}-") and f.endswith(".parquet")
    )
    buf, written, seen = [], 0, set()

    def flush():
        nonlocal buf, written
        path = out / "data" / f"{split}-{written:05d}-of-{n_shards:05d}.parquet"
        pq.write_table(
            pa.Table.from_pylist(buf, schema=schema), path, row_group_size=500
        )
        print(f"[{split}] {path.name}: {len(buf)} rows", flush=True)
        written += 1
        buf = []

    for src in sources:
        local = hf_hub_download(SOURCE, src, repo_type="dataset")
        for batch in pq.ParquetFile(local).iter_batches(
            columns=["image"], batch_size=256
        ):
            for im in batch.column("image").to_pylist():
                name = os.path.basename(im.get("path") or "")
                if name not in rows or name in seen:
                    continue
                seen.add(name)
                buf.append(
                    {"image": {"bytes": im["bytes"], "path": name}, **rows[name]}
                )
                if len(buf) == per_shard:
                    flush()
        os.remove(os.path.realpath(local))
    if buf:
        flush()
    missing = set(rows) - seen
    if missing or written != n_shards:
        raise SystemExit(
            f"[{split}] {len(missing)} images not in {SOURCE}: {sorted(missing)[:3]}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--ann", required=True, help="directory with hicodet_{train,test}_coco.json"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-shard", type=int, default=3000)
    a = ap.parse_args()

    out = Path(a.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(exist_ok=True)
    splits = {
        s: json.load(open(Path(a.ann) / f"hicodet_{s}_coco.json"))
        for s in ("train", "test")
    }
    names = {r["name"] for s in splits.values() for r in s["rel_categories"]}
    predicates = sorted(names)
    schema = template_schema()
    for split, raw in splits.items():
        coco = canonical(raw, predicates)
        json.dump(coco, open(out / "annotations" / f"{split}.json", "w"))
        write_split(split, rows_of(coco), schema, out, a.per_shard)
    cats = canonical(splits["train"], predicates)
    json.dump(
        {"categories": cats["categories"], "rel_categories": cats["rel_categories"]},
        open(out / "categories.json", "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
