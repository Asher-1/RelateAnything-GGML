#!/usr/bin/env python3
"""Extract a fixed, diverse 50-image RA-4M validation suite with complete GT."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "cpp_ggml/test_data/official/maelic_RA-4M"
DESTINATION = ROOT / "cpp_ggml/test_data/ra4m_50"
SOURCE_QUOTAS = {"coco": 17, "objects365": 17, "openimages": 16}
DIFFICULTY_QUOTAS = {"moderate": 12, "hard": 38}

SCENE_GROUPS = {
    "people": {"person", "man", "woman", "boy", "girl", "human face"},
    "transport": {"car", "bus", "truck", "train", "bicycle", "motorcycle", "airplane", "boat"},
    "animals": {"dog", "cat", "horse", "bird", "cow", "sheep", "elephant"},
    "food": {"food", "pizza", "cake", "banana", "apple", "bowl", "plate", "cup"},
    "interior": {"chair", "couch", "bed", "table", "lamp", "cabinet", "sofa"},
    "sports": {"ball", "tennis racket", "skateboard", "surfboard", "baseball bat"},
    "work": {"laptop", "keyboard", "computer", "phone", "book", "monitor"},
    "outdoor": {"tree", "flower", "mountain", "road", "building", "traffic light"},
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def overlaps(boxes: list[list[float]]) -> int:
    count = 0
    # With at most 16 regions this pairwise geometry check is bounded by 120 pairs.
    for left in range(len(boxes)):
        x1, y1, w1, h1 = boxes[left]
        for right in range(left + 1, len(boxes)):
            x2, y2, w2, h2 = boxes[right]
            iw = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
            ih = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
            intersection = iw * ih
            union = w1 * h1 + w2 * h2 - intersection
            count += intersection / union > 0.25 if union > 0 else 0
    return count


def candidate(row: dict, shard: str, row_index: int) -> dict | None:
    objects = row["objects"] or []
    relations = row["relations"] or []
    if not (3 <= len(objects) <= 16 and len(relations) >= 5 and
            row["width"] >= 256 and row["height"] >= 256):
        return None
    boxes = [obj["bbox"] for obj in objects]
    if any(w <= 0 or h <= 0 or x < -1 or y < -1 or
           x + w > row["width"] + 1 or y + h > row["height"] + 1
           for x, y, w, h in boxes):
        return None
    if any(not (0 <= rel["subject"] < len(objects) and
                0 <= rel["object"] < len(objects)) for rel in relations):
        return None
    categories = {obj["category"].lower() for obj in objects}
    predicates = {rel["predicate"] for rel in relations}
    groups = {group for group, names in SCENE_GROUPS.items() if categories & names}
    small = sum(w * h / (row["width"] * row["height"]) < 0.025
                for x, y, w, h in boxes)
    overlap = overlaps(boxes)
    has_both_types = len({bool(rel["spatial"]) for rel in relations}) == 2
    hard = len(objects) >= 6 or len(relations) >= 11 or small >= 2 or overlap >= 2
    return {
        "image_id": row["image_id"], "file_name": row["file_name"],
        "source": row["source"], "shard": shard, "row_index": row_index,
        "objects": len(objects), "relations": len(relations),
        "categories": categories, "predicates": predicates, "scene_groups": groups,
        "small_objects": small, "overlapping_pairs": overlap,
        "spatial_and_semantic": has_both_types,
        "difficulty": "hard" if hard else "moderate",
    }


def choose(candidates: list[dict], category_freq: Counter, predicate_freq: Counter) -> list[dict]:
    selected = []
    seen_categories: set[str] = set()
    seen_predicates: set[str] = set()
    seen_groups: set[str] = set()
    source_counts = Counter()
    difficulty_counts = Counter()
    shard_counts = Counter()

    # Fifty bounded greedy rounds reward previously unseen labels and hard geometry.
    # Source and difficulty quotas prevent the greedy score from selecting one scene type.
    for _ in range(50):
        best = None
        best_score = -math.inf
        for item in candidates:
            if item.get("selected") or source_counts[item["source"]] >= SOURCE_QUOTAS[item["source"]]:
                continue
            if difficulty_counts[item["difficulty"]] >= DIFFICULTY_QUOTAS[item["difficulty"]]:
                continue
            novelty = 5 * len(item["categories"] - seen_categories)
            novelty += 3 * len(item["predicates"] - seen_predicates)
            novelty += 12 * len(item["scene_groups"] - seen_groups)
            rarity = 4 * sum(1 / math.sqrt(category_freq[name]) for name in item["categories"])
            rarity += 2 * sum(1 / math.sqrt(predicate_freq[name]) for name in item["predicates"])
            difficulty = min(item["objects"], 12) + min(item["relations"], 20) / 2
            difficulty += min(item["small_objects"], 4) * 2
            difficulty += min(item["overlapping_pairs"], 4) * 2
            difficulty += 3 if item["spatial_and_semantic"] else 0
            coverage = 8 * (SOURCE_QUOTAS[item["source"]] - source_counts[item["source"]])
            coverage += 5 * (10 - shard_counts[item["shard"]])
            score = novelty + rarity + difficulty + coverage
            key = (score, -int(item["image_id"]))
            if best is None or key > (best_score, -int(best["image_id"])):
                best, best_score = item, score
        if best is None:
            raise RuntimeError("not enough valid candidate images for all source/difficulty quotas")
        best["selected"] = True
        best["selection_score"] = round(best_score, 4)
        selected.append(best)
        source_counts[best["source"]] += 1
        difficulty_counts[best["difficulty"]] += 1
        shard_counts[best["shard"]] += 1
        seen_categories.update(best["categories"])
        seen_predicates.update(best["predicates"])
        seen_groups.update(best["scene_groups"])
    if source_counts != SOURCE_QUOTAS or difficulty_counts != DIFFICULTY_QUOTAS:
        raise RuntimeError(f"quota mismatch: {source_counts}, {difficulty_counts}")
    return selected


def verify(destination: Path) -> dict:
    from PIL import Image

    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"]
    if len(samples) != 50 or len({sample["image_id"] for sample in samples}) != 50:
        raise RuntimeError("compact suite must contain 50 distinct image IDs")
    if Counter(sample["source"] for sample in samples) != SOURCE_QUOTAS:
        raise RuntimeError("source quotas changed")
    if Counter(sample["difficulty"] for sample in samples) != DIFFICULTY_QUOTAS:
        raise RuntimeError("difficulty quotas changed")
    expected_images = {sample["image"] for sample in samples}
    actual_images = {str(path.relative_to(destination)) for path in (destination / "images").iterdir()}
    if expected_images != actual_images:
        raise RuntimeError("image set does not match the 50-image manifest")
    for sample in samples:
        path = destination / sample["image"]
        raw = path.read_bytes()
        if sha256(raw) != sample["image_sha256"]:
            raise RuntimeError(f"image hash mismatch: {path}")
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
            if image.size != (sample["width"], sample["height"]):
                raise RuntimeError(f"image dimensions mismatch: {path}")
        count = len(sample["objects"])
        if not sample["relations"] or any(
            not (0 <= rel["subject"] < count and 0 <= rel["object"] < count)
            for rel in sample["relations"]
        ):
            raise RuntimeError(f"invalid relation annotation: {path}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--dest", type=Path, default=DESTINATION)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--prune-original", action="store_true",
                        help="remove the full snapshot only after verifying all 50 extracted samples")
    args = parser.parse_args()
    if args.verify_only:
        manifest = verify(args.dest)
    else:
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise SystemExit(f"pyarrow is required: {exc}") from exc
        if (args.dest / "manifest.json").exists():
            raise SystemExit(f"compact suite already exists: {args.dest}; use --verify-only")
        shards = sorted((args.source / "data").glob("val-*.parquet"))
        if len(shards) != 5:
            raise SystemExit(f"expected five RA-4M validation shards, found {len(shards)}")
        candidates = []
        category_freq: Counter = Counter()
        predicate_freq: Counter = Counter()
        for shard in shards:
            row_index = 0
            for batch in parquet.ParquetFile(shard).iter_batches(
                batch_size=256,
                columns=["image_id", "file_name", "width", "height", "source", "objects", "relations"],
            ):
                for row in batch.to_pylist():
                    item = candidate(row, shard.name, row_index)
                    if item is not None:
                        candidates.append(item)
                        category_freq.update(item["categories"])
                        predicate_freq.update(item["predicates"])
                    row_index += 1
            print(f"scanned {shard.name}: {row_index} rows")
        selected = choose(candidates, category_freq, predicate_freq)
        lookup = {(item["shard"], item["row_index"]): item for item in selected}
        args.dest.mkdir(parents=True, exist_ok=True)
        (args.dest / "images").mkdir(exist_ok=True)
        samples = []
        for shard in shards:
            if not any(item["shard"] == shard.name for item in selected):
                continue
            row_index = 0
            for batch in parquet.ParquetFile(shard).iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    item = lookup.get((shard.name, row_index))
                    row_index += 1
                    if item is None:
                        continue
                    raw = row["image"]["bytes"]
                    if not raw:
                        raise RuntimeError(f"missing image bytes: {item['image_id']}")
                    suffix = Path(row["file_name"]).suffix.lower()
                    if suffix not in (".jpg", ".jpeg", ".png"):
                        suffix = ".jpg"
                    name = f"images/{item['image_id']}{suffix}"
                    (args.dest / name).write_bytes(raw)
                    samples.append({
                        "image_id": item["image_id"], "image": name,
                        "image_sha256": sha256(raw), "file_name": row["file_name"],
                        "width": row["width"], "height": row["height"],
                        "source": row["source"], "shard": shard.name,
                        "row_index": item["row_index"],
                        "selection_score": item["selection_score"],
                        "difficulty": item["difficulty"],
                        "scene_groups": sorted(item["scene_groups"]),
                        "small_objects": item["small_objects"],
                        "overlapping_pairs": item["overlapping_pairs"],
                        "objects": row["objects"], "relations": row["relations"],
                    })
            print(f"extracted {shard.name}: {len(samples)} cumulative images")
        revision = json.loads((ROOT / "cpp_ggml/benchmarks/ra4m_verification.json").read_text())["revision"]
        manifest = {
            "dataset": "maelic/RA-4M", "revision": revision,
            "split": "val", "selection": "deterministic diversity and difficulty greedy v1",
            "source_quotas": SOURCE_QUOTAS, "difficulty_quotas": DIFFICULTY_QUOTAS,
            "candidate_count": len(candidates),
            "samples": sorted(samples, key=lambda sample: sample["image_id"]),
        }
        (args.dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        manifest = verify(args.dest)
    if args.prune_original:
        if args.source.resolve() != SOURCE.resolve() or args.source.is_symlink():
            raise SystemExit("pruning is allowed only for the exact local RA-4M snapshot directory")
        if args.source.exists():
            shutil.rmtree(args.source)
            print(f"removed full snapshot: {args.source}")
    print(json.dumps({"status": "verified", "images": len(manifest["samples"]),
                      "source_counts": dict(Counter(x["source"] for x in manifest["samples"])),
                      "difficulty_counts": dict(Counter(x["difficulty"] for x in manifest["samples"])),
                      "bytes": sum((args.dest / x["image"]).stat().st_size for x in manifest["samples"])},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
