#!/usr/bin/env python3
"""Evaluate every retained RA-4M image with the released Python checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODELS = ("relsgg-vits16", "relsgg-vits16plus", "relsgg-vitb16")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(rows: list[dict], key: str) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return {name: {
        "images": len(group),
        "gt_relations": sum(item["gt_count"] for item in group),
        "exact_matches": sum(item["matches"] for item in group),
        "exact_gt_recall_at_50": (
            sum(item["matches"] for item in group) / sum(item["gt_count"] for item in group)
        ),
        "latency_ms_median": statistics.median(item["latency_ms"] for item in group),
    } for name, group in sorted(grouped.items())}


def evaluate(model, sample: dict, image_path: Path, device: str, repeats: int,
             torch, np, Image) -> dict:
    timings = []
    predictions = None
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
        boxes = np.asarray([[x, y, x + w, y + h] for x, y, w, h in
                            (obj["bbox"] for obj in sample["objects"])], dtype=np.float32)
        labels = [obj["category"] for obj in sample["objects"]]
        predictions = model.predict(image, boxes, box_labels=labels, topk=50)
        if device == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - started) * 1000)
    gt = {(int(r["subject"]), int(r["object"]), r["predicate"])
          for r in sample["relations"]}
    pred = [(int(p.subject_idx), int(p.object_idx), p.predicate, float(p.score))
            for p in predictions]
    matched = gt & {(s, o, name) for s, o, name, _ in pred}
    matched_at_20 = gt & {(s, o, name) for s, o, name, _ in pred[:20]}
    pair_matches = {(s, o) for s, o, _ in gt} & {(s, o) for s, o, _, _ in pred}
    return {
        "image_id": sample["image_id"], "source": sample["source"],
        "difficulty": sample["difficulty"], "scene_groups": sample["scene_groups"],
        "objects": len(sample["objects"]), "gt_count": len(gt),
        "pred_count": len(pred), "matches": len(matched),
        "matches_at_20": len(matched_at_20), "pair_matches_at_50": len(pair_matches),
        "gt_pairs": len({(s, o) for s, o, _ in gt}),
        "latency_ms": statistics.median(timings), "latency_samples_ms": timings,
        "matched_gt": [list(item) for item in sorted(matched)],
        "predictions": [{"subject": s, "object": o, "predicate": name, "score": score}
                        for s, o, name, score in pred],
    }


def predicate_diagnostics(rows: list[dict], samples: list[dict]) -> dict:
    support: Counter = Counter()
    hits: Counter = Counter()
    for row, sample in zip(rows, samples):
        for _, _, predicate in {
            (int(r["subject"]), int(r["object"]), r["predicate"])
            for r in sample["relations"]
        }:
            support[predicate] += 1
        hits.update(predicate for _, _, predicate in row["matched_gt"])
    buckets = {"rare_1": [], "common_2_to_4": [], "frequent_5_plus": []}
    for predicate, count in support.items():
        bucket = "rare_1" if count == 1 else "common_2_to_4" if count < 5 else "frequent_5_plus"
        buckets[bucket].append(predicate)
    return {
        "macro_exact_gt_recall_at_50": statistics.mean(hits[name] / count
                                                          for name, count in support.items()),
        "predicate_classes": len(support),
        "support_buckets": {bucket: {
            "classes": len(names),
            "gt_relations": sum(support[name] for name in names),
            "exact_matches": sum(hits[name] for name in names),
            "macro_exact_gt_recall_at_50": (statistics.mean(hits[name] / support[name]
                                                               for name in names) if names else None),
        } for bucket, names in buckets.items()},
    }


def render_contact_sheet(data: Path, samples: list[dict], output: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    columns, cell_w, cell_h = 5, 230, 208
    canvas = Image.new("RGB", (columns * cell_w, 10 * cell_h), "#f5f7f7")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("DejaVuSans.ttf", 12)
    for index, sample in enumerate(samples):
        x0, y0 = index % columns * cell_w, index // columns * cell_h
        with Image.open(data / sample["image"]) as raw:
            image = raw.convert("RGB")
            image.thumbnail((cell_w - 10, 162))
        canvas.paste(image, (x0 + (cell_w - image.width) // 2, y0 + 3))
        draw.text((x0 + 6, y0 + 166),
                  f'{sample["source"]} | {sample["difficulty"]} | {sample["image_id"]}',
                  fill="#19333d", font=font)
        draw.text((x0 + 6, y0 + 183),
                  f'{len(sample["objects"])} objects / {len(sample["relations"])} GT edges',
                  fill="#36545c", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def render_metrics(results: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    colors = ("#007a74", "#cd7940", "#4c68a5")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for index, (model, result) in enumerate(results.items()):
        axes[0].bar(index, result["overall"]["exact_gt_recall_at_50"], color=colors[index])
        axes[1].bar(index, result["overall"]["latency_ms_median"], color=colors[index])
        sources = result["by_source"]
        for j, source in enumerate(("coco", "objects365", "openimages")):
            axes[2].bar(j + (index - 1) * 0.26,
                        sources[source]["exact_gt_recall_at_50"], width=0.25,
                        color=colors[index], label=model if j == 0 else None)
    for axis, title, ylabel in zip(axes[:2],
                                    ("Exact GT recall @ 50", "Median end-to-end latency"),
                                    ("Recall", "ms / image")):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(range(3), MODELS, rotation=25, ha="right")
    axes[2].set_title("Exact GT recall by source")
    axes[2].set_ylabel("Recall")
    axes[2].set_xticks(range(3), ("COCO", "Objects365", "Open Images"))
    axes[2].legend(fontsize=8)
    fig.suptitle("RA-4M 50-image diagnostic | 803 annotated relations | full vocabulary")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "cpp_ggml/test_data/ra4m_50")
    parser.add_argument("--models-root", type=Path, default=ROOT / "cpp_ggml/models/pytorch")
    parser.add_argument("--model", action="append", choices=MODELS, help="repeatable; all three by default")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--repeats", type=int, default=3, help="complete image-to-triplet calls per sample")
    parser.add_argument("--output", type=Path, default=ROOT / "cpp_ggml/benchmarks/ra4m_50_python.json")
    parser.add_argument("--plot", type=Path, default=ROOT / "cpp_ggml/benchmarks/ra4m_50_python.png")
    parser.add_argument("--contact-sheet", type=Path,
                        default=ROOT / "cpp_ggml/benchmarks/ra4m_50_contact_sheet.png")
    args = parser.parse_args()
    try:
        import numpy as np
        import torch
        from PIL import Image
        from relsgg import RelateAnything
    except ImportError as exc:
        parser.error(f"inference dependency unavailable: {exc}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    data = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    samples = data["samples"]
    if len(samples) != 50:
        parser.error("the compact dataset must contain exactly 50 images")
    render_contact_sheet(args.data, samples, args.contact_sheet)
    results = {}
    for name in args.model or MODELS:
        checkpoint = args.models_root / name / "model.pth"
        model = RelateAnything.from_checkpoint(str(checkpoint), device=args.device,
                                                full_vocabulary=True)
        with Image.open(args.data / samples[0]["image"]) as raw:
            warmup_image = raw.convert("RGB")
        boxes = np.asarray([[x, y, x + w, y + h] for x, y, w, h in
                            (obj["bbox"] for obj in samples[0]["objects"])], dtype=np.float32)
        model.predict(warmup_image, boxes, topk=50)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        rows = []
        for index, sample in enumerate(samples, 1):
            rows.append(evaluate(model, sample, args.data / sample["image"], args.device, args.repeats,
                                 torch, np, Image))
            print(f"{name}: {index}/50 image={sample['image_id']} latency={rows[-1]['latency_ms']:.1f}ms", flush=True)
        overall = {
            "images": len(rows), "gt_relations": sum(row["gt_count"] for row in rows),
            "exact_matches": sum(row["matches"] for row in rows),
            "exact_gt_recall_at_50": sum(row["matches"] for row in rows) /
                                      sum(row["gt_count"] for row in rows),
            "exact_gt_recall_at_20": sum(row["matches_at_20"] for row in rows) /
                                      sum(row["gt_count"] for row in rows),
            "pair_recall_at_50": sum(row["pair_matches_at_50"] for row in rows) /
                                 sum(row["gt_pairs"] for row in rows),
            "latency_ms_median": statistics.median(row["latency_ms"] for row in rows),
            "latency_ms_p95": sorted(row["latency_ms"] for row in rows)[47],
            "estimated_throughput_images_per_s_from_medians": (
                1000 * len(rows) / sum(row["latency_ms"] for row in rows)
            ),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated())
                                         if args.device == "cuda" else None,
        }
        overall.update(predicate_diagnostics(rows, samples))
        results[name] = {
            "checkpoint_sha256": sha256(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "vocabulary_size": len(model.predicates),
            "parameter_dtype": str(next(model.model.parameters()).dtype),
            "overall": overall,
            "by_source": metrics(rows, "source"), "by_difficulty": metrics(rows, "difficulty"),
            "rows": rows,
        }
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
    report = {
        "dataset": data["dataset"], "dataset_revision": data["revision"],
        "compact_manifest_sha256": sha256(args.data / "manifest.json"),
        "device": args.device,
        "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU",
        "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "repeats_per_image": args.repeats,
        "warmup": "one first-image prediction per model before peak-memory reset",
        "timing_scope": "PIL image load/decode, box conversion, model.predict and triplet decoding with CUDA synchronization; excludes checkpoint and vocabulary loading; per-image median of repeats",
        "protocol": "exact (subject index, object index, predicate string) GT recall @20/@50; pair recall @50; per-class macro recall; full checkpoint vocabulary; GT boxes; no masks",
        "limitations": "RA-4M annotation coverage is incomplete; this diagnostic is not upstream OVS-F1 or precision. No C++ full-graph output exists to compare.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if len(results) == len(MODELS):
        render_metrics(results, args.plot)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
