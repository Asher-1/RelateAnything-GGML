#!/usr/bin/env python3
"""Measure the official RelateAnything Python pipeline for every local checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_ORDER = ("relsgg-vits16", "relsgg-vits16plus", "relsgg-vitb16")
VOCABULARY = ["riding", "carrying a rider", "beside", "in front of"]
BOXES = [[470.0, 130.0, 650.0, 630.0], [90.0, 310.0, 1010.0, 875.0]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, default=ROOT / "cpp_ggml/models/pytorch")
    parser.add_argument("--model-name", action="append", choices=MODEL_ORDER,
                        help="checkpoint to run (repeatable; defaults to all three)")
    parser.add_argument("--image", type=Path, default=ROOT / "assets/reel/images/horse.jpg")
    parser.add_argument("--boxes-json", help="JSON array of [x1,y1,x2,y2] boxes")
    parser.add_argument("--box-label", action="append", help="display label for each box (repeatable)")
    parser.add_argument("--vocabulary", action="append", help="relation phrase (repeatable)")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "cpp_ggml/benchmarks/pytorch_cuda_results.jsonl")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be >= 0 and --repeats must be >= 1")

    try:
        import numpy as np
        import torch
        from PIL import Image
        from relsgg import RelateAnything
    except ImportError as exc:
        raise SystemExit(f"Python benchmark dependencies are unavailable: {exc}") from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    image_path = args.image if args.image.is_absolute() else ROOT / args.image
    if image_path.resolve() != (ROOT / "assets/reel/images/horse.jpg").resolve() and not args.boxes_json:
        parser.error("a custom --image requires --boxes-json")
    image = Image.open(image_path).convert("RGB")
    boxes_data = json.loads(args.boxes_json) if args.boxes_json else BOXES
    boxes = np.asarray(boxes_data, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or len(boxes) < 2 or not np.isfinite(boxes).all():
        parser.error("--boxes-json must contain at least two finite [x1,y1,x2,y2] boxes")
    if np.any(boxes[:, 2:] <= boxes[:, :2]):
        parser.error("each box must have x2>x1 and y2>y1")
    labels = args.box_label or (["person", "horse"] if not args.boxes_json else
                                [f"object {index}" for index in range(len(boxes))])
    if len(labels) != len(boxes):
        parser.error("the number of --box-label values must equal the number of boxes")
    vocabulary = args.vocabulary or VOCABULARY
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for model_name in tuple(args.model_name or MODEL_ORDER):
        started = time.perf_counter()
        checkpoint = (args.models_root / model_name / "model.pth").resolve()
        record: dict[str, object] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "backend": "pytorch-cuda" if args.device == "cuda" else "pytorch-cpu",
            "device": args.device,
            "model": model_name,
            "checkpoint": str(checkpoint),
            "image": str(image_path),
            "boxes": boxes.tolist(),
            "box_labels": labels,
            "vocabulary": vocabulary,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "host": platform.platform(),
            "status": "error",
        }
        try:
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            record["checkpoint_sha256"] = sha256(checkpoint)
            if args.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            model = RelateAnything.from_checkpoint(str(checkpoint), device=args.device)
            model.set_vocabulary(vocabulary)
            for _ in range(args.warmup):
                model.predict(image, boxes, box_labels=labels, topk=5)
            if args.device == "cuda":
                torch.cuda.synchronize()
            timings: list[float] = []
            output: object = None
            for _ in range(args.repeats):
                if args.device == "cuda":
                    torch.cuda.synchronize()
                begin = time.perf_counter()
                output = model.predict(image, boxes, box_labels=labels, topk=5)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                timings.append((time.perf_counter() - begin) * 1000.0)
            record.update(
                {
                    "status": "passed",
                    "latency_ms_mean": statistics.mean(timings),
                    "latency_ms_median": statistics.median(timings),
                    "latency_ms_p95": percentile(timings, 0.95),
                    "latency_ms_min": min(timings),
                    "latency_ms_max": max(timings),
                    "peak_device_memory_bytes": (
                        int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else None
                    ),
                    "triplets": [str(item) for item in output] if output is not None else [],
                    "torch_version": torch.__version__,
                    "numpy_version": np.__version__,
                }
            )
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:  # retain a complete matrix even if one model cannot load
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["wall_time_s"] = time.perf_counter() - started
        records.append(record)
        print(json.dumps(record, sort_keys=True))

    args.output.write_text("\n".join(json.dumps(row, sort_keys=True) for row in records) + "\n", encoding="utf-8")
    passed = sum(row["status"] == "passed" for row in records)
    print(f"wrote {args.output} ({passed}/{len(records)} passed)")
    return 0 if passed == len(records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
