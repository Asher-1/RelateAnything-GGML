"""Generate detector-inclusive latency tables from checked, complete raw records."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from release.update_readme_latency import (
    ARMS,
    MODELS,
    RECOMMENDED,
    bold_fastest,
    column_headers,
    configurations,
    thermal_delta,
    update_block,
    vocabularies,
)

START = "<!-- BEGIN GENERATED E2E LATENCY -->"
END = "<!-- END GENERATED E2E LATENCY -->"
DETECTORS = {"yolo26": "YOLO26m", "yolo-world": "YOLO-World v2-m", "yoloe": "YOLOE-26m"}


def load_records(source):
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt") as stream:
        records = json.load(stream)["records"]
    expected = {(detector, model) for detector in DETECTORS for model in MODELS}
    keys = [(r["detector"]["family"], r["checkpoint"]) for r in records]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Expected every detector/checkpoint combination exactly once")
    records.sort(
        key=lambda r: (
            list(DETECTORS).index(r["detector"]["family"]),
            list(MODELS).index(r["checkpoint"]),
        )
    )
    shared = (
        "gpu",
        "cpu",
        "torch",
        "cuda",
        "tensorrt",
        "onnxruntime",
        "ultralytics",
        "batch",
        "image_size",
        "max_relation_boxes",
        "padded_boxes",
        "pair_budget",
        "input_images",
        "images",
        "threads",
        "opencv_threads",
        "rounds",
        "warmup_per_round",
        "timed_iterations_per_round",
        "tf32",
        "torch_compile",
        "arm_shuffle_seeds",
        "ort_cuda_options",
        "cooldown_temperature_c",
        "timing",
        "stage_timing",
        "vocabularies",
    )
    detector_metadata = {}
    for record in records:
        if not record["complete"]:
            raise ValueError("Cannot publish an incomplete end-to-end benchmark")
        if any(record[key] != records[0][key] for key in shared):
            raise ValueError("End-to-end benchmark conditions differ")
        detector = record["detector"]
        if any(
            detector[key] != records[0]["detector"][key]
            for key in (
                "classes",
                "predict",
                "precision",
                "runtime",
                "size",
                "masks_used_by_relation",
            )
        ):
            raise ValueError(
                "Detector vocabulary or inference settings differ between families"
            )
        if len(detector["classes"]) != 80 or detector["masks_used_by_relation"]:
            raise ValueError(
                "Expected COCO-80 detectors supplying boxes to the relation model"
            )
        if (
            detector["runtime"] != "pytorch"
            or detector["precision"] != "fp32"
            or detector["size"] != "m"
        ):
            raise ValueError("Expected medium PyTorch FP32 detectors")
        family = detector["family"]
        if detector != detector_metadata.setdefault(family, detector):
            raise ValueError("Detector configuration changed between checkpoints")
        expected = set(configurations(records))
        rows = {(row["backend"], row["predicates"]): row for row in record["rows"]}
        if len(rows) != len(record["rows"]) or set(rows) != expected:
            raise ValueError(
                "Expected four relation backends on the bank and the PyTorch arms on the full vocabulary"
            )
        expected_n = record["rounds"] * record["timed_iterations_per_round"]
        if record["timed_iterations_per_round"] % record["images"]:
            raise ValueError("Every image must be timed equally")
        for row in rows.values():
            samples = row["pipeline_samples"]
            if len(samples) != expected_n or row["n"] != expected_n:
                raise ValueError("Missing pipeline samples")
            if [s["total_ms"] for s in samples] != row["samples_ms"]:
                raise ValueError("Pipeline and overall totals disagree")
            coverage = Counter((s["round"], s["image_index"]) for s in samples)
            if coverage != Counter(
                {
                    (r, i): record["timed_iterations_per_round"] // record["images"]
                    for r in range(1, record["rounds"] + 1)
                    for i in range(record["images"])
                }
            ):
                raise ValueError("Unequal per-image or per-round coverage")
            for sample in samples:
                stages = [
                    sample[key + "_ms"] for key in ("detector", "relation", "decode")
                ]
                if (
                    not all(math.isfinite(x) and x >= 0 for x in stages)
                    or not math.isfinite(sample["total_ms"])
                    or sample["total_ms"] <= 0
                    or sum(stages) > sample["total_ms"] + 1e-6
                ):
                    raise ValueError("Invalid stage timing")
                used = sample["used_boxes"]
                if (
                    not 0
                    <= used
                    <= min(sample["detected_boxes"], record["max_relation_boxes"])
                ):
                    raise ValueError("Invalid detected/used box counts")
                if sample["relation_skipped"] != (used < 2):
                    raise ValueError("Inconsistent skipped relation flag")
                if sample["relation_skipped"] and (
                    sample["relation_ms"] or sample["decode_ms"] or sample["triplets"]
                ):
                    raise ValueError("Skipped relations contain measured relation work")
        blocks = record["measurement_blocks"]
        if Counter(
            (b["backend"], b["predicates"], b["round"]) for b in blocks
        ) != Counter(
            {
                (arm, vocab, r): 1
                for arm, vocab in expected
                for r in range(1, record["rounds"] + 1)
            }
        ):
            raise ValueError("Missing measurement-block telemetry")
        for block in blocks:
            thermal_delta(block)
        checks = record["fp32_cross_backend_checks"]
        expected_checks = {
            (arm, vocabularies(records)[0], Path(image["path"]).name)
            for arm in (ARMS[0], ARMS[2])
            for image in record["input_images"]
        }
        if (
            len(checks) != len(expected_checks)
            or {(c["backend"], c["predicates"], c["image"]) for c in checks}
            != expected_checks
        ):
            raise ValueError("Incomplete FP32 parity checks")
        if not all(
            c["same_valid_pairs"] and math.isfinite(c["max_logit_delta"])
            for c in checks
        ):
            raise ValueError("Failed FP32 parity checks")
    return records


def samples(record, arm, vocab):
    return next(
        row["pipeline_samples"]
        for row in record["rows"]
        if row["backend"] == arm and row["predicates"] == vocab
    )


def stats(values):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * 0.95
    lo, hi = math.floor(index), math.ceil(index)
    return statistics.median(values), values[lo] + (values[hi] - values[lo]) * (
        index - lo
    )


def cell(values):
    result = stats(values)
    return "—" if result is None else " / ".join(f"{value:.2f}" for value in result)


def all_samples(record):
    return [sample for row in record["rows"] for sample in row["pipeline_samples"]]


def total_ms(record, arm, vocab):
    return stats([s["total_ms"] for s in samples(record, arm, vocab)])[0]


def render_overview(records):
    first = records[0]
    columns = configurations(records)
    lines = [
        "Median ms per image, batch 1, detection included. **Bold**: the fastest",
        "checkpoint with each detector in each column.",
        "",
        "| Detector | Checkpoint | " + " | ".join(column_headers(records)) + " |",
        "|---|---|" + "---:|" * len(columns),
    ]
    for family, label in DETECTORS.items():
        group = [r for r in records if r["detector"]["family"] == family]
        formatted = {c: bold_fastest([total_ms(r, *c) for r in group]) for c in columns}
        for i, record in enumerate(group):
            name = MODELS[record["checkpoint"]]
            lines.append(
                "| "
                + " | ".join([label, name] + [formatted[c][i] for c in columns])
                + " |"
            )
    bank, full = vocabularies(records)
    boxes = [s["used_boxes"] for record in records for s in all_samples(record)]
    lines += [
        "",
        f"The detector runs in PyTorch FP32; the columns choose the relation runtime and vocabulary "
        f"({bank} released predicates, or the full {full:,} in the Python API). "
        f"Measured on {first['images']} photos with {min(boxes)} to {max(boxes)} detected objects.",
    ]
    return "\n".join(lines)


def render_summary(records):
    bank, _ = vocabularies(records)
    group = [r for r in records if r["detector"]["family"] == "yolo26"]
    trt = {r["checkpoint"]: total_ms(r, "TensorRT FP32", bank) for r in group}
    fastest = min(trt, key=trt.get)
    return "\n".join(
        [
            f"With **{DETECTORS['yolo26']} detection** included, a frame takes **{trt[fastest]:.1f} ms** with",
            f"**{MODELS[fastest]}** and **{trt[RECOMMENDED]:.1f} ms** with **{MODELS[RECOMMENDED]}** (TensorRT relations).",
        ]
    )


def render_details(records):
    bank, _ = vocabularies(records)
    detection = {}
    for family, label in DETECTORS.items():
        group = [r for r in records if r["detector"]["family"] == family]
        detection[label] = statistics.median(
            stats([s["detector_ms"] for s in samples(r, "TensorRT FP32", bank)])[0]
            for r in group
        )
    decode = max(
        stats([s["decode_ms"] for s in samples(r, arm, vocab)])[0]
        for r in records
        for arm, vocab in configurations(records)
    )
    first = records[0]
    return "\n".join(
        [
            render_overview(records),
            "",
            "Detection alone takes "
            + ", ".join(f"{ms:.0f} ms with {label}" for label, ms in detection.items())
            + f"; decoding takes under {decode:.1f} ms. The rest is the relation model.",
            "",
            f"PyTorch {first['torch']}, CUDA {first['cuda']}, TensorRT {first['tensorrt']}, "
            f"ONNX Runtime {first['onnxruntime']}, Ultralytics {first['ultralytics']}; {first['cpu']}.",
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="local benchmark records JSON"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    records = load_records(args.source)
    update_block(ROOT / "README.md", render_summary(records), args.check, START, END)
    update_block(
        ROOT / "docs/benchmarks/end-to-end.md",
        render_details(records),
        args.check,
        START,
        END,
    )


if __name__ == "__main__":
    main()
