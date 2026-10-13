"""Generate the README sentence and the laptop latency table from raw samples.

Pass --source with a local result file; --check fails if either page is stale.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- BEGIN GENERATED LATENCY -->"
END = "<!-- END GENERATED LATENCY -->"
MODELS = {
    "maelic/relsgg-vits16": "ViT-S",
    "maelic/relsgg-vits16plus": "ViT-S+",
    "maelic/relsgg-vitb16": "ViT-B",
    "maelic/relsgg-convnext-tiny": "ConvNeXt-T",
    "maelic/relsgg-convnext-small": "ConvNeXt-S",
    "maelic/relsgg-convnext-base": "ConvNeXt-B",
    "maelic/relsgg-convnext-large": "ConvNeXt-L",
}
RECOMMENDED = "maelic/relsgg-vits16plus"
ARMS = ["PyTorch eager FP32", "PyTorch eager BF16", "ONNX CUDA FP32", "TensorRT FP32"]
#: The full training vocabulary exists only in the Python API.
FULL_ARMS = ARMS[:2]
HEADERS = {
    "PyTorch eager FP32": "PyTorch FP32",
    "PyTorch eager BF16": "PyTorch BF16",
    "ONNX CUDA FP32": "ONNX Runtime FP32",
    "TensorRT FP32": "TensorRT FP32",
}


def vocabularies(records):
    """(bank, full): the released predicate bank and the full vocabulary."""
    sizes = {(r["vocabularies"]["bank"], r["vocabularies"]["full"]) for r in records}
    if len(sizes) != 1 or None in next(iter(sizes)):
        raise ValueError("Every record must time the same bank and full vocabulary")
    return next(iter(sizes))


def configurations(records):
    """(arm, predicates) of every column, in table order."""
    bank, full = vocabularies(records)
    return [(arm, bank) for arm in ARMS] + [(arm, full) for arm in FULL_ARMS]


def column_headers(records):
    bank, full = vocabularies(records)
    return [HEADERS[arm] for arm in ARMS] + [
        f"{HEADERS[arm]}, {full:,} predicates" for arm in FULL_ARMS
    ]


def bold_fastest(values):
    """Format a column, the fastest value in bold."""
    best = min(values)
    return [f"**{v:.1f}**" if v == best else f"{v:.1f}" for v in values]


def load_records(source):
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt") as stream:
        records = json.load(stream)["records"]
    if len(records) != len(MODELS) or {r["checkpoint"] for r in records} != set(MODELS):
        raise ValueError(
            "Expected exactly one completed record per released checkpoint"
        )
    records.sort(key=lambda r: list(MODELS).index(r["checkpoint"]))
    shared = [
        "gpu",
        "cpu",
        "torch",
        "cuda",
        "tensorrt",
        "onnxruntime",
        "batch",
        "image_size",
        "valid_boxes",
        "padded_boxes",
        "pair_budget",
        "images",
        "threads",
        "opencv_threads",
        "rounds",
        "warmup_per_round",
        "timed_iterations_per_round",
        "tf32",
        "torch_compile",
        "input_images",
        "box_seed",
        "arm_shuffle_seeds",
        "ort_cuda_options",
        "cooldown_temperature_c",
        "timing",
        "vocabularies",
    ]
    for record in records:
        if not record["complete"]:
            raise ValueError("Cannot publish an incomplete benchmark")
        if record.get("detector") is not None:
            raise ValueError(
                "End-to-end detector records cannot populate relation-only tables"
            )
        if any(record[key] != records[0][key] for key in shared):
            raise ValueError("Benchmark conditions differ between checkpoints")
    expected = set(configurations(records))
    for record in records:
        rows = {(r["backend"], r["predicates"]): r for r in record["rows"]}
        if len(rows) != len(record["rows"]) or set(rows) != expected:
            raise ValueError(
                "Expected four backends on the bank and the PyTorch arms on the full vocabulary"
            )
        for row in rows.values():
            samples = row["samples_ms"]
            expected_n = record["rounds"] * record["timed_iterations_per_round"]
            if (
                len(samples) != row["n"]
                or row["n"] != expected_n
                or not all(math.isfinite(x) and x > 0 for x in samples)
            ):
                raise ValueError("Invalid latency samples")
    return records


def stats(record, backend, predicates):
    row = next(
        r
        for r in record["rows"]
        if r["backend"] == backend and r["predicates"] == predicates
    )
    values = sorted(row["samples_ms"])
    # Linear interpolation, matching numpy.percentile(..., 95).
    index = (len(values) - 1) * 0.95
    lo, hi = math.floor(index), math.ceil(index)
    p95 = values[lo] + (values[hi] - values[lo]) * (index - lo)
    return statistics.median(values), p95


def thermal_delta(block):
    total = 0
    for kind in ("sw_thermal_slowdown", "hw_thermal_slowdown"):
        key = "clocks_event_reasons_counters." + kind
        before = int(block["gpu_before_warmup"][key].split()[0])
        after = int(block["gpu_after_timing"][key].split()[0])
        if after < before:
            raise ValueError("Thermal counter reset during a measurement block")
        total += after - before
    return total


def render_summary(records):
    bank, full = vocabularies(records)
    trt = {r["checkpoint"]: stats(r, "TensorRT FP32", bank)[0] for r in records}
    fastest = min(trt, key=trt.get)
    extra = [
        stats(r, arm, full)[0] - stats(r, arm, bank)[0]
        for r in records
        for arm in FULL_ARMS
    ]
    gpu = records[0]["gpu"].removeprefix("NVIDIA GeForce ")
    return "\n".join(
        [
            f"On an **{gpu}** with TensorRT, the relation model takes **{trt[fastest]:.1f} ms**",
            f"per image with **{MODELS[fastest]}**, the fastest checkpoint, and **{trt[RECOMMENDED]:.1f} ms**",
            f"with the recommended **{MODELS[RECOMMENDED]}**. The full {full:,}-predicate vocabulary",
            f"(Python API) adds at most {max(extra):.1f} ms.",
        ]
    )


def render_details(records):
    columns = configurations(records)
    cells = {(r["checkpoint"], c): stats(r, *c)[0] for r in records for c in columns}
    formatted = {
        c: bold_fastest([cells[r["checkpoint"], c] for r in records]) for c in columns
    }
    bank, full = vocabularies(records)
    lines = [
        f"Median ms per image, batch 1. **Bold**: the fastest checkpoint in each column.",
        "",
        "| Checkpoint | " + " | ".join(column_headers(records)) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for i, record in enumerate(records):
        label = MODELS[record["checkpoint"]]
        lines.append(
            "| " + " | ".join([label] + [formatted[c][i] for c in columns]) + " |"
        )
    data = records[0]
    lines += [
        "",
        f"The first four columns use the released {bank}-predicate vocabulary on every runtime; "
        f"the last two the full {full:,}-predicate training vocabulary, which only the Python API offers.",
        "",
        f"PyTorch {data['torch']}, CUDA {data['cuda']}, TensorRT {data['tensorrt']}, "
        f"ONNX Runtime {data['onnxruntime']}; {data['cpu']}.",
    ]
    return "\n".join(lines)


def update_block(path, body, check, start=START, end=END):
    original = path.read_text()
    if original.count(start) != 1 or original.count(end) != 1:
        raise ValueError(f"Expected exactly one latency block in {path}")
    before, remainder = original.split(start)
    _, after = remainder.split(end)
    updated = before + start + "\n" + body + "\n" + end + after
    if check:
        if updated != original:
            raise SystemExit(
                f"{path.relative_to(ROOT)} is stale; regenerate its latency table"
            )
        print(f"{path.relative_to(ROOT)} matches the measured samples")
    else:
        path.write_text(updated)
        print(f"Updated {path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="local benchmark records JSON"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    records = load_records(args.source)
    update_block(ROOT / "README.md", render_summary(records), args.check)
    update_block(
        ROOT / "docs/benchmarks/README.md", render_details(records), args.check
    )


if __name__ == "__main__":
    main()
