#!/usr/bin/env python3
"""Plot measured full-pipeline PyTorch CUDA latency and peak allocation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    rows = [row for row in rows if row.get("status") == "passed"]
    if not rows:
        raise SystemExit("no passing PyTorch CUDA measurements in input")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required to render charts") from exc

    labels = [row["model"] for row in rows]
    positions = range(len(rows))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar([x - 0.2 for x in positions], [row["latency_ms_median"] for row in rows],
                width=0.4, color="#0f766e", label="Median")
    axes[0].bar([x + 0.2 for x in positions], [row["latency_ms_p95"] for row in rows],
                width=0.4, color="#2563eb", label="P95")
    axes[0].set_xticks(list(positions), labels)
    axes[0].set_ylabel("end-to-end latency (ms)")
    axes[0].set_title("Measured Python CUDA latency")
    axes[0].legend()
    axes[1].bar(labels, [row["peak_device_memory_bytes"] / 1024**2 for row in rows],
                color="#f59e0b")
    axes[1].set_ylabel("peak PyTorch allocation (MiB)")
    axes[1].set_title("Measured Python CUDA memory")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
    figure.suptitle("Official checkpoints, same image and vocabulary, RTX 3060")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
