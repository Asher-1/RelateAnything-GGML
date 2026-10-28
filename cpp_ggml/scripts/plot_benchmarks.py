#!/usr/bin/env python3
"""Plot measured benchmark JSONL records without inventing missing results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not records:
        raise SystemExit("benchmark JSONL is empty")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required to render charts") from exc
    labels = [f"{r['backend']}:{r['dtype']}" for r in records]
    values = [r["latency_ms_mean"] for r in records]
    fig, axis = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4.5))
    axis.bar(labels, values, color="#2563eb")
    axis.set_ylabel("mean inference latency (ms)")
    axis.set_title("RelateAnything GGML measured latency")
    axis.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
