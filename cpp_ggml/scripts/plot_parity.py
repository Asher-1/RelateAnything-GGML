#!/usr/bin/env python3
"""Plot measured raw-logit max-absolute errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.input.read_text(encoding="utf-8"))
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required to render charts") from exc
    labels = [f"{row['dtype']}:{row['backend']}" for row in records]
    values = [row["max_abs"] for row in records]
    figure, axis = plt.subplots(figsize=(max(7, len(labels) * 1.05), 4.5))
    axis.bar(labels, values, color="#0f766e")
    axis.set_yscale("log")
    axis.set_ylabel("max absolute raw-logit error (log scale)")
    axis.set_title("RelateAnything GGML measured parity")
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
