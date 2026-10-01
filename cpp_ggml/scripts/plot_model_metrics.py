#!/usr/bin/env python3
"""Render the upstream published model quality/speed comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required to render charts") from exc
    labels = [row["model"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(labels, [row["a40_batch1_ms"] for row in rows], color="#2563eb", label="A40 batch 1")
    axes[0].set_ylabel("relation-head latency (ms)")
    axes[0].set_title("Published A40 relation-head latency")
    axes[0].tick_params(axis="x", rotation=20)
    positions = range(len(rows))
    axes[1].bar([x - 0.2 for x in positions], [row["ovs_f1"] for row in rows],
                width=0.4, color="#0f766e", label="OVS-F1")
    axes[1].bar([x + 0.2 for x in positions], [row["hico_f1"] for row in rows],
                width=0.4, color="#f59e0b", label="HICO F1")
    axes[1].set_xticks(list(positions), labels)
    axes[1].set_ylabel("F1 score")
    axes[1].set_title("Published quality metrics")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(fontsize=8)
    figure.suptitle("RelateAnything official checkpoint comparison (upstream published values)")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
