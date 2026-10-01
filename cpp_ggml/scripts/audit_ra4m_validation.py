#!/usr/bin/env python3
"""Audit retained RA-4M annotations or an optional complete validation split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "cpp_ggml/test_data/ra4m_50")
    parser.add_argument("--output", type=Path, default=ROOT / "cpp_ggml/benchmarks/ra4m_50_audit.json")
    parser.add_argument("--plot", type=Path, default=ROOT / "cpp_ggml/benchmarks/ra4m_50_audit.png")
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(f"matplotlib is required: {exc}") from exc
    manifest_path = args.data / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shards = []
        batches = [("ra4m_50", manifest["samples"])]
        expected_rows = 50
    else:
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise SystemExit(f"pyarrow is required for full parquet audit: {exc}") from exc
        shards = sorted(args.data.glob("val-*.parquet"))
        if len(shards) != 5:
            raise SystemExit(f"expected five RA-4M validation shards, found {len(shards)}")
        batches = ((shard.name, (row for batch in parquet.ParquetFile(shard).iter_batches(
            batch_size=256, columns=["source", "width", "height", "objects", "relations"])
            for row in batch.to_pylist())) for shard in shards)
        expected_rows = 24964

    sources = Counter()
    predicates = Counter()
    object_counts = Counter()
    relation_counts = Counter()
    issues = Counter()
    boundary_sources = Counter()
    boundary_examples = []
    maximum_overflow = 0.0
    rows = 0
    objects = 0
    relations = 0
    for shard_name, batch_rows in batches:
        for row in batch_rows:
                rows += 1
                sources[row["source"]] += 1
                nodes = row["objects"] or []
                edges = row["relations"] or []
                objects += len(nodes)
                relations += len(edges)
                object_counts[len(nodes)] += 1
                relation_counts[len(edges)] += 1
                for node in nodes:
                    x, y, width, height = node["bbox"]
                    if width <= 0 or height <= 0:
                        issues["nonpositive_box_extent"] += 1
                    overflow = max(-x, -y, x + width - row["width"],
                                   y + height - row["height"], 0)
                    if overflow > 1:
                        issues["box_outside_image_by_over_1px"] += 1
                        boundary_sources[row["source"]] += 1
                        maximum_overflow = max(maximum_overflow, overflow)
                        if len(boundary_examples) < 5:
                            boundary_examples.append({"source": row["source"],
                                                      "category": node["category"],
                                                      "overflow_px": overflow})
                for edge in edges:
                    if not (0 <= edge["subject"] < len(nodes) and
                            0 <= edge["object"] < len(nodes)):
                        issues["invalid_relation_endpoint"] += 1
                    predicates[edge["predicate"]] += 1
        print(f"audited {shard_name}: {rows} cumulative rows")

    report = {
        "dataset": "maelic/RA-4M",
        "split": "val",
        "scope": "50 retained validation images" if expected_rows == 50 else
                 "all validation annotation rows; no model evaluation or image decoding",
        "shards": [shard.name for shard in shards] if shards else
                  sorted({sample["shard"] for sample in manifest["samples"]}),
        "images": rows,
        "objects": objects,
        "relations": relations,
        "distinct_predicates": len(predicates),
        "source_images": dict(sorted(sources.items())),
        "issues": dict(sorted(issues.items())),
        "boundary_warnings_by_source": dict(sorted(boundary_sources.items())),
        "maximum_boundary_overflow_px": maximum_overflow,
        "boundary_warning_examples": boundary_examples,
        "objects_per_image": dict(sorted(object_counts.items())),
        "relations_per_image": dict(sorted(relation_counts.items())),
        "top_predicates": predicates.most_common(20),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(f"RA-4M annotation coverage | {rows:,} images, {relations:,} relations", fontsize=17)
    source_items = sorted(sources.items(), key=lambda item: item[1])
    axes[0, 0].barh([name for name, _ in source_items], [count for _, count in source_items], color="#167f82")
    axes[0, 0].set_title("Images by source corpus")
    axes[0, 0].set_xlabel("Images")
    axes[0, 1].bar([str(n) for n in range(1, 21)], [object_counts[n] for n in range(1, 21)], color="#ae7330")
    axes[0, 1].set_title("Object count per image (1-20)")
    axes[0, 1].set_xlabel("Objects")
    axes[0, 1].tick_params(axis="x", labelrotation=45)
    axes[1, 0].bar([str(n) for n in range(1, 31)], [relation_counts[n] for n in range(1, 31)], color="#4d6fa6")
    axes[1, 0].set_title("Relation count per image (1-30)")
    axes[1, 0].set_xlabel("Relations")
    axes[1, 0].tick_params(axis="x", labelrotation=45)
    common = predicates.most_common(12)
    axes[1, 1].barh([name for name, _ in reversed(common)],
                    [count for _, count in reversed(common)], color="#a34d5c")
    axes[1, 1].set_title(f"Top predicates of {len(predicates):,} distinct strings")
    axes[1, 1].set_xlabel("Relations")
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=160)
    plt.close(fig)
    print(f"wrote {args.output} and {args.plot}")
    critical = issues["invalid_relation_endpoint"] + issues["nonpositive_box_extent"]
    return 0 if rows == expected_rows and not critical else 2


if __name__ == "__main__":
    raise SystemExit(main())
