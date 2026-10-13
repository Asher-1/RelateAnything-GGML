"""Per-run OVS from the files benchmark/run_benchmark.py writes, with and without A5."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def score(run, out):
    subprocess.run(
        [
            sys.executable,
            "benchmark/overall_score.py",
            "--runs",
            str(run),
            "--metric",
            "F1",
            "--latency",
            str(run / "none.json"),
            "--out",
            str(out),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return json.load(open(out))["rows"][0]


def test_run_directory_is_scored_on_four_axes_then_five(tmp_path):
    run = tmp_path / "model"
    for src in ("vg150", "psg", "indoorvg", "hicodet"):
        write(
            run / f"zeroshot_{src}_test_gc.json",
            {"metrics": {"R@50": 0.4, "mR@50": 0.3}},
        )
    write(run / "haystack_sigmoid.json", {"metrics": {"mfAP": 0.7}})
    write(
        run / "hico_fap" / "haystack_sigmoid.json",
        {"metrics": {"mfAP": 0.6, "n_cells": 100449}},
    )
    write(run / "zeroshot_detbox_psg_yoloworld_gc.json", {"lenient": {"wR@50": 0.18}})
    write(
        run / "spatialsense.json", {"AUC": 0.7, "per_predicate": {"on": {"AUC": 0.72}}}
    )

    row = score(run, tmp_path / "ovs4.json")
    assert row["composite_axes"] == [
        "A1 transfer",
        "A2 precision",
        "A4 detector",
        "A6 spatial",
    ]
    assert row["missing_axes"] == ["A5 graph quality"] and 0 < row["OVS"] < 1

    write(
        run / "a5" / "annotation_reference.json",
        {"per_system": {"model": {"share_of_annotation": 0.6}}},
    )
    row5 = score(run, tmp_path / "ovs5.json")
    assert row5["complete"] and row5["axes"]["A5 graph quality"] == 0.6
    assert row5["OVS"] != row["OVS"]
