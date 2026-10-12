#!/usr/bin/env python3
"""Run OV-SGG-Bench on one checkpoint and print its composite score.

    python benchmark/setup_data.py                                     # once: data and images
    python benchmark/run_benchmark.py --checkpoint maelic/relsgg-convnext-tiny
    python benchmark/run_benchmark.py --checkpoint runs/train/my_run/checkpoint_last.pth
    python benchmark/run_benchmark.py --checkpoint ... --axes a1 a6    # a subset
    python benchmark/run_benchmark.py --checkpoint ... --judge         # also A5

Without ``--judge`` the composite is OVS-F1 over A1, A2, A4 and A6, the number of the model
comparison in docs/results.md. ``--judge`` adds A5 (graph quality, judged by a VLM) for the
five-axis OVS of the paper; it needs vLLM and about 60 GB of GPU memory for the default judge
(``--judge-tp 2`` splits it over two GPUs; ``--judge-python`` runs it from a separate venv).

Every axis writes into ``--out`` (default ``runs/benchmark/<model>``) under the names
benchmark/overall_score.py reads; a finished step is skipped on a rerun (``--force`` redoes it).
The axes are defined in benchmark/SPEC.md.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
AXES = ["a1", "a2", "a3", "a4", "a6"]
DETECTIONS = "runs/detect/yoloworld_ov_psg_test.npz"  # shared YOLO-World boxes (A4, A5)
TEXT_SPACE = "runs/packed/datamix_v22/text_space/pred_embeds_studentv2_512_photo.npz"
TAU = "runs/packed/text_student_v2_512/tau_calibration.json"
# The judge of the paper's A5 (chosen on PSG val by benchmark/judge_selection.py).
JUDGE = dict(
    model="Qwen/Qwen3.6-27B",
    rubric="truth",
    threshold=0.20,
    n=200,
    max_rel=10,
    control_frac=0.12,
    seed=0,
)


def resolve_checkpoint(name: str) -> Path:
    """A local file, or a Hugging Face model id whose model.pth is downloaded."""
    if Path(name).is_file():
        return Path(name).resolve()
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        name,
        allow_patterns=[
            "model.pth",
            "text_student.pt",
            "tokenizer*",
            "calibration.json",
            "predicate_embeddings.npz",
        ],
    )
    return Path(local) / "model.pth"


def records_text_space(ckpt: Path) -> bool:
    import torch

    ck = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    args = ck.get("args") or {}
    args = args if isinstance(args, dict) else vars(args)
    return bool(args.get("pred_embeds"))


def step(name: str, out: Path, done: Path, cmd: list, force: bool) -> None:
    """Run one stage, logged to out/logs/<name>.log; skipped when `done` exists."""
    if done.exists() and not force:
        print(f"[bench] {name}: done ({done.relative_to(out)})")
        return
    log = out / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[bench] {name} ...", flush=True)
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run(
            [str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT, cwd=REPO
        ).returncode
    if rc != 0 or not done.exists():
        tail = "".join(open(log).readlines()[-15:])
        sys.exit(f"[bench] {name} failed (log: {log}):\n{tail}")
    print(f"[bench] {name}: {time.time() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--checkpoint",
        required=True,
        help="model.pth, a training checkpoint, "
        "or a Hugging Face id such as maelic/relsgg-vits16plus",
    )
    ap.add_argument("--axes", nargs="+", default=AXES, choices=AXES)
    ap.add_argument("--judge", action="store_true", help="also run A5 (needs vLLM)")
    ap.add_argument("--judge-model", default=JUDGE["model"])
    ap.add_argument("--judge-tp", type=int, default=1, help="GPUs for the judge")
    ap.add_argument(
        "--judge-python",
        default=PY,
        help="Python with vLLM, when it lives in a separate venv",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="redo finished steps")
    a = ap.parse_args()

    os.chdir(REPO)
    ckpt = resolve_checkpoint(a.checkpoint)
    name = (
        a.checkpoint.rstrip("/").split("/")[-1]
        if not Path(a.checkpoint).is_file()
        else ckpt.parent.name
    )
    out = Path(a.out or f"runs/benchmark/{name}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"[bench] {ckpt} -> {out}")
    nw = ["--num_workers", a.num_workers]
    axes = set(a.axes)

    if "a1" in axes:
        step(
            "a1",
            out,
            out / "zeroshot_hicodet_test_gc.json",
            [
                PY,
                "benchmark/eval_zeroshot.py",
                "--checkpoint",
                ckpt,
                "--data_roots",
                "runs/packed/vg150",
                "runs/packed/psg",
                "runs/packed/indoorvg",
                "runs/packed/hicodet",
                "--split",
                "test",
                "--graph_constraint",
                "--out_dir",
                out,
                *nw,
            ],
            a.force,
        )
    if "a2" in axes:
        step(
            "a2_haystack",
            out,
            out / "haystack_sigmoid.json",
            [
                PY,
                "benchmark/eval_haystack.py",
                "--checkpoint",
                ckpt,
                "--pack",
                "runs/packed/haystack",
                "--negatives",
                "runs/datamix/haystack_negatives.json",
                "--out_dir",
                out,
                *nw,
            ],
            a.force,
        )
        step(
            "a2_hicodet",
            out,
            out / "hico_fap" / "haystack_sigmoid.json",
            [
                PY,
                "benchmark/eval_haystack.py",
                "--checkpoint",
                ckpt,
                "--pack",
                "runs/packed/hicodet",
                "--negatives",
                "runs/datamix/hicodet_negatives.json",
                "--out_dir",
                out / "hico_fap",
                *nw,
            ],
            a.force,
        )
    if "a3" in axes:
        space = [] if records_text_space(ckpt) else ["--pred_embeds", TEXT_SPACE]
        step(
            "a3",
            out,
            out / "zeroshot_indoorvg_test_gc_ov.json",
            [
                PY,
                "benchmark/eval_zeroshot.py",
                "--checkpoint",
                ckpt,
                "--data_roots",
                "runs/packed/vg150",
                "runs/packed/psg",
                "runs/packed/indoorvg",
                "--split",
                "test",
                "--open_vocab",
                "--tau_calibration",
                TAU,
                *space,
                "--batch_size",
                8,
                "--out_dir",
                out,
                *nw,
            ],
            a.force,
        )
    if "a4" in axes:
        step(
            "a4",
            out,
            out / "zeroshot_detbox_psg_yoloworld_gc.json",
            [
                PY,
                "benchmark/eval_zeroshot_detbox.py",
                "--checkpoint",
                ckpt,
                "--dataset_root",
                "runs/packed/psg",
                "--dataset_name",
                "psg",
                "--split",
                "test",
                "--det",
                DETECTIONS,
                "--det_vocab",
                "pack",
                "--det_conf",
                0.05,
                "--score_mode",
                "softmax",
                "--out",
                out / "zeroshot_detbox_psg_yoloworld_gc.json",
                *nw,
            ],
            a.force,
        )
    if "a6" in axes:
        step(
            "a6",
            out,
            out / "spatialsense.json",
            [
                PY,
                "benchmark/eval_spatialsense.py",
                "--checkpoint",
                ckpt,
                "--out",
                out / "spatialsense.json",
                *nw,
            ],
            a.force,
        )
    if a.judge:
        a5 = out / "a5"
        dump = a5 / "psg_test_yoloworld_pack.npz"
        step(
            "a5_predictions",
            out,
            dump,
            [
                PY,
                "benchmark/dump_relsgg_interchange.py",
                "--checkpoint",
                ckpt,
                "--dataset_root",
                "runs/packed/psg",
                "--split",
                "test",
                "--dataset_name",
                "psg_test",
                "--det",
                DETECTIONS,
                "--det_vocab",
                "pack",
                "--det_conf",
                0.05,
                "--out",
                dump,
                *nw,
            ],
            a.force,
        )
        rp = a5 / "relation_precision.json"
        step(
            "a5_judge",
            out,
            rp,
            [
                a.judge_python,
                "benchmark/relation_precision.py",
                "--system",
                f"{name}={dump}",
                "--pack",
                "runs/packed/psg/test",
                "--model",
                a.judge_model,
                "--tp",
                a.judge_tp,
                "--n",
                JUDGE["n"],
                "--seed",
                JUDGE["seed"],
                "--control_frac",
                JUDGE["control_frac"],
                "--rubric",
                JUDGE["rubric"],
                "--threshold",
                JUDGE["threshold"],
                "--max_rel",
                JUDGE["max_rel"],
                "--rel_frac",
                0.0,
                "--out",
                rp,
            ],
            a.force,
        )
        info = a5 / "info.json"
        step(
            "a5_bits",
            out,
            info,
            [
                PY,
                "benchmark/a5b_information.py",
                "--run",
                rp,
                "--expected",
                "--ref",
                "runs/packed/psg/train/meta.json",
                "--out",
                info,
            ],
            a.force,
        )
        step(
            "a5_share",
            out,
            a5 / "annotation_reference.json",
            [
                PY,
                "benchmark/a5b_annotation_reference.py",
                "--run",
                rp,
                "--info",
                info,
                "--pack",
                "runs/packed/psg/test",
                "--ref",
                "runs/packed/psg/train/meta.json",
                "--out",
                a5 / "annotation_reference.json",
            ],
            a.force,
        )

    summary = out / "ovs.json"
    subprocess.run(
        [
            PY,
            "benchmark/overall_score.py",
            "--runs",
            str(out),
            "--metric",
            "F1",
            "--latency",
            str(out / "latency.json"),
            "--out",
            str(summary),
        ],
        cwd=REPO,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    rows = json.load(open(summary))["rows"]
    if not rows:
        sys.exit("[bench] no axis produced a result")
    row = rows[0]
    print(f"\n{name}  (chance-corrected, x100)")
    for axis in (
        "A1 transfer",
        "A2 precision",
        "A3 open-vocab",
        "A4 detector",
        "A5 graph quality",
        "A6 spatial",
    ):
        v = row["axes"].get(axis)
        print(
            f"  {axis:17s} {100 * v:5.1f}" if v is not None else f"  {axis:17s}    --"
        )
    comp = [k.split()[0] for k in row["composite_axes"]]
    print(f"  OVS-F1 over {', '.join(comp)}: {100 * row['OVS']:.1f}")
    if comp == ["A1", "A2", "A4", "A6"]:
        print(
            "  (comparable with the OVS-F1 column of docs/results.md; add --judge for the "
            "five-axis OVS of the paper)"
        )
    elif len(comp) == 5:
        print("  (five axes, as in the paper)")
    else:
        print("  (a subset of the axes: compare only with runs scored on the same set)")
    print(f"saved: {summary}")


if __name__ == "__main__":
    main()
