#!/usr/bin/env python3
"""Record reproducible C++ GGML latency metadata as JSON."""

from __future__ import annotations

import argparse
import json
import platform
import re
import struct
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LATENCY = re.compile(r"^latency_ms=([0-9.eE+-]+)$", re.MULTILINE)


def _resolved(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _input_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(16)
    if len(header) != 16 or header[:8] != b"RAIOv1\0\0":
        raise RuntimeError(f"unsupported RAIO input: {path}")
    return struct.unpack_from("<II", header, 8)


def _ggml_commit() -> str:
    completed = subprocess.run(
        ["git", "-C", str(ROOT / "cpp_ggml/third_party/ggml"), "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--dtype", choices=("f32", "f16", "q8_0"), required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    binary = _resolved(args.binary)
    model = _resolved(args.model)
    input_path = _resolved(args.input)
    command = [str(binary), "--backend", args.backend, "--model", str(model),
               "--input", str(input_path), "--warmup", str(args.warmup), "--repeats", str(args.repeats)]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    elapsed = time.perf_counter() - started
    match = LATENCY.search(completed.stdout)
    if not match:
        raise RuntimeError("C++ output did not contain latency_ms")
    object_count, feature_dim = _input_shape(input_path)
    reference_status = "not_requested"
    reference_stdout = ""
    reference_stderr = ""
    if args.reference_command:
        reference = subprocess.run(
            args.reference_command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        reference_status = "passed" if reference.returncode == 0 else "failed"
        reference_stdout = reference.stdout
        reference_stderr = reference.stderr
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "dtype": args.dtype,
        "model": str(args.model),
        "input": str(args.input),
        "model_size_bytes": model.stat().st_size,
        "input_size_bytes": input_path.stat().st_size,
        "object_count": object_count,
        "feature_dim": feature_dim,
        "ggml_commit": _ggml_commit(),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "latency_ms_mean": float(match.group(1)),
        "wall_time_s": elapsed,
        "host": platform.platform(),
        "python_reference_status": reference_status,
        "parity_status": "not_run",
        "peak_device_memory_bytes": None,
        "cudnn": False,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "python_reference_stdout": reference_stdout,
        "python_reference_stderr": reference_stderr,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
