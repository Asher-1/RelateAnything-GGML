#!/usr/bin/env python3
"""Compare two RAOOv1 files and print raw-logit parity metrics."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = path.read_bytes()
    if data[:8] != b"RAOOv1\0\0":
        raise ValueError(f"{path}: invalid RAOOv1 magic")
    pairs, predicates = struct.unpack_from("<II", data, 8)
    offset = 16
    subjects = np.frombuffer(data, dtype="<i4", count=pairs, offset=offset).copy()
    offset += pairs * 4
    objects = np.frombuffer(data, dtype="<i4", count=pairs, offset=offset).copy()
    offset += pairs * 4
    pair_logits = np.frombuffer(data, dtype="<f4", count=pairs, offset=offset).copy()
    offset += pairs * 4
    predicate_logits = np.frombuffer(data, dtype="<f4", count=pairs * predicates, offset=offset).copy()
    if predicate_logits.size != pairs * predicates:
        raise ValueError(f"{path}: truncated logits")
    return subjects, objects, pair_logits, predicate_logits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    ref = load(args.reference)
    got = load(args.candidate)
    if any(a.shape != b.shape or not np.array_equal(a, b) for a, b in zip(ref[:2], got[:2])):
        raise SystemExit("PARITY FAIL pair indices differ")
    errors = np.concatenate([got[2] - ref[2], got[3] - ref[3]])
    max_abs = float(np.max(np.abs(errors))) if errors.size else 0.0
    mae = float(np.mean(np.abs(errors))) if errors.size else 0.0
    rmse = float(np.sqrt(np.mean(errors * errors))) if errors.size else 0.0
    scale = max(float(np.max(np.abs(ref[2]))), float(np.max(np.abs(ref[3]))), 1.0)
    passed = bool(max_abs <= args.atol + args.rtol * scale)
    print(f"max_abs={max_abs:.8g} mae={mae:.8g} rmse={rmse:.8g} scale={scale:.8g}")
    print("PARITY PASS" if passed else "PARITY FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
