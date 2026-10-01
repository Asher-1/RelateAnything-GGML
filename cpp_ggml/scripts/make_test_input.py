#!/usr/bin/env python3
"""Create deterministic RAIOv1 object features and boxes."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--objects", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    if args.objects < 2 or args.feature_dim < 1:
        parser.error("--objects must be >= 2 and --feature-dim must be positive")
    rng = np.random.default_rng(args.seed)
    features = rng.standard_normal((args.objects, args.feature_dim), dtype=np.float32)
    boxes = rng.uniform(0.05, 0.95, (args.objects, 4)).astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as stream:
        stream.write(b"RAIOv1\0\0")
        stream.write(struct.pack("<II", args.objects, args.feature_dim))
        stream.write(features.tobytes(order="C"))
        stream.write(boxes.tobytes(order="C"))
    print(f"wrote {args.out} objects={args.objects} feature_dim={args.feature_dim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
