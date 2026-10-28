#!/usr/bin/env python3
"""Independent NumPy reference for relation_pair_linear_v1.

This is a tensor-level reference for the explicit adapter contract. It is not
the full DINOv3 RelateAnything Python pipeline.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

def read_input(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    if data[:8] != b"RAIOv1\0\0":
        raise ValueError("invalid RAIOv1 magic")
    objects, features = struct.unpack_from("<II", data, 8)
    offset = 16
    x = np.frombuffer(data, dtype="<f4", count=objects * features, offset=offset).reshape(objects, features)
    offset += objects * features * 4
    boxes = np.frombuffer(data, dtype="<f4", count=objects * 4, offset=offset).reshape(objects, 4)
    return x, boxes


def write_output(path: Path, subjects: np.ndarray, objects: np.ndarray, pair: np.ndarray, predicate: np.ndarray) -> None:
    with path.open("wb") as stream:
        stream.write(b"RAOOv1\0\0")
        stream.write(struct.pack("<II", len(subjects), predicate.shape[1]))
        stream.write(subjects.astype("<i4").tobytes())
        stream.write(objects.astype("<i4").tobytes())
        stream.write(pair.astype("<f4").tobytes())
        stream.write(predicate.astype("<f4").tobytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True, help="NPZ with six ra.* arrays")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    weights = dict(np.load(args.weights))
    names = ("ra.pair_proj.weight", "ra.pair_proj.bias", "ra.predicate.weight",
             "ra.predicate.bias", "ra.pair.weight", "ra.pair.bias")
    missing = [name for name in names if name not in weights]
    if missing:
        raise SystemExit("missing reference weights: " + ", ".join(missing))
    features, boxes = read_input(args.input)
    subjects, objects, rows = [], [], []
    for subject in range(len(features)):
        for object_ in range(len(features)):
            if subject == object_:
                continue
            rows.append(np.concatenate((features[subject], features[object_], boxes[subject], boxes[object_])))
            subjects.append(subject)
            objects.append(object_)
    pair_input = np.asarray(rows, dtype=np.float32)
    hidden = pair_input @ weights["ra.pair_proj.weight"].T + weights["ra.pair_proj.bias"]
    # Match ggml_gelu's tanh approximation exactly instead of NumPy's erf GELU.
    hidden = 0.5 * hidden * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (hidden + 0.044715 * hidden ** 3)))
    predicate = hidden @ weights["ra.predicate.weight"].T + weights["ra.predicate.bias"]
    pair = (hidden @ weights["ra.pair.weight"].T + weights["ra.pair.bias"]).reshape(-1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_output(args.output, np.asarray(subjects), np.asarray(objects), pair, predicate)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
