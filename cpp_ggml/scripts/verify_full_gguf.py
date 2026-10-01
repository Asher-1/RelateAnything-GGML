#!/usr/bin/env python3
"""Verify that a full RelateAnything GGUF contains the released checkpoint.

This is a container check, separate from inference parity. It checks every
tensor name, logical shape and stored value against the official runtime model
(including its sidecar vocabulary) under the declared quantization policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        import torch
        from gguf import GGUFReader, GGMLQuantizationType as Q
    except ImportError as exc:
        parser.error(f"verification dependencies unavailable: {exc}")

    from export_gguf import _load_checkpoint
    import numpy as np
    from gguf.quants import quantize
    payload = _load_checkpoint(args.checkpoint)
    state = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise SystemExit("checkpoint does not contain model state")
    expected = {f"ra.{k}": tuple(v.shape) for k, v in state.items() if hasattr(v, "shape")}
    reader = GGUFReader(args.gguf)
    # GGUF stores ne[0] as the fastest dimension, while checkpoint arrays use
    # the conventional PyTorch order.  Reverse the reader shape for comparison.
    # Long checkpoint names are emitted as stable hashes because ggml 0.21
    # limits tensor names to 63 bytes.  Restore the original names before
    # comparing the container with the PyTorch state dict.
    name_map_field = reader.fields.get("ra.tensor_name_map")
    name_map = name_map_field.contents() if name_map_field is not None else []
    emitted_to_original = {
        str(name_map[index + 1]): str(name_map[index])
        for index in range(0, len(name_map) - 1, 2)
    }
    actual = {
        emitted_to_original.get(t.name, t.name): tuple(reversed(tuple(int(x) for x in t.shape)))
        for t in reader.tensors
    }
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    shape_mismatch = {
        name: {"expected": expected[name], "actual": actual[name]}
        for name in sorted(set(expected) & set(actual))
        if expected[name] != actual[name]
    }
    graph_kind = reader.fields["ra.graph_kind"].contents()
    if isinstance(graph_kind, bytes):
        graph_kind = graph_kind.decode()
    type_counts: dict[str, int] = {}
    value_mismatch = []
    for tensor in reader.tensors:
        type_counts[str(tensor.tensor_type.name)] = type_counts.get(str(tensor.tensor_type.name), 0) + 1
        original = emitted_to_original.get(tensor.name, tensor.name)[3:]
        if original not in state:
            continue
        source = state[original].detach().cpu().float().numpy()
        if tensor.tensor_type == Q.Q8_0:
            expected_value = quantize(source, Q.Q8_0)
        elif tensor.tensor_type in (Q.F32, Q.F16):
            expected_value = source.astype(np.float32 if tensor.tensor_type == Q.F32 else np.float16)
        else:
            value_mismatch.append(tensor.name + ': unexpected type')
            continue
        if not np.array_equal(expected_value.reshape(-1), tensor.data.reshape(-1)):
            value_mismatch.append(tensor.name)
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "gguf": str(args.gguf),
        "gguf_sha256": sha256(args.gguf),
        "graph_kind": graph_kind,
        "expected_tensor_count": len(expected),
        "actual_tensor_count": len(actual),
        "missing": missing,
        "extra": extra,
        "shape_mismatch": shape_mismatch,
        "type_counts": type_counts,
        "value_mismatch": value_mismatch,
        "runtime_vocabulary": True,
        "gguf_bytes": args.gguf.stat().st_size,
        "download_url": f"https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/gguf/{args.gguf.name}",
        "passed": not missing and not extra and not shape_mismatch and not value_mismatch and graph_kind == "relateanything_full_v1",
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
