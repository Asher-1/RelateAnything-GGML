#!/usr/bin/env python3
"""Capture official Python intermediate tensors for future GGML stage parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
STAGES = (
    "spatial_pool", "sampler", "geo_encoder",
    "pair_proj", "rel_transformer", "deformable_read", "rel_interaction",
    "spa_proj",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flatten(value, prefix: str, tensors: dict, torch) -> None:
    if isinstance(value, torch.Tensor):
        if value.numel():
            tensors[prefix] = value.detach().cpu().contiguous().numpy()
    elif isinstance(value, Mapping):
        for key, child in value.items():
            flatten(child, f"{prefix}.{key}", tensors, torch)
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            flatten(child, f"{prefix}.{index}", tensors, torch)


def tensor_stats(array, np) -> dict:
    finite = np.isfinite(array)
    values = array[finite].astype(np.float64)
    return {
        "shape": list(array.shape), "dtype": str(array.dtype),
        "finite_values": int(finite.sum()), "nonfinite_values": int(array.size - finite.sum()),
        "nan_values": int(np.isnan(array).sum()),
        "positive_inf_values": int(np.isposinf(array).sum()),
        "negative_inf_values": int(np.isneginf(array).sum()),
        "minimum_finite": float(values.min()) if values.size else None,
        "maximum_finite": float(values.max()) if values.size else None,
        "mean_finite": float(values.mean()) if values.size else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "cpp_ggml/test_data/ra4m_50")
    parser.add_argument("--image-index", type=int, default=0)
    parser.add_argument("--model-name", default="relsgg-vits16plus",
                        choices=("relsgg-vits16", "relsgg-vits16plus", "relsgg-vitb16"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "cpp_ggml/benchmarks/python_stage_trace.json")
    parser.add_argument("--tensors", type=Path, default=ROOT / "cpp_ggml/test_data/python_stage_trace.npz")
    args = parser.parse_args()
    try:
        import numpy as np
        import torch
        from PIL import Image
        from relsgg import RelateAnything
    except ImportError as exc:
        parser.error(f"stage trace dependency unavailable: {exc}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not 0 <= args.image_index < len(manifest["samples"]):
        parser.error("--image-index is out of range")
    sample = manifest["samples"][args.image_index]
    checkpoint = ROOT / "cpp_ggml/models/pytorch" / args.model_name / "model.pth"
    model = RelateAnything.from_checkpoint(str(checkpoint), device=args.device,
                                            full_vocabulary=True)
    tensors = {}
    calls = {}
    handles = []
    wrapped = []
    def wrap_method(owner, method_name: str, stage: str):
        original = getattr(owner, method_name)

        def traced(*method_args, **method_kwargs):
            output = original(*method_args, **method_kwargs)
            index = calls.get(stage, 0)
            calls[stage] = index + 1
            flatten(output, f"{stage}.{index}", tensors, torch)
            return output

        setattr(owner, method_name, traced)
        wrapped.append((owner, method_name, original))

    wrap_method(model.model.backbone, "preprocess", "normalized_image")
    wrap_method(model.model.backbone, "extract", "fused_patch_map")
    wrap_method(model.model.box_prompt_encoder, "encode_pairs", "box_prompt_tokens")
    wrap_method(model.model.vocab_head, "score_query_dual", "predicate_logits")
    for name in STAGES:
        module = getattr(model.model, name)

        def capture(_module, _inputs, output, stage=name):
            index = calls.get(stage, 0)
            calls[stage] = index + 1
            flatten(output, f"{stage}.{index}", tensors, torch)

        handles.append(module.register_forward_hook(capture))
    handles.append(model.model.register_forward_hook(
        lambda _module, _inputs, output: flatten(output, "final", tensors, torch)))
    # Keep the two query branches in the golden trace as well.  These are
    # needed to distinguish a relation-context error from a vocabulary
    # projection/layout error in the C++ implementation.
    for name in ("vocab_head.proj", "sub_text_proj", "obj_text_proj",
                 "compose_norm", "spa_proj"):
        owner = model.model
        parts = name.split(".")
        for part in parts:
            owner = getattr(owner, part)
        handles.append(owner.register_forward_hook(
            lambda _module, _inputs, output, stage=name: flatten(output, stage, tensors, torch)))
    with Image.open(args.data / sample["image"]) as raw:
        image = raw.convert("RGB")
    boxes = np.asarray([[x, y, x + w, y + h] for x, y, w, h in
                        (obj["bbox"] for obj in sample["objects"])], dtype=np.float32)
    try:
        model.predict(image, boxes, topk=50)
        if args.device == "cuda":
            torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
        for owner, method_name, original in wrapped:
            setattr(owner, method_name, original)
    if not tensors:
        raise RuntimeError("no official model stage emitted a tensor")
    summary = {
        "checkpoint": args.model_name, "checkpoint_sha256": sha256(checkpoint),
        "dataset_revision": manifest["revision"], "manifest_sha256": sha256(manifest_path),
        "image_id": sample["image_id"], "image_sha256": sample["image_sha256"],
        "objects": len(sample["objects"]), "vocabulary_size": len(model.predicates),
        "device": args.device, "torch_version": torch.__version__,
        "module_calls": calls,
        "tensors": {name: tensor_stats(array, np) for name, array in tensors.items()},
        "raw_tensors": (str(args.tensors.relative_to(ROOT)) if args.tensors.is_relative_to(ROOT)
                        else str(args.tensors)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.tensors.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.tensors, **tensors)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"traced {len(tensors)} intermediate tensors for image {sample['image_id']}")
    print(f"wrote {args.output} and {args.tensors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
