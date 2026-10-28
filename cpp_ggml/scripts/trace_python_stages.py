#!/usr/bin/env python3
"""Capture official Python intermediate tensors for future GGML stage parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from provenance import fingerprint, model_assets, python_fingerprint


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
STAGES = (
    "spatial_pool", "sampler", "geo_encoder",
    "pair_proj", "rel_transformer", "deformable_read", "rel_interaction",
    "spa_proj",
)
# First backbone block internals, in execution order, as (module suffix, side).
# The o_proj and down_proj inputs are the attention and activation outputs.
BLOCK0 = (
    ("model.layer.0.attention.q_proj", "output"), ("model.layer.0.attention.k_proj", "output"),
    ("model.layer.0.attention.v_proj", "output"), ("model.layer.0.attention.o_proj", "input"),
    ("model.layer.0.attention.o_proj", "output"), ("model.layer.0.layer_scale1", "output"),
    ("model.layer.0.norm2", "output"), ("model.layer.0.mlp.up_proj", "output"),
    ("model.layer.0.mlp.down_proj", "input"), ("model.layer.0.mlp.down_proj", "output"),
    ("model.layer.0.layer_scale2", "output"),
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
    # Choices follow the released model matrix (three ViT towers, four
    # ConvNeXt towers); RelateAnything.from_checkpoint handles both families.
    parser.add_argument("--model-name", default="relsgg-vits16plus",
                        choices=("relsgg-vits16", "relsgg-vits16plus", "relsgg-vitb16",
                                 "relsgg-convnext-tiny", "relsgg-convnext-small",
                                 "relsgg-convnext-base", "relsgg-convnext-large"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "cpp_ggml/outputs/probes/traces/python_stage_trace.json")
    parser.add_argument("--tensors", type=Path, default=ROOT / "cpp_ggml/outputs/probes/traces/python_stage_trace.npz")
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
    inputs = fingerprint([manifest_path,args.data/sample['image'],*model_assets(args.model_name)])
    sources = python_fingerprint(__file__)
    calls = {}
    wrapped = []
    # Captures wrap the instance attribute instead of registering module
    # hooks: a hook anywhere inside a TransformerEncoderLayer disables its
    # fused inference path, so a hooked trace would not be the official output.
    def wrap(owner, method_name: str, emit):
        original = getattr(owner, method_name)
        shadowed = method_name in vars(owner)

        def traced(*method_args, **method_kwargs):
            output = original(*method_args, **method_kwargs)
            emit(method_args, output)
            return output

        setattr(owner, method_name, traced)
        wrapped.append((owner, method_name, original if shadowed else None))

    def wrap_method(owner, method_name: str, stage: str):
        def emit(_args, output):
            index = calls.get(stage, 0)
            calls[stage] = index + 1
            flatten(output, f"{stage}.{index}", tensors, torch)
        wrap(owner, method_name, emit)

    wrap_method(model.model.backbone, "preprocess", "normalized_image")
    wrap_method(model.model.backbone, "extract", "fused_patch_map")
    wrap_method(model.model.box_prompt_encoder, "encode_pairs", "box_prompt_tokens")
    wrap_method(model.model.vocab_head, "score_query_dual", "predicate_logits")
    for name in STAGES:
        wrap_method(getattr(model.model, name), "forward", name)
    wrap(model.model, "forward", lambda model_args, output: (
        flatten(model_args[1], "input.boxes", tensors, torch),
        flatten(output, "final", tensors, torch)))
    for name, module in model.model.named_modules():
        if name.endswith('embeddings.patch_embeddings') or name.endswith('model.layer.0.norm1') or name.endswith('model.layer.0'):
            wrap(module, "forward", lambda _args, output, stage=name: flatten(output, stage, tensors, torch))
        if name.startswith('rel_transformer.') and name.count('.') == 2:
            wrap(module, "forward", lambda _args, output, stage=name: flatten(output, stage, tensors, torch))
        for suffix, side in BLOCK0:
            if name.endswith(suffix):
                wrap(module, "forward", lambda inputs, output, stage=name, side=side: flatten(
                    inputs[0] if side == "input" else output, f"{stage}:{side}", tensors, torch))
    # The ConvNeXt tower has a different module tree (stages of downsample +
    # gated blocks) and no attention internals, so its stage-boundary trace
    # wraps the four stage outputs directly; each is saved token-major
    # [C, H*W] to match the C++ convnext_stage_N taps exactly.
    tower = getattr(getattr(model.model, 'backbone', None), 'model', None)
    stages_module = getattr(getattr(tower, 'model', None), 'stages', None)
    if stages_module is not None:
        for index, stage in enumerate(stages_module):
            def emit_stage(_args, output, stage_index=index):
                # [B, C, H, W][0] -> [C, H*W]: the row-major layout already
                # matches the C++ token-major [C, h*w] tap byte for byte.
                tokens = output[0].reshape(output.shape[1], -1)
                flatten(tokens, f"convnext_stage_{stage_index}", tensors, torch)
            wrap(stage, "forward", emit_stage)
    # Keep the two query branches in the golden trace as well.  These are
    # needed to distinguish a relation-context error from a vocabulary
    # projection/layout error in the C++ implementation.
    for name in ("vocab_head.proj", "sub_text_proj", "obj_text_proj",
                 "compose_norm", "spa_proj",
                 "sampler.f_sub.0", "sampler.f_sub.1", "sampler.f_sub.2", "sampler.f_sub.3",
                 "sampler.f_obj.0", "sampler.f_obj.1", "sampler.f_obj.2", "sampler.f_obj.3"):
        owner = model.model
        parts = name.split(".")
        for part in parts:
            owner = getattr(owner, part)
        wrap(owner, "forward", lambda _args, output, stage=name: flatten(output, stage, tensors, torch))
    with Image.open(args.data / sample["image"]) as raw:
        image = raw.convert("RGB")
    boxes = np.asarray([[x, y, x + w, y + h] for x, y, w, h in
                        (obj["bbox"] for obj in sample["objects"])], dtype=np.float32)
    try:
        model.predict(image, boxes, topk=50)
        if args.device == "cuda":
            torch.cuda.synchronize()
    finally:
        for owner, method_name, original in reversed(wrapped):
            if original is None:
                delattr(owner, method_name)
            else:
                setattr(owner, method_name, original)
    if not tensors:
        raise RuntimeError("no official model stage emitted a tensor")
    summary = {
        "checkpoint": args.model_name, "checkpoint_sha256": sha256(checkpoint),
        "dataset_revision": manifest["revision"], "manifest_sha256": sha256(manifest_path),
        "image_id": sample["image_id"], "image_sha256": sample["image_sha256"],
        "objects": len(sample["objects"]), "vocabulary_size": len(model.predicates),
        "device": args.device, "torch_version": torch.__version__,
        "inputs": inputs, "python": sources,
        "math": {"tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                 "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                 "flash_sdp": torch.backends.cuda.flash_sdp_enabled()},
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
