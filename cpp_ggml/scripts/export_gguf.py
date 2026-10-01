#!/usr/bin/env python3
"""Export RelateAnything checkpoints to GGUF.

``--full`` preserves every tensor from an official RelateAnything checkpoint,
including the fine-tuned DINOv3 tower, relation head and dynamic vocabulary.
The runtime vocabulary and calibration match the official full-vocabulary API.
``--demo`` creates a deterministic adapter fixture, separate from real models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from gguf import GGUFWriter, GGMLQuantizationType as Q
from gguf.quants import quantize


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONTRACT = "relation_pair_linear_v1"
FULL_CONTRACT = "relateanything_full_v1"


def _tensor_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _load_state(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to inspect a PyTorch checkpoint") from exc
    # The official release checkpoint stores NumPy metadata alongside weights.
    # Match relsgg.checkpoint.load_checkpoint for explicitly supplied files.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        payload = payload["state_dict"]
    elif isinstance(payload, dict) and isinstance(payload.get("ema_model"), dict):
        payload = payload["ema_model"]
    elif isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        payload = payload["model"]
    if not isinstance(payload, dict):
        raise RuntimeError("checkpoint must contain a state_dict mapping")
    export = payload.get("ggml_export")
    if isinstance(export, dict):
        payload = export
    return payload


def _load_checkpoint(path: Path) -> dict[str, Any]:
    """Load the complete release payload without dropping checkpoint metadata."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to inspect a PyTorch checkpoint") from exc
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise RuntimeError("official checkpoint must contain a model state mapping")
    # The release checkpoint contains the training-time vocabulary head.  The
    # official Python full-vocabulary path immediately reparameterizes that
    # head with predicate_embeddings.npz (and recomputes alpha through the
    # gate MLP).  Export those runtime values so C++ consumes the same model
    # that the Python reference actually evaluates.
    try:
        from relsgg import RelateAnything
        runtime = RelateAnything.from_checkpoint(
            str(path), device="cpu", full_vocabulary=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"failed to materialize the official full vocabulary for {path}; "
            "the sidecar predicate_embeddings.npz and relsgg package are "
            "required for a parity-safe GGUF export"
        ) from exc
    runtime_state = runtime.model.state_dict()
    for key in ("vocab_head.W", "vocab_head.alpha"):
        value = runtime_state.get(key)
        if value is None:
            raise RuntimeError(f"runtime full-vocabulary model is missing {key}")
        payload["model"][key] = value.detach().cpu().float().clone()
    payload["pred_names"] = list(runtime.predicates)
    payload["ggml_calibration"] = {"a": runtime.calib_a, "b": runtime.calib_b}
    payload["ggml_provenance"] = {
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "vocabulary_sha256": hashlib.sha256((path.parent / "predicate_embeddings.npz").read_bytes()).hexdigest(),
    }
    return payload


def _resolve_adapter(state: dict[str, Any]) -> dict[str, np.ndarray]:
    aliases = {
        "ra.pair_proj.weight": ("ra.pair_proj.weight", "ggml_pair_proj.weight"),
        "ra.pair_proj.bias": ("ra.pair_proj.bias", "ggml_pair_proj.bias"),
        "ra.predicate.weight": ("ra.predicate.weight", "ggml_predicate.weight"),
        "ra.predicate.bias": ("ra.predicate.bias", "ggml_predicate.bias"),
        "ra.pair.weight": ("ra.pair.weight", "ggml_pair.weight"),
        "ra.pair.bias": ("ra.pair.bias", "ggml_pair.bias"),
    }
    missing = [name for name, keys in aliases.items() if not any(key in state for key in keys)]
    if missing:
        prefixes = sorted({name.split(".", 1)[0] for name in state})
        raise RuntimeError(
            "checkpoint does not contain the explicit relation_pair_linear_v1 adapter; "
            "use --full for official DINOv3 checkpoints. Missing: " +
            ", ".join(missing) + ". Available state tensors: " + str(len(state)) +
            "; top-level prefixes: " + ", ".join(prefixes)
        )
    return {name: _tensor_array(next(state[key] for key in keys if key in state)) for name, keys in aliases.items()}


def _demo_tensors() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260930)
    feature_dim, hidden_dim, predicates = 12, 32, 4
    return {
        "ra.pair_proj.weight": (rng.standard_normal((hidden_dim, feature_dim * 2 + 8), dtype=np.float32) * 0.04),
        "ra.pair_proj.bias": np.zeros(hidden_dim, dtype=np.float32),
        "ra.predicate.weight": (rng.standard_normal((predicates, hidden_dim), dtype=np.float32) * 0.04),
        "ra.predicate.bias": np.zeros(predicates, dtype=np.float32),
        "ra.pair.weight": (rng.standard_normal((1, hidden_dim), dtype=np.float32) * 0.04),
        "ra.pair.bias": np.zeros(1, dtype=np.float32),
    }


def _add_tensor(writer: GGUFWriter, name: str, value: np.ndarray, dtype: str) -> None:
    # GGUF writes dimensions in reverse order. A PyTorch-style [out, in]
    # array therefore becomes GGML ne[0]=in, ne[1]=out automatically.
    if value.ndim == 2:
        raw_shape = value.shape
    else:
        raw_shape = value.shape
    if dtype == "q8_0" and value.ndim == 2 and value.shape[1] % 32 == 0:
        if value.shape[1] % 32:
            raise RuntimeError(f"Q8_0 requires the input dimension to be divisible by 32: {name} shape={value.shape}")
        data = quantize(value, Q.Q8_0)
        # add_tensor_info converts the byte shape back to the logical shape for
        # quantized tensors, so pass the quantized byte array shape here.
        writer.add_tensor(name, data, raw_shape=data.shape, raw_dtype=Q.Q8_0)
    else:
        data = value.astype(np.float16 if dtype == "f16" else np.float32, copy=False)
        writer.add_tensor(name, data, raw_shape=raw_shape)


def _full_tensors(payload: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    state = payload["model"]
    tensors: dict[str, np.ndarray] = {}
    tensor_name_map: list[str] = []
    for key, value in state.items():
        if not hasattr(value, "detach"):
            continue
        array = _tensor_array(value)
        if array.dtype.kind not in "fiu":
            raise RuntimeError(f"unsupported checkpoint tensor dtype for {key}: {array.dtype}")
        original_name = f"ra.{key}"
        # GGUF v3 in ggml 0.21 caps tensor names at 63 bytes.  Keep short
        # names readable and hash only the rare long names; the alternating
        # original/emitted list is stored as metadata for a lossless lookup.
        emitted_name = original_name
        if len(emitted_name.encode("utf-8")) >= 64:
            emitted_name = "ra.t." + hashlib.sha256(emitted_name.encode()).hexdigest()[:24]
        tensors[emitted_name] = array
        if emitted_name != original_name:
            tensor_name_map.extend((original_name, emitted_name))
    if not tensors:
        raise RuntimeError("checkpoint contains no tensor weights")
    cfg = payload.get("backbone_config") or {}
    pair_proj = tensors.get("ra.pair_proj.weight")
    if pair_proj is None or pair_proj.ndim != 2:
        raise RuntimeError("checkpoint is missing pair_proj.weight; cannot describe full graph")
    hidden = int(pair_proj.shape[0])
    backbone_dim = int(cfg.get("hidden_size", 0))
    if backbone_dim <= 0:
        raise RuntimeError("backbone_config.hidden_size is missing")
    ffn_type = "swiglu" if "ra.backbone.model.model.layer.0.mlp.gate_proj.weight" in tensors else "gelu"
    meta = {
        "hidden_size": backbone_dim,
        "intermediate_size": int(cfg.get("intermediate_size", 0)),
        "depth": int(cfg.get("num_hidden_layers", 0)),
        "num_heads": int(cfg.get("num_attention_heads", 0)),
        "patch_size": int(cfg.get("patch_size", 16)),
        "num_register_tokens": int(cfg.get("num_register_tokens", 4)),
        "rope_theta": float(cfg.get("rope_theta", 100.0)),
        "layer_norm_eps": float(cfg.get("layer_norm_eps", 1e-5)),
        "relation_dim": hidden,
        "text_dim": int(tensors.get("ra.vocab_head.W", np.empty((0, 0))).shape[-1]),
        "predicate_count": int(tensors.get("ra.vocab_head.W", np.empty((0, 0))).shape[0]),
        "ffn_type": ffn_type,
        "pred_names": list(payload.get("pred_names") or []),
        "tensor_name_map": tensor_name_map,
    }
    required = ("depth", "num_heads", "intermediate_size", "predicate_count", "text_dim")
    if any(int(meta[k]) <= 0 for k in required):
        raise RuntimeError(f"incomplete full graph metadata: {meta}")
    return tensors, meta


def export_full(payload: dict[str, Any], output: Path, dtype: str, source: str) -> None:
    tensors, meta = _full_tensors(payload)
    writer = GGUFWriter(output, "relateanything")
    writer.add_string("general.name", output.stem)
    writer.add_string("general.description", "RelateAnything official full checkpoint weights")
    writer.add_string("ra.graph_kind", FULL_CONTRACT)
    writer.add_string("ra.model_family", "dinov3_vit")
    writer.add_string("ra.quantization", dtype)
    writer.add_string("ra.quantization_policy", "q8_0 rank-2 tensors with input dimension divisible by 32; other tensors keep f16/f32")
    writer.add_string("ra.source", source)
    writer.add_string("ra.vocabulary_source", "official predicate_embeddings.npz with runtime gate reparameterization")
    for key, value in payload.get("ggml_provenance", {}).items():
        writer.add_string("ra." + key, value)
    calibration = payload.get("ggml_calibration", {"a": 1.0, "b": 0.0})
    writer.add_float32("ra.calibration_a", calibration["a"])
    writer.add_float32("ra.calibration_b", calibration["b"])
    writer.add_uint32("ra.hidden_size", meta["hidden_size"])
    writer.add_uint32("ra.intermediate_size", meta["intermediate_size"])
    writer.add_uint32("ra.depth", meta["depth"])
    writer.add_uint32("ra.num_heads", meta["num_heads"])
    writer.add_uint32("ra.patch_size", meta["patch_size"])
    writer.add_uint32("ra.num_register_tokens", meta["num_register_tokens"])
    writer.add_float32("ra.rope_theta", meta["rope_theta"])
    writer.add_float32("ra.layer_norm_eps", meta["layer_norm_eps"])
    writer.add_uint32("ra.relation_dim", meta["relation_dim"])
    writer.add_uint32("ra.text_dim", meta["text_dim"])
    writer.add_uint32("ra.predicate_count", meta["predicate_count"])
    writer.add_string("ra.ffn_type", meta["ffn_type"])
    writer.add_array("ra.layer_offsets", [-6, -3, -1])
    if meta["pred_names"]:
        writer.add_array("ra.predicate_names", meta["pred_names"])
    if meta["tensor_name_map"]:
        writer.add_array("ra.tensor_name_map", meta["tensor_name_map"])
    for name, value in tensors.items():
        _add_tensor(writer, name, value, dtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(json.dumps({"output": str(output), "sha256": digest,
                      "graph_kind": FULL_CONTRACT, "dtype": dtype,
                      "tensor_count": len(tensors), "predicate_count": meta["predicate_count"]},
                     sort_keys=True))


def export(tensors: dict[str, np.ndarray], output: Path, dtype: str, source: str,
           reference_npz: Path | None = None) -> None:
    pair_proj = tensors["ra.pair_proj.weight"]
    predicate = tensors["ra.predicate.weight"]
    if pair_proj.ndim != 2 or predicate.ndim != 2:
        raise RuntimeError("adapter matrices must be rank-2")
    if pair_proj.shape[1] <= 8 or (pair_proj.shape[1] - 8) % 2:
        raise RuntimeError("pair projection shape must be [hidden, 2*feature_dim+8]")
    feature_dim = (pair_proj.shape[1] - 8) // 2
    hidden_dim = pair_proj.shape[0]
    if predicate.shape[1] != hidden_dim or tensors["ra.pair.weight"].shape != (1, hidden_dim):
        raise RuntimeError("adapter hidden dimensions do not agree")
    writer = GGUFWriter(output, "relateanything")
    writer.add_string("general.name", output.stem)
    writer.add_string("general.description", "RelateAnything relation_pair_linear_v1 adapter")
    writer.add_string("ra.graph_kind", CONTRACT)
    writer.add_uint32("ra.feature_dim", feature_dim)
    writer.add_uint32("ra.hidden_dim", hidden_dim)
    writer.add_uint32("ra.predicate_count", predicate.shape[0])
    writer.add_string("ra.quantization", dtype)
    writer.add_string("ra.source", source)
    for name, value in tensors.items():
        _add_tensor(writer, name, value, dtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    if reference_npz:
        reference_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(reference_npz, **tensors)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(json.dumps({"output": str(output), "sha256": digest, "graph_kind": CONTRACT, "dtype": dtype}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=("f32", "f16", "q8_0"), default="f32")
    parser.add_argument("--demo", action="store_true", help="write a deterministic adapter smoke-test model")
    parser.add_argument("--full", action="store_true", help="export every tensor in an official checkpoint")
    parser.add_argument("--reference-npz", type=Path, help="also write raw adapter weights for the NumPy reference")
    args = parser.parse_args()
    if args.demo and args.full:
        parser.error("--demo and --full are mutually exclusive")
    if args.demo and args.checkpoint:
        parser.error("--demo and --checkpoint are mutually exclusive")
    if not args.demo and not args.checkpoint:
        parser.error("--checkpoint is required unless --demo is used")
    official_dir = ROOT / "cpp_ggml/models/gguf"
    if args.demo and args.out.resolve().is_relative_to(official_dir.resolve()):
        parser.error("random-weight test fixtures belong under cpp_ggml/test_data/fixtures, not models/gguf")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.full:
        if args.reference_npz:
            parser.error("--reference-npz only applies to the adapter fixture")
        export_full(_load_checkpoint(args.checkpoint), args.out, args.dtype, str(args.checkpoint))
    else:
        tensors = _demo_tensors() if args.demo else _resolve_adapter(_load_state(args.checkpoint))
        export(tensors, args.out, args.dtype, "demo" if args.demo else str(args.checkpoint), args.reference_npz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
