# GGML porting notes

The full graph is being split at the same numerical boundaries used by the
working GGML ports already present under `/home/ludahai/develop/code/github/dl`.
These references are design inputs, not copied dependencies:

| RelateAnything stage | Local reference | Reuse boundary |
|---|---|---|
| DINOv3 patch stem, token layout, attention and RoPE | `depth-anything-cpp/src/dino_backbone.cpp`, `src/vit_block.cpp`; `rf-detr-ggml/src/dinov2.cpp` | GGML tensor layout, Q/K/V reshape, layer scale and RoPE math |
| multihead self/cross attention and transformer FFN | `rf-detr-ggml/src/heads.cpp`, `src/decoder.cpp` | projection, head packing and masked softmax kernels |
| full-graph GGUF loading and stage dumps | `map-anything-ggml/cpp_ggml/src/gguf_loader.cpp`, `src/vggt_graph.cpp` | explicit options, backend allocation, input lifetime and raw tensor dumps |
| v0.21 conversion and parity gates | `General-Keypoint-Detection-GGML/cpp_ggml`, `sam3-ggml` | exporter shape convention, patch replay and raw tensor comparison |

## Implemented graph

The deployment path is native C++ and uses the same released weights as the
Python API. No Python inference is invoked by `FullModel`.

| Stage | Implementation and verified boundary |
|---|---|
| Image preprocessing | libjpeg / GGML's stb decoder, Pillow-compatible separable resize, normalized CHW; 50/50 pixel and box arrays bit equal |
| DINOv3 | patch convolution, CLS/register tokens, 2-D rotary positions, 12 attention/FFN blocks, layer scale, tap fusion; GELU and S+ SwiGLU variants |
| Spatial pool | object/union/contact pools on the fused patch map |
| Box prompt | learned corner embeddings and Fourier positional projection |
| Pair sampler | GGML learned geometry and relation scoring; native stable TopK, 400 geometry candidates then 128 relation slots |
| Relation transformer | self/cross attention, masked invalid slots, scene positions and FFN |
| Deformable read | native GGML bilinear gathers, learned offsets/weights and null samples |
| Dynamic vocabulary | semantic composition, spatial branch, learned gate, normalized embeddings and calibrated triplets |

`validate_full_graph.py` compares eleven intermediate boundaries against
independent official Python traces for all three architectures. It also
replaces the vocabulary with scaled known embeddings and a new composite
embedding to exercise normalization, the gate MLP and a changed output shape.
The full image benchmark compares all 50 images and all 18 accelerator/storage
combinations. See [current measurements](full_graph/README.md).

## Root causes resolved

The earlier adapter used six artificial `ra.*` tensors and precomputed object
features. The real checkpoints contain 392 or 416 state tensors, image tower
weights and a 512-D relation head. Supporting them required the real graph;
renaming tensors or wrapping the adapter could not produce equivalent results.

The checkpoint's stored vocabulary weights are also not sufficient to match
`full_vocabulary=True`: upstream replaces `vocab_head.W` using the released
embedding sidecar and recomputes alpha. The exporter now performs exactly that
runtime preparation and embeds the resulting bank, gates, names and calibration.
Every exported value is checked after the declared F16/Q8 storage conversion.

GGML tensor layouts, contiguous rotary negation, invalid-slot masks, boundary
sampling, box normalization order and decimal-to-F32 conversion were checked
against intermediate traces. Native calibrated output is independently checked
against separately exported raw logits on every benchmark image.

## Performance choices and limits

All learned operations use the chosen GGML backend. Host pair selection creates
two graph executions per image and transfers fused/object features across that
boundary; this cost is included. Fixed positional encodings are cached. DINOv3 Q/K/V weights are packed once
at load; relation self-attention QKV and cross-attention KV use combined
projections to reduce dispatches. The added packed-weight memory is measured. GPU
flash attention uses F16 K/V with F32 accumulation; explicit F32 attention is
available with `--no-flash-attention`. Matrix products request F32 accumulation.
Weight storage names do not imply bit-exact arithmetic throughout the graph.

JPEG decoding and the fused RGB resize loops reproduce the official PIL path.
The resize algorithm is attributed in `third_party/patches/PILLOW_LICENSE`.
No GGML operation patch was needed: stock v0.21.0 implements the learned graph.
The ordered CMake patch path remains in place and currently applies only the
integration marker. cuDNN is neither found nor linked by this module.

The supported profile is batch 1, 448x448, GT/detector boxes, released DINOv3
architectures and 128 pair slots. The optional upstream masks, detector score
weighting and decomposed output are outside this box-only profile. Text
embeddings can be supplied dynamically; text-to-embedding execution remains
available through the official Python API.
