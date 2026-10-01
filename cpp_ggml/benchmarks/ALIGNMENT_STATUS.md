# Full graph alignment status

Validated on 2026-09-30 with RTX 3060 and GGML v0.21.0.

| Requested component | Native implementation | Evidence |
|---|---|---|
| DINOv3 | S / S+ / B, 12 blocks, rotary positions, learned tap fusion | Fused patch-map traces |
| Box prompt | Fourier corners and learned embeddings | Prompt-token traces |
| Spatial pool | Object, union and contact pools | Object/pair-stage traces |
| Pair sampler | Learned geometry/relation scores, 400→128 selection | 50 images plus 60-region dense test |
| Relation transformer | Self/cross attention, masked slots, FFN | Transformer traces |
| Deformable read | Learned offsets, bilinear gathers, null samples | Deformable-stage traces |
| Dynamic vocabulary head | Semantic/spatial score mixture, new-bank normalization and gating | 27 custom vocabulary configurations |

All seven requested components execute in C++; all learned tensor operations use the selected GGML backend. Geometry construction and TopK selection use C++ host code. There is no Python inference callback. The explicit historical adapter remains solely for its fixtures.

## Completed evidence

- Nine full official GGUF models verified against prepared official runtime weights and published with matching remote SHA256/size.
- 900 real-image GGML configurations and 150 independent PyTorch-CUDA references; calibrated native decode checked against raw logits.
- 27 architecture/backend/storage stage checks: 297 finite intermediate boundaries, plus 27 custom vocabulary checks.
- Dense 60-region pair sampling across 21 configurations, exposing quantization and close-score candidate changes.
- 50/50 image and box preprocessing arrays bit equal; 0/1/2/60-object and malformed-input contracts on CPU/CUDA/Vulkan.
- CPU/CUDA/Vulkan builds and CTest pass. Link inspection finds no cuDNN. Pristine pinned patch replay restores the original tree on reverse application.
- 150 full comparison grids, per-image predictions, timing samples, CSV/JSON metrics and all model cards.

## Measured requirements that are not exact identities

| Model | Python ms | CUDA F32 / F16 / Q8 ms | Vulkan F32 / F16 / Q8 ms |
|---|---:|---|---|
| relsgg-vits16 | 28.01 | 24.37 / 25.69 / 20.87 | 29.17 / 29.07 / 28.66 |
| relsgg-vits16plus | 30.68 | 26.02 / 28.39 / 22.35 | 31.17 / 30.25 / 31.30 |
| relsgg-vitb16 | 49.65 | 41.02 / 44.41 / 30.50 | 42.57 / 42.85 / 42.28 |

All nine CUDA medians beat the corresponding Python baseline. Vulkan B is faster; S/S+ are close, with five of six S/S+ rows up to about 4% slower. Therefore the strict requirement that every Vulkan configuration be faster has not been demonstrated.

All six F32 GPU rows reproduce Python’s aggregate exact GT R@50 on this set. F16/Q8 rows differ by at most one of 803 annotated triples at R@50. Predicate ranks are not identical: low-margin logits can switch under different kernels and quantized weights. On the additional dense synthetic-box test, S+ Vulkan F32/F16 change one of 128 selected pairs, which changes contextual attention; the larger logit RMSE is retained in the stage report. These observations are not relabeled as exact numerical equality.

The box-only full graph is integrated. Optional upstream masks, detector-score weighting, decomposed output and text-to-embedding encoding inside C++ are outside the validated interface profile. The dynamic head accepts embeddings. The unavailable six-axis OV-SGG pack does not block the implemented regression suite, but its negative-dependent metrics and OVS composite are not fabricated.

[Full measured report](full_graph/README.md) · [Stage and dense tests](full_graph_validation/README.md) · [Model cards](../models/MODEL_CARDS.md)
