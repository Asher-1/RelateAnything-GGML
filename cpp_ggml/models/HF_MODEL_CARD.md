---
library_name: ggml
license: other
license_name: dinov3-license
license_link: https://ai.meta.com/resources/models-and-libraries/dinov3-license/
tags:
- gguf
- scene-graph-generation
- visual-relationship-detection
- open-vocabulary
- cuda
- vulkan
---

# RelateAnything native full-graph GGUF

Nine downstream conversions of the three official RelateAnything models run
with native C++17/GGML on CPU, CUDA and Vulkan. The graph implements DINOv3,
box prompt, spatial pool, pair sampler, relation transformer, deformable read,
interaction and the dynamic vocabulary head, including calibrated triplets.
C++ inference needs no PyTorch, Python model execution or cuDNN.

Source: [Asher-1/RelateAnything-GGML](https://github.com/Asher-1/RelateAnything-GGML).
All tensors have passed the declared storage conversion check. The full released
vocabulary and calibration match the official full-vocabulary Python API.

## Model families and intended scenarios

| Official checkpoint | Backbone / upstream parameters | Scenario |
|---|---|---|
| [relsgg-vits16](https://huggingface.co/maelic/relsgg-vits16) | DINOv3 ViT-S/16, 46.1M | Lowest-capacity released tower; latency and CPU-sensitive deployment |
| [relsgg-vits16plus](https://huggingface.co/maelic/relsgg-vits16plus) | DINOv3 ViT-S/16+, 53.2M, SwiGLU | Upstream recommended balance of relation quality and cost |
| [relsgg-vitb16](https://huggingface.co/maelic/relsgg-vitb16) | DINOv3 ViT-B/16, 113.8M | Larger visual capacity; use when memory and latency permit |

The checked-out upstream [release manifest](https://github.com/Asher-1/RelateAnything-GGML/blob/main/deploy/release_manifest.json) identifies exactly these three published `final` models. Its three `*-zeroshot` siblings are explicitly `unreleased`, have no `hf_repo`, and have no downloadable weights. They are not omitted supported releases. ONNX/TensorRT exports are deployment formats, not additional checkpoint families; the optional S+ ONNX file is not needed by GGML.

Each local official snapshot contains `model.pth`, `config.json`, `relateanything.json`, `calibration.json`, `thresholds.json`, `predicate_embeddings.npz`, `predicate_bank.npz`, `text_student.pt`, `tokenizer.json` and `tokenizer_config.json`. Text assets support official Python vocabulary encoding; the C++ head accepts a precomputed replacement embedding bank.

## Nine direct downloads and fingerprints

Every artifact contains DINOv3, prompt/pool/sampler, relation transformer, deformable read, interaction and dynamic vocabulary weights. Export initializes the official full-vocabulary API: the 19,103-row released embedding bank and recomputed alpha replace the checkpoint bank. Predicate names, calibration and source hashes are embedded.

| Model download | Size MiB | Tensors | SHA256 |
|---|---:|---:|---|
| [relsgg-vitb16-f16.gguf](https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/gguf/relsgg-vitb16-f16.gguf) | 236.22 | 392 | `61222e4099c277d5272524507721fc6d65649051bae23984a06fe5cacb8e0806` |
| [relsgg-vitb16-f32.gguf](https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/gguf/relsgg-vitb16-f32.gguf) | 471.96 | 392 | `1537ba73ae9042b9553774269a4bf44ecfba7ea8776cac44acbea02090fe0934` |
| [relsgg-vitb16-q8_0.gguf](https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/gguf/relsgg-vitb16-q8_0.gguf) | 128.03 | 392 | `61bc2a07233a1d0e189dbcab96736d38afa9f9ade8d8028b327e3aa7a12c3dd5` |
| [relsgg-vits16-f16.gguf](https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/gguf/relsgg-vits16-f16.gguf) | 107.17 | 392 | `e00c896a5ad816cdd6af285d63346baab27e5c002935dba59ab0a8e124828a59` |
| [relsgg-vits16-f32.gguf](https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/gguf/relsgg-vits16-f32.gguf) | 213.86 | 392 | `ce90fb86c08716a45b9ca85f0bbfe74787966bf17eb267c3ad23f91094ced081` |
| [relsgg-vits16-q8_0.gguf](https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/gguf/relsgg-vits16-q8_0.gguf) | 58.44 | 392 | `ee168035fa5fa2dc334e159c70ab4785e1bf9c4758427c1264b59878200fa429` |
| [relsgg-vits16plus-f16.gguf](https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/gguf/relsgg-vits16plus-f16.gguf) | 120.71 | 416 | `4e5631450416ec2496623631b7f7e2bde9f399570fe747fad1dd9a5f20fbcb37` |
| [relsgg-vits16plus-f32.gguf](https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/gguf/relsgg-vits16plus-f32.gguf) | 240.94 | 416 | `996960494629efd0035379c2fa8ae7ca1ff0039b53d8ab61637e86b7e05ae52e` |
| [relsgg-vits16plus-q8_0.gguf](https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/gguf/relsgg-vits16plus-q8_0.gguf) | 65.69 | 416 | `bb70680859abf090bb3942871a5932aa07155f6236e87c6999218ed14400c8f9` |

[The machine-readable manifest](benchmarks/full_gguf_manifest.json) records the exact source checkpoint SHA, type counts, missing/extra tensors and numerical verification. All nine files pass the declared storage-conversion check.

## Storage policy and selection

| Storage | Conversion | Suitable use |
|---|---|---|
| F32 | Prepared runtime floating tensors stored in F32 | Numerical reference, closest weight representation to official PyTorch |
| F16 | Floating tensors stored in F16; auxiliary values converted as needed by native ops | Smaller files and weight buffers; latency depends on backend kernels |
| Q8_0 | Rank-2 matrices with inner dimension divisible by 32 use Q8_0; other tensors stay F32 | Lower weight bandwidth/storage, particularly useful on CUDA; inspect ranking drift for your vocabulary |

GPU flash attention uses F16 K/V and F32 accumulation for every storage type. `--no-flash-attention` selects explicit F32 attention. Q8/F16 cannot reproduce original F32 weights bit-for-bit, and close-score predicates/candidates may change. Packed Q/K/V weights are created once at load without changing the stored model and consume additional runtime memory.

## Measured full-pipeline evidence

The [full report](benchmarks/full_graph/README.md) and [metrics CSV](benchmarks/full_graph/metrics.csv) are the current source for all 21 rows: three independent official PyTorch-CUDA baselines plus 18 native CUDA/Vulkan variants. All 50 retained RA-4M images, GT boxes and the full vocabulary are evaluated. Native timing includes JPEG decode, exact preprocessing, graph construction/allocation/transfers, inference and calibrated ranking. Loading and output files are excluded on both sides.

![Measured pipeline latency and accuracy](benchmarks/full_graph/metrics.png)

![Metrics by image source](benchmarks/full_graph/source_breakdown.png)

Per-image grids cover every retained frame and every engine/storage variant. Numerical metrics include MAE, RMSE, max absolute error, cosine, predicate agreement and pair recall/Jaccard. Task metrics include R/mR@20/50/100, their harmonic F1, weighted recall and explicit local support buckets. Median/P95, throughput and allocator-scoped memory measurements accompany the accuracy rows.

GGML memory is the sum of native weight/packed-QKV/scheduler buffers; PyTorch reports peak allocator bytes. These do not have identical scope and are not advertised as directly comparable total VRAM peaks. Shared-machine hardware/process snapshots are recorded. Refer to individual speedups instead of assuming every lower-precision format is faster.

The [stage report](benchmarks/full_graph_validation/README.md) checks eleven intermediate boundaries, custom vocabulary normalization/gating and 60-region dense candidate selection. The compact GT set has incomplete positive annotations; it cannot establish negative-set AP, OVS composite or detector accuracy. [Metric coverage](benchmarks/METRIC_COVERAGE.md) explains those distinctions.


### Measured model/backend/storage matrix

RTX 3060; 50 GT-box images, 19,103 predicates; 2 warmups, 3 repeats per image. Speedup is each model’s Python median divided by its GGML median. All accepted timing runs were monitored and had no transient GPU job; two persistent system GPU services were explicitly allowed on both sides.

| Model | Engine/storage | Median / P95 ms | Speedup | R@50 | mR@50 | Logit RMSE | Top-1 | GGML buffers MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| relsgg-vits16 | cuda-f16 | 25.69 / 27.72 | 1.09x | 14.57% | 11.38% | 0.00432 | 99.74% | 141.4 |
| relsgg-vits16 | cuda-f32 | 24.37 / 27.34 | 1.15x | 14.45% | 11.37% | 0.00267 | 99.64% | 258.3 |
| relsgg-vits16 | cuda-q8_0 | 20.87 / 22.79 | 1.34x | 14.45% | 11.37% | 0.04015 | 97.38% | 88.0 |
| relsgg-vits16 | python | 28.01 / 30.26 | 1.00x | 14.45% | 11.37% | 0.00000 | 100.00% | — |
| relsgg-vits16 | vulkan-f16 | 29.07 / 32.03 | 0.96x | 14.45% | 11.37% | 0.00331 | 99.82% | 141.4 |
| relsgg-vits16 | vulkan-f32 | 29.17 / 32.08 | 0.96x | 14.45% | 11.37% | 0.00256 | 99.87% | 258.3 |
| relsgg-vits16 | vulkan-q8_0 | 28.66 / 31.96 | 0.98x | 14.45% | 11.37% | 0.02609 | 98.16% | 88.0 |
| relsgg-vits16plus | cuda-f16 | 28.39 / 30.47 | 1.08x | 16.94% | 10.47% | 0.00451 | 99.69% | 155.0 |
| relsgg-vits16plus | cuda-f32 | 26.02 / 27.95 | 1.18x | 17.06% | 10.96% | 0.00357 | 99.69% | 285.3 |
| relsgg-vits16plus | cuda-q8_0 | 22.35 / 24.30 | 1.37x | 17.19% | 11.43% | 0.03588 | 97.68% | 95.2 |
| relsgg-vits16plus | python | 30.68 / 32.71 | 1.00x | 17.06% | 10.96% | 0.00000 | 100.00% | — |
| relsgg-vits16plus | vulkan-f16 | 30.25 / 33.91 | 1.01x | 16.94% | 10.47% | 0.00437 | 99.75% | 155.0 |
| relsgg-vits16plus | vulkan-f32 | 31.17 / 34.81 | 0.98x | 17.06% | 10.96% | 0.00332 | 99.61% | 285.3 |
| relsgg-vits16plus | vulkan-q8_0 | 31.30 / 34.78 | 0.98x | 16.94% | 10.47% | 0.02446 | 98.36% | 95.2 |
| relsgg-vitb16 | cuda-f16 | 44.41 / 47.47 | 1.12x | 17.43% | 12.98% | 0.00363 | 99.57% | 304.8 |
| relsgg-vitb16 | cuda-f32 | 41.02 / 43.23 | 1.21x | 17.56% | 13.00% | 0.00255 | 99.74% | 581.1 |
| relsgg-vitb16 | cuda-q8_0 | 30.50 / 32.71 | 1.63x | 17.56% | 12.88% | 0.03382 | 97.65% | 177.7 |
| relsgg-vitb16 | python | 49.65 / 52.44 | 1.00x | 17.56% | 13.00% | 0.00000 | 100.00% | — |
| relsgg-vitb16 | vulkan-f16 | 42.85 / 45.43 | 1.16x | 17.43% | 12.98% | 0.00284 | 99.67% | 306.9 |
| relsgg-vitb16 | vulkan-f32 | 42.57 / 46.24 | 1.17x | 17.56% | 13.00% | 0.00227 | 99.79% | 581.2 |
| relsgg-vitb16 | vulkan-q8_0 | 42.28 / 44.82 | 1.17x | 17.56% | 12.87% | 0.01913 | 98.60% | 177.8 |

CUDA is faster for all nine measured variants. Vulkan is close to Python for S/S+ and faster for B; some S/S+ rows are slightly slower. No claim that every configuration exceeds Python is made. F32 does not imply bit-identical GPU kernels; full errors and candidate changes are disclosed.

## Supported interface and limits

- Batch 1, 448×448, boxes, up to `Options.max_boxes` (default 60), 128 candidate slots; all three released architectures.
- CPU, CUDA and Vulkan execute native GGML graphs; no Python inference or cuDNN dependency in C++.
- `Options` configures model/backend/precision mode/threads/vocabulary; `ImageOptions` supplies image and pixel XYXY boxes.
- Replacement vocabularies use finite row-major F32 `[V,512]` embeddings. The C++ head normalizes them and computes its learned gate. Free-form text encoding is available with the Python engine.
- Optional upstream masks, detector-score weighting and decomposed outputs are outside this box-only deployment profile.
- GT-box evaluation measures relation prediction, not detector localization or end-to-end detection recall.

```sh
./run_ggml.sh --backend cuda --model-name relsgg-vits16plus --dtype q8_0 --download
./run_ggml.sh --backend vulkan --model-name relsgg-vitb16 --dtype f16
./run_ggml.sh --engine python --backend cuda
```


## Terms

This is a downstream GGUF conversion, not an upstream-official GGUF release. The source model cards identify the DINOv3 model license; conversion does not replace those terms. See the [source model card](https://huggingface.co/maelic/relsgg-vits16plus) and [DINOv3 license](https://ai.meta.com/resources/models-and-libraries/dinov3-license/). Source code licensing remains governed by this repository’s LICENSE. Retained dataset images keep their original COCO/Objects365/Open Images terms.
