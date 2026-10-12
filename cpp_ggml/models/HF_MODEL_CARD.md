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

Execution provenance and storage conversion are verified. Numerical and performance gates, effective auto precision modes, and any accepted deviation are recorded in the [release acceptance status](benchmarks/RELEASE_STATUS.md).

## Model families and intended scenarios

| Official checkpoint | Backbone / upstream parameters | Scenario |
|---|---|---|
| [relsgg-vits16](https://huggingface.co/maelic/relsgg-vits16) | DINOv3 ViT-S/16, 46.1M | Lowest-capacity released tower; latency and CPU-sensitive deployment |
| [relsgg-vits16plus](https://huggingface.co/maelic/relsgg-vits16plus) | DINOv3 ViT-S/16+, 53.2M, SwiGLU | Upstream recommended balance of relation quality and cost |
| [relsgg-vitb16](https://huggingface.co/maelic/relsgg-vitb16) | DINOv3 ViT-B/16, 113.8M | Larger visual capacity; use when memory and latency permit |

The checked-out upstream [release manifest](https://github.com/Asher-1/RelateAnything-GGML/blob/main/deploy/release_manifest.json) identifies exactly these three published `final` models. Its three `*-zeroshot` siblings are explicitly `unreleased`, have no `hf_repo`, and have no downloadable weights. They are not omitted supported releases. ONNX/TensorRT exports are deployment formats, not additional checkpoint families; the optional S+ ONNX file is not needed by GGML.

Each local official snapshot contains `model.pth`, `config.json`, `relateanything.json`, `calibration.json`, `thresholds.json`, `predicate_embeddings.npz`, `predicate_bank.npz`, `text_student.pt`, `tokenizer.json` and `tokenizer_config.json`. C++ accepts a replacement embedding bank or encodes free text with the exported text-student GGUF and official tokenizer.

## Nine direct downloads and fingerprints

Every artifact contains DINOv3, prompt/pool/sampler, relation transformer, deformable read, interaction and dynamic vocabulary weights. Export initializes the official full-vocabulary API: the 19,103-row released embedding bank and recomputed alpha replace the checkpoint bank. Predicate names, calibration and source hashes are embedded.

| Model download | Size MiB | Tensors | SHA256 |
|---|---:|---:|---|
| [relsgg-vitb16-f16.gguf](https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/gguf/relsgg-vitb16-f16.gguf) | 236.22 | 392 | `fbd6e868e171024182535c281b6ef142b166ea4e67060209d8cfd4b01ac5e911` |
| [relsgg-vitb16-f32.gguf](https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/gguf/relsgg-vitb16-f32.gguf) | 471.96 | 392 | `f7d2aa5765877e945d15fc9433c4c947a10e948091692b3f5208c71c24b8e17f` |
| [relsgg-vitb16-q8_0.gguf](https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/gguf/relsgg-vitb16-q8_0.gguf) | 136.78 | 392 | `44f40492955b493c53d796c7b4ad65d968dce301cd09567920313bf88e4c2b61` |
| [relsgg-vits16-f16.gguf](https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/gguf/relsgg-vits16-f16.gguf) | 107.17 | 392 | `bd5d0eabc41747af6d0243c755f789917c3f53864f763989bee5a4f0371767cf` |
| [relsgg-vits16-f32.gguf](https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/gguf/relsgg-vits16-f32.gguf) | 213.86 | 392 | `b87e7cff5e5678e2871cbfcf2804be22e5a9b814a89be763c54c95b9c39468f4` |
| [relsgg-vits16-q8_0.gguf](https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/gguf/relsgg-vits16-q8_0.gguf) | 67.19 | 392 | `d281ace4ad9a51edc78da26660ffd89a98d073df352de735711812b48e2dee53` |
| [relsgg-vits16plus-f16.gguf](https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/gguf/relsgg-vits16plus-f16.gguf) | 120.71 | 416 | `36ed9f832324dff4d06edf216782eb692645f8f10c0bd7d1b4c5fa58d44d7d39` |
| [relsgg-vits16plus-f32.gguf](https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/gguf/relsgg-vits16plus-f32.gguf) | 240.94 | 416 | `24970d851decd61bfdf4b109f5353ff81596ae8d7c9a35635f9de89d17f935e4` |
| [relsgg-vits16plus-q8_0.gguf](https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/gguf/relsgg-vits16plus-q8_0.gguf) | 74.43 | 416 | `c19c25acae196e3a762fcbb4032ba450e716847bc66540988d730e4c75572177` |

### Native text-student downloads

Free-text encoding uses one additional F32 student for the selected model family, plus its official tokenizer.json. Image weights remain unchanged.

| Text student | Size MiB | Tensors | SHA256 |
|---|---:|---:|---|
| [relsgg-vitb16-text-student-f32.gguf](https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/gguf/relsgg-vitb16-text-student-f32.gguf) | 42.86 | 78 | `8cd368fa8692a987a4d8739fcc642dc74fd2902d73ac5544a65f24018656475b` |
| [relsgg-vits16-text-student-f32.gguf](https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/gguf/relsgg-vits16-text-student-f32.gguf) | 42.86 | 78 | `8cd368fa8692a987a4d8739fcc642dc74fd2902d73ac5544a65f24018656475b` |
| [relsgg-vits16plus-text-student-f32.gguf](https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/gguf/relsgg-vits16plus-text-student-f32.gguf) | 42.86 | 78 | `8cd368fa8692a987a4d8739fcc642dc74fd2902d73ac5544a65f24018656475b` |

[The machine-readable manifest](benchmarks/full_gguf_manifest.json) records the exact source checkpoint SHA, type counts, missing/extra tensors and numerical verification. All nine files pass the declared storage-conversion check.

## Storage policy and selection

| Storage | Conversion | Suitable use |
|---|---|---|
| F32 | Prepared runtime floating tensors stored in F32 | Numerical reference, closest weight representation to official PyTorch |
| F16 | Floating tensors stored in F16; auxiliary values converted as needed by native ops | Smaller files and weight buffers; latency depends on backend kernels |
| Q8_0 | Rank-2 matrices with inner dimension divisible by 32 use Q8_0, except the `vocab_head.W` predicate bank (F16, keeps close top-1 predicates stable under CPU Q8_0 activation rounding); other tensors stay F32 | Lower weight bandwidth/storage, particularly useful on CUDA; inspect ranking drift for your vocabulary |

Default GPU flash attention uses F16 K/V and F32 accumulation. CUDA with F32 weights defaults to `--strict-f32`, which preserves F32 matrix operands and selects explicit attention, so close top-128 candidate pairs do not flip; `--no-strict-f32` restores the faster accelerated path. `--strict-f32` can also be forced for other storage types. `--no-flash-attention` changes attention only. Q8/F16 cannot reproduce original F32 weights bit-for-bit, and close-score predicates/candidates may change. Packed Q/K/V weights consume additional runtime memory.

## Measured full-pipeline evidence

The [portable full report](benchmarks/summary/README.md) contains three independent official PyTorch-CUDA baselines, 27 primary CPU/CUDA/Vulkan variants, and three additional CUDA Q8 F32-input diagnostics. Each row covers all 50 retained RA-4M images, GT boxes and the full vocabulary. Native timing includes JPEG decode, exact preprocessing, graph construction/allocation/transfers, inference and calibrated ranking. Loading and output files are excluded on both sides. The default `auto` request enables strict arithmetic only for CUDA F32; CPU, Vulkan, F16 and Q8_0 remain non-strict. Each result records both the requested and effective precision mode, so timing is interpreted using the actual path.

[Offline detailed gallery](benchmarks/gallery/index.html)

![All backend pipeline latency](benchmarks/gallery/latency_all_backends.png)

![Numerical and candidate agreement](benchmarks/gallery/accuracy_candidates.png)

Per-image grids cover every retained frame and every engine/storage variant. Numerical metrics include MAE, RMSE, max absolute error, cosine, predicate agreement and pair recall/Jaccard. Task metrics include R/mR@20/50/100, their harmonic F1, weighted recall and explicit local support buckets. Median/P95, throughput and allocator-scoped memory measurements accompany the accuracy rows.

Allocator-scoped GGML buffers and PyTorch peaks remain separately named. Process measurements in the [portable full report](benchmarks/summary/README.md) additionally sample both engines from startup through load, warmup, inference and output: RSS/HWM plus driver process device memory at 200 ms cadence. These sampled peaks can miss shorter spikes. Shared-machine hardware/process snapshots and every slower native image are retained. Refer to individual speedups instead of assuming every lower-precision format is faster.

The stage report in the [portable full report](benchmarks/summary/README.md) checks 17 intermediate boundaries in all 27 model/backend/storage configurations, custom vocabulary normalization/gating and 60-region dense candidate selection. 9/9 F32 dense runs retain every candidate and agree on top-1 predicates against the CPU reference. Fixed-candidate traces separate arithmetic error from changed contextual attention. The compact GT set has incomplete positive annotations; it cannot establish negative-set AP, OVS composite or detector accuracy. Public PSG/IndoorVG/VG150 and labelled SpatialSense subsets add independent evaluator regressions. [Metric coverage](benchmarks/METRIC_COVERAGE.md) explains their limited scope.


### Measured model/backend/storage matrix

RTX 3060; 50 GT-box images, 19,103 predicates; 2 warmups, 5 repeats per image. Speedup is each model’s Python median divided by its GGML median. Accepted timing runs had no unapproved transient GPU process; allowed persistent PIDs and Python bracket drift are recorded per run.

| Model | Engine/storage | Median / P95 ms | Speedup | R@50 | mR@50 | Logit RMSE | Top-1 | GGML buffers MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| relsgg-vits16 | cpu-f16 | 668.92 / 766.16 | 0.04x | 14.45% | 11.37% | 0.00314 | 99.89% | 163.6 |
| relsgg-vits16 | cpu-f32 | 631.53 / 693.03 | 0.05x | 14.45% | 11.37% | 0.00130 | 99.97% | 280.4 |
| relsgg-vits16 | cpu-q8_0 | 607.02 / 708.95 | 0.05x | 14.57% | 11.38% | 0.04172 | 97.09% | 110.1 |
| relsgg-vits16 | cuda-f16 | 31.42 / 34.00 | 0.91x | 14.45% | 11.37% | 0.00313 | 99.86% | 163.6 |
| relsgg-vits16 | cuda-f32 | 31.93 / 34.57 | 0.89x | 14.45% | 11.37% | 0.00130 | 99.97% | 280.4 |
| relsgg-vits16 | cuda-q8_0 | 26.93 / 29.73 | 1.06x | 14.57% | 11.38% | 0.03892 | 97.06% | 110.1 |
| relsgg-vits16 | cuda-q8_0-f32-input | 32.33 / 36.41 | 0.88x | 14.45% | 11.37% | 0.02598 | 98.19% | 110.1 |
| relsgg-vits16 | python | 28.48 / 31.01 | 1.00x | 14.45% | 11.37% | 0.00000 | 100.00% | — |
| relsgg-vits16 | vulkan-f16 | 34.05 / 38.74 | 0.84x | 14.45% | 11.37% | 0.00323 | 99.79% | 161.3 |
| relsgg-vits16 | vulkan-f32 | 73.31 / 77.75 | 0.39x | 14.45% | 11.37% | 0.00130 | 99.97% | 278.6 |
| relsgg-vits16 | vulkan-q8_0 | 35.47 / 39.89 | 0.80x | 14.45% | 11.37% | 0.02633 | 98.19% | 108.3 |
| relsgg-vits16plus | cpu-f16 | 665.08 / 704.25 | 0.05x | 16.94% | 10.47% | 0.00502 | 99.66% | 177.1 |
| relsgg-vits16plus | cpu-f32 | 697.34 / 920.58 | 0.05x | 17.06% | 10.96% | 0.00120 | 99.98% | 307.5 |
| relsgg-vits16plus | cpu-q8_0 | 628.70 / 655.25 | 0.05x | 17.43% | 11.95% | 0.03480 | 97.32% | 117.3 |
| relsgg-vits16plus | cuda-f16 | 33.54 / 36.69 | 0.94x | 16.94% | 10.47% | 0.00504 | 99.66% | 177.1 |
| relsgg-vits16plus | cuda-f32 | 35.50 / 39.05 | 0.89x | 17.06% | 10.96% | 0.00120 | 99.98% | 307.5 |
| relsgg-vits16plus | cuda-q8_0 | 28.43 / 31.44 | 1.11x | 16.81% | 10.90% | 0.03608 | 98.04% | 117.4 |
| relsgg-vits16plus | cuda-q8_0-f32-input | 35.60 / 42.42 | 0.89x | 17.06% | 10.96% | 0.02444 | 98.37% | 117.4 |
| relsgg-vits16plus | python | 31.63 / 35.66 | 1.00x | 17.06% | 10.96% | 0.00000 | 100.00% | — |
| relsgg-vits16plus | vulkan-f16 | 36.23 / 40.94 | 0.87x | 16.94% | 10.47% | 0.00507 | 99.61% | 174.8 |
| relsgg-vits16plus | vulkan-f32 | 80.33 / 86.09 | 0.39x | 17.06% | 10.96% | 0.00120 | 99.98% | 305.7 |
| relsgg-vits16plus | vulkan-q8_0 | 37.38 / 41.50 | 0.85x | 17.19% | 11.45% | 0.02478 | 98.38% | 115.5 |
| relsgg-vitb16 | cpu-f16 | 1358.71 / 1436.32 | 0.04x | 17.56% | 13.00% | 0.00260 | 99.82% | 333.7 |
| relsgg-vitb16 | cpu-f32 | 1439.37 / 1495.20 | 0.04x | 17.56% | 13.00% | 0.00093 | 99.97% | 609.9 |
| relsgg-vitb16 | cpu-q8_0 | 1299.55 / 1347.96 | 0.04x | 17.56% | 12.96% | 0.03351 | 97.80% | 206.5 |
| relsgg-vitb16 | cuda-f16 | 55.20 / 57.81 | 0.94x | 17.43% | 12.98% | 0.00257 | 99.66% | 333.7 |
| relsgg-vitb16 | cuda-f32 | 60.77 / 63.75 | 0.85x | 17.56% | 13.00% | 0.00093 | 99.97% | 609.9 |
| relsgg-vitb16 | cuda-q8_0 | 42.32 / 45.28 | 1.22x | 17.56% | 12.98% | 0.03382 | 97.73% | 206.6 |
| relsgg-vitb16 | cuda-q8_0-f32-input | 56.23 / 63.00 | 0.92x | 17.56% | 12.87% | 0.01926 | 98.62% | 206.6 |
| relsgg-vitb16 | python | 51.74 / 55.31 | 1.00x | 17.56% | 13.00% | 0.00000 | 100.00% | — |
| relsgg-vitb16 | vulkan-f16 | 54.82 / 60.15 | 0.94x | 17.56% | 13.00% | 0.00263 | 99.79% | 329.1 |
| relsgg-vitb16 | vulkan-f32 | 121.68 / 127.14 | 0.43x | 17.56% | 13.00% | 0.00093 | 99.97% | 603.0 |
| relsgg-vitb16 | vulkan-q8_0 | 55.35 / 58.63 | 0.93x | 17.56% | 12.87% | 0.01959 | 98.28% | 199.6 |

Some measured native configurations do not beat Python; consult the individual rows. Shared-workstation results are specific to the recorded process load. F32 does not imply bit-identical GPU kernels; full errors and candidate changes are disclosed.

The [portable full report](benchmarks/summary/README.md) includes weight-rounding versus same-weight arithmetic for every retained image through the official Python graph with all GGUF weights restored, plus the CUDA Q8 F32-activation precision isolation. Superseded experiments were removed after replacement; the cleanup manifest records their identities.

## Supported interface and limits

- Batch 1, 448×448, boxes, up to `ModelOptions.max_boxes` (default 60), 128 candidate slots; all three released architectures.
- CPU, CUDA and Vulkan execute native GGML graphs; no Python inference or cuDNN dependency in C++.
- `ModelOptions` configures model/backend/precision mode/threads/vocabulary, with explicit `RuntimeOptions` for backend policy. `ImageInput` supplies image and pixel XYXY boxes.
- The versioned C ABI accepts borrowed RGB/RGBA/GRAY/BGR/BGRA memory with stride; results have independent ownership. A host can bind its existing backend lease. See [integration contract](benchmarks/PORTING_NOTES.md).
- Replacement vocabularies use finite row-major F32 `[V,512]` embeddings or native CLIP BPE/text-student encoding. The head normalizes them and computes its learned gate.
- Native masks, detector-score weighting and decomposed spatial/semantic outputs are independently compared with Python on all three models and backends. The 50-image timing profile uses boxes only.
- GT-box evaluation measures relation prediction, not detector localization or end-to-end detection recall.

```sh
./run_ggml.sh --backend cuda --model-name relsgg-vits16plus --dtype q8_0 --download
./run_ggml.sh --backend vulkan --model-name relsgg-vitb16 --dtype f16
./run_ggml.sh --engine python --backend cuda
```


## Terms

This is a downstream GGUF conversion, not an upstream-official GGUF release. The source model cards identify the DINOv3 model license; conversion does not replace those terms. See the [source model card](https://huggingface.co/maelic/relsgg-vits16plus) and [DINOv3 license](https://ai.meta.com/resources/models-and-libraries/dinov3-license/). Source code licensing remains governed by this repository’s LICENSE. Retained dataset images keep their original COCO/Objects365/Open Images terms.
