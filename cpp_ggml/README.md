# RelateAnything C++ / GGML

The three released checkpoints run as a native image-to-relation pipeline:
JPEG/PNG decode → Pillow-compatible 448×448 resize/normalization → DINOv3 →
spatial pooling and box prompts → pair sampler → relation transformer →
deformable read → relation interaction → dynamic vocabulary → calibrated
triplets. The runtime needs C++17, GGML, libjpeg, jsoncpp and ICU; it does not use PyTorch,
Python model inference or cuDNN.

[Current alignment status](benchmarks/ALIGNMENT_STATUS.md) and
[acceptance checklist](benchmarks/NEXT_WINDOW_TODOS.md) distinguish the
implemented graph from the remaining numerical, performance and protocol
criteria. The latest publishable execution identity and report download are recorded in
[the benchmark release summary](benchmarks/summary/README.md).

Open [all comparison charts](benchmarks/ALL_COMPARISONS.html) for every
backend/storage chart and 150 original-image grids. The
[strict optimization report](benchmarks/optimization/index.html) tracks the
new CUDA operator fixes and fresh measurements separately from the complete
earlier baseline. [outputs](outputs/index.html) retains reproducible evidence
and the downloadable test ZIP, with a visual entry point.

## Build and run

```sh
git submodule update --init --recursive
./run_ggml.sh --doctor
./run_ggml.sh --quickstart
cmake -S cpp_ggml -B cpp_ggml/build-cuda -DCMAKE_BUILD_TYPE=Release -DRA_GGML_CUDA=ON
cmake --build cpp_ggml/build-cuda --parallel 8
./run_ggml.sh --backend cuda --model-name relsgg-vits16plus --dtype q8_0 --download
./run_ggml.sh --backend vulkan --dtype f16
./run_ggml.sh --backend cpu --threads 8
./run_ggml.sh --engine python --backend cuda
```

`--quickstart` 自动发现 CUDA、Vulkan 或 CPU，下载 `relsgg-vits16plus-f16.gguf`，并使用维护的 horse 演示图片、person/horse 框完成推理。若需单独复现该输入，使用 `--demo horse`；`--doctor` 输出工具、设备、演示资产、默认测试数据和推荐模型的本地可用性，且不执行构建或下载。

The launcher configures/builds absent targets and defaults to the first real
RA-4M sample. `--download` fetches an absent official GGUF. For an explicit
build, use `-DRA_GGML_VULKAN=ON` in a separate build directory; leave both
accelerator flags OFF for CPU. A missing requested device fails clearly.
GGML is pinned at v0.21.0 (`8599e0ea3756c4bac4ef813af2241cb1a8bbfb0b`). CMake
applies the single `third_party/patches/0001-relateanything-ggml.patch` and recognizes the complete patched tree.
Every configure verifies that the GGML checkout exactly matches the pinned
commit plus the patch, including untracked files. Audit it separately with
`python3 cpp_ggml/scripts/audit_ggml_patch_state.py`; stable evidence and the
complete-report identity are published through the [benchmark release summary](benchmarks/summary/README.md).

```sh
./run_ggml.sh --backend cuda --image photo.jpg --boxes-json '[[20,30,180,240],[120,150,330,300]]'
./run_ggml.sh --backend cuda --vocabulary-embeddings bank.f32 --vocabulary holding --vocabulary above
./run_ggml.sh --backend cuda --raw-logits --output raw.rafo
./run_ggml.sh --backend cuda --no-flash-attention --warmup 3 --repeats 10
./run_ggml.sh --backend cuda --vocabulary holding --vocabulary above --strict-f32
./run_ggml.sh --backend cuda --decompose --box-scores-json '[0.8,0.9]' --image photo.jpg --boxes-json '[[20,30,180,240],[120,150,330,300]]'
```

`bank.f32` is little-endian row-major `[V,512]` text embeddings. C++ normalizes
them and executes the learned gate MLP for the new vocabulary. The optional
names label the output in the launcher. Free-form strings are encoded by the
native CLIP tokenizer and text student when no embedding bank is supplied.
The student GGUF and official tokenizer.json must be available; override them
with `--text-student` and `--tokenizer`.
The full released bank and calibration are embedded in each GGUF.

## Public C++ API

```cpp
#include "relateanything/full_model.hpp"
relateanything::ModelOptions options;
options.model = "relsgg-vits16plus-q8_0.gguf";
options.backend = "cuda"; // cpu or vulkan
options.max_boxes = 60;
relateanything::FullModel model(options);
relateanything::ImageOptions image;
image.image = "photo.jpg";
image.boxes_xyxy = {20,30,180,240, 120,150,330,300};
auto result = model.infer(relateanything::prepare_image(image));
for (const auto & r : result.relations) {
    // r.subject, r.object, model.predicates()[r.predicate], r.score
}
```

For borrowed image memory, pass `ImageView{data,width,height,stride,capacity,format}`
to `prepare_image(view, regions)`. RGB/RGBA/GRAY/BGR/BGRA and padded strides are
supported. The image is borrowed during preprocessing only.

The [pure C API](include/relateanything/c_api.h) uses opaque contexts, ABI-tagged
options, readiness/error queries and typed owned results with `ra_result_free`.
Results and predicate names survive context destruction. Cancellation is checked
before/after graph execution; model creation/inference catches exceptions.
`ra_timings.valid_fields` marks measured subdivisions only. Callers serialize a
context; future AICore adapters lock the shared device lease group.

A host may provide `ra_backend_binding` from its common `BackendLease` registry.
The model retains/releases the lease and owns only its scheduler and model
buffers, leaving discovery, physical backend destruction and thread policy to
the host. Handles must use the same patched GGML ABI. Link the static runtime
into AICore's monolithic library; no Qt/STL/OpenCV/GGML types cross the C ABI and
no separate per-task shared library is introduced. See [porting details](benchmarks/PORTING_NOTES.md).

All configuration enters `ModelOptions` and ABI-tagged `RuntimeOptions`.
Compiled project/GGML CPU/CUDA/Vulkan sources use no getenv/setenv feature switches.
The runtime freezes policy before backend discovery; conflicting configuration fails. Models own weights/backends and cached fixed positional inputs;
use one model instance per concurrent caller. The learned operations execute
on the selected backend. Pair candidate sorting and geometry construction use
C++; the fused feature map stays on the selected device across the sampler
boundary. Object features and candidate scores cross the host boundary for
candidate selection. All transfers are included in timings. Use
`--no-retain-features` to measure the historical feature-map round trip.

`ImageInput` accepts already normalized RGB/CHW floats and normalized cxcywh
boxes. Empty/one-object inputs return no relations. Inputs above `max_boxes`
are rejected explicitly; select the desired detector boxes before inference.
The implemented deployment profile uses batch 1, 448×448, 128 pair slots and
the released architectures. `ImageOptions.masks` accepts binary instance masks
in original-image `[N,H,W]` order; `box_scores` weights each relation by the
two detector confidences. `ModelOptions.decompose` returns independently ranked
spatial and semantic streams. The launcher accepts raw mask bytes with `--masks`.

F32/F16/Q8 describe weight storage. Q8_0 quantizes rank-2 matrices whose inner
dimension is divisible by 32; other tensors stay F32. F16 stores floating
weights in F16. Accelerated flash attention uses F16 K/V and F32 accumulation;
`--no-flash-attention` uses explicit F32 attention. Quantized predictions are
measured against independent official PyTorch F32, so rounding and close-score
rank changes remain visible in the report.

`ModelOptions::quant_f32_input` (`--quant-f32-input`, CUDA only) selects F32 activation
GEMM for quantized matrices, avoiding the additional activation quantization in
the default CUDA Q8 kernels. Its GGML change is part of the single consolidated patch, applied by CMake.
This is an explicit precision diagnostic; it does not restore the original
unquantized weights or promise bit identity with a different arithmetic engine.

`ModelOptions.strict_f32` (`--strict-f32`) preserves F32 operands during matrix
multiplication and uses explicit F32 attention. The consolidated patch routes CUDA F32
matrices through cuBLAS with default math and Vulkan through a scalar F32 shader,
avoiding TF32/F16 input conversion. It applies to native text encoding too.
This is an explicit precision mode with a measured latency cost; F16/Q8 stored
weights remain rounded, and floating-point reduction order can still differ.

## Reproduce validation

```sh
python3 cpp_ggml/scripts/run_alignment.py --run-id 20261009-local-alignment --current --warmup 2 --repeats 5
ctest --test-dir cpp_ggml/build-cuda --output-on-failure
```

[Offline detailed model/backend/storage gallery](benchmarks/gallery/index.html),
[visual outputs entry](outputs/index.html),
[test images and labels](test_data/index.html), and
[test ZIP/cache/autofill contract](test_data/PACKAGE.md),
[complete 50-image report download and identity](benchmarks/summary/README.md),
[metric coverage](benchmarks/METRIC_COVERAGE.md),
[model downloads](models/MODEL_CARDS.md), and
[porting decisions](benchmarks/PORTING_NOTES.md) document the actual behavior.

The default graph computes only valid relation rows and omits all-zero attention
masks. RAFO output remains 128 slots; ignored rows contain zero predicate logits
and are excluded by `valid_mask`. Intermediate dumps contain the computed rows.
`ModelOptions.compact_pairs` (`--no-compact-pairs`) and `ModelOptions.matmul_sampler`
(`--matmul-sampler`) retain diagnostic paths for reproducible ablation.
The sampler uses separate F32 products and row reduction, matching the upstream
operation sequence. The stored-weight oracle is included in the
[complete benchmark report](benchmarks/summary/README.md) and distinguishes quantization from backend arithmetic on all 50 images.
Vulkan also uses F32 products/reduction in the small geometry MLP controlling
Top400 selection. This avoids the observed close-candidate swap in its GEMM
path; the 400/128 budgets remain unchanged. ModelOptions.matmul_geometry
(--matmul-geometry) selects that historical GEMM for ablation.
The native timing includes decode, resize, normalization, allocation/transfers,
full graph and calibrated ranking. Raw tensor dumps run separately from timing.
Fixed scene positions are computed once on the selected backend and retained in
its weight buffer. Semantic/spatial vocabulary queries share one projection.
`--no-cache-positions` and `--no-fused-vocabulary` expose the corresponding
historical operations for measured ablations.

## File protocols

- `RAIP1`: text header, quoted image path, box count, then pixel xyxy rows.
  A relative image path is resolved against the request file. `.raip` requests
  measure the whole native image pipeline; a directory runs with one loaded model.
- `RAIP2`: RAIP1 fields followed by a 0/1 score flag, optional N detector
  confidences, and a quoted raw binary mask path (empty for box-only).
- `RAIMv1\0\0`: eight magic bytes, little-endian uint32 width/height/object count,
  F32 normalized RGB CHW, then F32 normalized cxcywh. `--preprocess-only`
  converts requests to RAIM for independent pixel verification.
- `RAIMv2`: RAIMv1 dimensions followed by uint32 flags (coverage=1, scores=2),
  pixels and boxes, optional `[N,32,32]` F32 coverage plus N fill values, and
  optional N F32 detector scores.
- `RAFOv1\0\0`: uint32 pair/predicate counts after the eight-byte magic;
  valid/sub/object int32 arrays, pair F32 logits, `[pairs,predicates]` F32 logits.
  Emitted only with `--raw-logits`; default output is calibrated JSON.
- The historical RAIO/RAOO adapter protocol remains for explicit fixtures in
  `test_data/fixtures/`; it is not the default launcher or model benchmark.
