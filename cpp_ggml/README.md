# RelateAnything C++ / GGML

The three released checkpoints run as a native image-to-relation pipeline:
JPEG/PNG decode → Pillow-compatible 448×448 resize/normalization → DINOv3 →
spatial pooling and box prompts → pair sampler → relation transformer →
deformable read → relation interaction → dynamic vocabulary → calibrated
triplets. The runtime needs C++17, GGML and libjpeg; it does not use PyTorch,
Python model inference or cuDNN.

## Build and run

```sh
git submodule update --init --recursive
cmake -S cpp_ggml -B cpp_ggml/build-cuda -DCMAKE_BUILD_TYPE=Release -DRA_GGML_CUDA=ON
cmake --build cpp_ggml/build-cuda --parallel 8
./run_ggml.sh --backend cuda --model-name relsgg-vits16plus --dtype q8_0 --download
./run_ggml.sh --backend vulkan --dtype f16
./run_ggml.sh --backend cpu --threads 8
./run_ggml.sh --engine python --backend cuda
```

The launcher configures/builds absent targets and defaults to the first real
RA-4M sample. `--download` fetches an absent official GGUF. For an explicit
build, use `-DRA_GGML_VULKAN=ON` in a separate build directory; leave both
accelerator flags OFF for CPU. A missing requested device fails clearly.
GGML is pinned at v0.21.0 (`8599e0ea3756c4bac4ef813af2241cb1a8bbfb0b`). CMake
applies `third_party/patches/*.patch` in order, with reverse checks for replay.

```sh
./run_ggml.sh --backend cuda --image photo.jpg --boxes-json '[[20,30,180,240],[120,150,330,300]]'
./run_ggml.sh --backend cuda --vocabulary-embeddings bank.f32 --vocabulary holding --vocabulary above
./run_ggml.sh --backend cuda --raw-logits --output raw.rafo
./run_ggml.sh --backend cuda --no-flash-attention --warmup 3 --repeats 10
```

`bank.f32` is little-endian row-major `[V,512]` text embeddings. C++ normalizes
them and executes the learned gate MLP for the new vocabulary. The optional
names label the output in the launcher. Text encoding from free-form strings
is available in the official Python engine; the C++ API consumes embeddings.
The full released bank and calibration are embedded in each GGUF.

## Public C++ API

```cpp
#include "relateanything/full_model.hpp"
relateanything::Options options;
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

All configuration enters options objects. No application getenv/setenv knobs
are used. Models own weights/backends and cached fixed positional inputs;
use one model instance per concurrent caller. The learned operations execute
on the selected backend. Pair candidate sorting and geometry construction use
C++; the sampler boundary currently copies the fused feature map and object
features to host and back. It is included in timings.

`ImageInput` accepts already normalized RGB/CHW floats and normalized cxcywh
boxes. Empty/one-object inputs return no relations. Inputs above `max_boxes`
are rejected explicitly; select the desired detector boxes before inference.
The implemented deployment profile uses boxes, batch 1, 448×448, 128 pair
slots and the released architectures. Optional upstream instance masks,
decomposed output and detector-score weighting are outside this profile.

F32/F16/Q8 describe weight storage. Q8_0 quantizes rank-2 matrices whose inner
dimension is divisible by 32; other tensors stay F32. F16 stores floating
weights in F16. Accelerated flash attention uses F16 K/V and F32 accumulation;
`--no-flash-attention` uses explicit F32 attention. Quantized predictions are
measured against independent official PyTorch F32, so rounding and close-score
rank changes remain visible in the report.

## Reproduce validation

```sh
python3 cpp_ggml/scripts/benchmark_full_graph.py --stage all
python3 cpp_ggml/scripts/check_runtime_contracts.py --backend cpu
python3 cpp_ggml/scripts/validate_full_graph.py --dynamic
ctest --test-dir cpp_ggml/build-cuda --output-on-failure
```

[All 50-frame comparisons](benchmarks/full_graph/README.md),
[metric coverage](benchmarks/METRIC_COVERAGE.md),
[model downloads](models/MODEL_CARDS.md), and
[porting decisions](benchmarks/PORTING_NOTES.md) document the actual behavior.
The native timing includes decode, resize, normalization, allocation/transfers,
full graph and calibrated ranking. Raw tensor dumps run separately from timing.

## File protocols

- `RAIP1`: text header, quoted image path, box count, then pixel xyxy rows.
  A relative image path is resolved against the request file. `.raip` requests
  measure the whole native image pipeline; a directory runs with one loaded model.
- `RAIMv1\0\0`: eight magic bytes, little-endian uint32 width/height/object count,
  F32 normalized RGB CHW, then F32 normalized cxcywh. `--preprocess-only`
  converts requests to RAIM for independent pixel verification.
- `RAFOv1\0\0`: uint32 pair/predicate counts after the eight-byte magic;
  valid/sub/object int32 arrays, pair F32 logits, `[pairs,predicates]` F32 logits.
  Emitted only with `--raw-logits`; default output is calibrated JSON.
- The historical RAIO/RAOO adapter protocol remains for explicit fixtures in
  `test_data/fixtures/`; it is not the default launcher or model benchmark.
