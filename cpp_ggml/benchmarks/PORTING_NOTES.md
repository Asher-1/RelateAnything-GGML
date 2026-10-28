# Native graph and embedding boundaries

The full graph implements upstream image resize/normalization, DINOv3 tower,
layer fusion, prompt/ROI pooling, pair geometry and sampler, relation stacks,
deformable reads, interaction head, vocabulary gates and calibrated ranking.
All three released architectures and all exported state tensors are used.
The six-tensor adapter remains an isolated fixture; it is not a real-model path.

Full-vocabulary export follows `full_vocabulary=True`: load the released
embedding sidecar, normalize the bank, replace `vocab_head.W`, and recompute
alpha. Checkpoint-bank-only comparisons would use the wrong reference.
Geometry and sampler reductions retain separate F32 products to avoid
backend GEMM rounding that changes Top400 selection. Budgets remain 400/128.
Candidate sorting runs on the host; cached scene positions, fused vocabulary
projection and retained feature maps avoid redundant device work/transfers.

`ModelOptions` contains session configuration. CLI `Options` adds file I/O and
benchmark controls; a model inference call executes once. `RuntimeOptions`
contains ABI-tagged CPU/CUDA/Vulkan policy. First configuration or backend read
freezes process policy; identical settings succeed, conflicts return errors.
There are no environment switches in the compiled project/GGML sources and
CUDA headers. The host manages OpenMP and exception-termination policy.
All dependency changes replay from one [patch](../third_party/patches/0001-relateanything-ggml.patch).

`strict_f32` selects explicit F32 attention, CUDA default cuBLAS math and scalar
Vulkan F32 kernels. F16 and Q8 storage still changes original weight values;
strict arithmetic cannot restore discarded information. CUDA `quant_f32_input`
avoids extra activation quantization; it does not restore original FP32 weights.
Compare original PyTorch-CUDA, PyTorch CPU F32, same stored-weight CPU oracles
and native outputs separately; those error comparisons cannot be added as
independent linear components.

The public [C API](../include/relateanything/c_api.h) exposes versioned options,
opaque contexts, borrowed image memory with stride/capacity, typed independently
owned results, error/readiness queries and cancellation. It exposes no STL, Qt,
OpenCV or GGML types. The C++ `FullModel` uses a PImpl and accepts `ImageView`.
Result strings/arrays survive context destruction and have one release function.
Timings mark only measurable fields valid; graph-plus-ranking is not labelled
pure inference. A single context is serialized by its caller.

For future AICore integration, link this static runtime into the monolithic
AICore target and adapt result structs to its common ABI. Supply
`ra_backend_binding` from `aicore::runtime::BackendLease`, retain/release the
lease, and lock its device group around create/infer/free. Bound sessions create
only schedulers and model buffers; they do not discover/free physical backends
or modify host thread policy. The host and model must share the same patched
GGML ABI. The standalone runtime is prepared for this adapter; no AICore plugin
or new task shared library is installed by this repository.

The stage traces and dense decomposition in the [complete report](summary/README.md),
together with [original/same-weight plots](gallery/README.md), retain numerical differences.
Speed and precision claims are tied to the complete measured configuration.
