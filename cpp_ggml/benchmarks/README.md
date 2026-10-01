# RelateAnything full-graph benchmarks

[Implementation and requirement status](ALIGNMENT_STATUS.md).

[The current full-graph report](full_graph/README.md) compares all three official
checkpoints with native C++ CUDA/Vulkan F32/F16/Q8_0 on all 50 retained RA-4M
images. It contains 150 complete per-image comparison grids and all raw metrics.

![Full pipeline speed and accuracy](full_graph/metrics.png)

![Accuracy by source corpus](full_graph/source_breakdown.png)

- [Metrics CSV](full_graph/metrics.csv): latency, R/mR/F1/wR, numerical drift and pair/predicate agreement.
- [Metric definitions and scope](METRIC_COVERAGE.md): what the positive-only compact set can establish.
- [Intermediate and dynamic-vocabulary validation](full_graph_validation/README.md): every architecture and backend.
- [Explicit F32 attention checks](explicit_f32_attention/README.md): six additional model/backend numerical diagnostics.
- [Model artifacts and conversion](EXPORT_STATUS.md), [model cards](../models/MODEL_CARDS.md).
- [Porting decisions](PORTING_NOTES.md), [retained dataset](RA4M_50.md), [official data sources](OFFICIAL_DATA.md).

```sh
python3 cpp_ggml/scripts/benchmark_full_graph.py --stage all
python3 cpp_ggml/scripts/validate_full_graph.py --backends cpu cuda vulkan --dtypes f32 f16 q8_0 --dynamic
python3 cpp_ggml/scripts/check_runtime_contracts.py --backend cuda
python3 cpp_ggml/scripts/verify_benchmark_release.py
python3 cpp_ggml/scripts/report_full_validation.py
python3 cpp_ggml/scripts/update_model_cards.py
```

The native timing covers image decode, preprocessing, full graph, transfers,
calibration and ranking. Model load and artifact writes are excluded on both
sides. Raw-logit validation runs separately. CUDA and Vulkan use the same
physical RTX 3060. Shared-machine load, precision and hardware snapshots are
recorded; speedups are measurements under this protocol, not universal claims.

## Historical evidence

`results.jsonl`, `results.png`, `parity_results.json`, `parity.png` and the
adapter section of [PARITY.md](PARITY.md) describe the earlier synthetic adapter.
They are retained as development history and contribute no numbers to the
current model report. `pytorch_cuda_results.jsonl`, `pytorch_cuda.png`,
`pytorch_outputs.png`, `ra4m_50_python.json` and `ra4m_50_python.png` are earlier
Python baseline runs. `model_metrics.json/png` reproduce upstream A40 protocol
numbers and are not comparable to local RTX 3060 timings.

## Release checks

- [Release audit](full_graph/release_audit.json): 900 GGML image/configuration comparisons, 150 Python images, current binary and model hashes, independent native decode validation.
- [Source fingerprint](source_fingerprint.json): exact runtime source/headers/CMake/patch files used by the build.
- [Dependency inspection](dependencies.json): actual CPU/CUDA/Vulkan link dependencies; no cuDNN.
- [Patch replay](patch_replay.json): pristine v0.21.0 forward application, idempotent reverse check and restored-tree equality.
- [Published artifacts](hf_release_verification.json): remote revision, every GGUF SHA256 and size checked against the local manifest.

Controlled timing is available with `--wait-gpu-idle`. If persistent system
GPU services cannot be stopped, pass their PIDs explicitly with
`--allow-gpu-pid`; the accepted run and all discarded attempts are logged.
No script terminates other users' GPU processes. The current measurements
allowed the same two persistent services for Python and C++.
