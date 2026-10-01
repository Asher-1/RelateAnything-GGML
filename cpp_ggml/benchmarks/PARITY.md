# Numerical validation

Current official model validation is in [full_graph/README.md](full_graph/README.md)
and [full_graph_validation/README.md](full_graph_validation/README.md). It compares
independent official PyTorch predictions with native C++ full-graph results.

## Historical synthetic adapter parity

These results compare binary `RAOOv1` outputs from the same four-object,
12-feature input. The F32 CPU result passes an independent NumPy adapter
reference using the GGML-compatible approximate GELU. CUDA and Vulkan F32 are
compared with that CPU output. F16 and Q8_0 are compared with the matching F32
CPU output, then their CUDA/Vulkan results are compared with the matching
quantized CPU output.

| Dtype | Backend | Reference | Max abs | MAE | RMSE | Gate |
|---|---|---|---:|---:|---:|---|
| F32 | CPU | NumPy | 2.436e-05 | 5.412e-06 | 7.918e-06 | pass, 1e-4 |
| F32 | CUDA | CPU F32 | 4.277e-05 | 1.459e-05 | 1.818e-05 | pass, 1e-4 |
| F32 | Vulkan | CPU F32 | 3.783e-05 | 8.700e-06 | 1.197e-05 | pass, 1e-4 |
| F16 | CPU | CPU F32 | 3.318e-05 | 7.104e-06 | 1.044e-05 | pass, 2e-2 |
| F16 | CUDA | CPU F16 | 4.911e-05 | 1.095e-05 | 1.472e-05 | pass, 2e-2 |
| F16 | Vulkan | CPU F16 | 4.148e-05 | 9.986e-06 | 1.312e-05 | pass, 2e-2 |
| Q8_0 | CPU | CPU F32 | 8.187e-04 | 2.088e-04 | 2.728e-04 | pass, 2e-1 |
| Q8_0 | CUDA | CPU Q8_0 | 2.615e-04 | 2.217e-05 | 5.388e-05 | pass, 2e-1 |
| Q8_0 | Vulkan | CPU Q8_0 | 1.087e-03 | 1.666e-04 | 2.465e-04 | pass, 2e-1 |

The machine was an NVIDIA RTX 3060. These are adapter-contract results, not a
claim of full DINOv3/dynamic-vocabulary model parity. Re-run the commands in
`README.md` and regenerate `parity_results.json` for a release checkpoint.

```sh
python3 cpp_ggml/scripts/plot_parity.py \
  cpp_ggml/benchmarks/parity_results.json \
  --output cpp_ggml/benchmarks/parity.png
```
