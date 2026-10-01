# Full graph intermediate validation

Eleven boundaries are compared against independent upstream PyTorch traces.
The trace uses the first retained real image and all its GT boxes. CPU F32 uses explicit attention; GPU variants use F16 K/V flash attention with F32 accumulation. Stage matrices compare matching subject/object IDs, excluding invalid slots.

| Model | Backend/storage | Fused map RMSE | Transformer RMSE | Deformable RMSE | Final RMSE | Final max abs | Custom vocabulary RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| relsgg-vitb16 | cpu/f16 | 0.0015213 | 0.00911541 | 0.00987932 | 0.00268512 | 0.01425 | 0.00256391 |
| relsgg-vitb16 | cpu/f32 | 2.52908e-06 | 0.00166348 | 0.00169262 | 0.000754657 | 0.00460958 | 3.15619e-06 |
| relsgg-vitb16 | cpu/q8_0 | 0.0189835 | 0.149826 | 0.158904 | 0.0454394 | 0.257971 | 0.0513014 |
| relsgg-vitb16 | cuda/f16 | 0.0021494 | 0.0142812 | 0.0152807 | 0.00455565 | 0.023778 | 0.00451711 |
| relsgg-vitb16 | cuda/f32 | 0.00211021 | 0.0128274 | 0.0139819 | 0.00350916 | 0.0198498 | 0.00399829 |
| relsgg-vitb16 | cuda/q8_0 | 0.0188729 | 0.142901 | 0.152126 | 0.045796 | 0.259845 | 0.0509504 |
| relsgg-vitb16 | vulkan/f16 | 0.00173464 | 0.0103754 | 0.0110822 | 0.00259892 | 0.0191538 | 0.00261282 |
| relsgg-vitb16 | vulkan/f32 | 0.00157956 | 0.00769268 | 0.00818935 | 0.00179114 | 0.0103617 | 0.0021784 |
| relsgg-vitb16 | vulkan/q8_0 | 0.0111467 | 0.0898964 | 0.0970783 | 0.0187074 | 0.147127 | 0.0208614 |
| relsgg-vits16 | cpu/f16 | 0.0014168 | 0.00828347 | 0.00893033 | 0.00310176 | 0.0169201 | 0.00396124 |
| relsgg-vits16 | cpu/f32 | 5.23811e-06 | 0.00151711 | 0.0015418 | 0.0011792 | 0.00808263 | 1.59307e-05 |
| relsgg-vits16 | cpu/q8_0 | 0.0253222 | 0.164642 | 0.176092 | 0.045192 | 0.446494 | 0.069329 |
| relsgg-vits16 | cuda/f16 | 0.00226212 | 0.0112199 | 0.0120904 | 0.00326637 | 0.0246553 | 0.00375256 |
| relsgg-vits16 | cuda/f32 | 0.00213818 | 0.00941761 | 0.0101312 | 0.00350665 | 0.0180135 | 0.00249789 |
| relsgg-vits16 | cuda/q8_0 | 0.0240371 | 0.14055 | 0.149908 | 0.0382873 | 0.272067 | 0.0414771 |
| relsgg-vits16 | vulkan/f16 | 0.00201131 | 0.0100287 | 0.0108703 | 0.002995 | 0.0178328 | 0.00357395 |
| relsgg-vits16 | vulkan/f32 | 0.00182907 | 0.0102919 | 0.0108676 | 0.00341722 | 0.0237486 | 0.00342964 |
| relsgg-vits16 | vulkan/q8_0 | 0.0139606 | 0.0982414 | 0.107179 | 0.0257539 | 0.182943 | 0.0344531 |
| relsgg-vits16plus | cpu/f16 | 0.00205832 | 0.00870347 | 0.00947017 | 0.00424066 | 0.0171633 | 0.00292824 |
| relsgg-vits16plus | cpu/f32 | 6.15816e-06 | 0.00148675 | 0.00151891 | 0.00127406 | 0.00499344 | 6.40523e-06 |
| relsgg-vits16plus | cpu/q8_0 | 0.0225578 | 0.129462 | 0.140401 | 0.0428845 | 0.429646 | 0.0499851 |
| relsgg-vits16plus | cuda/f16 | 0.00311444 | 0.0138356 | 0.014639 | 0.00408802 | 0.0255167 | 0.00355256 |
| relsgg-vits16plus | cuda/f32 | 0.0055184 | 0.0175494 | 0.0186006 | 0.00416457 | 0.0285306 | 0.00389659 |
| relsgg-vits16plus | cuda/q8_0 | 0.0234466 | 0.134969 | 0.146537 | 0.0365373 | 0.273012 | 0.04149 |
| relsgg-vits16plus | vulkan/f16 | 0.00294021 | 0.0103678 | 0.0110459 | 0.00382983 | 0.0230241 | 0.00274885 |
| relsgg-vits16plus | vulkan/f32 | 0.00555742 | 0.018794 | 0.0199021 | 0.0044624 | 0.0290446 | 0.00453673 |
| relsgg-vits16plus | vulkan/q8_0 | 0.0147651 | 0.0923768 | 0.100917 | 0.022649 | 0.169847 | 0.0275506 |

Each model/backend directory contains `stages/comparison.json` with all eleven boundaries and `dynamic/comparison.json`. Dynamic tests use five nonadjacent bank rows with non-unit scales plus an unseen composite direction. The independent CPU Python API performs its own normalization and gate computation.

## Dense candidate selection

A real retained image with 60 deterministic distinct boxes exercises 3,540 valid ordered pairs, geometry TopK=400 and relation TopK=128. Generated boxes have no task GT; this is a numerical sampler test. Close quantized scores can change the candidate set, which in turn changes contextual attention for otherwise matching pairs. All errors and pair changes are retained.

| Model | Backend/storage | Pair recall | Predicate agreement | Logit RMSE |
|---|---|---:|---:|---:|
| relsgg-vits16 | cpu/f32 | 100.00% | 100.00% | 1.16676e-05 |
| relsgg-vits16 | cuda/f32 | 100.00% | 100.00% | 0.00353289 |
| relsgg-vits16 | cuda/f16 | 100.00% | 99.22% | 0.00504989 |
| relsgg-vits16 | cuda/q8_0 | 98.44% | 99.21% | 0.0930231 |
| relsgg-vits16 | vulkan/f32 | 100.00% | 99.22% | 0.00375509 |
| relsgg-vits16 | vulkan/f16 | 100.00% | 99.22% | 0.00373719 |
| relsgg-vits16 | vulkan/q8_0 | 98.44% | 99.21% | 0.0848224 |
| relsgg-vits16plus | cpu/f32 | 100.00% | 100.00% | 9.12657e-06 |
| relsgg-vits16plus | cuda/f32 | 100.00% | 97.66% | 0.00960363 |
| relsgg-vits16plus | cuda/f16 | 100.00% | 100.00% | 0.00679208 |
| relsgg-vits16plus | cuda/q8_0 | 98.44% | 92.06% | 0.0788907 |
| relsgg-vits16plus | vulkan/f32 | 99.22% | 97.64% | 0.162227 |
| relsgg-vits16plus | vulkan/f16 | 99.22% | 96.85% | 0.160172 |
| relsgg-vits16plus | vulkan/q8_0 | 98.44% | 95.24% | 0.0664888 |
| relsgg-vitb16 | cpu/f32 | 100.00% | 100.00% | 4.55589e-06 |
| relsgg-vitb16 | cuda/f32 | 100.00% | 100.00% | 0.00560009 |
| relsgg-vitb16 | cuda/f16 | 100.00% | 99.22% | 0.00588426 |
| relsgg-vitb16 | cuda/q8_0 | 98.44% | 97.62% | 0.0911227 |
| relsgg-vitb16 | vulkan/f32 | 100.00% | 100.00% | 0.00296696 |
| relsgg-vitb16 | vulkan/f16 | 100.00% | 99.22% | 0.00406229 |
| relsgg-vitb16 | vulkan/q8_0 | 99.22% | 96.85% | 0.0339939 |

## Input and output contracts

The runtime checks all 50 preprocessed image/box arrays for bit equality, exercises 0/1/2/60 objects on CPU/CUDA/Vulkan, and rejects malformed dimensions. `../full_graph/contracts-*.json` and `preprocessing_parity.json` contain the actual results. Native top-predicate and calibrated scores are independently compared to raw logits for every image in the full benchmark.

## Reproduce

```sh
python3 cpp_ggml/scripts/validate_full_graph.py --backends cpu cuda vulkan --dtypes f32 f16 q8_0 --dynamic
python3 cpp_ggml/scripts/validate_dense_sampler.py
python3 cpp_ggml/scripts/report_full_validation.py
```

These are observed numerical differences, not a claim of bit identity between different kernels or quantized weights. Full-dataset GT metrics and speed are in [the image report](../full_graph/README.md).
