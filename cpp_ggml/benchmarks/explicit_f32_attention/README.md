# Explicit F32 attention diagnostics

Independent official-model comparison on the first retained image. `--no-flash-attention` uses explicit attention matrix products and softmax; it does not force every vendor matrix kernel to reproduce the same floating-point order as PyTorch. This is a numerical diagnostic, not a substitute for the controlled 50-image deployment timing.

| Model | Backend | Final logits RMSE | Max abs | Custom vocabulary RMSE |
|---|---|---:|---:|---:|
| relsgg-vits16 | cuda | 0.00192984 | 0.0113707 | 0.00178109 |
| relsgg-vits16 | vulkan | 0.00231143 | 0.0167923 | 0.00234383 |
| relsgg-vits16plus | cuda | 0.00200907 | 0.014431 | 0.00181213 |
| relsgg-vits16plus | vulkan | 0.00501591 | 0.0319135 | 0.00525641 |
| relsgg-vitb16 | cuda | 0.00194636 | 0.0129361 | 0.00143498 |
| relsgg-vitb16 | vulkan | 0.00153541 | 0.0130844 | 0.00130344 |

```sh
python3 cpp_ggml/scripts/validate_full_graph.py --backends cuda vulkan --dtypes f32 --no-flash --dynamic --output cpp_ggml/benchmarks/explicit_f32_attention
```

[Default deployment graph validation](../full_graph_validation/README.md) · [Controlled full-image comparison](../full_graph/README.md)
