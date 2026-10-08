# Latency of the released checkpoints

The relation model alone on an **RTX 3080 Laptop GPU**: one image with 20 given
boxes, from preprocessing to decoded triplets. With a detector:
[end to end](end-to-end.md). Against other scene-graph systems on one A100:
[cross-system comparison](a100-systems.md).

<!-- BEGIN GENERATED LATENCY -->
Median ms per image, batch 1. **Bold**: the fastest checkpoint in each column.

| Checkpoint | PyTorch FP32 | PyTorch BF16 | ONNX Runtime FP32 | TensorRT FP32 | PyTorch FP32, 19,103 predicates | PyTorch BF16, 19,103 predicates |
|---|---:|---:|---:|---:|---:|---:|
| ViT-S | 20.8 | 18.5 | 22.8 | 15.1 | 22.2 | 19.5 |
| ViT-S+ | 23.2 | 18.6 | 25.8 | 16.8 | 24.7 | 19.6 |
| ViT-B | 42.4 | 20.0 | 45.2 | 34.2 | 43.7 | 21.2 |
| ConvNeXt-T | **19.9** | **15.9** | **21.4** | **14.7** | **21.3** | **16.7** |
| ConvNeXt-S | 27.5 | 18.1 | 29.0 | 20.6 | 28.7 | 19.5 |
| ConvNeXt-B | 39.6 | 22.0 | 40.1 | 32.3 | 40.0 | 23.3 |
| ConvNeXt-L | 70.3 | 33.6 | 72.4 | 55.0 | 71.4 | 34.7 |

The first four columns use the released 243-predicate vocabulary on every runtime; the last two the full 19,103-predicate training vocabulary, which only the Python API offers.

PyTorch 2.14.0+cu130, CUDA 13.0, TensorRT 10.16.1.11, ONNX Runtime 1.30.0; 11th Gen Intel(R) Core(TM) i9-11950H @ 2.60GHz.
<!-- END GENERATED LATENCY -->

## How it was measured

- **Inputs:** the six photos in `assets/reel/images/`, 448 × 448, with 20 fixed
  random boxes each.
- **Timing:** 3 rounds of 20 warm-up and 60 timed calls per configuration, in
  shuffled order; the median of the 180 calls. Synchronized wall time, batch 1.
- **Runtimes:** FP32 runs without TF32; BF16 is PyTorch autocast. The PyTorch
  columns pick each pair's best predicate on the GPU, as the Python API does;
  ONNX Runtime and TensorRT decode on the CPU. All give the same triplets.
- **Hardware:** a laptop with unlocked clocks, cooled to 70 °C before each block.
  Compare models and runtimes within a table, not across machines.

## Reproduce

Use CUDA PyTorch, TensorRT 10 and the GPU build of ONNX Runtime
([TensorRT setup](../../deploy/README.md#tensorrt-nvidia-gpu)):

```bash
pip install -e ".[tensorrt,hub]"
pip uninstall -y onnxruntime
pip install "onnxruntime-gpu==1.30.0"
```

Then, for each `model_id` (`relsgg-vits16`, `relsgg-vits16plus`, `relsgg-vitb16`,
`relsgg-convnext-{tiny,small,base,large}`), one at a time. Export the ONNX graph
as below rather than reusing one exported before 2026-09-22: older graphs can
return duplicate pairs under ONNX Runtime CUDA when few boxes are given.

```bash
model_id=relsgg-vits16
bundle_dir="runs/benchmark/gpu/$model_id"
mkdir -p "$bundle_dir"

hf download "maelic/$model_id" model.pth calibration.json predicate_embeddings.npz \
  --local-dir "checkpoints/$model_id"
cp "deploy/dist/$model_id/predicate_bank.npz" "$bundle_dir/"

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python deploy/export_onnx.py \
  --checkpoint "checkpoints/$model_id/model.pth" \
  --vocab-npz "$bundle_dir/predicate_bank.npz" --vocab-mode input \
  --out "$bundle_dir/relateanything.onnx" --check

python deploy/export_tensorrt.py --onnx "$bundle_dir/relateanything.onnx" \
  --check-images assets/reel/images/*.jpg

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python deploy/bench_gpu_backends.py \
  --checkpoint "checkpoints/$model_id/model.pth" --bundle "$bundle_dir" \
  --full-vocabulary --cooldown-temperature 70 \
  --out "runs/benchmark/gpu/$model_id.json"
```

Collect the seven records, then regenerate the tables and the model plot:

```bash
python - <<'PY'
import json
from pathlib import Path

models = ["relsgg-vits16", "relsgg-vits16plus", "relsgg-vitb16",
          "relsgg-convnext-tiny", "relsgg-convnext-small",
          "relsgg-convnext-base", "relsgg-convnext-large"]
records = [json.loads(Path(f"runs/benchmark/gpu/{model}.json").read_text())
           for model in models]
Path("runs/benchmark/gpu/family.json").write_text(
    json.dumps({"records": records}, indent=2) + "\n")
PY
python release/update_readme_latency.py --source runs/benchmark/gpu/family.json
python release/plot_model_family.py --source runs/benchmark/gpu/family.json
```

`update_readme_latency.py --check` verifies that the published tables match
your records.
