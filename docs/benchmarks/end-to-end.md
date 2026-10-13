# Latency with a detector

One image through detection, the relation model and triplet decoding on an
**RTX 3080 Laptop GPU**, for the seven checkpoints and three Ultralytics
detectors. Without a detector: [relation model only](README.md).

<!-- BEGIN GENERATED E2E LATENCY -->
Median ms per image, batch 1, detection included. **Bold**: the fastest
checkpoint with each detector in each column.

| Detector | Checkpoint | PyTorch FP32 | PyTorch BF16 | ONNX Runtime FP32 | TensorRT FP32 | PyTorch FP32, 19,103 predicates | PyTorch BF16, 19,103 predicates |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLO26m | ViT-S | 42.9 | 41.6 | 45.6 | 37.2 | 44.3 | 42.1 |
| YOLO26m | ViT-S+ | 45.3 | 41.8 | 48.5 | 38.7 | 46.9 | 43.1 |
| YOLO26m | ViT-B | 60.9 | 39.2 | 64.8 | 54.9 | 62.5 | 40.0 |
| YOLO26m | ConvNeXt-T | **38.6** | **33.8** | **40.2** | **33.5** | **40.0** | **34.5** |
| YOLO26m | ConvNeXt-S | 46.6 | 37.2 | 48.6 | 39.9 | 47.8 | 37.9 |
| YOLO26m | ConvNeXt-B | 58.3 | 40.9 | 60.0 | 51.5 | 59.9 | 42.1 |
| YOLO26m | ConvNeXt-L | 89.4 | 51.9 | 91.9 | 74.1 | 90.5 | 53.0 |
| YOLO-World v2-m | ViT-S | 38.7 | 34.9 | 40.6 | **33.0** | 39.9 | 35.6 |
| YOLO-World v2-m | ViT-S+ | 41.1 | 35.1 | 43.6 | 34.6 | 42.3 | 36.2 |
| YOLO-World v2-m | ViT-B | 61.1 | 38.2 | 64.3 | 53.1 | 61.8 | 39.6 |
| YOLO-World v2-m | ConvNeXt-T | **38.0** | **33.2** | **39.9** | 33.2 | **39.2** | **34.1** |
| YOLO-World v2-m | ConvNeXt-S | 45.5 | 36.9 | 47.4 | 41.6 | 46.9 | 36.9 |
| YOLO-World v2-m | ConvNeXt-B | 57.8 | 40.6 | 59.9 | 51.0 | 58.9 | 42.0 |
| YOLO-World v2-m | ConvNeXt-L | 88.8 | 51.3 | 91.2 | 73.6 | 90.0 | 52.5 |
| YOLOE-26m | ViT-S | 47.8 | 42.7 | 49.1 | 41.5 | 48.7 | 44.9 |
| YOLOE-26m | ViT-S+ | 49.7 | 44.9 | 52.1 | 42.7 | 50.5 | 44.8 |
| YOLOE-26m | ViT-B | 68.6 | 46.5 | 71.9 | 62.4 | 70.4 | 48.2 |
| YOLOE-26m | ConvNeXt-T | **46.0** | **40.8** | **47.6** | **40.9** | **47.2** | **42.1** |
| YOLOE-26m | ConvNeXt-S | 53.7 | 43.5 | 55.5 | 46.9 | 54.8 | 44.8 |
| YOLOE-26m | ConvNeXt-B | 65.8 | 48.9 | 67.3 | 59.5 | 67.0 | 50.1 |
| YOLOE-26m | ConvNeXt-L | 97.5 | 59.7 | 99.9 | 82.2 | 99.0 | 60.6 |

The detector runs in PyTorch FP32; the columns choose the relation runtime and vocabulary (243 released predicates, or the full 19,103 in the Python API). Measured on 6 photos with 2 to 8 detected objects.

Detection alone takes 20 ms with YOLO26m, 19 ms with YOLO-World v2-m, 26 ms with YOLOE-26m; decoding takes under 0.4 ms. The rest is the relation model.

PyTorch 2.14.0+cu130, CUDA 13.0, TensorRT 10.16.1.11, ONNX Runtime 1.30.0, Ultralytics 8.4.159; 11th Gen Intel(R) Core(TM) i9-11950H @ 2.60GHz.
<!-- END GENERATED E2E LATENCY -->

## How it was measured

- **Detectors:** YOLO26m (`yolo26m.pt`), YOLO-World v2-m (`yolov8m-worldv2.pt`)
  and YOLOE-26m (`yoloe-26m-seg.pt`, mask computation included), all with the
  COCO-80 classes, PyTorch FP32, 640 px, confidence 0.25, at most 100
  detections. The 20 most confident boxes go to the relation model.
- **Timing:** every call reruns detection. Same sampling as the
  [relation-only benchmark](README.md#how-it-was-measured): the six repository
  photos, 3 rounds of 20 warm-up and 60 timed calls, medians of 180 calls.
  Serial latency, not pipelined video throughput.
- **Scope:** latency only, not detection or scene-graph accuracy. The three
  detectors differ in training data and quality.

## Reproduce

Set up the [relation-only benchmark](README.md#reproduce) and its seven bundles,
then prepare the detectors once (weights download, COCO-80 prompts encoded on
the CPU; detector code and weights keep their
[licenses](../../THIRD_PARTY_NOTICES.md)):

```bash
pip install "ultralytics==8.4.159"
pip install "git+https://github.com/ultralytics/CLIP.git@a13192f8cb767260d7dfd98c843b0716593169e7"
CUDA_VISIBLE_DEVICES="" python -m deploy.e2e --size m --out checkpoints/detectors/e2e
```

Run the 21 detector and checkpoint pairs one at a time (add `--dry-run` to print
the plan only):

```bash
for detector in yolo26 yolo-world yoloe; do
  case "$detector" in
    yolo26) weights=yolo26m-coco80.pt ;;
    yolo-world) weights=yolov8m-worldv2-coco80.pt ;;
    yoloe) weights=yoloe-26m-seg-coco80.pt ;;
  esac
  for model_id in relsgg-vits16 relsgg-vits16plus relsgg-vitb16 \
      relsgg-convnext-tiny relsgg-convnext-small relsgg-convnext-base relsgg-convnext-large; do
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python deploy/bench_gpu_backends.py \
      --checkpoint "checkpoints/$model_id/model.pth" \
      --bundle "runs/benchmark/gpu/$model_id" --full-vocabulary \
      --detector "$detector" --detector-weights "checkpoints/detectors/e2e/$weights" \
      --cooldown-temperature 70 \
      --out "runs/benchmark/e2e/$detector-$model_id.json" || break 2
  done
done
```

Then regenerate the tables (`--check` verifies the published ones):

```bash
python - <<'PY'
import json
from pathlib import Path

models = ["relsgg-vits16", "relsgg-vits16plus", "relsgg-vitb16",
          "relsgg-convnext-tiny", "relsgg-convnext-small",
          "relsgg-convnext-base", "relsgg-convnext-large"]
records = [json.loads(Path(f"runs/benchmark/e2e/{detector}-{model}.json").read_text())
           for detector in ("yolo26", "yolo-world", "yoloe") for model in models]
Path("runs/benchmark/e2e/combined.json").write_text(
    json.dumps({"records": records}, indent=2) + "\n")
PY
python release/update_readme_e2e.py --source runs/benchmark/e2e/combined.json
```
