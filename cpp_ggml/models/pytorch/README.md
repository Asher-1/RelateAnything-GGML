# PyTorch checkpoints

The official model snapshots are stored in `relsgg-vits16/`,
`relsgg-vits16plus/`, `relsgg-vitb16/`, `relsgg-convnext-tiny/`,
`relsgg-convnext-small/`, `relsgg-convnext-base/`, and `relsgg-convnext-large/`.
They come from the corresponding
`maelic/relsgg-*` Hugging Face model repositories. Each contains `model.pth`
plus configuration, calibration, text-student, tokenizer, and vocabulary
assets (all seven share the identical text-student weights). The optional
per-model ONNX file was not fetched.

The [release manifest](../../../deploy/release_manifest.json) lists seven
published final models. The three `*-zeroshot` training siblings are explicitly
unreleased and have no published weight repository. They are not missing downloads.

`export_gguf.py --full` prepares the official runtime vocabulary and exports
runnable F32/F16/Q8_0 image-graph models. See [model cards](../MODEL_CARDS.md)
for family choice, inventory, hashes, quantization and measured behavior.
