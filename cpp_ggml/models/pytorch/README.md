# PyTorch checkpoints

The official model snapshots are stored in `relsgg-vits16/`,
`relsgg-vits16plus/`, and `relsgg-vitb16/`. They come from the corresponding
`maelic/relsgg-*` Hugging Face model repositories. Each contains `model.pth`
plus configuration, calibration, text-student, tokenizer, and vocabulary
assets. The optional 208 MB ONNX file in `relsgg-vits16plus` was not fetched.

The [release manifest](../../../deploy/release_manifest.json) lists three
published final models. The three `*-zeroshot` training siblings are explicitly
unreleased and have no published weight repository. They are not missing downloads.

`export_gguf.py --full` prepares the official runtime vocabulary and exports
runnable F32/F16/Q8_0 image-graph models. See [model cards](../MODEL_CARDS.md)
for family choice, inventory, hashes, quantization and measured behavior.
