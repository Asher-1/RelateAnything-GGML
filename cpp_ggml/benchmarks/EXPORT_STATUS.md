# Official full-model export and execution

All three official checkpoint families have runnable native C++ full graphs
and F32/F16/Q8_0 GGUF exports. The nine artifacts are inventoried in
[full_gguf_manifest.json](full_gguf_manifest.json); all nine pass names, shapes,
tensor counts and numerical storage verification. Each per-file `*-container.json`
records exact checkpoint and GGUF hashes, size and tensor types.

| Checkpoint | State tensors | Storage variants |
|---|---:|---|
| relsgg-vits16 | 392 | F32, F16, Q8_0 |
| relsgg-vits16plus | 416 | F32, F16, Q8_0 |
| relsgg-vitb16 | 392 | F32, F16, Q8_0 |

The exporter initializes the official full vocabulary, replacing the checkpoint
bank and alpha exactly as the Python API does. It embeds 19,103 predicate names,
normalized runtime embeddings, calibration and source hashes. Thus a direct
comparison to the original checkpoint bank alone would be the wrong oracle.
F32 preserves these prepared values; F16 and Q8_0 are checked against the
specified conversion, not described as lossless original-weight storage.

```sh
python3 cpp_ggml/scripts/export_gguf.py --full \
  --checkpoint cpp_ggml/models/pytorch/relsgg-vits16plus/model.pth \
  --dtype f32 --out cpp_ggml/models/gguf/relsgg-vits16plus-f32.gguf
python3 cpp_ggml/scripts/verify_full_gguf.py \
  --checkpoint cpp_ggml/models/pytorch/relsgg-vits16plus/model.pth \
  --gguf cpp_ggml/models/gguf/relsgg-vits16plus-f32.gguf
./run_ggml.sh --backend cuda --model-name relsgg-vits16plus --dtype f32
```

[Native full-pipeline measurements](full_graph/README.md) cover every retained
image and all 18 CUDA/Vulkan variants. [Intermediate validation](full_graph_validation/README.md)
covers DINOv3, spatial pool, box prompt, geometry, projection, transformer,
deformable read, interaction and vocabulary stages, plus replacement vocabularies.
The [model cards](../models/MODEL_CARDS.md) provide all nine download links,
quantization policy, scenarios and measured limitations.
