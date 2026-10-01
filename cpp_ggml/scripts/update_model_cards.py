#!/usr/bin/env python3
"""Generate all model cards from verified artifact and measurement manifests."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[2];models=root/'cpp_ggml/models'
manifest=json.loads((root/'cpp_ggml/benchmarks/full_gguf_manifest.json').read_text())['models']
lines=['# RelateAnything GGUF model cards','',
'Three official checkpoint families run through the native C++ image graph. Each has F32, F16 and Q8_0 storage: nine deployment models, no `demo` suffixes. Source weights and sidecars are under `pytorch/`; GGUFs are under `gguf/` and published in [Asher-1/relateanything-ggml-full](https://huggingface.co/Asher-1/relateanything-ggml-full).','',
'## Model families and intended scenarios','',
'| Official checkpoint | Backbone / upstream parameters | Scenario |','|---|---|---|',
'| [relsgg-vits16](https://huggingface.co/maelic/relsgg-vits16) | DINOv3 ViT-S/16, 46.1M | Lowest-capacity released tower; latency and CPU-sensitive deployment |',
'| [relsgg-vits16plus](https://huggingface.co/maelic/relsgg-vits16plus) | DINOv3 ViT-S/16+, 53.2M, SwiGLU | Upstream recommended balance of relation quality and cost |',
'| [relsgg-vitb16](https://huggingface.co/maelic/relsgg-vitb16) | DINOv3 ViT-B/16, 113.8M | Larger visual capacity; use when memory and latency permit |','',
'The checked-out upstream [release manifest](../../deploy/release_manifest.json) identifies exactly these three published `final` models. Its three `*-zeroshot` siblings are explicitly `unreleased`, have no `hf_repo`, and have no downloadable weights. They are not omitted supported releases. ONNX/TensorRT exports are deployment formats, not additional checkpoint families; the optional S+ ONNX file is not needed by GGML.','',
'Each local official snapshot contains `model.pth`, `config.json`, `relateanything.json`, `calibration.json`, `thresholds.json`, `predicate_embeddings.npz`, `predicate_bank.npz`, `text_student.pt`, `tokenizer.json` and `tokenizer_config.json`. Text assets support official Python vocabulary encoding; the C++ head accepts a precomputed replacement embedding bank.','',
'## Nine direct downloads and fingerprints','',
'Every artifact contains DINOv3, prompt/pool/sampler, relation transformer, deformable read, interaction and dynamic vocabulary weights. Export initializes the official full-vocabulary API: the 19,103-row released embedding bank and recomputed alpha replace the checkpoint bank. Predicate names, calibration and source hashes are embedded.','',
'| Model download | Size MiB | Tensors | SHA256 |','|---|---:|---:|---|']
for m in sorted(manifest,key=lambda x:x['gguf']):
 name=Path(m['gguf']).name
 lines.append(f'| [{name}]({m["download_url"]}) | {m["gguf_bytes"]/2**20:.2f} | {m["actual_tensor_count"]} | `{m["gguf_sha256"]}` |')
lines+=['','[The machine-readable manifest](../benchmarks/full_gguf_manifest.json) records the exact source checkpoint SHA, type counts, missing/extra tensors and numerical verification. All nine files pass the declared storage-conversion check.','',
'## Storage policy and selection','',
'| Storage | Conversion | Suitable use |','|---|---|---|',
'| F32 | Prepared runtime floating tensors stored in F32 | Numerical reference, closest weight representation to official PyTorch |',
'| F16 | Floating tensors stored in F16; auxiliary values converted as needed by native ops | Smaller files and weight buffers; latency depends on backend kernels |',
'| Q8_0 | Rank-2 matrices with inner dimension divisible by 32 use Q8_0; other tensors stay F32 | Lower weight bandwidth/storage, particularly useful on CUDA; inspect ranking drift for your vocabulary |','',
'GPU flash attention uses F16 K/V and F32 accumulation for every storage type. `--no-flash-attention` selects explicit F32 attention. Q8/F16 cannot reproduce original F32 weights bit-for-bit, and close-score predicates/candidates may change. Packed Q/K/V weights are created once at load without changing the stored model and consume additional runtime memory.','',
'## Measured full-pipeline evidence','',
'The [full report](../benchmarks/full_graph/README.md) and [metrics CSV](../benchmarks/full_graph/metrics.csv) are the current source for all 21 rows: three independent official PyTorch-CUDA baselines plus 18 native CUDA/Vulkan variants. All 50 retained RA-4M images, GT boxes and the full vocabulary are evaluated. Native timing includes JPEG decode, exact preprocessing, graph construction/allocation/transfers, inference and calibrated ranking. Loading and output files are excluded on both sides.','',
'![Measured pipeline latency and accuracy](../benchmarks/full_graph/metrics.png)','',
'![Metrics by image source](../benchmarks/full_graph/source_breakdown.png)','',
'Per-image grids cover every retained frame and every engine/storage variant. Numerical metrics include MAE, RMSE, max absolute error, cosine, predicate agreement and pair recall/Jaccard. Task metrics include R/mR@20/50/100, their harmonic F1, weighted recall and explicit local support buckets. Median/P95, throughput and allocator-scoped memory measurements accompany the accuracy rows.','',
'GGML memory is the sum of native weight/packed-QKV/scheduler buffers; PyTorch reports peak allocator bytes. These do not have identical scope and are not advertised as directly comparable total VRAM peaks. Shared-machine hardware/process snapshots are recorded. Refer to individual speedups instead of assuming every lower-precision format is faster.','',
'The [stage report](../benchmarks/full_graph_validation/README.md) checks eleven intermediate boundaries, custom vocabulary normalization/gating and 60-region dense candidate selection. The compact GT set has incomplete positive annotations; it cannot establish negative-set AP, OVS composite or detector accuracy. [Metric coverage](../benchmarks/METRIC_COVERAGE.md) explains those distinctions.','',
'## Supported interface and limits','',
'- Batch 1, 448×448, boxes, up to `Options.max_boxes` (default 60), 128 candidate slots; all three released architectures.','- CPU, CUDA and Vulkan execute native GGML graphs; no Python inference or cuDNN dependency in C++.','- `Options` configures model/backend/precision mode/threads/vocabulary; `ImageOptions` supplies image and pixel XYXY boxes.','- Replacement vocabularies use finite row-major F32 `[V,512]` embeddings. The C++ head normalizes them and computes its learned gate. Free-form text encoding is available with the Python engine.','- Optional upstream masks, detector-score weighting and decomposed outputs are outside this box-only deployment profile.','- GT-box evaluation measures relation prediction, not detector localization or end-to-end detection recall.','',
'```sh','./run_ggml.sh --backend cuda --model-name relsgg-vits16plus --dtype q8_0 --download','./run_ggml.sh --backend vulkan --model-name relsgg-vitb16 --dtype f16','./run_ggml.sh --engine python --backend cuda','```','',
'## Old demo artifacts','',
'`demo-f32.gguf` and `demo-f32-v2.gguf` were obsolete random adapter fixtures and are absent locally. The three old adapter storage fixtures remain explicitly named `relation_pair_linear_v1-fixture-*` under `test_data/fixtures/`. They are smoke-test inputs, contain no official DINOv3 checkpoint, and contribute no numbers to the full-model reports. The historical `Asher-1/relateanything-gguf` repository is not the deployment repository.','',
'## Provenance and terms','',
'This is a downstream GGUF conversion, not an upstream-official GGUF release. The source model cards identify the DINOv3 model license; conversion does not replace those terms. See the [source model card](https://huggingface.co/maelic/relsgg-vits16plus) and [DINOv3 license](https://ai.meta.com/resources/models-and-libraries/dinov3-license/). Source code licensing remains governed by this repository’s LICENSE. Retained dataset images keep their original COCO/Objects365/Open Images terms.']
summary_path=root/'cpp_ggml/benchmarks/full_graph/summary.json'
if summary_path.exists():
    summary=json.loads(summary_path.read_text())['models']
    table=['', '### Measured model/backend/storage matrix', '',
           'RTX 3060; 50 GT-box images, 19,103 predicates; 2 warmups, 3 repeats per image. Speedup is each model’s Python median divided by its GGML median. All accepted timing runs were monitored and had no transient GPU job; two persistent system GPU services were explicitly allowed on both sides.', '',
           '| Model | Engine/storage | Median / P95 ms | Speedup | R@50 | mR@50 | Logit RMSE | Top-1 | GGML buffers MiB |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for model,entries in summary.items():
        reference=entries['python']['latency_median_ms']
        for engine,row in entries.items():
            memory=row.get('ggml_backend_buffer_bytes')
            memory=f'{memory/2**20:.1f}' if memory else '—'
            table.append(f'| {model} | {engine} | {row["latency_median_ms"]:.2f} / {row["latency_p95_ms"]:.2f} | {reference/row["latency_median_ms"]:.2f}x | {row["R@50"]:.2%} | {row["mR@50"]:.2%} | {row.get("logits_rmse",0):.5f} | {row.get("predicate_top1_agreement",1):.2%} | {memory} |')
    table += ['', 'CUDA is faster for all nine measured variants. Vulkan is close to Python for S/S+ and faster for B; some S/S+ rows are slightly slower. No claim that every configuration exceeds Python is made. F32 does not imply bit-identical GPU kernels; full errors and candidate changes are disclosed.', '']
    index=lines.index('## Supported interface and limits')
    lines[index:index]=table

(models/'MODEL_CARDS.md').write_text('\n'.join(lines)+'\n')
header='''---
library_name: ggml
license: other
license_name: dinov3-license
license_link: https://ai.meta.com/resources/models-and-libraries/dinov3-license/
tags:
- gguf
- scene-graph-generation
- visual-relationship-detection
- open-vocabulary
- cuda
- vulkan
---

# RelateAnything native full-graph GGUF

Nine downstream conversions of the three official RelateAnything models run
with native C++17/GGML on CPU, CUDA and Vulkan. The graph implements DINOv3,
box prompt, spatial pool, pair sampler, relation transformer, deformable read,
interaction and the dynamic vocabulary head, including calibrated triplets.
C++ inference needs no PyTorch, Python model execution or cuDNN.

Source: [Asher-1/RelateAnything-GGML](https://github.com/Asher-1/RelateAnything-GGML).
All tensors have passed the declared storage conversion check. The full released
vocabulary and calibration match the official full-vocabulary Python API.

'''
card=header+'\n'.join(lines[lines.index('## Model families and intended scenarios'):lines.index('## Old demo artifacts')])+'\n'
card=card.replace('(../benchmarks/', '(benchmarks/').replace('(../../deploy/', '(https://github.com/Asher-1/RelateAnything-GGML/blob/main/deploy/')
card+='\n## Terms\n\n'+lines[-1]+'\n'
(models/'HF_MODEL_CARD.md').write_text(card)
print('Model cards updated:',len(manifest),'verified artifacts')
