#!/usr/bin/env python3
"""Generate all model cards from verified artifact and measurement manifests."""
from pathlib import Path
import argparse
import json
root=Path(__file__).resolve().parents[2];models=root/'cpp_ggml/models'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--benchmark',type=Path,default=root/'cpp_ggml/benchmarks/current/full_graph')
parser.add_argument('--stages',type=Path,help='current intermediate-stage report')
args=parser.parse_args()
benchmark=args.benchmark.resolve()
audit=json.loads((benchmark/'release_audit.json').read_text())
if (audit.get('status')!='complete' or not audit.get('all_complete') or
        audit.get('primary_ggml_image_configurations') != 1350):
    raise SystemExit('model cards require a complete current release audit')
stage_folder=(args.stages or benchmark.parent/'stages').resolve()
stage_rows=json.loads((stage_folder/'summary.json').read_text())
dense_rows=json.loads((stage_folder.parent/'dense/results.json').read_text())
if len(stage_rows)!=27 or len(dense_rows)!=27:
    raise SystemExit('model cards require all 27 stage and dense configurations')
f32_dense=[row for row in dense_rows if row['dtype']=='f32']
f32_dense_agree=sum(row['pair_set_jaccard']==1.0 and row['predicate_top1_agreement']==1.0
                    for row in f32_dense)
stage_boundary_counts=sorted({len(row['stages']) for row in stage_rows})
stage_boundary_text='/'.join(map(str,stage_boundary_counts))
report='../benchmarks/summary/README.md'
manifest=json.loads((root/'cpp_ggml/benchmarks/full_gguf_manifest.json').read_text())['models']
lines=['# RelateAnything GGUF model cards','',
'Execution provenance and storage conversion are verified. Numerical and performance gates, effective auto precision modes, and any accepted deviation are recorded in the [release acceptance status](../benchmarks/RELEASE_STATUS.md).','',
'Three official checkpoint families run through the native C++ image graph. Each has F32, F16 and Q8_0 storage: nine deployment models, no `demo` suffixes. Source weights and sidecars are under `pytorch/`; GGUFs are under `gguf/` and published in [Asher-1/relateanything-ggml-full](https://huggingface.co/Asher-1/relateanything-ggml-full).','',
'## Model families and intended scenarios','',
'| Official checkpoint | Backbone / upstream parameters | Scenario |','|---|---|---|',
'| [relsgg-vits16](https://huggingface.co/maelic/relsgg-vits16) | DINOv3 ViT-S/16, 46.1M | Lowest-capacity released tower; latency and CPU-sensitive deployment |',
'| [relsgg-vits16plus](https://huggingface.co/maelic/relsgg-vits16plus) | DINOv3 ViT-S/16+, 53.2M, SwiGLU | Upstream recommended balance of relation quality and cost |',
'| [relsgg-vitb16](https://huggingface.co/maelic/relsgg-vitb16) | DINOv3 ViT-B/16, 113.8M | Larger visual capacity; use when memory and latency permit |','',
'The checked-out upstream [release manifest](../../deploy/release_manifest.json) identifies exactly these three published `final` models. Its three `*-zeroshot` siblings are explicitly `unreleased`, have no `hf_repo`, and have no downloadable weights. They are not omitted supported releases. ONNX/TensorRT exports are deployment formats, not additional checkpoint families; the optional S+ ONNX file is not needed by GGML.','',
'Each local official snapshot contains `model.pth`, `config.json`, `relateanything.json`, `calibration.json`, `thresholds.json`, `predicate_embeddings.npz`, `predicate_bank.npz`, `text_student.pt`, `tokenizer.json` and `tokenizer_config.json`. C++ accepts a replacement embedding bank or encodes free text with the exported text-student GGUF and official tokenizer.','',
'## Nine direct downloads and fingerprints','',
'Every artifact contains DINOv3, prompt/pool/sampler, relation transformer, deformable read, interaction and dynamic vocabulary weights. Export initializes the official full-vocabulary API: the 19,103-row released embedding bank and recomputed alpha replace the checkpoint bank. Predicate names, calibration and source hashes are embedded.','',
'| Model download | Size MiB | Tensors | SHA256 |','|---|---:|---:|---|']
for m in sorted(manifest,key=lambda x:x['gguf']):
 name=Path(m['gguf']).name
 lines.append(f'| [{name}]({m["download_url"]}) | {m["gguf_bytes"]/2**20:.2f} | {m["actual_tensor_count"]} | `{m["gguf_sha256"]}` |')
lines += ['', '### Native text-student downloads', '',
          'Free-text encoding uses one additional F32 student for the selected model '
          'family, plus its official tokenizer.json. Image weights remain unchanged.', '',
          '| Text student | Size MiB | Tensors | SHA256 |', '|---|---:|---:|---|']
for path in sorted((models/'gguf').glob('relsgg-*-text-student-f32.json')):
    value=json.loads(path.read_text());artifact=path.with_suffix('.gguf')
    lines.append(f'| [{artifact.name}](https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/gguf/{artifact.name}) | '
                 f'{artifact.stat().st_size/2**20:.2f} | {value["tensors"]} | `{value["gguf_sha256"]}` |')
lines+=['','[The machine-readable manifest](../benchmarks/full_gguf_manifest.json) records the exact source checkpoint SHA, type counts, missing/extra tensors and numerical verification. All nine files pass the declared storage-conversion check.','',
'## Storage policy and selection','',
'| Storage | Conversion | Suitable use |','|---|---|---|',
'| F32 | Prepared runtime floating tensors stored in F32 | Numerical reference, closest weight representation to official PyTorch |',
'| F16 | Floating tensors stored in F16; auxiliary values converted as needed by native ops | Smaller files and weight buffers; latency depends on backend kernels |',
'| Q8_0 | Rank-2 matrices with inner dimension divisible by 32 use Q8_0, except the `vocab_head.W` predicate bank (F16, keeps close top-1 predicates stable under CPU Q8_0 activation rounding); other tensors stay F32 | Lower weight bandwidth/storage, particularly useful on CUDA; inspect ranking drift for your vocabulary |','',
'Default GPU flash attention uses F16 K/V and F32 accumulation. CUDA with F32 weights defaults to `--strict-f32`, which preserves F32 matrix operands and selects explicit attention, so close top-128 candidate pairs do not flip; `--no-strict-f32` restores the faster accelerated path. `--strict-f32` can also be forced for other storage types. `--no-flash-attention` changes attention only. Q8/F16 cannot reproduce original F32 weights bit-for-bit, and close-score predicates/candidates may change. Packed Q/K/V weights consume additional runtime memory.','',
'## Measured full-pipeline evidence','',
'The [portable full report](../benchmarks/summary/README.md) contains three independent official PyTorch-CUDA baselines, 27 primary CPU/CUDA/Vulkan variants, and three additional CUDA Q8 F32-input diagnostics. Each row covers all 50 retained RA-4M images, GT boxes and the full vocabulary. Native timing includes JPEG decode, exact preprocessing, graph construction/allocation/transfers, inference and calibrated ranking. Loading and output files are excluded on both sides. The default `auto` request enables strict arithmetic only for CUDA F32; CPU, Vulkan, F16 and Q8_0 remain non-strict. Each result records both the requested and effective precision mode, so timing is interpreted using the actual path.','',
'[Offline detailed gallery](../benchmarks/gallery/index.html)\n\n![All backend pipeline latency](../benchmarks/gallery/latency_all_backends.png)','',
'![Numerical and candidate agreement](../benchmarks/gallery/accuracy_candidates.png)','',
'Per-image grids cover every retained frame and every engine/storage variant. Numerical metrics include MAE, RMSE, max absolute error, cosine, predicate agreement and pair recall/Jaccard. Task metrics include R/mR@20/50/100, their harmonic F1, weighted recall and explicit local support buckets. Median/P95, throughput and allocator-scoped memory measurements accompany the accuracy rows.','',
'Allocator-scoped GGML buffers and PyTorch peaks remain separately named. Process measurements in the [portable full report](../benchmarks/summary/README.md) additionally sample both engines from startup through load, warmup, inference and output: RSS/HWM plus driver process device memory at 200 ms cadence. These sampled peaks can miss shorter spikes. Shared-machine hardware/process snapshots and every slower native image are retained. Refer to individual speedups instead of assuming every lower-precision format is faster.','',
f'The stage report in the [portable full report](../benchmarks/summary/README.md) checks {stage_boundary_text} intermediate boundaries in all 27 model/backend/storage configurations, custom vocabulary normalization/gating and 60-region dense candidate selection. {f32_dense_agree}/{len(f32_dense)} F32 dense runs retain every candidate and agree on top-1 predicates against the CPU reference. Fixed-candidate traces separate arithmetic error from changed contextual attention. The compact GT set has incomplete positive annotations; it cannot establish negative-set AP, OVS composite or detector accuracy. Public PSG/IndoorVG/VG150 and labelled SpatialSense subsets add independent evaluator regressions. [Metric coverage](../benchmarks/METRIC_COVERAGE.md) explains their limited scope.','',
'## Supported interface and limits','',
'- Batch 1, 448×448, boxes, up to `ModelOptions.max_boxes` (default 60), 128 candidate slots; all three released architectures.','- CPU, CUDA and Vulkan execute native GGML graphs; no Python inference or cuDNN dependency in C++.','- `ModelOptions` configures model/backend/precision mode/threads/vocabulary, with explicit `RuntimeOptions` for backend policy. `ImageInput` supplies image and pixel XYXY boxes.','- The versioned C ABI accepts borrowed RGB/RGBA/GRAY/BGR/BGRA memory with stride; results have independent ownership. A host can bind its existing backend lease. See [integration contract](../benchmarks/PORTING_NOTES.md).','- Replacement vocabularies use finite row-major F32 `[V,512]` embeddings or native CLIP BPE/text-student encoding. The head normalizes them and computes its learned gate.','- Native masks, detector-score weighting and decomposed spatial/semantic outputs are independently compared with Python on all three models and backends. The 50-image timing profile uses boxes only.','- GT-box evaluation measures relation prediction, not detector localization or end-to-end detection recall.','',
'```sh','./run_ggml.sh --backend cuda --model-name relsgg-vits16plus --dtype q8_0 --download','./run_ggml.sh --backend vulkan --model-name relsgg-vitb16 --dtype f16','./run_ggml.sh --engine python --backend cuda','```','',
'## Fixture isolation','',
'The three `relation_pair_linear_v1-fixture-*` files under `test_data/fixtures/` are small format/loader inputs. They contain no official DINOv3 checkpoint and contribute no numbers to the full-model reports.','',
'## Provenance and terms','',
'This is a downstream GGUF conversion, not an upstream-official GGUF release. The source model cards identify the DINOv3 model license; conversion does not replace those terms. See the [source model card](https://huggingface.co/maelic/relsgg-vits16plus) and [DINOv3 license](https://ai.meta.com/resources/models-and-libraries/dinov3-license/). Source code licensing remains governed by this repository’s LICENSE. Retained dataset images keep their original COCO/Objects365/Open Images terms.']
summary_path=benchmark/'summary.json'
if summary_path.exists():
    measured=json.loads(summary_path.read_text())
    summary=measured['models']
    table=['', '### Measured model/backend/storage matrix', '',
           f'RTX 3060; 50 GT-box images, 19,103 predicates; {measured["warmup"]} warmups, {measured["repeats"]} repeats per image. Speedup is each model’s Python median divided by its GGML median. Accepted timing runs had no unapproved transient GPU process; allowed persistent PIDs and Python bracket drift are recorded per run.', '',
           '| Model | Engine/storage | Median / P95 ms | Speedup | R@50 | mR@50 | Logit RMSE | Top-1 | GGML buffers MiB |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for model,entries in summary.items():
        reference=entries['python']['latency_median_ms']
        for engine,row in entries.items():
            memory=row.get('ggml_backend_buffer_bytes')
            memory=f'{memory/2**20:.1f}' if memory else '—'
            table.append(f'| {model} | {engine} | {row["latency_median_ms"]:.2f} / {row["latency_p95_ms"]:.2f} | {reference/row["latency_median_ms"]:.2f}x | {row["R@50"]:.2%} | {row["mR@50"]:.2%} | {row.get("logits_rmse",0):.5f} | {row.get("predicate_top1_agreement",1):.2%} | {memory} |')
    all_faster=all(entries['python']['latency_median_ms']>r['latency_median_ms']
                   for entries in summary.values() for engine,r in entries.items() if engine!='python')
    table += ['', ('All measured native medians are faster than their corresponding Python baseline in this run.' if all_faster else 'Some measured native configurations do not beat Python; consult the individual rows.')+
              ' Shared-workstation results are specific to the recorded process load. F32 does not imply bit-identical GPU kernels; full errors and candidate changes are disclosed.', '',
              f'The [portable full report]({report}) includes weight-rounding versus same-weight arithmetic for every retained image through the official Python graph with all GGUF weights restored, plus the CUDA Q8 F32-activation precision isolation. Superseded experiments were removed after replacement; the cleanup manifest records their identities.', '']
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
card=header+'\n'.join(lines[lines.index('## Model families and intended scenarios'):lines.index('## Fixture isolation')])+'\n'
card=card.replace('## Model families and intended scenarios',
    'Execution provenance and storage conversion are verified. Numerical and performance '
    'gates, effective auto precision modes, and any accepted deviation are recorded in the '
    '[release acceptance status](../benchmarks/RELEASE_STATUS.md).\n\n## Model families and intended scenarios',1)
card=card.replace('(../benchmarks/', '(benchmarks/').replace('(../outputs/', '(outputs/').replace('(../../deploy/', '(https://github.com/Asher-1/RelateAnything-GGML/blob/main/deploy/')
card+='\n## Terms\n\n'+lines[-1]+'\n'
(models/'HF_MODEL_CARD.md').write_text(card)
print('Model cards updated:',len(manifest),'verified artifacts')
