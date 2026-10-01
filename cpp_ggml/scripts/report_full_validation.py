#!/usr/bin/env python3
"""Collect all measured stage, custom-bank and dense-sampler validation."""
import json
from pathlib import Path
from benchmark_full_graph import ROOT, save


def main():
    folder=ROOT/'cpp_ggml/benchmarks/full_graph_validation'
    rows=[]
    for path in sorted(folder.glob('relsgg-*/*/stages/comparison.json')):
        row=json.loads(path.read_text())
        dynamic=path.parent.parent/'dynamic/comparison.json'
        if dynamic.exists():row['dynamic_vocabulary']=json.loads(dynamic.read_text())
        rows.append(row)
    save(folder/'summary.json',rows)
    lines=['# Full graph intermediate validation','',
      'Eleven boundaries are compared against independent upstream PyTorch traces.','The trace uses the first retained real image and all its GT boxes. CPU F32 uses explicit attention; GPU variants use F16 K/V flash attention with F32 accumulation. Stage matrices compare matching subject/object IDs, excluding invalid slots.','',
      '| Model | Backend/storage | Fused map RMSE | Transformer RMSE | Deformable RMSE | Final RMSE | Final max abs | Custom vocabulary RMSE |',
      '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        st=r['stages']; f=r['final']; d=r.get('dynamic_vocabulary',{})
        lines.append(f'| {r["model"]} | {r["backend"]}/{r["dtype"]} | {st["fused_patch_map"]["rmse"]:.6g} | {st["rel_transformer"]["rmse"]:.6g} | {st["deformable_read"]["rmse"]:.6g} | {f["logits_rmse"]:.6g} | {f["logits_max_abs"]:.6g} | {d.get("logits_rmse",float("nan")):.6g} |')
    lines += ['', 'Each model/backend directory contains `stages/comparison.json` with all eleven boundaries and `dynamic/comparison.json`. Dynamic tests use five nonadjacent bank rows with non-unit scales plus an unseen composite direction. The independent CPU Python API performs its own normalization and gate computation.', '',
      '## Dense candidate selection', '', 'A real retained image with 60 deterministic distinct boxes exercises 3,540 valid ordered pairs, geometry TopK=400 and relation TopK=128. Generated boxes have no task GT; this is a numerical sampler test. Close quantized scores can change the candidate set, which in turn changes contextual attention for otherwise matching pairs. All errors and pair changes are retained.', '',
      '| Model | Backend/storage | Pair recall | Predicate agreement | Logit RMSE |', '|---|---|---:|---:|---:|']
    dense=folder/'dense_sampler/results.json'
    if dense.exists():
        for r in json.loads(dense.read_text()):lines.append(f'| {r["model"]} | {r["backend"]}/{r["dtype"]} | {r["pair_set_recall"]:.2%} | {r["predicate_top1_agreement"]:.2%} | {r["logits_rmse"]:.6g} |')
    lines += ['', '## Input and output contracts', '',
      'The runtime checks all 50 preprocessed image/box arrays for bit equality, exercises 0/1/2/60 objects on CPU/CUDA/Vulkan, and rejects malformed dimensions. `../full_graph/contracts-*.json` and `preprocessing_parity.json` contain the actual results. Native top-predicate and calibrated scores are independently compared to raw logits for every image in the full benchmark.', '',
      '## Reproduce', '', '```sh',
      'python3 cpp_ggml/scripts/validate_full_graph.py --backends cpu cuda vulkan --dtypes f32 f16 q8_0 --dynamic',
      'python3 cpp_ggml/scripts/validate_dense_sampler.py',
      'python3 cpp_ggml/scripts/report_full_validation.py', '```', '',
      'These are observed numerical differences, not a claim of bit identity between different kernels or quantized weights. Full-dataset GT metrics and speed are in [the image report](../full_graph/README.md).']
    (folder/'README.md').write_text('\n'.join(lines)+'\n')
    print(f'{len(rows)} stage/dynamic configurations summarized')
if __name__=='__main__':main()
