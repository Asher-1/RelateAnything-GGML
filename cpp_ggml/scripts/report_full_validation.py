#!/usr/bin/env python3
"""Collect all measured stage, custom-bank and dense-sampler validation."""
import argparse
import json
import os
from pathlib import Path
from benchmark_full_graph import ROOT, save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,default=ROOT/'cpp_ggml/outputs/probes/stages')
    parser.add_argument('--benchmark',type=Path,default=(ROOT/'cpp_ggml/benchmarks/current').resolve().parent/'artifacts/full_graph')
    parser.add_argument('--geometry-baseline',type=Path,help='dense sampler results using historical Vulkan GEMM')
    parser.add_argument('--dense',type=Path,help='explicit dense sampler results')
    args=parser.parse_args();folder=args.folder.resolve()
    rows=[]
    for path in sorted(folder.glob('relsgg-*/*/stages/comparison.json')):
        row=json.loads(path.read_text())
        dynamic=path.parent.parent/'dynamic/comparison.json'
        if dynamic.exists():row['dynamic_vocabulary']=json.loads(dynamic.read_text())
        rows.append(row)
    save(folder/'summary.json',rows)
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    stages=list(rows[0]['stages'])
    errors=np.array([[r['stages'][s]['rmse'] for s in stages] for r in rows])
    fig,ax=plt.subplots(figsize=(13, max(5, len(rows)*.3)),layout='constrained')
    plot=ax.imshow(np.log10(np.maximum(errors,1e-10)),aspect='auto',cmap='viridis')
    ax.set_xticks(range(len(stages)),stages,rotation=35,ha='right')
    ax.set_yticks(range(len(rows)),[r['model'].removeprefix('relsgg-')+'/'+r['backend']+'/'+r['dtype'] for r in rows])
    fig.colorbar(plot,ax=ax,label='log10(RMSE), matched valid pairs')
    fig.savefig(folder/'stage_errors.png',dpi=160);plt.close(fig)
    lines=['# Full graph intermediate validation','',
      f'{len(stages)} boundaries are compared against independent upstream PyTorch traces.','The trace uses the first retained real image and all its GT boxes. Each run records its exact attention and precision options in run_provenance.json. Strict F32 uses explicit attention and preserves F32 matrix operands. Stage matrices compare matching subject/object IDs, excluding invalid slots.','',
      '![Intermediate error by stage](stage_errors.png)','',
      '| Model | Backend/storage | Fused map RMSE | Transformer RMSE | Deformable RMSE | Final RMSE | Final max abs | Custom vocabulary RMSE |',
      '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        st=r['stages']; f=r['final']; d=r.get('dynamic_vocabulary',{})
        lines.append(f'| {r["model"]} | {r["backend"]}/{r["dtype"]} | {st["fused_patch_map"]["rmse"]:.6g} | {st["rel_transformer"]["rmse"]:.6g} | {st["deformable_read"]["rmse"]:.6g} | {f["logits_rmse"]:.6g} | {f["logits_max_abs"]:.6g} | {d.get("logits_rmse",float("nan")):.6g} |')
    lines += ['', 'Each model/backend directory contains `stages/comparison.json` with every measured boundary and `dynamic/comparison.json`. Dynamic tests use five nonadjacent bank rows with non-unit scales plus an unseen composite direction. The independent Python API performs its own normalization and gate computation.', '',
      '## Dense candidate selection', '', 'A real retained image with 60 deterministic distinct boxes exercises 3,540 valid ordered pairs, geometry TopK=400 and relation TopK=128. Generated boxes have no task GT; this is a numerical sampler test. Close quantized scores can change the candidate set, which in turn changes contextual attention for otherwise matching pairs. All errors and pair changes are retained.', '',
      '| Model | Backend/storage | Pair recall | Predicate agreement | Logit RMSE |', '|---|---|---:|---:|---:|']
    dense=args.dense or folder/'dense_sampler/results.json'
    if not dense.exists():dense=folder.parent/'dense_sampler/results.json'
    if dense.exists():
        dense_rows=json.loads(dense.read_text())
        reference_devices=sorted({r.get('reference_device','cpu') for r in dense_rows})
        lines.insert(lines.index('## Dense candidate selection')+2,
                     'Independent official reference device: '+', '.join(reference_devices)+'.')
        for r in dense_rows:lines.append(f'| {r["model"]} | {r["backend"]}/{r["dtype"]} | {r["pair_set_recall"]:.2%} | {r["predicate_top1_agreement"]:.2%} | {r["logits_rmse"]:.6g} |')
        if args.geometry_baseline:
            previous=json.loads(args.geometry_baseline.read_text())
            keyed={(r['model'],r['backend'],r['dtype']):r for r in dense_rows}
            compared=[(old,keyed[(old['model'],old['backend'],old['dtype'])]) for old in previous]
            fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
            x=np.arange(len(compared))
            for i,(label,color) in enumerate((('Vulkan GEMM','#777777'),('F32 products','#087e8b'))):
                axes[0].bar(x+(i-.5)*.36,[pair[i]['logits_rmse'] for pair in compared],.36,label=label,color=color)
                axes[1].bar(x+(i-.5)*.36,[pair[i]['matched_pairs'] for pair in compared],.36,color=color)
            for ax in axes:
                ax.set_xticks(x,[a['model'].removeprefix('relsgg-')+'/'+a['dtype'] for a,b in compared],rotation=25)
                ax.grid(axis='y',alpha=.2)
            axes[0].set_ylabel('Matched-pair logit RMSE');axes[0].legend()
            axes[1].set_ylabel('Selected pairs matching Python / 128');axes[1].set_ylim(124,128.5)
            fig.savefig(folder/'dense_geometry.png',dpi=160);plt.close(fig)
            lines += ['', '### Geometry scorer regression', '',
                      'The same binary is run with --matmul-geometry and with deployment defaults. '
                      'Only the Vulkan geometry MLP changes: F32 products/reduction avoid a close-score Top400 swap. '
                      'The official 400 then 128 budgets and model weights are unchanged.', '',
                      '![Dense geometry regression](dense_geometry.png)', '',
                      '| Model/storage | Historical / corrected matched pairs | Historical / corrected logit RMSE |',
                      '|---|---:|---:|']
            for a,b in compared:
                lines.append(f'| {a["model"]}/{a["dtype"]} | {a["matched_pairs"]} / {b["matched_pairs"]} | {a["logits_rmse"]:.6g} / {b["logits_rmse"]:.6g} |')
    lines += ['', '## Input and output contracts', '',
      'The runtime checks all 50 preprocessed image/box arrays for bit equality, exercises 0/1/2/60 objects on CPU/CUDA/Vulkan, and rejects malformed dimensions. The image benchmark directory contains `contracts-*.json` and `preprocessing_parity.json`. Native top-predicate and calibrated scores are independently compared to raw logits for every image in the full benchmark.', '',
      '## Reproduce', '', '```sh',
      'python3 cpp_ggml/scripts/validate_full_graph.py --backends cpu cuda vulkan --dtypes f32 f16 q8_0 --dynamic',
      'python3 cpp_ggml/scripts/validate_dense_sampler.py',
      'python3 cpp_ggml/scripts/report_full_validation.py', '```', '',
      'These are observed numerical differences, not a claim of bit identity between different kernels or quantized weights. Full-dataset GT metrics and speed are in [the current image report]('+os.path.relpath(args.benchmark/'README.md',folder)+').']
    (folder/'README.md').write_text('\n'.join(lines)+'\n')
    print(f'{len(rows)} stage/dynamic configurations summarized')
if __name__=='__main__':main()
