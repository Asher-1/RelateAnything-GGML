#!/usr/bin/env python3
"""Interleave historical and compact native graphs on all retained images."""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np

from benchmark_full_graph import ROOT, MODELS, DTYPES, native_timing_run, save, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', type=Path, default=(ROOT/'cpp_ggml/benchmarks/current').resolve().parent/'artifacts/full_graph')
    p.add_argument('--output', type=Path)
    p.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    p.add_argument('--dtypes', nargs='+', choices=DTYPES, default=list(DTYPES))
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--ablation', choices=('deployment', 'geometry'), default='deployment')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--wait-gpu-idle', action='store_true')
    p.add_argument('--allow-gpu-pid', type=int, nargs='*', default=[])
    args = p.parse_args()
    if args.repeats < 1: p.error('--repeats must be positive')
    args.output = args.output or args.benchmark/'compaction_ablation'
    binary = ROOT/'cpp_ggml/build-vulkan/relateanything-ggml'
    summary_path = args.output/'summary.json'
    summary = json.loads(summary_path.read_text()) if args.resume and summary_path.exists() else []
    for model in args.models:
        for dtype in args.dtypes:
            if any(s['model'] == model and s['dtype'] == dtype for s in summary):
                continue
            times = {'historical': {}, 'compact': {}}
            predictions = {}
            # ABBA cancels first-order drift without timing two GPU jobs together.
            for index, mode in enumerate(('historical', 'compact', 'compact', 'historical')):
                dest = args.output/model/dtype/f'{index}-{mode}'
                dest.mkdir(parents=True, exist_ok=True)
                cmd = [str(binary), '--backend', 'vulkan', '--model',
                       str(ROOT/f'cpp_ggml/models/gguf/{model}-{dtype}.gguf'),
                       '--input', str(args.benchmark/'image_requests'), '--output', str(dest),
                       '--warmup', '2', '--repeats', str(args.repeats)]
                if mode == 'historical':
                    cmd += ['--matmul-geometry'] if args.ablation == 'geometry' else [
                        '--no-compact-pairs', '--matmul-sampler', '--matmul-geometry',
                        '--no-fused-vocabulary', '--no-cache-positions', '--no-retain-features']
                run = native_timing_run(args, cmd, dest)
                (dest/'run.log').write_text(run.stdout + run.stderr)
                if run.returncode: raise RuntimeError(run.stderr[-2000:])
                name = None
                for line in run.stdout.splitlines():
                    if line.startswith('sample='): name = line.split('=', 1)[1]
                    elif line.startswith('iteration_ms='):
                        times[mode].setdefault(name, []).extend(map(float, line.split('=', 1)[1].split(',')))
                predictions[mode] = {path.stem: json.loads(path.read_text())['predictions']
                                     for path in dest.glob('*.json') if path.stem[0].isdigit()}
            if len(times['compact']) != 50 or set(times['compact']) != set(times['historical']):
                raise RuntimeError('ablation does not cover the same 50 images')
            rows = []
            for name in sorted(times['compact']):
                old = predictions['historical'][name]; new = predictions['compact'][name]
                a = {(r['subject'], r['object']): r for r in old}
                b = {(r['subject'], r['object']): r for r in new}
                common = a.keys() & b.keys()
                rows.append({'sample': name,
                             'historical_ms': statistics.median(times['historical'][name]),
                             'compact_ms': statistics.median(times['compact'][name]),
                             'historical_samples_ms': times['historical'][name],
                             'compact_samples_ms': times['compact'][name],
                             'same_pairs': a.keys() == b.keys(),
                             'predicate_changes': sum(a[k]['predicate_id'] != b[k]['predicate_id'] for k in common),
                             'score_max_abs': max(abs(a[k]['score'] - b[k]['score']) for k in common)})
            historical = statistics.median(r['historical_ms'] for r in rows)
            compact = statistics.median(r['compact_ms'] for r in rows)
            entry = {'model': model, 'dtype': dtype, 'ablation': args.ablation, 'historical_ms': historical,
                     'compact_ms': compact, 'speedup': historical / compact,
                     'same_pairs_images': sum(r['same_pairs'] for r in rows),
                     'predicate_changes': sum(r['predicate_changes'] for r in rows)}
            save(args.output/model/dtype/'results.json',
                 {**entry, 'binary_sha256': sha256(binary), 'order': 'ABBA', 'rows': rows})
            summary.append(entry); save(args.output/'summary.json', summary)
            print(model, dtype, f'{historical:.2f} -> {compact:.2f} ms', flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5), layout='constrained')
    labels = [r['model'].removeprefix('relsgg-')+'/'+r['dtype'] for r in summary]
    x = np.arange(len(summary))
    ax.bar(x-.18, [r['historical_ms'] for r in summary], .36, label='Historical graph', color='#777777')
    ax.bar(x+.18, [r['compact_ms'] for r in summary], .36, label='Deployment graph', color='#087e8b')
    ax.set_xticks(x, labels, rotation=30, ha='right'); ax.set_ylabel('50-image median pipeline ms')
    ax.legend(); ax.grid(axis='y', alpha=.2); fig.savefig(args.output/'latency.png', dpi=160)
    plt.close(fig)
    lines = ['# Native graph '+args.ablation+' ablation', '',
             'Same current binary, all 50 images, Vulkan, two warmups, '+str(args.repeats)+
             ' repeats per image per pass, ABBA pass order. Historical uses '+
             ('--matmul-geometry' if args.ablation == 'geometry' else
              '--no-compact-pairs --matmul-sampler --matmul-geometry --no-fused-vocabulary --no-cache-positions --no-retain-features')+
             '; compact uses the deployment defaults. '
             'Persistent GPU services are allowed explicitly; transient overlaps are discarded.', '',
             '![Interleaved pipeline latency](latency.png)', '',
             '| Model | Storage | Historical ms | Compact ms | Speedup | Same-pair images | Predicate changes |',
             '|---|---|---:|---:|---:|---:|---:|']
    for s in summary:
        lines.append(f'| {s["model"]} | {s["dtype"]} | {s["historical_ms"]:.2f} | '
                     f'{s["compact_ms"]:.2f} | {s["speedup"]:.3f}x | '
                     f'{s["same_pairs_images"]}/50 | {s["predicate_changes"]} |')
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__': main()
