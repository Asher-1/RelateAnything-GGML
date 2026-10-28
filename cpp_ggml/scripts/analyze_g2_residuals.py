#!/usr/bin/env python3
"""Classify residual G2 performance failures after a release run.

Compares a candidate run against a baseline run on the 42 GPU (cuda/vulkan)
rows of the G2 gate. The ViT towers act as an environment control group: the
GELU-epilogue fusion only touches the ConvNeXt blocks, so the python-bracket
drift measured on ViT rows estimates the shared-GPU noise epsilon. Each
baseline failure is then classified as cleared / noise-band / structural
using that control, and the verdict on the zero-tolerance gate is reported
alongside the drift caveat.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = ('relsgg-vits16', 'relsgg-vits16plus', 'relsgg-vitb16',
          'relsgg-convnext-tiny', 'relsgg-convnext-small',
          'relsgg-convnext-base', 'relsgg-convnext-large')
# Baseline (20261014) failures pending a data-driven verdict. The six
# ConvNeXt vulkan rows are the shader-ceiling group, tracked separately.
PENDING = {
    'structural_cuda': [
        ('relsgg-convnext-small', 'cuda-f32'),
        ('relsgg-convnext-base', 'cuda-f32'),
        ('relsgg-convnext-large', 'cuda-f32'),
        ('relsgg-convnext-large', 'cuda-f16')],
    'noise_vit': [
        ('relsgg-vits16', 'cuda-f32'),
        ('relsgg-vits16plus', 'cuda-f32'),
        ('relsgg-vits16', 'vulkan-f32')],
}
VIT = ('relsgg-vits16', 'relsgg-vits16plus', 'relsgg-vitb16')


def load(run: Path) -> dict:
    summary = json.loads((run / 'artifacts/full_graph/summary.json').read_text())
    rows = {}
    for model, configs in summary['models'].items():
        bracket = configs.get('python', {}).get('latency_median_ms')
        for name, row in configs.items():
            if name == 'python' or name.endswith('-f32-input'):
                continue
            if row['backend'] in ('cuda', 'vulkan'):
                rows[(model, name)] = (row.get('latency_median_ms'), bracket)
    return rows


def drift(current: dict, baseline: dict) -> float:
    """Median python-bracket drift on ViT rows = shared-GPU noise epsilon."""
    values = []
    for model in VIT:
        _, new = current.get((model, 'cuda-f32'), (None, None))
        _, old = baseline.get((model, 'cuda-f32'), (None, None))
        if new and old:
            values.append(new / old - 1.0)
    values.sort()
    return values[len(values) // 2] if values else 0.0


def classify(model: str, config: str, current: dict, baseline: dict,
             epsilon: float, persistent: set) -> dict:
    new_ms, new_py = current.get((model, config), (None, None))
    old_ms, old_py = baseline.get((model, config), (None, None))
    record = {'model': model, 'config': config,
              'native_ms': new_ms, 'python_ms': new_py,
              'baseline_native_ms': old_ms, 'baseline_python_ms': old_py}
    if not all(isinstance(v, float) for v in (new_ms, new_py, old_ms, old_py)):
        return record | {'verdict': 'missing_data'}
    record['ratio'] = new_ms / new_py
    record['baseline_ratio'] = old_ms / old_py
    record['native_drift'] = new_ms / old_ms - 1.0
    record['python_drift'] = new_py / old_py - 1.0
    if new_ms <= new_py:
        record['verdict'] = 'gate_pass'
    elif (model, config) in persistent:
        # Failing in every run is not noise, whatever the epsilon estimate says.
        record['verdict'] = 'persistent'
    elif record['native_drift'] <= epsilon:
        record['verdict'] = 'noise_band'
    else:
        record['verdict'] = 'structural'
    return record


def persistence(runs: list[dict]) -> dict:
    """Per-row ratio across runs, failing rows only. A row that fails in every
    run is persistent; sign consistency does not depend on any noise-epsilon
    estimate."""
    table = {}
    for key in sorted(set().union(*runs)):
        ratios = [ms / py for run in runs for ms, py in [run.get(key, (None, None))]
                  if isinstance(ms, float) and isinstance(py, float)]
        fails = sum(r > 1.0 for r in ratios)
        if fails:
            persistent = fails == len(ratios) and len(ratios) > 1
            table[key] = ('persistent' if persistent else 'intermittent', ratios)
    return table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--history', type=Path, nargs='*', default=[],
                        help='extra runs for the cross-run persistence table')
    args = parser.parse_args()
    current, baseline = load(args.candidate), load(args.baseline)
    epsilon = drift(current, baseline)
    runs = [baseline, *map(load, args.history), current]
    table = persistence(runs)
    persistent = {k for k, (label, _) in table.items() if label == 'persistent'}
    print(f'control-group python-bracket drift epsilon = {epsilon:+.3f}\n')
    gpu_rows = {k: v for k, v in current.items() if k[1].split("-")[0] in ('cuda', 'vulkan')}
    gate_failures = [(m, c) for (m, c), (ms, py) in sorted(gpu_rows.items())
                     if isinstance(ms, float) and isinstance(py, float) and ms > py]
    print(f'G2 zero-tolerance gate: {len(gpu_rows) - len(gate_failures)}/{len(gpu_rows)} pass'
          f' ({len(gate_failures)} fail) — measured under shared-GPU drift {epsilon:+.3f}\n')
    for group, keys in PENDING.items():
        print(f'[{group}]')
        for model, config in keys:
            row = classify(model, config, current, baseline, epsilon, persistent)
            print(json.dumps(row))
        print()
    print(f'[cross-run ratio, oldest -> candidate, {len(runs)} runs; failing rows only]')
    for (model, config), (label, ratios) in table.items():
        print(f'{model:24s} {config:11s} {label:12s} ' + ' '.join(f'{r:.3f}' for r in ratios))
    return 0


if __name__ == '__main__':
    main()
