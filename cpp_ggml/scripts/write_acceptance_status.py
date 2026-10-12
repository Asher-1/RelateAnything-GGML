#!/usr/bin/env python3
"""Write a machine-readable status for the seven acceptance boundaries.

The report deliberately distinguishes execution evidence from protocol assets.
Inference outputs cannot manufacture missing labels, detector predictions or
held-out calibration pairs.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

MODELS = ('relsgg-vits16', 'relsgg-vits16plus', 'relsgg-vitb16')
DTYPES = ('f32', 'f16', 'q8_0')


def acceptance(run):
    run = Path(run).resolve()
    full = run/'artifacts/full_graph'
    summary = json.loads((full/'summary.json').read_text())
    models = summary.get('models', {})
    record = json.loads((run/'run.json').read_text())
    limits = {'f32': (0.005, 0.995, 0.995),
              'f16': (0.010, 0.990, 0.990),
              'q8_0': (0.050, 0.970, 0.980)}
    expected = {(model, f'{backend}-{dtype}') for model in MODELS
                for backend in ('cpu', 'cuda', 'vulkan') for dtype in DTYPES}
    present = {(model, config) for model, values in models.items()
               for config in values if config != 'python' and not config.endswith('-f32-input')}
    numeric_failures = []
    performance_failures = []
    precision_failures = []
    for model, config in sorted(expected & present):
        row = models[model][config]
        dtype = row['dtype']
        rmse, top1, jaccard = limits[dtype]
        if (row.get('logits_rmse', float('inf')) > rmse or
                row.get('predicate_top1_agreement', 0) < top1 or
                row.get('pair_set_jaccard', 0) < jaccard):
            numeric_failures.append({
                'model': model, 'config': config,
                'logits_rmse': row.get('logits_rmse'),
                'predicate_top1_agreement': row.get('predicate_top1_agreement'),
                'pair_set_jaccard': row.get('pair_set_jaccard')})
        backend = row['backend']
        if backend in ('cuda', 'vulkan'):
            python_ms = models[model]['python']['latency_median_ms']
            if row.get('latency_median_ms', float('inf')) > python_ms:
                performance_failures.append({
                    'model': model, 'config': config,
                    'native_ms': row.get('latency_median_ms'), 'python_ms': python_ms})
        result_path = full/model/config/'results.json'
        effective = json.loads(result_path.read_text()) if result_path.exists() else row
        expected_strict = backend == 'cuda' and dtype == 'f32'
        if (effective.get('strict_f32_request') != 'auto' or
                effective.get('strict_f32') is not expected_strict):
            precision_failures.append({
                'model': model, 'config': config,
                'request': effective.get('strict_f32_request'),
                'effective': effective.get('strict_f32'),
                'expected_effective': expected_strict})
    complete = present == expected and set(models) == set(MODELS)
    g1_passed = len(expected) - len(numeric_failures) if complete else 0
    gpu_total = len(MODELS) * 2 * len(DTYPES)
    g2_passed = gpu_total - len(performance_failures) if complete else 0
    known_deviation = (len(performance_failures) == 1 and
                       performance_failures[0]['model'] == 'relsgg-vitb16' and
                       performance_failures[0]['config'] == 'cuda-f32')
    g1_complete = complete and not numeric_failures
    auto_complete = complete and record.get('config', {}).get('precision_request') == 'auto' and not precision_failures
    if complete and not performance_failures:
        g2_status = 'complete'
        decision = 'complete' if g1_complete and auto_complete else 'blocked'
    elif complete and g1_complete and auto_complete and known_deviation:
        g2_status = 'accepted_deviation'
        decision = 'accepted_with_one_performance_deviation'
    else:
        g2_status = 'open' if complete else 'pending'
        decision = 'blocked'
    statuses = [
        {'id': 'g1_numeric', 'status': 'complete' if g1_complete else ('open' if complete else 'pending'),
         'passed': g1_passed, 'total': len(expected), 'failed': numeric_failures,
         'evidence': str(full/'summary.json')},
        {'id': 'g2_gpu_performance', 'status': g2_status,
         'passed': g2_passed, 'total': gpu_total, 'failed': performance_failures,
         'evidence': str(full/'summary.json')},
        {'id': 'auto_precision', 'status': 'complete' if auto_complete else ('open' if complete else 'pending'),
         'passed': len(expected) - len(precision_failures) if complete else 0,
         'total': len(expected), 'failed': precision_failures,
         'evidence': str(full/'summary.json')},
    ]
    return {'schema': 2, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'run_id': run.name, 'execution': {'status': record.get('status'),
                                              'primary_rows': len(present)},
            'release_decision': decision, 'statuses': statuses,
            'interpretation': ('accepted_deviation preserves the measured failure; '
                               'it records an explicit precision-first release decision')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    report = acceptance(args.run)
    artifacts = args.run.resolve()/'artifacts'
    out = artifacts/'acceptance_status.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(out)


if __name__ == '__main__':
    main()
