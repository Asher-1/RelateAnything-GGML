#!/usr/bin/env python3
"""Write a machine-readable status for the seven acceptance boundaries.

The report deliberately distinguishes execution evidence from protocol assets.
Inference outputs cannot manufacture missing labels, detector predictions or
held-out calibration pairs.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path

MODELS = ('relsgg-vits16', 'relsgg-vits16plus', 'relsgg-vitb16',
          'relsgg-convnext-tiny', 'relsgg-convnext-small', 'relsgg-convnext-base', 'relsgg-convnext-large')
DTYPES = ('f32', 'f16', 'q8_0')
# G2 separates two questions the raw native_ms > python_ms comparison merges:
# is the gap real (resolved against the reference's own drift inside the run
# window), and is it large enough to block a release. MATERIALITY is the
# release policy (decided 20261025); a gap must be confidently above it to
# block. Six-run backtest: it blocks every run whose failures were judged
# real by hand (vulkan slow mode, 1014 tiny vulkan, 1018 fast-mode tiny/small)
# and passes 20261024-current-r2 without any hand-maintained accept list.
MATERIALITY = 0.05
BOOTSTRAP = 2000


def paired_ratio(native_rows, python_rows):
    """Median of per-image native/python ratios and its bootstrap 95% lower
    bound. Pairing on the same image removes image-difficulty spread, which a
    ratio of two medians mixes in. A fixed seed per row keeps the verdict a
    pure function of the measurements."""
    python = {row['sample']: row['latency_ms'] for row in python_rows}
    logs = [math.log(row['latency_ms'] / python[row['sample']])
            for row in native_rows if row['sample'] in python]
    if not logs:
        return None, None
    rng = random.Random(0)
    medians = sorted(statistics.median(rng.choices(logs, k=len(logs))) for _ in range(BOOTSTRAP))
    return math.exp(statistics.median(logs)), math.exp(medians[int(0.025 * BOOTSTRAP)])


def classify_failure(full, failure):
    """block: confidently slower than the reference by more than MATERIALITY.
    inconclusive: per-image evidence is missing or the reference itself moved
    by more than MATERIALITY inside the run window, so no gap of policy size
    can be resolved; the row needs a rerun rather than a pass. tracked: real
    or not, the gap is below MATERIALITY; it stays visible in `failed` and in
    cross-run persistence."""
    model, config = failure['model'], failure['config']
    native_path = full/model/config/'results.json'
    python_path = full/model/'python'/'results.json'
    record = dict(failure)
    if not native_path.exists() or not python_path.exists():
        return record | {'verdict': 'inconclusive', 'reason': 'missing per-image results'}
    python = json.loads(python_path.read_text())
    drift = abs((python.get('timing_bracket') or {}).get('relative_drift', float('inf')))
    ratio, lower = paired_ratio(json.loads(native_path.read_text())['rows'], python['rows'])
    record |= {'paired_ratio': ratio, 'ratio_ci_lower': lower, 'python_drift': drift}
    if ratio is None:
        return record | {'verdict': 'inconclusive', 'reason': 'no paired images'}
    if not drift <= MATERIALITY:
        return record | {'verdict': 'inconclusive', 'reason': 'reference drift above materiality'}
    if lower > 1 + MATERIALITY:
        return record | {'verdict': 'block', 'reason': 'confidently slower than materiality'}
    return record | {'verdict': 'tracked', 'reason': 'below materiality'}


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
    classified = [classify_failure(full, failure) for failure in performance_failures]
    blocking = [row for row in classified if row['verdict'] == 'block']
    inconclusive = [row for row in classified if row['verdict'] == 'inconclusive']
    tracked = [row for row in classified if row['verdict'] == 'tracked']
    g1_complete = complete and not numeric_failures
    auto_complete = complete and record.get('config', {}).get('precision_request') == 'auto' and not precision_failures
    if complete and not performance_failures:
        g2_status = 'complete'
        decision = 'complete' if g1_complete and auto_complete else 'blocked'
    elif complete and g1_complete and auto_complete and not blocking and not inconclusive:
        g2_status = 'accepted_deviation'
        decision = 'accepted_with_performance_deviations'
    elif complete and g1_complete and auto_complete and not blocking:
        g2_status = 'rerun_required'
        decision = 'rerun_required'
    else:
        g2_status = 'open' if complete else 'pending'
        decision = 'blocked'
    statuses = [
        {'id': 'g1_numeric', 'status': 'complete' if g1_complete else ('open' if complete else 'pending'),
         'passed': g1_passed, 'total': len(expected), 'failed': numeric_failures,
         'evidence': str(full/'summary.json')},
        {'id': 'g2_gpu_performance', 'status': g2_status,
         'passed': g2_passed, 'total': gpu_total, 'failed': performance_failures,
         'materiality': MATERIALITY, 'classified': classified,
         'accepted': tracked, 'blocking': blocking, 'inconclusive': inconclusive,
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
            'interpretation': ('failed keeps every raw native_ms > python_ms row; each is '
                               'classified by its paired per-image ratio: block when the '
                               'bootstrap lower bound exceeds 1+materiality, inconclusive '
                               'when evidence is missing or the python reference drifted '
                               'beyond materiality, otherwise tracked (accepted)')}


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
