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
    artifacts = run/'artifacts'
    full = artifacts/'full_graph'
    summary = full/'summary.json'
    rows = json.loads(summary.read_text()) if summary.exists() else {}
    models = rows.get('models', {})
    strict_rows = [v for values in models.values() for k, v in values.items()
                   if k not in ('python',) and isinstance(v, dict)]
    oracle_file = full/'quantization_audit/summary.json'
    oracles = json.loads(oracle_file.read_text()) if oracle_file.exists() else []
    bit_rows = [row.get(backend+'_same_weights', {}) for row in oracles
                for backend in ('cpu','cuda','vulkan')]
    original_file=artifacts/'gallery/numerical_first_divergence.json'
    originals=json.loads(original_file.read_text()) if original_file.exists() else []
    expected_native={backend+'-'+dtype for backend in ('cpu','cuda','vulkan') for dtype in DTYPES} | {'cuda-q8_0-f32-input'}
    bit_complete=({(r['model'],r['config']) for r in originals}==
                  {(m,k) for m in MODELS for k in expected_native} and
                  all(all(key in r for key in ('bit_equal_fraction','pair_logit_bit_equal_fraction','pair_set_jaccard')) for r in originals))
    bit_equal=bit_complete and all(r['bit_equal_fraction']==1.0 and
        r['pair_logit_bit_equal_fraction']==1.0 and r['pair_set_jaccard']==1.0 for r in originals)
    python_rows = [v for values in models.values() for k, v in values.items()
                   if k == 'python' and isinstance(v, dict)]
    speed_rows = [v for values in models.values() for k, v in values.items()
                  if k.startswith(('cpu-', 'cuda-', 'vulkan-')) and isinstance(v, dict)]
    expected_configs = {backend+'-'+dtype for backend in ('cpu','cuda','vulkan')
                        for dtype in DTYPES} | {'cuda-q8_0-f32-input', 'python'}
    speed_complete = (set(models) == set(MODELS) and
                      all(set(values) == expected_configs
                          for values in models.values()))
    slower_rows = [
        {'model': model, 'config': config,
         'native_ms': row.get('latency_median_ms'),
         'python_ms': values.get('python', {}).get('latency_median_ms')}
        for model, values in models.items() for config, row in values.items()
        if config.startswith(('cpu-', 'cuda-', 'vulkan-')) and isinstance(row, dict) and
        row.get('latency_median_ms', float('inf')) >=
        values.get('python', {}).get('latency_median_ms', 0)]
    speed_faster = speed_complete and not slower_rows

    statuses = [
        {'id': 'strict_bit_equality',
         'status': 'complete' if bit_equal else ('open' if bit_complete else 'pending'),
         'reason': ('all native predicate logits, pair logits and candidate sets equal original PyTorch-CUDA' if bit_equal else
                    'original PyTorch-CUDA comparisons retain arithmetic/storage residuals; same-weight oracles diagnose them separately'
                    if bit_complete else 'the complete original PyTorch-CUDA bit-comparison matrix is not available'),
         'evidence': str(original_file)},
        {'id': 'strict_faster_than_python',
         'status': 'complete' if speed_faster else ('open' if speed_complete else 'pending'),
         'reason': ('every measured strict row beats its paired Python median' if speed_faster else
                    'the complete matrix is measured, but at least one strict row remains slower'
                    if speed_complete else 'the complete paired timing matrix is not available'),
         'evidence': str(full/'summary.json')},
        {'id': 'soft_open_vocab', 'status': 'missing_asset',
         'reason': 'labelled synonym/inverse corpus, ontology and text-space calibration are absent',
         'evidence': 'upstream benchmark/calibrate_match_tau.py contract'},
        {'id': 'haystack_precision', 'status': 'missing_asset',
         'reason': 'adjudicated positive/negative judgment pack and split are unavailable',
         'evidence': 'benchmark/SPEC.md and benchmark/eval_haystack.py'},
        {'id': 'sgdet_detector_metrics', 'status': 'missing_asset',
         'reason': 'fixed detector outputs and operating point are unavailable',
         'evidence': 'benchmark/SPEC.md'},
        {'id': 'judge_hico_ovs', 'status': 'missing_asset',
         'reason': 'held-out pairings, judge controls and six-axis normalizers are unavailable',
         'evidence': 'benchmark/SPEC.md'},
        {'id': 'ece_brier', 'status': 'missing_asset',
         'reason': 'label-complete held-out deployment calibration distribution is unavailable',
         'evidence': 'benchmark/eval_operating_point.py'},
    ]
    return {'schema': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'run_id': run.name, 'execution': {'status': json.loads((run/'run.json').read_text()).get('status'),
                                                 'strict_rows': len(strict_rows),
                                                 'speed_rows': len(speed_rows)},
              'statuses': statuses,
              'details': {
                  'exact_precision_and_speed':{'measured_native_configs':len(originals),
                      'passing_native_configs':[{'model':r['model'],'config':r['config']} for r in originals
                        if r.get('bit_equal_fraction')==1 and r.get('pair_logit_bit_equal_fraction')==1
                        and r.get('pair_set_jaccard')==1
                        and models.get(r['model'],{}).get(r['config'],{}).get('latency_median_ms',float('inf'))
                          < models.get(r['model'],{}).get('python',{}).get('latency_median_ms',0)]},
                  'strict_bit_equality': {
                      'original_cuda_comparisons': len(originals),
                      'original_cuda_minimum_logit_bit_equal_fraction':min((r.get('bit_equal_fraction',0) for r in originals),default=0),
                      'same_weight_oracle_configs': len(oracles),
                      'backend_comparisons': len(bit_rows),
                      'minimum_logit_bit_equal_fraction': min(
                          (value.get('logit_bit_equal_fraction', 0.0) for value in bit_rows),
                          default=0.0),
                      'pair_set_equality_is_reported_separately': True},
                  'strict_faster_than_python': {
                      'native_rows': len(speed_rows), 'python_rows': len(python_rows),
                      'slower_or_equal_rows': slower_rows}},
              'interpretation': 'missing_asset is a protocol boundary, not a failed inference run'}


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
