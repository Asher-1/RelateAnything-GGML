#!/usr/bin/env python3
"""Separate reference-device arithmetic from storage changes in saved oracles."""
import argparse
import json
from pathlib import Path

import numpy as np
from audit_quantized_graph import identity
from benchmark_full_graph import MODELS, DTYPES
from provenance import recorded_run, save, validate_run


def arrays(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True)
    args = parser.parse_args()
    audit = args.benchmark/'quantization_audit'
    rows = []
    inputs = []
    for model in MODELS:
        validate_run(args.benchmark/model/'python')
        inputs.extend([args.benchmark/model/'python/run_provenance.json',
                       args.benchmark/model/'python/vocabulary.json'])
        for dtype in DTYPES:
            folder = audit/model/dtype/'oracle'
            validate_run(folder)
            inputs.append(folder/'run_provenance.json')
    with recorded_run(audit/'components', inputs=inputs, config={'device': 'cpu'}, script=__file__):
        for model in MODELS:
            vocab = json.loads((args.benchmark/model/'python/vocabulary.json').read_text())
            for dtype in DTYPES:
                per_image = []
                for path in sorted((audit/model/dtype/'oracle').glob('*.npz')):
                    f32 = arrays(audit/model/'f32/oracle'/path.name)
                    current = arrays(path)
                    original = arrays(args.benchmark/model/'python'/path.name)
                    per_image.append({'sample': path.stem,
                        'reference_device_and_runtime': identity(original, f32, vocab['calibration']),
                        'storage_only_cpu': identity(f32, current, vocab['calibration'])})
                row = {'model': model, 'dtype': dtype, 'rows': per_image}
                for key in ('reference_device_and_runtime', 'storage_only_cpu'):
                    row[key] = {metric: float(np.mean([r[key][metric] for r in per_image]))
                                for metric in per_image[0][key]}
                rows.append(row)
        save(audit/'components/results.json', rows)
    lines = ['# Reference device and storage components', '',
        'The original benchmark reference uses PyTorch-CUDA. Same-weight oracles '
        'use PyTorch-CPU. Their difference includes reference-device arithmetic '
        'and runtime vocabulary preparation. It is not entirely weight loss. '
        'The storage-only comparison below instead uses CPU F32 GGUF weights '
        'against CPU F16/Q8 GGUF weights with the same graph and device.', '',
        '| Model/storage | CUDA reference to CPU F32 RMSE | CPU storage-only RMSE | CPU storage top-1 |',
        '|---|---:|---:|---:|']
    for row in rows:
        a = row['reference_device_and_runtime']; b = row['storage_only_cpu']
        lines.append(f'| {row["model"]}/{row["dtype"]} | {a["logits_rmse"]:.6g} | '
                     f'{b["logits_rmse"]:.6g} | {b["predicate_top1_agreement"]:.3%} |')
    lines += ['', 'Candidate changes and calibrated-score errors are retained '
              'per image in [results.json](components/results.json). Native arithmetic '
              'on each same-weight oracle is in [the primary audit](README.md).']
    (audit/'COMPONENTS.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
