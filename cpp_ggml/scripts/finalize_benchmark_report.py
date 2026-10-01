#!/usr/bin/env python3
"""Derive precision descriptions from saved runs without changing their producer."""
import argparse
import json
import os
from pathlib import Path

from provenance import ROOT, validate_run


def quantization_report(folder):
    summary = folder/'summary.json'
    if not summary.exists():
        return
    rows = json.loads(summary.read_text())
    lines = ['# Stored-Weight and Native Arithmetic Comparison', '',
        'Every retained image is evaluated by the official Python-CPU graph '
        'with its GGUF tensors restored. The original baseline uses Python-CUDA. '
        'Their difference includes device/runtime arithmetic as well as storage '
        'changes. It is not entirely weight rounding; the separate primary '
        'COMPONENTS.md report compares CPU F32 against CPU F16/Q8 directly.', '',
        '![Measured error components](error_sources.png)', '',
        '| Model/storage | Changed values | CUDA reference to CPU oracle RMSE | CPU same-weight RMSE | CUDA same-weight RMSE | Vulkan same-weight RMSE |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["model"]}/{row["dtype"]} | {row["changed_weight_values"]:,} | '
                     f'{row["weight_conversion"]["logits_rmse"]:.6g} | '
                     f'{row["cpu_same_weights"]["logits_rmse"]:.6g} | '
                     f'{row["cuda_same_weights"]["logits_rmse"]:.6g} | '
                     f'{row["vulkan_same_weights"]["logits_rmse"]:.6g} |')
    lines += ['', 'F32 stored weights preserve the prepared runtime values exactly. '
              'Different arithmetic engines still have nonzero residuals. F16/Q8 '
              'discard original F32 information. Candidate sets, IEEE-754 equality, '
              'top-1 margins and stable-winner checks remain in the machine results.', '',
              '## Same-Weight Task Metrics', '',
              '| Model/storage | Oracle R@50 | mR@50 | F1@50 | wR@50 | CPU / CUDA / Vulkan calibrated score RMSE |',
              '|---|---:|---:|---:|---:|---:|']
    worst = []
    for row in rows:
        metrics = row['oracle_metrics']
        lines.append(f'| {row["model"]}/{row["dtype"]} | {metrics["R@50"]:.3%} | '
                     f'{metrics["mR@50"]:.3%} | {metrics["F1@50"]:.3%} | '
                     f'{metrics["wR@50"]:.3%} | '
                     f'{row["cpu_same_weights"]["calibrated_score_rmse"]:.6g} / '
                     f'{row["cuda_same_weights"]["calibrated_score_rmse"]:.6g} / '
                     f'{row["vulkan_same_weights"]["calibrated_score_rmse"]:.6g} |')
        for backend in ('cpu', 'cuda', 'vulkan'):
            bad = row['worst_images'][backend+'_same_weights'][0]
            worst.append(f'- {row["model"]}/{row["dtype"]}/{backend}: '
                         f'`{bad["sample"]}`, RMSE {bad["logits_rmse"]:.6g}, '
                         f'pair recall {bad["pair_set_recall"]:.3%}.')
    lines += ['', '## Worst Images', '', *worst, '',
              '[Complete summary](summary.json) includes every candidate change '
              'and the worst five images per component. Raw arrays and per-image '
              'comparisons remain beside the independent oracle runs.']
    (folder/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True)
    folder = parser.parse_args().benchmark
    native = []
    for path in sorted(folder.glob('relsgg-*/*/results.json')):
        validate_run(path.parent)
        result = json.loads(path.read_text())
        if path.parent.name != 'python':
            native.append(result)
    if not native:
        raise SystemExit('no verified native runs')
    strict = {row.get('strict_f32', False) for row in native}
    if len(strict) != 1:
        raise SystemExit('mixed strict/default runs require separate reports')
    is_strict = strict.pop()
    mode = ('All native rows use --strict-f32: F32 matrix operands retain F32 '
            'backend precision and attention is explicit. F16/Q8 weight storage '
            'still loses information; strict mode alone does not remove Q8 '
            'activation quantization. CUDA q8_0-f32-input is the separate F32 '
            'activation diagnostic.' if is_strict else
            'Native rows use the default matrix precision. Attention mode is '
            'recorded in each results.json; flash attention uses F16 K/V and '
            'F32 accumulation. F32/F16/Q8 name weight storage.')
    path = folder/'README.md'
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if line.startswith('Accelerated attention uses ') or line.startswith('All native rows use --strict-f32:') or line.startswith('Native rows use the default matrix precision.'):
            lines[index] = mode + ' Timing attempts retain sampled GPU processes, allowed persistent PIDs and discarded attempts. These shared-workstation measurements are not isolated-device guarantees.'
        if 'benchmark_full_graph.py --stage all ' in line:
            if is_strict and '--strict-f32' not in line:
                lines[index] += ' --strict-f32'
    if not any(line.startswith(mode) for line in lines):
        raise SystemExit('expected report precision paragraph not found')
    marker = '## Current Acceptance Scope'
    status_link = Path(os.path.relpath(ROOT/'cpp_ggml/benchmarks/ALIGNMENT_STATUS.md',
                                      folder.resolve())).as_posix()
    if marker not in lines:
        lines += ['', marker, '',
                  'The release audit verifies execution identities and raw/decoded '
                  'consistency. It does not certify bit equality, a speed advantage '
                  'for every row, or the unavailable six-axis benchmark. See '
                  f'[current alignment status]({status_link}).', '',
                  '[Sampled process memory and slower images](PROCESS_MEASUREMENTS.md). '
                  '[Reference-device and storage error components](quantization_audit/COMPONENTS.md).']
    text = '\n'.join(lines)+'\n'
    text = text.replace('[current alignment status](../../ALIGNMENT_STATUS.md)',
                        f'[current alignment status]({status_link})')
    path.write_text(text)
    for name in ('quantization_audit', 'quantization_audit_f32_input'):
        quantization_report(folder/name)
    print('Precision description derived from', len(native), 'verified native rows')


if __name__ == '__main__':
    main()
