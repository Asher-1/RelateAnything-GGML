#!/usr/bin/env python3
"""One immutable output tree for the native/Python alignment acceptance suites."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from benchmark_layout import OUTPUTS, allocate, now, publish_reports, set_current
from provenance import ROOT, build_fingerprint, fingerprint, save

MODELS = ('relsgg-vits16', 'relsgg-vits16plus', 'relsgg-vitb16')
SCRIPTS = ROOT/'cpp_ggml/scripts'
SUITES = ('full', 'stages', 'dense', 'interfaces', 'contracts', 'metrics')


def execute(script, arguments, folder, name, record):
    command = [sys.executable, str(SCRIPTS/script), *map(str, arguments)]
    log = folder/'logs'/f'{name}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    entry = {'name': name, 'command': command, 'status': 'running', 'started_utc': now()}
    record['steps'].append(entry)
    save(folder/'run.json', record)
    with log.open('w') as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, env={**os.environ, 'OPENBLAS_NUM_THREADS': '1',
                                                   'OMP_NUM_THREADS': '4'})
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                stream.write(line)
                stream.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    entry.update(status='complete' if code == 0 else 'failed', returncode=code, finished_utc=now())
    save(folder/'run.json', record)
    if code:
        raise RuntimeError(f'{name} failed with exit code {code}; see {log}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id')
    parser.add_argument('--root', type=Path, default=OUTPUTS)
    parser.add_argument('--suites', nargs='+', choices=SUITES, default=list(SUITES))
    parser.add_argument('--backends', nargs='+', choices=('cpu', 'cuda', 'vulkan'),
                        default=['cpu', 'cuda', 'vulkan'])
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--current', action='store_true', help='advance report view after all selected suites succeed')
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error('invalid warmup/repeats')
    if 'full' in args.suites and not {'cuda', 'vulkan'}.issubset(args.backends):
        parser.error('the full release protocol requires CUDA and Vulkan')
    if args.current and (set(args.suites) != set(SUITES) or
                         set(args.backends) != {'cpu', 'cuda', 'vulkan'}):
        parser.error('--current requires every suite and all three backends')
    folder = allocate(args.run_id, root=args.root)
    artifacts = folder/'artifacts'
    artifacts.mkdir()
    record = json.loads((folder/'run.json').read_text())
    record.update(status='running', started_utc=now(), steps=[], config={
        'suites': args.suites, 'backends': args.backends,
        'strict_f32': True, 'warmup': args.warmup, 'repeats': args.repeats},
        orchestrator=fingerprint([Path(__file__), SCRIPTS/'benchmark_layout.py']))
    save(folder/'run.json', record)
    print(f'run={folder}', flush=True)
    def run(script, arguments, name):
        execute(script, arguments, folder, name, record)
    try:
        record['builds'] = {backend: build_fingerprint(ROOT/f'cpp_ggml/build-{backend}/relateanything-ggml', backend)
                            for backend in args.backends}
        save(folder/'run.json', record)
        run('audit_ggml_patch_state.py', ['--out', artifacts/'ggml_patch_audit.json'], 'patch-state')
        run('verify_patch_replay.py', ['--output', artifacts/'patch_replay.json'], 'patch-replay')
        run('audit_runtime_options.py', ['--output', artifacts/'runtime_options_audit.json'], 'runtime-options-audit')
        full = artifacts/'full_graph'
        timing = ['--strict-f32', '--warmup', args.warmup, '--repeats', args.repeats,
                  '--wait-gpu-idle']
        if 'full' in args.suites:
            run('benchmark_full_graph.py', ['--stage', 'all', '--output', full,
                                           '--bracket-python', '--backends', *args.backends, *timing], 'full-primary')
            run('benchmark_full_graph.py', ['--stage', 'ggml', '--output', full,
                                           '--backends', 'cuda', '--dtypes', 'q8_0',
                                           '--quant-f32-input', *timing], 'full-q8-f32-input')
            run('audit_quantized_graph.py', ['--benchmark', full, '--device', 'cpu'], 'stored-weight-oracles')
            run('audit_quantized_graph.py', ['--benchmark', full, '--device', 'cpu',
                                            '--output', full/'quantization_audit_f32_input',
                                            '--dtypes', 'q8_0', '--cuda-variant=-f32-input',
                                            '--reuse-oracles', '--oracle-folder', full/'quantization_audit'],
                'q8-f32-input-oracles')
            run('report_quantization_components.py', ['--benchmark', full], 'quantization-components')
            run('verify_benchmark_release.py', ['--folder', full], 'release-audit')
            run('benchmark_full_graph.py', ['--stage', 'report', '--output', full,
                                           '--strict-f32', '--warmup', args.warmup,
                                           '--repeats', args.repeats], 'full-report')
            run('report_alignment_gallery.py', ['--benchmark', full, '--output', artifacts/'gallery'], 'detailed-gallery')
            run('report_process_measurements.py', ['--benchmark', full], 'process-measurements')
            run('finalize_benchmark_report.py', ['--benchmark', full], 'precision-report')
        if 'stages' in args.suites or 'contracts' in args.suites:
            if not (full/'inputs').exists():
                run('benchmark_full_graph.py', ['--stage', 'prepare', '--output', full], 'prepare-inputs')
        if 'stages' in args.suites:
            traces = artifacts/'traces_cpu'
            traces.mkdir()
            for model in MODELS:
                stem = 'python_stage_trace_'+model.removeprefix('relsgg-')
                run('trace_python_stages.py', ['--model-name', model, '--device', 'cpu',
                                              '--output', traces/(stem+'.json'),
                                              '--tensors', traces/(stem+'.npz')], 'trace-'+model)
            run('validate_full_graph.py', ['--output', artifacts/'stages',
                                          '--input', full/'inputs/000_9282.raim', '--traces', traces,
                                          '--backends', *args.backends, '--dtypes', 'f32', 'f16', 'q8_0',
                                          '--dynamic', '--strict-f32'], 'stages-dynamic')
        if 'dense' in args.suites:
            run('validate_dense_sampler.py', ['--output', artifacts/'dense', '--device', 'cpu',
                                             '--backends', *args.backends, '--dtypes', 'f32', 'f16', 'q8_0',
                                             '--strict-f32'], 'dense-sampler')
            run('audit_dense_candidates.py', ['--dense', artifacts/'dense',
                                             '--output', artifacts/'dense_candidate_audit'], 'dense-candidate-audit')
        if 'interfaces' in args.suites:
            run('validate_optional_interfaces.py', ['--output', artifacts/'interfaces',
                                                   '--backends', *args.backends, '--strict-f32'], 'optional-interfaces')
        if 'contracts' in args.suites:
            for backend in args.backends:
                run('check_runtime_contracts.py', ['--backend', backend, '--benchmark', full], 'contracts-'+backend)
        if 'contracts' in args.suites:
            for backend in args.backends:
                run('validate_c_api.py', ['--backend', backend, '--benchmark', full, '--output', artifacts/('c_api-'+backend+'.json')], 'c-api-'+backend)
        if 'metrics' in args.suites:
            common = ['--backends', *args.backends, '--dtypes', 'f32', 'f16', 'q8_0']
            run('evaluate_metric_subsets.py', ['--output', artifacts/'metric_subsets', *common], 'public-subsets')
            run('evaluate_metric_subsets.py', ['--data', ROOT/'cpp_ggml/test_data/metric_subsets',
                                             '--datasets', 'SpatialSense-valid', 'SpatialSense-test',
                                             '--output', artifacts/'spatialsense', *common], 'spatialsense-subsets')
        for backend, expected in record['builds'].items():
            if build_fingerprint(ROOT/f'cpp_ggml/build-{backend}/relateanything-ggml') != expected:
                raise RuntimeError(f'{backend} build changed during the suites')
        record.update(status='complete', finished_utc=now(),
                      scope='selected execution suites; bit equality, speed and protocol readiness are separate')
        save(folder/'run.json', record)
        run('write_acceptance_status.py', ['--run', folder], 'acceptance-status')
        record['acceptance_status'] = 'artifacts/acceptance_status.json'
        save(folder/'run.json', record)
        publish_reports(folder)
        if args.current:
            set_current(folder)
    except BaseException as exc:
        record.update(status='failed', finished_utc=now(), failure={'type': type(exc).__name__, 'message': str(exc)})
        save(folder/'run.json', record)
        raise
    print(f'complete={folder}', flush=True)


if __name__ == '__main__':
    main()
