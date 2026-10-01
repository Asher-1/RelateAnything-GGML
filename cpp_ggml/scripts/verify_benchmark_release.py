#!/usr/bin/env python3
"""Verify execution identities and raw outputs before publishing a release audit."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from benchmark_full_graph import ROOT, MODELS, DTYPES, load_cpp, parity, save, sha256, decode
from provenance import validate_run


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def audit(folder, checks):
    samples = json.loads((ROOT/'cpp_ggml/test_data/ra4m_50/manifest.json').read_text())['samples']
    expected = [f'{i:03d}_{x["image_id"]}' for i, x in enumerate(samples)]
    for model in MODELS:
        py = folder/model/'python'
        reference = json.loads((py/'results.json').read_text())
        validate_run(py, script=ROOT/'cpp_ggml/scripts/benchmark_full_graph.py')
        require([r['sample'] for r in reference['rows']] == expected, f'Python coverage: {model}')
        vocab = json.loads((py/'vocabulary.json').read_text())
        for backend in ('cpu', 'cuda', 'vulkan'):
            configurations = [(dtype, '') for dtype in DTYPES]
            if backend == 'cuda' and (folder/model/'cuda-q8_0-f32-input/results.json').exists():
                configurations.append(('q8_0', '-f32-input'))
            for dtype, variant in configurations:
                dest = folder/model/f'{backend}-{dtype}{variant}'
                result = json.loads((dest/'results.json').read_text())
                binary = ROOT/f'cpp_ggml/build-{backend}/relateanything-ggml'
                provenance = validate_run(dest, binary=binary)
                require(result['binary_sha256'] == provenance['build']['binary_sha256'],
                        f'result/binary mismatch: {dest}')
                require(result['gguf_sha256'] == sha256(ROOT/f'cpp_ggml/models/gguf/{model}-{dtype}.gguf'),
                        f'result/weights mismatch: {dest}')
                require([r['sample'] for r in result['rows']] == expected, f'native coverage: {dest}')
                attempts = json.loads((dest/'timing_attempts.json').read_text())
                require(attempts[-1]['transient_pids'] == [] and attempts[-1]['returncode'] == 0,
                        f'failed or contaminated timing: {dest}')
                for row in result['rows']:
                    name = row['sample']
                    candidate = load_cpp(dest/(name+'.rafo'))
                    with np.load(py/(name+'.npz')) as ref, threadpool_limits(limits=1):
                        numerical = parity(ref, candidate)
                    for key, value in numerical.items():
                        require(np.isclose(value, row['parity'][key], rtol=1e-10, atol=1e-10),
                                f'raw tensor/parity mismatch: {dest}/{name}/{key}')
                    decoded = decode(candidate, vocab['names'], vocab['calibration'])
                    native = json.loads((dest/(name+'.json')).read_text())['predictions']
                    lookup = {(x['subject'], x['object']): x for x in native}
                    require(set(lookup) == {(x['subject'], x['object']) for x in decoded},
                            f'calibrated pair mismatch: {dest}/{name}')
                    error = max((abs(lookup[(x['subject'], x['object'])]['score']-x['score'])
                                 for x in decoded), default=0.)
                    require(error < 1e-5, f'calibration mismatch: {dest}/{name}/{error}')
                    require(all(lookup[(x['subject'], x['object'])]['predicate_id'] == x['predicate_id']
                                for x in decoded), f'predicate decode mismatch: {dest}/{name}')
                checks.append({'model': model, 'backend': backend, 'dtype': dtype,
                               'variant': variant, 'images': len(result['rows']),
                               'provenance_sha256': sha256(dest/'run_provenance.json'),
                               'native_decode': True, 'raw_parity': True})
                print(model, backend, dtype+variant, 'verified 50/50', flush=True)
        for name in ('timing_attempts.json', 'timing-before.json', 'timing-after.json'):
            attempts = json.loads((py/name).read_text())
            require(attempts[-1]['transient_pids'] == [] and attempts[-1]['returncode'] == 0,
                    f'failed or contaminated Python timing: {py/name}')
    primary = sum(c['images'] for c in checks if not c['variant'])
    diagnostic = sum(c['images'] for c in checks if c['variant'])
    require(primary == 1350, f'primary coverage {primary}/1350')
    return {'primary_ggml_image_configurations': primary,
            'diagnostic_ggml_image_configurations': diagnostic,
            'ggml_image_configurations': primary+diagnostic, 'python_images': 150,
            'python_timing_passes': 2, 'checks': checks,
            'scope': 'execution provenance and raw/decoded consistency; numerical equality is reported separately'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=(ROOT/'cpp_ggml/benchmarks/current').resolve().parent/'artifacts/full_graph')
    folder = parser.parse_args().folder.resolve()
    path = folder/'release_audit.json'
    started = datetime.now(timezone.utc).isoformat()
    save(path, {'status': 'running', 'all_complete': False, 'started_utc': started})
    checks = []
    try:
        result = audit(folder, checks)
        save(path, {**result, 'status': 'complete', 'all_complete': True,
                    'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat()})
    except BaseException as exc:
        save(path, {'status': 'failed', 'all_complete': False, 'checks': checks,
                    'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
                    'failure': {'type': type(exc).__name__, 'message': str(exc)}})
        raise


if __name__ == '__main__':
    main()
