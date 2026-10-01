#!/usr/bin/env python3
"""Compare complete outputs with capture policy enabled/disabled after warmup."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from benchmark_full_graph import load_cpp, MODELS, DTYPES
from provenance import ROOT, build_fingerprint, fingerprint, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.inputs.is_dir() or len(list(args.inputs.glob('*.raim'))) != 50:
        parser.error('--inputs must contain all 50 normalized RAIM images')
    binary = ROOT/'cpp_ggml/build-cuda/relateanything-ggml'
    build = build_fingerprint(binary, 'cuda')
    rows = []
    for model in MODELS:
        for dtype in DTYPES:
            weight = ROOT/'cpp_ggml/models/gguf'/f'{model}-{dtype}.gguf'
            with tempfile.TemporaryDirectory(prefix='ra-replay-bits-') as temporary:
                folder = Path(temporary)
                commands = []
                for enabled in (False, True):
                    output = folder/str(enabled)
                    cmd = [str(binary), '--model', str(weight), '--backend', 'cuda',
                           '--input', str(args.inputs.resolve()), '--output', str(output),
                           '--strict-f32', '--raw-logits', '--warmup', '3', '--repeats', '1']
                    if not enabled:
                        cmd.append('--no-cuda-graphs')
                    result = subprocess.run(cmd, text=True, capture_output=True)
                    if result.returncode:
                        raise RuntimeError(result.stderr[-3000:])
                    commands.append({'command': cmd, 'stdout': result.stdout, 'stderr': result.stderr})
                originals = sorted((folder/'False').glob('*.rafo'))
                assert len(originals) == 50
                files, matches = [], []
                for before in originals:
                    after = folder/'True'/before.name
                    a, b = load_cpp(before), load_cpp(after)
                    matches.append({'sample': before.stem, 'equal': all(
                        np.array_equal(a[k].view(np.uint8), b[k].view(np.uint8)) for k in a)})
                    files += [before, after]
                row = {'model': model, 'dtype': dtype, 'images': len(matches),
                       'all_bits_equal': all(m['equal'] for m in matches),
                       'matches': matches, 'raw_hashes': fingerprint(files),
                       'weight': fingerprint([weight]), 'commands': commands}
                rows.append(row)
                save(args.output, {'status': 'running', 'rows': rows})
                print(model, dtype, 'replay raw bits', sum(m['equal'] for m in matches), '/ 50', flush=True)
    assert build == build_fingerprint(binary, 'cuda')
    passed = all(row['all_bits_equal'] for row in rows)
    save(args.output, {'status': 'pass' if passed else 'fail', 'rows': rows, 'build': build,
                       'inputs': fingerprint(args.inputs.glob('*.raim')),
                       'scope': '3 warmups allow capture before raw export; every tensor and pair index; 450 images. Actual capture requires separate profiler evidence.',
                       'source': fingerprint([__file__])})
    if not passed:
        raise RuntimeError('CUDA capture/replay changed raw inference bits')


if __name__ == '__main__':
    main()
