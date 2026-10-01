#!/usr/bin/env python3
"""Compare actual GGML CUDA softmax bits against independent PyTorch CUDA."""
import argparse
from pathlib import Path
import subprocess

import numpy as np
import torch

from provenance import ROOT, build_fingerprint, fingerprint, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    binary = ROOT/'cpp_ggml/build-cuda/relateanything-softmax-probe'
    rng = np.random.default_rng(12893)
    rows = []
    shapes = [(c, 19) for c in (1, 2, 3, 16, 24, 31, 32, 33, 72, 127, 128,
                                129, 255, 256, 512, 789, 856, 1023, 1024, 1025)]
    shapes.append((789, 4734))
    for cols, count in shapes:
        for scale in (1.0, 0.125, float(np.float32(1/np.sqrt(32)))):
            name = f'{cols}x{count}-scale{scale}'
            x = rng.normal(0, 3, (count, cols)).astype(np.float32)
            # Equal logits, overflow-resistant large shifts, and near ties.
            x[0] = 0
            x[1] += 1000
            x[2] -= 1000
            x[3] = np.linspace(-80, 80, cols, dtype=np.float32)
            x[4] = 1 + rng.normal(0, 1e-7, cols).astype(np.float32)
            input_path = args.output/(name+'.input.f32')
            x.tofile(input_path)
            gpu = torch.from_numpy(x).cuda()
            with torch.inference_mode():
                ref = torch.softmax(gpu * scale, dim=-1).cpu().numpy()
            reference_path = args.output/(name+'.torch.f32')
            ref.tofile(reference_path)
            result = {'columns': cols, 'rows': count, 'scale': scale,
                      'scope': 'persistent warp' if cols <= 1024 else 'general fallback'}
            for strict in (False, True):
                dst = args.output/(name+f'.strict-{strict}.f32')
                run = subprocess.run([str(binary), str(input_path), str(dst), str(cols),
                                      str(count), str(scale), str(int(strict)),
                                      '100' if count > 19 else '5'],
                                     text=True, capture_output=True, check=True)
                actual = np.fromfile(dst, '<f4').reshape(ref.shape)
                equal = actual.view(np.uint32) == ref.view(np.uint32)
                result[str(strict)] = {'bit_equal': bool(equal.all()),
                                      'bit_equal_fraction': float(equal.mean()),
                                      'max_abs': float(np.max(np.abs(actual-ref))),
                                      'median_ms': float(run.stdout.split('median_ms=')[-1]),
                                      'files': fingerprint([input_path, reference_path, dst])}
                if strict and cols <= 1024 and not equal.all():
                    result[str(strict)]['first_difference'] = np.argwhere(~equal)[0].tolist()
            rows.append(result)
            print(name, 'strict bits', result['True']['bit_equal_fraction'],
                  'baseline/strict ms', result['False']['median_ms'],
                  result['True']['median_ms'], flush=True)
    precise = [r for r in rows if r['columns'] <= 1024]
    passed = all(r['True']['bit_equal'] for r in precise)
    save(args.output/'summary.json', {
        'status': 'pass' if passed else 'fail', 'torch': torch.__version__,
        'device': torch.cuda.get_device_name(),
        'model_build': build_fingerprint(ROOT/'cpp_ggml/build-cuda/relateanything-ggml', 'cuda'),
        'probe': fingerprint([binary, Path(__file__), ROOT/'cpp_ggml/tests/strict_softmax_probe.cpp']),
        'rows': rows, 'covered_cases': len(precise),
        'scope': 'one unmasked F32 operator, not full-model bit equality'})
    if not passed:
        raise RuntimeError('strict warp softmax disagrees with the original PyTorch CUDA operator')


if __name__ == '__main__':
    main()
