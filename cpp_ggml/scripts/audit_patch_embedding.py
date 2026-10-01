#!/usr/bin/env python3
"""Isolate official CUDA Conv2d from layout, GEMM and TF32 effects."""
import argparse
import json
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from provenance import ROOT, fingerprint, save


def difference(actual, expected):
    actual = actual.detach().cpu().contiguous().numpy().astype(np.float32)
    expected = expected.detach().cpu().contiguous().numpy().astype(np.float32)
    bits = actual.view(np.uint32) == expected.view(np.uint32)
    error = actual.astype(np.float64) - expected
    first = None
    if not bits.all():
        index = tuple(int(v) for v in np.argwhere(~bits)[0])
        first = {'index': index, 'actual': float(actual[index]), 'expected': float(expected[index]),
                 'actual_bits': int(actual.view(np.uint32)[index]),
                 'expected_bits': int(expected.view(np.uint32)[index])}
    return {'bit_equal_fraction': float(bits.mean()), 'rmse': float(np.sqrt(np.mean(error**2))),
            'max_abs': float(np.max(np.abs(error))), 'first_difference': first}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path,
                        default=ROOT/'cpp_ggml/outputs/runs/20261009-options-alignment')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    for model in ('relsgg-vits16', 'relsgg-vits16plus', 'relsgg-vitb16'):
        checkpoint = ROOT/'cpp_ggml/models/pytorch'/model/'model.pth'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)['model']
        trace = args.run/'artifacts/stages/traces_cuda'/f'python_stage_trace_{model[7:]}.npz'
        with np.load(trace) as arrays:
            x = torch.from_numpy(arrays['normalized_image.0'].copy()).cuda()
            expected = torch.from_numpy(arrays['backbone.model.embeddings.patch_embeddings'].copy()).cuda()
        weight = state['backbone.model.embeddings.patch_embeddings.weight'].cuda()
        bias = state['backbone.model.embeddings.patch_embeddings.bias'].cuda()
        w = weight.flatten(1)
        columns = F.unfold(x, kernel_size=16, stride=16)[0]
        variants = {}
        for tf32 in (True, False):
            torch.backends.cudnn.allow_tf32 = tf32
            variants['conv2d_tf32_'+str(tf32)] = F.conv2d(x, weight, bias, stride=16)
        shape = expected.shape
        variants['w_mm_columns_plus_bias'] = (w @ columns + bias[:, None]).reshape(shape)
        variants['columns_T_mm_w_T_plus_bias'] = (columns.T.contiguous() @ w.T + bias).T.reshape(shape)
        variants['addmm_bias_epilogue'] = torch.addmm(bias, columns.T.contiguous(), w.T).T.reshape(shape)
        # cuDNN's implicit-GEMM may traverse channel/kernel dimensions in a
        # different order. Keep values fixed and test all six K traversals,
        # including GEMM operand-storage choices, without replacing the oracle.
        for order in permutations(range(3)):
            permutation = torch.arange(768, device='cuda').reshape(3, 16, 16).permute(order).flatten()
            cp, wp = columns[permutation], w[:, permutation]
            key = ''.join('cyx'[i] for i in order)
            variants['k_order_'+key+'_TN'] = (cp.T.contiguous() @ wp.T + bias).T.reshape(shape)
            variants['k_order_'+key+'_NN'] = (cp.T.contiguous() @ wp.T.contiguous() + bias).T.reshape(shape)
        # CPU/device/storage comparisons keep the original reference fixed.
        variants['conv2d_cpu'] = F.conv2d(x.cpu(), weight.cpu(), bias.cpu(), stride=16)
        native = args.run/'artifacts/stages'/model/'cuda-f32/stages/backbone_patch.f32'
        if native.is_file():
            values = np.fromfile(native, '<f4').reshape(1, -1, shape[1]).transpose(0, 2, 1).reshape(shape)
            variants['native_cuda_f32'] = torch.from_numpy(values.copy())
        torch.cuda.synchronize()
        rows.append({'model': model, 'weight_shape': list(weight.shape),
                     'comparisons': {name: difference(value, expected) for name, value in variants.items()},
                     'native_vs_torch_gemm': (difference(variants['native_cuda_f32'],
                                                       variants['columns_T_mm_w_T_plus_bias'])
                                             if 'native_cuda_f32' in variants else None),
                     'inputs': fingerprint([checkpoint, trace, trace.with_suffix('.json')])})
    save(args.output, {'scope': 'same pixels and original FP32 weights; first patch-embedding primitive only',
                       'torch': torch.__version__, 'cudnn': torch.backends.cudnn.version(),
                       'device': torch.cuda.get_device_name(), 'rows': rows})
    for row in rows:
        print(row['model'], {name: round(value['bit_equal_fraction'], 6)
                             for name, value in row['comparisons'].items()})


if __name__ == '__main__':
    main()
