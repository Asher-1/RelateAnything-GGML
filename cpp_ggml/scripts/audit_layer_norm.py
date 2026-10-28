#!/usr/bin/env python3
"""Isolate the next F32 boundary using identical official CUDA LayerNorm inputs."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from audit_patch_embedding import difference
from provenance import ROOT, fingerprint, save
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stages', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for model in ('relsgg-vits16', 'relsgg-vits16plus', 'relsgg-vitb16'):
        checkpoint = ROOT/'cpp_ggml/models/pytorch'/model/'model.pth'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)['model']
        trace = args.stages/'traces_cuda'/f'python_stage_trace_{model[7:]}.npz'
        base = 'backbone.model.'
        with np.load(trace) as tensors:
            patch = torch.from_numpy(tensors[base+'embeddings.patch_embeddings'].copy()).cuda()
            expected = torch.from_numpy(tensors[base+'model.layer.0.norm1'].copy())
        x = torch.cat((state[base+'embeddings.cls_token'].cuda(),
                       state[base+'embeddings.register_tokens'].cuda(),
                       patch.flatten(2).transpose(1, 2)), dim=1)
        weight = state[base+'model.layer.0.norm1.weight'].cuda()
        bias = state[base+'model.layer.0.norm1.bias'].cuda()
        # DINOv3 LayerNorm epsilon is recorded in the converted model metadata.
        from relsgg import RelateAnything
        api = RelateAnything.from_checkpoint(str(checkpoint), device='cpu', full_vocabulary=True)
        eps = api.model.backbone.model.model.layer[0].norm1.eps
        del api
        width = x.shape[-1]
        direct = F.layer_norm(x, (width,), weight, bias, eps)
        no_affine = F.layer_norm(x, (width,), None, None, eps)
        mean = x.mean(-1, keepdim=True)
        var = (x*x).mean(-1, keepdim=True) - mean*mean
        alternatives = {
            'official_layer_norm': direct,
            'welford_then_separate_affine': no_affine*weight+bias,
            'one_pass_sum_square_separate_affine': ((x-mean)*torch.rsqrt(var+eps))*weight+bias,
        }
        native = args.stages/model/'cuda-f32/stages/backbone_norm1.f32'
        actual = torch.from_numpy(np.fromfile(native, '<f4').reshape(tuple(expected.shape)))
        rows.append({'model': model, 'epsilon': eps, 'shape': list(x.shape),
                     'against_official': {name: difference(v, expected) for name, v in alternatives.items()},
                     'native_against_official': difference(actual, expected),
                     'native_against_welford_separate_affine': difference(actual, alternatives['welford_then_separate_affine']),
                     'source': fingerprint([checkpoint, trace, native, __file__])})
        print(model, rows[-1]['against_official'], flush=True)
    save(args.output, {'scope': 'same official F32 input; next boundary after exact native patch embedding',
                       'torch': torch.__version__, 'rows': rows,
                       'upstream_algorithm': 'https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/cuda/layer_norm_kernel.cu'})


if __name__ == '__main__':
    main()
