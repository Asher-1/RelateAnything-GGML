#!/usr/bin/env python3
"""Test precise serial/partitioned FMA against the original CUDA patch Conv2d."""
import argparse
import ctypes
from itertools import permutations
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from audit_patch_embedding import difference
from provenance import ROOT, save, fingerprint

CUDA = r'''
#include <cuda_runtime.h>
template<int parts, bool interleave>
__global__ void accumulate(const float* w, const float* x, const float* bias,
                          const int* permutation, float* out, int channels) {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= channels*784) return;
    int c = id/784, p = id%784;
    float a[parts] = {};
    for (int base=0; base<768/parts; ++base) {
        #pragma unroll
        for (int part=0; part<parts; ++part) {
            int k = permutation[interleave ? base*parts+part : part*(768/parts)+base];
            a[part] = __fmaf_rn(w[c*768+k],x[k*784+p],a[part]);
        }
    }
    // Ordered sum of independently rounded partial accumulations.
    for (int i=1; i<parts; ++i) a[0] = __fadd_rn(a[0],a[i]);
    out[id] = __fadd_rn(a[0],bias[c]);
}
extern "C" int probe(const float* w,const float* x,const float* b,const int* perm,
                     float* out,int c,int parts,int interleave,void* stream) {
    #define CASE(n) case n: if(interleave) accumulate<n,true><<<(c*784+127)/128,128,0,(cudaStream_t)stream>>>(w,x,b,perm,out,c); else accumulate<n,false><<<(c*784+127)/128,128,0,(cudaStream_t)stream>>>(w,x,b,perm,out,c); break
    switch(parts) { CASE(1);CASE(2);CASE(4);CASE(8);CASE(16); }
    #undef CASE
    return int(cudaGetLastError());
}
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=ROOT/'cpp_ggml/outputs/runs/20261009-options-alignment')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    records = []
    with tempfile.TemporaryDirectory(prefix='ra-conv-fma-') as folder:
        src, library = Path(folder)/'probe.cu', Path(folder)/'probe.so'
        src.write_text(CUDA)
        subprocess.run(['/usr/local/cuda/bin/nvcc','-shared','-Xcompiler','-fPIC','-O3',
                        '-std=c++17','-arch=sm_86',str(src),'-o',str(library)], check=True)
        impl = ctypes.CDLL(str(library)).probe
        impl.argtypes = [ctypes.c_void_p]*5+[ctypes.c_int]*3+[ctypes.c_void_p]
        impl.restype = ctypes.c_int
        for model in ('relsgg-vits16','relsgg-vits16plus','relsgg-vitb16'):
            checkpoint = ROOT/'cpp_ggml/models/pytorch'/model/'model.pth'
            trace = args.run/'artifacts/stages/traces_cuda'/f'python_stage_trace_{model[7:]}.npz'
            state = torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)['model']
            with np.load(trace) as ref:
                x = torch.from_numpy(ref['normalized_image.0'].copy()).cuda()
                expected = torch.from_numpy(ref['backbone.model.embeddings.patch_embeddings'].copy()).cuda()
            w = state['backbone.model.embeddings.patch_embeddings.weight'].cuda().flatten(1)
            b = state['backbone.model.embeddings.patch_embeddings.bias'].cuda()
            columns = F.unfold(x,kernel_size=16,stride=16)[0].contiguous()
            out = torch.empty_like(expected)
            results = {}
            for order in permutations(range(3)):
                perm = torch.arange(768,device='cuda',dtype=torch.int32).reshape(3,16,16).permute(order).flatten().contiguous()
                for parts in (1,2,4,8,16):
                    for interleave in ((0,) if parts==1 else (0,1)):
                        code = impl(w.data_ptr(),columns.data_ptr(),b.data_ptr(),perm.data_ptr(),out.data_ptr(),
                                    w.shape[0],parts,interleave,torch.cuda.current_stream().cuda_stream)
                        assert code == 0, code
                        torch.cuda.synchronize()
                        key = ''.join('cyx'[i] for i in order)+f'-parts{parts}-interleave{interleave}'
                        results[key] = difference(out,expected)
            best = max(results,key=lambda k:results[k]['bit_equal_fraction'])
            records.append({'model':model,'results':results,'best':best,'inputs':fingerprint([checkpoint,trace])})
            print(model,best,results[best],flush=True)
    save(args.output,{'scope':'diagnostic FMA accumulation only; no inference implementation changed',
                      'torch':torch.__version__,'device':torch.cuda.get_device_name(),
                      'kernel_source':CUDA,'rows':records})


if __name__ == '__main__':
    main()
