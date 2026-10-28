#!/usr/bin/env python3
"""Real checkpoint C/C++ memory/file API parity on all three storage formats."""
import argparse, json, subprocess
from pathlib import Path
from provenance import ROOT, build_fingerprint, fingerprint, save
from benchmark_full_graph import MODELS, DTYPES

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backend',required=True,choices=('cpu','cuda','vulkan'));p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    binary=ROOT/f'cpp_ggml/build-{a.backend}/relateanything-capi-contract';request=a.benchmark/'image_requests/000_9282.raip'
    rows=[]
    for model in MODELS:
        for dtype in DTYPES:
            weight=ROOT/f'cpp_ggml/models/gguf/{model}-{dtype}.gguf';cmd=[str(binary),str(weight),a.backend,str(request)]
            proc=subprocess.run(cmd,capture_output=True,text=True)
            rows.append({'model':model,'dtype':dtype,'backend':a.backend,'command':cmd,'returncode':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr,'inputs':fingerprint([weight,request])})
            print(model,a.backend,dtype,'PASS' if proc.returncode==0 else 'FAIL',flush=True)
            save(a.output,{'producer':fingerprint([Path(__file__),ROOT/'cpp_ggml/tests/c_api_contract.cpp']),
                           'binary':fingerprint([binary]),'runtime_build':build_fingerprint(binary.parent/'relateanything-ggml',a.backend),'rows':rows})
            if proc.returncode:raise RuntimeError(proc.stderr[-2000:])
if __name__=='__main__':main()
