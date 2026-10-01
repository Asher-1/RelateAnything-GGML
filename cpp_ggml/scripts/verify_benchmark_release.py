#!/usr/bin/env python3
"""Audit complete measured coverage against current files; refresh raw parity."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from benchmark_full_graph import ROOT,MODELS,DTYPES,load_cpp,parity,save,sha256,decode


def main():
    folder=ROOT/'cpp_ggml/benchmarks/full_graph'
    samples=json.loads((ROOT/'cpp_ggml/test_data/ra4m_50/manifest.json').read_text())['samples']
    expected=[f'{i:03d}_{x["image_id"]}' for i,x in enumerate(samples)]
    check=[]
    for model in MODELS:
        py=folder/model/'python'
        reference=json.loads((py/'results.json').read_text())
        assert [r['sample'] for r in reference['rows']]==expected
        vocab=json.loads((py/'vocabulary.json').read_text())
        for backend in ('cuda','vulkan'):
            for dtype in DTYPES:
                dest=folder/model/f'{backend}-{dtype}'
                path=dest/'results.json';result=json.loads(path.read_text())
                assert result['binary_sha256']==sha256(ROOT/f'cpp_ggml/build-{backend}/relateanything-ggml')
                assert result['gguf_sha256']==sha256(ROOT/f'cpp_ggml/models/gguf/{model}-{dtype}.gguf')
                assert [r['sample'] for r in result['rows']]==expected
                attempts=json.loads((dest/'timing_attempts.json').read_text())
                assert attempts[-1]['transient_pids']==[] and attempts[-1]['returncode']==0
                for row in result['rows']:
                    name=row['sample'];candidate=load_cpp(dest/(name+'.rafo'))
                    with np.load(py/(name+'.npz')) as ref:
                        with threadpool_limits(limits=1):row['parity']=parity(ref,candidate)
                    row['reference_sha256']=sha256(py/(name+'.npz'))
                    decoded=decode(candidate,vocab['names'],vocab['calibration'])
                    native=json.loads((dest/(name+'.json')).read_text())['predictions']
                    lookup={(x['subject'],x['object']):x for x in native}
                    error=max(abs(lookup[(x['subject'],x['object'])]['score']-x['score']) for x in decoded)
                    assert error<1e-5
                    assert all(lookup[(x['subject'],x['object'])]['predicate_id']==x['predicate_id'] for x in decoded)
                    row['native_decode_max_error']=error
                save(path,result)
                check.append({'model':model,'backend':backend,'dtype':dtype,'images':len(result['rows']),'native_decode':True,'current_reference':True,'binary_and_weights_hash':True,'no_transient_gpu_jobs':True})
                print(model,backend,dtype,'verified 50/50',flush=True)
        attempts=json.loads((py/'timing_attempts.json').read_text())
        assert attempts[-1]['transient_pids']==[] and attempts[-1]['returncode']==0
    save(folder/'release_audit.json',{'ggml_image_configurations':900,'python_images':150,'all_complete':True,'checks':check})
if __name__=='__main__':main()
