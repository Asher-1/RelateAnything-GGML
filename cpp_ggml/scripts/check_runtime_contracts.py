#!/usr/bin/env python3
"""Exercise native preprocessing and full-graph edge/error contracts."""
from pathlib import Path
import argparse
import json
import struct
import subprocess
import tempfile
import numpy as np
from benchmark_full_graph import ROOT, save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=('cpu','cuda','vulkan'),default='cpu')
    args=p.parse_args();binary=ROOT/f'cpp_ggml/build-{args.backend}/relateanything-ggml'
    folder=ROOT/'cpp_ggml/benchmarks/full_graph'
    result=subprocess.run([str(binary),'--preprocess-only','--input',str(folder/'image_requests'),'--output',str(folder/'native_inputs')],capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr)
    rows=[]
    for path in sorted((folder/'inputs').glob('*.raim')):
        a=np.frombuffer(path.read_bytes(),'<f4',offset=20)
        b=np.frombuffer((folder/'native_inputs'/path.name).read_bytes(),'<f4',offset=20)
        error=np.abs(a-b)
        rows.append({'sample':path.stem,'pixel_max_abs':float(error[:602112].max()),
                     'pixels_bit_equal':bool(np.array_equal(a[:602112],b[:602112])),
                     'boxes_max_abs':float(error[602112:].max())})
    if len(rows)!=50 or not all(r['pixels_bit_equal'] and r['boxes_max_abs']==0 for r in rows):
        raise RuntimeError('native/Python preprocessing mismatch')
    save(folder/'preprocessing_parity.json',rows)
    checks=[]
    with tempfile.TemporaryDirectory(prefix='ra-contract-') as tmp:
        tmp=Path(tmp);image=np.frombuffer((folder/'inputs/000_9282.raim').read_bytes(),'<f4',602112,20)
        for count in (0,1,2,60):
            boxes=np.array([[.5,.5,.25,.25]]*count,np.float32)
            path=tmp/f'{count}.raim';path.write_bytes(struct.pack('<8sIII',b'RAIMv1\0\0',448,448,count)+image.tobytes()+boxes.tobytes())
            cmd=[str(binary),'--model',str(ROOT/'cpp_ggml/models/gguf/relsgg-vits16-f32.gguf'),'--input',str(path),'--backend',args.backend,'--warmup','0','--repeats','1','--threads','8','--output',str(tmp/'out.json')]
            out=subprocess.run(cmd,capture_output=True,text=True)
            if out.returncode:raise RuntimeError(out.stderr[-2000:])
            predictions=json.loads((tmp/'out.json').read_text())['predictions']
            if count<2 and predictions:raise RuntimeError('zero/one region must return no relations')
            if any(r['subject']==r['object'] or not 0<=r['score']<=1 for r in predictions):raise RuntimeError('invalid decoded relation')
            checks.append({'objects':count,'relations':len(predictions),'passed':True})
        malformed=tmp/'bad.raim';malformed.write_bytes(struct.pack('<8sIII',b'RAIMv1\0\0',0xffffffff,448,2))
        cmd[cmd.index('--input')+1]=str(malformed)
        bad=subprocess.run(cmd,capture_output=True,text=True)
        if bad.returncode==0:raise RuntimeError('malformed dimensions were accepted')
        checks.append({'case':'malformed dimensions','passed':True,'error':bad.stderr.strip().splitlines()[-1]})
    save(folder/f'contracts-{args.backend}.json',checks)
    print(args.backend,'preprocessing 50/50 bit exact; input contracts PASS')
if __name__=='__main__':main()
