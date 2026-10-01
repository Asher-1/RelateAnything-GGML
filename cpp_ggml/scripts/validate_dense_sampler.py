#!/usr/bin/env python3
"""Exercise both sampler TopK stages with 60 distinct regions on a real image."""
import json
import struct
import subprocess
import numpy as np
from PIL import Image
from benchmark_full_graph import ROOT, MODELS, load_cpp, parity, raw_python, save


def main():
    import torch
    from relsgg import RelateAnything
    torch.set_num_threads(4)
    dst=ROOT/'cpp_ggml/benchmarks/full_graph_validation/dense_sampler'
    dst.mkdir(parents=True,exist_ok=True)
    sample=json.loads((ROOT/'cpp_ggml/test_data/ra4m_50/manifest.json').read_text())['samples'][0]
    image=Image.open(ROOT/'cpp_ggml/test_data/ra4m_50'/sample['image']).convert('RGB')
    rng=np.random.default_rng(20260930)
    lo=rng.uniform(-.03,.78,(60,2)).astype(np.float32)
    hi=lo+rng.uniform(.04,.4,(60,2)).astype(np.float32)
    xyxy=np.concatenate((lo,hi),1)*np.array([image.width,image.height]*2,np.float32)
    normalized=xyxy/np.array([image.width,image.height]*2,np.float32)
    boxes=np.concatenate(((normalized[:,:2]+normalized[:,2:])*.5,normalized[:,2:]-normalized[:,:2]),1)
    pixels=(ROOT/'cpp_ggml/benchmarks/full_graph/inputs/000_9282.raim').read_bytes()[20:20+602112*4]
    (dst/'input.raim').write_bytes(struct.pack('<8sIII',b'RAIMv1\0\0',448,448,60)+pixels+boxes.astype('<f4').tobytes())
    save(dst/'boxes.json',{'seed':20260930,'boxes_xyxy':xyxy.tolist(),'scope':'real image with synthetic varied boxes; no task GT for generated regions'})
    rows=[]
    for name in MODELS:
        api=RelateAnything.from_checkpoint(str(ROOT/'cpp_ggml/models/pytorch'/name/'model.pth'),device='cpu',full_vocabulary=True)
        captured={}
        handle=api.model.register_forward_hook(lambda m,i,o:captured.update(out=o))
        api.predict(image,xyxy,topk=128)
        reference=raw_python(captured['out']);handle.remove()
        np.savez(dst/(name+'-python.npz'),**reference)
        del api
        for backend in ('cpu','cuda','vulkan'):
            for dtype in (('f32',) if backend=='cpu' else ('f32','f16','q8_0')):
                out=dst/f'{name}-{backend}-{dtype}.rafo'
                cmd=[str(ROOT/f'cpp_ggml/build-{backend}/relateanything-ggml'),'--backend',backend,
                     '--model',str(ROOT/f'cpp_ggml/models/gguf/{name}-{dtype}.gguf'),
                     '--input',str(dst/'input.raim'),'--output',str(out),'--raw-logits','--warmup','0','--repeats','1','--threads','8']
                result=subprocess.run(cmd,capture_output=True,text=True)
                if result.returncode:raise RuntimeError(result.stderr[-2000:])
                record={'model':name,'backend':backend,'dtype':dtype,**parity(reference,load_cpp(out))}
                rows.append(record);print(name,backend,dtype,record['pair_set_recall'],record['logits_rmse'],flush=True)
        save(dst/'results.json',rows)
if __name__=='__main__':main()
