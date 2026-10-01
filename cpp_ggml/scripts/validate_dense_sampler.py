#!/usr/bin/env python3
"""Exercise both sampler TopK stages with 60 distinct regions on a real image."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import numpy as np
from PIL import Image
from benchmark_full_graph import ROOT, MODELS, load_cpp, parity, raw_python, save, sha256, prepare_image
from provenance import recorded_run, model_assets


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    parser.add_argument('--backends',nargs='+',choices=('cpu','cuda','vulkan'),default=['cpu','cuda','vulkan'])
    parser.add_argument('--dtypes',nargs='+',choices=('f32','f16','q8_0'))
    parser.add_argument('--matmul-geometry',action='store_true',help='historical Vulkan geometry scorer for ablation')
    parser.add_argument('--strict-f32',action='store_true')
    args=parser.parse_args()
    import torch
    from relsgg import RelateAnything
    torch.set_num_threads(4)
    dst=args.output or ROOT/'cpp_ggml/outputs/probes/dense'
    dst.mkdir(parents=True,exist_ok=True)
    sample=json.loads((ROOT/'cpp_ggml/test_data/ra4m_50/manifest.json').read_text())['samples'][0]
    image=Image.open(ROOT/'cpp_ggml/test_data/ra4m_50'/sample['image']).convert('RGB')
    rng=np.random.default_rng(20260930)
    lo=rng.uniform(-.03,.78,(60,2)).astype(np.float32)
    hi=lo+rng.uniform(.04,.4,(60,2)).astype(np.float32)
    xyxy=np.concatenate((lo,hi),1)*np.array([image.width,image.height]*2,np.float32)
    normalized=xyxy/np.array([image.width,image.height]*2,np.float32)
    boxes=np.concatenate(((normalized[:,:2]+normalized[:,2:])*.5,normalized[:,2:]-normalized[:,:2]),1)
    pixels=prepare_image(ROOT/'cpp_ggml/test_data/ra4m_50'/sample['image'], sample)[0].astype('<f4').tobytes()
    (dst/'input.raim').write_bytes(struct.pack('<8sIII',b'RAIMv1\0\0',448,448,60)+pixels+boxes.astype('<f4').tobytes())
    save(dst/'boxes.json',{'seed':20260930,'boxes_xyxy':xyxy.tolist(),'scope':'real image with synthetic varied boxes; no task GT for generated regions'})
    rows=[]
    for name in args.models:
        api=RelateAnything.from_checkpoint(str(ROOT/'cpp_ggml/models/pytorch'/name/'model.pth'),device=args.device,full_vocabulary=True)
        captured={}
        handle=api.model.register_forward_hook(lambda m,i,o:captured.update(out=o))
        stages={}
        hooks=[]
        for key,module in [('geo_scores',api.model.sampler.geo_scorer),
                           ('zs',api.model.sampler.f_sub),('zo',api.model.sampler.f_obj)]:
            hooks.append(module.register_forward_hook(lambda m,i,o,key=key:stages.update({key:o.detach()})))
        api.predict(image,xyxy,topk=128)
        reference=raw_python(captured['out']);handle.remove()
        for hook in hooks:hook.remove()
        scores=(stages['zs'].unsqueeze(2)*stages['zo'].unsqueeze(1)).sum(-1)/(stages['zs'].shape[-1]**.5)
        np.savez(dst/(name+'-python.npz'),**reference,
                 geo_scores=stages['geo_scores'].cpu().numpy().reshape(-1),
                 rel_scores=scores.cpu().numpy().reshape(-1))
        del api
        for backend in args.backends:
            for dtype in (args.dtypes or (('f32',) if backend=='cpu' else ('f32','f16','q8_0'))):
                out=dst/f'{name}-{backend}-{dtype}.rafo'
                cmd=[str(ROOT/f'cpp_ggml/build-{backend}/relateanything-ggml'),'--backend',backend,
                     '--model',str(ROOT/f'cpp_ggml/models/gguf/{name}-{dtype}.gguf'),
                     '--input',str(dst/'input.raim'),'--output',str(out),'--raw-logits','--warmup','0','--repeats','1','--threads','8',
                     '--dump-directory',str(dst/f'{name}-{backend}-{dtype}-stages')]
                if args.matmul_geometry:cmd.append('--matmul-geometry')
                if args.strict_f32:cmd.append('--strict-f32')
                run_folder=dst/f'{name}-{backend}-{dtype}-stages'
                with recorded_run(run_folder, inputs=[dst/'input.raim', Path(cmd[cmd.index('--model')+1]),
                                  dst/(name+'-python.npz')],config={'command':cmd},binary=Path(cmd[0]),outputs=[out]):
                    result=subprocess.run(cmd,capture_output=True,text=True)
                    (run_folder/'run.log').write_text(result.stdout+result.stderr)
                    if result.returncode:raise RuntimeError(result.stderr[-2000:])
                record={'model':name,'backend':backend,'dtype':dtype,'reference_device':args.device,
                        'matmul_geometry':args.matmul_geometry,'command':cmd,
                        'binary_sha256':sha256(Path(cmd[0])),
                        'reference_sha256':sha256(dst/(name+'-python.npz')),**parity(reference,load_cpp(out))}
                with np.load(dst/(name+'-python.npz')) as oracle:
                    for score in ('geo_scores','rel_scores'):
                        x=oracle[score]; y=np.fromfile(run_folder/(score+'.f32'),'<f4')
                        valid=np.arange(3600)//60!=np.arange(3600)%60
                        selected=np.flatnonzero(valid)[np.argsort(-x[valid])[:400]]
                        actual=np.flatnonzero(valid)[np.argsort(-y[valid])[:400]]
                        record[score]={'rmse':float(np.sqrt(np.mean((y.astype(np.float64)-x)**2))),
                                      'top400_missing':list(map(int,sorted(set(selected)-set(actual)))),
                                      'top400_extra':list(map(int,sorted(set(actual)-set(selected))))}
                        if score=='geo_scores':
                            boundary=np.flatnonzero(valid)[np.argsort(-x[valid])[398:402]]
                            record[score]['boundary']=[{'pair':int(ix),'python':float(x[ix]),'native':float(y[ix])} for ix in boundary]
                rows.append(record);print(name,backend,dtype,record['pair_set_recall'],record['logits_rmse'],record['geo_scores'],flush=True)
        save(dst/'results.json',rows)
if __name__=='__main__':main()
