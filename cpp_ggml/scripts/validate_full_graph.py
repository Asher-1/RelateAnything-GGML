#!/usr/bin/env python3
"""Compare real upstream intermediate tensors and custom-vocabulary inference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from benchmark_full_graph import ROOT, MODELS, load_cpp, parity, raw_python, save

STAGES = {
 'fused_patch_map': 'fused_patch_map.0', 'spatial_pool_0': 'spatial_pool.0',
 'geo_encoder': 'geo_encoder.0', 'box_prompt_tokens': 'box_prompt_tokens.0',
 'pair_proj': 'pair_proj.0', 'rel_transformer': 'rel_transformer.0',
 'deformable_read': 'deformable_read.0', 'rel_interaction': 'rel_interaction.0',
 'vocab_query': 'compose_norm', 'spa_proj': 'spa_proj', 'predicate_logits': 'final.logits',
}

def trace_path(model):
    suffix = {'relsgg-vits16':'_vits16','relsgg-vitb16':'_vitb16','relsgg-vits16plus':''}[model]
    return ROOT/f'cpp_ggml/test_data/python_stage_trace{suffix}.npz'

def run(args, model, backend, dtype, name, extra=()):
    dest = args.output/model/f'{backend}-{dtype}'/name
    dest.mkdir(parents=True, exist_ok=True)
    cmd=[str(ROOT/f'cpp_ggml/build-{backend}/relateanything-ggml'),'--backend',backend,
        '--model',str(ROOT/f'cpp_ggml/models/gguf/{model}-{dtype}.gguf'),
        '--input',str(ROOT/'cpp_ggml/benchmarks/full_graph/inputs/000_9282.raim'),
        '--output',str(dest/'output.rafo'),'--raw-logits','--warmup','0','--repeats','1','--threads','8']+list(extra)
    if name=='stages': cmd+=['--dump-directory',str(dest)]
    if args.no_flash: cmd+=['--no-flash-attention']
    result=subprocess.run(cmd,capture_output=True,text=True)
    (dest/'run.log').write_text(result.stdout+result.stderr)
    if result.returncode: raise RuntimeError(result.stderr[-2000:])
    return dest

def stage_check(args,model,backend,dtype):
    dest=run(args,model,backend,dtype,'stages')
    out=load_cpp(dest/'output.rafo')
    with np.load(trace_path(model)) as ref:
        reference={'logits':ref['final.logits'][0],'pairs':ref['final.pair_logits'][0],
                   'sub':ref['final.sub_idx'][0],'obj':ref['final.obj_idx'][0],'valid':ref['final.valid_mask'][0]}
        pairs={tuple(map(int,(out['sub'][i],out['obj'][i]))):i for i in np.flatnonzero(out['valid'])}
        ia=[i for i in np.flatnonzero(reference['valid']) if (int(reference['sub'][i]),int(reference['obj'][i])) in pairs]
        ib=[pairs[(int(reference['sub'][i]),int(reference['obj'][i]))] for i in ia]
        stages={}
        for cpp,py in STAGES.items():
            target=ref[py]; actual=np.fromfile(dest/(cpp+'.f32'),'<f4').reshape(target.shape)
            if cpp not in ('fused_patch_map','spatial_pool_0'):
                target=target[0,ia]; actual=actual[0,ib]
            error=actual.astype(np.float64)-target
            stages[cpp]={'shape':list(target.shape),'rmse':float(np.sqrt(np.mean(error**2))),
                         'max_abs':float(np.max(np.abs(error))), 'finite':bool(np.isfinite(actual).all())}
        result={'model':model,'backend':backend,'dtype':dtype,'stages':stages,'final':parity(reference,out)}
    save(dest/'comparison.json',result)
    print(model,backend,dtype,'stages RMSE',result['final']['logits_rmse'],flush=True)
    return result

def dynamic_reference(args,model):
    import torch
    from relsgg import RelateAnything
    from PIL import Image
    torch.set_num_threads(4)
    checkpoint=ROOT/'cpp_ggml/models/pytorch'/model/'model.pth'
    model_api=RelateAnything.from_checkpoint(str(checkpoint),device='cpu',full_vocabulary=True)
    # Nonadjacent known rows plus an unseen composite direction exercise
    # normalization, new V shapes and the native gate-MLP alpha computation.
    weights=model_api.model.vocab_head.W.detach().cpu().numpy()
    indices=[0,13,291,4001,11113]
    bank=np.concatenate([weights[indices]*np.array([2,.4,1,3,.7],np.float32)[:,None],
                         (weights[0]+weights[17])[None]],axis=0).astype(np.float32)
    names=[model_api.predicates[i] for i in indices]+['custom composite embedding']
    folder=args.output/model/'dynamic';folder.mkdir(parents=True,exist_ok=True)
    bank.tofile(folder/'vocabulary.f32');save(folder/'names.json',names)
    model_api.set_vocabulary(names,embeddings=bank)
    sample=json.loads((ROOT/'cpp_ggml/test_data/ra4m_50/manifest.json').read_text())['samples'][0]
    boxes=np.array([[x,y,x+w,y+h] for x,y,w,h in (o['bbox'] for o in sample['objects'])],np.float32)
    captured={}
    handle=model_api.model.register_forward_hook(lambda m,i,o:captured.update(out=o))
    with Image.open(ROOT/'cpp_ggml/test_data/ra4m_50'/sample['image']) as im:
        model_api.predict(im.convert('RGB'),boxes)
    handle.remove();out=raw_python(captured['out']);np.savez(folder/'reference.npz',**out)
    return folder

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    p.add_argument('--backends',nargs='+',choices=('cpu','cuda','vulkan'),default=['cpu','cuda','vulkan'])
    p.add_argument('--dtypes',nargs='+',choices=('f32','f16','q8_0'),default=['f32'])
    p.add_argument('--output',type=Path,default=ROOT/'cpp_ggml/benchmarks/full_graph_validation')
    p.add_argument('--no-flash',action='store_true');p.add_argument('--dynamic',action='store_true')
    args=p.parse_args();results=[]
    for model in args.models:
        folder=dynamic_reference(args,model) if args.dynamic else None
        for backend in args.backends:
            for dtype in args.dtypes:
                result=stage_check(args,model,backend,dtype)
                if folder:
                    dest=run(args,model,backend,dtype,'dynamic',['--vocabulary',str(folder/'vocabulary.f32')])
                    with np.load(folder/'reference.npz') as ref: dynamic=parity(ref,load_cpp(dest/'output.rafo'))
                    save(dest/'comparison.json',dynamic);result['dynamic_vocabulary']=dynamic
                    print('dynamic',model,backend,dtype,dynamic['logits_rmse'],flush=True)
                results.append(result)
    save(args.output/'summary.json',results)
    return 0
if __name__=='__main__':raise SystemExit(main())
