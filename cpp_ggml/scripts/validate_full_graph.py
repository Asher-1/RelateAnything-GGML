#!/usr/bin/env python3
"""Compare real upstream intermediate tensors and custom-vocabulary inference."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from benchmark_full_graph import ROOT, MODELS, load_cpp, parity, raw_python, save, sha256
from provenance import recorded_run, model_assets, validate_files, python_fingerprint

STAGES = {
 'fused_patch_map': 'fused_patch_map.0', 'spatial_pool_0': 'spatial_pool.0',
 'geo_encoder': 'geo_encoder.0', 'box_prompt_tokens': 'box_prompt_tokens.0',
 'pair_proj': 'pair_proj.0', 'rel_transformer': 'rel_transformer.0',
 'rel_self0':'rel_transformer.self_layers.0', 'rel_self1':'rel_transformer.self_layers.1',
 'rel_cross0':'rel_transformer.cross_layers.0',
 'deformable_read': 'deformable_read.0', 'rel_interaction': 'rel_interaction.0',
 'vocab_query': 'compose_norm', 'spa_proj': 'spa_proj', 'predicate_logits': 'final.logits',
}
# First-block internals in execution order; values are trace_python_stages.BLOCK0 keys.
BLOCK0_PREFIX = 'model.layer.0.'
BLOCK0 = {
 'backbone_q': 'attention.q_proj:output', 'backbone_k': 'attention.k_proj:output',
 'backbone_v': 'attention.v_proj:output', 'backbone_attn': 'attention.o_proj:input',
 'backbone_o_proj': 'attention.o_proj:output', 'backbone_ls1': 'layer_scale1:output',
 'backbone_norm2': 'norm2:output', 'backbone_up': 'mlp.up_proj:output',
 'backbone_act': 'mlp.down_proj:input', 'backbone_down': 'mlp.down_proj:output',
 'backbone_ls2': 'layer_scale2:output',
}

def trace_path(args, model):
    return args.traces/f'python_stage_trace_{model.removeprefix("relsgg-")}.npz'

@contextmanager
def run(args, model, backend, dtype, name, extra=()):
    dest = args.output/model/f'{backend}-{dtype}'/name
    dest.mkdir(parents=True, exist_ok=True)
    cmd=[str(ROOT/f'cpp_ggml/build-{backend}/relateanything-ggml'),'--backend',backend,
        '--model',str(ROOT/f'cpp_ggml/models/gguf/{model}-{dtype}.gguf'),
        '--input',str(args.input),
        '--output',str(dest/'output.rafo'),'--raw-logits','--warmup','0','--repeats','1','--threads','8']+list(extra)
    if name=='stages': cmd+=['--dump-directory',str(dest)]
    if args.no_flash: cmd+=['--no-flash-attention']
    if args.strict_f32: cmd+=['--strict-f32']
    inputs=[args.input, Path(cmd[cmd.index('--model')+1])]
    if '--vocabulary' in cmd:
        bank=Path(cmd[cmd.index('--vocabulary')+1])
        inputs += [bank, bank.parent/'reference.npz', bank.parent/'run_provenance.json']
    else:
        inputs += [trace_path(args,model),trace_path(args,model).with_suffix('.json')]
        if args.cuda_reference:
            secondary=args.output/'traces_cuda'/trace_path(args,model).name
            inputs += [secondary,secondary.with_suffix('.json')]
    with recorded_run(dest, inputs=inputs, config={'command':cmd}, binary=Path(cmd[0]),script=__file__):
        result=subprocess.run(cmd,capture_output=True,text=True)
        (dest/'run.log').write_text(result.stdout+result.stderr)
        if result.returncode: raise RuntimeError(result.stderr[-2000:])
        yield dest

def stage_check(args,model,backend,dtype):
    with run(args,model,backend,dtype,'stages') as dest:
        result=compare_stages(args,model,backend,dtype,dest,write=False)
        if args.cuda_reference:
            reference=args.output/'traces_cuda'/trace_path(args,model).name
            original=compare_stages(args,model,backend,dtype,dest,reference_path=reference,write=False)
            result['original_cuda_reference']={k:original[k] for k in
                ('stages','final','reference_device','reference_sha256','reference_math',
                 'reference_torch_version','first_nonidentical_boundary')}
        save(dest/'comparison.json',result)
        return result


def stage_statistics(actual,target):
    error=actual.astype(np.float64)-target
    a=np.asarray(actual,dtype=np.float32);b=np.asarray(target,dtype=np.float32)
    different=a.view(np.uint32)!=b.view(np.uint32)
    first=None
    if different.any():
        index=tuple(int(v) for v in np.argwhere(different)[0])
        first={'index':list(index),'reference':float(b[index]),'candidate':float(a[index]),
               'reference_bits':int(b.view(np.uint32)[index]),'candidate_bits':int(a.view(np.uint32)[index])}
    return {'shape':list(target.shape),'rmse':float(np.sqrt(np.mean(error**2))),
            'max_abs':float(np.max(np.abs(error))),'finite':bool(np.isfinite(a).all()),
            'bit_equal_fraction':float(np.mean(~different)),'first_nonidentical_value':first}

def compare_stages(args,model,backend,dtype,dest,reference_path=None,write=True):
    trace=Path(reference_path) if reference_path else trace_path(args,model)
    out=load_cpp(dest/'output.rafo')
    metadata=json.loads(trace.with_suffix('.json').read_text())
    validate_files(metadata['inputs'])
    if metadata['python']!=python_fingerprint(ROOT/'cpp_ggml/scripts/trace_python_stages.py'):
        raise RuntimeError('stage trace uses obsolete Python source')
    if metadata['checkpoint']!=model or metadata['checkpoint_sha256']!=sha256(ROOT/'cpp_ggml/models/pytorch'/model/'model.pth'):
        raise RuntimeError('stage trace model/checkpoint mismatch')
    with np.load(trace) as ref:
        pixels=np.fromfile(args.input,'<f4',offset=20,count=3*448*448).reshape(1,3,448,448)
        if not np.array_equal(pixels,ref['normalized_image.0']):
            raise RuntimeError('stage trace pixels differ from native input')
        boxes=np.fromfile(args.input,'<f4',offset=20+3*448*448*4).reshape(1,-1,4)
        if not np.array_equal(boxes,ref['input.boxes']):
            raise RuntimeError('stage trace boxes differ from native input')
        reference={'logits':ref['final.logits'][0],'pairs':ref['final.pair_logits'][0],
                   'sub':ref['final.sub_idx'][0],'obj':ref['final.obj_idx'][0],'valid':ref['final.valid_mask'][0]}
        pairs={tuple(map(int,(out['sub'][i],out['obj'][i]))):i for i in np.flatnonzero(out['valid'])}
        ia=[i for i in np.flatnonzero(reference['valid']) if (int(reference['sub'][i]),int(reference['obj'][i])) in pairs]
        ib=[pairs[(int(reference['sub'][i]),int(reference['obj'][i]))] for i in ia]
        stages={}
        backbone={'backbone_patch':('embeddings.patch_embeddings',None),
                  'backbone_norm1':('model.layer.0.norm1',None),
                  **{cpp:(BLOCK0_PREFIX+suffix,None) for cpp,suffix in BLOCK0.items()},
                  'backbone_block0':('model.layer.0',None)}
        for cpp,(suffix,_) in backbone.items():
            key=next(k for k in ref.files if k.endswith(suffix))
            target=ref[key]
            if cpp=='backbone_patch': target=target.reshape(target.shape[0],target.shape[1],-1).transpose(0,2,1)
            actual=np.fromfile(dest/(cpp+'.f32'),'<f4').reshape(target.shape)
            stages[cpp]=stage_statistics(actual,target)
        for cpp,py in STAGES.items():
            target=ref[py]
            values=np.fromfile(dest/(cpp+'.f32'),'<f4')
            shape=target.shape
            if cpp not in ('fused_patch_map','spatial_pool_0'):
                width=int(np.prod(shape[2:]))
                shape=(shape[0],values.size//width,*shape[2:])
            actual=values.reshape(shape)
            if cpp not in ('fused_patch_map','spatial_pool_0'):
                target=target[0,ia]; actual=actual[0,ib]
            stages[cpp]=stage_statistics(actual,target)
        result={'model':model,'backend':backend,'dtype':dtype,'stages':stages,'final':parity(reference,out),
                'reference_math':metadata.get('math',{}),'reference_torch_version':metadata['torch_version'],
                'binary_sha256':sha256(ROOT/f'cpp_ggml/build-{backend}/relateanything-ggml'),
                'reference_sha256':sha256(trace),'reference_device':metadata['device'],
                'first_nonidentical_boundary':next((k for k,v in stages.items() if v['first_nonidentical_value'] is not None),None),
                'reference_metadata':metadata,'input_sha256':sha256(args.input)}
    if write:save(dest/'comparison.json',result)
    print(model,backend,dtype,'stages RMSE',result['final']['logits_rmse'],flush=True)
    return result

def dynamic_reference(args,model):
    folder=args.output/model/'dynamic'
    data=ROOT/'cpp_ggml/test_data/ra4m_50'
    sample=json.loads((data/'manifest.json').read_text())['samples'][0]
    with recorded_run(folder, inputs=[*model_assets(model), args.input,
                      data/'manifest.json',data/sample['image']],
                      config={'device':'cpu','model':model,'vocabulary':'six scaled/composite rows'},script=__file__):
        return _dynamic_reference(args,model)


def _dynamic_reference(args,model):
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
    p.add_argument('--output',type=Path,default=ROOT/'cpp_ggml/outputs/probes/stages')
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--traces',type=Path,required=True)
    p.add_argument('--no-flash',action='store_true');p.add_argument('--dynamic',action='store_true')
    p.add_argument('--strict-f32',action='store_true')
    p.add_argument('--cuda-reference',action=argparse.BooleanOptionalAction,default=True,
                   help='also locate the first boundary against original PyTorch-CUDA')
    args=p.parse_args();results=[]
    for model in args.models:
        if args.cuda_reference:
            traces=args.output/'traces_cuda';traces.mkdir(parents=True,exist_ok=True)
            stem=trace_path(args,model).stem
            command=[sys.executable,str(ROOT/'cpp_ggml/scripts/trace_python_stages.py'),
                     '--model-name',model,'--device','cuda','--output',str(traces/(stem+'.json')),
                     '--tensors',str(traces/(stem+'.npz'))]
            completed=subprocess.run(command,text=True,capture_output=True)
            (traces/(stem+'.log')).write_text(json.dumps(command)+'\n'+completed.stdout+completed.stderr)
            if completed.returncode:raise RuntimeError(completed.stderr[-2000:])
        folder=dynamic_reference(args,model) if args.dynamic else None
        for backend in args.backends:
            for dtype in args.dtypes:
                result=stage_check(args,model,backend,dtype)
                if folder:
                    with run(args,model,backend,dtype,'dynamic',['--vocabulary',str(folder/'vocabulary.f32')]) as dest:
                        with np.load(folder/'reference.npz') as ref: dynamic=parity(ref,load_cpp(dest/'output.rafo'))
                        save(dest/'comparison.json',dynamic);result['dynamic_vocabulary']=dynamic
                    print('dynamic',model,backend,dtype,dynamic['logits_rmse'],flush=True)
                results.append(result)
                save(args.output/'summary.json',results)
    save(args.output/'summary.json',results)
    stage_overview(args.output,results)
    return 0

def stage_overview(folder,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys=list(rows[0]['stages']);labels=[r['model']+'/'+r['backend']+'/'+r['dtype'] for r in rows]
    first=[]
    for reference in ('cpu','cuda'):
        if reference=='cuda' and not all('original_cuda_reference' in r for r in rows):continue
        values=[r if reference=='cpu' else r['original_cuda_reference'] for r in rows]
        errors=np.array([[v['stages'][k]['rmse'] for k in keys] for v in values])
        fig,ax=plt.subplots(figsize=(17,12),layout='constrained')
        im=ax.imshow(np.log10(np.maximum(errors,1e-12)),aspect='auto',cmap='viridis')
        ax.set_xticks(range(len(keys)),keys,rotation=55,ha='right',fontsize=8)
        ax.set_yticks(range(len(rows)),labels,fontsize=8)
        ax.set_title('Independent PyTorch-'+reference.upper()+' reference | all instrumented boundaries')
        fig.colorbar(im,ax=ax,label='log10(matched tensor RMSE)')
        for suffix in ('png','svg'):fig.savefig(folder/('stage_errors_'+reference+'.'+suffix),dpi=155)
        plt.close(fig)
        for r,v in zip(rows,values):
            boundary=v['first_nonidentical_boundary']
            first.append({'model':r['model'],'backend':r['backend'],'dtype':r['dtype'],
                'reference_device':reference,'first_nonidentical_boundary':boundary,
                'difference':v['stages'][boundary]['first_nonidentical_value'] if boundary else None,
                'scope':'first instrumented boundary; not an unobserved internal primitive'})
    save(folder/'first_numerical_divergence.json',first)
    (folder/'README.md').write_text('# Intermediate and dynamic-vocabulary parity\n\n'
        'All 27 model/backend/storage configurations use the same real image. '
        'Normalized pixels and boxes must be equal before comparing any boundary. '
        'Original PyTorch-CUDA and independent CPU references are labelled separately. '
        'First differences include coordinates, values and IEEE-754 bits; they locate '
        'the first instrumented boundary, not an unobserved internal primitive.\n\n'
        '[Every boundary and dynamic bank](summary.json) · [First numerical differences](first_numerical_divergence.json)\n\n'
        '![CPU reference](stage_errors_cpu.png)\n\n'
        +('![Original CUDA reference](stage_errors_cuda.png)\n' if any('original_cuda_reference' in r for r in rows) else ''))
if __name__=='__main__':raise SystemExit(main())
