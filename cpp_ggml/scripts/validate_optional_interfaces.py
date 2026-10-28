#!/usr/bin/env python3
"""Compare the upstream mask, score, decomposition and free-text interfaces."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import numpy as np
import torch
from PIL import Image
from benchmark_full_graph import ROOT, MODELS, load_cpp, parity, raw_python, save
from provenance import recorded_run, model_assets


def execute(command, folder, inputs):
    with recorded_run(folder,inputs=inputs,config={'command':command},binary=Path(command[0])):
        result=subprocess.run(command,capture_output=True,text=True)
        (folder/'run.log').write_text(result.stdout+result.stderr)
        if result.returncode: raise RuntimeError(result.stderr[-3000:])


def compare_triplet_streams(reference, native):
    identity=lambda row:(row['subject'],row['object'],row['predicate'])
    reference_ids=[identity(row) for row in reference]
    native_ids=[identity(row) for row in native]
    reference_scores={identity(row):row['score'] for row in reference}
    native_scores={identity(row):row['score'] for row in native}
    if len(reference_scores)!=len(reference) or len(native_scores)!=len(native):
        raise RuntimeError('duplicate triplet identity')
    same_set=reference_scores.keys()==native_scores.keys()
    return {'triplet_set_equal':same_set,
            'ranked_triplets_equal':reference_ids==native_ids,
            'max_score_error':max((abs(reference_scores[key]-native_scores[key])
                                   for key in reference_scores),default=0.0) if same_set else None}


def main():
    from relsgg import RelateAnything
    from relsgg.text.student import PredicateTextStudent, encode_texts_student
    from relsgg.vocabulary import TRAIN_TEMPLATES
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backends',nargs='+',default=['cpu','cuda','vulkan'])
    parser.add_argument('--models',nargs='+',default=list(MODELS))
    precision=parser.add_mutually_exclusive_group()
    precision.add_argument('--strict-f32',action='store_true')
    precision.add_argument('--no-strict-f32',action='store_true')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    data=ROOT/'cpp_ggml/test_data/ra4m_50'
    sample=json.loads((data/'manifest.json').read_text())['samples'][0]
    image=Image.open(data/sample['image']).convert('RGB')
    boxes=np.array([[x,y,x+w,y+h] for x,y,w,h in (o['bbox'] for o in sample['objects'])],np.float32)
    scores=np.linspace(.19,.97,len(boxes),dtype=np.float32)
    yy,xx=np.mgrid[:image.height,:image.width]
    masks=np.array([(xx>=b[0])&(xx<b[2])&(yy>=b[1])&(yy<b[3])&((xx+yy+i)%3!=0)
                    for i,b in enumerate(boxes)],np.uint8)
    masks.tofile(args.output/'masks.u8')
    request=['RAIP2',json.dumps(str((data/sample['image']).resolve())),str(len(boxes))]
    request += [' '.join(map(str,b)) for b in boxes]
    request += ['1',' '.join(map(str,scores)),json.dumps(str((args.output/'masks.u8').resolve()))]
    (args.output/'regions.raip').write_text('\n'.join(request)+'\n')
    texts=['above','below','holding','not touching','a'*150,'cafe\u0301','\u4e0a\u9762','&amp;',
           '<|endoftext|>','!',"isn't near",'A\tB\nC','\u03a3\u039f\u03a3','']
    save(args.output/'texts.json',texts)
    results=[]
    for model in args.models:
        assets=ROOT/'cpp_ggml/models/pytorch'/model
        student=PredicateTextStudent.from_checkpoint(str(assets/'text_student.pt'),device='cpu')
        expected_tokens=student._ensure_tokenizer()(texts,truncation=True,max_length=32)['input_ids']
        refs={False:student.encode_texts(texts).numpy(),
              True:encode_texts_student(texts,str(assets/'text_student.pt'),templates=TRAIN_TEMPLATES,device='cpu').numpy()}
        api=RelateAnything.from_checkpoint(str(assets/'model.pth'),device='cpu',full_vocabulary=True)
        captured={}; hook=api.model.register_forward_hook(lambda m,i,o:captured.update(out=o))
        streams=api.predict(image,boxes,masks=masks,box_scores=scores,decompose=True,topk=20)
        raw=raw_python(captured['out']);hook.remove()
        reference={tag:[{'subject':p.subject_idx,'object':p.object_idx,'predicate':p.predicate,'score':p.score}
                        for p in rows] for tag,rows in streams.items()}
        selected_streams=api.predict(image,boxes,masks=masks,box_scores=scores,decompose=True,topk=20,max_boxes=5)
        selected_reference={tag:[{'subject':p.subject_idx,'object':p.object_idx,'predicate':p.predicate,'score':p.score}
                                for p in rows] for tag,rows in selected_streams.items()}
        cov,fill=api._rasterize(masks,boxes,image.width,image.height)
        np.savez(args.output/(model+'-reference.npz'),**raw,coverage=cov.numpy(),fill=fill.numpy())
        for backend in args.backends:
            folder=args.output/model/backend;folder.mkdir(parents=True,exist_ok=True)
            binary=ROOT/f'cpp_ggml/build-{backend}/relateanything-ggml'
            base=[str(binary),'--backend',backend,'--threads','8','--warmup','0','--repeats','1']
            if args.strict_f32:base.append('--strict-f32')
            if args.no_strict_f32:base.append('--no-strict-f32')
            command=base+['--input',str(args.output/'regions.raip'),'--model',str(ROOT/f'cpp_ggml/models/gguf/{model}-f32.gguf'),
                         '--output',str(folder/'regions.json'),'--decompose','--topk','20']
            with recorded_run(folder,inputs=[args.output/'regions.raip',args.output/'masks.u8',data/sample['image'],
                              *model_assets(model),args.output/'texts.json',
                              ROOT/f'cpp_ggml/models/gguf/{model}-text-student-f32.gguf',
                              Path(command[command.index('--model')+1])],config={'command':command},binary=binary,script=__file__):
                for extra,dest in [([],folder/'regions.json'),(['--raw-logits'],folder/'regions.rafo')]:
                    cmd=list(command);cmd[cmd.index('--output')+1]=str(dest)
                    run=subprocess.run(cmd+extra,capture_output=True,text=True)
                    (folder/(dest.suffix[1:]+'.log')).write_text(run.stdout+run.stderr)
                    if run.returncode:raise RuntimeError(run.stderr[-3000:])
                pre=base+['--preprocess-only','--input',str(args.output/'regions.raip'),'--output',str(folder/'regions.raim')]
                subprocess.run(pre,check=True,capture_output=True)
                blob=(folder/'regions.raim').read_bytes(); magic,w,h,n,flags=struct.unpack_from('<8sIIII',blob)
                if magic!=b'RAIMv2\0\0' or flags!=7:raise RuntimeError('optional region serialization mismatch')
                values=np.frombuffer(blob,'<f4',offset=24+4*(3*448*448+4*n))
                np.testing.assert_array_equal(values[:n*1024].reshape(cov.shape),cov.numpy())
                np.testing.assert_array_equal(values[n*1024:n*1024+n].reshape(fill.shape),fill.numpy())
                native=json.loads((folder/'regions.json').read_text())
                record={'model':model,'backend':backend,'masks':parity(raw,load_cpp(folder/'regions.rafo')),'streams':{}}
                for tag in ('spatial','semantic'):
                    record['streams'][tag]=compare_triplet_streams(reference[tag],native[tag])
                limited=list(command)+['--max-boxes','5']
                limited[limited.index('--output')+1]=str(folder/'selected.json')
                run=subprocess.run(limited,capture_output=True,text=True)
                if run.returncode:raise RuntimeError(run.stderr[-3000:])
                selected=json.loads((folder/'selected.json').read_text())
                record['selected_streams']={}
                for tag in ('spatial','semantic'):
                    comparison=compare_triplet_streams(selected_reference[tag],selected[tag])
                    if not comparison['triplet_set_equal']:
                        raise RuntimeError('score/mask region selection triplet-set mismatch')
                    record['selected_streams'][tag]=comparison
                record['text']={}
                for ensemble in (False,True):
                    prefix='ensemble' if ensemble else 'single'
                    cmd=base+['--encode-text-only','--text-student',str(ROOT/f'cpp_ggml/models/gguf/{model}-text-student-f32.gguf'),
                         '--tokenizer',str(assets/'tokenizer.json'),'--texts',str(args.output/'texts.json'),
                         '--output',str(folder/(prefix+'.f32')),'--token-output',str(folder/(prefix+'-tokens.json'))]
                    if not ensemble:cmd+=['--text-template','{p}']
                    run=subprocess.run(cmd,capture_output=True,text=True)
                    (folder/(prefix+'.log')).write_text(run.stdout+run.stderr)
                    if run.returncode:raise RuntimeError(run.stderr[-3000:])
                    tokens=json.loads((folder/(prefix+'-tokens.json')).read_text())
                    wanted=expected_tokens if not ensemble else [student._ensure_tokenizer()([t.format(p=p)],truncation=True,max_length=32)['input_ids'][0]
                             for p in texts for t in TRAIN_TEMPLATES]
                    if tokens!=wanted:raise RuntimeError(f'actual tokenizer ID mismatch: {model}/{backend}/{prefix}')
                    actual=np.fromfile(folder/(prefix+'.f32'),'<f4').reshape(refs[ensemble].shape)
                    error=actual.astype(float)-refs[ensemble]
                    record['text'][prefix]={'token_ids_equal':True,'rmse':float(np.sqrt(np.mean(error**2))),
                                           'max_abs':float(np.max(abs(error)))}
                save(folder/'comparison.json',record)
            results.append(record);save(args.output/'results.json',results)
            print(model,backend,record,flush=True)


if __name__=='__main__':main()
