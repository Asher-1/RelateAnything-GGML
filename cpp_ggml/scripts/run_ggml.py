#!/usr/bin/env python3
"""Build and run the official full image graph, or the upstream Python API."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
MODELS = ('relsgg-vits16', 'relsgg-vits16plus', 'relsgg-vitb16')


def run(command):
    print('$', ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', choices=('cpp','python'), default='cpp')
    p.add_argument('--backend', choices=('cpu','cuda','vulkan'), default='cuda')
    p.add_argument('--dtype', choices=('f32','f16','q8_0'), default='f32')
    p.add_argument('--model-name', choices=MODELS, default='relsgg-vits16plus')
    for name in ('model','input','output','image','build-dir','vocabulary-embeddings'):
        p.add_argument('--'+name, type=Path)
    p.add_argument('--boxes-json', help='JSON file or JSON list of pixel [x1,y1,x2,y2] boxes')
    p.add_argument('--box-label', action='append')
    p.add_argument('--vocabulary', action='append', help='ordered predicate names; C++ also requires --vocabulary-embeddings')
    p.add_argument('--sample-index', type=int, default=0)
    p.add_argument('--warmup', type=int, default=2); p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--threads', type=int, default=0); p.add_argument('--max-boxes', type=int, default=60)
    for name in ('configure','build','self-test','no-flash-attention','raw-logits','profile','download'):
        p.add_argument('--'+name, action='store_true')
    args = p.parse_args()
    if args.repeats < 1 or args.warmup < 0: p.error('invalid timing counts')
    if args.threads < 0 or args.max_boxes < 1: p.error('invalid threads/max-boxes')
    if args.engine == 'python' and args.dtype != 'f32': p.error('the official Python reference uses F32 weights')
    if bool(args.image) != bool(args.boxes_json): p.error('--image and --boxes-json must be supplied together')
    if args.input and args.image: p.error('--input and --image are mutually exclusive')
    if args.vocabulary and args.engine == 'cpp' and not args.vocabulary_embeddings:
        p.error('C++ accepts precomputed text embeddings with --vocabulary-embeddings')
    output = (args.output or ROOT/'cpp_ggml/benchmarks/inference'/f'{args.model_name}-{args.engine}-{args.backend}.{ "rafo" if args.raw_logits else "json"}').resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.input or args.self_test:
        image=None; boxes=[]
    elif args.image:
        image = args.image.resolve()
        value = args.boxes_json
        boxes = json.loads(value if value.lstrip().startswith('[') else Path(value).read_text())
    else:
        data = ROOT/'cpp_ggml/test_data/ra4m_50'
        samples = json.loads((data/'manifest.json').read_text())['samples']
        if not 0 <= args.sample_index < len(samples): p.error('--sample-index is out of range')
        sample = samples[args.sample_index]
        image = data/sample['image']
        boxes = [[x,y,x+w,y+h] for x,y,w,h in (o['bbox'] for o in sample['objects'])]
    if any(len(b) != 4 for b in boxes): p.error('every box needs four coordinates')
    if not args.input and len(boxes) > args.max_boxes: p.error('box count exceeds --max-boxes')
    if args.box_label and len(args.box_label) != len(boxes): p.error('box label count must match boxes')
    if args.vocabulary_embeddings and args.vocabulary:
        if args.vocabulary_embeddings.stat().st_size != len(args.vocabulary)*512*4:
            p.error('vocabulary names must match the [V,512] F32 embedding bank')
    if args.engine == 'python':
        if args.backend == 'vulkan': p.error('upstream Python supports cpu/cuda')
        if args.input or args.self_test or args.model or args.raw_logits: p.error('input/model/self-test/raw-logits are C++ options')
        import numpy as np
        import torch
        if args.threads: torch.set_num_threads(args.threads)
        from PIL import Image
        sys.path.insert(0,str(ROOT))
        from relsgg import RelateAnything
        model = RelateAnything.from_checkpoint(str(ROOT/'cpp_ggml/models/pytorch'/args.model_name/'model.pth'), device=args.backend, full_vocabulary=True)
        if args.vocabulary_embeddings:
            embeddings = np.fromfile(args.vocabulary_embeddings, dtype='<f4').reshape(-1,512)
            model.set_vocabulary(args.vocabulary or [str(i) for i in range(len(embeddings))],embeddings=embeddings)
        elif args.vocabulary: model.set_vocabulary(args.vocabulary)
        times=[]
        for i in range(args.warmup+args.repeats):
            if args.backend == 'cuda': torch.cuda.synchronize()
            start=time.perf_counter()
            with Image.open(image) as im: predictions=model.predict(im.convert('RGB'),np.asarray(boxes,np.float32),topk=128,max_boxes=args.max_boxes,box_labels=args.box_label)
            if args.backend == 'cuda': torch.cuda.synchronize()
            if i>=args.warmup: times.append((time.perf_counter()-start)*1000)
        result={'iterations_ms':times,'latency_ms':float(np.median(times)), 'predictions':[
            {'subject':r.subject_idx,'object':r.object_idx,'predicate':r.predicate,'score':float(r.score),
             'subject_label':r.subject_label,'object_label':r.object_label} for r in predictions]}
        output.write_text(json.dumps(result,indent=2)+'\n'); print(output); return 0
    build=(args.build_dir or ROOT/f'cpp_ggml/build-{args.backend}').resolve()
    if args.configure or not (build/'CMakeCache.txt').exists():
        run(['cmake','-S',ROOT/'cpp_ggml','-B',build,'-DCMAKE_BUILD_TYPE=Release',
             f'-DRA_GGML_CUDA={"ON" if args.backend=="cuda" else "OFF"}',
             f'-DRA_GGML_VULKAN={"ON" if args.backend=="vulkan" else "OFF"}'])
    if args.build or not (build/'relateanything-ggml').exists(): run(['cmake','--build',build,'--parallel','8'])
    cmd=[build/'relateanything-ggml','--backend',args.backend]
    if args.self_test: run(cmd+['--self-test']); return 0
    model=(args.model or ROOT/'cpp_ggml/models/gguf'/f'{args.model_name}-{args.dtype}.gguf').resolve()
    if not model.exists() and args.download:
        from huggingface_hub import hf_hub_download
        import shutil
        downloaded=hf_hub_download(repo_id='Asher-1/relateanything-ggml-full',filename='gguf/'+model.name)
        model.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(downloaded,model)
    if not model.exists(): raise SystemExit(f'Missing model: {model}. Download links: cpp_ggml/models/MODEL_CARDS.md')
    cmd+=['--model',model,'--output',output,'--warmup',args.warmup,'--repeats',args.repeats,'--threads',args.threads,'--max-boxes',args.max_boxes]
    for flag in ('no_flash_attention','raw_logits','profile'):
        if getattr(args,flag): cmd+=['--'+flag.replace('_','-')]
    if args.vocabulary_embeddings: cmd+=['--vocabulary',args.vocabulary_embeddings.resolve()]
    with tempfile.TemporaryDirectory(prefix='relateanything-') as temp:
        input_path=args.input.resolve() if args.input else Path(temp)/'image.raip'
        if not args.input: input_path.write_text('\n'.join(['RAIP1',json.dumps(str(image)),str(len(boxes))]+[' '.join(map(str,b)) for b in boxes])+'\n')
        run(cmd+['--input',input_path])
    if (args.vocabulary or args.box_label) and not args.raw_logits:
        paths=sorted(output.glob('*.json')) if output.is_dir() else [output]
        for path in paths:
            result=json.loads(path.read_text())
            for row in result['predictions']:
                if args.vocabulary: row['predicate']=args.vocabulary[row['predicate_id']]
                if args.box_label:
                    row['subject_label']=args.box_label[row['subject']]
                    row['object_label']=args.box_label[row['object']]
            path.write_text(json.dumps(result,indent=2)+'\n')
    print(output)
    return 0

if __name__=='__main__': raise SystemExit(main())
