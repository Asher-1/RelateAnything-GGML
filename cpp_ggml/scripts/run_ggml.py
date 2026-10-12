#!/usr/bin/env python3
"""Build and run the official full image graph, or the upstream Python API."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
MODELS = ('relsgg-vits16', 'relsgg-vits16plus', 'relsgg-vitb16')
DEMO_REQUESTS = {
    'horse': (ROOT / 'assets/reel/images/horse.jpg',
              [[470, 130, 650, 630], [90, 310, 1010, 875]],
              ['person', 'horse']),
}


def demo_request(name):
    """Return the maintained newcomer demo image and its pixel-XYXY boxes."""
    return DEMO_REQUESTS[name]


def select_quickstart_backend(available):
    """Use the fastest locally discoverable backend without requiring flags."""
    if available('nvidia-smi'):
        return 'cuda'
    if available('vulkaninfo'):
        return 'vulkan'
    return 'cpu'


def doctor_report(available=shutil.which):
    """Report launcher prerequisites without configuring or downloading anything."""
    tools = {name: bool(available(name)) for name in ('python3', 'cmake', 'c++', 'git')}
    tools.update({name: bool(available(name)) for name in ('nvidia-smi', 'vulkaninfo')})
    assets = {
        'horse_demo': DEMO_REQUESTS['horse'][0].is_file(),
        'default_test_data': (ROOT / 'cpp_ggml/test_data/ra4m_50/manifest.json').is_file(),
        'recommended_model_f16': (ROOT / 'cpp_ggml/models/gguf/relsgg-vits16plus-f16.gguf').is_file(),
    }
    return {'tools': tools, 'assets': assets,
            'recommended_backend': select_quickstart_backend(available),
            'quickstart': './run_ggml.sh --quickstart'}


def result_summary(output):
    """Produce a compact, human-readable summary of calibrated JSON output."""
    result = json.loads(output.read_text())
    predictions = sum(len(result.get(key, [])) for key in ('predictions', 'spatial', 'semantic'))
    latency = result.get('latency_ms')
    if latency is None:
        return f'完成：{predictions} 条关系'
    return f'完成：{predictions} 条关系，延迟 {float(latency):.2f} ms'


def run(command):
    print('$', ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True)


def input_box_count(path: Path) -> int:
    with path.open('rb') as stream:
        header = stream.read(20)
        if header[:8] in (b'RAIMv1\0\0',b'RAIMv2\0\0'):
            if len(header) != 20: raise ValueError('truncated RAIM header')
            return struct.unpack('<8sIII', header)[3]
        if header.startswith((b'RAIP1\n',b'RAIP2\n')):
            stream.seek(0)
            lines = stream.read().decode('utf-8').splitlines()
            if len(lines) < 3: raise ValueError('truncated RAIP input')
            count = int(lines[2])
            if count < 0: raise ValueError('negative RAIP box count')
            return count
    raise ValueError('input must be RAIP1/2 or RAIMv1/2')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', choices=('cpp','python'), default='cpp')
    p.add_argument('--backend', choices=('cpu','cuda','vulkan'))
    p.add_argument('--device-index', type=int, default=0)
    p.add_argument('--backend-profile', action='store_true')
    p.add_argument('--dtype', choices=('f32','f16','q8_0'))
    p.add_argument('--model-name', choices=MODELS)
    for name in ('model','input','output','image','build-dir','vocabulary-embeddings','masks','text-student','tokenizer'):
        p.add_argument('--'+name, type=Path)
    p.add_argument('--boxes-json', help='JSON file or JSON list of pixel [x1,y1,x2,y2] boxes')
    p.add_argument('--box-label', action='append')
    p.add_argument('--vocabulary', action='append', help='ordered predicate names; native student encodes strings when no embedding bank is supplied')
    p.add_argument('--box-scores-json',help='JSON file or list of detector confidences')
    p.add_argument('--topk',type=int,default=128)
    p.add_argument('--sample-index', type=int, default=0)
    p.add_argument('--warmup', type=int, default=2); p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--threads', type=int, default=0); p.add_argument('--max-boxes', type=int, default=60)
    for name in ('configure','build','self-test','no-flash-attention','no-compact-pairs','matmul-sampler','matmul-geometry','no-fused-vocabulary','no-cache-positions','no-retain-features','quant-f32-input','strict-f32','no-strict-f32','no-cuda-graphs','decompose','raw-logits','profile','download'):
        p.add_argument('--'+name, action='store_true')
    p.add_argument('--demo', choices=tuple(DEMO_REQUESTS), help='run a maintained image-and-box demo')
    p.add_argument('--quickstart', action='store_true', help='download and run the recommended demo with an auto-selected backend')
    p.add_argument('--doctor', action='store_true', help='print prerequisite and backend discovery information, then exit')
    args = p.parse_args()
    if args.doctor:
        print(json.dumps(doctor_report(), indent=2))
        return 0
    if args.quickstart:
        if args.input or args.image or args.boxes_json or args.demo:
            p.error('--quickstart cannot be combined with another input selection')
        args.demo, args.download = 'horse', True
        args.backend = args.backend or select_quickstart_backend(shutil.which)
        args.dtype = args.dtype or 'f16'
    args.backend = args.backend or 'cuda'
    args.dtype = args.dtype or 'f32'
    args.model_name = args.model_name or 'relsgg-vits16plus'
    if args.repeats < 1 or args.warmup < 0: p.error('invalid timing counts')
    if args.threads < 0 or args.max_boxes < 1: p.error('invalid threads/max-boxes')
    if args.engine == 'python' and args.dtype != 'f32': p.error('the official Python reference uses F32 weights')
    if bool(args.image) != bool(args.boxes_json): p.error('--image and --boxes-json must be supplied together')
    if args.input and args.image: p.error('--input and --image are mutually exclusive')
    if args.demo and (args.input or args.image or args.boxes_json): p.error('--demo selects its own image and boxes')
    if args.input and (args.masks or args.box_scores_json):p.error('mask/scores are supplied by --input request')
    output = (args.output or ROOT/'cpp_ggml/outputs/probes/inference'/f'{args.model_name}-{args.engine}-{args.backend}.{ "rafo" if args.raw_logits else "json"}').resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.input or args.self_test:
        image=None; boxes=[]
    elif args.demo:
        image, boxes, demo_labels = demo_request(args.demo)
        if not image.is_file():
            p.error(f'demo image is unavailable: {image}; restore the repository assets')
        if args.box_label is None:
            args.box_label = demo_labels
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
    try:
        box_count = input_box_count(args.input) if args.input and not args.self_test else len(boxes)
    except (OSError, UnicodeError, ValueError) as exc:
        p.error(f'invalid --input: {exc}')
    if args.box_label and len(args.box_label) != box_count: p.error('box label count must match boxes')
    if args.input and args.box_label and box_count > args.max_boxes:
        p.error('labels for truncated --input requests require preselected boxes')
    scores=None
    if args.box_scores_json:
        value=args.box_scores_json
        scores=json.loads(value if value.lstrip().startswith('[') else Path(value).read_text())
        if len(scores)!=box_count or any(not 0<=v<=1 for v in scores):p.error('box scores must match boxes and be in [0,1]')
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
            with Image.open(image) as im:
                masks=np.fromfile(args.masks,np.uint8).reshape(box_count,im.height,im.width) if args.masks else None
                predictions=model.predict(im.convert('RGB'),np.asarray(boxes,np.float32),topk=args.topk,
                    max_boxes=args.max_boxes,box_labels=args.box_label,masks=masks,
                    box_scores=None if scores is None else np.asarray(scores,np.float32),decompose=args.decompose)
            if args.backend == 'cuda': torch.cuda.synchronize()
            if i>=args.warmup: times.append((time.perf_counter()-start)*1000)
        def encode(rows):return [
            {'subject':r.subject_idx,'object':r.object_idx,'predicate':r.predicate,'score':float(r.score),
             'subject_label':r.subject_label,'object_label':r.object_label} for r in rows]
        result={'iterations_ms':times,'latency_ms':float(np.median(times))}
        if args.decompose:result.update({key:encode(rows) for key,rows in predictions.items()})
        else:result['predictions']=encode(predictions)
        output.write_text(json.dumps(result,indent=2)+'\n'); print(output); print(result_summary(output)); return 0
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
        downloaded=hf_hub_download(repo_id='Asher-1/relateanything-ggml-full',filename='gguf/'+model.name)
        model.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(downloaded,model)
    if not model.exists(): raise SystemExit(f'Missing model: {model}. Download links: cpp_ggml/models/MODEL_CARDS.md')
    cmd+=['--model',model,'--output',output,'--warmup',args.warmup,'--repeats',args.repeats,'--threads',args.threads,'--max-boxes',args.max_boxes,'--topk',args.topk,'--device-index',args.device_index]
    if args.backend_profile: cmd+=['--backend-profile']
    for flag in ('no_flash_attention','no_compact_pairs','matmul_sampler','matmul_geometry','no_fused_vocabulary','no_cache_positions','no_retain_features','quant_f32_input','strict_f32','no_strict_f32','no_cuda_graphs','decompose','raw_logits','profile'):
        if getattr(args,flag): cmd+=['--'+flag.replace('_','-')]
    if args.vocabulary_embeddings: cmd+=['--vocabulary',args.vocabulary_embeddings.resolve()]
    with tempfile.TemporaryDirectory(prefix='relateanything-') as temp:
        input_path=args.input.resolve() if args.input else Path(temp)/'image.raip'
        if args.vocabulary and not args.vocabulary_embeddings:
            texts=Path(temp)/'texts.json';texts.write_text(json.dumps(args.vocabulary))
            student=(args.text_student or ROOT/f'cpp_ggml/models/gguf/{args.model_name}-text-student-f32.gguf').resolve()
            tokenizer=(args.tokenizer or ROOT/f'cpp_ggml/models/pytorch/{args.model_name}/tokenizer.json').resolve()
            if not student.exists() and args.download:
                from huggingface_hub import hf_hub_download
                downloaded=hf_hub_download(repo_id='Asher-1/relateanything-ggml-full',filename='gguf/'+student.name)
                student.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(downloaded,student)
            if not tokenizer.exists() and args.download:
                from huggingface_hub import hf_hub_download
                downloaded=hf_hub_download(repo_id='maelic/'+args.model_name,filename='tokenizer.json')
                tokenizer.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(downloaded,tokenizer)
            cmd+=['--texts',texts,'--text-student',student,'--tokenizer',tokenizer]
        elif args.vocabulary:
            names=Path(temp)/'names.json';names.write_text(json.dumps(args.vocabulary));cmd+=['--vocabulary-names',names]
        if not args.input:
            request=['RAIP2' if args.masks or scores is not None else 'RAIP1',json.dumps(str(image)),str(len(boxes))]
            request+=[' '.join(map(str,b)) for b in boxes]
            if args.masks or scores is not None:
                request += [str(int(scores is not None))]
                if scores is not None:request+=[' '.join(map(str,scores))]
                request += [json.dumps(str(args.masks.resolve()) if args.masks else '')]
            input_path.write_text('\n'.join(request)+'\n')
        run(cmd+['--input',input_path])
    if (args.vocabulary or args.box_label) and not args.raw_logits:
        labels=args.box_label
        if labels and box_count > args.max_boxes:
            if scores is None:labels=labels[:args.max_boxes]
            else:
                import numpy as np
                labels=[labels[i] for i in np.argsort(-np.asarray(scores,np.float32))[:args.max_boxes]]
        paths=sorted(output.glob('*.json')) if output.is_dir() else [output]
        for path in paths:
            result=json.loads(path.read_text())
            for row in [r for key in ('predictions','spatial','semantic') for r in result.get(key,[])]:
                if args.vocabulary: row['predicate']=args.vocabulary[row['predicate_id']]
                if labels:
                    row['subject_label']=labels[row['subject']]
                    row['object_label']=labels[row['object']]
            path.write_text(json.dumps(result,indent=2)+'\n')
    if not output.exists():
        raise SystemExit(f'推理未生成输出：{output}')
    print(output)
    if output.suffix == '.json':
        print(result_summary(output))
    return 0

if __name__=='__main__': raise SystemExit(main())
