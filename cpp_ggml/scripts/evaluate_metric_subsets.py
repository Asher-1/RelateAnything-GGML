#!/usr/bin/env python3
"""Evaluate revision-pinned positive-only subsets with upstream SGClsEvaluator."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import numpy as np
import torch
from PIL import Image
from benchmark_full_graph import ROOT, MODELS, raw_python, load_cpp, parity, prepare_image
from provenance import recorded_run, model_assets, save


def tensor_output(raw):
    return {key: torch.from_numpy(np.asarray(raw[source])).unsqueeze(0).to(dtype)
            for key, source, dtype in [('logits','logits',torch.float32),
                ('pair_logits','pairs',torch.float32),('sub_idx','sub',torch.long),
                ('obj_idx','obj',torch.long),('valid_mask','valid',torch.bool)]}


def binary_cells(raws, samples, names):
    index = {name: i for i, name in enumerate(names)}
    scores, labels, predicates = [], [], []
    covered = 0
    for raw, sample in zip(raws, samples):
        pairs = {(int(raw['sub'][i]), int(raw['obj'][i])): i for i in np.flatnonzero(raw['valid'])}
        for cell in sample['cells']:
            row = pairs.get((cell['subject'], cell['object']))
            pred = index[cell['predicate']]
            if row is None:
                score = 0.0
            else:
                covered += 1
                logit = float(raw['logits'][row, pred])+float(raw['pairs'][row])
                score = float(1/(1+np.exp(-np.clip(logit, -80, 80))))
            scores.append(score)
            labels.append(int(cell['label']))
            predicates.append(pred)
    return scores, labels, predicates, covered/max(len(scores), 1)


def main():
    from relsgg import RelateAnything
    from relsgg.eval.evaluator import SGClsEvaluator
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'cpp_ggml/test_data/metric_subsets')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--models',nargs='+',default=list(MODELS))
    parser.add_argument('--backends',nargs='+',default=['cpu','cuda','vulkan'])
    parser.add_argument('--dtypes',nargs='+',default=['f32','f16','q8_0'])
    parser.add_argument('--datasets',nargs='+',default=['PSG','IndoorVG','VG150'])
    args=parser.parse_args();torch.set_num_threads(4)
    results=[]
    thresholds = {}
    for dataset in args.datasets:
        data=args.data/dataset
        if dataset.startswith('SpatialSense-'):
            data=args.data/'spatialsense'/dataset
        manifest=json.loads((data/'manifest.json').read_text())
        samples=manifest['samples'];cats=json.loads((data/'categories.json').read_text())
        names=[c['name'] for c in cats['rel_categories']];index={p:i for i,p in enumerate(names)}
        folder=args.output/dataset;folder.mkdir(parents=True,exist_ok=True)
        inputs=[];targets=[]
        for i,sample in enumerate(samples):
            image=data/sample['image']
            if sample['objects']:
                pixels,boxes=prepare_image(image,sample)
            else:
                pixels,_=prepare_image(image,{'objects':[{'bbox':[0,0,1,1]}]})
                boxes=np.zeros((0,4),dtype=np.float32)
            path=folder/f'{i:03d}.raim'
            path.write_bytes(struct.pack('<8sIII',b'RAIMv1\0\0',448,448,len(boxes))+
                             pixels.astype('<f4').tobytes()+boxes.astype('<f4').tobytes())
            inputs.append(path)
            targets.append({'relations':torch.tensor([[r['subject'],r['object'],index[r['predicate']]]
                                                     for r in sample['relations']],dtype=torch.long).reshape(-1,3)})
        for model in args.models:
            dst=folder/model;dst.mkdir(parents=True,exist_ok=True)
            assets=model_assets(model)
            with recorded_run(dst/'python',inputs=[data/'manifest.json',data/'categories.json',
                              *(data/s['image'] for s in samples),*assets],config={'device':'cpu',
                              'vocabulary':'dataset categories','graph_constraint':True,
                              'score_mode':'sigmoid','revision':manifest['revision']},script=__file__):
                api=RelateAnything.from_checkpoint(str(ROOT/'cpp_ggml/models/pytorch'/model/'model.pth'),
                                                  device='cpu',full_vocabulary=True)
                api.set_vocabulary(names)
                bank=dst/'python/vocabulary.f32';api.model.vocab_head.W.detach().numpy().astype('<f4').tofile(bank)
                save(dst/'python/vocabulary.json',names)
                captured={};hook=api.model.register_forward_hook(lambda m,i,o:captured.update(out=o))
                evaluator=SGClsEvaluator(num_predicates=len(names),score_mode='sigmoid',
                                         graph_constraint=True,contract=api.contract)
                references=[]
                for i,sample in enumerate(samples):
                    image=Image.open(data/sample['image']).convert('RGB')
                    xyxy=np.array([[x,y,x+w,y+h] for x,y,w,h in (o['bbox'] for o in sample['objects'])],np.float32).reshape(-1,4)
                    captured.clear()
                    api.predict(image,xyxy,topk=128)
                    if len(xyxy) < 2:
                        raw = {'logits':np.zeros((0,len(names)),np.float32),
                               'pairs':np.zeros(0,np.float32),
                               'sub':np.zeros(0,np.int64),'obj':np.zeros(0,np.int64),
                               'valid':np.zeros(0,bool)}
                    else:
                        raw=raw_python(captured['out'])
                    references.append(raw)
                    np.savez(dst/'python'/f'{i:03d}.npz',**raw)
                    evaluator.update(tensor_output(raw),[targets[i]])
                py_metrics=evaluator.compute();save(dst/'python/metrics.json',py_metrics)
                if dataset.startswith('SpatialSense-'):
                    from relsgg.eval.spatialsense import best_threshold, summarise
                    scores, labels, preds, coverage = binary_cells(references, samples, names)
                    if dataset.endswith('-valid'):
                        thresholds[(model, 'python')] = best_threshold(scores, labels)[0]
                    tau = thresholds[(model, 'python')]
                    py_metrics['binary_cells'] = summarise(scores, labels, preds, names, tau, coverage)
                    save(dst/'python/metrics.json', py_metrics)
                contract=api.contract;hook.remove();del api
            for backend in args.backends:
                for dtype in args.dtypes:
                    run=dst/f'{backend}-{dtype}';gguf=ROOT/f'cpp_ggml/models/gguf/{model}-{dtype}.gguf'
                    binary=ROOT/f'cpp_ggml/build-{backend}/relateanything-ggml'
                    evaluator=SGClsEvaluator(num_predicates=len(names),score_mode='sigmoid',
                                             graph_constraint=True,contract=contract)
                    rows=[];native_raws=[]
                    with recorded_run(run,inputs=[*inputs,gguf,bank,data/'manifest.json',
                                      dst/'python/run_provenance.json'],config={'strict_f32':True,
                                      'vocabulary':'dataset categories','graph_constraint':True},binary=binary):
                        for i,path in enumerate(inputs):
                            output=run/f'{i:03d}.rafo'
                            command=[str(binary),'--backend',backend,'--model',str(gguf),'--input',str(path),
                                     '--vocabulary',str(bank),'--output',str(output),'--raw-logits',
                                     '--strict-f32','--warmup','0','--repeats','1','--threads','8']
                            result=subprocess.run(command,capture_output=True,text=True)
                            (run/f'{i:03d}.log').write_text(result.stdout+result.stderr)
                            if result.returncode:raise RuntimeError(result.stderr[-2000:])
                            raw=load_cpp(output);evaluator.update(tensor_output(raw),[targets[i]])
                            native_raws.append(raw)
                            if np.any(references[i]['valid']) and np.any(raw['valid']):
                                comparison = parity(references[i], raw)
                            else:
                                if np.any(references[i]['valid']) != np.any(raw['valid']):
                                    raise RuntimeError('native/reference empty-pair mismatch')
                                comparison = {'pair_set_recall': 1.0,
                                              'pair_set_jaccard': 1.0,
                                              'logits_rmse': 0.0,
                                              'logits_max_abs': 0.0,
                                              'predicate_top1_agreement': 1.0}
                            rows.append({'image_id':samples[i]['image_id'],**comparison})
                        metrics=evaluator.compute()
                        if dataset.startswith('SpatialSense-'):
                            scores, labels, preds, coverage = binary_cells(native_raws, samples, names)
                            key = (model, backend, dtype)
                            if dataset.endswith('-valid'):
                                thresholds[key] = best_threshold(scores, labels)[0]
                            metrics['binary_cells'] = summarise(scores, labels, preds, names,
                                                               thresholds[key], coverage)
                        record={'dataset':dataset,'revision':manifest['revision'],'model':model,
                                'backend':backend,'dtype':dtype,'images':len(samples),
                                'metrics':metrics,'python_metrics':py_metrics,'parity':rows,
                                'scope':'first N original split rows, GT boxes, closed vocabulary, graph constrained; '
                                        +('verified binary cells, own valid-subset threshold' if dataset.startswith('SpatialSense-')
                                          else 'positive-only')}
                        save(run/'results.json',record)
                    results.append(record);save(args.output/'summary.json',results)
                    print(dataset,model,backend,dtype,metrics,flush=True)


if __name__=='__main__':main()
