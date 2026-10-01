#!/usr/bin/env python3
"""Separate weight rounding from native graph arithmetic on every retained image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark_full_graph import (ROOT, MODELS, DTYPES, decode, gt_metrics,
                                  load_cpp, parity, raw_python, save, sha256)
from provenance import recorded_run, validate_run, model_assets


def restore_weights(api, path):
    import torch
    from gguf import GGUFReader
    from gguf.quants import dequantize

    reader = GGUFReader(path)
    aliases = reader.fields.get('ra.tensor_name_map')
    names = aliases.contents() if aliases else []
    names = dict(zip(names[1::2], names[::2]))
    state = api.model.state_dict()
    changed = total = 0
    squared_error = squared_source = 0.0
    restored = set()
    with torch.no_grad():
        for tensor in reader.tensors:
            name = names.get(tensor.name, tensor.name)[3:]
            target = state[name]
            value = dequantize(tensor.data, tensor.tensor_type).reshape(tuple(target.shape))
            source = target.detach().cpu().float().numpy()
            error = value.astype(np.float64) - source
            changed += int(np.count_nonzero(error))
            total += value.size
            squared_error += float(np.sum(error * error))
            squared_source += float(np.sum(source.astype(np.float64) ** 2))
            target.copy_(torch.from_numpy(value.copy()).to(target.device))
            restored.add(name)
    if restored != set(state):
        raise RuntimeError(f'incomplete oracle weights: {sorted(set(state) - restored)}')
    return {'gguf_sha256': sha256(path), 'tensor_count': len(restored),
            'weight_values': total, 'changed_weight_values': changed,
            'weight_relative_l2_error': (squared_error / squared_source) ** .5}


def identity(reference, candidate, calibration=None):
    result = parity(reference, candidate)
    def indices(out):
        return {(int(out['sub'][i]), int(out['obj'][i])): i
                for i in np.flatnonzero(out['valid'])}
    a, b = indices(reference), indices(candidate)
    common = sorted(a.keys() & b.keys())
    x = reference['logits'][[a[k] for k in common]]
    y = candidate['logits'][[b[k] for k in common]]
    # Compare IEEE-754 representations, including signed zeros.
    equal = x.view(np.uint32) == y.view(np.uint32)
    top = np.argpartition(x, -2, axis=1)[:, -2:]
    best_two = np.take_along_axis(x, top, axis=1)
    margin = np.abs(best_two[:, 0] - best_two[:, 1])
    mismatched = x.argmax(1) != y.argmax(1)
    error = np.max(np.abs(y.astype(np.float64) - x), axis=1)
    result.update(logit_bit_equal_fraction=float(equal.mean()),
                  pair_logit_bit_equal_fraction=float(np.mean(
                      reference['pairs'][[a[k] for k in common]].view(np.uint32)==
                      candidate['pairs'][[b[k] for k in common]].view(np.uint32))),
                  predicate_mismatches=int(mismatched.sum()),
                  minimum_top1_margin=float(margin.min()),
                  maximum_changed_top1_margin=float(margin[mismatched].max()) if mismatched.any() else 0.,
                  stable_margin_rows=int(np.sum(margin > 2 * error)),
                  stable_margin_mismatches=int(np.sum(mismatched & (margin > 2 * error))))
    if calibration is not None:
        def scores(raw, ids):
            logit = raw['logits'][ids].max(1)+raw['pairs'][ids]
            z = logit.astype(np.float64)*calibration['a']+calibration['b']
            return 1/(1+np.exp(-np.clip(z, -80, 80)))
        delta = scores(candidate, [b[k] for k in common])-scores(reference, [a[k] for k in common])
        result['calibrated_score_rmse'] = float(np.sqrt(np.mean(delta**2)))
    return result


def main():
    import torch
    from relsgg import RelateAnything

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, default=(ROOT/'cpp_ggml/benchmarks/current').resolve().parent/'artifacts/full_graph')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--data', type=Path, default=ROOT/'cpp_ggml/test_data/ra4m_50')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--dtypes', nargs='+', choices=DTYPES, default=list(DTYPES))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--reuse-oracles', action='store_true', help='reuse stored Python tensors after native-only changes')
    parser.add_argument('--cuda-variant', default='', choices=('', '-f32-input'))
    parser.add_argument('--oracle-folder', type=Path, help='read existing independent oracles without overwriting their results')
    args = parser.parse_args()
    args.output = args.output or args.benchmark/'quantization_audit'
    samples = json.loads((args.data/'manifest.json').read_text())['samples']
    torch.set_num_threads(4)
    summaries = []
    for model in args.models:
        for dtype in args.dtypes:
            folder = args.output/model/dtype
            checkpoint = ROOT/f'cpp_ggml/models/gguf/{model}-{dtype}.gguf'
            folder.mkdir(parents=True, exist_ok=True)
            oracle_folder=(args.oracle_folder/model/dtype/'oracle') if args.oracle_folder else folder/'oracle'
            source=ROOT/'cpp_ggml/scripts/benchmark_full_graph.py'
            validate_run(args.benchmark/model/'python',script=source)
            if args.reuse_oracles:
                validate_run(oracle_folder,script=__file__)
                previous = json.loads((oracle_folder/'metadata.json').read_text())
                weights = previous['weights']
                if weights['gguf_sha256'] != sha256(checkpoint) or previous['device'] != args.device:
                    raise RuntimeError('stored oracle has different weights or reference device')
            else:
                with recorded_run(oracle_folder,inputs=[checkpoint,*model_assets(model),args.data/'manifest.json',
                                  *(args.data/s['image'] for s in samples)],
                                  config={'model':model,'dtype':dtype,'device':args.device,
                                          'torch_version':torch.__version__},script=__file__):
                    api = RelateAnything.from_checkpoint(
                        str(ROOT/'cpp_ggml/models/pytorch'/model/'model.pth'),
                        device=args.device, full_vocabulary=True)
                    weights = restore_weights(api, checkpoint)
                    captured = {}
                    handle = api.model.register_forward_hook(lambda m, i, o: captured.update(out=o))
                    for index,sample in enumerate(samples):
                        name=f'{index:03d}_{sample["image_id"]}'
                        boxes=np.array([[x,y,x+w,y+h] for x,y,w,h in
                                        (o['bbox'] for o in sample['objects'])],np.float32)
                        with Image.open(args.data/sample['image']) as image:
                            api.predict(image.convert('RGB'),boxes,topk=128)
                        np.savez(oracle_folder/(name+'.npz'),**raw_python(captured['out']))
                    handle.remove()
                    save(oracle_folder/'metadata.json',{'weights':weights,'device':args.device})
                    del api
                    if args.device=='cuda':torch.cuda.empty_cache()
            builds={}
            native_folders={}
            for backend in ('cpu','cuda','vulkan'):
                suffix=args.cuda_variant if backend=='cuda' else ''
                native_folders[backend]=args.benchmark/model/f'{backend}-{dtype}{suffix}'
            if args.cuda_variant:native_folders['cuda_default']=args.benchmark/model/f'cuda-{dtype}'
            for tag,native_folder in native_folders.items():
                binary=ROOT/f'cpp_ggml/build-{tag.split("_")[0]}/relateanything-ggml'
                builds[tag]=validate_run(native_folder,binary=binary)['build']
            analysis_inputs=[checkpoint,oracle_folder/'run_provenance.json',
                             args.benchmark/model/'python/run_provenance.json',
                             *(p/'run_provenance.json' for p in native_folders.values())]
            with recorded_run(folder,inputs=analysis_inputs,config={'cuda_variant':args.cuda_variant},script=__file__):
                result,entry=compare_outputs(args,samples,model,dtype,oracle_folder,weights,builds)
                save(folder/'results.json',result)
            summaries.append(entry)
            save(args.output/'summary.json', summaries)
            print(model, dtype, 'weight RMSE', entry['weight_conversion']['logits_rmse'], flush=True)
    plot_report(args,summaries)


def compare_outputs(args,samples,model,dtype,oracle_folder,weights,builds):
            import torch
            rows = []
            vocab = json.loads((args.benchmark/model/'python/vocabulary.json').read_text())
            for index, sample in enumerate(samples):
                name = f'{index:03d}_{sample["image_id"]}'
                with np.load(oracle_folder/(name+'.npz')) as saved:
                    oracle = {key: saved[key] for key in saved.files}
                with np.load(args.benchmark/model/'python'/(name+'.npz')) as original:
                    row = {'sample': name, 'weight_conversion': identity(original, oracle, vocab['calibration'])}
                pair_ids = lambda raw: {(int(raw['sub'][i]),int(raw['obj'][i])) for i in np.flatnonzero(raw['valid'])}
                row['candidate_changes']={}
                for backend in ('cpu','cuda','vulkan'):
                    suffix = args.cuda_variant if backend == 'cuda' else ''
                    native = load_cpp(args.benchmark/model/f'{backend}-{dtype}{suffix}'/(name+'.rafo'))
                    row[backend+'_same_weights'] = identity(oracle, native, vocab['calibration'])
                    row['candidate_changes'][backend]={'missing':sorted(pair_ids(oracle)-pair_ids(native)),
                                                       'extra':sorted(pair_ids(native)-pair_ids(oracle))}
                if args.cuda_variant:
                    baseline=load_cpp(args.benchmark/model/f'cuda-{dtype}'/(name+'.rafo'))
                    row['cuda_default_same_weights']=identity(oracle,baseline,vocab['calibration'])
                predictions = decode(oracle, vocab['names'], vocab['calibration'])
                row['oracle_gt'] = gt_metrics(predictions, sample)
                rows.append(row)
            result = {'model': model, 'dtype': dtype, 'device': args.device,
                      'cuda_variant': args.cuda_variant,
                      'native_builds': builds,
                      'torch_version': torch.__version__, 'weights': weights, 'rows': rows,
                      'scope': 'official PyTorch graph; all stored GGUF tensors dequantized to F32; untimed precision oracle'}
            entry = {'model': model, 'dtype': dtype, **weights}
            from benchmark_full_graph import summary
            task=summary({'backend':'pytorch-same-weights','dtype':dtype,
                 'rows':[{'latency_ms':1,**r['oracle_gt']} for r in rows]})
            entry['oracle_metrics']={k:v for k,v in task.items()
                                    if k.startswith(('R@','mR@','F1@','wR@')) or k in ('gt_relations','classes')}
            entry['worst_images']={key:sorted([{'sample':r['sample'],**r[key]} for r in rows],
                                              key=lambda r:r['logits_rmse'],reverse=True)[:5]
                                   for key in ('weight_conversion','cpu_same_weights','cuda_same_weights','vulkan_same_weights')}
            keys=['weight_conversion', 'cpu_same_weights', 'cuda_same_weights', 'vulkan_same_weights']
            if args.cuda_variant:keys.append('cuda_default_same_weights')
            for key in keys:
                entry[key] = {metric: float(np.mean([r[key][metric] for r in rows]))
                              for metric in rows[0][key]}
            return result,entry


def plot_report(args,summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    x = np.arange(len(summaries))
    labels = [s['model'].removeprefix('relsgg-')+'/'+s['dtype'] for s in summaries]
    plots=[('weight_conversion', 'Stored weights in PyTorch', '#777777'),
           ('cuda_same_weights', 'CUDA F32 activation' if args.cuda_variant else 'CUDA same weights', '#087e8b'),
           ('vulkan_same_weights', 'Vulkan same weights', '#b54f67')]
    if args.cuda_variant:plots.append(('cuda_default_same_weights', 'CUDA default Q8 activation', '#b86b35'))
    width=.8/len(plots)
    for i,(key,title,color) in enumerate(plots):
        offset=(i-(len(plots)-1)/2)*width
        axes[0].bar(x+offset, [s[key]['logits_rmse'] for s in summaries], width, label=title, color=color)
        axes[1].bar(x+offset, [100*(1-s[key]['predicate_top1_agreement']) for s in summaries], width, color=color)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=40, ha='right'); ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Mean per-image raw logit RMSE'); axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Mean per-image predicate top-1 changes (%)')
    fig.savefig(args.output/'error_sources.png', dpi=160); plt.close(fig)
    lines = ['# Weight rounding and native arithmetic', '',
             'All 50 images use the official Python graph with every tensor replaced by its stored GGUF value. '
             'This untimed F32 arithmetic oracle separates weight rounding from native arithmetic; '
             'it is not a Python mixed-precision speed baseline.', '',
             '![Separated weight and arithmetic errors](error_sources.png)', '',
             '| Model | Storage | Changed vs CUDA runtime weights | Python stored-weight logit RMSE | CUDA same-weight RMSE | Vulkan same-weight RMSE | Python stored-weight top-1 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f'| {s["model"]} | {s["dtype"]} | {s["changed_weight_values"]:,} | '
                     f'{s["weight_conversion"]["logits_rmse"]:.6f} | '
                     f'{s["cuda_same_weights"]["logits_rmse"]:.6f} | '
                     f'{s["vulkan_same_weights"]["logits_rmse"]:.6f} | '
                     f'{s["weight_conversion"]["predicate_top1_agreement"]:.3%} |')
    if args.cuda_variant:
        lines += ['', '## CUDA activation precision and latency', '',
                  'The consolidated GGML patch selects F32 activation GEMM for quantized matrices and excludes incompatible quantized fusions. Both paths use the same Q8 GGUF and the same attention mode recorded in each native results.json. These are full 50-image runs, not single-image projections.', '',
                  '| Model | Default / F32-input median ms | Python median ms | Default / F32-input same-weight RMSE | Default / F32-input top-1 agreement |',
                  '|---|---:|---:|---:|---:|']
        from benchmark_full_graph import summary
        for s in summaries:
            model=s['model']; dtype=s['dtype']
            timing={engine:summary(json.loads((args.benchmark/model/engine/'results.json').read_text()))
                    for engine in ('python',f'cuda-{dtype}',f'cuda-{dtype}{args.cuda_variant}')}
            a=s['cuda_default_same_weights'];b=s['cuda_same_weights']
            lines.append(f'| {model} | {timing[f"cuda-{dtype}"]["latency_median_ms"]:.2f} / {timing[f"cuda-{dtype}{args.cuda_variant}"]["latency_median_ms"]:.2f} | '
                         f'{timing["python"]["latency_median_ms"]:.2f} | {a["logits_rmse"]:.6f} / {b["logits_rmse"]:.6f} | '
                         f'{a["predicate_top1_agreement"]:.3%} / {b["predicate_top1_agreement"]:.3%} |')
        lines += ['', 'This option reduces backend activation-quantization error. Original-F32 weight rounding remains, and the same-weight GPU arithmetic is still not bit identical to the independent Python kernels. The default retains its measured faster Q8 path; select `--quant-f32-input` explicitly when prioritizing accuracy.', '',
                  '```sh', './run_ggml.sh --backend cuda --dtype q8_0 --quant-f32-input', '```']
    lines += ['', 'F32 storage preserves the prepared CPU runtime weights exactly. The nonzero '
              'F32 changed-value count here compares them with the CUDA runtime vocabulary '
              'normalization/gate computation; relative weight L2 differences are approximately '
              '1e-8 to 2e-8, not F32 storage quantization.', '',
              'Each result includes IEEE-754 logit equality, pair-set recall, top-1 margins, '
              'and same-weight backend residuals. Invalid padded slots are excluded. '
              'When a reference top-1 margin exceeds twice the maximum row error, '
              'its winner cannot change; every such row is checked explicitly.', '',
              'F16/Q8 changes to outputs reproduced by the official graph establish that '
              'original-F32 bit equality is not a property of those converted models. '
              'Backend residuals remain visible and are not attributed to weight rounding.', '',
              '[Machine-readable summary](summary.json)']
    lines += ['', '## Same-weight task metrics and worst images', '',
              '| Model/storage | Oracle R@50 | mR@50 | F1@50 | wR@50 | CUDA / Vulkan calibrated score RMSE |',
              '|---|---:|---:|---:|---:|---:|']
    for s in summaries:
        m=s['oracle_metrics']
        lines.append(f'| {s["model"]}/{s["dtype"]} | {m["R@50"]:.3%} | {m["mR@50"]:.3%} | '
                     f'{m["F1@50"]:.3%} | {m["wR@50"]:.3%} | '
                     f'{s["cuda_same_weights"]["calibrated_score_rmse"]:.6g} / '
                     f'{s["vulkan_same_weights"]["calibrated_score_rmse"]:.6g} |')
        for backend in ('cpu','cuda','vulkan'):
            worst=s['worst_images'][backend+'_same_weights'][0]
            lines.append(f'\nWorst {s["model"]}/{s["dtype"]}/{backend}: `{worst["sample"]}`, '
                         f'RMSE {worst["logits_rmse"]:.6g}, pair recall {worst["pair_set_recall"]:.3%}.')
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
