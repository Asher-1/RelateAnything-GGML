#!/usr/bin/env python3
"""Separate dense candidate propagation from same-weight native arithmetic."""
import argparse
import json
from pathlib import Path
import types

import numpy as np
import torch
from PIL import Image
from audit_quantized_graph import restore_weights
from benchmark_full_graph import ROOT, load_cpp, parity, raw_python
from provenance import model_assets, recorded_run, save, validate_run
from relsgg import RelateAnything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dense', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    rows = json.loads((args.dense/'results.json').read_text())
    sample = json.loads((ROOT/'cpp_ggml/test_data/ra4m_50/manifest.json').read_text())['samples'][0]
    image = Image.open(ROOT/'cpp_ggml/test_data/ra4m_50'/sample['image']).convert('RGB')
    boxes = np.asarray(json.loads((args.dense/'boxes.json').read_text())['boxes_xyxy'], np.float32)
    summary = []
    for model, dtype in sorted({(r['model'], r['dtype']) for r in rows}):
        gguf = ROOT/f'cpp_ggml/models/gguf/{model}-{dtype}.gguf'
        folder = args.output/model/dtype
        group = [r for r in rows if (r['model'], r['dtype']) == (model, dtype)]
        native_files = [args.dense/f'{model}-{r["backend"]}-{dtype}.rafo' for r in group]
        runs = [args.dense/f'{model}-{r["backend"]}-{dtype}-stages' for r in group]
        for row, run in zip(group, runs):
            validate_run(run, binary=ROOT/f'cpp_ggml/build-{row["backend"]}/relateanything-ggml')
        with recorded_run(folder, inputs=[gguf, *model_assets(model), args.dense/'input.raim',
                          args.dense/'boxes.json', *native_files, *(p/'run_provenance.json' for p in runs)],
                          config={'device': 'cpu', 'forced_pairs': True}, script=__file__):
            api = RelateAnything.from_checkpoint(str(ROOT/f'cpp_ggml/models/pytorch/{model}/model.pth'),
                                                  device='cpu', full_vocabulary=True)
            weights = restore_weights(api, gguf)
            captured = {}
            hook = api.model.register_forward_hook(lambda m, i, o: captured.update(out=o))
            api.predict(image, boxes, topk=128)
            natural = raw_python(captured['out'])
            np.savez(folder/'natural.npz', **natural)
            original = api.model.sampler.forward
            entries = []
            for row, native_file in zip(group, native_files):
                native = load_cpp(native_file)

                def forced(sampler, boxes, obj_feats, *unused, **kwargs):
                    normalized, features = boxes, obj_feats
                    sub = torch.as_tensor(native['sub'], device=features.device).long().unsqueeze(0)
                    obj = torch.as_tensor(native['obj'], device=features.device).long().unsqueeze(0)
                    valid = torch.as_tensor(native['valid'], device=features.device).bool().unsqueeze(0)
                    zs, zo = sampler.f_sub(features), sampler.f_obj(features)
                    zs = torch.gather(zs, 1, sub.unsqueeze(-1).expand(-1, -1, sampler.rel_dim))
                    zo = torch.gather(zo, 1, obj.unsqueeze(-1).expand(-1, -1, sampler.rel_dim))
                    logits = (zs*zo).sum(-1)/(sampler.rel_dim**.5)
                    zero = normalized.new_zeros(())
                    return sub, obj, valid, torch.full_like(sub, -1), zero, zero, logits

                api.model.sampler.forward = types.MethodType(forced, api.model.sampler)
                api.predict(image, boxes, topk=128)
                fixed = raw_python(captured['out'])
                np.savez(folder/f'{row["backend"]}-fixed.npz', **fixed)
                entries.append({'model': model, 'dtype': dtype, 'backend': row['backend'],
                                'native_vs_natural_same_weights': parity(natural, native),
                                'native_vs_fixed_same_weights': parity(fixed, native),
                                'candidate_context_propagation': parity(natural, fixed)})
            api.model.sampler.forward = original
            hook.remove()
            save(folder/'results.json', {'weights': weights, 'rows': entries,
                 'scope': 'same GGUF restored in CPU PyTorch; forced native pair order; excludes storage rounding'})
        summary.extend(entries)
        save(args.output/'summary.json', summary)
        print(model, dtype, [(x['backend'], x['native_vs_fixed_same_weights']['logits_rmse'])
                            for x in entries], flush=True)


if __name__ == '__main__':
    main()
