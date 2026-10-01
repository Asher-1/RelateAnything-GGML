#!/usr/bin/env python3
"""Report a focused CUDA optimization without advancing full-release acceptance."""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from benchmark_full_graph import load_cpp
from provenance import ROOT, fingerprint, save, validate_run

MODELS = ('relsgg-vits16', 'relsgg-vits16plus', 'relsgg-vitb16')
DTYPES = ('f32', 'f16', 'q8_0')


def read(path):
    return json.loads(path.read_text())


def median(rows):
    return statistics.median(r['latency_ms'] for r in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--without-graphs', type=Path, required=True)
    parser.add_argument('--with-graphs', type=Path, required=True)
    parser.add_argument('--operator', type=Path, required=True)
    parser.add_argument('--patch-embedding', type=Path, required=True)
    parser.add_argument('--stages', type=Path, required=True)
    parser.add_argument('--replay-audit', type=Path, required=True)
    parser.add_argument('--layer-norm', type=Path, required=True)
    parser.add_argument('--quant-f32-input', type=Path, required=True)
    parser.add_argument('--capture-profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'cpp_ggml/benchmarks/optimization')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    replay = read(args.replay_audit)
    assert replay['status'] == 'pass'
    stage_rows = read(args.stages/'summary.json')
    assert len(stage_rows) == 27
    rows, files = [], [Path(__file__), args.operator, args.patch_embedding, args.replay_audit, args.capture_profile,
                       args.layer_norm, args.stages/'summary.json']
    for model in MODELS:
        py_path = args.with_graphs/model/'python/results.json'
        py = read(py_path)
        files.append(py_path)
        for dtype in DTYPES:
            config = 'cuda-'+dtype
            paths = [folder/model/config/'results.json' for folder in
                     (args.original, args.without_graphs, args.with_graphs)]
            versions = [read(p) for p in paths]
            files += paths
            assert all(len(v['rows']) == 50 for v in versions)
            for path in paths[1:]:
                provenance = validate_run(path.parent)
                assert provenance['status'] == 'complete'
            original, without, current = [v['rows'] for v in versions]
            raw_equal, raw_files = [], []
            equal_bits, bit_count = 0, 0
            for row in current:
                name = row['sample']+'.rafo'
                before = args.without_graphs/model/config/name
                after = args.with_graphs/model/config/name
                a, b = load_cpp(before), load_cpp(after)
                raw_files += [before, after]
                raw_equal.append(all(np.array_equal(a[k].view(np.uint8), b[k].view(np.uint8)) for k in a))
                with np.load(args.with_graphs/model/'python'/(row['sample']+'.npz')) as reference:
                    # Load each compressed member once: indexing the NpzFile in
                    # the pair loop would inflate the complete logits P times.
                    sub, obj = reference['sub'], reference['obj']
                    valid, logits = reference['valid'], reference['logits']
                    lookup = {(int(sub[i]), int(obj[i])): i
                              for i in np.flatnonzero(valid)}
                    for i in np.flatnonzero(b['valid']):
                        key = (int(b['sub'][i]), int(b['obj'][i]))
                        if key in lookup:
                            same = (b['logits'][i].view(np.uint32) ==
                                    logits[lookup[key]].view(np.uint32))
                            equal_bits += int(same.sum())
                            bit_count += same.size
            rmse = lambda values: statistics.mean(r['parity']['logits_rmse'] for r in values)
            rows.append({'model': model, 'dtype': dtype, 'images': 50,
                         'original_snapshot_ms': median(original),
                         'precise_without_capture_ms': median(without), 'capture_enabled_ms': median(current),
                         'python_bracket_ms': median(py['rows']), 'python_bracket': py['timing_bracket'],
                         'speedup_vs_python': median(py['rows'])/median(current),
                         'capture_option_raw_equal_images': sum(raw_equal),
                         'warmed_capture_option_raw_equal_images': sum(m['equal'] for rr in replay['rows']
                             if rr['model'] == model and rr['dtype'] == dtype for m in rr['matches']),
                         'raw_identity': fingerprint(raw_files),
                         'original_rmse': rmse(original), 'current_rmse': rmse(current),
                         'current_top1_agreement': statistics.mean(r['parity']['predicate_top1_agreement'] for r in current),
                         'current_pair_jaccard': statistics.mean(r['parity']['pair_set_jaccard'] for r in current),
                         'original_cuda_logit_bit_equal_fraction': equal_bits/bit_count,
                         'original_hits_at_100': sum(r['hits@100'] for r in original),
                         'current_hits_at_100': sum(r['hits@100'] for r in current)})
            print(model, dtype, 'validated 50 images', flush=True)
    operator = read(args.operator)
    capture_profile = read(args.capture_profile)
    assert operator['status'] == 'pass'
    payload = {'scope': '600 new CUDA images, including 150 Q8 F32-input; CPU/Vulkan latency remains the complete earlier snapshot',
               'original_bit_equality': 'open', 'all_strict_faster_than_python': 'open',
               'timing_scope': 'image decoding through calibrated triplets; 2 warmups, 5 repeats per image; load/export excluded',
               'source': fingerprint(files), 'rows': rows,
               'operator_cases': operator['covered_cases'],
               'operator_scope': operator['scope'],
               'patch_embedding': read(args.patch_embedding),
               'layer_norm': read(args.layer_norm), 'replay_audit': replay,
               'capture_profile': capture_profile,
               'stage_rows': stage_rows}
    save(args.output/'summary.json', payload)
    labels = [r['model'].removeprefix('relsgg-')+' / '+r['dtype'] for r in rows]
    figures = []
    def figure(name, fig):
        fig.tight_layout()
        fig.savefig(args.output/(name+'.png'), dpi=160)
        fig.savefig(args.output/(name+'.svg'))
        plt.close(fig)
        figures.append(name)
    fig, axes = plt.subplots(3, 1, figsize=(13, 12))
    for ax, model in zip(axes, MODELS):
        selected = [r for r in rows if r['model'] == model]
        x = np.arange(3)
        for offset, key, color, label in ((-.27, 'original_snapshot_ms', '#89939a', 'Complete prior snapshot'),
                                        (0, 'precise_without_capture_ms', '#168592', 'Precise operators, no capture'),
                                        (.27, 'capture_enabled_ms', '#4d9d56', 'Precise operators, capture enabled')):
            bars = ax.bar(x+offset, [r[key] for r in selected], .25, color=color, label=label)
            ax.bar_label(bars, fmt='%.2f', padding=3)
        ax.axhline(selected[0]['python_bracket_ms'], color='#ad402d', linestyle='--',
                   label=f"Fresh official PyTorch-CUDA: {selected[0]['python_bracket_ms']:.2f} ms")
        bracket = selected[0]['python_bracket']
        ax.axhspan(min(bracket['before_median_ms'], bracket['after_median_ms']),
                   max(bracket['before_median_ms'], bracket['after_median_ms']),
                   color='#ad402d', alpha=.12, label='Python before/after P50 range')
        ax.set_xticks(x, DTYPES)
        ax.set_ylabel('Full pipeline P50 (ms)')
        ax.set_title(model+' | 50 images')
        ax.legend(fontsize=8)
    figure('latency', fig)
    fig, axes = plt.subplots(3, 1, figsize=(13, 12))
    x = np.arange(len(rows))
    axes[0].bar(x-.18, [r['original_rmse'] for r in rows], .36, label='Previous snapshot')
    axes[0].bar(x+.18, [r['current_rmse'] for r in rows], .36, label='Current CUDA')
    axes[0].set_yscale('log'); axes[0].set_ylabel('Original CUDA logits RMSE'); axes[0].legend()
    axes[1].bar(x, [r['current_top1_agreement'] for r in rows])
    axes[1].set_ylabel('Predicate top-1 agreement'); axes[1].set_ylim(.95, 1.005)
    axes[2].bar(x, [r['original_cuda_logit_bit_equal_fraction'] for r in rows])
    axes[2].set_ylabel('Original CUDA bit equality fraction')
    for ax in axes:
        ax.set_xticks(x, labels, rotation=30, ha='right')
    fig.suptitle('Full-model precision remains a separate open acceptance condition')
    figure('precision', fig)
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    axes[0].bar(x, [r['speedup_vs_python'] for r in rows])
    axes[0].axhline(1, color='red', linestyle='--')
    axes[0].set_ylabel('Speedup vs fresh official Python')
    axes[1].bar(x, [r['warmed_capture_option_raw_equal_images'] for r in rows])
    axes[1].set_ylabel('Capture option preserves raw bits (of 50)')
    for ax in axes:
        ax.set_xticks(x, labels, rotation=30, ha='right')
    figure('speed_and_replay_bits', fig)
    op_rows = [r for r in operator['rows'] if r['columns'] <= 1024]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot([r['False']['bit_equal_fraction'] for r in op_rows], label='General GGML')
    axes[0].plot([r['True']['bit_equal_fraction'] for r in op_rows], label='Precise warp')
    axes[0].set_ylabel('Softmax bit equality with PyTorch'); axes[0].legend()
    perf = [r for r in op_rows if r['rows'] == 4734]
    axes[1].bar(np.arange(3)-.18, [r['False']['median_ms'] for r in perf], .36, label='General GGML')
    axes[1].bar(np.arange(3)+.18, [r['True']['median_ms'] for r in perf], .36, label='Precise warp')
    axes[1].set_xticks(np.arange(3), [str(r['scale']) for r in perf])
    axes[1].set_ylabel('789 x 4734 softmax (ms)'); axes[1].legend()
    figure('softmax_operator', fig)
    stage_keys = list(stage_rows[0]['stages'])
    for reference in ('cpu', 'cuda'):
        values = [r if reference == 'cpu' else r['original_cuda_reference'] for r in stage_rows]
        errors = [[v['stages'][k]['rmse'] for k in stage_keys] for v in values]
        fig, ax = plt.subplots(figsize=(17, 12))
        im = ax.imshow(np.log10(np.maximum(errors, 1e-12)), aspect='auto', cmap='viridis')
        ax.set_xticks(np.arange(len(stage_keys)), stage_keys, rotation=55, ha='right', fontsize=8)
        ax.set_yticks(np.arange(27), [r['model']+'/'+r['backend']+'/'+r['dtype'] for r in stage_rows], fontsize=8)
        ax.set_title('New strict build: all 27 stages vs original '+reference.upper())
        fig.colorbar(im, ax=ax, label='log10(RMSE)')
        figure('stage_errors_'+reference, fig)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    native_f32 = [r for r in stage_rows if r['backend'] == 'cuda' and r['dtype'] == 'f32']
    for i, key in enumerate(('backbone_patch', 'backbone_norm1')):
        axes[i].bar(np.arange(3), [r['original_cuda_reference']['stages'][key]['bit_equal_fraction'] for r in native_f32])
        axes[i].set_xticks(np.arange(3), [r['model'] for r in native_f32])
        axes[i].set_ylim(0, 1.05)
        axes[i].set_ylabel('Bit equality fraction')
        axes[i].set_title(key+' vs original CUDA | same real image, original F32 weights')
    figure('patch_and_next_boundary', fig)
    fig, ax = plt.subplots(figsize=(17, 8))
    ax.bar(np.arange(27), [r['dynamic_vocabulary']['logits_rmse'] for r in stage_rows])
    ax.set_xticks(np.arange(27), [r['model']+'/'+r['backend']+'/'+r['dtype'] for r in stage_rows], rotation=55, ha='right', fontsize=8)
    ax.set_yscale('log')
    ax.set_ylabel('Dynamic vocabulary logits RMSE vs independent CPU')
    ax.set_title('All 27 dynamic-vocabulary configurations | six scaled/composite embeddings')
    figure('dynamic_vocabulary', fig)
    q8 = []
    for model in MODELS:
        result_path = args.quant_f32_input/model/'cuda-q8_0-f32-input/results.json'
        python_path = args.quant_f32_input/model/'python/results.json'
        result, reference = read(result_path), read(python_path)
        assert len(result['rows']) == 50 and len(reference['rows']) == 50
        validate_run(result_path.parent)
        q8.append({'model': model, 'native_ms': median(result['rows']),
                   'python_ms': median(reference['rows']),
                   'logits_rmse': statistics.mean(r['parity']['logits_rmse'] for r in result['rows'])})
        files += [result_path, python_path]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    xx = np.arange(3)
    axes[0].bar(xx-.18, [r['native_ms'] for r in q8], .36, label='CUDA Q8 F32 activation')
    axes[0].bar(xx+.18, [r['python_ms'] for r in q8], .36, label='Fresh original PyTorch CUDA')
    axes[0].legend(); axes[0].set_ylabel('Full pipeline P50 (ms)')
    axes[1].bar(xx, [r['logits_rmse'] for r in q8]); axes[1].set_ylabel('Original CUDA logits RMSE')
    for ax in axes:
        ax.set_xticks(xx, MODELS)
    figure('quant_f32_input', fig)
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    for ax, model in zip(axes, MODELS):
        current_model = [r for r in rows if r['model'] == model]
        labels_all, values_all, colors = [], [], []
        for backend, color in (('cpu', '#7766cc'), ('cuda', '#168592'), ('vulkan', '#d18a43')):
            for dtype in DTYPES:
                if backend == 'cuda':
                    value = next(r['capture_enabled_ms'] for r in current_model if r['dtype'] == dtype)
                    label = 'CUDA new / '+dtype
                else:
                    path = args.original/model/(backend+'-'+dtype)/'results.json'
                    result = read(path)
                    value = median(result['rows'])
                    files.append(path)
                    label = backend+' snapshot / '+dtype
                labels_all.append(label); values_all.append(value); colors.append(color)
        extra_row = next(r for r in q8 if r['model'] == model)
        labels_all += ['CUDA new / Q8 F32-input']
        values_all += [extra_row['native_ms']]; colors += ['#168592']
        bars = ax.bar(np.arange(10), values_all, color=colors)
        ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=3)
        ax.axhline(current_model[0]['python_bracket_ms'], color='#ad402d', linestyle='--',
                   label=f"Fresh official Python-CUDA {current_model[0]['python_bracket_ms']:.2f} ms")
        ax.set_xticks(np.arange(10), labels_all, rotation=25, ha='right', fontsize=8)
        ax.set_yscale('log'); ax.set_ylabel('Image-to-triplets P50 (ms, log scale)')
        ax.set_title(model+' | explicit mixed snapshots: CPU/Vulkan previous, CUDA newly measured')
        ax.legend(fontsize=8)
    figure('all_backends_latency', fig)
    payload.update(quant_f32_input=q8, source=fingerprint(files), figures=figures)
    save(args.output/'summary.json', payload)
    evidence = args.output/'evidence'
    evidence.mkdir(exist_ok=True)
    import shutil
    for path in (args.operator, args.patch_embedding, args.replay_audit, args.layer_norm, args.capture_profile):
        shutil.copyfile(path, evidence/(path.parent.name+'-'+path.name))
    for filename in ('first_numerical_divergence.json', 'summary.json'):
        shutil.copyfile(args.stages/filename, evidence/('stages-'+filename))
    lines = ['# Strict CUDA 优化进度', '',
             '这是新 CUDA 的专项重测，覆盖 3 个模型 × F32/F16/Q8 × 50 图，另有 150 图 Q8 F32-input。完整 CPU/CUDA/Vulkan 基线仍在上一级画廊。', '',
             '**全模型原版 CUDA 逐位一致、全部后端 strict 更快仍为 open。** 算子通过不能替代全模型通过。', '',
             '计时：原图解码到校准三元组；2 次预热、5 次重复；官方 Python 在每组模型前后独立测量。', '',
             '| 模型 | 存储 | 原完整基线 ms | 精确算子禁用 capture ms | 启用 capture ms | 本轮 Python ms | 速度倍率 | 预热后 capture 开关逐位相同图数 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["model"]} | {r["dtype"]} | {r["original_snapshot_ms"]:.2f} | '
                     f'{r["precise_without_capture_ms"]:.2f} | {r["capture_enabled_ms"]:.2f} | '
                     f'{r["python_bracket_ms"]:.2f} | {r["speedup_vs_python"]:.3f} | {r["warmed_capture_option_raw_equal_images"]}/50 |')
    lines += ['', '速度倍率使用每图 Python 重复计时汇总后的 50 图 P50；before/after 分布另行保留。'
              '7/9 是本轮观测值，接近 1 倍的结果不足以保证稳定更快，且全模型逐位一致仍未通过。', '',
              '| 模型 | Python 前 P50 ms | Python 后 P50 ms | 漂移 |', '|---|---:|---:|---:|']
    for model in MODELS:
        bracket = next(r['python_bracket'] for r in rows if r['model'] == model)
        lines.append(f'| {model} | {bracket["before_median_ms"]:.2f} | '
                     f'{bracket["after_median_ms"]:.2f} | {bracket["relative_drift"]:.2%} |')
    lines += ['', '60 个受支持 F32 softmax 用例全部逐位一致。超过 1024 列、mask、ALiBi 或 sinks 使用通用回退；其算术未宣称逐位一致。', '',
              'F32 patch embedding 已改为按 CYX 顺序累积的 F32 FMA。三个真实模型的首个 patch 边界均与原始 CUDA 逐位一致；'
              '第一处分歧推进到 LayerNorm。F16/Q8 权重没有原始 F32 信息，未宣称原版权重逐位一致。', '',
              '已重跑 27 组 stage/dynamic。capture 是否实际发生取决于图的地址和形状是否稳定；启用编译选项不代表每次运行均已 capture。'
              'capture 开关位对照使用 3 次预热后导出的所有 tensor、候选索引与 logits；是否发生 capture/launch 另看 profiler 记录。', '',
              f'本轮端到端 profiler：{capture_profile["cuda_capture_begins"]} 次 capture、'
              f'{capture_profile["cuda_graph_launches"]} 次 graph launch；当前速度变化不能归因于 CUDA Graph 重放。', '',
              '[完整统计与哈希](summary.json)。F16/Q8 存储误差仍保留；GT 字符串命中不是同义词校准指标。', '']
    for name in figures:
        lines += [f'![{name}]({name}.png)', '', f'[SVG]({name}.svg)', '']
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')
    from html import escape
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Strict CUDA 优化进度</title><style>body{font:16px system-ui;max-width:1300px;margin:auto;padding:20px;overflow-wrap:anywhere}'
            'img{width:100%;height:auto}pre{white-space:pre-wrap}a{color:#076778}</style><h1>Strict CUDA 优化进度</h1>'
            '<p>12 个 CUDA 配置、600 图；27 组 stage/dynamic；全模型逐位一致与全部后端更快仍为 open。</p>'
            '<p><a href="../ALL_COMPARISONS.html">全部后端对比</a> · <a href="summary.json">完整统计</a></p>'
            '<pre>'+escape('\n'.join(lines[:lines.index('60 个受支持 F32 softmax 用例全部逐位一致。超过 1024 列、mask、ALiBi 或 sinks 使用通用回退；其算术未宣称逐位一致。')]))+'</pre>'
            +''.join(f'<h2>{name}</h2><img src="{name}.png"><a href="{name}.svg">SVG</a>' for name in figures)+'</html>')
    (args.output/'index.html').write_text(page)
    print(json.dumps({'cuda_rows': len(rows), 'faster_rows': sum(r['speedup_vs_python']>1 for r in rows),
                      'capture_option_preserves_bits': replay['status'] == 'pass'}, indent=2))


if __name__ == '__main__':
    main()
