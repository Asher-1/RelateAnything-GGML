#!/usr/bin/env python3
"""Expose every chart and original-image grid at the benchmarks root."""
from html import escape
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT/'benchmarks'
    snapshot = json.loads((folder/'gallery/release_snapshot.json').read_text())
    figures = [row['file'] for row in snapshot['files'] if row['file'].endswith('.png')]
    priority = ('latency_all_backends.png', 'latency_gpu_backends.png',
                'speedup_all_backends.png', 'accuracy_candidates.png',
                'precision_speed_tradeoff.png', 'stage_errors_cpu.png', 'stage_errors_cuda.png')
    figures.sort(key=lambda name: (priority.index(name) if name in priority else len(priority), name))
    titles = dict(zip(priority, ('全部模型与后端端到端延迟', 'CUDA / Vulkan 延迟放大图',
                                '各模型各后端速度倍率', '精度和候选对一致率', '逐位一致与速度联合验收',
                                '逐 stage 对 CPU 参考误差', '逐 stage 对原版 CUDA 参考误差')))
    run = ROOT/'outputs/runs'/snapshot['run_id']
    grids = sorted((run/'reports/full_graph/visualizations').glob('*/*.png'))
    md = ['# 全量对比图', '', f'运行：`{snapshot["run_id"]}`。{len(figures)} 张详细图、{len(grids)} 张原图配置网格。', '',
          '[浏览器打开全部图](ALL_COMPARISONS.html) · [交互切换 1500 个图像配置](gallery/index.html)', '']
    sections = []
    for index, name in enumerate(figures):
        title = titles.get(name, name.removesuffix('.png'))
        md += [f'## {title}', '', f'![{title}](gallery/{name})', '',
               f'[PNG](gallery/{name}) · [SVG](gallery/{name[:-4]}.svg)', '']
        sections.append(f'<section id="chart-{index}"><h2>{escape(title)}</h2>'
                        f'<a href="gallery/{escape(name)}"><img loading="lazy" '
                        f'src="gallery/{escape(name)}" alt="{escape(title)}"></a>'
                        f'<a href="gallery/{escape(name[:-4])}.svg">SVG</a></section>')
    optimization = folder/'optimization/summary.json'
    extra = []
    if optimization.is_file():
        report = json.loads(optimization.read_text())
        extra = report['figures']
        md += ['## 本轮 strict CUDA 修复与全后端 stage 重测', '',
               '[新算子、端到端速度和第一处分歧](optimization/index.html)。全模型验收仍为 open。', '']
        sections.insert(0, '<section><h2>本轮 strict 修复与重测</h2>'
                        '<p><a href="optimization/index.html">查看重测范围、来源和完整数据</a>；全模型验收仍为 open。</p>'
                        + ''.join(f'<h3>{escape(name)}</h3><a href="optimization/{escape(name)}.png">'
                                  f'<img loading="lazy" src="optimization/{escape(name)}.png" alt="{escape(name)}"></a>'
                                  f'<a href="optimization/{escape(name)}.svg">SVG</a>' for name in extra)+'</section>')
        for name in extra:
            md += [f'![本轮 {name}](optimization/{name}.png)', '', f'[SVG](optimization/{name}.svg)', '']
        md[2] += f' 另含本轮 {len(extra)} 张修复与重测图；新 CUDA 和原完整基线分别标注。'
    nav = ''.join(f'<li><a href="#chart-{i}">{escape(titles.get(n,n[:-4]))}</a></li>'
                  for i, n in enumerate(figures))
    for model in ('relsgg-vits16', 'relsgg-vits16plus', 'relsgg-vitb16'):
        model_grids = [p for p in grids if p.parent.name == model]
        md += [f'## {model}：50 张原图的全部配置网格', '']
        html = []
        for path in model_grids:
            relative = '../'+path.relative_to(ROOT).as_posix()
            md += [f'- [{path.stem}]({relative})']
            html.append(f'<details><summary>{escape(path.stem)}</summary>'
                        f'<a href="{escape(relative)}"><img loading="lazy" '
                        f'src="{escape(relative)}" alt="{escape(model+" "+path.stem)}"></a></details>')
        md.append('')
        sections.append(f'<section><h2>{model}：全部 50 张原图对照</h2>{"".join(html)}</section>')
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>RelateAnything 全量对比图</title><style>'
            'body{font:16px system-ui;max-width:1700px;margin:auto;padding:20px;color:#18242e;overflow-wrap:anywhere}'
            'img{width:100%;height:auto}a{color:#076778}li{margin:7px 0}'
            'section{margin:28px 0}summary{cursor:pointer;padding:10px;background:#eef3f5}'
            'details{margin:6px 0}nav{overflow-wrap:anywhere}</style><h1>全量对比图</h1>'
            f'<p>{escape(snapshot["run_id"])} · {len(figures)} 张完整基线 PNG/SVG 图 · '
            f'{len(extra)} 张本轮修复重测图 · '
            f'{len(grids)} 张原图配置网格。</p>'
            '<p><a href="gallery/index.html">交互查看 GT / 官方 Python / GGML：1500 个组合</a> · '
            '<a href="../test_data/index.html">查看测试数据</a></p>'
            '<p>逐位一致与全部 strict 配置更快仍未通过。下方图保留误差、慢配置和参考范围。</p>'
            f'<nav><ul>{nav}</ul></nav>'+''.join(sections)+'</html>')
    (folder/'ALL_COMPARISONS.html').write_text(page)
    (folder/'ALL_COMPARISONS.md').write_text('\n'.join(md)+'\n')
    print(f'Comparison index: {len(figures)} complete-snapshot figures, {len(extra)} new figures, {len(grids)} original-image grids')


if __name__ == '__main__':
    main()
