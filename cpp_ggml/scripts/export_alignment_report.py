#!/usr/bin/env python3
"""Export a verified compact gallery and regenerate current acceptance documents."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from benchmark_layout import verify
from provenance import ROOT, fingerprint, save, sha256

MODELS = ('relsgg-vits16', 'relsgg-vits16plus', 'relsgg-vitb16')
DTYPES = ('f32', 'f16', 'q8_0')
BACKENDS = ('cpu', 'cuda', 'vulkan')


def read(path):
    return json.loads(path.read_text())


def responsive_gallery(path):
    html = path.read_text()
    if html.count('<table>') != 1 or html.count('</table>') != 1:
        raise RuntimeError('unexpected gallery relation table layout')
    css = ('\n.table-scroll{max-width:100%;overflow-x:auto}'
           '.controls label{min-width:0;max-width:100%}'
           '.controls select{min-width:0;max-width:100%}'
           '.panel{min-width:0;overflow-wrap:anywhere}'
           '.stats{overflow-wrap:anywhere}'
           '@media(max-width:500px){main{padding:12px}'
           '.controls{padding:12px;position:static}'
           '.controls label{width:100%}'
           '.controls select{flex:1;width:0}'
           '.controls input{width:64px}}\n')
    html = html.replace('</style>', css+'</style>', 1)
    html = html.replace('<table>', '<div class="table-scroll"><table>', 1)
    html = html.replace('</table>', '</table></div>', 1)
    visual_appendix = ('<h2>全部配置的候选对与精度</h2>'
                       '<img class="chart" src="accuracy_candidates.png">'
                       '<h2>逐 stage 误差：独立 CPU 参考</h2>'
                       '<img class="chart" src="stage_errors_cpu.png">'
                       '<h2>逐 stage 误差：原版 PyTorch-CUDA 参考</h2>'
                       '<img class="chart" src="stage_errors_cuda.png">')
    html = html.replace('</main>', visual_appendix+'</main>', 1)
    path.write_text(html)


def export(run, output):
    if run == output or run in output.parents or output in run.parents:
        raise RuntimeError('the compact export must be outside the immutable run')
    if output.exists() and not (output/'release_snapshot.json').is_file():
        raise RuntimeError('refusing to replace a directory without an alignment export marker')
    verify(run)
    reports = run/'reports'
    audit = read(reports/'full_graph/release_audit.json')
    if not audit.get('all_complete') or audit.get('primary_ggml_image_configurations') != 1350:
        raise RuntimeError('a complete three-backend release audit is required')
    status = read(reports/'acceptance_status.json')
    if status['execution']['primary_rows'] != 27:
        raise RuntimeError('all 27 primary native precision/timing configurations are required')
    if status.get('release_decision') not in ('complete', 'accepted_with_performance_deviations'):
        raise RuntimeError('release acceptance decision is blocked')
    for backend in BACKENDS:
        rows = read(reports/f'c_api-{backend}.json')['rows']
        if len(rows) != 9 or any(row['returncode'] for row in rows):
            raise RuntimeError(f'incomplete {backend} real-checkpoint C API contract')
    numeric = read(reports/'gallery/numerical_first_divergence.json')
    stages = read(reports/'stages/first_numerical_divergence.json')
    if len(numeric) != 30 or len(stages) != 54:
        raise RuntimeError('complete original-CUDA and stage first-difference evidence is required')
    files = {path.name: path for path in (reports/'gallery').iterdir() if path.is_file()}
    if 'browser_audit.json' in files:
        files['evidence/source_browser_audit.json'] = files.pop('browser_audit.json')
    for name in ('stage_errors_cpu.png', 'stage_errors_cpu.svg',
                 'stage_errors_cuda.png', 'stage_errors_cuda.svg'):
        files[name] = reports/'stages'/name
    evidence = ('acceptance_status.json', 'ggml_patch_audit.json', 'patch_replay.json',
                'runtime_options_audit.json', 'build_test_audit.json', 'report_test_audit.json', 'linkage_audit.json',
                'protocol_asset_access.json', 'rejected_optimization.json',
                'c_api-cpu.json', 'c_api-cuda.json', 'c_api-vulkan.json',
                'full_graph/release_audit.json', 'full_graph/summary.json',
                'full_graph/metrics.csv', 'full_graph/quantization_audit/summary.json',
                'full_graph/quantization_audit/components/results.json',
                'stages/summary.json', 'stages/first_numerical_divergence.json',
                'dense_candidate_audit/summary.json', 'interfaces/results.json',
                'metric_subsets/summary.json', 'spatialsense/summary.json')
    files.update({'evidence/'+name: reports/name for name in evidence})
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='gallery-', dir=output.parent))
    try:
        rows = []
        for relative, source in sorted(files.items()):
            destination = temporary/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if sha256(source) != sha256(destination):
                raise RuntimeError('copy hash mismatch: '+relative)
            rows.append({'file': relative, 'source': source.relative_to(run).as_posix(),
                         'source_sha256': sha256(source), 'sha256': sha256(destination),
                         'bytes': destination.stat().st_size})
        appendix = ('\n## 全链路与接口证据\n\n'
                    '[七项验收状态](evidence/acceptance_status.json) · '
                    '[1500 个 native 输出审计](evidence/full_graph/release_audit.json) · '
                    '[首个 stage 差异](evidence/stages/first_numerical_divergence.json)\n\n'
                    '![所有 stage 对 CPU 参考](stage_errors_cpu.png)\n\n'
                    '![所有 stage 对原版 CUDA 参考](stage_errors_cuda.png)\n\n'
                    '纯 C 接口真实模型验证：'
                    '[CPU](evidence/c_api-cpu.json)、[CUDA](evidence/c_api-cuda.json)、'
                    '[Vulkan](evidence/c_api-vulkan.json)，每个后端九个模型存储组合。'
                    '包含借用图像内存与文件输入逐位比较、取消、结果生命周期和宿主后端 lease。\n\n'
                    '[单补丁重放](evidence/patch_replay.json) · '
                    '[编译源 Struct Options 审计](evidence/runtime_options_audit.json) · '
                    '[固定构建 CTest](evidence/build_test_audit.json) · '
                    '[协议资产访问结果](evidence/protocol_asset_access.json)\n')
        with (temporary/'README.md').open('a') as stream:
            stream.write(appendix)
        responsive_gallery(temporary/'index.html')
        for row in rows:
            path = temporary/row['file']
            row.update(sha256=sha256(path), bytes=path.stat().st_size)
        save(temporary/'release_snapshot.json', {
            'schema': 1, 'run_id': run.name, 'run_sha256': sha256(run/'run.json'),
            'reports_manifest_sha256': sha256(run/'reports.json'),
            'exporter': fingerprint([Path(__file__)]), 'files': rows,
            'exported_readme_sha256': sha256(temporary/'README.md'),
            'scope': 'compact derived reports; raw tensors remain in the single immutable run'})
        if output.exists():
            shutil.rmtree(output)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    documents(reports, status, run.name, len(list(output.glob("*.png"))))
    print(f'Exported {len(rows)} source files, {len(list(output.glob("*.png")))} PNG/SVG figures: {output}')


def timing_table(summary):
    lines = ['| 模型 | PyTorch CUDA | CPU F32 / F16 / Q8 | CUDA F32 / F16 / Q8 | Vulkan F32 / F16 / Q8 | CUDA Q8 F32-input |',
             '|---|---:|---|---|---|---:|']
    for model in MODELS:
        values = summary['models'][model]
        groups = [' / '.join(f'{values[b+"-"+d]["latency_median_ms"]:.2f}' for d in DTYPES)
                  for b in BACKENDS]
        lines.append(f'| {model.removeprefix("relsgg-")} | {values["python"]["latency_median_ms"]:.2f} | '
                     +' | '.join(groups)+f' | {values["cuda-q8_0-f32-input"]["latency_median_ms"]:.2f} |')
    return '\n'.join(lines)


def documents(reports, status, run_id, figure_count):
    folder = ROOT/'cpp_ggml/benchmarks'
    summary = read(reports/'full_graph/summary.json')
    statuses = {row['id']: row for row in status['statuses']}
    g1, g2, precision = (statuses[key] for key in ('g1_numeric', 'g2_gpu_performance', 'auto_precision'))
    decision = ('通过' if status['release_decision'] == 'complete' else
                f'通过，含低于 {g2["materiality"]:.0%} materiality 带的已跟踪性能偏差' if
                status['release_decision'] == 'accepted_with_performance_deviations' else '阻塞')
    failures = '\n'.join(
        f'- `{row["model"]}/{row["config"]}`：{row["native_ms"]:.2f} ms vs '
        f'PyTorch {row["python_ms"]:.2f} ms（{row["python_ms"]/row["native_ms"]:.2f}×）'
        for row in g2['failed']) or '- 无'
    status_text = f'''# GGML 发布验收状态

## 发布结论

- 运行：`{run_id}`
- 发布决策：**{decision}**
- 数值门槛 G1：**{g1["passed"]}/{g1["total"]}**（`{g1["status"]}`）
- GPU 性能门槛 G2：**{g2["passed"]}/{g2["total"]}**（`{g2["status"]}`）
- 默认 auto 精度模式：**{precision["passed"]}/{precision["total"]}**（`{precision["status"]}`）
- 模型工件：[9 个 GGUF 清单](full_gguf_manifest.json)，GGUF 发布 commit `2377ca3539bf5c4f09161a886a38dd0fc7517bd0`
- 报告身份与下载：[summary/current.json](summary/current.json)

`accepted_deviation` 不表示机器门槛通过；它保留失败测量，并按同图配对比值判定每一行均未以 95% 置信度超出 materiality 带、且 PyTorch 参考在窗口内的漂移未超出该带。

## 默认精度规则

默认请求为 `auto`：仅 CUDA 加载全 F32 矩阵时生效 `strict_f32=1`；其余 CPU、Vulkan、F16、Q8_0 配置生效 `strict_f32=0`。显式 `--strict-f32` / `--no-strict-f32` 仅作诊断或覆盖，不代表默认发布路径。

strict 保留 F32 GEMM/softmax/norm 与显式 F32 attention，避免 TF32/F16 中间算术触发 top-k 边界换对；non-strict 使用更快的 TF32、flash F16 K/V 或量化激活路径。F16/Q8 已丢失的存储精度无法由 strict 恢复。

## 已接受偏差

{failures}

## 覆盖范围

- 3 模型 × 3 后端 × 3 存储，共 27 个主配置，每配置 50 图。
- 2 次预热、5 次重复；GPU 使用前后 PyTorch bracket 和 GPU idle 检测。
- stage/dynamic、dense sampler、量化 oracle、可选接口、真实 C API、PSG/IndoorVG/VG150/SpatialSense 子集。
- R/mR/F1/wR、候选集合、logits 误差、P50/P95、吞吐、后端内存均保留在报告中。
- 离线图表：[gallery](gallery/README.md)，机器状态：[acceptance_status.json](gallery/evidence/acceptance_status.json)。

## 非目标

- 不承诺 F16/Q8 与原始 F32 逐位一致。
- 公共子集回归不等于完整 leaderboard；缺失的标注资产不会由推理输出替代。
- accepted deviation 不修改原始 G2 18/18 门槛。

## 复现

```sh
python3 cpp_ggml/scripts/run_alignment.py --run-id my-auto-release --warmup 2 --repeats 5
python3 cpp_ggml/scripts/export_alignment_report.py --run cpp_ggml/outputs/runs/my-auto-release
```

本次导出包含 {figure_count} 张 PNG/SVG 图；完整原始数组保留在不可覆盖的运行目录中。

## 端到端延迟

{timing_table(summary)}
'''
    (folder/'RELEASE_STATUS.md').write_text(status_text)
    redirect = '# 兼容入口\n\n发布验收状态已统一迁移至 [RELEASE_STATUS.md](RELEASE_STATUS.md)。\n'
    for name in ('ALIGNMENT_STATUS.md', 'NEXT_WINDOW_TODOS.md'):
        (folder/name).write_text(redirect)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'cpp_ggml/benchmarks/gallery')
    args = parser.parse_args()
    export(args.run.resolve(), args.output.resolve())
    if args.output.resolve() == ROOT/'cpp_ggml/benchmarks/gallery':
        from report_comparison_index import main as comparison_index
        comparison_index()


if __name__ == '__main__':
    main()
