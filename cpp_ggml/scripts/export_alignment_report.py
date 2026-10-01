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
    if status['execution']['strict_rows'] != 30:
        raise RuntimeError('all 30 native precision/timing configurations are required')
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
    todo_path = folder/'NEXT_WINDOW_TODOS.md'
    continuation = ''
    if todo_path.is_file():
        previous = todo_path.read_text()
        start = '<!-- continuation-todos:start -->'
        end = '<!-- continuation-todos:end -->'
        if start in previous or end in previous:
            if previous.count(start) != 1 or previous.count(end) != 1:
                raise RuntimeError('invalid continuation TODO markers; refusing to overwrite the checklist')
            begin, finish = previous.index(start), previous.index(end)
            if finish < begin:
                raise RuntimeError('reversed continuation TODO markers; refusing to overwrite the checklist')
            # Execution summaries are generated; outstanding work is maintained
            # by the user/agent and must survive the next compact report export.
            continuation = previous[begin:finish+len(end)]+'\n\n'
    summary = read(reports/'full_graph/summary.json')
    versions = sorted({read(reports/f'full_graph/{m}/python/results.json')['torch_version'] for m in MODELS})
    bit = status['details']['strict_bit_equality']
    slow = status['details']['strict_faster_than_python']['slower_or_equal_rows']
    joint = status['details']['exact_precision_and_speed']['passing_native_configs']
    statuses = {row['id']: row for row in status['statuses']}
    common = (f'当前发布运行：`{run_id}`。'
              '[机器验收状态](gallery/evidence/acceptance_status.json)与'
              '[跨平台报告摘要](summary/README.md)分别记录稳定证据和下载身份。\n\n'
              '全部源修改和构建冻结后重新运行；执行完成不等于精度、速度或协议验收通过。'
              '离线详细图和机器证据保存在 [gallery](gallery/README.md)，无需旧日期目录。\n\n')
    table = timing_table(summary)
    coverage = ('- 27 主配置 × 50 图：S / S+ / B、CPU / CUDA / Vulkan、F32 / F16 / Q8_0。'
                '另有 3 × 50 图 CUDA Q8 F32-input；总计 1500 个 native 图像配置、150 个官方 Python 图像。\n'
                '- 27 组 stage/dynamic，对独立 CPU 和原始 CUDA 参考分别定位首个不同浮点位；'
                '27 dense、固定候选对误差分析和九组可选接口。\n'
                '- 27 组真实模型 C API 验证：内存/文件输入、五种像素格式、stride/capacity、取消、'
                '独立结果生命周期、宿主 backend lease。CTest：CPU 5/5、CUDA 6/6、Vulkan 6/6。\n'
                '- 9 个 GGUF 存储权重 CPU oracle × 50 图，对 CPU/CUDA/Vulkan 分别比较；'
                '存储误差、参考设备误差和 native 算术误差分别报告。\n'
                '- PSG / IndoorVG / VG150 共 81 个模型配置子集回归，SpatialSense valid/test 共 54 个；'
                '全部使用对应公开标注，不能代表完整 leaderboard。\n'
                '- GGML v0.21.0 (`8599e0ea3756c4bac4ef813af2241cb1a8bbfb0b`) 的'
                '一个合并补丁已完成 pristine 正向/反向/还原重放，当前依赖树逐文件一致。'
                '401 个编译源/头文件通过环境开关审计，运行策略由版本化 Struct Options 显式配置。\n\n')
    numerical = (f'原版 PyTorch-CUDA 的逐位对照覆盖 {bit["original_cuda_comparisons"]} 个 native 配置，'
                 f'最小 logits 逐位相同比例为 {bit["original_cuda_minimum_logit_bit_equal_fraction"]:.8%}。'
                 '同时比较候选对、pair logits 和谓词 logits，近似误差或相同候选集不代表逐位一致。\n\n'
                 'F16/Q8 权重转换会丢失原始 FP32 信息；strict F32 控制算术精度，不能恢复已丢失的权重位。'
                 'F32 也保留不同 kernel/reduction 的浮点残差。'
                 '[逐 stage 首个差异](gallery/evidence/stages/first_numerical_divergence.json)'
                 '记录坐标、值和 IEEE-754 位，仅定位已观测边界，未把它称为内部算子的最终根因。\n\n')
    performance = ('上游 Python 推理源码在本机 PyTorch '+', '.join(versions)+' 环境执行。'
                   +f'{summary["warmup"]} 次预热、{summary["repeats"]} 次重复，50 图 P50，单位 ms。'
                   '完整图像解码到校准三元组，包含预处理、图构造/分配、传输、推理和排序；'
                   '模型加载与写文件不计入。官方 Python 在每个模型 native 矩阵前后独立计时。\n\n'
                   +table+'\n\n'
                   +f'30 个 native 配置中 {len(slow)} 个未快于 PyTorch-CUDA；'
                   f'同时满足原始 CUDA 逐位一致和速度更快的配置为 {len(joint)} 个。'
                   '这是 RTX 3060 共享工作站的实测结果。'
                   '[全局图与逐次分布](gallery/README.md)保存 P50/P95、慢图和未剔除的测量分布。\n\n')
    assets = ('五类协议资产仍缺失：同义词/反义词及 text-space 校准、裁定正负例 Haystack、'
              '固定 detector 与 operating point、judge/HICO/OVS 控制及归一化、'
              '标签完备的 held-out ECE/Brier 分布。'
              '不能把未标注样本当负例，也不能用自动生成的同义词代替对应标注资产。'
              '[资产来源与访问结果](OFFICIAL_DATA.md)、[指标定义](METRIC_COVERAGE.md)。\n\n')
    (folder/'ALIGNMENT_STATUS.md').write_text('# 对齐状态入口\n\n'
        '验收状态统一维护在 [NEXT_WINDOW_TODOS.md](NEXT_WINDOW_TODOS.md)，避免两份清单产生冲突。\n\n'
        '[全部对比图](ALL_COMPARISONS.html)覆盖完整基线与专项重测；'
        '[接口与 AICore 集成边界](PORTING_NOTES.md)。\n')
    focused = ''
    optimization = folder/'optimization/summary.json'
    if optimization.is_file():
        progress = read(optimization)
        # Attach only the optimization measured against this complete baseline.
        # A later baseline must not inherit an unrelated earlier progress claim.
        prefix = str((reports.parent/'artifacts/full_graph').resolve())+'/'
        if any(path.startswith(prefix) for path in progress['source']['files']):
            faster = sum(row['speedup_vs_python'] > 1 for row in progress['rows'])
            capture = progress['capture_profile']
            focused = ('## 本轮 strict 专项重测\n\n'
                '[修复与图表](optimization/index.html)：'+progress['scope']+'。\n\n'
                f'新 CUDA 主矩阵 {faster}/{len(progress["rows"])} 的 P50 快于同期官方 Python；'
                '全模型原版逐位一致和全部 strict 配置更快仍未达成。'
                f'另有 {len(progress["stage_rows"])} 组 stage/dynamic 和 '
                f'{progress["operator_cases"]} 个精确 softmax 用例。\n\n'
                f'端到端 profiler 记录 {capture["cuda_capture_begins"]} 次 capture、'
                f'{capture["cuda_graph_launches"]} 次 graph launch；'
                '编译启用 capture 不代表图重放加速已发生。'
                '完整基线与专项重测分别标明来源，专项重测不替换完整验收状态。\n\n')
    labels = [('strict_bit_equality', '原始 PyTorch-CUDA 逐位一致', 'gallery/numerical_first_divergence.json'),
              ('strict_faster_than_python', '每个 strict 配置快于 PyTorch-CUDA', 'gallery/evidence/full_graph/summary.json'),
              ('soft_open_vocab', 'SoftR / SoftmR / SoftF1、open-vocabulary A3', 'OFFICIAL_DATA.md'),
              ('haystack_precision', 'Haystack fAP / rare fAP、P-AUC / PDD / PDO', 'OFFICIAL_DATA.md'),
              ('sgdet_detector_metrics', 'SGDet、box AP/mAP、detector retention', 'OFFICIAL_DATA.md'),
              ('judge_hico_ovs', 'judge、counterfactual、HICO/zero-shot、OVS', 'OFFICIAL_DATA.md'),
              ('ece_brier', 'ECE / Brier', 'OFFICIAL_DATA.md')]
    lines = ['| 验收项 | 状态 | 证据 / 所需资产 |', '|---|---|---|']
    for key, label, link in labels:
        row = statuses[key]
        lines.append(f'| {label} | **{row["status"]}** | [{row["reason"]}]({link}) |')
    todo_path.write_text('# GGML / upstream Python 最终验收\n\n'+common
        +continuation+focused
        +'## 七项验收\n\n'+'\n'.join(lines)+'\n\n'
        +'`missing_asset` 是协议输入缺失，不能标作模型通过或失败。取得资产后须记录版本、许可、'
        'split、标注哈希和评测脚本哈希，创建新运行。\n\n'
        +'## 已完成的重构与验证\n\n'+coverage
        +'## 完整基线精度与性能结论\n\n'+numerical+performance+assets
        +'## 复现与结果保留\n\n'
        +'```sh\npython3 cpp_ggml/scripts/run_alignment.py --run-id my-alignment --current --warmup 2 --repeats 5\n'
        +'python3 cpp_ggml/scripts/export_alignment_report.py --run cpp_ggml/outputs/runs/my-alignment\n```\n\n'
        +'运行目录不可覆盖。完整 suite 和审计完成后原子更新 `benchmarks/current`；'
        f'导出脚本只复制紧凑报告和 {figure_count} 张 PNG/SVG 图，不复制大数组。'
        '替代结果验证完成后，按 [清理清单](cleanup_manifest.json)移除旧运行和 probes。\n\n'
        +'## 可视化与测试数据\n\n'
        +'在浏览器打开 [全量图表](ALL_COMPARISONS.html)、[输出入口](../outputs/index.html) 或 [离线画廊](gallery/index.html)。'
        '全部模型/后端/存储配置的原图关系对照、逐配置数值图、端到端延迟和 stage 图均可查看。\n\n'
        +'[测试数据预览](../test_data/index.html)包含 100 个样本入口；'
        '99 个具备至少两个标注对象，可以自动填充。'
        '[ZIP 与缓存接口](../test_data/PACKAGE.md)记录目录、校验和 XYXY 框/类别字段。'
        '本地包已生成，远端 URL 尚未发布；ACloudViewer 按钮连接不作为已完成项目。\n\n'
        +'```sh\npython3 cpp_ggml/scripts/package_test_data.py\n'
        +'python3 cpp_ggml/scripts/package_test_data.py --verify-only\n'
        +'python3 cpp_ggml/scripts/verify_visual_outputs.py\n```\n\n'
        +'浏览器验证需要 Playwright 和 Chrome，仅用于检查；查看 HTML 和生成 ZIP 无此依赖。\n')


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
