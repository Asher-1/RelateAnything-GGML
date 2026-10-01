# RelateAnything GGML 对齐与测量

**全部图表入口：[ALL_COMPARISONS.html](ALL_COMPARISONS.html)**。
[可直接预览的 Markdown](ALL_COMPARISONS.md)包含 42 张完整基线图和 10 张新修复重测图；
关键图表使用 [gallery](gallery/README.md) 中的普通 PNG/SVG 文件。

本轮 strict 算子修复与专项重测见 [optimization/index.html](optimization/index.html)：
新 CUDA 端到端延迟、原版 CUDA 数值精度、Q8 F32-input、27 组 stage/dynamic、
精确 patch embedding 与下一处分歧。原完整基线和新测量有明确标签。

**[后续全部 TODO、执行顺序与验收条件](NEXT_WINDOW_TODOS.md#后续需要完成的全部-todo)**
统一记录数值/性能修复、五类协议资产、完整回归/图表更新、测试包发布和 ACloudViewer 接入。

在浏览器打开 [index.html](index.html) 或下方画廊即可查看结果。
[测试图片与标注预览](../test_data/index.html)、[测试 ZIP 与自动填充说明](../test_data/PACKAGE.md)
提供原图、对象框、关系标注和可缓存的输入数据。

完整运行的可下载报告、来源、哈希和验收身份由
[跨平台发布摘要](summary/README.md)固定。图像、数组与命令日志位于本地
`outputs/runs/<run-id>/artifacts` 和 `logs`；`reports` 是可重建的报告视图。

[离线交互对比画廊](gallery/index.html)覆盖 S / S+ / B、F32 / F16 / Q8_0、
CPU / CUDA / Vulkan 以及 CUDA Q8 F32-input，提供所有 50 张图的 GT、官方
PyTorch-CUDA 和 GGML 关系图、谓词、分数、候选对与逐关系排名。

![全部模型与后端的端到端延迟](gallery/latency_all_backends.png)

![候选对与精度](gallery/accuracy_candidates.png)

- [全部详细图与数据](gallery/README.md)：PNG / SVG、P50 / P95、逐图与逐次分布、原始 logits 误差、同权重 oracle、首个逐位差异。
- [完整报告下载、哈希与运行身份](summary/README.md)。
- [协议与指标范围](METRIC_COVERAGE.md)、[资产来源](OFFICIAL_DATA.md)、[九个模型导出](EXPORT_STATUS.md)。
- [验收清单](NEXT_WINDOW_TODOS.md)与 [机器验收状态](gallery/evidence/acceptance_status.json)。

计时范围是图像解码到校准三元组的完整过程，包含图构造、分配、传输与后处理，
不含模型加载和导出。官方 Python 在每个模型的 native 矩阵前后独立计时，保存漂移。
严格数值一致和全部配置更快是独立验收条件；图中红色速度行为未通过。
GT 是不完备正例；精确字符串命中不能替代带标注资产的同义词校准。

```sh
python3 cpp_ggml/scripts/run_alignment.py --run-id my-alignment --current --warmup 2 --repeats 5
```

旧日期目录、过时图表、历史数组和失败探测的中间结果已清理；
[删除清单](cleanup_manifest.json)记录路径、大小和身份文件哈希。
新探测写入 `outputs/probes/`，完整运行写入唯一 run ID；完整运行审核后才更新 current。
