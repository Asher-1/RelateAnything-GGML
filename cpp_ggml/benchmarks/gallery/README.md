# 完整模型、后端与量化对比

[打开离线交互画廊](index.html) · [逐行数据](comparison.csv) · [首个逐位差异](numerical_first_divergence.json)

每个模型、后端与存储精度均包含全部 50 图。箭头表示方向，结果表显示谓词、分数、参考排名和精确字符串 GT 命中。GT 是不完备正例标注，未命中不能据此判为错误。所有耗时是图像到校准三元组的端到端时间，不含模型加载和导出。速度图红色行未满足快于官方 Python 的目标。原始 FP32 和同量化权重 oracle 是两个参考，不能混用。

![accuracy_candidates.png](accuracy_candidates.png)

![latency_all_backends.png](latency_all_backends.png)

![latency_gpu_backends.png](latency_gpu_backends.png)

![precision_speed_tradeoff.png](precision_speed_tradeoff.png)

![reference_and_storage_precision.png](reference_and_storage_precision.png)

![relsgg-vitb16-cpu-f16-numerics.png](relsgg-vitb16-cpu-f16-numerics.png)

![relsgg-vitb16-cpu-f32-numerics.png](relsgg-vitb16-cpu-f32-numerics.png)

![relsgg-vitb16-cpu-q8_0-numerics.png](relsgg-vitb16-cpu-q8_0-numerics.png)

![relsgg-vitb16-cuda-f16-numerics.png](relsgg-vitb16-cuda-f16-numerics.png)

![relsgg-vitb16-cuda-f32-numerics.png](relsgg-vitb16-cuda-f32-numerics.png)

![relsgg-vitb16-cuda-q8_0-f32-input-numerics.png](relsgg-vitb16-cuda-q8_0-f32-input-numerics.png)

![relsgg-vitb16-cuda-q8_0-numerics.png](relsgg-vitb16-cuda-q8_0-numerics.png)

![relsgg-vitb16-latency_distribution.png](relsgg-vitb16-latency_distribution.png)

![relsgg-vitb16-vulkan-f16-numerics.png](relsgg-vitb16-vulkan-f16-numerics.png)

![relsgg-vitb16-vulkan-f32-numerics.png](relsgg-vitb16-vulkan-f32-numerics.png)

![relsgg-vitb16-vulkan-q8_0-numerics.png](relsgg-vitb16-vulkan-q8_0-numerics.png)

![relsgg-vits16-cpu-f16-numerics.png](relsgg-vits16-cpu-f16-numerics.png)

![relsgg-vits16-cpu-f32-numerics.png](relsgg-vits16-cpu-f32-numerics.png)

![relsgg-vits16-cpu-q8_0-numerics.png](relsgg-vits16-cpu-q8_0-numerics.png)

![relsgg-vits16-cuda-f16-numerics.png](relsgg-vits16-cuda-f16-numerics.png)

![relsgg-vits16-cuda-f32-numerics.png](relsgg-vits16-cuda-f32-numerics.png)

![relsgg-vits16-cuda-q8_0-f32-input-numerics.png](relsgg-vits16-cuda-q8_0-f32-input-numerics.png)

![relsgg-vits16-cuda-q8_0-numerics.png](relsgg-vits16-cuda-q8_0-numerics.png)

![relsgg-vits16-latency_distribution.png](relsgg-vits16-latency_distribution.png)

![relsgg-vits16-vulkan-f16-numerics.png](relsgg-vits16-vulkan-f16-numerics.png)

![relsgg-vits16-vulkan-f32-numerics.png](relsgg-vits16-vulkan-f32-numerics.png)

![relsgg-vits16-vulkan-q8_0-numerics.png](relsgg-vits16-vulkan-q8_0-numerics.png)

![relsgg-vits16plus-cpu-f16-numerics.png](relsgg-vits16plus-cpu-f16-numerics.png)

![relsgg-vits16plus-cpu-f32-numerics.png](relsgg-vits16plus-cpu-f32-numerics.png)

![relsgg-vits16plus-cpu-q8_0-numerics.png](relsgg-vits16plus-cpu-q8_0-numerics.png)

![relsgg-vits16plus-cuda-f16-numerics.png](relsgg-vits16plus-cuda-f16-numerics.png)

![relsgg-vits16plus-cuda-f32-numerics.png](relsgg-vits16plus-cuda-f32-numerics.png)

![relsgg-vits16plus-cuda-q8_0-f32-input-numerics.png](relsgg-vits16plus-cuda-q8_0-f32-input-numerics.png)

![relsgg-vits16plus-cuda-q8_0-numerics.png](relsgg-vits16plus-cuda-q8_0-numerics.png)

![relsgg-vits16plus-latency_distribution.png](relsgg-vits16plus-latency_distribution.png)

![relsgg-vits16plus-vulkan-f16-numerics.png](relsgg-vits16plus-vulkan-f16-numerics.png)

![relsgg-vits16plus-vulkan-f32-numerics.png](relsgg-vits16plus-vulkan-f32-numerics.png)

![relsgg-vits16plus-vulkan-q8_0-numerics.png](relsgg-vits16plus-vulkan-q8_0-numerics.png)

![same_weight_precision.png](same_weight_precision.png)

![speedup_all_backends.png](speedup_all_backends.png)

## 全链路与接口证据

[七项验收状态](evidence/acceptance_status.json) · [1500 个 native 输出审计](evidence/full_graph/release_audit.json) · [首个 stage 差异](evidence/stages/first_numerical_divergence.json)

![所有 stage 对 CPU 参考](stage_errors_cpu.png)

![所有 stage 对原版 CUDA 参考](stage_errors_cuda.png)

纯 C 接口真实模型验证：[CPU](evidence/c_api-cpu.json)、[CUDA](evidence/c_api-cuda.json)、[Vulkan](evidence/c_api-vulkan.json)，每个后端九个模型存储组合。包含借用图像内存与文件输入逐位比较、取消、结果生命周期和宿主后端 lease。

[单补丁重放](evidence/patch_replay.json) · [编译源 Struct Options 审计](evidence/runtime_options_audit.json) · [固定构建 CTest](evidence/build_test_audit.json) · [协议资产访问结果](evidence/protocol_asset_access.json)
