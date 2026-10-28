# GGML 发布验收状态

## 当前结论

- **最新完整验收（20261028 核验）：`20261026-dw7vk-release` 已在重启前完成，6 suites × 3 backend、33/33 步骤；G1 63/63、auto_precision 63/63、G2 42/42，`release_decision=complete`，原始 fail、tracked、block、inconclusive 均为 0。CUDA/Vulkan 7×7 depthwise 工程验收闭合，无需恢复或重复长跑。**
- **重启后复核**：三后端二进制 SHA256 与 run 登记一致；`benchmark_release.py verify-run` 通过；以只读 `verify_benchmark_release.audit()` 重新核验 3150 个主配置图像 + 350 个诊断配置图像，70/70 配置的原始数值、解码、权重/输入/源码身份与最终无竞争 attempt 均通过。Python 回归测试 30/30，三后端 ctest 5/5、6/6、6/6。`benchmarks/current` 已切至本 run；HF 指针及 checked-in gallery 尚未同步，本任务未执行外部发布。
- **清理收口（20261028）**：用户已删除 `outputs/scratch/dwconv/` 下的 `precheck/`、`rafo-final/`、`lib-old/`、`lib-old-vk/`，四目录原占约 4.4 GiB；逐路径核验均不存在，`verify-run` 再次通过。Vulkan 草稿与失效调参脚本此前已删除，清理待办关闭，详情见 `cleanup_manifest.json`。正式 run、A/B 日志、新旧 RAFO 对照与 profiling 原始证据继续保留。
- 21 个 GGUF 已发布（7 个 checkpoint × F32/F16/Q8_0：三个 DINOv3 ViT 塔 + 四个 DINOv3 ConvNeXt 塔）并经远端回下载 SHA-256 12/12 校验（ConvNeXt 新增部分）。
- ConvNeXt 全链路适配完成：导出 metadata、C++ token-major backbone（cwhn dwconv）、Stage 融合，下游 head 与 ViT 共用。
- 图层优化已合入（纯 C++ 层，无 GGML 源码新增改动）：token-major（CWHN）深度卷积零拷贝、LayerNorm affine 与 LayerScale 的加载期折叠——图节点数 497 → 约 400。
- 完整发布运行 `20261014-pure-cpp-release`：G1 63/63、auto_precision 63/63、G2 28/42（14 项未达标，其中 8 项边缘偏差遗留待定）。
- 完整发布运行 `20261016-gelu-fusion-release`（GELU epilogue 融合工程验收，含 c-api/metrics 全部 suites 收尾）：G1 63/63、auto_precision 63/63、G2 32/42。
- 正式判定运行 `20261017-clean-release`（无外部计算进程时段，`--suites full`）：**G1 63/63、auto_precision 63/63、G2 27/42，发布决策 blocked**。此前“7 项为污染假 fail、干净口径 39/42、cuda 三项在噪声带”的结论被本次 run 推翻，更正见下文。
- 捕获运行 `20261018-vk-slowmode-capture`（`--suites full`，日间、无外部计算进程）：**慢模式未出现**，vulkan 全程快模式且仪器健康（P0、SM 1935–1950 MHz、mem 7500 MHz、60–67°C）。**G1 63/63、auto_precision 63/63、G2 32/42（10 fail），blocked**。
- 高判据运行 `20261019-repeats15-r2`（`--repeats 15`，75 帧/配置，含 infer() 分段探针）：**慢模式未再现**，vulkan 全程快模式。**G1 63/63、auto 63/63、G2 34/42（8 fail，白名单 accepted=3 后 remaining=5），blocked**。关键变化：**tiny vulkan-f16/f32 转正**（高精度中位数落回 python bracket 内，证实此前 1.05–1.10 的 fail 有相当部分是 5 帧中位数的窗口摆动，与双窗口探针结论一致）；small vulkan ×3 仍 fail（1.021–1.047，真实小幅差距证据增强）；tiny/vulkan-q8_0 1.020 临界。cuda 三行稳定（base f32 1.034、large f32 1.041、small f32 1.005 → 已接受；large f16 1.007 → 独立工程）。**发布决策更新（20261020）**：small vulkan ×3 与 tiny/vulkan-q8_0 按"测量窗口摆动带内的临界差距"并入 accepted deviation（`write_acceptance_status.py` 具名 (model, config) 集合，证据=双窗口探针符号翻转 + repeats=15 持久 1.02–1.05）——r2 重生成后 **accepted=7、remaining=1（large/cuda-f16）**，blocked 仅剩该独立工程项。
- **处置更新（20261024，第一性原理评审）**：large/cuda-f16 从“独立工程立项”降级为 accepted deviation——borderline 集合加入 `(relsgg-convnext-large, cuda-f16)` 后 r2 重生成：**accepted=8、remaining=0，decision=accepted_with_performance_deviations**。理由：差距 1.007（≈0.5 ms/帧）低于全部已接受项（1.02–1.05）；代码证据显示缺口形态是图级 `GGML_EPILOGUE_GELU` hint 已打在 pwconv1 add 上、Vulkan 已消费（`add_gelu.comp`）而 CUDA F16 无消费者，且 strict cublasLt `GELU_BIAS` epilogue 为 F32 domain + tanh 近似、与模型 GELU_ERF 不符——原“cuBLASLt/epilogue”方案命名存在已验证冲突。工程面随之重定义：**归因探针先行**（kernel 级 native vs python 对照：GELU 链合计差 ≥0.3 ms 坐实 → 天级 CUDA 写出融合修复，复用已有 hint；弥散 → 关案），不再是发布阻塞项。**探针当日完成并坐实，见下文处置小节。**
- **终态运行 `20261024-current-r2`（20261025 判读）**：全 6 suites × 3 backend × 33 步骤 complete（`--repeats 15`，75 帧/配置；`RELATE_ALLOW_GPU_PIDS` 放行常驻 nxnode）。**G1 63/63、auto_precision 63/63、G2 39/42（3 fail 全部落在白名单内）、decision=accepted_with_performance_deviations——run 内原生达成，remaining=0**。3 fail：base/cuda-f32 1.014、large/cuda-f32 1.045（strict 类）、small/vulkan-f16 1.000（borderline，0.05 ms 压线）。**处置验证**：20261024 降级的 large/cuda-f16 本次 **0.960 pass**——归因是融合修复而非摆动翻转：r2 为全后端重建后的融合二进制，native 较 1014 基线 −12.3% vs python −7.8% 的差值与预测收益（−2.9 ms/帧 ≈ −3.9pp）吻合（预测带 0.96–0.98 兑现）；修复前 1.007“小幅、临界”的量级判断仍然成立。全部 vulkan 行 0.73–1.00 且快模式（SM 1927–1942/P0、mem 7500，慢模式第 5 窗口未再现，出现率维持 2/5 run）。persistent 仅剩 base/large cuda-f32 两行（20261025 归因更正：同等精度下的 depthwise 卷积 kernel 效率差，非 strict 精度代价，见下文）。
- **G2 判据重设计（20261025 落地）**：二值 `native > python` + 两个手工白名单（cuda-f32 整类、borderline 具名集合）**全部退役**，改为逐图配对比值 + materiality 带 m=5%（block / inconclusive→rerun / tracked 三态，见“G2 materiality 判据”）。r2 按新规则：**3 fail 全部 tracked、decision=accepted_with_performance_deviations**（与旧判定一致，但不再依赖任何白名单）；新判定写在旁路文件 `outputs/runs/20261024-current-r2/g2_materiality_rejudge.json`，已发布 run 的登记证据未改动（verify-run 通过）。
- **报告视图切换（20261025）**：`set_current` 完成，[benchmarks/current](current) symlink 由 1014-reports 切至 **20261024-current-r2/reports**（本次全量 run 即为正式 gallery/报告视图；此前发布窗口缺 `--current` 标志，已补）。**HF 发布同步完成（20261025）**：[summary/current.json](summary/current.json) 已切至本 run 发布包 `reports/20261024-current-r2.zip`（212 MB / 2276 files，zip_sha256 `cb7b2543…`，HF commit `b3c7a07f`，URL 302→200 自检通过）。

## 默认精度规则

默认请求为 `auto`：仅 CUDA 加载全 F32 矩阵时生效 `strict_f32=1`；其余 CPU、Vulkan、F16、Q8_0 配置生效 `strict_f32=0`。显式 `--strict-f32` / `--no-strict-f32` 仅作诊断或覆盖，不代表默认发布路径。

strict 保留 F32 GEMM/softmax/norm 与显式 F32 attention，避免 TF32/F16 中间算术触发 top-k 边界换对；non-strict 使用更快的 TF32、flash F16 K/V 或量化激活路径。F16/Q8 已丢失的存储精度无法由 strict 恢复。

## 发布门槛

最新 `20261026-dw7vk-release`：G1 **63/63**、G2 **42/42**、auto_precision **63/63**；下表保留此前 run 的历史判定，不代表当前残留项。

| 门槛 | 上一稳定 run | 20261014-pure-cpp-release | 20261016-gelu-fusion-release | 20261017-clean-release | 20261018-vk-slowmode-capture | 20261019-repeats15-r2 | 20261024-current-r2 | 说明 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| G1 数值 | 27/27 | 63/63 | 63/63 | 63/63 | 63/63 | 63/63 | **63/63** | 7 模型 × 9 配置 |
| G2 GPU 性能 | 17/18 | 28/42 | 32/42 | 27/42 | 32/42 | 34/42 | **39/42** | 原始 fail 计数（`failed` 语义不变）；r2 的 3 fail 按 materiality 判据全部 tracked（block=0、inconclusive=0），**decision=accepted_with_performance_deviations**，见下表 |
| 工件身份 | 9/9 | 21/21 | 21/21 | 21/21 | 21/21 | 21/21 | **21/21** | [full_gguf_manifest.json](full_gguf_manifest.json) 与 Hugging Face 一致（release-audit complete） |

## G2 残留：六 run 持久性（20261025 历史；20261026 新 kernel 验收全部通过）

判据为跨 run 符号一致性（native/python 比值在每个 run 都 >1 即 persistent），不依赖噪声 ε 估计——ViT 对照组 ε（本次 +6.3%）过宽，曾把 1.115× 判成 noise_band，现 [analyze_g2_residuals.py](../scripts/analyze_g2_residuals.py) 以 `--history` 持久性优先。比值顺序 20261014 / 20261016 / 20261017 / 20261018（20261019-r15 与 20261024-r15 见上两条目）：

| 配置 | 比值 | 判定 | 归因 |
|---|---|---|---|
| `convnext-base/cuda-f32` | 1.039 / 1.019 / 1.037 / 1.034 | persistent | depthwise 卷积 kernel 效率差（20261025 探针，见下；原“strict F32 代价”归因撤回） |
| `convnext-large/cuda-f32` | 1.033 / 1.045 / 1.115 / 1.040 | persistent | 同上；1.115 为 1017 孤例 |
| `convnext-large/cuda-f16` | 1.010 / 1.017 / 1.055 / 1.009（r15: 1.007；r2: **0.960**） | 已修复（persistent → intermittent） | 融合修复前为真实小幅差距（5-run 符号一致）；20261025 gate_pass，borderline 集合移除（见下） |
| `convnext-tiny/vulkan×3` | 1.05–1.14 / 1.71–1.84 / 1.79–1.86 / 1.05–1.10 | persistent | 快模式残余 1.05–1.14×，慢模式时 1.8×（见下节） |
| `convnext-small/vulkan×3` | 1.10–1.12 / 0.96–0.98 / 1.87–1.91 / 1.02–1.05 | intermittent | 快模式临界 + 慢模式 |
| `convnext-base/vulkan×3` | 0.98–1.02 / 0.89–0.92 / 1.75–1.79 / 0.91–0.95 | intermittent | 慢模式时 fail，快模式过线（融合收益 0.82–0.95×） |
| `convnext-small/cuda-f32` | 1.083 / 0.977 / 1.010 / 1.006 | intermittent | 临界（同为 strict 路径） |
| `convnext-large/cuda-q8_0` | 0.972 / 0.976 / 1.016 / 0.969 | intermittent | 临界 |
| `vits16plus/vulkan-f32` | 0.984 / 0.870 / 1.016 / 0.971 | intermittent | 临界 |

ViT 全部行在 20261018 过线（cuda-f32 0.905/0.941/0.994，vulkan 0.808–0.892）；遗留 ViT 三项在 20261017 也已过线（0.917 / 0.985 / 0.942）。

**六 run 汇总（20261025）**：persistent 仅剩 `base/cuda-f32`（六值 1.014–1.039）与 `large/cuda-f32`（1.033–1.115）两行——归因见下文 20261025 更正（depthwise 卷积 kernel，非精度代价），按 materiality 判据为 tracked。`large/cuda-f16` 第 6 run 0.960 转正——归因是融合修复而非摆动翻转（r2 为融合二进制，native −12.3% vs python −7.8% 的差值与预测收益吻合，且修复前 5-run 符号一致支持真实差距判定）；borderline 放行在修复落地前是正确的风险处置，实跑兑现为 gate_pass。全部 vulkan 行在 20261024 run 0.73–1.00（vitb16 vulkan 0.731–0.765 大幅优于 python）——tiny/small 快模式残余未再现。

**快模式基线（20261018 无慢模式 run 确认）**：base/large vulkan 0.82–0.95×——GELU 融合收益在干净 run 复现；tiny vulkan 1.05–1.10×（四个 run 全部 >1）、small vulkan 1.02–1.05×——两个最小塔对 vulkan 后端存在真实小幅差距，疑似与逐 op 提交开销在小模型上相对占比更高有关（CUDA 走 graph replay 掩盖了同类开销，tiny/cuda 0.86–0.99×），未验证。

**strict F32 代价（配对实测，同输入同 binary，交替 2×）**：`convnext-base` strict 46.76/46.96 ms vs `--no-strict-f32` 39.82/39.92 ms（+17%）；`convnext-large` strict 82.84/82.93 vs 76.87/77.37（+7.5%）。即 base/large 的 cuda-f32 残差与已接受偏差 `vitb16/cuda-f32` 同类——默认 auto 在 CUDA 全 F32 权重下启用 strict 的精度优先代价，non-strict 路径可过线（base 39.8 vs python 41.4；large 77 vs 73.5，独立运行 CUDA 绝对值较 run 内偏慢约 5%，属临界）。**白名单已落地**：`write_acceptance_status.py` 改为按配置类接受全部 cuda-f32 行（strict-F32 精度优先代价，非逐行手列）；1018 run 重生成后 accepted=3（base/large/small cuda-f32），remaining=7（large/cuda-f16 + tiny/small vulkan×6），g2 仍 open、decision 仍 blocked——gate 计数与 `failed` 列表语义不变，仅新增 `accepted` 字段。（该白名单已于 20261025 退役，见下。）

### 更正：cuda-f32 残差不是 strict 精度代价（20261025，kernel 级归因探针）

“strict 代价”成立的前提是 PyTorch 参考没有付出同等精度代价。实测参考**同样是严格 F32**：`torch.backends.cuda.matmul.allow_tf32=False`、`float32_matmul_precision=highest`（torch 2.6 默认，`benchmark_full_graph.py` 未改）。所以上文的 +17% / +7.5% 是 non-strict（TF32/F16 中间算术）**低于参考精度**换来的收益；“关掉 strict 可以过线”的本质是降精度掩盖差距，而不是精度本身的价格。

nsys 探针（4 图 × (2 warmup + 5 repeats)，同 .raip 输入；native 默认 auto=strict，PyTorch 只抓计时区间；脚本与 trace 在 `outputs/probes/cuda-f32/`，分析脚本为 `analyze.py`），单位 ms/帧：

| 类别 | base native | base python | Δ | large native | large python | Δ |
|---|---:|---:|---:|---:|---:|---:|
| depthwise 7×7 conv | 4.30 | 2.27 | **+2.03** | 7.76 | 3.61 | **+4.15** |
| GEMM（双方同为 `ampere_sgemm_128x64_tn` 主核） | 25.28 | 24.72 | +0.56 | 50.79 | 49.94 | +0.85 |
| copy/cast + other | 3.94 | 3.20 | +0.74 | 4.89 | 4.24 | +0.65 |
| elementwise（native 已折叠/融合） | 0.38 | 1.89 | −1.50 | 0.42 | 2.44 | −2.02 |
| norm + gelu + softmax | 3.38 | 3.84 | −0.46 | 4.64 | 5.14 | −0.50 |
| **GPU kernel 合计（= busy，单流串行）** | 37.28 | 35.92 | **+1.37（+3.8%）** | 68.51 | 65.38 | **+3.13（+4.8%）** |

- **单一主因**：ggml `conv2d_dw_kernel<float, cwhn_layout>`（通用 kernel，kernel_w/h 为运行时参数，逐像素双重循环）比 PyTorch 的 `conv2d_c1_k1_nhwc_kernel<…, 7, …>`（7×7 编译期特化）慢 **1.9–2.1×**，单这一项就超过整个差距（base 148%、large 133%），被 native 的 elementwise 折叠收益部分抵消。GEMM 双方是同一族 SGEMM，差距 2%。
- **与 strict 无关**：depthwise kernel 只有 F32 实例化（权重加载期打包为 F32，[full_model.cpp](../src/full_model.cpp) L445），strict、non-strict、F16、Q8_0 的 CUDA 配置都走这条路径。
- **修复面（20261025 已落地，见下节）**：为 CWHN 布局的 7×7 depthwise 写编译期特化 kernel。按 materiality 判据，这两行修复前是 tracked（4% < 5%），不阻塞发布。

### depthwise 7×7 特化 kernel（CUDA/Vulkan 已落地；`20261026-dw7vk-release` 完整验收通过）

- **机制**：现通用 kernel 每个 tap 两次标量 load，地址由运行时 kernel 尺寸/stride/dilation 现算，循环边界运行时决定、无法展开——large stage 3（C=768、28×28，每帧 27 次）每次 192 µs，而 DRAM 下限约 13 µs、FMA 下限约 5 µs，瓶颈是加载指令数。新 kernel `conv2d_dw_7x7_cwhn_kernel`（[conv2d-dw.cu](../third_party/ggml/src/ggml-cuda/conv2d-dw.cu)）：每线程 4 通道（float4）× 7 个连续输出列，输入行流经寄存器，每个输入 float4 每 kernel 行只读一次；每输出 L1 字节从 392 B 降到约 80 B。条件（cwhn、7×7、stride 1、dilation 1、C%4==0、16 B 对齐）不满足即回退原 kernel。
- **数值**：每个输出的累加顺序与原 kernel 相同（ky 外层、kx 内层升序、越界 tap 跳过、bias 最后加），**位级恒等**——RAFO 4 ConvNeXt × f32/f16/q8_0 × 4 图 = 48/48 字节一致（strict 与 non-strict 路径都覆盖），G1 不受影响。
- **op 级**：`test-backend-ops` 新增 18 个 7×7 cwhn 用例（整块/尾块/batch 2/图小于 kernel/非 same padding/C%4≠0、stride 2、dilation 2 回退/WHCN，各含 bias 与无 bias）+ 4 个 ConvNeXt-large stage 性能用例，22/22 与 CPU 对拍通过。单 op 耗时（µs，旧 → 新）：112×112×192 499 → 163、56×56×384 320 → 85、28×28×768 206 → 60、14×14×1536 109 → 30（3.0–3.6×；tile 宽 2/4 与 64/128 线程块在 ±10% 的轮间波动内没有稳定优势，取 7 —— 整除 112/56/28/14）。
- **整帧（nsys，同协议）**：dwconv 每帧 base 4.30 → **1.40 ms**、large 7.76 → **2.36 ms**，比 PyTorch/cuDNN（2.08/3.31 ms）快 33–29%。同一会话交替 A/B（同可执行文件、只换 `libggml-cuda.so`，3 轮）：base-f32 每轮 −4.0/−6.5/−5.4 ms、large-f32 −12.0/−9.8/−8.4 ms、large-f16 −10.1/−6.3/−25.9 ms（机器 load 17，绝对值偏高、只看轮内差）。所有 CUDA ConvNeXt 配置（含 f16/q8_0）都走此 kernel。
- **验证**：patch 重生成（34 文件）+ patch-replay PASS；三后端重建且指纹与源码一致；三后端 self-test 与 ctest 全过。G2 判定以全量 run 为准。
- **CUDA 判定 run `20261025-dw7-release`（22:44 → 次日 07:08，单二进制干净 run）**：G1 63/63、auto_precision 63/63；**base/large cuda-f32 转正**（0.952 / 0.964，r2 为 1.014 / 1.045），全部 CUDA ConvNeXt 行 native 较 r2 降 1.1–5.8 ms。G2 36/42，decision=**blocked**：tiny/vulkan×3 判 block（配对 1.061 / 1.073 / 1.101，lo 1.051 / 1.054 / 1.065，δ 2.2%），small/vulkan×3 tracked（1.026–1.056）。成因两半：本窗口 python 参考比 r2 快 12%（tiny 27.46 → 24.09 ms），而 vulkan 二进制未变、native 反慢 0.9 ms——原来被参考偏慢掩盖的 tiny vulkan 小幅差距（见下文“快模式基线”1.05–1.10×）在 materiality 下露出；且 Vulkan 的 `conv2d_dw.comp` 与 CUDA 原 kernel 同样是逐输出、逐 tap 标量 load 的朴素实现，tiny 每帧 18 次共 1.86 ms（`--backend-profile`）。
- **Vulkan 移植（20261026）**：`conv2d_dw.comp` 新增 `DW7` 变体（f32 / f16 kernel × 有无 bias 四条 pipeline，[ggml-vulkan.cpp](../third_party/ggml/src/ggml-vulkan/ggml-vulkan.cpp) `ggml_vk_conv2d_dw7_eligible`）：与 CUDA 同构——每 invocation 一个 vec4 通道组 × 7 列，输入行经寄存器复用，tap 顺序与 fma 与原 shader 一致；条件（cwhn、7×7、stride 1、dilation 1、C%4==0、src/dst 16 B 对齐、dst F32、kernel F32/F16）不满足即走原 shader。dispatch 数按 tile 数计（`ggml_vk_conv2d_dw7_invocations`）。
  - op 级：`test-backend-ops` 补 6 个 F16 kernel 的 7×7 用例，Vulkan 32/32 与 CPU 对拍通过（CUDA 22/22，F16 kernel 在 CUDA 上 not supported 属既有状态）；单 op 490 → 130、282 → 67、182 → 44、92 → 25 µs（3.7–4.2×）。
  - 数值：RAFO 4 ConvNeXt × f32/f16/q8_0 × 4 图 Vulkan **48/48 字节一致**。
  - 整帧：tiny 每帧 dwconv 1.73–1.86 → **0.60 ms**（f32/f16/q8_0 三种 GGUF 都走新 shader）；同会话交替 A/B（只换 `libggml-vulkan.so`，3 轮，repeats 15）新库 11/12 轮更快，tiny −0.5…−1.8 ms、small −1.4…−2.2 ms。
  - 预检（`benchmark_full_graph.py` tiny/small × vulkan × 3 dtype，repeats 15，bracket python）：配对比值 tiny 0.947 / 0.930 / 0.993、small 0.959 / 0.941 / 0.916。按 dw7 run 偏快的参考（24.09 ms）折算，tiny 仍约 0.98–1.04，最坏落在 tracked。
  - 验证：patch 重生成（34 文件）+ patch-replay PASS；三后端重建，指纹 cpu 7482413b / cuda 1abca16b / vulkan 688ac624；三后端 self-test 与 ctest 全过。
- **最终判定（20261027 16:17 UTC 生成 acceptance，20261028 重启后复核）**：33/33 步骤 complete，77 个结果目录 = 63 个主配置 + 7 个 Python 参考 + 7 个 CUDA Q8/F32-input 诊断；不是完整 suite 的百分比分母。G1 63/63、auto_precision 63/63、G2 42/42，无偏差放行。
- **最终整帧延迟（ms，括号为 native/Python 两中位数之比，不冒充逐图配对比值）**：base CUDA F32 43.18/44.76（0.965）、large CUDA F32 76.13/78.26（0.973）；tiny Vulkan F32/F16/Q8_0 为 24.20/23.40/26.54，对应参考 27.14（0.891/0.862/0.978）。单 op/profile 和交替 A/B 已证明寄存器输入复用的机制收益；跨 run 总延迟同时受参考窗口漂移影响，不能把全部比值变化归为 kernel 收益。
- **数值对照复核**：清理前 CUDA old/new、CUDA new/final、Vulkan old/new 各 48/48 RAFO 字节一致。仅删除与 CUDA new 重复的 final 输出，CUDA old/new 与 Vulkan old/new 对照继续保留；登记 run 的 acceptance、reports.json 与报告源未重写。驱动侧 Vulkan slow-mode 仍是未复现、根因未定的历史风险，本 kernel 工程不宣称修复驱动问题。

### G2 materiality 判据（20261025 落地，[write_acceptance_status.py](../scripts/write_acceptance_status.py)）

旧判据把两个问题混成一个二值比较：“差距是否真实”和“够不够大到挡发布”。结果是噪声附近反复 blocked，每个 run 都靠手工白名单兜底；而 borderline 具名集合**没有幅度上限**（慢模式 1.8× 落到集合内会被静默放行，见上文慢模式节）。新判据对每个原始 fail 行计算**同图配对比值**（50 图 native/python 的中位数，避免两中位数相除时混入图像难度分布）与其 bootstrap 95% 下界 lo，并读取 python bracket 的窗口内漂移 δ：

| 判定 | 条件 | 发布含义 |
|---|---|---|
| block | δ ≤ m 且 lo > 1+m | 有把握慢于 m，挡发布 → `blocked` |
| inconclusive | 缺逐图证据，或 δ > m（参考自身漂移超出政策带） | 无法判断，**重跑而不是放行** → `rerun_required` |
| tracked | 其余 | 不挡发布；保留在 `failed`，并进入跨 run 持久性表 |

m=5%（20261025 拍板）。六 run 回测（真实函数，不改登记证据）：1014 / 1016 / 1017 / 1018 → **blocked**（block 3 / 3 / 9 / 4 行：tiny vulkan 1.18–1.20、慢模式 1.49–1.86、快模式 tiny/small vulkan 1.06–1.12），与当时人工认定的真实问题逐一吻合；1019 → **rerun_required**（tiny 参考漂移 9%）；r2 → **accepted_with_performance_deviations**（base/cuda-f32 配对 1.010、lo 0.995；large/cuda-f32 1.040、lo 1.035；small/vulkan-f16 0.995——全部 tracked）。m=2–3% 时 r2 也会因 large/cuda-f32 被阻断，且漂移中位数为 2.4%，会频繁出现 inconclusive。`export_alignment_report.py` 已同步：放行条件改为 `accepted_with_performance_deviations`（原 `accepted_with_one_performance_deviation` 是陈旧名称，生成器早已不再输出它）。测试见 `tests/test_benchmark_release.py`（tracked / block / 漂移→重跑 / 缺证据→重跑四例）。

### large/cuda-f16 处置（20261024：探针先行 + 发布放行；探针已坐实）

- 判定：真实（5-run 符号一致，持久性判据不依赖 ε），非噪声；量级 0.7%（≈0.5 ms/帧），低于全部已接受项。
- **归因探针（20261024，nsys，4 图 × (2 warmup + 5 repeats)，[outputs/probes/large-f16-attribution.nsys-rep](../outputs/probes/large-f16-attribution.nsys-rep)）**：`strict_f32=0` 确认 non-strict F16 路径；每图 GPU 密集两段（graphA ≈50 ms + graphB ≈7 ms）。**每图 48 个独立 GELU_ERF kernel（`unary_op_kernel<&op_gelu_erf>`，合计 3.0 ms）100% 由独立 `k_bin_bcast` bias-ADD 喂入（193/193 配对）**——机制坐实，可回收 3.0 ms ≫ 0.3 ms 判据，为实测差距（0.5 ms）的 6 倍。
- 结构事实：图级 `GGML_EPILOGUE_GELU` hint 只在 pwconv1 add 打标（graphA 36 site），decoder FFN 的 12 site 未打；Vulkan 消费（`add_gelu.comp`，仅 F32），CUDA 无消费者。cutlass F16 GEMM 占 GPU 时间 58%（cublas GemmEx，与 python 同族 kernel，非差距来源）；其余 ADD（每图 ≈306 个，7.1 ms，residual/LN affine 等）不在本次修复面；D2H/H2D memcpy 为权重上传与 logits 读回，backend 中性。
- 方案约束（维持）：cublasLt 仅 tanh 近似 GELU epilogue，模型为 GELU_ERF——修复形态为 CUDA 自定义写出融合（erf），消费已有 hint；decoder FFN 补打 hint 后覆盖全部 48 site，预期收益 2.3–3.0 ms/图 → 该行预期由 1.007 转正至 ≈0.96–0.98（gate_pass），而非仅关案。
- 程序：修复 = fused add+gelu_erf kernel（对齐 `add_gelu.comp` 语义），过 G1 parity + 全量 acceptance 后重判该行。白名单变更及证据注释见 [write_acceptance_status.py](../scripts/write_acceptance_status.py)。
- **修复已落地（20261024）**：CUDA 消费者在 `ggml_cuda_try_fuse` 注册（ADD 持 `GGML_EPILOGUE_GELU` hint + 后继 GELU_ERF + `ggml_can_fuse` 单消费者校验），fused kernel（`unary.cu` `add_gelu_erf_bias_kernel`）复用 `op_gelu_erf` 并镜像 k_bin_bcast 的 1-D bias 索引——位级恒等；域检查不满足即回退 unfused。decoder FFN 12 site 补打 hint（[full_model.cpp](../src/full_model.cpp)），vulkan 顺带获得同批融合（A&S 同近似，数值中性）。strict 路径不受影响：strict 下 MUL_MAT+bias 先被 cublasLt 融合并跳过 ADD 节点。
- **验证（修复当日）**：patch-replay PASS；RAFO 位级 diff 在 large cuda f16（non-strict 融合路径）与 f32（strict 路径）双路径均 BIT-IDENTICAL（4 图基线对照，[outputs/probes/parity/](../outputs/probes/parity/)）；`--self-test` PASS；nsys 复查 48 site 中 38 个融合发射（时间覆盖 98%，10 个微小张量 site 被域检查保守回退，合计 0.06 ms/帧），融合收益实测 ≈2.91 ms/帧（直执行口径），与归因探针预测（2.3–3.0 ms）一致。
- **判定 run（20261024 22:20 启动，20261025 07:03 完成）**：`20261024-current-r2`（full 全 suite，repeats 15，与 r2 同判据）——启动前三个后端于 22:18–22:20 全部重建（含融合 CUDA 修改），launch 时 provenance 指纹与盘上 binary 逐一比对一致（cpu dcdee05a / cuda 37eed3ef / vulkan 4259e361），为单二进制干净 run。原 `20261024-current`（19:48 启动）因 21:52 的 CUDA 重建成为混合二进制 run，已终止并删除，不作为判据；为它设置的 watcher 链已退役（其后续阶段会在 r2 运行中重建后端，属污染风险）。
- **收口（20261025）**：acceptance 重生成——**G1 63/63、auto_precision 63/63、G2 39/42（3 fail = base/large cuda-f32 + small/vulkan-f16 1.0004，accepted=3、remaining=0），decision=accepted_with_performance_deviations**；该行 gate_pass（0.9599）后 `(relsgg-convnext-large, cuda-f16)` 已从 borderline 集合移除（[write_acceptance_status.py](../scripts/write_acceptance_status.py)，注释保留五 run 历史与修复兑现记录）。六 run 持久性标签 persistent → intermittent（[analyze_g2_residuals.py](../scripts/analyze_g2_residuals.py)，ε=+6.9%）。

## Vulkan 快模式差距根因（20261019，`--profile` 逐帧分解，50 帧中位数）

| 配置 | 帧延迟 | graphA | graphB | graph 外余量 |
|---|---:|---:|---:|---:|
| vulkan tiny | 26.71 | 12.02 | 7.99 | **6.68** |
| cuda tiny | 25.81 | 13.24 | 8.46 | **4.13** |
| vulkan small | 34.40 | 19.11 | 8.48 | **7.09** |
| cuda small | 34.66 | 21.74 | 8.60 | **4.34** |

结论：vulkan 图内计算比 cuda 还快（融合有效），全部快模式差距是一个 **+2.6–2.7 ms/帧的宿主侧固定项**（graph 外），tiny/small 等量。两段 graph 逐帧节点数恒定（365+531）→ 头号机制：每帧 `ggml_new_graph_custom` + `build_forward_expand` + `sched_reset/alloc_graph`（[full_model.cpp `outputs()/execute()`](../src/full_model.cpp)）对 vulkan 意味着约 900 节点逐帧命令缓冲重录；CUDA 走 graph replay 缓存了同等工作。修复设计：形状与输出集合不变时跨帧缓存 `GraphRun`（图 + sched 分配），与 `no-cache-positions` 消融机制同族；落地后需过 G1 parity + 全量 acceptance run。

**量化边界（20261019 补测，约束修复预期）**：profile 显示 rebuild（build+expand+alloc+readback）仅 ≈1.4 ms/帧（graphA ≈0.67 + graphB ≈0.74）→ 请求框数 3–16+ 变化，单槽签名缓存（同一图 5 次 repeat 命中 4 次，≈80%）最多收回 ≈1.1 ms。**勘误：此前报的“graph 外余量 vulkan 6.68/7.09 vs cuda 4.13/4.34”不可信**——main.cpp 计时环内每次 repeat 都重新 `read_image_input(path)`（磁盘解码+resize+coverage），且 warmup 帧也产出 profile 行，跨流按 2 图/帧配对被 warmup 污染。下一步探针改为在 `infer()` 内部单行输出分段（prep/graphA/graphB/post，`options.profile` 门控），backend 间同口径对比后再定修复面；已排除：`ggml_vk_command_pool_cleanup`（`cleanup_frequency=10`，`resetCommandPool` 为 µs 级）。

**`.raim` 预解码对照（20261019，50 帧中位数）**：vulkan tiny 25.56→22.12、cuda tiny 23.95→20.77（图像 IO ≈3.2–3.4 ms，backend 无关）；vulkan small 32.95→29.95、cuda small 31.57→30.28。delta 在 .raim 下 tiny +1.34、small -0.33（反常待复测）。

**`infer()` 内部分段探针（20261019 落地，`--profile` 门控，逐帧单行：`frameA prep_ms/graphA_ms`、`frameB infer_ms/graphBplus_post_ms`）**：prep≈0 排除 CPU 准备。两个时间窗口重采后判定：**backend 间 delta 不稳定**——窗口 A（干净，.raim）tiny +1.34/small -0.33，窗口 B（全体膨胀 25–40%、SM 1807）tiny -0.52/small +1.56，符号翻转、幅度 ±1.5–2 ms 与门禁余量同量级。结论：**不存在可定位的固定 vulkan 差距**；gated fail（tiny 1.05–1.10、small 1.02–1.05）落在窗口摆动带内，属测量临界而非可修复缺口。可复现分量：prep≈0、rebuild≈1.4 ms、图像 IO≈3 ms（均 backend 无关）。处置归用户决策：接受窗口摆动带内的 vulkan 临界项，或增加采样次数（如 repeats=15）后再判。

### 更正：20261016 “测量污染 / 干净口径 39/42” 结论撤回

20261016 run 期间确有外部 GPU 进程（full-primary.log 542 条 waiting），但其“7 项假 fail”判定依据是 run 外独立复测。20261017 在无外部计算进程条件下重现了同一组慢值（tiny/vulkan 45.4 → 45.9 ms），而独立复测始终为快值——即“独立复测 = 干净”这一前提不成立，复测协议不能替代 run 内判定。cuda 三项“噪声带”判定同样被三 run 全部 >1 的符号一致性推翻。

## Vulkan 双模态 slow mode（root cause 未定）

现象：ConvNeXt（及 20261016 的 vitb16）vulkan 三种 dtype 同步慢约 1.8×，单图 iteration 稳定（非间歇抢占）；binary sha、device memory、stderr 与快模式一致。20261017 run 内：tiny 45.9 / small 61 / base 73.5 ms；快模式：tiny 25.4–26.7 / small 33 / base 37–40.6 ms（20261018 run 内 25.4 / 33.0 / 37.9，与独立运行 26.7 / 34.0 / 40.6 一致）。出现率：GELU 融合 patch 后 2/5 个正式 run 出现（1016、1017；1018、1019-repeats15、1024-current 三个后续窗口均未再现），融合前 0/1；独立运行约 50 次 0 次。五项仪器中位数常驻待命，再现即捕获。

**窗口签名定位（20261025，mtime 时间线 × journal 对齐）**：慢模式是**限时状态**而非 run 恒定态——1016 慢窗口 ≈18:33–19:07（vitb16 vulkan 1.52、tiny vulkan 1.79–1.84，**仅约 35 分钟**）；1017 慢窗口从 [21:18, 21:44] 间进入、[22:31, 次日 00:26] 间自发退出（≤2.5 小时）。窗口内 CUDA/CPU 全部正常、退出后 large vulkan 0.846–0.860 比常态更好；同窗口内门禁比值不同（vitb16 1.5× vs tiny 1.8×；**20261025 更正：相对自身快模式的减速因子一致 1.71–1.94×，见下文判别实验**）。已排除：boot/重启边界（5 run 同 boot）、驱动与内核包变更（dpkg/apt 10-14~19 为空）、二进制差异（1017 vs 1018 build 指纹 `e2ffdd3b` 完全一致、源码 mtime 窗口为空）、stderr/stdout/初始化差异（逐字节一致）、两个切换窗口内的任何内核/电源/锁屏/显示异常（journal 仅常态 Xorg 轮询）。**头号假设（H8）**：第三方 Vulkan 应用（同期 ACloudViewer 的 onnxruntime GPU 测试活跃）退出后留下**驱动级 VK 残留态**，使后续逐 op 提交的 vulkan 进程变慢、CUDA graph-replay 路径免疫，数十分钟~数小时后自发恢复；并发期样本天然被 wait/discard 机制剔除，故保留的慢 attempt 均显示"无外部进程"——与全部观测自洽。判定实验（20261025 首轮）：① vkcube（VulkanSDK 版）因 GLIBC_2.34 不可用，未执行；② **glxgears 退出后残留窗口首测**——此前 gfx_contention 只测渲染进行中（证伪），本次测退出后：基线 23.56/23.56，退出后 post0=24.47、**post60=32.59（+38%）**、post180=23.14（恢复）——方向与残留态假设一致但单点采样、未达 1.8× 签名，需重复实验（多基线 + post 序列加密采样）后再判。

**重复实验 v1 判读（20261025 晚，3 轮 × post0/30/60/120/300 + cuda 同时刻对照）**：**无一轮满足残留态判据**（事先定死：post ≥ +8% 且 cuda 持平、≥2 轮复现）。Round 1 的 p60/p120 抬升（+9%/+16-20%）与 cuda 同步（pc +9.7%）；Round 2 基线整体抬高 ~20%（vk+cuda 同步）后回落，Round 3 p300 低于基线——post 波动完全由**系统 CPU 负载漂移**主导（实验期间外部 LTO/pybind-CUDA 构建启停，load 飙至 15.6）。**偶得方法性发现：load 15 环境下 vk/cuda 同步 +20%——CPU 负载漂移本身即是 gated fail（1.02–1.10）窗口摆动的充分解释源之一，反向支持白名单 borderline 集合的合理性**。结论：GL 上下文退出后残留分支**未获支持**。v2（load<2 门控）已撤销——共享机 load 长时间 15–48、从未满足门控，且下述 vk/cuda 配对比值本身抵消 CPU 负载漂移，门控不再必要（v1 按比值重读同为阴性：p60 vk/cuda 0.98）。

**判别实验 H9–H12（20261025 夜，tiny-f32，vk/cuda 配对比值为判据——抵消 load 漂移；脚本 `outputs/scratch/h9_*.sh`、`h10_video.sh`、`h12_powermizer.sh`）**：

| 假设 | 条件 | vulkan / cuda（ms） | 判读 |
|---|---|---|---|
| H9 外部 GPU 上下文忙碌 | 自建 cuda / vulkan 推理循环并发 | 34.5–37.1 / 35.5–39.6（双方 +70%） | **对称**，非 vk-only |
| H9 外部上下文空闲持有 | 同上进程 SIGSTOP（上下文+显存保持） | 20.9 / 20.4–21.1 | 无影响 |
| H9b 上下文 churn | `--self-test` 紧循环建/毁设备（ctest 形态） | 20.2–21.5 / 20.7–21.5；停止后 p0/p60 同 | 无影响、无残留 |
| H10 视频引擎 | NVENC/NVDEC 实时（会议形态）/ 双饱和 | 19.9–23.6 / 20.6–23.4；饱和 23.6–29.7 / 23.1–27.7 | 实时无影响；饱和**对称** |
| H10 远程桌面 | nxd.log 会话 × 慢窗口 | 1016/1017 慢窗口内无 NX 会话；10-24、10-25 有会话时快模式 | 无关联 |
| H11 显存回落系统内存 | 慢/快 run 推理期 `device_bytes` | tiny vk-f32 五 run 均 431 MB、base 均 924 MB | 放置一致，证伪（"265 vs 1 MB"为 load 期采样时刻差） |
| H12 PowerMizer 策略 | 切 Auto/Adaptive/MaxPerf（已恢复 Auto） | vk 恒 P0 SM 1931–1944 / mem 7500；cuda 恒 P2 / mem 7300；延迟不变 | 用户态策略无法诱发 |

**run 内括号证据（`timing_attempts.json` 0.4 s 采样 + results 时间戳）**：慢窗口内同一模型的执行序为 cuda×3 → vulkan×3 → python（torch-CUDA），全部在 ~2 分钟内——1017 tiny 21:43:30 cuda 24.84 正常 → 21:44:21 vulkan 45.87 慢 → 21:45:22 python 24.91 正常，small/base 连续重复同一形态，全部采样 foreign=[]。外部竞争须恰好只覆盖 3 个模型各自的 vulkan 段，排除；且慢态跨越 10–80 分钟纯 CPU 段与多个 CUDA/torch 进程而不重置——**状态在驱动/GPU 侧持久、仅作用于 Vulkan 客户端**。H8 前提"并发样本被 wait/discard 剔除"由此撤回。

**减速形态更正**：相对同二进制快模式（20261018）的减速因子为 **乘性 1.71–1.94×**——vitb16 1.87–1.89、tiny 1.71–1.79（1016 与 1017 两日几乎逐位一致）、small 1.82–1.87、base 1.92–1.94，dtype 无关，增量随模型线性增长（+20/+28/+35 ms）。此前"小塔受害更重（vitb16 1.5× vs tiny 1.8×）"是门禁比值（vitb16 快模式本就 0.79× python）误读，撤回。签名等价于 **GPU 侧有效吞吐约减半**（host 侧固定段稀释小模型因子）。

**预注册判别（再现即判，无需新实验）**：常驻仪器的 `sm_clock_mhz`/`pstate`/`graphics_pids` 对慢 attempt 只剩两种可区分形态——① SM ≈ 1000 MHz 或非 P0 → 驱动对 Vulkan 客户端的时钟/性能级态（H12 族，非用户策略可诱发）；② P0 满频 + 吞吐减半 → 驱动侧分时（查 `graphics_pids` 中 Type=G 客户端；compute-apps 守护对其不可见）。两者均为驱动内部态，修复面不在本仓库。**发布判定交互（已核）**：HEAD 版 `write_acceptance_status.py` 的 borderline 具名集合按 (model, config) 放行、**无幅度上限**——慢模式若落在 small vulkan×3 / tiny-q8_0 上（1.8×）会被静默 accepted；materiality 判据（20261025 已落地：逐图配对比值 bootstrap 下界 > 1.05 即 block）关闭了此缺口，慢模式行判 block → decision=blocked（1016/1017 回测已验证）。

已证伪假设（均为配对因果实验，GPU 空闲时段）：

| 假设 | 实验 | 结果 |
|---|---|---|
| 外部计算进程 | run 内 compute-apps 守护 | 无外部进程仍出现 |
| CPU 争用 | `stress -c 16/32` | Vulkan 与 CUDA 同为约 +20%，非 1.8× |
| 监控观察者效应 | 裸调用 vs `native_timing_run` 交替 A/B | 无差异 |
| 图形引擎负载 | 无垂直同步 glxgears 1600×1000 | Vulkan +19% / CUDA +25%，对称 |
| 时钟降频 | `nvidia-smi -lms 100` | 快模式 P0、SM≈1940 MHz；慢模式未能在探针下复现 |
| run 注入环境变量 | run 内自带复刻（OPENBLAS_NUM_THREADS=1 / OMP_NUM_THREADS=4） | 快（1.05×） |
| host 线程饥饿 | `taskset -c 0-3`（±run-env） | 快（1.08–1.09×）；Vulkan 逐 op 提交虽重，饥饿不足以降速 1.8× |
| torch 上下文相邻 | python bracket（完整 torch-CUDA 生命周期）→ vulkan ×3 周期 | 快（1.07–1.08×） |

单模型迷你复现阶梯（20261018，8/8 全快）排除全部单变量触发；慢模式只在完整 run 的长会话中出现——指向多模型多上下文创建/销毁的驱动累积状态或时间窗因素，单模型复现不可能，定位只能依赖 run 内仪器。

仪器补齐：`benchmark_full_graph.py` 监控现记录 `sm_clock_mhz`、`mem_clock_mhz`、`gpu_temp_c`、`pstate`、`graphics_pids`、`host_mem_available_kb`、`load1`，attempt 记录各推理段中位数——20261018 捕获 run 直接区分时钟/P-state、host 压力与驱动侧调度。该问题是用户可见的真实风险，不按测量噪声处理。

## 复测协议

`build-{cuda,vulkan}/relateanything-ggml --model <gguf> --input <run>/artifacts/full_graph/image_requests --backend <b> --warmup 2 --repeats 5 --max-boxes 60`，50 图逐条 latency 取 median。独立复测仅用于因果实验（配对、交替），不覆盖 run 内判定。持久性表：`python3 scripts/analyze_g2_residuals.py --candidate <run> --baseline outputs/runs/20261014-pure-cpp-release --history <中间 runs>`。

**产物裁剪（20261020 第一轮：runs 252G→5.3G；20261025 第二轮：全量 run 收尾后 62G→4.4G）**：两轮均按判据保留式执行——删除逐图原始输出（`.rafo`/`.npz`/逐图 predictions json）与 stages/dense/traces/interfaces/gallery 等 suite 原始数据（`publish_reports` 为 copy 语义，`reports/` 有完整副本；HF 发布包输入源 = `reports/`）、旧 run 的 `reports/` 视图副本（current 已切走）、残壳 run 与 probes 探针产物。各 run 保留 run.json、logs、summary.json、timing_attempts.json（20261024 为 77 个 = 7 模型 × 11 配置，已验证）、process_memory、inputs/image_requests（同输入复测能力完整）、acceptance/审计 JSON；G2 判定、持久性表与 HF 发布链不受影响。

## ConvNeXt Vulkan 性能备注

此前定性“F32 GEMM 后端天花板（1.10–1.14×）”在快模式下被修正：融合后 base/large vulkan 稳定在 **0.82–0.95×**（1016 与 1018 两个快模式 run 一致），small 0.96–1.05× 临界，tiny 1.05–1.14× 仍持续略慢——收益随模型变小而衰减，与逐 op 提交开销相对占比上升一致（未验证）。落地项：token-major CWHN 深度卷积零拷贝、depthwise bias 融合进卷积（`ggml_conv_2d_dw_direct_bias`）、pwconv1 bias-add 挂 GELU epilogue hint（CUDA 走 cublasLt `GELU_BIAS` epilogue，Vulkan 走 add 写出融合 `add_gelu.comp`）与新增 subgroup-per-row 归一化 shader（`norm_row.comp`）。这些均需维护 GGML patch（[0001-relateanything-ggml.patch](../third_party/patches/0001-relateanything-ggml.patch)，含 patch-replay 审计），历史记录中“暂缓 shader 级方案”的权衡已被本次收益逆转。

## 覆盖与非目标

完整发布运行覆盖 7 模型 × 3 后端 × 3 存储、50 图、stage/dynamic、dense sampler、量化 oracle、可选接口、真实 C API 和公开标注子集。R/mR/F1/wR、候选集合、logits 误差、P50/P95、吞吐和后端内存均保留在报告中。

不承诺 F16/Q8 与原始 F32 逐位一致；公共子集回归不等于完整 leaderboard；缺失标注不会由推理输出替代。
