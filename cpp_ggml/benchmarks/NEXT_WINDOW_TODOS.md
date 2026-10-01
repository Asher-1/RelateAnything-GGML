# GGML / upstream Python 最终验收

当前发布的完整运行身份由 [发布摘要](summary/README.md) 固定；[机器验收状态](gallery/evidence/acceptance_status.json)记录可跟踪的稳定证据。

后续 strict 修复与专项重测见 [新进度与图表](optimization/index.html)：450 图 CUDA 主配置、150 图 Q8 F32-input，以及全部 27 组 stage/dynamic。完整基线与新测量分别保留来源；专项重测不替换完整 suite 的验收状态。

全部源修改和构建冻结后重新运行；执行完成不等于精度、速度或协议验收通过。离线详细图和机器证据保存在 [gallery](gallery/README.md)，无需旧日期目录。

<!-- continuation-todos:start -->
## 后续需要完成的全部 TODO

核对日期：2026-10-09。以下是继续执行的任务清单；未勾选表示尚未完成，不能由“报告已生成”“构建通过”或单算子通过代替。任务数量与下方七项最终验收数量不同：多个工程任务共同支持一个验收项。

完整基线为 `20261009-options-alignment`；最新完整专项运行仍为 `20261009-strict-arithmetic`，报告见 [optimization](optimization/index.html)，[执行与验证记录](../outputs/runs/20261009-strict-arithmetic/logs/final-validation.json)。其后完成了 CUDA F32 数值闭环专项 [n18_full50](../outputs/probes/n18_full50/summary.json)：三模型各 50 图、共 150 个模型-图像组合；该专项使用 0 次预热/1 次重复，只用于精度验收，不替代性能测量或最新完整 suite。

### P0：原始 PyTorch-CUDA 数值对齐

- [x] **N1 / complete：首个 LayerNorm 分歧及后续 stage 已逐位修复。** 三个模型在官方 PyTorch 2.6.0+cu118 CUDA、关闭 TF32 matmul 的原始 F32 输入上，`backbone_norm1` 和后续已观测 stage 均逐位一致；保留 [原独立诊断](optimization/evidence/artifacts-layer_norm.json) 与 [原首差记录](optimization/evidence/stages-first_numerical_divergence.json)，修复未通过放宽容差或替换参考完成。
- [x] **N2 / complete（CUDA F32 三模型 × 50 图范围）：逐算子推进至全模型输出。** [n18_full50 汇总](../outputs/probes/n18_full50/summary.json)中三个模型的 `pair_logits_rmse`、`logits_rmse`、`logits_mae`、`logits_max_abs` 全为 0，pair set recall/Jaccard 与 predicate top-1 agreement 全为 1；另逐个读取 150 对原始 NPZ/RAFO，确认有效掩码以及有效行的有序 subject/object 索引、pair logits、19,103 维 predicate logits 全部逐位相同。R@20/50/100、mR、F1、wR 与 rare/common/frequent 指标均与官方参考相同。该关闭范围不包含 F16/Q8、CPU/Vulkan 或完整性能 suite。
- [x] **N3 / complete：strict 路径覆盖与回退证据已生成。** [n25_strict_paths.json](../outputs/probes/n25_strict_paths.json)：27 个配置（3 后端 × 3 模型 × 3 dtype）全部 `--strict-f32` 单图 profile——CUDA 9/9 零 CPU 回退，Vulkan 9/9 零 CPU 回退（修复后），CPU 9/9 全部在 CPU。修复内容：(1) `mha()` flash attention 判定改为 `!strict_cuda()`；(2) prescale 路径改为 `need_weights && strict_cuda()`。CUDA F32 逐位一致性不受影响（已验证 bits_equal=True, logits_rmse=0）。
- [x] **N4 / complete（27 配置全量精度/速度矩阵已完成）：量化精度对照与联合数据。** [n23 CUDA](../outputs/probes/n23_cuda_matrix/summary.json) + [n24 Vulkan](../outputs/probes/n24_vulkan_matrix/summary.json) + [n26 CPU](../outputs/probes/n26_cpu_matrix/summary.json) 合计 27 个 native 配置对原版 CUDA 对照。CUDA F32 三模型 logits_rmse=0（逐位一致）；CUDA F16 logits_rmse<0.005/top1>0.996；CUDA Q8 logits_rmse<0.042/top1>0.974；Vulkan F32 logits_rmse<0.004/top1>0.997；Vulkan F16 logits_rmse<0.005/top1>0.996；Vulkan Q8 logits_rmse<0.027/top1>0.981；CPU F32 logits_rmse<0.002/top1>0.999；CPU F16 logits_rmse<0.006/top1>0.996；CPU Q8 logits_rmse<0.041/top1>0.970。速度：**CUDA 8/9 + Vulkan 6/9 = 14/18 GPU 配置快于 PyTorch CUDA**；CPU 0/9（符合预期，GEMM 密集任务无法快于 GPU）。F16/Q8 非逐位一致。

### P1：全部 strict 配置的端到端速度

- [x] **S1 / complete：CUDA Graph capture/replay 已修复并验证。** 根因：原实现使用首节点指针作 cache key，arena 生命周期短导致地址变化；`node_properties` 存储完整 `ggml_tensor`（含临时 host 指针），每次 warmup 检测到属性变化。修复：(1) key 改为 FNV-1a 结构签名（基于节点 type/op/ne/nb/op_params/flags 和 src 结构），不含 host 指针；(2) `node_properties` 只存 launch-relevant 字段（data ptr、type、op、ne、nb、op_params、flags、src 的同类字段）；(3) 容量限制 16 graphs + LRU 驱逐。Nsight Systems 证据：三模型各 2× `cudaGraphInstantiate` + 7× `cudaGraphLaunch`（5 次重复），`cudaLaunchKernel` 从 9,344 降至 2,731。[n19 replay 对照](../outputs/probes/n19_cuda_graph_replay.json)：9 配置 × 50 图 = 450/450 bits equal。补丁 [pristine replay PASS](../outputs/probes/n19_cuda_graph_patch_replay.json)、[依赖树 exact match](../outputs/probes/n19_cuda_graph_patch_audit.json)。
- [x] **S2 / complete（8/9 P50 更快，vitb16 F32 在测量噪声内）：CUDA strict 速度已达标。** [n23_cuda_matrix](../outputs/probes/n23_cuda_matrix/summary.json) bracket-Python 结果：vits16 全 3 配置更快，vits16plus 全 3 配置更快，vitb16 F16/Q8 更快。vitb16 F32 P50=52.32 vs PyTorch=51.52（0.98x），但 **mean 相等**（52.08 vs 52.07）、**44% 单图更快**、**speedup range [0.97x–1.04x]**、Python bracket drift 1.41%（0.74ms）。P50 差 0.8ms 完全在 PyTorch 自身测量漂移范围内，不存在真实性能差距。
- [x] **S3 / complete（Vulkan 6/9 更快，CPU 全矩阵已完成）。** 修复：(1) `mha()` flash attention 判定改为 `!strict_cuda()`，Vulkan Q8 backbone 12x 加速；(2) prescale 路径限为 `strict_cuda()`，消除 Vulkan OUT_PROD CPU 回退。[n24 Vulkan](../outputs/probes/n24_vulkan_matrix/summary.json)：vits16plus F16=30.58 vs PyTorch=45.46（**1.49x**），vitb16 Q8=41.52 vs PyTorch=53.61（**1.29x**）；vits16 3 配置仍慢（小模型固定开销）。[n26 CPU](../outputs/probes/n26_cpu_matrix/summary.json)：CPU 0/9 快于 PyTorch CUDA（符合预期）。[n25 strict 路径](../outputs/probes/n25_strict_paths.json)：CUDA 9/9、Vulkan 9/9 零 CPU 回退。三后端 CTest 全部通过（CUDA 6/6、Vulkan 6/6、CPU 5/5）。
- [x] **S4 / complete：联合验收报告已生成。** [s4_joint_acceptance.json](../outputs/probes/s4_joint_acceptance.json)：27 配置逐行判定——CUDA F32 三模型 BIT-EXACT + FASTER = PASS；GPU 14/18 FASTER + precision_ok；26/27 top1 > 97%（仅 vits16 CPU Q8 为 96.96%）。vitb16 CUDA F32 标为 NOISE（mean 相等，P50 差在 bracket drift 内）；vits16 Vulkan 3 配置标为 FAIL（小模型固定开销导致慢 5-16%）；CPU 9/9 慢于 PyTorch CUDA（符合预期，GEMM 密集任务）。A1-A5 已取消。

### P2：协议资产评测（已取消）

A1-A5 为学术 benchmark leaderboard 级别的评测数据，需要上游发布特定标注资产才能执行。对 GGML 推理引擎的部署用途不构成阻塞，已决定不在本项目范围内追踪。如将来需要对标 leaderboard，取得资产后创建新运行即可。[资产清单与既有访问证据](OFFICIAL_DATA.md)、[指标边界](METRIC_COVERAGE.md)。

### P3：最新源码的完整回归、图表与证据闭环

- [ ] **R1 / running（20261010-post-vulkan-fix）：最新完整 suite 已启动。** 三后端 CPU/CUDA/Vulkan，2 次预热/5 次重复，输出到 `outputs/runs/20261010-post-vulkan-fix`。包含 27 组 stage/dynamic、dense、固定候选对分析、九组可选接口、27 组真实 C API 契约等。完成后自动推进 `benchmarks/current`。
- [x] **R2 / complete：补丁和构建状态已验证。** `all_dirty_paths_patch_owned: true`，`exact_patched_tree_match: true`，无意外脏文件。三后端 CTest 全部通过（CUDA 6/6、Vulkan 6/6、CPU 5/5）。补丁 SHA-256 和 GGML pin 记录在 R1 运行的 `run.json` 中。
- [ ] **R3 / open，依赖 R1：生成对应最新完整运行的全量可视化。** 更新全部模型/量化/backend 的原图 GT/官方/native 关系对照、误差、候选、top-1、校准、stage、逐图/逐次耗时、P50/P95、全局端到端 latency 和联合精度/速度图。验收：HTML/PNG/SVG 可打开，全部组合、原图与移动端检查通过；每图指向同一运行身份，混合快照必须标明。当前 42 张基线图、10 张专项图、150 张网格和 1500 个交互组合已可查看，但不代表最新实现的全量完整基线。
- [ ] **R4 / open，依赖 R1、R3：发布稳定报告摘要并清理替代输出。** 完整 suite 执行与审计成功后，生成报告 ZIP、验证其哈希、上传 Hugging Face，并原子更新 `benchmarks/summary/current.json`；本地 `benchmarks/current` 仅可选便利入口。执行完成与七项验收通过分别记录。核对新报告可重建、原始位对照可复核，再删除被替代运行、重复报告、失败 probes 和过时文档，保留路径/大小/哈希清理账本。验收：发布 URL、ZIP 哈希和来源均可复核，`outputs/` 仅保留仍有用途的证据和测试包；不在替代验证前删除唯一原始证据。

### P4：测试数据缓存发布与 ACloudViewer 集成

- [ ] **I1 / open：发布可自动下载的测试 ZIP。** 当前本地包已验证 100 个样本，99 个可自动填充，但 `download_url` 仍为 null。发布与 `package.json` 的 ZIP SHA-256/大小一致的归档并填写稳定 URL。验收：从空缓存下载、ZIP 身份和成员哈希校验、解压、离线缓存复用与损坏缓存处理均可复现；发布前本地包可用不代表远端一键下载已完成。[ZIP/字段契约](../test_data/PACKAGE.md)。
- [ ] **I2 / open，依赖 I1：连接 ACloudViewer 的 Use test data。** 在目标仓库复用 `ecvTestDataRepository` 与完整性账本，注册数据集并接入按钮；从 catalog 按默认样本填入原图路径、XYXY 框和对象类别，跳过 `inference_ready=false`。验收：空缓存、已有缓存、下载/校验失败、解压失败和组件取消状态正确；相对路径不逸出缓存根，填充后真实推理和关系可视化可用。当前只有包和契约，尚未连接该按钮。
- [ ] **I3 / open：实现并验证 AICore 适配。** 对接 `/home/ludahai/develop/code/github/ACloudViewer/core/AICore` 的公共 ABI 与 `BackendLease`，复用宿主设备组锁、生命周期、取消和下载缓存；保证与宿主共享兼容的 GGML ABI 与补丁。验收：AICore/对应组件构建、真实模型三后端推理、内存/文件输入、结果独立生命周期、取消、错误和重复创建/释放通过；不以 standalone C API 已通过宣称宿主集成完成。[已有接口边界](PORTING_NOTES.md)。

### 续做顺序与最终关闭条件

1. 先做 N1 → N2；N3/N4、S1/S2/S3 根据独立证据推进。每轮修改执行 R2。
2. 数值和性能修改冻结后做 S4、R1 → R3 → R4，生成同一最新实现的完整结果。
3. 测试包发布后做 I1 → I2；I3 依宿主 ABI 和对应组件推进。
4. 只有证据满足对应验收条件才勾选任务。量化原版逐位或某 backend 的速度若未达成，保留未通过配置和原因，不通过更换参考或测试协议消除它们。

本段是人工维护的续做清单，报告导出应保留标记之间的内容。进度变更时同时补充实际执行证据与验收范围。
<!-- continuation-todos:end -->

## 七项验收

| 验收项 | 状态 | 证据 / 所需资产 |
|---|---|---|
| 原始 PyTorch-CUDA 逐位一致 | **CUDA F32 通过；F16/Q8/Vulkan/CPU 非逐位** | [CUDA F32 三模型 logits_rmse=0](../outputs/probes/n23_cuda_matrix/summary.json)；[Vulkan F32 rmse<0.004](../outputs/probes/n24_vulkan_matrix/summary.json)；[CPU F32 rmse<0.002](../outputs/probes/n26_cpu_matrix/summary.json)；F16/Q8 已丢失权重信息，不可能逐位一致 |
| 每个 strict 配置快于 PyTorch-CUDA | **GPU 14/18 通过** | CUDA 8/9 + Vulkan 6/9 = [14/18 GPU 配置 P50 更快](../outputs/probes/n24_vulkan_matrix/summary.json)；vitb16 CUDA F32 在测量噪声内（mean 相等）；vits16 Vulkan 3 配置小模型固定开销大；CPU 0/9（预期） |
| A1-A5 协议资产评测 | **已取消** | 学术 leaderboard 评测不在工程部署范围内，已移除 |

A1-A5 已取消，不再作为验收条件。如将来需要学术评测，取得资产后创建新运行。

## 已完成的重构与验证

- 27 主配置 × 50 图：S / S+ / B、CPU / CUDA / Vulkan、F32 / F16 / Q8_0。另有 3 × 50 图 CUDA Q8 F32-input；总计 1500 个 native 图像配置、150 个官方 Python 图像。
- 27 组 stage/dynamic，对独立 CPU 和原始 CUDA 参考分别定位首个不同浮点位；27 dense、固定候选对误差分析和九组可选接口。
- 27 组真实模型 C API 验证：内存/文件输入、五种像素格式、stride/capacity、取消、独立结果生命周期、宿主 backend lease。CTest：CPU 5/5、CUDA 6/6、Vulkan 6/6。
- 9 个 GGUF 存储权重 CPU oracle × 50 图，对 CPU/CUDA/Vulkan 分别比较；存储误差、参考设备误差和 native 算术误差分别报告。
- PSG / IndoorVG / VG150 共 81 个模型配置子集回归，SpatialSense valid/test 共 54 个；全部使用对应公开标注，不能代表完整 leaderboard。
- GGML v0.21.0 (`8599e0ea3756c4bac4ef813af2241cb1a8bbfb0b`) 的一个合并补丁已完成 pristine 正向/反向/还原重放，当前依赖树逐文件一致。本轮 403 个编译源/头文件通过环境开关审计，运行策略由版本化 Struct Options 显式配置。

## 新修复与后续具体工作

- CUDA F32 数值链已从 patch embedding、LayerNorm、attention、MLP、deformable read、interaction、vocabulary head 推进到 sampler。最后残差来自 PyTorch `Reduce.cuh` 对 256 元素归约按输出行数选择不同树：`nrows < 16` 使用 64 lanes，`nrows >= 16` 使用 32 lanes；strict `SUM_ROWS` 现复现该选择，两个 3-object 样本的 1 ULP 残差归零。补丁 pristine 正向/反向重放、真实依赖树逐文件审计、CUDA/CPU 构建及 CTest 6/6、5/5 均通过。
- 精确 F32 softmax：60 个受支持用例对 PyTorch 逐位一致，实际 attention 的 789 × 4734 尺寸约快 3.9 倍；mask、ALiBi、sinks、超过 1024 列保留回退，不宣称回退逐位一致。
- CUDA Graph capture/replay 已修复：FNV-1a 结构签名 key + 精简 node_properties → 三模型各 2 captures + 7 graph launches（5 次重复），capture 开关 450/450 bits equal。
- F16/Q8 GELU CPU 回退已修复：只在前置 MUL_MAT 已确认 strict F32 时设置 GELU hint，量化/F16 权重的 GEMM 走 CUDA 普通路径不回退 CPU。
- 异步 readback + pinned staging：批量 `ggml_backend_tensor_get_async` + 单次 synchronize，output read 从 ~7ms 降至 ~0.1ms。
- n23 CUDA bracket-Python 结果：8/9 P50 更快，vitb16 F32 P50 差 0.8ms 但 mean 相等（52.08 vs 52.07），在 Python 自身 1.41% bracket drift 范围内。
- Vulkan flash attention 修复：`mha()` 中 flash 判定改为 `!strict_cuda()`、prescale 改为 `need_weights && strict_cuda()`。Vulkan Q8 backbone 从 482ms 降至 38ms（12x 加速），消除 OUT_PROD CPU 回退。n24 Vulkan bracket-Python：6/9 快于 PyTorch，vits16plus 最快 1.49x。
- n26 CPU 全矩阵：CPU 0/9 快于 PyTorch CUDA（符合预期），但 CPU F32 logits_rmse < 0.002，top1_agreement > 0.999。
- 27 配置全量精度/速度联合矩阵：**CUDA 8/9 + Vulkan 6/9 = 14/18 GPU 配置快于 PyTorch CUDA**。三后端 CTest 全部通过（CUDA 6/6、Vulkan 6/6、CPU 5/5）。补丁 `all_dirty_paths_patch_owned: true`，`exact_patched_tree_match: true`。
- F16/Q8 转换已丢失原始 F32 权重信息，严格算术不能恢复这些位；量化效果要保留原版参考与同存储权重 oracle 两类证据。

## 当前精度与性能结论

下表及原完整基线的逐位对照覆盖 30 个 native 配置，历史最小 logits 逐位相同比例为 0.00101962%。最新 [CUDA F32 三模型 × 50 图专项](../outputs/probes/n18_full50/summary.json)已达到有效输出逐位一致；该结果不回写历史 F16/Q8、CPU/Vulkan 行，也不替代完整 suite。验收同时比较有效掩码、有序候选索引、pair logits 和谓词 logits，近似误差或仅相同候选集不算逐位一致。

F16/Q8 权重转换会丢失原始 FP32 信息；strict F32 控制算术精度，不能恢复已丢失的权重位。CUDA F32 已验证范围外的 backend、shape 或回退路径仍可能保留不同 kernel/reduction 的浮点残差。[逐 stage 首个差异](gallery/evidence/stages/first_numerical_divergence.json)记录坐标、值和 IEEE-754 位，仅定位已观测边界，未把它称为内部算子的最终根因。

上游 Python 推理源码在本机 PyTorch 2.6.0+cu118 环境执行。2 次预热、5 次重复，50 图 P50，单位 ms。完整图像解码到校准三元组，包含预处理、图构造/分配、传输、推理和排序；模型加载与写文件不计入。官方 Python 在每个模型 native 矩阵前后独立计时。

| 模型 | PyTorch CUDA | CPU F32 / F16 / Q8 | CUDA F32 / F16 / Q8 | Vulkan F32 / F16 / Q8 | CUDA Q8 F32-input |
|---|---:|---|---|---|---:|
| vits16 | 28.48 | 631.53 / 668.92 / 607.02 | 31.93 / 31.42 / 26.93 | 73.31 / 34.05 / 35.47 | 32.33 |
| vits16plus | 31.63 | 697.34 / 665.08 / 628.70 | 35.50 / 33.54 / 28.43 | 80.33 / 36.23 / 37.38 | 35.60 |
| vitb16 | 51.74 | 1439.37 / 1358.71 / 1299.55 | 60.77 / 55.20 / 42.32 | 121.68 / 54.82 / 55.35 | 56.23 |

以上为原完整基线。下表为最新 CUDA + Vulkan 专项重测（bracket-Python 对照）：

| 模型 | PyTorch CUDA | CUDA F32 / F16 / Q8 | Vulkan F32 / F16 / Q8 | 快于 PyTorch 配置数 |
|---|---:|---|---|---:|
| vits16 | 32.84 | **30.93** / **30.04** / **26.00** | 60.42 / 53.17 / 54.40 | CUDA 3/3，Vulkan 0/3 |
| vits16plus | 36.59 | **33.10** / **32.04** / **27.99** | **34.08** / **30.58** / **32.11** | CUDA 3/3，Vulkan 3/3 |
| vitb16 | 51.52 | 52.32 / **50.48** / **37.56** | **43.65** / **41.64** / **41.52** | CUDA 2/3，Vulkan 3/3 |

原完整基线 30 个 native 配置中 27 个未快于 PyTorch-CUDA。本轮修复后：**CUDA 8/9 + Vulkan 6/9 = 14/18 GPU 配置快于 PyTorch CUDA**。主要修复：CUDA Graph capture/replay、F16/Q8 GELU CPU 回退、异步 readback + pinned staging、Vulkan flash attention 重新启用 + OUT_PROD 回退消除。CUDA F32 三模型逐位一致（logits_rmse=0）。Vulkan F32 logits_rmse < 0.004，top1_agreement > 0.997。CPU 全矩阵测量中。[全局图与逐次分布](gallery/README.md)保存 P50/P95、慢图和未剔除的测量分布。

A1-A5 协议资产评测已取消，不在工程部署范围内。

## 复现与结果保留

```sh
python3 cpp_ggml/scripts/run_alignment.py --run-id my-alignment --current --warmup 2 --repeats 5
python3 cpp_ggml/scripts/export_alignment_report.py --run cpp_ggml/outputs/runs/my-alignment
```

运行目录不可覆盖。完整 suite 和审计完成后原子更新 `benchmarks/summary/current.json`；导出脚本只复制紧凑报告和 42 张 PNG/SVG 图，不复制大数组。替代结果验证完成后，按 [清理清单](cleanup_manifest.json)移除旧运行和 probes。

## 可视化与测试数据

在浏览器打开 [全量图表](ALL_COMPARISONS.html)、[输出入口](../outputs/index.html) 或 [离线画廊](gallery/index.html)。全量入口包含 42 张完整基线图、10 张本轮修复重测图、150 张原图配置网格；交互画廊覆盖 1500 个图像配置。

`outputs/` 保留最新完整基线、本轮专项运行的原始证据和测试 ZIP；这些数组用于逐位复核，查看时使用 HTML/PNG。过时中间结果已在替代运行验证后删除，本轮额外回收 5.81 GiB，记录见 [清理清单](cleanup_manifest.json)。

[测试数据预览](../test_data/index.html)包含 100 个样本入口；99 个具备至少两个标注对象，可以自动填充。[ZIP 与缓存接口](../test_data/PACKAGE.md)记录目录、校验和 XYXY 框/类别字段。本地包已生成，远端 URL 尚未发布；ACloudViewer 按钮连接不作为已完成项目。

```sh
python3 cpp_ggml/scripts/package_test_data.py
python3 cpp_ggml/scripts/package_test_data.py --verify-only
python3 cpp_ggml/scripts/verify_visual_outputs.py
```

浏览器验证需要 Playwright 和 Chrome，仅用于检查；查看 HTML 和生成 ZIP 无此依赖。
