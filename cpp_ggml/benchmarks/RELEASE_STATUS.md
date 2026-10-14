# GGML 发布验收状态

## 当前结论

- 21 个 GGUF 已发布（7 个 checkpoint × F32/F16/Q8_0：三个 DINOv3 ViT 塔 + 四个 DINOv3 ConvNeXt 塔）并经远端回下载 SHA-256 12/12 校验（ConvNeXt 新增部分）。
- ConvNeXt 全链路适配完成：导出 metadata、C++ token-major backbone（cwhn dwconv）、Stage 融合，下游 head 与 ViT 共用。
- 图层优化已合入（纯 C++ 层，无 GGML 源码新增改动）：token-major（CWHN）深度卷积零拷贝、LayerNorm affine 与 LayerScale 的加载期折叠——图节点数 497 → 约 400。
- 带图层优化的完整发布运行将重新启动（先前的 `20261014-optimized-release` 因源码回退已终止）；在发布前，[summary/current.json](summary/current.json) 仍指向上一份稳定报告。

## 默认精度规则

默认请求为 `auto`：仅 CUDA 加载全 F32 矩阵时生效 `strict_f32=1`；其余 CPU、Vulkan、F16、Q8_0 配置生效 `strict_f32=0`。显式 `--strict-f32` / `--no-strict-f32` 仅作诊断或覆盖，不代表默认发布路径。

strict 保留 F32 GEMM/softmax/norm 与显式 F32 attention，避免 TF32/F16 中间算术触发 top-k 边界换对；non-strict 使用更快的 TF32、flash F16 K/V 或量化激活路径。F16/Q8 已丢失的存储精度无法由 strict 恢复。

## 发布门槛

| 门槛 | 上一稳定 run | 本次发布 run | 说明 |
|---|---:|---|---|
| G1 数值 | 27/27 | 63 项待 20261014 run | 判定面扩至 7 模型 × 9 配置 |
| G2 GPU 性能 | 17/18 | 42 项待 20261014 run | 判定面扩至 7 模型 × 6 GPU 配置 |
| 工件身份 | 9/9 | 21/21 | [full_gguf_manifest.json](full_gguf_manifest.json) 与 Hugging Face 一致 |

已知偏差与处理：`relsgg-vitb16/cuda-f32` 在上一 run 为 0.96×（唯一 G2 未过项，优化后重测）；ConvNeXt 的 Vulkan 配置预期落后 PyTorch 10–20%（F32 GEMM 效率为后端天花板，详见下节），将逐项以 `accepted_deviation` 记录而不是放宽门槛。

## ConvNeXt Vulkan 性能备注

ConvNeXt 的深度卷积 + 逐位置 LayerNorm 结构在 Vulkan 通用 shader 上效率低于 cuDNN/cuBLAS：端到端约 31 ms 对 PyTorch 约 25.4 ms（约 0.83×）。per-op 归因显示 FFN 的两个 GEMM 已在设备峰值的 27–82% 运行，剩余差距集中于 F32 GEMM 效率（F32 coopmat 在 NVIDIA 上实为 TF32 精度，违反 strict_f32 语义，不可用）。已完成纯 C++ 层优化（token-major CWHN 深度卷积零拷贝、LayerNorm affine 与 LayerScale 加载期折叠）；进一步的 shader 级方案（subgroup norm、depthwise bias 融合、FFN 融合）原型验证过收益（合计约 -3.5 ms）但需要维护 GGML patch，权衡后暂缓，如后续需要可从历史记录恢复。

## 覆盖与非目标

完整发布运行覆盖 3 模型 × 3 后端 × 3 存储、50 图、stage/dynamic、dense sampler、量化 oracle、可选接口、真实 C API 和公开标注子集。R/mR/F1/wR、候选集合、logits 误差、P50/P95、吞吐和后端内存均保留在报告中。

不承诺 F16/Q8 与原始 F32 逐位一致；公共子集回归不等于完整 leaderboard；缺失标注不会由推理输出替代。
