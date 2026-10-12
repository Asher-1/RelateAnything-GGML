# GGML 发布验收状态

## 当前结论

- 9 个 GGUF 已发布并经远端回下载 SHA-256/字节数 9/9 校验。
- 当前模型哈希对应的专项证据：G1 数值门槛 27/27；G2 GPU 性能门槛 17/18。
- 唯一 G2 未通过项：`relsgg-vitb16/cuda-f32` 52.27 ms，对 PyTorch 50.27 ms（0.96×）。
- 新的默认 auto 完整发布运行正在生成；在其发布前，[summary/current.json](summary/current.json) 仍指向上一份稳定报告。

## 默认精度规则

默认请求为 `auto`：仅 CUDA 加载全 F32 矩阵时生效 `strict_f32=1`；其余 CPU、Vulkan、F16、Q8_0 配置生效 `strict_f32=0`。显式 `--strict-f32` / `--no-strict-f32` 仅作诊断或覆盖，不代表默认发布路径。

strict 保留 F32 GEMM/softmax/norm 与显式 F32 attention，避免 TF32/F16 中间算术触发 top-k 边界换对；non-strict 使用更快的 TF32、flash F16 K/V 或量化激活路径。F16/Q8 已丢失的存储精度无法由 strict 恢复。

## 发布门槛

| 门槛 | 当前结果 | 说明 |
|---|---:|---|
| G1 数值 | 27/27 | F32/F16/Q8 × CPU/CUDA/Vulkan 均通过各自门槛 |
| G2 GPU 性能 | 17/18 | 原门槛仍为 18/18；唯一偏差不会被改写为机器通过 |
| 工件身份 | 9/9 | [full_gguf_manifest.json](full_gguf_manifest.json) 与 Hugging Face 一致 |

若受限优化不能在保持 G1 的同时使 G2 达到 18/18，最终发布将明确记录一项 `accepted_deviation`，表示接受 CUDA F32 精度优先的性能代价，而不是放宽门槛。

## 覆盖与非目标

完整发布运行覆盖 3 模型 × 3 后端 × 3 存储、50 图、stage/dynamic、dense sampler、量化 oracle、可选接口、真实 C API 和公开标注子集。R/mR/F1/wR、候选集合、logits 误差、P50/P95、吞吐和后端内存均保留在报告中。

不承诺 F16/Q8 与原始 F32 逐位一致；公共子集回归不等于完整 leaderboard；缺失标注不会由推理输出替代。
