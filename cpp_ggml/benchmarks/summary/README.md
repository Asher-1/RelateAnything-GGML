# 跨平台基准报告发布

此目录保存 Git 可跟踪的稳定发布入口。完整 benchmark 报告以 ZIP 形式发布到 Hugging Face；当前发布身份、下载 URL、文件大小和 SHA-256 固定在 [current.json](current.json)。

`outputs/runs/<run-id>/artifacts/` 包含本地原始执行证据，可能很大，既不进入 Git，也不包含在报告 ZIP 中。报告 ZIP 仅包含经过运行验证后的 `reports/` 视图：Markdown、HTML、CSV、JSON、PNG 和 SVG。

下载报告后使用以下命令校验：

```sh
python3 cpp_ggml/scripts/benchmark_release.py verify-package \
  --archive <downloaded-report.zip> \
  --descriptor <matching-package.json>
```

本地 `benchmarks/current` 若存在，只是开发工作区的便捷入口；它不属于发布接口，也不应被文档或跨平台工具依赖。
