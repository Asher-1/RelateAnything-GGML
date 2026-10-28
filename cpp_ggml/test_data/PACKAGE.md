# 可缓存测试数据与自动填充

[查看原图、框和标注](index.html) · [下载 ZIP](../outputs/packages/relateanything_test_data.zip) · [下载身份与哈希](package.json)

包包含 100 个样本入口：RA-4M 50 图，PSG / IndoorVG / VG150 各 10 图，
SpatialSense valid/test 各 10 个图像记录。SpatialSense 同时保留显式正负关系标签。
模型权重单独下载；此包未补齐同义词校准、Haystack、detector、judge/HICO/OVS 和 ECE/Brier 的完整协议资产。

## ZIP 结构与校验

```text
relateanything_test_data/
  index.html             # 离线图片、框、关系标注预览
  catalog.json           # 自动填充与样本选择
  manifest.json          # 每个内容文件的大小与 SHA-256
  ra4m_50/               # 原图与原始 manifest
  metric_subsets/        # 原图、原始标注、类别、split、来源
  official/              # 下载的原始数据说明与元数据
  README.md
```

固定成员顺序、日期和权限使相同输入生成相同 ZIP；按 1 MiB 块压缩和校验。
外部 `relateanything_test_data.package.json` 固定完整 ZIP 的 SHA-256 和大小。
包内 `manifest.json` 校验全部内容文件；验证器同时检查成员路径、重复项和样本图片哈希。
数据来源版本、split 与原始条款继续保留，不把未标注关系当负例。

```sh
python3 cpp_ggml/scripts/package_test_data.py
python3 cpp_ggml/scripts/package_test_data.py --verify-only
```

## ACloudViewer 缓存路径

现有 `ecvTestDataRepository::extractDataset()` 解压到 `extract/`，
再检查 `extractDirName`。因此 ZIP 内含一个 `relateanything_test_data/` 根目录：

```text
~/cloudViewer_data/download/relateanything_test_data.zip
~/cloudViewer_data/extract/relateanything_test_data/catalog.json
~/cloudViewer_data/extract/relateanything_test_data/index.html
~/cloudViewer_data/extract/relateanything_test_data/ra4m_50/images/9282.jpg
```

注册数据集时从 `package.json` 取 `zip_file_name`、`extract_dir_name`、
`zip_sha256` 和发布后的 `download_url`，构造现有 `DatasetInfo` 与
`ecvAssetIntegrity::Anchor{QCryptographicHash::Sha256, digestHex}`。
复用 `startDownload()`、`extractDataset()` 和完整性账本。

当前 ZIP 已在本地生成并验证，`download_url` 为 null；尚未发布远端文件，
ACloudViewer 尚无 RelateAnything 数据集枚举和按钮连接，因而远端一键下载尚未启用。
此包已提供其所需目录和字段，不把计划中的按钮当成已实现功能。

## 组件填充契约

读取 `catalog.json`，选择 `default_sample` 指定的记录。路径相对数据集解压根目录：

```json
{
  "image": "ra4m_50/images/9282.jpg",
  "boxes_xyxy": [[279.0, 181.0, 298.0, 226.0]],
  "object_names": ["person"]
}
```

示例仅展示一个对象，实际记录保存该图全部对象。`boxes_xyxy` 和 `object_names`
长度、顺序相同，坐标单位是原图像素。`relations` 和 `labelled_cells` 的
subject/object 是数组下标；`objects[].id` 保留原始对象 ID。
`inference_ready=false` 表示原始记录不足两个标注对象，保留原图供预览，
自动填充时跳过此记录；不会凭空补框或标注。
`annotation_manifest` 与 `annotation_sample_index` 指向原始标注，避免把自动填充字段作为第二份标注源。
谓词库默认由所选模型提供；这些小子集的标签不能替代模型完整谓词库。
