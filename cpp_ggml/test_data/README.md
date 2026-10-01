# Test data

Open [index.html](index.html) to view every packaged sample, its original image,
labelled object boxes and relations. [catalog.json](catalog.json) supplies
portable image paths, pixel XYXY boxes and matching object names for component
autofill. [ZIP and ACloudViewer integration](PACKAGE.md) explains packaging,
integrity checks and the existing shared cache contract.

`ra4m_50/` contains 50 retained original RA-4M validation images, their hashes,
source revision, object boxes and positive relation annotations. The original
49.35 GB snapshot has been pruned. Verify the retained inputs with:

```sh
python3 cpp_ggml/scripts/compact_ra4m.py --verify-only
```

`metric_subsets/` contains bounded labelled PSG/IndoorVG/VG150 samples and
SpatialSense valid/test cells; each manifest records source, split and image
hashes. `official/` holds downloaded source metadata. `fixtures/` holds small
synthetic pair-adapter contract inputs, separately from real-model evaluation.
The source datasets' terms apply to retained images/annotations.

Build the deterministic archive, including its offline preview:

```sh
python3 cpp_ggml/scripts/package_test_data.py
python3 cpp_ggml/scripts/package_test_data.py --verify-only
```

The archive is generated at
`outputs/packages/relateanything_test_data.zip`. The adjacent `.package.json`
and [package.json](package.json) record its pinned SHA-256, byte size and layout.

Generated traces, masks, raw tensors, predictions and temporary files belong
under `outputs/runs/<run-id>/artifacts` or `outputs/probes`, never this input
asset tree. [Portable full-validation report](../benchmarks/summary/README.md),
[visual/numerical gallery](../benchmarks/gallery/index.html) and
[protocol asset requirements](../benchmarks/OFFICIAL_DATA.md) are separate.
