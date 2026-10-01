# Official GGUF models

Nine runnable full-graph models are stored here: the three official prefixes
`relsgg-vits16`, `relsgg-vits16plus` and `relsgg-vitb16`, each in `f32`, `f16`
and `q8_0` storage. There are no demo model aliases in this directory.

[Model cards and direct downloads](../MODEL_CARDS.md) describe every artifact.
[The manifest](../../benchmarks/full_gguf_manifest.json) records hashes,
numerical tensor verification and sizes. Binaries remain outside source Git;
the launcher can fetch them using `--download`.

The embedded vocabulary matches official `full_vocabulary=True`, including
runtime embedding replacement and alpha recomputation. C++ executes the image
tower and relation graph natively. Legacy random adapter fixtures remain
explicitly named under `test_data/fixtures/` and are not deployment models.
