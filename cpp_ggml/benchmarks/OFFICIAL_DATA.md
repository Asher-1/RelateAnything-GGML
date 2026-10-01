# Official data and evaluation coverage

`official_data_manifest.json` is the machine-readable source of truth for the
current local data. The initial 122-file, 49,350,823,236-byte RA-4M snapshot
was fully verified and audited, then pruned after extracting 50 original
validation images with complete GT to `cpp_ggml/test_data/ra4m_50/`.
[`ra4m_verification.json`](ra4m_verification.json) documents the **historical**
full download, not a file set that is still present. The retained suite and
all-image Python-CUDA measurements are in [`RA4M_50.md`](RA4M_50.md).

[`ra4m_validation_audit.json`](ra4m_validation_audit.json) is a historical scan
of every validation annotation: 24,964 images from COCO (1,393), Objects365 (14,073),
and Open Images (9,498); 140,912 objects, 226,203 relations, and 2,222 distinct
predicate strings. Relation endpoints and box extents passed structural checks.
There are 307 source boxes outside an image edge by more than one pixel, with a
maximum 34-pixel overflow; downstream evaluators must apply their stated
box-clipping policy consistently. The distribution plot is
[`ra4m_validation_audit.png`](ra4m_validation_audit.png).

[`ra4m_python_samples.png`](ra4m_python_samples.png) puts retained original images,
source annotations, and full-vocabulary Python predictions from all three
official checkpoints side by side for one COCO, one Objects365, and one Open
Images validation row. Its structured counterpart records the exact dataset
revision, checkpoint hashes, boxes, GT relations, and predicted scores in
[`ra4m_python_samples.json`](ra4m_python_samples.json). Three examples cannot
substitute for a full 24,964-image accuracy evaluation.

The [OV-SGG-Bench dataset URL](https://huggingface.co/datasets/maelic/OV-SGG-Bench)
returned HTTP 401 to an unauthenticated Hugging Face API request on
2026-09-30; an earlier attempt returned 404. Neither establishes public access. It is the named pack in
the [upstream six-axis protocol](../../benchmark/SPEC.md), including explicit
negative examples, fixed splits, calibration and matching assets. It is needed
only to reproduce those **exact published protocol scores**. It is not needed
for same-image Python/GGML parity, latency, the RA-4M 50-image diagnostic, or
new task evaluations with a stated protocol. The local 50-image evaluation
already ran independently of this URL.

Public, accessible alternatives from the same publisher are
[VG150 COCO format](https://huggingface.co/datasets/maelic/VG150-coco-format),
[PSG COCO format](https://huggingface.co/datasets/maelic/PSG-coco-format), and
[IndoorVG COCO format](https://huggingface.co/datasets/maelic/IndoorVG-coco-format).
Their formats and sample protocols differ from the missing OV-SGG pack; report
their scores separately. Their README and category metadata have been fetched
locally (about 1.1 MB together with RA-4M metadata); image-bearing test shards
have not been downloaded. The downloader defaults to metadata from these
available repositories and RA-4M. To retry the missing exact pack explicitly:

```sh
python3 cpp_ggml/scripts/download_test_data.py \
  --dataset maelic/OV-SGG-Bench --dest cpp_ggml/test_data/official --allow-large
```

## Required evaluation matrix

The task metrics and protocols below follow the upstream
[`benchmark/SPEC.md`](../../benchmark/SPEC.md) and
[`docs/evaluation.md`](../../docs/evaluation.md). The A40 model ladder's
OVS-F1 is a chance-corrected harmonic mean over A1, A2, A4, and A6; it is not
an F1 score measured on RA-4M validation images.

The full released-tower OVS composite instead spans A1, A2, A4, A5, and A6.
Those two composites are not interchangeable. Required reporting also includes:

| Scope | Report together | Interpretation |
|---|---|---|
| Transfer and long tail | graph-constrained R@50, mR@50, F1@50, wR@50, rare/common/frequent R@50 and class support | micro recall is secondary; tail results need support counts |
| Explicit negatives | federated AP with `n_pos >= 5`, rare fAP, P-AUC, PDD, PDO | Haystack negatives assess hard-negative discrimination, not real-world precision |
| Open vocabulary | 19,103-string mR@50, synonym threshold and text-embedding revision | the deployed vocabulary and calibrated matcher are part of the protocol |
| Deployment and spatial | shared-detector pair-recall ceiling, PSG wR@50/mR@50, SpatialSense macro and pooled AUC | keep detector and threshold settings identical |
| Graph quality | accepted true bits per image, annotation-information share, judge/control results | judge provenance and blinded prompt are part of the measurement |
| Composite | axis vector, OVS harmonic score, OVS arithmetic score, weakest axis, balance | compare only scores with the same axis set |
| Numerical/runtime parity | raw-logit MAE/RMSE/max error, pair ranking, p50/p95 latency, throughput, peak memory, exact checkpoint and GGML commit | use identical images, boxes, vocabulary, math mode and postprocessing |

The upstream protocol explicitly excludes zero-shot triplet recall as a claim
for this transfer setting and warns against ECE/Brier on the federated negative
set. See the [official metric definitions](../../benchmark/SPEC.md) and
[evaluation guide](../../docs/evaluation.md).

| Axis | Required metric | Required official source | Current local status |
|---|---|---|---|
| A1 transfer | wR@50 plus overlap-bucket split across VG150, PSG, IndoorVG, HICO-DET | OV-SGG packs and source images | exact published protocol unavailable; public VG150/PSG/IndoorVG alternatives exist; full C++ graph implemented |
| A2 precision | federated AP, P-AUC, PDD, PDO on explicit negatives | Haystack and HICO-DET negative cells | exact negative pack unavailable; RA-4M positives cannot replace it |
| A3 open vocabulary | mR@50 with 19,103-string vocabulary and calibrated synonym matcher | OV-SGG packs and text assets | 50-image Python/GGML full-vocabulary exact recall measured; calibrated axis matcher not evaluated |
| A4 deployment | SGDet wR@50 against shared detector pair-recall ceiling | PSG pack, images, shared detector | public PSG available for a separate protocol; full C++ graph implemented |
| A5 graph quality | judge-accepted true bits per image under controls | interchange predictions and VLM judge | full C++ predictions available; requires a defined judge protocol |
| A6 spatial | SpatialSense balanced triple AUC | SpatialSense test/valid cells and images | exact axis pack unavailable; full C++ graph implemented |

Numerical parity needs raw pair/predicate logits, MAE, RMSE, maximum absolute
error, and ranking agreement on the same image, boxes, vocabulary, and
postprocessing. Performance needs warmed median/p95 latency, throughput,
peak device memory, model size, hardware, dtype, backend, and exact model and
GGML revisions. The current fixture parity and single-image Python-CUDA
baseline are documented separately; their graphs are not equivalent.

The repository sample image `assets/reel/images/horse.jpg` is used for the
official Python smoke benchmark. The resulting `pytorch_cuda_results.jsonl`
records each of the three official checkpoints separately. C++ adapter results
are in `results.jsonl` and `parity_results.json`; their input is precomputed
object features, so they are not a full DINOv3 pipeline comparison. No
six-axis OV-SGG comparison is claimed. The current [50-image full-graph report](full_graph/README.md) now compares all
three models and all CUDA/Vulkan F32/F16/Q8 configurations against Python.
