# Evaluation scope and metric coverage

The full-graph comparison evaluates all 50 retained RA-4M GT images (803
annotated triplets, 205 predicate strings, COCO/Objects365/Open Images sources)
with the same boxes and 19,103-predicate vocabulary. The implementation uses
one argmax predicate per ordered pair, then the official calibrated score
`sigmoid(a * (predicate_logit + pair_logit) + b)`.

The metric definitions are taken from the upstream sources
[`docs/evaluation.md`](../../docs/evaluation.md),
[`benchmark/SPEC.md`](../../benchmark/SPEC.md) and
[`relsgg/eval/evaluator.py`](../../relsgg/eval/evaluator.py).
The published protocol is also available in the
[official scoring guide](https://github.com/Maelic/RelateAnything/blob/main/docs/evaluation.md).
No unavailable evaluation pack is required to run the local comparison.

| Metric / question | Current evidence | Interpretation |
|---|---|---|
| Pixel and normalized box parity | All 50 inputs, bit equality | C++ decode/resize/normalization vs PIL/NumPy |
| Intermediate stage RMSE/max error | All released architectures, independent Python traces | Locates numerical drift along the actual image graph |
| Final logits MAE/RMSE/max/cosine | Every image × 18 GPU configurations | Pairs matched by subject/object IDs; unmatched pairs reported separately |
| Pair set recall/Jaccard, predicate top-1 agreement | Every configuration | Isolates candidate selection and close-score rank flips |
| R@20/50/100 | Micro exact GT recall | Fraction of annotated triples recovered |
| mR@20/50/100 | Macro exact GT recall | Each of the 205 observed predicates has equal weight |
| F1@K | Harmonic mean of R@K and mR@K | Upstream SGG definition, not precision/recall F1 |
| wR@K | Log-frequency weighted class recall | Local support weighting |
| Rare/common/frequent mR | Support 1 / 2–4 / ≥5 in this subset | Explicit local buckets; not training-corpus frequency categories |
| Source breakdown | COCO, Objects365, Open Images | No source is omitted from evaluation |
| Median/P95 wall time, iterations, weight size | All models/backends/storage | Native full pipeline; model load and file writes excluded |
| Per-image visual comparison | 150 grids with GT, Python and six GGML variants | Inspect all images, not a selected success gallery |
| Custom vocabulary | New six-row bank, scaled and composite embeddings | Validates normalization, learned routing and scoring |
| SoftR/SoftmR/SoftF1 | Not reported as calibrated benchmark scores | Needs the matcher/ontology/inverse masks and threshold calibrated for this text space |
| Federated AP, P-AUC, accuracy, calibration ECE/Brier | Not identifiable from this subset | Need adjudicated negatives and held-out calibration; absent annotation is not a negative |
| OVS/OVS-F1 composite (six-axis protocol) | Not claimed | Requires the specified axis packs/normalizers and comparable axis set |
| SGDet, detector retention, box AP/mAP | Not measured by a relation-only GT-box test | Requires a fixed detector and its recall/operating point |
| Robustness interventions and zero-shot HICO | Not a property of this 50-frame subset | Need paired counterfactuals or held-out split provenance |

The upstream OV-SGG-Bench address is
[maelic/OV-SGG-Bench](https://huggingface.co/datasets/maelic/OV-SGG-Bench).
It provides axis-specific packs, negative judgments and calibration assets.
The previous download attempt returned 404; a fresh unauthenticated API check
on 2026-09-30 returned 401. The exact pack remains publicly unavailable to this check. This is not a dependency of model
conversion, full-graph validation, or the retained-image benchmark. Fabricating
negative labels or substituting a new synonym threshold would change the
metric rather than complete that upstream benchmark. Local exact recall,
raw numerical parity and visual comparisons are available independently.

The compact suite was chosen from validation data for diversity/difficulty.
It is an engineering regression set, not a new representative leaderboard or
proof of generalization. All metrics and speed statements carry this scope.

## Broader metric literature

The upstream evaluation guide and six-axis specification were cross-checked
against their public source pages on 2026-09-30. Two primary research sources
explain why positive-only exact recall cannot justify a complete metric claim:

- [Tackling the Unannotated: Scene Graph Generation with Bias-Reduced Models](https://arxiv.org/abs/2008.07832), 2020: annotation sparsity and missing valid relations affect measured recall and training bias.
- [Rethinking the Evaluation of Unbiased Scene Graph Generation](https://arxiv.org/abs/2208.01909), 2022: cross-category ranking and compositional diversity affect mR; proposes independent mean recall and weighted variants.

Those alternative metrics are not silently substituted for the upstream
protocol. Reproducing them requires their candidate ranking and class-support
rules; this report retains the defined R/mR and reports numerical equivalence
separately from benchmark completeness.
