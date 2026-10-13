---
license: cc-by-nc-4.0
language:
- en
task_categories:
- object-detection
size_categories:
- 100K<n<1M
tags:
- scene-graph
- visual-relationship-detection
- open-vocabulary
- relation-detection
- coco-format
- arxiv:2609.12552
pretty_name: RA-4M
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: val
    path: data/val-*
---

# RA-4M — a verified free-text relation corpus

**472,344 images · 4,282,531 relations · 10,102 free-text predicates · 497 object
categories · 9.03 relations per image** in the training split, plus a
24,964-image validation split (226,203 relations). Machine-annotated by an
open-weight vision–language model, then filtered by deterministic geometric
gates that reject 11.3% of raw candidates. Every object also carries a
box-prompted SAM 3 mask.

RA-4M exists because open-vocabulary relation models cannot be trained on
benchmark vocabularies. Visual Genome's annotators wrote 36,549 distinct
predicate strings; the community kept the 50 most frequent, and the discarded
strings are exactly what an open-vocabulary model is supposed to cover. So
**synonyms are never collapsed here** — `riding` and `riding on` stay separate
labels. Surface-form diversity *is* the label space.

| | train | val |
|---|---|---|
| images | 472,344 | 24,964 |
| objects | 2,668,852 | 140,912 |
| relations | 4,282,531 | 226,203 |
| predicates | 10,102 | 2,222 (268 of them not in train) |

- **Paper:** [RelateAnything: Real-Time Open-Vocabulary Relation Prediction From Any Inputs](https://arxiv.org/abs/2609.12552) (Neau, 2026)
- **Code:** https://github.com/Maelic/RelateAnything
- **Models:** https://huggingface.co/collections/maelic/relateanything

## How it was built

Generation runs an open-weight VLM (`gemma-4-26B-A4B-it`) on images with
**drawn, numbered boxes**, so grounding is an input to the annotator rather than
an inference from its output. Every proposed relation then passes a
deterministic geometric gate that rejects 11.3% of raw candidates. A gate fires
only where box geometry logically constrains the predicate: it rejects a
relation or swaps its roles, never rewrites a predicate. Predicates geometry
cannot constrain pass through unchecked and are counted as residual risk.
Total annotation cost: 104 GPU-hours.

Each relation carries the provenance of that decision:

| field | meaning |
|---|---|
| `spatial` | the predicate is a spatial one, subject to the geometric gates |
| `source` | `vlm` (as generated) or `geometric` (role-swapped or emitted by the gate) |
| `round` | which generation round produced it |

## Layout

```
data/train-*.parquet, data/val-*.parquet   images + boxes + masks + relations, one row per image
annotations/ra4m_{train,val}_coco.json     COCO-SGG, one shared free-text vocabulary
packs/megasg/{train,val}/                  the corpus as RelateAnything memmap packs (val: checkpoint validation)
packs/megasg_clean/train/                  the 463,657 training images of the released models (megasg minus clean_train_excluded.json)
packs/megasg_proxy50k/train/               a 50,000-image subset of megasg_clean, the development proxy
packs/vg_raw/train/                        raw Visual Genome relations, the second training source (Visual Genome images)
masks/megasg_{train,val}_rle.jsonl         SAM 3 RLE masks, keyed by pack image index
predicates.json, categories.json           the 10,370 predicates and 497 object categories
clean_train_excluded.json                  8,687 images excluded for evaluation leakage
```

The parquet rows and the COCO json refer to each object by **the same
annotation id**, so the two artifacts join without matching boxes.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("maelic/RA-4M", split="train", streaming=True)
row = next(iter(ds))
row["image"]                       # PIL image
row["objects"][0]["bbox"]          # [x, y, w, h] in pixels
row["objects"][0]["mask_rle"]      # COCO RLE counts; size is [height, width]
[r["predicate"] for r in row["relations"]]
```

To train RelateAnything, use the packs: write the images to disk under the file
names the packs record with
[`release/ra4m/export_images.py`](https://github.com/Maelic/RelateAnything/blob/main/release/ra4m/export_images.py),
then follow
[Get the data](https://github.com/Maelic/RelateAnything/blob/main/docs/installation.md#get-the-data).
The third training source, `hicodet`, and the evaluation packs are in
[`maelic/OV-SGG-Bench`](https://huggingface.co/datasets/maelic/OV-SGG-Bench).

COCO-SGG consumers (`SGG-Benchmark`, `maelic/PSG-coco-format` tooling) want the
json instead:

```python
import json
d = json.load(open("annotations/ra4m_train_coco.json"))
d["rel_categories"][:3]            # free-text predicates, ids into rel_annotations
```

## Image provenance and licensing

**The annotations, masks and packs in this repository are released under
CC BY-NC 4.0**, except `packs/vg_raw`, which holds Visual Genome's own relations
and stays under its CC BY 4.0 license. The images are redistributed for non-commercial research use and
**remain under the terms of their source corpora**, which are not ours to relicense:

| source | images | share | terms |
|---|---|---|---|
| [Objects365](https://www.objects365.org/) | 282,764 | 56.9% | academic / non-commercial research use |
| [Open Images V6](https://storage.googleapis.com/openimages/web/index.html) | 187,202 | 37.6% | images CC BY 2.0 by their Flickr authors; annotations CC BY 4.0 |
| [COCO](https://cocodataset.org/) | 27,342 | 5.5% | images per Flickr terms; annotations CC BY 4.0 |

No image was re-encoded: the JPEG bytes are the source corpora's own. If you
hold rights to a photograph here and want it removed, open a discussion on this
repository and it will be taken down.

## Masks

Masks are **box-prompted** `facebook/sam3` — each existing box is handed to the
segmenter, so masks come back 1:1 with annotation ids and every entity id,
category and relation stays valid. (SAM's automatic mode emits ~100 part masks
per image with no correspondence to our 5.6 entities, which is why it is not
used.) Coverage is 100% of packed images in both splits. They are *predicted*
masks, not human segmentations.

## Evaluation leakage

`clean_train_excluded.json` lists **8,687 training images that appear in an
evaluation benchmark** (PSG, VG150, IndoorVG, Haystack, SpatialSense). The
models in the paper were trained on the filtered set — **463,657 images /
4,183,035 relations** — and never on these. Filter them out before training if
you intend to report on any of those benchmarks:

```python
excluded = set(json.load(open("clean_train_excluded.json"))["file_names"])
```

## Known limitations

- **No human-audited precision number.** The corpus is machine-annotated and
  verified structurally; every precision claim about it is structural, not the
  result of a person reading a sample. The gates bound one class of error
  (geometric contradiction), not annotator judgement.
- **Predicate counts differ by split.** The train split has 10,102 distinct
  predicates and the val split 2,222, written against separate vocabularies at
  packing time. Both are remapped here onto **one 10,370-string union** so a
  `predicate_id` means the same thing in both; the paper quotes train's 10,102.
- **472,344 vs the paper's 474,413.** The paper counts every image the annotator
  processed. 2,069 of them had all their relations rejected by the gates and so
  carry no supervision; they are not published here.
- **0.10% of boxes** (2,764 train / 134 val) could not be matched back to the
  source object annotations bit-for-bit and are reconstructed from the packed
  normalised box instead; their ids start at 900,000,000.
- **Directional predicates are present** (`to the left of`, `to the right of`),
  so horizontal-flip augmentation silently falsifies labels. Do not use it.
- The predicate distribution is Zipfian by design: 4,391 predicates occur once.
  Treat tail strings as vocabulary coverage, not as a balanced label set.

## Citation

```bibtex
@article{neau2026relateanything,
  title  = {RelateAnything: Real-Time Open-Vocabulary Relation Prediction From Any Inputs},
  author = {Neau, Ma\"elic},
  journal = {arXiv preprint arXiv:2609.12552},
  eprint  = {2609.12552},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  year   = {2026},
  url    = {https://arxiv.org/abs/2609.12552}
}
```
