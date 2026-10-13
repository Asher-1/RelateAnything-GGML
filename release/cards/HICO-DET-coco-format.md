---
license: other
license_name: hico-det
license_link: https://www-personal.umich.edu/~ywchao/hico/
task_categories:
  - object-detection
tags:
  - scene-graph
  - visual-relationship-detection
  - human-object-interaction
  - coco-format
language:
  - en
pretty_name: HICO-DET (COCO format)
size_categories:
  - 10K<n<100K
---

# HICO-DET (COCO format)

The **HICO-DET** human-object interaction benchmark
([Chao et al., WACV 2018](https://arxiv.org/abs/1702.05448)) as a scene-graph dataset in
COCO format: object boxes, and each annotated interaction as a directed
`(person, predicate, object)` relation. It is the HICO-DET data
[RelateAnything](https://github.com/Maelic/RelateAnything) trains and evaluates on (the
`hicodet` pack of [`maelic/OV-SGG-Bench`](https://huggingface.co/datasets/maelic/OV-SGG-Bench)),
in the layout of [`maelic/PSG-coco-format`](https://huggingface.co/datasets/maelic/PSG-coco-format)
and [`maelic/VG150-coco-format`](https://huggingface.co/datasets/maelic/VG150-coco-format).

## Statistics

| Split | Images | Object annotations | Relations |
|-------|-------:|-------------------:|----------:|
| train | 37 633 | 86 799 | 64 929 |
| test  | 9 546 | 22 843 | 18 846 |

## Layout

```
data/train-*.parquet, data/test-*.parquet   image, image_id, width, height, file_name, objects, relations
annotations/{train,test}.json               the same annotations as COCO-SGG json
categories.json                              80 object categories, 116 predicates
```

Boxes are COCO `[x, y, width, height]` in pixels. A relation's `subject_id` and `object_id`
are the `id`s of the image's objects; `predicate_id` indexes `rel_categories`. Both splits
share the category ids (the 80 COCO classes in COCO order, ids 1-80, with HICO-DET's
spellings such as `dining_table`) and the predicate ids (alphabetical, 0-115).

```python
from datasets import load_dataset

ds = load_dataset("maelic/HICO-DET-coco-format", split="test")
row = ds[0]
row["image"], row["objects"][0]["bbox"], row["relations"][0]
```

## From HICO-DET to scene graphs

- Each annotated interaction becomes one relation from the person box to the object box.
  The predicate is the verb's gerund (`ride` -> `riding`, `sit_on` -> `sitting on`):
  the 116 verbs of HICO-DET without `no_interaction`.
- HICO-DET stores one person box and one object box per interaction, so the same object
  appears several times with slightly different coordinates. Boxes of the same category
  that overlap at IoU >= 0.5 (HICO-DET's own matching criterion) are merged into one object
  (their mean box); a relation whose two ends merge is dropped.
- Invisible instances are skipped, and an image keeps at most 40 boxes.
- `no_interaction` annotations and the image-level negative captions are not relations
  here; RelateAnything turns them into the explicit negatives of
  [`maelic/OV-SGG-Bench`](https://huggingface.co/datasets/maelic/OV-SGG-Bench)
  (`datamix/hicodet_negatives*.json`).

The conversion is
[`training/convert_hicodet.py`](https://github.com/Maelic/RelateAnything/blob/main/training/convert_hicodet.py);
this repository is built by
[`release/build_hicodet_coco.py`](https://github.com/Maelic/RelateAnything/blob/main/release/build_hicodet_coco.py).

## Predicates (116)

> adjusting · assembling · blocking · blowing · boarding · breaking · brushing with · buying · carrying · catching · chasing · checking · cleaning · controling · cooking · cutting · cutting with · directing · dragging · dribbling · drinking with · driving · drying · eating · eating at · exiting · feeding · filling · flipping · flushing · flying · greeting · grinding · grooming · herding · hitting · holding · hopping on · hosing · hugging · hunting · inspecting · installing · jumping · kicking · kissing · lassoing · launching · licking · lifting · lighting · loading · losing · lying on · making · milking · moving · opening · operating · packing · painting · parking · paying · peeling · petting · picking · picking up · pointing · pouring · pulling · pushing · racing · reading · releasing · repairing · riding · rowing · running · sailing · scratching · serving · setting · shearing · signing · sipping · sitting at · sitting on · sliding · smelling · spining · squeezing · stabbing · standing on · standing under · sticking · stirring · stopping at · straddling · swinging · tagging · talking on · teaching · texting on · throwing · toasting · training · turning · tying · typing on · walking · washing · watching · waving · wearing · wielding · zipping

## Images and license

The images are those of HICO-DET as distributed in
[`zhimeng/hico_det`](https://huggingface.co/datasets/zhimeng/hico_det); their JPEG bytes
are copied unchanged. Images and annotations remain under the terms of the original
HICO-DET release ([project page](https://www-personal.umich.edu/~ywchao/hico/)); use them
for research and cite the original work:

```bibtex
@inproceedings{chao2018learning,
  title     = {Learning to Detect Human-Object Interactions},
  author    = {Chao, Yu-Wei and Liu, Yunfan and Liu, Xieyang and Zeng, Huayi and Deng, Jia},
  booktitle = {IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2018}
}
```
