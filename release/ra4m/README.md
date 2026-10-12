# RA-4M image manifest

RA-4M's relations are annotated on 500,000 images (475,000 train + 25,000 val) taken from
MEGASG, i.e. the HF dataset [JosephZ/mega_1m](https://huggingface.co/datasets/JosephZ/mega_1m)
(988,531 images, 444 parquet shards, ~221 GB). The images are not redistributed here; this
directory lets you rebuild the exact image set from the upstream dataset.

The images are also inline in [`maelic/RA-4M`](https://huggingface.co/datasets/maelic/RA-4M)
(~44 GB); `export_images.py` writes them to `<out>/{train,val}/<file_name>` from there, which
is the shorter route. The manifest below rebuilds the same set from the upstream dataset.

| file | contents |
|---|---|
| `ra4m_image_manifest.tsv.gz` | one row per image: `split, coco_image_id, file_name, original_id, source, width, height, num_relations, in_megasg_clean` |
| `ra4m_image_filenames.txt.gz` | the 500,000 `file_name`s only |
| `extract_from_mega1m.py` | streams the shards and writes `<out>/{train,val}/<file_name>` |
| `export_images.py` | the same output, from the `maelic/RA-4M` parquet shards |

* `coco_image_id` is the `id` in `DATASETS/MEGASG/<split>/_annotations.coco.json`;
  `original_id` equals mega_1m's `image_id`.
* `source`: objects365 282,802 · openimages 189,856 · coco 27,342.
* `num_relations`: relations in the coco json (498,096 images have >= 1).
* `in_megasg_clean`: 1 for the 463,657 images in the `megasg_clean` training pack.

Verified 2026-09-01 on shard `train-00000-of-00444`: all 2,227 of its file names are in the manifest.
