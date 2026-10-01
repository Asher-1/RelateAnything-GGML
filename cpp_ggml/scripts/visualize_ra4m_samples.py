#!/usr/bin/env python3
"""Run released PyTorch models on reproducible RA-4M validation examples."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODELS = ("relsgg-vits16", "relsgg-vits16plus", "relsgg-vitb16")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render(rows: list[dict], image_bytes: dict[int, bytes], path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype("DejaVuSans.ttf", 16)
    title_font = ImageFont.truetype("DejaVuSans.ttf", 19)
    cell_w, photo_h, note_h, margin = 450, 310, 180, 16
    cell_h = photo_h + note_h
    canvas = Image.new("RGB", (4 * cell_w + 5 * margin, 3 * cell_h + 4 * margin), "#f4f7f8")
    draw = ImageDraw.Draw(canvas)
    colors = ("#008d83", "#b86400", "#2366a8", "#aa4160", "#6652a1")

    for sample_idx, row in enumerate(rows):
        with Image.open(io.BytesIO(image_bytes[row["row_index"]])) as source:
            source = source.convert("RGB")
            scale = min(cell_w / source.width, photo_h / source.height)
            photo = source.resize((round(source.width * scale), round(source.height * scale)))
        boxes = row["objects"]
        panels = [("Ground truth", [
            f'#{r["subject"]} --{r["predicate"]}--> #{r["object"]}'
            for r in row["ground_truth"][:5]])]
        panels += [(model, [
            f'#{r["subject"]} --{r["predicate"]}--> #{r["object"]} ({r["score"]:.2f})'
            for r in row["predictions"][model][:5]]) for model in MODELS]
        for column, (title, relations) in enumerate(panels):
            left = margin + column * (cell_w + margin)
            top = margin + sample_idx * (cell_h + margin)
            canvas.paste(photo, (left, top))
            for object_idx, obj in enumerate(boxes):
                x, y, w, h = obj["bbox_xywh"]
                color = colors[object_idx % len(colors)]
                draw.rectangle((left + round(x * scale), top + round(y * scale),
                                left + round((x + w) * scale), top + round((y + h) * scale)),
                               outline=color, width=3)
                draw.text((left + round(x * scale) + 2, top + round(y * scale) + 2),
                          f'#{object_idx} {obj["category"]}', font=font, fill=color,
                          stroke_width=2, stroke_fill="white")
            caption = f'{row["source"]} / {row["image_id"]} / {title}'
            draw.text((left, top + photo_h + 5), caption, font=title_font, fill="#1b3039")
            line_y = top + photo_h + 34
            for relation in relations:
                for line in textwrap.wrap(relation, width=48, max_lines=2, placeholder="..."):
                    if line_y + 19 > top + cell_h:
                        break
                    draw.text((left, line_y), line, font=font, fill="#243b44")
                    line_y += 22
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "cpp_ggml/test_data/ra4m_50")
    parser.add_argument("--models-root", type=Path, default=ROOT / "cpp_ggml/models/pytorch")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "cpp_ggml/benchmarks/ra4m_python_samples.json")
    parser.add_argument("--plot", type=Path, default=ROOT / "cpp_ggml/benchmarks/ra4m_python_samples.png")
    args = parser.parse_args()
    try:
        import numpy as np
        import torch
        from PIL import Image
        from relsgg import RelateAnything
    except ImportError as exc:
        raise SystemExit(f"sample inference dependencies are unavailable: {exc}") from exc
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = [next(row for row in manifest["samples"] if row["source"] == source)
               for source in ("coco", "objects365", "openimages")]
    selected = range(len(samples))
    image_bytes = {idx: (args.data / row["image"]).read_bytes()
                   for idx, row in enumerate(samples)}
    rows = []
    checkpoint_sha256 = {}
    for idx in selected:
        row = samples[idx]
        objects = [{"category": obj["category"], "bbox_xywh": obj["bbox"]}
                   for obj in row["objects"]]
        rows.append({
            "row_index": idx,
            "image_id": row["image_id"],
            "file_name": row["file_name"],
            "source": row["source"],
            "objects": objects,
            "ground_truth": [{"subject": rel["subject"], "object": rel["object"],
                              "predicate": rel["predicate"]} for rel in row["relations"]],
            "predictions": {},
        })

    for model_name in MODELS:
        checkpoint = args.models_root / model_name / "model.pth"
        checkpoint_sha256[model_name] = file_sha256(checkpoint)
        model = RelateAnything.from_checkpoint(str(checkpoint), device=args.device,
                                                full_vocabulary=True)
        for result in rows:
            with Image.open(io.BytesIO(image_bytes[result["row_index"]])) as source:
                image = source.convert("RGB")
            boxes = np.array([[x, y, x + w, y + h]
                              for x, y, w, h in (obj["bbox_xywh"] for obj in result["objects"])],
                             dtype=np.float32)
            labels = [obj["category"] for obj in result["objects"]]
            triplets = model.predict(image, boxes, box_labels=labels, topk=5)
            result["predictions"][model_name] = [
                {"subject": int(t.subject_idx), "object": int(t.object_idx),
                 "predicate": t.predicate, "score": float(t.score)} for t in triplets]
            print(f'{model_name} {result["source"]}/{result["image_id"]}: {len(triplets)} triplets')
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    report = {
        "dataset": "maelic/RA-4M",
        "dataset_revision": manifest["revision"],
        "split": "val",
        "compact_manifest": str(manifest_path),
        "compact_manifest_sha256": file_sha256(manifest_path),
        "device": args.device,
        "checkpoint_sha256": checkpoint_sha256,
        "torch_version": torch.__version__,
        "vocabulary": "each checkpoint's full released vocabulary",
        "scope": "three qualitative examples; not an official metric or full-validation result",
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    render(rows, image_bytes, args.plot)
    print(f"wrote {args.output} and {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
