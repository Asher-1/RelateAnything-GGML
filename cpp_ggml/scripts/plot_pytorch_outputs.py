#!/usr/bin/env python3
"""Render observed triplets from the measured official Python checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    rows = [row for row in rows if row.get("status") == "passed"]
    if not rows:
        raise SystemExit("no passing Python predictions in input")
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise SystemExit("Pillow is required to render predictions") from exc

    font = ImageFont.truetype("DejaVuSans.ttf", 18)
    title_font = ImageFont.truetype("DejaVuSans.ttf", 21)
    width, image_height, footer_height, gap = 480, 340, 132, 18
    canvas = Image.new("RGB", (len(rows) * width + (len(rows) + 1) * gap,
                                image_height + footer_height + 2 * gap), "#f7f9fa")
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(rows):
        left = gap + index * (width + gap)
        top = gap
        with Image.open(row["image"]) as source:
            source = source.convert("RGB")
            scale = min(width / source.width, image_height / source.height)
            resized = source.resize((round(source.width * scale), round(source.height * scale)))
            canvas.paste(resized, (left, top))
        colors = ("#00d0b0", "#ffd166", "#79a7f1", "#f28fa4")
        fallback_labels = (["person", "horse"] if Path(row["image"]).name == "horse.jpg"
                           and len(row["boxes"]) == 2 else
                           [f"object {idx}" for idx in range(len(row["boxes"]))])
        labels = row.get("box_labels", fallback_labels)
        for box_idx, (box, label) in enumerate(zip(row["boxes"], labels)):
            color = colors[box_idx % len(colors)]
            x1, y1, x2, y2 = [round(value * scale) for value in box]
            draw.rectangle((left + x1, top + y1, left + x2, top + y2), outline=color, width=4)
            draw.text((left + x1 + 3, top + y1 + 3), label, font=font, fill=color,
                      stroke_width=2, stroke_fill="#172b35")
        text_top = top + image_height + 8
        draw.text((left, text_top), row["model"], font=title_font, fill="#172b35")
        for triplet_index, triplet in enumerate(row["triplets"][:3]):
            draw.text((left, text_top + 32 + 27 * triplet_index), triplet,
                      font=font, fill="#263d47")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
