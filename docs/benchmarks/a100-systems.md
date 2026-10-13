# Latency across scene-graph systems on one A100

RelateAnything against other scene-graph systems on the same NVIDIA
A100-SXM4-40GB, the same PSG test images and one timing protocol: batch 1,
median and 90th percentile, PSG's 56 predicates for every system. **Bold**: the
fastest system in each column.

| system | params | end to end, p50 / p90 (ms) | FPS | relation stage on shared boxes, p50 (ms) |
|---|---:|---:|---:|---:|
| RelateAnything ViT-S/16+, `torch.compile` | 243 M | **25.6** / 26.8 | **39.1** | **13.2** |
| RelateAnything ViT-S/16 | 236 M | 46.2 / 47.4 | 21.7 | 34.1 |
| RelateAnything ViT-S/16+ | 243 M | 47.9 / 49.3 | 20.9 | 34.8 |
| RelateAnything ViT-B/16 | 303 M | 46.0 / 47.4 | 21.7 | 33.4 |
| RelateAnything ConvNeXt-T | 248 M | 41.5 / 43.1 | 24.1 | 29.5 |
| RelateAnything ConvNeXt-S | 270 M | 47.8 / 49.5 | 20.9 | 35.0 |
| RelateAnything ConvNeXt-B | 313 M | 47.9 / 50.7 | 20.9 | 35.5 |
| RelateAnything ConvNeXt-L | 435 M | 47.6 / 48.8 | 21.0 | 35.7 |
| OvSGTR Swin-T | 177 M | 231.5 / 240.1 | 4.3 | 223.1 |
| OvSGTR Swin-B | 237 M | 243.3 / 251.7 | 4.1 | 235.5 |
| ACC Swin-T | 177 M | 498.1 / 510.1 | 2.0 | 497.1 |
| ACC Swin-B | 237 M | 520.2 / 540.7 | 1.9 | 517.7 |
| ROBIN-3B | 3,515 M | 38.8 s / 39.2 s | 0.026 | 38.8 s |
| Qwen3-VL-8B (prompted) | 8,790 M | 44.1 s / 66.6 s | 0.023 | — |

- **End to end** is pixels to scene graph. RelateAnything and ROBIN-3B use the
  same YOLO-World-v2-M detector, included in time and parameters. OvSGTR and ACC
  detect their own objects.
- **Relation stage on shared boxes** gives every system the same YOLO-World boxes,
  so only the relation model differs. OvSGTR and ACC still run their whole
  network; ROBIN-3B's value is its end-to-end time minus the detector.
- **The full 19,103-predicate vocabulary** adds 0.4–0.9 ms to RelateAnything's
  relation stage.
- **Batch 1 is CPU-bound on this host** (AMD EPYC 7J13): the relation stage
  costs the same from 45 M to 244 M parameters, and `torch.compile` cuts it from
  34.8 to 13.2 ms. Compare numbers within this table.

## How each system was run

Each system runs as its authors release it, one per process on an otherwise idle
GPU: 20 warm-up and 200 timed images (5 and 50 for the two generative systems),
decoded into memory before timing. Time runs from the in-memory image to a list
of (subject box, predicate, object box, score), with preprocessing and decoding
included and model loading excluded.

| system | configuration |
|---|---|
| RelateAnything | bf16 eager PyTorch (first row: `torch.compile`); YOLO-World-v2-M with the 497 MegaSG classes, 640 px, confidence 0.10, at most 60 boxes to the relation model |
| OvSGTR | `vg-ovdr-swin{t,b}-mega-best.pth`, fp32 with TF32 as their code sets it, at most 300 pairs |
| ACC | `ACC_Swin{T,B}_ovdr.pth`, two-pass self-query inference (one pass: 269 / 279 ms) |
| ROBIN-3B | `jamepark3922/robin-qwen2.5-3b`, greedy, at most 1,024 new tokens and 60 regions |
| Qwen3-VL-8B | `Qwen/Qwen3-VL-8B-Instruct`, bf16, greedy, asked to answer with PSG's predicates |

Measured 2026-10-07.
