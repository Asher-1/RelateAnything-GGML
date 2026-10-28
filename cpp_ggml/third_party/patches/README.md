# Dependency and algorithm provenance

GGML is the only inference dependency, pinned to v0.21.0
(`8599e0ea3756c4bac4ef813af2241cb1a8bbfb0b`). All dependency edits are in
**one patch**, `0001-relateanything-ggml.patch`. CMake replays it from the pin,
recognizes the complete patched tree for idempotence, and rejects any byte
outside that tree, including untracked files. The audit and pristine forward /
reverse replay records are included in the portable report identified by `benchmarks/summary/current.json`.

The patch contains the integration marker, explicit `GGML_HINT_QUANT_F32_INPUT`
(CUDA stored-weight dequantization followed by F32 activation GEMM), and
`GGML_HINT_STRICT_F32` (CUDA default math and scalar Vulkan F32 arithmetic).
Strict Vulkan selects 32/64/128 scalar tiles; aligned variants reuse the verified
unaligned pipeline. Quantized weight rounding remains irreversible.

Strict CUDA additionally selects an unmasked F32 persistent-warp softmax for
up to 1024 columns. Its lane accumulation, XOR reduction and division follow
the arithmetic order examined in PyTorch 2.6
[`PersistentSoftmax.cuh`](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/cuda/PersistentSoftmax.cuh).
The isolated `strict-arithmetic.cu` translation unit compiles without fast-math;
other kernels retain the upstream build policy. Unsupported shapes, masks,
ALiBi and sinks use the general kernel. Sixty independent CUDA oracle cases
cover supported widths and the real 789-column attention shape.

`GGML_HINT_SERIAL_F32` preserves increasing-K F32 FMA accumulation for the
original F32 patch embedding. Shared-memory tiling changes traffic, while
each output retains its accumulation order. All three real F32 checkpoints
match official CUDA at this boundary; LayerNorm remains the next difference.
This does not make quantized storage or the entire model bit-identical.

`RA_GGML_CUDA_GRAPHS` compiles upstream capture/replay support. The explicit
`RuntimeOptions::cuda_disable_graphs` policy and CLI `--no-cuda-graphs` control
its use. Compilation does not prove capture occurred: tensor addresses and
graph properties must stabilize first. Focused operator and replay checks
are linked from `benchmarks/optimization/`, separately from the complete
earlier release snapshot.

CPU/CUDA/Vulkan and the common scheduler/registry use
`ggml_runtime_options` rather than environment feature switches. Configure the
ABI-tagged struct before backend discovery; the first configuration or read
freezes it. Identical reconfiguration succeeds and conflicting policy fails.
The host owns OpenMP and exception termination policy. Unbuilt upstream
backends are retained at the pin and are outside this project's compiled API.

Native text encoding uses ICU Unicode normalization and the CLIP BPE helper
adapted from the pinned GGML examples; see `CLIP_TOKENIZER_LICENSE`.
Image decoding uses system libjpeg and pinned GGML `examples/stb_image.h`.
No cuDNN is linked. The 22-bit bilinear coefficient construction in
`src/image_io.cpp` follows Pillow `src/libImaging/Resample.c`, adapted to
448 x 448 and validated against the real JPEG inputs. Pillow's license follows.

Copyright © 1997-2011 by Secret Labs AB
Copyright © 1995-2011 by Fredrik Lundh
Copyright © 2010 by Alex Clark and contributors

By obtaining, using, and/or copying this software and/or its associated
documentation, you agree that you have read, understood, and will comply with
the following terms and conditions:

Permission to use, copy, modify, and distribute this software and its associated
documentation for any purpose and without fee is hereby granted, provided that
the above copyright notice appears in all copies, and that both that copyright
notice and this permission notice appear in supporting documentation, and that
the name of Secret Labs AB or the author not be used in advertising or publicity
pertaining to distribution of the software without specific, written prior
permission.

SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
