# Dependency and algorithm provenance

GGML is the only inference dependency. It is a Git submodule pinned to v0.21.0
(`8599e0ea3756c4bac4ef813af2241cb1a8bbfb0b`). CMake applies numbered patches in
lexical order and recognizes an already applied patch through reverse checks.
The current patch adds integration provenance only; the full model uses stock
GGML operators. No changes to generated build sources are required.

Image decoding uses system libjpeg (JPEG) and `examples/stb_image.h` from the
same pinned GGML submodule (PNG and other formats). No cuDNN is linked.
The separable 22-bit bilinear coefficient construction in `src/image_io.cpp`
follows Pillow's `src/libImaging/Resample.c`. Local reference implementations
were inspected in General-Keypoint-Detection-GGML and recognize-anything-ggml.
The implementation here is adapted to the model's fixed 448 x 448 input and
validated against every retained JPEG. Pillow's license is preserved below.

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
