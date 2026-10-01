#!/usr/bin/env python3
"""Vendor the exact header closure of PyTorch 2.6.0's F32 mem-efficient attention.

Official F32 SDPA on sm80-sm99 runs
AttentionKernel<float, Sm80, true, 64, 64, 64, true, true> compiled from
PyTorch v2.6.0 and its pinned CUTLASS commit. Its 3xTF32 tensor-core GEMMs and
online-softmax tiling define the result bits, so the strict GGML path compiles
the same source instead of re-deriving it. Only headers reached by that
instantiation are kept; every file is hashed in UPSTREAM.json.

  vendor_mem_eff_attention.py --fetch [--cache DIR]   # download, verify, copy
  vendor_mem_eff_attention.py                         # verify the vendored tree
"""
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1] / "third_party" / "pytorch_mem_eff"
PYTORCH_REF = "v2.6.0"
PYTORCH_TREE = "1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340"
CUTLASS_COMMIT = "bbe579a9e3beb6ea6626d9227ec32d0dae119a49"  # pytorch v2.6.0 third_party/cutlass
CUTLASS_SHA256 = "9fa1da6be3d2d9207b801d5768cbced59c202444a8c84b82325b0670f47f9d48"
MEM_EFF = "aten/src/ATen/native/transformers/cuda/mem_eff_attention"
PROBE = f"#include <{MEM_EFF.removeprefix('aten/src/')}/kernel_forward.h>\n"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_sha1(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            path.write_bytes(response.read())
    return path


def fetch(cache):
    tree_url = f"https://api.github.com/repos/pytorch/pytorch/git/trees/{PYTORCH_TREE}?recursive=1"
    tree = json.loads(download(tree_url, cache / "tree.json").read_text())
    assert tree["sha"] == PYTORCH_TREE and not tree["truncated"]
    blobs = {e["path"]: e["sha"] for e in tree["tree"] if e["type"] == "blob"}
    wanted = [p for p in blobs if p == "LICENSE" or (p.startswith(MEM_EFF + "/") and p.endswith(".h"))]
    pytorch = cache / "pytorch"
    for rel in wanted:
        data = download(f"https://raw.githubusercontent.com/pytorch/pytorch/{PYTORCH_REF}/{rel}",
                        pytorch / rel).read_bytes()
        if git_blob_sha1(data) != blobs[rel]:
            raise SystemExit(f"{rel}: blob does not match PyTorch tree {PYTORCH_TREE}")
    tgz = download(f"https://github.com/NVIDIA/cutlass/archive/{CUTLASS_COMMIT}.tar.gz", cache / "cutlass.tgz")
    if sha256(tgz) != CUTLASS_SHA256:
        raise SystemExit("CUTLASS archive hash mismatch")
    cutlass = cache / f"cutlass-{CUTLASS_COMMIT}"
    if not cutlass.exists():
        with tarfile.open(tgz) as archive:
            archive.extractall(cache, filter="data")

    # The shim is maintained here, not upstream; it survives re-vendoring.
    shim = ROOT / "shim"
    with tempfile.TemporaryDirectory() as tmp:
        probe = pathlib.Path(tmp) / "probe.cu"
        probe.write_text(PROBE)
        deps = subprocess.run(
            ["nvcc", "-std=c++17", "-arch=sm_80", "--expt-relaxed-constexpr", "--expt-extended-lambda",
             "-I", str(shim), "-I", str(pytorch / "aten/src"), "-I", str(cutlass / "include"),
             "-M", str(probe)], check=True, capture_output=True, text=True).stdout
    deps = [pathlib.Path(d).resolve() for d in deps.replace("\\\n", " ").split()[1:]]
    files = {}
    for dep in deps:
        for base, dest in ((pytorch.resolve(), "pytorch"), ((cutlass / "include").resolve(), "cutlass/include")):
            if dep.is_relative_to(base):
                files[f"{dest}/{dep.relative_to(base)}"] = dep
    files["pytorch/LICENSE"] = pytorch / "LICENSE"
    files["cutlass/LICENSE.txt"] = cutlass / "LICENSE.txt"
    for rel in sorted(p.relative_to(shim) for p in shim.rglob("*") if p.is_file()):
        files[f"shim/{rel}"] = shim / rel

    for upstream in ("pytorch", "cutlass"):
        shutil.rmtree(ROOT / upstream, ignore_errors=True)
    for rel, src in files.items():
        if rel.startswith("shim/"):
            continue
        (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, ROOT / rel)
    manifest = {
        "purpose": "Header closure of PyTorch 2.6.0 F32 mem-efficient attention "
                   "(fmha_cutlassF_f32_aligned_64x64_rf_sm80) for the strict GGML CUDA path",
        "pytorch": {"url": "https://github.com/pytorch/pytorch", "ref": PYTORCH_REF,
                    "tree": PYTORCH_TREE, "license": "BSD-3-Clause (pytorch/LICENSE)"},
        "cutlass": {"url": f"https://github.com/NVIDIA/cutlass/archive/{CUTLASS_COMMIT}.tar.gz",
                    "commit": CUTLASS_COMMIT, "archive_sha256": CUTLASS_SHA256,
                    "license": "BSD-3-Clause (cutlass/LICENSE.txt)"},
        "shim": "Local replacements for ATen PhiloxUtils and TORCH_CHECK; dropout is never enabled",
        "files": {rel: sha256(ROOT / rel) for rel in sorted(files)},
    }
    (ROOT / "UPSTREAM.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"vendored {len(files)} files into {ROOT}")


def verify():
    manifest = json.loads((ROOT / "UPSTREAM.json").read_text())
    actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file()} - {"UPSTREAM.json"}
    expected = set(manifest["files"])
    bad = [rel for rel in sorted(expected & actual) if sha256(ROOT / rel) != manifest["files"][rel]]
    if bad or actual != expected:
        raise SystemExit(f"vendored tree mismatch: changed={bad} missing={sorted(expected - actual)} "
                         f"extra={sorted(actual - expected)}")
    print(f"verified {len(expected)} vendored files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--cache", type=pathlib.Path, default=pathlib.Path(tempfile.gettempdir()) / "ra_mem_eff_cache")
    args = parser.parse_args()
    fetch(args.cache) if args.fetch else verify()
