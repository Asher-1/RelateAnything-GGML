from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import zipfile

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from benchmark_layout import set_current
from benchmark_release import (
    build_report_archive,
    verify_report_archive,
    write_current_summary,
)


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_run_alignment():
    return load_script("run_alignment")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_strict_f32_compacts_only_valid_relation_pairs():
    source = (Path(__file__).resolve().parents[1] / "src" / "full_model.cpp").read_text()
    assert "bool compact_pairs() const { return options.compact_pairs; }" in source


def test_export_gguf_writes_convnext_metadata_by_model_type():
    source = (Path(__file__).resolve().parents[1] / "scripts" / "export_gguf.py").read_text()
    assert 'model_family = str(cfg.get("model_type") or "dinov3_vit")' in source
    assert '"layer_offsets": [-4, -3, -2, -1],' in source
    assert 'writer.add_array("ra.hidden_sizes", meta["hidden_sizes"])' in source
    assert 'writer.add_array("ra.depths", meta["depths"])' in source


def test_runtime_implements_convnext_backbone_on_every_backend_path():
    source = (Path(__file__).resolve().parents[1] / "src" / "full_model.cpp").read_text()
    assert 'model_family == "dinov3_convnext"' in source
    assert "ggml_conv_2d_dw_direct" in source
    assert "GGML_SCALE_MODE_BILINEAR" in source
    # The stage fusion must keep the shared ViT combiner: pixel_unshuffle folds
    # into the r×r projection, so both families return the same fused layout.
    assert 'return backbone_convnext(run, ctx, image, taps, height, width, trace);' in source
    assert 't(p + ".depthwise_conv.bias")' in source


def test_native_argmax_check_tolerates_bit_equal_ties_only():
    source = (Path(__file__).resolve().parents[1] / "scripts" / "benchmark_full_graph.py").read_text()
    # A bit-equal top-1 tie is implementation-defined across devices; the check
    # must compare the two candidate logits before declaring a disagreement.
    assert "chosen['predicate_id'] == p['predicate_id']: continue" in source
    assert "out['logits'][i, chosen['predicate_id']] != out['logits'][i, p['predicate_id']]" in source


def test_cuda_strict_backbone_reuses_packed_qkv_projection():
    source = (Path(__file__).resolve().parents[1] / "src" / "full_model.cpp").read_text()
    assert "if (options.strict_f32) {\n                const auto projection" not in source
    assert "if (!options.strict_f32) {\n        packed_qkv.ctx" not in source
    assert "auto * qkv = linear(ctx, y, packed_qkv.weights[layer], packed_qkv.biases[layer]);" in source


def test_cuda_patch_embedding_uses_parallel_strict_f32_matmul():
    source = (Path(__file__).resolve().parents[1] / "src" / "full_model.cpp").read_text()
    patch_hint = (
        "if (options.strict_f32 && options.backend == \"cuda\" && "
        "kernel->type == GGML_TYPE_F32)\n"
        "            ggml_mul_mat_set_hint(patches, GGML_HINT_STRICT_F32);"
    )
    assert patch_hint in source


def test_optional_interface_comparison_aligns_scores_by_triplet_identity():
    optional = load_script("validate_optional_interfaces")
    reference = [
        {"subject": 0, "object": 4, "predicate": "right", "score": 0.38},
        {"subject": 0, "object": 1, "predicate": "right", "score": 0.57},
    ]
    native = [
        {"subject": 0, "object": 1, "predicate": "right", "score": 0.571},
        {"subject": 0, "object": 4, "predicate": "right", "score": 0.379},
    ]

    comparison = optional.compare_triplet_streams(reference, native)

    assert comparison["triplet_set_equal"] is True
    assert comparison["ranked_triplets_equal"] is False
    assert comparison["max_score_error"] == pytest.approx(0.001)


def make_completed_run(tmp_path: Path, files: dict[str, bytes] | None = None) -> Path:
    run = tmp_path / "run-1"
    reports = run / "reports"
    reports.mkdir(parents=True)
    for relative, content in (files or {"README.md": b"# report\n"}).items():
        path = reports / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    artifact_rows = []
    for report in reports.rglob("*"):
        if report.is_file():
            relative = report.relative_to(reports)
            artifact = run / "artifacts" / relative
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(report.read_bytes())
            artifact_rows.append({
                "path": relative.as_posix(),
                "source": (Path("artifacts") / relative).as_posix(),
                "source_sha256": digest(artifact),
                "report_sha256": digest(report),
                "bytes": report.stat().st_size,
            })
    run_json = {
        "schema": 1,
        "run_id": "run-1",
        "status": "complete",
        "historical": False,
    }
    (run / "run.json").write_text(json.dumps(run_json))
    (run / "reports.json").write_text(json.dumps({"schema": 1, "files": artifact_rows}))
    return run


def test_build_report_archive_is_portable_and_hash_pinned(tmp_path: Path):
    run = make_completed_run(tmp_path, {
        "README.md": b"# report\n",
        "full_graph/metrics.csv": b"latency_ms\n1.0\n",
    })
    archive = tmp_path / "run-1.zip"
    descriptor = build_report_archive(run, archive)

    assert descriptor["run_id"] == "run-1"
    assert descriptor["zip_file_name"] == "run-1.zip"
    assert descriptor["report_files"] == 2
    assert descriptor["zip_sha256"] == digest(archive)
    assert len(descriptor["zip_sha256"]) == 64
    verify_report_archive(archive, descriptor)

    with zipfile.ZipFile(archive) as packed:
        names = packed.namelist()
        assert names == sorted(names)
        assert all(name.startswith("relateanything_benchmark_report/") for name in names)
        assert "relateanything_benchmark_report/manifest.json" in names
        assert all((packed.getinfo(name).external_attr >> 16) & 0o170000 == 0o100000 for name in names)


def test_build_report_archive_rejects_symlink(tmp_path: Path):
    run = make_completed_run(tmp_path)
    (run / "reports" / "alias.md").symlink_to("README.md")

    with pytest.raises(ValueError, match="symlinks"):
        build_report_archive(run, tmp_path / "report.zip")


def test_set_current_does_not_fail_when_symlink_creation_is_unavailable(monkeypatch, tmp_path: Path):
    run = make_completed_run(tmp_path)
    link = tmp_path / "current"

    def denied(*args, **kwargs):
        raise OSError("symlink privilege unavailable")

    monkeypatch.setattr(Path, "symlink_to", denied)
    assert set_current(run, link=link) is None
    assert not link.exists()


def test_write_current_summary_uses_huggingface_report_url(tmp_path: Path):
    descriptor = {
        "schema": 1,
        "run_id": "run-1",
        "zip_file_name": "run-1.zip",
        "zip_bytes": 123,
        "zip_sha256": "a" * 64,
        "report_files": 2,
        "source_run_json_sha256": "b" * 64,
    }
    path = write_current_summary(
        tmp_path,
        descriptor,
        "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/reports/run-1.zip",
    )

    assert path == tmp_path / "summary" / "current.json"
    data = json.loads(path.read_text())
    assert data["report_url"].endswith("/reports/run-1.zip")
    assert data["run_id"] == "run-1"
    assert data["zip_sha256"] == "a" * 64


def tracked_documents() -> list[Path]:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "ls-files", "cpp_ggml"], cwd=root, check=True,
        text=True, capture_output=True,
    )
    return [root / line for line in result.stdout.splitlines()
            if Path(line).suffix in {".md", ".html"}]


def test_tracked_benchmark_documents_do_not_link_to_local_current():
    offenders = [path.relative_to(Path(__file__).resolve().parents[2])
                 for path in tracked_documents() if "current/" in path.read_text(errors="ignore")]
    assert offenders == []


def test_run_alignment_defaults_to_auto_precision_without_force_flag():
    alignment = load_run_alignment()

    args = alignment.parse_args([])

    assert args.precision == "auto"
    assert alignment.parse_args(["--strict-f32"]).precision == "on"
    assert alignment.parse_args(["--no-strict-f32"]).precision == "off"
    assert alignment.precision_arguments(args.precision) == []
    assert alignment.precision_arguments("on") == ["--strict-f32"]
    assert alignment.precision_arguments("off") == ["--no-strict-f32"]


def test_run_alignment_resume_skips_complete_and_preserves_failed_attempt():
    alignment = load_run_alignment()
    record = {"steps": [
        {"name": "full-primary", "command": ["python", "full.py"],
         "status": "complete", "returncode": 0},
        {"name": "optional-interfaces", "command": ["python", "interfaces.py"],
         "status": "failed", "returncode": 1},
    ]}

    assert alignment.prepare_step(record, "full-primary", ["python", "full.py"], True) is None
    entry = alignment.prepare_step(
        record, "optional-interfaces", ["python", "interfaces.py"], True
    )

    assert len(record["steps"]) == 2
    assert entry["status"] == "running"
    assert entry["attempts"] == [{
        "name": "optional-interfaces", "command": ["python", "interfaces.py"],
        "status": "failed", "returncode": 1,
    }]
    assert alignment.parse_args(["--resume", "--run-id", "release"]).resume is True


def write_acceptance_run(tmp_path: Path, slow_ratio: float, drift: float = 0.01,
                         per_image: bool = True) -> Path:
    """A complete 63-row run where every GPU row is faster than PyTorch except
    relsgg-vitb16/cuda-f32, which is slower by `slow_ratio` on every image."""
    run = tmp_path / "release-run"
    full = run / "artifacts" / "full_graph"
    full.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"status": "complete", "config": {"precision_request": "auto"}}))
    samples = [f"{index:03d}" for index in range(50)]
    models = {}
    for model in ("relsgg-vits16", "relsgg-vits16plus", "relsgg-vitb16",
                  "relsgg-convnext-tiny", "relsgg-convnext-small",
                  "relsgg-convnext-base", "relsgg-convnext-large"):
        configs = {"python": {"latency_median_ms": 100.0}}
        if per_image:
            folder = full / model / "python"
            folder.mkdir(parents=True)
            (folder / "results.json").write_text(json.dumps({
                "timing_bracket": {"relative_drift": drift},
                "rows": [{"sample": sample, "latency_ms": 100.0 + index % 7}
                         for index, sample in enumerate(samples)]}))
        for backend in ("cpu", "cuda", "vulkan"):
            for dtype in ("f32", "f16", "q8_0"):
                key = f"{backend}-{dtype}"
                ratio = slow_ratio if model == "relsgg-vitb16" and key == "cuda-f32" else 0.8
                configs[key] = {
                    "backend": backend, "dtype": dtype, "latency_median_ms": 100.0 * ratio,
                    "logits_rmse": 0.001, "predicate_top1_agreement": 1.0,
                    "pair_set_jaccard": 1.0,
                    "strict_f32_request": "auto",
                    "strict_f32": backend == "cuda" and dtype == "f32",
                }
                if per_image:
                    folder = full / model / key
                    folder.mkdir(parents=True)
                    (folder / "results.json").write_text(json.dumps(dict(configs[key], rows=[
                        {"sample": sample, "latency_ms": (100.0 + index % 7) * ratio}
                        for index, sample in enumerate(samples)])))
        configs["cuda-q8_0-f32-input"] = dict(configs["cuda-q8_0"], quant_f32_input=True)
        models[model] = configs
    (full / "summary.json").write_text(json.dumps({"models": models}))
    return run


def g2_report(run: Path) -> tuple[dict, dict]:
    report = load_script("write_acceptance_status").acceptance(run)
    statuses = {row["id"]: row for row in report["statuses"]}
    return report, statuses


def test_acceptance_tracks_a_real_gap_below_materiality_without_an_accept_list(tmp_path: Path):
    report, statuses = g2_report(write_acceptance_run(tmp_path, slow_ratio=1.01))
    g2 = statuses["g2_gpu_performance"]

    assert statuses["g1_numeric"]["status"] == "complete"
    assert statuses["g1_numeric"]["passed"] == 63
    assert statuses["auto_precision"]["status"] == "complete"
    assert g2["passed"] == 41
    # The raw gate result is preserved; classification only adds evidence.
    assert g2["failed"] == [{
        "model": "relsgg-vitb16", "config": "cuda-f32",
        "native_ms": 101.0, "python_ms": 100.0,
    }]
    assert [row["verdict"] for row in g2["classified"]] == ["tracked"]
    assert g2["classified"][0]["paired_ratio"] == pytest.approx(1.01)
    assert g2["status"] == "accepted_deviation"
    assert report["release_decision"] == "accepted_with_performance_deviations"


def test_acceptance_blocks_a_gap_confidently_above_materiality(tmp_path: Path):
    report, statuses = g2_report(write_acceptance_run(tmp_path, slow_ratio=1.10))
    g2 = statuses["g2_gpu_performance"]

    assert [row["verdict"] for row in g2["classified"]] == ["block"]
    assert g2["classified"][0]["ratio_ci_lower"] > 1.05
    assert g2["status"] == "open"
    assert report["release_decision"] == "blocked"


def test_acceptance_requires_a_rerun_when_the_reference_drifts_beyond_materiality(tmp_path: Path):
    report, statuses = g2_report(write_acceptance_run(tmp_path, slow_ratio=1.01, drift=0.20))
    g2 = statuses["g2_gpu_performance"]

    # A small measured gap is not a pass when the reference moved by 20%.
    assert [row["verdict"] for row in g2["classified"]] == ["inconclusive"]
    assert g2["status"] == "rerun_required"
    assert report["release_decision"] == "rerun_required"


def test_acceptance_without_per_image_evidence_cannot_accept_a_failure(tmp_path: Path):
    report, statuses = g2_report(write_acceptance_run(tmp_path, slow_ratio=1.01, per_image=False))
    g2 = statuses["g2_gpu_performance"]

    assert g2["classified"][0]["verdict"] == "inconclusive"
    assert g2["classified"][0]["reason"] == "missing per-image results"
    assert report["release_decision"] == "rerun_required"


def test_release_status_is_authoritative_and_legacy_paths_are_redirects():
    root = Path(__file__).resolve().parents[1]
    status = root / "benchmarks" / "RELEASE_STATUS.md"
    assert status.is_file()
    for name in ("ALIGNMENT_STATUS.md", "NEXT_WINDOW_TODOS.md"):
        text = (root / "benchmarks" / name).read_text()
        assert "RELEASE_STATUS.md" in text
        assert len(text.splitlines()) <= 6


def test_model_cards_describe_the_auto_precision_matrix():
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "models" / "MODEL_CARDS.md",
        root / "models" / "HF_MODEL_CARD.md",
        root / "scripts" / "update_model_cards.py",
    ]
    for path in paths:
        text = path.read_text()
        assert "All current native timing rows use strict F32 precision" not in text
        assert "CUDA F32" in text
        assert "CPU, Vulkan, F16 and Q8_0" in text


def test_report_index_uses_release_acceptance_vocabulary():
    text = (SCRIPTS / "benchmark_layout.py").read_text()
    assert "Seven acceptance conditions" not in text
    assert "Release acceptance status" in text


def test_write_current_summary_rejects_invalid_descriptor(tmp_path: Path):
    descriptor = {
        "schema": 1,
        "run_id": "run-1",
        "zip_file_name": "wrong.zip",
        "zip_bytes": 0,
        "zip_sha256": "not-a-hash",
        "report_files": 0,
        "source_run_json_sha256": "b" * 64,
    }
    with pytest.raises(ValueError):
        write_current_summary(tmp_path, descriptor, "http://example.invalid/report.zip")
