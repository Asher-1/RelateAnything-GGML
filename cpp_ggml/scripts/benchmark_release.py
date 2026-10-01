#!/usr/bin/env python3
"""Build, verify, and publish portable benchmark report archives."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
from urllib.parse import urlparse
import zipfile

PACKAGE = "relateanything_benchmark_report"
CHUNK = 1024 * 1024


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def encoded(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def safe_relative(path: Path) -> str:
    relative = PurePosixPath(path.as_posix())
    if relative.is_absolute() or ".." in relative.parts or str(relative) in ("", "."):
        raise ValueError(f"unsafe report path: {path}")
    return str(relative)


def report_files(reports: Path) -> list[Path]:
    if not reports.is_dir() or reports.is_symlink():
        raise ValueError(f"reports must be a real directory: {reports}")
    entries = sorted(reports.rglob("*"), key=lambda path: path.relative_to(reports).as_posix())
    links = [path for path in entries if path.is_symlink()]
    if links:
        raise ValueError(f"report archive cannot contain symlinks: {links[0]}")
    return [path for path in entries if path.is_file()]


def zip_info(relative: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(f"{PACKAGE}/{relative}", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def run_record(run_dir: Path) -> dict:
    record_path = run_dir / "run.json"
    if not record_path.is_file():
        raise ValueError(f"missing run record: {record_path}")
    record = json.loads(record_path.read_text())
    run_id = record.get("run_id")
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id:
        raise ValueError("run record has an invalid run_id")
    if record.get("status") not in ("complete", "relocated"):
        raise ValueError("portable report requires a completed run")
    return record


def build_report_archive(run_dir: Path, output: Path) -> dict:
    """Create a deterministic ZIP from a completed run's ordinary report files."""
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    record = run_record(run_dir)
    reports = run_dir / "reports"
    files = report_files(reports)
    rows = [{"file": safe_relative(path.relative_to(reports)),
             "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files]
    sources = {safe_relative(path.relative_to(reports)): path for path in files}
    manifest = encoded({"schema": 1, "package_root": PACKAGE, "files": rows})
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="benchmark-report-", dir=output.parent) as temporary:
        candidate = Path(temporary) / output.name
        entries = sorted([*sources, "manifest.json"])
        with zipfile.ZipFile(candidate, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as packed:
            for relative in entries:
                payload = manifest if relative == "manifest.json" else sources[relative].read_bytes()
                packed.writestr(zip_info(relative), payload)
        candidate.replace(output)

    descriptor = {
        "schema": 1,
        "run_id": record["run_id"],
        "zip_file_name": output.name,
        "zip_bytes": output.stat().st_size,
        "zip_sha256": sha256(output),
        "report_files": len(rows),
        "report_bytes": sum(int(row["bytes"]) for row in rows),
        "source_run_json_sha256": sha256(run_dir / "run.json"),
        "package_root": PACKAGE,
    }
    verify_report_archive(output, descriptor)
    return descriptor


def validate_descriptor(descriptor: dict) -> None:
    required = {"schema", "run_id", "zip_file_name", "zip_bytes", "zip_sha256",
                "report_files", "source_run_json_sha256"}
    if not required <= descriptor.keys() or descriptor["schema"] != 1:
        raise ValueError("invalid report descriptor")
    run_id = descriptor["run_id"]
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id:
        raise ValueError("invalid descriptor run_id")
    if descriptor["zip_file_name"] != f"{run_id}.zip":
        raise ValueError("ZIP filename must match run_id")
    for field in ("zip_sha256", "source_run_json_sha256"):
        value = descriptor[field]
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"invalid {field}")
    if not isinstance(descriptor["zip_bytes"], int) or descriptor["zip_bytes"] <= 0:
        raise ValueError("invalid ZIP size")
    if not isinstance(descriptor["report_files"], int) or descriptor["report_files"] < 0:
        raise ValueError("invalid report file count")
    if "package_root" in descriptor and descriptor["package_root"] != PACKAGE:
        raise ValueError("unexpected package root")


def verify_report_archive(archive: Path, descriptor: dict) -> None:
    archive = Path(archive)
    validate_descriptor(descriptor)
    if archive.name != descriptor["zip_file_name"]:
        raise ValueError("archive name differs from descriptor")
    if not archive.is_file() or archive.stat().st_size != descriptor["zip_bytes"]:
        raise ValueError("archive size differs from descriptor")
    if sha256(archive) != descriptor["zip_sha256"]:
        raise ValueError("archive SHA-256 differs from descriptor")
    with zipfile.ZipFile(archive) as packed:
        names = packed.namelist()
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("archive member order or uniqueness is invalid")
        manifest_name = f"{PACKAGE}/manifest.json"
        if manifest_name not in names:
            raise ValueError("archive lacks manifest")
        manifest = json.loads(packed.read(manifest_name))
        rows = manifest.get("files")
        if manifest.get("schema") != 1 or manifest.get("package_root") != PACKAGE or not isinstance(rows, list):
            raise ValueError("invalid package manifest")
        expected = [f"{PACKAGE}/{row['file']}" for row in rows] + [manifest_name]
        if names != sorted(expected) or len(rows) != descriptor["report_files"]:
            raise ValueError("archive members differ from manifest")
        for row in rows:
            relative = safe_relative(Path(row["file"]))
            info = packed.getinfo(f"{PACKAGE}/{relative}")
            if (info.external_attr >> 16) & 0o170000 != 0o100000:
                raise ValueError(f"archive member is not a regular file: {relative}")
            payload = packed.read(info)
            if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
                raise ValueError(f"archive member digest differs: {relative}")


def write_current_summary(benchmarks: Path, descriptor: dict, report_url: str) -> Path:
    """Atomically publish a verified Hugging Face report pointer."""
    validate_descriptor(descriptor)
    parsed = urlparse(report_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("report URL must be HTTPS")
    expected_suffix = f"/reports/{descriptor['zip_file_name']}"
    if not parsed.path.endswith(expected_suffix):
        raise ValueError("report URL must end with the descriptor ZIP path")
    summary = Path(benchmarks) / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": 1,
        "run_id": descriptor["run_id"],
        "generated_utc": now(),
        "report_url": report_url,
        "zip_file_name": descriptor["zip_file_name"],
        "zip_bytes": descriptor["zip_bytes"],
        "zip_sha256": descriptor["zip_sha256"],
        "report_files": descriptor["report_files"],
        "source_run_json_sha256": descriptor["source_run_json_sha256"],
    }
    target = summary / "current.json"
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(encoded(result))
    temporary.replace(target)
    return target


def write_descriptor(archive: Path, descriptor: dict) -> Path:
    path = Path(archive).with_suffix(".package.json")
    path.write_bytes(encoded(descriptor))
    return path


def load_descriptor(archive: Path, descriptor: Path | None) -> dict:
    path = Path(descriptor) if descriptor else Path(archive).with_suffix(".package.json")
    if not path.is_file():
        raise ValueError(f"missing report descriptor: {path}")
    return json.loads(path.read_text())


def verify_run(run_dir: Path) -> None:
    from benchmark_layout import verify
    verify(Path(run_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    package = commands.add_parser("package", help="build a portable report ZIP")
    package.add_argument("--run", type=Path, required=True)
    package.add_argument("--output", type=Path, required=True)
    verify_package = commands.add_parser("verify-package", help="verify a report ZIP")
    verify_package.add_argument("--archive", type=Path, required=True)
    verify_package.add_argument("--descriptor", type=Path)
    verify_run_parser = commands.add_parser("verify-run", help="verify completed run evidence")
    verify_run_parser.add_argument("--run", type=Path, required=True)
    publish = commands.add_parser("publish", help="write the stable report pointer")
    publish.add_argument("--archive", type=Path, required=True)
    publish.add_argument("--descriptor", type=Path)
    publish.add_argument("--benchmarks", type=Path, default=Path(__file__).resolve().parents[1] / "benchmarks")
    publish.add_argument("--report-url", required=True)
    args = parser.parse_args()

    if args.command == "verify-run":
        verify_run(args.run)
        print(json.dumps({"status": "verified", "run": str(args.run)}))
    elif args.command == "package":
        verify_run(args.run)
        descriptor = build_report_archive(args.run, args.output)
        write_descriptor(args.output, descriptor)
        print(json.dumps({"status": "packaged", "archive": str(args.output), **descriptor}))
    elif args.command == "verify-package":
        descriptor = load_descriptor(args.archive, args.descriptor)
        verify_report_archive(args.archive, descriptor)
        print(json.dumps({"status": "verified", "archive": str(args.archive), **descriptor}))
    else:
        descriptor = load_descriptor(args.archive, args.descriptor)
        verify_report_archive(args.archive, descriptor)
        path = write_current_summary(args.benchmarks, descriptor, args.report_url)
        print(json.dumps({"status": "published", "summary": str(path), **descriptor}))


if __name__ == "__main__":
    main()
