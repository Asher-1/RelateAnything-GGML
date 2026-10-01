#!/usr/bin/env python3
"""Immutable run allocation, reversible evidence relocation, and report views."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil

from provenance import ROOT, save, sha256

OUTPUTS = ROOT / 'cpp_ggml/outputs'
LINK = re.compile(r'(!?\[[^\]]*\]\()([^\s)]+)(\))')
RUN_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z')


def now():
    return datetime.now(timezone.utc).isoformat()


def allocate(run_id=None, *, root=OUTPUTS, archive=False):
    run_id = run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-alignment')
    if not RUN_ID.fullmatch(run_id):
        raise ValueError('run ID must be a simple filename of at most 96 characters')
    folder = Path(root) / ('archive' if archive else 'runs') / run_id
    folder.mkdir(parents=True, exist_ok=False)
    save(folder/'run.json', {'schema': 1, 'run_id': run_id, 'created_utc': now(),
                            'status': 'allocated', 'historical': archive})
    (folder/'reports').mkdir()
    return folder


def inventory(folder):
    rows = []
    for path in sorted(folder.rglob('*')):
        if path.is_file():
            rows.append({'path': path.relative_to(folder).as_posix(),
                         'bytes': path.stat().st_size, 'sha256': sha256(path),
                         'symlink': os.readlink(path) if path.is_symlink() else None})
        elif path.is_symlink() and not path.exists():
            raise RuntimeError(f'broken evidence link: {path}')
    return rows


def verify_inventory(folder, rows):
    for row in rows:
        path = folder/row['path']
        if not path.is_file() or path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            raise RuntimeError(f'missing or changed relocated evidence: {path}')
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    if actual != {row['path'] for row in rows}:
        raise RuntimeError('evidence file set changed during relocation')


def report_file(path):
    if 'hf_package' in path.parts:
        return False
    if path.suffix in ('.md', '.csv', '.png', '.svg', '.html'):
        return True
    # Individual image outputs belong to the execution evidence, not the report view.
    return path.suffix == '.json' and not re.match(r'\d{3}[_\.]', path.name)


def publish_reports(folder):
    folder = Path(folder).resolve()
    artifacts, reports = folder/'artifacts', folder/'reports'
    if not artifacts.is_dir():
        raise RuntimeError(f'no artifacts in {folder}')
    sources = [p for p in artifacts.rglob('*') if p.is_file() and report_file(p.relative_to(artifacts))]
    selected = {p.resolve(): reports/p.relative_to(artifacts) for p in sources}
    relocation = folder/'relocation.json'
    legacy = Path(json.loads(relocation.read_text())['source']) if relocation.exists() else artifacts
    rows = []
    for source in sorted(sources):
        destination = reports/source.relative_to(artifacts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + f'.tmp-{os.getpid()}')
        if source.suffix == '.md':
            def relocate(match):
                link = match[2]
                if link.startswith(('#', 'http:', 'https:', 'mailto:', 'data:')):
                    return match[0]
                filename, separator, fragment = link.partition('#')
                original = legacy/source.relative_to(artifacts)
                target = Path(os.path.normpath(original.parent/filename)).resolve()
                if not target.exists():
                    return match[0]
                mapped = selected.get(target, target)
                relative = Path(os.path.relpath(mapped, destination.parent)).as_posix()
                return match[1]+relative+(separator+fragment if separator else '')+match[3]
            temporary.write_text(LINK.sub(relocate, source.read_text()))
        else:
            shutil.copyfile(source, temporary)
        temporary.replace(destination)
        rows.append({'path': destination.relative_to(reports).as_posix(),
                     'source': source.relative_to(folder).as_posix(),
                     'source_sha256': sha256(source), 'report_sha256': sha256(destination),
                     'bytes': destination.stat().st_size})
    if not (artifacts/'README.md').exists():
        record = json.loads((folder/'run.json').read_text())
        lines = [f'# Alignment run {record["run_id"]}', '',
                 f'Execution status: **{record["status"]}**. '
                 '[Run record](../run.json). Raw execution artifacts and command logs '
                 'remain in the sibling `artifacts/` and `logs/` directories.', '',
                 'This is a derived report view. Numerical, performance and protocol '
                 'acceptance are reported separately from execution completion.', '']
        entries = [('gallery/index.html', 'Offline all-model/backend/storage gallery'),
                   ('runtime_options_audit.json', 'Compiled environment-switch audit'),
                   ('acceptance_status.json', 'Seven acceptance conditions'),
                   ('full_graph/README.md', 'Full 50-image benchmark and galleries'),
                   ('stages/summary.json', 'Stage and dynamic-vocabulary comparisons'),
                   ('dense/results.json', 'Dense sampler comparisons'),
                   ('dense_candidate_audit/summary.json', 'Fixed-candidate error decomposition'),
                   ('interfaces/results.json', 'Optional interfaces and native text'),
                   ('metric_subsets/summary.json', 'Public SGCls subset regressions'),
                   ('spatialsense/summary.json', 'SpatialSense binary subset regressions'),
                   ('ggml_patch_audit.json', 'Exact patch tree audit'),
                   ('patch_replay.json', 'Pristine patch replay')]
        lines += [f'- [{label}]({path})' for path, label in entries if (reports/path).exists()]
        destination = reports/'README.md'
        destination.write_text('\n'.join(lines)+'\n')
        rows.append({'path': 'README.md', 'source': 'run.json',
                     'source_sha256': sha256(folder/'run.json'),
                     'report_sha256': sha256(destination), 'bytes': destination.stat().st_size})
    save(folder/'reports.json', {'schema': 1, 'created_utc': now(), 'files': rows,
                                'scope': 'derived view; execution provenance remains in artifacts'})
    return rows


def verify(folder):
    folder = Path(folder).resolve()
    record = json.loads((folder/'run.json').read_text())
    if record.get('status') not in ('complete', 'relocated'):
        raise RuntimeError('cannot verify an incomplete run')
    if any(step.get('status') != 'complete' or step.get('returncode') != 0
           for step in record.get('steps', [])):
        raise RuntimeError('run contains an incomplete or failed step')
    relocated = folder/'relocation.json'
    relocated_files = 0
    if relocated.exists():
        identity = json.loads(relocated.read_text())
        if identity.get('status') != 'complete':
            raise RuntimeError('evidence relocation did not complete')
        verify_inventory(folder/'artifacts', identity['files'])
        relocated_files = len(identity['files'])
    rows = json.loads((folder/'reports.json').read_text())['files']
    for row in rows:
        source = folder/row.get('source', 'artifacts/'+row['path'])
        if not source.is_file() or sha256(source) != row['source_sha256']:
            raise RuntimeError(f'changed report source: {source}')
        report = folder/'reports'/row['path']
        if not report.is_file() or sha256(report) != row['report_sha256']:
            raise RuntimeError(f'changed report: {report}')
    return {'status': 'complete', 'relocated_files': relocated_files,
            'report_files': len(rows),
            'scope': 'report/source hashes and relocated inventory; inference audits are separate'}


def migrate(source, run_id, *, root=OUTPUTS, archive=False, apply=False):
    source = Path(source).absolute()
    destination = Path(root)/('archive' if archive else 'runs')/run_id
    if not RUN_ID.fullmatch(run_id):
        raise ValueError('invalid run ID')
    if source.is_symlink():
        raise RuntimeError(f'already relocated or symbolic source: {source}')
    if not source.is_dir() or destination.exists():
        raise RuntimeError('source must exist and destination must be new')
    if source == Path(root).resolve() or source in Path(root).resolve().parents:
        raise RuntimeError('destination cannot be inside source')
    plan = {'source': str(source), 'destination': str(destination/'artifacts'),
            'compatibility_link': str(source), 'delete_files': False}
    if not apply:
        return plan
    folder = allocate(run_id, root=root, archive=archive)
    if source.stat().st_dev != folder.stat().st_dev:
        raise RuntimeError('relocation requires the same filesystem; no implicit large copy')
    rows = inventory(source)
    record = {**plan, 'schema': 1, 'status': 'prepared', 'started_utc': now(),
              'files': rows, 'bytes': sum(row['bytes'] for row in rows)}
    save(folder/'relocation.json', record)
    source.rename(folder/'artifacts')
    try:
        source.symlink_to(os.path.relpath(folder/'artifacts', source.parent), target_is_directory=True)
    except BaseException:
        (folder/'artifacts').rename(source)
        raise
    try:
        verify_inventory(source, rows)
        run = json.loads((folder/'run.json').read_text())
        save(folder/'run.json', {**run, 'status': 'relocated', 'legacy_path': str(source),
                                'scope': 'existing execution identities preserved; no new inference claim'})
        # Generated README records run.json's hash; finalize the run metadata
        # before publishing, so a source without README also verifies correctly.
        reports = publish_reports(folder)
        record.update(status='complete', finished_utc=now(), report_files=len(reports))
    except BaseException as exc:
        record.update(status='failed', failure=str(exc))
        run = json.loads((folder/'run.json').read_text())
        save(folder/'run.json', {**run, 'status': 'failed', 'failure': str(exc)})
        raise
    finally:
        save(folder/'relocation.json', record)
    return {**plan, 'status': record['status'], 'files': len(rows), 'bytes': record['bytes']}


def set_current(folder, *, link=ROOT/'cpp_ggml/benchmarks/current'):
    folder, link = Path(folder).resolve(), Path(link)
    record = json.loads((folder/'run.json').read_text())
    if record['status'] not in ('complete', 'relocated') or record.get('historical'):
        raise RuntimeError('current view requires a completed nonhistorical run')
    config = record.get('config')
    if config and (set(config.get('suites', [])) !=
                   {'full', 'stages', 'dense', 'interfaces', 'contracts', 'metrics'} or
                   set(config.get('backends', [])) != {'cpu', 'cuda', 'vulkan'}):
        raise RuntimeError('partial execution cannot replace the full current report')
    if config and not (folder/'artifacts/acceptance_status.json').is_file():
        raise RuntimeError('current report requires the separate acceptance status')
    if link.exists() and not link.is_symlink():
        raise RuntimeError(f'refusing to replace a real directory: {link}')
    verify(folder)
    temporary = link.with_name(link.name+f'.tmp-{os.getpid()}')
    try:
        temporary.symlink_to(os.path.relpath(folder/'reports', link.parent), target_is_directory=True)
        temporary.replace(link)
    except OSError:
        if temporary.is_symlink():
            temporary.unlink()
        return None
    return link


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    move = sub.add_parser('migrate')
    move.add_argument('--source', type=Path, required=True)
    move.add_argument('--run-id', required=True)
    move.add_argument('--root', type=Path, default=OUTPUTS)
    move.add_argument('--archive', action='store_true')
    move.add_argument('--apply', action='store_true')
    for name in ('reports', 'verify', 'current'):
        child = sub.add_parser(name)
        child.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'migrate':
        text = migrate(args.source, args.run_id, root=args.root, archive=args.archive, apply=args.apply)
    elif args.command == 'reports':
        text = {'files': len(publish_reports(args.run))}
    elif args.command == 'verify':
        text = verify(args.run)
    else:
        set_current(args.run)
        text = {'current': str(args.run/'reports')}
    print(json.dumps(text, indent=2))


if __name__ == '__main__':
    main()
