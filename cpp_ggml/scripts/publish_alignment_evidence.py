#!/usr/bin/env python3
"""Prepare and optionally publish audited compact alignment evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import posixpath
import re

from provenance import ROOT, save, sha256, validate_run
from benchmark_layout import verify


def git_blob(path):
    data = path.read_bytes()
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def main():
    from huggingface_hub import CommitOperationAdd, HfApi
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--repo-id', default='Asher-1/relateanything-ggml-full')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--prepare-only', action='store_true',
                        help='build and check the compact package without contacting the remote repository')
    args = parser.parse_args()
    benchmark = args.benchmark.resolve()
    audit = json.loads((benchmark/'release_audit.json').read_text())
    if audit.get('status') != 'complete' or not audit.get('all_complete'):
        raise SystemExit('publication requires a complete current execution audit')
    if audit['primary_ggml_image_configurations'] != 1350 or audit['diagnostic_ggml_image_configurations'] != 150:
        raise SystemExit('publication requires all primary and diagnostic runs')
    for check in audit['checks']:
        run = benchmark/check['model']/(check['backend']+'-'+check['dtype']+check['variant'])
        if sha256(run/'run_provenance.json') != check['provenance_sha256']:
            raise SystemExit('execution record changed after audit: '+str(run))
        validate_run(run, binary=ROOT/f'cpp_ggml/build-{check["backend"]}/relateanything-ggml')
    for model in {check['model'] for check in audit['checks']}:
        validate_run(benchmark/model/'python', script=ROOT/'cpp_ggml/scripts/benchmark_full_graph.py')
    folder = ROOT/'cpp_ggml/benchmarks'
    paths = {}
    for name in ('README.md', 'RELEASE_STATUS.md', 'ALIGNMENT_STATUS.md', 'NEXT_WINDOW_TODOS.md',
                 'METRIC_COVERAGE.md', 'EXPORT_STATUS.md', 'PORTING_NOTES.md',
                 'RA4M_50.md', 'OFFICIAL_DATA.md', 'full_gguf_manifest.json',
                 'official_data_manifest.json', 'cleanup_manifest.json'):
        path = folder/name
        paths['benchmarks/'+name] = path
    immutable_run = benchmark.parent.parent if benchmark.parent.name == 'artifacts' else None
    if not immutable_run or not (immutable_run/'run.json').exists():
        raise SystemExit('--benchmark must name a completed immutable run/artifacts/full_graph directory')
    verify(immutable_run)
    for path in (immutable_run/'reports').rglob('*'):
        if path.is_file():
            paths[path.relative_to(ROOT/'cpp_ggml').as_posix()] = path
    for name in ('run.json', 'reports.json'):
        path = immutable_run/name
        paths[path.relative_to(ROOT/'cpp_ggml').as_posix()] = path
    paths['README.md'] = ROOT/'cpp_ggml/models/HF_MODEL_CARD.md'
    paths['deploy/release_manifest.json'] = ROOT/'deploy/release_manifest.json'
    for path in (ROOT/'cpp_ggml/models/gguf').glob('relsgg-*-text-student-f32.*'):
        paths['gguf/'+path.name] = path
        if path.suffix == '.gguf':
            verification = json.loads(path.with_suffix('.json').read_text())
            if verification['gguf_sha256'] != sha256(path):
                raise SystemExit('text-student export hash mismatch: '+path.name)
    package = args.output.parent/'hf_package'
    pending = list(paths.items())
    seen = set()
    while pending:
        remote, path = pending.pop()
        if remote in seen:
            continue
        seen.add(remote)
        if path.suffix != '.md':
            continue
        text = path.read_text()
        def rewrite(match):
            link = match.group(1)
            if link.startswith(('http:', 'https:', '#', 'mailto:')):
                return match.group(0)
            target, separator, anchor = link.partition('#')
            base = ROOT/'cpp_ggml' if remote == 'README.md' else path.parent
            source = (base/target).resolve()
            if source.is_dir() and (source/'README.md').is_file():
                source = source/'README.md'
            # Execution links use the corresponding compact report whenever possible.
            for parent in source.parents:
                if parent.name == 'artifacts' and (parent.parent/'reports').is_dir():
                    view = parent.parent/'reports'/source.relative_to(parent)
                    if view.is_file():
                        source = view
                    break
            if not source.is_file():
                raise RuntimeError('missing publication link: '+str(source))
            if source.suffix in ('.npz', '.npy', '.f32', '.raim', '.raoo', '.raio'):
                raise RuntimeError('raw tensor/input link cannot enter a compact package: '+str(source))
            if source == ROOT/'cpp_ggml/models/HF_MODEL_CARD.md':
                destination = 'README.md'
            else:
                local = source.relative_to(ROOT).as_posix()
                destination = local.removeprefix('cpp_ggml/')
            if destination not in paths:
                if source == args.output.resolve():
                    snapshot = package/destination
                    snapshot.parent.mkdir(parents=True, exist_ok=True)
                    snapshot.write_bytes(source.read_bytes())
                    paths[destination] = snapshot
                else:
                    paths[destination] = source
                pending.append((destination, source))
            new_link = posixpath.relpath(destination, posixpath.dirname(remote) or '.')
            if separator:
                new_link += '#'+anchor
            return ']('+new_link+')'
        text = re.sub(r'\]\(([^)]+)\)', rewrite, text)
        generated = package/remote
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_text(text)
        paths[remote] = generated
    if 'test_data/index.html' in paths:
        # The offline preview embeds its catalog but references original images.
        # Markdown link traversal alone does not discover these HTML resources.
        inputs = ROOT/'cpp_ggml/test_data'
        catalog = json.loads((inputs/'catalog.json').read_text())
        for sample in catalog['samples']:
            for key in ('image', 'annotation_manifest'):
                source = (inputs/sample[key]).resolve()
                if not source.is_file() or inputs.resolve() not in source.parents:
                    raise RuntimeError('missing or unsafe test preview resource: '+str(source))
                paths['test_data/'+source.relative_to(inputs).as_posix()] = source
        for name in ('catalog.json', 'package.json'):
            paths['test_data/'+name] = inputs/name
        archive = ROOT/'cpp_ggml/outputs/packages/relateanything_test_data.zip'
        from package_test_data import verify as verify_test_package
        descriptor = json.loads((inputs/'package.json').read_text())
        verify_test_package(archive, descriptor)
        paths['outputs/packages/'+archive.name] = archive
    files = [{'remote': remote, 'local': str(path), 'bytes': path.stat().st_size,
              'sha256': sha256(path), 'git_blob': git_blob(path)}
             for remote, path in sorted(paths.items())]
    manifest = {'repo': args.repo_id, 'status': 'prepared', 'files': files,
                'bytes': sum(row['bytes'] for row in files),
                'scope': 'compact reports, offline test preview resources, pinned test ZIP and three text-student models; historical raw tensors and unchanged image GGUFs are excluded',
                'release_audit_sha256': sha256(benchmark/'release_audit.json')}
    if args.prepare_only:
        if args.upload:
            raise SystemExit('--prepare-only and --upload are mutually exclusive')
        save(args.output, manifest)
        print('Prepared', len(files), 'files,', round(manifest['bytes']/2**20, 2), 'MiB', flush=True)
        return
    api = HfApi()
    before = api.model_info(args.repo_id, files_metadata=True)
    manifest['parent_revision'] = before.sha
    existing = {row.rfilename: row for row in before.siblings}
    changed = []
    for row in files:
        remote = existing.get(row['remote'])
        actual = (remote.lfs.sha256 if remote.lfs else remote.blob_id) if remote else None
        if not remote or remote.size != row['bytes'] or actual != (row['sha256'] if remote.lfs else row['git_blob']):
            changed.append(row)
    manifest['changed_files'] = len(changed)
    save(args.output, manifest)
    print('Prepared', len(files), 'files,', round(manifest['bytes']/2**20, 2), 'MiB', flush=True)
    if not args.upload:
        return
    try:
        manifest['status'] = 'uploading'
        save(args.output, manifest)
        commit = api.create_commit(repo_id=args.repo_id, repo_type='model',
            parent_commit=before.sha,
            operations=[CommitOperationAdd(path_in_repo=row['remote'], path_or_fileobj=row['local'])
                        for row in changed],
            commit_message='Publish audited strict precision runs and native optional interfaces',
            commit_description='27 stage/dynamic and dense configurations, 1500 native image configurations, 30 detailed numerical galleries, independent references, error decomposition and bounded public subset regressions. Speed and wider protocol gaps remain explicit.')
        info = api.model_info(args.repo_id, revision=commit.oid, files_metadata=True)
        remote_files = {row.rfilename: row for row in info.siblings}
        for row in files:
            remote = remote_files[row['remote']]
            if remote.size != row['bytes']:
                raise RuntimeError('remote size mismatch: '+row['remote'])
            expected = row['sha256'] if remote.lfs else row['git_blob']
            actual = remote.lfs.sha256 if remote.lfs else remote.blob_id
            if actual != expected:
                raise RuntimeError('remote hash mismatch: '+row['remote'])
            row['verified'] = True
        retained = []
        official = json.loads((folder/'full_gguf_manifest.json').read_text())['models']
        for model in official:
            name = 'gguf/'+Path(model['gguf']).name
            remote = remote_files[name]
            if not remote.lfs or remote.lfs.sha256 != model['gguf_sha256'] or remote.size != model['gguf_bytes']:
                raise RuntimeError('retained image weights mismatch: '+name)
            retained.append({'file': name, 'sha256': remote.lfs.sha256,
                             'bytes': remote.size, 'verified': True})
        manifest.update(status='complete', revision=commit.oid, commit_url=commit.commit_url,
                        retained_image_models=retained,
                        published_comparison_grids=len(list((benchmark/'visualizations').rglob('*.png'))))
        save(args.output, manifest)
        print('Verified remote revision', commit.oid, flush=True)
    except BaseException as exc:
        manifest.update(status='failed', failure={'type': type(exc).__name__, 'message': str(exc)})
        save(args.output, manifest)
        raise


if __name__ == '__main__':
    main()
