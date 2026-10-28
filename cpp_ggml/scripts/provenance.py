"""Build-time and execution-time identities for reproducible graph evidence."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def fingerprint(files):
    hashes = {str(path.resolve()): sha256(path) for path in sorted(set(map(Path, files)))}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {'sha256': digest, 'files': hashes}


def source_fingerprint():
    return fingerprint([ROOT/'cpp_ggml/CMakeLists.txt', Path(__file__),
                        ROOT/'relsgg/corpus_type_map.json',
                        ROOT/'cpp_ggml/third_party/clip_tokenizer.hpp',
                        *ROOT.glob('cpp_ggml/src/*.cpp'),
                        *ROOT.glob('cpp_ggml/include/relateanything/*.hpp'),
                        *ROOT.glob('cpp_ggml/include/relateanything/*.h'),
                        *ROOT.glob('cpp_ggml/third_party/patches/*.patch')])


def python_fingerprint(script):
    return fingerprint([*ROOT.glob('relsgg/**/*.py'), Path(script), Path(__file__)])


def model_assets(model):
    folder = ROOT/'cpp_ggml/models/pytorch'/model
    return [p for p in folder.iterdir() if p.is_file() and p.suffix in
            ('.pth', '.pt', '.json', '.npz', '.txt')]


def build_fingerprint(binary, backend=None):
    binary = Path(binary)
    value = json.loads(subprocess.check_output([str(binary), '--build-info'], text=True))
    if value['source'] != source_fingerprint():
        raise RuntimeError(f'binary was built from different source; rebuild {binary.parent}')
    if backend is not None and value['backend'] != backend:
        raise RuntimeError(f'binary has wrong backend: {binary}')
    validate_files(value['build_files'])
    return {**value, 'binary': str(binary.resolve()), 'binary_sha256': sha256(binary)}


def validate_files(identity):
    for path, expected in identity['files'].items():
        if not Path(path).is_file() or sha256(path) != expected:
            raise RuntimeError(f'missing or changed provenance file: {path}')


@contextmanager
def recorded_run(folder, *, inputs, config, binary=None, script=None, outputs=()):
    folder = Path(folder)
    record = {'schema': 1, 'status': 'running',
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'config': config}
    save(folder/'run_provenance.json', record)
    try:
        record['inputs'] = fingerprint(inputs)
        if binary is not None:
            record['build'] = build_fingerprint(binary)
        if script is not None:
            record['python'] = python_fingerprint(script)
        save(folder/'run_provenance.json', record)
        yield record
        validate_files(record['inputs'])
        if binary is not None:
            if record['build'] != build_fingerprint(binary):
                raise RuntimeError('binary changed during inference')
        if script is not None and record['python'] != python_fingerprint(script):
            raise RuntimeError('Python source changed during inference')
        artifacts = [p for p in folder.iterdir() if p.is_file()
                     and p.name != 'run_provenance.json' and '.tmp-' not in p.name]
        record.update(status='complete', artifacts=fingerprint([*artifacts,*outputs]))
    except BaseException as exc:
        record.update(status='failed', failure={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(folder/'run_provenance.json', record)


def validate_run(folder, *, binary=None, script=None):
    record = json.loads((Path(folder)/'run_provenance.json').read_text())
    if record['status'] != 'complete':
        raise RuntimeError(f'incomplete run: {folder}')
    validate_files(record['inputs'])
    validate_files(record['artifacts'])
    if 'python' in record:
        validate_files(record['python'])
    if binary is not None and record['build'] != build_fingerprint(binary):
        raise RuntimeError(f'run has obsolete binary/build: {folder}')
    if script is not None and record['python'] != python_fingerprint(script):
        raise RuntimeError(f'run has obsolete Python graph: {folder}')
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--header', type=Path, required=True)
    parser.add_argument('--backend', required=True)
    parser.add_argument('--compiler', required=True)
    parser.add_argument('--flags', default='')
    parser.add_argument('--build-type', required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    value = {'source': source_fingerprint(), 'backend': args.backend,
             'compiler': subprocess.check_output([args.compiler, '--version'], text=True),
             'flags': args.flags, 'build_type': args.build_type,
             'build_files': fingerprint([args.cache]),
             'ggml_commit': subprocess.check_output(['git', '-C', str(ROOT/'cpp_ggml/third_party/ggml'),
                                                     'rev-parse', 'HEAD'], text=True).strip()}
    args.header.parent.mkdir(parents=True, exist_ok=True)
    text = '#pragma once\nstatic constexpr const char * ra_build_info = R"RAJSON('
    text += json.dumps(value, sort_keys=True) + ')RAJSON";\n'
    corpus = json.loads((ROOT/'relsgg/corpus_type_map.json').read_text())
    text += 'static constexpr const char * ra_corpus_type_map = R"RACORPUS('
    text += json.dumps(corpus, sort_keys=True) + ')RACORPUS";\n'
    if not args.header.exists() or args.header.read_text() != text:
        args.header.write_text(text)


if __name__ == '__main__':
    main()
