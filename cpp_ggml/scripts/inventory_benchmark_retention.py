#!/usr/bin/env python3
"""Index historical evidence and protect files referenced by current runs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from provenance import save, sha256


def references(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from references(child)
    elif isinstance(value, list):
        for child in value:
            yield from references(child)
    elif isinstance(value, str):
        yield value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical', type=Path, required=True)
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    historical = args.historical.resolve()
    protected = set()
    for path in args.current.rglob('run_provenance.json'):
        value = json.loads(path.read_text())
        for ref in references(value):
            if isinstance(ref, str) and ref.startswith(str(historical)+'/'):
                protected.add(ref)
    rows = []
    groups = defaultdict(lambda: defaultdict(int))
    raw_suffixes = {'.rafo', '.npz', '.f32', '.raim', '.raip'}
    for path in sorted(historical.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(historical)
        category = ('current_dependency' if str(path) in protected else
                    'archive_candidate' if path.suffix in raw_suffixes else 'retain_report')
        size = path.stat().st_size
        rows.append({'path': relative.as_posix(), 'bytes': size,
                     'sha256': sha256(path), 'category': category})
        groups[relative.parts[0]][category] += size
    save(args.output.with_suffix('.json'), {'historical': str(historical),
         'current': str(args.current.resolve()), 'files': rows,
         'policy': 'read-only inventory; archive candidates remain on disk; preserve provenance identities'})
    lines = ['# Historical Benchmark Retention', '',
             'This inventory makes no deletions. Keep every current dependency '
             'and all reports, metrics, logs, fingerprints and galleries. Raw '
             'historical tensors can be archived only with this original-path '
             'and SHA256 index. An archived run is historical; its missing local '
             'artifacts cannot continue to pass the original local validator.', '',
             '| Directory / file | Current dependencies MiB | Archivable raw MiB | Reports / other MiB |',
             '|---|---:|---:|---:|']
    for group, sizes in sorted(groups.items()):
        lines.append(f'| {group} | {sizes["current_dependency"]/2**20:.2f} | '
                     f'{sizes["archive_candidate"]/2**20:.2f} | {sizes["retain_report"]/2**20:.2f} |')
    totals = {key: sum(row['bytes'] for row in rows if row['category'] == key)
              for key in ('current_dependency', 'archive_candidate', 'retain_report')}
    lines += ['', f'Total {sum(totals.values())/2**30:.2f} GiB; '
              f'archivable raw {totals["archive_candidate"]/2**30:.2f} GiB; '
              f'current dependencies {totals["current_dependency"]/2**20:.2f} MiB. '
              'This is a storage plan, not evidence that archived runs match the current binary.', '',
              f'[Every path and SHA256]({args.output.with_suffix(".json").name}).']
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(totals))


if __name__ == '__main__':
    main()
