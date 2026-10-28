#!/usr/bin/env python3
"""Build and verify the portable, labelled ACloudViewer test-data archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

DATA = Path(__file__).resolve().parents[1] / 'test_data'
PACKAGE = 'relateanything_test_data'
CHUNK = 1024 * 1024
SUITES = (
    'ra4m_50', 'metric_subsets/PSG', 'metric_subsets/IndoorVG',
    'metric_subsets/VG150', 'metric_subsets/spatialsense/SpatialSense-valid',
    'metric_subsets/spatialsense/SpatialSense-test',
)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def digest_file(path):
    with path.open('rb') as stream:
        return digest_stream(stream)[0]


def digest_stream(stream):
    digest, size = hashlib.sha256(), 0
    while block := stream.read(CHUNK):
        digest.update(block)
        size += len(block)
    return digest.hexdigest(), size


def catalog(source):
    """Normalize only input fields; retain the original annotation manifests."""
    samples, suites = [], []
    for relative in SUITES:
        folder = source / relative
        manifest = json.loads((folder / 'manifest.json').read_text())
        categories = {}
        if (folder / 'categories.json').exists():
            categories = {row['id']: row['name'] for row in
                          json.loads((folder / 'categories.json').read_text()).get('categories', [])}
        suites.append({'id': relative, 'samples': len(manifest['samples']),
                       'manifest': relative + '/manifest.json',
                       'source': manifest.get('dataset', manifest.get('repo')),
                       'revision': manifest['revision'], 'split': manifest['split']})
        for index, sample in enumerate(manifest['samples']):
            image = (folder / sample['image']).resolve()
            image_relative = image.relative_to(source.resolve()).as_posix()
            if digest_file(image) != sample['image_sha256']:
                raise ValueError('input image hash mismatch: ' + image_relative)
            objects = []
            for obj in sample['objects']:
                x, y, width, height = obj['bbox']
                name = obj.get('category', obj.get('name', categories.get(obj.get('category_id'))))
                if not name or width <= 0 or height <= 0:
                    raise ValueError('invalid object in ' + image_relative)
                objects.append({'id': obj['id'], 'name': name,
                                'box_xyxy': [x, y, x + width, y + height]})
            if len(objects) > 60:
                raise ValueError('sample exceeds the native box contract: ' + image_relative)
            relations = sample.get('relations', [])
            cells = sample.get('cells', [])
            for row in relations + cells:
                if not (0 <= row['subject'] < len(objects) and 0 <= row['object'] < len(objects)):
                    raise ValueError('invalid annotation object index: ' + image_relative)
            samples.append({'suite': relative, 'image_id': sample['image_id'],
                            'inference_ready': len(objects) >= 2,
                            'image': image_relative, 'image_sha256': sample['image_sha256'],
                            'annotation_manifest': relative + '/manifest.json',
                            'annotation_sample_index': index, 'objects': objects,
                            'boxes_xyxy': [obj['box_xyxy'] for obj in objects],
                            'object_names': [obj['name'] for obj in objects],
                            'relations': relations, 'labelled_cells': cells})
    return {'schema': 1, 'box_format': 'pixel_xyxy', 'relation_indices': 'object array indices',
            'predicate_vocabulary_policy': 'model_default', 'default_sample': 0,
            'suites': suites, 'samples': samples,
            'scope': 'bounded labelled test subsets; original dataset terms apply; '
                     'unlabelled relations are not negatives; missing full protocols are not supplied'}


def zip_info(relative):
    info = zipfile.ZipInfo(PACKAGE + '/' + relative, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build(source, output):
    """O(total bytes); file compression and hashing use a 1 MiB buffer."""
    source, output = source.resolve(), output.resolve()
    data = catalog(source)
    template = Path(__file__).with_name('test_data_gallery.html').read_text()
    html = template.replace('__TEST_DATA__', json.dumps(data, ensure_ascii=False)
                            .replace('<', '\\u003c').replace('>', '\\u003e')).encode()
    generated = {'catalog.json': encoded(data), 'index.html': html,
                 'README.md': ('# RelateAnything test data\n\n'
                               'Open [index.html](index.html) to view images, object boxes and labels.\n'
                               '[catalog.json](catalog.json) provides image paths, pixel XYXY boxes, '\
                               'object names and original annotation references for automatic input filling.\n\n'
                               'This archive contains bounded RA-4M, PSG, IndoorVG, VG150 and '\
                               'SpatialSense subsets, plus source metadata. Original dataset terms apply. '\
                               'Unlabelled relations are not negatives. It supplies no model weights or '\
                               'missing full benchmark protocol assets. Records with fewer than two '\
                               'annotated objects have inference_ready=false and remain visible.\n\n'
                               'manifest.json pins every payload file with its size and SHA-256. '\
                               'The external .package.json pins the complete ZIP for download caching.\n').encode()}
    inputs = sorted(p for name in ('ra4m_50', 'metric_subsets', 'official')
                    for p in (source / name).rglob('*')
                    if p.is_file() and not any(part.startswith('.')
                                              for part in p.relative_to(source).parts))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='test-data-', dir=output.parent) as temporary:
        archive, rows = Path(temporary) / output.name, []
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as packed:
            for path in inputs:
                if path.is_symlink():
                    raise ValueError('test archive cannot contain symlinks: ' + str(path))
                relative = path.relative_to(source).as_posix()
                digest, size = hashlib.sha256(), 0
                with path.open('rb') as stream, packed.open(zip_info(relative), 'w') as target:
                    while block := stream.read(CHUNK):
                        target.write(block)
                        digest.update(block)
                        size += len(block)
                rows.append({'file': relative, 'bytes': size, 'sha256': digest.hexdigest()})
            for relative, content in sorted(generated.items()):
                packed.writestr(zip_info(relative), content)
                rows.append({'file': relative, 'bytes': len(content),
                             'sha256': hashlib.sha256(content).hexdigest()})
            packed.writestr(zip_info('manifest.json'), encoded({'schema': 1, 'files': rows}))
        descriptor = {'schema': 1, 'zip_file_name': output.name, 'extract_dir_name': PACKAGE,
                      'zip_bytes': archive.stat().st_size, 'zip_sha256': digest_file(archive),
                      'download_url': None, 'publication_status': 'local_archive_ready',
                      'cache_relative_path': 'download/' + output.name,
                      'extract_relative_path': 'extract/' + PACKAGE,
                      'catalog_relative_path': 'catalog.json', 'preview_relative_path': 'index.html',
                      'autofill_fields': ['image', 'boxes_xyxy', 'object_names'],
                      'samples': len(data['samples']), 'suites': data['suites'],
                      'inference_ready_samples': sum(sample['inference_ready'] for sample in data['samples']),
                      'payload_files': len(rows), 'payload_bytes': sum(row['bytes'] for row in rows),
                      'scope': data['scope']}
        verify(archive, descriptor)
        archive.replace(output)
        output.with_suffix('.package.json').write_bytes(encoded(descriptor))
        for relative in ('catalog.json', 'index.html'):
            (source / relative).write_bytes(generated[relative])
        (source / 'package.json').write_bytes(encoded(descriptor))
    return descriptor


def verify(archive, descriptor):
    if archive.stat().st_size != descriptor['zip_bytes'] or digest_file(archive) != descriptor['zip_sha256']:
        raise ValueError('archive size/SHA-256 mismatch')
    with zipfile.ZipFile(archive) as packed:
        names = packed.namelist()
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name or path.parts[0] != PACKAGE:
                raise ValueError('unsafe archive member: ' + name)
            mode = packed.getinfo(name).external_attr >> 16
            if mode & 0o170000 != 0o100000:
                raise ValueError('archive member is not a regular file: ' + name)
        rows = json.loads(packed.read(PACKAGE + '/manifest.json'))['files']
        expected = [PACKAGE + '/' + row['file'] for row in rows] + [PACKAGE + '/manifest.json']
        if len(names) != len(set(names)) or sorted(names) != sorted(expected):
            raise ValueError('archive member list differs from manifest')
        for row in rows:
            with packed.open(PACKAGE + '/' + row['file']) as stream:
                digest, size = digest_stream(stream)
            if digest != row['sha256'] or size != row['bytes']:
                raise ValueError('payload hash mismatch: ' + row['file'])
        data = json.loads(packed.read(PACKAGE + '/catalog.json'))
        hashes = {row['file']: row['sha256'] for row in rows}
        if len(data['samples']) != descriptor['samples']:
            raise ValueError('sample count mismatch')
        for sample in data['samples']:
            if hashes.get(sample['image']) != sample['image_sha256']:
                raise ValueError('catalog image mismatch: ' + sample['image'])
            if sample['annotation_manifest'] not in hashes:
                raise ValueError('catalog annotation missing')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=DATA.parent / 'outputs/packages' / (PACKAGE + '.zip'))
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        descriptor = json.loads(args.output.with_suffix('.package.json').read_text())
        verify(args.output, descriptor)
    else:
        descriptor = build(DATA, args.output)
    print(json.dumps({'status': 'verified', 'archive': str(args.output),
                      'samples': descriptor['samples'], 'bytes': descriptor['zip_bytes'],
                      'sha256': descriptor['zip_sha256']}))


if __name__ == '__main__':
    main()
