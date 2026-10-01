#!/usr/bin/env python3
"""Retain bounded valid/test SpatialSense samples from the official archive."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from urllib.parse import urlparse

import requests
from PIL import Image
from provenance import save, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--max-download-mib', type=int, default=256)
    args = parser.parse_args()
    if not 2 <= args.limit <= 50:
        parser.error('--limit must be in [2,50]')
    args.output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    api = 'https://zenodo.org/api/records/8104370'
    response = session.get(api, timeout=60)
    response.raise_for_status()
    record = response.json()
    save(args.output/'record.json', record)
    files = {x['key']: x for x in record['files']}
    response = session.get(files['annotations.json']['links']['self'], timeout=60)
    response.raise_for_status()
    if 'md5:'+hashlib.md5(response.content).hexdigest() != files['annotations.json']['checksum']:
        raise RuntimeError('annotation checksum mismatch')
    (args.output/'annotations.json').write_bytes(response.content)
    annotations = response.json()
    names = sorted({a['predicate'] for r in annotations for a in r['annotations']})
    selected = {split: [] for split in ('valid', 'test')}
    source = {Path(urlparse(row['url']).path).name: (i, row) for i, row in enumerate(annotations)
              if row['split'] in selected}
    images = args.output/'images'
    images.mkdir(exist_ok=True)
    archive = files['images.tar.gz']['links']['self']
    class BoundedReader:
        def __init__(self, stream):self.stream=stream;self.bytes=0
        def read(self, size):
            result=self.stream.read(size);self.bytes+=len(result)
            if self.bytes > args.max_download_mib*2**20:
                raise RuntimeError('archive scan exceeded download byte limit')
            return result
    # Archive order is deterministic and avoids scanning 1.8 GB for scattered IDs.
    with session.get(archive, timeout=(30, 120), stream=True) as response:
            response.raise_for_status()
            bounded=BoundedReader(response.raw)
            with tarfile.open(fileobj=bounded, mode='r|gz') as stream:
                for member in stream:
                    name = Path(member.name).name
                    if member.isfile() and name in source:
                        index,row=source[name]
                        if len(selected[row['split']]) >= args.limit:continue
                        if member.size > 32*2**20:
                            raise RuntimeError('unexpected image size')
                        (images/name).write_bytes(stream.extractfile(member).read())
                        selected[row['split']].append((index,row))
                        print('retained', row['split'], len(selected[row['split']]), name, flush=True)
                    if all(len(x)==args.limit for x in selected.values()):
                        break
    if any(len(x)!=args.limit for x in selected.values()):
        raise RuntimeError('archive lacks enough selected split images')
    for split, rows in selected.items():
        folder = args.output/f'SpatialSense-{split}'
        folder.mkdir(exist_ok=True)
        save(folder/'categories.json', {'rel_categories': [{'id': i, 'name': name}
                                                          for i, name in enumerate(names)]})
        samples = []
        for index, row in rows:
            name = Path(urlparse(row['url']).path).name
            image = images/name
            with Image.open(image) as decoded:
                if decoded.size != (row['width'], row['height']):
                    raise RuntimeError(f'image dimensions mismatch: {name}')
            objects, ids, cells = [], {}, []
            for annotation in row['annotations']:
                pair = []
                for tag in ('subject', 'object'):
                    entity = annotation[tag]
                    y0, y1, x0, x1 = map(float, entity['bbox'])
                    key = (x0, y0, x1, y1, entity['name'])
                    if key not in ids:
                        ids[key] = len(objects)
                        objects.append({'id': len(objects), 'name': entity['name'],
                                        'bbox': [x0, y0, x1-x0, y1-y0]})
                    pair.append(ids[key])
                cells.append({'subject': pair[0], 'object': pair[1],
                              'predicate': annotation['predicate'], 'label': bool(annotation['label'])})
            samples.append({'image': '../images/'+name, 'image_id': name, 'source': 'SpatialSense',
                            'objects': objects, 'relations': [{k: c[k] for k in ('subject','object','predicate')}
                                                             for c in cells if c['label']],
                            'cells': cells, 'image_sha256': sha256(image), 'row_index': index})
        save(folder/'manifest.json', {'repo': api, 'revision': record['revision'], 'split': split,
             'samples': samples, 'annotation_sha256': sha256(args.output/'annotations.json'),
             'archive': files['images.tar.gz'], 'license': record['metadata']['license'],
             'downloaded_archive_bytes': bounded.bytes,
             'scope': 'first N valid/test images encountered in official archive order; verified positive/negative cells; not full published protocol'})
        print(split, len(samples), 'images', sum(len(x['cells']) for x in samples), 'cells', flush=True)


if __name__ == '__main__':
    main()
