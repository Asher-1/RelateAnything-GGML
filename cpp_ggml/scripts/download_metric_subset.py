#!/usr/bin/env python3
"""Download bounded, revision-pinned public scene-graph test subsets."""
import argparse
from pathlib import Path
import requests
from PIL import Image
from provenance import save, sha256

REVISIONS={'PSG':'8a7e2f92840de2d267244c3ea26fa2daa0772120',
           'IndoorVG':'63105f42838e6ce0867f8ab9e2956c2dcec043fa',
           'VG150':'ea6fb3a56a0876eee98165ea17792fc6ec8460e6'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=10)
    args=parser.parse_args()
    if not 1<=args.limit<=50:parser.error('--limit must be in [1,50]')
    session=requests.Session()
    for name,revision in REVISIONS.items():
        repo=f'maelic/{name}-coco-format'; folder=args.output/name
        folder.mkdir(parents=True,exist_ok=True)
        metadata={}
        for filename in ('categories.json','README.md'):
            url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}'
            r=session.get(url,timeout=60);r.raise_for_status()
            (folder/filename).write_bytes(r.content)
            metadata[filename]={'url':url,'sha256':sha256(folder/filename)}
        categories=(folder/'categories.json').read_text()
        import json
        categories=json.loads(categories)
        predicates={x['id']:x['name'] for x in categories['rel_categories']}
        response=session.get('https://datasets-server.huggingface.co/rows',
                             params={'dataset':repo,'config':'default','split':'test',
                                     'offset':0,'length':args.limit,'revision':revision},timeout=60)
        response.raise_for_status();payload=response.json();samples=[]
        save(folder/'source_rows.json',payload)
        for item in payload['rows']:
            row=item['row'];url=row['image']['src']
            if revision not in url:raise RuntimeError('dataset server served another revision')
            target=folder/f'{item["row_idx"]:03d}_{row["image_id"]}.jpg'
            image_response=session.get(url,timeout=60);image_response.raise_for_status()
            target.write_bytes(image_response.content)
            with Image.open(target) as image:
                if image.size!=(row['width'],row['height']):raise RuntimeError('image/annotation shape mismatch')
            objects=row['objects'];indices={o['id']:i for i,o in enumerate(objects)}
            relations=[{'subject':indices[r['subject_id']],'object':indices[r['object_id']],
                        'predicate':predicates[r['predicate_id']]} for r in row['relations']]
            samples.append({'image':target.name,'image_id':row['image_id'],'row_index':item['row_idx'],
                            'image_sha256':sha256(target),'source':name,'objects':objects,'relations':relations})
        save(folder/'manifest.json',{'repo':repo,'revision':revision,'split':'test','samples':samples,
                                    'metadata':metadata,'source_rows_sha256':sha256(folder/'source_rows.json'),
                                    'scope':'first N test rows, engineering regression subset, positive annotations only',
                                    'license':'see pinned dataset README; original image rights remain with their owners'})
        print(name,len(samples),'images',sum(len(x['relations']) for x in samples),'relations',flush=True)


if __name__=='__main__':main()
