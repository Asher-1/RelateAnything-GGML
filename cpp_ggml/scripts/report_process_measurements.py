#!/usr/bin/env python3
"""Report comparable sampled process memory and every slower native image."""
import argparse
import json
from pathlib import Path
from benchmark_full_graph import MODELS, summary
from provenance import save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',type=Path,required=True)
    folder=parser.parse_args().benchmark
    rows=[];slower=[];brackets=[]
    lines=['# Process memory and per-image latency', '',
           'Memory is sampled every 200 ms in the isolated child process, from startup '
           'through output writing. RSS/HWM comes from /proc; device memory comes from '
           'the driver process list including graphical contexts. These are sampled '
           'peaks, not guaranteed sub-200-ms maxima. Both engines use the same scope. '
           'Allocator-specific PyTorch/GGML counters remain separately named.', '',
           '| Model | Engine | Process RSS peak MiB | Driver device peak MiB | Slower images / 50 |',
           '|---|---|---:|---:|---:|']
    for model in MODELS:
        py=json.loads((folder/model/'python/results.json').read_text())
        baseline={r['sample']:r for r in py['rows']}
        for path in sorted((folder/model).glob('*/results.json')):
            engine=path.parent.name
            result=json.loads(path.read_text())
            memories=[json.loads(p.read_text()) for p in
                      ([path.parent/'memory-before.json',path.parent/'memory-after.json'] if engine=='python'
                       else [path.parent/'process_memory.json'])]
            peak={key:max((phase.get(source) or 0 for m in memories for phase in m['phases'].values()),default=0)
                  for key,source in (('rss_peak_bytes','rss_bytes'),('device_peak_bytes','device_bytes'),
                                     ('rss_hwm_peak_bytes','rss_hwm_bytes'))}
            own=[]
            if engine!='python':
                for r in result['rows']:
                    ref=baseline[r['sample']]['latency_ms']
                    if r['latency_ms']>ref:
                        own.append({'model':model,'engine':engine,'sample':r['sample'],
                                    'python_ms':ref,'native_ms':r['latency_ms'],'ratio':r['latency_ms']/ref})
                slower.extend(own)
            rows.append({'model':model,'engine':engine,**peak,'phases':[m.get('phases',{}) for m in memories],
                         'slower_images':len(own),'timing':summary(result)})
            lines.append(f'| {model} | {engine} | {peak["rss_peak_bytes"]/2**20:.1f} | '
                         f'{peak["device_peak_bytes"]/2**20:.1f} | {len(own) if engine!="python" else "reference"} |')
        bracket=py.get('timing_bracket')
        if bracket:
            brackets += ['',f'{model} Python bracket drift: {bracket["relative_drift"]:.2%}. '
                      'Ratios close to this drift should not support a categorical speed claim.']
    save(folder/'process_measurements.json',{'memory':rows,'slower_images':sorted(slower,key=lambda r:-r['ratio']),
          'scope':'sampled isolated child process; startup/load/warmup/inference/output; 200 ms cadence'})
    lines += brackets + ['', '[Every slower image and phase memory values](process_measurements.json). '
              'A faster aggregate median does not imply every image or P95 is faster.']
    (folder/'PROCESS_MEASUREMENTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
