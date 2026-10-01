#!/usr/bin/env python3
"""Reproducible real-checkpoint, all-frame Python / GGML full-graph benchmark.

The C++ graph receives identical normalized pixels and GT boxes. Raw logits
are matched by subject/object IDs, never by TopK row position. Artifacts include
every prediction, task metrics, timings, per-image comparisons and plots.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import struct
import subprocess
import sys
import threading
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MODELS = ('relsgg-vits16', 'relsgg-vits16plus', 'relsgg-vitb16')
DTYPES = ('f32', 'f16', 'q8_0')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''): h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def hardware_snapshot():
    result = {'platform': platform.platform(), 'time_utc': datetime.now(timezone.utc).isoformat()}
    for name, query in [('gpu','--query-gpu=name,driver_version,pstate,temperature.gpu,clocks.sm,utilization.gpu,memory.used'),
                        ('processes','--query-compute-apps=pid,process_name,used_memory')]:
        response = subprocess.run(['nvidia-smi',query,'--format=csv'],capture_output=True,text=True)
        result[name] = response.stdout.strip()
    return result


def gpu_processes():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True)
    return [{'pid': int(row.split(',', 1)[0]), 'name': row.split(',', 1)[1].strip()}
            for row in result.stdout.splitlines() if row.strip()]


def native_timing_run(args, cmd, dst):
    """Retry whole runs if transient GPU jobs overlap; preserve every attempt."""
    allowed = set(args.allow_gpu_pid)
    attempts = []
    while True:
        if args.wait_gpu_idle:
            while True:
                busy = [p for p in gpu_processes() if p['pid'] not in allowed]
                if not busy: break
                print('waiting for transient GPU processes:', busy, flush=True)
                time.sleep(5)
        samples = []; stop = threading.Event()
        started = time.perf_counter()
        proc = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        def monitor():
            while not stop.is_set():
                try: samples.append({'seconds': time.perf_counter()-started, 'processes': gpu_processes()})
                except (subprocess.SubprocessError, ValueError): pass
                stop.wait(.2)
        worker = threading.Thread(target=monitor, daemon=True); worker.start()
        stdout, stderr = proc.communicate(); stop.set(); worker.join()
        interference = sorted({p['pid'] for sample in samples for p in sample['processes']
                               if p['pid'] not in allowed and p['pid'] != proc.pid})
        attempt = {'returncode': proc.returncode, 'samples': samples,
                   'transient_pids': interference, 'allowed_gpu_pids': sorted(allowed)}
        attempts.append(attempt); save(dst/'timing_attempts.json', attempts)
        if proc.returncode or not args.wait_gpu_idle or not interference:
            return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
        (dst/f'interference-{len(attempts)}.log').write_text(stdout+stderr)
        print('discard timing attempt with concurrent GPU jobs:', interference, flush=True)


def prepare_image(path, sample):
    with Image.open(path) as raw: image = raw.convert('RGB')
    w, h = image.size
    pixels = np.asarray(image.resize((448, 448), Image.Resampling.BILINEAR), np.float32).transpose(2, 0, 1) / 255
    normalized = (pixels - np.array([.485, .456, .406], np.float32)[:, None, None]) / np.array([.229, .224, .225], np.float32)[:, None, None]
    xyxy = np.array([[x,y,x+bw,y+bh] for x,y,bw,bh in (o['bbox'] for o in sample['objects'])], np.float32)
    xyxy /= np.array([w, h, w, h], np.float32)
    boxes = np.concatenate(((xyxy[:, :2]+xyxy[:, 2:])*.5,xyxy[:, 2:]-xyxy[:, :2]),axis=1)
    return np.ascontiguousarray(normalized), boxes


def prepare(args, samples):
    dst = args.output / 'inputs'
    dst.mkdir(parents=True, exist_ok=True)
    requests = args.output / 'image_requests'
    requests.mkdir(parents=True, exist_ok=True)
    times = {}
    for i, sample in enumerate(samples):
        tt = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            pixels, boxes = prepare_image(args.data / sample['image'], sample)
            tt.append((time.perf_counter() - start) * 1000)
        name = f'{i:03d}_{sample["image_id"]}'
        with (dst / (name + '.raim')).open('wb') as f:
            f.write(struct.pack('<8sIII', b'RAIMv1\0\0', 448, 448, len(boxes)))
            f.write(pixels.astype('<f4').tobytes()); f.write(boxes.astype('<f4').tobytes())
        request = ['RAIP1', json.dumps(str((args.data/sample['image']).resolve())), str(len(boxes))]
        request += [' '.join(map(str, (x,y,x+w,y+h))) for x,y,w,h in (o['bbox'] for o in sample['objects'])]
        (requests/(name+'.raip')).write_text('\n'.join(request)+'\n')
        times[name] = {'preprocess_ms': statistics.median(tt), 'samples_ms': tt,
                       'image_sha256': sha256(args.data / sample['image'])}
    save(args.output / 'inputs.json', times)


def raw_python(out):
    return {k: out[v][0].detach().float().cpu().numpy() for k, v in (
        ('logits', 'logits'), ('pairs', 'pair_logits'), ('sub', 'sub_idx'),
        ('obj', 'obj_idx'), ('valid', 'valid_mask'))}


def load_cpp(path):
    data = path.read_bytes()
    magic, k, v = struct.unpack_from('<8sII', data)
    if magic != b'RAFOv1\0\0' or len(data) != 16 + k * (16 + 4 * v):
        raise ValueError(f'Invalid full graph output: {path}')
    r = {}
    for j, key in enumerate(('valid', 'sub', 'obj', 'pairs')):
        r[key] = np.frombuffer(data, '<f4' if key == 'pairs' else '<i4', k, 16 + j * 4 * k).copy()
    r['logits'] = np.frombuffer(data, '<f4', k * v, 16 + 16 * k).reshape(k, v).copy()
    return r


def decode(out, names, calibration):
    logits, pair = out['logits'], out['pairs']
    best = logits.argmax(-1)
    valid = out['valid'].astype(bool) & (out['sub'] != out['obj'])
    scores = (logits[np.arange(len(best)), best] + pair) * calibration['a'] + calibration['b']
    scores = 1 / (1 + np.exp(-np.clip(scores, -80, 80)))
    rows = [{'subject': int(out['sub'][i]), 'object': int(out['obj'][i]),
             'predicate': names[int(best[i])], 'predicate_id': int(best[i]), 'score': float(scores[i])}
            for i in np.flatnonzero(valid)]
    return sorted(rows, key=lambda x: -x['score'])


def gt_metrics(predictions, sample):
    gt = {(int(r['subject']), int(r['object']), r['predicate']) for r in sample['relations']}
    result = {'gt_count': len(gt), 'predicate_support': dict(Counter(x[2] for x in gt))}
    for k in (20, 50, 100):
        predicted = {(r['subject'], r['object'], r['predicate']) for r in predictions[:k]}
        hits = gt & predicted
        result[f'hits@{k}'] = len(hits)
        result[f'predicate_hits@{k}'] = dict(Counter(x[2] for x in hits))
        result[f'precision@{k}'] = len(hits) / max(1, len(predicted))
    return result


def parity(reference, candidate):
    def lookup(out):
        return {(int(out['sub'][i]), int(out['obj'][i])): i for i in np.flatnonzero(out['valid'])}
    a, b = lookup(reference), lookup(candidate)
    common = sorted(a.keys() & b.keys())
    if not common: raise ValueError('no matching valid pair')
    ix, iy = [a[k] for k in common], [b[k] for k in common]
    x, y = reference['logits'][ix], candidate['logits'][iy]
    if not np.isfinite(y).all(): raise ValueError('nonfinite GGML predicate logits')
    e = y.astype(np.float64) - x
    ep = candidate['pairs'][iy].astype(np.float64) - reference['pairs'][ix]
    topx, topy = x.argmax(-1), y.argmax(-1)
    flatx, flaty = x.ravel().astype(np.float64), y.ravel().astype(np.float64)
    return {'matched_pairs': len(common), 'reference_pairs': len(a), 'candidate_pairs': len(b),
            'pair_set_recall': len(common) / len(a), 'pair_set_jaccard': len(common) / len(a.keys() | b.keys()),
            'logits_rmse': float(np.sqrt(np.mean(e*e))), 'logits_mae': float(np.mean(abs(e))),
            'logits_max_abs': float(abs(e).max()), 'pair_logits_rmse': float(np.sqrt(np.mean(ep*ep))),
            'predicate_top1_agreement': float(np.mean(topx == topy)),
            'logits_cosine': float(np.dot(flatx, flaty) / (np.linalg.norm(flatx) * np.linalg.norm(flaty)))}


def python_reference(args, samples, model_name):
    import torch
    from relsgg import RelateAnything
    checkpoint = ROOT / 'cpp_ggml/models/pytorch' / model_name / 'model.pth'
    model = RelateAnything.from_checkpoint(str(checkpoint), device='cuda', full_vocabulary=True)
    dst = args.output / model_name / 'python'
    dst.mkdir(parents=True, exist_ok=True)
    rows = []
    captured = {}
    def hook(_module, _inputs, output): captured['out'] = output
    handle = model.model.register_forward_hook(hook)
    torch.cuda.reset_peak_memory_stats()
    names = list(model.predicates)
    calibration = {'a': model.calib_a, 'b': model.calib_b}
    save(dst / 'vocabulary.json', {'names': names, 'calibration': calibration})
    for index, sample in enumerate(samples):
        name = f'{index:03d}_{sample["image_id"]}'
        xyxy = np.array([[x,y,x+w,y+h] for x,y,w,h in (o['bbox'] for o in sample['objects'])], np.float32)
        times = []
        if index == 0:
            with Image.open(args.data / sample['image']) as img:
                for _ in range(args.warmup): model.predict(img.convert('RGB'), xyxy, topk=100)
        for _ in range(args.repeats):
            torch.cuda.synchronize(); start = time.perf_counter()
            with Image.open(args.data / sample['image']) as img:
                preds = model.predict(img.convert('RGB'), xyxy, topk=100)
            torch.cuda.synchronize(); times.append((time.perf_counter()-start)*1000)
        out = raw_python(captured['out'])
        np.savez(dst / (name + '.npz'), **out)
        predicted = decode(out, names, calibration)
        # Check the independent decoder against upstream's public result.
        actual = [(p.subject_idx, p.object_idx, p.predicate) for p in preds]
        decoded = [(p['subject'], p['object'], p['predicate']) for p in predicted[:100]]
        if actual != decoded: raise RuntimeError(f'public Python decode mismatch: {name}')
        rows.append({'sample': name, 'image_id': sample['image_id'], 'source': sample['source'],
                     'latency_ms': statistics.median(times), 'iterations_ms': times,
                     'predictions': predicted[:100], **gt_metrics(predicted, sample)})
        print(f'python {model_name} {index+1}/{len(samples)} {statistics.median(times):.2f} ms', flush=True)
    handle.remove()
    save(dst / 'results.json', {'backend': 'pytorch-cuda', 'dtype': 'f32', 'checkpoint_sha256': sha256(checkpoint),
         'torch_version': torch.__version__, 'device': torch.cuda.get_device_name(),
         'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'hardware': hardware_snapshot(), 'rows': rows})
    del model
    torch.cuda.empty_cache()


def ggml_benchmark(args, samples, model_name, backend, dtype):
    model = ROOT / 'cpp_ggml/models/gguf' / f'{model_name}-{dtype}.gguf'
    dst = args.output / model_name / f'{backend}-{dtype}'
    dst.mkdir(parents=True, exist_ok=True)
    cmd = [str(ROOT / f'cpp_ggml/build-{backend}/relateanything-ggml'), '--model', str(model),
           '--input', str(args.output / 'image_requests'), '--output', str(dst), '--backend', backend,
           '--warmup', str(args.warmup), '--repeats', str(args.repeats), '--max-boxes', '60']
    if args.no_flash: cmd.append('--no-flash-attention')
    started = time.perf_counter()
    result = native_timing_run(args, cmd, dst)
    (dst / 'stdout.log').write_text(result.stdout); (dst / 'stderr.log').write_text(result.stderr)
    if result.returncode: raise RuntimeError(f'GGML failed {backend}/{dtype}: {result.stderr[-3000:]}')
    times, current = {}, None
    for line in result.stdout.splitlines():
        if line.startswith('sample='): current = line.split('=',1)[1]
        elif line.startswith('iteration_ms='):
            times[current] = [float(x) for x in line.split('=',1)[1].split(',')]
    vocab = json.loads((args.output / model_name / 'python/vocabulary.json').read_text())
    raw_cmd = list(cmd)
    raw_cmd[raw_cmd.index('--repeats')+1] = '1'
    raw_cmd[raw_cmd.index('--warmup')+1] = '0'
    raw = subprocess.run(raw_cmd+['--raw-logits'],text=True,capture_output=True)
    (dst/'raw_stderr.log').write_text(raw.stderr)
    if raw.returncode: raise RuntimeError(f'raw validation run failed: {raw.stderr[-3000:]}')
    rows = []
    for i, sample in enumerate(samples):
        name = f'{i:03d}_{sample["image_id"]}'
        out = load_cpp(dst / (name + '.rafo'))
        with np.load(args.output / model_name / 'python' / (name + '.npz')) as ref:
            numerical = parity(ref, out)
        decoded = decode(out, vocab['names'], vocab['calibration'])
        native_result = json.loads((dst/(name+'.json')).read_text())
        predictions = native_result['predictions']
        # The same raw logits must produce the native calibrated triplets.
        native = {(p['subject'],p['object']):p for p in predictions}
        native_score_error = max(abs(native[(p['subject'],p['object'])]['score']-p['score']) for p in decoded)
        if any(native[(p['subject'],p['object'])]['predicate_id']!=p['predicate_id'] for p in decoded):
            raise RuntimeError(f'native argmax disagrees with raw logits: {name}')
        if native_score_error > 1e-5: raise RuntimeError(f'calibration mismatch: {native_score_error}')
        elapsed = statistics.median(times[name])
        rows.append({'sample': name, 'image_id': sample['image_id'], 'source': sample['source'],
            'latency_ms': elapsed, 'native_decode_max_error': native_score_error,
            'backend_buffer_bytes': native_result.get('backend_buffer_bytes'),
            'iterations_ms': times[name], 'parity': numerical,
            'predictions': predictions[:100], **gt_metrics(predictions, sample)})
    save(dst / 'results.json', {'backend': backend, 'dtype': dtype, 'gguf_sha256': sha256(model),
        'gguf_bytes': model.stat().st_size, 'flash_attention': not args.no_flash,
        'timing_scope': 'native C++ JPEG/PNG decode, bilinear resize, normalize, full GGML graph, calibrated triplets',
        'binary_sha256': sha256(Path(cmd[0])),
        'hardware': hardware_snapshot(),
        'command': cmd, 'wall_seconds': time.perf_counter()-started, 'rows': rows})
    print(f'{model_name}/{backend}/{dtype}: median pipeline {statistics.median(r["latency_ms"] for r in rows):.2f} ms', flush=True)


def summary(result):
    rows = result['rows']; n = len(rows)
    support = Counter(); hits = {k: Counter() for k in (20, 50, 100)}
    for row in rows:
        support.update(row['predicate_support'])
        for k in hits: hits[k].update(row[f'predicate_hits@{k}'])
    s = {'images': n, 'backend': result['backend'], 'dtype': result['dtype'],
         'latency_median_ms': float(np.median([r['latency_ms'] for r in rows])),
         'latency_p95_ms': float(np.percentile([r['latency_ms'] for r in rows], 95)),
         'gt_relations': sum(support.values()), 'classes': len(support)}
    for k in hits:
        s[f'R@{k}'] = sum(hits[k].values()) / sum(support.values())
        s[f'mR@{k}'] = float(np.mean([hits[k][p]/c for p,c in support.items()]))
        r, mr = s[f'R@{k}'], s[f'mR@{k}']
        s[f'F1@{k}'] = 2*r*mr/(r+mr) if r+mr else 0.0
        weights = {p: np.log(sum(support.values())/count) for p,count in support.items()}
        s[f'wR@{k}'] = sum(weights[p]*hits[k][p]/count for p,count in support.items())/sum(weights.values())
        # These are explicitly local support buckets, not the upstream train
        # frequency buckets (the 50-image subset has a different distribution).
        for label,lo,hi in [('rare',1,2),('common',2,5),('frequent',5,float('inf'))]:
            recalls=[hits[k][p]/count for p,count in support.items() if lo<=count<hi]
            s[f'mR@{k}_{label}'] = float(np.mean(recalls)) if recalls else None
    if 'parity' in rows[0]:
        for key in rows[0]['parity']:
            s[key] = float(np.mean([r['parity'][key] for r in rows]))
        s['logits_max_abs'] = max(r['parity']['logits_max_abs'] for r in rows)
    s['throughput_fps'] = 1000 / float(np.mean([r['latency_ms'] for r in rows]))
    s['weight_bytes'] = result.get('gguf_bytes')
    s['torch_peak_allocated_bytes'] = result.get('peak_allocated_bytes')
    buffers=[r['backend_buffer_bytes'] for r in rows if r.get('backend_buffer_bytes') is not None]
    s['ggml_backend_buffer_bytes'] = max(buffers) if buffers else None
    return s


def report(args, samples):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summaries, all_results = {}, {}
    lines = ['# Official full-graph PyTorch / GGML comparison', '',
        f'Generated {datetime.now(timezone.utc).isoformat()}. Hardware: {platform.platform()}.', '',
        'All rows use the same 50 retained RA-4M images, GT boxes, 448 x 448 pixels, 19,103 predicates, and official checkpoints.',
        'Python measures image decode through calibrated triplets with CUDA synchronization. GGML measures native C++ image decode, Pillow-compatible resize, normalization, full graph allocation/transfers/execution, and calibrated triplets in one wall-clock interval. Model loading and artifact writes are excluded for both. Numerical validation uses a separate raw-logit pass; exporting 9.8 MB per image does not burden the native deployment timing.',
        'Accelerated attention uses F16 K/V with F32 accumulation for all storage types. Use --no-flash for explicit F32 attention. F32/F16/Q8 label weight storage, not every activation. For controlled runs, timing_attempts.json records sampled GPU processes and discarded attempts. Explicit --allow-gpu-pid values permit persistent services on both sides. Hardware/process snapshots are included in each results.json; measurements on this shared workstation are not isolated-device guarantees.',
        'Raw logits are aligned by subject/object IDs. A changed pair set is reported separately. Quantized storage cannot be bit identical to F32. Exact-string R/mR are local diagnostics on incomplete positive annotations, not upstream OVS-F1 or negative-set AP.', '',
        '| Model | Engine/storage | Median ms | P95 ms | R@50 | mR@50 | Logit RMSE | Top-1 agreement | Pair recall |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for model_name in args.models:
        summaries[model_name] = {}; all_results[model_name] = {}
        for p in sorted((args.output / model_name).glob('*/results.json')):
            result = json.loads(p.read_text()); s = summary(result)
            summaries[model_name][p.parent.name] = s; all_results[model_name][p.parent.name] = result
            lines.append(f'| {model_name} | {s["backend"]}/{s["dtype"]} | {s["latency_median_ms"]:.2f} | {s["latency_p95_ms"]:.2f} | {s["R@50"]:.2%} | {s["mR@50"]:.2%} | {s.get("logits_rmse",0):.5f} | {s.get("predicate_top1_agreement",1):.2%} | {s.get("pair_set_recall",1):.2%} |')
    save(args.output / 'summary.json', {'models': summaries, 'warmup': args.warmup, 'repeats': args.repeats,
        'manifest_sha256': sha256(args.data/'manifest.json'),
        'ggml_commit': subprocess.check_output(['git','-C',str(ROOT/'cpp_ggml/third_party/ggml'),'rev-parse','HEAD'],text=True).strip()})
    csv_keys=['latency_median_ms','latency_p95_ms','throughput_fps','weight_bytes','torch_peak_allocated_bytes','ggml_backend_buffer_bytes','R@20','R@50','R@100','mR@20','mR@50','mR@100','F1@50','wR@50','mR@50_rare','mR@50_common','mR@50_frequent','logits_mae','logits_rmse','logits_max_abs','predicate_top1_agreement','pair_set_recall']
    import csv
    with (args.output/'metrics.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['model','engine']+csv_keys)
        for model,entries in summaries.items():
            for engine,s in entries.items():writer.writerow([model,engine]+[s.get(k,'') for k in csv_keys])
    groups={}
    for model,entries in all_results.items():
        groups[model]={}
        for engine,result in entries.items():
            groups[model][engine]={source:summary({**result,'rows':[r for r in result['rows'] if r['source']==source]}) for source in sorted({r['source'] for r in result['rows']})}
    save(args.output/'source_breakdown.json',groups)
    fig, axes = plt.subplots(len(args.models), 3, figsize=(16, 4*len(args.models)), squeeze=False, layout='constrained')
    for row, model in enumerate(args.models):
        entries = summaries[model]; labels = list(entries); ss = list(entries.values()); x = np.arange(len(ss))
        for col, (key, title) in enumerate([('latency_median_ms','Median pipeline ms'), ('R@50','Exact GT recall@50'), ('predicate_top1_agreement','Predicate agreement')]):
            ax=axes[row,col]; ax.bar(x,[s.get(key,1) for s in ss],color=['#777777' if a=='python' else '#087e8b' if a.startswith('cuda') else '#b86b35' for a in labels])
            vals=[s.get(key,1) for s in ss]
            ax.bar_label(ax.containers[0],labels=[f'{v:.1f}' if col==0 else f'{v:.1%}' for v in vals],fontsize=8,padding=2)
            ax.set_xticks(x,labels,rotation=40,ha='right');ax.set_title(model+' | '+title);ax.grid(axis='y',alpha=.2)
            if col:ax.set_ylim(0,1.08)
    fig.savefig(args.output/'metrics.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(len(args.models),3,figsize=(16,4*len(args.models)),squeeze=False,layout='constrained')
    for row,model in enumerate(args.models):
        for col,source in enumerate(('coco','objects365','openimages')):
            entries=groups[model];labels=list(entries);ax=axes[row,col]
            ax.bar(np.arange(len(labels)),[entries[e][source]['mR@50'] for e in labels]);ax.set_xticks(np.arange(len(labels)),labels,rotation=40,ha='right')
            ax.set_title(model+' / '+source+' mR@50');ax.set_ylim(0,1);ax.grid(axis='y',alpha=.2)
    fig.savefig(args.output/'source_breakdown.png',dpi=160);plt.close(fig)
    lines += ['', 'Memory fields use different allocator scopes: PyTorch peak allocated bytes versus GGML weight (including packed QKV) plus scheduler buffers. Neither includes every driver allocation; do not treat them as directly comparable process VRAM peaks.', '', '![Measured metrics](metrics.png)', '', '![Source breakdown](source_breakdown.png)', '',
              '[All metrics CSV](metrics.csv) includes R/mR at 20, 50 and 100, their harmonic F1, log-frequency weighted recall, and local support buckets. F1 follows upstream: harmonic R and mR, not precision/recall F1. [Source breakdown](source_breakdown.json) covers all three source corpora.', '', '## Complete per-image comparisons', '',
              'Each image below links to three complete grids. Every grid has ground truth, official PyTorch, and every measured backend/storage combination.', '']
    for i,sample in enumerate(samples):
        links=[]
        for model in args.models:
            file=render_image(args,sample,i,model,all_results[model]);links.append(f'[{model}]({file.relative_to(args.output).as_posix()})')
        lines.append(f'- {sample["image_id"]} ({sample["source"]}, {sample["difficulty"]}): '+ ' | '.join(links))
    lines += ['', '### Full illustrated galleries', '']
    for model in args.models:
        gallery=[f'# {model}: every retained frame', '',
                 '[Return to metrics and protocol](README.md)', '',
                 'Each grid shows GT, independent PyTorch, CUDA F32/F16/Q8, and Vulkan F32/F16/Q8. Full predictions and numerical errors are in the adjacent result JSON files.', '']
        for i,sample in enumerate(samples):
            gallery += [f'## {i+1}. {sample["image_id"]} — {sample["source"]} / {sample["difficulty"]}', '',
                        f'![Full comparison for {sample["image_id"]}](visualizations/{model}/{i:03d}_{sample["image_id"]}.png)', '']
        name=f'GALLERY-{model}.md';(args.output/name).write_text('\n'.join(gallery)+'\n')
        lines.append(f'- [{model}: all {len(samples)} grids embedded]({name})')
    lines += ['', '## Reproduce', '', '```sh', 'python cpp_ggml/scripts/benchmark_full_graph.py --stage all', '```', '',
        'Raw per-image outputs, predictions, timing samples and logit comparisons are stored beside this report. `summary.json` is the machine-readable source for the chart. No values from adapter fixtures enter this report.']
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')


def render_image(args, sample, index, model, results):
    import textwrap
    panels=[('Ground truth',sample['relations'])]
    ordered=['python']+[key for key in results if key!='python']
    for name in ordered:
        result=results[name]; panels.append((name,result['rows'][index]['predictions']))
    cw,ch=440,420; cols=4; canvas=Image.new('RGB',(cols*cw,((len(panels)+cols-1)//cols)*ch),'#f4f7f8'); draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('DejaVuSans.ttf',13); title=ImageFont.truetype('DejaVuSans.ttf',16)
    with Image.open(args.data/sample['image']) as im: photo=im.convert('RGB')
    scale=min((cw-16)/photo.width,245/photo.height); photo=photo.resize((round(photo.width*scale),round(photo.height*scale)))
    for j,(name,predictions) in enumerate(panels):
        x=(j%cols)*cw;y=(j//cols)*ch; canvas.paste(photo,(x+8,y+30))
        draw.text((x+8,y+6),f'{model} / {name}',font=title,fill='#17343a')
        for oi,obj in enumerate(sample['objects']):
            bx,by,bw,bh=obj['bbox'];r=[x+8+bx*scale,y+30+by*scale,x+8+(bx+bw)*scale,y+30+(by+bh)*scale]
            draw.rectangle(r,outline='#00ad9e',width=2);draw.text((r[0],r[1]),str(oi),font=font,fill='black',stroke_width=1,stroke_fill='white')
        yy=y+283
        for pred in predictions[:5]:
            text=f'#{pred["subject"]} -> {pred["predicate"]} -> #{pred["object"]}'
            if 'score' in pred:text+=f'  {pred["score"]:.4f}'
            for line in textwrap.wrap(text,56)[:2]:draw.text((x+8,yy),line,font=font,fill='#17343a');yy+=16
    dest=args.output/'visualizations'/model/f'{index:03d}_{sample["image_id"]}.png';dest.parent.mkdir(parents=True,exist_ok=True);canvas.save(dest)
    return dest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('all','prepare','python','ggml','report'),default='all')
    p.add_argument('--data',type=Path,default=ROOT/'cpp_ggml/test_data/ra4m_50')
    p.add_argument('--output',type=Path,default=ROOT/'cpp_ggml/benchmarks/full_graph')
    p.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    p.add_argument('--backends',nargs='+',choices=('cpu','cuda','vulkan'),default=['cuda','vulkan'])
    p.add_argument('--dtypes',nargs='+',choices=DTYPES,default=list(DTYPES))
    p.add_argument('--warmup',type=int,default=2);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--no-flash',action='store_true');p.add_argument('--limit',type=int)
    p.add_argument('--wait-gpu-idle',action='store_true',help='retry native timing runs that overlap other GPU jobs')
    p.add_argument('--allow-gpu-pid',type=int,nargs='*',default=[],help='explicit persistent GPU processes allowed during timing')
    args=p.parse_args()
    if args.warmup<0 or args.repeats<1:p.error('invalid warmup/repeats')
    args.output=args.output.resolve();args.data=args.data.resolve()
    samples=json.loads((args.data/'manifest.json').read_text())['samples']
    if args.limit:samples=samples[:args.limit]
    if args.stage in ('all','prepare'):prepare(args,samples)
    if args.stage in ('all','python'):
        for model in args.models:
            if args.wait_gpu_idle:
                dst=args.output/model/'python'; dst.mkdir(parents=True,exist_ok=True)
                command=[sys.executable,str(Path(__file__).resolve()),'--stage','python','--models',model,
                         '--data',str(args.data),'--output',str(args.output),'--warmup',str(args.warmup),
                         '--repeats',str(args.repeats)]
                if args.limit:command+=['--limit',str(args.limit)]
                result=native_timing_run(args,command,dst)
                (dst/'stdout.log').write_text(result.stdout)
                (dst/'stderr.log').write_text(result.stderr)
                if result.returncode:raise RuntimeError(result.stderr[-3000:])
                print(model,'controlled Python reference completed',flush=True)
            else:python_reference(args,samples,model)
    if args.stage in ('all','ggml'):
        for model in args.models:
            for backend in args.backends:
                for dtype in args.dtypes:ggml_benchmark(args,samples,model,backend,dtype)
    if args.stage in ('all','report'):report(args,samples)
    return 0


if __name__=='__main__':raise SystemExit(main())
