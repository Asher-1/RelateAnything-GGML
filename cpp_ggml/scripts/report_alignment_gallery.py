#!/usr/bin/env python3
"""Offline model/backend/storage gallery and exportable plots from measured rows."""
from __future__ import annotations
import argparse, base64, csv, hashlib, io, json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmark_full_graph import ROOT, MODELS, load_cpp

COLORS={'cpu':'#6a5acd','cuda':'#087e8b','vulkan':'#c56c2a','python':'#555555'}

def read(path): return json.loads(path.read_text())
def savefig(fig,folder,name):
    fig.savefig(folder/(name+'.png'),dpi=155)
    fig.savefig(folder/(name+'.svg'))
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=ROOT/'cpp_ggml/test_data/ra4m_50')
    args=p.parse_args();src=args.benchmark.resolve();dst=args.output.resolve();dst.mkdir(parents=True,exist_ok=True)
    samples=read(args.data/'manifest.json')['samples'];summary=read(src/'summary.json')['models']
    results={m:{f.parent.name:read(f) for f in (src/m).glob('*/results.json')} for m in MODELS}
    expected=[f'{i:03d}_{s["image_id"]}' for i,s in enumerate(samples)]
    for m,configs in results.items():
        for k,value in configs.items():
            if [r['sample'] for r in value['rows']]!=expected:raise ValueError(f'gallery sample order: {m}/{k}')
    identities={str(f.relative_to(src)):hashlib.sha256(f.read_bytes()).hexdigest()
                for m in MODELS for f in (src/m).glob('*/results.json')}
    identities['summary.json']=hashlib.sha256((src/'summary.json').read_bytes()).hexdigest()
    native=sorted(set(k for rows in summary.values() for k in rows if k!='python'),key=lambda k:(k.split('-')[0],k))
    labels=['python']+native
    # E2E P50/P95, including CPU instead of silently omitting the slower backend.
    fig,axes=plt.subplots(3,1,figsize=(16,14),layout='constrained')
    speedfig,speedaxes=plt.subplots(3,1,figsize=(16,13),layout='constrained')
    export=[]
    for m,ax,sax in zip(MODELS,axes,speedaxes):
        ss=summary[m];present=[k for k in labels if k in ss];x=np.arange(len(present))
        med=[ss[k]['latency_median_ms'] for k in present];hi=[ss[k]['latency_p95_ms'] for k in present]
        bars=ax.bar(x,med,color=[COLORS[k.split('-')[0]] for k in present],alpha=.9)
        ax.scatter(x,hi,color='#151515',marker='_',s=160,label='P95 across image medians')
        ax.bar_label(bars,labels=[f'{v:.2f}' for v in med],fontsize=9,padding=3)
        ax.axhline(ss['python']['latency_median_ms'],color='#555',ls='--',label='Official PyTorch-CUDA P50')
        ax.set_xticks(x,present,rotation=35,ha='right');ax.set_ylabel('image-to-triplets ms');ax.set_title(m+' | full pipeline latency');ax.legend(loc='upper left');ax.grid(axis='y',alpha=.2)
        speed=[ss['python']['latency_median_ms']/v for v in med]
        bars=sax.bar(x,speed,color=['#219653' if v>1 else '#c0392b' for v in speed]);sax.bar_label(bars,fmt='%.2fx',fontsize=9,padding=3)
        sax.axhline(1,color='black',ls='--');sax.set_xticks(x,present,rotation=35,ha='right');sax.set_ylabel('PyTorch-CUDA P50 / native P50');sax.set_title(m+' | values above 1 are faster');sax.grid(axis='y',alpha=.2)
        for k in present:export.append({'model':m,'config':k,'p50_ms':ss[k]['latency_median_ms'],'p95_ms':ss[k]['latency_p95_ms'],'speedup':ss['python']['latency_median_ms']/ss[k]['latency_median_ms'],'top1_agreement':ss[k].get('predicate_top1_agreement',1),'pair_jaccard':ss[k].get('pair_set_jaccard',1),'logit_rmse':ss[k].get('logits_rmse',0),'R50':ss[k]['R@50'],'mR50':ss[k]['mR@50']})
    savefig(fig,dst,'latency_all_backends');savefig(speedfig,dst,'speedup_all_backends')
    # A separate GPU scale keeps small accelerator differences readable beside CPU.
    fig,axes=plt.subplots(3,1,figsize=(16,13),layout='constrained')
    trade,tradeaxes=plt.subplots(1,3,figsize=(19,7),layout='constrained')
    for m,ax,tax in zip(MODELS,axes,tradeaxes):
        ss=summary[m];present=[k for k in labels if k=='python' or k.startswith(('cuda-','vulkan-'))]
        x=np.arange(len(present));med=[ss[k]['latency_median_ms'] for k in present]
        bars=ax.bar(x,med,color=[COLORS[k.split('-')[0]] for k in present]);ax.bar_label(bars,fmt='%.2f',padding=3)
        ax.scatter(x,[ss[k]['latency_p95_ms'] for k in present],color='black',marker='_',s=160,label='P95 across image medians')
        ax.axhline(ss['python']['latency_median_ms'],color='#555',ls='--',label='Official PyTorch-CUDA P50')
        ax.set_xticks(x,present,rotation=35,ha='right');ax.set_ylabel('image-to-triplets ms');ax.set_title(m+' | accelerator scale');ax.legend();ax.grid(axis='y',alpha=.2)
        for k in native:
            tax.scatter(ss[k]['logits_rmse'],ss['python']['latency_median_ms']/ss[k]['latency_median_ms'],
                        color=COLORS[k.split('-')[0]],marker={'f32':'o','f16':'s','q8_0':'^'}[k.split('-')[1]],
                        s=80,label=k,edgecolors='black',linewidths=.4)
        tax.axhline(1,color='black',ls='--');tax.axvline(0,color='black',ls=':')
        tax.set_xscale('symlog',linthresh=1e-5);tax.set_xlabel('matched logit RMSE vs original CUDA (0 = equal values)')
        tax.set_ylabel('PyTorch-CUDA P50 / native P50');tax.set_title(m+' | precision and speed together');tax.grid(alpha=.2)
        tax.legend(fontsize=7,loc='upper right')
    savefig(fig,dst,'latency_gpu_backends');savefig(trade,dst,'precision_speed_tradeoff')
    with (dst/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(export[0]));w.writeheader();w.writerows(export)
    # Accuracy and candidate agreement side by side, no positive-only precision claim.
    fig,axes=plt.subplots(3,4,figsize=(19,12),layout='constrained')
    for row,m in enumerate(MODELS):
        ss=summary[m];kk=[k for k in labels if k in ss];x=np.arange(len(kk))
        for col,(key,title,default) in enumerate([('predicate_top1_agreement','predicate top-1 vs official CUDA',1),('pair_set_jaccard','candidate Jaccard vs official CUDA',1),('logits_rmse','matched raw logit RMSE',0),('mR@50','exact-string GT mR@50',0)]):
            ax=axes[row,col];v=[ss[k].get(key,default) for k in kk];ax.bar(x,v,color=[COLORS[k.split('-')[0]] for k in kk]);ax.set_xticks(x,kk,rotation=65,ha='right',fontsize=7);ax.set_title(m+'\n'+title,fontsize=10);ax.grid(axis='y',alpha=.2)
            if col==2:ax.set_yscale('symlog',linthresh=1e-4)
    savefig(fig,dst,'accuracy_candidates')
    for m in MODELS:
        fig,axes=plt.subplots(2,3,figsize=(20,10),layout='constrained')
        for backend,ax in zip(('cpu','cuda','vulkan'),axes[0]):
            for k in ['python']+[k for k in native if k.startswith(backend+'-')]:
                if k not in results[m]:continue
                rows=results[m][k]['rows'];ax.plot([r['latency_ms'] for r in rows],label=k,linewidth=1)
            ax.set_title(m+' / '+backend+' | every image E2E');ax.set_xlabel('retained image index');ax.set_ylabel('ms');ax.legend(fontsize=8);ax.grid(alpha=.2)
        for backend,ax in zip(('cpu','cuda','vulkan'),axes[1]):
            kk=['python']+[k for k in native if k.startswith(backend+'-') and k in results[m]]
            vv=[[t for r in results[m][k]['rows'] for t in r['iterations_ms']] for k in kk]
            ax.boxplot(vv,labels=kk,showfliers=True);ax.set_title('all timed iterations, including outliers');ax.set_ylabel('ms');ax.grid(axis='y',alpha=.2)
        savefig(fig,dst,m+'-latency_distribution')
    # Every CPU/CUDA/Vulkan storage configuration gets all-image error distributions.
    rawstats=[]
    for m in MODELS:
        names=read(src/m/'python/vocabulary.json')['names']
        for k in native:
            if k not in results[m]:continue
            print('numeric plot',m,k,flush=True)
            radius=max(summary[m][k]['logits_max_abs'],1e-8)*1.00001
            histogram=np.zeros(200,dtype=np.int64);edges=np.linspace(-radius,radius,201);under=over=equal=count=pair_equal=pair_count=0;maximum=0;scatter_x=[];scatter_y=[];first=None;first_pair=None;worst=None;worst_error=-1
            for row in results[m][k]['rows']:
                name=row['sample'];ref=dict(np.load(src/m/'python'/(name+'.npz')));cand=load_cpp(src/m/k/(name+'.rafo'))
                a={(int(ref['sub'][i]),int(ref['obj'][i])):i for i in np.flatnonzero(ref['valid'])};b={(int(cand['sub'][i]),int(cand['obj'][i])):i for i in np.flatnonzero(cand['valid'])};common=sorted(a.keys()&b.keys());ix=[a[pair] for pair in common];iy=[b[pair] for pair in common]
                x=ref['logits'][ix];y=cand['logits'][iy];err=y.astype(np.float64)-x;abs_err=np.abs(err);v=float(abs_err.max());maximum=max(maximum,v);bits=x.view(np.uint32)==y.view(np.uint32);equal+=int(bits.sum());count+=bits.size
                pair_bits=ref['pairs'][ix].view(np.uint32)==cand['pairs'][iy].view(np.uint32);pair_equal+=int(pair_bits.sum());pair_count+=pair_bits.size
                if first_pair is None and not pair_bits.all():
                    index=int(np.flatnonzero(~pair_bits)[0]);a_pair=ref['pairs'][ix];b_pair=cand['pairs'][iy]
                    first_pair={'sample':name,'pair':common[index],'reference':float(a_pair[index]),
                        'candidate':float(b_pair[index]),'reference_bits':int(a_pair.view(np.uint32)[index]),
                        'candidate_bits':int(b_pair.view(np.uint32)[index])}
                if first is None and not bits.all():
                    i,j=np.argwhere(~bits)[0];first={'sample':name,'pair':common[int(i)],'predicate_id':int(j),'predicate':names[int(j)],'reference':float(x[i,j]),'candidate':float(y[i,j]),'reference_bits':int(x.view(np.uint32)[i,j]),'candidate_bits':int(y.view(np.uint32)[i,j])}
                histogram+=np.histogram(err,edges)[0];under+=int((err<-radius).sum());over+=int((err>radius).sum())
                stride=max(1,x.size//1200);scatter_x.extend(x.ravel()[::stride].tolist());scatter_y.extend(y.ravel()[::stride].tolist())
                if v>worst_error:
                    worst_error=v;top=np.argsort(abs_err.max(axis=0))[-40:];worst=(name,err[:,top],top,common)
            fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained');ax=axes[0,0];ax.hexbin(scatter_x,scatter_y,gridsize=90,bins='log',mincnt=1,cmap='viridis');lo=min(scatter_x);hi=max(scatter_x);ax.plot([lo,hi],[lo,hi],'r--',lw=1);ax.set_xlabel('official PyTorch-CUDA logits');ax.set_ylabel(k+' logits');ax.set_title('deterministic strided sample from all 50 images')
            ax=axes[0,1];ax.stairs(histogram,edges,fill=True);ax.set_yscale('log');ax.set_xlabel('candidate − official CUDA');ax.set_ylabel('all matched logits');ax.set_title(f'full error range ±{radius:.5g}; outside: {under} / {over}')
            ax=axes[1,0];image_errors=[r['parity']['logits_rmse'] for r in results[m][k]['rows']];ax.plot(image_errors,label='RMSE');ax.plot([r['parity']['logits_max_abs'] for r in results[m][k]['rows']],label='max |error|');ax.set_yscale('log');ax.set_xlabel('image index');ax.legend();ax.grid(alpha=.2)
            ax=axes[1,1];name,matrix,top,pairs=worst;limit=max(float(np.abs(matrix).max()),1e-10);im=ax.imshow(matrix,aspect='auto',cmap='coolwarm',vmin=-limit,vmax=limit);fig.colorbar(im,ax=ax,label='signed logit error');ax.set_title(name+' | 40 predicates with largest error');ax.set_ylabel('matched subject/object pair');ax.set_xticks(range(0,len(top),4),[names[top[i]][:22] for i in range(0,len(top),4)],rotation=55,ha='right',fontsize=6);step=max(1,len(pairs)//8);ax.set_yticks(range(0,len(pairs),step),[str(pairs[i]) for i in range(0,len(pairs),step)],fontsize=7)
            fig.suptitle(m+' / '+k+' | original FP32 reference; weight rounding is included');savefig(fig,dst,m+'-'+k+'-numerics')
            rawstats.append({'model':m,'config':k,'matched_logit_values':count,'equal_bits':equal,'bit_equal_fraction':equal/max(count,1),'pair_logit_bit_equal_fraction':pair_equal/max(pair_count,1),'pair_set_jaccard':summary[m][k]['pair_set_jaccard'],'max_abs_error':maximum,'first_nonidentical_logit':first,'first_nonidentical_pair_logit':first_pair})
    # Keep original-device arithmetic, storage rounding and native same-weight
    # arithmetic separate; these are comparisons, not additive error terms.
    oracles=read(src/'quantization_audit/summary.json')
    components=read(src/'quantization_audit/components/results.json')
    identities['quantization_audit/summary.json']=hashlib.sha256((src/'quantization_audit/summary.json').read_bytes()).hexdigest()
    identities['quantization_audit/components/results.json']=hashlib.sha256((src/'quantization_audit/components/results.json').read_bytes()).hexdigest()
    fig,axes=plt.subplots(3,2,figsize=(16,12),layout='constrained')
    for m,(ax,bax) in zip(MODELS,axes):
        rr=[r for r in oracles if r['model']==m];labels2=[];values=[];bits=[]
        for r in rr:
            for backend in ('cpu','cuda','vulkan'):
                if backend+'_same_weights' not in r:continue
                z=r[backend+'_same_weights'];labels2.append(backend+'-'+r['dtype']);values.append(z['logits_rmse']);bits.append(z['logit_bit_equal_fraction']*100)
        ax.bar(labels2,values,color=[COLORS[k.split('-')[0]] for k in labels2]);ax.set_yscale('log');ax.set_title(m+' | native vs same stored weights, CPU oracle');ax.set_ylabel('matched logits RMSE');ax.tick_params(axis='x',rotation=35)
        bax.bar(labels2,bits,color=[COLORS[k.split('-')[0]] for k in labels2]);bax.set_title('IEEE-754 identical predicate-logit values');bax.set_ylabel('% of matched values');bax.tick_params(axis='x',rotation=35)
    savefig(fig,dst,'same_weight_precision')
    fig,axes=plt.subplots(3,1,figsize=(15,12),layout='constrained')
    for m,ax in zip(MODELS,axes):
        rr=[r for r in components if r['model']==m];x=np.arange(len(rr));a=[r['reference_device_and_runtime']['logits_rmse'] for r in rr];b=[r['storage_only_cpu']['logits_rmse'] for r in rr]
        ax.bar(x-.18,a,.36,label='official CUDA original vs CPU F32 oracle');ax.bar(x+.18,b,.36,label='CPU stored weights vs CPU F32 oracle');ax.set_xticks(x,[r['dtype'] for r in rr]);ax.set_title(m+' | independent comparisons, not additive decomposition');ax.set_ylabel('matched logits RMSE');ax.legend();ax.grid(axis='y',alpha=.2)
    savefig(fig,dst,'reference_and_storage_precision')
    lookup={(r['model'],r['config']):r for r in rawstats}
    for row in export:
        z=lookup.get((row['model'],row['config']),{})
        row['original_cuda_logit_bit_equal_fraction']=z.get('bit_equal_fraction','')
        row['original_cuda_pair_logit_bit_equal_fraction']=z.get('pair_logit_bit_equal_fraction','')
        exact=z.get('bit_equal_fraction')==1 and z.get('pair_logit_bit_equal_fraction')==1 and row['pair_jaccard']==1
        row['exact_precision_and_faster']=bool(exact and row['speedup']>1)
    with (dst/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(export[0]));w.writeheader();w.writerows(export)
    # Embed image thumbnails and predictions: no server, external JS or raw tensor loading.
    images=[]
    for sample in samples:
        with Image.open(args.data/sample['image']) as im:
            im=im.convert('RGB');im.thumbnail((900,900));buf=io.BytesIO();im.save(buf,format='JPEG',quality=88)
        images.append({**{key:sample[key] for key in ('image_id','width','height','source','difficulty','relations')},
            'objects':[{key:o[key] for key in ('bbox','category')} for o in sample['objects']],
            'thumbnail':'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()})
    embedded={m:{k:{'rows':[{key:r[key] for key in ('sample','latency_ms','predictions','parity') if key in r}
                          for r in value['rows']]} for k,value in configs.items()} for m,configs in results.items()}
    payload={'models':MODELS,'summary':summary,'results':embedded,'samples':images,'numerics':rawstats}
    encoded=json.dumps(payload,separators=(',',':')).replace('</','<\\/')
    template=Path(__file__).with_name('alignment_gallery.html').read_text()
    (dst/'index.html').write_text(template.replace('__ALIGNMENT_DATA__',encoded))
    (dst/'numerical_first_divergence.json').write_text(json.dumps(rawstats,indent=2)+'\n')
    (dst/'provenance.json').write_text(json.dumps({'benchmark':str(src),'inputs':identities,'manifest_sha256':hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest(),'report_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'template_sha256':hashlib.sha256(Path(__file__).with_name('alignment_gallery.html').read_bytes()).hexdigest()},indent=2)+'\n')
    figures=sorted(f.name for f in dst.glob('*.png'))
    (dst/'README.md').write_text('# 完整模型、后端与量化对比\n\n[打开离线交互画廊](index.html) · [逐行数据](comparison.csv) · [首个逐位差异](numerical_first_divergence.json)\n\n每个模型、后端与存储精度均包含全部 50 图。箭头表示方向，结果表显示谓词、分数、参考排名和精确字符串 GT 命中。GT 是不完备正例标注，未命中不能据此判为错误。所有耗时是图像到校准三元组的端到端时间，不含模型加载和导出。速度图红色行未满足快于官方 Python 的目标。原始 FP32 和同量化权重 oracle 是两个参考，不能混用。\n\n'+ '\n\n'.join('!['+f+']('+f+')' for f in figures)+'\n')
    print('gallery configurations',sum(len(v)-1 for v in summary.values()),'numeric figures',len(rawstats),'figures',len(figures),flush=True)
if __name__=='__main__':main()
