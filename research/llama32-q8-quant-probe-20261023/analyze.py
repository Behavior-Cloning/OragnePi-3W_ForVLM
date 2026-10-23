"""Analyze owner-returned raw files. Never connects to board."""
from pathlib import Path
import json,argparse,re,numpy as np
from model import ROOT,BUILD
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--manifest',type=Path,default=ROOT/'fixtures-manifest.json');p.add_argument('--build',type=Path,default=BUILD);a=p.parse_args();BUILD=a.build;meta=json.loads(a.manifest.read_text());res={}
def error(z,y):
 return dict(relative_l2=float(np.linalg.norm(z-y)/max(np.linalg.norm(y),1e-20)),max_abs=float(abs(z-y).max()),cosine=float(np.dot(z,y)/max(np.linalg.norm(z)*np.linalg.norm(y),1e-20)))
def logs(d):
 rows=[]
 for p in d.glob('*.log'):
  for line in p.read_text().splitlines():
   if line.startswith('{'):
    try:rows.append(json.loads(line))
    except json.JSONDecodeError:pass
 return rows
def query_io(d):
 records=[]
 for p in d.glob('*.log'):
  for line in p.read_text().splitlines():
   m=re.match(r'(in|out)\[(\d+)\] (\S+) size=(\d+) dtype=(\d+) quant=(\d+) fl=(-?\d+) shape=(.*)',line)
   if m:
    direction,index,name,size,dtype,quant,fl,shape=m.groups();records.append(dict(direction=direction,index=int(index),name=name,count=int(size),data_format={0:'FP32',1:'FP16',5:'INT16'}.get(int(dtype),str(dtype)),quant_format=int(quant),fl=int(fl),scale=2.**-int(fl) if int(quant)==1 else None,shape=shape))
 return records
for variant in ['layer0-allw8','layer0-mixed']:
 d=a.results/variant
 if not d.exists():continue
 rows=logs(d);records=[]
 for rec in meta['decoder_fixtures']:
  for oi in range(3):
   f=d/f'p{rec["fixture_id"]}-out{oi}.f32'
   if not f.exists():continue
   z=np.fromfile(f,dtype='<f4');y=np.fromfile(BUILD/'fixtures'/f'p{rec["fixture_id"]}-ref{oi}.f32',dtype='<f4');assert np.isfinite(z).all() and z.shape==y.shape
   records.append(dict(fixture_id=rec['fixture_id'],split=rec['split'],seen_in_calibration=rec['seen_in_calibration'],output=oi,**error(z,y)))
 timing=[x['run_seconds'] for x in rows if x.get('event')=='timing' and x['repeat']>0]
 groups={}
 for split in ['cal','heldout','heldout_unseen']:
  for oi in range(3):
   xx=[x['relative_l2'] for x in records if (x['split']==split or (split=='heldout_unseen' and x['split']=='heldout' and not x['seen_in_calibration'])) and x['output']==oi]
   if xx:groups[f'{split}_output{oi}']=dict(count=len(xx),min=float(min(xx)),max=float(max(xx)),median=float(np.median(xx)))
 res[variant]=dict(actual_io=query_io(d),complete=len(records)==120 and any(x.get('event')=='closed' and x['driver_released']==0 for x in rows),timing_samples=len(timing),hot_run_p50=float(np.median(timing)) if timing else None,hot_run_p95=float(np.percentile(timing,95)) if timing else None,errors=groups,records=records,closed=[x for x in rows if x.get('event')=='closed'])
head=[]
for rec in meta['head_fixtures']:
 fid=rec['fixture_id'];fs=[a.results/f'head{j}-w8'/f'h{fid}-s{j}-output.f32' for j in range(4)]
 if not all(f.exists() for f in fs):continue
 z=np.concatenate([np.fromfile(f,dtype='<f4') for f in fs]);y=np.concatenate([np.fromfile(BUILD/'fixtures'/f'h{fid}-s{j}-reference.f32',dtype='<f4') for j in range(4)]);assert z.size==128256 and np.isfinite(z).all();top=np.argsort(-y)[:2]
 head.append(dict(fixture_id=fid,split=rec['split'],**error(z,y),argmax=int(z.argmax()),reference_argmax=int(y.argmax()),argmax_equal=bool(z.argmax()==y.argmax()),reference_top1_margin=float(y[top[0]]-y[top[1]]),reference_token_rank_in_npu=int(np.sum(z>z[y.argmax()]))+1))
head_logs={j:logs(a.results/f'head{j}-w8') for j in range(4)}
head_closed={j:[x for x in rows if x.get('event')=='closed'] for j,rows in head_logs.items()}
head_timing={j:{(x['fixture_id'],x['repeat']):x['run_seconds'] for x in rows if x.get('event')=='timing' and x['repeat']>0} for j,rows in head_logs.items()}
shared=set.intersection(*(set(x) for x in head_timing.values()))
sums=[sum(head_timing[j][k] for j in range(4)) for k in shared]
res['full_vocabulary_head']=dict(complete=len(head)==16 and all(any(x.get('driver_released')==0 for x in rows) for rows in head_closed.values()),closed=head_closed,actual_io={j:query_io(a.results/f'head{j}-w8') for j in range(4)},shard_run_sum_samples=len(sums),shard_run_sum_p50=float(np.median(sums)) if sums else None,shard_run_sum_p95=float(np.percentile(sums,95)) if sums else None,timing_note='Sum of matching frozen-state shard runs from separate processes; not full model decode speed',records=head,cal_agreement=sum(x['argmax_equal'] for x in head if x['split']=='cal'),heldout_agreement=sum(x['argmax_equal'] for x in head if x['split']=='heldout'),note='Frozen Q8 all16 FP32 hidden states; not accumulated NPU generation quality')
(a.results/'summary.json').write_text(json.dumps(res,indent=2));print(json.dumps({k:{x:y for x,y in v.items() if x!='records'} for k,v in res.items()},indent=2))
