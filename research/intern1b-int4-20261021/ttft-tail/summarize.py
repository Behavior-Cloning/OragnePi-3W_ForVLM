from pathlib import Path
import json,statistics,hashlib
import numpy as np
from tokenizers import Tokenizer
r=Path(__file__).resolve().parent;parent=r.parent
p=sorted((r/'results').glob('*/events.jsonl'))[-1];rows=[json.loads(s) for s in p.read_text().splitlines()]
tok=Tokenizer.from_file('/opt/opi-lab/source-validation-20261017/official-InternVL3-1B/tokenizer.json')
def stats(xs):return {'n':len(xs),'p50':statistics.median(xs),'p95':float(np.quantile(xs,.95)),'min':min(xs),'max':max(xs)}
summary={'complete':any(x['event']=='complete' for x in rows),'errors':[x for x in rows if x['event']=='error'],'performance':{},'quality':[],'diagnostic':{},'closed':[x for x in rows if x['event']=='closed'],'result_file':str(p)}
for v in ['w8','w4']:
 summary['performance'][v]={}
 for mode in [0,8]:
  summary['performance'][v][str(mode)]={}
  for case in ['dynamic-64','dynamic-128','dynamic-256']:
   group=[x for x in rows if x['event']=='measure' and x['variant']==v and x['mode']==mode and x['case']==case];assert len(group)==5
   fields=['ttft_seconds','setup_seconds','prefill_seconds','scalar_prompt_seconds','first_head_seconds','cpu_seconds','batch_calls','tail_rows','scalar_prompt_tokens','driver_bytes']
   summary['performance'][v][str(mode)][case]={key:stats([x['result'][key] for x in group]) for key in fields}
   assert all(x['result']['reused_tokens']==32 and x['result']['generated_tokens']==1 for x in group)
for x in rows:
 if x['event']=='quality':summary['quality'].append({'variant':x['variant'],'mode':x['mode'],'case':x['case'],'text':tok.decode(x['result']['tokens']),'tokens':x['result']['tokens'],'ttft':x['result']['ttft_seconds']})
def load(x):
 q=p.parent/Path(x['dump']).parent.name/Path(x['dump']).name
 return np.fromfile(q,np.float32).reshape(16,151674)
def compare(a,b):
 exact=bool(np.array_equal(a,b));a=a.astype(np.float64);b=b.astype(np.float64)
 logs=[]
 for x in [a,b]:
  z=x-x.max(axis=1,keepdims=True);logs.append(z-np.log(np.exp(z).sum(axis=1,keepdims=True)))
 kl=(np.exp(logs[0])*(logs[0]-logs[1])).sum(axis=1);cos=(a*b).sum(axis=1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-30)
 return {'bit_identical':exact,'finite':bool(np.isfinite(b).all()),'relative_l2':float(np.linalg.norm(a-b)/np.linalg.norm(a)),'mean_kl_nats':float(kl.mean()),'argmax_matches':int((a.argmax(axis=1)==b.argmax(axis=1)).sum()),'steps':len(a),'min_cosine':float(cos.min())}
diag={(x['variant'],x['mode'],x['case']):x for x in rows if x['event']=='diagnostic'}
for v in ['w8','w4']:
 summary['diagnostic'][v]={}
 for case in ['dynamic-64','dynamic-128','dynamic-256']:
  summary['diagnostic'][v][case]=compare(load(diag[(v,0,case)]),load(diag[(v,8,case)]))
old=json.loads((parent/'summary.json').read_text());oldp=Path(old['result_file']);oldrows=[json.loads(x) for x in oldp.read_text().splitlines()]
summary['instrumentation_control']={}
for v in ['w8','w4']:
 oldd=next(x for x in oldrows if x['event']=='diagnostic' and x['variant']==v);oldfile=oldp.parent/Path(oldd['dump']).parent.name/Path(oldd['dump']).name
 a=np.fromfile(oldfile,np.float32).reshape(64,151674)[:16];b=load(diag[(v,0,'dynamic-64')]);summary['instrumentation_control'][v]=compare(a,b)
summary['speedups']={}
for v in ['w8','w4']:
 summary['speedups'][v]={}
 for case in ['dynamic-64','dynamic-128','dynamic-256']:
  a=summary['performance'][v]['0'][case]['ttft_seconds']['p50'];b=summary['performance'][v]['8'][case]['ttft_seconds']['p50'];summary['speedups'][v][case]={'reduction_percent':100*(1-b/a),'saved_seconds':a-b}
quality=summary['quality'];summary['natural_output_identical']={case:next(x['tokens'] for x in quality if x['mode']==0 and x['case']==case)==next(x['tokens'] for x in quality if x['mode']==8 and x['case']==case) for case in ['daily-'+str(i) for i in range(5)]}
telemetry=next(x['samples'] for x in rows if x['event']=='telemetry');valid=[x for x in telemetry if 'temperature' in x];summary['telemetry']={'temperature_max':max(x['temperature'] for x in valid),'available_min':min(x['available'] for x in valid),'swap_in_delta':max(x['swap_in'] for x in valid)-min(x['swap_in'] for x in valid),'swap_out_delta':max(x['swap_out'] for x in valid)-min(x['swap_out'] for x in valid)}
assert summary['complete'] and not summary['errors'] and len(summary['closed'])==8 and all(x['result']['driver_bytes']==0 for x in summary['closed'])
(r/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ['quality','performance','closed']},ensure_ascii=False,indent=2))
