"""Read returned standalone tensor-probe outputs; never connects to the board."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args()
m=json.loads((ROOT/'prefill8-fixtures.json').read_text());F=Path('/opt/opi-lab/intern-nextopt-20261025/fixtures')
def split(path,meta):
 x=np.fromfile(path,np.float32);offset=0;out={};assert np.isfinite(x).all()
 for item in meta['output_order']:
  end=offset+item['count'];out[item['name']]=x[offset:end].reshape(item['shape']);offset=end
 assert offset==x.size;return out
def metric(x,y):
 x=x.astype(np.float64).ravel();y=y.astype(np.float64).ravel()
 return {'relative_l2':float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-20)),
  'cosine':float(x@y/max(np.linalg.norm(x)*np.linalg.norm(y),1e-20)),
  'max_abs':float(np.max(np.abs(x-y))),'exact_elements':int(np.count_nonzero(x==y)),'elements':x.size}
def valid(d):return {k:v[:,:8] if 'hidden' in k else v[:,:,:8] for k,v in d.items()}
rows=[]
for case in m['cases']:
 c={'id':case['id'],'static_tokens':case['static_tokens'],'body_tokens':case['body_tokens'],'dynamic_template_tokens':8,
    'history':case['history'],'variants':{}};outputs={}
 for label,n in [('t16',16),('t8',8),('t16-repeat',16)]:
  d=a.results/case['id']/label
  if not (d/'output.f32').exists():continue
  meta=case['variants'][str(n)];out=split(d/'output.f32',meta);ref=split(F/case['id']/f't{n}'/'reference.f32',meta)
  tm=json.loads((d/'timing.json').read_text());times=np.asarray(tm['api_seconds']);assert len(times)==tm['repeats']
  outputs[label]=valid(out);v={'timing':tm,'api_p50_seconds':float(np.median(times)),
   'api_p95_seconds':float(np.percentile(times,95)),'all_output_vs_exported_weight_fp32':{k:metric(out[k],ref[k]) for k in out},
   'valid8_vs_exported_weight_fp32':{k:metric(valid(out)[k],valid(ref)[k]) for k in out},
   'output_sha256':hashlib.sha256((d/'output.f32').read_bytes()).hexdigest()}
  c['variants'][label]=v
 if 't8' in outputs and 't16' in outputs:
  c['t8_valid8_vs_t16']={k:metric(outputs['t8'][k],outputs['t16'][k]) for k in outputs['t8']}
  c['api_reduction_fraction']=1-c['variants']['t8']['api_p50_seconds']/c['variants']['t16']['api_p50_seconds']
 if 't16-repeat' in outputs:
  c['t16_repeat_vs_t16']={k:metric(outputs['t16-repeat'][k],outputs['t16'][k]) for k in outputs['t16']}
 rows.append(c)
report={'scope':m['history_scope'],'pair_scope':m['pair_scope'],
 'timing_scope':'Single fused two-layer vip_run_network API only; excludes load/prepare/flush/invalidate/input conversion and host dispatch, not full-stack TTFT.',
 'cases':rows,'full_stack_approval':False,'notes':'Weight quantization identical, but different GEMM/query shapes may alter rounding. Full-stack natural-output/logits validation still required.'}
(a.results/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
for c in rows:print(c['id'],{k:round(v['api_p50_seconds']*1000,3) for k,v in c['variants'].items()},'pair worst L2',max([x['relative_l2'] for x in c.get('t8_valid8_vs_t16',{}).values()] or [0]))
