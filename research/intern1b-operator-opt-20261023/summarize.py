from pathlib import Path
import json,statistics as st,numpy as np
from tokenizers import Tokenizer
r=Path(__file__).resolve().parent;run=sorted((r/'results').iterdir())[-1];rows=[json.loads(l) for l in (run/'events.jsonl').read_text().splitlines()]
def group(name):return [x for x in rows if x.get('event')==name]
by={};base={}
for x in group('measure'):by.setdefault((x['variant'],x['case']),[]).append(x)
perf=[]
for (variant,case),rs in by.items():
 def med(k):return st.median(x['result'][k] for x in rs)
 z=dict(variant=variant,case=case,n=len(rs),prompt=rs[0]['result']['prompt_tokens'],reused=rs[0]['result']['reused_tokens'],ttft=med('ttft_seconds'),decode=med('decode_tokens_per_second'),cpu_cores=st.median(x['result']['cpu_seconds']/x['result']['total_seconds'] for x in rs),setup=med('setup_seconds'),prefill=med('prefill_seconds'),scalar=med('scalar_prompt_seconds'),head=med('first_head_seconds'),ttft_min=min(x['result']['ttft_seconds'] for x in rs),ttft_max=max(x['result']['ttft_seconds'] for x in rs))
 perf.append(z)
 if variant=='baseline':base[case]=z
for x in perf:x['ttft_gain_percent']=100*(1-x['ttft']/base[x['case']]['ttft']);x['decode_gain_percent']=100*(x['decode']/base[x['case']]['decode']-1)
tok=Tokenizer.from_file('/opt/opi-lab/source-validation-20261017/official-InternVL3-1B/tokenizer.json')
qual={}
for x in group('quality'):
 ids=x['result']['tokens'];qual.setdefault(x['case'],{})[x['variant']]=dict(tokens=ids,text=tok.decode(ids,skip_special_tokens=True))
for case,vs in qual.items():
 for v,x in vs.items():x['same_tokens_as_baseline']=x['tokens']==vs['baseline']['tokens']
diags={};checks=[]
for x in group('diagnostic'):
 p=run/Path(x['dump']).parent.name/Path(x['dump']).name;a=np.fromfile(p,dtype='<f4').reshape(-1,151674);diags[(x['variant'],x['case'])]=a
for (v,c),a in diags.items():
 b=diags[('baseline',c)];shift=lambda x:x-x.max(axis=1,keepdims=True)
 la=shift(a);lb=shift(b);la-=np.log(np.exp(la).sum(axis=1,keepdims=True));lb-=np.log(np.exp(lb).sum(axis=1,keepdims=True))
 checks.append(dict(variant=v,case=c,bitwise=bool(np.array_equal(a,b)),relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(b)),argmax_same=int(np.sum(a.argmax(1)==b.argmax(1))),steps=len(a),kl_base_to_candidate=float(np.mean((np.exp(lb)*(lb-la)).sum(1)))))
samples=group('telemetry')[-1]['samples'];samples=[x for x in samples if 'error' not in x]
resource=dict(samples=len(samples),max_temp=max(v for x in samples for v in x['temperature'].values()),max_rss=sum([]) if not samples else max(sum(p['rss'] for p in x['processes']) for x in samples),max_pss=max(sum(p['pss'] for p in x['processes']) for x in samples),available_min=min(x['available'] for x in samples),swap_in_delta=samples[-1]['swap_in']-samples[0]['swap_in'],swap_out_delta=samples[-1]['swap_out']-samples[0]['swap_out'],npu_frequency=sorted(set(x['npu_frequency'] for x in samples)))
out=dict(complete=bool(group('complete')),errors=group('error'),performance=perf,quality=qual,diagnostic=checks,resources=resource,closed=group('closed'),run=run.name)
(r/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['quality']},ensure_ascii=False,indent=2))
