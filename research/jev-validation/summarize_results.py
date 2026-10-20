import json,statistics,collections
from pathlib import Path
r=Path(__file__).resolve().parent

def percentile(a,q):
 a=sorted(a)
 if not a:return None
 x=(len(a)-1)*q;i=int(x);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)
def stats(a):return dict(n=len(a),p50=percentile(a,.5),p95=percentile(a,.95),minimum=min(a) if a else None,maximum=max(a) if a else None)
summary={}
for f in sorted((r/'results/board').glob('*.jsonl')):
 rows=[]
 for line in f.read_text().splitlines():
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:pass
 blocks=collections.defaultdict(list)
 for z in rows:
  if 'block' in z:blocks[(z['event'],z['block'])].append(z)
 rec=dict(events=len(rows),completed=any(x.get('event')=='complete' for x in rows),errors=[z for z in rows if 'error' in z],blocks={},loaded=[x for x in rows if x.get('event')=='loaded'],native_controls=[x for x in rows if x.get('event')=='native_control'])
 continuous=[x for z in rows if z.get('event')=='telemetry' for x in z.get('samples',[]) if 'rss_tree_bytes' in x]
 if continuous:rec['continuous_resources']=dict(samples=len(continuous),rss_tree_max=max(x['rss_tree_bytes'] for x in continuous),available_min=min(x['system_available_bytes'] for x in continuous),swap_max=max(x['swap_used_bytes'] for x in continuous),temp_max=max(v for x in continuous for v in x['temperature_c'].values()))
 for (event,block),items in blocks.items():
  valid=[x for x in items if 'seconds' in x];res=[x['resource'] for x in items if 'resource' in x]
  z=dict(attempted=len(items),valid=len(valid),errors=len(items)-len(valid),seconds=stats([x['seconds'] for x in valid]),accuracy=sum(x.get('correct',False) for x in valid)/len(valid) if valid else None,state_tokens=sorted(set(x.get('state_tokens') for x in valid)))
  z['by_suite']={suite:dict(n=len(a),correct=sum(x.get('correct',False) for x in a)) for suite in sorted(set(x.get('suite') for x in valid)-{None}) for a in [[x for x in valid if x.get('suite')==suite]]}
  if any(x.get('seconds') and 'cpu_seconds' in x for x in valid):z['process_cpu_cores_mean']=statistics.mean(x['cpu_seconds']/x['seconds'] for x in valid if x.get('seconds') and 'cpu_seconds' in x)
  if res:z['resources']=dict(rss_tree_max=max(x['rss_tree_bytes'] for x in res),available_min=min(x['system_available_bytes'] for x in res),swap_max=max(x['swap_used_bytes'] for x in res),temp_max=max(v for x in res for v in x['temperature_c'].values()))
  rec['blocks'][event+':'+block]=z
 if f.name.startswith('opt-'):
  timed=[x for x in rows if x.get('event')=='opt_measure'];quality=[x for x in rows if x.get('event')=='opt_quality']
  samples=[x for z in rows if z.get('event')=='telemetry' for x in z.get('samples',[]) if 'temp' in x]
  rec['optimization']=dict(seconds=stats([x['seconds'] for x in timed]),timing_accuracy=sum(x['correct'] for x in timed)/len(timed) if timed else None,quality_n=len(quality),quality_correct=sum(x['correct'] for x in quality),quality_by_suite={k:dict(n=sum(x['suite']==k for x in quality),correct=sum(x['correct'] for x in quality if x['suite']==k)) for k in sorted(set(x['suite'] for x in quality))},answers={x['case_id']:x['answer'] for x in quality},cpu_cores_mean=statistics.mean(x['cpu_seconds']/x['seconds'] for x in timed) if timed else None)
  if samples:rec['optimization']['resources']=dict(rss_max=max(x['rss'] for x in samples),pss_max=max(x.get('pss',0) for x in samples),available_min=min(x['available'] for x in samples),temp_max=max(x['temp'] for x in samples),swap_max=max(x['swap'] for x in samples),swap_in_delta=max(x.get('swap_in',0) for x in samples)-min(x.get('swap_in',0) for x in samples),swap_out_delta=max(x.get('swap_out',0) for x in samples)-min(x.get('swap_out',0) for x in samples),sample_count=len(samples))
  rec['provenance']=[x for x in rows if x.get('event')=='provenance']
 if f.name.startswith('co-resident'):
  pairs=collections.defaultdict(list)
  for z in rows:
   if z.get('event')=='pair_measure':pairs[(z['llm'],z['jev'],z['mode'])].append(z)
  rec['pairs']={}
  for k,items in pairs.items():
   j=[x['result']['jev']['seconds'] for x in items if 'jev' in x['result']];ll=[x['result']['llm'] for x in items if 'llm' in x['result']]
   rec['pairs']['/'.join(k)]=dict(n=len(items),jev_seconds=stats(j),llm_ttft_seconds=stats([x['ttft_seconds'] for x in ll if x.get('ttft_seconds') is not None]),decode_tokens_per_second=stats([x.get('decode_tokens_per_second',x.get('final',{}).get('timings',{}).get('predicted_per_second')) for x in ll if x.get('decode_tokens_per_second',x.get('final',{}).get('timings',{}).get('predicted_per_second')) is not None]),overlap_seconds=stats([x['overlap_seconds'] for x in items if x.get('overlap_seconds') is not None]),available_min=min(x['after']['available'] for x in items))
 summary[f.name]=rec
(r/'results/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))
