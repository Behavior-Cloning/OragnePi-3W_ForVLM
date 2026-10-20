import json,statistics
from pathlib import Path
r=Path(__file__).resolve().parent
from summarize_results import percentile
runs=[]
for p in sorted((r/'results/board').glob('opt-*.jsonl')):
 rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()];load=next((x for x in rows if x.get('event')=='loaded'),None)
 if not load:continue
 timed=[x for x in rows if x.get('event')=='opt_measure'];quality=[x for x in rows if x.get('event')=='opt_quality'];sample=[x for z in rows if z.get('event')=='telemetry' for x in z.get('samples',[]) if 'temp' in x]
 model=p.name[4:].split('-a76')[0].split('-a55')[0]
 z=dict(file=p.name,model=model,label=load['label'],loaded=load,complete=any(x.get('event')=='complete' for x in rows),errors=[x for x in rows if x.get('event')=='error'],n=len(timed),p50=percentile([x['seconds'] for x in timed],.5),p95=percentile([x['seconds'] for x in timed],.95),timing_correct=sum(x['correct'] for x in timed),quality_n=len(quality),quality_correct=sum(x['correct'] for x in quality),answers={x['case_id']:x['answer'] for x in quality},cpu_cores=statistics.mean(x['cpu_seconds']/x['seconds'] for x in timed) if timed else None)
 if sample:z['resource']=dict(rss_max=max(x['rss'] for x in sample),pss_max=max(x.get('pss',0) for x in sample),temperature_max=max(x['temp'] for x in sample),available_min=min(x['available'] for x in sample))
 runs.append(z)
for z in runs:
 base=next((x for x in reversed(runs) if x['model']==z['model'] and x['label']=='a76-t2' and x['complete']),None)
 if base and base['p50'] and z['p50']:
  shared=set(base['answers'])&set(z['answers']);z['baseline_file']=base['file'];z['speedup']=base['p50']/z['p50'];z['paired_quality_n']=len(shared);z['changed_decisions']=sum(base['answers'][k]!=z['answers'][k] for k in shared);z['quality_correct_delta']=z['quality_correct']-base['quality_correct'] if set(base['answers'])==set(z['answers']) else None
(r/'results/optimization-comparison.json').write_text(json.dumps(runs,ensure_ascii=False,indent=2))
print(json.dumps([{k:v for k,v in x.items() if k not in ['loaded','answers']} for x in runs],ensure_ascii=False,indent=2))
