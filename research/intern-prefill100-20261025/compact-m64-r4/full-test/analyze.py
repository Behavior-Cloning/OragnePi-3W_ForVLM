from pathlib import Path
import json, hashlib, statistics, math
import numpy as np
R=Path(__file__).resolve().parent
read=lambda n:[json.loads(x) for x in (R/n).read_text().splitlines()]
E=read('results/events.jsonl');res=read('results/resources.jsonl')
status=json.loads((R/'results/status.json').read_text());control=json.loads((R/'controller.json').read_text())
assert control['returncode']==0 and status['status']=='complete' and not status['errors']
variants=['parent','compact','parent-repeat']
def stat(x):
 a=np.asarray(x,dtype=float)
 return {'n':len(a),'min':float(a.min()),'p50':float(np.median(a)),'p95_nearest_rank':float(sorted(a)[math.ceil(.95*len(a))-1]),'p95_linear':float(np.percentile(a,95)),'max':float(a.max())}
out={'scope':'Intern NPU text; synthetic streaming frontend; RAG resident but not queried; no images or motors', 'performance':[], 'diagnostics':[], 'quality':{},'resources':{},'counts':{k:sum(e['event']==k for e in E) for k in sorted(set(e['event'] for e in E))},'controller':control,'audio':status['audio']}
measures=[e for e in E if e['event']=='measure'];assert len(measures)==117
for v in variants:
 for s,d in json.loads((R/'inputs.json').read_text())['performance_grid']:
  a=[e for e in measures if e['variant']==v and e['system_total']==s and e['dynamic_body']==d];assert len(a)==3
  for e in a:
   m=e['metrics'];assert m['generated_tokens']==32 and m['fixed_feedback'] and not m['diagnostic']
   assert m['new_tokens']==d+8 and m['reused_tokens']==s and m['prompt_tokens']==s+d+8
   assert m['scalar_prompt_tokens']==0 and m['scalar_prompt_seconds']==0
   assert m['true_tail8_calls']==1 and m['compact_calls']==(min((d+8)//64,(128-s)//64+1) if v=='compact' else 0)
   assert abs(m['decode_tokens_per_second']-31/(m['total_seconds']-m['ttft_seconds']))<.001
  z={'variant':v,'system_body':s-5,'system_template':5,'system_total':s,'dynamic_body':d,'dynamic_template':8,'new_prefill':d+8,'reused':s,'total_prompt':s+d+8,'output':32}
  for k in ['setup_seconds','prefill_seconds','first_head_seconds','ttft_seconds','decode_tokens_per_second','total_seconds','true_tail8_seconds','compact_seconds','batch_calls','compact_calls']:
   z[k]=stat([e['metrics'][k] for e in a])
  z['lt1s']=sum(e['metrics']['ttft_seconds']<1 for e in a);z['lt500ms']=sum(e['metrics']['ttft_seconds']<.5 for e in a)
  out['performance'].append(z)
D={}
for e in [e for e in E if e['event']=='diagnostic']:
 m=e['metrics'];assert m['diagnostic'] and m['fixed_feedback'] and m['generated_tokens']==4
 p=R/'results'/e['path'];raw=p.read_bytes();assert len(raw)==2426784
 digest=hashlib.sha256(raw).hexdigest();assert digest==e['sha256'];arr=np.frombuffer(raw,dtype='<f4');assert np.isfinite(arr).all()
 D[e['variant'],e['case']]=arr
 out['diagnostics'].append({'variant':e['variant'],'case':e['case'],'path':e['path'],'bytes':len(raw),'sha256':digest,'finite':True})
assert len(D)==18
for d in out['diagnostics']:
 a=D[d['variant'],d['case']];b=D['parent',d['case']]
 d['same_bytes_as_original']=bool(np.array_equal(a.view(np.uint8),b.view(np.uint8)))
 d['relative_l2']=float(np.linalg.norm(a.astype(np.float64)-b)/np.linalg.norm(b.astype(np.float64)))
Q={(e['variant'],e['case']):e for e in E if e['event']=='quality'}
for v in variants:
 rows=[]
 for (variant,case),e in Q.items():
  if variant!=v:continue
  ref=Q['parent',case]
  rows.append({'case':case,'text':e['text'],'reference':ref['text'],'equal_sequence':e['metrics']['tokens']==ref['metrics']['tokens'],'system_total':e['system_total'],'dynamic_body':e['dynamic_body'],'dynamic_template':8,'actual_new':e['metrics']['new_tokens'],'output_tokens':e['metrics']['generated_tokens'],'expected_action':e['expected_action'],'forbidden_actions':e['forbidden_actions']})
 assert len(rows)==12
 out['quality'][v]={'equal':sum(x['equal_sequence'] for x in rows),'n':len(rows),'cases':rows}
# Pairwise fixed-feedback output equality does not assess semantic correctness.
out['fixed_feedback_equal']={v:sum(e['metrics']['tokens']==next(x['metrics']['tokens'] for x in measures if x['variant']=='parent' and x['case']==e['case']) for e in measures if e['variant']==v) for v in variants}
CPU=[]
for prev,cur in zip(res,res[1:]):
 a=list(map(int,prev['cpu'][0].split()[1:9]));b=list(map(int,cur['cpu'][0].split()[1:9]));d=[y-x for x,y in zip(a,b)];total=sum(d)
 if total<=0:continue
 busy=total-d[3]-d[4]
 CPU.append({'at':cur['at'],'variant':cur['variant'],'phase':cur['phase'],'system_busy_pct':100*busy/total,'system_iowait_pct':100*d[4]/total,'one_core_equivalent_pct':800*busy/total})
for v in ['all']+variants:
 rows=res if v=='all' else [r for r in res if r['variant']==v]
 cp=CPU if v=='all' else [r for r in CPU if r['variant']==v]
 x={'samples':len(rows),'span_s':rows[-1]['at']-rows[0]['at'],'temperature_C':stat([r['temp_mC']/1000 for r in rows]),'cpu_busy_pct_8cores':stat([r['system_busy_pct'] for r in cp]),'iowait_pct':stat([r['system_iowait_pct'] for r in cp]),'load1':stat([float(r['loadavg'].split()[0]) for r in rows]),'available_MiB':stat([r['meminfo']['MemAvailable']/1024 for r in rows]),'swap_used_MiB':stat([(r['meminfo']['SwapTotal']-r['meminfo']['SwapFree'])/1024 for r in rows]),'sum_process_RSS_MiB':stat([sum(p.get('VmRSS',0) for p in r['processes'].values())/1024 for r in rows]),'sum_process_PSS_MiB':stat([sum(p.get('Pss',0) for p in r['processes'].values())/1024 for r in rows])}
 x['swap_in_pages_delta']=rows[-1]['vmstat']['pswpin']-rows[0]['vmstat']['pswpin'];x['swap_out_pages_delta']=rows[-1]['vmstat']['pswpout']-rows[0]['vmstat']['pswpout']
 x['audio_frames_delta']=rows[-1]['frontend']['frames']-rows[0]['frontend']['frames'];x['audio_overruns_delta']=rows[-1]['frontend']['overruns']-rows[0]['frontend']['overruns'];x['max_audio_lag_cumulative_s']=max(r['frontend']['max_lag_s'] for r in rows)
 x['cpu_frequency_khz']={c:sorted(set(r['freq_khz'].get(c) for r in rows if c in r['freq_khz'])) for c in rows[0]['freq_khz']}
 out['resources'][v]=x
out['closed']=[e for e in E if e['event']=='closed'];assert len(out['closed'])==3 and all(e['metrics']['driver_bytes']==0 for e in out['closed'])
out['snapshot_audio_caveat']='overruns are frontend processing-frame budget failures, max_lag is cumulative since start; final queue=1 is queued playback input, not established ASR backlog'
out['evidence_sha256']={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'config.json',R/'inputs.json',R/'deployment-sha.json',R/'board_matrix.py',R/'results/events.jsonl',R/'results/resources.jsonl',R/'results/status.json',R/'results/post-health.txt']}
assert all(x['same_bytes_as_original'] for x in out['diagnostics'] if x['variant']=='parent-repeat')
assert out['fixed_feedback_equal']['parent-repeat']==39 and out['quality']['parent-repeat']['equal']==12
out['driver_ready']={e['variant']:e['metrics']['driver_bytes'] for e in E if e['event']=='ready'}
prior_root=R.parent.parent/'down-outlier-split-r2/full-test'
prior=json.loads((prior_root/'ANALYSIS.json').read_text())
old=next(x for x in prior['diagnostics'] if x['variant']=='outlier8' and x['case']=='s128-d256-0')
current=next(x for x in out['diagnostics'] if x['variant']=='parent' and x['case']=='s128-d256-0')
out['old_runtime_control']={'prior_report':'down-outlier-split-r2/full-test/ANALYSIS.json','case':'s128-d256-0','prior_sha256':old['sha256'],'current_sha256':current['sha256'],'same_bytes':old['sha256']==current['sha256']}
assert out['old_runtime_control']['same_bytes'],'disabled compact runtime changed existing parent result'
(R/'ANALYSIS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
for s,d in json.loads((R/'inputs.json').read_text())['performance_grid']:
 rows=[next(x for x in out['performance'] if x['variant']==v and x['system_total']==s and x['dynamic_body']==d) for v in variants]
 print(s,d,[(x['variant'],round(x['prefill_seconds']['p50']*1000,2),round(x['ttft_seconds']['p50']*1000,2),round(x['decode_tokens_per_second']['p50'],2)) for x in rows])
print('PARITY',[(x['variant'],x['case'],x['same_bytes_as_original']) for x in out['diagnostics']]);print('QUALITY', {v:x['equal'] for v,x in out['quality'].items()})
print('RESOURCES',json.dumps(out['resources']['all'],ensure_ascii=False));print('FIXED',out['fixed_feedback_equal'])
