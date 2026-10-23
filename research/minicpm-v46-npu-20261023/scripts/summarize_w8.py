from pathlib import Path
import json,statistics,re,hashlib
r=Path(__file__).resolve().parents[1];out={'scope':'short text-only NPU microbenchmarks; not full vision or speech','runtime':{},'telemetry':{}}
for name in ['w8cal','w8scaled','w8grouped','w8fixed','w8fixed-linear','w8fixed-attn']:
 p=r/'results'/(name+'-runtime.jsonl')
 if not p.exists():continue
 rows=[json.loads(x) for x in p.read_text().splitlines()];ready=next(x for x in rows if x['event']=='ready');v=[x for x in rows if x['event']=='round' and x['round']>0]
 out['runtime'][name]={'warm_rounds':len(v),'ttft_median':statistics.median(x['ttft_seconds'] for x in v),'decode_median':statistics.median(x['decode_tokens_per_second'] for x in v),'load':ready['load_seconds'],'driver_bytes':ready['npu_bytes'],'cpu_cores':statistics.median(x['cpu_seconds']/x['elapsed_seconds'] for x in v),'closed_zero':rows[-1].get('event')=='closed' and rows[-1].get('npu_bytes')==0,'tokens_identical_all_rounds':len({tuple(x['ids']) for x in rows if x['event']=='round'})==1}
for name in ['intern-comparison','intern-same-text']:
 rows=[json.loads(x) for x in (r/'results'/(name+'.jsonl')).read_text().splitlines()];v=[x['details'] for x in rows if x['event']=='round' and x['round']>0]
 out['runtime'][name]={'warm_rounds':len(v),'prompt_tokens':v[0]['prompt_tokens'],'ttft_median':statistics.median(x['ttft_seconds'] for x in v),'decode_median':statistics.median(x['decode_tokens_per_second'] for x in v),'cpu_cores':statistics.median(x['cpu_seconds']/x['total_seconds'] for x in v),'load':rows[0]['details']['load_seconds'],'driver_bytes':rows[0]['details']['driver_bytes'],'closed_zero':rows[-1]['details']['driver_bytes']==0}
for p in (r/'results').glob('w8fixed-*-telemetry.jsonl'):
 rows=[json.loads(x) for x in p.read_text().splitlines()];first=list(map(int,rows[0]['cpu_stat'].split()[1:9]));last=list(map(int,rows[-1]['cpu_stat'].split()[1:9]));diff=[b-a for a,b in zip(first,last)];total=sum(diff)
 def mem(x,k):return int(re.search('^'+k+':\s*(\d+)',x['meminfo'],re.M)[1])*1024
 out['telemetry'][p.name]={'samples':len(rows),'duration_seconds':rows[-1]['at']-rows[0]['at'],'allzone_peak_c':max(v for x in rows for v in x['temperatures_mC'].values())/1000,'available_min':min(mem(x,'MemAvailable') for x in rows),'swap_used_start':mem(rows[0],'SwapTotal')-mem(rows[0],'SwapFree'),'swap_used_end':mem(rows[-1],'SwapTotal')-mem(rows[-1],'SwapFree'),'system_cpu_percent_excluding_iowait':100*(total-diff[3]-diff[4])/total,'iowait_percent':100*diff[4]/total,'includes_load_and_reset':True,'runtime_monitor_exit_race':p.name=='w8fixed-runtime-telemetry.jsonl'}
(r/'results/w8-final-summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
