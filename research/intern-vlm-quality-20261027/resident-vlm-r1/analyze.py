from pathlib import Path
import json,statistics as st,re,hashlib
P=Path(__file__).resolve().parent;j=json.loads((P/'results/summary.json').read_text());assert j['status']=='complete' and not j['monitor_errors'];assert j['language_closed']['driver_bytes']==0 and j['vision_closed']['event']=='closed'
rows=[x for x in j['records'] if not x['warmup']];assert len(rows)==12
fields=['preprocess_seconds','vision_wall_seconds','embedding_read_tokenize_seconds','language_first_event_seconds','end_to_end_first_seconds','end_to_end_full_seconds']
stats={k:{'median':st.median(x[k] for x in rows),'min':min(x[k] for x in rows),'max':max(x[k] for x in rows)} for k in fields}
samples=[json.loads(s) for s in (P/'results/resources.jsonl').read_text().splitlines()];hot=[x for x in samples if x['phase']=='measure'];assert hot
values=lambda s:{k:int(v) for k,v in re.findall(r'^(\w+):?\s+(\d+)',s,re.M)}
peak_pss={};peak_rss={}
for s in hot:
 for pid,x in s['processes'].items():
  peak_pss[pid]=max(peak_pss.get(pid,0),values(x['smaps'])['Pss']);peak_rss[pid]=max(peak_rss.get(pid,0),values(x['status']).get('VmRSS',0))
cpu=[]
for a,b in zip(hot,hot[1:]):
 x=list(map(int,a['cpu'].split()[1:9]));y=list(map(int,b['cpu'].split()[1:9]));dt=sum(y)-sum(x)
 if dt>0:cpu.append(100*(dt-(y[3]-x[3])-(y[4]-x[4]))/dt)
v1=values(hot[0]['vmstat']);v2=values(hot[-1]['vmstat'])
audit={'status':'completed_short_co_residency_not_long_term','hot_requests':len(rows),'timing_seconds':stats,'temperature_max_mC':max(max(x['temperatures'].values()) for x in hot),'mem_available_min_kib':min(values(x['meminfo'])['MemAvailable'] for x in hot),'cpu_8core_normalized_median_percent':st.median(cpu),'process_peak_pss_kib':peak_pss,'process_peak_rss_kib':peak_rss,'swapin_pages_during_measure':v2['pswpin']-v1['pswpin'],'swapout_pages_during_measure':v2['pswpout']-v1['pswpout'],'language_driver_max_bytes':max(x['metrics']['driver_bytes'] for x in rows),'decode_median_tokens_per_second':st.median(x['metrics']['decode_tokens_per_second'] for x in rows),'summary_sha256':hashlib.sha256((P/'results/summary.json').read_bytes()).hexdigest()}
(P/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
