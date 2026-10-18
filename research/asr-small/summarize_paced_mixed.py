import json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]/'results/asr-small-20261018'
def q(x,p):return sorted(x)[max(0,__import__('math').ceil(len(x)*p)-1)]
summary={'paced':{},'mixed':{}}
for p in ROOT.glob('paced-*/*/result.json'):
 d=json.loads(p.read_text());rs=[]
 for r in d['runs']:
  cs=r['chunk_records'];rs.append({'id':r['id'],'reference':r['reference'],'text':r['text'],'duration':r['duration_seconds'],'first_partial':r['first_partial_seconds'],'stable_final':r['stable_final_seconds'],'clip_end_to_stable':r['clip_end_to_stable_seconds'],'endpoints':r['natural_endpoint_transitions'],'flush_changed':r['flush_changed_text'],'chunk_compute_p95_ms':q([c['compute_ms'] for c in cs],.95),'completion_lag_p95_ms':q([c['lag_ms'] for c in cs],.95),'completion_lag_max_ms':max(c['lag_ms'] for c in cs),'start_lag_p95_ms':q([max(0,c['start_seconds']-c['due_seconds'])*1000 for c in cs],.95),'start_lag_max_ms':max(max(0,c['start_seconds']-c['due_seconds'])*1000 for c in cs),'speech_compute_seconds':r['speech_chunk_compute_seconds'],'tail_compute_seconds':r['tail_chunk_compute_seconds']})
 summary['paced'][str(p.relative_to(ROOT))]=rs
for p in ROOT.glob('mixed-*/summary.json'):
 rows=json.loads(p.read_text());base=[r['decode_tokens_per_second'] for r in rows if r['condition']=='baseline'];b=statistics.mean(base);stats={}
 for cond in sorted({r['condition'] for r in rows}):
  rs=[r for r in rows if r['condition']==cond];resources=[]
  for r in rs:
   resources+=json.loads((p.parent/f'{cond}-{r["repeat"]}'/'system.json').read_text())
  stats[cond]={'runs':len(rs),'tps':[r['decode_tokens_per_second'] for r in rs],'tps_mean':statistics.mean(r['decode_tokens_per_second'] for r in rs),'tps_change_percent':100*(statistics.mean(r['decode_tokens_per_second'] for r in rs)/b-1),'ttft':[r['ttft_seconds'] for r in rs],'response_temp_mC':[r['response_start_temperature_mC'] for r in rs],'audio_overlap_seconds':[r.get('audio_overlap_seconds') for r in rs],'audio_rtf_p50':[r.get('audio_rtf_p50') for r in rs],'min_available_mib':min(z['mem_available_kib'] for z in resources)/1024,'max_swap_mib':max(z['swap_used_kib'] for z in resources)/1024,'peak_sum_process_pss_mib':max(sum(v.get('Pss_kib',0) for v in z['processes'].values()) for z in resources)/1024}
 summary['mixed'][p.parent.name]=stats
(ROOT/'paced-mixed-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))
