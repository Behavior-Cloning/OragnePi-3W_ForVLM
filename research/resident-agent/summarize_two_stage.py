import argparse,json,statistics,math,sys
from pathlib import Path
from summarize import normalize,distance,distribution,cpu_values,cpu_delta
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('name');a=p.parse_args();base=a.root;name=a.name
rows=[json.loads(x) for x in (base/(name+'-turns.jsonl')).read_text().splitlines()];meta=json.loads((base/(name+'-meta.json')).read_text())
out={'meta':meta,'policies':{},'pairs':[]}
for pol in ['baseline','speculative']:
 rs=[r for r in rows if r['policy']==pol];n=sum(len(normalize(r['reference'])) for r in rs);errors=sum(distance(normalize(r['reference']),normalize(r['response']['text'])) for r in rs)
 out['policies'][pol]={'turns':len(rs),'proxy_to_first_s':distribution([r['proxy_to_first_s'] for r in rs]),'endpoint_to_first_s':distribution([r['endpoint_to_first_s'] for r in rs]),'proxy_to_complete_s':distribution([r['proxy_to_complete_s'] for r in rs]),'cache_hits':sum(r['response']['cache_hit'] for r in rs),'candidate_count':sum(r['candidate_count'] for r in rs),'stale_candidates':sum(r['stale_candidates'] for r in rs),'speech_resumptions':sum(r['speech_resumptions'] for r in rs),'gate_violations':sum(r['publication_gate_violations'] for r in rs),'cer_errors':errors,'cer_characters':n,'cer':errors/n if n else None}
for sid in sorted({r['source_id'] for r in rows}):
 rs={r['policy']:r for r in rows if r['source_id']==sid}
 if len(rs)<2:continue
 b,s=rs['baseline'],rs['speculative'];out['pairs'].append({'source_id':sid,'reference':b['reference'],'asr_equal':b['response']['text']==s['response']['text'],'prompt_equal':b['response']['prompt']==s['response']['prompt'],'tokens_equal':b['response']['tokens']==s['response']['tokens'],'first_saved_s':b['proxy_to_first_s']-s['proxy_to_first_s'],'baseline_first_s':b['proxy_to_first_s'],'speculative_first_s':s['proxy_to_first_s'],'speculative_hit':s['response']['cache_hit'],'baseline_reply':b['response']['reply'],'speculative_reply':s['response']['reply'],'tentative_texts':[x['text'] for x in s['asr'] if x['kind']=='tentative'],'final_text':s['response']['text'],'candidate_statuses':[x['status'] for x in s['candidates']],'stale_candidates':s['stale_candidates']})
out['paired_first_saved_s']=distribution([x['first_saved_s'] for x in out['pairs']])
rp=base/(name+'-resources.jsonl')
if rp.exists():
 rr=[json.loads(x) for x in rp.read_text().splitlines()];f,l=rr[0],rr[-1]
 out['resources']={'whole_cpu_mean_percent':cpu_delta(cpu_values(f),cpu_values(l))['cpu'],'temperature_C':distribution([x['temp_mC']/1000 for x in rr]),'mem_available_MiB':distribution([x['meminfo']['MemAvailable']/1024 for x in rr]),'swap_io_pages':{k:l['vmstat'][k]-f['vmstat'][k] for k in f['vmstat']},'frontend':l['frontend']}
 for pol in ['baseline','speculative']:
  values=[];durations=[];temps=[]
  for row in [z for z in rows if z['policy']==pol]:
   subset=[x for x in rr if row['start']<=x['at']<=row['response']['end']]
   if len(subset)>1:
    dt=subset[-1]['at']-subset[0]['at'];values.append(cpu_delta(cpu_values(subset[0]),cpu_values(subset[-1]))['cpu']);durations.append(dt);temps.extend(x['temp_mC']/1000 for x in subset)
  out['policies'][pol]['active_turn_cpu_percent']=sum(v*d for v,d in zip(values,durations))/sum(durations) if durations else None
  out['policies'][pol]['active_turn_temperature_C']=distribution(temps)
 for pid in f['processes']:
  namep=f['processes'][pid]['stat'].split('(',1)[1].split(')',1)[0];out['resources'][namep+'_rss_MiB']=distribution([x['processes'][pid]['VmRSS']/1024 for x in rr if pid in x['processes']])
(base/(name+'-summary.json')).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False,indent=2))
