import json,sys,math,statistics
from pathlib import Path
from collections import Counter
root=Path(sys.argv[1]);rows=[json.loads(x) for x in (root/'events.jsonl').read_text().splitlines()];res=[json.loads(x) for x in (root/'resources.jsonl').read_text().splitlines()];meta=json.loads((root/'meta.json').read_text())
def stats(xs):
 xs=sorted(x for x in xs if x is not None)
 return {'n':len(xs),'p50':statistics.median(xs),'p95':xs[max(0,math.ceil(.95*len(xs))-1)],'max':max(xs),'min':min(xs),'mean':statistics.mean(xs)} if xs else {}
b=[x for x in rows if x['event']=='brain_done'];frames=[x for x in rows if x['event']=='frame'];asr=[x for x in rows if x['event']=='asr_final'];actions=[x for x in rows if x['event']=='action_start'];cancels=[x for x in rows if x['event']=='action_cancel'];triggers=[x for x in rows if x['event']=='trigger']
summary={'meta':meta,'counts':dict(Counter(x['event'] for x in rows)),'timings':{key:stats([x.get(key) for x in b]) for key in ['queue_s','compression_s','prefix_restore_s','llm_first_s','first_visible_s','event_to_first_s','llm_s','rag_s','event_to_action_s','total_s']},'decode_tokens_s':stats([x['metrics']['decode_tokens_per_second'] for x in b]),'prompt_tokens':stats([x['prompt_tokens'] for x in b]),'output_tokens':stats([x['metrics']['generated_tokens'] for x in b]),'reused_tokens':stats([x['metrics']['reused_tokens'] for x in b]),'finish_reason':dict(Counter(x['metrics']['finish_reason'] for x in b)),'frame_compute_s':stats([x['compute_s'] for x in frames]),'frame_lag_s':stats([x['lag_s'] for x in frames]),'frame_over_budget_fraction':sum(x['compute_s']>.016 for x in frames)/max(1,len(frames)),'audio_processed_s':len(frames)*.016,'asr_drain_s':stats([x['drain_s'] for x in asr]),'application_pending_samples_max':max((x.get('application_pending_samples',0) for x in frames),default=0),'preempt_s':stats([x.get('preempt_s') for x in cancels]),'actual_trigger_interval_s':stats([y['at']-x['at'] for x,y in zip(triggers,triggers[1:])])}
summary['compression']={'output_tokens':stats([x['context']['tokens'] for x in b]),'input_tokens':stats([x['context']['input_tokens'] for x in b]),'over_budget':sum(x['context']['tokens']>x['context'].get('budget',meta.get('summary_budget',64)) for x in b),'critical_clauses_dropped':sum(x['context']['dropped_critical_clauses'] for x in b),'latest_clause_dropped_nonempty':sum(x['context']['input_rows']>0 and not x['context']['latest_clause_kept'] for x in b),'calls':len(b),'semantics':'extractive proxy flags, not independently human-scored factual retention'}
# System CPU capacity normalized to all eight cores = 100%.
cpu=[]
for x,y in zip(res,res[1:]):
 aa=[int(v) for v in x['cpu'][0].split()[1:9]];bb=[int(v) for v in y['cpu'][0].split()[1:9]];d=[v-u for u,v in zip(aa,bb)];total=sum(d)
 if total>0:cpu.append(100*(total-d[3]-d[4])/total)
def pss(x):return sum(v.get('Pss',0) for v in x['processes'].values())
def rss(x):return sum(v.get('VmRSS',0) for v in x['processes'].values())
summary['resources']={'system_cpu_percent':stats(cpu),'temp_C':stats([x['temp_mC']/1000 for x in res]),'sum_pss_MiB':stats([pss(x)/1024 for x in res]),'sum_rss_MiB':stats([rss(x)/1024 for x in res]),'available_MiB':stats([x['meminfo']['MemAvailable']/1024 for x in res]),'max_driver_bytes':max((x['metrics']['driver_bytes'] for x in b),default=0),'swap_in_pages':res[-1]['vmstat']['pswpin']-res[0]['vmstat']['pswpin'] if res else None,'swap_out_pages':res[-1]['vmstat']['pswpout']-res[0]['vmstat']['pswpout'] if res else None,'pss_last_minus_first_MiB':(pss(res[-1])-pss(res[0]))/1024 if res else None,'max_load1':max((float(x['loadavg'].split()[0]) for x in res),default=0)}
# Audit generation IDs: no old mock step may follow cancellation.
canceled=set();stale=[]
for x in rows:
 if x['event']=='action_cancel':canceled.add(x['old_id'])
 if x['event']=='action_tick' and x['id'] in canceled:stale.append(x)
summary['stale_action_ticks']=len(stale)
trigger_by_id={x['id']:x for x in triggers}
summary['event_to_actual_action_start_s']=stats([x['at']-trigger_by_id[x['id']]['at'] for x in actions if x['id'] in trigger_by_id])
summary['rag_ready_to_action_start_s']=stats([x['at']-x['ready_at'] for x in actions])
summary['capture_start_lateness_s']=stats([max(0,x['lag_s']-x['compute_s']) for x in frames])
starts={};speech=[]
for x in rows:
 if x['event']=='source_start':starts[x['source_id']]=x['at']
 if x['event']=='source_end' and x['source_id'] in starts:speech.append((starts.pop(x['source_id']),x['at']))
summary['completed_source_speech_s']=sum(z-a for a,z in speech)
summary['speech_brain_overlap_s']=sum(max(0,min(z,x['at'])-max(a,x['start'])) for a,z in speech for x in b)
summary['five_minute_windows']=[]
if frames:
 start=frames[0]['at']
 for i in range(math.ceil(meta["requested_s"]/300)):
  rr=[x for x in res if i*300<=x['at']-start<(i+1)*300];bb=[x for x in b if i*300<=x['start']-start<(i+1)*300];ff=[x for x in frames if i*300<=x['at']-start<(i+1)*300]
  if rr:summary['five_minute_windows'].append({'minutes':[i*5,(i+1)*5],'brain_calls':len(bb),'event_to_first_s':stats([x['event_to_first_s'] for x in bb]),'decode_tokens_s':stats([x['metrics']['decode_tokens_per_second'] for x in bb]),'temp_C':stats([x['temp_mC']/1000 for x in rr]),'pss_MiB':stats([pss(x)/1024 for x in rr]),'lag_s':stats([x['lag_s'] for x in ff])})
(root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))
