"""Summarize saved resident-agent logs without contacting the board."""
import argparse,collections,json,math,statistics,unicodedata
from pathlib import Path

def distribution(values):
 v=sorted(x for x in values if x is not None)
 if not v:return None
 return {'n':len(v),'min':v[0],'p50':statistics.median(v),'p95':v[math.ceil(.95*len(v))-1],'max':v[-1],'mean':statistics.mean(v)}

def normalize(s):
 return ''.join(c.lower() for c in unicodedata.normalize('NFKC',s) if unicodedata.category(c)[0] in 'LN')

def distance(a,b):
 row=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  nxt=[i]
  for j,y in enumerate(b,1):nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(x!=y)))
  row=nxt
 return row[-1]

def summarize_turns(rows):
 keys=['file_end_to_first_s','file_end_to_complete_s','segment_proxy_to_first_s','vad_to_first_s','vad_to_complete_s','asr_seconds','endpoint_drain_seconds','memory_seconds','prefix_prepare_seconds','llm_first_text_s','llm_complete_s','wake_s','dynamic_tokens','memory_added_tokens']
 result={k:distribution([x.get(k) for x in rows]) for k in keys}
 result['segment_proxy_to_complete_s']=distribution([x['segment_proxy_to_first_s']+x['file_end_to_complete_s']-x['file_end_to_first_s'] for x in rows if x['segment_proxy_to_first_s'] is not None])
 result['endpoint_wait_from_proxy_s']=distribution([x['vad_endpoint']-x['command_start']-x['segment_end_sample']/16000 for x in rows])
 for k in ['decode_tokens_per_second','reused_tokens','new_tokens','generated_tokens','driver_bytes']:
  result[k]=distribution([x['metrics'][k] for x in rows if k!='decode_tokens_per_second' or x['metrics']['generated_tokens']>1])
 result['finish_reasons']=dict(collections.Counter(x['metrics']['finish_reason'] for x in rows))
 result['turns']=len(rows)
 scored=[]
 for x in rows:
  a,b=normalize(x['reference']),normalize(x['asr_text'])
  scored.append({'turn':x['turn'],'source_id':x['source_id'],'reference':x['reference'],'asr_text':x['asr_text'],'errors':distance(a,b),'reference_characters':len(a),'reply':x['reply']})
 total=sum(x['reference_characters'] for x in scored)
 result['cer']={'normalization':'Unicode NFKC, lowercase, retain letters/numbers; punctuation/space removed; no number verbalization or semantic normalization','errors':sum(x['errors'] for x in scored),'reference_characters':total,'rate':sum(x['errors'] for x in scored)/total if total else None,'exact_transcripts':sum(x['errors']==0 for x in scored),'cases_with_errors':[x for x in scored if x['errors']]}
 result['command_stage_compute_s']={k:distribution([x['command_stage_compute_s'][k] for x in rows if 'command_stage_compute_s' in x]) for k in ['denoise','vad','kws','stream_asr']}
 result['memory_nonzero_added_tokens']=distribution([x['memory_added_tokens'] for x in rows if x['memory_added_tokens']])
 return result

def cpu_values(row):
 return {s.split()[0]:[int(x) for x in s.split()[1:9]] for s in row['cpu']}

def cpu_delta(a,b):
 out={}
 for name,x in a.items():
  d=[j-i for i,j in zip(x,b[name])];total=sum(d)
  if total>0:out[name]=100*(total-d[3]-d[4])/total
 return out

def summarize_resources(rows):
 if len(rows)<2:return {}
 first,last=rows[0],rows[-1];intervals=collections.defaultdict(list)
 for a,b in zip(rows,rows[1:]):
  for k,v in cpu_delta(cpu_values(a),cpu_values(b)).items():intervals[k].append(v)
 result={'sampling_seconds':last['at']-first['at'],'samples':len(rows),'cpu_percent_8core_total100_time_weighted':cpu_delta(cpu_values(first),cpu_values(last)), 'cpu_interval_percent':{k:distribution(v) for k,v in intervals.items()},'temperature_C':distribution([x['temp_mC']/1000 for x in rows]),'mem_available_MiB':distribution([x['meminfo']['MemAvailable']/1024 for x in rows]),'swap_used_MiB':distribution([(x['meminfo']['SwapTotal']-x['meminfo']['SwapFree'])/1024 for x in rows]),'swap_io_pages':{k:last['vmstat'][k]-first['vmstat'][k] for k in first['vmstat']},'loadavg_1min':distribution([float(x['loadavg'].split()[0]) for x in rows]),'processes':{},'cpu_freq_khz':{k:distribution([x['freq_khz'][k] for x in rows]) for k in first['freq_khz']}}
 for pid in set().union(*(x['processes'] for x in rows)):
  ps=[(x['at'],x['processes'][pid]) for x in rows if pid in x['processes']]
  proc={'name':ps[0][1]['stat'].split('(',1)[1].split(')',1)[0]}
  for key in ['VmRSS','Pss']:proc[key+'_MiB']=distribution([p[key]/1024 for _,p in ps]);proc[key+'_first_last_MiB']=[ps[0][1][key]/1024,ps[-1][1][key]/1024]
  def ticks(p):
   fields=p['stat'].rsplit(')',1)[1].split();return int(fields[11])+int(fields[12])
  proc['cpu_seconds_assuming_CLK_TCK100']=(ticks(ps[-1][1])-ticks(ps[0][1]))/100
  proc['one_core_cpu_percent']=100*proc['cpu_seconds_assuming_CLK_TCK100']/(ps[-1][0]-ps[0][0])
  result['processes'][pid]=proc
 f=last['frontend'];g=first['frontend'];frames=f['frames']-g['frames']
 result['frontend']={'frames_sampled':frames,'overruns_sampled':f['overruns']-g['overruns'],'overrun_fraction':(f['overruns']-g['overruns'])/frames,'max_lag_s':f['max_lag_s'],'max_compute_s':f['max_compute_s'],'stage_mean_ms_per_16ms_frame':{k:1000*(v-g['stage_seconds'][k])/frames for k,v in f['stage_seconds'].items()},'caveat':'vad includes endpoint stream-drain time, which also appears in stream_asr; do not sum stage means. Lag is replay backlog, not measured microphone loss.'}
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('name');a=p.parse_args();base=a.directory;n=a.name
 meta=json.loads((base/(n+'-meta.json')).read_text());rows=[json.loads(x) for x in (base/(n+'-turns.jsonl')).read_text().splitlines()]
 valid=[x for x in rows if x['segments']==1 and x.get('reply','').strip() and x.get('file_end_to_first_s') is not None]
 out={'name':n,'meta':meta,'invalid_turns':[x['turn'] for x in rows if x not in valid],'all_valid':summarize_turns(valid),'by_mode':{},'by_mode_budget':{}}
 for mode in sorted({x.get('mode',meta['mode']) for x in valid}):
  subset=[x for x in valid if x.get('mode',meta['mode'])==mode];out['by_mode'][mode]=summarize_turns(subset)
  out['by_mode_budget'][mode]={str(b):summarize_turns([x for x in subset if x['dynamic_budget']==b]) for b in [64,128,256] if any(x['dynamic_budget']==b for x in subset)}
 rp=base/(n+'-resources.jsonl')
 if rp.exists():
  out['resources']=summarize_resources([json.loads(x) for x in rp.read_text().splitlines()])
  if meta.get('stage_timing_version',1)>=2:out['resources']['frontend']['caveat']='Stage timers exclude stream-drain double counting. Lag is replay backlog, not measured microphone loss.'
 (base/(n+'-summary.json')).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'name':n,'status':meta['status'],'turns':len(rows),'invalid':out['invalid_turns'],'elapsed':meta['elapsed_seconds']},ensure_ascii=False))
if __name__=='__main__':main()
