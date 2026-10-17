import json,statistics
from pathlib import Path
R=Path(__file__).resolve().parents[2]/'results/audio-models-20261018';result={}
for version in ['paced-v1','paced-v2']:
 p=R/version
 if not (p/'summary.json').exists():continue
 d=json.loads((p/'summary.json').read_text());frames=json.loads((p/'frames.json').read_text());monitor=json.loads((p/'system-resources.json').read_text());entry={k:v for k,v in d.items() if k not in ['llm','segments','backend']};entry['segment_count']=len(d['segments']);entry['forced_ids']=[x['index'] for x in d['segments'] if x['forced_at_end']];entry['backend']={};entry['llm']=[]
 for kind in ['asr','speaker']:
  b=json.loads((p/(kind+'.json')).read_text());rs=[r for r in b['runs'] if r['segment'] not in entry['forced_ids']]
  entry['backend'][kind]={'nonforced_calls':len(rs),'queue_wait_max_seconds':max(r['queue_delay_seconds'] for r in rs),'emission_to_result_p50_seconds':statistics.median(r['end']-r['queued_at'] for r in rs),'segment_end_to_result_p50_seconds':statistics.median(r['end']-d['segments'][r['segment']]['vad_segment_end_due'] for r in rs),'resources':b['resources']}
 for llm in d['llm']:
  start=llm['response_start_monotonic_seconds'];end=start+llm['total_seconds'];overlap={'frontend':sum(max(0,min(f['end'],end)-max(f['start'],start)) for f in frames)}
  for kind in ['asr','speaker']:
   rs=json.loads((p/(kind+'.json')).read_text())['runs'];overlap[kind]=sum(max(0,min(x['end'],end)-max(x['start'],start)) for x in rs)
  entry['llm'].append({k:llm[k] for k in ['new_prompt_tokens','ttft_seconds','decode_tokens_per_second','response_start_temperature_mC','driver_after_release_bytes','npu_frequency_before_hz','npu_frequency_after_hz']})
  entry['llm'][-1]['audio_overlap_seconds']=overlap
 entry['min_available_mib']=min(r['mem_available_kib'] for r in monitor)/1024;entry['max_swap_used_mib']=max(r['swap_used_kib'] for r in monitor)/1024;entry['max_sum_process_pss_mib']=max(sum(s.get('Pss_kib',0) for s in r['processes'].values()) for r in monitor)/1024
 result[version]=entry
(R/'paced-comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
