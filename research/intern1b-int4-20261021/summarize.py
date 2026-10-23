from pathlib import Path
import json,statistics,sys
import numpy as np
from tokenizers import Tokenizer
r=Path(__file__).resolve().parent;files=sorted((r/'results').glob('*/events.jsonl'));assert files
p=files[-1];rows=[json.loads(x) for x in p.read_text().splitlines()];tok=Tokenizer.from_file('/opt/opi-lab/source-validation-20261017/official-InternVL3-1B/tokenizer.json')
def stats(v):return {'n':len(v),'p50':statistics.median(v),'p95':float(np.quantile(v,.95)),'min':min(v),'max':max(v)}
summary={'result_file':str(p),'complete':any(x.get('event')=='complete' for x in rows),'errors':[x for x in rows if x.get('event')=='error'],'performance':{},'quality':[],'diagnostics':[],'loaded':[x for x in rows if x.get('event')=='loaded'],'closed':[x for x in rows if x.get('event')=='closed']}
for v in ['w8','w4']:
 out={}
 for case in ['dynamic-64','dynamic-128','dynamic-256']:
  group=[x for x in rows if x.get('event')=='measure' and x['variant']==v and x['case']==case]
  if not group:continue
  data={'dynamic_tokens':group[0]['dynamic_tokens'],'prompt_tokens':group[0]['result']['prompt_tokens'],'reused_tokens':group[0]['result']['reused_tokens']}
  for key in ['ttft_seconds','decode_tokens_per_second','total_seconds','cpu_seconds','rss_kib','driver_bytes','temperature_start_mC','temperature_end_mC']:data[key]=stats([x['result'][key] for x in group])
  data['cpu_cores']=stats([x['result']['cpu_seconds']/x['result']['total_seconds'] for x in group]);data['identical_feedback']=all(x['result']['fixed_feedback'] and x['result']['generated_tokens']==64 for x in group);out[case]=data
 summary['performance'][v]=out
for x in rows:
 if x.get('event')=='quality':summary['quality'].append({'variant':x['variant'],'case':x['case'],'tokens':x['result']['tokens'],'text':tok.decode(x['result']['tokens']),'generated_tokens':x['result']['generated_tokens'],'finish_reason':x['result']['finish_reason']})
summary['request_resource_deltas']={}
for variant in ['w8','w4']:
 group=[x for x in rows if x.get('event')=='measure' and x['variant']==variant]
 summary['request_resource_deltas'][variant]={key:sum(x['after'][key]-x['before'][key] for x in group) for key in ['swap_in','swap_out']}
 summary['request_resource_deltas'][variant]['system_loadavg_max']=[max(x['after']['loadavg'][i] for x in group) for i in range(3)] if group else []
summary['speedups']={}
for case in summary['performance']['w4']:
 a=summary['performance']['w8'][case];b=summary['performance']['w4'][case];ratio=b['decode_tokens_per_second']['p50']/a['decode_tokens_per_second']['p50'];summary['speedups'][case]={'decode_ratio':ratio,'gain_percent':100*(ratio-1),'meets_50_percent_target':ratio>=1.5,'ttft_reduction_percent':100*(1-b['ttft_seconds']['p50']/a['ttft_seconds']['p50'])}
diag={x['variant']:x for x in rows if x.get('event')=='diagnostic'}
if set(diag)=={'w8','w4'}:
 tensors={v:np.fromfile(p.parent/Path(x['dump']).parent.name/Path(x['dump']).name,np.float32).reshape(64,151674) for v,x in diag.items()};a=tensors['w8'].astype(np.float64);b=tensors['w4'].astype(np.float64);logs=[]
 for data in [a,b]:
  z=data-data.max(axis=1,keepdims=True);logs.append(z-np.log(np.exp(z).sum(axis=1,keepdims=True)))
 pa=np.exp(logs[0]);kl=(pa*(logs[0]-logs[1])).sum(axis=1);norms=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1);cos=(a*b).sum(axis=1)/np.maximum(norms,1e-30)
 summary['diagnostics']={'comparison':'same fixed feedback sequence, NPU W4 versus current NPU W8; not original FP16/BF16 oracle','argmax_matches':int((a.argmax(axis=1)==b.argmax(axis=1)).sum()),'steps':64,'mean_kl_nats':float(kl.mean()),'relative_l2':float(np.linalg.norm(a-b)/np.linalg.norm(a)),'min_cosine':float(cos.min()),'mean_cosine':float(cos.mean())}
samples=[x for row in rows if row.get('event')=='telemetry' for x in row['samples'] if 'temperature' in x];summary['telemetry']={}
if samples:
 summary['telemetry']['whole_run']={'samples':len(samples),'pss_max':max(sum(z['pss'] for z in x['processes']) for x in samples),'rss_max':max(sum(z['rss'] for z in x['processes']) for x in samples),'temperature_max':max(max(x['temperature'].values()) for x in samples),'available_min':min(x['available'] for x in samples),'swap_used_max':max(x['swap_used'] for x in samples),'swap_in_delta':max(x['swap_in'] for x in samples)-min(x['swap_in'] for x in samples),'swap_out_delta':max(x['swap_out'] for x in samples)-min(x['swap_out'] for x in samples)}
 for v in ['w8','w4']:
  windows=[(x['start'],x['end']) for x in rows if x.get('event')=='measure' and x['variant']==v];ss=[s for s in samples if any(a<=s['at']<=b for a,b in windows)]
  if ss:summary['telemetry'][v]={'scope':'measurement request windows only','samples':len(ss),'pss_max':max(sum(z['pss'] for z in x['processes']) for x in ss),'rss_max':max(sum(z['rss'] for z in x['processes']) for x in ss),'temperature_max':max(max(x['temperature'].values()) for x in ss),'available_min':min(x['available'] for x in ss),'system_cpu_percent_p50':statistics.median(x['cpu_percent'] for x in ss)}
if samples:
 summary['frequency_ranges']={'npu_hz':sorted(set(int(x['npu_frequency']) for x in samples)),'cpu_khz':{cpu:{'min':min(x['cpu_freq'][cpu] for x in samples),'max':max(x['cpu_freq'][cpu] for x in samples)} for cpu in samples[0]['cpu_freq']}}
summary['protocol']={'system_prompt_tokens':len(json.loads((r/'inputs.json').read_text())['prefix']),'output_tokens':64,'repeats_per_variant_per_dynamic_case':5,'session_order':'W8(3) W4(3) W4(2) W8(2)','target_decode_ratio':1.5,'native_packed_int4_proven':False,'quality_scope':'five natural daily requests and 64-step fixed-feedback logits vs W8; not complete production acceptance'}
(r/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))
