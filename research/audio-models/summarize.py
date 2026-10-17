"""Rebuild comparison tables from retained board measurements; never invent missing runs."""
import json,statistics
from pathlib import Path
R=Path(__file__).resolve().parents[2]/'results/audio-models-20261018'
def median(xs):return statistics.median(xs)
def main():
 out={'standalone':{},'mixed':{},'quality':{},'paced':{}}
 for kind in ['vad','denoise','asr','speaker']:
  d=json.loads((R/'standalone-v1'/kind/'result.json').read_text());resources=json.loads((R/'standalone-v1'/kind/'system-resources.json').read_text());hot=[r for r in d['runs'] if r['sample']=='zh'][1:]
  out['standalone'][kind]={'zh_seconds':d['by_sample']['zh']['p50_seconds'],'zh_p95_seconds':d['by_sample']['zh']['p95_seconds'],'zh_rtf':d['by_sample']['zh']['rtf_p50'],'load_once_seconds':d['load_seconds'],'peak_rss_mib':d['final_resources']['peak_rss_kib']/1024,'max_sampled_pss_mib':max(p.get('Pss_kib',0) for r in resources for p in r['processes'].values())/1024,'zh_cpu_percent_one_core':median(r['cpu_seconds']/r['seconds']*100 for r in hot),'zh_frame_p95_ms':median(r['frame_p95_ms'] for r in hot) if hot[0]['frame_p95_ms'] else None,'temperature_range_mC':[min(r['temperature_mC'] for r in resources),max(r['temperature_mC'] for r in resources)]}
 if (R/'mixed-v1/summary.json').exists():
  mixed=json.loads((R/'mixed-v1/summary.json').read_text());out['mixed_status']=mixed['status']
  for dynamic in [64,256]:
   base=[r for r in mixed['runs'] if r['dynamic']==dynamic and r['condition']=='baseline']
   if not base:continue
   baseline_ttft=median(r['ttft_seconds'] for r in base);baseline_tps=median(r['decode_tokens_per_second'] for r in base)
   for condition in sorted({r['condition'] for r in mixed['runs']}):
    rs=[r for r in mixed['runs'] if r['dynamic']==dynamic and r['condition']==condition]
    if not rs:continue
    ttft=median(r['ttft_seconds'] for r in rs);tps=median(r['decode_tokens_per_second'] for r in rs);key=f'd{dynamic}-{condition}'
    monitors=[row for r in rs for row in json.loads((R/'mixed-v1'/r['case']/'system-resources.json').read_text())]
    # Sampling contains idle thermal preparation too; report full trial footprint explicitly.
    out['mixed'][key]={'n':len(rs),'ttft_seconds':ttft,'tps':tps,'ttft_change_percent':(ttft/baseline_ttft-1)*100,'decode_loss_percent':(1-tps/baseline_tps)*100,'response_start_temperature_mC':[r['response_start_temperature_mC'] for r in rs],'min_system_available_mib':min(row['mem_available_kib'] for row in monitors)/1024,'max_swap_used_mib':max(row['swap_used_kib'] for row in monitors)/1024,'max_sum_process_pss_mib':max(sum(p.get('Pss_kib',0) for p in row['processes'].values()) for row in monitors)/1024,'driver_bytes':[r['driver_allocated_bytes'] for r in rs],'audio':{}}
    for kind in ['vad','denoise','asr','speaker']:
     calls=[]
     for r in rs:
      f=R/'mixed-v1'/r['case']/(kind+'.json')
      if not f.exists():continue
      d=json.loads(f.read_text());start=r['response_start_monotonic_seconds'];end=start+r['total_seconds'];calls.extend(c for c in d['runs'] if c['end_monotonic']>start and c['start_monotonic']<end)
     if calls:
      out['mixed'][key]['audio'][kind]={n:{'n':len(cs),'p50_seconds':median(c['seconds'] for c in cs),'rtf_p50':median(c['rtf'] for c in cs),'cpu_share_one_core_percent':median(c['cpu_seconds']/c['seconds']*100 for c in cs)} for n in sorted({c['sample'] for c in calls}) for cs in [[c for c in calls if c['sample']==n]]}
 if (R/'quality-v1/result.json').exists():out['quality']=json.loads((R/'quality-v1/result.json').read_text())
 for version in ['paced-v1','paced-v2']:
  if (R/version/'summary.json').exists():out['paced'][version]=json.loads((R/version/'summary.json').read_text())
 (R/'comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 lines=['## 同一5.592秒中文音频的板端独跑结果','', '| 模型 | 热态p50秒 | 热态p95秒（4次） | RTF | 单次加载秒 | 进程峰值RSS MiB | 采样峰值PSS MiB |','|---|---:|---:|---:|---:|---:|---:|']
 for k,v in out['standalone'].items():lines.append(f"| {k} | {v['zh_seconds']:.3f} | {v['zh_p95_seconds']:.3f} | {v['zh_rtf']:.3f} | {v['load_once_seconds']:.3f} | {v['peak_rss_mib']:.1f} | {v['max_sampled_pss_mib']:.1f} |")
 lines+=['','## NPU LLM与语音的并发压力测试','','| 动态token | 并发条件 | 次数 | TTFT秒 | decode token/s | TTFT变化 | decode下降 |','|---:|---|---:|---:|---:|---:|---:|']
 for k,v in out['mixed'].items():
  dynamic,condition=k.split('-',1);lines.append(f"| {dynamic[1:]} | {condition} | {v['n']} | {v['ttft_seconds']:.3f} | {v['tps']:.3f} | {v['ttft_change_percent']:+.2f}% | {v['decode_loss_percent']:+.2f}% |")
 (R/'tables.md').write_text('\n'.join(lines)+'\n')
 print('\n'.join(lines))
if __name__=='__main__':main()
