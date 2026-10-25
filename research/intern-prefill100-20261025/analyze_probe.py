#!/usr/bin/env python3
"""Recompute actual probe output hashes and same-query errors; no TTFT extrapolation."""
from pathlib import Path
import argparse,hashlib,json,math,statistics
import numpy as np
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();out=R/'results'/a.run
rows=[json.loads(x) for x in (out/'events.jsonl').read_text().splitlines()]
meta=json.loads((out/'meta.json').read_text());assert meta['status']=='complete'
results=[x for x in rows if x['event']=='result'];assert len(results)==12
cfg=json.loads((out/'config.json').read_text());spec={x['run_id']:x for x in cfg['sequence']}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def errors(x,y):
 e=y.astype('float64')-x.astype('float64')
 return {'byte_equal':x.tobytes()==y.tobytes(),'relative_l2':float(np.linalg.norm(e)/max(np.linalg.norm(x.astype('float64')),1e-30)),'max_abs':float(np.abs(e).max())}
summary=[];first={};drift=[]
for row in results:
 name=row['case'];m=row['query_tokens'];layer=row['first_layer'];metrics=row['metrics'];f=out/(name+'.f32');ref=R/'fixtures-canonical'/f'c512-m{m}-l{layer}'/'probe-reference.f32'
 assert metrics['scope']=='two_layer_probe' and metrics['capacity']==512 and metrics['query_rows']==m
 assert sha(f)==row['output_sha256'] and sha(ref)==spec[name]['reference_sha256']
 x=np.fromfile(ref,dtype='<f4');y=np.fromfile(f,dtype='<f4')
 assert x.size==y.size==m*(896+4*128) and np.isfinite(x).all() and np.isfinite(y).all()
 timings=metrics['api_seconds'];assert len(timings)==30 and all(math.isfinite(t) and t>0 for t in timings)
 outputs=[];off=0
 for label,width in [('hidden',896),('layer0_new_k',128),('layer1_new_k',128),('layer0_new_v',128),('layer1_new_v',128)]:
  n=m*width;outputs.append({'tensor':label,**errors(x[off:off+n],y[off:off+n])});off+=n
 q={'case':name,'query_tokens':m,'first_layer':layer,'n':30,'p50_api_s':statistics.median(timings),'p95_api_s':sorted(timings)[math.ceil(.95*len(timings))-1],'min_api_s':min(timings),'max_api_s':max(timings),'p50_api_s_per_input_row':statistics.median(timings)/m,'allocated_bytes':metrics['allocated_bytes'],'released_bytes':metrics['released_bytes'],'temperature_before_mC':metrics['temperature_before_mC'],'temperature_after_mC':metrics['temperature_after_mC'],'against_exported_weights_fp32':outputs,'output_sha256':sha(f)}
 summary.append(q)
 key=(layer,m)
 if key in first:
  prior=first[key];drift.append({'first_layer':layer,'query_tokens':m,'p50_ratio':q['p50_api_s']/prior['p50_api_s'],'byte_equal':q['output_sha256']==prior['output_sha256']})
 else:first[key]=q
max_vs_m256=[]
for layer in [0,2]:
 big=np.fromfile(out/f'c512-m256-l{layer}.f32',dtype='<f4')
 for m in [16,32,64,128]:
  small=np.fromfile(out/f'c512-m{m}-l{layer}.f32',dtype='<f4');pieces=[];offset=0
  for idx,width in enumerate([896,128,128,128,128]):
   chunk=big[offset:offset+256*width];offset+=256*width
   pieces.append(chunk.reshape(1,256,896)[:,:m].ravel() if idx==0 else chunk.reshape(1,2,256,64)[:,:,:m].ravel())
  compare=np.concatenate(pieces);max_vs_m256.append({'first_layer':layer,'query_tokens':m,**errors(compare,small)})
report={'scope':'12 repeated-input single fused2 probes, no whole-model TTFT; no speech/RAG load','status':'measurement_complete_not_quality_qualified','rows':summary,'baseline_repeat_drift':drift,'cross_batch_common_rows_vs_m256':max_vs_m256,'raw_sha256':{n:sha(out/n) for n in ['events.jsonl','resources.jsonl','meta.json','config.json']}}
(out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# C512 大batch单块初筛','', '这是两层fused block的重复相同输入NPU API时间，不是24层完整prefill、首token或完整系统延迟。未开启语音/RAG背景；同层采用唯一canonical FP32 history输入；3 warmup +30计时，前后M16复测。','', '|起始层|M|API P50 ms|P95 ms|每输入行 ms|driver MiB|','|---:|---:|---:|---:|---:|---:|']
for q in summary:lines.append(f"|{q['first_layer']}|{q['query_tokens']}|{q['p50_api_s']*1000:.3f}|{q['p95_api_s']*1000:.3f}|{q['p50_api_s_per_input_row']*1000:.4f}|{q['allocated_bytes']/2**20:.1f}|")
lines+=['','所有真实f32输出重新核对SHA、长度、finite并对比同一输入下实际导出权重的FP32参考；分张量数值误差和跨batch共同输出比较见analysis.json。独立FP32历史不是板端捕获KV；不能由此宣称完整模型质量通过或100ms目标达成。','', '必须根据图吞吐、量化误差和资源证据筛选，再编译全层进行同输入计时。每API/M只是该图的批内平均成本，不能将更少调用次数当成所有请求同等倍数加速。']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'rows':[{k:x[k] for k in ['case','p50_api_s','p95_api_s','allocated_bytes']} for x in summary],'drift':drift,'scope':report['scope']}))
