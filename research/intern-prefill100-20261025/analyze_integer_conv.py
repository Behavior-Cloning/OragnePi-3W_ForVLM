#!/usr/bin/env python3
"""Analyze Integer Conv timing and five tensor outputs, not model quality."""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np
R=Path(__file__).resolve().parent
PIN='eed89e71c0030ea15fb90d490611b6d6efdcf88a530906fdc57a1c42498e4da3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ports(data):
 shapes=[('hidden',(1,32,896)),('layer0_new_k',(1,2,32,64)),('layer1_new_k',(1,2,32,64)),('layer0_new_v',(1,2,32,64)),('layer1_new_v',(1,2,32,64))];o=0;result={}
 for name,shape in shapes:
  n=int(np.prod(shape));result[name]=data[o:o+n].reshape(shape);o+=n
 assert o==data.size;return result
def compare(a,b):
 result=[]
 for name,x in a.items():
  y=b[name];assert np.isfinite(x).all() and np.isfinite(y).all()
  d=x.astype(np.float64)-y.astype(np.float64);norm=float(np.linalg.norm(y.astype(np.float64)));result.append({'tensor':name,'finite':True,'byte_equal':x.tobytes()==y.tobytes(),'different_elements':int(np.sum(x!=y)),'elements':int(x.size),'relative_l2':float(np.linalg.norm(d)/max(norm,1e-30)),'max_abs':float(np.max(abs(d)))})
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('run');a=ap.parse_args();assert a.run.replace('-','').replace('_','').isalnum();d=R/'results'/a.run
 cfg=json.loads((d/'config.json').read_text());meta=json.loads((d/'meta.json').read_text());assert meta['status']=='complete' and not meta['monitor_errors']
 assert cfg['binary_sha256']==PIN and cfg['warmup']==3 and cfg['repeats']==30
 assert [x['run_id'] for x in cfg['sequence']]==['original-before','candidate','original-after']
 assert len({x['input_sha256'] for x in cfg['sequence']})==1 and len({x['reference_sha256'] for x in cfg['sequence']})==1
 assert cfg['sequence'][0]['nb_sha256']==cfg['sequence'][2]['nb_sha256']
 assert all(x['capacity']==512 and x['first_layer']==0 and x['query_tokens']==32 for x in cfg['sequence'])
 refpath=Path(cfg['sequence'][0]['local_reference']);assert sha(refpath)==cfg['canonical_reference_sha256'];reference=ports(np.fromfile(refpath,np.float32))
 events=[json.loads(line) for line in (d/'events.jsonl').read_text().splitlines()];results=[e for e in events if e['event']=='result'];assert len(results)==3
 assert not any(e['event'] in ['error','failed'] for e in events)
 rows=[];dumps={}
 for case in cfg['sequence']:
  name=case['run_id'];metrics=json.loads((d/(name+'.json')).read_text());event=next(e for e in results if e['case']==name);assert metrics==event['metrics']
  assert metrics['repeats']==30 and metrics['warmup_repeats']==3 and metrics['released_bytes']==0
  assert metrics['capacity']==512 and metrics['query_rows']==32 and metrics['fusion_layers']==2 and metrics['scope']=='two_layer_probe'
  assert metrics['npd_enabled'] is False and metrics['io_profile_enabled'] is False and metrics['resident_copies']==0
  assert metrics['mismatched_elements']==0 and metrics['differing_repeats']==0
  times=np.asarray(metrics['api_seconds'],np.float64);assert times.size==30 and np.isfinite(times).all() and (times>0).all()
  out=d/(name+'.f32');assert out.stat().st_size==180224 and sha(out)==event['output_sha256'];values=ports(np.fromfile(out,np.float32));dumps[name]=values
  rows.append({'case':name,'variant':case['variant'],'nb_sha256':case['nb_sha256'],'n':30,'p50_api_ms':float(np.quantile(times,.5)*1000),'query_tokens':32,'p50_api_ms_per_input_row':float(np.quantile(times,.5)*1000/32),'p95_api_ms':float(np.quantile(times,.95)*1000),'min_api_ms':float(times.min()*1000),'max_api_ms':float(times.max()*1000),'allocated_bytes':metrics['allocated_bytes'],'released_bytes':0,'output_sha256':sha(out),'against_exported_weights_fp32':compare(values,reference)})
 before,conv,after=rows;drift=(after['p50_api_ms']/before['p50_api_ms']-1)*100
 resources=[json.loads(x) for x in (d/'resources.jsonl').read_text().splitlines()];temps=[v for x in resources for v in x['temperatures_mC'].values()]
 memory=[int(x['memory']['MemAvailable'].split()[0])*1024 for x in resources if 'MemAvailable' in x['memory']]
 report={'scope':'same-input single fused2 layers0/1 M32 API only; excludes load/conversion/host IO, no full-model TTFT, no audio/RAG residency','status':'measurement_complete_not_full_model_quality_qualified','geometry':'physical query rows32, history128 from canonical fixture; not full prompt TTFT','rows':rows,'same_input_sha256':cfg['canonical_input_sha256'],'same_reference_sha256':cfg['canonical_reference_sha256'],'probe_sha256':PIN,'candidate_vs_original_before':compare(dumps['candidate'],dumps['original-before']),'original_repeat_vs_before':compare(dumps['original-after'],dumps['original-before']),'baseline_repeat_p50_drift_percent':drift,'candidate_speedup_vs_before':before['p50_api_ms']/conv['p50_api_ms'],'candidate_speedup_vs_after':after['p50_api_ms']/conv['p50_api_ms'],'environment_drift_note':'A-B-A checks this short interval only; differing temperatures/scheduling can still affect results','resource_samples':len(resources),'temperature_peak_C':max(temps)/1000 if temps else None,'mem_available_min_bytes':min(memory) if memory else None,'three_releases_zero':True,'diagnostic_samples_included':False,'no_full_model_accuracy_inference':True}
 (d/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 text='# 保持原整数数值的Conv单块配对测试\n\n同一C512、M32、真实层0/1 fused块，固定canonical输入、历史128个位置。本表物理query32行不等于完整动态输入；目标整模型仍为system正文59/123+模板5=64/128、动态正文64/128/192/256+模板8、新72/136/200/264。3次预热后各30次纯API计时。没有视觉、语音/RAG并发，也不是完整模型TTFT。\n\n|顺序|P50 API (ms)|P95 API (ms)|\n|---|---:|---:|\n'
 for row in rows:text+=f"|{row['case']}|{row['p50_api_ms']:.3f}|{row['p95_api_ms']:.3f}|\n"
 text+=f"\n原图前后P50漂移：{drift:+.3f}%；Conv候选相对前/后原图速度比：{report['candidate_speedup_vs_before']:.3f}× / {report['candidate_speedup_vs_after']:.3f}×。\n\n"
 text+='所有5个输出必须分开看，避免大幅值KV掩盖hidden差异：\n\n|输出|候选对原NPU relative L2|最大abs|不同元素|\n|---|---:|---:|---:|\n'
 for row in report['candidate_vs_original_before']:text+=f"|{row['tensor']}|{row['relative_l2']:.8g}|{row['max_abs']:.8g}|{row['different_elements']}/{row['elements']}|\n"
 text+='\n原图重复是否逐位相同：'+str(all(x['byte_equal'] for x in report['original_repeat_vs_before']))+'。3次进程释放值均为0；NPD/IO诊断样本未混入计时。'
 if report['temperature_peak_C'] is not None:text+=f"采样最高温度{report['temperature_peak_C']:.3f}°C。"
 text+='\n\n对实际导出权重FP32参考的误差逐端口列于analysis.json。该参考不模拟全部NPU激活舍入，单块相近也不能证明完整模型中文回答或动作正确。后续若扩展需完整固定反馈logits与自然任务质量验证。\n'
 (d/'REPORT.md').write_text(text);print(json.dumps({k:v for k,v in report.items() if k not in ['rows','candidate_vs_original_before','original_repeat_vs_before']},indent=2))
if __name__=='__main__':main()
