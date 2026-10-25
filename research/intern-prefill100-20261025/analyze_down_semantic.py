from pathlib import Path
import json,sys,numpy as np
from analyze_down_i16 import sha,ports,compare
R=Path(__file__).resolve().parent;D=R/'results'/(sys.argv[1] if len(sys.argv)>1 else 'down-i16-semantic-r1');cfg=json.loads((D/'config.json').read_text());meta=json.loads((D/'meta.json').read_text())
assert meta['status']=='complete' and not meta['monitor_errors'];assert len(cfg['sequence'])==96 and cfg['repeats']==2 and cfg['warmup']==1
es=[json.loads(x) for x in (D/'events.jsonl').read_text().splitlines()];ev={x['case']:x for x in es if x['event']=='result'};assert len(ev)==96 and not any(x['event'] in ['error','failed'] for x in es)
rows=[];group={}
for c in cfg['sequence']:
 id=c['run_id'];p=D/(id+'.f32');e=ev[id];m=json.loads((D/(id+'.json')).read_text());assert e['metrics']==m
 assert sha(p)==e['output_sha256'] and p.stat().st_size==180224
 assert m['released_bytes']==0 and m['warmup_repeats']==1 and m['repeats']==2 and m['differing_repeats']==0 and m['mismatched_elements']==0
 assert not m['npd_enabled'] and not m['io_profile_enabled'] and m['resident_copies']==0
 assert m['capacity']==512 and m['query_rows']==32 and m['fusion_layers']==2
 x=ports(np.fromfile(p,dtype='<f4'));ref=Path(c['local_reference']);assert sha(ref)==c['reference_sha256']
 row={'id':id,'fixture_id':c['fixture_id'],'variant':c['variant'],'static_tokens':c['static_tokens'],'dynamic_body_tokens':c['body_tokens'],'dynamic_template_tokens':8,'history':c['history'],'window':c['window'],'output_sha256':sha(p),'against_fp32_reference':compare(x,ports(np.fromfile(ref,dtype='<f4')))}
 rows.append(row);group.setdefault(c['fixture_id'],{})[c['variant']]=x
pairs=[{'fixture':k,'candidate_vs_original':compare(v['candidate'],v['original'])} for k,v in group.items()];assert len(pairs)==48
agg=[]
for name in group[next(iter(group))]['original']:
 values=[next(v for v in p['candidate_vs_original'] if v['tensor']==name) for p in pairs];z=np.array([v['relative_l2'] for v in values]);worst=int(np.argmax(z))
 agg.append({'tensor':name,'n':48,'relative_l2_p50':float(np.median(z)),'relative_l2_max':float(z.max()),'relative_l2_p95':float(np.quantile(z,.95)),'max_abs':max(v['max_abs'] for v in values),'byte_equal_cases':sum(v['byte_equal'] for v in values),'worst_case':pairs[worst]['fixture']})
resources=[json.loads(x) for x in (D/'resources.jsonl').read_text().splitlines()]
report={'scope':cfg['scope'],'cases':48,'runs':96,'independent_scenarios':12,'finite_all':True,'all_releases_zero':True,'aggregates':agg,'pairs':pairs,'rows':rows,'temperature_peak_C':max(v for x in resources for v in x['temperatures_mC'].values())/1000,'memory_available_min_bytes':min(int(x['memory']['MemAvailable'].split()[0])*1024 for x in resources),'limitations':['physical query32 windows, not complete dynamic processing','FP32 source history, not NPU-captured KV','single-block outputs, not full-model logits or task quality','two repeats for determinism; not latency percentile benchmark','calibration fixed from one other trajectory, no evaluation refit']}
(D/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
text='# INT16 down：48个输入窗口数值对照\n\n静态正文59/123+模板5=64/128；动态正文64/128/192/256+模板8=完整新增72/136/200/264。本次只截取每提示首/末各32行测层2/3，不能当作上述完整prefill。24个提示来自12种场景，共48窗口、96次原版/候选运行。每次预热1次、重复2次检查确定性，不用于速度分位数宣称。\n\n|输出|relative L2 中位数|最大值|最大绝对误差|逐位相同窗口|\n|---|---:|---:|---:|---:|\n'
for x in agg:text+=f"|{x['tensor']}|{100*x['relative_l2_p50']:.5f}%|{100*x['relative_l2_max']:.5f}%|{x['max_abs']:.7g}|{x['byte_equal_cases']}/48|\n"
text+=f"\n全部输出finite、96次释放后driver=0。采样最高{report['temperature_peak_C']:.3f}°C。\n\n输入history由Q8源权重FP32参考生成，未模拟上游NPU全部舍入。校准仅一条轨迹，以上是单块数值扩展测试，不能证明完整模型回答正确，也不能证明100ms TTFT。\n"
text+='\n该48窗口现已用于范围调整，统一按调参回归集合解释；独立样例与整模型质量尚待验证。\n'
(D/'REPORT.md').write_text(text);print(json.dumps({'aggregates':agg,'temperature_peak_C':report['temperature_peak_C']},indent=2))
