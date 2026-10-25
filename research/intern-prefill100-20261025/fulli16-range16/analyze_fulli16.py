"""Read recovered evidence only; no board access or automatic quality approval."""
from pathlib import Path
import json,sys,hashlib,importlib.util,collections,numpy as np
from contract import validate_aba,sha,BINARY_SHA,NAMES
P=Path(__file__).resolve().parent;R=P.parent
spec=importlib.util.spec_from_file_location('metrics',R/'fullconv/analyze_fullconv.py');H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
run=sys.argv[1];D=P/'results'/run;J=lambda p:json.loads(p.read_text())
ctl=J(P/(run+'-controller.json'));cfg=J(D/'config.json');meta=J(D/'meta.json');fixtures=J(D/'fixtures.json');prov=J(D/'provenance.json')
assert ctl['status']=='complete' and ctl['returncode']==0 and meta['status']=='complete' and not meta['background_errors'];validate_aba(cfg)
events=H.rows(D/'events.jsonl');resources=H.rows(D/'resources.jsonl');counts=collections.Counter(e['event'] for e in events)
assert {k:counts[k] for k in ['measure','quality','diagnostic','loaded','closed','complete']}==dict(measure=72,quality=72,diagnostic=24,loaded=3,closed=3,complete=1)
assert not any(counts[k] for k in ['error','failed','audio_error','monitor_error','closed_on_failure','release_error'])
for n in ['board_fulli16.py','resident_worker_v1.py','context_rag.py','contract.py']:assert sha(P/n)==prov[n]==ctl['source_sha256'][n]
assert sha(P/'controller_fulli16.py')==cfg['controller_source_sha256'];assert sha(P/'inputs.json')==prov['fixture_sha256']==cfg['expected_fixture_sha256']
assert sha(P/(run+'-config.json'))==ctl['source_sha256']['config.json'];assert J(P/(run+'-config.json'))==cfg
assert sha(Path(cfg['candidate']['candidate_nb']))==cfg['candidate']['candidate_sha256']
assert [e['variant'] for e in events if e['event']=='loaded']==NAMES
assert [e['variant'] for e in events if e['event']=='closed']==NAMES
assert all(e['metrics']['driver_bytes']==0 for e in events if e['event']=='closed')
bycase={c['id']:c for c in fixtures['performance']+fixtures['quality']};measure=[e for e in events if e['event']=='measure'];quality=[e for e in events if e['event']=='quality'];diags=[e for e in events if e['event']=='diagnostic']
for e in measure+quality:
 c=bycase[e['case']];m=e['metrics'];assert m['prompt_tokens']==c['prompt_tokens'] and m['new_tokens']==c['body_tokens']+8 and m['reused_tokens']==c['static_tokens']
 assert not m['diagnostic'] and not e['prefix_prepare_metrics']['diagnostic'] and e['prefix_prepare_metrics']['reused_tokens']==0
 assert m['true_tail8_calls']==0 and np.isfinite([m['ttft_seconds'],m['total_seconds'],m['prefill_seconds'],m['scalar_prompt_seconds']]).all()
 if e['event']=='measure':assert m['fixed_feedback'] and m['generated_tokens']==32 and len(e['token_events'])==32 and e['feedback_tokens']==cfg['feedback_tokens']
 else:assert not m['fixed_feedback'] and m['generated_tokens']==len(m['tokens'])<=32
 assert m['new_tokens']+m['reused_tokens']==m['prompt_tokens']
cells=[]
for static in [64,128]:
 for body in [64,128,192,256]:
  v={}
  for name in NAMES:
   rows=[e for e in measure if e['variant']==name and e['static_tokens']==static and e['body_tokens']==body];assert len(rows)==3;v[name]=H.cell_stats(rows)
  cells.append({'static_tokens':static,'system_body_tokens':static-5,'system_template_tokens':5,'dynamic_body_tokens':body,'dynamic_template_tokens':8,'new_tokens':body+8,'variants':v})
logits={};diagcomparisons=[]
for e in diags:
 m=e['metrics'];c=bycase[e['case']];assert m['diagnostic'] and m['fixed_feedback'] and m['generated_tokens']==4
 assert m['new_tokens']==c['body_tokens']+8 and m['reused_tokens']==c['static_tokens']
 p=D/e['logits_file'];assert p.stat().st_size==2426784==e['bytes'] and sha(p)==e['sha256'];x=np.fromfile(p,dtype='<f4').reshape(4,151674);assert np.isfinite(x).all()
 logits[(e['variant'],e['case'])]=x
for name in NAMES[1:]:
 for case in [e['case'] for e in diags if e['variant']==NAMES[0]]:diagcomparisons.append({'variant':name,'case':case,**H.logit_compare(logits[(NAMES[0],case)],logits[(name,case)])})
qrows=[]
for c in fixtures['quality']:
 rows={n:next(e for e in quality if e['variant']==n and e['case']==c['id']) for n in NAMES};qrows.append({'id':c['id'],'input':c['body'],'gold':c['gold'],'replies':{n:e['reply'] for n,e in rows.items()},'tokens':{n:e['metrics']['tokens'] for n,e in rows.items()},'candidate_exact_baseline':rows[NAMES[0]]['metrics']['tokens']==rows[NAMES[1]]['metrics']['tokens'],'repeat_exact_baseline':rows[NAMES[0]]['metrics']['tokens']==rows[NAMES[2]]['metrics']['tokens'],'rag_top3':{n:e['top3'] for n,e in rows.items()},'manual_quality_status':'pending review; token equality alone is not semantic correctness'})
ivs=H.intervals(resources);res=H.resource_bucket(resources,ivs)
report={'scope':cfg['scope'],'counts':dict(counts),'cells':cells,'natural_quality':qrows,'diagnostics':diagcomparisons,'resources':res,'audio_final':meta['audio'],'three_driver_releases_zero':True,'small_sample_note':'3 different frozen cost prompts per cell, one pass; not sustained latency stability','independence':'24 natural presentations/12 scenes already used during range debugging; independent-household-v2 frozen separately, not tested here','full_goal_reached':all(e['metrics']['ttft_seconds']<=.1 and e['metrics']['prefill_seconds']+e['metrics']['scalar_prompt_seconds']<=.1 for e in measure if e['variant']==NAMES[1])}
(D/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
text='# 修复范围I16 down：完整模型配对\n\nIntern NPU文本，C1280、prefill16；只替换层2/3 prefill块，decode/head/embedding和runtime保持相同。语音前端流式回放合成WAV，RAG常驻且只在自然输出后查询。不是视觉、真实麦克风、电机或语音触发完整agent。每次清空并准备system缓存，只复用固定前缀；system构建排除热请求TTFT，但计入全段资源统计。\n\nsystem正文59/123＋模板5＝静态64/128；动态正文64/128/192/256＋模板8＝新72/136/200/264。性能输出32固定反馈token；自然输出上限32、实际含EOS数量见analysis。每格3案，未据此声称长期稳定。\n\n|静态|动态正文|新prefill|原prefill→I16(ms)|原TTFT→I16(ms)|原decode→I16(tok/s)|原图重复TTFT(ms)|\n|---:|---:|---:|---:|---:|---:|---:|\n'
for c in cells:
 a,b,z=[c['variants'][n] for n in NAMES];f=lambda x,k:x[k]['p50'];text+=f"|{c['static_tokens']}|{c['dynamic_body_tokens']}|{c['new_tokens']}|{1000*f(a,'dynamic_prefill_seconds'):.1f}→{1000*f(b,'dynamic_prefill_seconds'):.1f}|{1000*f(a,'ttft_seconds'):.1f}→{1000*f(b,'ttft_seconds'):.1f}|{f(a,'decode_tokens_s'):.2f}→{f(b,'decode_tokens_s'):.2f}|{1000*f(z,'ttft_seconds'):.1f}|\n"
text+=f"\n候选自然输出与原图完全相同：{sum(x['candidate_exact_baseline'] for x in qrows)}/24；原图前后相同：{sum(x['repeat_exact_baseline'] for x in qrows)}/24。尚需人工按各高层指令评估是否可执行、是否遗漏约束；不能把token差异直接当错误率。此集合已用于调试，独立新集未跑。\n"
text+=f"\n最高温度{res['temperature_C']['max']:.3f}°C；全8核CPU加权占用{res['cpu_weighted_mean_percent_whole_8cores']:.2f}%；所采两进程RSS合计峰值{res['process_RSS_sum_MiB']['max']:.1f}MiB、PSS{res['process_PSS_sum_MiB']['max']:.1f}MiB；MemAvailable最低{res['MemAvailable_MiB']['min']:.1f}MiB。这些数据包含加载、system重建和诊断阶段，不是纯生产热循环。3次NPU释放均0。\n"
text+='\n全部原始logits文件实际重算SHA、检查finite后分析；详见analysis.json。首token与prefill100ms目标是否在当前24请求全部满足：'+str(report['full_goal_reached'])+'；即便满足，本次小样本也不能证明全部场景稳定达标。\n'
(D/'REPORT.md').write_text(text);print(text)
