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
assert {k:counts[k] for k in ['measure','quality','diagnostic','loaded','closed','complete']}==dict(measure=72,quality=144,diagnostic=24,loaded=3,closed=3,complete=1)
assert not any(counts[k] for k in ['error','failed','audio_error','monitor_error','closed_on_failure','release_error'])
for n in ['board_fulli16.py','resident_worker_v1.py','context_rag.py','contract.py']:assert sha(P/n)==prov[n]==ctl['source_sha256'][n]
assert sha(P/'controller_fulli16.py')==cfg['controller_source_sha256'];assert sha(P/'inputs.json')==prov['fixture_sha256']==cfg['expected_fixture_sha256'];assert fixtures==J(P/'inputs.json') and prov['tokenizer_sha256']==fixtures['tokenizer_sha256']
assert sha(P/(run+'-config.json'))==ctl['source_sha256']['config.json'];assert J(P/(run+'-config.json'))==cfg
assert sha(Path(cfg['candidate']['candidate_nb']))==cfg['candidate']['candidate_sha256']
assert [e['variant'] for e in events if e['event']=='loaded']==NAMES
assert [e['variant'] for e in events if e['event']=='closed']==NAMES
assert all(e['metrics']['driver_bytes']==0 for e in events if e['event']=='closed')
bycase={c['id']:c for c in fixtures['performance']+fixtures['quality']};measure=[e for e in events if e['event']=='measure'];quality=[e for e in events if e['event']=='quality'];diags=[e for e in events if e['event']=='diagnostic']
perf_ids={c['id'] for c in fixtures['performance']};quality_ids={c['id'] for c in fixtures['quality']}
assert len(perf_ids)==24 and len(quality_ids)==48 and not perf_ids&quality_ids
diag_ids={f's{s}-d{d}-0' for s in [64,128] for d in [64,128,192,256]}
for rows,ids in [(measure,perf_ids),(quality,quality_ids),(diags,diag_ids)]:
 keys=[(e['variant'],e['case']) for e in rows]
 assert len(keys)==len(set(keys)) and set(keys)=={(n,c) for n in NAMES for c in ids}
graphs=J(D/'graph-provenance.json');assert [g['variant'] for g in graphs]==NAMES
for g,v in zip(graphs,cfg['variants']):
 assert len(g['files'])==29 and {x['declared_path'] for x in g['files']}==set(v['expected_graph_sha256'])
 for f in g['files']:assert f['sha256']==f['expected_sha256']==v['expected_graph_sha256'][f['declared_path']] and f['bytes']>0
 ge=[e for e in events if e['event']=='graph_verified' and e['variant']==g['variant']];assert len(ge)==1 and ge[0]['files']==g['files']
 loaded=[e for e in events if e['event']=='loaded' and e['variant']==g['variant']];assert len(loaded)==1 and loaded[0]['binary_sha256']==BINARY_SHA
 assert loaded[0]['settings']==dict(cfg['defaults'],**v)
 assert loaded[0]['ready']['layers']==24 and loaded[0]['ready']['capacity']==1280 and not loaded[0]['ready']['true_tail8_enabled'] and loaded[0]['ready']['single_pass_greedy_enabled']
extra=[e for e in events if e['event']=='extra_first_chunk_graph'];assert len(extra)==1 and extra[0]['variant']==NAMES[1] and extra[0]['sha256']==cfg['candidate']['baseline_nb_sha256'] and extra[0]['path']==cfg['variants'][1]['env']['OPI_FIRST_PREFILL_L2']
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
 assert x.argmax(axis=1).tolist()==m['tokens'] and e['timing_eligible'] is False
 logits[(e['variant'],e['case'])]=x
for name in NAMES[1:]:
 for case in [e['case'] for e in diags if e['variant']==NAMES[0]]:diagcomparisons.append({'variant':name,'case':case,**H.logit_compare(logits[(NAMES[0],case)],logits[(name,case)])})
qrows=[]
for c in fixtures['quality']:
 rows={n:next(e for e in quality if e['variant']==n and e['case']==c['id']) for n in NAMES};qrows.append({'id':c['id'],'quality_set':c['quality_set'],'input':c['body'],'gold':c['gold'],'replies':{n:e['reply'] for n,e in rows.items()},'tokens':{n:e['metrics']['tokens'] for n,e in rows.items()},'candidate_exact_baseline':rows[NAMES[0]]['metrics']['tokens']==rows[NAMES[1]]['metrics']['tokens'],'repeat_exact_baseline':rows[NAMES[0]]['metrics']['tokens']==rows[NAMES[2]]['metrics']['tokens'],'rag_top3':{n:e['top3'] for n,e in rows.items()},'manual_quality_status':'pending review; token equality alone is not semantic correctness'})
ivs=H.intervals(resources);res=H.resource_bucket(resources,ivs)
report={'scope':cfg['scope'],'counts':dict(counts),'cells':cells,'natural_quality':qrows,'diagnostics':diagcomparisons,'resources':res,'audio_final':meta['audio'],'three_driver_releases_zero':True,'small_sample_note':'3 different frozen cost prompts per cell, one pass; not sustained latency stability','independence':'48 natural presentations: 24 development regression and 24 new frozen household v2. Separate sets; length-composed text, not spontaneous speech or proven training-disjoint','full_goal_reached':all(e['metrics']['ttft_seconds']<=.1 and e['metrics']['prefill_seconds']+e['metrics']['scalar_prompt_seconds']<=.1 for e in measure if e['variant']==NAMES[1])}
# Recompute branch use and distinguish frozen quality sets before any report.
assert collections.Counter(c['quality_set'] for c in fixtures['quality'])=={'development-regression':24,'new-frozen-household-v2':24}
for e in quality: assert e['quality_set']==bycase[e['case']]['quality_set']
for e in events:
 if e['event']=='closed':
  expected=81 if e['variant']==NAMES[1] else 0
  log=(D/(e['variant']+'.log')).read_text()
  assert log.count('FIRST_CHUNK_ORIGINAL block=1 pos=0 rows=16 valid=16')==e['first_chunk_original_calls']==expected
  assert log.count('FIRST_CHUNK_SUFFIX_GROUP blocks=10')==e['suffix_group_first_calls']==expected
report['quality_sets']={k:{'count':len(v),'candidate_exact_baseline':sum(q['candidate_exact_baseline'] for q in v),'repeat_exact_baseline':sum(q['repeat_exact_baseline'] for q in v)} for k in ['development-regression','new-frozen-household-v2'] for v in [[q for q in qrows if q['quality_set']==k]]}
report['first_chunk_original_and_suffix_markers']=81
report['performance_baseline_repeat_token_exact']=sum(next(z for z in measure if z['variant']==NAMES[2] and z['case']==a['case'])['metrics']['tokens']==a['metrics']['tokens'] for a in measure if a['variant']==NAMES[0])
report['resources_by_variant']={n:H.resource_bucket([x for x in resources if x['variant']==n],[x for x in ivs if x['variant']==n and x['same_bucket']]) for n in NAMES}
report['resources_performance_phase']={n:H.resource_bucket([x for x in resources if x['variant']==n and x['phase']=='performance'],[x for x in ivs if x['variant']==n and x['phase']=='performance' and x['same_bucket']]) for n in NAMES}
report['paired_performance']=[]
for c in fixtures['performance']:
 byname={n:next(e['metrics'] for e in measure if e['case']==c['id'] and e['variant']==n) for n in NAMES}
 a,b,z=[byname[n] for n in NAMES]
 report['paired_performance'].append({'case':c['id'],'candidate_ttft_ratio':b['ttft_seconds']/a['ttft_seconds'],'repeat_ttft_ratio':z['ttft_seconds']/a['ttft_seconds'],'candidate_decode_ratio':(a['total_seconds']-a['ttft_seconds'])/(b['total_seconds']-b['ttft_seconds']),'repeat_decode_ratio':(a['total_seconds']-a['ttft_seconds'])/(z['total_seconds']-z['ttft_seconds'])})
report['measured_grid_all_below100ms']=report.pop('full_goal_reached')
report['whole_objective_verified']=False
report['analysis_source_sha256']={str(p.relative_to(R)):sha(p) for p in [Path(__file__),R/'fullconv/analyze_fullconv.py',P/'contract.py']}
report['coverage_limit']='24 fixed cost requests per variant; only D64/128/192/256, S64/128. Boundary20 and continuous-length coverage not run; no long stability qualification.'
(D/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
text='# 首段原精度＋后续I16：完整模型配对\n\nIntern NPU文本，C1280、prefill16；只替换层2/3 prefill块的后续片段，位置0保留原块；候选每批3次提交（块0、块1、余下10块组），原图为整组提交。decode/head/embedding和binary相同。语音前端流式回放合成WAV，RAG常驻且只在自然输出后查询。无视觉、真实麦克风或电机。每次重置并准备system缓存，只复用固定前缀；system构建排除热请求TTFT，但计入资源。\n\nsystem正文59/123＋模板5＝静态64/128；动态正文64/128/192/256＋模板8＝新72/136/200/264。性能输出32固定反馈token；自然输出上限32、实际含EOS数量单列。每格3案，不证明长期稳定。诊断阶段有logits转储，不参与速度统计。\n\n|静态|动态正文|新prefill|原prefill→修复I16(ms)|原TTFT→修复I16(ms)|原decode→修复I16(tok/s)|原图重复TTFT(ms)|\n|---:|---:|---:|---:|---:|---:|---:|\n'
for c in cells:
 a,b,z=[c['variants'][n] for n in NAMES];f=lambda x,k:x[k]['p50'];text+=f"|{c['static_tokens']}|{c['dynamic_body_tokens']}|{c['new_tokens']}|{1000*f(a,'dynamic_prefill_seconds'):.1f}→{1000*f(b,'dynamic_prefill_seconds'):.1f}|{1000*f(a,'ttft_seconds'):.1f}→{1000*f(b,'ttft_seconds'):.1f}|{f(a,'decode_tokens_s'):.2f}→{f(b,'decode_tokens_s'):.2f}|{1000*f(z,'ttft_seconds'):.1f}|\n"
for label,qs in report['quality_sets'].items():
 text+=f"\n{label}：{qs['count']}例，候选与原图token完全一致{qs['candidate_exact_baseline']}例，原图前后完全一致{qs['repeat_exact_baseline']}例。token一致不是语义正确率，需按高层指令人工核查可执行性、约束遗漏及RAG结果。\n"
text+=f"\n最高温度{res['temperature_C']['max']:.3f}°C；全8核CPU加权占用{res['cpu_weighted_mean_percent_whole_8cores']:.2f}%；采样进程RSS合计峰值{res['process_RSS_sum_MiB']['max']:.1f}MiB、PSS{res['process_PSS_sum_MiB']['max']:.1f}MiB；MemAvailable最低{res['MemAvailable_MiB']['min']:.1f}MiB。包含加载、system重建和诊断，不是纯生产热循环。3次NPU释放均0。\n"
text+='\n全部24个4步logits文件实际重算SHA、finite和argmax后分析；详见analysis.json。目标100ms在当前24请求全部满足：'+str(report['measured_grid_all_below100ms'])+'。\n'
(D/'REPORT.md').write_text(text);print(text)
