"""Read recovered evidence only; no board access or automatic quality approval."""
from pathlib import Path
import json,sys,hashlib,importlib.util,collections,tarfile,numpy as np
from contract import validate_aba,sha,BINARY_SHA
NAMES=['original-m32','row-scale-m32','protected-system','original-m32-repeat']
P=Path(__file__).resolve().parent;R=P.parent
spec=importlib.util.spec_from_file_location('metrics',R/'fullconv/analyze_fullconv.py');H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
run=sys.argv[1];D=P/'results'/run;J=lambda p:json.loads(p.read_text())
ctl=J(P/(run+'-controller.json'));cfg=J(D/'config.json');meta=J(D/'meta.json');fixtures=J(D/'fixtures.json');prov=J(D/'provenance.json')
assert ctl['status']=='complete' and ctl['returncode']==0 and meta['status']=='complete' and not meta['background_errors'];validate_aba(cfg)
archive=P/(run+'.tar.gz');assert sha(archive)==ctl['archive_sha256'] and archive.stat().st_size==ctl['archive_bytes']
archive_files=[]
with tarfile.open(archive) as tar:
 for member in tar.getmembers():
  path=(D/member.name).resolve();assert path==D.resolve() or D.resolve() in path.parents
  assert member.isdir() or member.isfile()
  if member.isfile():
   assert path.stat().st_size==member.size
   source=tar.extractfile(member);digest=hashlib.sha256()
   for chunk in iter(lambda:source.read(4194304),b''):digest.update(chunk)
   assert digest.hexdigest()==sha(path),member.name
   archive_files.append(member.name)
assert len(archive_files)==len(set(archive_files))
events=H.rows(D/'events.jsonl');resources=H.rows(D/'resources.jsonl');counts=collections.Counter(e['event'] for e in events)
assert {k:counts[k] for k in ['measure','quality','diagnostic','prefix_diagnostic','cache_boundary_probe','loaded','closed','complete']}==dict(measure=96,quality=192,diagnostic=32,prefix_diagnostic=8,cache_boundary_probe=4,loaded=4,closed=4,complete=1)
assert not any(counts[k] for k in ['error','failed','audio_error','monitor_error','closed_on_failure','release_error'])
for n in ['board_fulli16.py','resident_worker_v1.py','context_rag.py','contract.py']:assert sha(P/n)==prov[n]==ctl['source_sha256'][n]
assert sha(P/'controller.py')==ctl['source_sha256']['controller.py'];assert sha(P/'inputs.json')==prov['fixture_sha256']==cfg['expected_fixture_sha256'];assert fixtures==J(P/'inputs.json') and prov['tokenizer_sha256']==fixtures['tokenizer_sha256']
assert sha(P/'config.json')==ctl['source_sha256']['config.json'];assert J(P/'config.json')==cfg
assert cfg['candidate']['sha256']==ctl['source_sha256']['candidate.nb']
assert sha(P/'static-audit.json')==cfg['candidate']['static_audit_sha256']==ctl['source_sha256']['static-audit.json']
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
 assert loaded[0]['ready']['layers']==24 and loaded[0]['ready']['capacity']==512 and loaded[0]['ready']['true_tail8_enabled'] and loaded[0]['ready']['single_pass_greedy_enabled']
extra=[e for e in events if e['event']=='extra_prefix_graph_verified'];assert len(extra)==1 and extra[0]['variant']=='protected-system'
v=cfg['variants'][2];base=cfg['variants'][0];expected={v['env']['OPI_FIRST_PREFILL_L2']:base['expected_graph_sha256'][v['env']['OPI_FIRST_PREFILL_L2']],v['env']['OPI_FIRST_PREFILL8_L2']:v['expected_tail_sha256'][v['env']['OPI_FIRST_PREFILL8_L2']]}
assert {f['declared_path']:f['sha256'] for f in extra[0]['files']}==expected and all(not f['cache_reused'] for f in extra[0]['files'])
for v in cfg['variants']:
 ts=[e for e in events if e['event']=='tail_graph_verified' and e['variant']==v['name']];assert len(ts)==1 and len(ts[0]['files'])==12
 assert {f['declared_path']:f['sha256'] for f in ts[0]['files']}==v['expected_tail_sha256']

for e in measure+quality:
 c=bycase[e['case']];m=e['metrics'];assert m['prompt_tokens']==c['prompt_tokens'] and m['new_tokens']==c['body_tokens']+8 and m['reused_tokens']==c['static_tokens']
 assert not m['diagnostic'] and not e['prefix_prepare_metrics']['diagnostic'] and e['prefix_prepare_metrics']['reused_tokens']==0
 assert m['true_tail8_calls']==1 and m['tail_rows']==8 and m['scalar_prompt_tokens']==0
 batch=next(v['prefill_batch'] for v in cfg['variants'] if v['name']==e['variant']);assert m['batch_calls']==c['body_tokens']//batch+1
 assert np.isfinite([m['ttft_seconds'],m['total_seconds'],m['prefill_seconds'],m['scalar_prompt_seconds']]).all()
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
 batch=next(v['prefill_batch'] for v in cfg['variants'] if v['name']==e['variant'])
 assert m['true_tail8_calls']==1 and m['tail_rows']==8 and m['scalar_prompt_tokens']==0 and m['batch_calls']==c['body_tokens']//batch+1
 p=D/e['logits_file'];assert p.stat().st_size==2426784==e['bytes'] and sha(p)==e['sha256'];x=np.fromfile(p,dtype='<f4').reshape(4,151674);assert np.isfinite(x).all()
 assert x.argmax(axis=1).tolist()==m['tokens'] and e['timing_eligible'] is False
 logits[(e['variant'],e['case'])]=x
for name in NAMES[1:]:
 for case in [e['case'] for e in diags if e['variant']==NAMES[0]]:diagcomparisons.append({'variant':name,'case':case,**H.logit_compare(logits[(NAMES[0],case)],logits[(name,case)])})

# Independently verify prefix-only logits and protected repeated-input cache caps.
prefix_rows=[e for e in events if e['event']=='prefix_diagnostic'];prefix_arrays={};prefix_results=[]
assert {(e['variant'],e['static_tokens']) for e in prefix_rows}=={(n,S) for n in NAMES for S in [64,128]}
for e in prefix_rows:
 p=D/e['logits_file'];raw=p.read_bytes();x=np.frombuffer(raw,dtype='<f4');m=e['metrics']
 assert len(raw)==606696==e['bytes'] and sha(p)==e['sha256'] and np.isfinite(x).all()
 assert m['diagnostic'] and m['generated_tokens']==1 and m['reused_tokens']==0 and m['new_tokens']==e['static_tokens']
 assert m['tokens']==[int(x.argmax())] and e['timing_eligible'] is False
 prefix_arrays[(e['variant'],e['static_tokens'])]=raw
for n in NAMES[1:]:
 for S in [64,128]:
  prefix_results.append({'variant':n,'static_tokens':S,'byte_equal':prefix_arrays[(n,S)]==prefix_arrays[(NAMES[0],S)]})
probes=[e for e in events if e['event']=='cache_boundary_probe']
assert {(e['static_tokens'],e['iteration']) for e in probes}=={(S,i) for S in [64,128] for i in [0,1]}
for e in probes:
 assert e['variant']=='protected-system' and e['timing_eligible'] is False
 c=bycase[e['case']];m=e['metrics'];assert m['reused_tokens']==c['static_tokens'] and m['new_tokens']==c['body_tokens']+8 and m['generated_tokens']==1
import re
for e in events:
 if e['event']!='closed':continue
 log=(D/(e['variant']+'.log')).read_text()
 actual=[tuple(map(int,m)) for m in re.findall(r'PREFIX_ORIGINAL block=1 pos=(\d+) rows=(\d+) valid=(\d+) limit=(\d+)',log)]
 expected=[]
 if e['variant']=='protected-system':
  requests=[fixtures['performance'][0]]+fixtures['performance']+fixtures['quality']+[c for c in fixtures['performance'] if c['id'] in diag_ids]
  for c in requests:expected.extend((pos,32,32,c['static_tokens']) for pos in range(0,c['static_tokens'],32))
  for system in fixtures['systems']:
   S=len(system['prefix_tokens']);expected.extend((pos,32,32,S) for pos in range(0,S,32))
 assert actual==expected and e['original_prefix_branches']==e['expected_original_prefix_branches']==len(expected)
 assert log.count('PREFIX_SUFFIX_GROUP blocks=10')==len(expected)
qrows=[]
for c in fixtures['quality']:
 rows={n:next(e for e in quality if e['variant']==n and e['case']==c['id']) for n in NAMES}
 qrows.append({'id':c['id'],'quality_set':c['quality_set'],'input':c['body'],'gold':c['gold'],'replies':{n:e['reply'] for n,e in rows.items()},'tokens':{n:e['metrics']['tokens'] for n,e in rows.items()},'exact_baseline':{n:rows[n]['metrics']['tokens']==rows[NAMES[0]]['metrics']['tokens'] for n in NAMES},'rag_top3':{n:e['top3'] for n,e in rows.items()},'manual_quality_status':'pending; parity not accuracy'})
ivs=H.intervals(resources);res=H.resource_bucket(resources,ivs)
report={'scope':cfg['scope'],'counts':dict(counts),'cells':cells,'natural_quality':qrows,'diagnostics':diagcomparisons,'prefix_logit_comparisons':prefix_results,'cache_boundary_probes':probes,'resources':res,'audio_final':meta['audio'],'four_driver_releases_zero':True,'archive_verified':{'sha256':ctl['archive_sha256'],'files':len(archive_files)},'quality_exact_baseline':{n:sum(q['exact_baseline'][n] for q in qrows) for n in NAMES},'whole_objective_verified':False,'coverage_limit':'3 frozen cost prompts per cell, 48 reused regression cases per variant. Not long stability; prefix-logit byte equality is not all-KV byte proof.'}
report['resources_by_variant']={n:H.resource_bucket([x for x in resources if x['variant']==n],[x for x in ivs if x['variant']==n and x['same_bucket']]) for n in NAMES}
report['performance_baseline_repeat_token_exact']=sum(next(z for z in measure if z['variant']==NAMES[-1] and z['case']==a['case'])['metrics']['tokens']==a['metrics']['tokens'] for a in measure if a['variant']==NAMES[0])
report['measured_grid_all_below100ms']={n:all(e['metrics']['ttft_seconds']<=.1 and e['dynamic_prefill_seconds']<=.1 for e in measure if e['variant']==n) for n in NAMES}
report['resources_performance_phase']={n:H.resource_bucket([x for x in resources if x['variant']==n and x['phase']=='performance'],[x for x in ivs if x['variant']==n and x['phase']=='performance' and x['same_bucket']]) for n in NAMES}
report['paired_performance']=[]
for c in fixtures['performance']:
 byname={n:next(e['metrics'] for e in measure if e['case']==c['id'] and e['variant']==n) for n in NAMES}
 a=byname[NAMES[0]]
 report['paired_performance'].append({'case':c['id'],'ratios_vs_original':{n:{'ttft':m['ttft_seconds']/a['ttft_seconds'],'decode':(a['total_seconds']-a['ttft_seconds'])/(m['total_seconds']-m['ttft_seconds'])} for n,m in byname.items()}})
report['analysis_source_sha256']={str(p.relative_to(R)):sha(p) for p in [Path(__file__),R/'fullconv/analyze_fullconv.py',P/'contract.py']}
(D/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
text='# C512/M32 保留原始 system 路径的逐行量化对照\n\n同一新binary四组：原图、全程row-scale、原始system＋动态row-scale、原图重复。静态正文59/123＋模板5＝64/128；动态正文64/128/192/256＋模板8＝新增72/136/200/264。仅复用system。性能输出固定反馈32token；自然输出上限32。system冷构建不计热TTFT。\n\n纯Intern NPU文本、合成语音前端和RAG驻留；无视觉/真实麦克风/电机。保护路径每个块分3次执行（原组为1次），因此包括这部分调度开销。各格3案，不能称长期稳定；质量待人工复核，不能将token一致率当准确率。\n\n|静态|动态正文|新增prefill|原图 TTFT ms|全程量化 TTFT ms|原system量化动态 TTFT ms|原图重复 TTFT ms|保护组 prefill ms|保护组 decode tok/s|\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
for c in cells:
 a,b,p,z=[c['variants'][n] for n in NAMES];f=lambda x,k:x[k]['p50']
 text+=f"|{c['static_tokens']}|{c['dynamic_body_tokens']}|{c['new_tokens']}|{1000*f(a,'ttft_seconds'):.1f}|{1000*f(b,'ttft_seconds'):.1f}|{1000*f(p,'ttft_seconds'):.1f}|{1000*f(z,'ttft_seconds'):.1f}|{1000*f(p,'dynamic_prefill_seconds'):.1f}|{f(p,'decode_tokens_s'):.2f}|\n"
text+='\n自然token完全一致数（各48例，非正确率）：'+json.dumps(report['quality_exact_baseline'],ensure_ascii=False)+'。\n'
text+='\nSystem-only logits检查：'+json.dumps(prefix_results,ensure_ascii=False)+'。不能据此证明所有KV逐字节一致。\n'
text+=f"\n最高温度{res['temperature_C']['max']:.3f}°C；全8核CPU加权占用{res['cpu_weighted_mean_percent_whole_8cores']:.2f}%；进程PSS峰{res['process_PSS_sum_MiB']['max']:.1f}MiB；MemAvailable最低{res['MemAvailable_MiB']['min']:.1f}MiB。含加载/诊断；不是纯热循环或全系统内存。四次driver释放0。\n"
text+='\n全部性能格≤100ms：'+json.dumps(report['measured_grid_all_below100ms'],ensure_ascii=False)+'。完整目标仍未验证。\n'
text+='\n音频结束统计：'+json.dumps(meta['audio'],ensure_ascii=False)+'。overrun与lag单列，不能假定全程实时。\n'
text+=f"\nSwap占用范围{res['swap_used_MiB']['min']:.1f}–{res['swap_used_MiB']['max']:.1f}MiB；监测区间swap-in{res['swapin_pages_delta_included_intervals']}页、swap-out{res['swapout_pages_delta_included_intervals']}页。观测时长{res['observed_span_s']/60:.2f}分钟，含加载、切换与诊断；不归因于单一候选。\n"
text+='\n|静态|动态正文|保护组TTFT P95 ms|保护组setup P50 ms|保护组head P50 ms|≤100ms次数/3|\n|---:|---:|---:|---:|---:|---:|\n'
for c in cells:
 es=[e for e in measure if e['variant']=='protected-system' and e['static_tokens']==c['static_tokens'] and e['body_tokens']==c['dynamic_body_tokens']]
 text+=f"|{c['static_tokens']}|{c['dynamic_body_tokens']}|{max(e['metrics']['ttft_seconds'] for e in es)*1000:.2f}|{np.median([e['metrics']['setup_seconds'] for e in es])*1000:.2f}|{np.median([e['metrics']['first_head_seconds'] for e in es])*1000:.2f}|{sum(e['metrics']['ttft_seconds']<=.1 for e in es)}/3|\n"
text+='\nP95为nearest rank，n=3时等于最大值。分phase资源与逐case原图重复漂移见analysis.json。\n'
(D/'REPORT.md').write_text(text);print(text)
