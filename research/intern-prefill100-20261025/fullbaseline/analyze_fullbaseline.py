#!/usr/bin/env python3
"""Offline WSL audit of recovered C1280 full-model baseline; never accesses board."""
from pathlib import Path
import argparse,collections,copy,hashlib,importlib.util,json,math,statistics
import numpy as np
from tokenizers import Tokenizer
P=Path(__file__).resolve().parent;RESEARCH=P.parents[1]
ap=argparse.ArgumentParser();ap.add_argument('run',nargs='?',default='fullbaseline-c1280-r1');args=ap.parse_args();R=P/'results'/args.run
J=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def stats(a):
 a=sorted(a)
 return {'n':len(a),'min':min(a),'p50':statistics.median(a),'p95':a[math.ceil(.95*len(a))-1],'max':max(a),'mean':statistics.mean(a)}
def near(a,b,tol=3e-6):assert abs(a-b)<=tol,(a,b)
E=[json.loads(l) for l in (R/'events.jsonl').read_text().splitlines()];RS=[json.loads(l) for l in (R/'resources.jsonl').read_text().splitlines()]
M=J(R/'meta.json');C=J(R/'config.json');F=J(R/'fixtures.json');PV=J(R/'provenance.json');CTL=J(P/(args.run+'-controller.json'))
assert M['status']==CTL['status']=='complete' and CTL['returncode']==0 and not M['background_errors']
counts=collections.Counter(e['event'] for e in E)
assert counts['measure']==48 and counts['quality']==12 and counts['diagnostic']==8 and counts['closed']==1 and counts['complete']==1
assert not any(counts[k] for k in ['error','release_error','audio_error','monitor_error','closed_on_failure'])
for n in ['board_fullbaseline.py','resident_worker_v1.py','context_rag.py']:assert sha(P/n)==PV[n]==CTL['source_sha256'][n]
assert sha(P/'controller_fullbaseline.py')==C['controller_source_sha256']
assert sha(P/'inputs.json')==PV['fixture_sha256']==C['expected_fixture_sha256']==CTL['source_sha256']['inputs.json']
assert sha(P/(args.run+'-config.json'))==CTL['source_sha256']['config.json']
assert J(P/(args.run+'-config.json'))==C  # Harness reserializes without trailing newline.
T=Path(J(P/'inputs.json')['tokenizer_path']);assert sha(T)==PV['tokenizer_sha256']==F['tokenizer_sha256']
assert sha(RESEARCH/'resident-agent/agent_replay.py')==PV['frontend_sha256']
assert sha(RESEARCH/'home-companion-atoms128-v2-20261022/atoms.json')==PV['rag_catalog_sha256']==CTL['source_sha256']['atoms.json']
spec=importlib.util.spec_from_file_location('bench',P/'board_fullbaseline.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
input_doc=J(P/'inputs.json');bench.attach_system_metadata(input_doc);assert input_doc==F
TOK=Tokenizer.from_file(str(T));diag_cases=bench.validate_fixture(copy.deepcopy(F),lambda x:TOK.encode(x,add_special_tokens=False).ids)
CASE={c['id']:c for c in F['performance']+F['quality']};perf=[e for e in E if e['event']=='measure'];qual=[e for e in E if e['event']=='quality'];diags=[e for e in E if e['event']=='diagnostic']
loaded=next(e for e in E if e['event']=='loaded');closed=next(e for e in E if e['event']=='closed')
assert loaded['ready']['layers']==24 and loaded['ready']['capacity']==1280 and not loaded['ready']['true_tail8_enabled'] and loaded['ready']['single_pass_greedy_enabled']
assert loaded['binary_sha256']=='ff758d8eb013b7ebd05eae1f95634ecef065b92f0bf1fca7b292256351a2900d'==CTL['source_sha256']['resident-combined-v2']
assert sha(RESEARCH/'intern-nextopt-20261025/runtime-combined-v2/resident-combined-v2')==loaded['binary_sha256']
assert closed['metrics']['driver_bytes']==0
G=J(R/'graph-provenance.json');assert len(G)==1 and len(G[0]['files'])==29
expected=C['variants'][0]['expected_graph_sha256'];oldcfg=J(RESEARCH/'intern-nextbench-20261025/config.candidates.json');oldmask=next(v for v in oldcfg['variants'] if v['name']=='mask-base');assert expected==oldmask['expected_graph_sha256']
assert {x['declared_path']:x['sha256'] for x in G[0]['files']}==expected
assert all(x['expected_sha256']==x['sha256'] for x in G[0]['files'])
for e in perf+qual+diags:
 c=CASE[e['case']];m=e['metrics'];assert m['prompt_tokens']==len(c['tokens']) and m['reused_tokens']==c['static_tokens'] and m['new_tokens']==c['body_tokens']+8
 assert m['prompt_tokens']==m['reused_tokens']+m['new_tokens'] and m['kv_tokens']==m['prompt_tokens']+m['generated_tokens']-1
 assert not m['true_tail8_calls'] and len(m['tokens'])==m['generated_tokens']
 assert m['diagnostic']==(e['event']=='diagnostic') and m['fixed_feedback']==(e['event']!='quality')
 assert all(math.isfinite(m[k]) and m[k]>=0 for k in ['prefill_seconds','scalar_prompt_seconds','ttft_seconds','total_seconds','setup_seconds','first_head_seconds'])
 if e['event']=='measure':
  assert m['generated_tokens']==32 and e['feedback_tokens']==C['feedback_tokens'] and e['speed_profileflag']=='none'
  assert e['prompt_sha256']==c['tokens_sha256'];near(e['dynamic_prefill_seconds'],m['prefill_seconds']+m['scalar_prompt_seconds'])
  assert m['batch_calls']==math.ceil(m['new_tokens']/16) and m['scalar_prompt_tokens']==0 and m['tail_rows']==8
 if e['event']=='diagnostic':assert m['generated_tokens']==4 and not e['timing_eligible']
 if e['event']!='diagnostic':
  prep=e['prefix_prepare_metrics'];assert prep['prompt_tokens']==prep['new_tokens']==c['static_tokens'] and prep['reused_tokens']==0 and prep['generated_tokens']==1
  tokens=e['token_events'];assert len(tokens)==m['generated_tokens'] and [x['token'] for x in tokens]==m['tokens'];near(tokens[0]['elapsed'],m['ttft_seconds'])
  assert all(b['elapsed']>a['elapsed'] for a,b in zip(tokens,tokens[1:]));near(m['decode_tokens_per_second'],(m['generated_tokens']-1)/(m['total_seconds']-m['ttft_seconds']),2e-4)
  if e['event']=='measure':
   seg=e['decode_segments'];assert sum(x['intervals'] for x in seg)==31;near(sum(x['seconds'] for x in seg),tokens[-1]['elapsed']-tokens[0]['elapsed'])
  else:
   assert TOK.decode(m['tokens'])==e['reply'];assert e['output_tokens_including_eos']==m['generated_tokens'];assert e['output_tokens_excluding_terminal_eos']==m['generated_tokens']-int(m['tokens'][-1]==151645)
   assert all(x['mock_only'] and not x['enabled_on_board'] for x in e['top3'])
groups=collections.defaultdict(list)
for e in perf:groups[(e['static_tokens'],e['body_tokens'])].append(e)
assert set(groups)=={(s,d) for s in [64,128] for d in [64,128,192,256]}
cells=[]
for (s,d),es in sorted(groups.items()):
 assert len(es)==6 and {(e['case'],e['repeat']) for e in es}=={(f's{s}-d{d}-{i}',r) for i in range(3) for r in range(2)}
 ms=[e['metrics'] for e in es];pref=[m['prefill_seconds']+m['scalar_prompt_seconds'] for m in ms]
 cells.append({'static_tokens':s,'system_body_tokens':s-5,'dynamic_body_tokens':d,'dynamic_template_tokens':8,'actual_new_tokens':d+8,'prompt_tokens':s+d+8,'n':6,'prefill_seconds':stats(pref),'ttft_seconds':stats([m['ttft_seconds'] for m in ms]),'caller_first_observed_s':stats([e['token_events'][0]['observed_s'] for e in es]),'decode_tokens_s':stats([31/(m['total_seconds']-m['ttft_seconds']) for m in ms]),'setup_seconds':stats([m['setup_seconds'] for m in ms]),'first_head_seconds':stats([m['first_head_seconds'] for m in ms]),'prefill_fraction_of_TTFT':stats([p/m['ttft_seconds'] for p,m in zip(pref,ms)]),'first_head_fraction_of_TTFT':stats([m['first_head_seconds']/m['ttft_seconds'] for m in ms]),'residual_TTFT_seconds':stats([m['ttft_seconds']-m['setup_seconds']-p-m['first_head_seconds'] for p,m in zip(pref,ms)]),'prefix_prepare_s':stats([e['prefix_prepare_metrics']['total_seconds'] for e in es]),'decode_segments':[{'positions':es[0]['decode_segments'][i]['output_positions'],'tokens_s':stats([e['decode_segments'][i]['tokens_s'] for e in es]),'fraction_of_decode':stats([e['decode_segments'][i]['seconds']/(e['metrics']['total_seconds']-e['metrics']['ttft_seconds']) for e in es])} for i in range(4)],'within_100ms_all_samples':max(pref)<=.1 and max(m['ttft_seconds'] for m in ms)<=.1})
repeat=[]
for c in F['performance']:
 a,b=sorted([e for e in perf if e['case']==c['id']],key=lambda e:e['repeat']);repeat.append({'case':c['id'],'tokens_equal':a['metrics']['tokens']==b['metrics']['tokens'],'TTFT_repeat_ratio':b['metrics']['ttft_seconds']/a['metrics']['ttft_seconds'],'decode_repeat_ratio':b['metrics']['decode_tokens_per_second']/a['metrics']['decode_tokens_per_second']})
diagnostic=[]
assert {e['case'] for e in diags}=={c['id'] for c in diag_cases}
for e in diags:
 f=R/e['logits_file'];assert sha(f)==e['sha256'] and f.stat().st_size==e['bytes']==4*151674*4
 arr=np.fromfile(f,dtype='<f4');assert np.isfinite(arr).all();assert arr.reshape(4,151674).argmax(1).tolist()==e['metrics']['tokens']
 predicted=next(x for x in perf if x['case']==e['case'])['metrics']['tokens'][:4];assert predicted==e['metrics']['tokens']
 diagnostic.append({'case':e['case'],'file':e['logits_file'],'bytes':e['bytes'],'sha256':sha(f),'finite':True,'argmax_matches':True,'matches_perf_first4':True})

def cpu(x):return [int(v) for v in x['cpu'][0].split()[1:9]]
cpus=[]
for a,b in zip(RS,RS[1:]):
 delta=[v-u for u,v in zip(cpu(a),cpu(b))];total=sum(delta)
 assert total>0
 cpus.append(100*(total-delta[3]-delta[4])/total)
delta=[v-u for u,v in zip(cpu(RS[0]),cpu(RS[-1]))];total=sum(delta)
audio=M['audio'];assert audio['error'] is None and all(x['frontend']['error'] is None for x in RS)
resources={'samples':len(RS),'sample_duration_s':RS[-1]['at']-RS[0]['at'],'sample_gap_s':stats([b['at']-a['at'] for a,b in zip(RS,RS[1:])]),'cpu_percent_whole_8cores':stats(cpus),'cpu_weighted_mean_percent':100*(total-delta[3]-delta[4])/total,'temperature_C':stats([x['temp_mC']/1000 for x in RS]),'temperature_scope':'thermal_zone0 only, not all zones','load1':stats([float(x['loadavg'].split()[0]) for x in RS]),'process_RSS_sum_MiB':stats([sum(z['VmRSS'] for z in x['processes'].values())/1024 for x in RS]),'process_PSS_sum_MiB':stats([sum(z['Pss'] for z in x['processes'].values())/1024 for x in RS]),'MemAvailable_MiB':stats([x['meminfo']['MemAvailable']/1024 for x in RS]),'swap_used_MiB':stats([(x['meminfo']['SwapTotal']-x['meminfo']['SwapFree'])/1024 for x in RS]),'swapin_pages_delta':RS[-1]['vmstat']['pswpin']-RS[0]['vmstat']['pswpin'],'swapout_pages_delta':RS[-1]['vmstat']['pswpout']-RS[0]['vmstat']['pswpout'],'frequency_kHz':{c:stats([x['freq_khz'][c] for x in RS]) for c in RS[0]['freq_khz']},'driver_ready_bytes':loaded['ready']['driver_bytes'],'driver_last_closed_bytes':closed['metrics']['driver_bytes'],'driver_sample_limit':'ready and request boundaries plus normal close, no continuous driver samples'}
resources['phases']={ph:{'n':len(rr:=[x for x in RS if x['phase']==ph]),'temperature_max_C':max(x['temp_mC'] for x in rr)/1000,'PSS_max_MiB':max(sum(z['Pss'] for z in x['processes'].values()) for x in rr)/1024,'available_min_MiB':min(x['meminfo']['MemAvailable'] for x in rr)/1024} for ph in sorted({x['phase'] for x in RS})}
frames=[x['frontend']['frames'] for x in RS];assert all(b>=a for a,b in zip(frames,frames[1:]))
audio_result={**audio,'effective_audio_seconds':audio['frames']*.016,'overrun_fraction':audio['overruns']/audio['frames'],'overrun_definition':'per-frame compute >16ms; not dropped frames','queue_semantics':'one queued whole WAV, not per-frame backlog','endpoints':counts['audio_endpoint'],'first_sample_max_lag_s':RS[0]['frontend']['max_lag_s'],'last_sample_max_lag_s':RS[-1]['frontend']['max_lag_s'],'stage_seconds_mean_per_audio_frame':{k:v/audio['frames'] for k,v in audio['stage_seconds'].items()},'not_speech_to_first_token_test':True}
quality=[]
miss={'s64-wave','s64-bend-not-look-up','s128-bend-not-look-up'}
for e in qual:
 quality.append({'case':e['case'],'input':e['input'],'expected':CASE[e['case']]['task_constraint_for_manual_audit'],'reply':e['reply'],'output_including_eos':e['output_tokens_including_eos'],'output_excluding_terminal_eos':e['output_tokens_excluding_terminal_eos'],'finish_reason':e['metrics']['finish_reason'],'task_following_pass_manual':e['case'] not in miss,'assessment':'safe under-response: requested action omitted' if e['case'] in miss else 'requested high-level action represented; no explicit forbidden movement in reply','ttft_seconds':e['metrics']['ttft_seconds'],'prefill_seconds':e['metrics']['prefill_seconds']+e['metrics']['scalar_prompt_seconds'],'total_seconds':e['metrics']['total_seconds'],'rag_seconds':e['rag_s'],'top3':[x['name'] for x in e['top3']],'top3_semantic_conflict':e['case']!='s128-wave','holdout':CASE[e['case']]['independent_holdout']})
comparison={'same_input_verified':False}
old=RESEARCH/'intern-nextbench-20261025/results/candidate-matrix-r3'
if (old/'events.jsonl').exists():
 oldF=J(old/'fixtures.json');oldE=[json.loads(l) for l in (old/'events.jsonl').read_text().splitlines()];oldPerf={(e['case'],e['repeat']):e for e in oldE if e['event']=='measure' and e['variant']=='mask-base'}
 oldCases={c['id']:c for c in oldF['performance']};pairs=[]
 assert F['systems']==oldF['systems']
 for e in perf:
  if e['body_tokens']!=64:continue
  c=CASE[e['case']];oc=oldCases[e['case']];assert c['tokens']==oc['tokens'] and c['prefix_tokens']==oc['prefix_tokens'] and c['tokens_sha256']==oc['tokens_sha256']
  o=oldPerf[(e['case'],e['repeat'])];a=o['metrics'];b=e['metrics'];assert o['feedback_tokens']==e['feedback_tokens'] and a['generated_tokens']==b['generated_tokens']==32
  pairs.append({'case':e['case'],'repeat':e['repeat'],'old_TTFT_s':a['ttft_seconds'],'new_TTFT_s':b['ttft_seconds'],'TTFT_ratio':b['ttft_seconds']/a['ttft_seconds'],'old_prefill_s':a['prefill_seconds']+a['scalar_prompt_seconds'],'new_prefill_s':b['prefill_seconds']+b['scalar_prompt_seconds'],'old_decode_tps':a['decode_tokens_per_second'],'new_decode_tps':b['decode_tokens_per_second'],'decode_ratio':b['decode_tokens_per_second']/a['decode_tokens_per_second'],'output_tokens_identical':a['tokens']==b['tokens']})
 od={e['case']:e for e in oldE if e['event']=='diagnostic' and e['variant']=='mask-base'};dcompare=[]
 for e in diags:
  if e['case'] not in od:continue
  o=od[e['case']];a=np.fromfile(old/o['logits_file'],dtype='<f4');b=np.fromfile(R/e['logits_file'],dtype='<f4');assert a.shape==b.shape and np.isfinite(a).all()
  delta=b.astype('float64')-a;dcompare.append({'case':e['case'],'bytes_equal':a.tobytes()==b.tobytes(),'relative_l2':float(np.linalg.norm(delta)/max(np.linalg.norm(a.astype('float64')),1e-30)),'max_abs':float(np.abs(delta).max())})
 comparison={'same_input_verified':True,'pairs':pairs,'same_systems':True,'diagnostic_overlap':dcompare,'source':str(old),'old_variant':'mask-base','new_variant':'c1280-fullbaseline','caveat':'Cross-run comparison. Same C1280 graphs and prompts, but old mask binary vs combined-v2 greedy head and close ordering. Not a new C512/bulk candidate speedup.','by_static':{str(s):{'n':len(z:=[x for x in pairs if x['case'].startswith(f's{s}-')]),'old_TTFT_s':stats([x['old_TTFT_s'] for x in z]),'new_TTFT_s':stats([x['new_TTFT_s'] for x in z]),'paired_TTFT_ratio':stats([x['TTFT_ratio'] for x in z]),'old_prefill_s':stats([x['old_prefill_s'] for x in z]),'new_prefill_s':stats([x['new_prefill_s'] for x in z]),'old_decode_tps':stats([x['old_decode_tps'] for x in z]),'new_decode_tps':stats([x['new_decode_tps'] for x in z]),'paired_decode_ratio':stats([x['decode_ratio'] for x in z])} for s in [64,128]}}
report={'status':'baseline_complete_not_100ms_goal','integrity_pass':True,'scope':M['scope'],'counts':dict(counts),'source_provenance_verified':True,'actual_fixture_retokenization_verified':True,'graph_records_verified':29,'graph_rehash_limit':'Recorded board actual SHA compared to frozen expected map; no fresh board access by analyst','binary_sha256':loaded['binary_sha256'],'load_seconds':loaded['ready']['load_seconds'],'cells':cells,'repeat_pairs':repeat,'repeat_exact_count':sum(x['tokens_equal'] for x in repeat),'diagnostic_files':diagnostic,'natural_quality':quality,'natural_task_pass_manual':sum(x['task_following_pass_manual'] for x in quality),'natural_quality_denominator':12,'quality_caveat':'six underlying short tasks repeated at two system lengths, existing prompts, not independent holdout; these are not 64..256-body quality validation','natural_RAG_seconds':stats([x['rag_seconds'] for x in quality]),'resources':resources,'audio':audio_result,'comparison_old_D64':comparison,'controller_seconds':CTL['ended']-CTL['started'],'raw_sha256':{n:sha(R/n) for n in ['events.jsonl','resources.jsonl','fixtures.json','config.json','provenance.json','graph-provenance.json','meta.json']},'limitations':['6 samples per cell; P95 nearest-rank equals observed max, not a production tail guarantee.','32 fixed feedback predictions do not form a free generated response.','System reset/prepared before each measurement; its separately logged cost contributes overall load but is excluded from hot TTFT.','48 timed calls plus diagnostics total about 8 minutes, not a long-run stability qualification.','No image encoder, true microphone, physical action execution or continuous RAG compute.','Frozen boundary20 not executed; no all-integer64..256 guarantee.','Graph/NPU operator breakdown absent in this unprofiled whole-model run.']}

perfR=[x for x in RS if x['phase']=='performance'];first=[x for x in perfR if x['at']<perfR[0]['at']+60];last=[x for x in perfR if x['at']>perfR[-1]['at']-60]
resources['performance_first_last_60s_PSS_MiB']=[stats([sum(z['Pss'] for z in x['processes'].values())/1024 for x in window]) for window in [first,last]]
resources['swapout_pages_by_observed_phase']={ph:sum(b['vmstat']['pswpout']-a['vmstat']['pswpout'] for a,b in zip(RS,RS[1:]) if b['phase']==ph) for ph in resources['phases']}
resources['swapout_MiB_if_page4096']=resources['swapout_pages_delta']*4096/2**20
report['quality_pass_definition']='Manual checking against frozen task_constraint: requested action must be represented, and explicit prohibited motion absent. Safe omission is a task failure, not automatically a conflicting action.'
report['quality_RAG_note']='All Top3 are mock/unadjudicated. 11/12 lists contain an opposite or forbidden action; the remaining arm-action list still has shared-resource/order constraints. Never execute all top3 unconditionally.'
(R/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
lines=['# Intern C1280：动态正文64～256完整模型基线独立报告','',
'本轮建立新输入范围的实测基线：全部8格运行完整，但没有达到100ms。热态首token中位数随正文长度由约1.58秒增至5.41秒；decode中位约14.65～14.69 token/s。采用原C1280图和既有group/mask/greedy运行时，不是C512或大batch候选的优化结果。','',
'## 实验与完整性','',
'常驻Intern 1B NPU文本decoder（24层、151674词表、C1280）＋流式语音前端（降噪/VAD/KWS/ASR）＋RAG模型。语音为预录WAV逐帧回放；本轮没有视觉encoder、真实麦克风、物理电机。RAG在性能阶段只是常驻，在12条自然质量阶段实际检索。','',
'系统前缀总长64/128，其中正文59/123、模板5。动态正文64/128/192/256，另有8模板token，实际新增72/136/200/264。每次reset并单独计算system，再提交完整prompt，实际复用严格等于64/128；没有复用动态KV。最大prompt392、32次预测占423 KV槽。system准备成本单独记录，不计入热请求TTFT，但会贡献本轮总负载；这与生产长期复用同一个system的总负载不同。','',
'独立核对48条性能（8格×3情景×2重复）、12条自然质量、8条各4步全词表诊断。24条性能case的两轮预测全部相同。输入经实际tokenizer重新校验，正文/模板/完整IDs/hash/cache/输出数断言通过；29项板端记录的实际graph SHA与旧冻结清单一致，二进制SHA为ff758d8eb013b7ebd05eae1f95634ecef065b92f0bf1fca7b292256351a2900d。source、原始fixture及controller上传SHA链通过。8份实际f32各2,426,784字节，SHA/finite/全词表argmax均通过，预测匹配性能前4步。1次正常closed，driver归零，rc0。','',
'## 八格性能','',
'完整prefill逐请求先计算prefill_seconds+scalar_prompt_seconds，再统计P50/P95。TTFT从runtime读完并验证token请求后开始，包含setup、全部动态prefill和首head/argmax；不包含tokenization、请求排队或system准备。每格仅6样本；这里P95采用nearest-rank，恰好等于观察到的最大值，不是生产尾延迟保证。','',
'|System总token|动态正文|实际新增|prefill P50/P95 秒|TTFT P50/P95 秒|decode P50 token/s|100ms全样本|','|---:|---:|---:|---:|---:|---:|---|']
for c in cells:
 p,t=c['prefill_seconds'],c['ttft_seconds'];lines.append(f"|{c['static_tokens']}|{c['dynamic_body_tokens']}|{c['actual_new_tokens']}|{p['p50']:.4f}/{p['p95']:.4f}|{t['p50']:.4f}/{t['p95']:.4f}|{c['decode_tokens_s']['p50']:.3f}|未通过|")
lines+=['','全部性能请求固定反馈32步、diagnostic=False；decode按31/(total−TTFT)计算。预测序列后续输入使用冻结反馈，并非自然回复，不能据此评价文本流畅性。','', '## 时间构成','', '|System/正文|prefill占TTFT P50|setup P50 ms|首head P50 ms|decode 2–8 / 9–16 / 17–24 / 25–32 token/s|','|---|---:|---:|---:|---|']
for c in cells:
 ds=' / '.join(f"{x['tokens_s']['p50']:.2f}" for x in c['decode_segments']);lines.append(f"|{c['static_tokens']}/{c['dynamic_body_tokens']}|{c['prefill_fraction_of_TTFT']['p50']*100:.2f}%|{c['setup_seconds']['p50']*1000:.2f}|{c['first_head_seconds']['p50']*1000:.2f}|{ds}|")
lines+=['','四段decode含7/8/8/8个间隔，合计31。各段占比及剩余计时开销在analysis.json。正文越长，prefill越占主导；本轮没有逐算子插桩，不能从此表推断某个NPU算子/引擎的耗时比例。','', '## 自然质量与RAG','',
'12条为6个短场景×两种system，不是独立holdout，也不是长正文64～256的自然质量验证。人工按冻结需求检查“是否表达所要求动作、是否违反明确禁止动作”：9/12完成要求；另3条选择保持静止，属于保守漏执行，不能写成正确完成。没有输出明确违反禁止移动的高层指令。','', '|Case|自然输出|含EOS token|要求是否完成|','|---|---|---:|---|']
for q in quality:lines.append(f"|{q['case']}|{q['reply']}|{q['output_including_eos']}|{'是' if q['task_following_pass_manual'] else '否：安全但漏动作'}|")
lines+=[f"RAG查询P50 {report['natural_RAG_seconds']['p50']*1000:.2f}ms、最大{report['natural_RAG_seconds']['max']*1000:.2f}ms。Top3仍不能全部执行：点头后召回摇头；保持静止后召回前移/后移。11/12列表有明显相反或禁止动作；其余摆臂列表仍需资源互斥和动作顺序裁定。全部保持mock。",'', '## 资源、音频与释放','',
f"915次约0.5秒采样覆盖{resources['sample_duration_s']:.2f}秒；controller含部署/回收共{report['controller_seconds']:.2f}秒。音频处理{audio['frames']}个16ms帧，等效{audio_result['effective_audio_seconds']:.3f}秒；音频在资源采样开始前已启动，因此不将这些时长混同。不是15分钟/1小时稳定性测试。",'',
f"整机8核CPU平均{resources['cpu_weighted_mean_percent']:.2f}%、P95 {resources['cpu_percent_whole_8cores']['p95']:.2f}%、峰值{resources['cpu_percent_whole_8cores']['max']:.2f}%；load1 P50 {resources['load1']['p50']:.2f}，不是CPU利用率。thermal_zone0温度P50 {resources['temperature_C']['p50']:.3f}°C、峰值{resources['temperature_C']['max']:.3f}°C，未触发78°C保护；未采所有温区，也未采NPU频率。CPU6/7全部样本2002MHz，CPU0–5在416～1794MHz变化。",'',
f"harness+worker总RSS峰{resources['process_RSS_sum_MiB']['max']:.2f}MiB、PSS峰{resources['process_PSS_sum_MiB']['max']:.2f}MiB；NPU ready占用{resources['driver_ready_bytes']/2**20:.2f}MiB，正常closed为0。系统MemAvailable最低{resources['MemAvailable_MiB']['min']:.2f}MiB。CPU进程PSS和driver应分列，不将其相加冒充整机唯一占用。",'',
f"swap已用从{resources['swap_used_MiB']['min']:.2f}增至{resources['swap_used_MiB']['max']:.2f}MiB；采样期间swap-in 0页、swap-out {resources['swapout_pages_delta']}页（若页长4096B，约{resources['swapout_MiB_if_page4096']:.2f}MiB）。因此不能声称无交换。各阶段差值和内存初末60秒统计在JSON；本轮时长不能排除慢性内存增长。",'',
f"语音无线程错误，73个endpoint。{audio['overruns']}帧计算超过16ms（{audio_result['overrun_fraction']*100:.2f}%），单帧最大计算{audio['max_compute_s']*1000:.2f}ms；这不等于丢了1495帧。累计最大lag {audio['max_lag_s']*1000:.2f}ms，资源首样本已是该最大值，随后未再升高；队列1表示一个整句WAV，不是单帧积压。不能写成每帧严格实时或零延迟。",'',
'降噪/VAD/KWS/ASR实际工作时间分别为'+ ' / '.join(f"{audio['stage_seconds'][k]:.3f}s" for k in ['denoise','vad','kws','stream_asr'])+'；这些是阶段墙钟累加，不能当成单句端到端延迟或CPU秒。本轮性能请求按fixture主动发起，与语音endpoint没有触发关联，不能给出“用户说完→首token”的链路延迟。','', '## 与旧D64基线的同内容比较','',
'已核实旧candidate-matrix-r3的mask-base与本轮D64：两份system完全相等，6条prompt IDs/hash/prefix、12个case-repeat的32反馈契约相等；12条预测序列完全相同，两个重叠诊断的完整4步logits逐字节相同。','', '|System|旧/本轮 prefill P50 秒|旧/本轮 TTFT P50 秒|旧/本轮 decode P50|','|---:|---:|---:|---:|']
for k,c in comparison['by_static'].items():lines.append(f"|{k}|{c['old_prefill_s']['p50']:.4f}/{c['new_prefill_s']['p50']:.4f}|{c['old_TTFT_s']['p50']:.4f}/{c['new_TTFT_s']['p50']:.4f}|{c['old_decode_tps']['p50']:.3f}/{c['new_decode_tps']['p50']:.3f}|")
lines+=['','这是跨轮参考：旧mask-base和本轮都为C1280，但本轮用了既有greedy head与v2关闭顺序。prefill几乎不变，decode约提高5～7%；不能记作新C512/bulk提速，也没有控制同轮热漂移的A-B-A证据。','', '## 结论边界','',
'此轮确认原C1280全模型在新动态范围上可完整运行，并给出继续优化所需的真实基线。全部8格都未达到prefill/TTFT≤100ms；冻结的20个batch边界记录尚未执行。小样本性能、短自然任务质量和约8分钟资源记录分别报告，不能互相替代。原始events/resources、source/fixture/graph provenance及实际诊断f32保留；机器可读分析见analysis.json。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'integrity_pass':True,'cells':len(cells),'repeat_pairs_exact':report['repeat_exact_count'],'natural_pass':report['natural_task_pass_manual'],'diagnostic_files':len(diagnostic),'driver_closed':closed['metrics']['driver_bytes'],'report':str(R/'REPORT.md'),'swap_phase':resources['swapout_pages_by_observed_phase']},ensure_ascii=False))
