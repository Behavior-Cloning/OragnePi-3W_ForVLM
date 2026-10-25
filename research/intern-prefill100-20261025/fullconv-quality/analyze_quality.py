#!/usr/bin/env python3
"""Offline long semantic quality analysis, with explicit human-reviewed judgments.
No board or network calls. Does not modify the runner/config or raw evidence.
"""
from pathlib import Path
import argparse,json,hashlib,importlib.util,collections,math,copy
import numpy as np
from tokenizers import Tokenizer
from contract import validate_quality,validate_fixture,sha,BINARY_SHA
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('aba_analysis_helpers',P.parent/'fullconv/analyze_fullconv.py');H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
NAMES=['c1280-original','c1280-conv-l2'];J=lambda p:json.loads(Path(p).read_text())

# Human semantic decisions for the exact observed response under each frozen
# scenario. This is an auditable review table, not a keyword-based evaluator.
# Candidate identities were visible to reviewer: do not claim blinded review.
REVIEW={
 'd64-return-home-nod':{'点头':('pass','complete','compatible','点头是一次回应的常用短语；rubric允许未重复明示一次。未发出其他部件动作。')},
 'd64-greeting-right-arm':{'抬起右臂':('compatible_partial','partial','compatible','主体和左右正确，但未保留低幅度要求，不能补成已完成幅度约束。')},
 'd64-edge-hold':{'保持全身静止':('pass','complete','compatible','明确保持全身静止，符合桌边无安全移动方向。')},
 'd128-waist-bow':{'鞠躬一次':('compatible_partial','partial','compatible','表达鞠躬一次；常见鞠躬可指腰部动作，但省略只用腰/小幅轻缓，按严格约束保守记部分完成。没有明确下令头臂动作。')},
 'd128-head-left':{'头部向左转一点':('pass','complete','compatible','头部、左方向和小角度均明确。'),
                  '左转':('underspecified','unknown','unknown','未指头部还是底盘，不能替模型补出主体。')},
 'd128-base-back':{'后退两厘米':('compatible_partial','partial','compatible','后退与距离正确；未保留缓慢要求，按rubric记部分完成。')},
 'd192-lower-arms':{'放下手臂':('compatible_partial','partial','compatible','放下方向及手臂部件正确，但未明确双臂、缓慢和自然垂下终点。未下令抬高或挥臂。')},
 'd192-base-turn-right':{'保持原地向右小角度转弯':('underspecified','unknown','unknown','原地/右/小角度正确，但未明确底盘主体；按冻结rubric未明确底盘的情形保守列unknown，供人工复审。'),
                        '保持静止':('safe_underresponse','not_done','compatible','静止不违反禁止动作，但没有完成有效的右转要求。')},
 'd192-child-near-arm-hold':{'保持全身静止':('pass','complete','compatible','明确静止，遵守已取消挥臂及全身不动。'),
                           '保持静止':('pass','complete','compatible','rubric明确接受保持静止作为全身不动的同义短语。')},
 'd256-head-neutral':{'缓慢回正头部':('pass','complete','compatible','头部、缓慢和回正中立终点均明确；不因从低头回正需要上移而误判抬过中立。')},
 'd256-left-arm-correction':{'保持静止':('safe_underresponse','not_done','compatible','静止安全，但没有执行纠正后的左臂低幅度招呼。'),
                            '低幅度抬起':('underspecified','unknown','unknown','只有幅度和抬起动作，缺左臂主体和左右，无法确认避开右臂障碍。')},
 'd256-base-left':{'缓慢移动底盘两厘米':('underspecified','unknown','unknown','速度、底盘和距离正确，但缺左方向；前后右均被禁止，不能自行补方向。')}
}

# Per-ranked-action decisions use the original long user instruction, not the
# potentially incomplete generated phrase. Compatibility is not exact-action
# hit rate, and partial/unknown are not silently treated as conflicts.
RAG={
 'd64-return-home-nod':{'R008':('compatible','目标点头一次。'),'R011':('conflicting','绕圈包含左右转动；用户只要点头并明确不要摇头。'),'R012':('conflicting','绕圈包含左右转动；用户只要点头并明确不要摇头。')},
 'd64-greeting-right-arm':{'R026':('conflicting','左臂被明确禁止，纠正后只右臂。'),'R046':('unknown','右臂正确，但上举姿态未给出低幅度参数，不能保证不超范围。'),'R053':('compatible_partial','仅右前臂竖起不涉及左臂，方向兼容；与低幅度整右臂招呼目标不完全相同。')},
 'd64-edge-hold':{'R108':('compatible_partial','底盘零运动兼容，但只覆盖底盘不覆盖全身静止。'),'R109':('unknown','中立姿态可能主动回正头/腰/臂；未给出当前全身关节与目标是否相等，无法保证静止。'),'R018':('conflicting','主动转腰违反全身不动。')},
 'd128-waist-bow':{'R110':('conflicting','复合鞠躬姿态包含头部下俯，而要求只用腰。'),'R035':('conflicting','左挥臂被禁止。'),'R055':('conflicting','右挥臂被禁止。')},
 'd128-head-left':{'R005':('compatible_partial','向左侧倾不等于向左转头，但不是明确禁止的向右转或底盘动作；不能算完成。'),'R006':('compatible_partial','向右侧倾轴不同于禁止的向右转头，属替代未完成，不能因方向字面就当yaw冲突。'),'R001':('compatible','头部左转；需下游保留小角度约束。'),'R101':('conflicting','底盘右移并左转，底盘必须不动。'),'R099':('conflicting','底盘左移并左转，底盘必须不动。'),'R102':('conflicting','底盘右移并右转，底盘必须不动。')},
 'd128-base-back':{'R097':('conflicting','后退同时左转，违反不转弯。'),'R098':('conflicting','后退同时右转，违反不转弯。'),'R009':('conflicting','摇头违反其余部件保持不动。')},
 'd192-lower-arms':{'R025':('compatible_partial','左臂放下可作为双臂目标的一部分。'),'R045':('compatible_partial','右臂放下可作为双臂目标的一部分。'),'R026':('conflicting','重新举起左臂违反禁止抬臂。')},
 'd192-base-turn-right':{'R093':('conflicting','原地左转与纠正后的右转相反。'),'R094':('compatible','原地右转，需下游保留小角度。'),'R107':('conflicting','左右往返包含被禁止左转。'),'R108':('compatible_partial','底盘不动兼容禁令但未完成右转。'),'R089':('conflicting','向前平移被禁止。'),'R090':('conflicting','向后平移被禁止。')},
 'd192-child-near-arm-hold':{'R108':('compatible_partial','底盘零运动兼容，但未覆盖全部关节。'),'R109':('unknown','回中立姿态可能主动移动关节，无法保证全身静止。'),'R018':('conflicting','腰部转动被禁止。'),'R089':('conflicting','前进被禁止且有桌沿。'),'R090':('conflicting','后退被禁止且有电线。')},
 'd256-head-neutral':{'R007':('compatible','头部中立回正，需下游保留缓慢。'),'R010':('compatible_partial','侧倾轻晃与回正任务不同；未命中明确禁止的yaw，不能据此计硬冲突，但不完成终点。'),'R008':('conflicting','再点头致意违反结束低头、不是再发起一轮致意的要求。')},
 'd256-left-arm-correction':{'R108':('compatible_partial','底盘不动兼容禁止底盘运动，但未完成左臂招呼。'),'R089':('conflicting','前进违反底盘保持原位。'),'R090':('conflicting','后退违反底盘保持原位。'),'R004':('conflicting','低头违反头部不动。'),'R082':('conflicting','双臂摆动包含禁止的右臂动作。'),'R125':('conflicting','抬头并双臂展开违反头部/右臂不动。')},
 'd256-base-left':{'R091':('compatible','底盘左平移，需保留缓慢两厘米参数。'),'R092':('conflicting','右移违反方向且右侧贴收纳箱。'),'R089':('conflicting','前进被明确禁止。')}
}


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=P/'results/fullconv-long-quality-r1');a=ap.parse_args();r=a.results
 events=H.rows(r/'events.jsonl');samples=H.rows(r/'resources.jsonl');cfg=J(r/'config.json');meta=J(r/'meta.json');prov=J(r/'provenance.json')
 validate_quality(cfg);assert meta['status']=='complete' and not meta['background_errors']
 counts=collections.Counter(x['event'] for x in events)
 assert all(counts[k]==v for k,v in {'quality':48,'quality_cost':2,'diagnostic':2,'closed':2,'loaded':2,'complete':1}.items()) and not counts['measure']
 doc=J(P/'semantic-inputs.json');old=J(P/'cost-inputs.json');scenes=J(P/'scenes.source.json')
 assert J(r/'fixtures.json')==doc and J(r/'scenes.json')==scenes
 assert sha(doc['tokenizer_path'])==doc['tokenizer_sha256']==prov['tokenizer_sha256']
 tok=Tokenizer.from_file(doc['tokenizer_path']);enc=lambda s:tok.encode(s,add_special_tokens=False).ids
 cost=validate_fixture(doc,enc,old,scenes);assert cost==J(r/'cost-fixture.json')
 for f in ['board_quality.py','resident_worker_v1.py','context_rag.py','contract.py','base_contract.py']:assert prov[f]==sha(P/f)
 assert prov['fixture_sha256']==sha(P/'semantic-inputs.json')==cfg['expected_fixture_sha256']
 assert prov['cost_fixture_sha256']==sha(P/'cost-inputs.json')==cfg['expected_cost_fixture_sha256']
 assert prov['scenes_sha256']==sha(P/'scenes.source.json')==cfg['expected_scenes_sha256']
 assert cfg['controller_source_sha256']==sha(P/'controller_quality.py')
 delivered=J(P/'delivery.json');assert sha(P/'GOLD_RUBRIC.md')==delivered['files']['GOLD_RUBRIC.md']['sha256']
 graph={};loaded={};closed={}
 for v in NAMES:
  ge=[x for x in events if x['event']=='graph_verified' and x['variant']==v];assert len(ge)==1 and len(ge[0]['files'])==29
  assert all(x['sha256']==x['expected_sha256'] for x in ge[0]['files']);graph[v]=ge[0]
  le=next(x for x in events if x['event']=='loaded' and x['variant']==v);loaded[v]=le
  assert le['binary_sha256']==BINARY_SHA and le['ready']['capacity']==1280 and le['ready']['layers']==24
  assert le['ready']['single_pass_greedy_enabled'] and not le['ready']['true_tail8_enabled']
  ce=[x for x in events if x['event']=='closed' and x['variant']==v];assert len(ce)==1 and ce[0]['metrics']['driver_bytes']==0;closed[v]=ce[0]
 ga={Path(x['declared_path']).name:x for x in graph[NAMES[0]]['files']};gb={Path(x['declared_path']).name:x for x in graph[NAMES[1]]['files']}
 assert set(ga)==set(gb) and [k for k in sorted(ga) if ga[k]['sha256']!=gb[k]['sha256']]==['prefill-layer2.nb']
 ov=J(r/'overlay-provenance.json');assert len(ov['files'])==29
 assert {x['name']:x['link_target'] for x in ov['files']}==cfg['overlay_contract']['symlinks']
 fixtures={x['id']:x for x in doc['quality']};fixtures[cost['id']]=cost
 selected=[x for x in events if x['event'] in ['quality','quality_cost','diagnostic']]
 for e in selected:
  c=fixtures[e['case']];m=e['metrics'];st=c['static_tokens'];new=c['body_tokens']+8;total=st+new
  assert m['prompt_tokens']==total and m['new_tokens']==new and m['reused_tokens']==st
  assert m['scalar_prompt_tokens']==0 and m['batch_calls']==math.ceil(new/16) and m['tail_rows']==8
  assert not m['true_tail8_calls'] and len(m['tokens'])==m['generated_tokens'] and m['kv_tokens']==total+m['generated_tokens']-1
  assert e['prompt_sha256']==c['tokens_sha256']
  if e['event']!='diagnostic':
   prep=e['prefix_prepare_metrics'];assert prep['prompt_tokens']==prep['new_tokens']==prep['kv_tokens']==st and prep['reused_tokens']==0 and prep['generated_tokens']==1
   assert not m['fixed_feedback'] and not m['diagnostic'] and m['generated_tokens']<=32
   assert tok.decode(m['tokens'],skip_special_tokens=True)==e['reply']
   assert m['finish_reason']=='eos' and m['tokens'][-1]==151645
  else:assert m['fixed_feedback'] and m['diagnostic'] and m['generated_tokens']==4 and not e['timing_eligible']
 q={v:{x['case']:x for x in selected if x['event']=='quality' and x['variant']==v} for v in NAMES}
 reviews=[];rag_reviews=[];pairs=[]
 for v in NAMES:
  assert len(q[v])==24
  for cid,e in q[v].items():
   cls,completion,compat,why=REVIEW[e['scenario']][e['reply'].rstrip('。')]
   reviews.append({'variant':v,'case':cid,'scenario':e['scenario'],'input':e['input'],'reply':e['reply'],'category':cls,'completion':completion,'compatibility':compat,'reason':why,'prompt_sha256':e['prompt_sha256'],'reply_sha256':hashlib.sha256(e['reply'].encode()).hexdigest(),'gold':e['gold'],'hardware_capability':'declared-capability-compatible','format':'single-action-phrase','physical_execution':False})
   for rank,act in enumerate(e['top3'],1):
    cl,reason=RAG[e['scenario']][act['id']]
    rag_reviews.append({'variant':v,'case':cid,'rank':rank,'id':act['id'],'name':act['name'],'text':act['text'],'score':act['score'],'classification':cl,'reason':reason})
 for cid,x in q[NAMES[0]].items():
  y=q[NAMES[1]][cid];pairs.append({'case':cid,'a':x['reply'],'b':y['reply'],'tokens_exact':x['metrics']['tokens']==y['metrics']['tokens'],'differences':H.differences(x['metrics']['tokens'],y['metrics']['tokens'])})
 diags={};raw={}
 for e in selected:
  if e['event']!='diagnostic':continue
  f=r/e['logits_file'];assert f.stat().st_size==e['bytes']==4*151674*4 and sha(f)==e['sha256']
  x=np.fromfile(f,dtype='<f4').reshape(4,151674);assert np.isfinite(x).all() and x.argmax(axis=1).tolist()==e['metrics']['tokens']
  diags[e['variant']]={k:e[k] for k in ['case','logits_file','sha256','bytes']};raw[e['variant']]=x
 cmp=H.logit_compare(raw[NAMES[0]],raw[NAMES[1]])
 for step in cmp['steps']:
  for ab in ['a','b']:
   step['top5_decoded_'+ab]=[{'id':i,'text':tok.decode([i],skip_special_tokens=False)} for i in step['top5_'+ab]]
   step['winner_decoded_'+ab]=tok.decode([step['argmax_'+ab]],skip_special_tokens=False)
 ids=[117057,114209];cmp['flipped_pair_first_logits']={v:{str(i):float(raw[v][0,i]) for i in ids} for v in NAMES}
 costs={v:next(x for x in selected if x['event']=='quality_cost' and x['variant']==v) for v in NAMES}
 assert all(costs[v]['metrics']['tokens'][0]==int(raw[v][0].argmax()) for v in NAMES)
 ivs=H.intervals(samples);resources={'global':H.resource_bucket(samples,ivs),'by_variant':{},'by_variant_phase':{}}
 for v in NAMES:
  resources['by_variant'][v]=H.resource_bucket([x for x in samples if x['variant']==v],[x for x in ivs if x['same_bucket'] and x['variant']==v])
  resources['by_variant_phase'][v]={phase:H.resource_bucket([x for x in samples if x['variant']==v and x['phase']==phase],[x for x in ivs if x['same_bucket'] and x['variant']==v and x['phase']==phase]) for phase in sorted({x['phase'] for x in samples if x['variant']==v})}
 semantic_counts={v:dict(collections.Counter(x['category'] for x in reviews if x['variant']==v)) for v in NAMES}
 rag_summary={v:{'by_rank':{str(k):dict(collections.Counter(x['classification'] for x in rag_reviews if x['variant']==v and x['rank']==k)) for k in [1,2,3]},'lists_with_explicit_conflict':len({x['case'] for x in rag_reviews if x['variant']==v and x['classification']=='conflicting'})} for v in NAMES}
 natural_timings={v:{str((s,d)):{'n':3,'ttft_s':H.stats(e['metrics']['ttft_seconds'] for e in q[v].values() if e['static_tokens']==s and e['body_tokens']==d),'dynamic_prefill_s':H.stats(H.dynamic(e['metrics']) for e in q[v].values() if e['static_tokens']==s and e['body_tokens']==d),'prefix_prepare_s':H.stats(e['prefix_prepare_metrics']['total_seconds'] for e in q[v].values() if e['static_tokens']==s and e['body_tokens']==d)} for s in [64,128] for d in [64,128,192,256]} for v in NAMES}
 result={'status':'complete_offline_analysis','analyzer_sha256':sha(__file__),'counts':dict(counts),'input_contract':{'system_total':[64,128],'system_body':[59,123],'system_template':5,'dynamic_body':[64,128,192,256],'dynamic_template':8,'actual_new':[72,136,200,264],'natural_max_new':32,'independent_scenarios':12,'presentations_per_variant':24},'source_provenance':prov,'verification':'All source/fixture/graph/cache/token accounting assertions passed; one changed prefill graph; two normal driver0 closes.','natural_pair_exact':sum(x['tokens_exact'] for x in pairs),'pairs':pairs,'semantic_counts':semantic_counts,'rag_summary':rag_summary,'cost_natural':costs,'diagnostic_files':diags,'logit_comparison':cmp,'resources':resources,'audio_final':meta['audio'],'loaded_driver_bytes':{v:loaded[v]['ready']['driver_bytes'] for v in NAMES},'closed_driver_bytes':{v:closed[v]['metrics']['driver_bytes'] for v in NAMES},'natural_timings_not_repeated_performance':natural_timings,'review_method':'Exact observed replies manually adjudicated against original input and frozen rubric; variant labels visible. No safety execution inference. One strictness disagreement with independent label-masked review is retained.'}
 (r/'analysis-quality.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 (r/'semantic-review.json').write_text(json.dumps({'method':result['review_method'],'rubric_sha256':sha(P/'GOLD_RUBRIC.md'),'responses':reviews,'ranked_actions':rag_reviews},ensure_ascii=False,indent=2)+'\n')
 write_report(r,result,reviews,rag_reviews)
 print(json.dumps({k:result[k] for k in ['status','natural_pair_exact','semantic_counts','rag_summary','closed_driver_bytes']},ensure_ascii=False))

def write_report(r,d,reviews,rag_reviews):
 A,B=NAMES;lc=d['logit_comparison'];s=lc['steps'][0];g=d['resources']['global'];audio=d['audio_final'];lines=[]
 lines += ['# C1280 FP16 Conv L2：长语义质量诊断', '', '结论：该候选仍属于实验优化，未通过完整动作决策质量验收。48次自然请求正常完成，两个版本均只有9/24严格完成最终要求；没有观察到明确冲突的自然短语，但遗漏与主体/方向不明不能作为已验证可执行。旧C256最快结果不由这轮C1280实验替代。', '', '## 实际输入和证据边界', '', '固定system总token为64/128，其中正文59/123、模板5；动态正文64/128/192/256，另有user/assistant模板8，实际新增prefill72/136/200/264。每次仅恢复system KV，未缓存动态内容。自然输出上限32，48条均EOS结束，未截断。', '', '12个新编家庭情景×2种system，共24条/版本；同一场景配对使用完全相同输入。各正文长度对应不同情景，且system128含原有重复指令，不能据此证明长度单一因素的质量影响。另有1条旧成本输入/版本，不计入24条语义集。', '', 'A=原C1280，B=仅prefill-layer2替换FP16 Conv1×1。其余28个依赖哈希、量化与runtime flags保持一致；system准备同样经过候选图，因此缓存策略相同不等于缓存数值相同。源码/输入/29依赖哈希、52请求token accounting、两个close driver0均核对通过。', '', '## 人工语义复核', '', '按原指令与冻结rubric评估完成、兼容性及明确冲突，不把“保持静止”一律判正确。以下是可见候选标签的复核，不是盲测。逐条原因及全部RAG排名见semantic-review.json。', '', '|类别|原C1280|Conv L2|', '|---|---:|---:|']
 for cls in ['pass','compatible_partial','safe_underresponse','underspecified','conflicting_or_infeasible']:
  lines.append(f"|{cls}|{d['semantic_counts'][A].get(cls,0)}|{d['semantic_counts'][B].get(cls,0)}|")
 lines += ['', '独立标签遮蔽审计将“保持原地向右小角度转弯”记compatible_partial，本复核按缺少底盘主体记underspecified；这是1条严格度分歧，双方完整通过数均9/24、明确冲突均0。未将分歧隐藏或改成通过。', '', '22/24自然回复token完全相同。两条变化：']
 for x in d['pairs']:
  if not x['tokens_exact']:lines.append(f"- {x['case']}：A「{x['a']}」→B「{x['b']}」。")
 lines += ['', '鞠躬一例仅标点变化；左臂纠正一例从安全未执行变为欠明确动作。B的“低幅度抬起”仍缺左臂主体，不能据此证明已正确避开右侧障碍。', '', '|案例|A回复|B回复|A类别|B类别|', '|---|---|---|---|---|']
 rr={(x['variant'],x['case']):x for x in reviews}
 for x in d['pairs']:lines.append(f"|{x['case']}|{x['a']}|{x['b']}|{rr[A,x['case']]['category']}|{rr[B,x['case']]['category']}|")
 lines += ['', '## 首token翻转与全词表诊断', '', '旧成本输入s64-d64-2：system64（59+5），正文64+模板8，prompt136，实际新增72。A自然输出「'+d['cost_natural'][A]['reply']+'」，B「'+d['cost_natural'][B]['reply']+'」。两者均是兼容招呼意图的表达变化；仍需幅度/碰撞约束执行层，不能自动执行Top3。该例不计入长语义完成率。', '', f"同一固定反馈的4步×151674全词表F32已核验大小、SHA、finite与argmax。首步A={s['argmax_a']}「{s['winner_decoded_a']}」，B={s['argmax_b']}「{s['winner_decoded_b']}」；A top1-top2 margin={s['top1_margin_a']:.8f}，B margin={s['top1_margin_b']:.8f}，原A赢家在B中的signed margin={s['baseline_winner_margin_in_b']:.8f}。", '', f"4步argmax一致{lc['argmax_equal_count']}/4，整体logits相对L2={lc['relative_l2']:.8f}，maxabs={lc['max_abs']:.8f}。这只是固定反馈诊断，第二步以后不是两条自由生成轨迹的直接配对。首步两token具体logits：", '', '```json',json.dumps(lc['flipped_pair_first_logits'],ensure_ascii=False,indent=2),'```', '', '## RAG与执行兼容性', '', 'RAG实际常驻并执行50次查询；它检索模型生成的动作短语，未直接使用完整原指令的禁止/纠正条件。下表人工判断仍以原始指令为准，cosine相似度不是动作兼容概率。没有物理电机执行。', '', '|版本|Top3中至少一个明确冲突的列表|Top1类别|', '|---|---:|---|']
 for v in NAMES:lines.append(f"|{v}|{d['rag_summary'][v]['lists_with_explicit_conflict']}/24|{json.dumps(d['rag_summary'][v]['by_rank']['1'],ensure_ascii=False)}|")
 lines += ['', '常见冲突包含：只点头却召回绕头；底盘不得移动却召回平移；已取消挥臂却召回转腰。部分动作虽可执行但没有完成目标，单列compatible_partial；参数/当前姿态不足列unknown，不直接算冲突。两个版本仍不能把Top3无条件全执行。', '', '## 系统资源和时序', '', '纯文本Intern NPU decoder＋录音WAV流式语音前端＋常驻RAG；没有图像encoder、真人麦克风或物理动作。语音输入独立于本文冻结文字请求，不能当作端到端“用户说完到回复”的测试。自然请求只有每格3次不同文本，报告的TTFT用于质量运行观测，不替代固定32输出性能AB；全logits诊断不纳入纯速度。system prefix准备单列并位于动态TTFT计时前。', '', '|版本|整机平均CPU（8核）|最高温度|合计峰值PSS MiB|最低可用内存 MiB|ready driver字节|', '|---|---:|---:|---:|---:|---:|']
 for v in NAMES:
  b=d['resources']['by_variant'][v]
  lines.append(f"|{v}|{b['cpu_weighted_mean_percent_whole_8cores']:.2f}%|{b['temperature_C']['max']:.3f}°C|{b['process_PSS_sum_MiB']['max']:.2f}|{b['MemAvailable_MiB']['min']:.2f}|{d['loaded_driver_bytes'][v]}|")
 lines += ['', f"全程语音{audio['frames']}帧，overrun={audio['overruns']}（{100*audio['overruns']/audio['frames']:.2f}%），最大lag={audio['max_lag_s']:.6f}s、单帧最大计算={audio['max_compute_s']:.6f}s；队列最多保留一整段WAV。不能声称语音全程实时无积压。", '', f"全程最高温度{g['temperature_C']['max']:.3f}°C；swap-in增加{g['swapin_pages_delta_included_intervals']}页、swap-out增加{g['swapout_pages_delta_included_intervals']}页。详细分阶段CPU/频率/温度/RSS/PSS/音频累计计数见analysis-quality.json。PSS合计只含显式Python和resident PID，driver另列，二者不能直接相加当独立内存。两个resident正常退出driver归零。", '', '## 验收结论', '', '有限样本未发现自然文本明确冲突，但完成率37.5%，关键主体/方向遗漏仍存在；候选还产生了非逐位logits变化和一条语义轨迹变化，因此不批准为质量无损默认版本。保留原基线及候选，后续应先完善约束保持与RAG筛选，而不是用输出一致性代替质量。', '', f"分析器SHA256：`{d['analyzer_sha256']}`。原始JSONL/两个F32全词表文件保留，全部分析为离线读取，没有新板端运行。"]
 (r/'QUALITY_REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
