import json,collections,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
j=json.loads((ROOT/'summary.json').read_text());metrics=j['metrics'];perf=j['performance']
labels={'bge_small':'BGE-small','gte_base':'GTE 中文 base','bge_base':'BGE-base','bge_large':'BGE-large','qwen06':'Qwen3-Embedding 0.6B'}
models=[m for m in labels if m in j['completed_models']]
def cell(m,pool,k,path='state'):return next(x for x in metrics if x['model']==m and x['pool']==pool and x['k']==k and x['path']==path)
def rate(a,b):return f'{a/b*100:.1f}% ({a}/{b})' if b else '无足够返回项'
def table(rows):return '\n'.join('|'+ '|'.join(map(str,r))+'|' for r in rows)+'\n'
complete=len(models)==5
out=['# 家庭／桌面陪伴动作 RAG 复测报告','',('五个编码器均已完成实板复测。' if complete else '进行中：已完成 '+','.join(labels[m] for m in models)+'。'),'','本轮去掉全部管理接口，并重新冻结家庭／桌面场景。结果来自 Orange Pi 板卡 CPU 实际推理；没有驱动电机或把模型排名当成动作执行成功。','',
'## 范围与版本','',
'- 原子库35项：日常表情、头部、手势、上身小幅动作和保持姿态。组合技能库20项：迎接、陪读、倾诉、安慰、庆祝、晚安姿态、舒展、音乐、感谢、道歉、告别、桌面物体观察等。两库独立检索，原子Top-K中不会混进高层技能。',
'- 移除查文档、查历史、改超时等192个管理接口；声源估计/视觉检测、急停/澄清等内部API不在动作词典。走动、底盘旋转、跳跃、抓握、触碰不在本轮桌面库。历史版本仍保留。',
'- 60条家庭高层输入=30个场景各两种说法。数据由助手编写并复核，非独立真人盲测；同场景两种说法有关联，重复推理也不增加质量样本数。',
'- 每模型、每库：60输入×2次=120次热态计时；每模型两库共240次，5模型合计1200次。候选预编码和3条输入×两库预热另记。','',
'## 相关性判断','',
'遵循用户口径：非最佳、非唯一标准答案或不在有限目标标签内，不直接算不相关。compatible 表示可作为目标/辅助/过渡动作安排；conditional 表示需补目标、方向、风格或顺序；incompatible 仅为明确无法执行、违背禁令或意图冲突。',
'主审与独立审计在查看排名前分别评审全部60×55=3300个指令—候选组合，展开组合技能。两者对304个硬负例完全一致；452个C/U分歧统一保留U。最终1268个C、1728个U、304个N。这些是全库配对数量，不能当作任何模型的错误率。',
'明确负例包括：左臂禁止运动却张开双臂；只能表情却转头；安静时拍手；明确表达同意却调用只定义否定/拒绝的摇头卡。拒绝场景点头可表示理解，保留U，不直接判错。','',
'## 原子动作：前K整组无明确冲突率','',
'以下是前K每项均无N的比例，包含U；不是第K名自身准确率，也不是五个动作已可直接执行。所有原子查询均有足够返回项，分母60。状态过滤使用样例提供的约束，不包含从自然语言提取约束的时间或错误。','']
rows=[['模型','Top-1','Top-2','Top-3','Top-4','Top-5'],['---']*6]
for m in models:rows.append([labels[m]]+[rate(cell(m,'atom',k)['no_hard_conflict'],cell(m,'atom',k)['n']) for k in range(1,6)])
out.append(table(rows))
out+=['## 原子动作：整组C / 含U无N / 含N','', '每格为“全部兼容 / 至少一个待定但无冲突 / 至少一个明确冲突”的查询组数。C仍是语义条件下可以安排，不等于实机成功。','']
rows=[['模型','Top-1','Top-2','Top-3','Top-4','Top-5'],['---']*6]
for m in models:rows.append([labels[m]]+[' / '.join(str(cell(m,'atom',k)[f]) for f in ['all_compatible','conditional_no_incompatible','any_incompatible']) for k in range(1,6)])
out.append(table(rows))
out+=['## 原子动作：每个名次自身无明确冲突率','', '本表仅看第1、2、3、4、5名自身，不能替代上面的整组指标。','']
rows=[['模型','第1名','第2名','第3名','第4名','第5名'],['---']*6]
for m in models:rows.append([labels[m]]+[rate(cell(m,'atom',k)['n']-cell(m,'atom',k)['rank_counts'].get('incompatible',0),cell(m,'atom',k)['n']) for k in range(1,6)])
out.append(table(rows))
out+=['## 原始召回与约束过滤的作用','', '完整排名先由模型打分，之后再检查已知执行器、安静、禁用动作和目标可用状态。双手动作已补齐左右手标签。未额外根据测试目标标签屏蔽“同意时摇头”，所以此类语义冲突会如实留下。','']
rows=[['模型','过滤前Top-3无冲突','过滤后Top-3无冲突','过滤前Top-5无冲突','过滤后Top-5无冲突'],['---']*5]
for m in models:rows.append([labels[m]]+[rate(cell(m,'atom',k,path)['no_hard_conflict'],cell(m,'atom',k,path)['n']) for k,path in [(3,'raw'),(3,'state'),(5,'raw'),(5,'state')]])
out.append(table(rows))
out+=['## 组合技能独立检索','', '每项是一个家庭组合技能，不与原子动作混排。前K仍使用全部项目无明确冲突率；部分严格限制场景只剩2或4项，按实际足K的输入统计，并显式给出分母。没有用一个返回项冒充五项正确。','']
rows=[['模型','Top-1','Top-2','Top-3','Top-4','Top-5'],['---']*6]
for m in models:rows.append([labels[m]]+[rate(cell(m,'skill',k)['no_hard_conflict'],cell(m,'skill',k)['n']) for k in range(1,6)])
out.append(table(rows))
rows=[['模型','技能Top-3 C/U/N','技能Top-5 C/U/N','技能Top-5返回不足5'],['---']*4]
for m in models:rows.append([labels[m]]+[' / '.join(str(cell(m,'skill',k)[f]) for f in ['all_compatible','conditional_no_incompatible','any_incompatible']) for k in [3,5]]+[str(cell(m,'skill',5)['shortfall'])])
out.append(table(rows))
out+=['## 目标线索与组合顺序','',
'“无冲突”不保证选中了实现主要目标所需的动作。另设冻结的目标线索标签，统计至少命中一个；这张表只辅助说明召回是否贴近目标，不用于把标签外动作判N，也不是完成任务率。技能目标线索仅计预先定义了对应技能线索且返回足K的输入；Top-3为52条，另外6条没有对应的技能目标线索、2条返回不足3项。','']
rows=[['模型','原子Top-1目标线索','原子Top-3目标线索','原子Top-5目标线索','技能Top-3目标线索'],['---']*5]
for m in models:rows.append([labels[m]]+[rate(cell(m,pool,k)['goal_signal_hit'],cell(m,pool,k)['goal_signal_n']) for pool,k in [('atom',1),('atom',3),('atom',5),('skill',3)]])
out.append(table(rows))
out+=['执行假设为串行，不能将共享头、手、屏幕等执行器的候选同时发出。检索排名并非执行顺序；例如“先挥手后倾听”“先观察后休息”仍需编排。组合技能可能共享点头、微笑等步骤。下面统计存在重复步骤、阶段顺序或最终眼部状态疑点的组数；这些都是待检查项，不强行当作硬负例，也不等价于完整计划验证。','']
rows=[['模型','原子Top-3需顺序检查','原子Top-5需顺序检查','技能Top-3需顺序/重复检查','技能Top-5需顺序/重复检查'],['---']*5]
for m in models:rows.append([labels[m]]+[str(cell(m,pool,k)['needs_sequence_or_duplicate_review'])+'/'+str(cell(m,pool,k)['n']) for pool,k in [('atom',3),('atom',5),('skill',3),('skill',5)]])
out.append(table(rows))
out+=['## 板卡性能与资源','',
'A76 CPU6/7、2线程、FP32、PyTorch 2.7.0+cpu。每请求重新分词和编码，不缓存query；当前实现统一对55条预编码向量计算点积，再按原子/技能类型取各自排名，检查提供状态并构造Top-5。包含这些步骤，排除模型初始化、候选建库、网络、感知、ASR、LLM和物理执行。状态字段直接提供，未测试其自动提取。未搜索提示或利用测试标签调参；这组结果不代表各模型的能力上限。','']
rows=[['模型','原子P50/P95 ms','技能P50/P95 ms','执行时CPU核数','热态/全程PSS峰值 MiB','起始/最高温度 °C'],['---']*6]
for m in models:
 a=next(x for x in perf if x['model']==m and x['pool']=='atom');s=next(x for x in perf if x['model']==m and x['pool']=='skill')
 rows.append([labels[m],f"{a['total_ms']['p50']:.2f} / {a['total_ms']['p95']:.2f}",f"{s['total_ms']['p50']:.2f} / {s['total_ms']['p95']:.2f}",f"{a['average_cpu_cores']:.2f}",f"{a['pss']/1024**2:.1f} / {a['pss_full']/1024**2:.1f}",f"{a['temperature_start']:.2f} / {a['temperature_max']:.2f}"])
out.append(table(rows))
rows=[['模型','全程最低系统可用内存 MiB','计时期间换入/换出 MiB'],['---']*3]
for m in models:
 a=next(x for x in perf if x['model']==m and x['pool']=='atom')
 rows.append([labels[m],f"{a['available_min']/1024**2:.1f}",f"{a['swap_in_delta']/1024**2:.2f} / {a['swap_out_delta']/1024**2:.2f}"])
out.append(table(rows))
out+=['CPU核数约2相当于Linux进程约200%，约占8核整机25%，不是整机满载200%。PSS包含模型、Python/运行库、索引与结果缓冲；全程PSS包括加载和建库，热态PSS不能作为共存内存上界。温度峰值取计时阶段1秒遥测与请求间温控即时读数的较大值；温度是顺序实验的观测峰值，非相同初温/长稳态对比。后台现有服务保留，本轮未主动施加语音或VLM并发负载。','']
rows=[['模型','进程加载 s','55条预编码 s','全程温控等待 s','两次排名稳定查询数'],['---']*5]
for m in models:
 a=next(x for x in perf if x['model']==m and x['pool']=='atom');s=next(x for x in perf if x['model']==m and x['pool']=='skill')
 rows.append([labels[m],f"{a['cold_load_s']:.2f}",f"{a['index_s']:.2f}",f"{a['cooldown_s']:.2f}",str(a['stable_rankings']+s['stable_rankings'])+'/120'])
out.append(table(rows))
out+=['温控等待在单次查询计时之外，不能忽略成持续满载吞吐。加载计时前已读取权重做哈希，且未清空文件页缓存，因此不是冷盘启动耗时；预编码时间可能包含准备期冷却。板上每个大模型单独进程，暂存权重在结果回传且SHA核验后删除；F盘源权重保持。','',
'## 限制与交付','',
'本轮验证的是家庭动作召回与语义兼容性，没有驱动真实机械动作，也没有验证全链路语音/VLM、长时间负载、情绪识别、传感器定位或真实家庭用户效果。高层“倾听”“晚安”卡只含姿态/表情步骤，不代表开启VAD或已经发出语音。','',
'这次词典和查询均已更换，不能把新比例与旧256项管理接口混合库直接比较成模型准确率提升。所有新结果和冻结数据在独立目录保存。','',
'Windows：`F:/Files/OPi-Lab/repos/OragnePi-3W_ForVLM/research/home-companion-rag-20261022`。原始记录在 `results/<model>/raw.jsonl`，逐查询评审在 `ranking-evaluation.jsonl`，汇总在 `summary.json`，全部语义判断在 `adjudicated-review.json`，独立审计原件为 `independent-review.json`。']
context=json.loads((ROOT/'report-context.json').read_text()) if (ROOT/'report-context.json').exists() else {}
report='\n'.join(out)+'\n'
report=report.replace('## 范围与版本',context.get('recommendations','')+'## 范围与版本')+'\n'+context.get('audit','')
(ROOT/'HOME_COMPANION_RAG_REPORT.md').write_text(report)
print('report written',complete)
