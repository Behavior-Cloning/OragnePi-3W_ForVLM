import json,math,statistics,hashlib
from pathlib import Path
from collections import Counter
p=Path(__file__).resolve().parent;r=p/'results/ttft-32-64-matrix'
rows=[json.loads(x) for x in (r/'events.jsonl').read_text().splitlines()];res=[json.loads(x) for x in (r/'resources.jsonl').read_text().splitlines()]
def st(xs):
 xs=sorted(xs);return {'n':len(xs),'p50':statistics.median(xs),'p95':xs[math.ceil(.95*len(xs))-1],'min':min(xs),'max':max(xs),'mean':statistics.mean(xs)} if xs else {}
def resource(xs):
 cpu=[]
 for a,b in zip(xs,xs[1:]):
  d=[int(y)-int(x) for x,y in zip(a['cpu'][0].split()[1:9],b['cpu'][0].split()[1:9])];t=sum(d)
  if t>0:cpu.append(100*(t-d[3]-d[4])/t)
 return {'system_cpu_percent':st(cpu),'temp_C':st([x['temp_mC']/1000 for x in xs]),'harness_sum_pss_MiB':st([sum(y.get('Pss',0) for y in x['processes'].values())/1024 for x in xs]),'available_MiB':st([x['meminfo']['MemAvailable']/1024 for x in xs]),'swap_in_pages':xs[-1]['vmstat']['pswpin']-xs[0]['vmstat']['pswpin'],'swap_out_pages':xs[-1]['vmstat']['pswpout']-xs[0]['vmstat']['pswpout'],'freq_khz':{c:sorted({x['freq_khz'][c] for x in xs if c in x['freq_khz']}) for c in xs[0]['freq_khz']}}
variants=['old','tail8','group','group-exact','old-repeat'];base={x['case']:x for x in rows if x['event']=='quality' and x['variant']=='old'};perfbase={(x['case'],x['repeat']):x for x in rows if x['event']=='measure' and x['variant']=='old'}
out={'scope':'Intern NPU text; synthetic streaming audio; RAG resident, queried only during quality; no vision, microphone or motors','counts':dict(Counter(x['event'] for x in rows)),'variants':{},'resources':resource(res),'complete':[x for x in rows if x['event']=='complete'],'input_hashes':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ['events.jsonl','resources.jsonl','fixtures.json']}}
for v in variants:
 q=[x for x in rows if x['event']=='quality' and x['variant']==v];me=[x for x in rows if x['event']=='measure' and x['variant']==v];ld=next(x for x in rows if x['event']=='loaded' and x['variant']==v);cl=next(x for x in rows if x['event']=='closed' and x['variant']==v)
 vr={'performance':{},'quality':[{'case':x['case'],'input':x['input'],'reply':x['reply'],'tokens_equal_old':x['metrics']['tokens']==base[x['case']]['metrics']['tokens'],'ttft_s':x['metrics']['ttft_seconds'],'decode_tps':x['metrics']['decode_tokens_per_second'],'top3_names':[a['name'] for a in x['top3']]} for x in q],'quality_token_parity_count':sum(x['metrics']['tokens']==base[x['case']]['metrics']['tokens'] for x in q),'performance_token_parity_count':sum(x['metrics']['tokens']==perfbase[(x['case'],x['repeat'])]['metrics']['tokens'] for x in me),'loaded':ld,'closed':cl,'resources':resource([x for x in res if ld['at']<=x['at']<=cl['at']])}
 for n in [32,48,64]:
  xx=[x for x in me if x['body_tokens']==n];mm=[x['metrics'] for x in xx]
  vr['performance'][str(n)]={'ttft_s':st([x['ttft_seconds'] for x in mm]),'under_1s':sum(x['ttft_seconds']<1 for x in mm),'prompt_tokens':sorted({x['prompt_tokens'] for x in mm}),'new_tokens':sorted({x['new_tokens'] for x in mm}),'reused_tokens':sorted({x['reused_tokens'] for x in mm}),'template_overhead':sorted({x['template_overhead'] for x in xx}),'prefix_prepare_ttft_s':st([x['prefix_prepare_metrics']['ttft_seconds'] for x in xx]),'stages':{k:st([x[k] for x in mm]) for k in ['setup_seconds','prefill_seconds','scalar_prompt_seconds','first_head_seconds','batch_calls','tail_rows'] if k in mm[0]}}
 out['variants'][v]=vr
(r/'analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
lines=['# 32–64 token 首字运行时优化实测','', '2026-10-25。135次计时、30次自然语义输出，控制器exit0。**尾部批处理有效，但当前32/48/64正文均未稳定达到首token<1秒。**','', '## 延迟与缓存边界','', '|配置|32正文 P50/P95|48正文 P50/P95|64正文 P50/P95|32正文<1s|','|---|---:|---:|---:|---:|']
for v in variants:
 a=out['variants'][v]['performance'];lines.append('|'+v+'|'+ '|'.join(f"{a[str(n)]['ttft_s']['p50']:.3f}/{a[str(n)]['ttft_s']['p95']:.3f}s" for n in [32,48,64])+f"|{a['32']['under_1s']}/9|")
lines += ['', '每格9次，3条中文场景各重复3次；正文用末尾标点补齐精确32/48/64，仅作计算成本测量。P50为中位数；P95为nearest-rank，此处等于9次中的最大值，不能代表长期SLO。所有48/64档均0/9低于1秒。', '', 'system正文81 token，system角色模板5 token，固定前缀合计86；动态正文精确32/48/64，user/assistant额外聊天模板8 token，总prompt126/142/158。非exact只复用80，真实新计算为46/62/78；exact复用86，真实新增40/56/72。每次请求前重置并重新计算同一system，排除动态缓存命中；准备system约2.15–2.30秒排除在TTFT外。这是常驻system-cache受控微基准，不能用作事件→首字或用户说完→首字延迟。', '', 'tail8将尾部8–15行由逐token改成一次16行图运行。32正文从1.450降至1.064秒（约26.6%）；64从2.103降至1.672秒（约20.5%）。group收益较小且受请求准备波动影响。exact在本轮长度上没有减少批次数，尚无足够收益支持切换默认。', '', '## 分段瓶颈','', '|group配置正文|setup P50|prefill P50|首输出头 P50|16行批次数|','|---|---:|---:|---:|---:|']
for n in [32,48,64]:
 a=out['variants']['group']['performance'][str(n)]['stages'];lines.append(f"|{n}|{a['setup_seconds']['p50']:.3f}s|{a['prefill_seconds']['p50']:.3f}s|{a['first_head_seconds']['p50']:.3f}s|{a['batch_calls']['p50']:g}|")
z=out['resources'];c=out['complete'][0]
lines += ['', '分段各自中位数不保证相加等于总中位数。32正文需要3批、64需要5批；64的prefill本身约1.53秒，单纯消除请求准备不足以到1秒。下一步先检查KV准备开销；64需要进一步prefill图/算子优化。', '', '## 质量与资源','', 'tail8、group、group-exact、old-repeat各6/6自然输出逐token与old相同，27次性能首token也各27/27一致。样本小，不能证明普遍无损；exact历史曾出现自然输出差异，保留实验属性。', '', '所有配置仍将“弯腰”检索到“抬头”；保持静止Top2/Top3仍包含前移/后移。这是既有RAG兼容性缺陷。此次只有查询/mock结果，不接电机，不把Top3直接当可一起执行动作。', '', f"整机CPU平均{z['system_cpu_percent']['mean']:.2f}%，最高温度{z['temp_C']['max']:.2f}°C；采样进程PSS峰{z['harness_sum_pss_MiB']['max']:.2f}MiB，系统可用内存最低{z['available_MiB']['min']:.2f}MiB。swap-in/out均{z['swap_in_pages']}/{z['swap_out_pages']}页。NPU driver各组753274680字节，5次关闭均归零。PSS仅harness与NPU子进程，不能代表包括其他驻留服务的整机模型内存，NPU driver单列也不简单相加。", '', f"音频前端持续运行{c['audio_frames']}帧，即{c['audio_frames']*.016:.3f}秒，98次ASR端点，最大音频滞后{c['max_audio_lag_s']:.3f}秒，高于此前配对约0.068秒。RAG常驻，只有自然输出阶段调用检索，性能阶段没有并发RAG查询。未做本优化配置的新一小时完整链路测试。", '', 'old在末尾复测的三个长度P50变化约−1.4%、+0.6%、−0.02%，未观察到明显持续热劣化；顺序固定且样本少，不能据此证明无温度/调度影响。CPU亲和：NPU宿主4,5，音频6，Python/RAG7；板卡频率见analysis.json。', '', '## 可复现文件','', '`ttft_matrix.py`、`controller_ttft.py`、`analyze_ttft.py`；原始fixtures/events/resources和各binary SHA均保留在本目录。语音为合成中文录音，Intern为NPU文本decoder，无视觉编码器、真实麦克风或物理执行器。此次结果不会自动将实验性exact作为默认。', '']
(r/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps({'resources':z,'complete':c},ensure_ascii=False))
