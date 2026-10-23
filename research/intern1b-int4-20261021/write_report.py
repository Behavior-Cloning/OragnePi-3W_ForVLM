from pathlib import Path
import json
r=Path(__file__).resolve().parent;s=json.loads((r/'summary.json').read_text());t=json.loads((r/'ttft-tail/summary.json').read_text());c=json.loads((r/'cache-study/summary.json').read_text())
lines=(r/'REPORT-notes.md').read_text().splitlines()
lines+=['## 完整解码器与首token结果','', '| 实际动态tokens | W8 tok/s | W4 tok/s | 吞吐提升 | W8 TTFT s | W4 TTFT s |','|---:|---:|---:|---:|---:|---:|']
for case in ['dynamic-64','dynamic-128','dynamic-256']:
 a=s['performance']['w8'][case];b=s['performance']['w4'][case]
 lines.append(f"| {a['dynamic_tokens']} | {a['decode_tokens_per_second']['p50']:.3f} | {b['decode_tokens_per_second']['p50']:.3f} | {s['speedups'][case]['gain_percent']:.2f}% | {a['ttft_seconds']['p50']:.3f} | {b['ttft_seconds']['p50']:.3f} |")
lines+=['','W4每token约44ms，W8约64ms；吞吐提升约45.7%对应每token耗时降低约31.4%。达到50%需要约23.4tok/s，目前未达标。完整模型还含激活/KV、输出头与宿主开销，不能照搬两层比例。','',
'## 首token优化：尾部合批','',
'在隔离resident副本中，至少8个且不足16个尾部tokens补齐成一批；因果遮罩屏蔽未来补齐行，只写回真实KV，取最后真实行hidden。权重/前缀复用策略/容量不变，原服务和二进制没有替换。每版本每长度每策略5次，仅生成1token；诊断不进入计时。','',
'| 版本 | 动态tokens | 原TTFT s | 合批TTFT s | 降低 |','|---|---:|---:|---:|---:|']
for variant in ['w8','w4']:
 for case in ['dynamic-64','dynamic-128','dynamic-256']:
  a=t['performance'][variant]['0'][case];b=t['performance'][variant]['8'][case]
  lines.append(f"| {variant.upper()} | {s['performance'][variant][case]['dynamic_tokens']} | {a['ttft_seconds']['p50']:.3f} | {b['ttft_seconds']['p50']:.3f} | {t['speedups'][variant][case]['reduction_percent']:.2f}% |")
lines+=['','268动态tokens只剩1个标量尾部，未触发合批，约0.0%～0.2%的差异视为波动。','',
'分阶段中位数（ms）：','', '| 权重/尾部策略/动态tokens | KV处理 | 批量prefill | 标量尾部 | 首输出头 |','|---|---:|---:|---:|---:|']
for variant in ['w8','w4']:
 for mode in ['0','8']:
  for case,a in t['performance'][variant][mode].items():
   nums=' | '.join(f"{a[k]['p50']*1000:.1f}" for k in ['setup_seconds','prefill_seconds','scalar_prompt_seconds','first_head_seconds'])
   lines.append(f"| {variant}/{mode}/{s['performance'][variant][case]['dynamic_tokens']} | {nums} |")
lines+=['','策略0=原逐token尾部，8=至少8个尾部tokens时合批。各阶段分别取中位数，和不一定等于TTFT中位数。W4的整批prefill仅快约4%～5%；长输入主要受prefill影响，不能用decode的46%收益来估算VLM首字。','',
'W8的5条自然回答在优化前后逐token一致。动态73/133场景的16步logits相对L2约4.30%/4.84%，平均KL约0.00030/0.00038，argmax均16/16；动态268未触发优化，logits逐位一致。新增计时代码关闭优化时，与首轮原二进制前16步logits逐位一致。','',
'W4的尾部策略变化仍会导致巨大logits差异；它的量化路径本来已不可用。尾部优化不能修复W4质量。本次开发样例也不能代替产品独立质量验收。','',
f"追加实验最高温{t['telemetry']['temperature_max']:.3f}°C，整段swap换入{t['telemetry']['swap_in_delta']}bytes、换出{t['telemetry']['swap_out_delta']}bytes；八次退出driver均归零。",'']
lines+=['## system KV复用配对：本轮重点','',
'采用同一W8权重、尾部合批候选、同一常驻进程。system分别64/128 tokens，动态文本73/133/268 tokens。每个场景3次配对，交替顺序，比较KV命中与在计时内reset重算前缀。两组均不冷加载模型。每次前缀准备单独记录并排除TTFT，准备仅用于控制测试缓存状态；部署首次准备的成本仍存在。','',
'| System tokens | 动态文本tokens | 重算system TTFT s | 复用system TTFT s | 节省 s | 降低 |','|---:|---:|---:|---:|---:|---:|']
for name,x in c['cases'].items():
 lines.append(f"| {x['system_tokens']} | {x['dynamic_tokens']} | {x['recompute']['ttft_seconds']['p50']:.3f} | {x['cached']['ttft_seconds']['p50']:.3f} | {x['saved_seconds']:.3f} | {x['reduction_percent']:.1f}% |")
lines+=['', '缓存命中与重算前缀的首token全词表logits逐位一致：'+str(all(x['first_token_logits']['bit_identical'] for x in c['cases'].values()))+'。每次复用长度都核对为64或128，未把重复用户输入额外命中的缓存算成system收益。','',
f"缓存配对请求边界采样最高温 {c['resources']['temperature_max']:.3f}°C、可用内存最低 {c['resources']['available_min']/1024**3:.2f}GiB，swap换入/换出均为0，退出driver归零。边界采样不能保证捕获瞬时峰值。",'',
'**这仍是文本decoder的system KV与prefill验证，不是完整VLM验收。** 当前resident只接收token IDs；`research/resident-agent/README.md`明确记录尚未接入视觉encoder/projector。不能把新增256个文本tokens当成一张图片的真实输入测试。','',
'完整VLM后续应分别测：图像缩放/切块、视觉编码器、projector、视觉embedding注入、动态图像+文本prefill、首输出头。system KV可复用，但每轮新图片的视觉tokens不能直接复用旧图KV；图像改变时还要验证失效边界。旧图片完全相同时的图像特征缓存，应独立于system KV统计。','',
'## 质量样例','',
'W4的5条自然提问均出现乱码、答非所问或极短空洞回答；W8基本可读，但起床建议样例回答了早餐，W8也不是正确性oracle。','',
'| 问题 | W8 | W4（前120字符） |','|---|---|---|']
inputs=json.loads((r/'inputs.json').read_text())
def esc(x):
 x=x.replace('\\','\\\\')
 for char in ['|','*','_','`','[',']']:x=x.replace(char,'\\'+char)
 return x.replace('\n','<br>')
for q in inputs['quality']:
 a=next(x for x in s['quality'] if x['variant']=='w8' and x['case']==q['id']);b=next(x for x in s['quality'] if x['variant']=='w4' and x['case']==q['id'])
 lines.append(f"| {esc(q['query'])} | {esc(a['text'])} | {esc(b['text'][:120])} |")
d=s['diagnostics']
lines+=['',f"相同固定反馈64步：W4/W8 argmax一致{d['argmax_matches']}/{d['steps']}，logits相对L2={d['relative_l2']:.4f}，平均KL={d['mean_kl_nats']:.4f}nats。这个差异不能描述成普通准确率轻微下降，当前候选不可用。",'',
'## 结论与材料','',
'优先保留W8及system KV复用。尾部合批在这些开发样例上有首字收益，投入产品前仍需独立留出、多轮缓存截断/失效与图像embedding回归。W4的速度结果仅作为失败候选诊断材料；应先恢复数值/回答质量，再追求50%decode目标。对VLM的重点应是视觉路径接入及动态prefill，而非只优化decode。','',
'权威目录：`F:/Files/OPi-Lab/repos/OragnePi-3W_ForVLM/research/intern1b-int4-20261021`。',
'WSL构建目录：`/opt/opi-lab/intern1b-int4-20261021`（F盘VHDX）。板端隔离目录：`/opt/opi-vlm/intern1b-int4-side-20261021`。','',
'- `summary.json`、`verification.json`、`results/20261021T035301/events.jsonl`：整模W8/W4配对。',
'- `all-graph-storage-audit.json`、`weight-reconstruction-audit.json`、`build-manifests/`、`deployment-manifest.json`：量化容器、重构与28图hash。',
'- `ttft-tail/summary.json`、`ttft-tail/resident_session.c`、`ttft-tail/results/*/events.jsonl`：首字优化与原始事件。',
'- `cache-study/summary.json`、`cache-study/inputs.json`、`cache-study/results/*/events.jsonl`：system KV命中/重算配对。',
'- `task-provenance.json`：源commit和工具链。独立实验目录未提交Git，未修改主线程队列/HANDOFF或现有服务。','']
# Put the current VLM/prefill focus before the detailed original decode experiment.
lead=['**本轮重点：输入prefill与固定system KV复用。真实图像encoder/projector尚未接入，以下不能称为完整VLM首字成绩。**','',
'W8尾部合批将原短输入TTFT从1.93s降到1.57s（−18.5%）；5条自然回答逐token一致。W4整批prefill仅快约4%～5%，且质量已失败，因此暂不采用。','']
lines[2:2]=lead
import re
body='\n'.join(lines)+'\n'
pieces=re.split(r'(?m)^(## .+)\n',body);sections={pieces[i]:pieces[i+1] for i in range(1,len(pieces),2)}
order=['## system KV复用配对：本轮重点','## 首token优化：尾部合批','## 完整解码器与首token结果','## 量化与精度说明','## 质量样例','## 存储与运行资源','## 协议与边界','## 结论与材料']
assert set(order)==set(sections)
(r/'REPORT.md').write_text(pieces[0]+''.join(name+'\n'+sections[name] for name in order));print('report written',len(lines),'lines')
