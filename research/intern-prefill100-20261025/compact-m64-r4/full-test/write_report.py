from pathlib import Path
import json
D=Path(__file__).resolve().parent
A=json.loads((D/'ANALYSIS.json').read_text());Q=json.loads((D/'QUALITY_REVIEW.json').read_text())
rows=A['performance']; lookup={(x['variant'],x['system_total'],x['dynamic_body']):x for x in rows}
L=['# Compact M64完整24层A/B/A实验', '', '结论：登记为性能候选，质量不合格，默认版本不变。system64、动态正文128时prefill P50=464.5ms，TTFT P50=485.6ms；system128同动态仍545.9ms，动态256仍约1.0–1.1秒。decode没有确定提速，完整范围500ms目标未达到。', '', '## 版本和条件', '', '- 实现提交：4a4c57e3434e3bedbbb6be89ece8008cd9c1bbb7。独立run compact-m64-r4；配置、部署图、编译来源与冻结输入分别见config.json、deployment-sha.json和上级graphs/*/build.json。', '- 父版本outlier8-c448-full-r1：L2/L3 down保留8个FP16离群通道，其余I16，已不同于原始量化。当前候选保持父版本量化和权重；改变分块与attention形状仍改变数值，不能叫原精度或无损。', '- 仍是Transformer。KV存储448；新增past128/query64图。有效history能全部放入128时才调用，不截断历史；后续回退main32，尾部tail8。system64能用2个compact块，system128只能用1个。', '- 同一新runtime依次关闭compact、开启compact、再关闭。旧runtime一个4步完整logits输入同字节，A/A六个诊断输入全部同字节。', '- 每格相同token IDs，只复用system KV；每请求重新准备system，冷构建不计入TTFT。性能输出固定反馈32token，decode=31/(总耗时-TTFT)。每格3样例，P95最近秩即最大值，不代表长期P95。', '- 共13格：覆盖system32/64×dynamic32/64/128/256以及system64/128×dynamic64/128/192/256。system32正文截断、system128重复并补标点、动态成本输入有标点填充；用于成本对齐，语义另用12条新冻结自然输入。', '- 语音前端合成录音流式常驻，RAG常驻但未检索；Intern NPU纯文本，无图像、真实麦克风、电机。该TTFT不含语音、摘要、排队、RAG或动作执行。', '', '## 完整token和候选性能', '', 'system正文+模板=实际复用；动态正文+模板=实际new。各行输出32token。', '', '|system正文|模板|总量/reused|动态正文|模板|new|总输入|prefill P50 ms|TTFT P50/P95 ms|decode P50 tok/s|TTFT<500ms|', '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
 if r['variant']!='compact':continue
 L.append(f"|{r['system_body']}|5|{r['system_total']}|{r['dynamic_body']}|8|{r['new_prefill']}|{r['total_prompt']}|{r['prefill_seconds']['p50']*1000:.2f}|{r['ttft_seconds']['p50']*1000:.2f}/{r['ttft_seconds']['p95_nearest_rank']*1000:.2f}|{r['decode_tokens_per_second']['p50']:.2f}|{r['lt500ms']}/3|")
L+=['','## 同输入A/B/A','', '以下按同格较快的父版本A/A作为保守参考。完整原量化基线不是这里的parent；不要跨量化把所有收益归因于M64。', '', '|system总量|动态正文|父版TTFT前/后 ms|候选TTFT ms|相对较快父版降低|compact调用数|', '|---:|---:|---:|---:|---:|---:|']
for r in rows:
 if r['variant']!='compact':continue
 s,b=r['system_total'],r['dynamic_body'];a=lookup['parent',s,b]['ttft_seconds']['p50'];c=lookup['parent-repeat',s,b]['ttft_seconds']['p50'];z=r['ttft_seconds']['p50'];pct=(1-z/min(a,c))*100
 L.append(f"|{s}|{b}|{a*1000:.2f}/{c*1000:.2f}|{z*1000:.2f}|{pct:.1f}%|{r['compact_calls']['p50']:.0f}|")
L+=['','动态32未调用compact；微小差异属于波动，不登记为算法收益。system64动态64首个A较慢，采用重复A的342.61ms作参考。动态256在past128以后回退大history图，仍是下一步主要瓶颈。','', '## 质量与数值', '', '12条新自然中文请求（6场景×system64/128），动态正文96–101、模板8、实际new104–109，输出实际2–5token，max32。预先冻结期望与禁用动作；逐条人工式语义审查由Codex完成，并非独立人类评测或总体准确率。','', '候选与父版输出10/12相同；固定反馈37/39相同。六个4步完整logits诊断×三版本全部有限、每文件2426784B，真实SHA复核；启用compact的五例relative L2约2.74%–2.96%，关闭compact的一例0。这不是准确率下降百分比。', '', '父版和候选各6/12直接完成指定动作、5/12保守不动漏动作、1/12与明确禁止方向冲突。候选在system128低头请求退化为静止，在弯腰请求从静止改善为鞠躬。两版system64的取消向前、改后退请求都输出向前移动。不得晋升默认，不执行电机。', '', '|案例|期望|父版|候选|候选判定|', '|---|---|---|---|---|']
parents={r['case']:r for r in Q['rows'] if r['variant']=='parent'}
for r in Q['rows']:
 if r['variant']=='compact':L.append(f"|{r['case']}|{r['expected_action']}|{parents[r['case']]['output']}|{r['output']}|{r['grade']}|")
L+=['','## 负载与释放','','资源采样覆盖约7.58分钟，非1小时重测；8核总容量归一到100%。PSS与驱动内存单列，不能直接相加。','', '|阶段|CPU P50/P95 %|峰温°C|最低可用MiB|峰PSS MiB|swap in/out页|', '|---|---:|---:|---:|---:|---:|']
for name in ['all','parent','compact','parent-repeat']:
 r=A['resources'][name];L.append(f"|{name}|{r['cpu_busy_pct_8cores']['p50']:.2f}/{r['cpu_busy_pct_8cores']['p95_nearest_rank']:.2f}|{r['temperature_C']['max']:.3f}|{r['available_MiB']['min']:.1f}|{r['sum_process_PSS_MiB']['max']:.1f}|{r['swap_in_pages_delta']}/{r['swap_out_pages_delta']}|")
L+=['', '驱动ready：parent1014351896B、compact1334332408B、repeat1014089752B，候选额外约305.2MiB。三次正常关闭driver=0，post-health分配/释放813/813、current0。新增图文件保存在TF独立目录，不覆盖现役图。', '', '全窗口swap已用约2915.6→2926.6MiB，音频28429帧中1391帧计算超过16ms、最大累计滞后1.567s；不能声称语音完全实时。最终音频快照29718帧/1479超预算包含资源窗口以外时间，队列1是待播放输入，不能据此推断ASR积压。', '', '## 推广决定与下一步', '', '更新最快性能候选指针，原始量化验证版本和默认部署保持不变，质量合格最优为空。M128实验未改善和softmax形状改写略慢均保留为否定结果。下一步针对past128之后的prefill、约70ms tail8及固定head开销分段优化；必须用相同输入A/B/A及新语义请求继续核查。长期稳定性、完整语音延迟和动作正确性尚未通过，不因单格低于500ms标记任务完成。', '', '原始f32和资源日志留Windows磁盘，ANALYSIS.json绑定SHA。源码、冻结输入、统计、质量逐案判定和报告提交Git；不可移动旧tag或改写已冻结原始证据。']
(D/'REPORT.md').write_text('\n'.join(L)+'\n')
