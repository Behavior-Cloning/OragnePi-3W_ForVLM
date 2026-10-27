from pathlib import Path
import json,statistics,collections,re
P=Path(__file__).resolve().parent;x=json.loads((P/'results/summary.json').read_text());assert x['status']=='complete' and x['vision_closed']['driver_bytes']==0
rs=[json.loads(l) for l in (P/'results/resources.jsonl').read_text().splitlines()];events=[json.loads(l) for l in (P/'results/events.jsonl').read_text().splitlines()]
def stats(vals):
 a=sorted(vals);n=len(a)
 def q(v):j=(n-1)*v;i=int(j);return a[i]+(a[min(i+1,n-1)]-a[i])*(j-i)
 return {'n':n,'min':a[0],'p50':q(.5),'p95':q(.95),'max':a[-1]}
def mem(t,k):return int(re.search('^'+k+r':\s*(\d+)',t,re.M).group(1))
s={};groups=collections.defaultdict(list)
for r in x['records']:groups[r['phase']].append(r)
for phase,rows in groups.items():
 vv=[v for v in x['vision_records'] if v['phase']==phase];overlaps=[sum(max(0,min(r['end'],v['end'])-max(r['start'],v['start'])) for v in vv) for r in rows];res=[q for q in rs if q['phase']==phase];cpu=[]
 for a,b in zip(res,res[1:]):
  aa=list(map(int,a['cpu'].split()[1:9]));bb=list(map(int,b['cpu'].split()[1:9]));delta=[v-u for u,v in zip(aa,bb)];total=sum(delta)
  if total:cpu.append(100*(total-delta[3]-delta[4])/total)
 s[phase]={'ttft_s':stats([r['metrics']['ttft_seconds'] for r in rows]),'decode_tps':stats([r['metrics']['decode_tokens_per_second'] for r in rows]),'total_s':stats([r['end']-r['start'] for r in rows]),'requests_with_vision_overlap':sum(t>0 for t in overlaps),'overlap_s':overlaps,'vision_n':len(vv),'vision_s':stats([v['metrics']['seconds'] for v in vv]) if vv else None,'cpu_allcores_pct':stats(cpu) if cpu else None,'temp_max_c':max(t for q in res for t in q['temps'].values())/1000,'memavailable_min_mib':min(mem(q['meminfo'],'MemAvailable') for q in res)/1024,'skip_vision_ticks':sum(e['count'] for e in events if e['kind']=='skip_stale_vision_ticks' and e['phase']==phase),'new_tokens':sorted(set(r['metrics']['new_tokens'] for r in rows))}
base=[v for v in x['vision_records'] if v['phase']=='vision-alone-llm-idle'][2:];profile=[{'component':'stem' if i==0 else 'projector' if i==25 else f'layer{i-1:02d}','api_ms':statistics.median(v['metrics']['layer_api_seconds'][i] for v in base)*1000} for i in range(26)]
vm=lambda r:dict(line.split() for line in r['vmstat'].splitlines());aa=vm(rs[0]);bb=vm(rs[-1]);delta={k:int(bb[k])-int(aa[k]) for k in ['pswpin','pswpout','oom_kill']}
audit={'phases':s,'layer_profile':profile,'vmstat_delta':delta,'vision_idle_s':stats([v['metrics']['seconds'] for v in base]),'readies':[e for e in events if e['kind'] in ['vision_ready','language_ready']],'final_release':x['vision_closed'],'natural_vector_completion_age_s':[r['vector_age_s'] for r in groups['natural-latest-image']],'scope':x['scope']}
(P/'ANALYSIS.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
lines=['# 224 W8 ViT 与 Intern LLM 双NPU并发','', '2026-10-27。两模型真实同时常驻NPU，视觉CPU6、语言CPU7，CPU仅预处理/搬运/调度。并发是应用提交时间重叠，不代表芯片有两个独立NPU计算单元。未加入语音/RAG。','', '|阶段|语言请求|TTFT P50/P95 ms|decode P50 token/s|视觉P50 ms|与视觉重叠的语言请求|','|---|---:|---:|---:|---:|---:|']
for phase,row in s.items():
 v=f"{row['vision_s']['p50']*1000:.1f}" if row['vision_s'] else '—';lines.append(f"|{phase}|{row['ttft_s']['n']}|{row['ttft_s']['p50']*1000:.1f}/{row['ttft_s']['p95']*1000:.1f}|{row['decode_tps']['p50']:.2f}|{v}|{row['requests_with_vision_overlap']}|")
lines+=['', '前五组同6幅图/问题各2遍、固定反馈32步，仅用于等工作量性能比较，不能把固定反馈文本当自然回答。最后6条为自然问答，使用异步最近完成的视觉向量。每请求重建固定system后仅复用system KV，没有复用动态输入。','', 'system正文30＋模板5＝35，实际reuse32；动态图像64＋问题正文4–15，另11模板/图像边界；实际new82/83/84/89/93。性能组输出32，含固定反馈；自然组最多32。冷加载/system prime不算模型TTFT。','', f"单独运行视觉（LLM驻留但空闲）P50 {audit['vision_idle_s']['p50']*1000:.1f}ms。每2秒触发组跳过{ s['concurrent-period2']['skip_vision_ticks']}个过期tick，没有排队积压；因此并未稳定达到0.5Hz，需依据实际完成间隔评估频率。每5秒组与LLM也有资源竞争，不能说低频就没有影响。",'', '|阶段|全机CPU P50/P95|最高温度°C|最低MemAvailable MiB|','|---|---:|---:|---:|']
for phase,row in s.items():
 c=row['cpu_allcores_pct'];lines.append(f"|{phase}|{c['p50']:.1f}%/{c['p95']:.1f}%|{row['temp_max_c']:.1f}|{row['memavailable_min_mib']:.1f}|")
lines+=['', f"全段vmstat差分：{delta}（页/事件）；原始资源含每0.5秒CPU、温度、RSS/PSS、内存、swap。语言关闭时仍有视觉驱动分配{ x['language_closed']['driver_bytes']}B，这是共驻进程的剩余分配；视觉关闭后总driver0。不能要求关闭第一个模型时全设备内存立即归零。",'', '发现并保留的问题：natural-latest-image首两条复用了约32/34秒前完成的图像，原因是中间空闲对照阶段没有更新latest缓存。原回答虽正确，但时间语义不正确。下一独立版本需要以图像采集开始时刻计龄、等待首帧、超过4秒拒绝使用；不能将此首轮称为已解决数据新鲜度。', '', '## 完整塔逐层NPU API定位','', '|组件|API中位ms|','|---|---:|']
for q in profile:lines.append(f"|{q['component']}|{q['api_ms']:.3f}|")
lines+=['', '这是完整26图实跑拆分，未以单层时间替代整塔延迟。权重不变、每次视觉向量SHA与先前224 W8冻结输出相同。120题扩展另跑，不混入本性能6图。r1在加载模型前内存门控失败且0推理，已回收错误并清理SHA验证的旧临时副本；本r2独立结果可用。']
(P/'REPORT.md').write_text('\n'.join(lines)+'\n');print((P/'REPORT.md').read_text()[:6000])
