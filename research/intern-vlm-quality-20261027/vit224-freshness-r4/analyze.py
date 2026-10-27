from pathlib import Path
import json,statistics,re,collections
P=Path(__file__).resolve().parent;load=lambda f:json.loads((P/f).read_text());x=load('results/summary.json');assert x['status']=='complete' and not x['monitor_errors'] and not x['cleanup_errors'] and x['vision_closed']['driver_bytes']==0
r=x['records'];assert len(r)==12 and all(z['vector_age_s']<=4 for z in r)
e=[json.loads(l) for l in (P/'results/events.jsonl').read_text().splitlines()];rs=[json.loads(l) for l in (P/'results/resources.jsonl').read_text().splitlines()]
def st(a):
 a=sorted(a);n=len(a)
 def q(p):f=(n-1)*p;i=int(f);return a[i]+(a[min(i+1,n-1)]-a[i])*(f-i)
 return {'n':n,'min':a[0],'p50':q(.5),'p95':q(.95),'max':a[-1]}
def mem(t,k):return int(re.search('^'+k+r':\s*(\d+)',t,re.M).group(1))
a={'phases':{},'capture_ttl_s':4,'requests':12,'injected_stale_timestamps':sum(z.get('kind')=='inject_stale_cache_age' for z in e),'wait_events':sum(z.get('kind')=='wait_fresh_image' for z in e),'final_driver_bytes':x['vision_closed']['driver_bytes']}
for phase in sorted({z['phase'] for z in r}):
 rows=[z for z in r if z['phase']==phase];v=[z for z in x['vision_records'] if z['phase']==phase]
 a['phases'][phase]={k:st([z[k] for z in rows]) for k in ['vector_age_s','frame_age_at_complete_s','fresh_wait_s']}
 a['phases'][phase].update(ttft_s=st([z['metrics']['ttft_seconds'] for z in rows]),decode_tps=st([z['metrics']['decode_tokens_per_second'] for z in rows]),wait_plus_ttft_s=st([z['fresh_wait_s']+z['metrics']['ttft_seconds'] for z in rows]),vision_s=st([z['metrics']['seconds'] for z in v]),skip_ticks=sum(z['count'] for z in e if z.get('kind')=='skip_stale_vision_ticks' and z['phase']==phase))
a['temp_max_c']=max(t for z in rs for t in z['temps'].values())/1000;a['memavailable_min_mib']=min(mem(z['meminfo'],'MemAvailable') for z in rs)/1024
vm=lambda z:dict(l.split() for l in z['vmstat'].splitlines());first,last=vm(rs[0]),vm(rs[-1]);a['vmstat_delta']={k:int(last[k])-int(first[k]) for k in ['pswpin','pswpout','oom_kill']}
(P/'ANALYSIS.json').write_text(json.dumps(a,indent=2)+'\n')
lines=['# 异步NPU视觉：按采集时间拒绝过期图像','', '两模型同时常驻NPU；224 W8视觉塔与Intern语言模型。12条自然问答；5秒/2秒视觉定时各6条。只验证图像新鲜度门控，不是120题质量分数，也不是长稳压力测试。','', '两阶段开始时各注入31秒旧时间戳的已知向量（故障注入，不是真实相机等待31秒），直到采集开始时间距当前≤4秒才派发给语言模型。单个视觉任务串行，跳过过期tick，不排队堆积。两次注入均被门控拦下；12/12派发帧龄≤4秒。','', '|定时|TTFT P50/P95 ms|等待新图+TTFT P50/P95 ms|等待新图最大 ms|派发帧龄最大 s|回答结束帧龄最大 s|decode P50 t/s|','|---|---:|---:|---:|---:|---:|---:|']
for phase,z in a['phases'].items():lines.append(f"|{phase}|{z['ttft_s']['p50']*1000:.1f}/{z['ttft_s']['p95']*1000:.1f}|{z['wait_plus_ttft_s']['p50']*1000:.1f}/{z['wait_plus_ttft_s']['p95']*1000:.1f}|{z['fresh_wait_s']['max']*1000:.1f}|{z['vector_age_s']['max']:.3f}|{z['frame_age_at_complete_s']['max']:.3f}|{z['decode_tps']['p50']:.2f}|")
lines+=['', '等待新图+TTFT仅将实际等待时长加模型TTFT，不含此前system prime及上游事件/调度，不冒充事件到首字完整延迟。5秒采样配4秒TTL会在更新间隔末尾等待下一帧，这是明确延迟代价。TTL只约束推理派发；生成完成时帧龄可能超过4秒。真正控制动作前仍需再次检查场景版本/时间并决定丢弃过期计划；本测试无电机。','', 'system正文30+模板5=35，实际reuse32；动态图像64+问题正文4–15；另11模板/图像边界；实际new82–93。自然输出最多32，没有动态KV复用。新图与对应问题配对，未将另一图的问题强加给latest向量。','', f"最高温度{a['temp_max_c']:.1f}°C，最低MemAvailable {a['memavailable_min_mib']:.1f} MiB；vmstat差分{a['vmstat_delta']}页/事件。模型正常关闭，最后driver=0。r3在加载前内存门控失败；清理Windows已校验SHA的板端副本后以r4独立重试。WSL控制器背景会话退出，板端完成summary/日志经双锁独立回收，状态标complete-recovered；不复用或伪造r3结果。",'', '答案与新图对应不等于答案都正确：home039仍把剪刀左右位置答错，保留原输出。先前r2发现32/34秒旧图被使用；此版处理派发新鲜度，不修复VQA认知错误。']
(P/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(a))
