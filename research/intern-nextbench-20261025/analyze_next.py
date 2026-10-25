"""Analyze recovered files only. No device access or model execution."""
import argparse, array, collections, hashlib, json, math, statistics, sys
from pathlib import Path

def stats(values):
    xs=sorted(x for x in values if x is not None and math.isfinite(x))
    if not xs:return {'n':0}
    return {'n':len(xs),'min':xs[0],'p50':statistics.median(xs),'p95':xs[math.ceil(.95*len(xs))-1],'max':xs[-1],'mean':statistics.mean(xs)}
def read_rows(path,issues):
    rows=[]
    if not path.exists():issues.append('missing '+path.name);return rows
    for n,line in enumerate(path.read_text().splitlines(),1):
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:issues.append('%s malformed line %d'%(path.name,n))
    return rows

def aggregate_stage(rows):
    profiles=[x['metrics']['stage_profile'] for x in rows if x.get('metrics',{}).get('stage_profile')]
    result={'profiled_requests':len(profiles),'network_query_seconds':stats([p.get('network_query_seconds',0) for p in profiles]),'phases':{},'coverage':'scalar prompt, decode decoder and heads only; batched prefill is separately measured by prefill_seconds, not internally split'}
    for name in ['scalar_prompt','decode_decoder','first_head','decode_head']:
        phases=[p['phases'][name] for p in profiles if name in p.get('phases',{})]
        if not phases:continue
        names=set().union(*(x['seconds'] for x in phases))
        seconds={n:sum(x['seconds'].get(n,0) for x in phases) for n in sorted(names)}
        calls=sum(x.get('calls',0) for x in phases)
        transfer_keys=['input_prepare','input_flush','output_invalidate','kv_publish','hidden_read']
        host_keys=['embedding_rope','sampling']
        sums=[];cycles=[]
        for x in phases:
            a=x.get('network_sum_inference_us',[]);b=x.get('network_sum_cycles',[])
            if not sums:sums=[0]*len(a);cycles=[0]*len(b)
            if len(sums)!=len(a) or len(cycles)!=len(b):raise ValueError('profile network shape changed')
            sums=[u+v for u,v in zip(sums,a)];cycles=[u+v for u,v in zip(cycles,b)]
        result['phases'][name]={'requests':len(phases),'calls':calls,'calls_per_request':stats([x.get('calls',0) for x in phases]),'seconds_total':seconds,'seconds_per_call':{k:v/calls if calls else None for k,v in seconds.items()},'seconds_per_request':{n:stats([x['seconds'].get(n,0) for x in phases]) for n in sorted(names)},'groups_seconds_total':{'npu_run_wall':seconds.get('npu_run',0),'transfer_and_kv_host_wall':sum(seconds.get(k,0) for k in transfer_keys),'embedding_and_sampling_host_wall':sum(seconds.get(k,0) for k in host_keys),'logit_dump':seconds.get('logit_dump',0)},'network_samples':sum(x.get('network_samples',0) for x in phases),'network_sum_inference_us':sums,'network_sum_cycles':cycles,'network_queries_enabled':any(p.get('network_queries_enabled',False) for p in profiles)}
    return result

def resources_stats(rows):
    cpu=[]
    for a,b in zip(rows,rows[1:]):
        if b['at']-a['at']>2:continue
        aa=list(map(int,a['cpu'][0].split()[1:9]));bb=list(map(int,b['cpu'][0].split()[1:9]));d=[y-x for x,y in zip(aa,bb)];total=sum(d)
        if total>0:cpu.append(100*(total-d[3]-d[4])/total)
    def memory(r,key):return sum(p.get(key,0) for p in r.get('processes',{}).values())/1024
    out={'samples':len(rows),'cpu_percent_all_cores':stats(cpu),'temperature_C':stats([r['temp_mC']/1000 for r in rows]),'harness_plus_worker_pss_MiB':stats([memory(r,'Pss') for r in rows]),'harness_plus_worker_rss_MiB':stats([memory(r,'VmRSS') for r in rows]),'mem_available_MiB':stats([r['meminfo']['MemAvailable']/1024 for r in rows]),'load1':stats([float(r['loadavg'].split()[0]) for r in rows])}
    if rows:
        a,b=rows[0],rows[-1]
        out['swap_in_pages']=b['vmstat']['pswpin']-a['vmstat']['pswpin'];out['swap_out_pages']=b['vmstat']['pswpout']-a['vmstat']['pswpout']
        out['pss_first_last_MiB']=[memory(a,'Pss'),memory(b,'Pss')]
        out['audio']={'first':a.get('frontend',{}),'last':b.get('frontend',{}),'note':'Frontend counters cumulative across variants; queue is whole WAV jobs, not raw audio samples'}
    return out

def analyze(root):
    issues=[];events=read_rows(root/'events.jsonl',issues);resources=read_rows(root/'resources.jsonl',issues)
    meta=json.loads((root/'meta.json').read_text()) if (root/'meta.json').exists() else {'status':'missing'}
    out={'root':str(root),'meta':meta,'issues':issues,'event_counts':dict(collections.Counter(r['event'] for r in events)),'variants':{},'resources':resources_stats(resources),'scope':'Intern NPU text; synthetic stream audio and resident RAG. No image encoder, microphone or motor. Fixed feedback performance is not natural generation quality.'}
    plan=next((x for x in events if x['event']=='resident_models'),{})
    enabled=set(plan.get('phases',[]));repeat_count=plan.get('repeats')
    for x in events:
        if x['event'] in ['error','audio_error','monitor_error','release_error']:issues.append('runtime '+x['event'])
    names=list(dict.fromkeys(r['variant'] for r in events if 'variant' in r))
    baseline=next((x['variant'] for x in events if x['event']=='loaded'),None);refs={};diag_refs={}
    fixture=json.loads((root/'fixtures.json').read_text()) if (root/'fixtures.json').exists() else {}
    by_case={x['id']:x for x in fixture.get('performance',[])}
    for name in names:
        rows=[x for x in events if x.get('variant')==name];perf=[x for x in rows if x['event']=='measure'];quality=[x for x in rows if x['event']=='quality'];diagnostic=[x for x in rows if x['event']=='diagnostic']
        v={'counts':dict(collections.Counter(x['event'] for x in rows)),'cells':{},'stage_profile':aggregate_stage(perf),'quality':[],'diagnostic':[],'resources':resources_stats([x for x in resources if x.get('variant')==name]),'close':[x.get('metrics',{}) for x in rows if x['event'] in ['closed','closed_on_failure']]}
        for kind,actual,expected in [('performance',len(perf),12*repeat_count if repeat_count else None),('quality',len(quality),12),('diagnostic',len(diagnostic),4)]:
            if kind in enabled and expected is not None and actual!=expected:issues.append('count mismatch '+name+' '+kind)
        for cell in [(64,32),(64,64),(128,32),(128,64)]:
            subset=[x for x in perf if (x['static_tokens'],x['body_tokens'])==cell];decode=[];parity=[]
            if 'performance' in enabled and repeat_count and len(subset)!=3*repeat_count:issues.append('cell count mismatch '+name+' '+str(cell))
            for x in subset:
                m=x['metrics'];count=m['generated_tokens'];denom=m['total_seconds']-m['ttft_seconds']
                if count>1 and denom>0:decode.append((count-1)/denom)
                if m.get('diagnostic') or not m.get('fixed_feedback') or count!=32:issues.append('invalid speed flags/count '+name+' '+x['case'])
                if m.get('stage_profile',{}).get('network_queries_enabled'):issues.append('SDK query contaminated speed '+name)
                expected=by_case.get(x['case'],{})
                for k in ['reused_tokens','new_tokens']:
                    if expected and m[k]!=expected['expected_'+k]:issues.append('cache mismatch '+name+' '+x['case'])
                key=(x['repeat'],x['case']);tokens=m['tokens']
                if name==baseline:refs[key]=tokens
                prior=refs.get(key)
                parity.append({'repeat':x['repeat'],'case':x['case'],'reference':name==baseline,'exact':tokens==prior if prior is not None else None,'matching_positions':sum(a==b for a,b in zip(tokens,prior or [])),'tokens':len(tokens)})
            metrics={key:stats([x['metrics'].get(key) for x in subset]) for key in ['ttft_seconds','total_seconds','prefill_seconds','scalar_prompt_seconds','first_head_seconds','setup_seconds','batch_calls','tail_rows','scalar_prompt_tokens','new_tokens','reused_tokens','generated_tokens','true_tail8_calls','true_tail8_seconds']}
            metrics['decode_tokens_s_recomputed']=stats(decode)
            v['cells']['s%d-d%d'%cell]={'requests':len(subset),'metrics':metrics,'true_tail8_note':'true_tail8_seconds is included in prefill_seconds, do not add twice','decode_segments':{str(seg):stats([d['tokens_s'] for x in subset for d in x.get('decode_segments',[]) if d['output_positions']==seg]) for seg in [[2,8],[9,16],[17,24],[25,32]]},'fixed_feedback_parity':parity,'stage_profile':aggregate_stage(subset)}
        for x in quality:
            key=('quality',x['case']);tokens=x['metrics']['tokens']
            if name==baseline:refs[key]=tokens
            v['quality'].append({'case':x['case'],'input':x['input'],'reply':x['reply'],'finish_reason':x['metrics']['finish_reason'],'generated_tokens':x['metrics']['generated_tokens'],'exact_tokens_vs_baseline':tokens==refs.get(key) if key in refs else None,'tokens':tokens,'top3':[(a['id'],a['name'],a['score']) for a in x['top3']],'rag_s':x['rag_s'],'token_accounting':{k:x.get(k) for k in ['static_tokens','system_body_tokens','system_template_tokens','body_tokens','template_overhead','dynamic_tokens','total_prompt_tokens','expected_new_tokens','actual_new_tokens','actual_reused_tokens','output_tokens_including_eos','output_tokens_excluding_terminal_eos']}})
        for x in diagnostic:
            file=root/x['logits_file'];d={'case':x['case'],'path':x['logits_file'],'exists':file.exists(),'logged_comparison':x.get('logits_parity')}
            if file.exists():
                raw=file.read_bytes();a=array.array('f');a.frombytes(raw)
                if sys.byteorder!='little':a.byteswap()
                finite=all(math.isfinite(t) for t in a);hashvalue=hashlib.sha256(raw).hexdigest()
                d.update(bytes=len(raw),sha256=hashvalue,hash_matches=hashvalue==x['sha256'],finite=finite,steps=x['metrics']['generated_tokens'],diagnostic=x['metrics'].get('diagnostic'))
                if not finite or not d['hash_matches'] or len(raw)!=4*151674*4:issues.append('diagnostic failure '+name+' '+x['case'])
                if name==baseline:diag_refs[x['case']]=file
                ref=diag_refs.get(x['case'])
                if ref and name!=baseline:
                    b=array.array('f');b.frombytes(ref.read_bytes())
                    if sys.byteorder!='little':b.byteswap()
                    if len(a)!=len(b):issues.append('logit size mismatch '+name)
                    else:
                        denom=sum(t*t for t in b);err=sum((u-v)**2 for u,v in zip(a,b));width=len(a)//4
                        top1=sum(max(range(width),key=lambda i:a[j*width+i])==max(range(width),key=lambda i:b[j*width+i]) for j in range(4))
                        d['independent_comparison']={'relative_l2':math.sqrt(err/max(denom,1e-60)),'max_abs':max(abs(u-v) for u,v in zip(a,b)),'top1_matches':top1,'steps':4}
            else:issues.append('missing diagnostic '+str(file))
            v['diagnostic'].append(d)
        driver_values=[x.get('ready',{}).get('driver_bytes') for x in rows if x['event']=='loaded']+[x.get('metrics',{}).get('driver_bytes') for x in rows if x['event'] in ['measure','quality','diagnostic']]
        v['driver_bytes']=stats(driver_values)
        v['driver_MiB']=stats([b/1048576 for b in driver_values if b is not None])
        v['graph_verification']=[x.get('files',[]) for x in rows if x['event']=='graph_verified']
        v['diagnostic_stage_profile']=aggregate_stage(diagnostic)
        if not v['close'] or any(x.get('driver_bytes')!=0 for x in v['close']):issues.append('release unconfirmed '+name)
        out['variants'][name]=v
    if meta.get('status')!='complete':issues.append('workload meta not complete')
    out['audit_pass']=not issues
    return out

def markdown(data):
    lines=['# Intern NPU 下一轮基准分析','', '只分析回收原始证据；没有启动板卡。原始文件保持不变。','',
        '状态：'+data['meta'].get('status','missing')+'；结构/释放审计：'+str(data['audit_pass']),
        '', '输入定义：static包含system模板；动态正文32/64，另有8个模板token，实际new应为40/72。',
        '速度仅32步固定反馈、diagnostic=False；自然质量和全词表诊断独立。',
        '', '| 配置 | static/dynamic正文 | n | TTFT P50/P95秒 | decode P50 tok/s | batch prefill P50秒 | 首head P50秒 |',
        '|---|---|---:|---:|---:|---:|---:|']
    def n(s,k='p50'):return '—' if not s or k not in s else ('%.4f'%s[k])
    for name,v in data['variants'].items():
        for cell,c in v['cells'].items():
            m=c['metrics'];lines.append('| %s | %s | %d | %s / %s | %s | %s | %s |'%(name,cell,c['requests'],n(m['ttft_seconds']),n(m['ttft_seconds'],'p95'),n(m['decode_tokens_s_recomputed']),n(m['prefill_seconds']),n(m['first_head_seconds'])))
    lines+=['','P95采用nearest-rank；每格仅3次的烟测P95等于样本最大值，不能当稳定尾延迟结论。','',
        '## 质量与诊断','', '一致性仅相对首个配置，不能据此认定模型决策正确。']
    for name,v in data['variants'].items():
        lines+=['', '真tail8计量（seconds已包含在prefill_seconds内）：', '', '| 配置/格 | calls P50 | tail8秒 P50 |', '|---|---:|---:|']
        for cell,c in v['cells'].items():
            m=c['metrics'];lines.append('| %s/%s | %s | %s |'%(name,cell,n(m.get('true_tail8_calls',{})),n(m.get('true_tail8_seconds',{}))))
        natural=v['quality'];measure=[q for c in v['cells'].values() for q in c['fixed_feedback_parity']]
        lines+=['','### '+name,'', '固定反馈完全一致：%d/%d；自然输出完全一致：%d/%d。'%(sum(x['exact'] is True for x in measure),len(measure),sum(x['exact_tokens_vs_baseline'] is True for x in natural),len(natural)), '', '| case | 输出 | Top1 | 与基线token一致 |','|---|---|---|---|']
        for q in natural:lines.append('| %s | %s | %s | %s |'%(q['case'],q['reply'].replace('|','/').replace('\n',' '),q['top3'][0][1] if q['top3'] else '—',q['exact_tokens_vs_baseline']))
        for d in v['diagnostic']:lines.append('- %s：存在=%s，finite=%s，SHA匹配=%s，独立对照=%s'%(d['case'],d['exists'],d.get('finite'),d.get('hash_matches'),d.get('independent_comparison','基线')))
        lines+=['','阶段profile：合批prefill未内部拆分；npu_run是SDK运行调用墙钟时间，不是净芯片算力时间。', '', '| phase | calls | NPU调用wall秒 | 搬运/KV host秒 | embedding/sampling host秒 |','|---|---:|---:|---:|---:|']
        for phase,z in v['stage_profile']['phases'].items():
            g=z['groups_seconds_total'];lines.append('| %s | %s | %.5f | %.5f | %.5f |'%(phase,z['calls'],g['npu_run_wall'],g['transfer_and_kv_host_wall'],g['embedding_and_sampling_host_wall']))
        r=v['resources'];lines+=['', '资源：CPU均值%s%%，最高温度%s°C，PSS峰值%sMiB，最低MemAvailable%sMiB；swap in/out=%s/%s页。'%(n(r['cpu_percent_all_cores'],'mean'),n(r['temperature_C'],'max'),n(r['harness_plus_worker_pss_MiB'],'max'),n(r['mem_available_MiB'],'min'),r.get('swap_in_pages'),r.get('swap_out_pages')), 'NPU driver常驻峰值：%s MiB（单列，不能未经核对直接与宿主PSS相加）；closed=%s。'%(n(v.get('driver_MiB',{}),'max'),v['close'])]
    lines+=['','## 限制与审计','', '- CPU按全8核容量归一化；PSS为harness+worker，不代表全板所有服务。', '- 音频由Frontend以真实时间播放预录WAV；RAG性能阶段仅常驻，质量阶段才查询。','- 无视觉、真实麦克风、电机；固定反馈不代表真实对话质量。','- SDK NPU查询仅诊断集合，查询时间不得纳入速度比较。','- diagnostic关闭时profile logit_dump的微秒值是空分支计时开销；实际目录检查确保性能阶段没有写词表。']
    lines += ['- '+x for x in data['issues']] or ['- 结构、有限值与释放检查无异常。']
    return '\n'.join(lines)+'\n'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run_dir');ap.add_argument('--out',required=True);args=ap.parse_args()
    result=analyze(Path(args.run_dir));out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'summary_next.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));(out/'REPORT_NEXT.md').write_text(markdown(result))
    print(json.dumps({'audit_pass':result['audit_pass'],'issues':result['issues'],'variants':list(result['variants']),'output':str(out)},ensure_ascii=False))
if __name__=='__main__':main()
