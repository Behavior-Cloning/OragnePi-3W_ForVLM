"""Current DFP candidate coverage, with exact length and CPU trajectory checks."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import struct
from matrix_paths import cpu_result_path
from report_system_load import summarize

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'results'


def read(path):return json.loads(path.read_text())


def rejection_reason(failure):
    validation=failure['validation'];q=failure['original_q8_development'];parts=[]
    if not validation['execution_gate_v1']['passed']:
        parts.append(f'执行L2 {validation["relative_l2_all_logits"]:.3%} / cosine {validation["min_step_cosine"]:.5f} / argmax {validation["argmax_match_fraction"]:.2%}')
    if q['kl']>.05:parts.append(f'原Q8 KL {q["kl"]:.5f}>0.05')
    if q['ppl_ratio']>1.05:parts.append(f'原Q8 PPL比 {q["ppl_ratio"]:.5f}>1.05')
    assert parts
    return '；'.join(parts)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--finalists',action='store_true',help='current CPU trajectories and the separately labelled 1B long-history combination')
    args=parser.parse_args()
    candidates={}
    failures={}
    def collect(path,label):
        if not path.exists():return
        d=read(path)
        if d.get('status')!='complete':return
        v=next(v for v in d['config']['variants'] if v['label']==label)
        # Exclude the older product-range repair baseline and all failed MLP/head trials.
        combo=args.finalists and v['remote'].endswith(('/full-intern1b-c1280-cap1-headmse','/full-intern1b-c768-cap1-headmse'))
        if not combo:
            if not any(s in v['remote'] for s in ('dfpgateup','dfpmse')):return
            if any(s in v['remote'] for s in ('headmse','dfpmlp')):return
        runs=[r for r in d['runs'] if r['phase']=='measure' and r['label']==label]
        assert len(runs)==5
        runtime=runs[0]['result'];model='smol500' if runtime['decoder_layers']==32 else 'intern1b'
        depth=32 if model=='smol500' else 24
        h,n=runtime['cached_prefix_tokens'],runtime['new_prompt_tokens']
        if args.finalists and model=='intern1b' and h==1024 and not combo:return
        assert all(r['result']['decoder_layers']==depth and r['result']['head_on_npu'] and
                   r['result']['generated_tokens']==128 and r['result']['fixed_feedback'] and
                   r['result']['cached_prefix_tokens']==h and r['result']['new_prompt_tokens']==n and
                   r['result']['feedback_tokens']==runtime['feedback_tokens'] and
                   r['result']['driver_after_release_bytes']==0 for r in runs)
        bounds=d['config']['temperature_band_mC'];band='low' if bounds==[40000,50000] else 'medium'
        assert bounds==([40000,50000] if band=='low' else [55000,64999])
        assert all(bounds[0]<=r['result']['response_start_temperature_mC']<=bounds[1] for r in runs)
        assert v['validation']['passed']
        key=(model,h,n,band)
        candidates[key]={'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                         'variant':v,'medians':d['medians'][label],'runtime':runtime,
                         'config':d['config'],'runs':[{k:v for k,v in r.items() if k!='system_samples'} for r in runs]}
    for path,label in [('smol500-dfpgateup-ab-20261013.json','dfp'),
                       ('dfpmse-prefill16-trial-20261013/performance.json','candidate'),
                       ('smol500-c1280-dfp-trial-20261013-attempt2/performance.json','candidate'),
                       ('dfpmse-productwide-trial-20261013-attempt2/performance.json','candidate')]:
        collect(ROOT/path,label)
    folders=('candidate-expansion-20261013','candidate-long-expansion-20261013','candidate-short-low-20261013')
    if args.finalists:folders+=('cpu-finalist-realignment-20261014','short-combination-expansion-20261014')
    for folder in folders:
        state=ROOT/folder/'state.json'
        if state.exists():
            for row in read(state)['cases'].values():
                if row.get('status')=='quality-failed':
                    cfg=row['config'];runtime=read(Path(row['reference'])/'tokens.json')
                    model='smol500' if runtime['decoder_layers']==32 else 'intern1b'
                    if args.finalists and model=='intern1b' and runtime['cached_prefix_tokens']==1024 and 'cap1-headmse' not in cfg['tag']:
                        continue
                    failure={'case':cfg['case'],'reference':row['reference'],'tag':cfg['tag'],
                             'validation':read(Path(row['reference'])/'full-validation-w8.json'),
                             'original_q8_development':row['original_q8_development']}
                    for band in ('low','medium'):
                        failures[(model,runtime['cached_prefix_tokens'],runtime['new_prompt_tokens'],band)]=failure
                for band in row.get('bands',{}).values():
                    collect(Path(band['result']),'candidate')
    if args.finalists:collect(ROOT/'cap1-headmse-short-trial-20261014/performance.json','candidate')
    search=read(ROOT/'cpu-q8-search-20261013/state.json')
    low_path=ROOT/'cpu-q8-low-finalists-20261013/state.json'
    low_search=read(low_path) if low_path.exists() else {'cases':{}}
    rows=[]
    title='最新CPU对齐与组合候选矩阵' if args.finalists else '当前DFP候选精确矩阵'
    description=('500M沿用gate/up峰值DFP；1B1024历史为cap1＋新输出头组合，512历史在各格完成验证后列出组合，否则保留原gate/up MSE＋prefill16并明确版本。与CPU配置搜索五轮入围的最快decode方案重新回灌对齐；本表不把旧1B长历史候选冒充组合方案。' if args.finalists else
                 '500M gate/up峰值DFP，1B gate/up MSE DFP＋prefill16，1024历史另有已验证product范围修复。此表不混入旧NPU基线、质量失败的MLP候选或尚未验证的新输出头。')
    lines=['# '+title,'',description+'全部32/24层、完整词表、固定128预测、预热＋五轮。','',
           f'当前候选五轮完成{len(candidates)}/24格；“待测”不是0速度。独立留出仍单独验收，不能以开发轨迹通过自动推广产品。','',
           '| 模型 / 历史＋新增 / 起温 | NPU配置 | NPU token/s | NPU首token s | 同温CPU token/s | CPU首token s | 对齐/基线状态 |',
           '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for model in ('smol500','intern1b'):
        for h in (512,1024):
            s=search['cases'].get(f'{model}-h{h}-n64',{})
            for n in (32,64,128):
                case=f'{model}-h{h}-n{n}'
                for band in ('low','medium'):
                    key=(model,h,n,band)
                    c=candidates.get(key)
                    display=f'{model} / {h}＋{n} / {band}'
                    if not c:
                        failed=failures.get(key)
                        if failed:
                            v=failed['validation']
                            lines.append(f'| {display} | — | 精度未过 | — | — | — | {rejection_reason(failed)}；不进入测速 |')
                        else:lines.append(f'| {display} | — | 待测 | — | — | — | — |')
                        continue
                    version=('cap1＋headMSE' if c['variant']['remote'].endswith('cap1-headmse') else
                             '峰值DFP' if model=='smol500' else 'DFP MSE＋t16')
                    cpu_path=cpu_result_path(ROOT/'thermal-aligned-20261013',case,band)
                    note='原矩阵Q8配置，尚非本格最优确认'
                    if band=='medium':
                        prefix='' if n==64 else f'n{n}-'
                        finals={k:v for k,v in s.get('runs',{}).items() if k.startswith(prefix+'final-') and
                                v.get('repeats')==5 and v.get('medians')}
                        if finals:
                            winner=max(finals,key=lambda k:finals[k]['medians']['decode_tokens_per_second'])
                            cpu_path=Path(finals[winner]['path'])
                            note=winner+'；配置搜索'+s.get('status','pending')
                    else:
                        low=low_search['cases'].get(case,{})
                        if low.get('status')=='complete':
                            winner=low['fastest_decode']
                            cpu_path=Path(low['runs'][winner]['path'])
                            note=winner+'；中温入围配置低温复核，非低温全搜索'
                    cpu=read(cpu_path) if cpu_path.exists() else {}
                    if cpu.get('status')!='complete':
                        m=c['medians']
                        lines.append(f'| {display} | {version} | {m["decode_tokens_per_second"]:.3f} | {m["ttft_seconds"]:.3f} | 待补测 | — | CPU当前attempt未完成 |')
                        rows.append({'model':model,'history':h,'new_tokens':n,'band':band,'npu':c,
                                     'cpu':None,'trajectory_aligned':None,'cpu_note':'pending current matrix attempt'})
                        continue
                    assert cpu['status']=='complete' and cpu['cached_prefix']==h and len(cpu['prompt']['token_ids'])==h+n
                    packed=struct.pack('<'+'i'*len(cpu['prompt']['token_ids']),*cpu['prompt']['token_ids'])
                    assert hashlib.sha256(packed).hexdigest()==c['config']['prompt_sha256']
                    aligned=cpu['nonstream_trajectory']['tokens']==c['runtime']['feedback_tokens']
                    if not aligned:note+='；反馈不同，待重新对齐'
                    else:note+='；反馈一致'
                    failed=failures.get(key)
                    if failed:
                        if c['variant']['remote'].endswith('/'+failed['tag']) and not aligned:
                            note+='；本配置的新反馈已拒绝：'+rejection_reason(failed)
                        elif not c['variant']['remote'].endswith('/'+failed['tag']):
                            note+='；另一个候选未过质量，详见下表'
                    measured=[r for r in cpu['runs'] if r['phase']=='measure'];assert len(measured)==5
                    ttft=statistics.median(r['timings']['prompt_ms']/1000 for r in measured)
                    speed=cpu['medians']['decode_tokens_per_second']
                    m=c['medians']
                    lines.append(f'| {display} | {version} | {m["decode_tokens_per_second"]:.3f} | {m["ttft_seconds"]:.3f} | {speed:.3f} | {ttft:.3f} | {note} |')
                    rows.append({'model':model,'history':h,'new_tokens':n,'band':band,'npu':c,
                                 'cpu':{'source':str(cpu_path),'source_sha256':hashlib.sha256(cpu_path.read_bytes()).hexdigest(),
                                        'decode_tokens_per_second':speed,'engine_ttft_seconds':ttft},
                                 'trajectory_aligned':aligned,'cpu_note':note})
    if failures:
        lines+=['','## 未通过的完整开发验证','',
                '每条固定轨迹先过执行门槛，再允许低、中温测速；同一精度失败同时阻止两温档，不能写作普通待测。保持L2≤5%、最低cosine≥0.99、argmax≥95%，原Q8 KL≤0.05/PPL比≤1.05的门槛不变。','',
                '| 开发场景 | 执行L2 | 最低cosine | argmax一致 | 原Q8 KL / PPL比 | 拒绝原因 |','| --- | ---: | ---: | ---: | ---: | --- |']
        for case in sorted({v['case'] for v in failures.values()}):
            f=next(v for v in failures.values() if v['case']==case);v=f['validation'];q=f['original_q8_development']
            lines.append(f'| {case} | {v["relative_l2_all_logits"]:.4%} | {v["min_step_cosine"]:.6f} | {v["argmax_match_fraction"]:.3%} | {q["kl"]:.6f} / {q["ppl_ratio"]:.6f} | {rejection_reason(f)} |')
    lines+=['','低温起点40–50°C，中温55–<65°C，响应期间自然变化。两端首token均使用引擎口径，不含加载、历史KV准备、调温和HTTP缓冲。低温最快CPU配置仍需同温复核，不能拿中温CPU直接替换低温成绩。','',
            '先筛64新增，再把入围CPU配置扩32/128；不等于穷举所有实现。若更快CPU改变贪心轨迹，保留更高目标线并要求NPU重新回灌和完整logits校验，不回退到慢CPU宣布胜出。','',
            '[独立质量与持续运行](独立质量与连续多轮验收.md) · [真实ASR并发](真实ASR与LLM并发验收.md) · [配对A/B收益与负载](全深度候选提速与负载.md)','',
            '重建：`npu/.venv/bin/python npu/kv-cache/report_candidate_matrix.py'+(' --finalists' if args.finalists else '')+'`。','']
    if args.finalists:
        completed=sum(r['trajectory_aligned'] is True and ('配置搜索complete' in r['cpu_note'] or '低温复核' in r['cpu_note']) for r in rows)
        lines.insert(4,f'当前同温五轮、CPU入围配置确认且反馈一致：{completed}/24格。未对齐项继续保留CPU更高目标线，不用于加速比。新组合的独立质量另验，不继承旧版本结论。\n')
        lines+=['## 当前候选与已确认CPU的直接负载','',
                '仅纳入上表同温、反馈一致且CPU入围配置确认的格子。性能和资源来自同一轮响应；不将旧版本的占用拼到新速度上。整机CPU按8核合计100%，NPU驱动分配与宿主RSS分别列出，二者不能简单相加视为去重后的物理内存。','',
                '| 模型 / 历史＋新增 / 起温 | NPU相对CPU速度 | CPU/NPU整机CPU % | CPU/NPU进程RSS MiB | NPU驱动 MiB | CPU/NPU可用内存最低 MiB | CPU/NPU响应内温度°C |',
                '| --- | ---: | --- | --- | ---: | --- | --- |']
        fmt=lambda x:'未采到' if x is None else f'{x:.1f}'
        for row in rows:
            if row['trajectory_aligned'] is not True or not ('配置搜索complete' in row['cpu_note'] or '低温复核' in row['cpu_note']):continue
            cp,np=Path(row['cpu']['source']),Path(row['npu']['source'])
            assert hashlib.sha256(cp.read_bytes()).hexdigest()==row['cpu']['source_sha256']
            assert hashlib.sha256(np.read_bytes()).hexdigest()==row['npu']['source_sha256']
            c=summarize(cp,'cpu');n=summarize(np,'npu',row['npu']['variant']['label'])
            row['direct_resources']={'cpu':c,'npu':n}
            key=f'{row["model"]} / {row["history"]}＋{row["new_tokens"]} / {row["band"]}'
            if not c or not n:
                lines.append(f'| {key} | — | 缺少直接采样 | — | — | — | — |');continue
            c,n=c['medians'],n['medians']
            gain=n['decode_tps']/c['decode_tps']-1
            lines.append(f'| {key} | {gain:+.1%} | {c["busy_percent"]:.2f} / {n["busy_percent"]:.2f} | {fmt(c["rss_peak_mib"])} / {fmt(n["rss_peak_mib"])} | {n["driver_mib"]:.1f} | {c["available_min_mib"]:.0f} / {n["available_min_mib"]:.0f} | {c["temperature_min_c"]:.1f}–{c["temperature_max_c"]:.1f} / {n["temperature_min_c"]:.1f}–{n["temperature_max_c"]:.1f} |')
        lines+=['','| 模型 / 历史＋新增 / 起温 | CPU/NPU PSS MiB | CPU/NPU换入＋换出 KiB | CPU/NPU磁盘读＋写 KiB | CPU/NPU采样覆盖率 |',
                '| --- | --- | --- | --- | --- |']
        for row in rows:
            resources=row.get('direct_resources',{})
            if not resources.get('cpu') or not resources.get('npu'):continue
            c,n=resources['cpu']['medians'],resources['npu']['medians']
            key=f'{row["model"]} / {row["history"]}＋{row["new_tokens"]} / {row["band"]}'
            lines.append(f'| {key} | {fmt(c["pss_peak_mib"])} / {fmt(n["pss_peak_mib"])} | {(c["swap_in_bytes"]+c["swap_out_bytes"])/1024:.0f} / {(n["swap_in_bytes"]+n["swap_out_bytes"])/1024:.0f} | {(c["disk_read_bytes"]+c["disk_write_bytes"])/1024:.0f} / {(n["disk_read_bytes"]+n["disk_write_bytes"])/1024:.0f} | {c["covered_fraction"]:.0%} / {n["covered_fraction"]:.0%} |')
        lines+=['','每个指标是五轮各自统计量的中位数。采样只统计完全落入响应内的区间，边界、模型加载及KV历史准备不包含在零换页/零I/O结论中；温度范围也仅覆盖这些区间。MemAvailable含系统缓存及后台服务影响，不是独立模型内存。生产CPU网页在两端均常驻，未测电功率。源文件SHA、逐轮资源和缺失字段保存在同名紧凑证据。','']
        lines+=['## 距离CPU目标线的耗时预算','',
                '取已对齐的中温64新增场景。每轮将127个decode间隔的总时间拆为decoder API、head API和其余时间，再取五轮中位。API包括设备调用等待，不能等同纯算子核时间。最后两列是条件推算，不是实测加速：假设API时间完全不变时，把其余时间降为零的理论收益；以及保持head和其余时间不变时，decoder为了达到CPU速度的1.15倍需减少的耗时比例。','',
                '| 模型 / 历史 | 每token总 / decoder / head / 其余 ms | 追平CPU需削减 ms/token | 超过CPU15%需削减 ms/token | 只消除其余时间的理论提速上限 | 仅decoder需降时 |',
                '| --- | --- | ---: | ---: | ---: | ---: |']
        for row in rows:
            if row['band']!='medium' or row['new_tokens']!=64 or row['trajectory_aligned'] is not True or '配置搜索complete' not in row['cpu_note']:continue
            cpu_speed=row['cpu']['decode_tokens_per_second'];budgets=[]
            for run in row['npu']['runs']:
                runtime=run['result'];intervals=runtime['generated_tokens']-1
                assert intervals==127
                total=(runtime['total_seconds']-runtime['ttft_seconds'])/intervals
                decoder=runtime['decode_decoder_api_seconds']/intervals
                head=runtime['decode_head_api_seconds']/intervals
                other=total-decoder-head
                assert decoder>0 and head>0 and other>=-1e-6
                budgets.append({'total_seconds':total,'decoder_seconds':decoder,'head_seconds':head,
                                'other_seconds':other,'match_cpu_saving_seconds':total-1/cpu_speed,
                                'beat_cpu_15pct_saving_seconds':total-1/(1.15*cpu_speed),
                                'other_elimination_speedup':total/(decoder+head)-1,
                                'decoder_only_required_reduction':(total-1/(1.15*cpu_speed))/decoder})
            b={key:statistics.median(v[key] for v in budgets) for key in budgets[0]}
            row['latency_budget']={'medians':b,'runs':budgets,'target_cpu_multiplier':1.15,
                                  'scope':'conditional time budget; not a measured candidate or a feasibility guarantee'}
            times=' / '.join(f'{b[k]*1000:.2f}' for k in ('total_seconds','decoder_seconds','head_seconds','other_seconds'))
            lines.append(f'| {row["model"]} / {row["history"]} | {times} | {b["match_cpu_saving_seconds"]*1000:.2f} | {b["beat_cpu_15pct_saving_seconds"]*1000:.2f} | {b["other_elimination_speedup"]:.1%} | {b["decoder_only_required_reduction"]:.1%} |')
        lines+=['','以上预算不保证硬件或编译器能实现。质量门槛与首字、端到端时间仍独立验收；未对齐格子不加入此推算。','']
    suffix='finalist-aligned' if args.finalists else 'candidate'
    (HERE/(title+'.md')).write_text('\n'.join(lines))
    (HERE/f'evidence-attack-20261013/{suffix}-matrix.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
    (HERE/f'evidence-attack-20261013/{suffix}-quality-failures.json').write_text(json.dumps(
        {v['case']:v for v in failures.values()},indent=2,ensure_ascii=False)+'\n')
    print(title,len(rows))


if __name__=='__main__':main()
