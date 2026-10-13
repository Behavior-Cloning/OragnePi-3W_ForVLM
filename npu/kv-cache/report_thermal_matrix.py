"""Archive temperature-stratified full-depth measurements and reject mislabeling."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys


def read(path):return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    here=Path(__file__).resolve().parent;archive=here/'evidence-thermal-20261013';archive.mkdir(exist_ok=True)
    state=read(a.root/'state.json');rows=[];system=[];gates=[];counts={'cpu':0,'npu':0,'failed_cases':0}
    def keep(data,name):(archive/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    fixtures=read(a.root.parent/'aligned-20261013r2/matrix.json')
    entries={c['case']:state['cases'].get(c['case'],{'status':'pending','config':c,'bands':{},
        'reference':str(a.root/('not-yet-deployed-'+c['case']))}) for c in fixtures['cases']}
    for case,entry in entries.items():
        c=entry['config'];depth=32 if c['model']=='smol500' else 24
        if entry['status'] in ('failed','quality-failed'):counts['failed_cases']+=1
        reference=Path(entry['reference']);gp=reference/'full-validation-w8.json'
        gate=read(gp) if gp.exists() else None
        if gate:
            r=gate['runtime'];assert r['decoder_layers']==depth and r['head_on_npu'] and r['generated_tokens']==128
            assert r['cached_prefix_tokens']==c['history'] and r['new_prompt_tokens']==c['new_tokens']
            assert r['capacity']==c['capacity'] and r['fixed_feedback']
            gates.append({'case':case,'l2':gate['relative_l2_all_logits'],'cos':gate['min_step_cosine'],
                          'argmax':gate['argmax_matches_fp32'],'passed':gate['execution_gate_v1']['passed']})
            keep(gate,case+'-validation.json')
            keep(read(reference/'runner-config.json'),case+'-runner.json')
        prompt=read(Path(c['folder'])/'prompt.json');assert digest(Path(c['folder'])/'prompt.i32')==c['prompt_sha256']
        keep(prompt,case+'-prompt.json')
        for band,(low,high) in state['bands_mC'].items():
            row={'case':case,'model':c['model'],'history':c['history'],'new_tokens':c['new_tokens'],'band':band}
            for backend in ('cpu','npu'):
                path=Path(entry['bands'].get(band,{}).get('cpu_path',str(a.root/(case+'-'+band+'-cpu'))))/'results.json' if backend=='cpu' else Path(entry['bands'].get(band,{}).get('performance',str(a.root/(case+'-'+band+'-npu.json'))))
                if backend=='npu' and (not path.exists() or read(path).get('status')!='complete'):
                    completed=[f for f in a.root.glob(case+'-'+band+'-npu-attempt*.json') if read(f).get('status')=='complete']
                    if completed:path=sorted(completed,key=lambda f:int(f.stem.rsplit('attempt',1)[1]))[-1]
                if not path.exists():continue
                data=read(path)
                if data.get('status')!='complete':continue
                measures=[r for r in data['runs'] if r['phase']=='measure'];assert len(measures)==5
                if backend=='npu':
                    assert gate and gate['execution_gate_v1']['passed']
                    assert data['config']['layers']==depth
                    assert data['config']['temperature_band_mC']==[low,high] or (band=='medium' and data['config']['temperature_band_mC']==[55000,65000] and high==64999)
                    med=data['medians']['batch32'];runtime=[r['result'] for r in measures]
                    assert all(r['decoder_layers']==depth and r['head_on_npu'] and r['generated_tokens']==128 for r in runtime)
                    assert all(r['driver_after_release_bytes']==0 and r['feedback_tokens']==gate['runtime']['feedback_tokens'] for r in runtime)
                    assert data['config']['prompt_sha256']==c['prompt_sha256']
                    diagnostic=next(r for r in data['runs'] if r['phase']=='diagnostic')
                    assert diagnostic['logits_sha256']==digest(reference/'logits.f32')
                    cpu_s=med['cpu_seconds'];rss=med['max_rss_kib']/1024;driver=med['driver_allocated_bytes']/2**20
                    allsamples=[s for r in measures for s in r['system_samples']]
                    response_samples=[s for r in measures for s in r['system_samples']
                        if s.get('monotonic_seconds',0)>=r['result'].get('response_start_monotonic_seconds',float('inf'))]
                    freq=[s['health'].get('frequencies',{}).get('policy6',{}).get('scaling_cur_freq') for s in response_samples]
                    conditioning=[r['response_cooldown_seconds'] for r in runtime]
                else:
                    assert data['temperature_band_mC']==[low,high] or (band=='medium' and data['temperature_band_mC']==[55000,65000] and high==64999)
                    assert data['steps']==128 and data['cached_prefix']==c['history']
                    assert data['prompt']['token_ids']==prompt['token_ids']
                    for r in measures:
                        assert r['timings']['cache_n']==c['history'] and r['timings']['prompt_n']==c['new_tokens']
                        assert r['timings']['predicted_n']==128
                        assert abs(r['timings']['predicted_per_second']-127000/r['timings']['predicted_ms'])<1e-6
                        if gate:assert r['tokens']==gate['runtime']['feedback_tokens']
                    med=data['medians'];runtime=measures;cpu_s=med['response_cpu_seconds'];driver=None
                    allsamples=[s for r in measures for s in r['system_samples']]
                    response_samples=[s for r in measures for s in r['system_samples']
                        if s['elapsed']>=r.get('response_observation_start_seconds',float('inf'))]
                    freq=[s.get('frequencies',{}).get('policy6',{}).get('scaling_cur_freq') for s in response_samples]
                    rss=max(s['process']['rss_kib'] for s in allsamples)/1024
                    conditioning=[r['response_cooldown_seconds'] for r in runtime]
                starts=[r['response_start_temperature_mC']/1000 for r in runtime]
                assert all(low/1000<=t<=high/1000 for t in starts),'wrong temperature label'
                temps=[s['health']['metrics']['temperature_c'] for s in response_samples]
                # Start/end sensors are also actual samples, not an inferred peak.
                temps+=starts+[r['temperature_after_mC']/1000 for r in runtime]
                frequencies=[int(f) for f in freq if f is not None]
                stats={'case':case,'band':band,'backend':backend,'start_min_c':min(starts),'start_max_c':max(starts),
                    'response_sample_min_c':min(temps),'response_sample_max_c':max(temps),
                    'response_time_series_available':bool(response_samples),
                    'a76_min_sample_mhz':min(frequencies)/1000 if frequencies else None,
                    'a76_max_sample_mhz':max(frequencies)/1000 if frequencies else None,
                    'cpu_seconds':cpu_s,'process_rss_mib':rss,'driver_mib':driver,
                    'min_available_ram_mib':min(s['health']['metrics']['ram_available_mib'] for s in allsamples),
                    'swap_used_mib_range':[min(s['health']['metrics']['swap_used_mib'] for s in allsamples),max(s['health']['metrics']['swap_used_mib'] for s in allsamples)],
                    'all_phase_peak_system_cpu_percent':max(s['health']['metrics']['system_cpu_percent'] for s in allsamples),
                    'conditioning_seconds_median':statistics.median(conditioning),
                    'speed_min':min(r['decode_tokens_per_second'] if backend=='npu' else r['timings']['predicted_per_second'] for r in runtime),
                    'speed_max':max(r['decode_tokens_per_second'] if backend=='npu' else r['timings']['predicted_per_second'] for r in runtime)}
                if backend=='npu':
                    npu_freq=[s['health']['npu_frequency_hz']/1e6 for s in response_samples if 'npu_frequency_hz' in s['health']]
                    stats['npu_frequency_mhz_range']=[min(npu_freq),max(npu_freq)] if npu_freq else None
                system.append(stats);row[backend]=med;counts[backend]+=1
                # Preserve all measured samples but omit repeated webpage catalog.
                for run in data['runs']:
                    for sample in run.get('system_samples',[]):
                        sample['health']={k:v for k,v in sample['health'].items() if k in
                            ('model_id','backend','ready','busy','loading','maintenance','metrics','frequencies','npu_frequency_hz')}
                    for key in ('health_before','health_after'):
                        if key in run:run[key]={k:v for k,v in run[key].items() if k in ('model_id','backend','ready','busy','loading','maintenance','metrics')}
                keep({'source_sha256':digest(path),'report':data,'summary':stats},case+'-'+band+'-'+backend+'.json')
            rows.append(row)
    keep({'status':state['status'],'counts':counts,'bands_mC':state['bands_mC'],'rows':rows,'system':system,'gates':gates},'summary.json')
    keep(state,'state.json')
    attempts=[]
    for path in sorted(a.root.glob('*npu*.json')):
        data=read(path)
        attempts.append({'file':path.name,'sha256':digest(path),'status':data.get('status'),
                         'error':data.get('error'),'recorded_runs':len(data.get('runs',[]))})
    for path in sorted(a.root.glob('*cpu*/results.json')):
        data=read(path)
        attempts.append({'file':str(path.relative_to(a.root)),'sha256':digest(path),'status':data.get('status','incomplete'),
                         'recorded_runs':len(data.get('runs',[]))})
    keep(attempts,'attempts.json')
    lines=['# 三温档、精确token、完整深度CPU/NPU对标','',
        '2026-10-13。正式范围为SmolVLM-500M全部32层、InternVL3-1B全部24层，最终Norm、完整词表输出头及采样；仅语言部分，不测视觉编码。','',
        f'完成CPU {counts["cpu"]}/36、通过完整数值门槛及五轮测速的NPU {counts["npu"]}/36；失败或精度未通过的输入场景 {counts["failed_cases"]}。空格为未完成，不能作为零速度。','',
        '| 起始温档 | 实测温度范围 |','| --- | --- |','| 低温 | 40–50°C |','| 中温 | 55–低于65°C |','| 高温 | 65–80°C |','',
        '原高温设70–80°C，停预热后结温陡降导致两次无法进入/保持起温范围，尚无成功高温数据时改为65–80°C；旧失败记录保留，不重新贴标签。之前中温成功数据仍逐条检查实际温度<65°C。','',
        '温度在历史KV已准备好、增量响应开始时归档。预热使用独立CPU负载，正式计时前终止并等待所有预热线程退出；等待散热/预热单列，KV始终驻留。停止CPU负载后结温下降很快，当前预热停止阈值最多80°C，实际起测仍须落入标注温档，预热采样峰值另外保存。初版85°C阈值难以达到，失败日志保留。不修改系统温控和频率限制。推理期间允许自然升降温，因此是起温档位试验，不是恒温/稳态试验。温区不可达时记录失败，不强行填表。','',
        '每模型：缓存512/1024 token × 新增32/64/128 token × 三温档，共36格。新增包含模板标记，CPU/NPU每格输入二进制相同；固定CPU greedy轨迹回灌，128步全词表预测，忽略EOS仅用于固定工作量。每温档独立诊断完整logits与已通过的参考逐字节相同，预热1次、正式5次。','',
        'CPU为历史筛选较快的llama.cpp Q8_0、2线程/两A76；尚未穷举长历史所有配置。NPU主体W8、敏感down FP16、KV与多数激活INT16；32-token批量prefill，500M转置K，1B普通K。CPU客户端TTFT含本机HTTP/SSE、可能受UTF-8输出缓冲影响；NPU是运行器首预测、不含HTTP。decode双方均统计首token之后127步。','',
        '## 五轮中位数','',
        '| 模型 | 历史 | 新增 | 温档 | CPU token/s | NPU token/s | NPU/CPU | CPU TTFT s | NPU TTFT s | CPU总响应 s | NPU总响应 s |',
        '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    if state.get('status')=='blocked-storage':
        recovery=('用户已授权重启，但SSH认证成功后无法启动命令，普通重启和systemd正常重启信号请求均超时，是否执行未能确认；网页也超时。已请求现场断电再上电，等待恢复核查。[重启尝试记录](evidence-thermal-20261013/reboot-attempt.json)。'
                  if state.get('reboot_attempt',{}).get('status')=='awaiting-physical-power-cycle'
                  else '已停止后续排队和预热，存储恢复尚未验证。')
        lines[4:4]=['**当前因板端TF存储I/O阻塞暂停。** 内核报告SDMMC `wait dma hold bit clear timeout`，结果写入阻塞于`ext4_setattr`，多个系统进程也处于D状态；降至约35°C后仍未恢复。'+recovery+'不能据此直接判定TF卡损坏，供电、控制器调谐和温度相关性尚未定位。[只读故障证据](evidence-thermal-20261013/storage-fault.json)。高温CPU仅完成预热和1次，未计正式结果。','']
    names={'low':'低','medium':'中','high':'高'}
    def fmt(x):return '—' if x is None else f'{x:.3f}'
    for r in rows:
        cpu=r.get('cpu',{});npu=r.get('npu',{});ratio=npu['decode_tokens_per_second']/cpu['decode_tokens_per_second'] if cpu and npu else None
        lines.append('| '+' | '.join([r['model'],str(r['history']),str(r['new_tokens']),names[r['band']],
            fmt(cpu.get('decode_tokens_per_second')),fmt(npu.get('decode_tokens_per_second')),fmt(ratio),
            fmt(cpu.get('ttft_seconds')),fmt(npu.get('ttft_seconds')),fmt(cpu.get('total_seconds')),fmt(npu.get('total_seconds'))])+' |')
    lines+=['','## 实测温度、频率与开销','',
        '1秒采样可能漏峰；频率降低可能是负载/调频策略变化，不能仅据频率低就判定热降频。CPU RSS为独立server进程，NPU RSS为宿主进程，驱动分配另列，不能仅比RSS。响应温度含起/终点；首格低温CPU缺分段时间标记，仅能给起/终点范围，完整流程采样仍归档。','',
        '| 场景 | 温档/后端 | 起温°C | 响应采样°C | A76采样MHz | 响应CPU s | RSS MiB | 驱动MiB | 温控准备 s |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for s in system:
        freq='—' if s['a76_min_sample_mhz'] is None else f'{s["a76_min_sample_mhz"]:.0f}–{s["a76_max_sample_mhz"]:.0f}'
        lines.append(f'| {s["case"]} | {names[s["band"]]}/{s["backend"]} | {s["start_min_c"]:.1f}–{s["start_max_c"]:.1f} | {s["response_sample_min_c"]:.1f}–{s["response_sample_max_c"]:.1f} | {freq} | {s["cpu_seconds"]:.3f} | {s["process_rss_mib"]:.1f} | {fmt(s["driver_mib"])} | {s["conditioning_seconds_median"]:.1f} |')
    lines+=['','## 完整数值门槛','',
        '参考重建实际导出权重；logits L2≤5%、每步cos≥0.99、argmax≥95%、全有限。三温档再做完整logits复现。此门槛不替代原始权重量化质量和独立陪伴留出验收。','',
        '| 场景 | L2 | 最低cos | argmax/128 | 门槛 |','| --- | ---: | ---: | ---: | --- |']
    for g in gates:lines.append(f'| {g["case"]} | {g["l2"]:.3%} | {g["cos"]:.6f} | {g["argmax"]} | {"通过" if g["passed"] else "失败"} |')
    failures=[(case,e.get('error',e['status'])) for case,e in state['cases'].items() if e['status'] in ('failed','quality-failed')]
    if failures:
        lines+=['','## 未完成原因','']+[f'- {case}: {error}' for case,error in failures]
    lines+=['','## 曲线','',
        '![完整模型decode对比](thermal-decode.png)','',
        '![完整模型首token延迟](thermal-ttft.png)','',
        '未分温档的过渡结果见[精确长度报告](精确长度全深度对标.md)，不混入本表。旧500M64新增对1B20新增的TTFT差异主要来自工作量不同；本轮用相同新增token档位替代比较。','',
        '图、模型、完整logits留本地；代码和小型证据进入Git。[归档汇总](evidence-thermal-20261013/summary.json)。重建命令：`npu/.venv/bin/python npu/kv-cache/report_thermal_matrix.py --root npu/results/thermal-aligned-20261013`。','']
    (here/'三温档全深度对标.md').write_text('\n'.join(lines))
    subprocess.run([sys.executable,str(here/'plot_thermal_matrix.py')],check=True)
    print(json.dumps(counts),flush=True)


if __name__=='__main__':main()
