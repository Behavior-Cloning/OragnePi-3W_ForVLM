"""Publish checked, paired full-depth results without promoting failed gates."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(p):
    return json.loads(p.read_text())


def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    here=Path(__file__).resolve().parent
    archive=here/'evidence-aligned-20261013'
    archive.mkdir(exist_ok=True)
    matrix=read(a.root/'matrix.json')
    states={s:read(a.root/f'state-{s}.json')['cases'] if (a.root/f'state-{s}.json').exists() else {} for s in ('cpu','npu')}
    rows=[];system=[];quality=[];evidence=[];counts={'cpu':0,'npu':0,'quality_failed':0}
    def keep(data,name):
        (archive/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    for c in matrix['cases']:
        case=c['case'];cpu_folder=a.root/(case+'-cpu');cpu_path=cpu_folder/'results.json'
        npup=a.root/(case+'-npu-performance.json');reference=states['npu'].get(case,{}).get('reference')
        cpu=read(cpu_path) if cpu_path.exists() else None
        npu=read(npup) if npup.exists() else None
        gatepath=Path(reference)/'full-validation-w8.json' if reference else None
        gate=read(gatepath) if gatepath and gatepath.exists() else None
        prompt=read(Path(c['folder'])/'prompt.json')
        assert len(prompt['token_ids'])==c['history']+c['new_tokens']
        assert digest(Path(c['folder'])/'prompt.i32')==c['prompt_sha256']
        expected_depth=32 if c['model']=='smol500' else 24
        item={'case':c,'cpu':None,'npu':None,'execution_gate':None}
        row={'case':case,'model':c['model'],'history':c['history'],'new_tokens':c['new_tokens']}
        if cpu and cpu.get('status')=='complete':
            measures=[r for r in cpu['runs'] if r['phase']=='measure']
            assert len(measures)==5 and cpu['steps']==128 and cpu['cached_prefix']==c['history']
            for r in measures:
                assert r['timings']['cache_n']==c['history'] and r['timings']['prompt_n']==c['new_tokens']
                assert r['timings']['predicted_n']==128
                # llama.cpp reports n-1 decode intervals, matching the NPU timer.
                assert abs(r['timings']['predicted_per_second']-127000/r['timings']['predicted_ms'])<1e-6
                assert r['response_start_temperature_mC']<=69000
            counts['cpu']+=1;row['cpu']=cpu['medians']
            item['cpu']={k:v for k,v in cpu.items() if k not in ('runs','nonstream_trajectory','prompt')}
            item['cpu']['results_sha256']=digest(cpu_path)
            item['cpu']['runs']=[{k:v for k,v in r.items() if k not in ('health_before','health_after','system_samples')} for r in cpu['runs']]
            item['cpu']['system_samples']=[{'repeat':r['repeat'],'phase':r['phase'],'samples':[
                {**{k:v for k,v in s.items() if k!='health'},'health':{k:s['health'][k] for k in ('model_id','backend','ready','busy','loading','maintenance','metrics')}}
                for s in r['system_samples']]} for r in cpu['runs']]
            samples=[s for r in measures for s in r['system_samples']]
            rss=max(s['process']['rss_kib'] for s in samples)/1024
            metrics=[s['health']['metrics'] for s in samples]
            system.append((case,'CPU',cpu['medians']['response_cpu_seconds'],rss,None,
                min(m['ram_available_mib'] for m in metrics),max(m['temperature_c'] for m in metrics),
                min(r['response_start_temperature_mC'] for r in measures)/1000,
                max(r['response_start_temperature_mC'] for r in measures)/1000,
                max(m['system_cpu_percent'] for m in metrics)))
        if gate:
            runtime=gate['runtime'];g=gate['execution_gate_v1']
            assert runtime['decoder_layers']==expected_depth and runtime['head_on_npu']
            assert runtime['generated_tokens']==128 and runtime['cached_prefix_tokens']==c['history']
            assert runtime['new_prompt_tokens']==c['new_tokens'] and runtime['capacity']==c['capacity']
            assert runtime['fixed_feedback'] and runtime['prefill_batch_size']==32
            assert (Path(reference)/'prompt.i32').read_bytes()==(Path(c['folder'])/'prompt.i32').read_bytes()
            assert runtime['feedback_tokens']==read(cpu_folder/'tokens.json')['tokens']
            row['gate']=g;item['execution_gate']={k:v for k,v in gate.items() if k not in ('constant_records','runtime','prompt')}
            item['execution_gate'].update(source_sha256=digest(gatepath),logits_sha256=digest(Path(reference)/'logits.f32'))
            item['diagnostic_runtime']=runtime
            quality.append((case,gate['relative_l2_all_logits'],gate['min_step_cosine'],gate['argmax_matches_fp32'],g['passed']))
            if not g['passed']:counts['quality_failed']+=1
        if npu and npu.get('status')=='complete':
            assert gate and gate['execution_gate_v1']['passed']
            measures=[r for r in npu['runs'] if r['phase']=='measure']
            assert len(measures)==5 and npu['config']['layers']==expected_depth
            assert all(r['result']['driver_after_release_bytes']==0 for r in npu['runs'])
            assert all(r['result']['generated_tokens']==128 for r in npu['runs'])
            counts['npu']+=1;row['npu']=npu['medians']['batch32']
            item['npu']=npu;item['npu']['results_sha256']=digest(npup)
            m=row['npu'];metrics=[s['health']['metrics'] for r in measures for s in r['system_samples']]
            system.append((case,'NPU',m['cpu_seconds'],m['max_rss_kib']/1024,m['driver_allocated_bytes']/2**20,
                min(x['ram_available_mib'] for x in metrics),max(x['temperature_c'] for x in metrics),
                min(r['result']['response_start_temperature_mC'] for r in measures)/1000,
                max(r['result']['response_start_temperature_mC'] for r in measures)/1000,
                max(x['system_cpu_percent'] for x in metrics)))
            row['npu_preparation_seconds']=statistics.median(r['result']['preparation_seconds'] for r in measures)
        row['status']={s:states[s].get(case,{}).get('status','pending') for s in states}
        rows.append(row);evidence.append(item);keep(item,case+'.json');keep(prompt,case+'-prompt.json')
    keep({'matrix':matrix,'rows':rows,'counts':counts},'summary.json')
    lines=['# 精确历史与新增 token：完整语言模型 CPU/NPU 对标','',
        '2026-10-13。500M全部32层、1B全部24层、最终Norm、完整输出头及采样。仅文本，视觉编码不在本轮范围内。',
        '',f'过渡记录：CPU {counts["cpu"]}/12，数值通过且完成五轮NPU {counts["npu"]}/12；NPU执行门槛失败 {counts["quality_failed"]} 项。用户随后增加温度场景，本矩阵停止扩跑，当前工作见[三温档全深度对标](三温档全深度对标.md)。未完成或未通过的格子不填合格性能。','',
        '已缓存历史精确512/1024 token，新增32/64/128 token，新增包含角色模板和assistant生成标记；容量768/1280。每格固定128次完整词表预测，两端输入及回灌CPU轨迹一致；同一模型/历史档的三个新增档共用相同历史。忽略EOS仅用于固定工作量，不是自然对话质量验收。','',
        'CPU：llama.cpp Q8_0，使用此前线程/绑核筛选较快的两A76（6、7核）/2线程重新测量；长历史所有配置尚未穷举。NPU：主体W8，敏感MLP down FP16、KV/激活主要INT16，softmax概率INT16；500M转置K、1B普通K、32-token批量图。两端权重编码不同，不能称为逐位相同INT8。','',
        '每格完整数值检查→诊断logits复现→预热→5次正式测量。执行门槛为全部logits L2≤5%、逐步cos≥0.99、argmax≥95%、全有限，参考使用实际导出权重；不等于相对原模型的独立留出质量验收。','',
        'CPU客户端TTFT包含本地HTTP/SSE；另保存首个可见文字时间，UTF-8缓冲可能延迟上报。NPU TTFT是运行器内首个预测，包含宿主转换、同步、输出头和采样，不含HTTP。CPU prompt_ms单独归档。decode两端均为首个预测之后的127步。','',
        '约68°C起测上限在历史准备后再次检查，保留KV并短暂等候散热，等待不计入增量响应。按用户要求控制大致温区，不要求温度完全相同。首次建立缓存、等待散热与加载均独立记录，初次历史准备仍可能明显升温；这不是连续30分钟稳定性试验。网页CPU服务保持在线空闲，CPU/NPU测速和部署串行。','',
        '## 配对性能（五次中位数）','',
        '| 模型 | 历史 | 新增 | CPU token/s | NPU token/s | NPU/CPU | CPU首token s | NPU首token s | CPU总响应 s | NPU总响应 s |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    def fmt(v):return '—' if v is None else f'{v:.3f}'
    for r in rows:
        cpu=r.get('cpu',{});npu=r.get('npu',{})
        ratio=npu.get('decode_tokens_per_second',0)/cpu['decode_tokens_per_second'] if npu and cpu else None
        lines.append('| '+' | '.join([r['model'],str(r['history']),str(r['new_tokens']),
            fmt(cpu.get('decode_tokens_per_second')),fmt(npu.get('decode_tokens_per_second')),fmt(ratio),
            fmt(cpu.get('ttft_seconds')),fmt(npu.get('ttft_seconds')),fmt(cpu.get('total_seconds')),fmt(npu.get('total_seconds'))])+' |')
    lines+=['','## 执行精度','', '| 场景 | logits L2 | 最低cos | argmax/128 | 门槛 |','| --- | ---: | ---: | ---: | --- |']
    for case,l2,cos,matched,passed in quality:
        lines.append(f'| {case} | {l2:.3%} | {cos:.6f} | {matched} | {"通过" if passed else "失败"} |')
    lines+=['','## 系统开销','',
        'CPU时间为增量响应消耗的CPU秒。CPU RSS是独立llama-server进程采样，NPU RSS为宿主进程峰值，NPU驱动分配另外列出；不能只比较两列RSS就断言NPU更省总内存。系统温度/CPU峰值及最低可用RAM覆盖准备与推理，1秒采样可能漏掉短峰值。', '',
        '| 场景 | 后端 | 响应CPU s | 进程RSS MiB | 驱动 MiB | 最低可用RAM MiB | 全流程最高°C | 响应起温°C | 系统CPU峰值% |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for case,backend,seconds,rss,driver,ram,temp,tmin,tmax,load in system:
        lines.append(f'| {case} | {backend} | {seconds:.3f} | {rss:.1f} | {fmt(driver)} | {ram:.0f} | {temp:.1f} | {tmin:.1f}–{tmax:.1f} | {load:.1f} |')
    lines+=['','## 首字差异与本轮改动','',
        '上一轮500M缓存1001、新增64，1B缓存1053、新增20；500M每20-token批量约0.388秒，1B约0.380秒。500M需3批再处理4个尾token，1B只需1批，旧1.411秒与0.409秒不能当作相同负载的模型大小对比。新矩阵替代该横向比较。','',
        '本轮将500M转置K与批量prefill结合，批量输出的新K仍按原布局输出，只将新增行散写进常驻转置缓存；没有每次转置整个历史。prefill/decode共享缓存前强制核对每个端口的形状、类型、量化尺度和布局。批量改32，三档新增输入均可整批处理。','',
        '小图探针仅供优化定位：1B转置K的两层收益约2%；四层融合探针未证明端到端收益，未放入本矩阵。正式表全部为完整模型，不能用探针替代。','',
        '最初未在增量响应前散热的CPU数据保留在本地aligned-20261013；1024历史准备后温度超过85°C，正式矩阵使用aligned-20261013r2。没有调低CPU频率，没有把前一批过热成绩混作本表基线。','',
        '独立留出质量、自然EOS多轮、CPU完整配置搜索、30分钟连续运行及网页NPU后端仍未验收。', '',
        '重建：`npu/.venv/bin/python npu/kv-cache/report_aligned_matrix.py --root npu/results/aligned-20261013r2`。小型证据见[evidence-aligned-20261013](evidence-aligned-20261013/summary.json)，大权重、编译图及完整logits仅本地保留。','']
    (here/'精确长度全深度对标.md').write_text('\n'.join(lines))
    print(json.dumps(counts))


if __name__=='__main__':main()
