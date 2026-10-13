"""Archive complete-model medium-temperature candidate comparisons."""
import hashlib
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
RESULTS=HERE.parent/'results'
ARCHIVE=HERE/'evidence-medium-20261013'
ARCHIVE.mkdir(exist_ok=True)


def read(path):return json.loads(path.read_text())
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def keep(data,name):(ARCHIVE/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


lines=['# 中温完整模型优化对照','',
       '2026-10-13。以响应起始55–低于65°C为对照条件，计时前停止预热负载；响应中温度允许自然变化，实际范围单列。高温试验暂缓。','',
       '全部decoder层、最终Norm、完整词表head和采样都执行；历史512 token、新增分别32/64 token，KV保持，固定CPU轨迹128步。CPU Q8_0与NPU W8主体/敏感down FP16例外；本轮不改变权重量化。独立留出/自由对话质量和30分钟稳定性未验收。','',
       'CPU TTFT为本机HTTP/SSE首预测事件，NPU为进程内首预测，细小差异不宜直接解释为产品端体验提升。性能不含首次权重加载、初始历史KV准备及温控等待；这些另存原始记录。','']
lines+=['入档起温和终温直接读取sysfs。早期响应时间序列来自网页2秒缓存，可能把预热阶段约70°C的值带到响应开始后，不能当作推理真实峰值；后续采样器已改为直接读取，并在JSON标记来源。此问题不改变入档判断和五轮时延。','']
cases=[('1B输出头合并提交','intern1b',24,32,'intern1b-h512-n32-ab.json',
        RESULTS/'medium-head-20261013/intern1b-h512-n32-cpu/results.json',
        RESULTS/'full-intern1b-sharded-v2-c128/w8-fused2-c768-splitkv-i16mm/board-medium-head-intern1b-h512-n32',
        'serial','grouped'),
       ('500M输入批次32→64','smol500',32,64,'smol500-h512-n64-batch-ab.json',
        RESULTS/'thermal-aligned-20261013/smol500-h512-n64-medium-cpu/results.json',
        RESULTS/'full-smol500-stable-v3-c128/w8-fused2-c768-splitkv-transk/board-medium-batch64-smol500-h512-n64',
        'batch32','batch64')]
summary=[]
for title,model,layers,new,filename,cpu_path,reference,baseline,candidate in cases:
    lines+=['## '+title,'']
    gate=read(reference/'full-validation-w8.json')
    assert gate['execution_gate_v1']['passed']
    runtime=gate['runtime']
    assert runtime['decoder_layers']==layers and runtime['head_on_npu']
    assert runtime['cached_prefix_tokens']==512 and runtime['new_prompt_tokens']==new
    assert runtime['generated_tokens']==128 and runtime['fixed_feedback']
    keep(gate,model+'-validation.json');keep(read(reference/'runner-config.json'),model+'-runner.json')
    keep({'logits_sha256':digest(reference/'logits.f32'),'prompt_sha256':digest(reference/'prompt.i32'),
          'deployment_manifest':(reference/'deployed-sha256.txt').read_text()},model+'-reference-hashes.json')
    lines+=[f'完整logits执行门槛通过：L2 {gate["relative_l2_all_logits"]:.3%}、最低cos {gate["min_step_cosine"]:.6f}、argmax {gate["argmax_matches_fp32"]}/128。参考重建导出常量，不能代替原始模型量化质量验收。','']
    cpu=read(cpu_path);assert cpu['status']=='complete'
    assert len([r for r in cpu['runs'] if r['phase']=='measure'])==5
    assert cpu['prompt']['token_ids']==read(reference/'prompt.json')['token_ids']
    keep(cpu,model+'-cpu.json')
    path=RESULTS/'medium-head-20261013'/filename
    if not path.exists():lines+=['性能对照尚未开始。',''];continue
    data=read(path)
    keep(data,filename)
    if data['status']!='complete':
        lines+=[f'状态：{data["status"]}，已记录{len(data["runs"])}次诊断/预热/测量；尚不形成五轮结论。','']
        continue
    assert data['config']['layers']==layers and data['config']['temperature_band_mC']==[55000,64999]
    assert data['config']['fixed_feedback'] and data['config']['cached_prefix']==512
    stats={}
    lines+=['| 后端/候选 | decode token/s | TTFT s | 总响应 s | 起温°C | 响应采样°C | 宿主CPU s | 驱动MiB |',
            '| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |']
    med=cpu['medians'];cr=[r for r in cpu['runs'] if r['phase']=='measure']
    starts=[r['response_start_temperature_mC']/1000 for r in cr]
    assert all(55<=t<65 for t in starts)
    lines+=[f'| CPU Q8_0 | {med["decode_tokens_per_second"]:.3f} | {med["ttft_seconds"]:.3f} | {med["total_seconds"]:.3f} | {min(starts):.1f}–{max(starts):.1f} | 见原始记录 | {med["response_cpu_seconds"]:.3f} | — |']
    for label in (baseline,candidate):
        runs=[r for r in data['runs'] if r['phase']=='measure' and r['label']==label]
        assert len(runs)==5
        runtime=[r['result'] for r in runs]
        assert all(r['decoder_layers']==layers and r['head_on_npu'] and r['generated_tokens']==128 and r['driver_after_release_bytes']==0 for r in runtime)
        assert all(r['cached_prefix_tokens']==512 and r['new_prompt_tokens']==new for r in runtime)
        starts=[r['response_start_temperature_mC']/1000 for r in runtime]
        assert all(55<=t<65 for t in starts)
        temps=starts+[r['temperature_after_mC']/1000 for r in runtime]
        temps += [s['health']['metrics']['temperature_c'] for r in runs for s in r['system_samples']
                  if s.get('monotonic_seconds',0)>=r['result']['response_start_monotonic_seconds']]
        med=data['medians'][label];stats[label]=med
        lines+=[f'| NPU {label} | {med["decode_tokens_per_second"]:.3f} | {med["ttft_seconds"]:.3f} | {med["total_seconds"]:.3f} | {min(starts):.1f}–{max(starts):.1f} | {min(temps):.1f}–{max(temps):.1f} | {med["cpu_seconds"]:.3f} | {med["driver_allocated_bytes"]/2**20:.1f} |']
    b,c=stats[baseline],stats[candidate]
    speed=100*(c['decode_tokens_per_second']/b['decode_tokens_per_second']-1)
    ttft=100*(1-c['ttft_seconds']/b['ttft_seconds'])
    total=100*(1-c['total_seconds']/b['total_seconds'])
    paired=[]
    for repeat in range(5):
        pair={r['label']:r['result'] for r in data['runs'] if r['phase']=='measure' and r['repeat']==repeat}
        paired.append({'repeat':repeat,'decode_gain_percent':100*(pair[candidate]['decode_tokens_per_second']/pair[baseline]['decode_tokens_per_second']-1),
                       'ttft_reduction_percent':100*(1-pair[candidate]['ttft_seconds']/pair[baseline]['ttft_seconds'])})
    wins=sum(r['decode_gain_percent']>0 for r in paired)
    lines+=['',f'同轮中位变化：decode {speed:+.2f}%（{wins}/5配对更快）；首token延迟缩短 {ttft:+.2f}%；总响应缩短 {total:+.2f}%。单次微小差异不当作可靠提速。','']
    summary.append({'model':model,'baseline':b,'candidate':c,'cpu':cpu['medians'],'decode_gain_percent':speed,'ttft_reduction_percent':ttft,'total_reduction_percent':total,'pairs':paired})
    if model=='intern1b':
        diagnostics=[r for r in data['runs'] if r['phase']=='diagnostic']
        assert len({r['logits_sha256'] for r in diagnostics})==1
        lines+=['分组与串行诊断的全词表logits逐字节一致；仅改变提交方式，每片量化scale和全部词表计算保持。','']
    else:lines+=['批次扩大可能改变中间舍入，两候选分别通过完整执行门槛；不声称logits逐字节一致。','']
keep(summary,'summary.json')
lines+=['代码入口：`full_model.c`/`--group-head`、`compile_fused_stack.py --tokens 64`。默认正式矩阵仍使用batch32且不开head group，实验候选未经确认不会替换。','',
        '复现报告：`npu/.venv/bin/python npu/kv-cache/report_medium_optimization.py`。原始JSON和源文件哈希归档于`evidence-medium-20261013`。','']
(HERE/'中温优化对照.md').write_text('\n'.join(lines))
print(json.dumps({'complete_comparisons':len(summary),'summary':summary}),flush=True)
