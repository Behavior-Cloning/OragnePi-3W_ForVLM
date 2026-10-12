"""Archive long-history development evidence without weights or SDK binaries."""
import hashlib
import json
from pathlib import Path
import shutil
import statistics

here=Path(__file__).resolve().parent
results=here.parent/'results'
archive=here/'evidence-long-context-20261012'
archive.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def copy(p,name):
    shutil.copyfile(p,archive/name)
intern=results/'full-intern1b-sharded-v2-c128'
smol=results/'full-smol500-stable-v3-c128'
cases=[
    ('1B',510,intern,768,'history510-fixed','batch20'),
    ('1B',1053,intern,1280,'history1053-fixed','batch20，失败保留'),
    ('1B',1053,intern,1280,'history1053-scalar-fixed','scalar'),
    ('1B',1053,intern,1280,'history1053-scalar-repeat','scalar独立复测'),
    ('1B',1053,intern,1280,'history1053-scalar-cache-batch','scalar建立缓存，batch20追加'),
    ('1B',1053,intern,1280,'history1053-initflush-fixed','初始化flush诊断'),
    ('1B',1053,intern,1280,'history1053-ungrouped-fixed','关闭decode分组'),
    ('1B',1053,intern,1280,'history1053-poison','输出哨兵+flush诊断'),
    ('1B',1053,intern,1280,'history1053-preflush','逐批输出flush诊断'),
    ('1B',1053,intern,1280,'history1053-clear','逐批清零+flush诊断'),
    ('1B',1053,intern,1280,'history1053-scalarblock1-v2','仅2/3层逐token，其余batch20'),
    ('500M',501,smol,768,'history501-fixed','batch20'),
    ('500M',1001,smol,1280,'history1001-fixed','batch20'),
]
lines=['# 512 / 1024 档长历史 KV 开发验证','',
    '2026-10-12。按用户最新要求扩展真实多轮历史。下列是开发证据，不代表全部质量、速度或稳定性验收完成。','',
    '## 场景与边界','',
    '复原此前陪伴场景的模板和token，核对原始prompt SHA及板端llama.cpp分词。约512/1024档的实际缓存：1B为510/1053，500M为501/1001；本轮新增分别为20及63/64 token，均固定预测128步。容量768/1280包括已有历史、新输入及输出，并非把窗口大小当成有效历史。','',
    '保留完整语言层、完整NPU输出头及W8主体/INT16激活和KV/敏感FP16例外。初始化建立缓存与后续增量响应分开计时，后续响应不重新计算已有历史。首次准备缓存可能很慢，不能因其不计入增量首字而隐瞒。当前运行器不是网页跨HTTP请求的会话服务。','',
    '## 完整模型数值','',
    '相对实际导出常量的独立FP32计算，阈值仍为logits L2≤5%、逐步余弦≥0.99、argmax≥95%。固定反馈使用CPU完整生成轨迹；其预测拼接不是自然回答。','',
    '| 模型 | 有效缓存 | 容量 | 路径 | logits L2 | 最低余弦 | argmax | 门槛 |',
    '| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |']
evidence=[]
for model,h,root,c,case,label in cases:
    folder=root/f'w8-fused2-c{c}-t1'/('board-'+case)
    name=model+'-'+case
    p=folder/'full-validation-w8.json'
    if not p.exists():
        lines.append(f'| {model} | {h} | {c} | {label} | — | — | — | 待验证 |')
        continue
    v=read(p);r=read(folder/'tokens.json')
    assert r==v['runtime'] and r['cached_prefix_tokens']==h and r['capacity']==c
    assert r['generated_tokens']==128 and r['fixed_feedback']
    passed=v['execution_gate_v1']['passed']
    lines.append(f'| {model} | {h} | {c} | {label} | {v["relative_l2_all_logits"]:.3%} | {v["min_step_cosine"]:.8f} | {v["argmax_matches_fp32"]}/128 | '+('通过' if passed else '未通过')+' |')
    for f in ('full-validation-w8.json','tokens.json','prompt.json','runner-config.json'):
        copy(folder/f,name+'-'+f)
    evidence.append({'case':name,'logits_sha256':digest(folder/'logits.f32'),
        'prompt_sha256':digest(folder/'prompt.i32'),'feedback_sha256':digest(folder/'feedback.i32')})
lines+=['','1B的千token历史在batch20路径明显失败，scalar路径单次通过。正确缓存+仅新增消息batch20、初始化flush及逐批flush均出现过单次通过，重复失败：因此这些方案均未证明稳定，不能由一次通过宣布修复。关闭decode分组也未解决；失败路径不能进入合格速度对标。固定反馈、模型图相同，不是自然采样差异。','',
    '## 重复性能','',
    '只把通过数值门槛、性能模式关闭大logits输出、五轮完成的结果列为正式NPU重复数据。诊断运行的首字/速度不混入该表。起温≤68°C补充协议，宿主A76核6/7，包含调度/同步/转换/采样。温度、RAM、swap、CPU负载等逐秒数据保存在原始JSON。','',
    '| 场景 | decode token/s | 增量首字秒 | 响应秒 | 驱动MiB | 首次历史准备秒 |',
    '| --- | ---: | ---: | ---: | ---: | ---: |']
system_rows=[]
for model,h,suffix in [('intern1b',510,''),('intern1b',1053,''),('intern1b',1053,'-scalarblock1'),('smol500',501,''),('smol500',1001,'')]:
    p=results/f'{model}-history{h}{suffix}-performance.json'
    if not p.exists():continue
    d=read(p);copy(p,p.name)
    if d['status']!='complete':continue
    for label,m in d['medians'].items():
        assert len([r for r in d['runs'] if r['phase']=='measure' and r['label']==label])>=5
        runs=[r for r in d['runs'] if r['phase']=='measure' and r['label']==label]
        prep=statistics.median(r['result']['preparation_seconds'] for r in runs)
        lines.append(f'| {model}/{h}/{label} | {m["decode_tokens_per_second"]:.3f} | {m["ttft_seconds"]:.4f} | {m["total_seconds"]:.3f} | {m["driver_allocated_bytes"]/2**20:.2f} | {prep:.2f} |')
        samples=[x['health']['metrics'] for r in runs for x in r['system_samples']]
        assert all(r['result']['driver_after_release_bytes']==0 for r in runs)
        system_rows.append(f'| {model}/{h} | {min(r["result"]["temperature_before_mC"] for r in runs)/1000:.1f}–{max(r["result"]["temperature_before_mC"] for r in runs)/1000:.1f} | {max(s["temperature_c"] for s in samples):.1f} | {min(s["ram_available_mib"] for s in samples):.1f} | {min(s["swap_used_mib"] for s in samples):.1f}–{max(s["swap_used_mib"] for s in samples):.1f} | {m["cpu_seconds"]:.3f} | {m["max_rss_kib"]/1024:.2f} |')
init_perf=results/'intern1b-history1053-initflush-performance.json'
if init_perf.exists():
    d=read(init_perf);copy(init_perf,init_perf.name)
    if d['status']=='complete':
        m=d['medians']['initflush'];runs=[r for r in d['runs'] if r['phase']=='measure']
        assert len(runs)==5 and all(r['result']['driver_after_release_bytes']==0 for r in runs)
        prep=statistics.median(r['result']['preparation_seconds'] for r in runs)
        lines.append(f'| intern1b/1053/initflush | {m["decode_tokens_per_second"]:.3f} | {m["ttft_seconds"]:.4f} | {m["total_seconds"]:.3f} | {m["driver_allocated_bytes"]/2**20:.2f} | {prep:.2f} |')
lines+=['','| 场景 | 起温°C | 采样最高温°C | 最低可用RAM MiB | swap MiB | 本轮CPU秒 | RSS MiB |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',*system_rows,'',
    'RSS不包含全部驱动分配，不能当成模型总内存；swap是系统总占用而非本模型新增量。所有表内成功轮次退出后驱动分配回到0。历史准备仅在首次建立缓存发生，不能每轮重做再称真实缓存服务；此处每个重复独立建立同一状态以控制测量。','']
lines+=['','CPU对照为历史最快候选两个A76/2线程Q8，尚非长历史各参数穷举。初版1B/510的部分CPU轮次与NPU文件部署重叠，保留为诊断数据，不据此宣称正式公平加速比；后续无部署干扰的复测单独归档。','',
    '| CPU场景 | decode token/s中位 | 客户端首字秒 | 五轮 |',
    '| --- | ---: | ---: | --- |']
cpu_paths=[('1B/1053',results/'intern1b-history1053-cpu-reference/cpu-history1053-q8'),
    ('1B/510，部署干扰',results/'intern1b-history510-cpu-reference'),
    ('1B/510，无部署干扰复测',results/'intern1b-history510-cpu-reference-clean'),
    ('500M/1001',results/'smol500-history1001-cpu-reference-v2'),
    ('500M/501',results/'smol500-history501-cpu-reference')]
for name,folder in cpu_paths:
    p=folder/'results.json'
    if not p.exists():continue
    d=read(p);copy(p,folder.name+'-results.json')
    if d.get('status')!='complete':continue
    m=d['medians'];count=len([r for r in d['runs'] if r['phase']=='measure'])
    lines.append(f'| {name} | {m["decode_tokens_per_second"]:.3f} | {m["ttft_seconds"]:.4f} | {count} |')
    copy(folder/'tokens.json',folder.name+'-tokens.json')
lines+=['','500M中文输出发现当前llama.cpp流式接口在UTF-8字符未拼齐时不发送部分token ID，最终块也不补回，不能用收到的ID个数充作生成步数。采集器改为单独取同输入greedy非流式完整128-token轨迹，逐项核对流式已发token的位置和值及文本前缀；性能仍用流式响应。明确未从流中直接观察缺失ID，固定反馈来自完整非流式运行，不通过文本重新分词猜测。','',
    '本轮500M的CPU中文固定128步本身已有复读：501历史为“说到这里，我们可以说得详细一些。我们有一些灯光……”循环，1001历史在短句后重复句点。ignore-EOS固定长度负载不能当作自然对话质量；完整数值通过不代表这个模型已适合中文陪伴。自然EOS回答和多轮质量需另行检查。','',
    '新增主机级互斥锁覆盖部署上传、哈希、板端推理和结果取回，与板端flock共同避免实验相互干扰。锁排他与进程退出释放已验证。旧诊断样例保留实际温度；未满足起温条件的诊断速度不计入性能成绩。','',
    '## 两层探针与融合RMSNorm','',
    '1B/500M的1280容量两层图先以1024历史核对hidden及新增K/V，1B另核对20-token批量图。它们仅为形状/位置/算子检查，不替代全模型。原生RMSNorm属于实验选项，尚未证明完整模型提速或极端值安全，不纳入当前候选；保持原稳定Norm为默认。','']
lines+=['K布局实验把缓存保留为[KV头, head_dim, capacity]，仅转置本轮新增K，保留量化边界。1B两层输出与原图逐字节一致，但初筛约9.85ms对原图9.35ms，没有显示收益；500M为16.52ms对旧16.88ms，不同轮次/重复次数不能证明微小收益，暂不推广到完整运行器。','',
    'Attention FP16：直接修改permute输出精度因INT16→FP16类型不支持而编译失败；显式插入dtype_converter后编译成功，两层1024历史精度通过且50次输出一致，但13.56ms慢于原路径约9.35ms，不采用。两次编译及板端结果均保留。','',
    '## 1B批量路径旧hidden定位','',
    'revision 10新增可选逐块输出指纹、实际输入/输出抓取和隔离重放。audit1/audit2前160 token所有块一致，第一次差异出现在position160、第2/3层融合块hidden；新增K/V仍一致。该块部分调用的hidden与上一批次逐字节相同，输入已变化。抓取的错误hidden相对独立参考L2约119.8%，而同一实际输入隔离重放100次hidden L2约0.917%、输出一致；交替两份不同输入重放100次也一致。不能由固定输入重放一致独自排除旧结果。','',
    '取消prefill hidden共享、改SDK host-backed输出缓冲、重新绑定输出均未消除旧hidden。逐批写哨兵并flush的诊断运行没有旧hidden、完整误差约3.20%；仅flush及清零+flush均单次通过但复测失败，根因及最小可靠修复尚未成立。所有候选都是显式诊断开关，默认运行路径未自动替换。完整失败/成功单次记录与哈希见stale-output-diagnostics.json，性能表不混入这些带诊断开销的运行。','']
lines+=['局部回退候选`--prefill-scalar-block 1`：仅第2/3层融合块用原单token图，其他22层继续batch20。所有矩阵计算仍在NPU，CPU仅搬运hidden和追加KV，不能称CPU/NPU算子混合。VIP组中的网络不能单独执行（首版返回NETWORK_NOT_PREPARED），因此另建同权重单token实例并检查/共享KV契约。两次完整运行logits逐字节相同，误差3.228%、最低余弦0.998636、124/128 argmax通过；但正式A76双核绑定下重复诊断再次失败，所以仍不是已验证修复，五轮速度被拒绝。另将原批量图绑定单A76核6，仍失败，不能只归因线程迁移。','']
natural=intern/'w8-fused2-c1280-t1/board-history1053-natural-scalarblock1'
if (natural/'full-validation-w8.json').exists():
    v=read(natural/'full-validation-w8.json');r=v['runtime']
    copy(natural/'full-validation-w8.json','1B-history1053-natural-scalarblock1-validation.json')
    copy(natural/'tokens.json','1B-history1053-natural-scalarblock1-tokens.json')
    lines+=['## 自然续聊开发样例','',
        '1B在1053历史、20新增的“我想再听一点，接着说吧”上，局部逐token候选自由生成'+str(r['generated_tokens'])+'个token并自然EOS。该次执行门槛'+('通过' if v['execution_gate_v1']['passed'] else '失败')+'，L2 '+f'{v["relative_l2_all_logits"]:.3%}'+f'、argmax {v["argmax_matches_fp32"]}/{r["generated_tokens"]}。回答如下；尚无同场景自然CPU配对或独立人工质量验收，且运行间稳定性问题仍在，不能算可靠陪伴服务。','',
        '> '+v['npu_answer'].replace('\n','\n> '),'']
for name in ('intern1b-decode-c1280-t1','intern1b-prefill-c1280-t20','smol500-decode-c1280-t1','intern1b-native-rms-c256',
             'intern1b-ktranspose-c1280-t1','smol500-ktranspose-c1280-t1',
             'intern1b-attnfp16-c1280-t1','intern1b-attnfp16-convert-c1280-t1'):
    folder=results/name/'l0'
    for f in ('board-validation.json','conversion.json','probe-fixture.json'):
        if (folder/f).exists():copy(folder/f,name+'-'+f)
    if name=='intern1b-attnfp16-c1280-t1':
        copy(folder/'export.log',name+'-failed-export.log')
for name in ('intern1b-capture160-replay','intern1b-capture-alternating'):
    for f in ('board-validation.json','probe-fixture.json'):
        copy(results/name/'l2'/f,name+'-'+f)
for folder in ('intern1b-history1053-repeat-mismatch','intern1b-history1053-initflush-repeat-mismatch','intern1b-history1053-scalarblock1-mismatch','smol500-history1001-cpu-failed-stream'):
    for f in ('tokens.json','results.json'):
        p=results/folder/f
        if p.exists():copy(p,folder+'-'+f)
graphs=[]
for root in (intern,smol):
    for c in (768,1280):
        for suffix in ('t1','t20-vslice'):
            folder=root/f'w8-fused2-c{c}-{suffix}'
            if not (folder/'stack-conversion.json').exists():continue
            rows=read(folder/'stack-conversion.json')
            graphs.append({'source':root.name,'variant':folder.name,'files':[
                {'layer':r['layer'],'bytes':Path(r['path']).stat().st_size,'sha256':digest(Path(r['path']))} for r in rows]})
(archive/'provenance.json').write_text(json.dumps({'cases':evidence,'graphs':graphs},indent=2)+'\n')
lines+=['## 未完成','',
    '千token批量路径修复、两个模型两档完整自然多轮、独立留出质量、充分CPU调优、30分钟稳定性与网页NPU会话仍需验收。继续投入decode优化，同时报告大窗口静态attention开销；不以短窗口成绩替代真实历史档。','',
    '原批量图单A55核0对照同样失败（L2约84.6%）；单A76核6约88.0%。单核绑定不是修复。收尾网页ready/CPU、无评测进程、驱动当前分配0；TF可用13.67GiB、RAM可用约4575MiB、温度41.4°C，均为收尾快照。','',
    '复现与证据：`restore_history_fixture.py --context ... --case-index ...`、`run_cpu_reference.py`、`compile_fused_stack.py --capacity 768/1280`、`run_full_model.py --cached-prefix ...`、`validate_full.py --reference-weights exported`；本报告可用`report_long_context.py`重建。SDK、GGUF、NBG和大logits仅留本地，小型数值/性能记录及哈希进入Git。','']
(here/'长历史KV验证.md').write_text('\n'.join(lines))
print(here/'长历史KV验证.md')
