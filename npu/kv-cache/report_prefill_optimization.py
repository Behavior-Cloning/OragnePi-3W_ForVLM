"""Archive complete-model prefill experiments, including rejected precision cases."""
import hashlib
import json
from pathlib import Path
import shutil
import statistics

here=Path(__file__).resolve().parent;results=here.parent/'results'
archive=here/'evidence-prefill-20261012';archive.mkdir(exist_ok=True)
def read(path):return json.loads(path.read_text())
def copy(path,name=None):
    target=archive/(name or path.name);shutil.copyfile(path,target);return target
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
smol=results/'full-smol500-stable-v3-c128';intern=results/'full-intern1b-sharded-v2-c128'
lines=['# 批量增量 prefill 与 CPU 短历史对标','',
    '2026-10-12。继续完整500M/1B语言链路的W8优化。本轮主要缩短首字等待，未达到超过CPU的目标；1B另一个首轮陪伴样例仍未过数值门槛，不能把批量路径作为全面验收完成的候选。','',
    '## 实现与验证边界','',
    '新增20-token因果prefill图，与逐token decode共用真实KV缓冲。加载时逐项核对形状、类型和DFP scale；缓存不兼容立即拒绝。每块只输出新增K/V，按绝对位置写入；历史不重新计算。剩余不足20个token走原decode图。全部decoder和完整词表head仍在NPU，CPU处理embedding、位置/掩码、调度和greedy。权重仍以INT8为主，激活/KV以INT16为主，敏感FP16例外与上一轮一致。','',
    '计时入口可以先处理一段历史，再在同一进程、同一KV状态上追加新token。分别记录加载、历史准备、增量首字和decode；这是跨阶段KV复用测量入口，尚不是网页跨HTTP请求的NPU会话服务。加载有文件缓存，不称断电冷启动。','',
    '原批量图的V外部输出全零，而hidden未随V修复改变。把V的批量permute改为等价的逐KV头slice/reshape/concat后恢复；未改V权重或放宽阈值。两层FP32图解释器与独立PyTorch因果计算先核对形状/布局，上板后核对全部hidden/K/V，随后验证全部语言层和输出头。不能从两层时间外推完整速度。','',
    'runtime revision 9只增量开启新增历史掩码列，不重复写完整掩码；分组执行时只回读最后的hidden及各层KV，诊断trace仍保留逐层读取。1B固定128步在revision 8/9之间全部logits逐字节一致。','',
    '## 完整链路数值结果','',
    '阈值未变：相对实际导出常量的FP32参考，完整logits L2≤5%、逐步余弦≥0.99、argmax≥95%。批量与decode的全部导出矩阵/bias/Norm常量逐项相同才允许使用同一参考。以下均为开发集执行检查，独立留出、相对原始Q8的量化质量和标准实现logits交叉验证仍未完成。','',
    '| 模型/容量 | 输入与方式 | logits L2 | 最低余弦 | argmax | 门槛 |',
    '| --- | --- | ---: | ---: | ---: | --- |']
cases=[
    ('500M/128',smol,'w8-fused2-gqa-mlp-h1','scalar128-fixed-r9','30输入、90步，逐token'),
    ('500M/128',smol,'w8-fused2-gqa-mlp-h1','batch128-fixed-r9','30输入、90步，batch20'),
    ('500M/256',smol,'w8-fused2-c256-t1','companion-batch','自由陪伴，96步上限'),
    ('500M/256',smol,'w8-fused2-c256-t1','companion-fixed-batch','30输入、90步，batch20'),
    ('1B/256',intern,'w8-fused2-c256-t1','companion-scalar','首轮陪伴，逐token'),
    ('1B/256',intern,'w8-fused2-c256-t1','companion-batch','首轮陪伴，batch20'),
    ('1B/256',intern,'w8-fused2-c256-t1','companion-batch-bos1','首token单算，其余batch20'),
    ('1B/256',intern,'w8-fused2-c256-t1','history104-batch','中文续聊，自由EOS'),
    ('1B/256',intern,'w8-fused2-c256-t1','history104-fixed-scalar-r9','84缓存+20新增、128步，逐token'),
    ('1B/256',intern,'w8-fused2-c256-t1','history104-fixed-batch-r9','84缓存+20新增、128步，batch20'),
]
answers={};evidence=[]
for label,root,variant,case,description in cases:
    folder=root/variant/('board-'+case);path=folder/'full-validation-w8.json';d=read(path)
    name=('smol500' if root==smol else 'intern1b')+'-'+case
    copy(path,name+'-validation.json');copy(folder/'tokens.json',name+'-tokens.json')
    if (folder/'runner-config.json').exists():copy(folder/'runner-config.json',name+'-config.json')
    evidence.append({'label':name,'logits_sha256':digest(folder/'logits.f32'),
        'prompt_sha256':digest(folder/'prompt.i32'),'feedback_sha256':digest(folder/'feedback.i32') if (folder/'feedback.i32').exists() else None})
    lines.append(f'| {label} | {description} | {100*d["relative_l2_all_logits"]:.3f}% | {d["min_step_cosine"]:.8f} | {d["argmax_matches_fp32"]}/{d["generated_token_count"]} | {"通过" if d["execution_gate_v1"]["passed"] else "未通过"} |')
    answers[name]=d
equivalence=[]
for label in ('scalar','batch'):
    folder=intern/'w8-fused2-c256-t1'
    old=folder/f'board-history104-fixed-{label}'/'logits.f32';new=folder/f'board-history104-fixed-{label}-r9'/'logits.f32'
    assert digest(old)==digest(new)
    equivalence.append({'variant':label,'r8_sha256':digest(old),'r9_sha256':digest(new),'byte_identical':True})
(archive/'output-sha256.json').write_text(json.dumps({'cases':evidence,'host_r8_r9':equivalence},indent=2)+'\n')
copy(results/'prefill-guards.json');assert read(results/'prefill-guards.json')['passed']
lines += ['', '1B首轮陪伴batch20最低余弦0.989989，虽64/64 argmax一致，仍按未通过记录；只把第一个token改回单算也未通过。未用四舍五入把失败变通过。中文续聊场景通过，不代表这个首轮失败已经解决。','',
    '## 五轮交替性能','',
    '先诊断完整logits一致性，再预热一次，交替测量五轮。性能模式不输出全词表logits；固定反馈仍执行全部层、KV写入、全词表预测和采样。每个变体的输入轨迹完全相同，预测可不同；固定反馈的预测拼接不是自然回答。起温≤68°C补充协议，频率/温度/网页空闲状态与系统内存逐秒记录；可能漏掉采样间隙的短干扰。','']
experiments=[('500M / 128容量 / 30输入+90预测','smol500-prefill-c128-performance.json'),
             ('500M / 256容量 / 同一30输入+90预测','smol500-prefill-fixed90-performance.json'),
             ('1B / 256容量 / 84缓存+20新增+128预测 / A76宿主','intern1b-prefill-history104-a76-performance.json'),
             ('1B / 同输入 / 默认宿主调度对照','intern1b-prefill-history104-performance.json')]
reports={}
for title,filename in experiments:
    d=read(results/filename);assert d['status']=='complete';copy(results/filename);reports[filename]=d
    lines += [f'### {title}','',
        '| 路径 | decode token/s 中位（范围） | 首字秒 | 本轮响应秒 | 驱动MiB | CPU秒 | 加载秒 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    groups={}
    for label,med in d['medians'].items():
        rows=[r for r in d['runs'] if r['phase']=='measure' and r['label']==label];assert len(rows)==5
        groups[label]=rows;speeds=[r['result']['decode_tokens_per_second'] for r in rows]
        assert all(r['result']['driver_after_release_bytes']==0 for r in rows)
        lines.append(f'| {label} | {med["decode_tokens_per_second"]:.3f}（{min(speeds):.3f}–{max(speeds):.3f}） | {med["ttft_seconds"]:.4f} | {med["total_seconds"]:.3f} | {med["driver_allocated_bytes"]/1048576:.2f} | {med["cpu_seconds"]:.3f} | {med["load_seconds"]:.3f} |')
    old,new=d['medians']['scalar'],d['medians']['batch20']
    wins=sum(b['result']['ttft_seconds']<a['result']['ttft_seconds'] for a,b in zip(groups['scalar'],groups['batch20']))
    lines += ['',f'首字缩短{100*(1-new["ttft_seconds"]/old["ttft_seconds"]):.1f}%，{wins}/5轮更快。宿主CPU亲和性：{d["config"].get("host_cpu_affinity") or "系统默认"}。加载和历史准备不计入首字/本轮响应。','',
        '| 路径 | 起温°C | 采样最高温°C | 推理后RSS MiB | 最低可用RAM MiB | swap MiB范围 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for label,rows in groups.items():
        snapshots=[s['health']['metrics'] for r in rows for s in r['system_samples']]
        temps=[r['result']['temperature_before_mC']/1000 for r in rows]
        rss=[r['result']['rss_after_inference_kib']/1024 for r in rows];swap=[s['swap_used_mib'] for s in snapshots]
        lines.append(f'| {label} | {min(temps):.1f}–{max(temps):.1f} | {max(s["temperature_c"] for s in snapshots):.1f} | {min(rss):.2f}–{max(rss):.2f} | {min(s["ram_available_mib"] for s in snapshots):.1f} | {min(swap):.1f}–{max(swap):.1f} |')
    lines.append('')
cpu_path=results/'intern1b-history104-cpu-reference/cpu-history104-q8-v2/results.json';cpu=read(cpu_path);assert cpu['status']=='complete'
copy(cpu_path,'intern1b-cpu-history104.json')
npu=reports['intern1b-prefill-history104-a76-performance.json']['medians']['batch20'];cm=cpu['medians']
lines += ['## 与 CPU Q8 对照','',
    '复原10月10日的实际中文对话、模板和token，先核对旧prompt SHA256及原始token，再由板端llama.cpp独立核对分词。1B新增20token，缓存84token，双方预测128步。CPU用这次实际greedy产生的128个token；NPU逐步回送同一序列，因此后续每步看到同样的文本历史。CPU忽略EOS以保证长度，另外展示NPU自然EOS回答。','',
    '| 实现 | decode token/s中位 | 首字秒中位 |', '| --- | ---: | ---: |',
    f'| CPU llama.cpp Q8_0、2个A76、2线程 | {cm["decode_tokens_per_second"]:.3f} | {cm["ttft_seconds"]:.4f} |',
    f'| NPU W8主体、A76宿主、batch20 | {npu["decode_tokens_per_second"]:.3f} | {npu["ttft_seconds"]:.4f} |','',
    'CPU五轮复测了历史配置，未宣称已完成所有线程/attention/KV参数的最优搜索。CPU缓存容量1024，NPU静态容量256，双方有效历史/输入/预测数一致、无截断；NPU256足够容纳231位置。CPU首字为本机HTTP客户端测量，NPU为C运行器内计时，不含HTTP开销。即使如此，NPU仍未追上CPU；不能用不同计时边界掩盖差距。','',
    'NPU decoder图没有因batch prefill而变快；默认调度组decode数字存在宿主执行波动，不能把差异全归功于prefill。另做A76亲和性对照以减少这项干扰。NPU API时间（decoder/head）保存在原始JSON。','',
    '500M的本轮固定轨迹是30输入、90预测；历史CPU短续聊实际为125缓存+64新增，128预测需至少316位置，当前256版本容不下，未冒充已完成500M同场景CPU对照。500M较大容量明显增加静态attention开销，所以保留128短容量版本；不能把128与256的decode速度差异归给batch优化。','',
    '双图常驻增加驱动内存，表内RSS不等于整个推理占用，不能忽略驱动分配或简单与之相加。全部运行退出后驱动分配回到0。','',
    '## 实际生成','', '### 1B中文续聊（109 token自然EOS）','',answers['intern1b-history104-batch']['npu_answer'],'',
    '### 500M英语陪伴（96-token上限，未自然EOS）','',answers['smol500-companion-batch']['npu_answer'],'',
    '## 失败证据与复现','',
    'V归零、FP16布局编译失败及未过门槛的完整样例均保留。新增板端边界检查：负缓存长度、把整个prompt当已缓存、非法batch大小、实际图与声明batch不符、非法单算前缀均拒绝。FP32图核对只证明布局语义，不证明NPU浮点/整数执行无误。','']
diagnostic=['## decode分块定位与CPU线程初筛','',
    '为决定下一步加速投入，关闭组提交作单次逐块诊断，分别统计127/89次decode；预填充不计入块时间。诊断吞吐不当作最终性能成绩，API墙钟也不是纯硬件内核时间。两个模型诊断的全部logits均与对应已通过门槛的分组路径逐字节一致。','',
    '| 模型 | 普通块每块ms范围（每块2层） | 敏感块ms | 全部decoder API ms/步 | head API ms/步 |',
    '| --- | ---: | ---: | ---: | ---: |']
for label,root,variant,case,previous,sensitive in [
        ('1B/256',intern,'w8-fused2-c256-t1','history104-decode-profile','history104-fixed-batch-r9',[1]),
        ('500M/128',smol,'w8-fused2-gqa-mlp-h1','scalar128-decode-profile','scalar128-fixed-r9',[1,15])]:
    folder=root/variant/('board-'+case);prior=root/variant/('board-'+previous)
    d=read(folder/'tokens.json');assert digest(folder/'logits.f32')==digest(prior/'logits.f32')
    times=[t/(d['generated_tokens']-1)*1000 for t in d['decode_block_api_seconds']]
    normal=[t for i,t in enumerate(times) if i not in sensitive]
    diagnostic.append(f'| {label} | {min(normal):.3f}–{max(normal):.3f} | '+', '.join(f'{times[i]:.3f}' for i in sensitive)+f' | {sum(times):.3f} | {d["decode_head_api_seconds"]/(d["generated_tokens"]-1)*1000:.3f} |')
    copy(folder/'tokens.json',label.replace('/','-')+'-decode-profile.json')
    (archive/(label.replace('/','-')+'-decode-profile-equivalence.json')).write_text(json.dumps({
        'diagnostic_sha256':digest(folder/'logits.f32'),'validated_grouped_sha256':digest(prior/'logits.f32'),
        'byte_identical':True},indent=2)+'\n')
diagnostic+=['','1B普通22层累计约51ms，敏感2层块约10ms，输出头约18ms。只把敏感块降到普通块水平，大约只省5.7ms，仍远不足以达到CPU约49ms/token；下步应同时集中普通decoder和head，不能继续只优化prefill或只盯敏感层。分组版decoder API略低于这里的逐块提交总和。','',
    'CPU追加九种线程/绑核初筛，每配置一次预热、一次128步测量，ctx=256，greedy Q8。它是筛选而非九配置都完成五轮质量验收；候选中的两个A76仍最快，其ctx=1024增量场景已在上面完成五轮。','',
    '| CPU线程 | 可用核 | decode token/s（初筛单次） |','| ---: | --- | ---: |']
screen_path=results/'intern1b-cpu-history104-thread-screen.json';screen=read(screen_path);copy(screen_path)
assert len(screen['candidates'])==9
for candidate in screen['candidates']:
    diagnostic.append(f'| {candidate["threads"]} | {candidate["affinity"]} | {candidate["median_decode_tps"]:.3f} |')
diagnostic+=['','所有候选都使用现有优化llama.cpp及同一Q8文件，未限制CPU为慢FP32基线。attention/KV参数和其他部署实现仍未穷举，不能称已证明超过一切CPU实现。','']
idx=lines.index('## 实际生成');lines[idx:idx]=diagnostic
for name in ('intern1b-prefill-c256-t20','intern1b-prefill-c256-t20-voutfp16','intern1b-prefill-c256-t20-vpc',
             'intern1b-prefill-c256-t20-vlayoutfp16','intern1b-prefill-c256-t20-vslice',
             'intern1b-decode-c256-t1','smol500-prefill-c256-t20-vslice'):
    folder=results/name/'l0'
    for file in ('conversion.json','board-validation.json','probe-fixture.json'):
        if (folder/file).exists():copy(folder/file,name+'-'+file)
    if name.endswith('vlayoutfp16'):copy(folder/'export.log',name+'-export.log')
for root,variant in ((smol,'w8-fused2-c128-t20-vslice'),(smol,'w8-fused2-c256-t20-vslice'),
                     (smol,'w8-fused2-c256-t1'),(intern,'w8-fused2-c256-t20-vslice'),(intern,'w8-fused2-c256-t1')):
    rows=[]
    for record in read(root/variant/'stack-conversion.json'):
        folder=root/variant/f'l{record["layer"]}';binary=folder/'export_nbg_unify/network_binary.nb'
        rows.append({'layer':record['layer'],'nbg_sha256':digest(binary),'nbg_bytes':binary.stat().st_size,
                     'conversion':read(folder/'conversion.json') if (folder/'conversion.json').exists() else None})
    (archive/(("smol500" if root==smol else "intern1b")+'-'+variant+'-compilation.json')).write_text(json.dumps(rows,indent=2)+'\n')
lines += ['主要入口：`compile_fused_stack.py --compact-gqa --capacity 256 --tokens 20 --value-layout-sliced`；运行：`run_full_model.py --prefill-variant <variant> --cached-prefix 84 --feedback-from <CPU tokens.json>`；交替测速：`benchmark_variants.py --prefill batch20 20 --cpu-affinity 6,7 --fixed-feedback ...`。','',
    '离线重建：`python3 npu/kv-cache/report_prefill_optimization.py`。源码、报告、完整数值摘要、每轮性能/系统采样及哈希进Git；GGUF/SDK/NBG/大logits保留本地忽略目录。','',
    '下一步重点仍是decode：定位剩余敏感FP16 down和其他decoder算子的实际时间，减少静态空槽attention，改善容量分档；同时修复1B首轮batch质量边界。长历史512/2048、独立留出、真实网页NPU会话、充分优化CPU搜索和30分钟稳定性尚未验收，不宣称整个计划已完成。']
(here/'批量Prefill与CPU对标.md').write_text('\n'.join(lines).rstrip()+'\n')
print(here/'批量Prefill与CPU对标.md')
