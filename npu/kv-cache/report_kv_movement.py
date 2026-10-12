"""Archive the KV movement experiment, including failed compiler candidates."""
import hashlib
import json
from pathlib import Path
import shutil
import statistics

here = Path(__file__).resolve().parent
results = here.parent / 'results'
archive = here / 'evidence-kv-movement-20261012'
archive.mkdir(exist_ok=True)


def read(path):
    return json.loads(path.read_text())


def keep(path, name):
    shutil.copyfile(path, archive / name)


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


lines = ['# KV 搬运与长历史 decode 优化', '',
    '2026-10-12。完整语言模型、对应CPU Q8_0、真实缓存历史约512/1024。新候选拆分历史/当前K和V，并把softmax概率输出设为INT16，显著改善decode；当前仍未达到超过CPU的目标。所有decoder层和完整输出头在NPU。', '',
    '权重仍为W8主体，沿用已披露的敏感FP16例外；KV主要INT16。概率输出由FP16改为DFP INT16、fl=15（步长2⁻¹⁵）。保留softmax公式和因果mask，但不能保证SDK选择相同内核或中间算术。不能把该变化说成无损或全INT8。', '',
    '## 硬件计数定位', '',
    '已接通 VIPLite NPD。板端输出 VNPD v2、128字节记录；SDK自带解析器按192字节读取而失败，新增解析器严格检查格式、记录长度和溢出位。', '',
    '下表是500M、两层、容量1280、有效历史1024的硬件忙碌计数，**不是整模型延迟占比**；可能未覆盖其他执行单元及调度等待。编译器UID在融合层之间重复，映射保留候选算子，不猜测具体层归属。', '',
    '| 两层图 | 合计忙碌周期 | concat周期 | concat占比 | 全连接周期 |',
    '| --- | ---: | ---: | ---: | ---: |']
for label, key, path in [('原图', 'baseline', results / 'smol500-decode-c1280-trace/npd-profile.json'),
                    ('拆分K', 'splitk', results / 'splitk-diagnostics/smol500-c1280-npd/profile.json'),
                    ('拆分KV+INT16概率', 'splitkv', results / 'splitk-diagnostics/smol500-splitkv-i16mm-npd/profile.json')]:
    if not path.exists():
        continue
    d = read(path); total = d['summed_busy_cycles']; concat = d['by_op']['concat']['cycles']
    lines.append(f'| {label} | {total:,} | {concat:,} | {concat / total:.2%} | {d["by_op"]["fullconnect"]["cycles"]:,} |')
    keep(path, 'profile-' + key + '.json')
baseline = results / 'smol500-decode-c1280-t1/l0/export_nbg_unify/network_binary.nb'
trace = results / 'smol500-decode-c1280-trace/l0/export_nbg_unify/network_binary.nb'
assert digest(baseline) == digest(trace), 'cannot apply trace mapping to a different binary'
(archive / 'baseline-trace-identity.json').write_text(json.dumps({'sha256': digest(trace),
    'baseline': str(baseline), 'trace': str(trace)}, indent=2) + '\n')
lines += ['', '拆分K利用 `Q·cat(Kpast,Knew)ᵀ = cat(Q·Kpastᵀ,Q·Knewᵀ)`，只拼接小得多的注意力分数。完整softmax、mask、RoPE位置和外部KV格式保留。新token点积用乘法和归约表达以绕开退化矩阵的编译问题，新增舍入经过独立FP32和完整模型检查。', '',
    '新方案进一步利用 `P·cat(Vpast,Vnew) = Ppast·Vpast + Pnew·Vnew`。无模型权重的最小复现定位到softmax→slice→AV的batch GEMM -8；把概率生产者输出设为INT16后消失。完整新V分支的广播乘仍报reshape -1，改成外积matmul后编译成功。原先失败候选保留，不能把它们说成性能结果。TP concat开关生成相同二进制，无收益；FP16常驻KV通过但更慢，不采用。', '',
    '## 全深度数值与初次诊断', '',
    '相同冻结CPU轨迹、固定128步预测，包含全部decoder层和完整NPU输出头。阈值保持logits L2≤5%、每步cos≥0.99、argmax≥95%、全部有限。参考重建实际导出常量；该门槛不覆盖相对原始模型的权重量化损失，也不是独立留出质量验收。固定轨迹的预测拼接不是自然自由回答。诊断速度不混入五轮正式成绩。', '',
    '| 模型/已缓存token/候选 | L2 | 最低cos | argmax | 门槛 | 诊断token/s | 增量首字秒 |',
    '| --- | ---: | ---: | ---: | --- | ---: | ---: |']
cases = [('smol500', 501, 768, 'history501-fixed'), ('smol500', 1001, 1280, 'history1001-fixed'),
         ('intern1b', 510, 768, 'history510-fixed'), ('intern1b', 1053, 1280, 'history1053-scalar-fixed')]
provenance = []
for variant, model, history, capacity, case in [(v, *c) for v in ('splitk', 'splitkv-i16mm') for c in cases]:
    source = results / ('full-smol500-stable-v3-c128' if model == 'smol500' else 'full-intern1b-sharded-v2-c128')
    folder = source / f'w8-fused2-c{capacity}-{variant}' / ('board-' + case)
    path = folder / 'full-validation-w8.json'
    if not path.exists():
        lines.append(f'| {model}/{history} | — | — | — | 待完成 | — | — |')
        continue
    d = read(path); r = read(folder / 'tokens.json')
    assert d['runtime'] == r and r['fixed_feedback'] and r['generated_tokens'] == 128
    assert r['cached_prefix_tokens'] == history and r['capacity'] == capacity
    gate = '通过' if d['execution_gate_v1']['passed'] else '失败'
    lines.append(f'| {model}/{history}/{variant} | {d["relative_l2_all_logits"]:.3%} | {d["min_step_cosine"]:.7f} | {d["argmax_matches_fp32"]}/128 | {gate} | {r["decode_tokens_per_second"]:.3f} | {r["ttft_seconds"]:.3f} |')
    for name in ('full-validation-w8.json', 'tokens.json', 'runner-config.json', 'deployed-sha256.txt'):
        keep(folder / name, f'{model}-{history}-{variant}-{name}')
    provenance.append({'case': f'{model}-{history}-{variant}', 'logits_sha256': digest(folder / 'logits.f32'),
                       'prompt_sha256': digest(folder / 'prompt.i32'), 'feedback_sha256': digest(folder / 'feedback.i32')})
lines += ['', '补查原GGUF Q8常量的FP32参考（包含重编码权重和执行引入的总差异），与上述导出常量门槛分开报告：', '',
    '| 模型/历史 | logits L2 | 最低cos | 对原常量FP32 argmax | 对实际CPU Q8固定轨迹预测 | KL nat/token | 开发轨迹困惑度比 |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for model, history, capacity, case in cases:
    source = results / ('full-smol500-stable-v3-c128' if model == 'smol500' else 'full-intern1b-sharded-v2-c128')
    path = source / f'w8-fused2-c{capacity}-splitkv-i16mm' / ('board-' + case) / 'full-validation.json'
    if not path.exists():continue
    d = read(path); r = d['runtime']; keep(path, f'{model}-{history}-original-weights-validation.json')
    cpu_matches = sum(a == b for a,b in zip(r['tokens'],r['feedback_tokens'],strict=True))
    ppl = d.get('fixed_trajectory_nll',{}).get('perplexity_ratio')
    ppl_text = '未计算' if ppl is None else f'{ppl:.5f}'
    lines.append(f'| {model}/{history} | {d["relative_l2_all_logits"]:.3%} | {d["min_step_cosine"]:.7f} | {d["argmax_matches_fp32"]}/128 | {cpu_matches}/128 | {d["mean_kl_nats"]:.6f} | {ppl_text} |')
lines += ['', '原常量参考显示额外权重编码差异。协议对该项采用独立留出KL和困惑度门槛，不与导出常量的执行门槛混用；上表是开发轨迹，不能声称整体质量验收已完成，也不为速度放宽门槛。']
lines += ['', '1B/1053采用逐token准备和追加输入，避开尚未解决的批量prefill旧hidden问题；这不是修复声明。其内存、首字和首次准备时间不能直接与加载双套图的批量方案比较。已有缓存不会在单次增量响应中重新计算。', '',
    '## 五轮重复与CPU对照', '',
    '沿用A76核6/7宿主、起温≤68°C补充协议。每个候选首先要求诊断完整logits逐字节复现，再预热和五轮无logits大输出的测量。驱动释放、预测轨迹、温度和逐秒系统状态均检查。下列CPU为此前无部署干扰的对应Q8结果，并非本轮重新穷举调优。', '',
    '| 场景 | 新NPU中位token/s | 五轮范围 | 旧NPU | 提升 | CPU Q8 | NPU/CPU |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
old = {('smol500', 501): (5.369263, 21.05210), ('smol500', 1001): (3.633722, 17.59973),
       ('intern1b', 510): (8.609991, 18.02623), ('intern1b', 1053): (None, 16.7543)}
formal = []
for model, history, _, _ in cases:
    path = results / f'{model}-splitkv-history{history}-performance.json'
    prior, cpu = old[model, history]; prior_text = '—' if prior is None else f'{prior:.3f}'
    if not path.exists():
        lines.append(f'| {model}/{history} | 待完成 | — | {prior_text} | — | {cpu:.3f} | — |')
        continue
    keep(path, path.name); d = read(path)
    if d['status'] != 'complete':
        lines.append(f'| {model}/{history} | {d["status"]} | — | {prior_text} | — | {cpu:.3f} | — |')
        continue
    measures = [x for x in d['runs'] if x['phase'] == 'measure']
    assert len(measures) == 5 and all(x['result']['driver_after_release_bytes'] == 0 for x in measures)
    assert all(x['result']['generated_tokens'] == 128 and x['result']['cached_prefix_tokens'] == history for x in measures)
    assert d['config']['fixed_feedback'] and d['config']['host_cpu_affinity'] == '6,7'
    med = d['medians']['splitkv']; speed = med['decode_tokens_per_second']
    speeds = [x['result']['decode_tokens_per_second'] for x in measures]
    gain = '—' if prior is None else f'{speed/prior-1:+.1%}'
    lines.append(f'| {model}/{history} | {speed:.3f} | {min(speeds):.3f}–{max(speeds):.3f} | {prior_text} | {gain} | {cpu:.3f} | {speed/cpu:.1%} |')
    formal.append((model, history, measures, med))
lines += ['', '旧正式NPU与CPU见[长历史KV验证](长历史KV验证.md)。1B/1053此前没有合格五轮，不以单次诊断替代。目标仍是逐模型/历史档超过最快合格CPU至少15%。', '',
    '| 场景 | 新增token | 增量TTFT秒 | CPU TTFT秒 | 总响应秒 | 一次性准备历史秒 | 图加载秒 |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
cpu_ttft = {('smol500',501):.795835, ('smol500',1001):1.029797, ('intern1b',510):.246879, ('intern1b',1053):.336721}
for model, history, measures, med in formal:
    prep = statistics.median(x['result']['preparation_seconds'] for x in measures)
    lines.append(f'| {model}/{history} | {measures[0]["result"]["new_prompt_tokens"]} | {med["ttft_seconds"]:.3f} | {cpu_ttft[model,history]:.3f} | {med["total_seconds"]:.3f} | {prep:.3f} | {med["load_seconds"]:.3f} |')
lines += ['', '增量TTFT不包含首次建立历史缓存。图加载受系统页缓存影响，不是统一断电冷启动。500M与1B新增输入长度不同，不能直接横比首字。', '',
    '| 场景 | decoder毫秒/token | 输出头毫秒/token | 驱动MiB | RSS MiB | 增量响应CPU秒 |',
    '| --- | ---: | ---: | ---: | ---: | ---: |']
for model, history, measures, med in formal:
    lines.append(f'| {model}/{history} | {med["decode_decoder_api_seconds"]/127*1000:.2f} | {med["decode_head_api_seconds"]/127*1000:.2f} | {med["driver_allocated_bytes"]/2**20:.2f} | {med["max_rss_kib"]/1024:.2f} | {med["cpu_seconds"]:.3f} |')
lines += ['', '驱动内存和进程RSS分开报告，不能用小RSS代替推理总占用。所有正式轮次驱动释放后为0。', '',
    '| 场景 | 起温范围°C | 采样最高温°C | 最低系统可用RAM MiB | 系统swap范围MiB | 采样系统CPU最高% |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for model, history, measures, med in formal:
    samples = [s['health']['metrics'] for x in measures for s in x['system_samples']]
    starts = [x['result']['temperature_before_mC']/1000 for x in measures]
    lines.append(f'| {model}/{history} | {min(starts):.1f}–{max(starts):.1f} | {max(x["temperature_c"] for x in samples):.1f} | {min(x["ram_available_mib"] for x in samples):.1f} | {min(x["swap_used_mib"] for x in samples):.1f}–{max(x["swap_used_mib"] for x in samples):.1f} | {max(x["system_cpu_percent"] for x in samples):.1f} |')
lines += ['', '系统1秒采样无法捕捉全部瞬时峰值；swap包含本轮开始前的占用，不能归因于本轮新增。网页CPU后端采样时保持ready且空闲。', '',
    '仅拆K的500M五轮作为消融保留：', '']
for history in (501, 1001):
    path = results / f'smol500-splitk-history{history}-performance.json'
    if path.exists():
        keep(path, path.name); d = read(path)
        if d['status'] == 'complete':
            lines.append(f'- 历史{history}：{d["medians"]["splitk"]["decode_tokens_per_second"]:.3f} token/s。')
lines += ['', '1B两个仅拆K五轮任务在获取锁前撤出队列，优先测更快候选；不是推理失败。']
lines += ['', '## 1B旧输出问题的缩小实验', '']
resident = results / 'splitk-diagnostics/intern1b-resident12/board-validation.json'
if resident.exists():
    d = read(resident); t = d['timing']; keep(resident, 'intern1b-resident12-validation.json')
    lines += [f'原捕获输入与另一个真实输入交替；主图加12份同图常驻，40轮合计520次执行。驱动分配{t["allocated_bytes"] / 2**20:.2f}MiB，主图变化{t["differing_repeats"]}轮，副本不同元素{t["resident_mismatched_elements"]}，退出分配{t["released_bytes"]}。这次未复现旧hidden，只说明该同图驻留条件不充分；不同图交互、更大驻留或特定执行轨迹仍未排除。', '']
lines += ['## 探针与失败记录', '', '| 探针 | API毫秒/两层 | 精度/重复门槛 | 范围 |', '| --- | ---: | --- | --- |']
for model in ('smol500', 'intern1b'):
    for variant, path in [('splitk', results / f'{model}-decode-c1280-splitk/l0/board-validation.json')] + [
            (v, results / f'splitk-diagnostics/{model}-{v}/board-validation.json') for v in
            ('kvfp16', 'splitkv-i16mm', 'probi16', 'splitkv-transk', 'prefill-splitkv-i16mm')]:
        if not path.exists():continue
        d = read(path); keep(path, f'{model}-{variant}-probe.json')
        lines.append(f'| {model}/{variant} | {1000*d["timing"]["mean_api_seconds"]:.3f} | {d["gate"]["passed"]} | 单个两层图，不是全模型 |')
lines += ['', 'probi16保留原KV concat，只改概率精度；说明收益来自格式与搬运两部分。prefill探针每次20token，1B选敏感2/3层，不能直接与单token/其他层比较。转置K已增加带严格形状检查的单token运行路径，批量路径仍明确拒绝；完整结果见下节。', '',
    '| 无权重最小复现 | 编译退出码 |', '| --- | ---: |']
for path in sorted((results / 'av-repro').glob('*/conversion.json')):
    d = read(path); name = path.parent.name; keep(path, f'av-repro-{name}.json')
    lines.append(f'| {name} | {d["exit_code"]} |')
lines += ['', '## 后续完整候选与自然生成', '',
    '下面单独列出新batch与转置K候选，不把单次诊断当正式五轮。batch即使单次通过，也须再次检查完整logits能否复现。自然生成用自己的token回灌并尊重EOS，与fixed轨迹分开。', '',
    '| 场景 | 执行门槛 | L2 | 最低cos | argmax | 诊断TTFT秒 | 诊断token/s |',
    '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
extra_cases = [('smol500','w8-fused2-c1280-splitkv-i16mm','history1001-batchsplit-fixed'),
               ('intern1b','w8-fused2-c1280-splitkv-i16mm','history1053-batchsplit-fixed'),
               ('smol500','w8-fused2-c1280-splitkv-transk','history1001-transk-fixed'),
               ('smol500','w8-fused2-c1280-splitkv-i16mm','history1001-natural'),
               ('intern1b','w8-fused2-c1280-splitkv-i16mm','history1053-natural')]
answers = []
for model, variant, case in extra_cases:
    source = results / ('full-smol500-stable-v3-c128' if model == 'smol500' else 'full-intern1b-sharded-v2-c128')
    folder = source / variant / ('board-' + case); path = folder / 'full-validation-w8.json'
    if not path.exists():continue
    d = read(path); r = d['runtime']; gate = '通过' if d['execution_gate_v1']['passed'] else '失败'
    assert r == read(folder / 'tokens.json')
    for name in ('full-validation-w8.json','runner-config.json','tokens.json','deployed-sha256.txt'):
        keep(folder / name,f'{model}-{case}-{name}')
    provenance.append({'case':f'{model}-{case}','logits_sha256':digest(folder/'logits.f32')})
    lines.append(f'| {model}/{case} | {gate} | {d["relative_l2_all_logits"]:.3%} | {d["min_step_cosine"]:.7f} | {d["argmax_matches_fp32"]}/{r["generated_tokens"]} | {r["ttft_seconds"]:.3f} | {r["decode_tokens_per_second"]:.3f} |')
    if d.get('npu_answer') is not None:
        assert not r.get('fixed_feedback')
        answers.append((model,r['generated_tokens'],d['npu_answer']))
for model, count, answer in answers:
    lines += ['', f'{model}，实际生成{count} token（上限128；EOS须看末尾特殊token）：', '', '```text', answer, '```']
lines += ['', '这些是开发场景的单次实际回答，不是独立留出、人工评分或跨请求多轮验收。模型本身的中文能力和NPU引入的额外退化需要CPU自然回答配对才能区分。']
lines += ['', '后续候选五轮结果（含失败；不覆盖上方已验证对照）：', '',
    '| 场景/候选 | 状态 | token/s中位 | TTFT秒 | 总响应秒 | 首次准备秒 | 驱动MiB |',
    '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
extra_system_rows = []
for model, history, label in [('smol500',1001,'batchsplit'),('intern1b',1053,'batchsplit'),('smol500',1001,'transk')]:
    path = results / f'{model}-{label}-history{history}-performance.json'
    if not path.exists():continue
    d = read(path); keep(path,path.name)
    if d['status'] != 'complete':
        lines.append(f'| {model}/{history}/{label} | {d["status"]} | — | — | — | — | — |')
        continue
    runs = [x for x in d['runs'] if x['phase'] == 'measure']
    assert len(runs) == 5 and all(x['result']['driver_after_release_bytes'] == 0 for x in runs)
    med = d['medians'][label]; prep = statistics.median(x['result']['preparation_seconds'] for x in runs)
    lines.append(f'| {model}/{history}/{label} | 五轮完成 | {med["decode_tokens_per_second"]:.3f} | {med["ttft_seconds"]:.3f} | {med["total_seconds"]:.3f} | {prep:.3f} | {med["driver_allocated_bytes"]/2**20:.2f} |')
    samples = [s['health']['metrics'] for x in runs for s in x['system_samples']]
    speeds = [x['result']['decode_tokens_per_second'] for x in runs]
    extra_system_rows.append(f'| {model}/{history}/{label} | {min(speeds):.3f}–{max(speeds):.3f} | {med["cpu_seconds"]:.3f} | {med["max_rss_kib"]/1024:.2f} | {max(x["temperature_c"] for x in samples):.1f} | {min(x["ram_available_mib"] for x in samples):.1f} |')
if extra_system_rows:
    lines += ['', '| 场景/候选 | 五轮token/s范围 | 响应CPU秒 | RSS MiB | 采样最高温°C | 最低可用RAM MiB |',
              '| --- | ---: | ---: | ---: | ---: | ---: |', *extra_system_rows]
lines += ['', '转置K目前只支持逐token输入；即使decode更快，也不能自动视为首字或端到端最佳方案。新batch重复通过的结论只覆盖本开发场景，旧batch故障根因及其他输入仍需验证。']
failure_rows = []
for path in sorted(results.glob('*decode-c*-split*/l0/conversion.json')):
    d = read(path)
    if d['exit_code'] == 0:continue
    name = path.parents[1].name; keep(path, name + '-conversion.json')
    errors = [line for line in (path.parent / 'export.log').read_text(errors='replace').splitlines()
              if any(s in line for s in ('CHECK STATUS', 'Failed to initialize Kernel', 'Fatal model generation error:'))]
    failure_rows.append({'name': name, 'exit_code': d['exit_code'], 'errors': errors[:8]})
(archive / 'compiler-failures.json').write_text(json.dumps(failure_rows, indent=2) + '\n')
(archive / 'full-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
lines += ['', '代码与小型证据进Git；模型、NBG、SDK、完整logits及捕获二进制仅本地保留。详情见[evidence-kv-movement-20261012](evidence-kv-movement-20261012)。', '',
    '下一步改善batch首字和1B敏感层/输出头，并处理相对原Q8参考的精度差距。独立留出质量、真实跨HTTP多轮、充分CPU调优及30分钟连续运行仍需完成。网页保持原CPU后端；本轮为命令行NPU研发，不是已部署NPU聊天服务。', '']
if len(formal) == 4:
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.hashsalt'] = 'opi-kv-movement-20261012'
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(4)
    new = [m['decode_tokens_per_second'] for _,_,_,m in formal]
    prior = [old[model,history][0] or float('nan') for model,history,_,_ in formal]
    cpu = [old[model,history][1] for model,history,_,_ in formal]
    low = [min(r['result']['decode_tokens_per_second'] for r in runs) for _,_,runs,_ in formal]
    high = [max(r['result']['decode_tokens_per_second'] for r in runs) for _,_,runs,_ in formal]
    fig, ax = plt.subplots(figsize=(10,5.6), layout='constrained')
    for offset, values, color, label in [(-.25,prior,'#a6afb8','Previous NPU W8'),
                                       (0,new,'#137c83','SplitKV NPU W8'),
                                       (.25,cpu,'#dd8b37','CPU Q8 baseline')]:
        bars = ax.bar(x+offset,values,width=.23,color=color,label=label,zorder=3)
        for b,v in zip(bars,values):
            if np.isfinite(v):ax.text(b.get_x()+b.get_width()/2,v+.35,f'{v:.2f}',ha='center',fontsize=9,
                                     bbox={'facecolor':'white','edgecolor':'none','pad':1})
    ax.errorbar(x,new,yerr=[np.array(new)-low,np.array(high)-new],fmt='none',color='#174c50',capsize=3,zorder=4)
    ax.axhline(6,color='#b53d4c',linestyle='--',linewidth=1.2,label='Speech reference: ~6 tok/s',zorder=2)
    transk = results/'smol500-transk-history1001-performance.json'
    if transk.exists() and read(transk)['status'] == 'complete':
        value = read(transk)['medians']['transk']['decode_tokens_per_second']
        ax.scatter([1],[value],marker='D',color='#764ba2',s=45,zorder=5,label='Transposed K (scalar input)')
        ax.annotate(f'{value:.2f}',(1,value),xytext=(-28,5),textcoords='offset points',color='#764ba2',fontsize=9)
    ax.text(3-.25,.4,'n/a',ha='center',color='#727b84',fontsize=9)
    ax.set_xticks(x,['500M / 501','500M / 1001','1B / 510','1B / 1053'])
    ax.set_xlabel('Model / actual cached history tokens')
    ax.set_ylabel('Decode tokens per second')
    ax.set_ylim(0,27)
    ax.grid(axis='y',alpha=.2,zorder=0)
    ax.spines[['top','right']].set_visible(False)
    ax.set_title('A733 full language inference: cached-history decode',loc='left',pad=14)
    ax.legend(loc='upper right',fontsize=8,ncols=2,frameon=False)
    fig.savefig(here/'kv-decode-comparison.png',dpi=160)
    svg = here/'kv-decode-comparison.svg'
    fig.savefig(svg,metadata={'Date':'2026-10-12'})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    at = lines.index('## 硬件计数定位')
    lines[at:at] = ['![完整语言链路解码对照](kv-decode-comparison.png)', '',
        '图中NPU为五轮中位，误差条是最小–最大，不是置信区间。6 token/s沿用此前粗略中文口语参考线，非统一实测标准；分词和输出语言会改变对应语速。主柱使用四档splitKV方案口径，新batch首字和转置K另列。', '']
(here / 'KV搬运优化.md').write_text('\n'.join(lines))
print(here / 'KV搬运优化.md')
