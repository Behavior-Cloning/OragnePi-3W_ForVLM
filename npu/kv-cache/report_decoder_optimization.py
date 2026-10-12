"""Archive graph-fusion/GQA experiments without promoting diagnostic speed."""
import hashlib
import json
from pathlib import Path
import shutil

here = Path(__file__).resolve().parent
results = here.parent / 'results'
archive = here / 'evidence-decoder-optimization-20261012'
archive.mkdir(exist_ok=True)
models = [('500M', 'full-smol500-stable-v3-c128', 'w8-head-wide', 'companion'),
          ('1B', 'full-intern1b-sharded-v2-c128', 'w8-balanced-headwide', 'companion-v2')]
lines = ['# Decoder 融合与 GQA 布局优化', '', '2026-10-12。完整语言模型、W8为主的相同混合精度、容量128、增量KV；CPU基准仍为对应Q8。', '',
    '这是继输出头和批量提交优化后的新实验。所有速度先区分诊断运行与受控重复测速，未完成的部分不算验收通过。', '',
    '## 实现', '',
    '- 将相邻两个已校准decoder图合为一个图，移除中间hidden输入/输出边界，保留原权重与每个张量的量化参数。500M为16个双层图，1B为12个双层图，仍执行全部语言层和输出头。',
    '- 每层仍保留独立KV、mask、RoPE端口，不假定各层KV量化scale相同。ACUITY会按节点编号重排接口，脚本从导出的C代码核对真实顺序，运行器按K分组、V分组的接口绑定。',
    '- GQA版本把Q整理为`[batch, KV头数, 每组Q头数, head_dim]`，直接与未展开的K/V相乘，移除重复K/V用的两组Tile/Reshape。500M去掉3倍展开，1B去掉7倍展开。权重位宽、KV逻辑、mask和RoPE不变；需要重新验证数值和实际性能。',
    '- 原逐层图、批量提交路径继续保留。融合后不提供虚构的逐层耗时，记录块耗时；逐层hidden trace请求会明确拒绝。', '',
    '另有等价贪心采样优化：每片输出头在INT16中找最大值，只转换每片最大值后按真实scale比较；并列时仍选最小token编号。对不支持的输出类型回退浮点扫描。诊断模式同时执行完整浮点argmax逐步核对；性能模式省去整词表转换。它仅适用于本轮greedy生成，不是top-p/temperature采样实现。板上NEON和主机标量测试覆盖负数、极值、向量尾部及并列最大值。','',
    '## 完整输出验证', '',
    '| 模型 | 变体 | 输入 | 输出token数 | 与原版本token一致 | 完整logits逐字节一致 | 对编译常量FP32执行门槛 |',
    '| --- | --- | --- | ---: | --- | --- | --- |']
comparisons = []
diagnostics = []
for model, source, base, companion in models:
    for variant in ('w8-fused2', 'w8-fused2-gqa'):
        variant_dir = results / source / variant
        if not variant_dir.exists():
            continue
        compiled = []
        for folder in sorted(variant_dir.glob('l*')):
            conversion = folder / 'conversion.json'
            if conversion.exists() and not folder.is_symlink():
                compiled.append(json.loads(conversion.read_text()))
        (archive / f'{model}-{variant}-conversion.json').write_text(json.dumps(compiled, indent=2) + '\n')
        for case in ('greeting', 'companion'):
            folder = variant_dir / ('board-' + case)
            if not (folder / 'tokens.json').exists():
                continue
            baseline_case = ('board-greeting' if model == '500M' else 'board') if case == 'greeting' else 'board-' + companion
            baseline = results / source / base / baseline_case
            actual = json.loads((folder / 'tokens.json').read_text())
            expected = json.loads((baseline / 'tokens.json').read_text())
            assert (folder / 'prompt.i32').read_bytes() == (baseline / 'prompt.i32').read_bytes()
            hashes = {name: hashlib.sha256((path / 'logits.f32').read_bytes()).hexdigest()
                      for name, path in [('baseline', baseline), ('candidate', folder)]}
            record = {'model': model, 'variant': variant, 'case': case,
                'same_tokens': actual['tokens'] == expected['tokens'], 'same_logits': hashes['baseline'] == hashes['candidate'],
                'logits_sha256': hashes}
            comparisons.append(record)
            gate = '未运行独立参考'
            validation = folder / 'full-validation-w8.json'
            if validation.exists():
                d = json.loads(validation.read_text())
                gate = ('通过' if d['execution_gate_v1']['passed'] else '未通过') + f'（L2 {d["relative_l2_all_logits"]*100:.3f}%）'
            elif record['same_logits']:
                gate = '与原通过样例逐值一致'
            lines.append(f'| {model} | {variant} | {case} | {actual["generated_tokens"]} | {record["same_tokens"]} | {record["same_logits"]} | {gate} |')
            diagnostics.append((model, variant, case, actual))
            dest = archive / f'{model}-{variant}-{case}'
            dest.mkdir(exist_ok=True)
            for name in ('tokens.json', 'prompt.json', 'exit.json', 'deployed-sha256.txt', 'run.stderr', 'full-validation-w8.json', 'full-validation.json'):
                if (folder / name).exists():
                    shutil.copyfile(folder / name, dest / name)
(archive / 'output-equivalence.json').write_text(json.dumps(comparisons, indent=2) + '\n')
lines += ['', '问候和陪伴均为开发样例；1B陪伴达到64-token上限，不是自然EOS完整回答。逐字节一致证明该样例没有新增差异，不代表独立留出或长历史已验收。', '',
          '## 诊断记录（非受控测速）', '',
          '| 模型 | 变体 | 输入 | token/s | 首字秒 | 驱动MiB | 起始温度°C |',
          '| --- | --- | --- | ---: | ---: | ---: | ---: |']
for model, variant, case, r in diagnostics:
    lines.append(f'| {model} | {variant} | {case} | {r["decode_tokens_per_second"]:.3f} | {r["ttft_seconds"]:.3f} | {r["driver_allocated_bytes"]/1048576:.2f} | {r["temperature_before_mC"]/1000:.1f} |')
lines += ['', '以上运行包含logits写盘，部分起始温度高于此前55°C控制线，不能与上轮中位成绩相除宣布提速。每次驱动释放状态见原始JSON。', '',
          '## 重复性能对照', '']
for name in ('smol500-fusion-performance.json', 'intern1b-fusion-performance.json', 'smol500-gqa-performance.json', 'intern1b-gqa-performance.json',
             'smol500-greedy-performance.json','intern1b-greedy-performance.json'):
    path = results / name
    if not path.exists():
        continue
    d = json.loads(path.read_text())
    shutil.copyfile(path, archive / name)
    lines += [f'### {name}', '', f'状态：{d["status"]}。已记录{len(d["runs"])}次（包含诊断/预热）。', '']
    limit=d['config'].get('max_start_temperature_mC',55000)
    lines += [f'起始温度上限：{limit/1000:.1f}°C。'+('沿用原55°C协议。' if limit==55000 else '补充对照：原版与新版均在当前散热条件下重新测量，不能与此前55°C成绩混算；未替代原条件验收。'),'']
    if d['status'] == 'complete':
        lines += ['| 变体 | 中位token/s（范围） | 首字秒 | 总响应秒 | CPU秒 | 驱动MiB | 起温范围°C |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for label, r in d['medians'].items():
            rows=[v['result'] for v in d['runs'] if v['phase']=='measure' and v['label']==label]
            speeds=[v['decode_tokens_per_second'] for v in rows];temps=[v['temperature_before_mC']/1000 for v in rows]
            lines.append(f'| {label} | {r["decode_tokens_per_second"]:.3f}（{min(speeds):.3f}–{max(speeds):.3f}） | {r["ttft_seconds"]:.3f} | {r["total_seconds"]:.3f} | {r["cpu_seconds"]:.3f} | {r["driver_allocated_bytes"]/1048576:.2f} | {min(temps):.1f}–{max(temps):.1f} |')
        labels=[v['label'] for v in d['config']['variants']]
        before,after=(d['medians'][label] for label in labels[:2])
        paired={label:{v['repeat']:v['result']['decode_tokens_per_second'] for v in d['runs'] if v['phase']=='measure' and v['label']==label} for label in labels[:2]}
        wins=sum(paired[labels[1]][i]>speed for i,speed in paired[labels[0]].items())
        lines += ['',f'相对本轮原版：decode中位提升{(after["decode_tokens_per_second"]/before["decode_tokens_per_second"]-1)*100:.1f}%；同轮{wins}/{len(paired[labels[0]])}次更快；首字缩短{(1-after["ttft_seconds"]/before["ttft_seconds"])*100:.1f}%。这不是相对CPU的加速比。']
        freq={v['result'].get(k) for v in d['runs'] if v['phase']=='measure' for k in ('npu_frequency_before_hz','npu_frequency_after_hz')}
        lines += [f'NPU开始/结束频率记录：{sorted(freq)} Hz；CPU大核开始/结束频率、每次时延和温度见JSON。这是端点采样，不冒充全程连续频率曲线。']
        rss={label:[v['result']['rss_after_inference_kib']/1024 for v in d['runs'] if v['phase']=='measure' and v['label']==label and 'rss_after_inference_kib' in v['result']] for label in labels}
        if all(rss.values()):
            lines += ['推理结束、释放NPU前的进程VmRSS范围：'+ '；'.join(f'{label} {min(values):.2f}–{max(values):.2f} MiB' for label,values in rss.items())+'。设备驱动分配另列，两者不能简单相加当作整机内存峰值。']
        lines.append('')
lines += ['受控脚本先验证相同提示词、token和全部logits，再各预热一次、交替至少五次测量。55°C原条件试跑等待300秒后仍未达标，运行器报`cooldown timeout`，该次失败保留。为继续研发，另建起温上限68°C的补充实验，默认55°C协议仍保留，未把补充实验当作原条件验收；正式CPU/NPU验收需重新统一温度条件。', '',
          '早期重复脚本的`max_rss_kib`可能带入启动器校验大文件时的fork内存高水位，不能解释为模型自身RSS。本轮后续改用分块SHA256，并新增读取`/proc/self/status`的推理结束VmRSS；驱动分配始终独立记录。', '',
          '## 实际生成样例','']
for model,source,_,_ in models:
    path=results/source/'w8-fused2-gqa/board-companion/full-validation-w8.json'
    if path.exists():
        d=json.loads(path.read_text())
        lines += [f'### {model} 紧凑GQA完整语言链路','',d['npu_answer'],'',
            '与原版输出逐值一致。'+('此回答达到64-token上限，显示为截断文本。' if model=='1B' else '此回答自然EOS结束。'),'']
lines += [
          '## 尚未完成', '',
          '独立留出和长历史验收、同轨迹最优CPU Q8比较、批量prefill、容量分档、30分钟稳定性仍未完成。当前结果不支持宣称NPU超过CPU。', '',
          '复现入口：`compile_fused_stack.py`、`run_full_model.py --fusion-width 2`、`validate_full.py`、`benchmark_variants.py`；GQA编译增加`--compact-gqa`。模型/SDK/NBG保持在Git忽略目录，源码、编译参数来源哈希、验证结果与性能JSON同仓归档。']
(here / 'Decoder融合与GQA优化.md').write_text('\n'.join(line.rstrip() for line in '\n'.join(lines).splitlines()).rstrip() + '\n')
print(here / 'Decoder融合与GQA优化.md')
