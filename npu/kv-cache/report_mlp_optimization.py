"""Archive selective MLP precision experiments, including rejected candidates."""
import hashlib
import json
from pathlib import Path
import shutil

here = Path(__file__).resolve().parent
results = here.parent / 'results'
archive = here / 'evidence-mlp-optimization-20261012'
archive.mkdir(exist_ok=True)
audit=json.loads((results/'mlp-precision-audit.json').read_text())
shutil.copyfile(results/'mlp-precision-audit.json',archive/'mlp-precision-audit.json')
guards=json.loads((results/'fixed-feedback-guards.json').read_text())
assert guards['passed']
shutil.copyfile(results/'fixed-feedback-guards.json',archive/'fixed-feedback-guards.json')
models = [
    ('500M', 'smol500', 'full-smol500-stable-v3-c128', 'w8-fused2-gqa-mlp-h1', 'w8-mlp-gateup-h1', [3, 31]),
    ('1B', 'intern1b', 'full-intern1b-sharded-v2-c128', 'w8-fused2-gqa-mlp-h0', 'w8-mlp-gateup-h0', [2, 3]),
]
lines = [
    '# MLP 选择性量化加速', '',
    '2026-10-12。继紧凑GQA和精确整数greedy之后，继续优化500M与1B的全部语言生成链路。对应CPU仍为GGUF Q8_0。本报告不是NPU超过CPU的验收。', '',
    '## 改动与适用范围', '',
    '原候选为了避免异常激活破坏生成，在每个模型的两个敏感层保留了整个MLP的FP16。本轮把这些层的gate/up矩阵恢复为逐输出通道对称INT8，输入输出使用DFP INT16；SiLU、门控乘积和down矩阵仍为FP16。最终实测候选还缩小了这两个层中attention/KV的INT16范围。其余层及输出头保持原产物。', '',
    '| 模型 | 修改的层索引（从0开始） | 敏感层中的FP16矩阵 | MLP激活范围余量 | 这两个层的其他INT16余量 |',
    '| --- | --- | --- | --- | --- |',
    '| 500M | 3、31 | 每层3个→1个 | 4 bit（校准范围×16） | 1 bit（原为4） |',
    '| 1B | 2、3 | 每层3个→1个 | 1 bit（校准范围×2） | 0 bit（原为1） |', '',
    '这是已有GGUF Q8权重上的PTQ，并非AWQ/GPTQ，也不是全模型所有算子都用INT8。更窄范围可降低舍入误差，但对未见输入的截断风险仍需留出和长历史测试。没有用留出集调参。', '',
    '编译复用先逐项核对原始图、权重、量化参数的SHA256以及融合/GQA选项，仅重编译变化的块，旧产物保持不变。', '',
    '配置审计更正：初版`--mlp-headroom-bits`误作用于非MLP的INT16范围循环，实际MLP仍用原余量。因此目录名h1/h0及旧conversion中的CLI参数不能直接当作有效MLP范围。v2已修正参数路由；用500M `--activation-headroom-bits 1 --mlp-headroom-bits 4`、1B `--activation-headroom-bits 0 --mlp-headroom-bits 1`重新编译四层，量化参数语义完全一致，NBG逐字节一致。完整差异与重编译哈希在`mlp-precision-audit.json`。本报告按真实产物说明组合调整，不把质量或速度变化单独归因于MLP。', '',
    '## 开发样例的精度门槛', '',
    '仍按完整logits相对L2≤5%、最低逐步余弦≥0.99、argmax一致率≥95%检查；参考使用实际编译常量重建的全深度FP32。它是执行误差门槛，不能替代相对原模型的量化质量验收。', '',
    '| 模型 | 候选 | 场景 | L2 | 最低余弦 | argmax一致 | 门槛 |',
    '| --- | --- | --- | ---: | ---: | ---: | --- |',
]
for label, short, source, candidate, unfused, layers in models:
    root = results / source
    for variant in ('w8-fused2-gqa-mlp', candidate, 'w8-fused2-gqa'):
        for case in ('greeting', 'companion', 'mlp-fixed'):
            if variant == 'w8-fused2-gqa' and case != 'mlp-fixed':
                continue
            folder = root / variant / ('board-' + case)
            file = folder / 'full-validation-w8.json'
            if not file.exists():
                continue
            d = json.loads(file.read_text())
            status = '通过' if d['execution_gate_v1']['passed'] else '拒绝'
            name = '原GQA基线' if variant == 'w8-fused2-gqa' else ('初版范围' if variant == 'w8-fused2-gqa-mlp' else '选择性MLP')
            scenario = '固定输入轨迹' if case == 'mlp-fixed' else case
            lines.append(f'| {label} | {name} | {scenario} | {100*d["relative_l2_all_logits"]:.3f}% | {d["min_step_cosine"]:.6f} | {d["argmax_matches_fp32"]}/{d["generated_token_count"]} | {status} |')
            target = archive / f'{short}-{variant}-{case}'
            target.mkdir(exist_ok=True)
            for name in ('tokens.json', 'prompt.json', 'exit.json', 'deployed-sha256.txt', 'run.stderr', 'full-validation-w8.json', 'full-validation.json'):
                if (folder / name).exists():
                    shutil.copyfile(folder / name, target / name)
            hashes = {f.name: hashlib.file_digest(f.open('rb'), 'sha256').hexdigest()
                      for f in folder.iterdir() if f.name in ('logits.f32', 'prompt.i32', 'feedback.i32')}
            (target / 'fixture-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    for variant, indices in (('w8-mlp-gateup', layers), (unfused, layers),
                             ('w8-fused2-gqa-mlp', sorted({n//2*2 for n in layers})),
                             (candidate, sorted({n//2*2 for n in layers}))):
        records = [json.loads((root / variant / f'l{i}/conversion.json').read_text()) for i in indices]
        (archive / f'{short}-{variant}-conversion.json').write_text(json.dumps(records, indent=2) + '\n')
lines += ['', '初版500M因argmax一致率低于95%被拒绝，1B因最低余弦低于0.99被拒绝；最终实测版缩小两个敏感层中非MLP的INT16范围后通过上述开发样例。未放宽阈值，未把失败方案纳入合格速度成绩。', '',
          '## 五轮完整链路速度对照', '',
          '每个模型使用原GQA候选的陪伴输出作为固定输入轨迹（teacher forcing），两版都计算全部decoder、KV更新、全词表head与greedy预测，再把相同的冻结token喂回下一步。预测仍被保存，但固定轨迹的预测拼接不作为自由生成回答。500M为30个prompt token加67次预测；1B为49加64。容量128，不冒充128步生成或长历史测试。', '',
          '运行器新增固定反馈功能；原模型重放自己的输出时，全部logits与原自由生成逐字节相同。默认仍是自然生成，EOS正常结束；固定模式明确记录feedback token、要求文件长度/词表范围合法，并执行约定的全部步数。板端另验证短/长反馈文件、负token、词表越界和KV超容全部拒绝；人为指定首个预测为EOS，自然模式停止而固定模式继续，完整logits仍匹配原轨迹。', '',
          '各变体先做完整诊断、再预热一次，交替五轮。每个变体在重复运行中要求自身预测和完整诊断logits与已通过数值门槛的冻结结果一致。不同量化变体允许预测不同，但输入轨迹、工作量完全相同。全部启用等价整数greedy，性能运行不写全词表logits。', '',
          '本轮仍使用起温≤68°C的补充协议，原/新版都重新测量，不混用此前55°C成绩。每秒采样网页健康状态和系统指标，繁忙或异常时拒绝该实验；短于采样间隔的干扰仍可能漏掉。', '']
for label, short, source, candidate, unfused, layers in models:
    path = results / f'{short}-mlp-performance.json'
    if not path.exists():
        lines += [f'{label}：未运行重复测速。', '']
        continue
    d = json.loads(path.read_text())
    shutil.copyfile(path, archive / path.name)
    lines += [f'### {label}', '', f'实验状态：{d["status"]}；已记录{len(d["runs"])}轮（含诊断和预热）。', '']
    if d['status'] != 'complete':
        continue
    lines += ['| 变体 | decode token/s 中位（范围） | 首字秒 | 总响应秒 | CPU秒 | 驱动MiB | 起温°C |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    groups = {}
    for variant, med in d['medians'].items():
        rows = [r for r in d['runs'] if r['phase'] == 'measure' and r['label'] == variant]
        groups[variant] = rows
        speeds = [r['result']['decode_tokens_per_second'] for r in rows]
        temps = [r['result']['temperature_before_mC']/1000 for r in rows]
        lines.append(f'| {variant} | {med["decode_tokens_per_second"]:.3f}（{min(speeds):.3f}–{max(speeds):.3f}） | {med["ttft_seconds"]:.3f} | {med["total_seconds"]:.3f} | {med["cpu_seconds"]:.3f} | {med["driver_allocated_bytes"]/1048576:.2f} | {min(temps):.1f}–{max(temps):.1f} |')
    b, c = (d['medians'][v] for v in ('baseline', 'selective'))
    wins = sum(c['result']['decode_tokens_per_second'] > b['result']['decode_tokens_per_second']
               for b,c in zip(groups['baseline'],groups['selective']))
    lines += ['', f'相对同轮基线decode中位提升{100*(c["decode_tokens_per_second"]/b["decode_tokens_per_second"]-1):.1f}%，{wins}/5轮更快；首字缩短{100*(1-c["ttft_seconds"]/b["ttft_seconds"]):.1f}%；驱动分配减少{(b["driver_allocated_bytes"]-c["driver_allocated_bytes"])/1048576:.2f}MiB。加载单独计时，首字/响应表不含加载。', '',
              '| 变体 | 推理后RSS MiB | 每秒采样最少可用RAM MiB | swap使用MiB范围 | 采样最高温°C |',
              '| --- | ---: | ---: | ---: | ---: |']
    for variant, rows in groups.items():
        rss = [r['result']['rss_after_inference_kib']/1024 for r in rows]
        snapshots = [s['health']['metrics'] for r in rows for s in r['system_samples']]
        swap = [s['swap_used_mib'] for s in snapshots]
        assert all(r['result']['driver_after_release_bytes']==0 for r in rows)
        lines.append(f'| {variant} | {min(rss):.2f}–{max(rss):.2f} | {min(s["ram_available_mib"] for s in snapshots):.1f} | {min(swap):.2f}–{max(swap):.2f} | {max(s["temperature_c"] for s in snapshots):.1f} |')
    freq = sorted({r['result'][k] for rows in groups.values() for r in rows for k in ('npu_frequency_before_hz','npu_frequency_after_hz')})
    lines += ['', f'NPU起止频率：{freq}Hz；所有测量释放后驱动分配为0。RSS与设备驱动分配分别记录，不能直接相加当作整机峰值；原始每token耗时、整机CPU负载、加载和系统采样均在JSON。', '']
lines += ['## 自由生成与原始Q8参考', '']
for label, short, source, candidate, unfused, layers in models:
    folder = results / source / candidate / 'board-companion'
    d = json.loads((folder / 'full-validation-w8.json').read_text())
    original = json.loads((folder / 'full-validation.json').read_text())
    lines += [f'### {label}', '', d['npu_answer'], '',
        ('500M在第90个token自然EOS结束。' if label == '500M' else '1B在64-token上限处截断，不是自然EOS完整回答。'),
        f'该自由轨迹相对原始GGUF Q8反量化FP32参考：logits L2 {100*original["relative_l2_all_logits"]:.3f}%，平均KL {original["mean_kl_nats"]:.6f}，argmax {original["argmax_matches_fp32"]}/{original["generated_token_count"]}。该值包括再次量化与执行误差，不等于“相对原模型无损”。不同自由轨迹不直接比较总响应时间。', '']
lines += ['## 未达成的目标与复现', '',
    '当前仍未证明超过优化CPU Q8；500M历史CPU约27.5、1B约20.3 token/s仅为参考，尚未完成同轨迹正式对照。长历史128/512/2048、独立留出陪伴质量、批量prefill及30分钟稳定性仍待完成。继续提速应优先处理剩余FP16 down矩阵、decoder算子开销与prefill；不能把短历史开发样例当成最终验收。', '',
    '编译：`compile_weight_int8.py --float-mlp --int8-mlp-projections gate up --mlp-headroom-bits N`；融合复用：`compile_fused_stack.py --reuse-blocks-from w8-fused2-gqa`。运行固定轨迹：`run_full_model.py --feedback-from <自然生成tokens.json>`。重复测速：`benchmark_variants.py --fixed-feedback --variant-reference <标签> <已验收目录>`。', '',
    '运行`python3 npu/kv-cache/report_mlp_optimization.py`从本地原始结果重建报告和小型证据。模型、SDK、NBG继续保留本地忽略目录；源码、编译哈希、成功/失败验证和原始性能数据归档Git。']
(here / 'MLP选择性量化加速.md').write_text('\n'.join(lines).rstrip() + '\n')
print(here / 'MLP选择性量化加速.md')
