"""Archive hardware attribution and bounded dense-kernel experiments."""
import hashlib
import json
from pathlib import Path
import statistics

from profile_npd import summarize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / 'results/intern1b-decode-profile-20261013'
ARCHIVE = HERE / 'evidence-attack-20261013'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    ARCHIVE.mkdir(exist_ok=True)
    traces = [summarize(ROOT / 'l0', path) for path in sorted((ROOT / 'npd-l0').glob('*.npd'))]
    assert len(traces) == 5
    profile = {'scope': 'ordinary layers 0/1, actual full-depth position 544, capacity 768; diagnostic only',
               'counter_warning': 'summed hardware busy cycles are not additive end-to-end latency or measured DRAM bandwidth',
               'traces': traces}
    profile['fullconnect_busy_percent'] = [100*t['by_op']['fullconnect']['cycles']/t['summed_busy_cycles'] for t in traces]
    base = sha(ROOT / 'l0/export_nbg_unify/network_binary.nb')
    trials = []
    for name in ('tpfc0', 'tpfc1', 'nn-transpose', 'burst128', 'gateup-a8', 'gateup-a8-range1', 'gateup-w8dfp', 'gateup-w8dfp-mse'):
        folder = ROOT / name
        conversion = folder / 'conversion.json'
        if not conversion.exists():
            continue
        record = json.loads(conversion.read_text())
        binary = folder / 'export_nbg_unify/network_binary.nb'
        trials.append({'name': name, 'source': str(folder), 'conversion': record,
                       'binary_sha256': sha(binary) if binary.exists() else None,
                       'binary_identical_to_baseline': sha(binary) == base if binary.exists() else False})
    probes = {}
    for name in ('preload-on', 'preload-off', 'probe-gateup-a8', 'probe-gateup-a8-range1', 'probe-gateup-w8dfp', 'probe-gateup-w8dfp-mse'):
        path = ROOT / name / 'board-validation.json'
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        record.update(source_path=str(path), source_sha256=sha(path))
        actual = ROOT / name / 'output.f32'
        record['output_sha256'] = sha(actual)
        probes[name] = record
    smol_root = HERE.parent / 'results/smol500-decode-profile-20261013'
    smol = [summarize(smol_root/'l0', path) for path in sorted((smol_root/'npd-l0').glob('*.npd'))]
    report = {'profile': profile, 'compiler_trials': trials, 'probes': probes,
              'smol500_profiles': smol}
    dfp=[summarize(ROOT/'gateup-w8dfp', path) for path in sorted((ROOT/'npd-gateup-w8dfp').glob('*.npd'))]
    report['dfp_profiles']=dfp
    fullroot=HERE.parent/'results/full-intern1b-sharded-v2-c128/w8-fused2-c768-dfpgateup/board-dfpgateup-intern1b-h512-n64'
    report['full_dfp']={}
    for name in ('full-validation-w8.json','full-validation.json'):
        path=fullroot/name
        if path.exists():
            d=json.loads(path.read_text())
            report['full_dfp'][name]={k:d[k] for k in ('relative_l2_all_logits','min_step_cosine','argmax_match_fraction','execution_gate_v1','fixed_trajectory_nll','mean_kl_nats')}
            report['full_dfp'][name].update(source_path=str(path),source_sha256=sha(path))
    mse_root=HERE.parent/'results/full-intern1b-sharded-v2-c128/w8-fused2-c768-dfpgateup-mse-v2/board-dfpmse-intern1b-h512-n64'
    report['full_dfp_mse']={}
    for name in ('full-validation-w8.json','full-validation.json'):
        path=mse_root/name
        if path.exists():
            d=json.loads(path.read_text())
            report['full_dfp_mse'][name]={k:d[k] for k in ('relative_l2_all_logits','min_step_cosine','argmax_match_fraction','execution_gate_v1','fixed_trajectory_nll','mean_kl_nats')}
            report['full_dfp_mse'][name].update(source_path=str(path),source_sha256=sha(path))
    report['smol500_dfp_probes']={}
    for name in ('probe-gateup-w8dfp','probe-gateup-w8dfp-mse'):
        path=smol_root/name/'board-validation.json'
        if path.exists():report['smol500_dfp_probes'][name]={**json.loads(path.read_text()),'source_path':str(path),'source_sha256':sha(path)}
    (ARCHIVE / 'decoder-dense-probes.json').write_text(json.dumps(report, indent=2) + '\n')
    pct = profile['fullconnect_busy_percent']
    lines = ['# Decoder矩阵瓶颈与候选筛选', '',
             '2026-10-13。以下为局部诊断，不能当成完整模型速度或产品验收。输入来自InternVL3-1B全部24层、512历史＋32新增、128步固定反馈运行中的位置544。', '',
             f'普通第0/1层的5份硬件计数：fullconnect占累计busy cycles中位{statistics.median(pct):.3f}%（{min(pct):.3f}–{max(pct):.3f}%）。主要大矩阵为MLP gate/up/down。重复编译器UID保留候选层标签，不猜测具体层归属。', '',
             '这是这一组普通层的归因，不能外推敏感层或500M；busy cycles不能直接相加当成端到端耗时，read counter也不是独立实测DDR带宽。', '',
             '## 编译候选', '', '| 候选 | 编译退出码 | 二进制相同 | 处理 |', '| --- | ---: | --- | --- |']
    for trial in trials:
        same = trial['binary_identical_to_baseline']
        lines.append(f'| {trial["name"]} | {trial["conversion"]["exit_code"]} | {"是" if same else "否"} | {"不重复计时，无生效证据" if same else "需实板精度和耗时检查"} |')
    lines += ['', '## 板端探针', '',
              '| 候选 | 次数 | API中位ms | 平均ms | 全输出数值门槛 | 起止温°C | 驱动MiB |',
              '| --- | ---: | ---: | ---: | --- | --- | ---: |']
    for name, record in probes.items():
        t = record['timing']
        samples = t['api_seconds']
        lines.append(f'| {name} | {t["repeats"]} | {statistics.median(samples)*1000:.4f} | {t["mean_api_seconds"]*1000:.4f} | {"通过" if record["gate"]["passed"] else "拒绝"} | {t["temperature_before_mC"]/1000:.1f}→{t["temperature_after_mC"]/1000:.1f} | {t["allocated_bytes"]/2**20:.2f} |')
    if smol:
        assert len(smol)==5
        fc = statistics.median(100*t['by_op']['fullconnect']['cycles']/t['summed_busy_cycles'] for t in smol)
        mm = statistics.median(100*t['by_op']['matmul']['cycles']/t['summed_busy_cycles'] for t in smol)
        lines += ['',f'500M补充：全32层诊断输出与正式基线逐字节一致，完整执行门槛通过。相同位置544、容量768的普通0/1层硬件计数，fullconnect中位{fc:.2f}%，matmul约{mm:.2f}%；不同模型分别归因。']
    if dfp:
        gain=1-statistics.median(t['summed_busy_cycles'] for t in dfp)/statistics.median(t['summed_busy_cycles'] for t in traces)
        lines+=['',f'DFP权重路径硬件计数：相同两层累计busy cycles减少{gain:.2%}，出现fullconnect+swish与fullconnect+multiply融合节点。这提供编译执行路径变化证据，不能把计数比例当作完整模型速度。']
    if report['full_dfp']:
        lines+=['','## 完整DFP候选：精度尚未接受','',
                '1B、512＋64、全部24层和完整输出头：导出常量执行门槛通过，但相对原Q8的完整logits L2约12.60%、开发轨迹PPL比1.05738，超过预设1.05质量线。诊断运行写大logits，速度不列为正式成绩；不推广这一候选。','',
                '下一版保持W8/A16，只用权重自身搜索DFP小数位，最小化权重均方误差；不查看留出对话。每矩阵记录全部候选的误差和截断比例。既有敏感层、head与KV格式保持基线来源；新版本仍须全模型验证。']
    if report['full_dfp_mse']:
        d=report['full_dfp_mse']['full-validation-w8.json'];o=report['full_dfp_mse']['full-validation.json']
        lines+=['',f'MSE缩放版完整24层已验证：执行门槛{"通过" if d["execution_gate_v1"]["passed"] else "未通过"}，L2 {d["relative_l2_all_logits"]:.3%}、最低cosine {d["min_step_cosine"]:.6f}；相对原Q8开发轨迹PPL比{o["fixed_trajectory_nll"]["perplexity_ratio"]:.6f}、logits L2 {o["relative_l2_all_logits"]:.3%}。该结果不等于独立留出通过，五轮干净测速另列。']
    for name,d in report['smol500_dfp_probes'].items():
        lines+=['',f'500M局部{name}：50次两层API平均{d["timing"]["mean_api_seconds"]*1000:.4f}ms，执行门槛{"通过" if d["gate"]["passed"] else "未通过"}。完整32层仍需验证，不与1B混合计算收益。']
    lines += ['', '探针3次预热后记录50次；只统计VIP API，不含完整模型宿主代价。开关分时执行、温度不同，不以微小差异宣告提速。SRAM设置返回成功仅证明接口接受，不证明实际预载字节数；最终仍须同温档、全深度五轮A/B。', '',
              'A8候选仅在选中的W8 gate/up矩阵输入前插入显式转换，原权重、外部KV、残差和矩阵输出格式不变。默认保持原校准动态范围；range1另将范围缩小1bit，须额外验证饱和风险。该变化不是新的权重量化算法，局部通过也不能替代完整logits和独立留出。', '',
              '实测gate/up A8无可见速度收益，hidden局部L2达9.42%、另一V输出达7.39%，超过5%门槛，拒绝。后续DFP候选保持W8位数和A16，单独比较单尺度权重与逐通道权重；必须另查原权重误差，不能只依赖导出常量参考。', '',
              '紧凑证据：[decoder-dense-probes.json](evidence-attack-20261013/decoder-dense-probes.json)。重建：`npu/.venv/bin/python npu/kv-cache/report_decoder_dense.py`。', '']
    (HERE / 'Decoder矩阵瓶颈与候选筛选.md').write_text('\n'.join(lines))
    print(json.dumps({'fullconnect_busy_percent_median': statistics.median(pct),
                      'compiler_candidates': len(trials), 'measured_probes': list(probes)}))


if __name__ == '__main__':
    main()
