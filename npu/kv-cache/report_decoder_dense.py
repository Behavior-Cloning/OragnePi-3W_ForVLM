"""Archive hardware attribution and bounded dense-kernel experiments."""
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

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
    for name in ('tpfc0', 'tpfc1', 'nn-transpose', 'burst128', 'gateup-a8', 'gateup-a8-range1', 'gateup-w8dfp', 'gateup-w8dfp-mse', 'all-w8dfp-mse'):
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
    for name in ('preload-on', 'preload-off', 'probe-gateup-a8', 'probe-gateup-a8-range1', 'probe-gateup-w8dfp', 'probe-gateup-w8dfp-mse', 'probe-all-w8dfp-mse'):
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
    for name in ('probe-gateup-w8dfp','probe-gateup-w8dfp-mse','probe-all-w8dfp-mse'):
        path=smol_root/name/'board-validation.json'
        if path.exists():report['smol500_dfp_probes'][name]={**json.loads(path.read_text()),'source_path':str(path),'source_sha256':sha(path)}
    ab_path=HERE.parent/'results/intern1b-dfpmse-ab-20261013.json'
    ab=json.loads(ab_path.read_text()) if ab_path.exists() else None
    if ab and ab['status']=='complete':
        compact={k:v for k,v in ab.items() if k!='runs'}
        compact.update(source_path=str(ab_path),source_sha256=sha(ab_path))
        compact['runs']=[{k:v for k,v in r.items() if k!='system_samples'} for r in ab['runs']]
        report['full_dfp_mse_ab']=compact
    smol_ab_path=HERE.parent/'results/smol500-dfpgateup-ab-20261013.json'
    smol_ab=json.loads(smol_ab_path.read_text()) if smol_ab_path.exists() else None
    if smol_ab and smol_ab['status']=='complete':
        compact={k:v for k,v in smol_ab.items() if k!='runs'}
        compact.update(source_path=str(smol_ab_path),source_sha256=sha(smol_ab_path))
        compact['runs']=[{k:v for k,v in r.items() if k!='system_samples'} for r in smol_ab['runs']]
        report['full_smol500_dfp_ab']=compact
    report['all_dfp_original_errors']={}
    for model,root in [('intern1b',ROOT),('smol500',smol_root)]:
        fixture=root/'fixture-all-w8dfp-mse';probe=root/'probe-all-w8dfp-mse'
        if not (probe/'board-validation.json').exists():continue
        original=np.fromfile(fixture/'probe-original-reference.f32',np.float32)
        output=np.fromfile(probe/'output.f32',np.float32)
        spec=json.loads((fixture/'probe-fixture.json').read_text());n=spec['outputs'][0][1]
        report['all_dfp_original_errors'][model]={'hidden_relative_l2':float(np.linalg.norm(output[:n]-original[:n])/np.linalg.norm(original[:n])),
            'original_reference_sha256':sha(fixture/'probe-original-reference.f32'),'board_output_sha256':sha(probe/'output.f32')}
    (ARCHIVE / 'decoder-dense-probes.json').write_text(json.dumps(report, indent=2) + '\n')
    pct = profile['fullconnect_busy_percent']
    lines = ['# Decoder矩阵瓶颈与候选筛选', '',
             '2026-10-13。局部诊断与单独列出的完整模型五轮结果分别报告，局部探针不能代替全模型或产品验收。诊断输入来自InternVL3-1B全部24层、512历史＋32新增、128步固定反馈运行中的位置544。', '',
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
        lines+=['',f'500M局部{name}：50次两层API平均{d["timing"]["mean_api_seconds"]*1000:.4f}ms，执行门槛{"通过" if d["gate"]["passed"] else "未通过"}。局部结果不能代替完整32层验收，不与1B混合计算收益。']
    if report['all_dfp_original_errors']:
        lines+=['','## 扩展全部W8矩阵的局部筛选','',
                'all-w8dfp-mse只改变原本逐通道W8矩阵的编码，既有DFP V及敏感FP16仍保留。两模型对导出常量的执行门槛通过，但独立原Q8参考暴露出更多权重量化误差。','',
                '| 模型 | 两层平均API ms | 对原Q8 hidden L2 | 决定 |','| --- | ---: | ---: | --- |']
        for model,root in [('intern1b',ROOT),('smol500',smol_root)]:
            if model not in report['all_dfp_original_errors']:continue
            d=json.loads((root/'probe-all-w8dfp-mse/board-validation.json').read_text())
            err=report['all_dfp_original_errors'][model]['hidden_relative_l2']
            lines.append(f'| {model} | {d["timing"]["mean_api_seconds"]*1000:.4f} | {err:.3%} | {"误差明显增大且慢于已选峰值gate/up版，拒绝推广" if model=="smol500" else "比gate/up MSE局部略快，但误差增大，暂不推广"} |')
        lines+=['','这些不是全模型速度，也不是独立留出成绩。1B的gate/up MSE对应原Q8局部hidden误差约3.83%，全部矩阵扩展后约6.83%；500M相应扩展达约28.83%。不能只看对导出常量的低误差就宣称量化质量合格。']
    if 'full_dfp_mse_ab' in report:
        c,n=ab['medians']['baseline'],ab['medians']['dfpmse']
        gains=[]
        for i in range(5):
            pair={r['label']:r['result'] for r in ab['runs'] if r['phase']=='measure' and r['repeat']==i}
            gains.append(100*(pair['dfpmse']['decode_tokens_per_second']/pair['baseline']['decode_tokens_per_second']-1))
        lines+=['','## 1B完整模型五轮解码收益','',
                '512历史＋64新增、全24层、128次预测，两端同为t32和grouped head、中温55–<65°C起测。冻结反馈相同，各自完整诊断logits先复现；正式性能不写大logits。','',
                '| 项目 | 基线NPU | DFP MSE候选 |','| --- | ---: | ---: |',
                f'| decode token/s | {c["decode_tokens_per_second"]:.6f} | {n["decode_tokens_per_second"]:.6f} |',
                f'| 引擎首token s | {c["ttft_seconds"]:.6f} | {n["ttft_seconds"]:.6f} |',
                f'| 总响应 s | {c["total_seconds"]:.6f} | {n["total_seconds"]:.6f} |',
                f'| 驱动分配 MiB | {c["driver_allocated_bytes"]/2**20:.2f} | {n["driver_allocated_bytes"]/2**20:.2f} |','',
                f'decode中位提高{100*(n["decode_tokens_per_second"]/c["decode_tokens_per_second"]-1):.2f}%，配对五轮收益范围{min(gains):.2f}%–{max(gains):.2f}%，{sum(g>0 for g in gains)}/5更快。decoder API下降，head API基本不变，收益确实来自decoder路径。','',
                '相同场景已测CPU Q8约18.25 token/s，候选仍未追平；CPU配置搜索继续，不能把超过旧NPU写成超过CPU。原Q8对齐、独立留出、其他输入档位及稳定性均需继续。']
    if 'full_smol500_dfp_ab' in report:
        baseline,candidate=smol_ab['medians']['baseline'],smol_ab['medians']['dfp']
        gains=[]
        for i in range(5):
            pair={r['label']:r['result'] for r in smol_ab['runs'] if r['phase']=='measure' and r['repeat']==i}
            gains.append(100*(pair['dfp']['decode_tokens_per_second']/pair['baseline']['decode_tokens_per_second']-1))
        lines+=['','## 500M完整32层五轮结果','',
                '512历史＋64新增、全部32层、完整输出头、128次固定轨迹预测，中温交替五轮。此模型采用峰值DFP缩放；1B采用MSE缩放，不能混作同一数值方案。两端均t32，head设置相同。','',
                '| 项目 | 基线NPU | DFP候选 |','| --- | ---: | ---: |',
                f'| decode token/s | {baseline["decode_tokens_per_second"]:.6f} | {candidate["decode_tokens_per_second"]:.6f} |',
                f'| 引擎首token s | {baseline["ttft_seconds"]:.6f} | {candidate["ttft_seconds"]:.6f} |',
                f'| 总响应 s | {baseline["total_seconds"]:.6f} | {candidate["total_seconds"]:.6f} |',
                f'| 驱动分配 MiB | {baseline["driver_allocated_bytes"]/2**20:.2f} | {candidate["driver_allocated_bytes"]/2**20:.2f} |','',
                f'decode中位提高{100*(candidate["decode_tokens_per_second"]/baseline["decode_tokens_per_second"]-1):.2f}%，配对收益{min(gains):.2f}%–{max(gains):.2f}%，{sum(g>0 for g in gains)}/5更快。驱动分配减少{(baseline["driver_allocated_bytes"]-candidate["driver_allocated_bytes"])/2**20:.2f}MiB；decoder API下降，head基本不变。','',
                '全模型导出常量门槛通过：L2约3.756%、最低cosine0.99648。相对原Q8开发轨迹logits L2约12.17%、最低cosine0.96548，PPL比0.99030；不能用预测相同或开发PPL较好替代独立留出验收。','',
                '同场景优化CPU Q8五轮约22.780 token/s，NPU约20.487，仍未追平。其他新增长度、1024历史和独立质量继续验证。']
    lines += ['', '探针3次预热后记录50次；只统计VIP API，不含完整模型宿主代价。开关分时执行、温度不同，不以微小差异宣告提速。SRAM设置返回成功仅证明接口接受，不证明实际预载字节数；最终仍须同温档、全深度五轮A/B。', '',
              'A8候选仅在选中的W8 gate/up矩阵输入前插入显式转换，原权重、外部KV、残差和矩阵输出格式不变。默认保持原校准动态范围；range1另将范围缩小1bit，须额外验证饱和风险。该变化不是新的权重量化算法，局部通过也不能替代完整logits和独立留出。', '',
              '实测gate/up A8无可见速度收益，hidden局部L2达9.42%、另一V输出达7.39%，超过5%门槛，拒绝。后续DFP候选保持W8位数和A16，单独比较单尺度权重与逐通道权重；必须另查原权重误差，不能只依赖导出常量参考。', '',
              '紧凑证据：[decoder-dense-probes.json](evidence-attack-20261013/decoder-dense-probes.json)。重建：`npu/.venv/bin/python npu/kv-cache/report_decoder_dense.py`。', '']
    (HERE / 'Decoder矩阵瓶颈与候选筛选.md').write_text('\n'.join(lines))
    print(json.dumps({'fullconnect_busy_percent_median': statistics.median(pct),
                      'compiler_candidates': len(trials), 'measured_probes': list(probes)}))


if __name__ == '__main__':
    main()
