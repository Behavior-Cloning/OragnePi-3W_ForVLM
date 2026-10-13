"""Retain single-shard screening separately from complete-vocabulary results."""
import hashlib
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'results'


def read(path):
    if not path.exists():return None
    raw=path.read_bytes()
    return {**json.loads(raw),'source_path':str(path),'source_sha256':hashlib.sha256(raw).hexdigest()}


def main():
    probes={name:read(ROOT/f'intern1b-head-profile-20261013/probe-{name}/board-validation.json') for name in ('baseline','dfpmse')}
    trial=read(ROOT/'headmse-productwide-trial-20261013/state.json')
    performance=read(ROOT/'headmse-productwide-trial-20261013/performance.json')
    evidence={'probes':probes,'trial':trial,'performance':performance}
    lines=['# 1B输出头DFP MSE候选','',
           '权重位数维持W8。将输出头从逐输出通道INT8改为单尺度DFP INT8，搜索使权重均方误差最小的小数位；激活余量不变，保留全部151674词表。四个分片都编译，decoder沿用通过开发门槛的gate/up DFP MSE与范围修复版，prefill16不变。','',
           '## 单片筛选','',
           '输入来自原head校准样本，不是独立留出。首49152词表片的结果只能用于筛选，不能当成完整模型速度。','',
           '| 候选 | 次数 | 平均API ms | 执行门槛 |','| --- | ---: | ---: | --- |']
    for name,p in probes.items():
        if p:
            t=p['timing'];lines.append(f'| {name} | {t["repeats"]} | {t["mean_api_seconds"]*1000:.4f} | {p["gate"]["passed"]} |')
    lines+=['','## 完整24层、全部词表','',
            f'开发试验状态：`{trial["status"] if trial else "未开始"}`。1024已有历史＋32新增、128步，先核对完整logits与原Q8 KL/PPL门槛，通过才做中温交替五轮。独立留出与自然生成仍需另验，不能以单片精度替代。']
    if trial and 'execution_gate' in trial:
        q=trial['original_q8_development']
        lines+=['',f'导出常量执行门槛：{trial["execution_gate"]["passed"]}。原Q8开发轨迹KL={q["mean_kl"]:.6f}，PPL比={q["perplexity_ratio"]:.6f}，logits相对L2={q["l2"]:.3%}。']
    if performance and performance['status']=='complete':
        b,c=performance['medians']['baseline'],performance['medians']['candidate']
        assert all(len([r for r in performance['runs'] if r['phase']=='measure' and r['label']==label])==5 for label in ('baseline','candidate'))
        gains=[]
        for repeat in range(5):
            rows={r['label']:r['result'] for r in performance['runs'] if r['phase']=='measure' and r['repeat']==repeat}
            assert rows['baseline']['feedback_tokens']==rows['candidate']['feedback_tokens']
            gains.append(100*(rows['candidate']['decode_tokens_per_second']/rows['baseline']['decode_tokens_per_second']-1))
        lines+=['','| 指标 | 原输出头 | 新输出头 |','| --- | ---: | ---: |']
        for key,label in [('decode_tokens_per_second','decode token/s'),('ttft_seconds','首token s'),('total_seconds','响应 s')]:
            lines.append(f'| {label} | {b[key]:.6f} | {c[key]:.6f} |')
        lines+=['',f'中位解码提升{100*(c["decode_tokens_per_second"]/b["decode_tokens_per_second"]-1):+.2f}%；配对收益{min(gains):+.2f}%–{max(gains):+.2f}%，{sum(g>0 for g in gains)}/5次更快。']
    lines+=['','任何失败原样保留，不放宽原数值门槛；尚未达到全模型质量要求时不把局部提速加入最终成绩。','',
            '复现：`npu/.venv/bin/python npu/kv-cache/report_head_mse.py`。','']
    if performance:
        performance['runs']=[{k:v for k,v in r.items() if k!='system_samples'} for r in performance['runs']]
    (HERE/'1B输出头DFP候选.md').write_text('\n'.join(lines))
    (HERE/'evidence-attack-20261013/head-dfpmse.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+'\n')
    print(trial['status'] if trial else 'pending')


if __name__=='__main__':main()
