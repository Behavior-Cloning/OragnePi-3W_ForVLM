"""Archive full-depth causal padded-tail experiments without extrapolating lengths."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / 'results'


def read(path):
    if not path.exists():
        return None
    raw = path.read_bytes()
    return {**json.loads(raw), 'source_path': str(path),
            'source_sha256': hashlib.sha256(raw).hexdigest()}


def main():
    evidence = {}
    lines = ['# 短输入尾块合并实验', '',
             '复用已有32/16-token prefill图，把不足一批的真实输入放在前面，后面重复末token补齐。因果mask屏蔽未来位置，只写回真实行的KV，首token取最后真实行的hidden。不会把padding加入对话历史。选项默认关闭，生产网页未修改。', '',
             '本轮是512历史＋20新增（500M）和512＋12新增（1B）的补充实验，含模板token。不能代替用户要求的32/64/128矩阵。分别运行全部32/24层和完整输出头，CPU冻结128步反馈；两个NPU版本分别核对完整logits及原Q8质量，通过后才做中温交替五轮。', '',
             '| 模型 | 状态 | 标量尾块执行门槛 | 合并尾块执行门槛 |',
             '| --- | --- | --- | --- |']
    for model in ('smol500', 'intern1b'):
        original = ROOT / f'prefill-tail-{model}-20261013'
        folder = original
        if (Path(str(original)+'-attempt2')/'state.json').exists():
            folder=Path(str(original)+'-attempt2')
        state = read(folder/'state.json')
        performance = read(folder/'performance.json')
        cpu = read(ROOT / f'prefill-tail-{model}-20261013/cpu/results.json')
        evidence[model] = {'state': state, 'performance': performance, 'cpu': cpu,
                           'original_attempt': read(original/'state.json') if folder!=original else None}
        variants = (state or {}).get('variants', {})
        gates = [str(variants.get(v, {}).get('execution_gate', {}).get('passed', '待验'))
                 for v in ('scalar', 'padded')]
        lines.append(f'| {model} | {(state or {}).get("status", "未启动")} | {gates[0]} | {gates[1]} |')
    lines += ['', '1B首个标量尾块诊断未过执行门槛，原记录保留；新attempt只复用该诊断及其CPU轨迹，继续检查批量padding自身的完整精度。只要任一侧未过，就不做正式速度比。']
    lines += ['', '| 模型 / 已有＋新增 | 标量→合并首token s | 首字降低 | 标量→合并decode token/s | 驱动MiB（标量→合并） |',
              '| --- | ---: | ---: | ---: | ---: |']
    for model, row in evidence.items():
        p = row['performance']
        if not p or p['status'] != 'complete':
            continue
        assert row['state']['status'] == 'complete'
        variants = row['state']['variants']
        assert all(v['execution_gate']['passed'] and v['mean_kl_nats'] <= .05 and
                   v['perplexity_ratio'] <= 1.05 for v in variants.values())
        measured = [r for r in p['runs'] if r['phase'] == 'measure']
        assert len(measured) == 10
        a, b = p['medians']['scalar'], p['medians']['padded']
        new_tokens = row['state']['fixture']['new_tokens']
        for label in ('scalar', 'padded'):
            runs = [r for r in measured if r['label'] == label]
            assert len(runs) == 5
            assert all(r['result']['decoder_layers'] == (32 if model == 'smol500' else 24)
                       and r['result']['generated_tokens'] == 128
                       and r['result']['new_prompt_tokens'] == new_tokens
                       and r['result']['cached_prefix_tokens'] == 512
                       and r['result']['driver_after_release_bytes'] == 0 for r in runs)
        lines.append(f'| {model} / 512＋{new_tokens} | {a["ttft_seconds"]:.6f}→{b["ttft_seconds"]:.6f} | '
                     f'{100*(1-b["ttft_seconds"]/a["ttft_seconds"]):+.2f}% | '
                     f'{a["decode_tokens_per_second"]:.3f}→{b["decode_tokens_per_second"]:.3f} | '
                     f'{a["driver_allocated_bytes"]/2**20:.2f}→{b["driver_allocated_bytes"]/2**20:.2f} |')
        gains = []
        for repeat in range(5):
            pair = {r['label']: r['result'] for r in measured if r['repeat'] == repeat}
            assert pair['scalar']['feedback_tokens'] == pair['padded']['feedback_tokens']
            gains.append(100*(1-pair['padded']['ttft_seconds']/pair['scalar']['ttft_seconds']))
        row['paired_ttft_reduction_percent'] = gains
    lines += ['', '首字是引擎增量口径，排除加载、512历史准备与调温；响应起温55–<65°C。只合并短尾块，decode图和权重不变。局部KV拷贝已做多头/两布局/边界canary及ASAN/UBSAN检查，最终仍以ARM实板全链路精度为准。', '',
              '需要扩展更多短长度及独立留出，才可决定是否接入常驻对话。当前20/12-token测试不作产品质量或全长度最优声明。', '',
              '复现：`npu/.venv/bin/python npu/kv-cache/report_prefill_tail.py`。', '']
    for row in evidence.values():
        if row['performance']:
            row['performance']['runs'] = [{k: v for k, v in r.items() if k != 'system_samples'}
                                           for r in row['performance']['runs']]
        if row['cpu']:
            row['cpu']['runs'] = [{k: v for k, v in r.items() if k != 'system_samples'}
                                  for r in row['cpu']['runs']]
    (HERE / '短输入尾块合并实验.md').write_text('\n'.join(lines))
    (HERE / 'evidence-attack-20261013/prefill-tail.json').write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + '\n')
    print({k: (v['state'] or {}).get('status', 'pending') for k, v in evidence.items()})


if __name__ == '__main__':
    main()
