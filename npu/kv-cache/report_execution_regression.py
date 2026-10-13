"""Keep the failed 1024+64 trajectory visible; attribute only actual traces."""
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / 'results'
STACK = ROOT / 'full-intern1b-sharded-v2-c128/w8-fused2-c1280-decode-dfpmse-productwide'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def summary(folder):
    path = folder / 'full-validation-w8.json'
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return {k: d[k] for k in ('execution_gate_v1', 'relative_l2_all_logits', 'min_step_cosine',
                              'argmax_match_fraction', 'mean_kl_nats', 'step_relative_l2')} | {
        'source': str(path), 'source_sha256': sha(path),
        'decode_block_errors': d.get('decode_block_errors')}


def main():
    baseline = STACK / 'board-dfpmse-productwide-intern1b-h1024-n64'
    trace = STACK / 'board-dfpmse-productwide-intern1b-h1024-n64-blocktrace'
    cases = {str(n): summary(STACK / f'board-dfpmse-productwide-intern1b-h1024-n{n}') for n in (32, 64)}
    evidence = {'cases': cases, 'trace': summary(trace), 'trace_identical': None}
    lines = ['# 1B 1024＋64执行误差定位', '',
             '全24层、完整词表、128步固定CPU反馈。比较对象为同一候选导出常量的FP32计算，另有原始Q8质量参考。两者不能替代：原Q8 KL/PPL通过，也不能覆盖执行L2超线。', '',
             '| 已有＋新增 | 全logits L2 | 最低cosine | argmax一致 | 平均KL | 执行门槛 |',
             '| --- | ---: | ---: | ---: | ---: | --- |']
    for n, d in cases.items():
        if d:
            lines.append(f'| 1024＋{n} | {d["relative_l2_all_logits"]:.4%} | {d["min_step_cosine"]:.6f} | '
                         f'{d["argmax_match_fraction"]:.3%} | {d["mean_kl_nats"]:.6f} | {d["execution_gate_v1"]["passed"]} |')
    lines += ['', '维持L2≤5%、最低cosine≥0.99、argmax≥95%的原门槛；1024＋64不进入正式性能表。该场景逐步误差没有单向增长，但这不足以排除KV误差或确定某层故障。需要逐层组trace或真实输入重放。', '']
    d = evidence['trace']
    if d:
        evidence['trace_identical'] = sha(baseline/'logits.f32') == sha(trace/'logits.f32')
        assert evidence['trace_identical'], 'trace changed logits; investigate capture before attributing original failure'
        assert d['decode_block_errors'] and len(d['decode_block_errors']) == 12
        lines += ['已取得127个后续decode步骤×12个两层组的完整trace；128步完整logits与原失败运行逐字节相同。下表是从模型输入累积到该层组的误差，不能当成每组独立局部误差。第一token来自prefill，不包含在decode层组trace内。', '',
                  '| 层组输出（0起始层号） | 全127步累计L2 | 每步L2中位 | 每步L2最大 | 最大处预测序号 |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for b in d['decode_block_errors']:
            e = b['step_relative_l2'];maximum = max(e)
            lines.append(f'| {b["last_layer"]-1}/{b["last_layer"]} | {b["relative_l2"]:.4%} | '
                         f'{statistics.median(e):.4%} | {maximum:.4%} | {e.index(maximum)+b["first_prediction_index"]} |')
    else:
        lines.append('完整decode层组trace已排队，尚无结果，不根据最终logits猜测异常层。')
    audit_path=trace/'head-execution-audit.json'
    if audit_path.exists():
        assert evidence['trace_identical']
        audit=json.loads(audit_path.read_text());m=audit['metrics']
        evidence['head_execution']={**audit,'source':str(audit_path),'source_sha256':sha(audit_path)}
        lines+=['','## 真实末层hidden上的输出头误差分解','',
                '在实际板端末层hidden上，用相同导出输出头常量做FP32计算。它与全FP32输出的差表示decoder误差传到logits后的影响；它与板端logits的差表示head执行及输入量化的额外影响。只是诊断，不把head搬到CPU作为候选。','',
                '| 项目 | 相对L2 |','| --- | ---: |',
                f'| 总执行误差（后续127步） | {m["all_error_relative_l2"]:.4%} |',
                f'| decoder误差传到logits | {m["decoder_propagated_error_relative_l2"]:.4%} |',
                f'| 输出头局部执行 | {m["local_head_error_relative_l2"]:.4%} |','',
                f'各项使用同一参考范数，L2不能直接相加；平方交叉项为{m["normalized_cross_term"]:.8f}，已核对误差向量分解恒等式。局部head KL={m["local_head_mean_kl_nats"]:.6f}。不包含prefill生成的第一token。']
    lines += ['', '新版完整输出头另扩展同一1024＋64/128轨迹，保留相同decoder与prefill，以区分输出头变化是否足够；不将32新增的通过外推至64新增。', '',
              '复现：`npu/.venv/bin/python npu/kv-cache/report_execution_regression.py`。', '']
    (HERE/'1B增量64执行误差定位.md').write_text('\n'.join(lines))
    (HERE/'evidence-attack-20261013/execution-regression.json').write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False)+'\n')
    print('trace-ready' if d else 'waiting-trace')


if __name__ == '__main__':
    main()
