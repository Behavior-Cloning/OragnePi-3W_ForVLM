"""Report fresh acceptance separately from tuning, old holdouts and board speed."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent/'results'


def read(path):
    if not path.exists():
        return None
    raw = path.read_bytes()
    try:
        return {**json.loads(raw), 'source_path': str(path),
                'source_sha256': hashlib.sha256(raw).hexdigest()}
    except json.JSONDecodeError:
        return {'status': 'snapshot-in-progress', 'source_path': str(path)}


def aggregate(rows):
    if not rows:
        return None
    count = sum(r['token_count'] for r in rows)
    return {'cases': len(rows), 'tokens': count,
            'mean_kl_nats': sum(r['mean_kl_nats']*r['token_count'] for r in rows)/count,
            'perplexity_ratio': math.exp(sum(math.log(r['perplexity_ratio'])*r['token_count'] for r in rows)/count),
            'individual_passed': sum(r['numeric_passed'] for r in rows)}


def main():
    suite_path = HERE/'quality-holdout-v2.json'
    suite_raw = suite_path.read_bytes()
    suite = json.loads(suite_raw)
    reference = read(ROOT/'holdout-v2-reference-headmse-intern1b-20261013/state.json')
    actual = read(ROOT/'holdout-v2-npu-headmse-intern1b-20261013/result.json')
    evidence = {'suite_sha256': hashlib.sha256(suite_raw).hexdigest(),
                'reference': reference, 'actual': actual, 'product_accepted': False}
    assert reference and reference['status'] == 'complete-reference-only'
    assert reference['holdout_sha256'] == evidence['suite_sha256'] and suite['version'] == 2
    expected_ids = {r['case'] for r in reference['cases']}
    assert len(expected_ids) == 36
    h = (actual or {}).get('holdout', {})
    rows = h.get('cases', [])
    if h:
        assert h['holdout_sha256'] == evidence['suite_sha256']
    assert len({r['case'] for r in rows}) == len(rows)
    assert {r['case'] for r in rows} <= expected_ids
    totals = {lang: aggregate([r for r in rows if lang == 'all' or r['case'].startswith(lang+'-')])
              for lang in ('all', 'en', 'zh')}
    evidence['aggregates'] = totals
    lines = ['# 1B新输出头：独立v2对话验收', '',
             '全部24层、完整151674词表、NPU W8主体及既有FP16例外。12个新中英场景，各三轮；套件在任何候选输出前冻结，只作验收，不用于校准、缩放选择或性能调优。旧输出头v1结果不迁移。', '',
             '参考是工作站标准Transformers对原Q8_0反量化后的FP32计算；本表不是板端CPU速度，也不是对训练原BF16的比较。数值测试采用参考的固定反馈，自由对话使用各自实际生成历史。', '',
             f'任务状态：`{(actual or {}).get("status", "等待板锁")}`；数值进度{len(rows)}/36。留出状态：`{h.get("status", "未开始")}`。执行任务结束不等于产品质量通过。', '',
             '| 语言 | 已测 / 预期 | token数 | KL | PPL比 | 单例通过 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for lang, d in totals.items():
        if d:
            lines.append(f'| {lang} | {d["cases"]}/{36 if lang=="all" else 18} | {d["tokens"]} | '
                         f'{d["mean_kl_nats"]:.6f} | {d["perplexity_ratio"]:.6f} | {d["individual_passed"]}/{d["cases"]} |')
    if h.get('status') == 'complete-numeric-and-free-generation':
        assert len(rows) == 36
        assert math.isclose(totals['all']['mean_kl_nats'], h['mean_kl_nats'], rel_tol=1e-5, abs_tol=1e-7)
        assert math.isclose(totals['all']['perplexity_ratio'], h['perplexity_ratio'], rel_tol=1e-5)
        lines += ['', f'聚合数值门槛通过：{h["numeric_passed"]}；全部单例通过：{h["all_cases_numeric_passed"]}。数值/自由生成结束后driver归零：{h["closed"]["driver_bytes"]==h["free_closed"]["driver_bytes"]==0}。']
    lines += ['', '门槛维持KL≤0.05、PPL比≤1.05；聚合与单例分别列。若未完成36条，不作整体质量结论。自然度、角色稳定及重复需看原文，数值通过不能替代产品验收；人工盲评尚未完成。']
    failures = [r for r in rows if not r['numeric_passed']]
    if failures:
        lines += ['', '| 未过单例 | KL | PPL比 |', '| --- | ---: | ---: |']
        for r in failures:
            lines.append(f'| {r["case"]} | {r["mean_kl_nats"]:.6f} | {r["perplexity_ratio"]:.6f} |')
    natural = {r['case']: r for r in h.get('natural', [])}
    if natural:
        lines += ['', '## 全部自由对话原文', '',
                  '同一套用户问题；后续轮分别接续各自回答，不能将两者视为同一token输入。参考在工作站，NPU完整语言计算在板卡，HTTP/分词在主机；尚非独立板端网页。']
        refs = {r['case']: r for r in reference['cases']}
        for scenario in suite['cases']:
            lines += ['', '### '+scenario['id'], '']
            for i, question in enumerate(scenario['turns'], 1):
                key = scenario['id']+f'-turn{i}'
                lines += [f'用户{i}：{question}', '', '标准Q8参考：', '', refs[key]['reference_answer'], '', 'NPU：', '',
                          natural.get(key, {}).get('answer', '尚无结果') or '（空回答）', '']
    lines += ['', '复现：`npu/.venv/bin/python npu/kv-cache/report_holdout_v2.py`。', '']
    # Store compact provenance and complete answer text, excluding token arrays.
    reference['cases'] = [{k: v for k, v in r.items() if k not in ('prompt_tokens', 'feedback_tokens')}
                          for r in reference['cases']]
    if actual:
        actual.pop('runs', None)
        for r in rows:
            r.pop('runtime', None)
    (HERE/'1B新输出头独立v2验收.md').write_text('\n'.join(lines))
    (HERE/'evidence-attack-20261013/holdout-v2-headmse.json').write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False)+'\n')
    print({'status': (actual or {}).get('status', 'pending'), 'numeric_cases': len(rows), 'natural': len(natural)})


if __name__ == '__main__':
    main()
