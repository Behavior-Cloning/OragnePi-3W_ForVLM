"""Separate sustained execution, independent numerical quality and dialogue quality.

Read partial runs without promoting them. Natural replies and context resets
are retained, so a fast cached median cannot hide expensive window rebuilds.
"""
import hashlib
import json
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / 'results'


def snapshot(path):
    if not path.exists():
        return None
    # The producer may still be updating its JSON. Never fall back to an older
    # successful file and label it as the current attempt.
    raw = path.read_bytes()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {'status': 'snapshot-in-progress', 'source_path': str(path)}
    data['source_path'] = str(path)
    data['source_sha256'] = hashlib.sha256(raw).hexdigest()
    return data


def dist(values):
    if not values:
        return None
    ordered = sorted(values)
    return {'n': len(values), 'min': ordered[0], 'median': statistics.median(values),
            'p95_nearest_rank': ordered[max(0, (95 * len(values) + 99) // 100 - 1)],
            'max': ordered[-1]}


def stability_summary(data):
    state = data.get('stability', {})
    runs = data.get('runs', [])[state.get('first_run_index', len(data.get('runs', []))):]
    rows = []
    for i, run in enumerate(runs, 1):
        m = run['metrics']
        rows.append({'turn': i, **{k: m[k] for k in (
            'prompt_tokens', 'reused_tokens', 'new_tokens', 'generated_tokens',
            'ttft_seconds', 'total_seconds', 'decode_tokens_per_second', 'rss_kib',
            'driver_bytes', 'temperature_start_mC', 'temperature_end_mC')},
            'dropped_messages': run['dropped_messages'],
            'http_first_text_seconds': run['http_first_text_seconds'],
            'http_total_seconds': run['http_total_seconds'],
            'answer': run['answer']})
    def group(label, selected):
        return {'group': label, 'turns': len(selected),
                'ttft_seconds': dist([r['ttft_seconds'] for r in selected]),
                'decode_tokens_per_second': dist([r['decode_tokens_per_second'] for r in selected
                                                  if r['generated_tokens'] > 1])}
    groups = [group('cached', [r for r in rows if r['reused_tokens'] > 0]),
              group('window-rebuild', [r for r in rows if r['dropped_messages'] > 0]),
              group('uncached-other', [r for r in rows if not r['reused_tokens'] and not r['dropped_messages']])]
    summary = {'status': state.get('status', data['status']),
               'elapsed_seconds': state.get('elapsed_seconds', 0), 'turns': len(rows),
               'completed_1800_seconds': state.get('status') == 'complete' and state.get('elapsed_seconds', 0) >= 1800,
               'driver_released_to_zero': state.get('closed', {}).get('driver_bytes') == 0,
               'source_path': data['source_path'], 'source_sha256': data.get('source_sha256'),
               'groups': groups, 'rows': rows,
               'quality_accepted': False, 'audio_concurrency': False}
    summary['provenance_scope'] = data.get('provenance_scope',
        'legacy hashes sampled after queue wait; Python files may have changed since launch; not guaranteed loaded Python bytes')
    if rows:
        q = max(1, len(rows) // 4)
        summary.update(rss_mib=dist([r['rss_kib'] / 1024 for r in rows]),
                       driver_mib=dist([r['driver_bytes'] / 2**20 for r in rows]),
                       temperature_c=dist([r[k] / 1000 for r in rows for k in
                                           ('temperature_start_mC', 'temperature_end_mC')]),
                       rss_last_minus_first_quarter_mib=(statistics.median(r['rss_kib'] for r in rows[-q:]) -
                                                        statistics.median(r['rss_kib'] for r in rows[:q])) / 1024)
    if data.get('error'):
        summary['error'] = data['error']
    return summary


def plot_stability(model, summary):
    if not summary['rows']:
        return None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.hashsalt': 'opi-resident-stability-v1', 'font.size': 9})
    rows = summary['rows']
    turns = [r['turn'] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(turns, [r['ttft_seconds'] for r in rows], color='#267ca5', linewidth=1)
    reset = [r for r in rows if r['dropped_messages']]
    ax.scatter([r['turn'] for r in reset], [r['ttft_seconds'] for r in reset], s=10,
               color='#cf6334', label='Whole-turn eviction + rebuild', zorder=3)
    ax.set_ylabel('Engine first token (s)')
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.plot(turns, [r['decode_tokens_per_second'] for r in rows], color='#267ca5')
    ax.axhline(6, color='#999999', linestyle=':', label='Approximate speech reference: 6 token/s')
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Decode (token/s)')
    ax.legend(fontsize=8)
    axes[1, 0].plot(turns, [r['rss_kib']/1024 for r in rows], color='#267ca5')
    axes[1, 0].set_ylabel('Board host process RSS (MiB)')
    d = summary['driver_mib']
    axes[1, 0].set_title(f'NPU driver separately: {d["min"]:.2f}–{d["max"]:.2f} MiB', fontsize=9)
    axes[1, 1].plot(turns, [r['temperature_start_mC']/1000 for r in rows], label='Turn start')
    axes[1, 1].plot(turns, [r['temperature_end_mC']/1000 for r in rows], label='Turn end')
    axes[1, 1].set_ylabel('SoC temperature (°C)')
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel('Natural conversation turn')
        ax.grid(alpha=.2)
    fig.suptitle(f'{model}: resident full-depth NPU, capacity 1280\n'
                 f'{summary["status"]}; {summary["elapsed_seconds"]:.1f} s; host HTTP over SSH; no temperature conditioning', fontsize=11)
    name = f'resident-{model}-stability'
    fig.savefig(HERE / (name + '.png'), dpi=160)
    svg = HERE / (name + '.svg')
    fig.savefig(svg, metadata={'Date': None})
    svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    return name + '.png'


def main():
    evidence = {'scope': 'full-depth resident NPU; host tokenizer and HTTP over SSH',
                'product_accepted': False, 'models': {}}
    lines = ['# 独立质量与连续多轮验收', '',
             '完整500M语言32层、1B语言24层及完整输出头。NPU与KV常驻开发板，分词和HTTP网关在工作电脑，SSH连接。未替换生产8080，也不是板端独立网页交付。', '',
             '30分钟持续执行、数值质量和陪伴对话质量分别判断。完成时长不代表能自然聊天；主机侧标准Transformers反量化Q8用于质量锚，不是板端优化CPU速度。[真实ASR并发](真实ASR与LLM并发验收.md)另有配对实测，仅对该文标明的图版本有效。', '',
             '## 连续多轮', '',
             '| 模型 | 状态 | 已测秒 / 轮 | 末次退出驱动归零 | RSS范围 MiB | RSS后1/4−前1/4 MiB | 温度范围 °C |',
             '| --- | --- | ---: | --- | ---: | ---: | ---: |']
    for model, label in [('smol500', '500M'), ('intern1b', '1B')]:
        row = evidence['models'].setdefault(model, {'label': label})
        data = snapshot(ROOT / f'resident-stability-{model}-20261013/result.json')
        if not data:
            lines.append(f'| {label} | 等待独占板卡 | 0 / 0 | 未测 | — | — | — |')
            continue
        s = row['stability'] = stability_summary(data)
        fmt_range = lambda k: f'{s[k]["min"]:.2f}–{s[k]["max"]:.2f}' if k in s else '—'
        delta = s.get('rss_last_minus_first_quarter_mib')
        delta_text = f'{delta:.3f}' if delta is not None else '—'
        lines.append(f'| {label} | {s["status"]} | {s["elapsed_seconds"]:.1f} / {s["turns"]} | '
                     f'{"是" if s["driver_released_to_zero"] else "未确认"} | {fmt_range("rss_mib")} | '
                     f'{delta_text} | {fmt_range("temperature_c")} |')
    lines += ['', 'RSS是板端宿主进程，驱动分配另列，不能把RSS当成NPU全部内存。记录的是每轮边界温度/RSS，不是持续采样峰值。前后区间内存变化只作泄漏线索，不自动定义通过。', '',
              '早期队列在获取板卡锁后才采源码哈希；等待期间Python文件若已更新，该哈希不保证等于已加载Python字节。原始结果保留此溯源限制；后续运行改为启动时快照Python、部署时快照C。行为参数和板端结果单独记录。', '',
              '| 模型 / 路径 | 轮数 | 首token中位 / P95 / 最大 s | decode中位 token/s |',
              '| --- | ---: | ---: | ---: |']
    for model, row in evidence['models'].items():
        s = row.get('stability', {})
        for g in s.get('groups', []):
            if not g['turns']:
                continue
            t, d = g['ttft_seconds'], g['decode_tokens_per_second']
            decode = f'{d["median"]:.3f}' if d else '无decode间隔'
            lines.append(f'| {row["label"]} / {g["group"]} | {g["turns"]} | '
                         f'{t["median"]:.3f} / {t["p95_nearest_rank"]:.3f} / {t["max"]:.3f} | {decode} |')
    for model, row in evidence['models'].items():
        if 'stability' in row:
            figure = plot_stability(model, row['stability'])
            if figure:
                lines += ['', f'![{row["label"]}连续多轮]({figure})', '']
    lines += ['', 'cached为实际前缀复用；window-rebuild为超窗整轮裁剪后的历史重建；uncached-other含首轮。重建轮不能归入32/64/128新增缓存命中首字。自然生成长度与内容不同，此表用于发现产品延迟，不与固定128步五轮CPU成绩直接相除。', '',
              '窗口满后当前实现重建保留历史，因此会出现明显首字尖峰。后续可比较提前裁剪保留余量、不同保留历史长度和位置/KV重用方案；需注明记忆损失并重验质量，不能隐去重建耗时。', '',
              '## 超窗保留余量试验', '',
              '默认策略保持不变。新试验只在实际超窗后按完整轮次裁剪到约768个prompt token，保留system和最新问题。比较原30分钟的前20轮与新策略20轮，各自使用真实生成历史；这会牺牲旧记忆，且后续回答轨迹可能不同，不当作同输入的内核加速比。', '',
              '| 模型 / 策略 | 状态 / 已测轮 | 重建次数 | 首token中位 / P95 s | 累计首token等待 s | prompt中位token / 丢弃消息数 |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for model,row in evidence['models'].items():
        data=snapshot(ROOT/f'resident-retain768-{model}-20261013/result.json')
        trial=stability_summary(data) if data else None
        row['retention_experiment']=trial or {'status':'queued'}
        if trial:
            trial['retain_after_overflow']=data.get('retain_after_overflow')
            trial['scope']='20-turn development experiment; own generated histories; not independent holdout or kernel speed'
        for label,summary in [('原策略前20轮',row.get('stability')),('超窗保留768',trial)]:
            if not summary or not summary.get('rows'):
                lines.append(f'| {row["label"]} / {label} | 等待实测 | — | — | — | — |');continue
            rows=summary['rows'][:20]
            first=dist([r['ttft_seconds'] for r in rows])
            lines.append(f'| {row["label"]} / {label} | {summary["status"]} / {len(rows)} | '
                         f'{sum(r["dropped_messages"]>0 for r in rows)} | {first["median"]:.3f} / {first["p95_nearest_rank"]:.3f} | '
                         f'{sum(r["ttft_seconds"] for r in rows):.3f} | {statistics.median(r["prompt_tokens"] for r in rows):.0f} / {sum(r["dropped_messages"] for r in rows)} |')
    lines += ['', '单轮重建时延、记忆保留、回答质量和重建频率需一起判断；少重建但丢更多旧对话不能称为无代价收益。试验未接入生产8080。', '',
              '## 独立留出', '',
              '冻结的v1为12个中英场景，每个3轮。固定标准Q8参考轨迹检查全部logits；自然生成另用NPU自己的历史。门槛KL≤0.05、PPL比≤1.05，人工对话评阅另列。看到留出后不得据其失败调参并继续声称v1独立。', '',
              '| 模型 | 状态 | 数值案例 / 36 | 聚合KL | 聚合PPL比 | 聚合通过 / 每例通过 |',
              '| --- | --- | ---: | ---: | ---: | --- |']
    for model, row in evidence['models'].items():
        ref = snapshot(ROOT / f'holdout-reference-{model}-20261013/state.json')
        data = snapshot(ROOT / f'holdout-npu-{model}-20261013/result.json')
        h = data.get('holdout', {}) if data else {}
        row['holdout'] = {'status': h.get('status', data.get('status') if data else 'queued'),
                          'source_path': data.get('source_path') if data else None,
                          'source_sha256': data.get('source_sha256') if data else None,
                          **h}
        cpu = snapshot(ROOT / f'holdout-cpu-{model}-20261013/state.json')
        row['board_cpu_natural'] = cpu or {'status': 'not-started', 'cases': []}
        if ref and 'cases' in ref:
            row['quality_reference'] = {k: v for k, v in ref.items() if k != 'cases'}
            row['quality_reference']['cases'] = [{k: v for k, v in c.items() if k not in
                                                ('prompt_tokens', 'feedback_tokens')} for c in ref['cases']]
        f = lambda k: f'{h[k]:.6f}' if k in h else '待完成'
        passed = lambda k: ('通过' if h[k] else '未通过') if k in h else '未验收'
        lines.append(f'| {row["label"]} | {row["holdout"]["status"]} | {len(h.get("cases", []))} | '
                     f'{f("mean_kl_nats")} | {f("perplexity_ratio")} | {passed("numeric_passed")} / {passed("all_cases_numeric_passed")} |')
    for model, row in evidence['models'].items():
        cpu = row['board_cpu_natural']
        lines += ['', f'{row["label"]}实际板端llama.cpp自然生成：`{cpu["status"]}`，'
                  f'{len(cpu.get("cases", []))}/36条。该任务用于配对质量，配置和原文独立记录，不作为最快CPU性能成绩。']
        h = row['holdout']
        cases = h.get('cases', [])
        if cases:
            lines += ['', '### ' + row['label'] + '数值例外', '']
            failed = [c for c in cases if not c['numeric_passed']]
            if failed:
                lines += ['| 案例 | KL | PPL比 | argmax一致率 |', '| --- | ---: | ---: | ---: |']
                lines += [f'| {c["case"]} | {c["mean_kl_nats"]:.6f} | {c["perplexity_ratio"]:.6f} | {c["argmax_fraction"]:.2%} |' for c in failed]
            else:
                lines += ['当前已完成案例未发现数值超限；不足36条时仍是部分结果。']
        if h.get('natural'):
            lines += ['', '### ' + row['label'] + '自然回答原文', '',
                      '三种实现各用自己的历史，因此后轮上下文可能不同。此处透明列出原文供评阅，不自动把文本差异判成错误。', '']
            refs = {c['case']: c for c in row['quality_reference']['cases']}
            cpus = {c['case']: c for c in cpu.get('cases', [])}
            for c in h['natural']:
                r = refs[c['case']]
                lines += ['**' + c['case'] + '**：' + r['messages'][-1]['content'], '',
                          '标准Q8参考：', '', '> ' + r['reference_answer'].replace('\n', '\n> '), '',
                          'NPU：', '', '> ' + (c['answer'] or '（空回答）').replace('\n', '\n> '), '']
                if c['case'] in cpus:
                    lines += ['板端llama.cpp Q8：', '', '> ' + (cpus[c['case']]['answer'] or '（空回答）').replace('\n', '\n> '), '']
    review_path=HERE/'quality-review-v1.json'
    if review_path.exists():
        review=json.loads(review_path.read_text())
        assert review['suite_sha256']==hashlib.sha256((HERE/'quality-holdout-v1.json').read_bytes()).hexdigest()
        evidence['qualitative_review']=review
        lines += ['', '## 对话原文质性评阅', '',
                  '以下为Codex按完整三轮原文逐项核对，未做盲评或人群统计。“满足”只指本场景要求，不代表模型整体质量；标准参考是反量化Q8的FP32，实际板端CPU回答另列。']
        for model,reviewed in review['models'].items():
            model_evidence=evidence['models'][model]
            assert reviewed['reference_sha256']==model_evidence['quality_reference']['source_sha256']
            assert reviewed['npu_result_sha256']==model_evidence['holdout']['source_sha256']
            cpu_review=reviewed.get('board_cpu_review',{})
            if isinstance(cpu_review,dict):
                assert cpu_review['source_sha256']==model_evidence['board_cpu_natural']['source_sha256']
            else:
                cpu_review={}
            lines += ['', model_evidence['label']+'：'+reviewed['summary'], '',cpu_review.get('summary','实际CPU尚待评阅。'),'',
                      '| 场景 | 标准Q8参考 | 板端CPU Q8 | NPU | 原文观察 |','| --- | --- | --- | --- | --- |']
            for c in reviewed['scenarios']:
                cpu_result=cpu_review.get('scenario_results',{}).get(c['id'],'待评阅')
                lines.append(f'| {c["id"]} | {c["reference"]} | {cpu_result} | {c["npu"]} | {c["observation"]} |')
    lines += ['', '产品状态：未验收。上述对话评阅已记录实际CPU/NPU差异与共同失败；其余矩阵、板端独立服务和真实音频并发按各项证据推进。聚合数值通过不能覆盖单例退化或真实对话失败。', '',
              '复现：`npu/.venv/bin/python npu/kv-cache/report_product_acceptance.py`。',
              '[紧凑证据](evidence-attack-20261013/product-acceptance.json)。', '']
    (HERE / '独立质量与连续多轮验收.md').write_text('\n'.join(line.rstrip() for line in '\n'.join(lines).splitlines())+'\n')
    (HERE / 'evidence-attack-20261013/product-acceptance.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    print({k: {'stability': v.get('stability', {}).get('status'), 'holdout': v['holdout']['status']}
           for k, v in evidence['models'].items()})


if __name__ == '__main__':
    main()
