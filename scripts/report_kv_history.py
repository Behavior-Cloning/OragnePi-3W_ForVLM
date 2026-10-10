#!/usr/bin/env python3
"""Validate the final incremental KV experiment and render its Markdown/plots."""
import argparse
import gzip
import json
from pathlib import Path


def table(headers, rows):
    def cell(v):
        return f'{v:.2f}' if isinstance(v, float) else str(v)
    return ['| ' + ' | '.join(headers) + ' |',
            '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + [
                '| ' + ' | '.join(map(cell, row)) + ' |' for row in rows] + ['']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    out = args.input.parent
    assert data['status'] == 'COMPLETE' and data['production_restored']
    assert len(data['models']) == 6
    rows = []
    samples = set()
    for model in data['models']:
        assert {c['context'] for c in model['contexts']} == {1024, 4096, 8192}
        for ctx in model['contexts']:
            assert len(ctx['cases']) == 4
            assert {c['requested_occupancy'] for c in ctx['cases']} == {0, .25, .5, .9}
            for case in ctx['cases']:
                inc = case['incremental']
                run = inc['run']
                t = run['timings']
                assert inc['status'] == 'COMPLETE' and not run.get('error')
                assert run['cache_prompt'] and t['cache_n'] > 0 and t['predicted_n'] == 128
                assert t['cache_n'] + t['prompt_n'] == inc['input_tokens']
                sample = out / 'samples' / Path(run['samples_file']).name
                with gzip.open(sample, 'rt') as f:
                    assert json.load(f)
                samples.add(sample)
                rows.append((model, ctx, case, inc, run))
    assert len(rows) == len(samples) == 72
    lines = ['# KV cache 增量对话：最终性能报告', '',
             '日期：2026-10-10。六模型 × 三档窗口 × 四档历史，共72/72项完成；72份压缩采样均已验证可解析。', '',
             '## 测量口径', '',
             'A733 / 6GB，已安装风扇；CPU6/7、2线程、单槽，未使用NPU。纯文本输入，VLM仍加载网页配置中的视觉投影器。',
             '先加载模拟中文历史并生成32 token作为不计分准备，再追加真实生成的助手回答和新问题“我想再听一点，接着说吧。”；仅统计这一轮KV缓存增量推理。未将完整历史重新预填充耗时混入结果。',
             '每次强制生成128 token，greedy、ignore_eos=true、repeat_penalty=1；这是性能测试，回答内容不用于质量评分。Qwen3关闭思考。',
             '窗口为1024/4096/8192；目标历史档位为短历史/25%/50%/90%。实际输入含模板、准备回答及新问题，短历史不等于0 token；小窗口为输出预留空间，实际占用以表格为准。不同tokenizer的token数不能直接当作相同字数。',
             '每格仅一次观测，无置信区间；顺序运行的温度与频率会影响结果。TTFT是客户端首个输出延迟，解码token/s采用服务端timings；总耗时含首字等待及生成。', '',
             '## 关键结论', '',
             'KV cache确实生效：全部72项都有缓存命中。但解码每个新token仍需访问较长的历史KV，历史增长后吞吐下降；缓存不等于长对话零成本。',
             'SmolVLM和Qwen3的部分准备回答未被复用；表中列出实际缓存/新算token数。模板或重新分词可能影响前缀匹配，但本实验未隔离根因，不能直接归因于未实现KV cache。',
             'InternVL3-1B与Qwen2.5-1.5B适合继续优化中文陪伴。3B更慢且内存更高；Qwen3在长历史下首字延迟尤为明显。性能不能替代对话质量验收。', '']
    summaries = []
    for window in (4096, 8192):
        part = [f'## {window}窗口：短历史与最长历史', '']
        values = []
        for m in data['models']:
            group = [r for r in rows if r[0]['id'] == m['id'] and r[1]['context'] == window]
            a, b = group[0], group[-1]
            values.append([m['id'], a[3]['input_tokens'], a[4]['ttft_seconds'], a[4]['timings']['predicted_per_second'],
                           b[3]['input_tokens'], b[4]['ttft_seconds'], b[4]['timings']['predicted_per_second'], b[4]['resources']['peak_rss_mib']])
        part += table(['模型', '短输入token', '短首字秒', '短token/s', '长输入token', '长首字秒', '长token/s', '长RSS MiB'], values)
        lines += part
        summaries += part
    lines += ['## 历史长度趋势', '', '![Latency and throughput](kv-performance.png)', '',
              '![Memory and CPU cost](kv-resources.png)', '',
              '横轴为新增用户轮次请求的实际输入token数，每条线对应一个配置窗口；图中点为单次实测。', '',
              '解码图红色虚线为6 token/s的粗略中文口语参考线，不是实测语速或统一标准。日常说话可粗估约3–5汉字/秒；若分词约1–2 token/汉字，对应约3–10 token/s。这里取6作为便于比较的参考点，各模型应以实际输出的汉字/token比例换算。',
              '高于参考线只表示持续生成吞吐可能跟得上口播，不保证对话自然或开口快；仍需单独看TTFT、分句缓冲和TTS延迟。本轮没有测试语音合成链路。', '',
              '## 配置加载与空闲', '',
              '加载时间是服务切换到ready的时间；未清空系统页缓存，不能称为断电冷启动。空闲内存取采样峰值。', '']
    lines += table(['模型', '窗口', '加载秒（非冷启动）', '空闲RSS MiB', '空闲cgroup MiB', '空闲最低可用 MiB'], [
        [m['id'], c['context'], c['startup']['ready_seconds'], c['idle']['resources']['peak_rss_mib'],
         c['idle']['resources']['peak_cgroup_mib'], c['idle']['resources']['min_available_mib']]
        for m in data['models'] for c in m['contexts']])
    for m in data['models']:
        group = [r for r in rows if r[0]['id'] == m['id']]
        lines += [f"## {m['id']}", '', '### 延迟、缓存和吞吐', '']
        lines += table(['窗口', '目标历史%', '历史轮对', '实际输入token', '实际占窗%', '缓存token', '新算token', '首字秒', '增量prefill秒', '解码token/s', '总秒'], [
            [c['context'], k['requested_occupancy'] * 100, k['history_turn_pairs'], i['input_tokens'], i['input_tokens'] / c['context'] * 100,
             r['timings']['cache_n'], r['timings']['prompt_n'], r['ttft_seconds'], r['timings']['prompt_ms'] / 1000,
             r['timings']['predicted_per_second'], r['total_seconds']] for _, c, k, i, r in group])
        lines += ['### 内存与交换', '']
        keys = ['peak_rss_mib', 'peak_cgroup_mib', 'min_available_mib', 'peak_system_used_mib', 'peak_swap_used_mib']
        lines += table(['窗口', '输入token', 'RSS峰值 MiB', 'cgroup峰值 MiB', '整机最低可用 MiB', '整机已用峰值 MiB', '整机Swap峰值 MiB'], [
            [c['context'], i['input_tokens']] + [r['resources'][key] for key in keys] for _, c, k, i, r in group])
        lines += ['### CPU、温度与原始采样', '']
        keys = ['cpu_seconds', 'cpu_percent_one_core_100', 'system_cpu_busy_percent', 'system_iowait_percent', 'peak_temperature_c']
        lines += table(['窗口', '输入token', 'CPU秒', '进程CPU%', '整机CPU%', 'IOwait%', '最高°C', 'A76最低MHz', 'A76最高MHz', '采样'], [
            [c['context'], i['input_tokens']] + [r['resources'][key] for key in keys] +
            [r['a76_frequency_khz']['min'] / 1000, r['a76_frequency_khz']['max'] / 1000,
             '[JSON.gz](samples/' + Path(r['samples_file']).name + ')'] for _, c, k, i, r in group])
    lines += ['## 解释与局限', '',
              '进程CPU以单核100%计，约200%表示占满两个核；整机CPU以8核合计100%计，两者并不矛盾。CPU秒是计算时间，不是电能。',
              'RSS、cgroup内存和整机可用内存口径不同，不能简单相加或用RSS推算剩余内存。Swap是整机已用峰值，可能含前次任务遗留，不代表本模型新增交换量。',
              '装风扇后仍有接近90°C的记录；没有受控的装风扇前后对照，不能宣称温度改善幅度或排除热影响。',
              '本轮未测电功率/能耗、统一断电冷启动、多并发、图像理解、超窗裁剪后的缓存损失或多seed重复；这些不在72项已完成的范围内。', '',
              '## 复现与证据', '',
              '- [完整原始结果（含各模型manifest）](results.json)',
              '- [固定模型来源与版本](../../../configs/vlm-models.json)',
              '- [测试脚本](../../../scripts/evaluate_history.py)',
              '- [本报告生成脚本](../../../scripts/report_kv_history.py)', '',
              '在仓库根目录运行：`python3 scripts/report_kv_history.py docs/benchmarks/2026-10-10-kv-fan/results.json`。', '']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for filename, metrics in [('kv-performance.png', [('ttft_seconds', 'First token (s)'), ('predicted_per_second', 'Decode (token/s)')]),
                              ('kv-resources.png', [('peak_rss_mib', 'Peak RSS (MiB)'), ('cpu_seconds', 'CPU time (s)')])]:
        fig, axes = plt.subplots(2, 6, figsize=(20, 7), constrained_layout=True)
        for col, m in enumerate(data['models']):
            for row, (key, label) in enumerate(metrics):
                ax = axes[row, col]
                for window in (1024, 4096, 8192):
                    group = [r for r in rows if r[0]['id'] == m['id'] and r[1]['context'] == window]
                    ys = [r[4].get(key, r[4]['timings'].get(key, r[4]['resources'].get(key))) for r in group]
                    ax.plot([r[3]['input_tokens'] for r in group], ys, 'o-', label=str(window), markersize=3)
                if key == 'predicted_per_second':
                    ax.axhline(6, color='#bd3030', linestyle='--', linewidth=1.4)
                    ax.text(.98, 6, 'Speech ref: ~6 tok/s', color='#a02020',
                            transform=ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=8)
                    ax.set_ylim(bottom=0, top=max(ax.get_ylim()[1], 6.9))
                ax.set_title(m['id'], fontsize=9)
                ax.set_ylabel(label)
                ax.set_xlabel('Actual input tokens')
                ax.grid(alpha=.25)
                if col == 0 and row == 0:
                    ax.legend(title='Context window', fontsize=8)
        fig.savefig(out / filename, dpi=150)
        plt.close(fig)
    (out / 'KV缓存增量性能报告.md').write_text('\n'.join(lines))
    (out / '性能摘要.md').write_text('# KV性能摘要\n\n' + '\n'.join(summaries))
    print('Validated 72/72 cached incremental runs and 72 sample files; reports and plots generated.')


if __name__ == '__main__':
    main()
