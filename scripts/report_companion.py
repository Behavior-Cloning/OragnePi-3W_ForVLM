#!/usr/bin/env python3
"""Render recorded companion replies and observed metrics; never manufacture quality scores."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    lines = ['# 陪伴对话板端实测', '', f"运行状态：{report['status']}。质量分未自动生成，需结合下方真实多轮对话评审。",
             '', '4096上下文，2个A76核心；统一角色、temperature=0.7、top_p=0.8、top_k=20、repeat_penalty=1.1。',
             '首字与解码速度取成功停止的回答中位数；不同回答长度和实际提示词长度会影响结果。',
             'CPU以单核100%计；温度、RSS和Swap为请求采样峰值，非整机电功率。启动受文件缓存影响，不是严格冷启动。', '',
             '| 模型 | 正常停止/已执行轮数 | 首字中位秒 | 解码中位token/s | 峰值RSS MiB | 峰值温度°C | 峰值Swap MiB | 平均CPU中位% | 启动秒 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    def number(values, operation=statistics.median):
        values = [v for v in values if isinstance(v, (int, float))]
        return f'{operation(values):.2f}' if values else '—'
    for model in report['models']:
        turns = [t for c in model['conversations'] for t in c['turns']]
        good = [t for t in turns if not t.get('error') and t.get('finish_reason') == 'stop' and not t.get('nonthinking_violation')]
        cells = [model['id'], f'{len(good)}/{len(turns)}', number([t.get('ttft_seconds') for t in good]),
                 number([t.get('timings', {}).get('predicted_per_second') for t in good])]
        for key in ('peak_rss_mib', 'peak_temperature_c', 'peak_swap_used_mib'):
            cells.append(number([t['resources'].get(key) for t in turns], max))
        cells += [number([t['resources'].get('cpu_percent_one_core_100') for t in turns]),
                  number([model.get('startup', {}).get('ready_seconds')])]
        lines.append('| ' + ' | '.join(cells) + ' |')
    for model in report['models']:
        lines += ['', '## ' + model['id'], '']
        turns = [t for c in model['conversations'] for t in c['turns']]
        fresh = [t.get('ttft_seconds') for t in turns if t.get('timings', {}).get('cache_n') == 0]
        cached = [t.get('ttft_seconds') for t in turns if t.get('timings', {}).get('cache_n', 0) > 0]
        prompts = [t.get('usage', {}).get('prompt_tokens') for t in turns]
        output = [t.get('usage', {}).get('completion_tokens') for t in turns]
        frequencies = [s.get('cpu_freq_khz', {}).get('policy6') for t in turns for s in t.get('samples', [])]
        lines += [f"无前缀缓存首字中位：{number(fresh)}秒（{len(fresh)}轮）；有前缀缓存：{number(cached)}秒（{len(cached)}轮）。",
                  f"输入token范围：{number(prompts, min)}–{number(prompts, max)}；输出token中位：{number(output)}。",
                  f"A76采样频率范围：{number(frequencies, min)}–{number(frequencies, max)} kHz。", '']
        if model.get('error'):
            lines += ['运行错误：' + model['error'], '']
        for conversation in model['conversations']:
            lines += ['### ' + conversation['id'], '', f"运行有效：{conversation['valid']}", '']
            for index, turn in enumerate(conversation['turns'], 1):
                lines += [f"**用户 {index}：** {turn['user']}", '', '**模型：**', '',
                          *['> ' + line for line in turn['answer'].splitlines()], '',
                          f"结束={turn.get('finish_reason')}；输入/输出usage={json.dumps(turn.get('usage', {}), ensure_ascii=False)}", '']
                if turn.get('error') or turn.get('nonthinking_violation'):
                    lines += [f"异常：{turn.get('error') or '非思考模式出现推理内容'}", '']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
