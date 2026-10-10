#!/usr/bin/env python3
"""Generate reviewable engineering tables/figures from raw board evaluation JSON."""
import argparse
import json
from pathlib import Path
import statistics as stats


def median(values):
    values=[v for v in values if v is not None]
    return stats.median(values) if values else None


def number(value):
    return '—' if value is None else f'{value:.2f}'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('inputs',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    reports=[json.loads(p.read_text()) for p in args.inputs]
    models=[m for report in reports for m in report['models']]
    lines=['# 板端系统与上下文分档实测','',
           '本文件由 `scripts/report_vlm_system.py` 从原始 JSON 生成；只展示实际记录，不外推未完成档位。',
           'CPU-only，A76 核 6/7，2 线程。语言量化：256M/500M/1B 为 Q8，3B 为 Q4_K_M；视觉均 F16。',
           '每次最多输出 128 token；固定 temperature=0.2、seed=42。网页常用输出上限为 256，不能混用总延迟。',
           '未测瓦数/焦耳。质量样例不等于标准数据集准确率；一张自然照片、图形与票据只覆盖有限视觉场景。','',
           '## 总览','',
           '| 模型 | 已记录请求 | 错误请求 | 文字解码中位数 token/s | 首图首字中位数 s | 首图完成中位数 s | 峰值 RSS MiB | 峰值温度 °C |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for m in models:
        tests=m['tests'];ok=[t for t in tests if not t.get('error')]
        texts=[t for t in ok if t['case'] in ['text_zh','text_en']]
        images=[t for t in ok if t['case'].startswith('first_image_')]
        lines.append('| '+' | '.join([m['id'],str(len(tests)),str(sum(bool(t.get('error')) for t in tests)),
            number(median([t.get('timings',{}).get('predicted_per_second') for t in texts])),
            number(median([t.get('ttft_seconds') for t in images])),number(median([t.get('total_seconds') for t in images])),
            number(max((t.get('resources',{}).get('peak_rss_mib',0) for t in tests),default=0)),
            number(max((t.get('resources',{}).get('peak_temperature_c',0) for t in tests),default=0))])+' |')
    lines += ['', '“首图”三次分别来自初始、保留文件缓存重启、清除权重文件缓存重启，不是三个完全相同启动条件。',
              '先完成下载再测；各服务重启前等待降温，但同一轮连续请求自然升温。长上下文档位按顺序递增，温度效应与长度效应并存。',
              '不能把全部速度差异都归因于 attention 计算量；原始采样包含频率、温度、系统内存与 CPU。','']
    for m in models:
        lines += ['## '+m['id'],'']
        if m.get('error'):lines += ['本模型未完整完成：`'+m['error'].replace('\n',' ')+'`','']
        idle=m.get('idle',{}).get('resources',{})
        lines += [f"加载后 10 秒空闲：RSS {number(idle.get('peak_rss_mib'))} MiB，模型 CPU {number(idle.get('cpu_percent_one_core_100'))}%（单核=100%）。",'',
                  '| 启动条件 | 就绪 s | 峰值 RSS MiB | CPU 秒 |','| --- | ---: | ---: | ---: |']
        for r in m['startups']:
            lines += [f"| {r['mode']} | {number(r['ready_seconds'])} | {number(r['resources'].get('peak_rss_mib'))} | {number(r['resources'].get('cpu_seconds'))} |"]
        lines += ['','### 上下文分档','',
                  '短对话、约25%、50%、90%占用。为保留128输出token及余量，1024窗口最高档实际低于90%；以实际输入列为准。',
                  '首次列指该档位新提示的首次处理（系统前缀可能仍命中）；缓存列是随后两次同一输入的中位数。',
                  '输入按模型自身模板/tokenizer 构造，最终实际计数采用后端 usage；重复填充是性能负载，不代表真实长文理解准确率。','',
                  '| 窗口 | 目标占用 | 实际输入 token | 首次缓存 token | 首次首字 s | 缓存首字 s | 首次 token/s | 缓存 token/s | 输出 token 范围 | 峰值 RSS MiB | 峰温 °C |',
                  '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for context in [1024,4096,8192]:
            for occupancy in [0,.25,.5,.9]:
                group=[t for t in m['tests'] if t['case']=='context_speed' and t['context']==context and t['occupancy']==occupancy and not t.get('error')]
                if not group:continue
                first=group[0];warm=group[1:]
                counts=[t.get('usage',{}).get('completion_tokens',t.get('timings',{}).get('predicted_n',0)) for t in group]
                lines.append('| '+' | '.join([str(context),'短' if not occupancy else f'{occupancy:.0%}',
                    str(first.get('usage',{}).get('prompt_tokens',first.get('tokenizer_input_count'))),str(first.get('timings',{}).get('cache_n','—')),
                    number(first.get('ttft_seconds')),number(median([t.get('ttft_seconds') for t in warm])),
                    number(first.get('timings',{}).get('predicted_per_second')),number(median([t.get('timings',{}).get('predicted_per_second') for t in warm])),
                    f'{min(counts)}–{max(counts)}',number(max(t['resources'].get('peak_rss_mib',0) for t in group)),number(max(t['resources'].get('peak_temperature_c',0) for t in group))])+' |')
        lines += ['','### 各请求指标与负载','',
                  '模型 CPU 百分比按单核100%；整机 CPU 按全8核满载100%，不含 iowait。RSS 与 cgroup 内存不能相加。',
                  '首字含预处理、视觉编码、prefill；解码 token/s 不含首字等待。CPU秒是累计计算开销，不是耗电量。','',
                  '| 场景 / 次数 | 窗口 | 首字 s | 总时长 s | token/s | 输出 token | 结束原因 | CPU秒 | 模型CPU % | 整机CPU % | iowait % | RSS MiB | cgroup MiB | 可用RAM MiB | Swap MiB | 温度 °C |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for t in m['tests']:
            r=t.get('resources',{});tim=t.get('timings',{})
            label=f"{t['case']} / {t['repetition']}"+(f" / {t['occupancy']}" if 'occupancy' in t else '')
            lines.append('| '+' | '.join([label,str(t['context']),number(t.get('ttft_seconds')),number(t.get('total_seconds')),number(tim.get('predicted_per_second')),str(tim.get('predicted_n','—')),t.get('finish_reason','error' if t.get('error') else '—')]+[number(r.get(k)) for k in ['cpu_seconds','cpu_percent_one_core_100','system_cpu_busy_percent','system_iowait_percent','peak_rss_mib','peak_cgroup_mib','min_available_mib','peak_swap_used_mib','peak_temperature_c']])+' |')
        lines += ['','### 原始效果样例','']
        for t in m['tests']:
            if t['case'] in ['arithmetic','memory_recall','unknown','visual_shapes','visual_receipt'] or t['case'].startswith('first_image_') or t.get('error'):
                answer=t.get('error') or t.get('answer','')
                lines += [f"- **{t['case']} / {t['repetition']}**：{answer.replace(chr(10),' ')}"]
        lines += ['']
    lines += ['## 原始数据','']+[f'- `{p.name}`' for p in args.inputs]
    lines += ['','文件保留每100ms采样、全部答案、prompt、缓存计数、结束原因、启动命令和服务日志。',
              '服务启动不是整板断电冷启动；本次没有重新进行上电测试。','']
    args.output.write_text('\n'.join(lines))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure,axes=plt.subplots(1,3,figsize=(15,4.5),sharey=True)
        for ax,ctx in zip(axes,[1024,4096,8192]):
            for m in models:
                xs,ys=[],[]
                for occ in [0,.25,.5,.9]:
                    group=[t for t in m['tests'] if t['case']=='context_speed' and t['context']==ctx and t['occupancy']==occ and t['repetition']>1 and not t.get('error')]
                    if group:
                        xs.append(median([t.get('usage',{}).get('prompt_tokens',t['tokenizer_input_count']) for t in group]))
                        ys.append(median([t['timings']['predicted_per_second'] for t in group]))
                if xs:ax.plot(xs,ys,marker='o',label=m['id'])
            ax.set_title(f'Context window {ctx}');ax.set_xlabel('Actual input tokens');ax.grid(alpha=.25)
        axes[0].set_ylabel('Decode tokens/s (cached-repeat median)')
        axes[-1].legend(fontsize=8);figure.tight_layout()
        figure.savefig(args.output.with_suffix('.png'),dpi=160);plt.close(figure)
        figure,axes=plt.subplots(1,3,figsize=(15,4.5),sharey=True)
        for ax,ctx in zip(axes,[1024,4096,8192]):
            for m in models:
                group=[t for t in m['tests'] if t['case']=='context_speed' and t['context']==ctx and t['repetition']==1 and not t.get('error')]
                if group:ax.plot([t.get('usage',{}).get('prompt_tokens',t['tokenizer_input_count']) for t in group],
                                 [t['ttft_seconds'] for t in group],marker='o',label=m['id'])
            ax.set_title(f'Context window {ctx}');ax.set_xlabel('Actual input tokens');ax.grid(alpha=.25)
        axes[0].set_ylabel('First request TTFT (seconds)');axes[-1].legend(fontsize=8);figure.tight_layout()
        figure.savefig(args.output.with_name(args.output.stem+'-ttft.png'),dpi=160);plt.close(figure)
        for m in models:
            group=[t for t in m['tests'] if t['case']=='steady_text' and not t.get('error')]
            samples=[s for t in group for s in t.get('samples',[])]
            if not samples:continue
            start=samples[0]['time']
            figure,axes=plt.subplots(3,1,figsize=(10,7),sharex=True)
            axes[0].plot([s['time']-start for s in samples],[s.get('temperature_c',float('nan')) for s in samples]);axes[0].set_ylabel('Temperature C')
            axes[1].plot([s['time']-start for s in samples],[s.get('rss_mib',float('nan')) for s in samples]);axes[1].set_ylabel('Model RSS MiB')
            valid=[s for s in samples if 'cpu_seconds' in s]
            axes[2].plot([b['time']-start for a,b in zip(valid,valid[1:]) if b['time']>a['time']],
                         [100*(b['cpu_seconds']-a['cpu_seconds'])/(b['time']-a['time']) for a,b in zip(valid,valid[1:]) if b['time']>a['time']]);axes[2].set_ylabel('Model CPU %\n100% = one core');axes[2].set_xlabel('Seconds since first steady request')
            for ax in axes:ax.grid(alpha=.25)
            figure.suptitle(m['id']+' / 8 consecutive requests');figure.tight_layout()
            figure.savefig(args.output.with_name(m['id']+'-load.png'),dpi=140);plt.close(figure)
    except ImportError:
        print('matplotlib unavailable; tables generated')
    print(args.output)


if __name__=='__main__':main()
