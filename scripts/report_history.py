#!/usr/bin/env python3
"""Readable tables and plots from measured history-length results; errors remain visible."""
import argparse
import json
from pathlib import Path


def fmt(value, digits=2):
    return f'{value:.{digits}f}' if isinstance(value,(int,float)) else '—'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    report=json.loads(args.input.read_text())
    lines=['# 所有模型：历史长度与系统开销', '', f"运行状态：{report['status']}。",
           'A733、CPU6/7、2线程、单槽。无图片输入，VLM仍加载其视觉投影器，保持与网页模型选择一致。',
           '固定中文模拟多轮历史；按各模型自己的tokenizer与聊天模板计数。相同比例不保证相同文字数或轮数。',
           '每档先cache_prompt=false完整预填充，再缓存重放相同请求。后者是理想前缀复用，不是用户新发一轮的端到端耗时。',
           '各请求强制输出128token（ignore_eos=true、greedy），用于速度测量，输出内容不用于质量评分。',
           '每个单元格仅一次测量，不能解读为统计均值；升温与上下文长度可能共同影响速度，频率和温度同步记录。',
           '短历史不是0token；90%档要为128输出token预留空间，按完整轮次向下取整，实际占用见表。',
           '模型CPU：单核100%；整机CPU：8核合计100%。RSS、cgroup、整机可用内存是不同口径。没有实测电功率。',
           '原始采样每100ms一次，分块保存于同名 `.samples` 目录，独立于指标JSON。', '',
           '## 配置空闲与启动', '',
           '| 模型 | 窗口 | 加载秒（非冷启动） | 空闲RSS MiB | 空闲cgroup MiB | 空闲最低可用MiB |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    rows=[]
    for m in report['models']:
        for ctx in m['contexts']:
            idle=ctx.get('idle',{}).get('resources',{})
            lines.append('| '+' | '.join([m['id'],str(ctx['context']),fmt(ctx.get('startup',{}).get('ready_seconds')),
                         fmt(idle.get('peak_rss_mib')),fmt(idle.get('peak_cgroup_mib')),fmt(idle.get('min_available_mib'))])+' |')
            for case in ctx['cases']:
                for run in case['runs']:
                    rows.append((m,ctx,case,run))
    summary=['## 4096窗口：短历史与接近满窗', '',
             '| 模型 | 短输入token | 短历史token/s | 长输入token | 长历史token/s | 长历史完整预填充首字秒 | 长历史缓存首字秒 | 长历史RSS MiB |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for m in report['models']:
        short=[(case,run) for model,ctx,case,run in rows if model['id']==m['id'] and ctx['context']==4096 and case['requested_occupancy']==0 and not run['cache_prompt'] and not run.get('error')]
        long=[(case,run) for model,ctx,case,run in rows if model['id']==m['id'] and ctx['context']==4096 and case['requested_occupancy']==.9 and not run['cache_prompt'] and not run.get('error')]
        cached=[run for model,ctx,case,run in rows if model['id']==m['id'] and ctx['context']==4096 and case['requested_occupancy']==.9 and run['cache_prompt'] and not run.get('error')]
        a,b=short[0] if short else ({},{});c,d=long[0] if long else ({},{})
        cells=[m['id'],str(a.get('input_tokens','—')),fmt(b.get('timings',{}).get('predicted_per_second')),
               str(c.get('input_tokens','—')),fmt(d.get('timings',{}).get('predicted_per_second')),
               fmt(d.get('ttft_seconds')),fmt(cached[0]['ttft_seconds'] if cached else None),fmt(d.get('resources',{}).get('peak_rss_mib'))]
        summary.append('| '+' | '.join(cells)+' |')
    summary += ['', '“—”代表尚未测得有效结果；不能把未完成项目当成0。', '']
    position=lines.index('## 配置空闲与启动')
    lines[position:position]=summary
    for m in report['models']:
        lines += ['', '## '+m['id'], '',
                  '| 窗口 | 目标% / 实际% | 历史轮对 | 输入token | 缓存模式 | 首字秒 | 解码token/s | 预填充token/s | 总秒 |',
                  '| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |']
        for model,ctx,case,run in rows:
            if model['id']!=m['id']:continue
            timing=run.get('timings',{})
            cells=[str(ctx['context']),f"{case['requested_occupancy']*100:g} / {case.get('actual_occupancy',0)*100:.1f}",
                   str(case.get('history_turn_pairs','—')),str(case.get('input_tokens','—')),
                   '缓存重放' if run['cache_prompt'] else '完整预填充',fmt(run['ttft_seconds']),fmt(timing.get('predicted_per_second')),
                   fmt(timing.get('prompt_per_second')),fmt(run['total_seconds'])]
            lines.append('| '+' | '.join(cells)+' |')
            if run.get('error'):lines += ['', '**本格测量无效：'+run['error']+'**', '']
        lines += ['', '| 窗口 | 实际输入 | 缓存 | RSS MiB | cgroup MiB | 整机最低可用MiB | Swap峰值MiB | 模型CPU% | 整机CPU% | I/O等待% | CPU秒 | 温度峰值°C | A76 MHz范围 |',
                  '| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
        for model,ctx,case,run in rows:
            if model['id']!=m['id']:continue
            resource=run['resources'];freq=run.get('a76_frequency_khz') or {}
            cells=[str(ctx['context']),str(case.get('input_tokens','—')),'是' if run['cache_prompt'] else '否']
            cells += [fmt(resource.get(k)) for k in ('peak_rss_mib','peak_cgroup_mib','min_available_mib','peak_swap_used_mib',
                                                     'cpu_percent_one_core_100','system_cpu_busy_percent','system_iowait_percent','cpu_seconds','peak_temperature_c')]
            cells += [f"{freq.get('min',0)/1000:g}–{freq.get('max',0)/1000:g}"]
            lines.append('| '+' | '.join(cells)+' |')
        if m.get('error'):lines += ['', '模型错误：'+m['error']]
        for ctx in m['contexts']:
            if ctx.get('error'):lines += ['', f"窗口{ctx['context']}未完成：{ctx['error']}"]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        for col,context in enumerate([1024,4096,8192]):
            for m in report['models']:
                selected=[(c,r) for model,ctx,c,r in rows if model['id']==m['id'] and ctx['context']==context and not r['cache_prompt'] and not r.get('error')]
                xs=[c['input_tokens'] for c,r in selected]
                axes[0,col].plot(xs,[r['timings']['predicted_per_second'] for c,r in selected],marker='o',label=m['id'])
                axes[1,col].plot(xs,[r['ttft_seconds'] for c,r in selected],marker='o',label=m['id'])
            axes[0,col].set_title(f'Context window: {context}')
            for row in [0,1]:
                axes[row,col].set_xlabel('Actual input tokens (template + history + last user)')
                axes[row,col].grid(alpha=.25)
            axes[0,col].set_ylabel('Decode tokens/s')
            axes[1,col].set_ylabel('Uncached time to first text (s)')
        axes[0,0].legend(fontsize=7)
        fig.suptitle('Orange Pi A733 CPU: fixed 128-token output, one observation per point')
        plot=args.output.with_suffix('.png');fig.savefig(plot,dpi=160);plt.close(fig)
        lines[3:3]=[f'![History performance]({plot.name})','']
        fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        for col,context in enumerate([1024,4096,8192]):
            for m in report['models']:
                selected=[(c,r) for model,ctx,c,r in rows if model['id']==m['id'] and ctx['context']==context and not r['cache_prompt'] and not r.get('error')]
                xs=[c['input_tokens'] for c,r in selected]
                axes[0,col].plot(xs,[r['resources'].get('peak_rss_mib',float('nan')) for c,r in selected],marker='o',label=m['id'])
                axes[1,col].plot(xs,[r['resources'].get('cpu_seconds',float('nan')) for c,r in selected],marker='o',label=m['id'])
            axes[0,col].set_title(f'Context window: {context}')
            for row in [0,1]:
                axes[row,col].set_xlabel('Actual input tokens')
                axes[row,col].grid(alpha=.25)
            axes[0,col].set_ylabel('Peak model RSS (MiB)')
            axes[1,col].set_ylabel('CPU-seconds (full prefill + 128-token decode)')
        axes[0,0].legend(fontsize=7)
        fig.suptitle('Observed resource cost; RSS includes mapped pages, CPU-seconds are not energy')
        resources=args.output.with_name(args.output.stem+'-resources.png')
        fig.savefig(resources,dpi=160);plt.close(fig)
        lines[5:5]=[f'![History resource cost]({resources.name})','']
    args.output.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':report['status'],'models':len(report['models']),'requests':len(rows),
                      'invalid_requests':sum(bool(r.get('error')) for m,c,k,r in rows)}))


if __name__=='__main__':main()
