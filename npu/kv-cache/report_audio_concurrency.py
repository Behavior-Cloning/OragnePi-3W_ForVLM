"""Report measured ASR interference only; distinguish setup from board results."""
import hashlib
import json
from pathlib import Path
import statistics


HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'results'


def read(path):
    if not path.exists():return None
    raw=path.read_bytes()
    data=json.loads(raw)
    return {**data,'source_path':str(path),'source_sha256':hashlib.sha256(raw).hexdigest()}


def plot_comparison(evidence):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.hashsalt':'opi-audio-concurrency-v1','font.size':9})
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    metrics=[('decode_tokens_per_second','Decode (token/s)'),
             ('engine_ttft_seconds','Engine first token (s)'),
             ('busy','Whole-system CPU (%)')]
    colors=['#446e9b','#cf875a','#459c89','#7bb6a8','#9b79b5']
    for row,model in enumerate(('smol500','intern1b')):
        d=evidence['models'][model]['result']
        groups=[]
        if d and d.get('status')=='complete-paired-concurrency':
            for backend,audio,label in [('cpu',False,'CPU'),('cpu',True,'CPU\n+ ASR'),
                                         ('npu',False,'NPU'),('npu',True,'NPU\n+ ASR')]:
                groups.append((label,[r for r in d['runs'] if r['phase']=='measure' and
                                      r['backend']==backend and r['audio']==audio]))
        extra=evidence['cpu_little_asr'][model]['result']
        if extra and extra.get('status')=='complete-paired-concurrency':
            groups.append(('CPU\nASR little',[r for r in extra['runs'] if r['phase']=='measure' and r['audio']]))
        for col,(key,title) in enumerate(metrics):
            ax=axes[row,col]
            peak=0
            for i,(label,runs) in enumerate(groups):
                assert len(runs)==5
                vals=[r['resource_summary']['cpu']['cpu']['busy_percent'] if key=='busy' else r[key] for r in runs]
                median=statistics.median(vals)
                peak=max(peak,max(vals))
                ax.bar(i,median,color=colors[i],width=.65)
                ax.errorbar(i,median,yerr=[[median-min(vals)],[max(vals)-median]],fmt='none',color='#222',capsize=3)
                ax.text(i,max(vals)+max(vals)*.04,f'{median:.2f}',ha='center',fontsize=8)
            ax.set_xticks(range(len(groups)),[x[0] for x in groups])
            ax.set_title(('500M' if row==0 else '1B')+' — '+title)
            ax.set_ylim(0,max(peak*1.23,8 if key=='decode_tokens_per_second' else .1))
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            if not groups:ax.text(.5,.5,'Pending',transform=ax.transAxes,ha='center')
            if key=='decode_tokens_per_second':
                ax.axhline(6,color='#777',linestyle='--',linewidth=1,label='~6 token/s speech reference')
                ax.legend(fontsize=7,loc='upper right')
    fig.suptitle('Full language models: 512 cached + 64 new tokens, 128 predictions\n'
                 '5-run medians; whiskers = min/max; start 55–<65°C; sustained 11 s ASR file',fontsize=11)
    fig.savefig(HERE/'audio-concurrency.png',dpi=160)
    fig.savefig(HERE/'audio-concurrency.svg',metadata={'Date':None})
    svg=HERE/'audio-concurrency.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    setup=read(ROOT/'audio-asr-board-20261013/state.json')
    evidence={'setup':setup,'models':{}}
    lines=['# 真实ASR与LLM并发验收','',
           '语音负载为whisper.cpp v1.8.0、多语言tiny Q5_1，两个CPU线程允许在0–7核调度。固定11秒、16kHz、单声道JFK录音，模型和录音常驻，连续重复识别。它是真实识别运算，但属于饱和离线负载，不是实时麦克风调度、中文准确率、AEC或TTS验收。','',
           '每个语言模型取512已有历史＋64新增、128次完整预测；500M32层、1B24层、完整词表头。CPU与NPU反馈轨迹相同，NPU先复现全部logits。ASR开关交替，预热一轮、正式五轮；响应起温55–<65°C，推理期间可变化。ASR模型在开关两种条件下都驻留，因此表中体现计算竞争而不混入ASR模型加载。','',
           'CPU使用与该NPU候选轨迹对齐的已测Q8配置，具体命令归档；与持续CPU搜索得到的最终最快成绩分开。生产网页CPU模型在两种后端测量期间均保持常驻。','',
           f'板端音频准备状态：`{setup["status"] if setup else "未启动"}`。本机编译/识别smoke通过不代替板端验证；setup通过也不等于并发通过。','',
           '| 模型 | 并发任务状态 | 已完成正式测量 / 20 |','| --- | --- | ---: |']
    for model,label in [('smol500','500M'),('intern1b','1B')]:
        state=read(ROOT/f'audio-concurrency-{model}-20261013/state.json')
        data=read(ROOT/f'audio-concurrency-{model}-20261013/result.json')
        evidence['models'][model]={'state':state,'result':data}
        count=sum(r['phase']=='measure' for r in data.get('runs',[])) if data else 0
        lines.append(f'| {label} | {(data or state or {}).get("status","未启动")} | {count} |')
    lines+=['','| 模型 / 后端 | ASR关→开 decode token/s | 解码变化 | ASR关→开首token s | ASR关→开响应 s |',
            '| --- | ---: | ---: | ---: | ---: |']
    for model,row in evidence['models'].items():
        d=row['result']
        if not d or d.get('status')!='complete-paired-concurrency':continue
        for backend in ('cpu','npu'):
            a,b=d['medians'][backend+'-alone'],d['medians'][backend+'-asr']
            speed='decode_tokens_per_second';ttft='engine_ttft_seconds';total='total_seconds'
            lines.append(f'| {model} / {backend} | {a[speed]:.3f}→{b[speed]:.3f} | '
                         f'{100*(b[speed]/a[speed]-1):+.2f}% | {a[ttft]:.3f}→{b[ttft]:.3f} | {a[total]:.3f}→{b[total]:.3f} |')
    lines+=['','| 模型 / 后端 / ASR | 整机CPU % | LLM / ASR RSS MiB | NPU驱动 MiB | 最低可用RAM MiB | 换入＋换出 KiB | ASR重叠比例 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for model,row in evidence['models'].items():
        data=row['result']
        if not data or data.get('status')!='complete-paired-concurrency':continue
        for backend in ('cpu','npu'):
            for enabled in (False,True):
                runs=[r for r in data['runs'] if r['phase']=='measure' and r['backend']==backend and r['audio']==enabled]
                assert len(runs)==5
                samples=[[s for s in r['resource_samples'] if s.get('process')] for r in runs]
                rss=statistics.median(max(s['process']['rss_kib'] for s in ss)/1024 for ss in samples)
                asr_peaks=[max(s['asr_process']['rss_kib'] for s in ss if s.get('asr_process'))/1024
                           for ss in samples if any(s.get('asr_process') for s in ss)]
                asr_rss=f'{statistics.median(asr_peaks):.2f}' if len(asr_peaks)==5 else '缺测'
                busy=statistics.median(r['resource_summary']['cpu']['cpu']['busy_percent'] for r in runs)
                avail=statistics.median(min(s['memory_kib']['MemAvailable'] for s in ss)/1024 for ss in samples)
                driver=statistics.median(max(s.get('npu_driver_bytes',0) for s in ss)/2**20 for ss in samples)
                swap=statistics.median((r['resource_summary']['swap_in_bytes']+r['resource_summary']['swap_out_bytes'])/1024 for r in runs)
                overlap=statistics.median(r['asr_overlap_fraction'] for r in runs)
                lines.append(f'| {model} / {backend} / {"开" if enabled else "关"} | {busy:.2f} | {rss:.2f} / {asr_rss} | {driver:.2f} | {avail:.1f} | {swap:.0f} | {overlap:.1%} |')
    lines+=['','整机CPU以8核合计100%计算。LLM与ASR的RSS分别列出，驱动不是独立显存；RSS与驱动不能直接相加当作去重物理内存。换页和系统负载只统计完全位于响应内的直接采样区间。不能由CPU占用推算整板功率。','',
            '| 模型 / 后端 / ASR | LLM平均占用核数 | ASR平均占用核数 | 响应采样A76 MHz范围 |',
            '| --- | ---: | ---: | ---: |']
    for model,row in evidence['models'].items():
        data=row['result']
        if not data or data.get('status')!='complete-paired-concurrency':continue
        for backend in ('cpu','npu'):
            for enabled in (False,True):
                rows=[r for r in data['runs'] if r['phase']=='measure' and r['backend']==backend and r['audio']==enabled]
                rates={'process':[],'asr_process':[]};freqs=[]
                for r in rows:
                    ss=r['resource_samples']
                    for key in rates:
                        samples=[s for s in ss if key in s]
                        if len(samples)>=2:
                            a,b=samples[0],samples[-1]
                            rates[key].append((b[key]['cpu_seconds']-a[key]['cpu_seconds'])/(b['monotonic_seconds']-a['monotonic_seconds']))
                    freqs.extend(s['frequencies_khz']['policy6']/1000 for s in ss if 'policy6' in s.get('frequencies_khz',{}))
                avg=lambda k:f'{statistics.median(rates[k]):.3f}' if len(rates[k])==5 else '缺测'
                freq=f'{min(freqs):.0f}–{max(freqs):.0f}' if freqs else '缺测'
                lines.append(f'| {model} / {backend} / {"开" if enabled else "关"} | {avg("process")} | {avg("asr_process")} | {freq} |')
    lines+=['','平均占用核数=两次直接进程CPU累计时间之差/采样墙钟间隔；2代表约两核计算时间，不表示固定在哪两核。只描述已采样时段；并发还可能有缓存、带宽和线程调度竞争，不能把所有降速归于单一原因。','',
            '| 模型 / 语言后端 | 与响应重叠的ASR调用 | ASR整次识别中位 s | ASR整次RTF中位 |',
            '| --- | ---: | ---: | ---: |']
    for model,row in evidence['models'].items():
        d=row['result']
        if not d or d.get('status')!='complete-paired-concurrency':continue
        for backend in ('cpu','npu'):
            calls=[call for r in d['runs'] if r['phase']=='measure' and r['audio'] and r['backend']==backend for call in r['asr']]
            lines.append(f'| {model} / {backend} | {len(calls)} | {statistics.median(c["inference_seconds"] for c in calls):.3f} | {statistics.median(c["rtf"] for c in calls):.3f} |')
    lines+=['','RTF为识别耗时/11秒音频时长，<1表示该文件的处理快于实时。每次ASR完整调用可能延续到LLM响应结束之后，因此RTF是混合时段的实测值，不能当成全程竞争下的严格实时保证。每轮记录重叠率（要求≥95%）、转写原文和单调时钟边界；两后端ASR任务参数相同。','',
            '## CPU与语音分核复核','',
            '默认ASR两线程可在全部核调度，可能与CPU语言模型争用A76；因此不能把上表的争用损失视为最优CPU并发的必然表现。补充试验固定语言模型原Q8配置，将ASR两线程限制在0–5小核。同样预热＋五轮、同温档、ASR开关交替，保留ASR识别延迟，不能只追求语言速度而忽略音频变慢。该表是调度试验，尚非穷举CPU/ASR联合配置。','',
            '| 模型 | 状态 | CPU单独→ASR小核并发 token/s | 首token s | ASR识别 s / RTF |',
            '| --- | --- | ---: | ---: | ---: |']
    evidence['cpu_little_asr']={}
    for model in ('smol500','intern1b'):
        state=read(ROOT/f'audio-cpu-little-{model}-20261013/state.json')
        d=read(ROOT/f'audio-cpu-little-{model}-20261013/result.json')
        evidence['cpu_little_asr'][model]={'state':state,'result':d}
        status=(d or state or {}).get('status','未启动')
        if not d or d.get('status')!='complete-paired-concurrency':
            lines.append(f'| {model} | {status} | — | — | — |');continue
        assert d['config']['backends']==['cpu'] and d['config']['asr_affinity']=='0-5'
        a,b=d['medians']['cpu-alone'],d['medians']['cpu-asr']
        rows=[r for r in d['runs'] if r['phase']=='measure' and r['audio']];assert len(rows)==5
        calls=[c for r in rows for c in r['asr']]
        lines.append(f'| {model} | {status} | {a["decode_tokens_per_second"]:.3f}→{b["decode_tokens_per_second"]:.3f} | '
                     f'{b["engine_ttft_seconds"]:.3f} | {statistics.median(c["inference_seconds"] for c in calls):.3f} / '
                     f'{statistics.median(c["rtf"] for c in calls):.3f} |')
    lines+=['',
            '![音频并发对照](audio-concurrency.png)','',
            '图中柱为五轮中位数，误差线为最小至最大；6 token/s虚线仅作口语节奏示意，实际换算随语言和分词器而变，不是质量或吞吐验收门槛。第五根柱（若已完成）为另一次CPU/ASR分核试验，需要结合ASR的RTF判断，不能只按语言token/s宣告更好。','',
            '完整原始采样位于本地`npu/results/audio-concurrency-*-20261013`；复现报告：`npu/.venv/bin/python npu/kv-cache/report_audio_concurrency.py`。','']
    plot_comparison(evidence)
    # Retain timing summaries and transcripts, without duplicating raw /proc counters into Git.
    for row in [*evidence['models'].values(),*evidence['cpu_little_asr'].values()]:
        if row['result']:
            row['result']['runs']=[{k:v for k,v in r.items() if k!='resource_samples'} for r in row['result']['runs']]
    (HERE/'真实ASR与LLM并发验收.md').write_text('\n'.join(lines))
    (HERE/'evidence-attack-20261013/audio-concurrency.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print({k:(v['result'] or v['state'] or {}).get('status') for k,v in evidence['models'].items()})


if __name__=='__main__':main()
