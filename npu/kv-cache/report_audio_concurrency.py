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
    lines+=['','| 模型 / 后端 / ASR | 整机CPU % | LLM RSS MiB | NPU驱动 MiB | 最低可用RAM MiB | 换入＋换出 KiB | ASR重叠比例 |',
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
                busy=statistics.median(r['resource_summary']['cpu']['cpu']['busy_percent'] for r in runs)
                avail=statistics.median(min(s['memory_kib']['MemAvailable'] for s in ss)/1024 for ss in samples)
                driver=statistics.median(max(s.get('npu_driver_bytes',0) for s in ss)/2**20 for ss in samples)
                swap=statistics.median((r['resource_summary']['swap_in_bytes']+r['resource_summary']['swap_out_bytes'])/1024 for r in runs)
                overlap=statistics.median(r['asr_overlap_fraction'] for r in runs)
                lines.append(f'| {model} / {backend} / {"开" if enabled else "关"} | {busy:.2f} | {rss:.2f} | {driver:.2f} | {avail:.1f} | {swap:.0f} | {overlap:.1%} |')
    lines+=['','整机CPU以8核合计100%计算。RSS只指待测语言进程，不包含ASR；驱动不是独立显存。换页和系统负载只统计完全位于响应内的直接采样区间。不能由CPU占用推算整板功率。','',
            '| 模型 / 语言后端 | 与响应重叠的ASR调用 | ASR整次识别中位 s | ASR整次RTF中位 |',
            '| --- | ---: | ---: | ---: |']
    for model,row in evidence['models'].items():
        d=row['result']
        if not d or d.get('status')!='complete-paired-concurrency':continue
        for backend in ('cpu','npu'):
            calls=[call for r in d['runs'] if r['phase']=='measure' and r['audio'] and r['backend']==backend for call in r['asr']]
            lines.append(f'| {model} / {backend} | {len(calls)} | {statistics.median(c["inference_seconds"] for c in calls):.3f} | {statistics.median(c["rtf"] for c in calls):.3f} |')
    lines+=['','RTF为识别耗时/11秒音频时长，<1表示该文件的处理快于实时。每次ASR完整调用可能延续到LLM响应结束之后，因此RTF是混合时段的实测值，不能当成全程竞争下的严格实时保证。每轮记录重叠率（要求≥95%）、转写原文和单调时钟边界；两后端ASR任务参数相同。','',
            '完整原始采样位于本地`npu/results/audio-concurrency-*-20261013`；复现报告：`npu/.venv/bin/python npu/kv-cache/report_audio_concurrency.py`。','']
    # Retain timing summaries and transcripts, without duplicating raw /proc counters into Git.
    for row in evidence['models'].values():
        if row['result']:
            row['result']['runs']=[{k:v for k,v in r.items() if k!='resource_samples'} for r in row['result']['runs']]
    (HERE/'真实ASR与LLM并发验收.md').write_text('\n'.join(lines))
    (HERE/'evidence-attack-20261013/audio-concurrency.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print({k:(v['result'] or v['state'] or {}).get('status') for k,v in evidence['models'].items()})


if __name__=='__main__':main()
