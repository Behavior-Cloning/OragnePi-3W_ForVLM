"""Compare only paired full-depth timing and resource samples from each run."""
import hashlib
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'results'


def read(path):return json.loads(path.read_text())


def summarize(path,backend):
    data=read(path);assert data['status']=='complete'
    runs=[r for r in data['runs'] if r['phase']=='measure'];assert len(runs)==5
    rows=[]
    for run in runs:
        runtime=run if backend=='cpu' else run['result']
        s=run.get('resource_summary',{})
        if not s.get('intervals'):return None
        start=runtime['response_start_monotonic_seconds']
        end=run.get('response_end_monotonic_seconds',start+runtime['total_seconds'])
        samples=[v['resources'] for v in run['system_samples'] if 'resources' in v and
                 start<=v['resources']['monotonic_seconds'] and
                 v['resources']['monotonic_seconds']+v['resources']['sampling_seconds']<=end]
        assert samples
        rss=[v['process']['rss_kib'] for v in samples if 'process' in v]
        pss=[v['process']['pss_kib'] for v in samples if 'pss_kib' in v.get('process',{})]
        # Older direct-v1 loses process data if kernel task I/O is disabled.
        # Keep its separately read RSS, and explicitly record the source.
        rss_source='direct-process'
        if not rss:
            rss_source='legacy-process-samples' if backend=='cpu' else 'runner-peak-RSS'
            rss=([v['process']['rss_kib'] for v in run['system_samples'] if 'process' in v]
                 if backend=='cpu' else [runtime['max_rss_kib']])
        rows.append({'busy_percent':s['cpu']['cpu']['busy_percent'],
                     'iowait_percent':s['cpu']['cpu']['iowait_percent'],
                     'rss_peak_mib':max(rss)/1024 if rss else None,'rss_source':rss_source,
                     'pss_peak_mib':max(pss)/1024 if pss else None,
                     'available_min_mib':min(v['memory_kib']['MemAvailable'] for v in samples)/1024,
                     'temperature_min_c':min(v['temperature_mC'] for v in samples)/1000,
                     'temperature_max_c':max(v['temperature_mC'] for v in samples)/1000,
                     'swap_in_bytes':s['swap_in_bytes'],'swap_out_bytes':s['swap_out_bytes'],
                     'disk_read_bytes':sum(v['read_bytes'] for v in s['disk_io_bytes'].values()),
                     'disk_write_bytes':sum(v['write_bytes'] for v in s['disk_io_bytes'].values()),
                     'covered_fraction':s['covered_seconds']/s['response_seconds'],
                     'decode_tps':run['timings']['predicted_per_second'] if backend=='cpu' else runtime['decode_tokens_per_second'],
                     'ttft_seconds':run['timings']['prompt_ms']/1000 if backend=='cpu' else runtime['ttft_seconds'],
                     'driver_mib':None if backend=='cpu' else runtime['driver_allocated_bytes']/2**20,
                     'sampling_errors':sorted({k for v in samples for k in v.get('errors',{})})})
    metrics={k:statistics.median(r[k] for r in rows) if all(r[k] is not None for r in rows) else None
             for k in rows[0] if k not in ('rss_source','sampling_errors')}
    return {'medians':metrics,'runs':rows,'source':str(path),
            'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    matrix=read(ROOT/'thermal-aligned-20261013/state.json')
    backfill=read(ROOT/'resource-backfill-20261013/state.json')
    result={}
    for case,entry in matrix['cases'].items():
        for band in ('low','medium'):
            row=entry.get('bands',{}).get(band,{})
            if row.get('status')!='complete':continue
            key=case+'-'+band
            replacement=backfill['cases'].get(key,{})
            if replacement.get('status')=='complete':
                cp,np=Path(replacement['cpu']),Path(replacement['npu'])
            else:
                cp=Path(row['cpu_path'])/'results.json'
                np=Path(row.get('performance',ROOT/'thermal-aligned-20261013'/(key+'-npu.json')))
            cpu,npu=summarize(cp,'cpu'),summarize(np,'npu')
            if cpu and npu:result[key]={'cpu':cpu,'npu':npu,'resource_backfill':replacement.get('status')=='complete'}
    lines=['# CPU与NPU直接系统负载对比','',
           '2026-10-13。全部语言层、完整输出头、相同冻结轨迹、128次预测，每格五轮。使用每次响应自己的计时与采样；补测整对替换展示，原始证据不改。这里的CPU为矩阵中已测Q8配置，搜索到更快配置后另做对应对照，不能称为最终最优CPU。','',
           '温档指响应开始时：低40–50°C、中55–<65°C。响应过程允许升降温，并非恒温；表末列实际采样范围。整机CPU利用率以8核合计为100%，两核跑满约25%。PSS缺失标“未采到”。驱动分配单列，不把NPU宿主RSS当成模型总内存。','',
           '| 场景/起温档 | CPU/NPU token/s | CPU/NPU 首token s | CPU/NPU 整机CPU % | CPU RSS / NPU RSS＋驱动 MiB | CPU/NPU PSS MiB | CPU/NPU 响应内温度°C |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    fmt=lambda x:'未采到' if x is None else f'{x:.1f}'
    for key,r in result.items():
        c,n=r['cpu']['medians'],r['npu']['medians']
        lines.append(f'| {key} | {c["decode_tps"]:.2f} / {n["decode_tps"]:.2f} | {c["ttft_seconds"]:.3f} / {n["ttft_seconds"]:.3f} | {c["busy_percent"]:.2f} / {n["busy_percent"]:.2f} | {fmt(c["rss_peak_mib"])} / {fmt(n["rss_peak_mib"])}＋{fmt(n["driver_mib"])} | {fmt(c["pss_peak_mib"])} / {fmt(n["pss_peak_mib"])} | {c["temperature_min_c"]:.1f}–{c["temperature_max_c"]:.1f} / {n["temperature_min_c"]:.1f}–{n["temperature_max_c"]:.1f} |')
    lines+=['','## 换页、I/O和采样覆盖','',
            '| 场景/起温档 | CPU/NPU 可用内存最低 MiB | CPU/NPU 换入＋换出 KiB | CPU/NPU 物理读＋写 KiB | CPU/NPU 覆盖率 |',
            '| --- | --- | --- | --- | --- |']
    for key,r in result.items():
        c,n=r['cpu']['medians'],r['npu']['medians']
        lines.append(f'| {key} | {c["available_min_mib"]:.0f} / {n["available_min_mib"]:.0f} | {(c["swap_in_bytes"]+c["swap_out_bytes"])/1024:.0f} / {(n["swap_in_bytes"]+n["swap_out_bytes"])/1024:.0f} | {(c["disk_read_bytes"]+c["disk_write_bytes"])/1024:.0f} / {(n["disk_read_bytes"]+n["disk_write_bytes"])/1024:.0f} | {c["covered_fraction"]:.0%} / {n["covered_fraction"]:.0%} |')
    lines+=['','以上为五轮各自统计量的中位数；零换页/零I/O仅适用于被覆盖的响应内区间，不表示加载、历史准备及边界完全没有I/O。完整各轮、来源哈希、采样错误见紧凑证据。网页CPU模型保持常驻，两端都包含系统后台负载。未测电功率，不能据CPU占用推导整板功耗。', '',
            '已发现板端缺少`/proc/<pid>/io`。direct-v2保留独立读取的CPU/RSS/PSS，缺失进程I/O明确记录；旧记录不追填不存在的PSS。物理盘I/O仍来自`/proc/diskstats`。', '',
            'direct-v1/v2的`npu_driver_bytes`采样字段误读了分配计数，已废弃；本表始终使用C运行器正确读取的`driver_allocated_bytes`。direct-v3严格读取以0x开头的前两列，保留旧原始记录用于追溯。','',
            '[紧凑证据](evidence-attack-20261013/system-load.json)。重建：`npu/.venv/bin/python npu/kv-cache/report_system_load.py`。','']
    archive=HERE/'evidence-attack-20261013';archive.mkdir(exist_ok=True)
    (archive/'system-load.json').write_text(json.dumps(result,indent=2)+'\n')
    (HERE/'CPU与NPU直接系统负载对比.md').write_text('\n'.join(lines))
    print('paired resource cells',len(result))


if __name__=='__main__':main()
