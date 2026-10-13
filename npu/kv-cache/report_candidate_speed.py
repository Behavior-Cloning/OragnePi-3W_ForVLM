"""Report full-depth measured gains and resources without promoting candidates."""
import hashlib
import json
from pathlib import Path

from report_system_load import summarize

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'results'


def main():
    trials=[
        ('500M DFP',ROOT/'smol500-dfpgateup-ab-20261013.json','dfp',
         ROOT/'cpu-q8-search-20261013/smol500-h512-n64-final-fa-off/results.json',
         '500M的512历史CPU配置筛选已完成，当前同场景最快decode确认配置；质量留出待验收'),
        ('1B DFP MSE',ROOT/'intern1b-dfpmse-ab-20261013.json','dfpmse',
         ROOT/'thermal-aligned-20261013/intern1b-h512-n64-medium-cpu/results.json',
         '1B CPU配置搜索尚未完成；这里只对照同场景已测Q8'),
        ('1B DFP MSE＋prefill16',ROOT/'dfpmse-prefill16-trial-20261013/performance.json','candidate',
         ROOT/'thermal-aligned-20261013/intern1b-h512-n64-medium-cpu/results.json',
         '1B CPU配置搜索尚未完成；这里只对照同场景已测Q8')]
    records=[]
    lines=['# 全深度加速候选：速度与系统开销','',
           '两模型均为512历史＋64新增、128次预测、Q8对应比较、中温55–<65°C起测、五轮。500M全32层，1B全24层，均包括完整词表输出头。温档约束响应开始温度，响应中自然升降温。','',
           '候选完整执行门槛已通过，独立质量留出尚未验收；这份报告不表示已超过最优CPU或完成产品验收。','',
           '| 候选 | 旧NPU→新NPU token/s | 解码提升 | 旧→新首token s | 同场景CPU token/s | CPU首token s |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,path,label,cpu_path,note in trials:
        if not path.exists():continue
        data=json.loads(path.read_text())
        if data['status']!='complete':continue
        cpu=json.loads(cpu_path.read_text());assert cpu['status']=='complete'
        measured=[r for r in data['runs'] if r['phase']=='measure' and r['label']==label]
        assert len(measured)==5
        assert all(r['result']['feedback_tokens']==cpu['nonstream_trajectory']['tokens'] for r in measured)
        b,n=data['medians']['baseline'],data['medians'][label]
        c=cpu['medians'];gain=100*(n['decode_tokens_per_second']/b['decode_tokens_per_second']-1)
        lines.append(f'| {name} | {b["decode_tokens_per_second"]:.3f}→{n["decode_tokens_per_second"]:.3f} | +{gain:.2f}% | {b["ttft_seconds"]:.3f}→{n["ttft_seconds"]:.3f} | {c["decode_tokens_per_second"]:.3f} | {c["engine_ttft_seconds"]:.3f} |')
        records.append({'name':name,'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'medians':data['medians'],'cpu':summarize(cpu_path,'cpu'),
                        'npu':summarize(path,'npu',label),'note':note,
                        'trajectory_matches_cpu':True,'independent_quality_accepted':False})
    lines+=['','首token使用引擎口径，CPU为llama.cpp prompt_ms，包含第一token采样；不混用HTTP首事件。加载、历史准备和控温不计入增量首字。各行是独立五轮A/B，不能把不同轮次的百分比相加。','',
            '| 候选 | CPU/NPU整机CPU % | CPU RSS MiB | NPU宿主RSS＋驱动 MiB | CPU/NPU PSS MiB |',
            '| --- | ---: | ---: | ---: | ---: |']
    fmt=lambda x:'缺测' if x is None else f'{x:.1f}'
    for record in records:
        c,n=record['cpu']['medians'],record['npu']['medians']
        lines.append(f'| {record["name"]} | {c["busy_percent"]:.2f} / {n["busy_percent"]:.2f} | {fmt(c["rss_peak_mib"])} | {fmt(n["rss_peak_mib"])}＋{fmt(n["driver_mib"])} | {fmt(c["pss_peak_mib"])} / {fmt(n["pss_peak_mib"])} |')
    lines+=['','整机CPU以8核总和为100%；NPU宿主RSS与驱动分配另列，不能把宿主RSS当成总模型内存，也不能把驱动字节与RSS简单相加视为去重后的物理内存。CPU网页模型在两端均保持常驻。CPU占用下降不是功耗测量。','',
            '| 候选 | CPU/NPU换入＋换出 KiB | CPU/NPU磁盘读＋写 KiB | CPU/NPU响应采样覆盖率 |',
            '| --- | ---: | ---: | ---: |']
    for record in records:
        c,n=record['cpu']['medians'],record['npu']['medians']
        lines.append(f'| {record["name"]} | {(c["swap_in_bytes"]+c["swap_out_bytes"])/1024:.0f} / {(n["swap_in_bytes"]+n["swap_out_bytes"])/1024:.0f} | {(c["disk_read_bytes"]+c["disk_write_bytes"])/1024:.0f} / {(n["disk_read_bytes"]+n["disk_write_bytes"])/1024:.0f} | {c["covered_fraction"]:.0%} / {n["covered_fraction"]:.0%} |')
    lines+=['','统计只覆盖完全落在响应中的采样区间；未覆盖边界、模型加载及KV历史准备阶段不包含在I/O、换页结论里。表中均为五轮中位数，逐轮采样摘要保存在证据中。','']
    lines.extend(f'- {r["name"]}：{r["note"]}。' for r in records)
    lines+=['','继续验收：扩展32/128新增与1024历史、最快CPU轨迹重对齐、独立留出与自然多轮、NPU常驻HTTP、30分钟稳定性、真实音频并发。缺少的项目不能用固定反馈速度替代。','',
            '复现：`npu/.venv/bin/python npu/kv-cache/report_candidate_speed.py`。',
            '[紧凑证据](evidence-attack-20261013/candidate-speed-load.json)。','']
    (HERE/'全深度候选提速与负载.md').write_text('\n'.join(lines))
    (HERE/'evidence-attack-20261013/candidate-speed-load.json').write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n')
    print('completed candidate comparisons',len(records))


if __name__=='__main__':main()
