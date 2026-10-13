"""Report full-depth measured gains and resources without promoting candidates."""
import hashlib
import json
from pathlib import Path
import statistics

from report_system_load import summarize

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'results'


def main():
    trials=[
        ('500M DFP',512,64,ROOT/'smol500-dfpgateup-ab-20261013.json','dfp',
         ROOT/'cpu-q8-search-20261013/smol500-h512-n64-final-fa-off/results.json',
         '500M的512历史CPU配置筛选已完成，当前同场景最快decode确认配置；质量留出待验收'),
        ('1B DFP MSE',512,64,ROOT/'intern1b-dfpmse-ab-20261013.json','dfpmse',
         ROOT/'thermal-aligned-20261013/intern1b-h512-n64-medium-cpu/results.json',
         '1B CPU配置搜索尚未完成；这里只对照同场景已测Q8'),
        ('1B DFP MSE＋prefill16',512,64,ROOT/'dfpmse-prefill16-trial-20261013/performance.json','candidate',
         ROOT/'thermal-aligned-20261013/intern1b-h512-n64-medium-cpu/results.json',
         '1B CPU配置搜索尚未完成；这里只对照同场景已测Q8'),
        ('500M DFP',1024,64,ROOT/'smol500-c1280-dfp-trial-20261013-attempt2/performance.json','candidate',
         ROOT/'thermal-aligned-20261013/smol500-h1024-n64-medium-cpu/results.json',
         '1024历史CPU配置搜索尚未完成；旧NPU与新NPU均t32，反馈轨迹对应本行已测CPU'),
        ('1B DFP MSE＋prefill16＋范围修复',1024,32,ROOT/'dfpmse-productwide-trial-20261013-attempt2/performance.json','candidate',
         ROOT/'thermal-aligned-20261013/intern1b-h1024-n32-medium-cpu/results.json',
         '旧NPU采用已通过质量门槛的product1920wide修复版，两端均grouped head；不与质量失败的旧栈计算收益。CPU配置搜索未完成'),
        ('1B输出头DFP MSE',1024,32,ROOT/'headmse-productwide-trial-20261013/performance.json','candidate',
         ROOT/'thermal-aligned-20261013/intern1b-h1024-n32-medium-cpu/results.json',
         '只改变完整词表输出头量化，decoder与prefill16相同；新输出头仅通过本开发轨迹，独立留出未验。CPU搜索未完成')]
    records=[]
    frontier=[]
    lines=['# 全深度加速候选：速度与系统开销','',
           '按表中精确历史/新增token计数，均为128次预测、Q8对应比较、中温55–<65°C起测、五轮。500M全32层，1B全24层，均包括完整词表输出头。温档约束响应开始温度，响应中自然升降温。','',
           '下表场景的候选完整执行门槛已通过。独立留出已完成执行：500M有单例数值失败，两模型陪伴自然度均未验收通过；这份报告不表示已超过最优CPU或完成产品验收。','',
           '| 候选 / 历史＋新增 | 旧NPU→新NPU token/s | 解码提升 | 旧→新首token s | 同场景CPU token/s | CPU首token s |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,history,new,path,label,cpu_path,note in trials:
        if not path.exists():continue
        data=json.loads(path.read_text())
        if data['status']!='complete':continue
        cpu=json.loads(cpu_path.read_text());assert cpu['status']=='complete'
        measured=[r for r in data['runs'] if r['phase']=='measure' and r['label']==label]
        assert len(measured)==5
        assert all(r['result']['feedback_tokens']==cpu['nonstream_trajectory']['tokens'] for r in measured)
        b,n=data['medians']['baseline'],data['medians'][label]
        c=cpu['medians'];gain=100*(n['decode_tokens_per_second']/b['decode_tokens_per_second']-1)
        engine_ttft=statistics.median(r['timings']['prompt_ms']/1000 for r in cpu['runs'] if r['phase']=='measure')
        display=f'{name} / {history}＋{new}'
        lines.append(f'| {display} | {b["decode_tokens_per_second"]:.3f}→{n["decode_tokens_per_second"]:.3f} | +{gain:.2f}% | {b["ttft_seconds"]:.3f}→{n["ttft_seconds"]:.3f} | {c["decode_tokens_per_second"]:.3f} | {engine_ttft:.3f} |')
        records.append({'name':display,'history_tokens':history,'new_tokens':new,
                        'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'cpu_timing':{'decode_tokens_per_second':c['decode_tokens_per_second'],
                                      'engine_ttft_seconds':engine_ttft,'source':str(cpu_path)},
                        'medians':data['medians'],'cpu':summarize(cpu_path,'cpu'),
                        'npu':summarize(path,'npu',label),'note':note,
                        'trajectory_matches_cpu':True,'independent_quality_accepted':False})
    lines+=['','首token使用引擎口径，CPU为llama.cpp prompt_ms，包含第一token采样；不混用HTTP首事件。加载、历史准备和控温不计入增量首字。各行是独立五轮A/B，不能把不同轮次的百分比相加。','',
            '| 候选 | CPU/NPU整机CPU % | CPU RSS MiB | NPU宿主RSS＋驱动 MiB | CPU/NPU PSS MiB |',
            '| --- | ---: | ---: | ---: | ---: |']
    fmt=lambda x:'缺测' if x is None else f'{x:.1f}'
    for record in records:
        if not record['cpu'] or not record['npu']:
            lines.append(f'| {record["name"]} | 缺少配对直接采样，待补测 | — | — | — |')
            continue
        c,n=record['cpu']['medians'],record['npu']['medians']
        lines.append(f'| {record["name"]} | {c["busy_percent"]:.2f} / {n["busy_percent"]:.2f} | {fmt(c["rss_peak_mib"])} | {fmt(n["rss_peak_mib"])}＋{fmt(n["driver_mib"])} | {fmt(c["pss_peak_mib"])} / {fmt(n["pss_peak_mib"])} |')
    lines+=['','整机CPU以8核总和为100%；NPU宿主RSS与驱动分配另列，不能把宿主RSS当成总模型内存，也不能把驱动字节与RSS简单相加视为去重后的物理内存。CPU网页模型在两端均保持常驻。CPU占用下降不是功耗测量。','',
            '| 候选 | CPU/NPU换入＋换出 KiB | CPU/NPU磁盘读＋写 KiB | CPU/NPU响应采样覆盖率 |',
            '| --- | ---: | ---: | ---: |']
    for record in records:
        if not record['cpu'] or not record['npu']:
            lines.append(f'| {record["name"]} | 待补测 | 待补测 | 待补测 |')
            continue
        c,n=record['cpu']['medians'],record['npu']['medians']
        lines.append(f'| {record["name"]} | {(c["swap_in_bytes"]+c["swap_out_bytes"])/1024:.0f} / {(n["swap_in_bytes"]+n["swap_out_bytes"])/1024:.0f} | {(c["disk_read_bytes"]+c["disk_write_bytes"])/1024:.0f} / {(n["disk_read_bytes"]+n["disk_write_bytes"])/1024:.0f} | {c["covered_fraction"]:.0%} / {n["covered_fraction"]:.0%} |')
    lines+=['','统计只覆盖完全落在响应中的采样区间；未覆盖边界、模型加载及KV历史准备阶段不包含在I/O、换页结论里。表中均为五轮中位数，逐轮采样摘要保存在证据中。','']
    lines.extend(f'- {r["name"]}：{r["note"]}。' for r in records)
    search_path=ROOT/'cpu-q8-search-20261013/state.json'
    if search_path.exists():
        search=json.loads(search_path.read_text())
        lines+=['','## 当前CPU目标线','',
                '上表CPU列保持与该次NPU测量相同的反馈轨迹，不能把较旧对照误读成最优CPU。下面单列持续搜索得到的最快decode配置；两轮仅为筛选，五轮确认也须检查反馈轨迹，不能静默换到旧NPU配对中。','',
                '| 场景 | 当前最快配置 | 轮数 | token/s | 引擎首token s | 配置搜索状态 |',
                '| --- | --- | ---: | ---: | ---: | --- |']
        for name,row in search['cases'].items():
            for repeats in (2,5):
                pool={k:v for k,v in row['runs'].items() if v.get('medians') and
                      v['repeats']==repeats and not k.startswith(('n32-','n128-'))}
                if not pool:continue
                label=max(pool,key=lambda k:pool[k]['medians']['decode_tokens_per_second'])
                value=pool[label];m=value['medians']
                frontier.append({'case':name,'label':label,'search_status':row['status'],**value})
                lines.append(f'| {name} | {label} | {repeats} | {m["decode_tokens_per_second"]:.3f} | {m["engine_ttft_seconds"]:.3f} | {row["status"]} |')
        lines+=['','500M的1024历史已有快于当前NPU的CPU配置；在最快配置确认和轨迹重对齐结束前，不能宣称长历史NPU胜出。完整配置及热保护拒绝记录见[CPU搜索报告](下一轮攻关进展.md)。']
    lines+=['','继续验收：扩展32/128新增与1024历史、最快CPU轨迹重对齐、真实音频并发，以及新优化候选的精度和性能。两模型原候选的独立留出、自然多轮、实验HTTP与30分钟稳定性已有结果，不能自动转移给后来改变权重或执行路径的新候选。','',
            '两模型真实HTTP/浏览器多轮已通过接口检查，见[HTTP报告](HTTP多轮接口验证.md)；独立质量和持续运行进度见[产品验收](独立质量与连续多轮验收.md)，接口通过不等于陪伴质量通过。','',
            '复现：`npu/.venv/bin/python npu/kv-cache/report_candidate_speed.py`。',
            '[紧凑证据](evidence-attack-20261013/candidate-speed-load.json)。','']
    (HERE/'全深度候选提速与负载.md').write_text('\n'.join(lines))
    (HERE/'evidence-attack-20261013/candidate-speed-load.json').write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n')
    (HERE/'evidence-attack-20261013/cpu-speed-frontier.json').write_text(json.dumps(frontier,indent=2,ensure_ascii=False)+'\n')
    print('completed candidate comparisons',len(records))


if __name__=='__main__':main()
