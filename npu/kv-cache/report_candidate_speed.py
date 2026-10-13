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
         '500M的512历史CPU配置筛选已完成，当前同场景最快decode确认配置；独立v1存在单例质量失败'),
        ('1B DFP MSE',512,64,ROOT/'intern1b-dfpmse-ab-20261013.json','dfpmse',
         ROOT/'thermal-aligned-20261013/intern1b-h512-n64-medium-cpu/results.json',
         '保留本次A/B匹配的原CPU轨迹；搜索已完成，最新更快CPU对齐见另表'),
        ('1B DFP MSE＋prefill16',512,64,ROOT/'dfpmse-prefill16-trial-20261013/performance.json','candidate',
         ROOT/'thermal-aligned-20261013/intern1b-h512-n64-medium-cpu/results.json',
         '保留本次A/B匹配的原CPU轨迹；搜索已完成，最新更快CPU对齐见另表'),
        ('500M DFP',1024,64,ROOT/'smol500-c1280-dfp-trial-20261013-attempt2/performance.json','candidate',
         ROOT/'thermal-aligned-20261013/smol500-h1024-n64-medium-cpu/results.json',
         '旧NPU与新NPU均t32，保留本次A/B匹配的原CPU；更快CPU结果与完整对齐见另表'),
        ('1B DFP MSE＋prefill16＋范围修复',1024,32,ROOT/'dfpmse-productwide-trial-20261013-attempt2/performance.json','candidate',
         ROOT/'thermal-aligned-20261013/intern1b-h1024-n32-medium-cpu/results.json',
         '旧NPU采用已通过开发门槛的product1920wide修复版，两端均grouped head；不与质量失败的旧栈计算收益。本行CPU是原匹配轨迹'),
        ('1B输出头DFP MSE',1024,32,ROOT/'headmse-productwide-trial-20261013/performance.json','candidate',
         ROOT/'thermal-aligned-20261013/intern1b-h1024-n32-medium-cpu/results.json',
         '只改变完整词表输出头量化，decoder与prefill16相同；独立v2为32/36单例通过，未过整套质量。本行CPU是原匹配轨迹'),
        ('1B decoder有界MSE cap1',1024,32,ROOT/'cap1-productwide-trial-20261013/performance.json','candidate',
         ROOT/'thermal-aligned-20261013/intern1b-h1024-n32-medium-cpu/results.json',
         '完整24层gate/up权重MSE额外小数位最多＋1；原Q8开发门槛通过，本行CPU是原匹配轨迹，不继承其他组合独立质量'),
        ('1B cap1＋输出头DFP MSE',1024,32,ROOT/'cap1-headmse-trial-20261013/performance.json','candidate',
         ROOT/'thermal-aligned-20261013/intern1b-h1024-n32-medium-cpu/results.json',
         '组合方案单独完整开发验证和五轮，与原gate/up MSE对照；不把两个独立收益相加。独立v2为33/36单例通过，未过整套质量。本行CPU是原匹配轨迹'),
        ('1B短历史cap1＋输出头DFP MSE',512,64,ROOT/'cap1-headmse-short-trial-20261014/performance.json','candidate',
         ROOT/'cpu-q8-search-20261013/intern1b-h512-n64-final-fa-off/results.json',
         '全24层容量768、t16、完整词表，直接使用当前最快CPU五轮轨迹；与原DFP MSE交替五轮。容量改变后的完整开发门槛单独检查，不能继承长历史组合独立质量'),
        ('1B短历史cap1＋原输出头',512,32,ROOT/'cap1-originalhead-short-trial-20261014/performance.json','candidate',
         ROOT/'cpu-q8-search-20261013/intern1b-h512-n32-final-fa-off/results.json',
         '组合headMSE在本开发场景PPL比1.053872未过；保留cap1 decoder、恢复原head后完整执行及原Q8门槛通过，PPL比1.048677，仍接近边界。无独立质量通过结论'),
        ('500M DFP＋prefill8',512,64,ROOT/'smol500-prefill8-trial-20261014/performance.json','candidate',
         ROOT/'cpu-q8-search-20261013/smol500-h512-n64-final-fa-off/results.json',
         '全部32层，原decode与head不变，仅t32改t8；同温五轮首字改善，decode基本持平。原Q8开发门槛通过，冻结陪伴套件回归另列，非产品验收')]
    records=[]
    frontier=[]
    lines=['# 全深度加速候选：速度与系统开销','',
           '按表中精确历史/新增token计数，均为128次预测、Q8对应比较、中温55–<65°C起测、五轮。500M全32层，1B全24层，均包括完整词表输出头。温档约束响应开始温度，响应中自然升降温。','',
           '下表场景的候选完整执行门槛已通过。早期候选独立v1留出已完成执行：500M有单例数值失败，两模型陪伴自然度均未验收通过。新head/cap1/组合不能继承该结论，各自状态另列；这份报告不表示已超过最优CPU或完成产品验收。','',
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
        lines.append(f'| {display} | {b["decode_tokens_per_second"]:.3f}→{n["decode_tokens_per_second"]:.3f} | {gain:+.2f}% | {b["ttft_seconds"]:.3f}→{n["ttft_seconds"]:.3f} | {c["decode_tokens_per_second"]:.3f} | {engine_ttft:.3f} |')
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
            '上表CPU列保持与该次NPU测量相同的反馈轨迹，不能把较旧对照误读成最优CPU。四个模型/历史档的规定CPU配置搜索已完成；下面单列搜索得到的最快decode配置。两轮仅为筛选，五轮确认也须检查反馈轨迹；[最新对齐矩阵与直接负载](最新CPU对齐与组合候选矩阵.md)单独收录重新回灌的结果。','',
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
    lines+=['','继续验收：扩展32/128新增与1024历史、最快CPU轨迹重对齐，以及新优化候选的精度和性能。两模型原候选的独立留出、自然多轮、实验HTTP、30分钟稳定性及真实ASR并发已有结果，不能自动转移给后来改变权重或执行路径的新候选。','',
            '两模型真实HTTP/浏览器多轮已通过接口检查，见[HTTP报告](HTTP多轮接口验证.md)；独立质量和持续运行进度见[产品验收](独立质量与连续多轮验收.md)，接口通过不等于陪伴质量通过。','',
            '复现：`npu/.venv/bin/python npu/kv-cache/report_candidate_speed.py`。',
            '[紧凑证据](evidence-attack-20261013/candidate-speed-load.json)。','']
    (HERE/'全深度候选提速与负载.md').write_text('\n'.join(lines))
    (HERE/'evidence-attack-20261013/candidate-speed-load.json').write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n')
    (HERE/'evidence-attack-20261013/cpu-speed-frontier.json').write_text(json.dumps(frontier,indent=2,ensure_ascii=False)+'\n')
    print('completed candidate comparisons',len(records))


if __name__=='__main__':main()
