"""Compact auditable snapshots of CPU screening and complete-model head A/B."""
import hashlib
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'results'
ARCHIVE=HERE/'evidence-attack-20261013'


def read(path): return json.loads(path.read_text())
def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    ARCHIVE.mkdir(exist_ok=True)
    matrix=read(ROOT/'thermal-aligned-20261013/state.json')
    cpu=read(ROOT/'cpu-q8-search-20261013/state.json')
    head=read(ROOT/'head-matrix-20261013/state.json')
    complete=sum(v.get('status')=='complete' for c in matrix['cases'].values()
                 for b,v in c['bands'].items() if b in ('low','medium'))
    lines=['# 下一轮攻关进展：测量、CPU搜索与NPU候选','',
           '2026-10-13。本报告为运行中快照；全语言深度、Q8对应比较、低/中温范围不变。','',
           f'低/中温CPU/NPU配对矩阵完成{complete}/24组；CPU搜索状态`{cpu["status"]}`，head扩展状态`{head["status"]}`。', '',
           '统一引擎首token及直接系统资源采样已落地，见[计时与采样说明](首字计时与直接资源采样.md)。完整矩阵和曲线见[全深度对标](三温档全深度对标.md)。','',
           '## 1B输出头：相同图、相同完整输出','',
           '串行与grouped在同场景交替5轮；每个候选先复现全部128步全词表logits。表中两端都为NPU，CPU对手仍需对应场景的最快合格配置。','',
           '| 历史＋新增 | 串行 token/s | 合并 token/s | 提升 | 配对更快次数 | 合并首token s | 合并整机CPU % |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for case, row in head['cases'].items():
        if row['status']!='complete':continue
        path=Path(row['result']);raw=read(path);assert raw['status']=='complete' and row['logits_identical']
        runs=[r for r in raw['runs'] if r['phase']=='measure' and r['label']=='grouped'];assert len(runs)==5
        busy=statistics.median(r['resource_summary']['cpu']['cpu']['busy_percent'] for r in runs)
        b,c=row['serial'],row['grouped']
        lines.append(f'| {case.removeprefix("intern1b-")} | {b["decode_tokens_per_second"]:.3f} | {c["decode_tokens_per_second"]:.3f} | {row["decode_gain_percent"]:+.2f}% | {sum(v>0 for v in row["paired_gains_percent"])}/5 | {c["ttft_seconds"]:.3f} | {busy:.2f} |')
        compact={k:v for k,v in raw.items() if k!='runs'}
        compact['source_path']=str(path);compact['source_sha256']=sha(path)
        compact['runs']=[{k:v for k,v in r.items() if k!='system_samples'} for r in raw['runs']]
        (ARCHIVE/(case+'-head-ab.json')).write_text(json.dumps(compact,indent=2)+'\n')
    lines+=['','此前独立512＋32 A/B约+2.30%；本轮复测保留新的配对结果，不把两轮相除拼成加速比。未完成场景不外推。', '',
            '## CPU Q8初筛与确认','',
            '初筛每配置预热一次、测两轮，入围再五轮；最终扩展32/128新增。预测与冻结轨迹不同仅表示不能直接沿用该NPU反馈对照，需要重新对齐和质量验证，不等于质量自动失败。不能忽略更快但尚未验证的配置就宣称超过最佳CPU。','',
            '| 场景 | 配置 | 轮数 | decode token/s | 引擎首token s | 冻结轨迹一致 |',
            '| --- | --- | ---: | ---: | ---: | --- |']
    for case,row in cpu['cases'].items():
        for label,result in row['runs'].items():
            med=result.get('medians',{})
            if not med:continue
            display_case=case
            display_label=label
            if label.startswith(('n32-','n128-')):
                new,display_label=label.split('-',1)
                display_case=case.rsplit('-n',1)[0]+'-'+new
            lines.append(f'| {display_case} | {display_label} | {result["repeats"]} | {med["decode_tokens_per_second"]:.3f} | {med.get("engine_ttft_seconds",float("nan")):.3f} | {"是" if result["aligned"] else "否，待对齐"} |')
            path=Path(result['path']);raw=read(path)
            compact={k:v for k,v in raw.items() if k not in ('runs','nonstream_trajectory')}
            compact.update(source_path=str(path),source_sha256=sha(path))
            compact['runs']=[{k:v for k,v in r.items() if k not in ('system_samples','health_before','health_after')}
                             for r in raw['runs']]
            (ARCHIVE/(case+'-'+label+'-cpu.json')).write_text(json.dumps(compact,indent=2,ensure_ascii=False)+'\n')
    lines+=['','6线程首次历史准备达80.948°C被保护终止，不计成绩；后续把准备起温降至50°C并给首次轨迹采集温控，正式响应仍55–<65°C。热停止与存储故障不能混为一谈，失败原始记录保留。','',
            '## Decoder定位与待完成工作','',
            '已从1B全24层512＋32运行抓取真实decode输入（位置544、block0）；抓取版的128步完整logits与已验收基线逐字节相同，独立全模型导出常量门槛也通过。5份硬件计数中fullconnect占busy cycles约91.55%，优先检查矩阵与权重搬运，见[Decoder矩阵瓶颈与候选筛选](Decoder矩阵瓶颈与候选筛选.md)。小图仅定位瓶颈，不进入全模型速度排行。','',
            '独立v1留出、实际CPU自然回答、实验HTTP/浏览器和两模型30分钟常驻多轮已完成执行，具体质量失败与接口边界见[产品验收](独立质量与连续多轮验收.md)。这不等于陪伴产品通过。[真实ASR并发及CPU分核复核](真实ASR与LLM并发验收.md)均已完成。剩余测量包括当前DFP矩阵、CPU配置搜索，以及新输出头、decoder缩放和短尾块候选的完整验收；新输出头使用另行冻结的[独立v2套件](1B新输出头独立v2验收.md)。','',
            '最新矩阵明确区分待测与质量失败，见[当前DFP矩阵](当前DFP候选精确矩阵.md)。1B 1024＋64的执行L2约5.208%超过5%门槛，正在定位；不放宽阈值、不列为合格速度。当前仍不能宣布超过最快CPU。','',
            '复现快照：`npu/.venv/bin/python npu/kv-cache/report_attack_iteration.py`。紧凑证据保存源报告SHA256、配置、每轮结果和响应内资源摘要；完整时间序列保留在本地`npu/results`。','']
    (ARCHIVE/'state-summary.json').write_text(json.dumps({'matrix_completed_pairs':complete,'cpu':cpu,'head':head},indent=2)+'\n')
    (HERE/'下一轮攻关进展.md').write_text('\n'.join(lines))
    print(json.dumps({'matrix_pairs':complete,'head_complete':sum(r['status']=='complete' for r in head['cases'].values()),
                      'cpu_cases_complete':sum(r['status']=='complete' for r in cpu['cases'].values())}))


if __name__=='__main__':main()
