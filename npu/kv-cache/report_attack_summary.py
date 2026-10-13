"""One readable checkpoint of the five-stage attack, preserving failed gates."""
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo
from report_candidate_matrix import rejection_reason

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'results'


def read(path):return json.loads(path.read_text())


def main():
    rows=read(HERE/'evidence-attack-20261013/finalist-aligned-matrix.json')
    failures=read(HERE/'evidence-attack-20261013/finalist-aligned-quality-failures.json')
    qualified=[r for r in rows if r['trajectory_aligned'] is True and
               ('配置搜索complete' in r['cpu_note'] or '低温复核' in r['cpu_note'])]
    cpu=read(ROOT/'cpu-q8-search-20261013/state.json')
    low=read(ROOT/'cpu-q8-low-finalists-20261013/state.json')
    clean=read(ROOT/'archived-board-dumps-prune-20261014.json')
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    lines=['# 本轮NPU技术攻关结果','',f'生成时间：{stamp}。','',
           '已有完整模型的实际提速，但尚未超过优化CPU，也没有通过完整陪伴产品验收。全部比较执行500M的32层或1B的24层及完整词表输出头；不拿前几层探针代替模型成绩。','',
           '## 1. 对齐测量','',
           f'CPU中温规定配置搜索状态：`{cpu["status"]}`；低温入围复核：`{low["status"]}`。最新候选中，CPU配置确认、同温五轮且反馈一致的格子为{len(qualified)}/24。正在运行、原轨迹未对齐和质量失败都不是0速度，也不算完成。各格可能为不同显式标记的候选版本，尚未形成同一固定权重覆盖全场景的产品；也未证明运行时切换版本无开销。','',
           '核心矩阵是512/1024已有KV × 32/64/128新增token × 低/中起温。每格预热＋5次、固定128预测/127decode间隔。低温响应起点40–50°C、中温55–<65°C；响应温度自然变化，不冒充恒温。CPU Q8_0与NPU W8为主的混合精度对照，激活/KV主要INT16，敏感位置FP16。','',
            '引擎首字不含模型加载、历史KV准备、控温及HTTP缓冲。直接系统采样与本轮响应配对，另报覆盖率；生产CPU网页在两端都驻留。[全部档位、版本、质量失败和资源](最新CPU对齐与组合候选矩阵.md)。','',
            '![中温完整语言链路：解码与首字，空位为尚未配对](finalist-aligned-medium.png)','',
           '## 2–3. 输出头与decoder的完整模型收益','',
           '| 场景 / 候选 | 同轮旧NPU→新NPU token/s | 提升 | 首字旧→新 s | 驱动减少 MiB |',
           '| --- | ---: | ---: | ---: | ---: |']
    trials=[('500M512＋64 / gate-up DFP','smol500-dfpgateup-ab-20261013.json','dfp'),
            ('1B1024＋32 / cap1＋headMSE','cap1-headmse-trial-20261013/performance.json','candidate'),
            ('1B512＋64 / cap1＋headMSE','cap1-headmse-short-trial-20261014/performance.json','candidate'),
            ('1B512＋32 / cap1＋原输出头','cap1-originalhead-short-trial-20261014/performance.json','candidate')]
    for name,path,label in trials:
        d=read(ROOT/path);assert d['status']=='complete'
        assert all(sum(r['phase']=='measure' and r['label']==v for r in d['runs'])==5 for v in ('baseline',label))
        b,c=d['medians']['baseline'],d['medians'][label]
        gain=c['decode_tokens_per_second']/b['decode_tokens_per_second']-1
        lines.append(f'| {name} | {b["decode_tokens_per_second"]:.3f}→{c["decode_tokens_per_second"]:.3f} | {gain:+.2%} | {b["ttft_seconds"]:.3f}→{c["ttft_seconds"]:.3f} | {(b["driver_allocated_bytes"]-c["driver_allocated_bytes"])/2**20:.2f} |')
    lines+=['','各行是独立配对实验，收益不能跨行相加。512＋32的原输出头版本仅证明该开发轨迹恢复（KL 0.009544、PPL比1.048677），尚无独立质量验收，不能写作组合方案全部过关。驱动是共享系统内存分配，不是独立显存；不能把它与进程RSS直接相加当作唯一物理内存。','',
            '| 当前已对齐中温64新增 | NPU / CPU token/s | NPU / CPU首字 s | NPU / CPU整机CPU % |',
            '| --- | ---: | ---: | ---: |']
    for r in qualified:
        if r['band']!='medium' or r['new_tokens']!=64:continue
        m=r['npu']['medians'];c=r['cpu'];resources=r.get('direct_resources',{})
        busy='缺测'
        if resources.get('npu') and resources.get('cpu'):
            busy=f'{resources["npu"]["medians"]["busy_percent"]:.2f} / {resources["cpu"]["medians"]["busy_percent"]:.2f}'
        lines.append(f'| {r["model"]} / 历史{r["history"]} | {m["decode_tokens_per_second"]:.3f} / {c["decode_tokens_per_second"]:.3f} | {m["ttft_seconds"]:.3f} / {c["engine_ttft_seconds"]:.3f} | {busy} |')
    lines+=['','整机CPU按8核合计为100%。NPU明显释放CPU核，但目前这不等于单任务速度超过CPU，也不能据此计算电功率。真实ASR并发已有[独立实测及CPU分核对照](真实ASR与LLM并发验收.md)，新图不能自动继承旧版本结果。','',
            '新cap1普通两层的FC及融合激活仍占91.51% busy cycles，attention matmul约4.00%。这只定位该层组，详细来源见[硬件计数](Decoder矩阵瓶颈与候选筛选.md)。按完整模型实测拆分，宿主剩余时间全消除也仅约几个百分点；需要减少矩阵执行与访存代价，不能只靠继续改宿主宣告达标。','',
            'Query/RoPE两处INT16范围修复，使原1B1024＋64的完整执行L2从5.208%降至4.961%，原Q8开发KL/PPL亦通过；该轨迹五轮14.725token/s、首字1.248秒，只作为精度修复，不与失败旧版算速度收益。','',
            '[A8通道缩放及选择性down筛选](MLP通道缩放开发筛选.md)没有得到可推广候选，主线保留W8/A16。'
            '新增[down逐通道缩放拆分](Down投影缩放拆分试验.md)保留了原INT8整数与有效权重，两模型局部精度通过，'
            '但耗时变化均不足0.2%，暂不扩展完整模型。新失败按原门槛记录：','']
    for case,r in failures.items():
        q=r['original_q8_development'];v=r['validation']
        lines.append(f'- {case}：拒绝原因为{rejection_reason(r)}。执行与原Q8质量两组门槛均须通过。')
    lines+=['','## 4. 首字与prefill','',
            '1B t16相比t32在512历史三种新增长度上完整五轮首字缩短约16%。500M t64已负收益，未采用。短尾块补充场景500M512＋20首字0.856→0.461秒（−46.13%），完整32层及原Q8开发门槛通过；1B512＋12未过执行门槛，未计正式性能。20/12新增是补充场景，不代替核心32/64/128矩阵，也不重复计算已有KV缓存收益。','',
            '当前量化路径8-token筛选：1B局部单位token成本反而高约39%，不扩大。500M已经扩展完整32层、512＋64、同温交替五轮：'
            '首字0.921722→0.752878秒（−18.32%），总响应7.110332→6.971551秒（−1.95%），'
            'decode20.521247→20.432035基本持平，驱动607.73→575.00MiB。当前CPU首字约0.704秒，仍更快。'
            '该性能结果只覆盖这个中温场景，其他输入/历史仍须另验。[批次与完整模型证据](Prefill批次筛选.md)。','',
            '## 5. 产品验收与未达标项','',
            '最新1B组合独立v2单例33/36通过、仅新输出头32/36，均未过整套数值门槛。500M既有候选v1有12条数值例外（24/36通过）；'
            '新prefill8已另做[冻结套件回归](500M-prefill8陪伴回归.md)，仍为24/36，通过数未改善，不作产品验收。'
            '标准Q8参考逐文件SHA验证后复用，不拿候选自身预测当参考，也不称新独立套件。'
            '实际CPU和NPU自由回答都存在模型本身的更正记忆、指令遵循或复读问题，详见[独立v2原文](1B%20cap1与新输出头组合独立v2验收.md)和[两模型v1结果](独立质量与连续多轮验收.md)。'
            '不能把聚合KL/PPL通过写成每条通过，也不能针对留出调参后继续称独立。','',
            '旧候选两模型30分钟、实际浏览器多轮和ASR并发已有证据；这些只验对应图版本，不自动转移到新图。当前实验HTTP是主机网关＋板端常驻NPU/KV，尚非板端独立NPU聊天网站。生产8080保持CPU，未推广未通过质量的候选。','',
            '继续完成待跑矩阵及输出头消融；保留同量化最强已测CPU目标线，合格候选再进入新版稳定性与产品流程。若质量或编译器限制持续存在，报告实际未达标项，不切换慢基线。','',
            '## 归档与可恢复清理','',
            f'本轮清理{len(clean["files"])}份已在本机完整SHA256验证的板端诊断副本，释放{clean["removed_bytes"]/2**30:.3f}GiB；结束时可用{clean["free_after"]/2**30:.3f}GiB，后续会变化。全部可按[恢复清单](TF容量恢复与归档.md)从本机还原，模型/系统/结果JSON保留。Git备份源码、报告和紧凑证据，大模型及完整logits不进入Git。','',
            '全部测量结束后的[只读状态核对](evidence-attack-20261013/closeout-health-20261014.json)：网页HTTP 200，温度38.936°C，NPU驱动分配归零；'
            '系统可用内存4618MiB，TF可用约2.75GiB。空闲swap已有1164MiB占用，不等于本轮响应正在换页；响应内实际换入/换出见配对采样。此处是空闲检查，不是推理峰值。','',
            '重建前先运行`report_candidate_matrix.py --finalists`，再运行`report_attack_summary.py`。','']
    (HERE/'本轮技术攻关结果.md').write_text('\n'.join(lines))
    print('summary aligned cells',len(qualified),'quality rejections',len(failures))


if __name__=='__main__':main()
