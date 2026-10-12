"""Archive current full-depth Q8 experiments, including failed precision candidates."""
import json
from pathlib import Path
import shutil

here=Path(__file__).resolve().parent;results=here.parent/'results'
archive=here/'evidence-q8-optimization-20261012';archive.mkdir(exist_ok=True)
lines=['# Q8 完整生成优化进展','',
    '2026-10-12。按用户最终口径：500M、1B 的历史 CPU 都是 Q8_0，未测 Q4；本轮优化 NPU W8，并与对应 CPU Q8 比较。Q4 编译探索及 FP16 下载已暂停，不加入本轮目标。', '',
    '这是研发进展，不是最终验收。核心历史档位、独立留出质量、30 分钟稳定性、最快 CPU Q8 的完整搜索尚未完成；不能宣称超过 CPU。', '',
    '## 完整生成和精度','',
    '下表执行误差以编译产物中的矩阵、QKV bias、Norm 常量及输出头权重重建 FP32 参考，按 NPU 实际生成轨迹逐步比较。词表分块仍覆盖全部 token。激活/KV 在参考端为 FP32。NPU 的敏感 FP16 例外另列；W8 并不表示激活与 KV 也是 8 位。', '',
    '| 模型/变体 | 样例 | 完整 logits L2 | 最低余弦 | 最高分 token 一致 | 开发执行门槛 |',
    '| --- | --- | ---: | ---: | ---: | --- |']
experiments=[
    ('500M','full-smol500-stable-v3-c128','w8-mixed-biasfix','companion-hostv2'),
    ('500M','full-smol500-stable-v3-c128','w8-head-a16','companion'),
    ('500M','full-smol500-stable-v3-c128','w8-head-a16','greeting'),
    ('500M','full-smol500-stable-v3-c128','w8-head-wide','greeting'),
    ('500M','full-smol500-stable-v3-c128','w8-head-wide','companion'),
    ('1B','full-intern1b-sharded-v2-c128','w8-balanced-headwide','companion-v2'),
    *[('1B','full-intern1b-sharded-v2-c128',v,None) for v in
      ('w8-sensitive','w8-vdfp-full','w8-mixed-l21','w8-tight','w8-mixed-early','w8-mixed-tight','w8-mixed-l8','w8-balanced','w8-balanced-head','w8-score-fix','w8-balanced-headnorm','w8-balanced-headwide')]]
answers=[]
for model,source,dtype,case in experiments:
    folder=results/source/dtype/('board'+('-'+case if case else ''))
    path=folder/'full-validation-w8.json'
    if not path.exists():continue
    d=json.loads(path.read_text());dest=archive/(model+'-'+dtype+'-'+(case or 'greeting'));dest.mkdir(exist_ok=True)
    for name in ('full-validation-w8.json','full-validation.json','tokens.json','prompt.json','exit.json','deployed-sha256.txt','run.stderr','local-layer-audit.json'):
        if (folder/name).exists():shutil.copyfile(folder/name,dest/name)
    lines.append(f'| {model} / {dtype} | {case or "问候"} | {d["relative_l2_all_logits"]*100:.3f}% | {d["min_step_cosine"]:.6f} | {d["argmax_matches_fp32"]}/{d["generated_token_count"]} | {"通过" if d["execution_gate_v1"]["passed"] else "未通过"} |')
    answers += [f'### {model} / {dtype} / {case or "问候"}', '',
        '> '+d['npu_answer'].replace('\n','\n> '), '',
        f'诊断运行 {d["runtime"]["decode_tokens_per_second"]:.3f} token/s，首字 {d["runtime"]["ttft_seconds"]:.3f} 秒。含 logits 写盘，不与干净性能成绩混用。','']
    original=folder/'full-validation.json'
    if original.exists():
        ref=json.loads(original.read_text())
        answers += [f'相对原始 GGUF Q8 反量化参考（包含 NPU 再量化的新增误差）：logits L2 {ref["relative_l2_all_logits"]*100:.3f}%，平均 KL {ref["mean_kl_nats"]:.6f} nat，最高分 token 一致 {ref["argmax_matches_fp32"]}/{ref["generated_token_count"]}。不是 BF16 原始训练权重参考。','']
lines += ['', '正常说话不等于数值通过：1B 的 V 修复先恢复可读文本，但完整误差仍需继续降低。所有上表样例均为开发集，未冒充独立留出验收。', '',
    '当前通过两种开发输入的提速候选：500M `w8-head-wide`（decoder 第 3/31 层 MLP FP16，head W8）；1B `w8-balanced-headwide`（decoder 第 2/3 层 MLP FP16，四片 head W8）。两者其他 decoder 矩阵 W8，残差/Norm/softmax 含 FP16。1B 陪伴样例达到 64-token 上限，回答被截断，不称自然 EOS 完整对话。', '',
    '## 已验证的改动','',
    '- 适配 Qwen2 的残差节点：按图依赖定位，避免沿用 Smol 的节点编号。masked softmax 也按算子定位。',
    '- 1B 原生逐通道 V 路径在首层输出 128 个零；仅将 V 改为 DFP INT8 后恢复。独立两层对照 L2 从 104.626% 降至 3.797%，其余六个矩阵保持逐通道 W8。两层只能用于定位。',
    '- 1B 的 3-bit 激活范围余量过宽；较紧的范围改善普通 token 精度。第 2/3 层（0 起算）MLP 的离群值需单独处理。逐层局部误差使用真实进入该层的 NPU hidden 历史，避免把上游误差全归给后层。',
    '- 500M 输出头成功采用 W8/A16（INT16 Norm 路径），保留 decoder 中第 3/31 层 MLP FP16。此前 FP16 Norm 与 W8 head 的两种混合尝试均在图验证阶段失败，原日志保留，不能据此声称所有 W8 head 不可用。',
    '- 初版 W8 head 陪伴通过、问候失败；在保持 W8 的前提下扩大 head 激活范围，两种输入均通过。1B 的 head 也需 FP16 内部 Norm 或扩大范围，不能只改矩阵权重就默认精度不变。',
    '- 运行时将同位置 RoPE 复用于所有层，mask 每步只打开一个历史槽；KV 在类型和 scale 相同的情况下直接复制，否则保留数值转换。500M 67-token 对话的输出 token 与完整 logits 对旧运行器逐值完全一致。',
    '- 新性能模式关闭 logits/trace/逐 token 日志，但仍执行全层、head、采样和 KV 更新。进入测量前在排他锁内等待温度降至 55°C 以下。', '',
    '## 500M 输出头速度 A/B','']
path=results/'smol500-head-performance-cooled.json'
if path.exists():
    d=json.loads(path.read_text());shutil.copyfile(path,archive/path.name)
    lines += ['两版本共用同一运行器；每版先预热一次，再交替五次。每次验证 67 个自然 EOS 输出 token 与原轨迹完全一致，且驱动释放后为 0。只比较当前短历史开发样例。','']
    if d['status']=='complete':
        lines += ['| 变体 | 中位 decode token/s | 中位首字秒 | 中位总响应秒 | CPU 秒 | 驱动分配 MiB |', '| --- | ---: | ---: | ---: | ---: | ---: |']
        for name,m in d['medians'].items():
            lines.append(f'| {name} | {m["decode_tokens_per_second"]:.3f} | {m["ttft_seconds"]:.3f} | {m["total_seconds"]:.3f} | {m["cpu_seconds"]:.3f} | {m["driver_allocated_bytes"]/1048576:.2f} |')
        lines += ['',f'本轮中位 decode 提升 {(d["speedup"]-1)*100:.1f}%。温度、每次 token 序列、时延及分配见原始 JSON。']
    else:lines += [f'尚未完成，仅归档 {len(d["runs"])} 次运行（含预热），不报最终中位收益。']
    lines += ['','第一次 A/B 因某次起始温度高于 55°C 被拒绝，保留失败记录；上表为加入锁内冷却后的另一次运行。','']
lines += ['## 最终候选的重复性能对照','']
for model,name in [('500M','smol500-headwide-performance-cooled.json'),('1B','intern1b-headwide-performance-cooled.json')]:
    path=results/name
    if not path.exists():continue
    d=json.loads(path.read_text());shutil.copyfile(path,archive/name)
    lines += [f'### {model}', '',
        '同一运行器、相同 token 轨迹、预热一次加五次交替测量。1B 为短问候，不能直接等同 500M 的 67-token 陪伴负载。','']
    if d['status']!='complete':
        lines += [f'进行中，已归档 {len(d["runs"])} 次（含预热）。',''];continue
    lines += ['| 变体 | 中位 decode token/s | 首字秒 | 总响应秒 | CPU 秒 | 驱动 MiB |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for variant,m in d['medians'].items():
        lines.append(f'| {variant} | {m["decode_tokens_per_second"]:.3f} | {m["ttft_seconds"]:.3f} | {m["total_seconds"]:.3f} | {m["cpu_seconds"]:.3f} | {m["driver_allocated_bytes"]/1048576:.2f} |')
    lines += ['',f'本场景 decode 中位提升 {(d["speedup"]-1)*100:.1f}%；不是与 CPU 的提升比。','']
lines += ['某次 1B 陪伴开发试跑与问候性能脚本共享远端 prompt 文件，实际执行了错误提示词；该记录作废，不用于模型精度结论。运行器现改用每个 case 独立 prompt/可执行文件，并将编译纳入排他锁；修复后重跑的 `companion-v2` 才纳入上表。','',
    '## 提交批处理与共享 hidden','',
    '新增可选 VIPLite group 路径：相邻层 FP16 hidden 直接绑定同一设备缓冲，整组只等待一次完成，各层 KV 在整栈执行后分别增量更新。权重和计算图未变；这是提交批处理，不是算子融合。保留原逐层路径作对照，不支持的 hidden 类型/形状立即拒绝。分组路径不伪造逐层时间，层时间记录为 null。','',
    '实验在同一板端排他锁内完成：先对每个模式验证生成 token 及所有输出 logits 逐字节一致，再各预热一次、交替五次测速。性能运行前温度不高于55°C，所有结果要求驱动释放为0。诊断写盘不计入性能成绩。','']
for model,name in [('500M','smol500-submission-performance.json'),('1B','intern1b-submission-performance.json')]:
    path=results/name
    if not path.exists():continue
    d=json.loads(path.read_text());shutil.copyfile(path,archive/name)
    lines += [f'### {model} 提交方式对照','']
    if d['status']!='complete':
        lines += [f'状态 {d["status"]}，保留 {len(d["runs"])} 次记录；不报最终收益。',''];continue
    lines += ['| 每组层数（0为原逐层路径） | decode 中位/范围 token/s | 首字秒 | 总响应秒 | CPU秒 | decoder API 毫秒/生成步 | head API 毫秒/生成步 |',
              '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    n=len(d['config']['tokens'])-1
    for group,m in d['medians'].items():
        speeds=[r['result']['decode_tokens_per_second'] for r in d['runs'] if r['phase']=='measure' and str(r['group'])==group]
        lines.append(f'| {group} | {m["decode_tokens_per_second"]:.3f} / {min(speeds):.3f}–{max(speeds):.3f} | {m["ttft_seconds"]:.3f} | {m["total_seconds"]:.3f} | {m["cpu_seconds"]:.3f} | {1000*m["decode_decoder_api_seconds"]/n:.3f} | {1000*m["decode_head_api_seconds"]/n:.3f} |')
    baseline=d['medians']['0'];best=max(d['medians'],key=lambda g:d['medians'][g]['decode_tokens_per_second'])
    winner=d['medians'][best]
    lines += ['',f'本场景最快为每组 {best} 层，相对逐层路径中位速度变化 {(winner["decode_tokens_per_second"]/baseline["decode_tokens_per_second"]-1)*100:.2f}%。API 时间包含运行时/驱动等待，不等于纯硬件算术时间；整栈并未因合并提交而少算任何层。',
        f'最快模式的总响应 {winner["total_seconds"]:.3f} 秒中，NPU调用累计中位数 {winner["npu_api_seconds"]:.3f} 秒（各指标分别取中位数，比例仅作定位）。主要瓶颈已落在调用内部，继续削减调用外Python/C编排开销无法单独追上CPU。','']
    if best!='0':
        paired={g:{r['repeat']:r['result']['decode_tokens_per_second'] for r in d['runs'] if r['phase']=='measure' and str(r['group'])==g} for g in ('0',best)}
        wins=sum(paired[best][i]>v for i,v in paired['0'].items())
        lines += [f'按同轮序号比较，分组路径 {wins}/{len(paired["0"])} 次更快。'+('每次均改善，但仍只代表当前样例。' if wins==len(paired['0']) else '存在倒退轮次，中位数改善不等于稳定加速；保留为可选实验路径。'),'']
lines += ['## 提速投入的下一优先级','',
    '按已测时间优先处理decoder内部执行：2/4层计算图融合、GQA展开/Concat/布局转换，以及少数敏感MLP的高效执行。当前将整个MLP保留FP16会增加耗时；后续应尝试仅保留必要的高精度边界，不能未经质量验证直接降回INT8。', '',
    'head已经从FP16改为W8；在500M最终候选原逐层路径中，decode平均每步decoder API约134ms、head约5ms；1B短问候相应约115ms、17ms。继续只优化500M head的上限很小，主要开发精力转向decoder。这里是API墙钟，尚未得到逐硬件算子的细分计时。', '',
    '首字仍采用逐token增量prefill，下一步需独立批量prefill图并验证与decode的KV兼容；容量分档和有效历史长度优化需新增长历史验证。本轮没有把这些未实现机制计入已达成收益。','']
for name in ('smol500-head-performance-ab.json','cpu-screen-smol500-q8-cooled.json'):
    path=results/name
    if path.exists():shutil.copyfile(path,archive/name)
lines += ['## CPU Q8 对照的边界','',
    '历史六模型评测中，SmolVLM-500M 与 InternVL3-1B 均为 Q8_0；有 Q4 的是其他 Qwen 模型，不能套用它们的 Q4 成绩。', '',
    '500M 新完成 9 组线程/绑核配置，每组预热后 5 次、128 个输出 token、起始温度 ≤55°C。最快为 2 线程绑 A76 核 6/7，27.516 token/s。它是线程初筛，尚未完成 attention/KV/batch 等全部优化；其提示词/输出轨迹与上述 NPU 陪伴 A/B 不同，不能直接计算正式 CPU/NPU 胜负。', '',
    '## 当前量化算法与 SDK 预制选项','',
    '当前是训练后量化 PTQ：先以现有 GGUF Q8_0 反量化权重导出 ONNX，ACUITY 校准激活，再覆写指定权重/敏感算子的量化配置。它不是 AWQ/GPTQ，不是量化感知训练 QAT，也不是直接执行 GGUF Q8_0 内核。校准仅少量开发输入，是目前泛化精度的一项限制。', '',
    '| 部分 | 当前实际使用 |', '| --- | --- |',
    '| 常规权重 | 逐输出通道对称 INT8，scale=max(abs(weight))/127，最近偶数舍入 rtne |',
    '| V 权重 | DFP INT8，2 的整数次幂 scale，规避已复现的完整图 V 归零路径 |',
    '| 激活/KV | SDK 校准的 DFP INT16，部分位置手动设置范围余量；敏感位置 FP16 |',
    '| 校准方法 | 脚本未显式指定 --algorithm，采用 SDK 默认流程；未系统 A/B 比较 KL/MLE 等 |',
    '| CPU 对照 | llama.cpp GGUF Q8_0：分块 INT8，与 NPU 的 per-channel/DFP 不是同一编码 |','',
    '已实际读取官方镜像 ACUITY 6.30.22 的 `pegasus.py quantize --help`：量化器有 dynamic_fixed_point、asymmetric_affine、symmetric_affine、perchannel_symmetric_affine，以及浮点格式选项；校准算法入口有 normal、moving_average、kl_divergence、auto，另有 hybrid 和 MLE（逐层误差最小化）选项。CLI 列出能力不表示 A733 每种算子组合都能执行，必须编译并上板确认。SDK 也列出 int4/uint4，但本轮按用户要求暂停 Q4。','',
    '用户要求增加提速投入：以通过精度门槛的 W8 为起点，优先矩阵/head执行路径、2/4 层融合和中间布局/拷贝、容量分档、批量 prefill；校准优化用于修复实际退化和减少高精度例外。当前尚未新增融合图、PagedAttention 或 FlashAttention 成功结果。','',
    '## 当前未完成','',
    '两模型在独立质量集和长历史上的精度验收、同轨迹 CPU Q8 比较、历史 128/512/2048（容量须包含输出）、完整首字/增量 prefill、资源曲线和 30 分钟持续运行仍待完成。CPU/NPU 同档位量化不代表编码相同，应同时报告相对原 CPU Q8 的新增量化误差。', '',
    '下一步按验收计划推进图融合、容量分档和批量 prefill；没有从仅两层的速度外推完整模型达标。网页保持 CPU 后端。','',
    '## 实际生成文本','',*answers]
for tag in ('intern1b-w8-native-correct-ref','intern1b-w8-vdfp-two-layer'):
    folder=results/'intern1b-firstlayers-c128'/tag;dest=archive/tag;dest.mkdir(exist_ok=True)
    for name in ('first-step-audit.json','board-result.json','run.stderr'):
        if (folder/name).exists():shutil.copyfile(folder/name,dest/name)
for dtype in ('w8-fasthead','w8-fasthead-dfp'):
    for name in ('conversion.json','export.log'):
        src=results/'full-smol500-stable-v3-c128'/dtype/'l32'/name
        if src.exists():shutil.copyfile(src,archive/(dtype+'-'+name))
(here/'Q8完整模型优化进展.md').write_text('\n'.join(line.rstrip() for line in '\n'.join(lines).splitlines()).rstrip()+'\n')
print(here/'Q8完整模型优化进展.md')
