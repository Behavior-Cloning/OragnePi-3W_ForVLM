"""Archive the INT8 isolation matrix and measured repair without hiding failures."""
import json
from pathlib import Path
import shutil
import statistics
import numpy as np

root=Path(__file__).resolve().parents[2];results=root/'npu/results';here=Path(__file__).resolve().parent
archive=here/'evidence-int8-repair-20261012';archive.mkdir(exist_ok=True)
variants=[('int16-matched-calibration','原INT16对照'),('w8a16','七矩阵逐通道INT8'),
          ('w8dfp','七矩阵DFP INT8'),('w8-attn','只量化attention四矩阵'),('w8-mlp','只量化MLP三矩阵'),
          ('w8-qk','只量化Q/K'),('w8-vo','只量化V/O'),('w8-o','只量化O'),('w8-no-o','其余六矩阵，O保留INT16'),
          ('w8-v','只量化V'),('w8-no-v','其余六矩阵，V保留INT16'),('w8-vdfp','全部INT8，V改DFP')]
lines=['# INT8误差定位与修复','',
       '2026-10-12。SmolVLM-500M语言部分，A733 VIP9000，ACUITY6.30.22、VIPLite2.0.3.2。',
       '用户要求修复Q8误差。本报告将两层数值修复与完整生成验收分开记录。','',
       '当前结论：500M采用W8为主、两处敏感MLP与head保留FP16后，完整陪伴样例执行误差3.13%，65/67最高分token一致，五次重复通过；问候误差4.48%，6/6一致。约5.4 token/s，未达到超过最优CPU的目标。1B另外修复大词表输出头归零，FP16完整问候误差0.45%，17/17一致；1B INT8尚未完成。留出质量、长历史、30分钟和CPU最优配置验收均未完成。','',
       '## 已定位问题','',
       '在当前编译配置下，将V投影权重改成逐通道INT8后，完整decoder的首token new_v输出320个元素全部为0，K输出则正常。',
       'V投影单独编译时可正常计算，说明不是INT8权重必然失真，也不能笼统说所有INT8矩阵乘不支持。',
       '只量化V即可重现两层58%误差；V保留INT16、其余六矩阵INT8时恢复到约3.49%。',
       '修复方案：其他六个矩阵保持逐通道INT8，V采用DFP INT8。七个大矩阵仍均为8位权重，激活/缓存以INT16为主，敏感算子依厂商图实现有FP16。',
       '修复后首token V不再归零，两层相对L2约3.87%，5次一致；基线INT16约3.41%。这解决了该配置下的V归零异常，但并不表示所有层、所有输入量化误差都解决。',
       '封闭编译器内部原因尚不可见，现有证据只能定位到逐通道V投影与完整图的组合路径；不能确认具体是融合、布局还是内部重标定的哪一条指令。','',
       '## 同输入消融矩阵','',
       '除指定权重量化路径外保持同一校准、图结构、128容量和128步追加KV输入。失败吞吐不计为可用模型速度。','',
       '| 变体 | 次数 | 两层ms/步 | 相对L2 | 最低余弦 | 层级门槛 |',
       '| --- | ---: | ---: | ---: | ---: | --- |']
for tag,label in variants:
    folder=results/'smol500-c128'/('smol500-c128-'+tag);path=folder/'board-result.json'
    if not path.exists():continue
    r=json.loads(path.read_text());runs=r['measurements'];dest=archive/tag;dest.mkdir(exist_ok=True)
    for name in ('board-result.json','run.stderr'):
        shutil.copyfile(folder/name,dest/name)
    lines.append(f'| {label} | {len(runs)} | {statistics.median(x["total_seconds"] for x in runs)*1000/128:.3f} | {runs[0]["relative_l2"]*100:.3f}% | {min(x["min_cosine"] for x in runs):.8f} | {"PASS" if r["exit_code"]==0 else "FAIL"} |')
lines+=['', '门槛沿用相对L2≤5%、每步余弦≥0.99；没有为修复放宽。两层通过不等于完整聊天无损。','',
        '## 单矩阵与V输出证据','']
for kind in ('q8-linear-audit','q8-v-audit'):
    p=results/kind/'linear-validation.json'
    if not p.exists():continue
    for name in ('linear-validation.json','fixture.json','run.log','deployed-sha256.txt'):
        shutil.copyfile(results/kind/name,archive/(kind+'-'+name))
    for r in json.loads(p.read_text()):
        lines.append(f'- {kind}/{r["variant"]}：相对L2 {r["relative_l2"]*100:.4f}%。独立矩阵含35个输入，其中32个来自未用于校准的真实token片段。')
trace=[]
for label,tag in [('异常','smol500-c128-w8-v-trace'),('修复','smol500-c128-w8-vdfp')]:
    path=results/'smol500-c128'/tag/'trace-layer0.f32'
    if not path.exists():continue
    a=np.fromfile(path,np.float32);v=a[1280:];k=a[960:1280]
    trace.append({'variant':label,'new_v_nonzero':int(np.count_nonzero(v)),
                  'new_v_l2':float(np.linalg.norm(v)),'new_k_l2':float(np.linalg.norm(k)),
                  'hidden':a[:960].tolist(),'new_k':k.tolist(),'new_v':v.tolist()})
    lines.append(f'- {label}首token：new_v非零 {np.count_nonzero(v)}/320，L2范数 {np.linalg.norm(v):.6f}。')
(archive/'first-token-trace.json').write_text(json.dumps(trace,indent=2,ensure_ascii=False)+'\n')
lines+=['', '所有测试都使用真实权重；单矩阵测试的RMSNorm输入由主机计算，仅用于定位，不是全模型运行方式。',
        'GGUF Q8_0本身采用分块量化，重新导出成NPU逐通道/DFP INT8并非同一种Q8表示，不能保证逐token与原CPU Q8一致。','',
        '## 全深度验收','']
source=results/'full-smol500-stable-v3-c128'
found=False
for dtype in ('w8-vdfp','w8-residual','w8-sensitive','w8-wide','w8-down-float','w8-mixed-biasfix'):
    for suffix,label in [('', '问候'),('-greeting','问候复测'),('-companion-complete','陪伴')]:
        folder=source/dtype/('board'+suffix);path=folder/'full-validation.json'
        if not path.exists():path=folder/'full-validation-w8.json'
        if not path.exists():continue
        found=True;r=json.loads(path.read_text());t=r['runtime'];dest=archive/(dtype+('-greeting' if not suffix else suffix));dest.mkdir(exist_ok=True)
        for name in ('full-validation.json','full-validation-w8.json','tokens.json','run.stderr','exit.json','prompt.json','deployed-sha256.txt'):
            if (folder/name).exists():shutil.copyfile(folder/name,dest/name)
        lines += [f'### {dtype} / {label}', '', '> '+r['npu_answer'].replace('\n','\n> '), '',
                  f'生成{t["generated_tokens"]}个token，解码{t["decode_tokens_per_second"]:.3f} token/s，首token {t["ttft_seconds"]:.3f}秒。',
                  f'FP32参考（{r.get("reference_weights","original")}权重）最高分token一致{r["argmax_matches_fp32"]}/{r["generated_token_count"]}，logits相对L2 {r["relative_l2_all_logits"]*100:.3f}%。',
                  f'加载 {t["load_seconds"]:.3f} 秒，进程 RSS {t["max_rss_kib"]/1024:.2f} MiB，驱动分配 {t["driver_allocated_bytes"]/1048576:.2f} MiB，释放后 {t["driver_after_release_bytes"]} 字节。RSS 不包含全部驱动内存，不能单报 RSS 充当模型内存。','']
        exact=folder/'full-validation-w8.json'
        if exact.exists():
            w=json.loads(exact.read_text())
            lines += [f'相同实际导出权重（含所标注FP16例外）、FP32 激活参考：logits L2 {w["relative_l2_all_logits"]*100:.3f}%，最低余弦 {w["min_step_cosine"]:.6f}，最高分一致 {w["argmax_matches_fp32"]}/{w["generated_token_count"]}。','']
            if 'execution_gate_v1' in w:lines += [f'执行门槛 v1：{"PASS" if w["execution_gate_v1"]["passed"] else "FAIL"}。这是调试轨迹，不是独立留出质量验收。','']
if not found:lines+=['尚未取得完整生成验收结果；不能以两层PASS宣称修复完成。','']
host=results/'full-smol500-w8-host-companion.json'
if host.exists():
    shutil.copyfile(host,archive/host.name)
    lines += ['### 权重与执行误差分离','',
              '独立主机 oracle 读取每层编译产物的实际 INT8 字节和尺度，使用 FP32 激活与 KV 可以生成连贯陪伴文本，原始 JSON 随证据归档。它不是 NPU 速度成绩。',
              'w8-vdfp 使用 INT16 激活和 head；w8-residual 将残差与 head 改为 FP16；w8-sensitive 进一步将 RMSNorm 内部和 masked softmax 改为 FP16，同时保留 INT16 线性层边界。',
              '层0原 masked score 的 INT16 fl=1：−10000 掩码占满动态范围，合法注意力分数步长变成 0.5。改用 FP16 可避免这项静态范围损失。未做单因素消融之前，不把组合修复的所有收益归给单一算子。','']
lines += ['### 激活截断与大动态范围修复','',
          '独立逐层 trace 显示：w8-sensitive 的第一层个别位置误差超过50%；FP32 ONNX范围审计发现 MLP Norm 输出最大21.825，但INT16范围约±2，发生截断；attention输出最大1.314，范围约±0.5。旧校准仅抽取少量位置，未覆盖这些激活。',
          'w8-wide 将非恒定INT16激活范围扩大16倍，第一层各位置误差降至约2%以内；但第4层和第32层的MLP包含上万的BOS离群值，扩大范围后down输出步长为8，普通token的小值又被舍弃。只将两处down输出改为FP16仍不足以解决MLP输入精度损失。',
          '**当前有效配置 w8-mixed-biasfix**：第4/32层MLP保留FP16，其余decoder大矩阵W8，V用DFP，敏感Norm/softmax/残差用FP16，其余INT16激活保留范围余量，head FP16。224个decoder大矩阵中218个为INT8；按参数个数约95.31%为INT8，不包含embedding/head，不能称全模型纯Q8。',
          '混合MLP第一次编译失败是沿用了整数校准的INT64 bias元数据；删除这三处浮点线性层的整数bias配置后，可由编译器按FP16路径推导并成功编译。该失败不能用来宣称SDK不支持所有混合精度。',
          '当前只在128容量的开发样例通过；静态范围余量仍需更充分校准和独立留出集验证。保留FP16敏感MLP是精度修复，尚不是速度优化。','']
for dtype in ('w8-sensitive','w8-wide'):
    folder=source/dtype/'board-companion-trace'
    for name in ('full-validation-w8.json','activation-ranges-l0.json'):
        if (folder/name).exists():shutil.copyfile(folder/name,archive/(dtype+'-trace-'+name))
for dtype in ('w8-down-float','w8-mixed','w8-mixed-biasfix'):
    for index in (3,31):
        for name in ('conversion.json',):
            path=source/dtype/f'l{index}'/name
            if path.exists():shutil.copyfile(path,archive/(dtype+f'-l{index}-'+name))
screen=results/'cpu-screen-smol500-q8.json'
if screen.exists():
    shutil.copyfile(screen,archive/screen.name)
    lines+=['## CPU线程/绑核初筛（非最终目标线）','',
            '同一SmolVLM-500M Q8，固定生成128 token，一次预热加三次记录。自然EOS被关闭，不能以这些尾部文本评价对话质量。初筛未控制统一起始温度，最高接近89°C，不能据此完成公平性能验收；新版脚本已加入每次运行≤55°C的起始温度约束和频率记录。','',
            '| 线程 | CPU亲和性 | 中位decode token/s | 记录温度范围°C |','| --- | --- | ---: | ---: |']
    for c in json.loads(screen.read_text())['candidates']:
        temperatures=[v['temperature_mC']/1000 for v in c['runs']]
        lines.append(f'| {c["threads"]} | {c["affinity"]} | {c["median_decode_tps"]:.3f} | {min(temperatures):.1f}–{max(temperatures):.1f} |')
    lines+=['','这里只覆盖现有Q8和线程/绑核，尚未覆盖其他量化、KV/attention参数及实现比较，不称“最好的CPU”。','']
repeats=source/'w8-mixed-biasfix/repeat-validation.json'
if repeats.exists():
    records=json.loads(repeats.read_text());shutil.copyfile(repeats,archive/'smol500-repeat-validation.json')
    lines+=['## 修复配置重复检查','',
            f'额外完成{len(records)}次陪伴开发样例重复；与首次运行合计{len(records)+1}次。所有已完成重复数值门槛：'+('PASS' if all(r['gate']['passed'] for r in records) else 'FAIL')+'。',
            '这些运行写完整logits用于数值校验，不是最终干净性能矩阵，也不是30分钟稳定性验收。','']
    for item in records:
        folder=source/'w8-mixed-biasfix'/('board-'+item['case']);dest=archive/item['case'];dest.mkdir(exist_ok=True)
        for name in ('full-validation-w8.json','tokens.json','exit.json','run.stderr','prompt.json','deployed-sha256.txt'):
            shutil.copyfile(folder/name,dest/name)
lines+=['## InternVL3-1B：完整输出头的独立问题','',
        '24层完整语言decoder与head已运行。未拆分的151674词输出头，仅前20602个logits非零，其余全部为0；20602 = 151674 mod 65536。证据指向大输出维度路径异常，封闭实现内部原因尚未确认。仅看前几个英文token会掩盖问题：连EOS也被置零，模型无法正常结束。',
        '修复为4个NPU输出头图，每块最多49152词，CPU仅按词表顺序拼接并argmax；全部词表矩阵计算仍在NPU。规范化重复4次，未迁移到CPU。','']
for tag,label in [('full-intern1b-stable-v3-c128','未拆分（失败）'),('full-intern1b-sharded-v2-c128','分块修复FP16')]:
    folder=results/tag/'fp16/board';path=folder/'full-validation.json'
    if not path.exists():continue
    r=json.loads(path.read_text());t=r['runtime'];dest=archive/tag;dest.mkdir(exist_ok=True)
    for name in ('full-validation.json','tokens.json','exit.json','run.stderr','prompt.json','deployed-sha256.txt'):
        shutil.copyfile(folder/name,dest/name)
    shutil.copyfile(results/tag/'trained-manifest.json',dest/'trained-manifest.json')
    lines += [f'### {label}','','> '+r['npu_answer'],'',
              f'logits相对L2 {r["relative_l2_all_logits"]*100:.3f}%，预测最高分一致{r["argmax_matches_fp32"]}/{r["generated_token_count"]}；decode {t["decode_tokens_per_second"]:.3f} token/s，首token {t["ttft_seconds"]:.3f}秒，驱动分配 {t["driver_allocated_bytes"]/1048576:.1f} MiB。','']
lines+=['1B目前是FP16完整问候样例，不是INT8迁移完成，也未完成陪伴、长历史或CPU最优配置对比。不能用500M的混合精度成绩替代1B验收。','']
lines += ['## 性能与验收边界','',
          '这些是容量128、短提示的调试运行；尚非五次固定128步、长历史或30分钟持续运行验收。旧 CPU Q8 约27.5 token/s仅为两大核旧参考，输出长度不同，且尚未完成 CPU 配置搜索。不得据此宣布 NPU 超过最优 CPU。',
          '最终目标、CPU 搜索范围、冻结的执行与质量门槛见[完整模型验收计划](完整模型验收计划.md)。500M与InternVL3-1B必须分别验收。网页仍使用 CPU。','']
measured=source/'w8-mixed-biasfix/board-companion-complete/tokens.json'
if measured.exists():
    t=json.loads(measured.read_text());decode=t['total_seconds']-t['ttft_seconds']
    api=t['decode_decoder_api_seconds']+t['decode_head_api_seconds']
    lines += [f'当前修复样例decode墙钟 {decode:.3f}秒，其中decoder VIP调用 {t["decode_decoder_api_seconds"]:.3f}秒、head VIP调用 {t["decode_head_api_seconds"]:.3f}秒，合计占 {100*api/decode:.1f}%。即使把这些调用以外的全部开销降到零，上限也只有约 {(t["generated_tokens"]-1)/api:.2f} token/s。',
              'VIP调用墙钟包含驱动提交/等待，不等于NPU算术核心利用率。此结果说明仅优化宿主RoPE、mask、日志不足以追上现有CPU，后续必须针对图/内核/搬运优化；不能用此理论上限冒充已实现速度。','']
health=results/'int8-repair-final-board-health.log'
if health.exists():shutil.copyfile(health,archive/health.name)
lines+=['## 实现与复现','',
        '`compile_weight_int8.py --value-dfp`沿new_v图依赖查找V的实际fullconnect名称，避免把旧图节点编号硬编码到所有模型。',
        '`--only-linear`提供消融，`--weight-mode dfp`提供独立量化形式对照。',
        '`run_board_probe.py --trace`额外保留第0步两个层的hidden/K/V，正常速度测试默认不启用trace。',
        '`compile_repaired_stack.py`将修正应用于32层；`--fp16-head`、`--float-residual`、`--float-sensitive`控制精度。不称作全模型所有张量均INT8。',
        '`validate_full.py --reference-weights exported-int8`区分实际W8执行误差与原GGUF权重量化差异；`run_full_model.py --trace`可保存全部位置逐层hidden，诊断时间不得冒充干净性能测试。',
        '扩全深度使用含BOS/对话特殊token的校准和稳定RMSNorm版本，不能把不同校准的全模型数字与本消融当成严格单变量对比。','']
report=here/'INT8误差定位与修复.md'
report.write_text('\n'.join(line.rstrip() for line in '\n'.join(lines).splitlines()).rstrip()+'\n')
print(report)
