"""Archive compact evidence and report full-depth generation separately from probes."""
import hashlib
import json
from pathlib import Path
import shutil

root=Path(__file__).resolve().parents[2];results=root/'npu/results';here=Path(__file__).resolve().parent
archive=here/'evidence-full-20261012';archive.mkdir(exist_ok=True)
lines=['# 全深度生成、数值算子与INT8适配','',
       '2026-10-12。模型为SmolVLM-500M-Instruct-Q8_0的完整语言部分：32层decoder及最终RMSNorm/LM head。',
       '32层和输出头均作为NBG图由VIPLite执行；CPU负责embedding查表、RoPE/mask、KV管理、argmax和文本解码。',
       '从GGUF Q8_0反量化后编译为NPU FP16，不是原生GGUF Q8执行。当前没有视觉编码器，没有接入网页。','',
       '## 能否正常生成','',
       '以下直接列出实际生成文本。每一步均把预测token回送下一步，全部decoder持续复用KV。',
       '上下文容量128；陪伴长样例最多生成96个token，达到上限属于人为截断，不代表自然EOS。',
       '只覆盖下列提示词，不能据此声称多轮角色稳定、中文或陪伴质量已通过系统评测。',
       '注意两种CPU参考不同：FP32参考反量化同一GGUF权重，用于检查图算术；独立llama.cpp CPU Q8用于实际产品速度与文本对照。与前者token一致不等于与后者完全相同。','']
cases=[]
for version,folder_name in [('v1','full-smol500-stable-c128'),('v3','full-smol500-stable-v3-c128')]:
    source=results/folder_name
    for name in ('trained-manifest.json','prompt.json'):
        if (source/name).exists():shutil.copyfile(source/name,archive/f'{version}-{name}')
    for suffix,label in [('', '问候'),('-companion','陪伴48'),('-companion-complete','陪伴96')]:
        board=source/'fp16'/('board'+suffix)
        path=board/'full-validation.json'
        if not path.exists():continue
        r=json.loads(path.read_text());key=version+('-greeting' if not suffix else suffix)
        dest=archive/key;dest.mkdir(exist_ok=True)
        for filename in ('full-validation.json','tokens.json','prompt.json','exit.json','run.stderr','deployed-sha256.txt'):
            if (board/filename).exists():shutil.copyfile(board/filename,dest/filename)
        runtime=r['runtime'];cases.append((version,label,r,runtime))
        lines += [f'### {version} · {label}', '', f'输入：{r["prompt"]}', '',
                  'NPU实际输出：', '', '> '+r['npu_answer'], '',
                  'CPU llama.cpp Q8输出：', '', '> '+r.get('llama_cpu_answer','未取得此提示词的独立CPU结果'), '',
                  f'与反量化权重FP32参考的逐步最高分token一致数：{r["argmax_matches_fp32"]}/{r["generated_token_count"]}。', '']
lines+=['## 实测开销','',
        '| 版本/样例 | 输入/输出token | 加载秒 | 首token秒 | decode token/s | CPU Q8 token/s | 进程CPU/单核 | RSS MiB | 驱动MiB |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for version,label,r,t in cases:
    cpu=r.get('llama_cpu_reference',{}).get('response',{}).get('timings',{}).get('predicted_per_second',0)
    lines.append(f'| {version}/{label} | {t["prompt_tokens"]}/{t["generated_tokens"]} | {t["load_seconds"]:.3f} | {t["ttft_seconds"]:.3f} | {t["decode_tokens_per_second"]:.3f} | {cpu:.3f} | {t["cpu_percent_one_core"]:.2f}% | {t["max_rss_kib"]/1024:.2f} | {t["driver_allocated_bytes"]/2**20:.2f} |')
lines+=['', '每行是一次实际运行，不是重复测试均值。96输出的CPU对照复用同提示词、48输出上限的记录，CPU已于39个token内自然EOS；两边实际生成长度不同。',
        '加载为进程/模型加载，受文件缓存影响；不是断电冷启动。首token不含模型加载。NPU目前逐token吸收新提示词，没有批量prefill优化。',
        'RSS不包含全部驱动映射，不能把约数MiB RSS误读为整个模型只占这些内存。驱动分配约720MiB，各次释放后均为0。',
        '板端温度、进程CPU、各token延迟在证据JSON中；没有电功率测量。','',
        '## FP32参考误差补充','',
        '| 版本/样例 | 完整logits相对L2 | 最低逐步余弦 | 最高分token一致 |',
        '| --- | ---: | ---: | ---: |']
for version,label,r,t in cases:
    lines.append(f'| {version}/{label} | {r["relative_l2_all_logits"]*100:.4f}% | {r["min_step_cosine"]:.8f} | {r["argmax_matches_fp32"]}/{r["generated_token_count"]} |')
lines+=['',
        '## 为什么没有加速','',
        '之前较快的是INT16两层与ONNX CPU FP32的对比；当前完整模型比较的是NPU FP16与llama.cpp CPU Q8。两者不是同一精度或软件后端。',
        '500M两层FP16此前约23.923ms，乘16约382.8ms，再加LM head便接近实际完整token延迟；两层INT16约6.781ms不能用于外推FP16。','']
for version,label,r,t in cases:
    if 'decode_decoder_api_seconds' in t and t['generated_tokens']>1:
        n=t['generated_tokens']-1
        dec=t['decode_decoder_api_seconds']/n*1000;head=t['decode_head_api_seconds']/n*1000
        total=1000/t['decode_tokens_per_second']
        lines.append(f'- {version}/{label}：32层VIP调用{dec:.2f}ms/token，输出头{head:.2f}ms/token，其余约{total-dec-head:.2f}ms/token。')
lines+=['', '这些是VIP API墙钟耗时，不是硬件逐指令分析。batch=1矩阵向量运算、FP16执行效率、静态完整容量attention、每层调度与缓冲刷新都有潜在影响；现有数据不能给每个因素分摊占比。',
        '尚未实现层融合、批量prefill、动态KV容量和可用的全模型INT8；不能把标称TOPS直接换算为本模型token/s。','',
        '## 激活与数值安全核对','',
        'CPU oracle使用PyTorch FP32标准softmax/SDPA、SiLU和RMSNorm；NPU采用FP16，不会逐位一致。',
        'ONNX Softmax编译为VSI_NN_OP_SOFTMAX，axis对应key维；SiLU融合为VSI_NN_OP_SWISH、beta=1；RMSNorm展开为reduce/divide/sqrt等图算子。',
        '厂商softmax内部实现未公开。本次额外用同形状15头×129位置的板测检查±1000/10000偏移、±60000极值、mask、零输入，并对照显式减max版本；只能证明测试覆盖范围，不能声称查到了内核源码。','']
audit=results/'activation-audit-fp16'
if (audit/'activation-validation.json').exists():
    records=json.loads((audit/'activation-validation.json').read_text())
    lines+=['| 算子 | 用例数 | 全部有限 | 最大绝对误差 | 整体相对L2 | 严格压力门槛 |',
            '| --- | ---: | --- | ---: | ---: | --- |']
    for r in records:
        lines.append(f'| {r["name"]} | {r["cases"]} | {r["finite"]} | {r["max_abs_error"]:.6g} | {r["relative_l2"]:.6g} | {"PASS" if r["passed"] else "FAIL"} |')
    for name in ('activation-manifest.json','activation-validation.json','activation-validation-original.json','activation-validation-v2.json','run.log','deployed-sha256.txt'):
        if (audit/name).exists():shutil.copyfile(audit/name,archive/('activation-'+name))
lines+=['', '原始未缩放RMSNorm在首token第5层出现-inf，不能生成。v1通过max(|x|,1)缩放避免大值平方溢出，但小值压力测试发现eps及中间平方下溢。',
        'v2仅降低缩放下限仍失败；v3同时使用d=max(max|x|,sqrt(eps))、y=x/d、归一化分母sqrt(mean(y²)+(sqrt(eps)/d)²)，避免直接用subnormal eps作除法分子。',
        'v3把约1e-3输入的误差显著降低，但FP16小于约6.1e-5的subnormal输入在此路径仍表现为归零；约1e-6输入的相对误差可达100%。严格压力检查仍FAIL，未通过放宽门槛隐藏。',
        '因此是“完整生成样例可用、极小输入仍存在数值边界”，不是所有输入等价CPU或全面精度通过。','',
        '## Q8/INT8尝试','',
        '| 路径 | 范围 | 结果 |', '| --- | --- | --- |',
        '| 原GGUF Q8_0 | 完整CPU | 网页原路径可用 |',
        '| NPU普通INT8 DFP | 前2层 | 约3.63ms/步，但相对L2约142.9%，失败 |',
        '| INT8逐通道权重＋FP16激活 | 1层编译 | SDK VerifyGraph拒绝当前配置，未生成NBG |',
        '| INT8逐通道权重＋INT16激活 | 前2层板测 | 约4.57ms/步，但相对L2约58.45%、最低余弦0.7518，失败 |',
        '| 相同校准INT16对照 | 前2层板测 | 相对L2约3.41%、最低余弦0.9974，通过层级门槛 |','',
        '检查编译产物中的INT8权重字节、逐通道scale及输出通道数后，在主机用FP32激活重算，两层相对L2约1.12%、最低余弦0.99989。权重布局核对通过，不足以解释板上的58%误差；还需定位混合精度算子/中间量化误差。',
        '这些失败仅针对当前实现与SDK配置，不证明芯片绝不支持INT8。没有可交付的全深度NPU Q8对话结果，失败吞吐不计为可用加速。','',
        '## 复现与后续','',
        '- `trained_probe.py --layers 0 --stable-norm --norm-version 3`导出所有层、LM head和KV输入输出；`compile_stack.py`逐图编译。',
        '- `run_full_model.py`部署、核对SHA256、使用共享评测锁执行；`--case`保留独立结果，`--skip-deploy`复用模型但更新提示词/运行器。',
        '- `full_cpu_reference.py`使用独立18083端口，退出时关闭临时CPU服务；`validate_full.py`核对实际token轨迹。',
        '- `activation_probe.py`/`run_activation_probe.py`执行独立算子压力检查，失败如实保留。',
        '- `compile_weight_int8.py`/`audit_int8_weights.py`保留Q8探索与权重布局证据。',
        '- 六模型均有前两层尝试，见[六模型层级实测](六模型NPU层级实测.md)；只有本文500M语言部分做了全深度生成。',
        '- 下一步应定位INT8混合路径的逐算子误差，再考虑全深度低比特；目前生产网页继续使用CPU。','']
for tag in ('smol500-w8f16-l0','smol500-w8a16-l0','smol500-w8a16-l1'):
    for name in ('conversion.json','export.log'):
        path=results/tag/name
        if path.exists():shutil.copyfile(path,archive/(tag+'-'+name))
for tag in ('smol500-c128-w8a16','smol500-c128-int16-matched-calibration','smol500-c128-int8'):
    for name in ('board-result.json','run.stderr'):
        path=results/'smol500-c128'/tag/name
        if path.exists():shutil.copyfile(path,archive/(tag+'-'+name))
if (results/'smol500-w8a16-weight-audit.json').exists():
    shutil.copyfile(results/'smol500-w8a16-weight-audit.json',archive/'smol500-w8a16-weight-audit.json')
for name in ('exit.json','run.stderr'):
    path=results/'full-smol500-c128/fp16/board'/name
    if path.exists():shutil.copyfile(path,archive/('original-fp16-overflow-'+name))
report=here/'全深度生成与算子核对.md';report.write_text('\n'.join(lines));print(report)
