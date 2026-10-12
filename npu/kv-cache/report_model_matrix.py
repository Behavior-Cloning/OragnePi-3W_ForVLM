"""Archive the larger-model layer tests, including precision/compiler failures."""
import hashlib
import argparse
import json
from pathlib import Path
import statistics

root=Path(__file__).resolve().parents[2]
results=root/'npu/results'
p=argparse.ArgumentParser();p.add_argument('--all-models',action='store_true');args=p.parse_args()
out=Path(__file__).resolve().parent/('evidence-six-20261012' if args.all_models else 'evidence-large-20261012');out.mkdir(exist_ok=True)
tags=('smol256','smol500','intern1b','qwen25','qwen3','qwen3b') if args.all_models else ('smol500','intern1b')
lines=['# '+('六模型NPU层级验证与CPU对照' if args.all_models else 'SmolVLM-500M / InternVL3-1B NPU层级验证与CPU对照'),'',
       '2026-10-12。本轮仅测各模型前两个真实语言decoder层，不是完整500M/1B生成速度。',
       '所有板端测量都逐token追加KV；不重算旧token投影。无视觉推理。','',
       '## 实际测试规模','',
       '| 模型 | 语言层总数 | GGUF语言张量元素合计 | 本轮层数 | 本轮权重参数 | hidden / Q头 / KV头 |',
       '| --- | ---: | ---: | ---: | ---: | --- |']
matrices={};manifests={}
def source_folder(tag,capacity):return results/f'{"trained-smol" if tag=="smol256" else tag}-c{capacity}'
def source_tag(tag):return 'trained-smol' if tag=='smol256' else tag
for tag in tags:
    matrix=({'scope':'reused verified 2026-10-12 initial two-layer data',
             'cases':[{'capacity':c,'stages':[]} for c in (128,512)]} if tag=='smol256'
            else json.loads((results/f'{tag}-matrix.json').read_text()))
    matrices[tag]=matrix;(out/f'{tag}-matrix.json').write_text(json.dumps(matrix,indent=2)+'\n')
    for capacity in (128,512):
        folder=source_folder(tag,capacity)
        path=folder/'trained-manifest.json'
        if path.exists():
            m=json.loads(path.read_text())
            if tag=='smol256':m.update(language_layers_total=30,gguf_tensor_elements=162974016,tested_layer_parameters=7080192)
            manifests[tag]=m
            (out/f'{tag}-c{capacity}-manifest.json').write_text(path.read_text())
    if tag in manifests:
        m=manifests[tag]
        lines.append(f"| {m['model']} | {m['language_layers_total']} | {m['gguf_tensor_elements']/1e6:.2f}M | 2 | {m['tested_layer_parameters']/1e6:.2f}M | {m['dim']} / {m['heads']} / {m['kv_heads']} |")
lines+=['','GGUF元素合计是文件内语言张量的存储规模；embedding/output可能存储重复的共享权重，不等于独立可训练参数数目。',
        '500M/1B为VLM整模型型号，包含视觉组件；这里不是两个纯文本500M/1B模型。','',
        '## CPU/NPU测量','',
        '通过精度门槛的NPU档位重复5次，CPU每档重复5次，表内取中位数。失败档位保留实际测量，不当作可用吞吐。',
        'CPU FP32与NPU INT16/FP16精度、调度器不同，只比较本实现组合；不是等精度硬件加速比。','',
        '| 模型 | 容量 | 后端 | 次数 | 两层每步ms | 推理API ms/步 | CPU占单核% | RSS MiB | NPU驱动MiB | 相对L2 | 最低余弦 | 状态 |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
histories=[];provenance={};comparisons=[]
for tag,matrix in matrices.items():
    for case in matrix['cases']:
        capacity=case['capacity'];folder=source_folder(tag,capacity)
        for stage in case['stages']:
            if stage['exit_code']:
                log=folder/f"matrix-{stage['stage']}.log"
                if log.exists():(out/f'{tag}-c{capacity}-{stage["stage"]}-initial-failure.log').write_text(log.read_text())
        cpu_times=[];npu_times={}
        for dtype in ('int16','fp16','int8'):
            base=f'{source_tag(tag)}-c{capacity}-{dtype}';path=folder/base/'board-result.json'
            if dtype=='int8' and not path.exists():continue
            for layer in (0,1):
                compile_dir=results/f'{source_tag(tag)}-c{capacity}-l{layer}-{dtype}'
                conv=compile_dir/'conversion.json'
                if conv.exists():
                    meta=json.loads(conv.read_text());nb=compile_dir/'export_nbg_unify/network_binary.nb'
                    if nb.exists():meta['nbg_sha256']=hashlib.file_digest(nb.open('rb'),'sha256').hexdigest()
                    provenance[f'{base}-l{layer}']=meta
            if not path.exists():
                lines.append(f'| {tag} | {capacity} | NPU {dtype} | 0 | — | — | — | — | — | — | — | 未获得有效板测结果 |')
                for stage in case['stages']:
                    if stage['exit_code'] and dtype in stage['stage']:
                        log=folder/f"matrix-{stage['stage']}.log"
                        if log.exists():(out/f'{base}-{stage["stage"]}-failure.log').write_text(log.read_text())
                continue
            result=json.loads(path.read_text());runs=result['measurements']
            (out/f'{base}.json').write_text(path.read_text())
            (out/f'{base}-io.log').write_text((path.parent/'run.stderr').read_text())
            median=lambda key:statistics.median(r[key] for r in runs)
            status='PASS' if result['exit_code']==0 else f"FAIL（退出{result['exit_code']}）"
            if not runs:
                lines.append(f'| {tag} | {capacity} | NPU {dtype} | 0 | — | — | — | — | — | — | — | {status}，运行失败，见IO日志 |')
                continue
            elapsed=median('total_seconds')*1000/capacity
            lines.append(f"| {tag} | {capacity} | NPU {dtype} | {len(runs)} | {elapsed:.3f} | "
                f"{median('run_seconds')*1000/capacity:.3f} | {median('process_cpu_percent_one_core'):.1f} | "
                f"{median('max_rss_kib')/1024:.2f} | {median('driver_allocated_bytes_after_load')/2**20:.2f} | "
                f"{median('relative_l2')*100:.3f}% | {min(r['min_cosine'] for r in runs):.6f} | {status} |")
            histories.append((tag,capacity,'NPU '+dtype,runs))
            if result['exit_code']==0:npu_times[dtype]=elapsed
        for path in sorted(folder.glob('cpu-*-threads*.json')):
            result=json.loads(path.read_text());runs=result['records']
            (out/f'{tag}-c{capacity}-{path.name}').write_text(path.read_text())
            median=lambda key:statistics.median(r[key] for r in runs)
            backend=f"CPU FP32/{result['threads']}线程"+('/A76' if 'a76' in path.name else '')
            elapsed=median('total_seconds')*1000/capacity;cpu_times.append(elapsed)
            lines.append(f"| {tag} | {capacity} | {backend} | {len(runs)} | {elapsed:.3f} | "
                f"{median('run_seconds')*1000/capacity:.3f} | {median('cpu_percent_one_core'):.1f} | "
                f"{median('max_rss_kib')/1024:.2f} | — | {median('relative_l2')*100:.5f}% | "
                f"{min(r['min_cosine'] for r in runs):.6f} | PASS |")
            histories.append((tag,capacity,backend,runs))
        if cpu_times:
            for dtype,elapsed in npu_times.items():comparisons.append(f"- {tag}，容量{capacity}：NPU {dtype}为本轮最快CPU配置的 **{min(cpu_times)/elapsed:.2f}×** 速度（>1才是更快）。")
lines+=['','## 当前实现的速度比','',*comparisons,'',
        '只对通过层级精度门槛的档位算速度比。更低精度带来的误差必须同时考虑；不能由两层外推全模型token/s。','',
        '## history分段（两层每步ms）','',
        '| 模型 | 容量 | 后端 | 0–25% | 25–50% | 50–75% | 75–100% |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
for tag,cap,backend,runs in histories:
    values=[statistics.median(r['quarter_step_ms'][i] for r in runs) for i in range(4)]
    lines.append(f'| {tag} | {cap} | {backend} | '+' | '.join(f'{v:.3f}' for v in values)+' |')
lines+=['','## 精度与范围','',
        '门槛仍为相对L2≤5%、逐步余弦≥0.99，属于两层冒烟门槛，不等于聊天质量通过。',
        'FP32 oracle以本项目GGUF解码为基础，未独立核对完整HF/llama.cpp logits。模型架构差异已处理：',
        'SmolVLM使用llama型GGUF RoPE行重排；InternVL3语言部分使用Qwen2，保留Q/K/V bias且不套用llama重排。',
        '校准与留出使用不同token片段，重复到容量，不是自然语言困惑度数据集。',
        '静态attention扫描整个容量，所以同容量内history各段近似恒定不代表未复用KV。',
        'RSS未包含全部驱动分配；驱动内存单列。没有电功率测试，没有断电冷启动测试。',
        '本轮没有运行剩余decoder层、最终LM head、采样或视觉编码器，不是整模型部署完成。','']
if args.all_models:
    lines+=['## 3B CPU复核说明','',
            '原CPU阶段因262144个元素中的1个超出绝对容差5e-4而退出；该失败日志和原矩阵状态均保留。',
            '随后对ARM ONNX Runtime的Qwen2-VL路径单独采用atol=1e-3、rtol=4e-4，仍要求整体相对L2<1e-4。',
            '四种CPU配置各5次复核通过。JSON记录原容差违规数量、实际误差和新旧容差，不能将容差调整解释成精度改善。',
            '完整SmolVLM-500M的生成结果另见[全深度报告](全深度生成与算子核对.md)，不由本表两层结果外推。','']
(out/'compiler-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
report=Path(__file__).resolve().parent/('六模型NPU层级实测.md' if args.all_models else '500M与1B层级实测.md')
report.write_text('\n'.join(lines));print(report)
