"""Collect small, reproducible evidence for Git; never infer absent results."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/'npu/results'
OUT=Path(__file__).resolve().parent/'evidence-20261012'
OUT.mkdir(exist_ok=True)
records=[]
provenance={}
random_cases=[]
for capacity in (32,128,512):
    folder=RESULTS/f'board-c{capacity}'
    path=folder/f'c{capacity}'/'run.log' if capacity==32 else folder/'run.log'
    if path.exists():
        measurements=[json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]
        assert measurements and all(x['min_cosine']>=.99 and x['relative_l2']<=.05 for x in measurements)
        random_cases.append({'capacity':capacity,'measurements':measurements})
(OUT/'random-decoder-board.json').write_text(json.dumps(random_cases,indent=2)+'\n')
for capacity in (128,512):
    folder=RESULTS/f'trained-smol-c{capacity}'
    manifest=folder/'trained-manifest.json'
    if manifest.exists():
        (OUT/f'trained-manifest-c{capacity}.json').write_text(manifest.read_text())
    for dtype in ('int16','fp16'):
        tag=f'trained-smol-c{capacity}-{dtype}'
        source=folder/tag/'board-result.json'
        if not source.exists():continue
        result=json.loads(source.read_text())
        assert result['exit_code']==0 and len(result['measurements'])==5, tag
        (OUT/f'{tag}.json').write_text(source.read_text())
        (OUT/f'{tag}-io.log').write_text((source.parent/'run.stderr').read_text())
        row={'capacity':capacity,'backend':'NPU '+dtype.upper(),'runs':result['measurements']}
        records.append(row)
        for layer in (0,1):
            conversion=RESULTS/f'trained-smol-c{capacity}-l{layer}-{dtype}'
            path=conversion/'export_nbg_unify/network_binary.nb'
            provenance[f'{tag}-l{layer}']={'bytes':path.stat().st_size,
                'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),
                'conversion':json.loads((conversion/'conversion.json').read_text())}
    for source in sorted(folder.glob('cpu-*-threads*.json')):
        result=json.loads(source.read_text());assert len(result['records'])==5
        (OUT/f'c{capacity}-{source.name}').write_text(source.read_text())
        records.append({'capacity':capacity,'backend':f"CPU FP32 / {result['threads']}线程" +
                        (' / A76' if 'a76' in source.name else ''),'runs':result['records']})
(OUT/'compiler-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
lines=['# 两层真实权重 CPU / NPU 原始测量汇总', '',
       'SmolVLM-256M 的语言模型前两层，576维、9个Q头/3个KV头。每档5次取中位数。',
       '这是增量层级测试，不是完整模型生成token/s，也不代表聊天质量或整模型加速比。', '',
       '| KV容量 | 后端 | 每步ms | 步/秒 | 推理API ms/步 | CPU占一核% | 进程峰值RSS MiB | NPU驱动分配MiB | 相对L2误差 |',
       '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for row in records:
    runs=row['runs'];cap=row['capacity'];median=lambda f:statistics.median(f(r) for r in runs)
    cpu=lambda r:r.get('process_cpu_percent_one_core',r.get('cpu_percent_one_core'))
    memory='—'
    if 'driver_allocated_bytes_after_load' in runs[0]:
        memory=f"{median(lambda r:r['driver_allocated_bytes_after_load']/2**20):.2f}"
        assert all(r['driver_allocated_bytes_after_release']==0 for r in runs)
    lines.append(f"| {cap} | {row['backend']} | {median(lambda r:r['total_seconds']*1000/cap):.3f} | "
                 f"{median(lambda r:r['two_layer_steps_per_second']):.1f} | {median(lambda r:r['run_seconds']*1000/cap):.3f} | "
                 f"{median(cpu):.1f} | {median(lambda r:r['max_rss_kib']/1024):.2f} | {memory} | "
                 f"{median(lambda r:r['relative_l2'])*100:.4f}% |")
lines+=['','## 各 history 段每步耗时（毫秒）','',
        '每段覆盖窗口的四分之一，属于连续累积KV的同一次运行；不是四个独立冷启动。','',
        '| 容量 | 后端 | 0–25% | 25–50% | 50–75% | 75–100% |',
        '| ---: | --- | ---: | ---: | ---: | ---: |']
for row in records:
    values=[statistics.median(r['quarter_step_ms'][i] for r in row['runs']) for i in range(4)]
    lines.append(f"| {row['capacity']} | {row['backend']} | "+' | '.join(f'{v:.3f}' for v in values)+' |')
lines+=['','CPU使用ORT FP32和Python调度；NPU使用VIPLite C调度及不同精度，API计时单列。',
        'RSS不包含全部NPU驱动分配，不可将NPU的几MiB RSS当作总占用。驱动列为测试网络加载后的分配值，未必覆盖所有固件/内核开销。',
        'NPU加载时间属于进程/网络加载，文件可被页缓存命中；本次没有测断电冷启动。',
        '测试保留原网页服务空闲运行。未测电功率，没有能耗/焦耳每token结论。','']
(OUT/'测量汇总.md').write_text('\n'.join(lines))
print('\n'.join(lines))
