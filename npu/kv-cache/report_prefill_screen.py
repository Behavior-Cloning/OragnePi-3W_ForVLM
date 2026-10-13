"""Compare block-size screening without labeling it complete-model TTFT."""
import hashlib
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'results'
ARCHIVE=HERE/'evidence-attack-20261013'


def main():
    results={}
    lines=['# Prefill批次筛选','',
           '2026-10-13。两个普通decoder层，容量768、512有效历史，按冻结对齐样例的真实token由独立FP32构造输入；不是捕获NPU KV。每个图3次预热、50次执行，所有hidden/K/V输出检查数值与重复一致性。局部探针用于挑候选，不作全深度首字成绩。','',
           '| 模型 | 每批token | 两层API中位ms | 每token估算ms | 数值门槛 |',
           '| --- | ---: | ---: | ---: | --- |']
    for model,sizes in (('smol500',(16,32)),('intern1b',(16,32,64))):
        for size in sizes:
            folder=f'{model}-prefill64-20261013' if size==64 else f'{model}-prefill16-20261013'
            path=ROOT/folder/f'probe-t{size}/board-validation.json'
            if not path.exists():continue
            record=json.loads(path.read_text());t=record['timing'];ms=statistics.median(t['api_seconds'])*1000
            record.update(source_path=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            results[f'{model}-t{size}']=record
            lines.append(f'| {model} | {size} | {ms:.3f} | {ms/size:.3f} | {"通过" if record["gate"]["passed"] else "拒绝"} |')
    lines+=['','500M的16批次每token成本更高，暂不扩整栈。1B的16批次有减少prefill计算的迹象，扩展完整24层验证；64批次没有初筛优势。实际完整模型还包括敏感层、KV传递、mask、宿主开销和head，必须再次检查完整logits、五轮TTFT与总响应。','',
            '这里只估算倍数关系，未把两个16批次的局部计时当作实际连续32-token完整响应。模型、权重、模板和历史不变；CPU的首字优胜配置仍需独立对标。','']
    paths=[ROOT/'intern1b-prefill16-ab-20261013.json']
    paths+=sorted((ROOT/'prefill16-matrix-20261013').glob('*-ab.json'))
    lines+=['## 完整24层、完整输出头、五轮交替对照','',
            '两端均使用grouped head，保持相同冻结128步反馈和全部词表预测；每个版本先独立通过导出常量执行门槛，正式计时关闭大logits写出。中温起点55–<65°C，加载、历史准备与调温不计入增量首字。','',
            '| 场景 | t32/t16 首token s | 首字缩短 | t32/t16 decode token/s | t32/t16 总响应 s | t32/t16 驱动MiB |',
            '| --- | --- | --- | --- | --- | --- |']
    for path in paths:
        if not path.exists():continue
        d=json.loads(path.read_text())
        if d['status']!='complete':continue
        c,n=d['medians']['t32'],d['medians']['t16']
        name='intern1b-h512-n64' if path==paths[0] else path.stem.removesuffix('-ab')
        lines.append(f'| {name} | {c["ttft_seconds"]:.3f} / {n["ttft_seconds"]:.3f} | {100*(1-n["ttft_seconds"]/c["ttft_seconds"]):.2f}% | {c["decode_tokens_per_second"]:.3f} / {n["decode_tokens_per_second"]:.3f} | {c["total_seconds"]:.3f} / {n["total_seconds"]:.3f} | {c["driver_allocated_bytes"]/2**20:.2f} / {n["driver_allocated_bytes"]/2**20:.2f} |')
        compact={k:v for k,v in d.items() if k!='runs'}
        compact.update(source_path=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        compact['runs']=[{k:v for k,v in r.items() if k!='system_samples'} for r in d['runs']]
        results[name+'-full-ab']=compact
    lines+=['','512＋64候选的独立全模型检查：导出常量L2 4.728%、最低cosine 0.99666、125/128最高分token一致；相对原Q8开发轨迹PPL比1.01045。该开发轨迹通过不替代独立留出/自由对话验收。首字收益不能算作decode提速，也不能算超过CPU。','']
    current_path=ROOT/'prefill-current-screen-20261014/state.json'
    if current_path.exists():
        current=json.loads(current_path.read_text())
        lines+=['## 当前量化路径的8-token补充筛选','',
                f'状态：`{current["status"]}`。1B采用cap1 gate/up，500M采用峰值DFP；比较同一量化路径的8-token图与已选16/32-token图。每图独立FP32构造512有效历史，3次预热＋50次；全部hidden/K/V与重复性先验。局部耗时不表示全部层TTFT，也不能把不同批次的单位token成本直接当成实际连续prefill性能。','',
                '| 模型 / 批次 | 两层API中位 ms | 单位token估算 ms | 数值门槛 | 起止温°C |',
                '| --- | ---: | ---: | --- | --- |']
        results['current_quantization_screen']={'state':current,'measurements':{}}
        for name,row in current['cases'].items():
            if row['status']!='complete':continue
            path=Path(row['result'])/'board-validation.json'
            raw=path.read_bytes();record=json.loads(raw);t=record['timing']
            ms=statistics.median(t['api_seconds'])*1000
            size=int(name.rsplit('-t',1)[1])
            record.update(source_path=str(path),source_sha256=hashlib.sha256(raw).hexdigest())
            results['current_quantization_screen']['measurements'][name]=record
            lines.append(f'| {name} | {ms:.4f} | {ms/size:.4f} | {record["gate"]["passed"]} | {t["temperature_before_mC"]/1000:.1f}→{t["temperature_after_mC"]/1000:.1f} |')
        lines+=['','这只是开发筛选，不使用独立留出。只有出现有余量的局部优势，才考虑编译完整32/24层并重新通过全词表门槛、五轮首字与decode；没有新完整结果前，正式候选批次保持原设置。','']
    trial_path=ROOT/'smol500-prefill8-trial-20261014/state.json'
    if trial_path.exists():
        state=json.loads(trial_path.read_text())
        results['smol500-current-t8-full']={'state':state}
        source=ROOT/'full-smol500-stable-v3-c128'
        graph_root=source/'w8-fused2-c768-t8-dfpgateup'
        if (graph_root/'stack-conversion.json').exists():
            provenance=[]
            for layer in range(0,32,2):
                folder=graph_root/f'l{layer}'
                new=json.loads((folder/'conversion.json').read_text())
                old=json.loads((source/'w8-fused2-c768-t32-dfpgateup'/f'l{layer}'/'conversion.json').read_text())
                assert new['sources']==old['sources']
                assert [(r['layer'],r['selection'],r['method'],r['changes']) for r in new['fc_weight_changes']]==[(r['layer'],r['selection'],r['method'],r['changes']) for r in old['fc_weight_changes']]
                binary=folder/'export_nbg_unify/network_binary.nb'
                provenance.append({'first_layer':layer,'width':2,'query_tokens':8,'capacity':768,
                    'source_graphs':new['sources'],'weight_decisions_identical_to_t32':True,
                    'conversion_sha256':hashlib.sha256((folder/'conversion.json').read_bytes()).hexdigest(),
                    'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'binary_bytes':binary.stat().st_size})
            results['smol500-current-t8-full']['compiled_prefill_blocks']=provenance
        reference=source/'w8-fused2-c768-dfpgateup/board-prefill8-dfpgateup-smol500-h512-n64'
        for name in ('full-validation-w8.json','full-validation.json'):
            path=reference/name
            if path.exists():
                raw=path.read_bytes()
                results['smol500-current-t8-full'][name]={**json.loads(raw),
                    'source_path':str(path),'source_sha256':hashlib.sha256(raw).hexdigest()}
        lines+=['## 500M当前量化8-token完整模型验证','',
                f'状态：`{state["status"]}`。512已有历史＋64新增、完整32层及完整词表，'
                'decode和输出头复用既有图，prefill全部32层独立编译；逐层核对来源SHA及权重量化决策与t32一致。'
                '采用当前最快CPU Q8反馈，先验执行误差和原Q8开发KL/PPL，通过后中温交替五轮。','']
        if 'execution_gate' in state:
            lines.append(f'执行门槛：{state["execution_gate"]["passed"]}；原Q8开发：`{state["original_q8_development"]}`。')
        if state['status']=='complete-development-trial':
            path=Path(state['performance']);d=json.loads(path.read_text())
            assert d['status']=='complete'
            b,c=d['medians']['baseline'],d['medians']['candidate']
            assert all(sum(r['phase']=='measure' and r['label']==label for r in d['runs'])==5 for label in ('baseline','candidate'))
            lines+=['','| 指标 | t32 | t8 |','| --- | ---: | ---: |',
                    f'| 引擎首token s | {b["ttft_seconds"]:.6f} | {c["ttft_seconds"]:.6f} |',
                    f'| decode token/s | {b["decode_tokens_per_second"]:.6f} | {c["decode_tokens_per_second"]:.6f} |',
                    f'| 总响应 s | {b["total_seconds"]:.6f} | {c["total_seconds"]:.6f} |',
                    f'| 驱动 MiB | {b["driver_allocated_bytes"]/2**20:.2f} | {c["driver_allocated_bytes"]/2**20:.2f} |','',
                    f'完整首字耗时变化{c["ttft_seconds"]/b["ttft_seconds"]-1:+.2%}，'
                    '独立留出、其他历史/输入与新版稳定性仍需分别验证。','']
            results['smol500-current-t8-full']['performance']={k:v for k,v in d.items() if k!='runs'}
            results['smol500-current-t8-full']['performance'].update(source_path=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                runs=[{k:v for k,v in r.items() if k!='system_samples'} for r in d['runs']])
    ARCHIVE.mkdir(exist_ok=True)
    (ARCHIVE/'prefill-block-screen.json').write_text(json.dumps(results,indent=2)+'\n')
    (HERE/'Prefill批次筛选.md').write_text('\n'.join(lines))
    print(list(results))


if __name__=='__main__':main()
