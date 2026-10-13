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
    ARCHIVE.mkdir(exist_ok=True)
    (ARCHIVE/'prefill-block-screen.json').write_text(json.dumps(results,indent=2)+'\n')
    (HERE/'Prefill批次筛选.md').write_text('\n'.join(lines))
    print(list(results))


if __name__=='__main__':main()
