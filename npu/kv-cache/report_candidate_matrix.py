"""Current DFP candidate coverage, with exact length and CPU trajectory checks."""
import hashlib
import json
from pathlib import Path
import statistics
import struct

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'results'


def read(path):return json.loads(path.read_text())


def main():
    candidates={}
    def collect(path,label):
        if not path.exists():return
        d=read(path)
        if d.get('status')!='complete':return
        v=next(v for v in d['config']['variants'] if v['label']==label)
        # Exclude the older product-range repair baseline and all failed MLP/head trials.
        if not any(s in v['remote'] for s in ('dfpgateup','dfpmse')):return
        if any(s in v['remote'] for s in ('headmse','dfpmlp')):return
        runs=[r for r in d['runs'] if r['phase']=='measure' and r['label']==label]
        assert len(runs)==5
        runtime=runs[0]['result'];model='smol500' if runtime['decoder_layers']==32 else 'intern1b'
        depth=32 if model=='smol500' else 24
        h,n=runtime['cached_prefix_tokens'],runtime['new_prompt_tokens']
        assert all(r['result']['decoder_layers']==depth and r['result']['head_on_npu'] and
                   r['result']['generated_tokens']==128 and r['result']['fixed_feedback'] and
                   r['result']['cached_prefix_tokens']==h and r['result']['new_prompt_tokens']==n and
                   r['result']['feedback_tokens']==runtime['feedback_tokens'] and
                   r['result']['driver_after_release_bytes']==0 for r in runs)
        bounds=d['config']['temperature_band_mC'];band='low' if bounds==[40000,50000] else 'medium'
        assert bounds==([40000,50000] if band=='low' else [55000,64999])
        assert all(bounds[0]<=r['result']['response_start_temperature_mC']<=bounds[1] for r in runs)
        assert v['validation']['passed']
        key=(model,h,n,band)
        candidates[key]={'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                         'variant':v,'medians':d['medians'][label],'runtime':runtime,
                         'config':d['config'],'runs':[{k:v for k,v in r.items() if k!='system_samples'} for r in runs]}
    for path,label in [('smol500-dfpgateup-ab-20261013.json','dfp'),
                       ('dfpmse-prefill16-trial-20261013/performance.json','candidate'),
                       ('smol500-c1280-dfp-trial-20261013-attempt2/performance.json','candidate'),
                       ('dfpmse-productwide-trial-20261013-attempt2/performance.json','candidate')]:
        collect(ROOT/path,label)
    for folder in ('candidate-expansion-20261013','candidate-long-expansion-20261013','candidate-short-low-20261013'):
        state=ROOT/folder/'state.json'
        if state.exists():
            for row in read(state)['cases'].values():
                for band in row.get('bands',{}).values():
                    collect(Path(band['result']),'candidate')
    search=read(ROOT/'cpu-q8-search-20261013/state.json')
    rows=[]
    lines=['# 当前DFP候选：精确长度低中温矩阵','',
           '500M gate/up峰值DFP，1B gate/up MSE DFP＋prefill16，1024历史另有已验证product范围修复。全部32/24层、完整词表、固定128预测、预热＋五轮。此表不混入旧NPU基线、质量失败的MLP候选或尚未验证的新输出头。','',
           f'当前候选五轮完成{len(candidates)}/24格；“待测”不是0速度。独立留出仍单独验收，不能以开发轨迹通过自动推广产品。','',
           '| 模型 / 历史＋新增 / 起温 | NPU token/s | NPU首token s | 同温CPU token/s | CPU首token s | 对齐/基线状态 |',
           '| --- | ---: | ---: | ---: | ---: | --- |']
    for model in ('smol500','intern1b'):
        for h in (512,1024):
            s=search['cases'].get(f'{model}-h{h}-n64',{})
            for n in (32,64,128):
                case=f'{model}-h{h}-n{n}'
                for band in ('low','medium'):
                    key=(model,h,n,band)
                    c=candidates.get(key)
                    display=f'{model} / {h}＋{n} / {band}'
                    if not c:
                        lines.append(f'| {display} | 待测 | — | — | — | — |');continue
                    cpu_path=ROOT/'thermal-aligned-20261013'/(case+'-'+band+'-cpu')/'results.json'
                    note='原矩阵Q8配置，尚非本格最优确认'
                    if band=='medium':
                        prefix='' if n==64 else f'n{n}-'
                        finals={k:v for k,v in s.get('runs',{}).items() if k.startswith(prefix+'final-') and
                                v.get('repeats')==5 and v.get('medians')}
                        if finals:
                            winner=max(finals,key=lambda k:finals[k]['medians']['decode_tokens_per_second'])
                            cpu_path=Path(finals[winner]['path'])
                            note=winner+'；配置搜索'+s.get('status','pending')
                    cpu=read(cpu_path)
                    assert cpu['status']=='complete' and cpu['cached_prefix']==h and len(cpu['prompt']['token_ids'])==h+n
                    packed=struct.pack('<'+'i'*len(cpu['prompt']['token_ids']),*cpu['prompt']['token_ids'])
                    assert hashlib.sha256(packed).hexdigest()==c['config']['prompt_sha256']
                    aligned=cpu['nonstream_trajectory']['tokens']==c['runtime']['feedback_tokens']
                    if not aligned:note+='；反馈不同，待重新对齐'
                    else:note+='；反馈一致'
                    measured=[r for r in cpu['runs'] if r['phase']=='measure'];assert len(measured)==5
                    ttft=statistics.median(r['timings']['prompt_ms']/1000 for r in measured)
                    speed=cpu['medians']['decode_tokens_per_second']
                    m=c['medians']
                    lines.append(f'| {display} | {m["decode_tokens_per_second"]:.3f} | {m["ttft_seconds"]:.3f} | {speed:.3f} | {ttft:.3f} | {note} |')
                    rows.append({'model':model,'history':h,'new_tokens':n,'band':band,'npu':c,
                                 'cpu':{'source':str(cpu_path),'source_sha256':hashlib.sha256(cpu_path.read_bytes()).hexdigest(),
                                        'decode_tokens_per_second':speed,'engine_ttft_seconds':ttft},
                                 'trajectory_aligned':aligned,'cpu_note':note})
    lines+=['','低温起点40–50°C，中温55–<65°C，响应期间自然变化。两端首token均使用引擎口径，不含加载、历史KV准备、调温和HTTP缓冲。低温最快CPU配置仍需同温复核，不能拿中温CPU直接替换低温成绩。','',
            '先筛64新增，再把入围CPU配置扩32/128；不等于穷举所有实现。若更快CPU改变贪心轨迹，保留更高目标线并要求NPU重新回灌和完整logits校验，不回退到慢CPU宣布胜出。','',
            '[独立质量与持续运行](独立质量与连续多轮验收.md) · [真实ASR并发](真实ASR与LLM并发验收.md) · [配对A/B收益与负载](全深度候选提速与负载.md)','',
            '重建：`npu/.venv/bin/python npu/kv-cache/report_candidate_matrix.py`。','']
    (HERE/'当前DFP候选精确矩阵.md').write_text('\n'.join(lines))
    (HERE/'evidence-attack-20261013/candidate-matrix.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
    print('current DFP candidate cells',len(rows))


if __name__=='__main__':main()
