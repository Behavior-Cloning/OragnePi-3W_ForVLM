"""Report repeated frozen-suite quality for the complete 500M prefill8 variant."""
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'results'


def read(path):
    raw=path.read_bytes()
    return {**json.loads(raw),'source_path':str(path),'source_sha256':hashlib.sha256(raw).hexdigest()}


def main():
    reference=read(ROOT/'holdout-reference-prefill8-smol500-20261014/state.json')
    path=ROOT/'holdout-npu-prefill8-smol500-20261014/result.json'
    actual=read(path) if path.exists() else {'status':'waiting'}
    holdout=actual.get('holdout',{})
    rows=holdout.get('cases',[]);natural=holdout.get('natural',[])
    assert len({r['case'] for r in rows})==len(rows)
    expected={r['case']:r for r in reference['cases']}
    assert len(expected)==36 and {r['case'] for r in rows}<=set(expected)
    if holdout:assert holdout['holdout_sha256']==reference['holdout_sha256']
    total=sum(r['token_count'] for r in rows)
    kl=sum(r['mean_kl_nats']*r['token_count'] for r in rows)/total if total else None
    ppl=math.exp(sum(math.log(r['perplexity_ratio'])*r['token_count'] for r in rows)/total) if total else None
    passed=sum(r['numeric_passed'] for r in rows)
    complete=holdout.get('status')=='complete-numeric-and-free-generation'
    if complete:
        assert len(rows)==36
        assert math.isclose(kl,holdout['mean_kl_nats'],rel_tol=1e-5,abs_tol=1e-7)
        assert math.isclose(ppl,holdout['perplexity_ratio'],rel_tol=1e-5)
        assert holdout['closed']['driver_bytes']==holdout['free_closed']['driver_bytes']==0
    lines=['# 500M prefill8：冻结陪伴套件回归','',
           '完整32层、完整词表、容量768、8-token prefill。沿用冻结v1的12个中英场景、每场景3轮；'
           '本次是重复套件回归，不称作新独立留出。候选按开发轨迹和速度选出，没有根据v1回答调整权重、提示词或阈值。','',
           '工作站标准Transformers原Q8反量化FP32参考复用已完成的36条轨迹，逐个校验logits SHA256、'
           'GGUF模型SHA及题目不变；参考不来自NPU预测。固定反馈数值与NPU自己接续历史的自由对话分开。'
           '本次完整候选与早期容量1280的v1结果不是只改批次的配对质量消融，不能将两者差异全归因于t8。','',
           f'执行状态：`{actual["status"]}`；数值{len(rows)}/36，自由回答{len(natural)}/36。'
           '执行结束不等于质量或产品通过。','',
           '| 数值指标 | 当前结果 |','| --- | ---: |',
           f'| 单例通过 / 已测 | {passed}/{len(rows)} |',
           f'| 聚合KL | {kl if kl is not None else "待测"} |',
           f'| 聚合PPL比 | {ppl if ppl is not None else "待测"} |','',
           '每例维持KL≤0.05、PPL比≤1.05；聚合通过不能覆盖单例失败。'
           'HTTP和分词在工作站，语言推理与常驻KV在板端；本结果不是板端独立网站交付，也不是CPU速度对照。','',
           '| 未过案例 | KL | PPL比 |','| --- | ---: | ---: |']
    for row in rows:
        if not row['numeric_passed']:
            lines.append(f'| {row["case"]} | {row["mean_kl_nats"]:.6f} | {row["perplexity_ratio"]:.6f} |')
    if complete and passed<36:
        lines+=['','当前候选未过全套质量，不推广生产，也不把旧版本30分钟稳定性当作该版本验收。']
    cpu=read(ROOT/'holdout-cpu-smol500-20261013/state.json')
    assert cpu['holdout_sha256']==reference['holdout_sha256']
    if natural:
        nmap={r['case']:r for r in natural};cmap={r['case']:r for r in cpu['cases']}
        lines+=['','## 对话抽查：与实际CPU区分','',
                '同一套用户问题，两端后续轮接续各自回答。这是质量对照和原文阅读，不是逐token误差归因或人工盲评。', '',
                '| 案例 | 板端CPU Q8 | 新NPU候选 |','| --- | --- | --- |']
        for case in ('en-listen-turn1','en-correction-turn3','zh-listen-turn1','zh-role-turn3','zh-memory-turn1'):
            clean=lambda text:text.replace('|','\\|').replace('\n','<br>')
            lines.append(f'| {case} | {clean(cmap[case]["answer"])} | {clean(nmap[case]["answer"])} |')
        lines+=['','英语角色Pip、猫名Bean等简单记忆可以保持；改称呼Robin→Sam后仍回Robin，实际CPU也有同样错误。'
                '中文倾听场景NPU连续三轮重复同一句，与题意不符；中文角色回忆CPU答出小豆，NPU反问复述。'
                '基模能力不足与NPU附加退化都需记录，不能全部归给任一方。']
    if natural:
        lines+=['','## 自由回答原文','',
                '这里是自然生成，EOS或96-token上限正常结束，不拼接固定轨迹预测当对话。']
        for row in natural:
            name=row['case'];question=expected[name]['messages'][-1]['content']
            lines+=['',f'### {name}','',f'用户：{question}','',f'NPU：{row["answer"] or "（空回答）"}','']
    lines+=['','复现：`report_prefill8_quality.py`；重用标准参考：`reuse_holdout_reference.py`。','']
    (HERE/'500M-prefill8陪伴回归.md').write_text('\n'.join(lines))
    compact={k:v for k,v in actual.items() if k!='runs'}
    if 'holdout' in compact:
        compact['holdout']={**holdout,'cases':[{k:v for k,v in r.items() if k!='runtime'} for r in rows]}
    evidence={'suite_sha256':reference['holdout_sha256'],'reused_reference':reference['reused_standard_reference'],
              'model_sha256':reference['model_sha256'],'reference_state_sha256':reference['source_sha256'],
              'actual':compact,'numeric_cases':len(rows),'passed_cases':passed,
              'natural_cases':len(natural),'mean_kl_nats':kl,'perplexity_ratio':ppl,'product_accepted':False,
              'cpu_reference':{'source_path':cpu['source_path'],'source_sha256':cpu['source_sha256'],
                  'cases':[{k:v for k,v in r.items() if k in ('case','messages','answer','checks')} for r in cpu['cases']]}}
    (HERE/'evidence-attack-20261013/prefill8-quality.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+'\n')
    print({k:evidence[k] for k in ('numeric_cases','passed_cases','natural_cases','mean_kl_nats','perplexity_ratio')})


if __name__=='__main__':main()
