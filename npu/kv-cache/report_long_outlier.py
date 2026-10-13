"""Archive the reproducible long-history outlier and its bounded repair."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'results'
SOURCE=ROOT/'full-intern1b-sharded-v2-c128'


def record(path):
    d=json.loads(path.read_text())
    return {'source_path':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            **{k:v for k,v in d.items() if k not in ('constant_records','runtime','npu_tokens','fp32_greedy_tokens_on_npu_history')}}


def main():
    trace=SOURCE/'w8-fused2-c1280-splitkv-i16mm/board-blocktrace-intern1b-h1024-n32'
    original=SOURCE/'w8-fused2-c1280-splitkv-i16mm/board-thermal3-intern1b-h1024-n32'
    captured=SOURCE/'w8-fused2-c1280-splitkv-i16mm/board-capture1141b9-intern1b-h1024-n32'
    hashes=[hashlib.sha256((p/'logits.f32').read_bytes()).hexdigest() for p in (original,trace,captured)]
    assert len(set(hashes))==1
    report={'scope':'all 24 layers, complete vocabulary, frozen CPU Q8 continuation; failed result excluded from accepted speed',
            'logits_sha256':hashes[0],'identical_diagnostic_runs':list(map(str,(original,trace,captured))),
            'validation':record(trace/'full-validation-w8.json'),
            'ranges':record(ROOT/'intern1b-range-1141b9-20261013/ranges.json')}
    lines=['# 1B长历史误差定位与定点范围修复','',
           '场景：1024历史＋32新增，全部24层、完整输出头、128次预测、同一CPU Q8反馈。原版、逐块hidden抓取版、第1141位置第9块输入抓取版的完整logits逐字节相同，说明此误差可稳定复现。诊断计时不进入速度排行。','',
           '原版总logits L2约4.76%，但第86/87步出现离群，最低cosine约0.9762，未通过既定0.99门槛。平均误差合格不能覆盖最差步失败。','',
           '| 融合块最后一层（从0起） | 第86步hidden相对L2 | 第87步hidden相对L2 |',
           '| ---: | ---: | ---: |']
    for b in report['validation']['decode_block_errors']:
        lines.append(f'| {b["last_layer"]} | {b["step_relative_l2"][85]:.3%} | {b["step_relative_l2"][86]:.3%} |')
    lines+=['','真实输入的独立FP32图诊断发现：第19层（第20层）MLP乘积最大23.0946，而INT16/FL11只覆盖−16～15.9995。4864个元素中1个被截断，这个张量的截断L2约22.43%；随后传入down矩阵，融合块hidden实际L2约10.45%。相同块K/V输出误差均低于0.14%，异常更集中于MLP分支。','',
            '修复候选只把第19层MLP乘积及到down输入的布局节点扩大1bit范围（FL11→10），保留W8权重、外部KV、其他层、head和完整计算深度。decode与prefill两图都明确记录范围修改。FP32范围审计本身不能证明实板修复，仍须完整logits重验与同温档五轮计时。','']
    repaired=SOURCE/'w8-fused2-c1280-product19wide/board-product19wide-intern1b-h1024-n32'
    for filename in ('full-validation-w8.json','full-validation.json'):
        path=repaired/filename
        if path.exists():report[filename]=record(path)
    if 'full-validation-w8.json' in report:
        d=report['full-validation-w8.json']
        lines += [f'候选全模型执行门槛：{"通过" if d["execution_gate_v1"]["passed"] else "未通过"}；logits L2 {d["relative_l2_all_logits"]:.3%}，最低cosine {d["min_step_cosine"]:.6f}，最高分token一致率 {d["argmax_match_fraction"]:.3%}。','']
    else:lines+=['候选全模型重验正在进行，尚不能称已修复。','']
    lines+=['该场景属于开发定位，不能再当作独立留出质量验收。其他历史、新增长度及自由生成须另验；不放宽精度门槛。','',
            '[紧凑证据](evidence-attack-20261013/long-outlier-repair.json)。重建：`npu/.venv/bin/python npu/kv-cache/report_long_outlier.py`。','']
    archive=HERE/'evidence-attack-20261013';archive.mkdir(exist_ok=True)
    (archive/'long-outlier-repair.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    (HERE/'1B长历史离群误差与修复.md').write_text('\n'.join(lines))
    print('report written; repair gate',report.get('full-validation-w8.json',{}).get('execution_gate_v1'))


if __name__=='__main__':main()
