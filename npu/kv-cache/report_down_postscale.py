"""Archive the scale-factorization screen without promoting a local probe."""
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent/'results'


def read(path):
    raw = path.read_bytes()
    return {**json.loads(raw), 'source_path': str(path),
            'source_sha256': hashlib.sha256(raw).hexdigest()}


def conversion_summary(path):
    record=read(path)
    fields=('source_path','source_sha256','exit_code','seconds','scope','first_layer',
            'fusion_width','capacity','query_tokens','sources','baseline_graph','binaries',
            'postscale_changes','postscale_intermediate','exported_integer_bytes_verified',
            'contract_verified_from_exported_c','input_order','output_order',
            'fc_weight_dfp','fc_weight_dfp_method','fc_weight_dfp_max_extra_bits')
    # Full source quantizer arrays remain in the immutable local compiler record.
    return {k:record[k] for k in fields if k in record}


def main():
    evidence = {'scope': 'two ordinary decoder layers; unchanged exported baseline reference',
                'full_model_accepted': False, 'models': {}}
    lines = ['# Down投影逐通道缩放拆分试验', '',
             '针对普通MLP down矩阵：读取既有编译产物的INT8整数q与逐输出通道scale s，'
             '将矩阵写成DFP q/128，矩阵输出再逐通道乘128s。数学上的有效权重不变；'
             '本机FP32重放与权重逐元素断言均通过。两个模型此处的128s可被FP16精确表示。'
             '新增中间INT16边界仍可能产生舍入或截断，必须由实板全部输出检查决定。', '',
             '编译后另核对导出的INT8整数逐字节未变、小数位仍为7。'
             '既有KV、Norm、gate/up和残差格式保持。参考是未改写基线的导出常量FP32，'
             '不是新图自己的导出常量，也不是独立对话或全模型精度。', '',
             '| 模型 | 原图/改写图两层API中位 ms | 耗时变化 | 全部输出门槛 | 最大输出L2 | 驱动MiB 原/新 |',
             '| --- | ---: | ---: | --- | ---: | ---: |']
    for model, baseline in [('intern1b', 'gateup-w8dfp-mse-cap1'), ('smol500', 'gateup-w8dfp')]:
        root = ROOT/'down-postscale-20261014'
        compiled = conversion_summary(root/(model+'-i16-attempt2')/'conversion.json')
        base = read(ROOT/(model+'-decode-profile-20261013')/('probe-'+baseline)/'board-validation.json')
        row = {'conversion': compiled, 'baseline': base, 'attempts': {}}
        binary = root/(model+'-i16-attempt2')/'export_nbg_unify/network_binary.nb'
        row['verified_candidate_binary'] = {'path': str(binary), 'bytes': binary.stat().st_size,
                                           'sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}
        # Early records inherited the baseline binaries list. Keep the source
        # JSON immutable and explicitly mark that list as non-authoritative.
        if not compiled['binaries'][0].get('sha256'):
            row['inherited_binaries_metadata_is_authoritative'] = False
        first = root/(model+'-i16')/'conversion.json'
        if first.exists():
            row['attempts']['invalid_node_id'] = conversion_summary(first)
        path = root/('probe-'+model+'-i16')/'board-validation.json'
        if path.exists():
            actual = read(path);row['actual'] = actual
            assert compiled['exit_code'] == 0 and compiled['exported_integer_bytes_verified']
            assert actual['binary_sha256'] == row['verified_candidate_binary']['sha256']
            assert actual['input_sha256'] == base['input_sha256']
            assert actual['timing']['repeats'] == base['timing']['repeats'] == 50
            b = statistics.median(base['timing']['api_seconds'])*1000
            c = statistics.median(actual['timing']['api_seconds'])*1000
            error = max(v['relative_l2'] for v in actual['errors'])
            verdict = '通过' if actual['gate']['passed'] else '拒绝'
            lines.append(f'| {model} | {b:.4f} / {c:.4f} | {c/b-1:+.2%} | {verdict} | {error:.3%} | '
                         f'{base["timing"]["allocated_bytes"]/2**20:.2f} / {actual["timing"]["allocated_bytes"]/2**20:.2f} |')
        else:
            lines.append(f'| {model} | 待实板 | — | — | — | — |')
        evidence['models'][model] = row
    lines += ['', '两图使用同一真实完整模型捕获输入，各3次预热＋50次。'
              '局部筛选分时测量，原图来自已有筛选，起止温度与逐次耗时存于证据；'
              '不能把局部变化外推为完整32/24层解码提速。若结果有优势，再做同温交替复测和完整模型验证。', '',
              '本轮两模型变化均不足0.2%，没有测得值得扩展的耗时优势；暂不编译完整模型。'
              '整数权重保真成立，但它不保证编译后的矩阵执行更快。', '',
              '初次编译因新增节点ID不以数字结束被ACUITY拒绝，未运行板卡；修正ID后保留原失败目录。'
              '量化参数与数值门槛未因失败改变。', '',
              '源码：`compile_down_postscale.py`；重建本报告：`report_down_postscale.py`。', '']
    (HERE/'Down投影缩放拆分试验.md').write_text('\n'.join(lines))
    (HERE/'evidence-attack-20261013/down-postscale.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print({k: v.get('actual', {}).get('gate') for k, v in evidence['models'].items()})


if __name__ == '__main__':
    main()
