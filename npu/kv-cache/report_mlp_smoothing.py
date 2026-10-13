"""Archive negative/positive host screening without inventing NPU performance."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'results'


def main():
    evidence={}
    lines=['# MLP通道缩放与A8开发筛选','',
           '本轮仅在工作站模拟普通第0/1层gate/up的W8A8量化。校准使用固定512＋64开发输入，检查使用此前公开使用过的问候/陪伴开发输入；未读取独立v1/v2留出。每个输入都先核对FP32重写路径与原层实现一致，再模拟量化误差。','',
           '把归一化输出按通道除以s、gate/up权重按对应输入通道乘s，FP32数学上保持MLP分支不变。s由校准激活峰值与权重峰值计算，并取2的幂；alpha=0明确为不缩放对照。权重使用现有DFP峰值/有界MSE规则，激活A8用校准峰值决定范围。down仍用原Q8反量化后的FP32，未模拟残差、KV及其他NPU算子，因此本表不是全模型误差或芯片仿真。','',
           '| 模型 / 层号 | alpha | 开发检查最差P95 MLP相对误差 | 只量化权重时最差P95误差 | A8最高截断比例 |',
           '| --- | ---: | ---: | ---: | ---: |']
    for model in ('smol500','intern1b'):
        path=ROOT/f'mlp-smoothing-{model}-20261014.json'
        if not path.exists():continue
        raw=path.read_bytes();data=json.loads(raw)
        assert not data['npu_execution_tested'] and not data['speed_measured']
        evidence[model]={**data,'source':str(path),'source_sha256':hashlib.sha256(raw).hexdigest()}
        for layer in range(data['layers']):
            for alpha in (0.,.25,.5,.75):
                rows=[r for r in data['records'] if r['layer']==layer and r['alpha']==alpha]
                errors=max(r['w8a8_mlp_error']['per_token_l2_p95'] for r in rows)
                weight=max(r['weight_only_mlp_error']['per_token_l2_p95'] for r in rows)
                clipped=max(r['activation_clipped_fraction'] for r in rows)
                lines.append(f'| {model} / {layer} | {alpha:.2f} | {errors:.2%} | {weight:.2%} | {clipped:.4%} |')
    lines+=['','500M第0层的A8误差能因缩放明显降低，但检查P95仍约11.5%，其他层也没有稳定的低误差候选；1B第1层缩放后反而恶化。没有据此改动NPU主线或声称A8可用。该局部MLP误差不能与完整logits的5%门槛直接比较，也不能解释为聊天准确率下降。','',
            '结论是先保留W8/A16，继续从已实测有效的DFP编译路径、输出头与decoder布局找速度。若后续继续A8，应先改进开发校准与异常通道处理，然后重新检查导出图、全部层、完整输出和新的独立质量数据；不能拿当前两层模拟宣布成功。','',
            '复现：`screen_mlp_smoothing.py`使用证据中的模型、校准和开发检查路径；`report_mlp_smoothing.py`生成本报告。模型/输入SHA与完整各层误差见[紧凑证据](evidence-attack-20261013/mlp-smoothing-screen.json)。','']
    down_path=ROOT/'selective-down-host-intern1b-20261014.json'
    if down_path.exists():
        raw=down_path.read_bytes();d=json.loads(raw)
        assert d['baseline_reproduces_existing_oracle'] and not d['npu_tested']
        evidence['selective_down']={**d,'source':str(down_path),'source_sha256':hashlib.sha256(raw).hexdigest()}
        lines+=['## 1B普通down投影的选择性筛选','',
                '另用当前cap1＋headMSE完整24层、512历史＋64新增、最快CPU的128步反馈，在FP32中重建实际导出常量。基线全部logits先与既有独立oracle文件核对；每层在相同MLP乘积输入上比较原逐通道W8与候选单尺度DFP W8。第2/3层已有敏感FP16继续保留。','',
                f'事先固定局部预测位置P95分支相对误差≤{d["selection_threshold_local_p95"]:.1%}才选入；这只是保守候选筛选线，不是完整logits或独立质量门槛。实际选中层：{d["selected_layers"]}。','',
                '| 层号 | 原down类型 | 改单尺度DFP后的预测位置P95分支误差 | 选中 |',
                '| --- | --- | ---: | --- |']
        for r in d['local_records']:
            value=f'{r["prediction_branch_l2_p95"]:.2%}' if 'prediction_branch_l2_p95' in r else '保留敏感FP16'
            lines.append(f'| {r["layer"]} | {r["down_dtype"]} | {value} | {r["selected"]} |')
        lines+=['','本次没有层通过选择线，因此没有编译或部署新down候选，也没有新增速度结果。局部比值可能受分支范数影响；它不能直接等同最终logits误差。完整主机重放仍执行了全部24层和全部词表，但只用于验证筛选与参考一致性，不能当作NPU执行验收。','']
    (HERE/'MLP通道缩放开发筛选.md').write_text('\n'.join(lines))
    (HERE/'evidence-attack-20261013/mlp-smoothing-screen.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('host-only screens',sorted(evidence))


if __name__=='__main__':main()
