"""Archive the standard-implementation checks separately from quality gates."""
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'results/oracle-standard-20261013'
ARCHIVE=HERE/'evidence-attack-20261013'


def main():
    ARCHIVE.mkdir(exist_ok=True)
    lines=['# 标准实现与完整KV缓存数值核对','',
           '2026-10-13。用Transformers 4.57.6的标准Llama/Qwen2实现，独立加载现有GGUF Q8反量化权重。在工作站运行，不是板端CPU性能测试，也不是相对训练原始BF16的无损证明。', '',
           '验证全部32/24层、最终Norm和完整词表head；每层hidden、自定义参考完整logits，以及标准实现“缓存历史→追加新消息→127次scalar decode”的128步logits均进行比较。映射严格检查所有GGUF语言权重，Llama Q/K显式还原RoPE行布局；标准库负责mask、RoPE、attention和KV缓存。', '',
           '| 场景 | 自定义/标准logits L2 | 标准缓存/整段L2 | 自定义argmax /128 | 数值参考检查 |',
           '| --- | ---: | ---: | ---: | --- |']
    results=[]
    for path in sorted(ROOT.glob('*.json')):
        data=json.loads(path.read_text());results.append((path.stem,data))
        shutil.copyfile(path,ARCHIVE/('standard-'+path.name))
        lines.append(f'| {path.stem} | {data["custom_vs_standard"]["relative_l2"]:.3g} | {data["standard_scalar_cache_vs_full"]["relative_l2"]:.3g} | {data["custom_argmax_matches_standard"]} | {"通过" if data["gate"]["passed"] else "失败"} |')
    lines+=['','所有逐层和完整logits相对L2门槛1e-4；缓存比较也是1e-4。只覆盖已列场景，不等于独立陪伴留出或真实跨HTTP会话验收。','',
            '## NPU相对标准Q8参考的额外差异','',
            '| 场景 | 完整logits L2 | 平均KL nat/token | argmax /128 |',
            '| --- | ---: | ---: | ---: |']
    for name,data in results:
        lines.append(f'| {name} | {100*data["npu_vs_standard_q8"]["relative_l2"]:.3f}% | {data["npu_mean_kl_vs_standard_q8"]:.6f} | {data["npu_argmax_matches_standard"]} |')
    lines+=['','这里包含NPU重新量化权重与执行的总差异。既有5% L2执行门槛的参考是NPU实际导出常量，不能把此表与那个门槛混用，也不能据开发轨迹KL小就宣布独立留出已通过。独立质量仍按冻结协议另测。','',
            '代码：[validate_standard_oracle.py](validate_standard_oracle.py)。证据含模型、脚本、自定义参考源码及NPU完整logits哈希。','']
    (HERE/'标准实现与KV缓存核对.md').write_text('\n'.join(lines))
    print([name for name,_ in results])


if __name__=='__main__':main()
