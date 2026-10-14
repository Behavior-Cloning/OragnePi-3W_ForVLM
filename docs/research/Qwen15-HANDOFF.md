# 1.5B NPU研究接续入口

本分支为`research/qwen15-rwkv`，工作目录`/home/ubuntu/Documents/ChatGPT/端侧minibot/qwen15-rwkv-worktree`。这是独立研究分支；根目录旧handoff的关机状态是历史记录，当前用户已上电并授权继续测试。生产网页仍为CPU，未替换。

## 当前结论

- 唯一新Transformer为Qwen2.5-1.5B-Instruct，完整28层+完整输出头。CPU既有Q4_K_M正常中文生成，短提示初测10.28token/s，未重新调优成最终CPU基线。
- NPU的W4不是CPU同算法Q4，且编译器以INT8存储低4bit数值；不得宣称已证原生INT4加速。
- strict-mask修复了合法分数低于旧mask值的问题；动态行最小值避免极大负哨兵毁坏校准。
- V per-channel路径几乎归零，改DFP W4后第0层首token V误差降到0.047%，hidden0.127%。敏感Norm/残差保留FP16。
- 全28层`w4-vdfp-sensitive`仍复读。最后两层W8未解决。**同一套导出常量在工作站FP32自由生成也复读**，不能只归因NPU执行或第26层。第26层实际输入重放误差约1.30%，全深度累计误差约20%不是这层局部误差。
- 关闭trace+group-head/fast-greedy后4.72token/s；再decoder-group4约4.93token/s，输出token一致。仍为失败质量模型的效率测量，不是可用产品成绩。
- NPU调用占总响应约99%；每token28层+完整head，scalar prefill导致44输入首字约8.2秒。纯宿主开销已很少；不能靠减少Python开销承诺追上CPU。

## 文件与复现

详见同目录《Qwen15适配实验-20261014.md》以及`npu/kv-cache/evidence-qwen15-20261014/summary.json`。原始大文件不上传GitHub，保留在工作目录`npu/results`。

- `compile_lowbit_stack.py`支持校准复用、W4/W8和敏感精度开关。
- `single_decoder_fixture.py`支持任意层的FP32前缀或真实捕获输入；这是局部诊断，不是全模型验收。
- `exported_greedy_reference.py`用于导出常量的FP32自由生成，重算prefix仅用于离线正确性，绝不能拿其速度当CPU对照。
- `run_full_model.py`、`validate_full.py`继续承担完整板端运行及轨迹对照。旧验证文件名`full-validation-w8.json`兼容历史命名，实际variant可能为W4，须看元数据。
- `summarize_qwen15_evidence.py`生成可提交的小型证据，不含凭据。

## 下一步优先级

1. 固定prompt/轨迹，对输出头、V定点量化、其他W4矩阵分别消融，先让导出常量FP32生成恢复，区分权重/常量退化与执行误差。
2. 如需新原始BF16权重，应明确来源，避免继续Q4再量化；探索保留组内尺度或量化感知校准，不能默认SDK逐通道W4等价于GGUF Q4_K_M。
3. 质量通过后优化decoder图融合、KV/布局与batch prefill，再扩512/1024历史。当前ctx128只是正确性起点。
4. 不扩大RWKV候选数量。用户关注HellaSwag、TextVQA和陪伴体验；视觉RWKV仍只是实验候选，尚无板端视觉部署结论。

## 空间及安全

TF接近满，所有测试用共享host lock和板端flock。仅清理了两个已失败Qwen实验的decoder远端副本，均逐文件校验本地SHA，可恢复；embedding、head、日志、生产模型未删除。当前候选部署标签`full-qwen15-vdfp-c128`与`full-qwen15-last2w8-c128`。不要并发板端测量，也不要为了腾空间删除历史500M/1B模型。
