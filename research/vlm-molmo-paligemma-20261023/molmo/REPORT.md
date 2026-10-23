# MolmoE-1B 尝试部署（2026-10-23）

当前结论：尚未完成整模部署，不能给出板端 TTFT/decode。名称中1B不代表实际常驻权重只有1B。

## 已核实
- 上游 allenai/MolmoE-1B-0924；本次小文件从 ModelScope LLM-Research/MolmoE-1B-0924 下载，15文件逐SHA校验。
- 总参数约7.2B、激活约1.5B；16层，D2048，64专家、每token选8；视觉CLIP336/patch14/23层，含多裁剪和投影。
- ModelScope manifest 原始 pytorch_model.bin 为28,888,362,419字节；当前F空间不足以保留该文件及转换副本，所以没有下载大权重。

## 常驻内存硬下限

按官方配置，单专家3个矩阵均为2048×1024：16层×64专家×3×2048×1024 = 6,442,450,944个专家参数。仅专家INT8权重就需要6.00GiB，已经超过板卡5853MiB总RAM，尚未算注意力、视觉、embedding、KV、图工作区和系统。本项目此前只验证了W4数值使用INT8容器，尚未证明NPU内部原生packed INT4能把常驻权重减半，不能据此绕过W8内存下限。因此当前W8路线无法满足完整MolmoE全专家常驻；需要真正打包INT4或专家换入方案，均属未验证的新实现。

## 实际兼容性试验
1. 当前 llama.cpp converter 对未修改官方配置 dry-run，选择 OLMoForCausalLM 路径但在tokenizer/vocab断言失败。不是可直接转换的完整Molmo多模态实现；未靠改名替换成普通OLMoE。
2. Acuity6.30.22 / VIP9000NANODI_PLUS_PID0X1000003B：D2048→64概率→Top8的小图通过FP16 NBG转换；softmax对照也通过。
3. 精确D2048、FF1024的一路专家静态MLP图通过FP16 NBG转换。
4. 按官方torch.where/索引选token方式构造一路动态专家分发，小图导入失败，见compiled/dynamic-one-expert/import.log。这说明现有整图直译路径有障碍；不能据此断言所有实现都不可行。可研究CPU负责路由、NPU静态专家图，之后仍需验证64专家存储、图调度与数据搬运开销。

这些NPU算子图全部是固定seed随机权重，只是编译器兼容性试验，没有预训练质量或板端性能含义。

## 后续要求
- 可追溯MolmoE量化权重/转换器，避免拿文本OLMoE或Molmo7B替代。
- 专家静态图调度实现、真实权重单层和全模型误差验证。
- 完整视觉路径、实际图片问答与6GB板RAM/带宽/磁盘适配。

来源：[官方模型](https://huggingface.co/allenai/MolmoE-1B-0924)、[官方论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Deitke_Molmo_and_PixMo_Open_Weights_and_Open_Data_for_State-of-the-Art_CVPR_2025_paper.pdf)。本地配置、源码哈希及失败/成功转换日志均在该目录保留。
