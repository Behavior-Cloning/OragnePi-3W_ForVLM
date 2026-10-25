# M256/L0/C512 W8 Conv：BITCAST 前段准备

只准备一个 fused2 单块，未访问板卡、未扩栈。对应原图 `../cases/c512-m256-l0`，输出在 `conv1x1-affine-c512-m256-l0/`。

M32 的 BITCAST Conv 已实测比原 FC 慢，因此 M256 仅用于检验大 batch 下的路径效率，不能提前认定提速。

## 已完成

- 14 个私有输入 reshape 改为 `[1,1,256,I]`，14 个 FC 改为 kernel1×1/stride1/group1 的 Conv，不跨 token。
- 保留 W8 整数权重、I64 bias 和输出量化；14 个内部 I16 DFP 激活、6 个 DFP I8 权重只改为等值 affine 元数据，scale=`2^-fl`、zero_point=0。全整数范围逐码验证数值相同，变化清单为 `affine-quantization-diff-m256.json`。
- 对每个源 tensor 的 shape、dtype、bytes 核对后，确认重塑常量与已有 M32 Conv data 相同，才使用 symlink，未重复复制大权重。
- canonical M256 输入下，原 FP32 图和改写图的最大相对 L2 误差为 2.103115e-6；用实际导出 FC 权重反量化作为基准时为 1.474034e-6。两组均覆盖 5 输出，不是板端精度验收。
- Pegasus 在统一编译锁下执行 10.843 s，按预期在 `node16 DATACONVERT` 被拒绝，返回 1；generated C/payload 已保留，无 NB。这是后续独立 BITCAST 替换需要的中间态，不是可运行模型。
- 对实际 generated C 再审计：14 权重和 14 bias 的 dtype、bytes、数值 scale 全同原；全部 14 输入/输出 I16 网格同原；16 个 PC weight/bias 的 axis 改写后 scale 数与 channel 维一致。整个导出 payload 与原 FC 完全同 SHA。

## 交付

- C：`conv1x1-affine-c512-m256-l0/export/vnn_decoder.c`
- C SHA：`d591153e7355e5184a4363888c7cbdc360c4d711f05db671103d03e86fa3c31b`
- payload SHA：`839fa51843b33140e864d01423f9da81e14a996cbe3d666c57fa0e84040a55e2`
- 原 NB SHA：`e9b2d8af40f96341eda1e7ff07b867ccde9bbcf9a755746c86e6dafec2de11d5`
- 状态：`status-affine-m256.json`；主审计：`m256-affine-generated-audit.json`；PC/FP32 补充审计：`m256-affine-channel-audit.json`。

`build_conv_affine_c512_m256_l0.py` 负责准备/有界编译，`audit_affine_m256.py` 读取实际导出常量审计。运行时智能体接手 14 个 converter 等网格证明、BITCAST 替换、官方 NB 导出、真实 NB 端口审计及部署清单；本阶段没有重复执行这些步骤。原失败与当前生成器证据均保留。
