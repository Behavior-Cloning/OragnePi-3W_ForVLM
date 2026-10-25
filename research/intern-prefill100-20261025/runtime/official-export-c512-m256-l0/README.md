# M256 BITCAST Conv 单块候选交付

2026-10-25：主机导出、实际 NB 端口查询及独立常量审计已完成；尚未运行此 M256 候选的板端性能/数值测试。`deployment-manifest.json` 的 complete 指候选交付完成，不指整模型或板端验收通过。

## 范围与输入

- InternVL3-1B decoder 的第 0/1 层合并块，容量 C512、query M256；不是完整 24 层模型。
- canonical 输入沿用 `fixtures-canonical/c512-m256-l0/probe-input.f32`，13,238,272 字节。历史 128 token 来自独立 FP32 全深度前缀状态，并非板上 NPU 捕获的 KV。
- M256 是图内 query 行数，不等同于用户动态正文 256 token：完整聊天输入还需计入 8 个动态模板 token。本单块没有请求 TTFT 或 decode 速度结论。
- 系统/动态正文的整模型 64/128 × 64/128/192/256 协议没有在本目录运行，也未预先缓存动态输入。

## 交付

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `bitcast/network_binary.nb` | 35,975,168 | `907689cdee9a1f767f4b64d8854bcbae27b5c16871f26500519ed977fee4c250` |
| `plain/network_binary.nb` | 35,976,976 | `e9b2d8af40f96341eda1e7ff07b867ccde9bbcf9a755746c86e6dafec2de11d5` |
| `deployment-manifest.json` | 以实际文件为准 | `b50c3194c9e33dd445af6f1a90475197f521278106f57d8e2c5ebb342c7ed215` |
| `INDEPENDENT_EXPORT_AUDIT.json` | 以实际文件为准 | `08c480712cc22273d30d54d617b1f3ce5fdf5f276e9639533a6954870d41bbad` |

plain 通过官方导出流程重新生成，NB 与之前实测的原始 M256 FC 图逐字节相同。`finalize_manifest.py` 重读源证明、独立审计、两个 NB、实际端口和 canonical 输入，全部断言通过；输出在 `delivery-validation.log`。该脚本不编译、不运行模型、不访问板卡。

## 改动与等价审计

原 14 个 FC 改为 1×1 Conv 后，生成 C 需要在 signed INT16 DFP 与数值等价的 affine 网格之间连接；本候选仅把已经逐边验证的 14 个 DATACONVERT opcode 改为 BITCAST。每边枚举全部 65,536 个 I16 整数值，证明输入输出数值网格一致。

14 个权重仍保存 I8 原始整数码；14 个偏置的整数码、offset、存储 dtype、反量化逐元素数值均与原始 FC 相同。whole payload SHA 为 `839fa51843b33140e864d01423f9da81e14a996cbe3d666c57fa0e84040a55e2`。6 个权重和 6 个偏置的 DFP 描述变为数值等价的 affine 描述；per-channel 轴随 Conv reshape 调整并核对通道数与 scale。**量化文件并非字节不变**，保持的是整数码及实际量化数值。

`inspect_nbg.c` 通过 SDK 导入真实 NB 并查询参数：11 输入、5 输出的顺序、维度、dtype 和有效量化属性逐项与原图一致；没有执行图推理。原始记录为 `*-binary-ports.json`，解释后的记录为 `binary-ports-normalized.json`。DFP 仅按 fl 解释；raw SCALE/ZERO_POINT 的 union 别名值不是有效量化字段。

## 可复用导出流程

保持原 SDK 生成 main，使用 Create → Verify → 11 个真实 canonical 输入的 PreProcess → RunGraph 自动保存。11 个文本 tensor 都由冻结 f32 分拆，以 `%.9g` 写入再回读验证 f32 逐位一致。输入顺序为 hidden、k0、k1、v0、v1、mask0、mask1、cos0、cos1、sin0、sin1。

关键环境为 `VIV_VX_ENABLE_SAVE_NETWORK_BINARY=1`、`VIV_VX_SAVE_NETWORK_BINARY_PATH=/work/network_binary.nb`（文件路径）、`VIV_VX_NBG_NAME=decoder`、`VIV_VX_NBG_INPUT_RANK=NCHW`；目标 `VIP9000NANODI_PLUS_PID0X1000003B`，shader core count 1。完整 argv/env、阶段返回码和耗时保存于 `status.json`。两个 variant 的 make/generate 均 rc0，生成约 25 秒；这是主机导出时间，不是板端推理时间。

不再调用此前失败的显式 NULL-size NBG 查询；旧失败证据保留在 M32 独立目录。`status.json` 保留生成时的 pending 状态作为历史证据，最终交付状态由已通过审计的 `deployment-manifest.json` 给出。

## 板端准入与限制

M32 的同路线此前在板上五端口逐位相同，但 API 耗时从约 10.829 ms 增至 13.523 ms，慢约 25%。因此本次只准备一个 M256 点，用来验证较大 batch 是否改善执行效率，不扩整栈、不替换默认运行图。

下一步由父控制器持双锁，使用原始 M256 同一 canonical 输入执行原图 → 候选 → 原图；分别报告 API 时间、五端口有限值/逐位一致或误差、温度/频率和 driver 释放。只有板端结果通过才考虑后续范围。本目录没有此 M256 板测结果、全模型提升或 100 ms TTFT 达标声明。
