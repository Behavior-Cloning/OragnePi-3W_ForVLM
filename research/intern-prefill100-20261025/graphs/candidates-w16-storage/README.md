# C512/M32/L0：原 W8 整数码扩展为 I16 存储的 Conv 候选

当前：静态审计、SDK make、Create、VerifyGraph 均通过；官方导出流程已成功生成 NB，真实二进制 11/5 端口已与原 FC NB 逐项核对一致；**未板测、未证明提速或输出等价**。本目录是独立候选，原 FC、失败的原 DFP Conv、失败的 affine Conv 均未修改。

14 个已量化 I8 矩阵以有符号方式扩展到 little-endian I16 存储，原整数码以及原 scale/fl/PC axis 保持不变；没有使用原 FP32 模型重新量化。全部 bias、其它常量、激活与输出网格、节点连接、11 输入/5 输出接口不变。权重的存储宽度增加，不是精度提高，也不能声称 raw bytes 相同。

## 当前验证

- 14/14 原码恢复 SHA 一致；14/14 原/新反量化 FP32 数值逐位相同。
- 原 FC export payload SHA 与直接 DFP Conv 的源 payload SHA 相同：`839fa51843b33140e864d01423f9da81e14a996cbe3d666c57fa0e84040a55e2`。
- 32 个常量引用、29 个唯一区间；别名保留，新区间连续、不重叠，所有非权重常量原字节保留。
- payload 29,890,566 → 59,709,446 B，增加 29,818,880 B（约 28.44 MiB）。
- make 返回 0（1.368 s）；Create 成功；VerifyGraph 返回 0（整个 verify 容器 6.232 s）。这些是工具链验证时间，不是 NPU 推理时间。
- Verify 前后 11/5 实际端口属性完全一致；候选源码和 payload hash 未变；正常释放退出。
- 初次 verify-only 阶段未调用 `vsi_nn_GenerateNBG`，该阶段记录 `nb_generated=false`。后续已用官方流程导出，见下节；两个阶段证据分别保留。共享编译锁已释放。

候选 C SHA：`e995dd7c770ad4b1f3af50ca2aaa3bb01dbe1d4f0aaa379da98dced402ae238f`。
候选 payload SHA：`08d1a7880e6924f938853df20a00f5a760527ca757345692ca789ccfb1c1e50f`。

## 文件与后续边界

`prepare.py` 生成唯一候选并逐常量审计；`static-audit.json` 保留每个 tensor 的码值、数值、scale 和 offset 证据；`decoder.patch` 是 C 的完整差异；`run_verify.py` 通过统一编译锁运行 SDK make/Verify-only，日志和命令在 `status.json`。`export/` 包含大产物，已 gitignore。

导出 NBG 需复用独立对照验证后的 SDK 官方流程。此前显式 `GenerateNBG(NULL)` 在 plain 和 BITCAST 上均崩溃，不能据此归因于 BITCAST，也不能断定未初始化输入就是根因；本候选没有复现该路径。

获准导出后仍需实际 NB 端口核对及原 FC → 候选 → 原 FC A-B-A。五个输出分别比较，有效 KV 区与 hidden 单独审计；不能由常量值相同推断 Conv 累加、溢出、舍入与 FC 逐位相同，也不能由单块通过推断全模型质量或 100 ms 目标达成。

## 官方导出更新

同一官方 flow 已在普通 FC 原图导出与原 NB 完全相同的 SHA，才用于本候选。独立 `official-export/` 使用原 SDK main、同一组真实 canonical 的 11 个 `.tensor`（FP32 文本往返回读 exact），环境包含 SAVE_NETWORK_BINARY=1、NBG_INPUT_RANK=NCHW；没有显式调用 GenerateNBG(NULL)。原 verify-only 目录及日志保留。

本候选 make/generate 均返回 0；NB 为 `official-export/network_binary.nb`，42,757,264 B，SHA `d9847c5cb9899ebcd08ca6f46a3106caa2fabc5fa33ad5a5e85b9fc24214e1a0`。导出工具输出的 RunGraph 时间属于 WSL SDK simulator，不是板端 NPU 性能。所有源 C、payload、main、makefile 和 11 输入 SHA 在导出后再次核对未变。

独立静态复核见 `INDEPENDENT_STATIC_AUDIT.json`：14 个权重码值、PC axis/layout/scales、其它 18 allocation 字节、29 唯一区间和所有别名均通过。最新导出日志与命令见 `official-status.json`。

## 板测交付

`deployment-manifest.json` 的 state=complete 表示单块候选准备完成，**不是端侧验收通过**。`actual-nbg-ports-audit.json` 通过 SDK 导入真实 NB 的 kernel metadata，确认 11 输入/5 输出的顺序、shape、dtype 和有效量化属性与原 FC NB 相同。DFP 只比较有效的 fl，不把 quant union 内不活跃的 scale 解释为量化比例。查询不执行图、没有使用板卡。

manifest 给出 NB、source、canonical 输入/参考路径及 SHA、14 权重码值与宽度变更、逐常量 offset 和独立审计证据。root 可据此运行原 FC→本候选→原 FC；本轮没有修改默认图或扩展到全栈。
