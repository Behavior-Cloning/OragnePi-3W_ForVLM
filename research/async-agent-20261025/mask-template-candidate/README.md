# Intern mask-template candidate (2026-10-25)

本候选只优化常驻 InternVL3-1B 请求前的 attention mask 恢复，尚未包含本轮板端计时结论。权威源码在 Windows F 盘；板测由主控制器在双锁下执行。

## 输入与缓存边界

当前比较固定 system 正文 **81 tokens**、system 模板 **5 tokens**，合计 **86 tokens**；动态 user/assistant 模板另 **8 tokens**。用户正文 32 / 48 / 64 tokens 时，总输入分别 126 / 142 / 158；原 batch16 路径只恢复 system 的 80-token 对齐前缀，实际重新处理 **46 / 62 / 78 tokens**。本候选不改变这一边界，也不缓存动态用户内容。

固定 user header 只有 3 tokens，拼入 system 后为 89；86 和 89 向下按 16 对齐都为 80，所以仅缓存 header 不减少本轮预填充批次。`fixed-template-token-counts.json` 保存真实 tokenizer 的结果。未启用历史 exact-prefix 模式。

## 改动和等价性

`truncate_session()` 原来不复制或清零 KV，而是恢复历史列和当前批次因果区域的 mask。容量 1280、batch16、12 个双层图、repeat7 时，恢复批量 mask 要调用 `put()` 3,483,648 次；另有 30,744 个 scalar mask 元素。若均为 2-byte 格式，最终写入约 7,028,784 字节。

新 `mask_templates.h` 用原 `put()` 分别编码 0 与 -10000；按完整 tensor contract（形状、数据类型、量化格式与参数）、capacity、batch、repeat 共享原始字节模板。模板保留原因果三角；keep 改变时重建，同 keep 命中时复用，最后 memcpy 到每个真实 mask。原 KV 布局、KV 写回、flush、图、group、批次划分、RoPE、embedding、取 token 和 LCP 逻辑均保持不变。

此法减少逐元素格式转换和判断，仍需写入全部 mask 字节。若全部 mask 共享同一 2-byte 合约，scalar 与 batch 模板合计约 292,866 字节；这只是结构估算，实际占用以 worker stderr 的 `mask_templates enabled=1 host_bytes=...` 为准，不以 host 多合约测试的分配总量充当板端占用。

## 开关与基线

- 不设置 `OPI_MASK_TEMPLATE`：完整执行原 mask `put()` 循环；仅新增一次开关判断、启动状态输出和关闭时空模板清理。
- 设置 `OPI_MASK_TEMPLATE=1`：启用模板。
- 同时设置 `OPI_MASK_TEMPLATE_VALIDATE=1`：每次构建模板时额外用原循环构建 reference 并逐字节比较；不同则退出。仅用于诊断，不能计入性能结果。

两开关按环境变量是否存在判定；值 `0` 仍表示启用，关闭必须 unset。板端 A/B 必须使用同一个候选 binary，以避免二进制或编译条件混杂。原 source 和 binary 未被本候选覆盖。

## 已完成验证

`byte-equivalence-test.json` 保存 408 例普通模式与 408 例 validation 模式结果，均 0 字节差异，边界 canary 完整。覆盖 scalar / batch 两种布局、小尺寸与实际 cap1280/batch16/repeat7、FP16/FP32/INT8/INT16、5 档整数 DFP scale、keep 改变、重复命中及 reset。`sanitizer-test.json` 另有 408 例 ASan + UBSan 通过。

WSL 的 GCC11 不支持本机 `_Float16`，故 host 测试对 mask 唯一允许的两个 FP16 值使用精确 IEEE16 编码（0 与 -10000 的 0xf0e2）。实际 ARM 候选仍使用原 `put()` 的 `__fp16` 转换。host 测试没有证明 ARM 浮点转换；板端 validation 阶段负责补齐这一验证。初次 host compiler 失败日志保留为 `byte-equivalence-first-compiler-failure.json`。上述均是正确性测试，不是速度结果。

## 构建与冻结版本

ARM64 交叉构建已以 `-O2 -Wall -Wextra -Werror` 完成，实际 SDK 动态链接成功。完整命令见 `build-result.json`，所有宏及链接目录均已记录。

- binary：`resident_mask`
- SHA256：`dc86c20e39707e617ef25453092bf652ffd619c293cf26809e27872d2ef1c49c`
- 原源码哈希：`baseline-source-manifest.json`
- 候选源码与验证文件哈希：`candidate-manifest.json`
- 对原 resident 的 diff：`resident-session.diff`

运行开始后本目录 C/H/Python 源及 binary 已冻结；后续只追加说明与结果。binary 和 object 留在工作区供部署，`.gitignore` 排除其 Git 提交。

## 板端通过门槛与限制

主控制器执行同一 binary 的 baseline → template → baseline，再单独执行 validation。固定 86-token system、正文 32/48/64，并保持语音常驻。需检查实际 prefix reuse、自然输出 token 一致性、全词表 logits 字节一致性、setup/TTFT/decode、温度、内存与 driver 释放。validation 诊断不纳入性能统计。

目前尚未在本说明中声称板端提速。旧 matrix 的正文32组 prefill 约919ms、head约26ms；即使降低 setup，也只能降低这两项之外的时间。正文64组 prefill 本身约1.53s，因此 mask 优化不能将该组完整首 token 延迟压到1s。
