# 容量256组合候选：真8尾图 + 单遍argmax

由已冻结的 `runtime-tail8` 为基础，仅合入 `runtime-greedy/greedy_single_pass.h` 及对应开关/调用；没有修改旧运行时、图或量化。此目录不含profile计时分支，便于同binary实验。完整NPU速度/数值仍由主任务串行板测，本文件不提前声明组合成功。

- ARM64 binary：`runtime-combined/resident-combined`，70,648 B。
- SHA256：`c222d12ef7876b483b6b5748fa439f75bf315baebd49e318adbe01515997aa8e`。
- 源SHA与ARM编译命令：`build-combined-runtime.json`。
- 相对冻结tail8源的差异：`combined-runtime.diff`。
- 合并后host验证：`combined-host-validation.json`。

## 同binary开关矩阵

所有方案保持主图capacity256、prefill_batch16、同system64/128、动态正文32/64+模板8、只复用system、相同固定反馈与自然输入。共同保留 `OPI_GROUP_PREFILL=1`、`OPI_MASK_TEMPLATE=1`、`OPI_TTFT_TAIL_MIN=8`。

| 方案 | OPI_GREEDY_I16_SINGLE_PASS | OPI_PREFILL8_DIR |
|---|---|---|
| 同binary基线 | 省略 | 省略 |
| 单遍argmax | 1 | 省略 |
| 真8尾图 | 省略 | 已验证的capacity256真8图目录 |
| 两者组合 | 1 | 同上 |

C开关按存在启用，**不能使用值0表达关闭，必须省略**。主图/头/embedding仍由原命令参数指定；尾目录须含12个 `prefill-layer{0,2,...,22}.nb`。

启用尾图时额外强制 `capacity==256`、主prefill_batch16，并拒绝scalar-block实验；每张尾图的KV输入count/dims/shape/dtype/quant/scale都必须与decoder契约一致。因此把c1280旧尾图路径传入c256组合版会失败，而不是悄悄执行不同容量。尾图关闭时不加载尾图，可作为同binary的独立argmax对照。

`ready`新增两个布尔字段：`true_tail8_enabled`、`single_pass_greedy_enabled`。controller应核对它们与配置一致，并继续核对capacity、layers、每个实际NBG SHA。`done`保留 `true_tail8_calls`、`true_tail8_seconds`；后者已经计入 `prefill_seconds`，不可重复相加。

system64/128准备阶段应0次真8；新增40/72的测量、4步logits和quality_cost每请求恰好1次真8。原12条quality没有8余数，应该0次；自然生成验证必须保留额外4条quality_cost。所有32输出的请求最大实际KV231≤256，没有截断输入或输出。

## 主机验证

- 从组合源提取实际reset/truncate函数，plain/template/validate各1,920个mask张量检查，全部通过ASan/UBSan。
- 覆盖两个prefill bank、keep反复128→64/200→128、旧mask污染后回退、KV reset归零、缓存释放；测试包含256与1280 mask结构，但运行时组合尾图仅允许256。
- 同一GCC vector单遍argmax源221,212例通过：全部INT16取值、长度1–8四值穷举、2,000随机、并列/最早索引、实际shard长度、65535边界和超界fallback。
- ARM自检已重新编译为 `runtime-combined/test-greedy-arm`，13,552 B，SHA `096435fe19729ebcbcc0cf63fc3174d141a3fc2b2544a3ef6deac4d53b8ed28f`（与已板测同源自检相同）；本agent没有再次操作板卡。

主机检查不替代组合图的全词表与自然输出实测。两优化单独通过，也不能自动推出组合无误差；必须检查完整NPU调用、共享KV与图释放。相对同binary基线分别报告TTFT/decode、峰值PSS/driver、音频lag、温度，并保留原4格system/动态token标注。

## 交接

现 `controller_next.py` 的候选复制表需要增加 `resident-combined` 对应此目录路径，确保实际复制的binary SHA和mode755；这由主任务修改，未在这里更改。新source/bin已冻结；所有本agent编译/测试进程均已退出。
