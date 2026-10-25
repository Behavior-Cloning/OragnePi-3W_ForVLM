# Softmax row224 / M256 源码独立审计

**通过本次有界两层 A/B/A 探针源码检查；未作数值、速度或整模资格判断。** 审计仅访问 Windows/WSL 文件，无板卡、SDK、编译或新模型执行。

两处原 softmax 被改为各八个连续行切片 `[1,2,224,768]`、同参数 softmax，再沿 axis2 拼回 `[1,2,1792,768]`。覆盖 0–1791 所有行恰好一次，顺序不变；完整 768-key 归约轴保留。原 axis2 中包含 7×256 的 query/group 行，切分可能跨原 query/group 边界，但各行独立归约，顺序恢复后数学语义不变。生成 C 进一步核对16个分块 softmax 的物理 axis0、两个 concat 的物理 axis1，未误切归约轴。

仅原两处 softmax 节点改变，其余原节点逐项相同，增加32节点。全部原量化条目保持，新增 slice 沿用 FP16 输入，新 softmax 沿用原 I16 fl15 输出。原 source.data、实际导出 payload 的完整 SHA 相同；生成 C 的32处常量声明逐字节同，所有16个外部端口声明逐字节同，`nbg_meta.json` 结构完全相同。NB 已改变，不能把数学等价或权重相同视为实际输出无损。

注意 `status.actual_ports_exact` 在构建器中来自 `verify_contract`，该函数检查生成 C 端口顺序；它不是独立 SDK 导入 NB 的运行查询。本审计补充了生成 C 全端口属性及 nbg_meta 对照，作为本次有界探针源契约。后续实际 probe 的 geometry/输出尺寸/finite/5端口数值必须另查。

控制器固定原图→候选→原图，各3次预热+30次计时，同一冻结 canonical 输入/探针。复用已审 runner，清除 OPI 环境，明确关闭 NPD/IO profile，CPU4/5；双锁、唯一 tmpfs 目录、全部上传前后 SHA、1GiB+100MiB tmpfs 预留及额外256MiB MemAvailable 预留、78°C监控、有界超时与失败回收均保留。控制器不自动重启、不清理失败证据。

当前候选 SHA `0588fb60e18097dc0c6c3e544c8bf11be0c544d84b8a82422e4a67ac9cbebcb7`。完整冻结源与配置 SHA 见 JSON。本探针无语音/RAG、只有两层和一个 FP32 派生历史轨迹，不可外推完整24层 TTFT 或100ms目标。
