# Mask 模板静态审计：64 / 128 静态前缀切换

2026-10-25。仅阅读冻结源码与既有 WSL 证据；未访问板卡、未改 C/H/Python 源或 binary、未新增执行测试。候选 manifest 的全部文件 SHA 重新核对通过，binary 仍为 `dc86c20e39707e617ef25453092bf652ffd619c293cf26809e27872d2ef1c49c`。

结论：在当前单 worker 顺序请求、连续映射 tensor、FP16/FP32 无量化或 INT8/INT16 DFP 的支持范围内，未发现因静态前缀 64↔128、不同层/通道共享模板而新增的 mask 错配或泄漏问题。可以继续冻结候选的板端 A/B；本审计不替代实际 logits、token 与内存验证。

## Contract 与布局

`batch_prefill.h:23` 的 `same_tensor_contract` 比较 count、num_of_dims、逐维 sizes、data_format、quant_format，并对 DFP 比较 fixed_point_pos。`mask_templates.h` 再比较 capacity、tokens、repeat。因此 scalar 与 batch、不同 batch/容量/广播次数、不同 DFP scale 均不会误共享。

SDK `vip_buffer_create_params_t` 的 memory_type 没有进入 key。这一字段描述存储类型，不改变 mask 的原始数值编码；模板只含 host 字节，复制目标仍是每个真实映射 buffer，且原 flush 路径保留。buffer 句柄、data 指针、层号/通道号不应作为 key：相同 mask 语义允许共用字节模板，但从不共用实际 NPU input buffer 或 KV。

该 SDK tensor 描述无 stride/layout 字段；当前原循环本身按连续线性 index 写，模板沿用同样顺序和精确逻辑 count。不能据此扩大为任意 strided/padded tensor 的通用支持。quant_format 为 none 或 DFP 才会匹配；affine/per-channel 不属于当前支持范围。未来如引入新格式，应先显式拒绝或实现其编码与 contract key，不能复用本次通过结论。

## 64↔128 与旧列关闭

每个 contract 只有一份 raw 和一个 `keep` 状态；keep 不是分配 key。restore 发现 keep 改变时会重建全模板：先将历史与未来区域全部写 blocked，再仅打开 `[0, keep)`，并建立当前批次原有因果三角。因此从 128 降到 64 会重新关闭 `[64,128)`；从 64 升到 128 会重新打开该区间。scalar 当前槽 capacity 始终开放，batch 的行 t 只开放 `capacity..capacity+t`。

`reset_session` 使用 keep=0，仍清零原 KV；`truncate_session` 更新 valid 与 previous_position 后覆盖全部 mask，仍不复制/清除 KV。有限的旧 KV 位于关闭的 mask 后方，行为与基线相同。推理过程中只修改实际 destination mask，不修改 cache raw；重复同 keep 仍 memcpy 整份模板，不能把前一次推理打开的未来列带入新请求。

同一进程切换 64/128 不增加模板数量；首次为每个 contract 分配后，后续仅重建与复制。既有 host 测试覆盖多次 keep 改变、降回0与重复命中，但未显式列入64/128这两个数；此处针对两值的结论来自上述源码不变量，板端新 aligned 矩阵需补实际覆盖。

## 资源、释放与并发

raw 只存 0 / -10000 的编码，不存文本、KV 或用户内容。缓存是 worker 进程内静态链表，不跨进程/请求通道传输。每个 entry 的 contract 副本不拥有 SDK buffer，free 只释放 raw 与 entry，不销毁复制的句柄，因而没有重复释放 buffer 的路径。

正常关闭路径先释放 graph/buffer/driver，再 `free_mask_templates()`，其循环清空列表与 raw-byte 计数。异常 `exit` 或外层 kill 由进程退出回收 host 内存；该候选未新增 NPU/driver allocation，driver 清理由原运行器负责，仍需查看本轮 closed/driver0 事件。`host_bytes` 只计 raw 数据，不包含小量链表结构与 allocator overhead，不能作为全部 RSS 增量。

当前 C worker 单线程、单 Session、顺序处理，`vip_run_group` 同步结束后才读取下一请求；不存在同时恢复和读取 mask 的新增竞争。本全局缓存未加锁，不适用于未来多线程同时执行多个 Session；此时应移入 Session 或加同步，不能直接扩用当前实现。

## 动态内容缓存边界：调用者必须继续维护

候选没有新增 prefix-limit 协议字段，仍使用原 history 的最长公共前缀。因此仅更换不同动态问题并不自动保证只缓存静态 system：两题共享开头时，原 runtime 可能复用动态共同前缀。这是既有行为，非模板新增。

当前 `ttft_mask_matrix.py` 在每一条 measure / quality / diagnostic 前调用 `generate(prefix_ids, 1, reset=True)`，再发送完整问题，故能限制本轮复用范围。新64/128 aligned矩阵须继续同一调用策略，并逐请求断言 reused_tokens 不超过对应静态token数；不要仅设置当前candidate并不读取的 `OPI_PREFIX_LIMIT` 环境变量。若64/128是含模板的实际静态token数，batch16均可完整对齐；动态token必须分别记录正文及模板，不能混称。

## 板端证据门槛

静态输入64/128与动态32/64的精确定义应来自冻结token IDs与实际tokenizer计数。每组需要检查实际reused/new、无用户内容复用、mask validation无差异、全词表logits/自然token一致、正常close/driver归零。现有86固定prefix的mask矩阵与新对齐矩阵分别报告；不能把减少token或批次数产生的收益全部记为mask算法加速。
