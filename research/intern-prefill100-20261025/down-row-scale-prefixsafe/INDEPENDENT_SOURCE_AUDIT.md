# C512/M32 protected-system runtime 独立源码审计

结论：两个运行时协议缺口已在 C 源码中封闭：保护前缀准备必须 max_new=1，保护模式续请求只允许复用完整且恰好 system 长度。源码与新构建记录一致，未发现本次增量引入新的别名/组成员/释放缺陷。这里只做文件读取与静态推导，**未执行 SDK、板卡或模型，也没有可运行 runner 的通过结论**。

## 已独立核对

- 与 capacity512-runtime/runtime 对比，只有 batch_prefill.h、resident_session.c 改变。其他九份 C/头文件/测试源码逐 SHA 相同；source.diff 由实际两树重新生成，字节完全一致。两文件的 prefixsafe-i16 父源 SHA 也匹配。
- build.json 记录 gcc -O2 -Wall -Wextra -Werror 成功；实际 78,920B binary SHA 为 0ed64448e560cd3b973491cc4bcd2901e27818ff45fb165659d95c5e5d5fdb67。只确认编译产物一致，未重新编译或执行。
- header7 非零仅允许 reset、max_new=1、请求长度恰等于该前缀、长度整除主 batch、且存在原 L2 graph。reset 将 original_prefix_tokens 同步到主 M32 与尾 M8 bank；续请求不能在已保护前缀之前回退。header7=0 仍为旧 first-chunk 模式。
- C512/M32 true M8 eligibility 与原基线相同；M8 原图从 OPI_FIRST_PREFILL8_L2 显式读取，缺失即拒绝，不能误用 M32 图。11 输入/5 输出按元素数、shape、dtype/DFP 属性核验；额外原网络输入输出绑定至敏感块相同有效 buffer。
- 固定 system64/128 均整除32，准备阶段没有 M8 尾调用。动态正文64/128/192/256加8模板后，各有2/4/6/8次主调用加一次真实 M8；所有动态位置均≥protected prefix，选择候选块。最大的 prompt+31 为423≤512。以上为静态 Python 数值推导，不是 C runtime 测试。
- group 永久拥有 blocks2..11，block0 和原/候选 block1 始终单独执行，没有移除再加入 group。session_release.h 逐字不变：先释放所有 group，再全部网络，再 owned buffers，两套 bank 的额外网络均纳入释放。运行时 driver 归零尚未验证。

## 必须由下一 runner 保证

保护模式现在先将 reused 上限裁到 prefix，再检查 reused 不足 prefix 即拒绝，因此接受的续请求 reused 恰好等于受保护长度，匹配的动态后缀也不能复用。header7 准备阶段 max_new!=1 直接拒绝。这两项增量反向撤销后 resident_session.c 的 SHA 精确恢复初版；其余10份源码 SHA 未变。独立 Python 按 C 条件推导2052个 reuse状态及384个准备条件组合全部符合预期；这些不是 C 二进制/SDK实测。

header7=0 故意保留旧逻辑，runner 仍须实际启用保护模式，并冻结64/128 system、同token与同输出契约、核对 actual reused/new。运行时并不限制受保护长度只能64/128，实验覆盖范围仍由配置/runner负责。初版审计保存在 INDEPENDENT_SOURCE_AUDIT-initial.md/json，不覆盖历史结论。

同一新 binary 比较原图/all-row/protected-system/原图复测；约束相同输入、同输出策略、无 EXACT_PREFIX。保护模式因永久 suffix10 架构有3次 NPU提交，相比不保护模式的一次整组提交，热路径调度代价必须保留在实测中，不能描述为零开销的缓存准备修改。

额外 M32/M8 原图及候选图需要 source SHA、实际端口证据；prefix-only logits 对照、整段固定反馈 logits、自然48与分组日志计数都仍需实测。前缀最后 logits 相等不是所有 KV 字节相等的证明。模型常驻、音频/RAG、资源、swap、关闭释放另需完整实验。

协议文件开头仍将 word7 注释为 reserved，正文已定义 protected-prefix；这是文档不一致，不影响当前8字 wire layout。模型目标 ≤100 ms 未由本次源码审计达成。

机器记录见 INDEPENDENT_SOURCE_AUDIT.json，包含完整构建/source SHA 与8格推导。
