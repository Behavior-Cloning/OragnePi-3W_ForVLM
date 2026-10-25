# C512/M32 主块与 M8 尾块 row-scale 独立 runner 审计

结论：未发现本次独立有界试验的源码阻断。此结论只依据 WSL 文件和12个内存合同负例，未访问板卡、SDK、锁，未启动控制器。当前前序prefixsafe板测与本目录分离；本目录没有端侧结果，不能声称精度或速度已通过。

## 对照隔离

固定顺序 original-m32 / protected-main / protected-main-tail / original-m32-repeat，均使用已审硬化 binary 0ed64448…，C512、M32、相同输入与输出策略。B/C的29个主图映射完全相同，只在12个M8图中改变 prefill-layer2.nb。实际新M8 NB SHA2732f92a…、28,736,560B，与已冻结独立静态审计6c5e773c…相符；候选M32仍f0bebd8b…。

flags函数先保留原OPI_FIRST_PREFILL8_L2，再切换C的OPI_PREFILL8_DIR为tail-candidate；因此两候选的system保护网络均为原M32/M8路径，不会误把候选尾块用作原图。runner在每候选load之前用空cache fresh hash两份原网络，SHA来源是原始variant的main/tail映射。当前S64/128整除32，system准备本身不调用M8；这一事实不能免去M8原图加载契约核查。

额外tail overlay严格要求12个精确symlink目标和无多余文件；只有敏感块指向新NB，其余11个仍为原图。主overlay保持原29条。两份候选NB、两份静态审计、fixture、binary均进入上传快照和远端SHA检查。实际runtime源码/二进制、Worker、输入、RAG代码与前序已审版本相同。

## 协议与数量

两候选均以非零header7准备纯system、max_new1，随后测量仅复用64/128，actual new=动态正文64/128/192/256+8模板。固定反馈性能32步，独立自然max32/EOS，完整logits诊断4步；所有请求要求真实M8调用一次、尾行8、scalar_prompt_tokens=0。保护runtime上限限制重复动态请求仍只复用system。

完整默认运行预期96性能、192自然、32四步诊断、8单步prefix诊断、8缓存边界probe、4关闭；完整logits40文件，共82,510,656B。两候选分别有83次前缀准备、248条原块分支：warm2+performance72+natural144+diag24+prefixdiag6；两个system各重复动态两次，probe不增加原块分支。原图两组分支为0。runner与analyzer均按variant和精确(pos,rows,valid,limit)顺序核对，不只看总数。

12个负例均拒绝：取消tail/protect标志、原M8误指候选、尾目录回原图、NB/hash/size/audit错误、多余/错误symlink、额外尾图变化、variant顺序、EXACT_PREFIX。没有修改真实配置或执行模型。

## 分析与运行边界

离线分析已正确扩展两个extra-prefix事件和8个cache probe的唯一组合；四套主图29与尾图12完整核查，真实f32 SHA/finite/argmax、原图A/A、前缀与动态token、driver释放继续保留。新增直接B↔C的8组完整logits、48自然原文/gold/reply/Top3，以及每case真实M8秒数、TTFT差值和主bulk秒数（prefill−tail）比较。tail秒数是运行时整个尾prefill墙钟，不是纯NPU kernel；scalar=0断言使该拆分成立。

控制器保留双锁、唯一tmpfs路径、源快照、可执行检查、入口内存余量、3600秒timeout、3750秒观察上限，无自动重启。发生异常仅标recovery_pending等待回收；scratch不能在原始输出核验前删除。finish仅观察本轮PID/cmdline及状态，不会另启动板测。

语音流与RAG仍常驻，但模型输入是固定fixture，非实时ASR输入；无图像、真实麦克风或电机。必须报告实际swap、lag、overrun。保护B/C都包含三次提交开销，可据配对数据比较尾块增量；与原图一次整组提交的差异不能删除。三成本场景/格、48个已用回归例和8个诊断例不构成长期稳定或新独立质量认证；完整100ms目标与默认晋升均仍未批准。

冻结SHA与条件详见 INDEPENDENT_RUNNER_AUDIT.json。当前审计未观察本目录新运行结果。
