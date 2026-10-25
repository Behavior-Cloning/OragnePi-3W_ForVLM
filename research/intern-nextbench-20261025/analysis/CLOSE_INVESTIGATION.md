# Combined runtime 关闭异常独立调查

结论：原 `combined-matrix-r1` 的关闭失败仍然成立，根因尚未定位。原行为的 ASan 与 O2 关闭追踪版本均未复现错误；保守调整关闭顺序的 v2 通过 3 次最小关闭循环。这是有限验证证据，不能表述为已经修复或根除故障。审计只读取 Windows/WSL 已回收文件，没有访问板卡。

## 证据链

|阶段|工作负载与结果|能够支持的结论|
|---|---|---|
|原 r1|baseline 与 tail8 各完成 24 性能、12 自然、12 成本自然、5 诊断；tail8 关闭时出现 `double free or corruption (out)`，无正常 closed；repeat 未启动|该运行失败；完成推理不能替代正常资源释放|
|close-asan-1|原关闭行为，O1 ASan，1 次 system64 预热 + 新增40 token / 固定32反馈；closed driver0，rc0|最小案例未复现|
|combined-asan-full-1|原关闭行为，O1 ASan，完整单 tail8 顺序：24 + 12 + 12 + 5；closed driver0、complete、rc0|完整单 worker 顺序未复现|
|combined-close-trace-1|原关闭行为，O2 详细追踪，MALLOC_CHECK_=3；同完整单 tail8 顺序；closed driver0、complete、rc0|原优化级别加追踪时未复现|
|close-v2-3cycles|保守关闭顺序调整，3 次独立最小循环，每次 system64 + 固定32；3/3 closed driver0，rc0|候选在三次最小循环能正常关闭；尚非长时间或全矩阵验证|

原失败首先是 worker 进程关闭管道，没有返回 closed。finally 再次调用 close 时出现 `I/O operation on closed epoll object`，这是后续清理异常，不是原生内存破坏的根因。进程死亡后驱动内存恢复为零，只能说明最终清理，不能算原 r1 正常关闭。

## 输出和文件核对

对两次完整调查运行分别按 `(case, repeat)` 与失败 r1 的 tail8 对齐，独立比较实际 tokens，不依赖单 variant 中自我参照的 `output_parity` 标签：

- 固定反馈32性能预测：各 24/24 完全相同，输入/复用/新增 token 计数相同。
- 自然结束的原12质量输入：各 12/12 完全相同。
- 标点填充的成本输入自然结束：各 12/12 完全相同。这不是额外12个独立真实质量场景。
- 五个诊断输入各生成4步：各 5/5 token 完全相同，包含 `s64-d64-2`。
- 直接读取10个 `.f32` 文件：每个 2,426,784 字节，即4步 ×151,674词表 ×4字节；全部有限，实际SHA等于记录SHA；分别与 r1 对应文件逐字节相同。
- 两次完整运行的全部图文件声明与校验哈希一致，图集合与 r1 tail8 相同；二进制记录SHA均与 Windows 中对应实际构建文件重新计算的SHA一致。具体完整SHA和逐案例记录见 JSON。
- 最小ASan以及 v2 三循环的固定32预测也与 r1 `s64-d32-0` 相同；这些最小试验没有 full-logits 文件。

O2 close trace 独立解析得到584次 buffer销毁、584个不同handle，40次network销毁、40个不同handle，40个完成标记全部配对。未发现重复handle，但“此次销毁唯一”不能排除其他运行中的生命周期问题、SDK内部问题或更早发生的堆损坏。

## 版本与限制

原失败二进制 SHA256：`c222d12ef7876b483b6b5748fa439f75bf315baebd49e318adbe01515997aa8e`。

ASan调查二进制：`719f390ffadace84f8afe68ffdfd18363e6efef3abb97fa38e5dd39b76b1871b`，O1 + address sanitizer + close日志；`ASAN_OPTIONS=detect_leaks=0:abort_on_error=1`，泄漏检测关闭。专有SDK、内核驱动及其内部池分配不代表均被ASan覆盖，因此没有ASan报错不等于不存在内存缺陷。

O2追踪二进制：`17b06d118ef79952112caf8ae74c0a6f62a80ff2403e7df64caf20cb5ec0aa62`。v2候选二进制：`ff758d8eb013b7ebd05eae1f95634ecef065b92f0bf1fca7b292256351a2900d`。保守调整在销毁全部group和network之前保留用户buffer，再释放拥有的buffer；这降低已识别的生命周期风险，但尚未证明该风险就是原失败原因。

两次完整复现都是单tail8 worker；原r1是在同一Python harness里先跑baseline再跑tail8。ASan、追踪、进程内分配历史都会影响布局和时间，因此不能称为完全相同二进制、完全相同运行条件下稳定复现的因果实验。调查版本不用于最终速度排名。

当前三循环共6次推理（每循环一次system准备和一次固定32），不是三次完整矩阵。新的 `combined-matrix-r2` 正在执行，必须等待实际回收、完整三variant数量、输出校验、正常closed和driver0，才能再提高候选可信度。即使该短矩阵成功，也不自动代表小时级稳定或已根除间歇错误。

本次计时仍为 Intern NPU 文本路径、录音语音前端和RAG常驻；没有视觉编码器、真实麦克风或物理电机。固定32反馈测试是受控性能/数值实验，自然输出质量另列；一致性不等于高层指令正确。capacity256适用于本矩阵最大200输入+32输出；不能外推任意长文本、图像token或64输出。

RAG Top3执行限制仍成立：`保持静止`的Top2/3可能是前移/后移，`点头回应`的Top2包含摇头，compatibility为unadjudicated且mock_only。不得因LLM token一致，就将Top3全部视为兼容动作。

## 追加：combined-matrix-r2 有限验证

同一v2二进制的base→tail8→repeat完整短矩阵已回收：72性能、36自然、36成本自然、15诊断、3次正常closed driver0，rc0/meta complete。15个实际f32全部finite、SHA正确；本轮三variant及原r1共有结果逐字节一致。root随后健康采集driver0、alloc/free550/550。独立审计见 `combined-matrix-r2/INDEPENDENT_RESULT_AUDIT.md/json`。

此结果覆盖先baseline再tail8的顺序，比先前单variant调查更完整，但仍只有一次约9.2分钟控制器周期、504.224秒前端音频。它是关闭候选的有限验证，不是原r1根因定位或小时级稳定认证；原r1失败结论不撤销。
