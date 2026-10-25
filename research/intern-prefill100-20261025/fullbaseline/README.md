# C1280完整模型新输入范围基线

只实现、未启动板卡。此版本复用旧runtime-combined-v2（SHA ff758d8eb013b7ebd05eae1f95634ecef065b92f0bf1fca7b292256351a2900d）、原C1280图、group/mask/greedy和M16 padded尾；不加载真8图、不使用新C512 runtime，不宣称100ms或候选加速。

root审查后运行（WSL、本目录）：

```sh
python3 controller_fullbaseline.py fullbaseline-c1280-r1 --repeats 2 --phases performance,quality,diagnostic --timeout 1800
```

默认不带--phases时为performance,quality。独立run目录，hostlock覆盖部署/等待/回收，板端flock覆盖部署与整个suite及归档；SSH或超时异常保留原错误并尝试重新获取双锁回收partial，仍运行时不抢锁、不重启。

- 核心24条：system总长64/128 × 动态正文64/128/192/256 ×3家庭情景；repeat2共48次32步固定反馈。每次reset prepare system，再提交完整prompt，强制reused=system、new=正文+8模板；最大prompt392、KV423。正反顺序重复。完整prefill逐次batch+scalar后统计。
- 自然quality独立12条，按EOS实际输出长度；不把固定32反馈当自然回答。RAG query返回Top3仅作记录，仍有冲突，禁止真实执行器。
- 可选diagnostic每8格首case各4步固定反馈，实际全词表f32单独保存；不将诊断时间混入性能。
- 与旧矩阵同一常驻模拟语音Frontend和RAG；音频是预录WAV按帧流式回放，最多一个整句队列。没有真实麦克风、图像encoder或电机。RAG只在自然quality阶段实际query，performance阶段只是常驻。
- 原inputs.json完整字节复制（含20个boundary冻结记录）；本轮不运行boundary20，不能据8主格声称连续所有长度达标。
- 29项graph/embedding SHA来自旧mask-base逐项原样复制；二进制SHA另外验证。资源记录继承0.5秒CPU/内存/温度/频率/音频监控；每个worker要求正常closed driver0。

host_preflight.py使用WSL /opt/opi-lab/export-venv/bin/python，验证真实tokenizer后的全部36条输入、8格、容量，四种破坏fixture均被拒绝；结果见host-preflight.json。控制器未执行，无板端结果。
