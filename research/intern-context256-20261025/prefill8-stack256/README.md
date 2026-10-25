# Context256 + true batch8 尾批图

12个双层true-batch8图已完成编译，覆盖Intern 1B的24层；完整端侧组合结果由父任务另测。本子任务没有访问板卡、没有抢锁、没有提交Git，也没有修改已有图或证据。

## 变更与精度范围

候选通过两条独立路径构造：

1. 已通过r3的c256-t16图仅修改41项query/M相关参数：16→8、112→56、mask宽272→264。
2. 已有c1280-t8图仅修改10项capacity相关参数，1280→256、mask宽1288→264。

所有12块的两个结果JSON完全一致，没有其他图变化。每块保留自己的层相关量化参数。`decoder.quantize`、`decoder.data`及编译器产生的实际`export/decoder.export.data`与其c256-t16基线均逐字节一致。原始权重与确认过的量化payload通过只读源引用复用，不重复拷贝权重；独立审计重新验证了全部552个数组。先完成layer0编译和以上审计，通过后再顺序编译余下11块。

这证明图形变更范围及常量一致，**不证明不同batch形状的NPU舍入结果或整模型回答完全一致**。需要同输入的全模型logits/token与自然回答验证。

## 交付

- `deployment-manifest.json`：12个NB绝对路径、目标文件名、SHA与大小。
- `full-audit.json`：独立审计结果。
- `cases/l0`、`cases/l2`…`cases/l22`：各自JSON、quantize、常量引用、编译日志、conversion、same-source-audit和NB。
- NB路径形式：`cases/lN/export_nbg_unify/network_binary.nb`；部署文件名为`prefill-layerN.nb`。
- `compile_stack.py`：默认preflight，`--execute`顺序编译；不会覆写已有候选目录。
- `audit_stack.py`：完成后独立重查所有图、payload、数组及NB SHA。

12个NB合计286,140,960 bytes（约272.9MiB），编译约160.6秒。当前目录新增约299.9MiB，含可选配对fixture；F盘余24.67GiB。量化payload复用避免再增加约342MiB编译中间产物。板端仅需新增12个NB，不需传源码、fixture或权重，除非父任务选择做单块检查。

## 运行时要求

使用同时支持true-tail8第二bank和已验证greedy优化的runtime。主模型仍用r3的context256 decode+prefill16 overlay，capacity256、batch_tokens16。设置：

```text
OPI_PREFILL8_DIR=<独立的c256-t8目录，含12个prefill-layerN.nb>
OPI_TTFT_TAIL_MIN=8
OPI_GROUP_PREFILL=1
OPI_MASK_TEMPLATE=1
```

不更换主t16图，不使用c1280的t8目录混搭。head、embedding与decode全部复用现有c256路径。保持静态64/128（含模板）、动态正文32/64+模板8的同ID输入；真实新prefill40/72。预计每个目标请求应记录`true_tail8_calls=1`、`tail_rows=8`、`scalar_prompt_tokens=0`；batch调用次数分别3或5，须以实际日志核查。所有模式最大输出32，最长prompt200+32不超容量；不要沿用64输出全矩阵。

新增bank会增加常驻NPU内存，不能仅以NB文件大小代替driver分配峰值。请检查所有bank关闭后的driver释放、两bank cache恢复、同输入输出parity、实际TTFT P95与温度漂移。

## 可选单块配对fixture

`fixture-manifest.json`记录8份输入：四个冻结场景×t8/t16，路径`fixtures/<case>/t8或t16/probe-input.f32`。由已有c1280 fixture裁去masked padding，保持有效KV、绝对位置、hidden及RoPE；history分别96/128/160/192。真实历史来自原Q8的独立FP32模型，非板端量化KV抓取。

t16有8个额外padding query；比较输出时必须对每个hidden/K/V tensor仅取前8 query行，不能把整个t16输出与t8直接字节比较。保留相同历史、同token和相同mask语义。t8输出没有尾部padding。
