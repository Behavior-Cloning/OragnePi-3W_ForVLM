# 真 batch8 尾部：单块板测契约

状态更新：单块板测已回传并通过有效8行逐位一致检查，结果位于主任务 `research/intern-nextbench-20261025/results/graph-probes/prefill8/summary.json`；下列内容保留可复现的输入与运行契约。当前只覆盖 Intern 第 0/1 层融合块，不包含整模型或语言质量结论。

## 冻结来源

- 候选 NBG：`/opt/opi-lab/intern-nextopt-20261025/prefill8/l0/export_nbg_unify/network_binary.nb`，24,140,496 字节，SHA256 `22ccfa0e941848bebc4b9c1ffac2d4e1192b3252df96d3dc1546a6e9b43eaec1`。
- 基线 NBG：`/opt/opi-lab/source-validation-20261017/gguf-fresh/export/w8-fused2-c1280-t16-cap1-productwide/l0/export_nbg_unify/network_binary.nb`，SHA256 `44271ccc82b1ac33037c25994f101966c6493041a60264ba705fa8ff5def6e4a`。可使用板上同 SHA 的 `prefill-layer0.nb` 符号链接复用。
- `tensor-probe`：本目录 ARM64 可执行文件，61,832 字节，SHA256 `2d53cafb9102f7af39569f1a153c94f3798928185e9156c38211c52a532dba09`。编译命令/依赖哈希见 `build-tensor-probe.json`。
- 两图 capacity1280；分别真8和16 query；14个FC矩阵/偏置和全部校准参数完全相同，详见 `prefill8-build-status.json`。使用相同导出权重的独立 FP32 参考。

## 输入与范围

输入位于 `/opt/opi-lab/intern-nextopt-20261025/fixtures/<case>/t8/input.f32` 和 `t16/input.f32`。
四例为 `s64-d32-0`、`s64-d64-0`、`s128-d32-0`、`s128-d64-0`。固定 system64/128 均包括5个模板token，动态正文32/64另有8个模板token，尾8起点分别96/128/160/192；无新语义填充，仅沿用已冻结成本测试输入。

历史 KV 来自相同源 Q8 权重解量化后、各层完整因果 FP32 前向，非零，**不是从现役 NPU 捕获的 KV**。这能验证同实际中文输入状态的块算术，但不能替代完整运行的缓存/输出验证。t16 的额外后8行按 resident 行为重复最后一个 embedding；有效前8行的 token、历史 KV、RoPE 位置及因果范围与 t8 相同。第二层历史也来自该层真实 FP32 前向。输入/参考/图 SHA 和每张量维度在 `prefill8-fixtures.json`。

11输入顺序：hidden、两层past_k、两层past_v、两层mask、两层cos、两层sin。5输出：layer1 hidden、两层new_k、两层new_v。t8 hidden[1,8,896]、KV[1,2,8,64]；t16对应16。比较 t16 的前8有效行，KV必须沿token维裁切，不可简单裁掉整个flatten后一半。

## 由唯一 board owner 执行

仅传候选图、probe、各例 `input.f32`（不需要传 `.data`、FP32 reference）。不得运行旧的 `run_tensor_probe.py`，其旧IP/锁接口不适用于本任务。

在外层 hostlock + 板端 flock 内，确认现resident退出、温度正常，顺序每例 t16 → t8 → t16-repeat。每个probe独立进程：

```sh
mkdir -p results/s64-d32-0/t16
timeout -k 5 60 env OPI_PROBE_WARMUP=3 taskset -c 6,7 ./tensor-probe t16.nb fixtures/s64-d32-0/t16/input.f32 results/s64-d32-0/t16/output.f32 50 results/s64-d32-0/t16/timing.json >results/s64-d32-0/t16/probe.log 2>&1
```

随后更换图/input/结果标签为t8，再用相同t16图/input保存到t16-repeat；依次其它3例。命令只示范单次，不自动申请板卡锁。每次退出码、温度、驱动释放量必查；若超时或错误，停止并由owner回收本任务进程。图在进入循环前已经 load/prepare；3 warmup不计50次。

`api_seconds` 只计 `vip_run_network`，不含图加载、输入量化/flush、输出invalidate或整体调度，**不能当整模型TTFT**。每次输出必须稳定且 `released_bytes=0`。`OPI_PROBE_NPD` 不开启，以免查询扰动。

回传结构为 `<results>/<case>/{t16,t8,t16-repeat}/{output.f32,timing.json,probe.log}`。

主机分析：

```sh
/opt/opi-lab/export-venv/bin/python research/intern-nextopt-20261025/analyze_prefill8_probe.py --results <returned-result-directory>
```

分析覆盖5个张量的完整参考误差、有效8行误差、t8/t16差异、重复t16漂移、逐次API时间。主机两形状的浮点运算顺序会带来约1e-6级差异，不宣称逐位相同。差异与速度都审查后，才决定敏感层2/3以及完整12块候选；最终还需相同冻结中文输入的全词表与自然生成对照。

## 空间与版本

本次编译15.26秒，单块scratch175.2MB；不重复复制整套现役图。新源均在此独立目录，未改冻结mask源/图。`prefill8-fixtures.log`保留首次过严2e-6浮点阈值失败，重试放宽到既有图参考1e-5后四例通过；实际误差完整留档，不抹除失败记录。
