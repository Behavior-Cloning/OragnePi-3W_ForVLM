# C512 L2 预填充异常诊断

范围：仅 Intern 1B 融合双层图，L0=层0/1，L2=层2/3；C512，M16/32/64/128/256。本文不是整24层 TTFT 成绩。完整目标仍为固定 system 64/128（含模板）+动态正文64/128/192/256+动态模板8，禁止动态缓存、删token。

## 已完成、可以确定的结论

纯计时见 `../results/single-block-c512-r1/analysis.json`。L2 的 API P50 随 M16→M256 为112.21/222.12/456.72/922.93/1890.17 ms，明显慢于 L0 的8.42/10.92/34.62/84.21/211.13 ms。输入、图权重和量化配置在各 M 之间保持既定源；这不能替代整模型质量验证。

已有4组独立 I/O 诊断，均3次预热+5次重复：

| 图 | `vip_run_network` P50 ms | flush+invalidate P50 ms | 同步 / wall |
|---|---:|---:|---:|
| L0 M16 | 8.4314 | 0.0378 | 0.447% |
| L2 M16 | 112.0869 | 0.0373 | 0.0333% |
| L0 M32 | 10.9062 | 0.0390 | 0.356% |
| L2 M32 | 222.0435 | 0.0923 | 0.0415% |

所以该探针的外部输入flush/输出invalidate不是L2多出的约100/211 ms的主要来源。`run_s`包括SDK提交、等待和图执行；并不是纯算术kernel。此证据不涵盖整模型的embedding/mask构造、KV更新或Python调度。

`single-block-profile-r1/IO_OUTPUT_AUDIT.json` 保存4组 I/O、4组 NPD 与纯计时输出的逐字节一致性；NPD诊断每组1次、无预热，不纳入正式速度。L0 NPD耗时明显变大（M16 13.76 ms，M32 18.05 ms），更不能混用。

## 精度例外：嫌疑点，尚非耗时归因

graph owner已确认 L0/L2 的原始 decoder.json 拓扑相同。L2 保留两处 MLP down（`b0/b1_fullconnect_116`，4864→896）及相邻路径 FP16，L0 对应FC为 I16/I8/I16。每个FP16权重矩阵8,716,288 B，周围10个新增FP16激活；其余12个FC仍I8。必须保持这些例外，或者把改量化作为独立精度候选；不能通过取消敏感层保护伪称等价加速。

仅凭FP16或总耗时不能判断CPU fallback，也不能判断NN/TP/SHADER执行。需要真实SDK node trace的 `target` 字段。

## NPD解析已完成及局限

本机SDK镜像 `ubuntu-npu:v2.0.10.2`，入口：

- `/root/acuity-toolkit-whl-6.30.22/bin/tools/device_profiler/nb_data_item.py`
- 同目录 `nb_data.py`、`node_trace.py`、`device_profiler.py`
- 本任务保存只读源快照于 `profile-analysis/sdk-source/`，哈希写入summary。

实际4个文件均为 VNPD v2、target `0x1000003b`、1 network/1 core、128 B/record。SDK新reader固定读取192 B；不能直接补零再声称存在其余counter。本任务用原SDK `ItemSegHEAD/0/1/2/3/4/5/15` 8个16B reader解码真实字段，并与既有A733 `profile_npd.py` 逐record交叉核对，全部相同、无overflow。

可复现命令（WSL）：

```sh
python3 research/intern-prefill100-20261025/runtime/profile-analysis/analyze.py
```

4份完整record/node JSON和summary已保存。计数器名称明确为 `NN_TOTAL_BUSY_CYCLE`、`NN_TOTAL_MAC_COUNT`、NN DDR/SRAM读写counter。SDK `DeviceProfiler.__calc_profile_info` 只把缺失NN profile视为错误，允许非NN target无profile。因此这些不是全引擎耗时计数。

| 图 | 记录数 | 有计数的node数 | 累加NN busy cycles |
|---|---:|---:|---:|
| L0 M16 | 1210 | 144 | 7,845,089 |
| L2 M16 | 1204 | 138 | 6,600,017 |
| L0 M32 | 2986 | 144 | 10,030,853 |
| L2 M32 | 2854 | 138 | 8,205,733 |

L2的已记录NN busy更低，不能用这些counter解释其更大的总wall time。累计busy可能重叠，不直接乘频率换算为端到端时间，也不把缺失的时间全部分给某引擎。`engine_time_fraction`暂为null，不编造L2 down耗时占比。

## 已完成node映射：执行引擎仍有明确证据缺口

原始10图未保留 `node_trace_db.json`。随后图侧已在 `graphs/trace-export/` 补齐 L0/L2 M16 的 node trace；两份最终NBG SHA均与实际板测原件逐位相同。`profile-analysis/map_trace.py` 已逐node映射原NPD，未把 M16 mapping套用到M32。

独立trace编译仅添加容器环境 `VIV_VX_DUMP_NODE_TRACE_DB=1`（入口核对于 `npu/kv-cache/compile_fused_decoder.py`）。保留：

1. 未包装 `export/vnn_decoder.c` 与 `node_trace_db.json`、fused trace（若SDK生成）。
2. 新旧最终NBG SHA、大小、完整编译命令、SDK image digest。
3. 每个profile node的 `uid.id/sub_id`、`target`、share_node_ids与候选原层名。

只有最终NBG SHA相同，才能直接给已有NPD套该node映射。若不同，先确认仅trace导致差异，再由root决定对新NBG重跑同输入NPD；不能用近似node编号硬映原记录。融合图中UID后缀可能重复，必须保留b0/b1候选标签，不能猜层号。

实际trace生产者没有写 `target` 字段，所有profile条目只有node_id。SDK新版reader的target枚举 NONE0、SHADER1、NN2、TP3、SW4、SCALER5、NBG6 不能补充不存在的事实。真实映射确认L0的UID116 down对应nodes318/319/320及339/340/341，6个都有NN计数，累计1,069,044 cycles；L2同一UID的两处FP16 down对应nodes232/294，二者均没有NN计数。它们究竟是TP、SHADER或其他路径仍未知，不能从缺失计数叫CPUfallback。拿到真实target后可以说明执行引擎；若其无NPD counter，依旧不能给总时延占比。若需精确归因，应做保持全部FP16精度的独立down子图/替换图对照，记录NPU API差分，并说明图切分会改变调度，差分不是精确单算子原位时间。

## 板端开关契约（仅root执行，本文未访问板卡）

保留已冻结 `tensor-probe-prefill100` SHA `eed89e71c0030ea15fb90d490611b6d6efdcf88a530906fdc57a1c42498e4da3`。参数仍：

```sh
tensor-probe-prefill100 graph.nb input.f32 output.f32 repeats result.json
```

- 正式纯计时：不设置 `OPI_PROBE_NPD` / `OPI_PROBE_IO_PROFILE`；`OPI_PROBE_WARMUP=3`，repeats30。
- I/O诊断：只设 `OPI_PROBE_IO_PROFILE=1`、warmup3、repeats5；记录flush/run/invalidate/wall。
- NPD诊断：只设 `OPI_PROBE_NPD=1`、warmup0、repeats1，在独立case工作目录运行并收回生成的`.npd`、output、result及stdout/stderr。当前探针禁止NPD同时warmup或resident copies。
- 布尔开关按环境是否存在判断，`FLAG=0`仍会启用，需unset。
- NPD属性在 `vip_prepare_network` 前设置，符合 `vip_lite.h` 的 `VIP_NETWORK_PROP_SET_ENABLE_NPD` 声明。
- 以上均需root的hostlock+board flock、温度/内存guard、最终driver归零。诊断不与整模型推理并行。
- `cpu_seconds`包括进程加载、转换和清理，不是每次推理CPU时间；不可据此断言CPU fallback。

## 优先级与停止条件

原L2 down节点已定位，真实target暂不可得；停止无界搜索，优先测保持FP16的FC→1×1 Conv等独立图改写；每次确认量化payload、端口、相同输入输出误差。若改写没有改善，不扩大完整24层编译。全模型最终仍须真实system64/128+动态正文64/128/192/256+8模板，以及完整KV、全词表、首token边界和中文质量验证。

当前交付：`profile-analysis/summary.json`（4份SDK counter解码与IO统计）、`mapping-summary.json`（2份exact-binary映射）、每份完整raw record/node JSON及可复现脚本。没有改动冻结resident/probe，没有额外板卡调用。
