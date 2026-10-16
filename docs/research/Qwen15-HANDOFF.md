# 1.5B NPU研究接续入口

2026-10-16文档更新；研究证据截至提交`ccab3543a03d72eb531df1f6288714cf6b0b0736`，已核对在远端研究分支。本次没有新跑实验。先读[项目总交接](../HANDOFF.md)、[当前TODO](../TODO.md)，性能证据见[提速报告](Qwen15提速-20261015.md)。

本分支为`research/qwen15-rwkv`，工作目录`/home/ubuntu/Documents/ChatGPT/端侧minibot/qwen15-rwkv-worktree`。生产板端8080仍保留CPU服务，实验NPU候选没有替换它。用户已在旧关机记录后重新上电并更换小风扇；实时在线/温度/磁盘状态接管时再核查，不恢复历史PID。

## 最新效率结论

完整28层及151936词输出头，ctx128、44输入、固定64输出，使用KV增量decode。当前最快是**单层图 + decoder-group=28 + group-head + fast-greedy**，性能模式关闭trace、宿主绑CPU6/7。

| 方案 | decode token/s | 引擎TTFT秒 | 证据边界 |
| --- | ---: | ---: | --- |
| group0 | 4.614207 | 8.600666 | 本轮单次对照 |
| group28 | 5.032149 | 8.042304 | 共5次中位，64个输出token完全相同 |
| 两层图融合 | 4.840659 | 8.363703 | 全28层单次，输出一致但较慢，不采用 |

group28较本轮单次group0约+9.1%解码、−6.5%首字，不是五轮配对显著性结论。所有方案仍复读，属于失败质量候选的效率研究，不是产品验收。重复试验起温29.2–32.5°C，不能混入旧低40–50°C/中55–<65°C矩阵。driver分配707732408字节（约675MiB），释放后0；进程CPU约单核1%，不是整机CPU百分比，也没有完整同轮换页/I/O/功耗对照。

## 当前结论

- 唯一新Transformer为Qwen2.5-1.5B-Instruct，纯文本，完整28层、hidden1536、Q12/KV2、4片完整输出头。CPU既有Q4_K_M正常中文生成，44输入、自然EOS18输出初测10.2769token/s，未重新调优成最终CPU基线，不能与固定64输出NPU直接计算正式加速比。
- NPU的W4不是CPU同算法Q4，且编译器以INT8存储低4bit数值；不得宣称已证原生INT4加速。
- strict-mask修复了合法分数低于旧mask值的问题；动态行最小值避免极大负哨兵毁坏校准。
- V per-channel路径几乎归零，改DFP W4后第0层首token V误差降到0.047%，hidden0.127%。敏感Norm/残差保留FP16。
- 全28层`w4-vdfp-sensitive`仍复读。最后两层W8未解决。**同一套导出常量在工作站FP32自由生成也复读**，不能只归因NPU执行或第26层。第26层实际输入重放误差约1.30%，全深度累计误差约20%不是这层局部误差。
- group4样本NPU API占总响应99.09%，包含驱动等待、同步和搬运，不能直接当算术时间。每token28层+完整head，scalar prefill使44输入首字仍约8秒；batch prefill未适配。
- 第26层真实输入孤立重放：fullconnect约89.2% busy cycles、前三个大矩阵79.4%，不是整模型墙钟分解。没有实测DDR带宽饱和的结论。
- 全矩阵DFP W4局部API约快35%，但多个矩阵权重误差80–99%，拒绝；DFP W8无提速且down权重误差77.8%。两层融合也没有收益，不继续无依据扩展这些方案。

## 文件与复现

详见[适配报告](Qwen15适配实验-20261014.md)、[适配证据](../../npu/kv-cache/evidence-qwen15-20261014/summary.json)、[效率证据](../../npu/kv-cache/evidence-qwen15-speed-20261015/summary.json)。原始大文件不上传GitHub，保留在工作目录`npu/results`。

| 本地相对路径/标识 | 用途 |
| --- | --- |
| `npu/results/full-qwen25-15b-sharded-c128` | 初始prompt、CPU完整参考；后续脚本仍复用其输入 |
| `npu/results/full-qwen25-15b-sharded-min-c128` | strict-min修复后的主要导出/编译根目录 |
| 上述目录的`w4-vdfp-sensitive` | 当前效率研究variant，含`board-*`完整输出 |
| `npu/results/qwen15-groups-20261015` | 提交分组筛选原始结果 |
| `npu/results/qwen15-g28-repeat*-20261015` | group28重复结果；计入筛选中的一次后共5次 |
| 板端tag `full-qwen15-vdfp-c128` | 现有最快研究图，不是生产服务 |
| 板端tag `full-qwen15-last2w8-c128` | 最后两层W8失败消融，不是推荐替换版本 |

- `compile_lowbit_stack.py`支持校准复用、W4/W8和敏感精度开关。
- `single_decoder_fixture.py`支持任意层的FP32前缀或真实捕获输入；这是局部诊断，不是全模型验收。
- `exported_greedy_reference.py`用于导出常量的FP32自由生成，重算prefix仅用于离线正确性，绝不能拿其速度当CPU对照。
- `run_full_model.py`、`validate_full.py`继续承担完整板端运行及轨迹对照。旧验证文件名`full-validation-w8.json`兼容历史命名，实际variant可能为W4，须看元数据。
- `summarize_qwen15_evidence.py`生成可提交的小型证据，不含凭据。
- `qwen15_group_search.py`和`summarize_qwen15_speed.py`复现分组与效率摘要；脚本含固定来源路径，先核对依赖和现有部署，运行时用新label保留旧结果，不盲目重跑。
- `qwen_fused_fixture.py`与`compile_lowbit_stack.py`已有两层融合支持；当前完整融合候选较慢，源码能力不等于应采用该配置。

## 下一步优先级

1. 固定prompt/轨迹，对输出头、V定点量化、其他W4矩阵分别消融，先让导出常量FP32生成恢复，区分权重/常量退化与执行误差。离线参考只用于正确性，不当作CPU性能基线。
2. 核对原始BF16/FP16权重来源、版本和SHA；以保留分组尺度、异常值和必要高精度的量化方案兼顾大矩阵执行效率。不能默认SDK逐通道W4等价于GGUF Q4_K_M。
3. 提速重点放在实际大矩阵路径、布局/转换和batch prefill。group28已经是提交调度当前最佳，二层融合和全DFP失败要保留，不重复无变化实验；任何新局部收益都需扩全28层核验。
4. 质量通过后扩512/1024有效历史×32/64/128新增，CPU Q4_K_M正常配置搜索、五轮同温对齐，记录完整资源和实际回答。既有500M/1B的Q8矩阵不能替代Qwen验收。
5. 再做新候选跨请求KV/HTTP、真实多轮、30分钟稳定和音频共存；ctx128短试验不具有这些结论。
6. 不扩大RWKV候选数量。用户关注HellaSwag、TextVQA和陪伴体验；视觉RWKV仍只是实验候选，尚无板端视觉部署结论。WSL和实体开关机/重启见[TODO](../TODO.md)。

## 空间及安全

TF历史余量很小，部署前重新读取实际空间、计算图/完整logits与512MiB额外预留。所有测试用共享host lock和板端flock；当前工作树host锁是指向原项目锁的符号链接，迁移时不能变成两把独立锁。

此前仅清理了两个已失败Qwen实验的decoder远端副本，均逐文件校验本地SHA，可恢复；embedding、head、日志、生产模型未删除。不要并发板端测量，也不要为腾空间删除历史500M/1B模型。NPU候选继续隔离于生产CPU服务。
