# 下一步 prefill 优化：先完成组合版验收，再筛选内部历史裁剪

本记录仅依据已回收的日志与现有源码，不启动编译、不访问板卡。当前组合版发生过关闭时内存错误；即使ASan复现通过，也须等正式O2版本的释放、精度与完整矩阵验收后才能升级基线。

## 当前瓶颈与口径

`intern-nextbench-20261025/results/candidate-matrix-r3/events.jsonl`，每格6次固定32输出，context256-greedy：

| 固定静态 / 动态正文 | TTFT P50 | prefill P50 | prefill占TTFT的逐次比例中位 |
|---|---:|---:|---:|
| 64 / 32 | 0.579s | 0.560s | 96.8% |
| 64 / 64 | 0.954s | 0.936s | 98.0% |
| 128 / 32 | 0.581s | 0.562s | 96.6% |
| 128 / 64 | 0.978s | 0.958s | 98.0% |

静态长度含system模板；动态正文之外另有8个模板token，实际新增prefill为40/72。首head约17.8–19.0ms，setup约0.45ms。继续优化prefill是主要方向，不能靠继续减少setup获得数十个百分点。

`combined-matrix-r1` 的true8组合已观测到TTFT约0.494/0.870/0.496/0.891s，true8尾批约0.101–0.110s，但该run关闭失败，**这些只是待验收的观测值**。先修复关闭/所有权问题并重新完成正常关闭、driver归零及同输入全词表/自然回答对照；不叠加新图掩盖内存问题。

## 首选可测候选：外部C256不变，只裁内部无效历史

保留decode容量256和prefill的外部K/V、mask输入形状。只在prefill图内部，将必然被mask屏蔽的历史尾部移除，再做QK、softmax和AV。

在**本轮冻结协议**（静态64/128、正文32/64、模板8、true8开启、最大完整prompt200、缓存复用按16对齐）下：

| 图 | 当前调用的最大起点pos | 内部past可保留的上界H | softmax总宽：当前→候选 |
|---|---:|---:|---:|
| t16主批 | 176 | 176 | 256+16=272 → 176+16=192 |
| t8尾批 | 192 | 192 | 256+8=264 → 192+8=200 |

例：静态128＋正文64＋模板8=200，t16起点为128/144/160/176，最后t8起点192。静态KV首次准备也在上述界内。decode仍可使用256槽，正常生成32个token。此提案不删除任何有效token，也不移动绝对RoPE位置。

实现轮廓：在`split_kv_attention.py`现有past/current拆分图中，给past K/V增加`[:H]`；mask由原输入的`[:H]`和`[256:256+T]`两段组成；相应修改softmax后past/current概率切片。保持所有原权重与原量化参数，新增slice/concat张量继承对应输入的精确量化签名。外部11输入/5输出契约不变，继续与decode共享同尺寸KV，不需要额外常驻第三套bank。

**必须加运行时边界断言：t16的pos<=176、t8的pos<=192，且输入协议仍满足完整prompt<=200。** 这些上界不是模型通用能力；更长输入、图片token、改变前缀对齐或批次调度时必须退回原图或重新证明，不能悄悄截断history。若未启用true8，pos192的8个有效尾token会落入padding-t16图，此时H176不合法。

收益未知：理论上少算29.4%/24.2%的attention列，但MLP、FC、当前token注意力、外部buffer flush均未减少；不能把列数变化写成同等TTFT收益。SDK可能将slice物化成设备搬运，抵消甚至反转收益；还可能改变softmax/matmul内核与舍入。需看实际NB/NPD和输出。

最小筛选：先普通l0及敏感l2，各t16/t8共4图。按既有量级新增NB约96MB、编译约1分钟仅作排期估计。通过后只扩24个prefill图（约572MB），复用现有decode/head/embedding；替换现有两bank，避免额外常驻权重。编译时间、NB大小与运行收益均以实际结果为准。

## 次选与成本

| 候选 | 可测内容及成本 | 主要限制 |
|---|---|---|
| 全容量C240 / C248，必要时C232 | C240+t16、C248+t8的softmax总列数恰为256，可先用单块查看SDK是否选到更好的形状；不存在已知收益。完整一档需12decode＋12t16＋12t8约858MB NB。 | C240/C248只比C256减少6.25%/3.125%历史槽，可能收益很小；新形状数值与布局风险仍在。当前combined runtime明确拒绝非256的t8配置，必须另建配套runtime，不能直接换图。 |
| 按精确契约共享mask/RoPE输入，预计算RoPE | 无图重编译；先分段计时再决定。现grouped代码每批约121次flush和49次invalidate，其中72次flush是各层重复的mask/cos/sin。按实际契约分组共享可减少重复写入与flush。 | 24层RoPE**实际有FL14/FL15两组**，不可盲目共享同一份量化数据；必须用`same_tensor_contract`逐端口验证并按组建立独立所有权。当前close问题未解决前不要叠加新的buffer alias。 |
| 内部KV更细分档、更多固定M图 | 可能减少更早批次的无效历史或调用次数，仅在上述单块结果支持时再研究。 | 每新增一个完整prefill bank约286MB NB且增加NPU常驻分配；按请求重载会重新引入加载延迟。不先建多档全栈。历史Intern t32/t64每token曾慢于t16，不能把“大batch”当确定提速。 |

容量下界不能误判：最长prompt200＋输出32，当前循环实际最多保留231个已输入KV；保守工程预算按232。C224不满足；C232虽满足序列界，未证明是SDK友好形状。C240是一个16对齐的保守候选，不保证比C256快。即便缩容，也必须保留原prompt/token序列与最大输出，不用缩短请求伪造提速。

现有profile只细分scalar/decode/head，没有直接细分batched prefill。建议补独立诊断项：embedding/RoPE、mask/input准备、flush、group API、invalidate、KV publish、最后hidden读取。旧C1280 profile用总API减去各已记录phase的NPU API得到的残差，仅提示prefill宿主侧约33/61ms量级，不能套作C256成绩。VIPLite当前公开头文件的`vip_flush_buffer`没有offset/range参数，不能凭设想跳过CPU写后flush或宣称已实现局部cache清理。

## 内部裁剪的fixture与准入协议

1. 从合格组合版采集真实NPU有效KV及输入，覆盖四格完整调度，重点t16 pos176、t8 pos192；另含更早pos和前缀恢复。保持同hidden、同KV有效区、同RoPE、同mask。现有独立FP32 fixture可作辅助，不能替代捕获的NPU KV。
2. 原图与新图的**外部输入完全同字节**；断言裁掉的mask列全部无效，并用非零哨兵污染这些无效KV槽测试是否漏屏蔽。只裁padding，绝不从有效history中抽样。
3. 先比FP32图与全部5个板端输出，记录finite、逐字节一致性、逐端口relative L2/max-abs、重复与交替输入稳定性；未做到字节一致时必须解释差异，再做全模型固定反馈logits/argmax及自然回答检查，不因快而放宽现有质量线。
4. 同温度/亲和条件做A-B-A，3次预热＋至少30次单块计时；NPD诊断与正式计时分开。只有API和完整TTFT均有可重复净收益，才扩图。最终保持四格同IDs/cache、输出32、音频前端常驻，报告P50/P95及资源、关闭后driver归零。当前每格6次不足以声称可靠尾延迟上限。

源码与证据：`intern-nextopt-20261025/runtime-combined/{resident_session.c,batch_prefill.h}`、`runtime_profile.h`；`npu/kv-cache/{split_kv_attention.py,Prefill批次筛选.md,Decoder矩阵瓶颈与候选筛选.md}`；本目录`SINGLE_BLOCK_REPORT.md`、`full-stack256/full-audit.json`、`prefill8-stack256/full-audit.json`。原splitKV、compact GQA、gate/up DFP和group/mask已在当前路径，不能重复记作新收益；A8及扩大down量化已有精度失败证据，本轮不重新包装成低风险提速。
