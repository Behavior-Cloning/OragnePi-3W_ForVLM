# Prefill优化进展：保留原最快配置，逐项验证候选

2026-10-25。目标仍是system总64/128、动态正文64～256时，完整新增prefill与模型首token均≤100ms；尚未达到，不把少输入或前轮动态KV收益当算法加速。

## 固定参照

system正文59/123+模板5=64/128；动态正文64/128/192/256另8模板，实际新prefill72/136/200/264。只复用system，模型TTFT不包括上游语音/排队/system准备。

旧C256+true8组合的S64/D64、S128/D64首token P50分别0.869842/0.8904535s，输出固定32。它仍是本项目已测短输入最快参照；C256指计算图容量，不是静态输入长度。最新C1280单项算子实验不是此配置的合并升级版。

## 已完成的新实验

1. C1280仅将prefill L2内两处FP16 down由FC改Conv，同容量原图→候选→原图配对：D64 TTFT约1.230/1.271s，D256约4.258/4.269s，较同C1280原图约降低21%。输出固定32，decode约14.7 tokens/s。当前仍远超100ms，且慢于旧C256短输入参照。
2. 24个长语义case/版本，48条自然输出上限32、实际2～10 tokens含EOS，全部EOS完成。两版严格完整表达都是9/24；其余区分部分完成、安全遗漏及主体/方向不明，不能把37.5%当“不相关率”。22/24预测序列全同，一处标点、一处左臂指令的动作主体遗漏。旧成本case“挥手/举起”跨过小margin，证明非逐位无损。RAG Top3有明确冲突，mock未执行，不能升级默认部署。
3. W8 FC轴/布局单块：保持原量化、全部权重和bias字节、外端口、canonical输入，C512/M32层0/1，3次warm+30次API。A/B/A中位10.8665455/10.8308580/10.8912750ms；约0.33%差，原前后漂移0.23%，无实质提速，不扩栈。5输出45056个元素逐字节一致，3释放0。此处M32是物理query块，history128来自独立FP32前缀，不能当完整模型32token TTFT。
4. 原始DFP A16/W8直接Conv和等网格affine＋DATACONVERT两次类型拒绝证据保留。显式GenerateNBG(NULL)在原图与BITCAST候选都崩溃；改用SDK官方main、真实11个canonical输入及SAVE_NETWORK_BINARY流程，原图导出的NB与原有NB SHA完全一致，随后BITCAST与I16存储候选均成功导出。此证据排除了“BITCAST必然无法导出”的结论，但未单独证明旧NULL-query崩溃根因。
5. 两种普通W8层Conv候选的M32板测已完成并独立审计。固定C512、真实层0/1融合块、同一canonical输入（128历史位置＋32物理query行），每组3次warm＋30次纯API计时：

| 候选 | 原图前 P50 ms | 候选 P50 ms | 原图后 P50 ms | 候选相对前基线耗时 |
| --- | ---: | ---: | ---: | ---: |
| W8保留＋等网格BITCAST Conv | 10.8287535 | 13.5229945 | 10.8320665 | 增加24.880% |
| 原W8码符号扩展为I16存储的Conv | 10.8661680 | 13.1410940 | 10.8389805 | 增加20.936% |

两组各自5输出共45,056个f32与原NPU逐字节一致，原前后重复也一致；每组3次release均为0。BITCAST前后基线漂移+0.031%，I16存储组−0.250%，不能解释约21%～25%的退步。这两个M32配置不扩整栈。W16这里只是原I8整数码符号扩展为I16存储，数值/scale未增精度，不能称原存储字节相同；候选driver分配45,351,216 B，比原多17,996,544 B。

这些结果只覆盖单块、一个canonical输入，排除加载/转换/宿主IO，无语音或RAG常驻；不是完整模型TTFT、全模型质量或长期稳定性结果，也不证明其它batch同样更慢。

## 1×1 Conv改写的含义

这里的1×1 Conv是每个token上Linear `y=Wx+b` 的另一种SDK算子与布局实现，不进行跨token卷积，不改变Transformer层数、attention逻辑或KV缓存策略。数学表达相同不保证低精度累加、舍入和输出逐位相同，也不保证选到更快kernel；FP16 L2曾提速而普通W8的M32两条路线与M256 BITCAST均未提速，必须按各图实测。

## M256单块已完成：仍未提速

`bitcast-c512-m256-r2` 已完成、控制器rc0，独立审计已完成。C512、真实层0/1融合块、同一canonical输入的128个历史位置＋256物理query行，3次预热后各30次纯API计时：

| 顺序 | P50 API ms | P95 API ms |
| --- | ---: | ---: |
| 原图前 | 210.7638185 | 211.0295107 |
| W8保留＋BITCAST Conv | 213.5066215 | 213.7834007 |
| 原图后 | 210.8479020 | 211.0506679 |

候选相对前基线耗时增加1.301%，原图前后P50漂移+0.040%；5输出共360,448个f32与原NPU逐字节一致，原前后重复也一致，3次release均为0。111个资源样本最高41.974°C。这是两层单块API，不是正文256token的整模测试，更不是完整264个新增prefill或TTFT；没有语音/RAG并发，不能外推整机负载或长稳。

r1因WSL保活失效在上传阶段结束，0有效测量；其失败证据保留，未混入r2。r2使用独立SSH保活，13,238,272 B canonical输入预先放入tmpfs，A/B/A共用同一输入且都在计时前预载；SHA已核对，收尾确认临时输入已删除。临时文件清理不表示所有NB/日志副本已删除。NB/接口与常量清单仍见 `runtime/official-export-c512-m256-l0/deployment-manifest.json`。

本轮普通整数Conv在M32和M256都没有测到速度收益，不扩到完整模型，不替换默认图。后续重点回到历史C256＋true8最快配置，评估能否在相同输入、容量和质量门槛下合并已证实有效的改动；或建立独立的量化校准候选与留出集。两条方向都需要新实测，不承诺能达到100ms。

## 资源与保全

长语义运行最高48.918°C，PSS峰600.83MiB，NPU约718.38MiB单列；swap-in/out本轮增0。预录语音18739帧，939帧处理超过16ms，最大lag约0.642s，非真实麦克风/图像/电机。W8布局单块无语音/RAG，9点最高40.982°C。新增BITCAST/I16存储M32单块分别采样10/9点，最高40.176/40.548°C；只代表短时单块观测，不能替代整机CPU/内存/温度或长稳报告。

为留存储，只有已结束实验的板端上传副本和已回收诊断被删除，Windows逐SHA保存；此前一次删除38904128B后可用1251676160B；这是历史快照，不是当前剩余容量。fullconv旧overlay指向的候选NB现须重新上传才能复用，production原图未改。详见cleanup-completed-fullconv-copies-20261025.json。

报告：[旧表及同容量对照](ORIGINAL_COMPARISON.md)、[长语义质量](fullconv-quality/results/fullconv-long-quality-r1/QUALITY_REPORT.md)、[独立质量审计](fullconv-quality/results/fullconv-long-quality-r1/INDEPENDENT_AUDIT.md)、[布局单块测试](results/w8axis-c512-m32-r2/REPORT.md)、[布局独立审计](results/w8axis-c512-m32-r2/INDEPENDENT_AUDIT.md)。

新增报告：[BITCAST M32板测](results/bitcast-c512-m32-r2/REPORT.md)、[BITCAST独立审计](results/bitcast-c512-m32-r2/INDEPENDENT_AUDIT.md)、[I16存储M32板测](results/w16conv-c512-m32-r2/REPORT.md)、[I16存储独立审计](results/w16conv-c512-m32-r2/INDEPENDENT_AUDIT.md)。 [BITCAST M256板测](results/bitcast-c512-m256-r2/REPORT.md)已补齐，独立审计待完成。

版本状态：Windows本地HEAD `57a45e075880d4182b0bf3c455bedae5d344f980` 尚未成功推送（网络错误），最近已核实的GitHub远端为 `d35fd319da7a636f41de281ff8dd6a88a5d652e8`。新增实验文件与本次文档按后续提交归档；本次只更新文档，没有提交或推送。
