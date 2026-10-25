# Prefill优化进展：保留原最快配置，逐项验证候选

2026-10-25。目标仍是system总64/128、动态正文64～256时，完整新增prefill与模型首token均≤100ms；尚未达到，不把少输入或前轮动态KV收益当算法加速。

## 固定参照

system正文59/123+模板5=64/128；动态正文64/128/192/256另8模板，实际新prefill72/136/200/264。只复用system，模型TTFT不包括上游语音/排队/system准备。

旧C256+true8组合的S64/D64、S128/D64首token P50分别0.869842/0.8904535s，输出固定32。它仍是本项目已测短输入最快参照；C256指计算图容量，不是静态输入长度。最新C1280单项算子实验不是此配置的合并升级版。

## 已完成的新实验

1. C1280仅将prefill L2内两处FP16 down由FC改Conv，同容量原图→候选→原图配对：D64 TTFT约1.230/1.271s，D256约4.258/4.269s，较同C1280原图约降低21%。输出固定32，decode约14.7 tokens/s。当前仍远超100ms，且慢于旧C256短输入参照。
2. 24个长语义case/版本，48条自然输出上限32、实际2～10 tokens含EOS，全部EOS完成。两版严格完整表达都是9/24；其余区分部分完成、安全遗漏及主体/方向不明，不能把37.5%当“不相关率”。22/24预测序列全同，一处标点、一处左臂指令的动作主体遗漏。旧成本case“挥手/举起”跨过小margin，证明非逐位无损。RAG Top3有明确冲突，mock未执行，不能升级默认部署。
3. W8 FC轴/布局单块：保持原量化、全部权重和bias字节、外端口、canonical输入，C512/M32层0/1，3次warm+30次API。A/B/A中位10.8665455/10.8308580/10.8912750ms；约0.33%差，原前后漂移0.23%，无实质提速，不扩栈。5输出45056个元素逐字节一致，3释放0。此处M32是物理query块，history128来自独立FP32前缀，不能当完整模型32token TTFT。
4. 普通W8 FC→Conv直接DFP类型被SDK拒绝；等整数网格affine候选自动DATACONVERT也被拒。最新BITCAST生成C最小替换通过Create/Verify，但GenerateNBG size查询阶段崩溃，尚无可部署NB。正在用原始未改图做同导出器对照，区分生成器问题与候选问题；不冒充板上提速。

## 资源与保全

长语义运行最高48.918°C，PSS峰600.83MiB，NPU约718.38MiB单列；swap-in/out本轮增0。预录语音18739帧，939帧处理超过16ms，最大lag约0.642s，非真实麦克风/图像/电机。W8布局单块无语音/RAG，9点最高40.982°C，不能替代整机负载报告。

为留存储，只有已结束实验的板端上传副本和已回收诊断被删除，Windows逐SHA保存；最近一次删除38904128B后可用1251676160B。fullconv旧overlay指向的候选NB现须重新上传才能复用，production原图未改。详见cleanup-completed-fullconv-copies-20261025.json。

报告：[旧表及同容量对照](ORIGINAL_COMPARISON.md)、[长语义质量](fullconv-quality/results/fullconv-long-quality-r1/QUALITY_REPORT.md)、[独立质量审计](fullconv-quality/results/fullconv-long-quality-r1/INDEPENDENT_AUDIT.md)、[布局单块测试](results/w8axis-c512-m32-r2/REPORT.md)、[布局独立审计](results/w8axis-c512-m32-r2/INDEPENDENT_AUDIT.md)。
