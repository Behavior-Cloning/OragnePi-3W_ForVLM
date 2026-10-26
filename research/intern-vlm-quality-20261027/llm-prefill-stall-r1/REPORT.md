# Prefill长尾诊断 r1：未得到有效分段结果

本候选仅增加run_prefill_rows分段计时和minor/major fault计数；没有改模型权重、量化或KV规则。独立runtime和二进制，不替换现役。

板端加载未在180秒内发出ready，Python wrapper中止自身worker，尚未进入任何测试请求，因此0有效分段样本。源码中embedding mmap按需读取是待检验线索，不能声称已定位长尾根因。

## 已确认的系统干扰

kernel-evidence.txt记录多次global OOM，后台opi-vlm-model.service的llama-server被杀；有记录由resident-visual触发OOM。最近一条单调时间274503.279748，属于本轮完整视觉向量VQA运行期间。该轮18秒慢请求时间274512.340454–274531.859964，发生在该OOM约9秒之后，不能据此精确归因到embedding读取或某个算子。

post-health.txt确认本实验结束后NPU current bytes=0，932次分配/释放相等；CPU服务active/running，NRestarts=6，MemoryCurrent924483584。服务未被本agent手动停止或重启。

停止新增高内存NPU语言测试，待用户确认CPU服务能否临停后进行独占/共驻对照。未将目标标为完成。首token0.31秒/18秒混跑数据不能被包装为稳定性能。PSI和进程io接口缺失，不能当零压力。

下一步：先协调并验证内存余量，再重跑本分段诊断；确认请求从加载、词向量收集、输入准备、NPU API、结果回写各阶段耗时。不能因为控制连接超时就重复启动未确认结束的任务。
