# PaliGemma2-3B 中文 text-only NPU 可行性（2026-10-23）

## 当前范围：先中文纯文本，视觉后置

最新用户要求只考虑text请求、中文。当前阶段不加载视觉，不编译视觉；下文含视觉的参数表仅保留为以后预算参考。NPU量化目前是官方BF16权重派生的混合W8方案，不是INT4；CPU SFP格式也不能冒称本次NPU INT4。

已用官方PaliGemmaModel._update_causal_mask做不加载权重、无pixel_values的真实4-token输入检查：mask[1,1,4,4]全0，第一query可见第四token。forward只在pixel分支处理特征，此后无条件调用该mask函数。因此纯文本官方Pali外层仍为prefix-LM；若绕过Pali只用Gemma2因果decoder，就是另一个推理语义。证据text-only-mask-check.txt。

当前18个探针来自3句中文（介绍自己、难过陪聊、桌上有什么），每句2种mask×3位置；按输入字节哈希只有13唯一snapshot。BOS/末token重复，不是18独立句子，更不是独立留出；见fixture-coverage.json。

中文text-only预算：decoder+head+scale+KV512的payload为2,675,762,432B（约2.49GiB）；若全部主机BF16 embedding页也驻留为3,861,013,760B（约3.60GiB）；若decode/prefill各存独立decoder常量，则不含主机embedding已4,703,339,776B（约4.38GiB）。这都未计编译打包、命令、激活、系统与语音，所以权重共享仍是重要门槛。纯文本FP16参数payload5,234,287,104B，已经很紧，若另加载FP16 prefill图就超过板卡RAM。

当前板测已回传：FP16/W8各18例有限输出、driver归零。FP16最坏hidden相对L2 0.5758%，W8为0.8523%；W8单层P50 11.861ms，FP16 61.515ms。该结果支持继续单层batch32门槛，尚未验证递归KV、26层、词表头或中文生成质量，不能外推tokens/s。

这是静态评估与实施门槛，不是已经实现的完整 NPU 支持。当前只有 layer0 FP16/W8 图已编译，18个冻结输入板测另见REPORT。本轮不启动额外大编译、下载或板卡作业，先待主任务压缩WSL VHDX。

## 判断与门槛

完整FP16常驻不适合约5.7GiB RAM的板卡：原始模型权重已有6,064,484,832B，尚未计KV、NPU工作区、系统、语音或主机embedding副本。可以探索真实BF16→经校验的W8 decoder/head，再迁移视觉；不能承诺Intern 1B的速度或固定system KV收益。

采用已确认门槛：先读板端FP16/W8误差；若W8明显异常，先定位不扩26层；若数值有希望，再做layer0 batch32双向prefix与vip_dup_network共享试验。共享失败或prefix不一致时停在这里，不先编译多套全模型。

## runtime复用与必须新增

|部分|可复用|必须新增|
|---|---|---|
|生命周期|VIP加载、IO contract检查、超时、run_group、driver释放|各新图dtype/shape/cache layout验证|
|decode|层间hidden alias、KV追加、RoPE输入、trace|26层、D2304、8Q/4KV/HD256、四处(1+w)RMSNorm、GELU-tanh、softcap50、Q缩放1/16；显式heads而非D/HD|
|embedding|token协议、按行读取|257216×2304，BF16/FP16 mmap读取，送Gemma前乘48；现runtime仅接受F32文件|
|词表头|多shard组、全词表logits与argmax对照|tied矩阵独立导出W8 head，8×32152词；最终(1+w)RMSNorm与softcap30；采样/logits保留softcap|
|停止|EOS/finish reason|外层generation EOS1与text_config另列107的任务约定需核对|
|局部attention|外置mask|交替sliding/full、4096窗口、绝对RoPE位置与cache索引；先容量768后跨4096边界|
|图像|常驻协议、计时|RGB224、官方bicubic、(pixel/255−0.5)/0.5、patch14、256位置、27层SigLIP D1152/16头/HD72、LayerNorm、GELU-tanh、projector1152→2304；接受特征而非仅token IDs|

视觉是LayerNorm而非Gemma RMSNorm，attention无causal mask。官方get_image_features在projector后除48，Gemma输入再乘48，最终进入解码层等效projector输出；必须避免图像特征重复乘48。

## prefix-LM与固定KV不兼容处

官方Transformers 4.57.6 PaliGemmaModel._update_causal_mask在推理prefill开放整个prompt prefix。system token的layer0后hidden及后续层K/V依赖本轮问题/图片，因此不能把独立system KV作为跨轮无损前缀。

权重可常驻；完全相同的整段prefix可按图像哈希+所有token IDs缓存；相同图片的视觉特征也可缓存。改变动态输入后只保留system KV不等价。强制causal prefix是模型行为变更，须单列实验。

现resident_session.c:42–57,205–212的复用依赖因果性；batch_prefill.h:130–160按短chunk发布历史KV。双向prefix要求每层先有全部prefix的该层输入。不能仅打开chunk内三角mask就认为支持跨chunk双向前缀。

正确方案可用静态全prefix长度图（32/384/512及有效列mask），或层优先两阶段分块：先算全部prefix Q/K/V，再分query块对全部K/V做attention+MLP。后者需拆图/共享权重，不能沿用现有跨层chunk流水。

## 精确payload与估算边界

参数/原始payload/KV按真实safetensors header精确计算；NBG/driver峰值必须编译加载后测。详见model-tensor-shapes.json、model-parameter-budget.json、full-npu-budget.json。

|组成|参数|BF16 bytes|计划混合payload bytes，未计scale/打包|
|---|---:|---:|---:|
|26 decoder层|2,024,515,584|4,049,031,168|2,024,755,200|
|tied embedding/head原矩阵|592,625,664|1,185,251,328|独立W8 head592,625,664|
|视觉|412,442,352|824,884,704|413,136,864|
|projector|2,656,512|5,313,024|2,658,816|
|最终norm|2,304|4,608|4,608|

视觉仅Conv/FC设INT8，其余位置embedding/norm/bias保留FP16，尚未实施。逐输出通道scale额外：decoder2,822,144B，head1,028,864B，vision1,091,520B，projector9,216B。编译器可能存INT64 bias、对齐/压缩，因此不是NBG或driver承诺。

FP16 KV=26×2×4×capacity×256×2B：容量512为54,525,952B，768为81,788,928B，1280为136,314,880B，4096为436,207,616B。先768以容纳256图像token、短system/动态输入及有限输出；必须约束最大输出。

|驻留情景|可计算payload合计|尚未包含|
|---|---:|---|
|一份decoder+head+视觉+projector+scale+KV768|3,119,921,824B，约2.91GiB|编译打包、命令/工作区、主机embedding、系统/语音|
|上项且全部主机BF16 embedding页面驻留|4,305,173,152B，约4.01GiB|同上其他开销|
|decode/prefill独立两份decoder权重+其余+KV768|5,147,499,168B，约4.79GiB|主机embedding、工作区、系统/语音；常驻余量不足|

embedding按需mmap只对访问页面计RSS，不必预触碰全1.185GB；要求完整锁页应按第二行。现Intern要求F32 mmap文件2,370,502,656B，虚拟映射大小不等于RSS。

layer0真实NBG为FP16 156,015,160B、W8 70,649,288B。乘26只是粗估4,056,394,160B / 1,836,881,488B；其他层压缩/范围不同，不能称全模型实际文件大小或driver。

**Intern的IO alias不等于共享权重。** load_prefill为每个prefill图独立load_net，仅vip_set_input连接KV/hidden；resident_session还加载独立decode/head。已查Intern和MiniCPM源码没有vip_dup_network或SET_MEMORY_POOL调用。初始预算必须按两套常量算，不能假定驱动自动去重。

driver即使共享，decode/prefill两个NBG文件通常仍重复保存常量。全W8模型+一套prefill文件+主机BF16 embedding按payload粗估约6.25GB；逐层编译临时峰值另预留0.8–1.0GiB。建议完整推进前F可用至少16GiB，始终保留8GiB。不要同时保留26份FP32 ONNX和26份import.data（单项各约8.1GB）；逐层保存哈希、NBG、参考后再清理可再生产中间件，原BF16保留。板卡若保留CPU SFP约4GB须另核算，完整NPU未验证前不先删除已验证模型。

## SDK共享接口的准确范围

原头文件：/opt/opi-lab/restored-slim3-20261017/board-runtime/usr/include/vip_lite.h。

- 152–158行：VIP_DUP_FOR_CMD_BY_NETWORK=0x01，原图与dup输入/输出shape相同，仅IO地址不同。
- 159–164行：VIP_DUP_FOR_CMD_BY_NBG=0x02，不同NBG共享权重，要求**相同网络拓扑**，允许不同输入尺寸。尺寸可变不表示任意两图可共享。
- 166–169行：VIP_DUP_FROM_NBG_FILE=0x100；MEMORY标志0x200。
- 723–741行：vip_dup_network(data,size,type,original_network,&dup_network)。dup仍运行/将运行时不能销毁原图。
- 233行：VIP_NETWORK_PROP_MEMORY_POOL_SIZE=6查询工作区；249–251行：VIP_NETWORK_PROP_SET_MEMORY_POOL=65设置可共用工作区。工作区共享与常量共享是两回事；仅严格串行且大小/对齐/生命周期符合SDK时尝试。

候选调用：`vip_dup_network(prefill_path,0,VIP_DUP_FOR_CMD_BY_NBG|VIP_DUP_FROM_NBG_FILE,decode_net,&prefill_net)`。必须检查返回值、IO contract、输出与driver增量；编译器可能因batch变化优化为不同拓扑/常量排列。失败保留错误码，不能去掉检查或静默退回普通加载后宣称共享。**此调用组合尚未在Pali实机成功验证。**

## compact之后可直接执行的最小一步（本轮未运行）

1. 先以当前FP16/W8原始hidden误差为基线，保留异常样本和driver归零。现在W8单层误差低于1%且未见几十％异常，因此值得继续，但不是整模型质量通过。
2. **仅导出layer0 batch32双向prefix。** 复用真实BF16权重：hidden[1,32,2304]，past K/V各[1,4,32,256]，mask[1,1,32,64]，cos/sin各[1,1,32,256]。past全部屏蔽，当前32列只开放有效prefix列；每个有效query看全部有效prefix。新增至少3句未参加当前校准的中文日常指令，保留原3句；官方完整层逐有效token对照，同时验证padding、改变末token对早token的影响。验证new K/V数值，不能继续只检查hidden。
3. 独立创建scalar/batch32图量driver增量，再按前述vip_dup_network尝试共享。比较返回码、IO、所有输出、关闭后的driver0。通过后再单独试memory pool，分清常量与工作区节省。新导出/import/NBG预计0.8–1.0GiB峰值；板新增约70–160MB图，实际driver待测。新编译在compact后再开。
4. 两图正确且能共享后，才加真实layer1，跑两层prefix→逐token decode递归KV；再扩26层、8词表shard，做完整logits、固定反馈、中文自然输出对照。两层接head也只是截断模型的算术测试，不能当完整模型中文质量。
5. 中文文本正确性、共享内存、常驻与TTFT/decode达标后才讨论视觉。先视觉单层，后27层和projector；不在本轮同时推进。

不优先选择26层NPU decode+板上CPU prefill：CPU gemma权重约4GB与新增约2.6GB文本NPU权重同时常驻有RAM冲突，cache布局/精度也不同。可用WSL官方模型生成离线prefix KV来诊断NPU decode，但这不代表板卡在线TTFT或端到端部署。

本次方案未引用旧Pali进程CPU核数统计；旧/proc/pid/stat空格切分对含空格线程名不可靠。CPU资源等待主任务修订，不使用旧0.13核数据。

## 下一步文件与实现入口（未创建运行任务）

- 新build根目录建议`/opt/opi-lab/paligemma2-npu-prefix32-20261023`，不覆盖本次scalar ONNX/图/18例证据。
- 从`export_layer0.py`抽出state加载与`Block`为无副作用模块；目前导入该脚本会自动执行导出，不能直接import后以为只取类。`Block.forward`的动态序列reshape可处理32行，但必须替换输入/掩码与校验，保留官方Gemma2DecoderLayer对照。新脚本`export_prefix32.py`分别保存32行hidden/new_k/new_v参考与padding信息。
- `prepare_w8.py`继续使用矩阵转置SHA映射7个FC；新batch校准范围重算且单独存档，不写回scalar范围。若要满足dup同常量条件，两个图必须用相同权重量化scale及dtype，并实际核对导出常量数据；激活范围不同可能引发不同图优化，不能仅比较原始权重SHA。
- 给`import_calibrate.py`传新source/output路径；`compile_nbg.py`目前硬编码build根目录，应新增`--build-root`和明确结果标签再用于新试验。不更改已完成两个NBG。
- 新`prefix32_probe.c`须核验6输入和3输出：hidden32×2304，K/V各4×32×256；原`layer_probe.c`是scalar2304输出，不能直接复用其硬编码长度。输出全部有效hidden和全部K/V供WSL重算。
- 新`dup_prefix_probe.c`按顺序测“scalar独立→prefill独立→关闭”和“scalar原图→dup prefill→关闭dup→关闭原图”，每步记录driver字节/内存池需求，任一返回码或IO不符即记录失败。需先验证单独图精度，再用相同输入对dup输出验证，防止共享错常量。共享memory pool放第三个独立条件，不能混入权重节省结果。
- 所有新板端步骤仍交唯一board owner，在hostlock+flock下、120秒进程deadline、温度上限与driver归零检查中执行；新增编译只能在主任务compact完成后开始。
