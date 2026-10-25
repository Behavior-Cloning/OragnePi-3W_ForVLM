# C512/M32 row-scale 整模 A/B/A 源码独立审计

当前冻结源码未发现新的实质阻断；运行结果和默认部署资格尚待验证。审计仅Windows/WSL读文件及本目录独立文档写入，无板卡、SDK、锁调用，无运行源码修改。审计末尾观察到controller为running。

配置与已经完成rc0的父实验c512-m16-m32-aba-r1的M32方案对应，baseline-config逐字节相同。三组顺序original-m32 → row-scale-m32 → original-m32-repeat；相同binary bf19922f…、C512/M32及环境变量。29个主文件中仅prefill-layer2.nb由34f3b3a2…改为f0bebd8b…；其余decode、head、embedding，以及12个原true-M8图完全相同。候选overlay路径/符号链接/实际NB字节大小与SHA均由contract检查，加载前runner逐SHA核对29主文件和12尾图。候选静态audit已绑定，避免只依赖文件名。

候选不仅用于动态输入，也用于固定system的M32预备步骤；每个测量请求均reset后只准备system。静态总token64/128（正文59/123+模板5）；动态正文64/128/192/256+模板8，实际新增72/136/200/264；正文与全部token/hash沿用父fixture。运行时会按真实tokenizer重新编码并断言。最大prompt+31=423，不超512。正文最后8token走原true-M8，已断言每请求true_tail8_calls=1、tail_rows=8、scalar_prompt_tokens=0；不能把候选M32与尾部混合精度隐去。

每版24性能请求（8格×3场景×1次），固定32 feedback且diagnostic=False；48自然请求（旧24+前轮已用new-frozen24），EOS或32上限，生成后查询RAG Top3；8请求×4步全词表诊断独立进行，不进入速度数据。三版期望72性能、144自然、24诊断及3次driver归零。32反馈的预测一致性不是自然任务准确率；“new-frozen”标签在本轮已经是回归语料，不是首次未见测试集。额外20个批边界fixture本轮未运行。

runner与已审capacity512-full相比只有两处说明文字变化；worker、RAG代码、输入文件逐字节一致。独立10个负例（容量、decode、M8、反馈、profile、顺序、candidate、overlay、重复次数、fixture）均被contract拒绝。采样和资源释放保留原逻辑；每测量前reset、worker异常abort、正常close需driver0；前端或监控错误导致失败。

语音前端常驻并流式播放合成/预录WAV，最多一条排队语音加当前活动语音；RAG嵌入模型常驻，自然阶段实际检索，性能阶段并非每次同时检索。性能输入是冻结文本，不来自此时ASR输出，因此这是模型常驻竞争测试，不是ASR→动作整链路延迟。无视觉编码、真实麦克风或电机。Top3的互斥方向、否定和参数缺失仍须人工审查。

controller具备hostlock及板端flock、独立目录、上传源快照及远端SHA核验、3600秒执行/3750秒观察上限，不自动重跑。异常保存recovery_pending及scratch，部分失败可能仍需手动回收。启动tmpfs阈值为staging+200MiB，MemAvailable为staging+2GiB；这是当前实验入口条件，不是保证运行期间始终保留该数值。系统/swap写入仍可能发生。需从实际资源、audio overrun及driver记录判断是否有效完成。

本轮候选对原基线含两处down I16权重差异和动态归一化，单块改善不能作为整模晋升依据。完整64～256正文及所有批边界的100ms目标仍未达成。
