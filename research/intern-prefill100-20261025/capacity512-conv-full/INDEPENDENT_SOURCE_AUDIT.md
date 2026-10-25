# C512 M32 FP16Conv-L2 整模候选独立源码审计

**结论：当前冻结源码无启动阻断；仅通过有条件的有界实验审查。必须先回收并审阅父轮结果，再由根节点决定执行。此审计没有访问板卡、没有启动新实验。**

## 实际证据核验

- 候选NB SHA 9b80e75c0ef95c367b5c1e423fd2ad76cb9ebb1bca4b115fda4b89dec70f9f7f，33,973,888字节；原NB SHA 34f3b3a2f8aec80ae712c5bd52a9bcbade6ca178bbff073ca54907cd1c08b250，与当前父轮M32的L2一致。
- 独立从生成C定位实际payload：14权重+14偏置的dtype及原始字节全部相同；对应dtype赋值和24个scale/zero-point数组也相同。量化文件逐字节相等。
- 图JSON只改变两处down的FC→1×1Conv及其两处reshape；group1、stride1、无padding。没有I16替换。
- nbg_meta一致；另外核对已回收fp16conv-m32-r1的VIPLite真实11输入/5输出日志，原/候选/原全部逐字节相同。此为既有板测证据复核，没有新SDK或板端操作。
- 冻结父配置SHA、main29、tail12及输入fixture完全对应；新overlay仅prefill-layer2.nb换候选，其他28主文件、全部12尾图及binary不变。
- execute先要求父run status=complete、rc0、meta complete；此判断先于board_access import。源文件上传前和远端上传后均绑定已冻结snapshot SHA，先前小缺口已修。
- runner与已审C512父runner仅说明文字不同，保持prefix-only、真实M8及token断言。三版均C512/M32：实际new72/136/200/264，最大KV423。
- 每版24性能/48自然/8四步诊断：总72/144/24；source/graph/binary与实际logitsSHA/finite/argmax、释放、分阶段资源、配对TTFT/decode由analyzer核验。诊断不混计时，自然语义不自动判合格。
- 双锁、唯一目录、超时、异常partial回收、768MiB+150MiB预留保留；本地dry preflight35,573,970字节（不含冻结配置），无板调用。
- 八个负例（容量、顺序、尾图、I16开关、额外主图变化、反馈长度、M16混入、候选geometry）全部被contract拒绝。

## 审计边界

- This is source launch review only; parent full M16/M32 result audit and root explicit execution decision remain prerequisites. Auditor did not run --execute or access board.
- Original quantizer and exported constants are preserved, but changing two FP16 down FCs to 1x1 Conv may change FP16 accumulation order. Prior single-block output already differs; mathematical equivalence is not bitwise or semantic qualification.
- Candidate main32 graph is used both for system KV preparation and dynamic input. System cache is reset/recomputed independently for every request; hot TTFT excludes that preparation but resources include it.
- M8 tail remains entirely original, including L2. This tests main32-only Conv, not a full Conv stack or any I16 route.
- Expected 3 variants x24 fixed-feedback performance /48 natural /8 four-step diagnostic requests. Fixed32 gives31 decode intervals; natural max32 can end on EOS. Each timing cell n=3 and no continuous-length/boundary20 or long-term stability coverage.
- All48 natural presentations are prior-used regression fixtures (labels preserve historical24 development+24 frozen-v2); exact-match counts are not semantic accuracy. Inspect direction, negation, omitted constraints and RAG Top3 separately.
- Read-only resource sample aggregation cannot prove absence of transient peaks or memory leaks; driver-zero must be observed at every actual close. No image encoder, true microphone, physical execution or liveASR-driven LLM prompt.
- Parent run graph files are live dependencies: retain parent m32/m8 artifacts and symlink targets. Actual source and storage are rechecked under locks; reserve stays768MiB+150MiB, no lowering to force launch.
- Controller gate checks completion/recovery, not a standalone human/independent quality signoff file; root must enforce the promised result review before choosing execution. No automatic continuation is included.
- Observation failure does not prove board computation failure. No automatic rerun; retain original errors, recover partial only after both locks and report driver availability honestly.

## 冻结源SHA

|文件|SHA256|
|---|---|
|contract.py|b92e6592025d91cda461e1082e74e884e5c759af975f7cdee831e201a858a359|
|controller_fulli16.py|c04f5c31b87977a93c53921bde836ead1cafa4d3f3620e84ddb3d8c2c39b838c|
|board_fulli16.py|73c2d8e358e95de2479cb58403f1dcab8fb4ebe3823a6b34c4ab3f967e4adbde|
|analyze_fulli16.py|7e41ad7bf1496d1e7170c5087bd292a0231bf722277a2baf39bde2f96b8a6f13|
|config.fulli16.json|afe94be71621a693677b96094c59c9340862a611e5286e19c3e48a0a7f206582|
|inputs.json|d314a51d07b4a39ca9533cb805540ac90ce132819355c7ffe67aa4591d8cd49c|
|resident_worker_v1.py|2c110a6821ce9ee21aa2c07b062f27a49e18a9289e0fd8aea12927f80448321d|
|context_rag.py|81afe39a529dd5de5ca7b3a0ca21c870fe42777684cdebb1daec3dd7e6f4998e|
|README.md|931d1673597590c57705f4391f34adaca54675660a884bbe2ebf39ee2c5cf1e3|

逐常量与既有实际端口证据：INDEPENDENT_CANDIDATE_EVIDENCE.json。

追加空白复审：独立读取 Git index 的已审版本与当前工作树，只有第18、35行的行末空白删除；全文件rstrip后字节完全相等，AST完全一致，行为不变。controller冻结SHA已更新为 c04f5c31b87977a93c53921bde836ead1cafa4d3f3620e84ddb3d8c2c39b838c。
