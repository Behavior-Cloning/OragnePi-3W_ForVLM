# BITCAST 最小生成 C 验证：Verify 通过，NBG 导出失败

只做 WSL SDK 验证，没有板卡访问。复制失败 affine export 到独立目录，原件保留。14 个 DFP signed-I16→affine signed-I16 converter逐项断言zero_point=0、scale=2^-fl，并穷举每个转换全部65536整数码，网格完全相同。只替换14个DATACONVERT opcode为BITCAST；C tensor属性、边、端口、权重/data不改。证明在patch-proof.json与decoder.patch。

实际make rc0，Create成功11输入5输出，VerifyGraph返回0，Verify前后端口metadata逐位同。随后按SDK头文件文档用vsi_nn_GenerateNBG(graph,NULL,&bytes)查询NBG大小，调用中崩溃，容器rc139。没有NBG生成，也没有实机速度/精度成绩。日志最后是STAGE nbg_size_query。Verify通过不能称可部署，也不能直接把崩溃归因于BITCAST；native exporter/SDK问题尚未排除。

首次启动在图创建前因libvdtproxy.so搜索路径缺失退出127，原.loader-attempt日志保留。补入同SDK ../common/lib后才执行真正验证。SDK在Verify时重写了copy内的graph.json追踪文件，原始graph.json未变；RESULT.json同时保留两SHA，不能把生成trace的变化说成输入图参数变更。

环境：ubuntu-npu:v2.0.10.2、VIP9000NANODI_PLUS_PID0X1000003B、1 shader core、trace enabled、4CPU/8GiB、network none。运行持/tmp/opi-prefill-compile.lock，已退出并释放。源补丁、主程序、日志、状态、端口均在本目录，export/是可再生构建目录，Git忽略。F实际余量约25.01GB。

复现：在无export的独立副本先运行prepare.py，复制main-export.c.source为export/main.c，再运行run_compile.py。禁止覆盖失败原件。Root已进一步授权独立plain原始图同harness对照，用以区分生成器路径与BITCAST图问题；该对照另放目录，不改本失败证据。

## 原图对照补充

独立 native-export-control-c512-m32-l0 原图也在同NULL查询阶段139；故该失败不是BITCAST特有。SDK vxGenerateNBG内部可能Verify/Process，main中no simulator inference注释不精确，不能当内部行为证据；无已完成数值结果、未上板的范围仍成立。下一步仅在审核官方auto-save流程后再试。
