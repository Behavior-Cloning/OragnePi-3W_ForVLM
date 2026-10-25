# 显式转换修复：I16 down候选已导出

本轮有实质进展，但没有新板端性能结果。完整目标不变：system正文59/123+模板5=64/128；动态正文64/128/192/256+模板8，实际新增72/136/200/264；仅复用固定system KV，完整prefill和TTFT均100ms，尚未实现。

## 定位证据

最小FC/Conv的FP16控制组与I16权重fl16/15/8共8个图全部成功导出。加入前后FP16边界后，FP16控制通过，两个I16配置Verify -3失败。再拆输入/输出边界：FC和Conv的输入FP16→I16 reshape均失败，输出I16→FP16 reshape均通过。故证据把本候选失败定位到输入边界这种写法，不能概括为NPU不支持I16权重。

仓库已有fc_input_precision.py的显式dtype_converter模式，旧attention精度实验也记录直接改变layout节点精度失败。保留两处reshape FP16，分别添加down_i16_2000/2001进行显式转换，再进入I16 FC。

## 新候选

C512，物理query32，真实层2/3融合块，两个权重为新的I16 DFP量化，另12个线性层原始导出权重字节不变。单轨迹前24行校准输入，加25%范围余量，后8行只用于范围检查，不足以确认对话质量。仍需多场景校准/验证。

SDK导出成功，NBG 29,748,736 B，SHA 55c03174538f31fe9dcd6a1828be71a3d076db47449463f8af50542ffb265d00。独立读取真实NB的16个端口，与原图接口逐字段相同；14权重检查确认12不变、2个I16码匹配预期round-to-nearest量化。这些都不代替实际NPU数值和性能测试。

首版最小图因layer ID缺少数字后缀在解析阶段失败，8项均不计SDK支持性证据；修复版包含FP16控制并遇控制失败即停止。所有失败保留。当前没有板卡测试进程。

下一步先清理已验证回收的M256板端上传副本，留足存储，再编写明确允许这两处精度变化的独立L2 A-B-A控制器。原图与候选使用相同canonical输入；3次预热、30次API；逐个比较hidden与4个KV输出，记录资源和driver释放。该单块成绩不称完整模型TTFT，也不能因快就跳过全词表logits和中文动作质量。

交付见 graphs/candidates-down-i16-explicit/deployment-manifest.json；最小图证据见 graphs/down-minimal-v2、down-minimal-bridges、down-minimal-boundaries。没有替换默认生产模型。
