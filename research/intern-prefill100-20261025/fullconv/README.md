# C1280 单L2 Conv完整模型 A-B-A 准备

**仅完成准备和host preflight；此目录未启动板测。** Root负责唯一board owner和双锁。

固定顺序：`c1280-original → c1280-conv-l2 → c1280-repeat`。相同resident binary、量化例外、affinity、模板、token IDs及全部flags。只在第二个variant把 `prefill-layer2.nb` 换为FP16 Conv，decoder/head/embedding和其余11个prefill不变。每variant重新计算system-only KV，因此是相同system输入和缓存策略，不能称三者KV数值逐位相同。

- System总64/128（正文59/123+模板5）。
- 性能动态正文64/128/192/256，加模板8，真实新增72/136/200/264；每格3个冻结case，总24/variant，fixed-feedback32。
- 自然质量12/variant，为原6个短家庭情景×2个system，最多32输出；不是长正文64～256自然质量覆盖。
- 全词表诊断8格×4步/variant，排除正式计时。
- 三variant总72性能、36自然、24诊断请求（96诊断步），另有system准备和预热，不计正式样本。
- 语音模型沿用原fullbaseline的常驻Frontend及mock WAV；RAG仅在quality结束查询；无麦克风/图像encoder/真实动作。

## 使用（仅root）

```sh
python3 research/intern-prefill100-20261025/fullconv/controller_fullconv.py RUN_NAME
# 上面只做host检查，不import board_access，不访问板卡。
python3 research/intern-prefill100-20261025/fullconv/controller_fullconv.py RUN_NAME --execute --timeout 3600
```

运行名唯一，默认3600s外层timeout，15s kill宽限。Host持hostlock覆盖上传、运行和回收，board每步flock；远程试验也持同一board flock。异常保存状态并尝试重新取得双锁回收partial，不自动重启。磁盘要求上传后保留1GiB，并另预留100MiB结果空间；root清理可再生文件后再启动，不能降低阈值绕过。

候选：C1280 M16融合L2（真实层2/3），34,050,560B，SHA `dc772225e8c2b9ec2f6376132ba12aa9938dbe966b678e2bd7672832299fcfa6`。同量化/14权重字节/端口声明在config与图侧deployment manifest交叉约束；本控制器还核实本地及上传后SHA。不能用C512图替代。

## 单图overlay验证

远程独立目录为 `/opt/opi-vlm/intern-prefill100-20261025/fullconv/runs/RUN_NAME/`：

- 新文件 `prefill-layer2-conv.nb`。
- `overlay/`恰好29个symlink，`prefill-layer2.nb`只指向该新文件；其余28个逐一指向原C1280目录。
- upload每文件重新SHA；启动前检查symlink类型、literal target、resolve后的实际文件、candidate大小/SHA。
- runner再次验证overlay，并核对每variant完整29文件SHA，保存declared/resolved路径及缓存复用记录；保证原→repeat哈希相同、候选只有一个哈希不同。

## Host证据与限制

`host-preflight.json`：tokenizer现场校验24性能/12自然、8诊断格，max prompt392、32输出对应max KV423；17个故障配置/fixture/物理symlink负例均拒绝。`dry-preflight.log`证明默认命令不执行board工作。`source-freeze.json`保留原baseline源码SHA，原目录和旧结果未修改。实际请求、计时和前缀准备代码由原runner冻结复制，只增加三variant/overlay契约、源清单、计数与进展打印。

主机验证不能替代SDK关闭、数值一致性、语义质量和板端系统资源；成功退出还需3个closed driver0、固定反馈/全logits与自然结果分析。继承原runner的温度guard及每请求timeout；外层超时可能缺失finally/meta，controller会保留失败/partial，不伪报complete。
