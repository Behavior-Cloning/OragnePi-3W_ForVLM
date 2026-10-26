# r7完整模型耗时分解（尚未运行）

依赖M64r4图和CPU7设置。只加入可关闭计时与SDK只读profiling查询，不改图/量化/有效历史；不作为速度优化候选。源码提交后运行off/on/off。

system正文59/123＋模板5=64/128；动态正文64/128/256＋模板8=new72/136/264。仅system KV，6格×3案例×3阶段54次性能、36次自然回归、15份4步full logits，成本输出固定反馈32token。CPU7 worker，CPU6合成流前端，RAG常驻未查询。

计时trace stage0=prefill批，stage1=decode标量，stage2=head。每条分prepare/API/publish/query，带绝对pos、query行数和past容量。SDK分网络时间若为0或返回非成功，则记不可用，不能假造内核时间。计时函数与query自身会有扰动，off/on/off检查量级；数据分解不能作为新的最优性能值。

原r6归档的匹配诊断SHA作为原runtime对照；本轮三阶段logits/自然序列需一致，正常关闭driver0。模型与rawf32不入Git，回收文件实际SHA核验。
