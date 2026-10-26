# C256/M256 compact probe

C256专用probe binary已编译并通过板端运行；仅两层API、history256、query256，不是完整Intern TTFT，也无语音/RAG。

|路径|P50 ms|P95 ms|
|---|---:|---:|
|m256-before|90.108|90.243|\n|m256-row|109.017|109.156|\n|m256-after|89.923|90.033|\n\nplain A/A输出字节一致；row输出finite=True，相对L2误差=0.4606（约46.1%），row版比plain慢约21%。因此本轮只保留C256 plain，不采用row组合。\n\nC256 plain约90.1ms是两层局部API成绩，不能外推整模；整模仍需worker和全层图接口改造。
