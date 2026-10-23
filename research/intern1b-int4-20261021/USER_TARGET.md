# 本轮验收目标

2026-10-21 用户明确要求检验两层收益能否推广到完整模型，并达到50%提速。

- 完整InternVL3-1B文本decoder：24层+完整词表输出头，模型与system prefix KV常驻。
- 主性能目标：同输入、相同固定反馈轨迹、相同输出64 token、相同CPU亲和及prefill设置，W4 decode tokens/s / W8 decode tokens/s >= 1.50。
- 用W8→W4→W4→W8交错会话，每动态输入长度合计5次/版本，报告P50/P95及每次原始值。
- 50%吞吐提升等于单token耗时减少33.3%，不与50%耗时降低混用。
- 同时单列TTFT、模型加载、NPU driver、PSS、CPU、温度、swap。
- 5条自然回答与固定轨迹logits诊断用于本轮失败筛查，不能视为完整语言质量验收。速度达标与可用性分别下结论。
- 当前W4由ACUITY i4请求导出，但已确认FC常量实际为INT8容器中的-7..7，不能声称原生packed INT4硬件算术。
