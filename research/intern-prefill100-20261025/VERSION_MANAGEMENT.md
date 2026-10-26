# 版本管理

Windows仓库为权威；Mac仅报告副本。baseline、有限验证版本、性能候选、smoke、局部探针分开登记。同名实验不得覆盖，源码/图/量化/tokenizer/fixture绑定SHA。

性能与质量分别判定。只在同输入token IDs、system-only缓存、同输出数量、同负载下比较。编译通过不等于部署通过；诊断时长不纳入性能统计。变更精度必须明确，原始导出data相同不能证明量化相同。

以VERSION_MANIFEST.json和各原始结果为准。BEST_VALIDATED.md包含已撤回的错误标签；没有当前证据就不得沿用历史“最优”。100ms全范围目标尚未完成。默认部署未改。
