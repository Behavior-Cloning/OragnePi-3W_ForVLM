# M256 BITCAST 独立实测审计

测量审计通过，候选未提速。存在一个交付 manifest 字段标签错误，详见末尾；实际参考文件与计算正确。

| 项目 | A 前 | BITCAST | A 后 |
|---|---:|---:|---:|
| P50 API (ms) | 210.7638185 | 213.5066215 | 210.8479020 |
| P95 API (ms，线性分位) | 211.0295106 | 213.7834007 | 211.0506679 |

每组3预热+30真实计时，候选比 A 前慢 1.3014%；A 前后漂移 +0.0399%。NPD/IO profile 和 resident-copy 都关闭，未混诊断计时。

五端口共360,448个f32（1,441,792B）全部finite，A/B/A逐字节相同，整份SHA `2548582d5782a9e74f36c02ad70d65a082e16c652c4779d8786d3cfc62934414`。实际源NB、候选NB、runner/probe/controller、canonical输入/参考和fixture元数据SHA全复核。

三组均使用同一已预载tmpfs canonical输入（13,238,272B），输入载入在API计时外。成功回收后控制器再次校验SHA再删除；其冻结源码与该运行config一致。前次r1只有中断上传，没有计入有效样本。

111个资源样本最高41.974°C；最低MemAvailable 4,573,208,576B；swap 112,721,920–112,721,920B、采样内pswpin/out增量0/0页。三组释放及post-health driver均为0。

限制：仅C512/M256、历史128的L0/1两层融合块、一个canonical输入；无语音/RAG，不是全模型TTFT，不能据此断言所有输入/层的数值一致或长期稳定。

交付元数据勘误：manifest 的 `reference_fixture_sha256` 实为 `probe-fixture.json` SHA `b6541753aaf63ca71b7952f731fe0daf1eaa754e3f3ff3b8355fa8cf1524ca36`；实际reference f32为 `376697d408c95617b0a96859eaef5ce2717ff5dd5907391c0e2a21e439c4e817`，运行config与分析器正确使用后者。已冻结manifest不覆盖，建议独立errata留痕。
