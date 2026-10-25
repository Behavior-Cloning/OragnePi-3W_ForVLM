# 首chunk原图诊断：源审查

未发现阻断项。仅pos0/block1切换同shape原M16图，输入输出绑定共享的真实buffer，KV追加仍沿原路径；禁止grouped与该开关共用。非速度测试，先核对original grouped/ungrouped输出一致。预期16固定＋16自然、4close、8次首chunk切换。

额外原图在前两个original variant已SHA核对，同锁持有；最终variant的29项map未重复列入，建议以后显式列extra graph provenance。当前源不修改。
