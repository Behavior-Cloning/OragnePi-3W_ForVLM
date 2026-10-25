# 逐token动态归一化INT16 down候选

目标仍为完整system64/128、动态正文64～256另加模板8的prefill/TTFT均≤100ms，尚未达成。本图仅C1280/M16层2/3局部试验，不能把两层API当24层TTFT。

按每row的max(abs(x))决定scale，下限2^-14；FP16 x/scale→INT16/fl14→原INT16权重FC（FP16输出）→FP16乘scale恢复。实际bias全为零，避免缩放bias改变数学函数。无校准拟合。静态审计确认46常量原始字节和payload相同、14矩阵权重相同、16静态外部端口相同；仅零bias派生fl28→30、26→29。实际SDK端口由probe另核。

离线数据只含两条复用开发prompt、528原NPU有效行，不是独立验收。编译成功不等于精度或性能可用。controller_down_row_scale_replay.py已启动104进程对照，使用原NPU冻结上游hidden/KV；不允许从局部误差推断候选整模传播误差。所有原模型、候选、失败证据均保留。
