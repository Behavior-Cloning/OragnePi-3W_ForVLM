# NPU视觉与语言共驻harness：已准备，未运行

目标：JPEG→常驻NPU视觉64token→复用固定system KV的常驻Intern NPU语言。两模型只加载一次，记录首token/完整回复、视觉和预处理、温度、CPU、内存与swap。本目录是候选，不是成功部署报告。

## 已实现
- 复用已板测的vision_resident/vision_worker和语言视觉输入runtime，不替换默认。
- 6个VQA场景，1预热+12热请求，每轮重新编码，无图像缓存。
- system30+5=35，reuse32；动态图64+问题5–9，模板/边界11，new83–87。
- SHA检查、内存温度监测、分别关闭两worker并保存清理错误。
- 双锁控制；连接超时标记远端未知，不自行重跑。

## 启动前门槛
视觉NPU707619272bytes、语言NPU1334594552bytes，加完整FP32 embedding映射与256MiB余量，共2854248896bytes。要求后台CPU服务inactive、NPU current0、MemAvailable足够；未知指标也不通过。门槛仅排除已知资源不足，不保证不会OOM，不能替代监测。

prepare.py只上传脚本并运行检查，未加载模型。板端结果：

    {
      "eligible": false,
      "reasons": [
        "background CPU model not confirmed inactive",
        "insufficient available memory for measured models plus embedding/reserve"
      ],
      "available_bytes": 1322737664,
      "required_bytes": 2854248896,
      "swap_free_bytes": 24403968,
      "background_service": "active",
      "driver_bytes": 0
    }

5项主机门槛测试及Python语法检查通过；共驻推理、VQA、资源释放与长时稳定性仍未验证，不能用检查代替。

## 恢复
等待用户明确后台CPU服务运行安排，不擅停服务。条件满足后运行run_prepared.py，它再次检查门槛。若已有controller-status.json，先确认原任务状态，不能重复启动。当前没有共驻成功指标。
