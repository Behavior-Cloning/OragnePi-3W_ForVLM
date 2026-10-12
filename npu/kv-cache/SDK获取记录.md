# ACUITY SDK获取记录

2026-10-12：经用户授权，从同事电脑临时HTTP共享拉取`docker_images_v2.0.x.zip`完成。

- 文件大小：2903273080字节；传输约135.94秒，平均21.36MB/s。
- 本地路径：`npu/results/sdk-downloads/docker_images_v2.0.x.zip`，属于Git忽略目录。
- 外层ZIP完整性：`unzip -tq`通过。
- 外层SHA-256：`d876a96a74905b92d09ae5681256dd5fc6c430b0ebbd612097142473ec87e287`（本地计算，未取得源主机SHA-256作对照）。
- 内层镜像包：`docker_images_v2.0.x/ubuntu-npu_v2.0.10.2.tar.zip`，已解出。
- 内层MD5：`170bb054e5fc78cb70d91dd6a2fc9397`，与包内厂商校验文件完全一致。
- 内层包含`ubuntu-npu_v2.0.10.2.tar`，解压大小7883452416字节。
- 包内说明：ACUITY v6.30.22、IDE v5.11.0；更新提及ScatterND INT16适配。实际decoder编译支持仍需验证。

此前的“拿不到编译工具包”阻塞已解除。Docker镜像`ubuntu-npu:v2.0.10.2`已加载成功；用包内ACUITY 6.30.22、IDE 5.11.0编译增量decoder成功，生成NBG已在板上运行。转换脚本为`compile_probe.py`，逐阶段日志保存在`npu/results/`，小型编译元数据及NBG哈希见`evidence-20261012/compiler-provenance.json`。无需保持同事电脑共享。
