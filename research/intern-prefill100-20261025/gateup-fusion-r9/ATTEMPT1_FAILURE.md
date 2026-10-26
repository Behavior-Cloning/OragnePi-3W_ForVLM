# 首轮编译失败

run gateup-fusion-r9首轮在加载图阶段退出1，编译器要求layer键以整数ID结尾，而新增键使用了r9。未产生NBG、未部署板卡、无性能样本。保留build.py、status.json、m32-p448/compile.log；修订build_v2.py写入独立attempt2目录，改为唯一整数ID12000，并在编译前校验。其他融合逻辑不变。
