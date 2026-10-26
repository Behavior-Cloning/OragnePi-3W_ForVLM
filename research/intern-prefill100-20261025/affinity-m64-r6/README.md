# CPU affinity-only test

固定M64r4 binary和图、量化、13格输入、system-only KV、输出32。CPU4/5→7→4/5；语音仍CPU6，harnessCPU7轻量控制。不加载额外图，不改变governor。每格3个相同案例配对，12自然回归、6输入四步full logits，每阶段正常释放。只有完整实测之后才作性能/资源判断。
