# C512/M32 后续优化状态

2026-10-26，原C1280 suffixfixed完整实验已结束并回收，控制器rc0。首次片段原精度修复后仍有“右臂挥动一次→摆弄桌子”新质量退化，候选不晋升。详见fulli16-suffixfix/results/suffixfixed-c1280-aba-r1/QUALIFICATION_DECISION.md与独立审计。

新试验capacity512-full采用全部原权重/量化参数，L2显式选original分支。48张NB构成decode12、M16 12、M8 12、M32 12；已完成本地SHA和实际端口合同预检。M16→M32→M16重复，同binary、C512、trueM8尾、mask/group/greedy和亲和。M32也用于system准备，故不声称只改动态块。

system正文59/123+模板5=64/128；动态正文64/128/192/256+模板8=新72/136/200/264，仅复用system；固定反馈输出32，最大位置423，无输入截断。每版24性能、48自然、8份4步full-logits。独立源码审计通过后，新run c512-m16-m32-aba-r1已由WSL PID34040启动；不重复此前C1280实验。

板端已删除有Windows逐SHA备份的3份派生Laya/Nano ONNX，共1,161,213,577B，清理后free2,220,736,512B。新图上传约1.162GB，控制器再次检查768MiB+150MiB保留空间，stream tar避免板端双份归档。

100ms完整目标未达到；不可将编译、单块或输入子集当作整模达标，不采用质量失败的I16分支。
