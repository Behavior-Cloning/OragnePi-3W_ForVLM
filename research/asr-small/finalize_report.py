import json,statistics
from pathlib import Path
R=Path(__file__).resolve().parents[2];D=R/'results/asr-small-20261018';p=R/'docs/research/小ASR离线部署与LLM竞争报告-20261018.md';s=p.read_text();common=json.loads((D/'common-qwen-subset.json').read_text());q=json.loads((D/'standalone-qwen-v1/qwen/result.json').read_text());big=json.loads((D/'qwen-bigcore.json').read_text());post=json.loads((D/'board-postcheck.json').read_text());win=json.loads((D/'windows-resources.json').read_text())
mix=json.loads((D/'paced-mixed-summary.json').read_text())['mixed'];qconds={k:v['qwen'] for k,v in mix.items() if 'qwen' in v}
block=['','**Qwen 0.6B不建议作为当前板端默认ASR。** 同11句190字，Qwen4个编辑错误、Paraformer/离线Zipformer各5个，只有1字差异，不能证明稳定准确率优势。Qwen峰RSS约1.86GiB、同核中位延迟17.42秒，且仍把朱圣祎写成朱圣伟，未解决触发试验的人名问题。','', 'Qwen仅对冻结子集11条干净句+2条同源噪声+静音跑1遍，以下对照重新截取所有模型的同11句；质量不拿它和其他模型24句总CER直接排序。下面同子集延迟统一取各模型预热后第一遍。','','| 模型 | 同11句错误/参考字 | 同子集CER | 同子集单句中位数 |','|---|---:|---:|---:|']
for row in common:
 g=row['groups']['clean'];block.append(f"| {row['kind']} | {g['errors']}/{g['chars']} | {g['cer']:.2%} | {g['p50_seconds']:.3f}s |")
block+=['','Qwen在这一个0dB白噪声句上错7/16字，而离线Zipformer错2/16、Paraformer错5/16；只作失败诊断，不据单句概括噪声能力。Qwen静音返回空文本，但耗时8.55秒，仍应加VAD门控。','', 'Qwen各输出见 `common-qwen-subset.json`。数字/日期若写为阿拉伯数字会按字面CER计错，报告需结合原文判断，未静默归一化。','', 'Qwen独跑期间系统vmstat增量（页大小4096B）：pswpin=1页、pswpout=32790页、pgmajfault=100、oom_kill=0。为整机计数，不作单进程归因；有换出增加但没有观察到OOM。内核未提供/proc/pressure/memory，压力指标标记缺失。','', '大核CPU6,7、仍单线程的额外3句探针：','']
for r in big['runs']:block.append(f"- `{r['id']}`：{r['duration_seconds']:.3f}s音频，推理{r['seconds']:.3f}s，输出“{r['text']}”。")
block+=['','大核仅3句、一遍，未声称统计稳定；若用于生产会占用当前LLM主机线程的大核，本轮大核探针未与LLM同跑。Qwen竞争仍用与其他模型一致的CPU2,3单线程，不能将大核独跑速度与该竞争条件混用。','']
block += ['Qwen与NPU同跑（CPU2,3单线程、各批匹配基线）：','']
for name,v in qconds.items():block.append(f"- {name}: LLM均值{v['tps_mean']:.3f}token/s，相对基线{v['tps_change_percent']:+.2f}%；最低可用内存{v['min_available_mib']:.0f}MiB，系统swap峰{v['max_swap_mib']:.0f}MiB；ASR重叠调用RTF中位{statistics.mean(v['audio_rtf_p50']):.2f}，仍慢于实时。")
block.append('\n少样本、温度有差异，百分比为描述性结果；NPU解码基本维持，不等于Qwen的自身延迟或内存可接受。')
s=s.replace('## 存储清理','\n'.join(block)+'\n\n## 存储清理')
end=['','## 收尾资源与交付状态','',f"板卡最终可用空间 {post['disk']['free']/2**30:.2f}GiB，可用内存 {post['meminfo_kib']['MemAvailable']/1024:.0f}MiB，swap存量 {(post['meminfo_kib']['SwapTotal']-post['meminfo_kib']['SwapFree'])/1024:.0f}MiB。实验ASR工作进程剩余{len(post['active_asr_workers'])}个。生产三个服务状态：{post['services'].strip().replace(chr(10),'/')}。",f"Windows F盘可用 {win['F_filesystem']['free']/2**30:.2f}GiB；WSL可用内存 {win['WSL_memory_kib']['MemAvailable']/1024:.0f}MiB；Windows可用物理内存 {win['Windows_memory_kib']['FreePhysicalMemory']/1024:.0f}MiB。新ASR资产合计 {win['ASR_assets_bytes']/2**30:.2f}GiB，原始结果 {win['ASR_result_bytes']/2**20:.1f}MiB。",'', '结论：已完成候选离线试验部署、固定语料质量/资源测量及NPU竞争；尚未替换生产ASR服务，未完成家庭麦克风域验证、连续流式分句、自动复核路由或新模型GTCRN联调。这些属于下一阶段集成，不伪装成已验收。','', '许可证来源笔记见 `source-license-notes.json`：runtime与模型权重分开记录；未能核实的两个转换权重许可明确留空，不能由runtime Apache许可推断。']
fclean=json.loads((D/'F-cleanup.json').read_text());end.append(f"另删除F盘三个已逐文件验证解压一致的重复下载包，合计{sum(r['bytes'] for r in fclean['files'])/2**20:.2f}MiB；清单见F-cleanup.json。");s+='\n'.join(end)+'\n';p.write_text(s);print(p)
