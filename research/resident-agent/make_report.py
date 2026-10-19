"""Generate Markdown from completed saved evidence, never from estimates."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
def read(n):return json.loads((a.results/n).read_text())
long=read('valid30-both-summary.json');opt=read('optimized-both-summary.json');mem=read('memoryonly-both-summary.json');sweep=read('endpoint-sweep.json');kv=read('prefix-validation.json')
for x in [long,opt,mem]:assert x['meta']['status']=='complete' and not x['invalid_turns'],x['name']
assert long['meta']['elapsed_seconds']>=1800 and kv['status']=='passed'
assert all(x['meta']['cache_policy'].startswith('restore fixed system') for x in [opt,mem])
def f(v,n=3):return '—' if v is None else f'{v:.{n}f}'
def dist(d,k):
 z=d.get(k);return '—' if not z else ' / '.join(f(z[x]) for x in ['p50','p95','max'])
def mm(d,k,n=1):
 z=d[k];return f"{f(z['min'],n)}–{f(z['max'],n)}"
L=[]
def add(s=''):L.append(s)
def table(headers,rows):
 add();add('| '+' | '.join(headers)+' |');add('|'+'|'.join(['---']*len(headers))+'|')
 for row in rows:add('| '+' | '.join(str(v) for v in row)+' |')
 add()
add('# 板端常驻语音、记忆与 NPU 文本模型测试报告\n\n2026-10-19；Windows F 盘为源码和原始证据存储，本文同步到 Mac 阅读。')
add(f"\n完成 {long['meta']['elapsed_seconds']/60:.2f} 分钟、{long['all_valid']['turns']} 回合长测，以及 {opt['all_valid']['turns']} 回合优化端点复测、{mem['all_valid']['turns']} 回合无历史对照。GTCRN 降噪一直开启；KWS、VAD、双 ASR、tokenizer、SQLite 记忆与 1B NPU 文本引擎都在板卡本地常驻。Windows/Mac 不承担 ASR 或语言生成。")
add('\n**已完成的是语音＋NPU 文本链。完整 VLM 尚未验收：视觉 encoder/projector 没有接入 NPU 常驻路径，没有图片输入与视觉资源竞争结果。** 本轮也不含声纹/人脸/ASD、多人隔离、TTS 出声、真实麦克风/回声/打断。')
add('\n**没有证明稳定达到“说完后1秒内出首字”。** 以下分别从实际 VAD 判停、VAD 语段末代理点计时；没有人工冻结真人音素终点。WAV 文件末尾自带静音，文件末尾计时会低估等待。')
add('\n## 配置与链路\n\nOrange Pi Zero 3W / A733，8核、约6 GiB RAM。Windows Huihui 合成普通话，16 kHz 单声道 PCM，16 ms 节奏实时重放；12个生活场景重复，不是独立真人测试集。每回合先播放“小白小白”固定唤醒正例，再输入命令。没有测误唤醒率或免唤醒连续会话。')
add('\nGTCRN→KWS/VAD→Paraformer Small INT8（端点后整段识别）或 Streaming Zipformer Small CTC INT8（边输入边识别）→SQLite 记忆检索和打包→InternVL3-1B 的24层 NPU 文本模型及完整输出头。两个 ASR 同时常驻、轮流识别；NPU 生成期间前端继续处理模拟静音。CPU6跑前端，CPU7跑离线ASR/分词/控制；NPU单宿主线程的亲和集合为CPU4/5。')
add('\nSystem固定64 token，先建KV；动态预算上限轮换64/128/256，真实token数由原1B Q8 GGUF tokenizer计算。动态内容包含记忆、最多4轮完整历史和本轮输入，两条ASR各自维护历史；无历史对照只含system＋记忆＋本轮输入。输出上限48 token。模型每次试验加载一次，结束后正常退出；这不是已设置为开机启动的生产服务。')
add(f"\n长测静音阈值1.6秒；开发集扫描后短测为{sweep['recommended_threshold_s']}秒。固定模型常驻消除了逐句冷加载，system KV复用消除了固定前缀重算；新动态输入仍须prefill。")
add('\n**长测缓存口径注意：** 两个ASR交替用同一NPU session处理同一WAV，流式排在后面，部分prompt命中了前一个非流式请求的额外KV。长测表中的路径延迟不能用于宣称流式加速，资源统计也是此混合缓存负载。优化短测已在每轮唤醒前将KV恢复到64-token system，确保两个路径新请求仅复用system；恢复操作在命令输入计时之前，额外准备耗时另存。\n')
add('\n## 延迟结果\n\n单位秒，单元格为P50 / P95 / 最大值。首字是板端首个非空可见解码文本，不包含UI传输或TTS。输出结束可能为EOS，也可能为48-token上限。')
rows=[]
for label,x in [('优化端点＋历史',opt),('优化端点无历史',mem)]:
 for mode,z in x['by_mode'].items():rows.append([label+'/'+mode,z['turns'],dist(z,'endpoint_wait_from_proxy_s'),dist(z,'segment_proxy_to_first_s'),dist(z,'vad_to_first_s'),dist(z,'segment_proxy_to_complete_s'),f(z['decode_tokens_per_second']['p50'],2)])
table(['试验/路径','回合','语段末→判停','语段末代理→首字','VAD判停→首字','语段末代理→输出结束','NPU decode P50 tok/s'],rows)
add('30分钟的路径计时存在共享KV缓存偏差，不能作为两种ASR的延迟对照；仅用于稳定性、整体资源和长时间缓存混合压力。优化短测恢复system前缀后再比较。\n\n语段末→端点等待＋端点后ASR/流式收尾＋记忆检索/分词＋NPU首字构成等待。降噪、KWS、VAD和流式ASR在说话期间已工作，整段累计计算不能再次全加到说完后的等待。无历史对照减少了上下文信息量，不能叫相同上下文下无损加速。ASR路径的识别、历史与输出可能不同，端到端差异不能全归因ASR。')
add('\n### 输入长度与LLM首字（优化短测，同一system KV起点）')
rows=[]
for mode,groups in opt['by_mode_budget'].items():
 for b,z in groups.items():rows.append([mode,b,z['turns'],mm(z,'dynamic_tokens',0),mm(z,'new_tokens',0),dist(z,'llm_first_text_s')])
table(['路径','预算上限','样本','实际动态token','NPU新算token','LLM首字P50/P95/最大'],rows)
add('预算上限不等于每轮实际长度。额外公共历史也可能复用，逐轮reused/new token已记录。历史15.5 tok/s是固定反馈场景，不是本次自然回答、语音并发的保证速度。')
add('\n### 模型计算拆分（优化短测）\n\n前端为命令开始到端点的累计API耗时，含尾部静音，通常与录音重叠；不是纯算子计时。')
rows=[]
for mode,z in opt['by_mode'].items():
 st=z['command_stage_compute_s'];tail=z['asr_seconds']['p50'] if mode=='para' else z['endpoint_drain_seconds']['p50']
 rows.append([mode,*[f(st[k]['p50']) for k in ['denoise','kws','vad','stream_asr']],f(tail),f(z['memory_seconds']['p50']),f(z['llm_first_text_s']['p50'])])
table(['路径','降噪累计','KWS累计','VAD累计','流式ASR累计','端点后ASR/收尾','记忆＋打包','LLM首字'],rows)

add('\n## 系统资源与稳定性（30分钟，混合缓存负载）')
r=long['resources'];fc=r['frontend']
table(['指标','实测'],[
 ['温度范围 / P95',mm(r,'temperature_C')+' / '+f(r['temperature_C']['p95'],1)+' °C'],
 ['整机CPU时间加权平均 / 采样P95 / 最大',f(r['cpu_percent_8core_total100_time_weighted']['cpu'],2)+' / '+f(r['cpu_interval_percent']['cpu']['p95'],2)+' / '+f(r['cpu_interval_percent']['cpu']['max'],2)+'%'],
 ['1分钟load average范围',mm(r,'loadavg_1min',2)],
 ['MemAvailable范围',mm(r,'mem_available_MiB')+' MiB'],
 ['已占swap范围',mm(r,'swap_used_MiB')+' MiB'],
 ['新增swap in / out',str(r['swap_io_pages'])+' pages'],
 ['NPU driver分配',f(long['all_valid']['driver_bytes']['max']/1048576,1)+' MiB'],
 ['音频帧超过16ms预算',f"{fc['overruns_sampled']} / {fc['frames_sampled']} ({100*fc['overrun_fraction']:.2f}%)"],
 ['最大帧计算 / 最大回放积压',f(fc['max_compute_s']*1000,1)+' / '+f(fc['max_lag_s']*1000,1)+' ms'],
 ['关闭后driver bytes',long['meta']['closed'].get('driver_bytes','未记录')]])
add('整机CPU按8核合计100%，下面单核各自100%；load average不是CPU百分比。回放器不丢帧而是记录积压，不能冒充真实音频设备零丢帧。没有电功率/环境温度实测，也未独立核验当前散热装配。')
table(['CPU核','平均占用','P95占用','频率范围MHz'],[[k,f(r['cpu_percent_8core_total100_time_weighted'][k],1)+'%',f(r['cpu_interval_percent'][k]['p95'],1)+'%',f(r['cpu_freq_khz'][k]['min']/1000,0)+'–'+f(r['cpu_freq_khz'][k]['max']/1000,0)] for k in sorted(r['cpu_freq_khz'])])
table(['进程','RSS范围MiB','PSS范围MiB','PSS首/末MiB','单核等效平均CPU'],[[z['name']+' ('+pid+')',mm(z,'VmRSS_MiB'),mm(z,'Pss_MiB'),f(z['Pss_first_last_MiB'][0],1)+' / '+f(z['Pss_first_last_MiB'][1],1),f(z['one_core_cpu_percent'],1)+'%'] for pid,z in r['processes'].items()])
add('python3包含全部语音模型、分词和记忆；resident为NPU宿主。原CPU llama-server保留背景内存，不参与本轮推理。RSS/PSS和driver映射不可直接相加当作唯一物理占用；MemAvailable是整体余量参考。30分钟运行成功不是数天老化证明，频率波动也不能单独证明热降频。')
add('\n## ASR准确率、记忆与回答质量\n\nCER为字符编辑距离（替换＋删除＋插入）除以参考字符数。做NFKC、小写、去标点空白，保留字母数字；不自动等同数字读法或近义词。重复回放不增加独立样本数。')
rows=[]
for label,x in [('30分钟',long),('端点优化',opt),('无历史',mem)]:
 for mode,z in x['by_mode'].items():
  nz=z['memory_nonzero_added_tokens'];span='无命中' if not nz else f"{nz['min']}–{nz['max']}"
  rows.append([label,mode,f(100*z['cer']['rate'],2)+'%',str(z['cer']['exact_transcripts'])+'/'+str(z['turns']),span,str(z['finish_reasons'])])
table(['试验','路径','CER','完全一致转写/回合','记忆命中时额外token','输出结束原因'],rows)
add('记忆token包含包装开销，未命中为0。合成资料为小林、杭州、绿茶、花生过敏、小白机器人项目。没有声纹鉴别，全部是同一个已确认的合成用户，不能宣称实现了多人隔离。LLM未做盲评，需检查原文中的人称误用、事实错误和长度截断；KV数值一致不等于语言质量全面达标。')
add('\n## 端点优化与失败记录')
table(['静音阈值','仅一个语段/12条','未通过样本ID'],[[th+'秒',str(z['one_segment_count'])+'/12',z['failed_source_ids']] for th,z in sweep['summary'].items()])
add('0.25秒试验将句中停顿当结束，产生负的文件末尾→首字，已作废。0.8秒前三条能通过，扩展至“我住在哪个城市？你还记得我说过的信息吗？”出现分段，旧full30-para第10回合终止，不算30分钟成功。当前先完成保守阈值长测，再扫描12条开发样本和paced复测。端点选择仍须独立真人、家庭噪声验证。')
add('\n## KV修复、版本与证据\n\n旧逻辑遇动态后缀变化会清空KV。现在保留公共完整16-token prefill块、关闭被丢弃后缀mask，并至少重算最后一个prompt token。变化发生在64/128 token后、短输入、重复输入、无共同前缀5组，cached与reset全logits逐字节一致。Python协议4项和记忆7项测试通过；板端C以-O2 -Wall -Wextra -Werror编译。')
add('\n每轮在唤醒前恢复system-only KV，准备耗时 P50/P95/max：'+', '.join(f"{mode} {dist(z,'prefix_prepare_seconds')}s" for mode,z in opt['by_mode'].items())+'。这项发生在用户说话之前。\n')
add('旧0.106秒来自CPU重复prompt缓存命中、0.836秒为旧端点假设估算，均不能代表本轮语音＋NPU结果，旧报告已补充纠正说明。')
add('\nWindows代码：`F:/Files/OPi-Lab/repos/OragnePi-3W_ForVLM/research/resident-agent/`；NPU修改：`npu/kv-cache/resident_session.c`。原始证据：`F:/Files/OPi-Lab/results/resident-agent-20261019/`，含逐轮输入/输出、资源JSONL、失败尝试、端点扫描、退出状态与摘要。仓库`results/resident-agent-20261019`保留紧凑证据和SHA清单。')
add(f"\n长测执行源码SHA256：`{long['meta']['source_sha256']}`；短测：`{opt['meta']['source_sha256']}`。长测快照为`executed-valid30-agent_replay.py`；短测增加去重的模型计时、prompt记录与可选无历史开关。")
add('\n## 当前建议与尚未完成的内容\n\n保留两条ASR路径，以实际CER和端点后耗时选型。已实现常驻、正确的system KV复用和历史预算限制。约1秒目标仍受可靠端点和动态prefill成本限制；稳定partial提前预填充、错误前缀撤销未实现，不能预支提速收益。视觉NPU还需要encoder/projector导出、校准、编译、数值验证以及图像embedding协议接入；现有文本runner只接受token ID，不能用加载mmproj文件冒充VLM已常驻NPU。')
add('\n## 补充验收记录\n\n30分钟运行结束时的SSH返回通道滞留，重新连接后取得板端complete记录、144行结果和3472个资源采样，NPU正常释放为0。原始SSH退出码未取回，`valid30-both.exit`中的0仅是恢复核验通过标志，不能作为原shell退出码。后续两组短测均实际返回exit 0。')
add('\n健康检查：板端剩余约12 GiB磁盘，试验退出后MemAvailable约4428 MiB。内核末尾包含vipcore的`seq_file: buggy .next function vipdrv_seq_next`警告以及释放内存日志，本轮未表现为推理错误，但不能宣称驱动日志完全干净。Windows采样时F盘剩余60.9 GiB、可用内存24.6 GiB；WSL约473 MiB已用、swap为0，WSL虚拟盘内显示的901 GiB不等于F盘实际可用空间。')
add('\n回答质量抽查发现：茶偏好被表述成“我最喜欢喝的是绿茶”，存在用户/助手人称混淆；早点睡觉的问题出现“帮助你保持清醒和清醒”，有内容和复读问题；菜谱、过敏、整理桌面等回复触及48-token上限被截断。城市“杭州”、项目“小白机器人”、3+2=5能够回答。故本轮通过的是运行/资源验证，不是对话产品质量验收；重要健康建议不能凭这组回答直接上线。')
add('\n无历史对照为前6个场景，每条ASR各6轮；有历史对照覆盖全部12个场景，每条各12轮。小样本P95按nearest-rank计算，12个样本时就是最大值，不具有大量独立数据的尾延迟置信度。无历史输入实际17–32 token，并未覆盖无历史128/256 token饱和负载。')
add('\n公平短测整体资源：')
table(['试验','平均整机CPU','温度最大°C','最低可用MiB'],[[x['name'],f(x['resources']['cpu_percent_8core_total100_time_weighted']['cpu'],2)+'%',f(x['resources']['temperature_C']['max'],2),f(x['resources']['mem_available_MiB']['min'],1)] for x in [opt,mem]])
add('\n建议：保留Paraformer作为当前准确率基线，流式Zipformer作为降低端点后识别等待的可选路径；本组流式主要节省约0.12–0.21秒，不能预期仅替换ASR就让长历史场景达到1秒。下一阶段应先解决端点可靠性与LLM动态prefill，再接入完整视觉NPU并重新做稳定性和质量验收。')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(L)+'\n');print(a.output)
