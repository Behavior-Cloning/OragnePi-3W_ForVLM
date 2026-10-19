"""Plot paired real speech timelines; aggregate frontend windows are not API traces."""
import argparse,json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
p=argparse.ArgumentParser();p.add_argument('rows',type=Path);p.add_argument('output',type=Path);p.add_argument('--font',required=True);p.add_argument('--source-id',type=int,default=2);a=p.parse_args()
font=fm.FontProperties(fname=a.font);plt.rcParams['font.family']=font.get_name();fm.fontManager.addfont(a.font)
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.unicode_minus':False,'svg.fonttype':'path'})
rows=[json.loads(x) for x in a.rows.read_text().splitlines()];pair=[next(x for x in rows if x['source_id']==a.source_id and x['mode']==m) for m in ['para','stream']]
assert pair[0]['prompt_token_ids']==pair[1]['prompt_token_ids']
assert pair[0]['reply']==pair[1]['reply']
colors={'speech':'#3277b3','wait':'#dce3e9','front':'#d4e8e5','asr':'#edb55f','memory':'#bc77af','prefill':'#9269ba','decode':'#4b9b78'}
fig,axs=plt.subplots(2,1,figsize=(16,10.5),sharex=True)
fig.subplots_adjust(left=.13,right=.975,top=.84,bottom=.20,hspace=.64)
fig.suptitle('同一句语音：非流式 vs 流式时序',fontsize=23,fontweight='bold',x=.13,ha='left',y=.975)
fig.text(.13,.928,'“'+pair[0]['reference']+'”',fontsize=15)
fig.text(.13,.893,'常驻模型 · 降噪开启 · system KV 64 token + 动态输入32 token（其中记忆增加14 token）· 相同输出42 token',fontsize=11.5,color='#455468')
intervals=[]
for ax,x in zip(axs,pair):
 mode=x['mode'];speech=x['segment_end_sample']/16000;ep=x['vad_endpoint']-x['command_start'];first=x['file_duration_s']+x['file_end_to_first_s'];end=x['file_duration_s']+x['file_end_to_complete_s'];llm=first-x['llm_first_text_s'];mem=llm-x['memory_seconds'];asr_end=ep+(x['asr_seconds'] if mode=='para' else x['endpoint_drain_seconds'])
 title='非流式：Paraformer' if mode=='para' else '流式：Zipformer Small'
 ax.set_title(title,loc='left',pad=23,fontsize=16,fontweight='bold')
 def bar(y,start,stop,color,label='',hatch=None):
  ax.barh(y,stop-start,left=start,height=.54,color=color,edgecolor='#648681' if hatch else 'none',hatch=hatch,linewidth=0,zorder=3)
  if label:ax.text((start+stop)/2,y,label,ha='center',va='center',fontsize=11,zorder=4,color='#152232')
 bar(2,0,speech,colors['speech'])
 ax.text(speech/2,2,f'说话 / 命令输入  [0, {speech:.3f}] s',color='white',ha='center',va='center')
 bar(2,speech,ep,colors['wait'],'等静音\n1.022 s')
 group='降噪 → KWS → VAD：逐帧串行' if mode=='para' else '降噪 → KWS → VAD：随输入逐帧处理'
 bar(1,0,ep,colors['front'],group,hatch='///')
 bar(1,ep,end,'#eaf2ef','继续处理模拟静音（与 LLM 并行）',hatch='///')
 # Aggregate post-endpoint chain. Memory boundary reconstructed backwards
 # from LLM start and measured packing duration; small dispatch gaps remain.
 if mode=='stream':
  # Orange interval is the streaming ASR processing window, not an invented
  # per-kernel trace. Hatching denotes intermittent chunk execution.
  ax.barh(0,ep,left=0,height=.54,color=colors['asr'],edgecolor='#a87629',hatch='///',linewidth=0,zorder=3)
  ax.text(ep/2,0,'流式 ASR：从 t≈0 按片段持续识别',ha='center',va='center',fontsize=12,zorder=4,color='#382b15')
  ax.text(.03,-.60,'↑ t≈0 开始处理语音',ha='left',fontsize=10.5,color='#91601c')
 bar(0,ep,asr_end,colors['asr'])
 if mem>asr_end:bar(0,asr_end,mem,'#dce3e9')
 bar(0,mem,llm,colors['memory'])
 bar(0,llm,first,colors['prefill'],'LLM 首字阶段')
 bar(0,first,end,colors['decode'],'LLM 后续生成')
 ax.annotate(('整段 ASR' if mode=='para' else 'ASR 收尾')+f" {1000*(asr_end-ep):.1f} ms",xy=((ep+asr_end)/2,-.28),xytext=(4.55,-.80),fontsize=10.5,ha='right',arrowprops={'arrowstyle':'-','color':'#a27229','lw':1})
 ax.annotate(f"记忆＋打包 {1000*x['memory_seconds']:.1f} ms",xy=((mem+llm)/2,-.28),xytext=(llm+.25,-.80),fontsize=10.5,ha='left',arrowprops={'arrowstyle':'-','color':colors['memory'],'lw':1})
 for v,label,color,yy in [(speech,f'话音末尾* {speech:.3f}s','#3277b3',2.55),(ep,f'VAD判停 {ep:.3f}s','#7f692f',2.91),(first,f'首字 {first:.3f}s','#754a9b',.51),(end,f'输出结束 {end:.3f}s','#2c7151',.51)]:
  ax.axvline(v,color=color,lw=.8,ls='--',alpha=.7,zorder=1);ax.text(v+.04,yy,label,color=color,fontsize=10.5,ha='left')
 ax.text(9.8,2.0,f'话音末尾*→首字\n{first-speech:.3f} s',ha='right',va='center',fontsize=14,fontweight='bold',color='#23364c')
 ax.set_yticks([2,1,0],['说话人 / 输入','前处理\n降噪 / KWS / VAD','识别与回复链\nASR → 记忆 → LLM'])
 ax.tick_params(axis='y',length=0,pad=12);ax.set_ylim(-1.0,3.15);ax.set_xlim(0,10);ax.set_xticks(range(11));ax.grid(axis='x',alpha=.17,zorder=0);ax.set_axisbelow(True)
 ax.tick_params(axis='x',labelbottom=True);ax.set_xlabel('从命令语音开始计时（秒）',labelpad=7)
 intervals.append({'mode':mode,'speech_proxy':[0,speech],'endpoint':ep,'asr_or_drain':[ep,asr_end],'dispatch_gap':[asr_end,mem],'memory_approx':[mem,llm],'llm_to_first':[llm,first],'llm_remaining':[first,end],'source_turn':x['turn'],'boundary_note':'Memory start/end reconstructed from duration and LLM start; frontend windows are not continuous busy spans.'})
fig.text(.13,.137,'* 话音末尾采用 VAD 语段末样本代理（4.088 s），不是人工标注；原 WAV 结束于4.810 s，末尾含静音。',fontsize=11,color='#455468')
fig.text(.13,.104,'橘色斜线带＝流式 ASR 的片段处理窗口；末端实色橘块＝判停后的收尾。没有逐片段时间戳，斜线不代表真实调用次数。',fontsize=11,color='#455468')
fig.text(.13,.071,'前处理与流式 ASR 都随输入推进，按模型所需块长执行；两行表示重叠处理窗口，不表示它们在不同 CPU 线程同时执行。',fontsize=11,color='#455468')
fig.text(.13,.038,'t≈0 为首帧输入的逻辑起点；毫秒级 API 边界按耗时还原。模型已常驻，此前唤醒、加载和 KV 准备不计入本图。',fontsize=11,color='#455468')
a.output.parent.mkdir(parents=True,exist_ok=True)
for ext in ['png','svg','pdf']:fig.savefig(a.output.with_suffix('.'+ext),dpi=180,facecolor='white')
a.output.with_suffix('.json').write_text(json.dumps({'source':str(a.rows),'reference':pair[0]['reference'],'intervals':intervals},ensure_ascii=False,indent=2))
print(json.dumps(intervals,ensure_ascii=False))
