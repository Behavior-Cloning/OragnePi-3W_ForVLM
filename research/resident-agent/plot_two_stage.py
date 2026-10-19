"""Measured endpoint/prefill overlap; frontend band is intermittent frame processing."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
p=argparse.ArgumentParser();p.add_argument('rows',type=Path);p.add_argument('output',type=Path);p.add_argument('--font',required=True);a=p.parse_args()
fm.fontManager.addfont(a.font);plt.rcParams.update({'font.family':fm.FontProperties(fname=a.font).get_name(),'axes.unicode_minus':False,'font.size':11,'svg.fonttype':'path'})
rows=[json.loads(s) for s in a.rows.read_text().splitlines()];pair=[next(x for x in rows if x['source_id']==2 and x['policy']==p) for p in ['baseline','speculative']]
fig,axs=plt.subplots(2,1,figsize=(15,9),sharex=True);fig.subplots_adjust(left=.16,right=.97,top=.82,bottom=.19,hspace=.65)
fig.suptitle('两阶段判停：把计算提前到静音确认期间',fontsize=21,y=.97)
fig.text(.16,.905,'“我想泡一杯茶，你还记得我喜欢喝什么吗？”  ·  干净语音配对实测',fontsize=13)
fig.text(.16,.865,'GTCRN 降噪始终开启；Paraformer 整段快照识别；NPU 文本模型常驻、system KV 64 token',fontsize=11)
for ax,row,title in zip(axs,pair,['基线：确认结束后才开始 ASR / NPU','两阶段：0.4 s 静音时先算，1.0 s 静音时确认']):
 s=row['start'];proxy=row['segment_end_sample']/16000;hard=row['endpoint_at']-s;resp=row['response'];first=resp['first']-s;end=resp['end']-s
 def bar(y,l,r,c,label=None,hatch=None):
  ax.barh(y,r-l,left=l,height=.48,color=c,edgecolor='white',hatch=hatch)
  if label:ax.text((l+r)/2,y,label,ha='center',va='center',fontsize=9)
 bar(3,0,proxy,'#5c9bc5','语音回放 → 话音结束代理点')
 bar(3,proxy,hard,'#d7dde3','静音等待')
 bar(2,0,hard,'#c3e2d9','从 t≈0 开始：16 ms 分帧降噪 / KWS，32 ms VAD（片段式处理）','///')
 for ar in row['asr']:bar(1,ar['begin']-s,ar['end']-s,'#e7a347')
 for c in row['candidates']:
  if 'request' in c:bar(0,c['request']-s,c['end']-s,'#a588cb','预计算')
 if not resp['cache_hit']:bar(0,resp['begin']-s,first,'#a588cb','prefill')
 bar(0,first,end,'#67a486','输出解码')
 for x in row['events']:
  if x['event']=='tentative':ax.axvline(x['at']-s,color='#b37a28',ls=':',lw=1)
 ax.axvline(hard,color='#66737d',ls='--',lw=1);ax.axvline(first,color='#287452',lw=1)
 ax.text(first,-.6,f'首 token {first:.3f}s\n结束后 {row["proxy_to_first_s"]:.3f}s',ha='center',va='top',color='#235b42',fontsize=10)
 ax.set_yticks([0,1,2,3],['NPU（独立执行）','ASR 快照识别','流式音频前端','模拟用户']);ax.set_title(title,loc='left',fontsize=13,pad=14)
 ax.set_ylim(-1.4,3.6);ax.set_xlim(0,9.5);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
 for sp in ['left','right','top']:ax.spines[sp].set_visible(False)
axs[-1].set_xlabel('相对语音回放起点 / 秒（唤醒样例在 t=0 之前完成）')
fig.legend(handles=[Patch(facecolor='#e7a347',label='ASR 实际调用区间'),Patch(facecolor='#a588cb',label='NPU prefill / 首 token 预计算'),Patch(facecolor='#67a486',label='正式输出')],loc='lower center',bbox_to_anchor=(.55,.075),ncol=3,frameon=False)
fig.text(.16,.035,'前端长条表示持续接收、分帧处理的时间窗口，并非 CPU 全程满载。ASR 与 NPU 各行允许并行。\n首 token 按首段非空解码文本计；话音结束为 VAD 样本代理点，未做人工声学标注。',fontsize=10,color='#53616a')
for ext in ['png','svg','pdf']:fig.savefig(a.output.with_suffix('.'+ext),dpi=160)
