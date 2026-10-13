"""Standalone figures for completed full-model temperature-stratified results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

here=Path(__file__).resolve().parent
data=json.loads((here/'evidence-thermal-20261013/summary.json').read_text())
plt.rcParams.update({'svg.hashsalt':'opi-thermal-matrix-v1','font.size':9})
colors={'low':'#1676b8','medium':'#df920d','high':'#c93d48'}
bands={'low':'40–50°C','medium':'55–<65°C','high':'65–80°C'}
for metric,label,name in [('decode_tokens_per_second','Decode (token/s)','decode'),('ttft_seconds','Time to first token (s)','ttft')]:
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
    fig.subplots_adjust(top=.80,bottom=.12,hspace=.42,wspace=.22)
    for ax,(model,history) in zip(axes.flat,[(m,h) for m in ('smol500','intern1b') for h in (512,1024)]):
        for band,color in colors.items():
            for backend,style,marker in [('cpu','-','o'),('npu','--','s')]:
                rows=sorted([r for r in data['rows'] if r['model']==model and r['history']==history and r['band']==band and backend in r],key=lambda r:r['new_tokens'])
                if rows:ax.plot([r['new_tokens'] for r in rows],[r[backend][metric] for r in rows],color=color,linestyle=style,marker=marker,markersize=5)
        if metric=='decode_tokens_per_second':ax.axhline(6,color='#888888',linestyle=':',linewidth=1)
        ax.set_title(('SmolVLM-500M' if model=='smol500' else 'InternVL3-1B')+f' · cached history {history}')
        ax.set_xticks([32,64,128]);ax.set_xlim(24,136);ax.set_ylim(bottom=0);ax.grid(alpha=.18)
        ax.set_xlabel('New input tokens (template included)');ax.set_ylabel(label)
        if not any(r['model']==model and r['history']==history and ('cpu' in r or 'npu' in r) for r in data['rows']):
            ax.text(.5,.5,'Pending',ha='center',va='center',transform=ax.transAxes,color='#777777')
    legend=[Line2D([0],[0],color=c,label=bands[b]) for b,c in colors.items()]
    legend += [Line2D([0],[0],color='black',marker='o',label='CPU Q8'),Line2D([0],[0],color='black',linestyle='--',marker='s',label='NPU W8 + FP16 exceptions')]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.89),ncols=3,frameon=False)
    fig.suptitle('Full language model · 128 predictions · median of 5 runs\nStart-temperature bands; temperature may change during inference',y=.98,fontsize=11)
    note='Only quality-passed, completed results shown. Missing points are pending/failed.'
    if metric=='decode_tokens_per_second':note+=' Dotted 6 token/s line: approximate speech reference, language/tokenizer dependent.'
    else:note+=' CPU includes local HTTP/SSE; NPU is in-process timing.'
    fig.text(.5,.02,note,ha='center',fontsize=7)
    fig.savefig(here/f'thermal-{name}.png',dpi=160,bbox_inches='tight')
    path=here/f'thermal-{name}.svg'
    fig.savefig(path,bbox_inches='tight',metadata={'Date':None})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)
