"""Plot only exact-feedback pairs; missing or rejected pairs are never zero bars."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    rows=json.loads((HERE/'evidence-attack-20261013/finalist-aligned-matrix.json').read_text())
    pairs={(r['model'],r['history'],r['new_tokens']):r for r in rows
           if r['band']=='medium' and r['trajectory_aligned'] is True and
           '配置搜索complete' in r['cpu_note']}
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.unicode_minus':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(13,8.2),sharex=True)
    configs=[(h,n) for h in (512,1024) for n in (32,64,128)]
    colors={'cpu':'#4274b3','npu':'#d38332'}
    for column,(model,title) in enumerate((('smol500','500M · 全32层'),('intern1b','1B · 全24层'))):
        for row_index,metric in enumerate(('decode','ttft')):
            ax=axes[row_index,column]
            for j,(history,new) in enumerate(configs):
                record=pairs.get((model,history,new))
                if not record:
                    ax.text(j,.05,'待配对',transform=ax.get_xaxis_transform(),ha='center',
                            va='bottom',color='#666666',fontsize=9)
                    continue
                source=Path(record['cpu']['source'])
                assert hashlib.sha256(source.read_bytes()).hexdigest()==record['cpu']['source_sha256']
                cpu=json.loads(source.read_text())
                cpu_runs=[r for r in cpu['runs'] if r['phase']=='measure']
                npu_runs=record['npu']['runs']
                assert len(cpu_runs)==len(npu_runs)==5
                values={'cpu':[r['timings']['predicted_per_second'] if metric=='decode' else r['timings']['prompt_ms']/1000 for r in cpu_runs],
                        'npu':[r['result']['decode_tokens_per_second'] if metric=='decode' else r['result']['ttft_seconds'] for r in npu_runs]}
                for backend,offset in (('cpu',-.19),('npu',.19)):
                    value=values[backend];median=float(np.median(value))
                    ax.bar(j+offset,median,width=.35,color=colors[backend],
                           yerr=[[median-min(value)],[max(value)-median]],capsize=2,
                           error_kw={'elinewidth':.7,'ecolor':'#333333'},
                           label=backend.upper() if j==0 and row_index==0 and column==0 else None)
            ax.set_xticks(range(6),[f'{h}+{n}' for h,n in configs],rotation=20)
            ax.set_xlim(-.65,5.65);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
            ax.set_ylabel('解码 token/s · 越高越好' if metric=='decode' else '引擎首token / 秒 · 越低越好')
            if metric=='decode':
                ax.set_title(title);ax.set_ylim(0,27)
                ax.axhline(6,color='#758564',linestyle='--',linewidth=1,
                           label='6 token/s 口语示意线' if column==0 else None)
            else:
                ax.set_ylim(0,2.9);ax.set_xlabel('已有KV token + 新增token（包含模板）')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.955),ncol=3,frameon=False)
    fig.suptitle('当前有效候选与最快已测CPU：中温起测、完整语言链路',fontsize=16,y=.995)
    fig.text(.06,.045,'每格预热＋5轮；柱为中位数，误差线为最小/最大值。固定128预测、相同CPU反馈；仅纳入已对齐且通过开发门槛的结果。',fontsize=9,color='#555555')
    fig.text(.06,.02,'起温55–<65°C，过程非恒温；首字不含加载/历史准备/HTTP。NPU具体版本见表；独立质量尚未全部通过。口语线并非固定分词换算。',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.075,1,.915))
    fig.savefig(HERE/'finalist-aligned-medium.png',dpi=160)
    fig.savefig(HERE/'finalist-aligned-medium.svg',metadata={'Date':None})
    svg=HERE/'finalist-aligned-medium.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    print('plotted paired medium cells',len(pairs))


if __name__=='__main__':main()
