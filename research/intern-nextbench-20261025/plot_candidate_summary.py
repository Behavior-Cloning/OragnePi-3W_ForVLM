"""Render a standalone comparison from audited JSON (no board access)."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('summary');ap.add_argument('--out',required=True);a=ap.parse_args()
d=json.loads(Path(a.summary).read_text());assert d['audit_pass']
variants=['mask-base','context256','tail8-c1280','context256-greedy','mask-repeat']
labels=['Baseline','Context 256','True tail 8 / C1280','Context 256 + greedy','Baseline repeat']
cells=['s64-d32','s64-d64','s128-d32','s128-d64'];colors=['#6c7d8c','#2182ad','#dba049','#218a58','#b2bcc5']
if 'combined-tail8' in d['variants']:
 variants=['combined-base','combined-tail8','combined-repeat']
 labels=['Context 256 + greedy','Context 256 + greedy + true tail 8','Baseline repeat']
 colors=['#2182ad','#218a58','#b2bcc5']
fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True);x=np.arange(4);width=.24 if len(variants)==3 else .15
for i,(v,label,color) in enumerate(zip(variants,labels,colors)):
 for ax,key in zip(axes,['ttft_seconds','decode_tokens_s_recomputed']):
  # The analyzer metric name is checked explicitly rather than silently falling back.
  rows=[d['variants'][v]['cells'][c]['metrics'][key] for c in cells]
  mid=np.array([r['p50'] for r in rows]);err=np.array([[m-r['min'] for m,r in zip(mid,rows)],[r['max']-m for m,r in zip(mid,rows)]])
  ax.bar(x+(i-(len(variants)-1)/2)*width,mid,width,label=label,color=color,yerr=err,capsize=2,error_kw={'lw':.7})
  ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].axhline(1,color='#ac3a37',ls='--',lw=1,label='1-second target');axes[0].set_ylabel('Model TTFT (seconds)');axes[1].set_ylabel('Decode (tokens/second)')
axes[0].legend(ncol=3,fontsize=8);axes[1].set_xticks(x,['S64 / D32','S64 / D64','S128 / D32','S128 / D64']);axes[1].set_xlabel('S: fixed prefix including 5 template tokens; D: dynamic body (+8 template tokens)')
fig.suptitle('Intern 1B NPU: matched-input prefill and decode candidates',fontsize=14)
fig.text(.5,.015,'6 observations per cell / 3 fixtures. Bars: median; whiskers: observed range. 32 fixed-feedback steps.\nSystem KV only; speech resident; RAG queries outside timing. Text-only, no camera or physical motors.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.055,1,.95]);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
for suffix in ['png','svg']:fig.savefig(out/('prefill-decode-comparison.'+suffix),dpi=170)

svg=out/'prefill-decode-comparison.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
