import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from summarize import cpu_values,cpu_delta
p=argparse.ArgumentParser();p.add_argument('resources',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
rows=[json.loads(x) for x in a.resources.read_text().splitlines()];t=[(x['at']-rows[0]['at'])/60 for x in rows]
cpu=[cpu_delta(cpu_values(x),cpu_values(y)) for x,y in zip(rows,rows[1:])]
fig,ax=plt.subplots(4,1,figsize=(11,10),sharex=True,layout='constrained')
ax[0].plot(t,[x['temp_mC']/1000 for x in rows],lw=.8);ax[0].set_ylabel('Temperature (C)')
for name,label in [('cpu','Whole board (8 cores = 100%)'),('cpu6','CPU6 audio frontend'),('cpu7','CPU7 ASR/control')]:ax[1].plot(t[1:],[x[name] for x in cpu],lw=.55,alpha=.65,label=label)
ax[1].set_ylabel('CPU (%)');ax[1].legend(fontsize=8)
ax[2].plot(t,[x['meminfo']['MemAvailable']/1024 for x in rows],label='System MemAvailable');ax[2].set_ylabel('Available RAM (MiB)');ax[2].legend(fontsize=8)
for pid in rows[0]['processes']:
 name=rows[0]['processes'][pid]['stat'].split('(',1)[1].split(')',1)[0]
 if name=='llama-server':continue
 ax[3].plot(t,[x['processes'].get(pid,{}).get('Pss',float('nan'))/1024 for x in rows],label=name+' PSS')
ax[3].set_ylabel('Process PSS (MiB)');ax[3].set_xlabel('Elapsed test time (minutes)');ax[3].legend(fontsize=8)
for z in ax:z.grid(alpha=.25)
fig.suptitle('Resident audio + memory + NPU text | mixed prompt cache | no vision')
fig.savefig(a.output,dpi=160)
