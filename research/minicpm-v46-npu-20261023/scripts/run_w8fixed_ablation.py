from pathlib import Path
import sys,json,shlex,fcntl
r=Path(__file__).resolve().parents[1];sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
lock=Path('/opt/opi-lab/restored-slim3-20261017/projects/orangepi-zero3w/npu/results/board-host.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);d='/opt/opi-vlm/minicpm-v46-npu-20261023'
def run(cmd,timeout=45):
 z=b.ssh('flock -n /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),capture_output=True,text=True,timeout=timeout);assert z.returncode==0,z.stderr;return z
for label,part in [('w8fixed-linear','linear'),('w8fixed-attn','attn')]:
 target=d+'/'+label+'-all';lines=['mkdir -p '+target]
 for n in ['l'+str(i)+'.nb' for i in range(24)]+['head'+str(i)+'.nb' for i in range(8)]+['embeddings.f16','prompt.i32']:
  fixed=n.startswith('l') and ((int(n[1:-3])%4!=3)==(part=='linear'))
  lines.append('ln '+d+('/w8fixed-all/' if fixed else '/w8scaled-all/')+n+' '+target+'/'+n)
 run(' && '.join(lines))
 for suffix,cmd in [('runtime','taskset -c 6,7 ./mini_runtime_grouped '+target+' '+target+'/prompt.i32 '+d+'/'+label+'-runtime.jsonl 3'),('holdouts','env MINI_PROMPTS_DIR='+d+'/holdout-prompts taskset -c 6,7 ./mini_runtime_cases '+target+' '+d+'/holdout-prompts/0.i32 '+d+'/'+label+'-holdouts.jsonl 4')]:
  z=run('cd '+d+' && timeout --kill-after=10s 180s '+cmd,195);(r/'results'/(label+'-'+suffix+'.log')).write_text(z.stdout+z.stderr);b.scp(b.HOST+':'+d+'/'+label+'-'+suffix+'.jsonl',r/'results'/(label+'-'+suffix+'.jsonl'),check=True,timeout=30);print(label,suffix,'COMPLETE',flush=True)
